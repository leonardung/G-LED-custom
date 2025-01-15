from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

from config.diff_args_typed import DiffTypedArgs
from config.seq_args_typed import SeqTypedArgs
from diffusion_models.trainer import ImagenTrainer


def test_final(
    args_seq: SeqTypedArgs,
    args_diff: DiffTypedArgs,
    trainer: ImagenTrainer,
    model,
    data_loader,
    down_sampler,
    up_sampler,
    save_flag=True,
):
    print(f"{args_diff.autoregressive_steps=}")
    contour_dir = args_diff.logging_path + "/contour"
    os.makedirs(contour_dir, exist_ok=True)
    loss_func = torch.nn.MSELoss()
    Nt = args_diff.autoregressive_steps
    LED_micro_list = []
    LED_macro_list = []
    Dif_recon_list = []
    CFD_micro_list = []
    CFD_macro_list = []
    with torch.no_grad():
        IDHistory = [i for i in range(1, args_seq.num_timesteps)]
        REs = []
        REs_fine = []
        print("total ite", len(data_loader))
        for iteration, batch in tqdm(enumerate(data_loader)):
            batch = batch.to(args_diff.device).float()
            b_size = batch.shape[0]
            assert b_size == 1
            num_time = batch.shape[1]
            num_velocity = batch.shape[2]
            image_dim = batch.shape[3:]
            z_micro = image_dim[-1] // 2
            z_macro = args_seq.coarse_dim[-1] // 2
            # z = 1
            extent1, extent2 = 4, 10
            batch = batch.reshape([b_size * num_time, num_velocity, *image_dim])

            batch_coarse = down_sampler(batch).reshape(
                [
                    b_size,
                    num_time,
                    num_velocity,
                    *args_seq.coarse_dim,
                ]
            )

            batch_coarse_flatten = batch_coarse.reshape(
                [
                    b_size,
                    num_time,
                    num_velocity * args_seq.coarse_product,
                ]
            )

            past = None
            xn = batch_coarse_flatten[:, 0:1, :]
            previous_len = 1
            mem = []
            print(f"{Nt=}")
            for j in tqdm(range(Nt)):
                if j == 0:
                    xnp1, past, _, _ = model(inputs_embeds=xn, past=past)
                elif past[0][0].shape[2] < args_seq.num_timesteps and j > 0:
                    xnp1, past, _, _ = model(inputs_embeds=xn, past=past)
                else:
                    past = [
                        [
                            past[l][0][:, :, IDHistory, :],
                            past[l][1][:, :, IDHistory, :],
                        ]
                        for l in range(args_seq.n_layer)
                    ]
                    xnp1, past, _, _ = model(inputs_embeds=xn, past=past)
                xn = xnp1
                mem.append(xn)
            mem = torch.cat(mem, dim=1)
            local_batch_size = mem.shape[0]
            for i in tqdm(range(local_batch_size)):
                er = loss_func(
                    mem[i : i + 1],
                    batch_coarse_flatten[
                        i : i + 1, previous_len : previous_len + Nt, :
                    ],
                )
                r_er = er / loss_func(
                    mem[i : i + 1] * 0,
                    batch_coarse_flatten[
                        i : i + 1, previous_len : previous_len + Nt, :
                    ],
                )
                REs.append(r_er.item())
                prediction = mem[i : i + 1]
                truth = batch_coarse_flatten[
                    i : i + 1, previous_len : previous_len + Nt, :
                ]
                # spatial recover
                prediction = prediction.reshape(
                    [
                        prediction.shape[0],
                        prediction.shape[1],
                        num_velocity,
                        *args_seq.coarse_dim,
                    ]
                )
                truth = truth.reshape(
                    [
                        truth.shape[0],
                        truth.shape[1],
                        num_velocity,
                        *args_seq.coarse_dim,
                    ]
                )

                assert prediction.shape[0] == truth.shape[0] == 1
                bsize_here = 1
                ntime = prediction.shape[1]
                print(f"{ntime=}")

                prediction_macro = up_sampler(prediction[0]).reshape(
                    [bsize_here, ntime, num_velocity, *image_dim]
                )
                truth_macro = up_sampler(truth[0]).reshape(
                    [bsize_here, ntime, num_velocity, *image_dim]
                )
                print(f"{truth.shape=}")
                print(f"{truth_macro.shape=}")

                if len(image_dim) == 3:
                    prediction_macro = prediction_macro.permute([0, 2, 1, 3, 4, 5])
                    truth_macro = truth_macro.permute([0, 2, 1, 3, 4, 5])
                    truth_micro = batch.permute([1, 0, 2, 3, 4]).unsqueeze(0)[:, :, 1:]
                elif len(image_dim) == 2:
                    prediction_macro = prediction_macro.permute([0, 2, 1, 3, 4])
                    truth_macro = truth_macro.permute([0, 2, 1, 3, 4])
                    truth_micro = batch.permute([1, 0, 2, 3]).unsqueeze(0)[:, :, 1:]
                else:
                    raise ValueError("image dim should be 2 or 3")
                recon_micro = []
                prediction_micro = []
                # pdb.set_trace()
                vf = args_diff.autoregressive_steps
                Nvf = args_diff.autoregressive_steps // vf
                for j in range(Nvf):
                    recon_micro.append(
                        trainer.sample(
                            video_frames=vf,
                            cond_images=truth_macro[:, :, vf * j : vf * j + vf],
                        )
                    )
                    prediction_micro.append(
                        trainer.sample(
                            video_frames=vf,
                            cond_images=prediction_macro[:, :, vf * j : vf * j + vf],
                        )
                    )
                recon_micro = torch.cat(recon_micro, dim=2)
                prediction_micro = torch.cat(prediction_micro, dim=2)
                # pdb.set_trace()

                seq_name = "batch" + str(iteration) + "sample" + str(i)
                try:
                    os.makedirs(contour_dir + "/" + seq_name)
                except:
                    pass

                DIC = {
                    "prediction_micro": prediction_micro.detach().cpu().numpy(),
                    "recon_micro": recon_micro.detach().cpu().numpy(),
                    "truth_micro": truth_micro.detach().cpu().numpy(),
                    "truth_macro": truth_macro.detach().cpu().numpy(),
                    "prediction_macro": prediction_macro.detach().cpu().numpy(),
                }
                pickle.dump(
                    DIC,
                    open(contour_dir + "/" + seq_name + "/DIC.npy", "wb"),
                    protocol=4,
                )
                if len(image_dim) == 3:
                    print(f"{prediction_micro.shape=}")
                    print(f"{recon_micro.shape=}")
                    print(f"{truth_micro.shape=}")
                    print(f"{truth_macro.shape=}")
                    print(f"{prediction_macro.shape=}")
                    print(f"{truth.shape=}")
                    print(f"{prediction.shape=}")
                    prediction_micro = prediction_micro[:, :, :, :, :, z_micro]
                    recon_micro = recon_micro[:, :, :, :, :, z_micro]
                    truth_micro = truth_micro[:, :, :, :, :, z_micro]
                    truth_macro = truth_macro[:, :, :, :, :, z_micro]
                    prediction_macro = prediction_macro[:, :, :, :, :, z_micro]
                    truth = truth[:, :, :, :, :, z_macro]
                    prediction = prediction[:, :, :, :, :, z_macro]
                for d in tqdm(range(num_velocity + 1)):
                    try:
                        os.makedirs(contour_dir + "/" + seq_name + "/" + str(d))
                    except:
                        pass
                    sub_seq_name = contour_dir + "/" + seq_name + "/" + str(d)
                    for t in tqdm(range(Nt)):
                        # pdb.set_trace()
                        if d == len(image_dim):
                            velo_micro_truth = np.sqrt(
                                truth_micro[0, 0, t, :, :].cpu().numpy() ** 2
                                + truth_micro[0, 1, t, :, :].cpu().numpy() ** 2
                            )
                            velo_micro_led = np.sqrt(
                                prediction_micro[0, 0, t, :, :].cpu().numpy() ** 2
                                + prediction_micro[0, 1, t, :, :].cpu().numpy() ** 2
                            )
                            velo_micro_rcons = np.sqrt(
                                recon_micro[0, 0, t, :, :].cpu().numpy() ** 2
                                + recon_micro[0, 1, t, :, :].cpu().numpy() ** 2
                            )
                            velo_macro_truth = np.sqrt(
                                truth[0, t, 0, :, :].cpu().numpy() ** 2
                                + truth[0, t, 1, :, :].cpu().numpy() ** 2
                            )
                            velo_macro_led = np.sqrt(
                                prediction[0, t, 0, :, :].cpu().numpy() ** 2
                                + prediction[0, t, 1, :, :].cpu().numpy() ** 2
                            )
                            vmin = 0  # truth[0,:,d,:,:].cpu().numpy().min()
                            vmax = 20  # truth[0,:,d,:,:].cpu().numpy().max()
                        else:
                            velo_micro_truth = truth_micro[0, d, t, :, :].cpu().numpy()
                            velo_micro_led = (
                                prediction_micro[0, d, t, :, :].cpu().numpy()
                            )
                            velo_micro_rcons = recon_micro[0, d, t, :, :].cpu().numpy()
                            velo_macro_truth = truth[0, t, d, :, :].cpu().numpy()
                            velo_macro_led = prediction[0, t, d, :, :].cpu().numpy()
                            vmin = truth[0, :, d, :, :].cpu().numpy().min()
                            vmax = truth[0, :, d, :, :].cpu().numpy().max()

                        fig, axes = plt.subplots(nrows=1, ncols=5)
                        fig.subplots_adjust(hspace=0.6)
                        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                        im0 = axes[0].imshow(
                            velo_macro_led[:, :],
                            extent=[0, extent1, 0, extent2],
                            cmap="jet",
                            interpolation=None,
                            norm=norm,
                        )
                        axes[0].set_title("LED Macro")
                        axes[0].set_ylabel("y")
                        axes[0].set_xticks([])

                        im1 = axes[1].imshow(
                            velo_macro_truth[:, :],
                            extent=[0, extent1, 0, extent2],
                            cmap="jet",
                            interpolation=None,
                            norm=norm,
                        )
                        axes[1].set_title(
                            "Truth Macro"
                        )  # ,rotation=-90, position=(1, -1), ha='left', va='center')
                        axes[1].set_ylabel("y")
                        axes[1].set_xticks([])

                        im2 = axes[2].imshow(
                            velo_micro_led[:, :],
                            extent=[0, extent1, 0, extent2],
                            cmap="jet",
                            # interpolation="bicubic",
                            norm=norm,
                        )
                        axes[2].set_title(
                            "LED Micro"
                        )  # ,rotation=-90, position=(1, -1), ha='left', va='center')
                        axes[2].set_ylabel("y")
                        axes[2].set_xticks([])

                        im3 = axes[3].imshow(
                            velo_micro_rcons[:, :],
                            extent=[0, extent1, 0, extent2],
                            cmap="jet",
                            # interpolation="bicubic",
                            norm=norm,
                        )
                        axes[3].set_title(
                            "Reconstruction Micro"
                        )  # ,rotation=-90, position=(1, -1), ha='left', va='center')
                        axes[3].set_ylabel("y")
                        axes[3].set_xticks([])

                        im4 = axes[4].imshow(
                            velo_micro_truth,
                            extent=[0, extent1, 0, extent2],
                            cmap="jet",
                            # interpolation="bicubic",
                            norm=norm,
                        )
                        axes[4].set_title(
                            "Truth Micro"
                        )  # ,rotation=-90, position=(1, -1), ha='left', va='center')
                        axes[4].set_xlabel("x")
                        axes[4].set_ylabel("y")

                        fig.subplots_adjust(right=0.8)
                        fig.colorbar(im0, orientation="vertical", ax=axes)
                        # fig.set_size_inches(20, 15, forward=True)
                        fig.savefig(
                            sub_seq_name + "/time" + str(t) + ".png",
                            bbox_inches="tight",
                            dpi=500,
                        )
                        plt.close(fig)

    