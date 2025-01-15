import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from config.diff_args_typed import DiffTypedArgs
from config.seq_args_typed import SeqTypedArgs
from diffusion_models.elucidated_imagen_3d import ElucidatedImagen
from diffusion_models.trainer import ImagenTrainer
from diffusion_models.unet3d import Unet3D
from model_registry import register_models, get_model
from dataset_registry import register_datasets, get_dataset
from train_test_diff.test_diff import test_final
from train_test_diff.train_diff import train_diff
from util.utils import load_config, save_args

register_datasets()
register_models()


if __name__ == "__main__":
    diff_config_path = "config/diff_3d_flow.yml"
    diff_config: DiffTypedArgs = load_config(diff_config_path)
    diff_config: DiffTypedArgs = load_config(
        Path("output") / diff_config.training_path / Path("diffusion/logging/args.yml")
    )
    diff_config.device = "cuda:0"
    diff_config.device = torch.device(diff_config.device)
    print(diff_config.device)
    diff_config.num_sample_steps = 20
    diff_config.unet_dim=64

    seq_config: SeqTypedArgs = load_config(
        os.path.join("output", diff_config.training_path, "logging/args.yml")
    )
    seq_config.batch_size = 1
    seq_config.batch_size_valid = 1

    val_dataset = get_dataset(
        seq_config.dataset, **{"config": seq_config, "mode": "val"}
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=seq_config.batch_size_valid,
        shuffle=False,
        num_workers=0,
    )
    image_dim = list(next(iter(val_dataloader)).shape[3:])
    print(f"{image_dim = }")
    print(f"{len(val_dataset) = }")
    print(f"{len(val_dataloader) = }")

    unet1 = Unet3D(
        dim=diff_config.unet_dim,
        cond_images_channels=seq_config.n_velocities,
        memory_efficient=True,
        dim_mults=(1, 2, 4, 8),
    ).to(diff_config.device)
    image_sizes, image_width, image_depth = image_dim
    imagen = ElucidatedImagen(
        unets=(unet1,),
        image_sizes=image_sizes,
        image_width=image_width,
        image_depth=image_depth,
        channels=seq_config.n_velocities,  # Han Gao add the input to this args explicity
        random_crop_sizes=None,
        num_sample_steps=diff_config.num_sample_steps,  # original is 10
        cond_drop_prob=0.1,
        sigma_min=0.002,
        sigma_max=(
            80
        ),  # max noise level, double the max noise level for upsampler  （80，160）
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
        condition_on_text=False,
        auto_normalize_img=False,  # Han Gao make it false
        device=diff_config.device,
    ).to(diff_config.device)
    trainer = ImagenTrainer(imagen, fp16=True, device=diff_config.device)
    trainer.load(path=diff_config.model_save_path + "/best_model_sofar")
    down_sampler = torch.nn.Upsample(
        size=seq_config.coarse_dim, mode=seq_config.coarse_mode
    )
    up_sampler = torch.nn.Upsample(size=image_dim, mode=seq_config.coarse_mode)
    model: torch.nn.Module = get_model(
        seq_config.model_name, **{"config": seq_config}
    ).to(diff_config.device)
    model.load_state_dict(
        torch.load(Path(seq_config.model_save_path) / diff_config.model_name)
    )
    model.eval()
    batch: torch.Tensor = next(iter(val_dataloader))
    b_size = batch.shape[0]
    num_time = batch.shape[1]
    num_velocity = batch.shape[2]
    image_dim = batch.shape[3:]
    batch = batch.reshape([b_size * num_time, num_velocity, *image_dim])
    batch_coarse: torch.Tensor = down_sampler(batch).reshape(
        [
            b_size,
            num_time,
            num_velocity,
            *seq_config.coarse_dim,
        ]
    )

    batch_coarse_flatten = batch_coarse.reshape(
        [
            b_size,
            num_time,
            num_velocity * seq_config.coarse_product,
        ]
    )
    truth_micro = batch.permute([1, 0, 2, 3, 4]).unsqueeze(0)[:, :, 1:]
    i = 0  # batch
    previous_len = 1
    Nt = diff_config.autoregressive_steps
    truth = batch_coarse_flatten[i : i + 1, previous_len : previous_len + Nt, :]
    truth = truth.reshape(
        [
            truth.shape[0],
            truth.shape[1],
            num_velocity,
            *seq_config.coarse_dim,
        ]
    )
    ntime = truth.shape[1]
    truth_macro: torch.Tensor = up_sampler(truth[0]).reshape(
        [1, ntime, num_velocity, *image_dim]
    )
    print(f"{truth_macro.shape=}")
    truth_macro = truth_macro.permute([0, 2, 1, 3, 4, 5])
    j = 0
    vf = diff_config.autoregressive_steps
    prediction: torch.Tensor = trainer.sample(
        video_frames=vf,
        cond_images=truth_macro[:, :, vf * j : vf * j + vf],
    )

    print(f"{truth_micro.shape=}")
    print(f"{truth_macro.shape=}")
    print(f"{prediction.shape=}")

    autoregressive_steps = 3
    fig, axes = plt.subplots(
        autoregressive_steps,
        3,
        figsize=(5, 4 * autoregressive_steps),
    )
    timestep = 0
    vmin = truth[0, :, 0, :, :].cpu().numpy().min()
    vmax = truth[0, :, 0, :, :].cpu().numpy().max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for timestep in range(autoregressive_steps):
        img_slice = truth_micro[0, 0, timestep, :, :, 8].cpu().detach().numpy()
        denoised_slice = prediction[0, 0, timestep, :, :, 8].cpu().detach().numpy()
        noised_slice = truth_macro[0, 0, timestep, :, :, 8].cpu().detach().numpy()

        # Plot the original image
        axes[timestep, 0].imshow(img_slice, cmap="jet", extent=[0, 4, 0, 10])
        axes[timestep, 0].set_title(f"truth {timestep})")
        axes[timestep, 0].axis("off")

        # Plohe denoised image
        axes[timestep, 1].imshow(
            denoised_slice, cmap="jet", interpolation="bicubic",norm=norm, extent=[0, 4, 0, 10]
        )
        axes[timestep, 1].set_title(f"prediction {timestep})")
        axes[timestep, 1].axis("off")

        # Plohe denoised image
        axes[timestep, 2].imshow(noised_slice, cmap="jet", extent=[0, 4, 0, 10])
        axes[timestep, 2].set_title(f"truth_macro {timestep})")
        axes[timestep, 2].axis("off")

        # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    print(f"{diff_config.autoregressive_steps=}")
    test_final(
        seq_config,
        diff_config,
        trainer,
        model,
        val_dataloader,
        down_sampler,
        up_sampler,
    )
