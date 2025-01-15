from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np

from config.diff_args_typed import DiffTypedArgs
from config.seq_args_typed import SeqTypedArgs
from diffusion_models.trainer import ImagenTrainer
from train_test_seq.train_seq import _plot_training_progress
from util.utils import save_loss


def train_diff(
    diff_args: DiffTypedArgs,
    seq_args: SeqTypedArgs,
    trainer: ImagenTrainer,
    data_loader,
):
    loss_list = []
    for epoch in range(diff_args.epoch_num):
        data_loader.dataset.load_new_file()
        down_sampler = torch.nn.Upsample(
            size=seq_args.coarse_dim, mode=seq_args.coarse_mode
        )
        for data in data_loader:
            volume_shape = data.shape
            break
        up_sampler = torch.nn.Upsample(
            size=[volume_shape[3], volume_shape[4], volume_shape[5]],
            mode=seq_args.coarse_mode,
        )
        model, loss = train_epoch(
            diff_args, seq_args, trainer, data_loader, down_sampler, up_sampler
        )
        if epoch % 10 == 0 and epoch > 0:
            # save_loss(diff_args, loss_list + [loss], epoch)
            model.save(
                path=os.path.join(
                    diff_args.model_save_path, "model_epoch_" + str(epoch)
                )
            )
        if epoch >= 1:
            if loss < min(loss_list):
                # save_loss(diff_args, loss_list + [loss], epoch)
                model.save(
                    path=os.path.join(diff_args.model_save_path, "best_model_sofar")
                )
                # np.savetxt(
                #     os.path.join(diff_args.model_save_path, "best_model_sofar_epoch"),
                #     np.ones(2) * epoch,
                # )
        loss_list.append(loss)
        _plot_training_progress(
            epoch + 1, loss_list, None, None, diff_args.model_save_path
        )
        print("finish training epoch {}".format(epoch))


def train_epoch(
    diff_args: DiffTypedArgs,
    seq_args: SeqTypedArgs,
    trainer: ImagenTrainer,
    data_loader,
    down_sampler,
    up_sampler,
):
    loss_epoch = []

    pbar = tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training", leave=True
    )
    for iteration, batch in pbar:
        batch = batch.to(diff_args.device).float()
        bsize, ntime, channels, D, H, W = batch.shape
        batch_coarse = down_sampler(batch.reshape([bsize * ntime, channels, D, H, W]))
        # Apply up-sampler and reshape back to original batch shape
        batch_coarse2fine = up_sampler(batch_coarse).reshape(batch.shape)
        # Reorder dimensions to [B x F x T x H x W x D]
        batch = batch.permute([0, 2, 1, 3, 4, 5])
        batch_coarse2fine = batch_coarse2fine.permute([0, 2, 1, 3, 4, 5])
        # Compute loss with trainer
        loss = trainer(
            batch, cond_images=batch_coarse2fine, unet_number=1, ignore_time=False
        )
        # Update trainer
        trainer.update(unet_number=1)
        loss_epoch.append(loss)
        pbar.set_postfix({"loss": sum(loss_epoch) / len(loss_epoch)})

    return trainer, sum(loss_epoch) / len(loss_epoch)
