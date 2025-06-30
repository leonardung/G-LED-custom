import os
from pathlib import Path
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
from train_test_diff.train_diff import train_diff
from util.utils import load_config, save_args

register_datasets()
register_models()


if __name__ == "__main__":
    diff_config_path = "config/diff_3d_flow.yml"
    diff_config: DiffTypedArgs = load_config(diff_config_path)
    diff_config.logging_path = str(
        Path(diff_config.dir_output)
        / diff_config.training_path
        / Path("diffusion/logging")
    )
    diff_config.model_save_path = str(
        Path(diff_config.dir_output)
        / diff_config.training_path
        / Path("diffusion/model_save")
    )
    os.makedirs(diff_config.logging_path, exist_ok=True)
    os.makedirs(diff_config.model_save_path, exist_ok=True)

    save_args(diff_config)
    diff_config.device = torch.device(diff_config.device)

    seq_config: SeqTypedArgs = load_config(
        os.path.join(
            diff_config.dir_output, diff_config.training_path, "logging/args.yml"
        )
    )
    seq_config.batch_size = 1
    seq_config.batch_size_valid = 1

    dataset = get_dataset(seq_config.dataset, **{"config": seq_config, "mode": "train"})
    dataloader = DataLoader(
        dataset,
        batch_size=seq_config.batch_size,
        shuffle=seq_config.shuffle,
        num_workers=0,
    )

    # val_dataset = get_dataset(
    #     seq_config.dataset, **{"config": seq_config, "mode": "val"}
    # )
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=seq_config.batch_size_valid,
    #     shuffle=False,
    #     num_workers=0,
    # )
    print(f"{len(dataset) = }")
    print(f"{len(dataloader) = }")
    # print(f"{len(val_dataset) = }")
    # print(f"{len(val_dataloader) = }")

    unet1 = Unet3D(
        dim=diff_config.unet_dim,
        cond_images_channels=seq_config.n_velocities,
        memory_efficient=True,
        dim_mults=(1, 2, 4, 8),
    ).to(diff_config.device)
    image_dim = list(next(iter(dataloader)).shape[3:])
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
    try:
        trainer.load(path=diff_config.model_save_path + "/best_model_sofar")
    except:
        print(f"Model: '{diff_config.model_save_path}/best_model_sofar' does not exist")
    train_diff(
        diff_args=diff_config,
        seq_args=seq_config,
        trainer=trainer,
        data_loader=dataloader,
    )
