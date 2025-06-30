import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model_registry import register_models, get_model
from dataset_registry import register_datasets, get_dataset
from train_test_seq.test_seq_vit import autoregressive_test

# from train_test_seq.test_seq import autoregressive_test
from util.utils import load_config

register_datasets()
register_models()


if __name__ == "__main__":
    training_path = "bfs_dataset_split_2d_2025_06_25_04_34_40_vit_64x32_autoreg_20"
    dir_output = "/shared/data1/Users/l1148900/G-LED/output/"
    model_name = "best_model.pth"
    autoregressive_steps = 80
    stride = 400
    is_valid = True

    model_path = os.path.join(dir_output, training_path, "model_save", model_name)
    config = load_config(os.path.join(dir_output, training_path, "logging/args.yml"))
    config.autoregressive_steps_valid = autoregressive_steps
    config.max_autoregressive_steps = autoregressive_steps
    config.trajec_max_len_valid = (
        config.num_timesteps + config.autoregressive_steps_valid + 1
    )
    config.trajec_max_len = config.num_timesteps + config.max_autoregressive_steps + 1

    coarse_dim = config.coarse_dim
    num_timesteps = config.num_timesteps
    num_autoregressive_steps = (
        config.autoregressive_steps_valid
        if is_valid
        else config.max_autoregressive_steps
    )
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    frame_patch_size = 4
    heads = 16
    dim_head = 512
    model: torch.nn.Module = get_model(
        config.model_name,
        **{
            "config": config,
            # "frame_patch_size": frame_patch_size,
            # "heads": heads,
            # "dim_head": dim_head,
        },
    ).to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_dataset = get_dataset(
        config.dataset, **{"config": config, "mode": "valid" if is_valid else "train"}
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    down_sampler = nn.Upsample(
        size=coarse_dim, mode=config.coarse_mode, align_corners=False
    )
    autoregressive_test(
        model=model,
        dataloader=test_dataloader,
        device=device,
        save_dir=config.logging_path,
        num_timesteps=num_timesteps,
        down_sampler=down_sampler,
        coarse_dim=config.coarse_dim,
        is_flatten=config.is_flatten,
        num_autoregressive_steps=num_autoregressive_steps,
        num_velocities=config.n_velocities,
        stride=stride,
    )
