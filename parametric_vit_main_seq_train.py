import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from train_test_seq.train_seq_vit import train_model
from util.utils import load_config, save_args, update_args
from dataset_registry import register_datasets, get_dataset
from model_registry import register_models, get_model

register_datasets()
register_models()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train parametric ViT on one GPU")
    p.add_argument(
        "--frame_patch_size",
        type=int,
        default=1,
        help="Size of the temporal patching",
    )
    p.add_argument("--dim", type=int, default=1024, help="Dimension")
    p.add_argument("--depth", type=int, default=6, help="Depth")
    p.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    p.add_argument(
        "--dim_head", type=int, default=64, help="Dimension of each attention head"
    )
    p.add_argument(
        "--num_timesteps", type=int, default=8, help="Dimension of each attention head"
    )
    p.add_argument(
        "--patch_dim",
        type=str,
        default="8,4,4",
        help="Size of the dimension of the volume patching",
    )
    p.add_argument("--gpu", type=int, default=0, help="which CUDA device to use")
    args = p.parse_args()

    patch_dim1, patch_dim2, patch_dim3 = map(int, args.patch_dim.split(","))
    print(
        f"****** Running with {args.frame_patch_size=}, {args.heads=}, {args.dim_head=}, "
        f"num_timesteps={args.num_timesteps}, dim={args.dim}, depth={args.depth}, "
        f"patch_dim={patch_dim1}x{patch_dim2}x{patch_dim3} ******"
    )
    config_path = "config/seq_3d_flow_vit.yml"
    config = load_config(config_path)
    config.device = f"cuda:{args.gpu}"

    config.run_name_postfix = (
        f"vit_64x32x32_parametric_list_framePatchDim={args.frame_patch_size}_"
        f"heads={args.heads}_dimHead={args.dim_head}_numTimesteps={args.num_timesteps}_"
        f"patchDim={patch_dim1}x{patch_dim2}x{patch_dim3}_"
        f"_dim={args.dim}_depth={args.depth}"
    )
    config = update_args(config)

    ### Overwrite config with command line args
    config.num_timesteps = args.num_timesteps

    save_args(config)

    dataset = get_dataset(config.dataset, **{"config": config, "mode": "train"})
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=0
    )

    val_dataset = get_dataset(config.dataset, **{"config": config, "mode": "val"})
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_valid,
        shuffle=False,
        num_workers=0,
    )
    print(f"{len(dataset) = }")
    print(f"{len(dataloader) = }")
    print(f"{len(val_dataset) = }")
    print(f"{len(val_dataloader) = }")
    model: torch.nn.Module = get_model(
        config.model_name,
        **{
            "config": config,
            "dim": args.dim,
            "depth": args.depth,
            "volume_patch_size": (
                patch_dim1,
                patch_dim2,
                patch_dim3,
            ),
            "frame_patch_size": args.frame_patch_size,
            "heads": args.heads,
            "dim_head": args.dim_head,
        },
    ).to(config.device)
    try:
        model.load_state_dict(torch.load(config.previous_model_name))
    except:
        print(f"No checkpoint found")

    # Define the model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_decay_factor,
        patience=config.lr_decay_patience,
    )

    # Run the training loop
    train_model(
        config=config,
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
