from datetime import datetime
import os
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

from data.data_bfs_preprocess import bfs_dataset
from transformer.sequentialModel import SequentialModel
from config.seq_args_typed import TypedArgs
from util.utils import load_config, save_args


def update_args(args: TypedArgs) -> TypedArgs:
    args.time = "{0:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    # output dataset
    args.dir_output = "output/"
    args.fname = args.dataset + "_" + args.time
    args.experiment_path = args.dir_output + args.fname
    args.model_save_path = os.path.join(args.experiment_path, "model_save/")
    args.logging_path = os.path.join(args.experiment_path, "logging/")
    args.current_model_save_path = args.model_save_path
    args.logging_epoch_path = os.path.join(args.logging_path, "epoch_history.csv")
    os.makedirs(args.logging_path, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def plot_training_progress(epoch, train_losses, val_losses, lrs, save_dir):
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epoch + 1), val_losses, label="Validation Loss")

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(
        range(1, epoch + 1),
        lrs,
        label="Learning Rate",
        color="green",
        linestyle="--",
    )
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Learning Rate")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss.png")
    plt.close()


def plot_input_target_pred(
    current_input: torch.Tensor, target: torch.Tensor, output: torch.Tensor, coarse_dim
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(
        current_input.squeeze()
        .cpu()
        .detach()
        .numpy()[-1]
        .reshape(3, *coarse_dim)[0, :, :, 5],
        cmap="viridis",
    )
    im1 = axes[1].imshow(
        target.squeeze().reshape(3, *coarse_dim)[0, :, :, 5].cpu().detach().numpy(),
        cmap="viridis",
    )
    im2 = axes[2].imshow(
        output.squeeze().reshape(3, *coarse_dim)[0, :, :, 5].cpu().detach().numpy(),
        cmap="viridis",
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.show()


def process_tensor(
    tensor: torch.Tensor, coarse_dim: tuple[int, int, int], down_sampler: Callable
) -> torch.Tensor:
    """
    Process the input tensor by reshaping, downsampling, and flattening.
    Args:
        tensor: [batch_size, seq_len, num_velocity, dim1, dim2, dim3]
        seq_len: Sequence length of the tensor
    Returns:
        Flattened tensor: [batch_size, seq_len, num_velocity * coarse_dim[0] * coarse_dim[1] * coarse_dim[2]]
    """
    batch_size = tensor.shape[0]
    seq_len = tensor.shape[1]
    num_velocity = tensor.shape[2]

    # Squeeze and downsample
    tensor_squeezed = tensor.float().reshape(
        batch_size * seq_len,
        num_velocity,
        tensor.shape[3],
        tensor.shape[4],
        tensor.shape[5],
    )
    tensor_coarse: torch.Tensor = down_sampler(tensor_squeezed)

    # Reshape back and flatten
    tensor_coarse = tensor_coarse.reshape(
        batch_size,
        seq_len,
        num_velocity,
        coarse_dim[0],
        coarse_dim[1],
        coarse_dim[2],
    )
    tensor_coarse_flatten = tensor_coarse.reshape(
        batch_size,
        seq_len,
        num_velocity * coarse_dim[0] * coarse_dim[1] * coarse_dim[2],
    )
    return tensor_coarse_flatten


if __name__ == "__main__":
    config_path = "config/seq_new.yml"
    config = load_config(config_path)
    # config = update_args(config)
    # save_args(config)

    save_dir = "test_models"
    coarse_dim = (16, 20, 4)
    config.n_embd = 3 * coarse_dim[0] * coarse_dim[1] * coarse_dim[2]
    num_timesteps = 9
    config.n_ctx = num_timesteps
    batch_size = 1
    val_interval = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters for autoregressive steps and loss threshold
    max_autoregressive_steps = 4
    loss_threshold = 0.0035
    current_autoregressive_steps = 2

    # Initialize model
    data_location = ["./data/global_domain_float16_all.npy"]
    trajec_max_len = num_timesteps + max_autoregressive_steps + 1
    dataset = bfs_dataset(
        data_location=data_location,
        trajec_max_len=trajec_max_len,
        n_span=650,
        start_n=0,
        stride=list(range(10, 30)),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = bfs_dataset(
        data_location=data_location,
        trajec_max_len=trajec_max_len,
        n_span=350,
        start_n=650,
        stride=20,
    )
    print(len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SequentialModel(config).to(device)
    model_save_path = f"{save_dir}/best_model.pth"
    model.load_state_dict(torch.load(model_save_path))

    # Define the model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")

    # Training loop
    num_epochs = 200
    train_losses = []
    val_losses = []
    lrs = []
    down_sampler = nn.Upsample(size=coarse_dim, mode="trilinear", align_corners=False)
    avg_val_loss = np.nan
    for epoch in tqdm(range(num_epochs), desc="Training Progress", leave=True):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch: torch.Tensor = batch.to(device)
            batch = process_tensor(batch, coarse_dim, down_sampler)
            input_data = batch[:, :num_timesteps]
            target_data = batch[
                :, num_timesteps : num_timesteps + current_autoregressive_steps
            ]

            current_input = input_data.clone()

            for step in range(current_autoregressive_steps):
                optimizer.zero_grad()
                output: torch.Tensor = model(current_input)[0][:, -1]
                target = target_data[:, step]
                loss: torch.Tensor = criterion(output, target)

                output = output.unsqueeze(1)  # Add time dimension
                current_input = torch.cat([current_input[:, 1:], output], dim=1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.3e}")

        # Compute validation loss every val_interval epochs
        if epoch % val_interval == 0 and epoch > 0:
            model.eval()

            # Pause training and increase autoregressive steps until avg_val_loss > loss_threshold
            while True:
                total_val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = val_batch.to(device)
                        val_batch = process_tensor(val_batch, coarse_dim, down_sampler)
                        input_data = val_batch[:, :num_timesteps]
                        target_data = val_batch[
                            :,
                            num_timesteps : num_timesteps
                            + current_autoregressive_steps,
                        ]

                        total_batch_loss = 0.0
                        current_input = input_data.clone()

                        for step in range(current_autoregressive_steps):
                            output: torch.Tensor = model(current_input)[0][:, -1]
                            target = target_data[:, step]

                            # plot_input_target_pred(current_input, target, output, coarse_dim)
                            val_loss = criterion(output, target)
                            total_batch_loss += val_loss.item()

                            # Update current_input
                            output = output.unsqueeze(1)
                            current_input = torch.cat(
                                [current_input[:, 1:], output], dim=1
                            )

                        avg_val_batch_loss = (
                            total_batch_loss / current_autoregressive_steps
                        )
                        total_val_loss += avg_val_batch_loss

                avg_val_loss = total_val_loss / len(val_dataloader)
                tqdm.write(
                    f"Validation Loss with {current_autoregressive_steps} autoregressive steps: {avg_val_loss:.3e}"
                )

                if avg_val_loss > loss_threshold:
                    tqdm.write(
                        f"Validation loss {avg_val_loss:.3e} exceeded threshold {loss_threshold}."
                    )
                    break

                elif current_autoregressive_steps >= max_autoregressive_steps:
                    tqdm.write(
                        f"Max autoregressive steps reach: {max_autoregressive_steps}"
                    )
                    break
                else:
                    current_autoregressive_steps += 1
                    tqdm.write(
                        f"Validation loss {avg_val_loss:.3e} below threshold {loss_threshold}, increasing autoregressive steps to {current_autoregressive_steps}"
                    )

            # Save the model if validation loss has improved
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(f"{save_dir}", exist_ok=True)
                model_save_path = f"{save_dir}/best_model.pth"
                torch.save(model.state_dict(), model_save_path)
                tqdm.write(
                    f"Best model saved with validation loss {best_val_loss:.3e} to {model_save_path}"
                )
        val_losses.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        train_losses.append(avg_loss)
        plot_training_progress(epoch + 1, train_losses, val_losses, lrs, save_dir)

    # Save the final model
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = f"{save_dir}/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
