import os
from typing import Callable
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from config.seq_args_typed import SeqTypedArgs

from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    current_autoregressive_steps: int,
    num_timesteps: int,
    coarse_dim: tuple[int, int, int],
    down_sampler: Callable,
    is_flatten: bool = True,
    iter_per_epoch: int = 1000,
):
    model.train()
    total_loss = 0.0
    total_length = min(len(dataloader), iter_per_epoch)
    with tqdm(total=total_length, desc="Training", unit="batch") as pbar:
        for iteration, batch in enumerate(dataloader):
            if iteration >= iter_per_epoch:
                break
            batch: torch.Tensor = batch.to(device)
            batch = _process_tensor(batch, coarse_dim, down_sampler, is_flatten)
            input_data = batch[:, :num_timesteps]
            target_data = batch[
                :, num_timesteps : num_timesteps + current_autoregressive_steps
            ]

            current_input = input_data.clone()

            for step in range(current_autoregressive_steps):
                optimizer.zero_grad()
                # output: torch.Tensor = model(current_input)[0][:, -1]
                # target = target_data[:, step]
                # loss: torch.Tensor = criterion(output, target)

                # output = output.unsqueeze(1)  # Add time dimension
                # current_input = torch.cat([current_input[:, 1:], output.detach()], dim=1)

                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # optimizer.step()
                # total_loss += loss.item()

                xn = batch[:, step : step + num_timesteps, :]
                xnp1, _, _, _ = model(inputs_embeds=xn, past=None)
                xn_label = batch[:, step + 1 : step + 1 + num_timesteps, :]
                loss: torch.Tensor = criterion(xnp1, xn_label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / ((iteration + 1) * current_autoregressive_steps)
            pbar.set_postfix({"loss": f"{avg_loss:.3e}"})
            pbar.update(1)

    avg_loss = total_loss / total_length / current_autoregressive_steps
    return avg_loss


def validate_model(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: str,
    criterion: torch.nn.Module,
    coarse_dim: tuple[int, int, int],
    num_timesteps: int,
    current_autoregressive_steps: int,
    max_autoregressive_steps: int,
    loss_threshold: float,
    down_sampler: str,
    is_flatten: bool = True,
):
    model.eval()

    # Pause training and increase autoregressive steps until avg_val_loss > loss_threshold
    while True:
        total_val_loss = 0.0
        val_dataloader_length_without_nan=0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = val_batch.to(device)
                val_batch = _process_tensor(
                    val_batch, coarse_dim, down_sampler, is_flatten
                )
                input_data = val_batch[:, :num_timesteps]
                target_data = val_batch[
                    :,
                    num_timesteps : num_timesteps + current_autoregressive_steps,
                ]
                current_input = input_data.clone()

                total_batch_loss = 0.0
                for step in range(current_autoregressive_steps):
                    # output: torch.Tensor = model(current_input)[0][:, -1]
                    # target = target_data[:, step]
                    # val_loss = criterion(output, target)
                    # total_batch_loss += val_loss.item()

                    # # Update current_input
                    # output = output.unsqueeze(1)
                    # current_input = torch.cat([current_input[:, 1:], output], dim=1)

                    xn = val_batch[:, step : step + num_timesteps, :]
                    xnp1, _, _, _ = model(inputs_embeds=xn, past=None)
                    xn_label = val_batch[:, step + 1 : step + 1 + num_timesteps, :]
                    val_loss: torch.Tensor = criterion(xnp1, xn_label)
                    total_batch_loss += val_loss.item()
                avg_val_batch_loss = total_batch_loss / current_autoregressive_steps
                if not np.isnan(avg_val_batch_loss):
                    total_val_loss += avg_val_batch_loss
                    val_dataloader_length_without_nan += 1
        avg_val_loss = total_val_loss / val_dataloader_length_without_nan
        
        tqdm.write(
            f"Validation Loss with {current_autoregressive_steps} autoregressive steps: {avg_val_loss:.3e}"
        )

        if avg_val_loss > loss_threshold:
            tqdm.write(
                f"Validation loss {avg_val_loss:.3e} exceeded threshold {loss_threshold}."
            )
            break
        elif current_autoregressive_steps >= max_autoregressive_steps:
            tqdm.write(f"Max autoregressive steps reached: {max_autoregressive_steps}")
            break
        else:
            current_autoregressive_steps += 1
            tqdm.write(
                f"Validation loss {avg_val_loss:.3e} below threshold {loss_threshold}, increasing autoregressive steps to {current_autoregressive_steps}"
            )
            break # only increase once every validation

    return avg_val_loss, current_autoregressive_steps


def train_model(
    config: SeqTypedArgs,
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):
    log_dir = os.path.join(config.model_save_path, "runs")
    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    lrs = []
    avg_val_loss = np.nan
    down_sampler = torch.nn.Upsample(
        size=config.coarse_dim, mode=config.coarse_mode, align_corners=False
    )
    current_autoregressive_steps = config.start_autoregressive_steps
    for epoch in tqdm(range(config.epoch_num), desc="Training Progress", leave=True):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            device=config.device,
            optimizer=optimizer,
            criterion=criterion,
            current_autoregressive_steps=current_autoregressive_steps,
            num_timesteps=config.num_timesteps,
            coarse_dim=config.coarse_dim,
            down_sampler=down_sampler,
            is_flatten=config.is_flatten,
            iter_per_epoch=config.iter_per_epoch,
        )

        # dataloader.dataset.load_new_file()
        # Compute validation loss every val_interval epochs (after some warmup)
        if epoch % config.epoch_valid_interval == 0 and epoch > 0:
            dataloader.dataset.load_new_file()
            avg_val_loss, current_autoregressive_steps = validate_model(
                model=model,
                val_dataloader=val_dataloader,
                device=config.device,
                criterion=criterion,
                coarse_dim=config.coarse_dim,
                num_timesteps=config.num_timesteps,
                current_autoregressive_steps=current_autoregressive_steps,
                max_autoregressive_steps=config.max_autoregressive_steps,
                loss_threshold=config.march_loss_threshold,
                down_sampler=down_sampler,
                is_flatten=config.is_flatten,
            )
            writer.add_scalar("Loss/val", avg_val_loss, epoch)

            # Adjust learning rate and save best model
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = f"{config.model_save_path}/best_model.pth"
                torch.save(model.state_dict(), model_save_path)
                tqdm.write(
                    f"Best model saved with validation loss {best_val_loss:.3e} to {model_save_path}"
                )
        if epoch % config.epoch_save_interval == 0 and epoch > 0:
            final_model_path = f"{config.model_save_path}/model_{epoch}.pth"
            torch.save(model.state_dict(), final_model_path)
            print(f"Model saved to {final_model_path}")
        val_losses.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        train_losses.append(avg_loss)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # _plot_training_progress(
        #     epoch + 1, train_losses, val_losses, lrs, config.model_save_path
        # )

    # Save the final model
    final_model_path = f"{config.model_save_path}/model_{epoch}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")
    writer.close()


def _plot_training_progress(epoch, train_losses, val_losses, lrs, save_dir):
    plt.figure(figsize=(10, 6))

    if train_losses is not None:
        plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
    if val_losses is not None:
        plt.plot(range(1, epoch + 1), val_losses, label="Validation Loss")

    ax1 = plt.gca()
    if lrs is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            range(1, epoch + 1),
            lrs,
            label="Learning Rate",
            color="green",
            linestyle="--",
        )
        ax2.set_yscale("log")
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="upper right")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss.png")
    plt.close()


def _plot_input_target_pred(
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


def _process_tensor(
    tensor: torch.Tensor,
    coarse_dim: tuple[int, int, int],
    down_sampler: Callable,
    is_flatten: bool = True,
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
    tensor_squeezed = (
        tensor.float()
        .reshape(
            batch_size * seq_len,
            num_velocity,
            tensor.shape[3],
            tensor.shape[4],
            tensor.shape[5] if len(tensor.shape) == 6 else 1,
        )
        .squeeze(-1)
    )
    tensor_coarse: torch.Tensor = down_sampler(tensor_squeezed)

    # Reshape back and flatten
    if len(coarse_dim) == 2:
        coarse_dim += (1,)
    tensor_coarse = tensor_coarse.reshape(
        batch_size,
        seq_len,
        num_velocity,
        coarse_dim[0],
        coarse_dim[1],
        coarse_dim[2],
    )
    if not is_flatten:
        return tensor_coarse
    tensor_coarse_flatten = tensor_coarse.reshape(
        batch_size,
        seq_len,
        num_velocity * coarse_dim[0] * coarse_dim[1] * coarse_dim[2],
    )
    return tensor_coarse_flatten
