import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from custom_models.spatiotemporal_transformer import SpatiotemporalTransformer
from data.data_bfs_preprocess import bfs_dataset
from transformer_test import process_tensor


def autoregressive_test(
    model,
    dataloader,
    device,
    save_dir,
    num_timesteps,
    down_sampler,
    num_autoregressive_steps=5,
    stride=5,
):
    """
    Test the model autoregressively and compare each step with the ground truth.

    Args:
        model: Trained model.
        dataloader: DataLoader for the test dataset.
        device: Computation device (CPU or GPU).
        save_dir: Directory to save the result images.
        num_autoregressive_steps: Number of autoregressive steps to perform.
        stride: Number of timesteps to skip before starting the autoregressive loop.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Create directory to save results
    os.makedirs(save_dir, exist_ok=True)
    pred_min, pred_max = float("inf"), float("-inf")
    label_min, label_max = float("inf"), float("-inf")
    error_min, error_max = float("inf"), float("-inf")
    all_results: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Skip timesteps based on stride
            if batch_idx % stride != 0:
                continue
            batch: torch.Tensor = batch.to(device)
            batch = process_tensor(batch, coarse_dim, down_sampler)
            input_data = batch[:, :num_timesteps].to(device)
            full_target_data = batch[:, num_timesteps:].to(device)
            autoregressive_input = input_data.clone()
            results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for step in range(num_autoregressive_steps):
                predicted: torch.Tensor = model(autoregressive_input)

                autoregressive_input = torch.cat(
                    (autoregressive_input[:, 1:], predicted.unsqueeze(1)), dim=1
                )

                target_data: torch.Tensor = full_target_data[:, step]
                predicted = predicted.reshape(3, *coarse_dim)
                target_data = target_data.reshape(3, *coarse_dim)
                error = torch.abs(predicted - target_data)

                # Compute error
                # Update global min/max for predictions, labels, and errors
                pred_min = min(pred_min, predicted.min().item())
                pred_max = max(pred_max, predicted.max().item())
                label_min = min(label_min, target_data.min().item())
                label_max = max(label_max, target_data.max().item())
                error_min = min(error_min, error.min().item())
                error_max = max(error_max, error.max().item())

                # Store results for plotting
                results.append(
                    (
                        predicted.squeeze().cpu(),
                        target_data.squeeze().cpu(),
                        error.squeeze().cpu(),
                    )
                )
            all_results.append(results)

    # Second pass: plot and save results
    z = 4
    for j, timesteps in enumerate(all_results):
        timestep_dir = os.path.join(save_dir, f"iteration_{j}")
        os.makedirs(timestep_dir, exist_ok=True)
        for direction_idx, direction in enumerate(["u", "v", "w"]):
            direction_dir = os.path.join(timestep_dir, direction)
            os.makedirs(direction_dir, exist_ok=True)
            for i, (predicted, target_data, error) in enumerate(timesteps):
                pred_img = predicted[direction_idx, :, :, z].numpy()
                label_img = target_data[direction_idx, :, :, z].numpy()
                error_img = error[direction_idx, :, :, z].numpy()

                # Plot and save with consistent colorbars
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                im0 = axes[0].imshow(
                    pred_img, cmap="viridis", vmin=label_min, vmax=label_max
                )
                axes[0].set_title("Predicted")
                axes[0].axis("off")

                im1 = axes[1].imshow(
                    label_img, cmap="viridis", vmin=label_min, vmax=label_max
                )
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                im2 = axes[2].imshow(
                    error_img, cmap="viridis", vmin=label_min, vmax=label_max
                )
                axes[2].set_title("Error")
                axes[2].axis("off")

                # Add colorbars
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                # Save the plot
                save_path = os.path.join(direction_dir, f"timestep_{i}.jpg")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)

        print(f"Results saved in {save_dir}")


if __name__ == "__main__":
    coarse_dim = (32, 32, 8)
    num_timesteps = 5
    num_autoregressive_steps = 39
    stride = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "test_models/best_model.pth"
    model = SpatiotemporalTransformer(
        volume_size=coarse_dim,
        num_timesteps=num_timesteps,
        d_model=2048,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
    ).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_data_location = ["./data/global_domain_float16_all.npy"]
    test_dataset = bfs_dataset(
        data_location=test_data_location,
        trajec_max_len=num_timesteps + num_autoregressive_steps + 1,
        n_span=1000,
        start_n=0,
        stride=20,
        mode="val",
    )
    print(len(test_dataset))
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    save_results_dir = "./test_results2"
    down_sampler = nn.Upsample(size=coarse_dim, mode="trilinear", align_corners=False)
    autoregressive_test(
        model,
        test_dataloader,
        device,
        save_results_dir,
        num_timesteps,
        down_sampler,
        num_autoregressive_steps,
        stride,
    )
