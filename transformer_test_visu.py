import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.data_bfs_preprocess import bfs_dataset
from transformer_test import SpatiotemporalTransformer


def test_and_visualize(model, dataloader, device, save_dir):
    """
    Test the model and save plots of predictions, ground truth, and errors with consistent colorbars.

    Args:
        model: Trained model.
        dataloader: DataLoader for the test dataset.
        device: Computation device (CPU or GPU).
        save_dir: Directory to save the result images.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Create directory to save results
    os.makedirs(save_dir, exist_ok=True)

    # Initialize variables for global min and max
    pred_min, pred_max = float('inf'), float('-inf')
    label_min, label_max = float('inf'), float('-inf')
    error_min, error_max = float('inf'), float('-inf')

    # Storage for predictions, labels, and errors
    all_results = []

    # First pass: compute global min/max and collect data
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.float()
            input_data = batch[:, :-1].to(device)  # All but the last timestep
            target_data = batch[:, -1].to(device)  # The last timestep is the target

            # Forward pass
            predicted = model(input_data)

            # Calculate error
            error = torch.abs(predicted - target_data)

            # Update global min/max for predictions, labels, and errors
            pred_min = min(pred_min, predicted.min().item())
            pred_max = max(pred_max, predicted.max().item())
            label_min = min(label_min, target_data.min().item())
            label_max = max(label_max, target_data.max().item())
            error_min = min(error_min, error.min().item())
            error_max = max(error_max, error.max().item())

            # Store results for plotting
            all_results.append((predicted.cpu(), target_data.cpu(), error.cpu()))

    # Second pass: plot and save results
    for i, (predicted, target_data, error) in enumerate(all_results):
        for b in range(predicted.size(0)):  # Iterate over batch
            direction = 0
            pred_img = predicted[b][direction, :, :, 1].numpy()
            label_img = target_data[b][direction, :, :, 1].numpy()
            error_img = error[b][direction, :, :, 1].numpy()

            # Plot and save with consistent colorbars
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            im0 = axes[0].imshow(pred_img, cmap="viridis", vmin=label_min, vmax=label_max)
            axes[0].set_title("Predicted")
            axes[0].axis("off")

            im1 = axes[1].imshow(label_img, cmap="viridis", vmin=label_min, vmax=label_max)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            im2 = axes[2].imshow(error_img, cmap="viridis", vmin=label_min, vmax=label_max)
            axes[2].set_title("Error")
            axes[2].axis("off")

            # Add colorbars
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            # Save the plot
            save_path = os.path.join(save_dir, f"test_{i}_batch_{b}.jpg")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

    print(f"Results saved in {save_dir}")



# Example usage
if __name__ == "__main__":
    # Test DataLoader
    volume_size = (20, 20, 3)
    num_timesteps = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "test_models/model.pth"
    model = (
        SpatiotemporalTransformer(volume_size=volume_size, num_timesteps=num_timesteps)
        .to(device)
        .float()
    )
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_data_location = ["./data/test_data.npy"]  
    test_dataset = bfs_dataset(
        data_location=test_data_location,
        trajec_max_len=num_timesteps + 1,
        start_n=500,
        n_span=100,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0
    )

    save_results_dir = "./test_results"

    test_and_visualize(model, test_dataloader, device, save_results_dir)
