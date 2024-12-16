import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from train_test_seq.train_seq import _process_tensor
import cv2

def autoregressive_test(
    model,
    dataloader,
    device,
    save_dir,
    num_timesteps,
    down_sampler,
    coarse_dim,
    num_autoregressive_steps=5,
    stride=5,
    num_velocities=3,
    z=None,
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
            batch = _process_tensor(batch, coarse_dim, down_sampler)
            input_data = batch[:, :num_timesteps].to(device)
            full_target_data = batch[:, num_timesteps:].to(device)
            autoregressive_input = input_data.clone()
            results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for step in range(num_autoregressive_steps):
                predicted: torch.Tensor = model(autoregressive_input)[0][:, -1]

                autoregressive_input = torch.cat(
                    (autoregressive_input[:, 1:], predicted.unsqueeze(1)), dim=1
                )

                target_data: torch.Tensor = full_target_data[:, step]
                predicted = predicted.reshape(num_velocities, *coarse_dim)
                target_data = target_data.reshape(num_velocities, *coarse_dim)
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
                        predicted.cpu(),
                        target_data.cpu(),
                        error.cpu(),
                    )
                )
            all_results.append(results)

    # Second pass: plot and save results
    z = coarse_dim[-1] // 2 - 1 if z is None else z
    avg_rmse_evolution = {direction: [] for direction in ["u", "v", "w"][:num_velocities]}
    for j, timesteps in enumerate(all_results):
        timestep_dir = os.path.join(save_dir, f"iteration_{j}")
        os.makedirs(timestep_dir, exist_ok=True)

        # Reset error evolution for this batch
        error_evolution = {direction: [] for direction in ["u", "v", "w"][:num_velocities]}

        for direction_idx, direction in enumerate(["u", "v", "w"][:num_velocities]):
            direction_dir = os.path.join(timestep_dir, direction)
            os.makedirs(direction_dir, exist_ok=True)

            images_for_video = []

            for i, (predicted, target_data, error) in enumerate(timesteps):
                if len(coarse_dim) == 3:
                    pred_img = predicted[direction_idx, :, :, z].numpy()
                    label_img = target_data[direction_idx, :, :, z].numpy()
                    error_img = error[direction_idx, :, :, z].numpy()
                else:
                    pred_img = predicted[direction_idx, :, :].numpy()
                    label_img = target_data[direction_idx, :, :].numpy()
                    error_img = error[direction_idx, :, :].numpy()

                # Compute RMSE for the error plot
                rmse = np.sqrt(np.mean(error_img ** 2))

                # Append mean error to the evolution list for this batch
                error_evolution[direction].append(rmse)

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
                axes[2].set_title(f"Error (RMSE: {rmse:.4f})")
                axes[2].axis("off")

                # Add colorbars
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                # Save the plot
                save_path = os.path.join(direction_dir, f"timestep_{i}.jpg")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)

                # Append to video frames
                images_for_video.append(cv2.imread(save_path))

            # Create video from images
            video_path = os.path.join(direction_dir, "evolution.avi")
            height, width, _ = images_for_video[0].shape
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), 10, (width, height))
            for image in images_for_video:
                video_writer.write(image)
            video_writer.release()


            # Generate and save the line plot for error evolution of this batch
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(error_evolution[direction])), error_evolution[direction], marker="o")
            ax.set_title(f"Error Evolution for Direction {direction} (Batch {j})")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Root Mean Squared Error")
            ax.grid(True)

            # Save the line plot
            lineplot_path = os.path.join(direction_dir, "error_evolution.jpg")
            plt.savefig(lineplot_path, bbox_inches="tight")
            plt.close(fig)

            # Compute average RMSE across all batches for each timestep
            if len(avg_rmse_evolution[direction]) < len(error_evolution[direction]):
                avg_rmse_evolution[direction].extend([0] * (len(error_evolution[direction]) - len(avg_rmse_evolution[direction])))
            for t in range(len(error_evolution[direction])):
                avg_rmse_evolution[direction][t] += error_evolution[direction][t]

    # Finalize and save average RMSE evolution plots for all batches
    for direction in avg_rmse_evolution:
        avg_rmse_evolution[direction] = [rmse / len(all_results) for rmse in avg_rmse_evolution[direction]]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(avg_rmse_evolution[direction])), avg_rmse_evolution[direction], marker="o")
        ax.set_title(f"Average RMSE Evolution for Direction {direction}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Average Root Mean Squared Error")
        ax.grid(True)

        # Save the average RMSE evolution plot
        avg_plot_path = os.path.join(save_dir, f"avg_error_evolution_{direction}.jpg")
        plt.savefig(avg_plot_path, bbox_inches="tight")
        plt.close(fig)



        print(f"Results saved in {save_dir}")