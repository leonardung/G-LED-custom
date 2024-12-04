import os
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data.data_bfs_preprocess import bfs_dataset
from transformer_test_2 import process_tensor, FlowPredictor
from transformer_test_2_medium import Transformer


def evaluate_and_plot(
    model: FlowPredictor,
    dataloader,
    down_sampler,
    coarse_dim,
    past_seq_len,
    future_seq_len,
    z_index,
    stride,
    output_dir,
    device,
):
    """
    Evaluate the model autoregressively and plot all the timestamps.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % stride != 0:
                continue  # Skip batches according to stride

            batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
            os.makedirs(batch_dir, exist_ok=True)

            batch: torch.Tensor = batch.to(device)
            batch = process_tensor(batch, coarse_dim, down_sampler)

            input_data = batch[:, :past_seq_len]  # [1, past_seq_len, input_dim]
            i = 0
            actual_future = batch[
                :, past_seq_len + i : past_seq_len + i + 1
            ]  # [1, future_seq_len, input_dim]

            # Initialize the target sequence with zeros
            tgt_input = torch.zeros(1, 1, model.input_dim).to(device)

            # List to store the predictions
            predictions = []

            # Forward pass
            output = model(src=input_data, tgt=tgt_input)
            print(f"{output.shape=}")
            print(f"{actual_future.shape=}")
            # for i in range(output.shape[1]):
            #     predictions.append(output[:,i,:])

            # Concatenate predictions along time dimension
            # predictions = torch.cat(
            #     output, dim=1
            # )  # [1, future_seq_len, input_dim]
            predictions = output
            # Compute the error
            print(f"{predictions.shape=}")
            print(f"{actual_future.shape=}")
            error = predictions - actual_future  # [1, future_seq_len, input_dim]
            error = error.cpu().numpy()

            # Compute mean error along the autoregressive steps
            mean_error = np.mean(np.square(error), axis=(2))  # [future_seq_len]
            mean_error = mean_error.squeeze()

            # Save the mean error plot
            # plt.figure()
            # plt.plot(range(future_seq_len), mean_error)
            # plt.xlabel("Time Step")
            # plt.ylabel("Mean Squared Error")
            # plt.title("Mean Error along Autoregressive Steps")
            # plt.savefig(os.path.join(batch_dir, "mean_error.png"))
            # plt.close()

            # For each timestep, plot predicted, actual, and error
            for t in range(1):
                pred = predictions[0, t].cpu().numpy()
                actual = actual_future[0, t].cpu().numpy()
                err = error[0, t]

                # Reshape to [num_velocity, coarse_dim[0], coarse_dim[1], coarse_dim[2]]
                pred = pred.reshape(3, *coarse_dim)
                actual = actual.reshape(3, *coarse_dim)
                err = err.reshape(3, *coarse_dim)

                # Extract the z plane
                pred_slice = pred[:, :, :, z_index]  # [3, coarse_dim[0], coarse_dim[1]]
                actual_slice = actual[:, :, :, z_index]
                err_slice = err[:, :, :, z_index]

                # Find the global min and max values for scaling
                vmin = min(pred_slice.min(), actual_slice.min())
                vmax = max(pred_slice.max(), actual_slice.max())

                # Plot for each velocity component
                for v in range(3):
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(
                        pred_slice[v],
                        vmin=vmin,
                        vmax=vmax,
                        cmap="viridis",
                        origin="lower",
                    )
                    plt.title(f"Predicted Velocity Component {v}")
                    plt.colorbar()

                    plt.subplot(1, 3, 2)
                    plt.imshow(
                        actual_slice[v],
                        vmin=vmin,
                        vmax=vmax,
                        cmap="viridis",
                        origin="lower",
                    )
                    plt.title(f"Actual Velocity Component {v}")
                    plt.colorbar()

                    plt.subplot(1, 3, 3)
                    plt.imshow(err_slice[v], cmap="viridis", origin="lower")
                    plt.title(f"Error Velocity Component {v}")
                    plt.colorbar()

                    plt.suptitle(f"Time Step {t}, Velocity Component {v}")
                    plt.savefig(os.path.join(batch_dir, f"time_{t}_velocity_{v}.png"))
                    plt.close()
            break


if __name__ == "__main__":
    coarse_dim = (15, 15, 4)
    past_seq_len = 10
    future_seq_len = 30
    batch_size = 2  # For training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    data_location = ["./data/test_data.npy"]  # Replace with your actual path
    down_sampler = nn.Upsample(size=coarse_dim, mode="trilinear", align_corners=False)
    input_dim = 3 * coarse_dim[0] * coarse_dim[1] * coarse_dim[2]
    model = Transformer(
        input_dim=input_dim,
        dim_model=512,
        num_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout_p=0.05,
    ).to(device)
    model_save_path = "test_models/model.pth"
    model.load_state_dict(torch.load(model_save_path))
    eval_dataset = bfs_dataset(
        data_location=data_location,
        trajec_max_len=past_seq_len + future_seq_len + 1,
        start_n=500,
        n_span=45,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Define z_index and stride
    z_index = 2  # Can be any integer between 0 and coarse_dim[2]-1
    stride = 1  # Adjust as needed
    output_dir = "evaluation_plots"

    evaluate_and_plot(
        model=model,
        dataloader=eval_dataloader,
        down_sampler=down_sampler,
        coarse_dim=coarse_dim,
        past_seq_len=past_seq_len,
        future_seq_len=future_seq_len,
        z_index=z_index,
        stride=stride,
        output_dir=output_dir,
        device=device,
    )
