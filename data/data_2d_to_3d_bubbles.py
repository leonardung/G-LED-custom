import random
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter

from config.seq_args_typed import TypedArgs


class data_2d_to_3d_bubbles(Dataset):
    def __init__(self, config: TypedArgs, mode: str = "train"):
        self.config=config
        if mode=="train":
            self.n_span = config.n_span
            self.trajec_max_len = config.trajec_max_len
        else:
            self.n_span = config.n_span_valid
            self.trajec_max_len = config.trajec_max_len_valid
        # data = np.load("data/3d_bubble_flow.npy")[
        #     start_n : start_n + n_span
        # ]  # .astype(np.float32)
        # data_u = gaussian_filter(data[:, 0], sigma=1, axes=(1, 2, 3))
        # data_v = gaussian_filter(data[:, 1], sigma=1, axes=(1, 2, 3))
        # data_u = data[:, 0]
        # data_v = data[:, 1]
        # self.solution = torch.from_numpy(np.stack([data_u, data_v], axis=1))
        # self.solution = torch.from_numpy(self._create_bubbles(timesteps=self.n_span))

    def __len__(self):
        return self.n_span - (self.trajec_max_len - 1)

    def __getitem__(self, index):
        # data = self.solution[index : index + self.trajec_max_len]
        data = torch.from_numpy(self._create_bubbles(timesteps=self.trajec_max_len))
        return data

    def _create_bubbles(self, timesteps=1000):
        flowss = []
        for _ in range(self.config.coarse_dim[2]):
            flows = []
            for _ in range(self.config.n_velocities):
                timesteps = 100
                grid_size = 13
                bubble_radius = 2
                num_bubbles = 1
                angle_variance = np.deg2rad(
                    [-45, -40, -35, -30, -25, -20, 20, 25, 30, 35, 40, 45]
                )
                blur_sigma = 1  # Standard deviation for Gaussian blur

                # Initialize bubble positions and directions
                bubble_positions = np.random.uniform(0, grid_size, (num_bubbles, 2))
                bubble_directions = np.array(
                    [
                        [
                            1,
                            np.tan(np.random.choice(angle_variance)),
                        ]
                        for _ in range(num_bubbles)
                    ]
                )
                bubble_directions = bubble_directions / np.linalg.norm(
                    bubble_directions, axis=1, keepdims=True
                )

                # Initialize random velocities for each bubble
                bubble_velocities = np.random.uniform(1, 3, (num_bubbles, 1))
                bubble_directions *= bubble_velocities  # Scale directions by velocity

                # Initialize 2D flow
                flow = np.zeros((timesteps, grid_size, grid_size), dtype=np.float32)

                # Simulate the flow
                for t in range(timesteps):
                    # Clear grid
                    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

                    # Update bubble positions and enforce periodic boundary conditions
                    bubble_positions += bubble_directions
                    bubble_positions %= grid_size

                    # Draw bubbles in the grid
                    for pos in bubble_positions:
                        x, y = np.round(pos).astype(int)
                        for dx in range(-bubble_radius, bubble_radius + 1):
                            for dy in range(-bubble_radius, bubble_radius + 1):
                                if dx**2 + dy**2 <= bubble_radius**2:
                                    nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
                                    grid[nx, ny] = 1

                    # Save grid to flow
                    flow[t] = grid
                flow = gaussian_filter(
                    flow.astype(np.float32), sigma=blur_sigma, axes=(1, 2)
                )
                flows.append(flow)
            flow = np.stack(flows, axis=1)
            flowss.append(flow)
        flow = np.stack(flowss, axis=-1)
        return flow


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import time

    def plot_input_target_pred(dataset: np.ndarray, start=0, stride=1):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        im0 = axes[0].imshow(dataset[start, 0, :, :, 0], cmap="viridis")
        im1 = axes[1].imshow(dataset[start + stride, 0, :, :, 0], cmap="viridis")
        im2 = axes[2].imshow(dataset[start + stride * 2, 0, :, :, 0], cmap="viridis")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.show()

    dset = data_2d_to_3d_bubbles()
    dloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    t0 = time.time()
    for iteration, batch in enumerate(dloader):
        print(time.time() - t0)
        print(f"{batch.shape=}")
        plot_input_target_pred(batch[0])
        t0 = time.time()
