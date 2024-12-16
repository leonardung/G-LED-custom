import random
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter

from config.seq_args_typed import TypedArgs


class data_3d_bubbles(Dataset):
    def __init__(
        self,
        config: TypedArgs,
        mode: str = "train",
    ):
        self.config = config
        if mode == "train":
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
        flows = []
        for _ in range(self.config.n_velocities):
            grid_size = self.config.coarse_dim  # Sizes for x, y, z directions
            num_bubbles = 1
            min_radius, max_radius = 3, 3  # Minimum and maximum bubble radii
            blur_sigma = 0.5  # Standard deviation for Gaussian blur

            # Initialize bubble positions and directions in 3D
            bubble_positions = np.random.uniform(0, 1, (num_bubbles, 3)) * grid_size
            bubble_directions = np.random.normal(0, 1, (num_bubbles, 3))
            bubble_directions = bubble_directions / np.linalg.norm(
                bubble_directions, axis=1, keepdims=True
            )

            # Initialize random velocities for each bubble
            bubble_velocities = np.random.uniform(1, 2, (num_bubbles, 1))
            bubble_directions *= bubble_velocities  # Scale directions by velocity

            # Initialize bubble radii and growth parameters
            bubble_sizes = np.random.uniform(min_radius, max_radius, num_bubbles)
            bubble_growth_directions = np.random.choice([-1, 1], num_bubbles)
            bubble_growth_rates = np.random.uniform(0.5, 1, num_bubbles)

            # Initialize 3D flow
            flow = np.zeros((timesteps, *grid_size), dtype=np.int8)

            # Simulate the flow
            for t in range(timesteps):
                # Clear grid
                grid = np.zeros(grid_size, dtype=np.int8)

                # Update bubble positions and enforce periodic boundary conditions
                bubble_positions += bubble_directions
                bubble_positions %= (
                    grid_size  # Periodic boundary condition for each dimension
                )

                # Update bubble sizes
                for i in range(num_bubbles):
                    bubble_sizes[i] += (
                        bubble_growth_directions[i] * bubble_growth_rates[i]
                    )
                    if bubble_sizes[i] >= max_radius or bubble_sizes[i] <= min_radius:
                        bubble_growth_directions[i] *= -1  # Reverse growth direction

                # Draw bubbles in the grid
                for i, pos in enumerate(bubble_positions):
                    x, y, z = np.round(pos).astype(int)
                    radius = int(bubble_sizes[i])
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            for dz in range(-radius, radius + 1):
                                if dx**2 + dy**2 + dz**2 <= radius**2:
                                    nx = (x + dx) % grid_size[0]
                                    ny = (y + dy) % grid_size[1]
                                    nz = (z + dz) % grid_size[2]
                                    grid[nx, ny, nz] = 1

                # Save grid to flow
                flow[t] = grid

            # Apply Gaussian blur to the flow
            flow = gaussian_filter(
                flow.astype(np.float32), sigma=blur_sigma, axes=(1, 2, 3)
            )
            flows.append(flow)
        flow = np.stack(flows, axis=1)
        return flow


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import time

    def plot_input_target_pred(dataset: np.ndarray, start=0, stride=1):
        fig, axes = plt.subplots(2, 5, figsize=(20, 5))
        axes = axes.flatten()
        for i in range(5):
            axes[i].imshow(dataset[i, 0, :, 4, :], cmap="viridis")
        for i in range(5):
            axes[i + 5].imshow(dataset[i, 0, :, 7, :], cmap="viridis")
        plt.show()

    dset = data_3d_bubbles()
    dloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    t0 = time.time()
    for iteration, batch in enumerate(dloader):
        print(time.time() - t0)
        print(f"{batch.shape=}")
        plot_input_target_pred(batch[0])
        t0 = time.time()
