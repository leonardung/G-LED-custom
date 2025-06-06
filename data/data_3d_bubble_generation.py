import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

from config.seq_args_typed import SeqTypedArgs


class bfs_3d_bubble(Dataset):
    def __init__(self, config: SeqTypedArgs, mode: str = "train"):
        self.config = config
        if mode == "train":
            self.n_span = config.n_span
            self.trajec_max_len = config.trajec_max_len
        else:
            self.n_span = config.n_span_valid
            self.trajec_max_len = config.trajec_max_len_valid
        self.mode = mode
        self.load_new_file()

    def __len__(self):
        return self.current.shape[0] - (self.trajec_max_len - 1)

    def load_new_file(self):
        self.current = self.create_3d_bubble(
            timesteps=self.n_span,
            grid_size=32,
            num_bubbles=3,
            min_radius=5,
            max_radius=5,
        )
        self.current = self.current.reshape(
            self.current.shape[0], 1, *self.current.shape[1:]
        )

    def create_3d_bubble(
        self, timesteps, grid_size, num_bubbles, min_radius, max_radius, blur_sigma=1
    ) -> np.ndarray:
        bubble_positions = np.random.uniform(0, grid_size, (num_bubbles, 3))
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
        bubble_growth_rates = np.random.uniform(0.1, 0.3, num_bubbles)

        # Initialize 3D flow
        flow = np.zeros((timesteps, grid_size, grid_size, grid_size))

        # Simulate the flow
        for t in range(timesteps):
            # Clear grid
            grid = np.zeros((grid_size, grid_size, grid_size))

            # Update bubble positions and enforce periodic boundary conditions
            bubble_positions += bubble_directions
            bubble_positions %= grid_size

            # Update bubble sizes
            for i in range(num_bubbles):
                bubble_sizes[i] += bubble_growth_directions[i] * bubble_growth_rates[i]
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
                                nx, ny, nz = (
                                    (x + dx) % grid_size,
                                    (y + dy) % grid_size,
                                    (z + dz) % grid_size,
                                )
                                grid[nx, ny, nz] = 1

            # Apply Gaussian blur
            grid = gaussian_filter(grid, sigma=blur_sigma)

            # Save grid to flow
            flow[t] = grid
        return flow

    def __getitem__(self, index):
        data = self.current[index : index + self.trajec_max_len]
        return data
