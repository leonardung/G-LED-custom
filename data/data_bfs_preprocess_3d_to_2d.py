import random
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader


class bfs_dataset_3d_to_2d(Dataset):
    def __init__(
        self,
        data_location: list[str] = ["./global_domain_float16_all.npy"],
        trajec_max_len: int = 10,
        start_n: int = 0,
        n_span: int = 1000,
        stride: list[int] | int = list(range(10,11)),
        cropped_probability=0.9,
        mode="train",
    ):
        assert n_span > trajec_max_len
        self.start_n = start_n
        self.n_span = n_span
        self.trajec_max_len = trajec_max_len
        if isinstance(stride, int):
            self.stride = [stride]
        else:
            self.stride = stride
        self.cropped_probability = cropped_probability
        self.mode = mode
        solutions = []
        for location in data_location:
            solutions.append(np.load(location, allow_pickle=True))
        solution = np.concatenate(solutions, axis=0)
        self.solution = torch.from_numpy(solution[start_n : start_n + n_span])[
            :, :2, :, :, 60
        ]
        print(f"{self.solution.shape=}")

    def __len__(self):
        return self.n_span - (self.trajec_max_len - 1) * max(self.stride)

    def __getitem__(self, index):

        selected_stride = random.choice(self.stride)

        # Extract the trajectory with the selected stride
        data = self.solution[
            index : index + self.trajec_max_len * selected_stride : selected_stride
        ]

        if self.mode == "train":
            axes_combinations = [
                (),  # No flip (original)
                (2,),  # x-axis (depth)
                (3,),  # y-axis (height)
                (2, 3),  # xy-plane
            ]

            flip_axes = random.choice(axes_combinations)
            if flip_axes:
                data = torch.flip(data, dims=flip_axes)
            if np.random.random() < self.cropped_probability:
                data = self.random_crop_and_resize(data)

        return data

    def random_crop_and_resize(self, data):
        """
        Apply a random crop and resize the volume back to its original size.
        Args:
            data (torch.Tensor): Input tensor of shape [time, velocity, depth, height, width].
        Returns:
            torch.Tensor: Transformed tensor with the same shape as input.
        """
        _, v, d, h = data.shape

        # Randomly select crop sizes within a range [half, full size] for each dimension
        crop_d = np.random.randint(d // 2, d + 1)
        crop_h = np.random.randint(h // 2, h + 1)

        crop_d = d // 2
        crop_h = h // 2

        # Randomly select start indices ensuring the crop fits within the bounds
        d_start = np.random.randint(0, d - crop_d + 1)
        h_start = np.random.randint(0, h - crop_h + 1)

        # Crop the data
        cropped = data[
            :,
            :,
            d_start : d_start + crop_d,
            h_start : h_start + crop_h,
        ]

        # Interpolate back to the original size
        resized = F.interpolate(
            cropped,  # Add batch dimension for interpolation
            size=(d, h),  # Target size
            mode="bilinear",  # Trilinear interpolation
            align_corners=False,
        )
        return cropped


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    def plot_input_target_pred(dataset: np.ndarray, start=0, stride=1):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        im0 = axes[0].imshow(dataset[start, 0, :, :], cmap="viridis")
        im1 = axes[1].imshow(dataset[start + stride, 0, :, :], cmap="viridis")
        im2 = axes[2].imshow(dataset[start + 2 * stride, 0, :, :], cmap="viridis")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.show()

    dset = bfs_dataset_3d_to_2d(data_location=["data/global_domain_float16_all.npy"])
    print(f"{len(dset)=}")
    dloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    for iteration, batch in enumerate(dloader):
        plot_input_target_pred(batch[0])
