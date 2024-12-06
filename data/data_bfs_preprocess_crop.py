import random
import numpy as np
import pdb
from torch.utils.data import Dataset, DataLoader
import torch


import torch.nn.functional as F


class bfs_dataset(Dataset):
    def __init__(
        self,
        data_location=["./global_domain_float16_all.npy"],
        trajec_max_len=10,
        start_n=0,
        n_span=600,
        stride=1,
        mode="train",
    ):
        assert n_span > trajec_max_len
        self.start_n = start_n
        self.n_span = n_span
        self.trajec_max_len = trajec_max_len
        self.stride = stride
        self.mode = mode
        solutions = []
        for location in data_location:
            solutions.append(np.load(location, allow_pickle=True))
        solution = np.concatenate(solutions, axis=0)
        self.solution = torch.from_numpy(solution[start_n : start_n + n_span])

    def __len__(self):
        return self.n_span - (self.trajec_max_len - 1) * self.stride

    def __getitem__(self, index):
        # Extract a trajectory of flow fields
        data = self.solution[
            index : index + self.trajec_max_len * self.stride : self.stride
        ]
        # Apply random crop and resize
        if self.mode == "train" and np.random.random() > 0.5:
            print("cropped")
            data = self.random_crop_and_resize(data)
        else:
            print("not cropped")

        return data

    def random_crop_and_resize(self, data):
        """
        Apply a random crop and resize the volume back to its original size.
        Args:
            data (torch.Tensor): Input tensor of shape [time, velocity, depth, height, width].
        Returns:
            torch.Tensor: Transformed tensor with the same shape as input.
        """
        _, v, d, h, w = data.shape

        # Randomly select crop sizes within a range [half, full size] for each dimension
        crop_d = np.random.randint(d // 2, d + 1)
        crop_h = np.random.randint(h // 2, h + 1)
        crop_w = np.random.randint(w // 2, w + 1)

        # Randomly select start indices ensuring the crop fits within the bounds
        d_start = np.random.randint(0, d - crop_d + 1)
        h_start = np.random.randint(0, h - crop_h + 1)
        w_start = np.random.randint(0, w - crop_w + 1)

        # Crop the data
        cropped = data[
            :,
            :,
            d_start : d_start + crop_d,
            h_start : h_start + crop_h,
            w_start : w_start + crop_w,
        ]

        # Interpolate back to the original size
        resized = F.interpolate(
            cropped,  # Add batch dimension for interpolation
            size=(d, h, w),  # Target size
            mode="trilinear",  # Trilinear interpolation
            align_corners=True,
        )
        return resized


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    def plot_input_target_pred(dataset: np.ndarray, start=0, stride=1):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        im0 = axes[0].imshow(dataset[start, 0, :, :, 50], cmap="viridis")
        im1 = axes[1].imshow(dataset[start + stride, 0, :, :, 50], cmap="viridis")
        im2 = axes[2].imshow(dataset[start + 2 * stride, 0, :, :, 50], cmap="viridis")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.show()

    dset = bfs_dataset(data_location=["data/global_domain_float16_all.npy"], stride=20)
    dloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    for iteration, batch in enumerate(dloader):
        plot_input_target_pred(batch[0])
