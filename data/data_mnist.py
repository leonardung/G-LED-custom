import random
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader


class mnist_dataset(Dataset):
    def __init__(
        self,
        data_location: str = "Video_diffusion_github\VPTR\dataset\mnist_test_seq.npy",
        start_n: int = 0,
        n_span: int = 1000,
        mode="train",
    ):
        self.start_n = start_n
        self.n_span = n_span
        self.mode = mode
        self.solution = torch.from_numpy(np.load(data_location, allow_pickle=True))/255
        self.solution = self.solution[:, start_n : start_n + n_span].unsqueeze(2)

    def __len__(self):
        return self.n_span

    def __getitem__(self, index):

        # Extract the trajectory with the selected stride
        data = self.solution[:, index]

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

        return data


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

    dset = mnist_dataset()
    dloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    for iteration, batch in enumerate(dloader):
        print(batch.shape)
        plot_input_target_pred(batch[0])
