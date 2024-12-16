import glob
import random
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

from config.seq_args_typed import TypedArgs


class bfs_dataset_split_2d(Dataset):
    def __init__(self, config: TypedArgs, mode: str = "train"):
        self.config = config
        if mode == "train":
            self.n_span = config.n_span
            self.trajec_max_len = config.trajec_max_len
        else:
            self.n_span = config.n_span_valid
            self.trajec_max_len = config.trajec_max_len_valid
        self.stride = config.stride_valid
        self.mode = mode
        self.files = sorted(glob.glob(f"{config.data_location}/*.npy"))
        self.n_file = len(self.files)
        assert self.n_file > 1
        self.file_number = 0
        self.load_new_file()

    def __len__(self):
        return self.current.shape[0] - (self.trajec_max_len - 1) * self.stride

    def load_new_file(self):
        if self.mode == "train":
            self.current: torch.Tensor = torch.from_numpy(
                np.load(self.files[self.file_number])
            )
        else:
            self.current: torch.Tensor = torch.from_numpy(np.load(self.files[-1]))
        self.current = self.current[:, :2, :, :, 64]
        print(f"{self.file_number=}")
        self.file_number+=1
        self.file_number%=self.n_file-1

    def __getitem__(self, index):
        data = self.current[
            index : index + self.trajec_max_len * self.stride : self.stride
        ]
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

        crop_d = d // 2
        crop_h = h // 2
        crop_w = w // 2

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
    from matplotlib.animation import FFMpegWriter
    from tqdm import tqdm
    import time

    def plot_input_target_pred(dataset: np.ndarray, start=0, stride=1):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        im0 = axes[0].imshow(dataset[start, 0, :, :, 50], cmap="viridis")
        im1 = axes[1].imshow(dataset[start + stride, 0, :, :, 50], cmap="viridis")
        im2 = axes[2].imshow(dataset[start + 2 * stride, 0, :, :, 50], cmap="viridis")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.show()

    def create_video(dloader, filename="output_video.mp4", z_slice=50, fps=10):
        fig, ax = plt.subplots(figsize=(5, 5))
        img = ax.imshow(
            next(iter(dloader)).numpy()[0, 0, 0, :, :, z_slice], cmap="viridis"
        )
        ax.axis("off")
        writer = FFMpegWriter(fps=fps, metadata={"title": "Dataset Video"})
        with writer.saving(fig, filename, dpi=100):
            for iteration, batch in tqdm(enumerate(dloader)):
                data = batch[0].numpy()
                frame_data = data[0, 0, :, :, z_slice]
                img.set_data(frame_data)
                ax.set_title(f"Frame {iteration + 1}", fontsize=12)
                writer.grab_frame()

        print(f"Video saved as {filename}")

    dset = bfs_dataset(start_n=0, n_span=100)
    dloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    create_video(dloader, filename="dataset_video.mp4", z_slice=50, fps=30)
    # t0 = time.time()
    # for iteration, batch in enumerate(dloader):
    #     print(time.time() - t0)
    #     plot_input_target_pred(batch[0])
    #     t0 = time.time()
