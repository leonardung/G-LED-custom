import glob
import random
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

from config.seq_args_typed import SeqTypedArgs


class bfs_dataset_split_2d(Dataset):
    def __init__(self, config: SeqTypedArgs, mode: str = "train"):
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
            file_loaded = np.random.choice(self.files[:-1])
        else:
            file_loaded = self.files[-1]
        self.current: torch.Tensor = torch.from_numpy(np.load(file_loaded))
        self.current = self.current[:, : self.config.n_velocities, :, :, 64]

    def __getitem__(self, index):
        data = self.current[
            index : index + self.trajec_max_len * self.stride : self.stride
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
        return data


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
