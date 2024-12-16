import numpy as np
import pdb
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from config.seq_args_typed import TypedArgs


class bfs_dataset_paper(Dataset):
    def __init__(
        self,
        config: TypedArgs, mode: str = "train",
    ):
        if mode == "train":
            self.start_n = config.start_n
            self.n_span = config.n_span
            self.trajec_max_len = config.trajec_max_len
        else:
            self.start_n = config.start_n_valid
            self.n_span = config.n_span_valid
            self.trajec_max_len = config.trajec_max_len_valid
        solutions = []
        for location in config.data_location:
            solutions.append(np.load(location, allow_pickle=True))
        solution = np.concatenate(solutions, axis=0)
        self.solution = torch.from_numpy(solution[self.start_n : self.start_n + self.n_span])

    def __len__(self):
        return self.n_span - self.trajec_max_len

    def __getitem__(self, index):
        return self.solution[index : index + self.trajec_max_len]


if __name__ == "__main__":
    dset = bfs_dataset_paper()
    dloader = DataLoader(dataset=dset, batch_size=20, shuffle=True)
    for iteration, batch in enumerate(dloader):
        print(iteration)
        print("Do something!")
        pdb.set_trace()
