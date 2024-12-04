# General Package
import argparse
import sys
import pdb
from datetime import datetime
import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import time

# Internal package
from config.seq_args_typed import TypedArgs
from util.utils import save_args, load_config
from data.data_bfs_preprocess import bfs_dataset

from transformer.sequentialModel import SequentialModel as transformer

from train_test_seq.train_seq import train_seq_shift


def update_args(args: TypedArgs) -> TypedArgs:
    args.time = "{0:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    # output dataset
    args.dir_output = "output/"
    args.fname = args.dataset + "_" + args.time
    args.experiment_path = args.dir_output + args.fname
    args.model_save_path = os.path.join(args.experiment_path, "model_save/")
    args.logging_path = os.path.join(args.experiment_path, "logging/")
    args.current_model_save_path = args.model_save_path
    args.logging_epoch_path = os.path.join(args.logging_path, "epoch_history.csv")
    os.makedirs(args.logging_path, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


if __name__ == "__main__":
    config_path = "config/seq_sample.yml"
    args = load_config(config_path)
    args = update_args(args)
    save_args(args)
    assert (
        args.coarse_dim[0] * args.coarse_dim[1] * args.coarse_dim[2] * 3 == args.n_embd
    )
    print("Start data_set")
    tic = time.time()
    data_set_train = bfs_dataset(
        data_location=args.data_location,
        trajec_max_len=args.trajec_max_len,
        start_n=args.start_n,
        n_span=args.n_span,
    )
    data_set_test_on_train = bfs_dataset(
        data_location=args.data_location,
        trajec_max_len=args.trajec_max_len_valid,
        start_n=args.start_n,
        n_span=args.n_span,
    )
    data_set_valid = bfs_dataset(
        data_location=args.data_location,
        trajec_max_len=args.trajec_max_len_valid,
        start_n=args.start_n_valid,
        n_span=args.n_span_valid,
    )
    data_loader_train = DataLoader(
        dataset=data_set_train, shuffle=args.shuffle, batch_size=args.batch_size
    )
    data_loader_test_on_train = DataLoader(
        dataset=data_set_test_on_train,
        shuffle=args.shuffle,
        batch_size=args.batch_size_valid,
    )
    data_loader_valid = DataLoader(
        dataset=data_set_valid, shuffle=args.shuffle, batch_size=args.batch_size_valid
    )
    print("Done data-set use time ", time.time() - tic)
    """
    create model
    """
    model = transformer(args).to(args.device).float()
    print(
        "Number of parameters: {:,}".format(model._num_parameters()).replace(",", "'")
    )

    """
    create loss function
    """
    loss_func = torch.nn.MSELoss()

    """
    create optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    """
    create scheduler
    """
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.gamma
    )
    """
    train
    """
    train_seq_shift(
        args=args,
        model=model,
        data_loader=data_loader_train,
        data_loader_copy=data_loader_test_on_train,
        data_loader_valid=data_loader_valid,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
    )
