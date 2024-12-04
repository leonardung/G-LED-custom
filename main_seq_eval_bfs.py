import torch
from torch.utils.data import DataLoader

# Internal package
from train_test_seq.test_seq import eval_seq_overall
from util.utils import load_config
from data.data_bfs_preprocess import bfs_dataset
from transformer.sequentialModel import SequentialModel as transformer





if __name__ == "__main__":
    """
    Fetch args
    """
    config_path = "config/seq_test.yml"
    args_sample = load_config(config_path)

    # Simulate reading training args from args.txt
    args_train = load_config(args_sample.train_args_txt)
    args_train.device = args_sample.device

    # Update args_sample with training experiment path
    args_sample.experiment_path = args_train.experiment_path


    """
	Pre-check
	"""
    assert args_train.coarse_dim[0] * args_train.coarse_dim[1] * args_train.coarse_dim[2] * 3 == args_train.n_embd

    """
	Fetch dataset
	"""
    data_set = bfs_dataset(
        data_location=args_train.data_location,
        trajec_max_len=args_sample.trajec_max_len,
        start_n=args_sample.start_n,
        n_span=args_sample.n_span,
    )
    data_loader = DataLoader(
        dataset=data_set, shuffle=args_sample.shuffle, batch_size=args_sample.batch_size
    )

    """
	Create and Load model
	"""
    model = transformer(args_train).to(args_sample.device).float()
    print("Number of parameters: {}".format(model._num_parameters()))
    model.load_state_dict(
        torch.load(
            args_train.current_model_save_path
            + "model_epoch_"
            + str(args_sample.Nt_read),
            map_location=torch.device(args_sample.device),
        )
    )

    """
	create loss function
	"""
    loss_func = torch.nn.MSELoss()

    """
	Eval
	"""
    eval_seq_overall(
        args_train=args_train,
        args_sample=args_sample,
        model=model,
        data_loader=data_loader,
        loss_func=loss_func,
    )
