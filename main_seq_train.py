import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from train_test_seq.train_seq import train_model
from util.utils import load_config, save_args, update_args
from dataset_registry import register_datasets, get_dataset
from model_registry import register_models, get_model

register_datasets()
register_models()


if __name__ == "__main__":
    config_path = "config/seq_3d_flow.yml"
    config = load_config(config_path)
    config = update_args(config)
    save_args(config)

    dataset = get_dataset(config.dataset, **{"config": config, "mode": "train"})
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=0
    )

    val_dataset = get_dataset(config.dataset, **{"config": config, "mode": "val"})
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size_valid, shuffle=False, num_workers=0
    )
    print(f"{len(dataset) = }")
    print(f"{len(dataloader) = }")
    print(f"{len(val_dataset) = }")
    print(f"{len(val_dataloader) = }")
    model: torch.nn.Module = get_model(config.model_name, **{"config": config}).to(
        config.device
    )
    try:
        model.load_state_dict(torch.load(config.previous_model_name))
    except:
        print(f"No checkpoint found")

    # Define the model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_decay_factor,
        patience=config.lr_decay_patience,
    )

    # Run the training loop
    train_model(
        config=config,
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
