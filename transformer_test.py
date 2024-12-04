import os
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_bfs_preprocess import bfs_dataset


def process_tensor(
    tensor: torch.Tensor, coarse_dim: tuple[int, int, int], down_sampler: Callable
) -> torch.Tensor:
    """
    Process the input tensor by reshaping, downsampling, and flattening.
    Args:
        tensor: [batch_size, seq_len, num_velocity, dim1, dim2, dim3]
        seq_len: Sequence length of the tensor
    Returns:
        Flattened tensor: [batch_size, seq_len, num_velocity * coarse_dim[0] * coarse_dim[1] * coarse_dim[2]]
    """
    batch_size = tensor.shape[0]
    seq_len = tensor.shape[1]
    num_velocity = tensor.shape[2]

    # Squeeze and downsample
    tensor_squeezed = tensor.float().reshape(
        batch_size * seq_len,
        num_velocity,
        tensor.shape[3],
        tensor.shape[4],
        tensor.shape[5],
    )
    tensor_coarse: torch.Tensor = down_sampler(tensor_squeezed)

    # Reshape back and flatten
    tensor_coarse = tensor_coarse.reshape(
        batch_size,
        seq_len,
        num_velocity,
        coarse_dim[0],
        coarse_dim[1],
        coarse_dim[2],
    )
    tensor_coarse_flatten = tensor_coarse.reshape(
        batch_size,
        seq_len,
        num_velocity * coarse_dim[0] * coarse_dim[1] * coarse_dim[2],
    )
    return tensor_coarse_flatten


class SpatiotemporalTransformer(nn.Module):
    def __init__(
        self,
        volume_size=(30, 15, 30),
        num_timesteps=10,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super(SpatiotemporalTransformer, self).__init__()
        self.num_timesteps = num_timesteps
        self.volume_size = volume_size
        # Flattened spatial volume
        self.input_dim = 3 * volume_size[0] * volume_size[1] * volume_size[2]

        # Embedding layers
        self.input_projection = nn.Linear(self.input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(
            num_timesteps, d_model
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )

        # Output layer
        self.output_projection = nn.Linear(d_model, self.input_dim)

    def _generate_positional_encoding(self, seq_len, d_model):
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_timesteps, -1)
        x = self.input_projection(x)  # Shape: (batch_size, num_timesteps, d_model)
        x = x + self.positional_encoding[:, : self.num_timesteps, :].to(x.device).half()
        target = x[:, -1, :].unsqueeze(1)  # Shape: (batch_size, 1, d_model)
        output = self.transformer(src=x, tgt=target)  # Shape: (batch_size, 1, d_model)
        output = self.output_projection(output).squeeze(1)
        return output


if __name__ == "__main__":
    coarse_dim = (15, 15, 4)
    num_timesteps = 20
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    data_location = ["./data/test_data.npy"]  # Replace with your actual path
    dataset = bfs_dataset(
        data_location=data_location,
        trajec_max_len=num_timesteps + 1,
        n_span=500,
        start_n=0,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = bfs_dataset(
        data_location=data_location,
        trajec_max_len=num_timesteps + 1,
        n_span=500,
        start_n=500,
    )
    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = SpatiotemporalTransformer(
        volume_size=coarse_dim, num_timesteps=num_timesteps
    ).to(device)

    # Define the model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 100
    down_sampler = nn.Upsample(size=coarse_dim, mode="trilinear", align_corners=False)

    for epoch in tqdm(range(num_epochs), desc="Training Progress", leave=True):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch: torch.Tensor = batch.to(device)
            batch = process_tensor(batch, coarse_dim, down_sampler)
            input_data = batch[:, :-1]
            target_data = batch[:, -1]
            output = model(input_data)
            loss: torch.Tensor = criterion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    os.makedirs("test_models", exist_ok=True)
    model_save_path = "test_models/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
