import os
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()
        # Create positional encoding matrix
        pos_encoding = torch.zeros(max_len, dim_model)
        positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim_model)
        )
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        if dim_model % 2 == 1:
            pos_encoding[:, 1::2] = torch.cos(positions * div_term[:-1])
        else:
            pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        pos_encoding = pos_encoding.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer("pe", pos_encoding)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[: x.size(0)]
        return x


class FlowPredictor(nn.Module):
    def __init__(
        self,
        down_sampler: Callable,
        coarse_dim=tuple[int, int, int],
        dim_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.down_sampler = down_sampler
        self.coarse_dim = coarse_dim
        self.input_dim = coarse_dim[0] * coarse_dim[1] * coarse_dim[2] * 3
        self.d_model = dim_model

        # Encoder input embedding
        self.encoder_input_linear = nn.Linear(self.input_dim, dim_model)

        # Decoder input embedding
        self.decoder_input_linear = nn.Linear(self.input_dim, dim_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(dim_model)
        self.pos_decoder = PositionalEncoding(dim_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_model * 4,
            dropout=dropout,
        )

        self.output_linear = nn.Linear(dim_model, self.input_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            src: [batch_size, src_seq_len, input_dim]
            tgt: [batch_size, tgt_seq_len, input_dim]
        Returns:
            output: [batch_size, tgt_seq_len, input_dim]
        """
        tgt_seq_len = tgt.shape[1]

        # Embed the source and target sequences
        # [batch_size, src_seq_len, d_model]
        src = self.encoder_input_linear(src)
        # [batch_size, tgt_seq_len, d_model]
        tgt = self.decoder_input_linear(tgt)

        # Transpose for transformer input
        src = src.permute(1, 0, 2)  # [src_seq_len, batch_size, d_model]
        tgt = tgt.permute(1, 0, 2)  # [tgt_seq_len, batch_size, d_model]

        # Apply positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        # Generate masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Forward through the transformer
        # Output: [tgt_seq_len, batch_size, d_model]
        output: torch.Tensor = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)  # [batch_size, tgt_seq_len, d_model]
        output = self.output_linear(output)  # [batch_size, tgt_seq_len, input_dim]

        return output


if __name__ == "__main__":
    coarse_dim = (15, 15, 4)
    past_seq_len = 10
    future_seq_len = 30
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    data_location = ["./data/test_data.npy"]  # Replace with your actual path
    dataset = bfs_dataset(
        data_location=data_location,
        trajec_max_len=past_seq_len + future_seq_len + 1,
        n_span=500,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    down_sampler = nn.Upsample(size=coarse_dim, mode="trilinear", align_corners=False)
    model = FlowPredictor(down_sampler, coarse_dim).to(device)

    # Define the model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Reduced LR

    # Training loop
    num_epochs = 50

    for epoch in tqdm(range(num_epochs), desc="Training Progress", leave=True):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch: torch.Tensor = batch.to(device)
            batch = process_tensor(batch, coarse_dim, down_sampler)
            optimizer.zero_grad()
            input_data = batch[:, :past_seq_len]
            future_timesteps = batch[:, past_seq_len:]
            target_input = future_timesteps[:, :-1]
            target_output = future_timesteps[:, 1:]
            output = model(input_data, target_input)
            loss: torch.Tensor = criterion(output, target_output)
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
