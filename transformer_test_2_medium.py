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
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


class Transformer(nn.Module):
    # Constructor
    def __init__(
        self,
        input_dim,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.input_dim = input_dim

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Linear(self.input_dim, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, self.input_dim)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src and Tgt size: (batch_size, sequence_length, input_dim)

        # Embedding + positional encoding
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Permute to (sequence_length, batch_size, dim_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks
        transformer_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        # Output layer
        out = self.out(transformer_out)

        # Permute back to (batch_size, sequence_length, input_dim)
        out = out.permute(1, 0, 2)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token


if __name__ == "__main__":
    coarse_dim = (15, 15, 4)
    past_seq_len = 20
    future_seq_len = 1
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
    input_dim = 3 * coarse_dim[0] * coarse_dim[1] * coarse_dim[2]
    model = Transformer(
        input_dim=input_dim,
        dim_model=512,
        num_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout_p=0.05,
    ).to(device)

    # Define the model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Reduced LR

    # Training loop
    num_epochs = 20

    for epoch in tqdm(range(num_epochs), desc="Training Progress", leave=True):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch: torch.Tensor = batch.to(device)
            batch = process_tensor(batch, coarse_dim, down_sampler)

            input_data = batch[:, :past_seq_len]
            future_timesteps = batch[:, past_seq_len:]
            target_input = future_timesteps[:, :-1]
            target_output = future_timesteps[:, 1:]
            tgt_mask = model.get_tgt_mask(target_input.size(1)).to(device)
            output: torch.Tensor = model(input_data, target_input, tgt_mask)
            loss: torch.Tensor = criterion(output, target_output)
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
