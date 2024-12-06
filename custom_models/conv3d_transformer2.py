import torch
import torch.nn as nn
import math

from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=6000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)} of PositionalEncoding."
            )
        return x + self.pe[:, :seq_len, :]


class Conv3dTransformer2(nn.Module):
    def __init__(
        self,
        velocity_channels=3,
        embed_dim=128,  # increased from 64 to 128
        nhead=8,  # increased from 4 to 8
        num_encoder_layers=8,  # increased from 4 to 8
        dim_feedforward=512,  # increased from 256 to 512
        dropout=0.1,
    ):
        super(Conv3dTransformer2, self).__init__()

        # ---- 3D Convolutional Encoder ----
        self.conv_encoder = nn.Sequential(
            nn.Conv3d(velocity_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ---- Transformer ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=100000)

        # ---- Final projection to predict next timestep ----
        self.final_project = nn.Linear(embed_dim, velocity_channels)

    def forward(self, x):
        # x: [B, T, V, D, H, W]
        B, T, V, D, H, W = x.shape

        # Encode spatially per timestep
        x = x.view(B * T, V, D, H, W)  # [B*T, V, D, H, W]
        x = self.conv_encoder(x)  # [B*T, embed_dim, D, H, W]
        x = x.view(B, T, -1, D, H, W)  # [B, T, embed_dim, D, H, W]

        # Flatten spatial dims
        num_tokens = D * H * W
        # Move embed_dim last for flatten
        x = x.permute(0, 1, 3, 4, 5, 2)  # [B, T, D, H, W, embed_dim]
        x = x.reshape(B, T * num_tokens, -1)  # [B, T*D*H*W, embed_dim]

        # Add positional encoding
        x = self.pos_encoder(x)  # [B, T*D*H*W, embed_dim]

        # Transformer encoding
        x = self.transformer_encoder(x)  # [B, T*D*H*W, embed_dim]

        # Extract last timestep tokens
        last_tokens = x[:, -num_tokens:, :]  # [B, D*H*W, embed_dim]

        # Map to velocity channels
        out = self.final_project(last_tokens)  # [B, D*H*W, V]

        # Reshape to [B, V, D, H, W]
        out = out.view(B, D, H, W, V)
        out = out.permute(0, 4, 1, 2, 3)  # [B, V, D, H, W]
        return out


if __name__ == "__main__":
    B, T, V, D, H, W = 1, 10, 3, 16, 32, 8
    x = torch.randn(B, T, V, D, H, W).half().to("cuda")

    model = (
        Conv3dTransformer2(
            velocity_channels=V,
            embed_dim=512,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=512,
        )
        .half()
        .to("cuda")
    )
    for i in tqdm(range(100)):
        y = model(x)
    print(y.shape)  # [B, V, D, H, W]
