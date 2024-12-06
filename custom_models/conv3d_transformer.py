from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    return tensor_coarse


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # shape: [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [sequence_length, batch_size, d_model]
        x = x + self.pe[: x.size(0)]
        return x


class Conv3dTransformer(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512, num_layers=4, num_heads=8):
        super(Conv3dTransformer, self).__init__()
        # 3D Conv layers for spatial feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(feature_dim),
            nn.AdaptiveAvgPool3d(1),
        )
        # Positional Encoding for Transformer
        self.pos_encoder = PositionalEncoding(feature_dim)
        # Transformer Encoder for temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Decoder to reconstruct the next timestep's 3D field
        self.decoder = Decoder(in_channels, feature_dim)
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for Conv and ConvTranspose layers
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        # x shape: [batch, time, channels, depth, height, width]
        batch_size, time_steps, channels, depth, height, width = x.shape
        features = []
        for t in range(time_steps):
            xt = x[:, t]  # shape: [batch, channels, depth, height, width]
            ft = self.conv3d(xt)  # shape: [batch, feature_dim, 1, 1, 1]
            ft = ft.view(batch_size, -1)  # shape: [batch, feature_dim]
            features.append(ft)
        features = torch.stack(features, dim=1)
        features = features.permute(1, 0, 2)  # shape: [time_steps, batch, feature_dim]

        features = self.pos_encoder(features)
        transformer_output = self.transformer_encoder(features)
        last_output = transformer_output[-1]  # shape: [batch, feature_dim]
        out = self.decoder(last_output, (depth, height, width))
        return out


class Decoder(nn.Module):
    def __init__(self, out_channels, feature_dim):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.relu = nn.ReLU()
        # Define Conv3D layers
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.Conv3d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
                nn.Conv3d(feature_dim // 2, feature_dim // 4, kernel_size=3, padding=1),
                nn.Conv3d(feature_dim // 4, out_channels, kernel_size=3, padding=1),
            ]
        )
        # Define BatchNorm layers
        self.bn_layers = nn.ModuleList(
            [
                nn.BatchNorm3d(feature_dim),
                nn.BatchNorm3d(feature_dim // 2),
                nn.BatchNorm3d(feature_dim // 4),
            ]
        )

    def forward(self, x, output_shape):
        # x shape: [batch_size, feature_dim]
        batch_size = x.size(0)
        depth, height, width = output_shape

        # Start from 1x1x1 spatial dimensions
        x = x.view(batch_size, self.feature_dim, 1, 1, 1)

        # Determine the number of upsampling steps required
        # We'll upsample in powers of 2 until we reach or exceed the target dimensions
        target_sizes = (depth, height, width)
        current_sizes = [1, 1, 1]
        upsample_steps = []

        for dim in range(3):  # For depth, height, width
            size = current_sizes[dim]
            target_size = target_sizes[dim]
            steps = 0
            while size < target_size:
                size *= 2
                steps += 1
            upsample_steps.append(steps)

        max_steps = max(upsample_steps)

        # Apply upsampling and Conv3D layers
        for i in range(max_steps):
            scale_factors = []
            for dim in range(3):
                if i < upsample_steps[dim]:
                    scale_factors.append(2)
                else:
                    scale_factors.append(1)
            x = F.interpolate(
                x, scale_factor=scale_factors, mode="trilinear", align_corners=False
            )
            if i < len(self.conv_layers) - 1:
                x = self.relu(self.bn_layers[i](self.conv_layers[i](x)))
            elif i == len(self.conv_layers) - 1:
                # Last Conv layer without activation
                x = self.conv_layers[i](x)
            else:
                # If we have more upsampling steps than conv layers, apply identity
                pass

        # Crop or pad to match the exact target dimensions
        x = self._crop_or_pad(x, target_sizes)
        return x

    def _crop_or_pad(self, x, target_sizes):
        # x shape: [batch_size, channels, depth, height, width]
        _, _, depth, height, width = x.shape
        target_depth, target_height, target_width = target_sizes

        # Crop if necessary
        x = x[:, :, :target_depth, :target_height, :target_width]

        # Pad if necessary
        pad_d = max(0, target_depth - depth)
        pad_h = max(0, target_height - height)
        pad_w = max(0, target_width - width)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        return x


# Example usage:
if __name__ == "__main__":
    # Input shape: [batch_size, time_steps, channels, depth, height, width]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    time_steps = 10
    coarse_dim = (32, 32, 16)
    input_data = torch.randn(batch_size, time_steps, 3, 32, 32, 32).to(device)
    down_sampler = nn.Upsample(size=coarse_dim, mode="trilinear", align_corners=False)
    downsampled_data = process_tensor(input_data, coarse_dim, down_sampler)
    model = Conv3dTransformer(feature_dim=1024).to(device)
    for i in range(100):
        output = model(downsampled_data)
    print("Output shape:", output.shape)
    # Expected output shape: [batch_size, channels, depth, height, width]
