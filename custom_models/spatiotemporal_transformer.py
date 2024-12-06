import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # Input shape: [batch_size, time_steps, channels, depth, height, width]
    batch_size = 2
    time_steps = 10
    channels = 3
    depth = 16
    height = 16
    width = 8
    input_data = torch.randn(batch_size, time_steps, channels, depth * height * width)
    model = SpatiotemporalTransformer(volume_size=(depth, height, width))
    output = model(input_data)
    print("Output shape:", output.shape)
