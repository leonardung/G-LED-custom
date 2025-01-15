import torch
import torch.nn as nn
import math

from config.seq_args_typed import SeqTypedArgs


class PositionalEncoding3D(nn.Module):
    """
    Learned positional encoding for 3D volumes (D, H, W) and time steps T.
    We’ll create separate embeddings for time and for each spatial dimension,
    and then combine them.
    """

    def __init__(
        self, embed_dim, max_time=50, max_depth=32, max_height=64, max_width=64
    ):
        super(PositionalEncoding3D, self).__init__()
        self.time_embed = nn.Embedding(max_time, embed_dim)
        self.depth_embed = nn.Embedding(max_depth, embed_dim)
        self.height_embed = nn.Embedding(max_height, embed_dim)
        self.width_embed = nn.Embedding(max_width, embed_dim)

        # Initialize embeddings
        nn.init.normal_(self.time_embed.weight, std=0.02)
        nn.init.normal_(self.depth_embed.weight, std=0.02)
        nn.init.normal_(self.height_embed.weight, std=0.02)
        nn.init.normal_(self.width_embed.weight, std=0.02)

    def forward(self, t_indices, d_indices, h_indices, w_indices):
        # t_indices: [batch, time]
        # d_indices, h_indices, w_indices: [1, depth], [1, height], [1, width]
        # We'll need to broadcast and combine them properly
        t_embed = self.time_embed(t_indices)  # [B, T, E]
        d_embed = self.depth_embed(d_indices)  # [D, E]
        h_embed = self.height_embed(h_indices)  # [H, E]
        w_embed = self.width_embed(w_indices)  # [W, E]

        # We want a combined embedding for each token (t,d,h,w)
        # final shape: [B, T, D, H, W, E]
        # Combine by summing: position = time_pos + depth_pos + height_pos + width_pos
        # We'll broadcast these embeddings across the dimensions:
        # t_embed: [B, T, 1, 1, 1, E]
        # d_embed: [1, 1, D, 1, 1, E]
        # h_embed: [1, 1, 1, H, 1, E]
        # w_embed: [1, 1, 1, 1, W, E]
        t_embed = t_embed.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, T, 1, 1, 1, E]
        d_embed = (
            d_embed.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4)
        )  # [1, 1, D, 1, 1, E]
        h_embed = (
            h_embed.unsqueeze(0).unsqueeze(0).unsqueeze(2).unsqueeze(4)
        )  # [1, 1, 1, H, 1, E]
        w_embed = (
            w_embed.unsqueeze(0).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )  # [1, 1, 1, 1, W, E]

        pos = t_embed + d_embed + h_embed + w_embed
        return pos


class FlowPredictorTransformer(nn.Module):
    def __init__(
        self,
        config: SeqTypedArgs,
    ):
        super(FlowPredictorTransformer, self).__init__()

        self.velocity_channels = 3
        self.depth = config.coarse_dim[0]
        self.height = config.coarse_dim[1]
        self.width = config.coarse_dim[2]
        self.time_steps = config.num_timesteps
        self.embed_dim = config.n_embd

        # Project each voxel velocity vector into embed_dim features.
        # For complexity, we use a small 3D conv net:
        self.input_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=self.velocity_channels,
                out_channels=self.embed_dim // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.embed_dim // 2,
                out_channels=self.embed_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        # Positional Encoding
        self.pos_encoding = PositionalEncoding3D(
            embed_dim=self.embed_dim,
            max_time=self.time_steps,
            max_depth=self.depth,
            max_height=self.height,
            max_width=self.width,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layer
        )

        # Final prediction layer: map embeddings back to velocity channels
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.velocity_channels),
        )

    def forward(self, x):
        # x: [B, T, V, D, H, W]
        B, T, V, D, H, W = x.shape

        # Apply 3D conv to each timestep independently, or stack them:
        # We can treat T as a batch dimension for convolution or loop over timesteps.
        # For simplicity, apply the conv over each timestep:
        # Result per timestep: [B, E, D, H, W]
        x_reshape = x.view(B * T, V, D, H, W)
        x_embed = self.input_conv(x_reshape)  # [B*T, E, D, H, W]

        # Flatten spatial dimensions to create tokens
        # tokens: [B*T, D*H*W, E]
        x_embed = x_embed.permute(0, 2, 3, 4, 1).contiguous()  # [B*T, D, H, W, E]
        x_embed = x_embed.view(B, T, D, H, W, self.embed_dim)

        # Create positional indices
        t_indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        d_indices = torch.arange(D, device=x.device)
        h_indices = torch.arange(H, device=x.device)
        w_indices = torch.arange(W, device=x.device)

        pos = self.pos_encoding(t_indices, d_indices, h_indices, w_indices)
        # pos: [B, T, D, H, W, E]

        x_embed = x_embed + pos

        # Now flatten [T, D, H, W] into a sequence dimension:
        # Sequence length = T * D * H * W
        seq_len = T * D * H * W
        x_embed = x_embed.view(B, seq_len, self.embed_dim)  # [B, T*D*H*W, E]

        # Transformer encoding
        # Key/Query/Value from the same sequence
        x_transformed = self.transformer_encoder(x_embed)  # [B, seq_len, E]

        # We want to predict the next timestep volume. One approach:
        # - Take the final representation and map it to a single predicted frame
        #   We assume the model has learned the temporal correlation to produce the next step.
        # Here we simply use the final transformer output and reshape back to [V, D, H, W].
        # For the next timestep, we want a single frame: that means we produce [B, V, D, H, W]

        # Apply output head to each token:
        x_out = self.output_head(x_transformed)  # [B, seq_len, V]

        # Reshape back to volume:
        x_out = x_out.view(B, T, D, H, W, self.velocity_channels)
        # We have T frames here, but we really want the next frame after the last input timestep.
        # A simple baseline: take the representation corresponding to the last timestep's positions
        # and try to predict the next frame from it.

        # However, since we haven't explicitly separated the next timestep token,
        # a simple approach is: use the final output token positions as is.
        # A more sophisticated approach might incorporate a separate "predict" token
        # or causal masking. For simplicity, let's just take the last T-th frame embedding:
        # Actually, we want a single frame as output. Let's just take the last T-th slice and treat
        # the output as the next prediction (in a real scenario, you'd train with a shift).

        # We'll just take the last T-th frame output as the predicted next frame
        # In practice, you'd want to align training so that you produce T+1 from first T.
        predicted_next = x_out[:, -1, ...]  # [B, D, H, W, V]
        predicted_next = predicted_next.permute(
            0, 4, 1, 2, 3
        ).contiguous()  # [B, V, D, H, W]

        return predicted_next


# Example usage:
# if __name__ == "__main__":
#     depth = 16
#     height = 16
#     width = 4
#     model = FlowPredictorTransformer(
#         velocity_channels=3,
#         depth=depth,
#         height=height,
#         width=width,
#         time_steps=10,
#         embed_dim=512,
#         n_heads=8,
#         num_layers=8,
#         dim_feedforward=1024,
#         dropout=0.1,
#         max_time=10,
#     ).to("cuda")

#     # Dummy input: batch=2, time=10, velocity=3, D=16, H=16, W=8
#     x = torch.randn(2, 10, 3, depth, height, width).to("cuda")
#     with torch.no_grad():
#         for i in range(100):
#             y_pred = model(x)
#     print(y_pred.shape)  # Expected: [2, 3, 16, 16, 8]
