import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from config.seq_args_typed import SeqTypedArgs


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViTNextFrame(nn.Module):
    def __init__(
        self,
        *,
        image_size,  # e.g., (H, W) = (224, 224) or int
        image_patch_size,  # e.g., (16, 16) or int
        frames,  # total number of input frames, e.g. 4
        frame_patch_size,  # e.g., 2 (temporal patching)
        dim,  # transformer embedding dimension
        depth,  # number of transformer layers
        heads,  # number of attention heads
        mlp_dim,  # hidden dim inside each FFN
        channels=3,  # usually 3 for RGB
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pool="cls"  # 'cls'‐pooling or 'mean'‐pooling
    ):
        super().__init__()
        # -------------------------------
        # 1) Compute and store the patch geometry
        # -------------------------------
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # temporal‐patch: frames must be divisible by frame_patch_size
        assert (
            frames % frame_patch_size == 0
        ), "Frames must be divisible by frame_patch_size"

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * (frames // frame_patch_size)
        )
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        # For decoding a single next frame:
        #   #patches_per_frame = (image_height / patch_height) * (image_width / patch_width)
        num_patches_per_frame = (image_height // patch_height) * (
            image_width // patch_width
        )
        #   each patch in the next frame has size (channels × patch_height × patch_width)
        patch_dim_per_frame = channels * patch_height * patch_width

        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channels = channels
        self.num_patches_per_frame = num_patches_per_frame
        self.patch_dim_per_frame = patch_dim_per_frame

        assert pool in {"cls", "mean"}, "pool type must be either cls or mean"

        # -------------------------------
        # 2) Build the ViT encoder
        # -------------------------------
        self.to_patch_embedding = nn.Sequential(
            # group spatiotemporal patches of size: (frame_patch_size, patch_height, patch_width)
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b (f h w) (pf p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        # -------------------------------
        # 3) Replace `mlp_head` with a “patch‐decoder” for next‐frame regression
        # -------------------------------
        # We will take the pooled latent (shape [B, dim]) → LayerNorm → Linear → (num_patches_per_frame × patch_dim_per_frame)
        # Finally we Rearrange to [B, C, H, W].
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_patches_per_frame * patch_dim_per_frame),
        )

    def forward(self, video):
        """
        video: Tensor of shape [B, C, F, H, W]
               F % frame_patch_size == 0
        returns → next_frame_pred: Tensor of shape [B, C, H, W]
        """
        # 1) Encode spatiotemporal patches:
        x = self.to_patch_embedding(video)  # → [B, NumInputPatches, dim]
        B, N, _ = x.shape

        # 2) Prepend CLS token and add position embeddings:
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat((cls_tokens, x), dim=1)  # → [B, 1 + N, dim]
        x = x + self.pos_embedding[:, : (N + 1)]
        x = self.dropout(x)

        # 3) Transformer encoder
        x = self.transformer(x)  # → [B, 1 + N, dim]

        # 4) Pooling: take [CLS] or mean over tokens
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]  # → [B, dim]
        x = self.to_latent(x)  # → [B, dim]

        # 5) Decode into next‐frame patches
        #    Output shape after Linear: [B, num_patches_per_frame * patch_dim_per_frame]
        y = self.decoder(x)  # → [B, num_patches_per_frame * patch_dim_per_frame]

        # 6) Rearrange patches back into [B, C, H, W]
        #    We know:
        #       num_patches_per_frame = (image_height//patch_height) * (image_width//patch_width)
        #       patch_dim_per_frame = channels * patch_height * patch_width
        H, W = self.image_height, self.image_width
        p_h, p_w = self.patch_height, self.patch_width
        C = self.channels

        # First reshape to “[B, num_patches_per_frame, patch_dim_per_frame]”
        y = y.view(B, self.num_patches_per_frame, self.patch_dim_per_frame)

        # Then “unpatchify” → [B, C, H, W]
        #   We have num_patches_per_frame = (H//p_h) * (W//p_w)
        h_patches = H // p_h
        w_patches = W // p_w
        # Now each patch is flattened as (C * p_h * p_w). We need to put them back:
        #  ‘b (h w (c ph pw)) -> b c (h ph) (w pw)’
        next_frame = rearrange(
            y,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_patches,
            w=w_patches,
            ph=p_h,
            pw=p_w,
            c=C,
        )

        return next_frame


class ViTWrapper(ViTNextFrame):
    def __init__(
        self,
        config: SeqTypedArgs,
        image_patch_size=16,
        frame_patch_size=4,
        heads=16,
        dim_head=256,
    ):
        super().__init__(
            image_size=config.coarse_dim,  # e.g., (H, W) = (224, 224) or int
            image_patch_size=image_patch_size,  # e.g., (16, 16) or int
            frames=config.num_timesteps,  # total number of input frames, e.g. 4
            frame_patch_size=frame_patch_size,  # e.g., 2 (temporal patching)
            dim=1024,  # transformer embedding dimension
            depth=6,  # number of transformer layers
            heads=heads,  # number of attention heads
            mlp_dim=2048,  # hidden dim inside each FFN
            channels=config.n_velocities,  # usually 3 for RGB
            dim_head=dim_head,
            dropout=0.0,
            emb_dropout=0.0,
            pool="cls",  # 'cls'‐pooling or 'mean'‐pooling
        )


# if __name__ == "__main__":
#     image_size = (128, 64)  # image size
#     model = ViTNextFrame(
#         image_size=image_size,
#         image_patch_size=4,
#         frames=4,
#         frame_patch_size=2,
#         dim=512,
#         depth=6,
#         heads=8,
#         mlp_dim=1024,
#         channels=3,
#         dim_head=64,
#         dropout=0.1,
#         emb_dropout=0.1,
#         pool="cls",
#     )

#     # Dummy check:
#     #   Input: batch of 2 videos, each with shape [3, 4, 64, 64]
#     x = torch.randn(2, 3, 4, *image_size)
#     y_pred = model(x)
#     print(y_pred.shape)  # → torch.Size([2, 3, 64, 64])

if __name__ == "__main__":
    from util.utils import load_config, update_args

    config_path = "config/seq_2d_flow_vit.yml"
    config: SeqTypedArgs = load_config(config_path)
    config = update_args(config)

    x = torch.randn(
        config.batch_size, config.n_velocities, config.num_timesteps, *config.coarse_dim
    )
    model = ViTWrapper(config)
    out = model(x)
    print(x.shape)
    print(out.shape)
