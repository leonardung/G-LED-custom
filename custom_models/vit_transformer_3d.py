import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from config.seq_args_typed import SeqTypedArgs


# helpers
def triple(t):
    """If t is an int, return (t, t, t)."""
    return t if isinstance(t, tuple) else (t, t, t)


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
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ============================================================
#                 ViT for NEXT-VOLUME PREDICTION
# ============================================================
class ViTNextVolume3D(nn.Module):
    r"""
    Predict the next 3-D flow field (volume) from a short video of 3-D volumes.

    Args
    ----
    volume_size          : int or (D, H, W)
    volume_patch_size    : int or (pd, ph, pw)           – spatial patch
    frames               : total input frames, F
    frame_patch_size     : temporal patch length, pf
    dim                  : embedding dimension
    depth                : # transformer layers
    heads                : # attention heads
    mlp_dim              : hidden dim in FFN
    channels             : velocity components, e.g. 3 (u,v,w)
    """

    def __init__(
        self,
        *,
        volume_size,
        volume_patch_size,
        frames,
        frame_patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pool="cls",
    ):
        super().__init__()

        # ----------------------------------------------------
        # 1) patch geometry
        # ----------------------------------------------------
        D, H, W = triple(volume_size)
        pd, ph, pw = triple(volume_patch_size)

        assert (
            D % pd == 0 and H % ph == 0 and W % pw == 0
        ), "Volume dims not divisible by patch size."
        assert frames % frame_patch_size == 0, "frames % frame_patch_size must be 0."

        num_patches = (D // pd) * (H // ph) * (W // pw) * (frames // frame_patch_size)
        patch_dim = channels * pd * ph * pw * frame_patch_size

        # patches in the *next* volume (for decoding)
        patches_per_vol = (D // pd) * (H // ph) * (W // pw)
        patch_dim_per_vol = channels * pd * ph * pw

        self.D, self.H, self.W = D, H, W
        self.pd, self.ph, self.pw = pd, ph, pw
        self.channels = channels
        self.patches_per_vol = patches_per_vol
        self.patch_dim_per_vol = patch_dim_per_vol

        assert pool in {"cls", "mean"}

        # ----------------------------------------------------
        # 2) spatio-temporal patch embedding
        # ----------------------------------------------------
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (d pd) (h ph) (w pw) -> b (f d h w) (pf pd ph pw c)",
                pf=frame_patch_size,
                pd=pd,
                ph=ph,
                pw=pw,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # CLS token + positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # backbone
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        # ----------------------------------------------------
        # 3) patch-decoder → next 3-D flow
        # ----------------------------------------------------
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patches_per_vol * patch_dim_per_vol),
        )

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(self, video):
        """
        video : Tensor [B, C, F, D, H, W]

        Returns
        -------
        next_vol_pred : Tensor [B, C, D, H, W]
        """
        # 1) embed spatio-temporal patches
        x = self.to_patch_embedding(video)  # [B, N, dim]
        B, N, _ = x.shape

        # 2) CLS + positions
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (N + 1)]
        x = self.dropout(x)

        # 3) transformer encoder
        x = self.transformer(x)  # [B, 1+N, dim]

        # 4) pool
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        # 5) decode flat patches
        y = self.decoder(x)  # [B, patches_per_vol * patch_dim_per_vol]

        # 6) reshape → [B, C, D, H, W]
        y = y.view(B, self.patches_per_vol, self.patch_dim_per_vol)

        d_p, h_p, w_p = self.D // self.pd, self.H // self.ph, self.W // self.pw
        next_volume = rearrange(
            y,
            "b (d h w) (c pd ph pw) -> b c (d pd) (h ph) (w pw)",
            d=d_p,
            h=h_p,
            w=w_p,
            pd=self.pd,
            ph=self.ph,
            pw=self.pw,
            c=self.channels,
        )

        return next_volume


# ------------------------------------------------------------
# A thin convenience wrapper that plugs in your SeqTypedArgs
# ------------------------------------------------------------
class ViT3DWrapper(ViTNextVolume3D):
    def __init__(
        self,
        config: SeqTypedArgs,  # SeqTypedArgs or similar (must expose .coarse_dim_3d, .num_timesteps, .n_velocities)
        dim=1024,
        depth=6,
        volume_patch_size=(8, 4, 4),  # int or (pd,ph,pw)
        frame_patch_size=4,
        heads=16,
        dim_head=256,
    ):
        super().__init__(
            volume_size=config.coarse_dim,  # (D, H, W)
            volume_patch_size=volume_patch_size,
            frames=config.num_timesteps,
            frame_patch_size=frame_patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=dim*2,
            channels=config.n_velocities,
            dim_head=dim_head,
            dropout=0.0,
            emb_dropout=0.0,
            pool="cls",
        )


if __name__ == "__main__":
    from util.utils import load_config, update_args

    config_path = "config/seq_3d_flow_vit.yml"
    config: SeqTypedArgs = load_config(config_path)
    config = update_args(config)

    x = torch.randn(
        config.batch_size, config.n_velocities, config.num_timesteps, *config.coarse_dim
    )
    print(x.shape)
    model = ViT3DWrapper(config)
    out = model(x)
    print(out.shape)
