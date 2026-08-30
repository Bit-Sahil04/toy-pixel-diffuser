"""Small, config-driven U-Net denoiser operating directly in pixel space.

No VAE / latent stage — this is a plain DDPM denoiser over RGB images.
All widths/depth/attention placement come from config.Config so the model can
be scaled up later without touching this file.

Contains a seam for v-prediction: `prediction_target` is a config value; the
network itself is identical for epsilon- and v-prediction (the difference is
the loss target, handled in diffusion.py).

Skip-connection bookkeeping follows the original DDPM implementation
(hojonathanho/diffusion): the initial conv output and every post-downsample
feature are pushed onto a skip stack in the encoder; the decoder pops them
LIFO, one per residual block (num_res_blocks + 1 per level). This balances
exactly: 1 + levels*num_res + (levels-1) pushes = levels*(num_res+1) pops.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Timestep embedding (standard DDPM sinusoidal + 2-layer MLP)
# ------------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ------------------------------------------------------------------------------
# Residual block with time-embedding injection
# ------------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ------------------------------------------------------------------------------
# Self-attention block (applied only at low resolutions per config)
# ------------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        attn = F.scaled_dot_product_attention(
            q.reshape(b, c, h * w).transpose(1, 2),
            k.reshape(b, c, h * w).transpose(1, 2),
            v.reshape(b, c, h * w).transpose(1, 2),
        )
        attn = attn.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(attn)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ------------------------------------------------------------------------------
# U-Net
# ------------------------------------------------------------------------------

class UNet(nn.Module):
    """Pixel-space DDPM U-Net.

    Config-driven: base_channels, channel_mults, num_res_blocks,
    attention_resolutions (feature-map sizes at which self-attention applies),
    dropout, image_size (used to compute per-level feature-map sizes).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.0,
        image_size: int = 128,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_sin = SinusoidalTimeEmbedding(base_channels)
        self.time_embed = TimeMLP(base_channels, time_dim)
        self.num_levels = len(channel_mults)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        levels = len(channel_mults)
        chans = [base_channels * m for m in channel_mults]
        feat_sizes = [image_size // (2**i) for i in range(levels)]
        # attention applies at any level whose feature map is <= the largest
        # configured resolution ("attention_resolutions" semantics, see config).
        # NOTE: changing this (or any channel/depth knob) rebuilds the model —
        # checkpoints from a differently-configured model will not load.
        attn_at = max(attention_resolutions) if attention_resolutions else 0
        assert image_size % (2 ** (levels - 1)) == 0, (
            f"image_size {image_size} not divisible by 2^{levels - 1}; "
            f"feature sizes {feat_sizes} would not be integral")

        # ---------------- encoder ----------------
        # Static record of every skip pushed in the encoder (pop order in the
        # decoder is the reverse of this push order).
        skip_chs = [base_channels]  # the init_conv output is the first skip
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        prev_ch = base_channels
        for lvl in range(levels):
            out_ch = chans[lvl]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(prev_ch, out_ch, time_dim, dropout))
                skip_chs.append(out_ch)
                prev_ch = out_ch
            self.down_blocks.append(blocks)
            self.down_attns.append(
                SelfAttention(prev_ch) if feat_sizes[lvl] <= attn_at else nn.Identity()
            )
            if lvl < levels - 1:
                self.down_samples.append(Downsample(prev_ch))
                skip_chs.append(prev_ch)   # skip across the downsample
            else:
                self.down_samples.append(nn.Identity())

        # ---------------- bottleneck ----------------
        self.mid_block1 = ResidualBlock(prev_ch, prev_ch, time_dim, dropout)
        self.mid_attn = SelfAttention(prev_ch)
        self.mid_block2 = ResidualBlock(prev_ch, prev_ch, time_dim, dropout)

        # ---------------- decoder ----------------
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        skip_iter = reversed(skip_chs)   # LIFO: mirror of forward-time pops
        for lvl in reversed(range(levels)):
            out_ch = chans[lvl]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResidualBlock(prev_ch + next(skip_iter), out_ch, time_dim, dropout))
                prev_ch = out_ch
            self.up_blocks.append(blocks)
            self.up_attns.append(
                SelfAttention(prev_ch) if feat_sizes[lvl] <= attn_at else nn.Identity()
            )
            if lvl > 0:
                self.up_samples.append(Upsample(prev_ch))
            else:
                self.up_samples.append(nn.Identity())

        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(self.time_sin(t))

        h = self.init_conv(x)
        skips = [h]  # init feature is the first skip (see module docstring)

        for lvl, (blocks, down, attn) in enumerate(
                zip(self.down_blocks, self.down_samples, self.down_attns)):
            for block in blocks:
                h = block(h, t_emb)
                skips.append(h)
            h = attn(h)
            if lvl < self.num_levels - 1:
                h = down(h)
                skips.append(h)   # skip across the downsample

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, t_emb)), t_emb)

        for blocks, up, attn in zip(self.up_blocks, self.up_samples, self.up_attns):
            for block in blocks:
                h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
            h = attn(h)
            h = up(h)

        return self.final_conv(F.silu(self.final_norm(h)))


# ------------------------------------------------------------------------------
# Model factory from Config
# ------------------------------------------------------------------------------

def build_model(cfg) -> UNet:
    """Instantiate the UNet from a config.Config.

    NOTE on prediction_target: the network is identical for "epsilon" and
    "v_prediction" — the target changes in diffusion.py, not here. This seam
    exists so unstable epsilon training can switch to v-prediction later
    without any architectural change.
    """
    assert cfg.prediction_target in ("epsilon", "v_prediction"), cfg.prediction_target
    return UNet(
        in_channels=3,
        out_channels=3,
        base_channels=cfg.base_channels,
        channel_mults=tuple(cfg.channel_mults),
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=tuple(cfg.attention_resolutions),
        dropout=cfg.dropout,
        image_size=cfg.image_size,
    )


if __name__ == "__main__":
    """Quick shape check: python model.py"""
    from config import Config

    cfg = Config()
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    t = torch.randint(0, cfg.timesteps, (2,))
    with torch.no_grad():
        y = model(x, t)
    print(f"params: {n_params/1e6:.2f}M")
    print(f"in : {tuple(x.shape)}")
    print(f"out: {tuple(y.shape)}")
    assert y.shape == x.shape, "UNet output shape mismatch"
    print("OK: UNet forward pass shape check passed.")
