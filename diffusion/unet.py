"""
diffusion/unet.py  —  Time-conditioned U-Net
=============================================
PROVIDED — do not modify.

A lightweight U-Net that accepts a 28×28 single-channel image and a scalar
time embedding.  Shared by both the VP score model (Part 5) and the
rectified flow velocity network (Part 6).

Usage::

    model = UNet(in_channels=1, base_channels=64)
    # For VP score model: output is the predicted score s_θ(x, t)
    # For rectified flow: output is the predicted velocity v_θ(x, t)
    out = model(x, t)   # x: (B,1,28,28),  t: (B,) float in [0,1] or {1..T}
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ------------------------------------------------------------------
# Time-step embedding
# ------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time step."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) — values can be continuous [0,1] or discrete {1..T}
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ------------------------------------------------------------------
# U-Net
# ------------------------------------------------------------------

class UNet(nn.Module):
    """Lightweight time-conditioned U-Net for 28×28 images.

    Args:
        in_channels:   Number of input image channels (1 for FashionMNIST).
        base_channels: Base feature-map width; doubled at each down-block.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        C = base_channels
        time_dim = C * 4

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(C),
            nn.Linear(C, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc_in  = nn.Conv2d(in_channels, C, 3, padding=1)
        self.enc1    = ResBlock(C,     C,     time_dim)  # 28×28
        self.down1   = nn.Conv2d(C,    C * 2, 3, stride=2, padding=1)  # 14×14
        self.enc2    = ResBlock(C * 2, C * 2, time_dim)
        self.down2   = nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1)  # 7×7

        # Bottleneck
        self.mid1 = ResBlock(C * 4, C * 4, time_dim)
        self.mid2 = ResBlock(C * 4, C * 4, time_dim)

        # Decoder
        self.up1     = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)   # 14×14
        self.dec1    = ResBlock(C * 4, C * 2, time_dim)  # skip from enc2
        self.up2     = nn.ConvTranspose2d(C * 2, C,     2, stride=2)   # 28×28
        self.dec2    = ResBlock(C * 2, C,     time_dim)  # skip from enc1

        self.out_norm = nn.GroupNorm(8, C)
        self.out_conv = nn.Conv2d(C, in_channels, 1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (B, C_in, 28, 28)
            t: (B,) — float time in [0,1] or integer step in {1..T}

        Returns:
            (B, C_in, 28, 28)
        """
        t_emb = self.time_embed(t)

        # Encoder
        h0  = self.enc_in(x)
        h1  = self.enc1(h0, t_emb)
        h2  = self.enc2(self.down1(h1), t_emb)
        h   = self.mid2(self.mid1(self.down2(h2), t_emb), t_emb)

        # Decoder with skip connections
        h   = self.dec1(torch.cat([self.up1(h), h2], dim=1), t_emb)
        h   = self.dec2(torch.cat([self.up2(h), h1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
