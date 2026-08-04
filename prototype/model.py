"""Lightweight 1D U-Net noise predictor for DDPM."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * torch.arange(half, device=t.device).float()
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class UNet1D(nn.Module):
    def __init__(self, channels: int = 1, base: int = 32, time_dim: int = 64):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)

        self.down1 = ConvBlock(channels, base)
        self.down2 = ConvBlock(base, base * 2)
        self.pool = nn.MaxPool1d(2)

        self.mid = ConvBlock(base * 2, base * 2)

        self.up2 = nn.ConvTranspose1d(base * 2, base, 2, stride=2)
        self.up1_block = ConvBlock(base * 2, base)

        self.time_proj = nn.Linear(time_dim, base * 2)
        self.out = nn.Conv1d(base, channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        te_mid = self.time_proj(te).unsqueeze(-1)

        h1 = self.down1(x)
        h2 = self.down2(self.pool(h1))
        h2 = self.mid(h2) + te_mid
        h = self.up2(h2)
        # Align length after upsample
        if h.shape[-1] != h1.shape[-1]:
            h = torch.nn.functional.interpolate(h, size=h1.shape[-1], mode="linear", align_corners=False)
        h = self.up1_block(torch.cat([h, h1], dim=1))
        return self.out(h)
