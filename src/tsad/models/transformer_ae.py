import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerAutoencoder(nn.Module):
    """Transformer baseline: encode -> latent bottleneck -> decode -> reconstruction error."""

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        latent_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.to_latent = nn.Linear(d_model, latent_size)
        self.from_latent = nn.Linear(latent_size, d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers)
        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        seq_len = x.shape[1]
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        latent = self.to_latent(h.mean(dim=1))
        dec_in = self.from_latent(latent).unsqueeze(1).repeat(1, seq_len, 1)
        dec_in = self.pos_encoder(dec_in)
        dec_out = self.decoder(dec_in)
        recon = self.output_proj(dec_out)
        return recon.transpose(1, 2)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2)
