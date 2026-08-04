"""TranAD: transformer + self-conditioning + dual decoders (VLDB 2022 wrapper)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsad.models.tranad_layers import PositionalEncoding, TransformerDecoderLayer, TransformerEncoderLayer


class TranAD(nn.Module):
    """
    Wrapper around the TranAD architecture for fixed-length windows.
    Input/output tensors inside forward use (L, B, C) — time-first, as in the original code.
    """

    def __init__(
        self,
        n_feats: int,
        n_window: int,
        dim_feedforward: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_window = n_window
        d_model = 2 * n_feats
        self.pos_encoder = PositionalEncoding(d_model, dropout, n_window)
        self.encoder_layer = TransformerEncoderLayer(d_model, n_feats, dim_feedforward, dropout)
        self.decoder_layer1 = TransformerDecoderLayer(d_model, n_feats, dim_feedforward, dropout)
        self.decoder_layer2 = TransformerDecoderLayer(d_model, n_feats, dim_feedforward, dropout)
        self.fcn = nn.Sequential(nn.Linear(d_model, n_feats), nn.Sigmoid())

    def _encode(self, src: torch.Tensor) -> torch.Tensor:
        return self.encoder_layer(src)

    def _decode(self, layer: TransformerDecoderLayer, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return layer(tgt, memory)

    def encode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor):
        src = torch.cat((src, c), dim=2)
        src = src * (self.n_feats**0.5)
        src = self.pos_encoder(src)
        memory = self._encode(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if tgt is None:
            tgt = src
        c = torch.zeros_like(src)
        x1 = self.fcn(self._decode(self.decoder_layer1, *self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self._decode(self.decoder_layer2, *self.encode(src, c, tgt)))
        return x1, x2

    def _to_seq_first(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> (L, B, C)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x.permute(2, 0, 1)

    def training_loss(self, x: torch.Tensor, epoch: int) -> torch.Tensor:
        src = self._to_seq_first(x)
        x1, x2 = self.forward(src, src)
        n = epoch + 1
        l1 = F.mse_loss(x1, src)
        l2 = F.mse_loss(x2, src)
        return (1.0 / n) * l1 + (1.0 - 1.0 / n) * l2

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Phase-2 reconstruction; x shape (B, C, L) or (C, L). Returns same layout as input."""
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        src = self._to_seq_first(x)
        _, x2 = self.forward(src, src)
        recon = x2.permute(1, 2, 0)
        return recon.squeeze(0) if single else recon
