import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """Baseline: encode sequence -> decode -> reconstruction error = anomaly score."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        latent_size: int = 32,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        enc_out, (h, c) = self.encoder(x)
        latent = self.to_latent(h[-1])
        h_dec = self.from_latent(latent).unsqueeze(0).repeat(self.encoder.num_layers, 1, 1)
        c_dec = torch.zeros_like(h_dec)
        dec_in = h_dec[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder(dec_in, (h_dec, c_dec))
        recon = self.out(dec_out)
        return recon.transpose(1, 2)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2)
