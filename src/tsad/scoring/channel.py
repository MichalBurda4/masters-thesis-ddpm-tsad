"""Per-channel reconstruction error (multivariate attribution)."""

import numpy as np
import torch

from tsad.models.ddpm import DDPM


@torch.no_grad()
def score_window_channels(
    model: torch.nn.Module,
    ddpm: DDPM,
    x: torch.Tensor,
    start_t: int,
    reconstruct_method: str = "ddpm",
    ddim_steps: int | None = None,
    ddim_eta: float = 0.0,
) -> np.ndarray:
    """Returns (C, L) absolute error per channel and timestep."""
    x = x.unsqueeze(0) if x.dim() == 2 else x
    st = min(start_t, ddpm.T - 1)
    recon = ddpm.reconstruct(
        model,
        x,
        start_t=st,
        method=reconstruct_method,
        ddim_steps=ddim_steps,
        eta=ddim_eta,
    )
    err = (x - recon).abs().squeeze(0).cpu().numpy()  # (C, L)
    return err
