"""DRE (Diffusion Reconstruction Error) + DAM (Denoising Attribution Map)."""

import numpy as np
import torch

from tsad.models.ddpm import DDPM


@torch.no_grad()
def score_window_dre(
    model: torch.nn.Module,
    ddpm: DDPM,
    x: torch.Tensor,
    start_steps: list[int],
    reconstruct_method: str = "ddpm",
    ddim_steps: int | None = None,
    ddim_eta: float = 0.0,
) -> dict:
    """
    x: (1, C, L)
    Returns point-wise scores aggregated over diffusion start steps.
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    errors_per_t = []
    dam_rows = []

    for st in start_steps:
        st = min(st, ddpm.T - 1)
        recon = ddpm.reconstruct(
            model,
            x,
            start_t=st,
            method=reconstruct_method,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
        )
        err = (x - recon).abs().mean(dim=1).squeeze(0).cpu().numpy()  # (L,)
        errors_per_t.append(err)
        dam_rows.append(err)

    dam = np.stack(dam_rows, axis=0)  # (n_steps, L)
    point_scores = dam.mean(axis=0)
    # Emphasize peak errors (helps point-level detection)
    point_scores = 0.6 * point_scores + 0.4 * dam.max(axis=0)
    series_score = float(point_scores.mean() + point_scores.max() * 0.3)

    return {
        "point_scores": point_scores,
        "series_score": series_score,
        "dam": dam,
        "reconstruction": ddpm.reconstruct(
            model,
            x,
            start_t=start_steps[len(start_steps) // 2],
            method=reconstruct_method,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
        )
        .squeeze()
        .cpu()
        .numpy(),
    }


@torch.no_grad()
def score_window_tranad(model: torch.nn.Module, x: torch.Tensor) -> dict:
    x = x.unsqueeze(0) if x.dim() == 2 else x
    recon = model.reconstruct(x)
    err = (x - recon).abs().mean(dim=1).squeeze(0).cpu().numpy()
    return {
        "point_scores": err,
        "series_score": float(err.mean() + err.max() * 0.3),
        "reconstruction": recon.squeeze().cpu().numpy(),
    }


@torch.no_grad()
def score_window_reconstruction(model: torch.nn.Module, x: torch.Tensor) -> dict:
    x = x.unsqueeze(0) if x.dim() == 2 else x
    recon = model(x)
    err = (x - recon).abs().mean(dim=1).squeeze(0).cpu().numpy()
    return {
        "point_scores": err,
        "series_score": float(err.mean() + err.max() * 0.3),
        "reconstruction": recon.squeeze().cpu().numpy(),
    }


@torch.no_grad()
def score_window_lstm_ae(model: torch.nn.Module, x: torch.Tensor) -> dict:
    return score_window_reconstruction(model, x)
