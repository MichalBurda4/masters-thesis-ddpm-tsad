"""Plot DRE + DAM heatmap for an anomalous test window."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import build_split
from tsad.data.tensor_utils import window_to_tensor
from tsad.models import DDPM, UNet1D
from tsad.scoring.diffusion import score_window_dre


def _plot_series(window: np.ndarray) -> np.ndarray:
    if window.ndim == 1:
        return window
    return window.mean(axis=1)


def plot_dam(cfg: dict, window_idx: int | None = None):
    device = torch.device(cfg["device"])
    split = build_split(cfg)
    ckpt_path = resolve_path(cfg, "checkpoint_dir") / "ddpm_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    channels = ckpt.get("n_channels") or cfg["model"]["channels"]

    model = UNet1D(channels=channels, base=cfg["model"]["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ddpm = DDPM(timesteps=ckpt.get("timesteps", 100))

    if window_idx is None:
        anom_idx = np.where(split.test_window_labels > 0.5)[0]
        window_idx = int(anom_idx[0]) if len(anom_idx) else 0

    window = split.test[window_idx]
    mask = split.test_point_labels[window_idx]
    series = _plot_series(window)
    r = score_window_dre(model, ddpm, window_to_tensor(window, device), cfg["scoring"]["diffusion_steps"])

    t = np.arange(len(series))
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1.2, 1.5, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, series, label="Input (mean ch.)", color="steelblue")
    recon = r["reconstruction"]
    if recon.ndim > 1:
        recon = recon.mean(axis=0)
    ax0.plot(t, recon, "--", label="DDPM recon", color="coral")
    ax0.set_title(f"Test window #{window_idx} — DRE + DAM")
    ax0.legend()

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.fill_between(t, 0, r["point_scores"], color="crimson", alpha=0.5)
    ax1.set_ylabel("DRE")
    ax1.set_title("Diffusion Reconstruction Error")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    im = ax2.imshow(
        r["dam"],
        aspect="auto",
        cmap="magma",
        extent=[0, len(series), 0, len(cfg["scoring"]["diffusion_steps"])],
        origin="lower",
    )
    ax2.set_ylabel("diffusion step idx")
    ax2.set_title("DAM: time × start_t")
    plt.colorbar(im, ax=ax2, label="|error|")

    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.fill_between(t, 0, mask, color="green", alpha=0.35, label="Ground truth")
    norm = r["point_scores"] / (r["point_scores"].max() + 1e-8)
    ax3.plot(t, norm, color="black", lw=1, label="DRE norm.")
    ax3.legend()
    ax3.set_xlabel("Time step")

    out = resolve_path(cfg, "output_dir") / "dam_example.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--index", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    plot_dam(cfg, args.index)
