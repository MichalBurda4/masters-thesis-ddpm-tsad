"""Plot example: signal, reconstruction, anomaly heatmap."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import make_test_set
from detect import score_series
from ddpm import DDPM
from model import UNet1D


def plot_example(ckpt_path: Path, index: int | None = None, out: Path = Path("outputs/example.png")):
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet1D().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ddpm = DDPM(timesteps=ckpt.get("timesteps", 100))

    test = make_test_set()
    if index is None:
        # First anomalous sample
        index = int(np.where(test["label"] == 1)[0][0])

    series = test["series"][index]
    label = test["label"][index]
    mask = test["mask"][index]
    res = score_series(model, ddpm, series, device, noise_t=ddpm.T // 2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t = np.arange(len(series))

    axes[0].plot(t, series, label="Input", color="steelblue")
    axes[0].plot(t, res["reconstruction"], "--", label="DDPM reconstruction", color="coral")
    axes[0].set_title(f"Sample #{index} — label: {'ANOMALY' if label else 'NORMAL'}")
    axes[0].legend()
    axes[0].set_ylabel("Value (norm.)")

    axes[1].fill_between(t, 0, res["point_scores"], color="crimson", alpha=0.5)
    axes[1].set_ylabel("|error|")
    axes[1].set_title("Explainability: per-timestep reconstruction error")

    axes[2].fill_between(t, 0, mask, color="green", alpha=0.4, label="True anomaly mask")
    axes[2].plot(t, res["point_scores"] / (res["point_scores"].max() + 1e-8), color="black", lw=1, label="Normalized error")
    axes[2].set_xlabel("Time step")
    axes[2].legend()
    axes[2].set_title("Ground truth vs model attribution")

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/ddpm_prototype.pt")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--out", type=str, default="outputs/example.png")
    args = p.parse_args()
    plot_example(Path(args.ckpt), args.index, Path(args.out))
