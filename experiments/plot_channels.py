"""Channel × time attribution heatmap (multivariate)."""

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
from tsad.eval.runner import load_model
from tsad.scoring.channel import score_window_channels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/smd.yaml")
    p.add_argument("--index", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"])
    split = build_split(cfg)
    if split.n_channels < 2:
        print("Need multivariate data for channel plot.")
        return

    ckpt = resolve_path(cfg, "checkpoint_dir") / "ddpm_best.pt"
    model, ddpm, channels, _ = load_model("ddpm", ckpt, cfg, device)
    steps = cfg["scoring"]["diffusion_steps"]
    k = steps[len(steps) // 2]

    if args.index is None:
        idxs = np.where(split.test_window_labels > 0.5)[0]
        index = int(idxs[0]) if len(idxs) else 0
    else:
        index = args.index

    window = split.test[index]
    mask = split.test_point_labels[index]
    ch_err = score_window_channels(model, ddpm, window_to_tensor(window, device), k)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = np.arange(window.shape[0])

    axes[0].imshow(ch_err, aspect="auto", cmap="magma", origin="lower")
    axes[0].set_ylabel("channel")
    axes[0].set_title(f"Channel attribution (|error|) — window #{index}, k={k}")

    axes[1].plot(t, ch_err.mean(axis=0), color="crimson")
    axes[1].set_ylabel("mean |err|")
    axes[1].set_title("Aggregated DRE over channels")

    axes[2].fill_between(t, 0, mask, color="green", alpha=0.35, label="GT")
    norm = ch_err.mean(axis=0)
    norm = norm / (norm.max() + 1e-8)
    axes[2].plot(t, norm, color="black", lw=1, label="score")
    axes[2].legend()
    axes[2].set_xlabel("time step")

    out = resolve_path(cfg, "output_dir") / "channel_attribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
