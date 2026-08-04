"""Anomaly detection + simple time-point explainability (reconstruction error map)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from data import make_test_set
from ddpm import DDPM
from model import UNet1D


def score_series(
    model: torch.nn.Module,
    ddpm: DDPM,
    series: np.ndarray,
    device: torch.device,
    noise_t: int | None = None,
) -> dict:
    """
    Returns:
      - series_score: global anomaly score (higher = more anomalous)
      - point_scores: per-timestep |x - x_recon| (explainability heatmap)
    """
    x = torch.from_numpy(series).float().unsqueeze(0).unsqueeze(0).to(device)
    recon = ddpm.reconstruct(model, x, start_t=noise_t)
    point_err = (x - recon).abs().squeeze().cpu().numpy()
    series_score = float(point_err.mean() + point_err.max() * 0.5)
    return {"series_score": series_score, "point_scores": point_err, "reconstruction": recon.squeeze().cpu().numpy()}


def evaluate_checkpoint(ckpt_path: Path, device: str = "cpu"):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = UNet1D().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(timesteps=ckpt.get("timesteps", 100))
    test = make_test_set()
    noise_t = ddpm.T // 2

    results = []
    for i, series in enumerate(test["series"]):
        s = score_series(model, ddpm, series, device, noise_t=noise_t)
        results.append({**s, "label": int(test["label"][i])})

    scores = np.array([r["series_score"] for r in results])
    labels = np.array([r["label"] for r in results])

    # Simple threshold at median of all scores (prototype only)
    thresh = np.median(scores)
    preds = (scores > thresh).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Threshold (median score): {thresh:.4f}")
    print(f"Accuracy: {acc:.2%}  Precision: {prec:.2%}  Recall: {rec:.2%}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

    return results, test, thresh


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/ddpm_prototype.pt")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    evaluate_checkpoint(Path(args.ckpt), args.device)
