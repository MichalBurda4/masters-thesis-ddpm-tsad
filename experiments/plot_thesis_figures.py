"""Generate publication-ready figures for the master's thesis.

Outputs (PNG + PDF) go to outputs/thesis_figures/ by default.

Examples:
  python experiments/plot_thesis_figures.py --config configs/smd.yaml
  python experiments/plot_thesis_figures.py --config configs/smd.yaml --skip-model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import build_split
from tsad.data.tensor_utils import window_to_tensor
from tsad.eval.runner import load_model
from tsad.scoring.channel import score_window_channels
from tsad.scoring.diffusion import score_window_dre

# Thesis-friendly palette (avoid default "AI purple")
C_BLUE = "#1f4e79"
C_TEAL = "#2a9d8f"
C_CORAL = "#e76f51"
C_AMBER = "#e9c46a"
C_SLATE = "#264653"
C_GRAY = "#6c757d"
C_GREEN = "#2d6a4f"
MODEL_COLORS = {
    "ddpm": C_BLUE,
    "lstm_ae": C_TEAL,
    "transformer_ae": C_CORAL,
    "tranad": C_AMBER,
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


def _pretty_model(name: str) -> str:
    return {
        "ddpm": "DDPM",
        "lstm_ae": "LSTM-AE",
        "transformer_ae": "Transformer-AE",
        "tranad": "TranAD",
    }.get(name, name)


def plot_model_comparison(summary_csv: Path, out_dir: Path) -> None:
    if not summary_csv.exists():
        print(f"Skip comparison: missing {summary_csv}")
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        print("Skip comparison: empty summary.csv")
        return

    models = [_pretty_model(m) for m in df["model"]]
    colors = [MODEL_COLORS.get(m, C_GRAY) for m in df["model"]]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width, df["test_precision"], width, label="Precision", color=C_TEAL, edgecolor="white")
    ax.bar(x, df["test_recall"], width, label="Recall", color=C_CORAL, edgecolor="white")
    ax.bar(x + width, df["test_f1"], width, label="F1", color=C_BLUE, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SMD — model comparison (point metrics)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    for i, f1 in enumerate(df["test_f1"]):
        ax.text(i + width, float(f1) + 0.02, f"{f1:.2f}", ha="center", va="bottom", fontsize=8, color=C_BLUE)
    _save(fig, out_dir, "fig_model_comparison")

    # AUC companion figure
    if "test_auc_roc" in df.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        vals = df["test_auc_roc"].astype(float)
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.65)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC-ROC")
        ax.set_title("SMD — AUC-ROC by model")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        _save(fig, out_dir, "fig_auc_roc")


def plot_k_ablation(ablation_json: Path, out_dir: Path) -> None:
    if not ablation_json.exists():
        print(f"Skip k-ablation: missing {ablation_json}")
        return

    data = json.loads(ablation_json.read_text(encoding="utf-8"))
    labels, f1s, aucs = [], [], []
    for key, row in data.items():
        steps = row.get("diffusion_steps", [])
        labels.append(f"k={steps[0]}" if len(steps) == 1 else key)
        f1s.append(float(row["test_f1"]))
        aucs.append(float(row.get("test_auc_roc", np.nan)))

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(x, f1s, "o-", color=C_BLUE, lw=2, markersize=8, label="Test F1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("F1", color=C_BLUE)
    ax1.set_ylim(min(f1s) - 0.02, max(f1s) + 0.03)
    ax1.set_title("Ablation of partial diffusion depth $k$ (DDPM on SMD)")

    if not np.all(np.isnan(aucs)):
        ax2 = ax1.twinx()
        ax2.plot(x, aucs, "s--", color=C_CORAL, lw=1.8, markersize=7, label="AUC-ROC")
        ax2.set_ylabel("AUC-ROC", color=C_CORAL)
        ax2.spines["top"].set_visible(False)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")
    else:
        ax1.legend(frameon=False)

    for i, v in enumerate(f1s):
        ax1.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _save(fig, out_dir, "fig_k_ablation")


def plot_training_curve(meta_json: Path, out_dir: Path) -> None:
    """If meta only has best_val_loss, skip; support optional history if present."""
    if not meta_json.exists():
        return
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    history = meta.get("history") or meta.get("val_losses")
    if not history:
        # Single-point summary card instead of fake curve
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.axis("off")
        text = (
            f"DDPM training summary ({meta_json.parent.name})\n\n"
            f"Best validation loss: {meta.get('best_val_loss', 'n/a'):.4f}\n"
            f"EMA enabled: {meta.get('use_ema', 'n/a')}\n"
            f"EMA decay: {meta.get('ema_decay', 'n/a')}"
        )
        ax.text(0.05, 0.55, text, fontsize=12, va="center", family="DejaVu Sans", color=C_SLATE)
        ax.set_title("Training checkpoint metadata")
        _save(fig, out_dir, "fig_training_meta")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(history, color=C_BLUE, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("DDPM validation loss")
    _save(fig, out_dir, "fig_training_curve")


def _pick_window(split, index: int | None) -> int:
    if index is not None:
        return index
    anom = np.where(split.test_window_labels > 0.5)[0]
    return int(anom[0]) if len(anom) else 0


def plot_case_study(cfg: dict, out_dir: Path, index: int | None, skip_model: bool) -> None:
    if skip_model:
        print("Skip DAM/channel case study (--skip-model)")
        return

    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")
    split = build_split(cfg)
    ckpt = resolve_path(cfg, "checkpoint_dir") / "ddpm_best.pt"
    if not ckpt.exists():
        print(f"Skip case study: missing {ckpt}")
        return

    model, ddpm, _, _ = load_model("ddpm", ckpt, cfg, device)
    steps = cfg["scoring"]["diffusion_steps"]
    wi = _pick_window(split, index)
    window = split.test[wi]
    mask = split.test_point_labels[wi]
    series = window if window.ndim == 1 else window.mean(axis=1)
    t = np.arange(len(series))

    r = score_window_dre(model, ddpm, window_to_tensor(window, device), steps)
    recon = r["reconstruction"]
    if getattr(recon, "ndim", 1) > 1:
        recon = recon.mean(axis=0) if recon.shape[0] != len(series) else recon.mean(axis=0)

    # ---- Figure: DAM case study ----
    fig = plt.figure(figsize=(9.5, 8.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.15, 1.7, 1.0], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, series, color=C_BLUE, lw=1.6, label="Input (channel mean)")
    ax0.plot(t, recon, color=C_CORAL, lw=1.4, ls="--", label="DDPM reconstruction")
    ymax = max(float(np.max(series)), float(np.max(recon)))
    ymin = min(float(np.min(series)), float(np.min(recon)))
    ax0.fill_between(t, ymin, ymax, where=mask > 0.5, color=C_GREEN, alpha=0.12, label="GT anomaly")
    ax0.set_title(f"Case study — SMD test window #{wi}")
    ax0.set_ylabel("Normalized value")
    ax0.legend(frameon=False, ncol=3, loc="upper right")

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.fill_between(t, 0, r["point_scores"], color=C_CORAL, alpha=0.55)
    ax1.plot(t, r["point_scores"], color=C_SLATE, lw=1.0)
    ax1.set_ylabel("DRE")
    ax1.set_title("Diffusion Reconstruction Error (point-wise)")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    dam = r["dam"]
    im = ax2.imshow(
        dam,
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=[0, len(series), -0.5, dam.shape[0] - 0.5],
        interpolation="nearest",
    )
    ax2.set_yticks(range(dam.shape[0]))
    ax2.set_yticklabels([str(s) for s in steps])
    ax2.set_ylabel(r"start $t$")
    ax2.set_title("DAM — reconstruction error over diffusion start steps")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    cbar.set_label("|error|")

    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    score_n = r["point_scores"] / (r["point_scores"].max() + 1e-8)
    ax3.fill_between(t, 0, mask, color=C_GREEN, alpha=0.35, label="Ground truth")
    ax3.plot(t, score_n, color="black", lw=1.2, label="DRE (normalized)")
    ax3.set_xlabel("Time step within window")
    ax3.set_ylabel("Score / label")
    ax3.set_ylim(-0.05, 1.15)
    ax3.legend(frameon=False, ncol=2, loc="upper right")
    _save(fig, out_dir, "fig_dam_case_study")

    # ---- Figure: channel attribution ----
    if split.n_channels >= 2:
        k = steps[len(steps) // 2]
        ch_err = score_window_channels(model, ddpm, window_to_tensor(window, device), k)
        # show top channels by mean error for readability
        order = np.argsort(ch_err.mean(axis=1))[::-1]
        top_n = min(16, ch_err.shape[0])
        top = order[:top_n]
        ch_show = ch_err[top]

        fig, axes = plt.subplots(
            3, 1, figsize=(9.5, 7.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.1, 1.0]}
        )
        im = axes[0].imshow(ch_show, aspect="auto", cmap="magma", origin="upper", interpolation="nearest")
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels([f"ch {i}" for i in top], fontsize=8)
        axes[0].set_ylabel("Channel (top by error)")
        axes[0].set_title(f"Channel attribution — window #{wi}, $k$={k}")
        fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02, label="|error|")

        axes[1].plot(t, ch_err.mean(axis=0), color=C_CORAL, lw=1.6)
        axes[1].set_ylabel("Mean |err|")
        axes[1].set_title("Error aggregated over all channels")

        axes[2].fill_between(t, 0, mask, color=C_GREEN, alpha=0.35, label="Ground truth")
        sn = ch_err.mean(axis=0)
        sn = sn / (sn.max() + 1e-8)
        axes[2].plot(t, sn, color="black", lw=1.2, label="Normalized score")
        axes[2].set_xlabel("Time step within window")
        axes[2].set_ylim(-0.05, 1.15)
        axes[2].legend(frameon=False, ncol=2)
        fig.tight_layout()
        _save(fig, out_dir, "fig_channel_attribution")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate thesis figures")
    p.add_argument("--config", type=str, default="configs/smd.yaml")
    p.add_argument("--index", type=int, default=None, help="Test window index for case study")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--skip-model", action="store_true", help="Only plot CSV/JSON figures (fast)")
    args = p.parse_args()

    _setup_style()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "thesis_figures"
    results = resolve_path(cfg, "results_dir")

    print(f"Writing figures to: {out_dir}")
    plot_model_comparison(results / "summary.csv", out_dir)
    plot_k_ablation(results / "k_ablation_ddpm.json", out_dir)

    # Prefer full-run meta if present
    for meta_candidate in (
        ROOT / "checkpoints" / "smd_full" / "ddpm_meta.json",
        resolve_path(cfg, "checkpoint_dir") / "ddpm_meta.json",
    ):
        if meta_candidate.exists():
            plot_training_curve(meta_candidate, out_dir)
            break

    plot_case_study(cfg, out_dir, args.index, args.skip_model)
    print("Done.")


if __name__ == "__main__":
    main()
