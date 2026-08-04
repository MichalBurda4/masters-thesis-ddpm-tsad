"""Evaluate DDPM vs LSTM-AE; threshold tuned on validation."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import build_split
from tsad.eval.runner import evaluate_model


def main(cfg: dict):
    device = torch.device(cfg["device"])
    split = build_split(cfg)
    ckpt_dir = resolve_path(cfg, "checkpoint_dir")
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {"dataset": cfg["data"]["name"], "n_channels": split.n_channels, "models": {}}

    for model_name, fname in [("ddpm", "ddpm_best.pt"), ("lstm_ae", "lstm_ae_best.pt")]:
        path = ckpt_dir / fname
        if not path.exists():
            print(f"Skip {model_name}: missing {path}")
            continue
        print(f"\n=== {model_name.upper()} ===")
        ev = evaluate_model(model_name, path, split, cfg, device)
        print(json.dumps(ev["test"]["point"], indent=2))
        report["models"][model_name] = ev

    out = results_dir / "metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    main(load_config(args.config))
