"""Sweep diffusion start_t (k) values and save F1 table."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import build_split
from tsad.eval.runner import evaluate_model

K_PRESETS = {
    "low": [10, 25, 40],
    "mid": [25, 50, 75],
    "high": [50, 75, 90],
    "single_25": [25],
    "single_50": [50],
    "single_75": [75],
}
QUICK_PRESETS = {"single_25": [25], "single_50": [50], "single_75": [75]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/smd.yaml")
    p.add_argument("--model", type=str, default="ddpm", choices=["ddpm", "lstm_ae"])
    p.add_argument("--quick", action="store_true", help="only 3 single-k runs")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"])
    ckpt = resolve_path(cfg, "checkpoint_dir") / f"{args.model}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}. Train first.")

    presets = QUICK_PRESETS if args.quick else K_PRESETS
    print(f"Ablation on {cfg['data']['name']}, presets={list(presets.keys())}", flush=True)
    split = build_split(cfg)
    results = {}
    for name, steps in presets.items():
        run_cfg = copy.deepcopy(cfg)
        run_cfg["scoring"]["diffusion_steps"] = steps
        out = evaluate_model(args.model, ckpt, split, run_cfg, device)
        results[name] = {
            "diffusion_steps": steps,
            "test_f1": out["test"]["point"]["f1"],
            "test_auc_roc": out["test"]["point"].get("auc_roc"),
            "val_f1": out["validation"]["point"]["f1"],
        }
        print(f"{name} {steps} -> test F1={results[name]['test_f1']:.4f}")

    out_dir = resolve_path(cfg, "results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"k_ablation_{args.model}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
