"""Compare DDPM vs reconstruction baselines; export summary CSV."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import build_split
from tsad.eval.runner import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"])
    split = build_split(cfg)
    ckpt_dir = resolve_path(cfg, "checkpoint_dir")
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {"dataset": cfg["data"]["name"], "n_channels": split.n_channels, "models": {}}
    rows = []

    for model_name in ("ddpm", "lstm_ae", "transformer_ae", "tranad"):
        ckpt = ckpt_dir / f"{model_name}_best.pt"
        if not ckpt.exists():
            print(f"Skip {model_name}: {ckpt} not found")
            continue
        ev = evaluate_model(model_name, ckpt, split, cfg, device)
        report["models"][model_name] = ev
        tp = ev["test"]["point"]
        rows.append(
            {
                "dataset": cfg["data"]["name"],
                "model": model_name,
                "threshold_method": ev.get("threshold_method"),
                "test_f1": tp["f1"],
                "test_precision": tp["precision"],
                "test_recall": tp["recall"],
                "test_auc_roc": tp.get("auc_roc"),
                "test_auc_pr": tp.get("auc_pr"),
                "range_f1": ev["test"].get("segment", {}).get("range_f1"),
                "vus_f1_mean": ev["test"].get("segment", {}).get("vus_f1_mean"),
            }
        )
        print(f"{model_name}: test F1={tp['f1']:.4f} P={tp['precision']:.4f} R={tp['recall']:.4f}")

    json_path = results_dir / "compare.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = results_dir / "summary.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
