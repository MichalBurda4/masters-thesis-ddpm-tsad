"""Run train/eval pipeline across multiple benchmark configs."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import build_split
from tsad.eval.runner import evaluate_model

BENCHMARKS = [
    "configs/smd.yaml",
    "configs/yahoo.yaml",
]

TRAIN_SCRIPTS = {
    "ddpm": "train_ddpm.py",
    "lstm_ae": "train_lstm_ae.py",
    "transformer_ae": "train_transformer_ae.py",
    "tranad": "train_tranad.py",
}


def _row(dataset: str, model: str, ev: dict) -> dict:
    tp = ev["test"]["point"]
    seg = ev["test"].get("segment", {})
    return {
        "dataset": dataset,
        "model": model,
        "threshold_method": ev.get("threshold_method"),
        "test_f1": tp["f1"],
        "test_precision": tp["precision"],
        "test_recall": tp["recall"],
        "test_auc_roc": tp.get("auc_roc"),
        "range_f1": seg.get("range_f1"),
        "vus_f1_mean": seg.get("vus_f1_mean"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, default=",".join(BENCHMARKS))
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--models", type=str, default="ddpm,lstm_ae,transformer_ae,tranad")
    args = p.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_rows = []

    for cfg_path in configs:
        cfg = load_config(cfg_path)
        ds = cfg["data"]["name"]
        device = torch.device(
            cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
        )
        print(f"\n========== {ds} (device={device}) ==========")

        if args.train:
            for model in models:
                script = TRAIN_SCRIPTS.get(model)
                if script is None:
                    print(f"Skip train {model}: unknown model")
                    continue
                subprocess.check_call(
                    [
                        sys.executable,
                        str(ROOT / "experiments" / script),
                        "--config",
                        cfg_path,
                        "--epochs",
                        str(args.epochs),
                    ],
                    cwd=ROOT,
                )

        split = build_split(cfg)
        ckpt_dir = resolve_path(cfg, "checkpoint_dir")

        for model in models:
            ckpt = ckpt_dir / f"{model}_best.pt"
            if not ckpt.exists():
                print(f"Skip {model}: no checkpoint")
                continue
            ev = evaluate_model(model, ckpt, split, cfg, device, show_progress=True)
            row = _row(ds, model, ev)
            all_rows.append(row)
            print(
                f"{model}: F1={row['test_f1']:.4f} range_f1={row.get('range_f1')} "
                f"method={row['threshold_method']}"
            )

            out_dir = resolve_path(cfg, "results_dir")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"eval_{model}.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")

    master = ROOT / "results" / "master_summary.csv"
    master.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with open(master, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nSaved: {master}")


if __name__ == "__main__":
    main()
