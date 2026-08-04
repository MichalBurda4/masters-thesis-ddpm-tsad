"""Evaluate DDPM on multiple SMD machines; aggregate metrics."""

import argparse
import csv
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import build_split
from tsad.data.smd import list_machines
from tsad.eval.runner import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/smd.yaml")
    p.add_argument("--machines", type=str, default="machine-1-1,machine-1-2,machine-1-3")
    p.add_argument("--train", action="store_true", help="train missing checkpoints")
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"])
    project = Path(cfg["_project_root"])
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]

    rows = []
    for machine in machines:
        print(f"\n=== {machine} ===")
        run_cfg = deepcopy(cfg)
        run_cfg["data"]["machine"] = machine
        run_cfg["paths"]["checkpoint_dir"] = f"{cfg['paths']['checkpoint_dir']}/{machine}"
        run_cfg["paths"]["results_dir"] = f"{cfg['paths']['results_dir']}/{machine}"

        ckpt_dir = project / run_cfg["paths"]["checkpoint_dir"]
        ckpt = ckpt_dir / "ddpm_best.pt"

        if args.train and not ckpt.exists():
            subprocess.check_call(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "train_ddpm.py"),
                    "--config",
                    args.config,
                    "--machine",
                    machine,
                    "--epochs",
                    str(args.epochs),
                ],
                cwd=project,
            )

        if not ckpt.exists():
            print(f"Missing {ckpt}")
            continue

        split = build_split(run_cfg)
        ev = evaluate_model("ddpm", ckpt, split, run_cfg, device)
        f1 = ev["test"]["point"]["f1"]
        rows.append(
            {
                "machine": machine,
                "test_f1": f1,
                "test_auc": ev["test"]["point"].get("auc_roc"),
                "test_precision": ev["test"]["point"]["precision"],
            }
        )
        print(f"test F1={f1:.4f}")

    out_dir = project / cfg["paths"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = out_dir / "smd_machines.csv"
    if rows:
        with open(agg, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {agg}")


if __name__ == "__main__":
    main()
