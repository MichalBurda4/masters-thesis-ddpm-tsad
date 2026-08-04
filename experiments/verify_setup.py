"""Quick health check: datasets, splits, checkpoints."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path

CONFIGS = [
    ("SMD", "configs/smd.yaml"),
    ("Yahoo", "configs/yahoo.yaml"),
    ("Weather", "configs/weather.yaml"),
    ("SensorDrift", "configs/sensor_drift.yaml"),
]


def check_data(cfg: dict) -> tuple[bool, list[str]]:
    lines: list[str] = []
    d = cfg["data"]
    ok = True
    project = Path(cfg["_project_root"])

    if d["name"] == "smd":
        root = project / d["smd_root"]
        machine = d.get("machine", "machine-1-1")
        for sub in ("train", "test", "test_label"):
            p = root / sub / f"{machine}.txt"
            exists = p.exists()
            lines.append(f"{sub}: {'OK' if exists else 'MISSING'} ({p})")
            ok &= exists
    elif d["name"] == "sensor_drift":
        base = project / d.get("data_dir", "data/sensor_drift")
        exists = base.exists()
        files = [f for f in base.rglob("*") if f.is_file() and f.name != "sensor_drift.zip"]
        lines.append(f"data_dir: {'OK' if exists else 'MISSING'} ({base})")
        lines.append(f"files found: {len(files)}")
        ok = exists and len(files) > 0
    else:
        p = project / d["csv_path"]
        lines.append(f"csv: {'OK' if p.exists() else 'MISSING'} ({p})")
        ok = p.exists()

    return ok, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true", help="Run quick DDPM eval on SMD (slow on CPU)")
    args = p.parse_args()

    print("=== DATASET HEALTH CHECK ===\n", flush=True)
    all_data_ok = True

    for name, cfg_path in CONFIGS:
        cfg = load_config(str(ROOT / cfg_path))
        data_ok, data_lines = check_data(cfg)
        split_ok = False
        split_lines: list[str] = []

        if data_ok:
            try:
                from tsad.data import build_split

                split = build_split(cfg)
                split_ok = True
                split_lines = [
                    f"train windows: {len(split.train)}",
                    f"val windows: {len(split.val)}",
                    f"test windows: {len(split.test)}",
                    f"channels: {split.n_channels}",
                    f"test anomaly window rate: {split.test_window_labels.mean():.3f}",
                ]
            except Exception as e:
                split_lines = [f"split ERROR: {e}"]
        else:
            split_lines = ["split: SKIPPED (download data first)"]

        ok = data_ok and split_ok
        all_data_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for line in data_lines + split_lines:
            print(f"  {line}")
        print()

    print("=== CHECKPOINTS ===")
    ckpts = sorted((ROOT / "checkpoints").rglob("*_best.pt"))
    if ckpts:
        for p in ckpts:
            print(f"  {p.relative_to(ROOT)}")
    else:
        print("  Brak checkpointów — uruchom trening (np. train_ddpm.py)")

    print("\n=== QUICK EVAL (if DDPM checkpoint exists) ===")
    smd_ckpt = ROOT / "checkpoints" / "smd" / "ddpm_best.pt"
    if args.eval and smd_ckpt.exists():
        import torch
        from tsad.data import build_split
        from tsad.eval.runner import evaluate_model

        cfg = load_config(str(ROOT / "configs/smd.yaml"))
        split = build_split(cfg)
        ev = evaluate_model("ddpm", smd_ckpt, split, cfg, torch.device("cpu"))
        tp = ev["test"]["point"]
        print(f"  SMD DDPM: F1={tp['f1']:.4f} P={tp['precision']:.4f} R={tp['recall']:.4f}")
    elif args.eval:
        print("  Skip — brak checkpoints/smd/ddpm_best.pt")
    else:
        print("  Skip (use --eval to run; slow on CPU)")

    print()
    if all_data_ok:
        print("All datasets ready.")
    else:
        print("Some datasets need download - see commands below.")


if __name__ == "__main__":
    main()
