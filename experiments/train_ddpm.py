"""Train DDPM on normal windows only."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import WindowDataset, build_split
from tsad.models import DDPM, UNet1D
from tsad.train.ema import ModelEMA


def _ema_cfg(cfg: dict) -> dict:
    return cfg.get("train", {}).get(
        "ema",
        {"enabled": False, "decay": 0.999},
    )


def train_ddpm(cfg: dict) -> Path:
    device = torch.device(
        cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    )
    split = build_split(cfg)
    channels = split.n_channels
    train_ds = WindowDataset(split.train)
    val_ds = WindowDataset(split.val)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    model = UNet1D(
        channels=channels,
        base=cfg["model"]["base_channels"],
    ).to(device)
    ddpm = DDPM(timesteps=cfg["model"]["timesteps"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    ema_settings = _ema_cfg(cfg)
    ema = ModelEMA(model, decay=ema_settings["decay"]) if ema_settings.get("enabled") else None

    out_dir = resolve_path(cfg, "checkpoint_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience = cfg["train"]["early_stop_patience"]
    stale = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"DDPM {epoch + 1}/{cfg['train']['epochs']}", leave=False):
            batch = batch.to(device)
            loss = ddpm.p_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            train_losses.append(loss.item())

        eval_model = ema.module() if ema is not None else model
        eval_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_losses.append(ddpm.p_loss(eval_model, batch).item())
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch + 1}: train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            ckpt = {
                "model_type": "ddpm",
                "model": eval_model.state_dict(),
                "use_ema": ema is not None,
                "ema_decay": ema_settings.get("decay") if ema is not None else None,
                "mean": split.mean,
                "std": split.std,
                "timesteps": ddpm.T,
                "n_channels": channels,
                "config": cfg,
            }
            path = out_dir / "ddpm_best.pt"
            torch.save(ckpt, path)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop at epoch {epoch + 1}")
                break

    meta = out_dir / "ddpm_meta.json"
    meta.write_text(
        json.dumps(
            {
                "best_val_loss": best_val,
                "use_ema": ema is not None,
                "ema_decay": ema_settings.get("decay") if ema is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {out_dir / 'ddpm_best.pt'}")
    return out_dir / "ddpm_best.pt"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--machine", type=str, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.machine:
        cfg.setdefault("data", {})["machine"] = args.machine
        cfg["paths"]["checkpoint_dir"] = str(
            Path(cfg["paths"]["checkpoint_dir"]) / args.machine
        )
    train_ddpm(cfg)
