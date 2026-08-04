"""Train TranAD baseline on normal windows."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.config import load_config, resolve_path
from tsad.data import WindowDataset, build_split
from tsad.models import TranAD


def _tranad_cfg(cfg: dict) -> dict:
    base = {
        "dim_feedforward": 16,
        "dropout": 0.1,
    }
    base.update(cfg.get("model", {}).get("tranad", {}))
    base["n_window"] = cfg["data"]["window_size"]
    return base


def train_tranad(cfg: dict) -> Path:
    device = torch.device(
        cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    )
    split = build_split(cfg)
    channels = split.n_channels
    tcfg = _tranad_cfg(cfg)
    n_window = tcfg.pop("n_window")
    train_ds = WindowDataset(split.train)
    val_ds = WindowDataset(split.val)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    model = TranAD(n_feats=channels, n_window=n_window, **tcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-5)

    out_dir = resolve_path(cfg, "checkpoint_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0
    patience = cfg["train"]["early_stop_patience"]

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"TranAD {epoch + 1}", leave=False):
            batch = batch.to(device)
            loss = model.training_loss(batch, epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(model.training_loss(batch.to(device), epoch).item())
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "model_type": "tranad",
                    "model": model.state_dict(),
                    "n_channels": channels,
                    "n_window": n_window,
                    "tranad_cfg": {**tcfg, "n_window": n_window},
                    "mean": split.mean,
                    "std": split.std,
                    "config": cfg,
                },
                out_dir / "tranad_best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break

    print(f"Saved: {out_dir / 'tranad_best.pt'}")
    return out_dir / "tranad_best.pt"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train_tranad(cfg)
