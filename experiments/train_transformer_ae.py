"""Train Transformer autoencoder baseline on normal windows."""

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
from tsad.models import TransformerAutoencoder


def _transformer_cfg(cfg: dict) -> dict:
    return cfg.get("model", {}).get(
        "transformer_ae",
        {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "latent_size": 32,
            "dropout": 0.1,
        },
    )


def train_transformer_ae(cfg: dict) -> Path:
    device = torch.device(
        cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    )
    split = build_split(cfg)
    channels = split.n_channels
    tcfg = _transformer_cfg(cfg)
    train_ds = WindowDataset(split.train)
    val_ds = WindowDataset(split.val)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    model = TransformerAutoencoder(input_size=channels, **tcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    out_dir = resolve_path(cfg, "checkpoint_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0
    patience = cfg["train"]["early_stop_patience"]

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"Transformer-AE {epoch + 1}", leave=False):
            batch = batch.to(device)
            loss = model.reconstruction_loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(model.reconstruction_loss(batch.to(device)).item())
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"Epoch {epoch + 1}: val_recon={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "model_type": "transformer_ae",
                    "model": model.state_dict(),
                    "n_channels": channels,
                    "transformer_cfg": tcfg,
                    "mean": split.mean,
                    "std": split.std,
                    "config": cfg,
                },
                out_dir / "transformer_ae_best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break

    print(f"Saved: {out_dir / 'transformer_ae_best.pt'}")
    return out_dir / "transformer_ae_best.pt"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train_transformer_ae(cfg)
