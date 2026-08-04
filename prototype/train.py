"""Train DDPM on normal time series only."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import NormalSeriesDataset
from ddpm import DDPM
from model import UNet1D


def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    out_dir: str = "checkpoints",
):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = NormalSeriesDataset(n_samples=800, length=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet1D().to(device)
    ddpm = DDPM(timesteps=100)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            batch = batch.to(device)
            loss = ddpm.p_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch + 1}: loss = {sum(losses) / len(losses):.4f}")

    ckpt = {
        "model": model.state_dict(),
        "mean": dataset.mean,
        "std": dataset.std,
        "timesteps": ddpm.T,
    }
    path = out / "ddpm_prototype.pt"
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
