"""Exponential moving average of model weights (common DDPM stabilizer)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain a shadow copy of weights: ema = decay * ema + (1 - decay) * model."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(d).add_(model_p, alpha=1.0 - d)

    def module(self) -> nn.Module:
        return self.shadow
