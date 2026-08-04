"""Minimal DDPM for 1D time series (noise prediction objective)."""

import math
import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_var = betas * (1.0 - torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])) / (1.0 - alpha_bar)

    def _gather(self, coeffs: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = coeffs.to(x.device)[t].view(-1, 1, 1)
        return c

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._gather(self.sqrt_alpha_bar, t, x0)
        sqrt_omb = self._gather(self.sqrt_one_minus_alpha_bar, t, x0)
        xt = sqrt_ab * x0 + sqrt_omb * noise
        return xt, noise

    def p_loss(self, model: torch.nn.Module, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def denoise_step(self, model: torch.nn.Module, xt: torch.Tensor, t: int) -> torch.Tensor:
        """Single reverse step: x_{t-1} from x_t."""
        batch = xt.shape[0]
        t_tensor = torch.full((batch,), t, device=xt.device, dtype=torch.long)
        beta_t = self.betas[t].to(xt.device)
        alpha_t = self.alphas[t].to(xt.device)
        alpha_bar_t = self.alpha_bar[t].to(xt.device)

        pred_noise = model(xt, t_tensor)
        mean = self.sqrt_recip_alphas[t].to(xt.device) * (
            xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        )
        if t == 0:
            return mean
        noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
        var = self.posterior_var[t].to(xt.device)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def reconstruct(self, model: torch.nn.Module, x0: torch.Tensor, start_t: int | None = None) -> torch.Tensor:
        """Partial forward noise then full reverse — used for anomaly scoring."""
        if start_t is None:
            start_t = self.T // 2
        b = x0.shape[0]
        t_start = torch.full((b,), start_t, device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t_start, noise)
        for t in reversed(range(start_t + 1)):
            xt = self.denoise_step(model, xt, t)
        return xt
