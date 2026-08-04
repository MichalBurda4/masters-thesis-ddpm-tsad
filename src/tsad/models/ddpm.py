import numpy as np
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
        return coeffs.to(x.device)[t].view(-1, 1, 1)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._gather(self.sqrt_alpha_bar, t, x0)
        sqrt_omb = self._gather(self.sqrt_one_minus_alpha_bar, t, x0)
        return sqrt_ab * x0 + sqrt_omb * noise, noise

    def p_loss(self, model: torch.nn.Module, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        return F.mse_loss(model(xt, t), noise)

    @torch.no_grad()
    def denoise_step(self, model: torch.nn.Module, xt: torch.Tensor, t: int) -> torch.Tensor:
        batch = xt.shape[0]
        t_tensor = torch.full((batch,), t, device=xt.device, dtype=torch.long)
        beta_t = self.betas[t].to(xt.device)
        alpha_bar_t = self.alpha_bar[t].to(xt.device)
        pred_noise = model(xt, t_tensor)
        mean = self.sqrt_recip_alphas[t].to(xt.device) * (
            xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        )
        if t == 0:
            return mean
        noise = torch.randn_like(xt)
        var = self.posterior_var[t].to(xt.device)
        return mean + torch.sqrt(var) * noise

    def _predict_x0(self, model: torch.nn.Module, xt: torch.Tensor, t: int) -> torch.Tensor:
        batch = xt.shape[0]
        t_tensor = torch.full((batch,), t, device=xt.device, dtype=torch.long)
        alpha_bar_t = self.alpha_bar[t].to(xt.device)
        pred_noise = model(xt, t_tensor)
        return (xt - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

    @torch.no_grad()
    def ddim_step(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        alpha_bar_t = self.alpha_bar[t].to(xt.device)
        alpha_bar_prev = self.alpha_bar[t_prev].to(xt.device) if t_prev >= 0 else torch.tensor(1.0, device=xt.device)
        pred_x0 = self._predict_x0(model, xt, t)
        if t_prev < 0:
            return pred_x0

        sigma_t = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        )
        batch = xt.shape[0]
        t_tensor = torch.full((batch,), t, device=xt.device, dtype=torch.long)
        pred_noise = model(xt, t_tensor)
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * pred_noise
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        if eta > 0 and t_prev > 0:
            x_prev = x_prev + sigma_t * torch.randn_like(xt)
        return x_prev

    @torch.no_grad()
    def reconstruct_ddim(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        start_t: int,
        num_steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Deterministic/faster reconstruction with DDIM subsampling."""
        b = x0.shape[0]
        t_start = min(start_t, self.T - 1)
        noise = torch.randn_like(x0)
        t_tensor = torch.full((b,), t_start, device=x0.device, dtype=torch.long)
        xt, _ = self.q_sample(x0, t_tensor, noise)

        if num_steps is None or num_steps >= t_start + 1:
            times = list(range(t_start, -1, -1))
        else:
            times = np.linspace(t_start, 0, num_steps, dtype=int).tolist()
            times = sorted(set(times), reverse=True)

        for i, t in enumerate(times):
            t_prev = times[i + 1] if i + 1 < len(times) else -1
            xt = self.ddim_step(model, xt, t, t_prev, eta=eta)
        return xt

    @torch.no_grad()
    def reconstruct(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        start_t: int,
        method: str = "ddpm",
        ddim_steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        if method == "ddim":
            return self.reconstruct_ddim(model, x0, start_t, num_steps=ddim_steps, eta=eta)
        b = x0.shape[0]
        t_start = torch.full((b,), start_t, device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t_start, noise)
        for t in reversed(range(start_t + 1)):
            xt = self.denoise_step(model, xt, t)
        return xt
