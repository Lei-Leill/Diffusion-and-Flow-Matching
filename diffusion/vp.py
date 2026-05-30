"""
diffusion/vp.py  —  Variance-Preserving (VP) SDE
=================================================
Part 5 of EE/CS 148B HW4.

Reference: Song et al. (2021) "Score-Based Generative Modeling through
Stochastic Differential Equations" (Song21), Appendix B & D.

Students implement every method marked TODO.  Methods marked PROVIDED
are complete and should not be modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VPSDE:
    """Variance-Preserving SDE forward process and samplers.

    The VP-SDE is:
        dx = -½ β(t) x dt + √β(t) dB_t

    with β(t) = β_min + (β_max - β_min) * t  (linear schedule).

    Args:
        beta_min: Minimum noise schedule value β_min.
        beta_max: Maximum noise schedule value β_max.
        T:        Number of discrete time steps (used by the EM/PC samplers).
    """

    def __init__(self, beta_min: float = 0.01, beta_max: float = 5.0, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    # ------------------------------------------------------------------
    # 5.A  Defining the VP SDE
    # ------------------------------------------------------------------

    def beta(self, t: Tensor) -> Tensor:
        """β(t) — the linear noise schedule.

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            β(t), same shape as t.

        Reference: Eq. (32) of Song21.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def c(self, t: Tensor) -> Tensor:
        """c(t) = exp(-½ ∫_0^t β(s) ds) — the signal decay factor.

        For a linear β schedule:
            ∫_0^t β(s) ds = β_min * t + ½ (β_max - β_min) * t²

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            c(t), same shape as t.

        Reference: Eq. (33) of Song21.
        """
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()
        return torch.exp(-0.5 * integral)

    def sigma(self, t: Tensor) -> Tensor:
        """σ(t) = √(1 - c(t)²) — the noise standard deviation.

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            σ(t), same shape as t.
        """
        return torch.sqrt(torch.clamp(1.0 - self.c(t).square(), min=0.0))

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Drift coefficient  f(x, t) = -½ β(t) x.

        Args:
            x: State tensor, shape (B, *).
            t: Time tensor, shape (B,) broadcast-compatible with x.

        Returns:
            Drift f(x, t), same shape as x.
        """
        beta_t = self.beta(t).reshape(t.shape[0], *([1] * (x.ndim - 1)))
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Diffusion coefficient  g(t) = √β(t).

        Args:
            t: Time tensor, shape (*).

        Returns:
            g(t), same shape as t.
        """
        return torch.sqrt(self.beta(t))

    def marginal(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Sample from the forward marginal  q(x_t | x_0).

        The marginal satisfies:
            x_t = c(t) * x_0 + σ(t) * ε,   ε ~ N(0, I)

        Args:
            x0: Clean data, shape (B, *).
            t:  Continuous time in [0, 1], shape (B,).

        Returns:
            (x_t, eps): noised sample and the noise used, both shape (B, *).
        """
        view_shape = (t.shape[0],) + (1,) * (x0.ndim - 1)
        c_t = self.c(t).reshape(view_shape)
        sigma_t = self.sigma(t).reshape(view_shape)
        eps = torch.randn_like(x0)
        return c_t * x0 + sigma_t * eps, eps

    # ------------------------------------------------------------------
    # 5.B  Samplers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_maruyama(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Euler-Maruyama reverse-SDE sampler (Problem 5.B.i).

        Starting from x(T=1) ~ N(0, σ(1)² I), integrates the reverse VP-SDE:
            dx = [-½ β(t) x - β(t) ∇_x log p_t(x)] dt + √β(t) dB̄_t

        Args:
            score_model: Trained score network s_θ(x, t).
                         Called as `score_model(x, t)` where t is a float
                         tensor of shape (B,) with values in [0, 1].
            shape:       Output shape (B, C, H, W).
            num_steps:   Number of discretisation steps (default: self.T).
            device:      Target device.

        Returns:
            Generated samples, shape (B, C, H, W), values in [-1, 1].
        """
        num_steps = num_steps or self.T
        device = torch.device(device)
        batch = shape[0]
        dt = 1.0 / num_steps
        t1 = torch.ones(batch, device=device)
        x = self.sigma(t1).reshape(batch, *([1] * (len(shape) - 1))) * torch.randn(
            shape, device=device
        )

        for i in range(num_steps):
            t_scalar = 1.0 - i * dt
            t = torch.full((batch,), t_scalar, device=device)
            beta_t = self.beta(t).reshape(batch, *([1] * (x.ndim - 1)))
            score = score_model(x, t)
            drift = 0.5 * beta_t * x + beta_t * score
            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = x + drift * dt + torch.sqrt(beta_t * dt) * noise
            else:
                x = x + drift * dt
        return x.clamp(-1, 1)

    @torch.no_grad()
    def predictor_corrector(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        n_corrector: int = 1,
        snr: float = 0.16,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Predictor-Corrector sampler with EM predictor (Problem 5.B.ii).

        Follows Algorithm 5 of Song21.  Each predictor step is an EM step;
        each corrector step is one step of annealed Langevin dynamics.

        Args:
            score_model:  Trained score network s_θ(x, t).
            shape:        Output shape (B, C, H, W).
            num_steps:    Number of predictor steps (default: self.T).
            n_corrector:  Number of Langevin corrector steps per predictor step.
            snr:          Signal-to-noise ratio for the corrector step size.
            device:       Target device.

        Returns:
            Generated samples, shape (B, C, H, W), values in [-1, 1].
        """
        num_steps = num_steps or self.T
        device = torch.device(device)
        batch = shape[0]
        dt = 1.0 / num_steps
        t1 = torch.ones(batch, device=device)
        x = self.sigma(t1).reshape(batch, *([1] * (len(shape) - 1))) * torch.randn(
            shape, device=device
        )

        for i in range(num_steps):
            t_scalar = 1.0 - i * dt
            t = torch.full((batch,), t_scalar, device=device)

            for _ in range(n_corrector):
                score = score_model(x, t)
                noise = torch.randn_like(x)
                grad_norm = torch.norm(score.reshape(batch, -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(batch, -1), dim=-1).mean()
                step_size = 2.0 * (snr * noise_norm / (grad_norm + 1e-12)).square()
                x = x + step_size * score + torch.sqrt(2.0 * step_size) * noise

            beta_t = self.beta(t).reshape(batch, *([1] * (x.ndim - 1)))
            score = score_model(x, t)
            drift = 0.5 * beta_t * x + beta_t * score
            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = x + drift * dt + torch.sqrt(beta_t * dt) * noise
            else:
                x = x + drift * dt
        return x.clamp(-1, 1)

    # ------------------------------------------------------------------
    # 5.D  Inverse problems (EC)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint(
        self,
        score_model: nn.Module,
        corrupted: Tensor,
        mask: Tensor,
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Conditional reverse diffusion for inpainting (EC Problem 5.D).

        At each reverse step, replaces the known pixels with their
        forward-diffused ground-truth values, conditioning the reverse
        process on the observed measurements.

        Reference: Song et al. (2022) "Solving Inverse Problems in Medical
        Imaging with Score-Based Generative Models".

        Args:
            score_model: Trained score network s_θ(x, t).
            corrupted:   Observed (corrupted) image, shape (B, C, H, W).
                         Unknown pixels are set to 0.
            mask:        Binary mask, shape (B, 1, H, W).
                         1 = observed pixel, 0 = missing pixel.
            num_steps:   Reverse steps (default: self.T).
            device:      Target device.

        Returns:
            Reconstructed images, shape (B, C, H, W).
        """
        num_steps = num_steps or self.T
        device = torch.device(device)
        corrupted = corrupted.to(device)
        mask = mask.to(device)
        batch = corrupted.shape[0]
        dt = 1.0 / num_steps
        t1 = torch.ones(batch, device=device)
        x = self.sigma(t1).reshape(batch, *([1] * (corrupted.ndim - 1))) * torch.randn_like(
            corrupted
        )

        for i in range(num_steps):
            t_scalar = 1.0 - i * dt
            t = torch.full((batch,), t_scalar, device=device)
            known_t, _ = self.marginal(corrupted, t)
            x = mask * known_t + (1.0 - mask) * x

            beta_t = self.beta(t).reshape(batch, *([1] * (x.ndim - 1)))
            score = score_model(x, t)
            drift = 0.5 * beta_t * x + beta_t * score
            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = x + drift * dt + torch.sqrt(beta_t * dt) * noise
            else:
                x = x + drift * dt

        return (mask * corrupted + (1.0 - mask) * x).clamp(-1, 1)
