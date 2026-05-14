"""
diffusion/rectflow.py  —  Rectified Flow
=========================================
Part 6 of EE/CS 148B HW4.

Reference: Liu et al. (2023) "Flow Straight and Fast: Learning to Generate
and Transfer Data with Rectified Flow" (ICLR 2023).

Students implement every method marked TODO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RectifiedFlow:
    """Rectified Flow forward process, training loss, and ODE sampler.

    The interpolation is:
        X_t = (1 - t) X_0 + t X_1,   t ∈ [0, 1]

    where X_0 ~ π_0 = N(0, I)  and  X_1 ~ π_1  (data).

    The regression target is the velocity  v = X_1 - X_0.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 6.A  Forward process and loss
    # ------------------------------------------------------------------

    def forward_process(
        self, x1: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample from the rectified flow interpolation at time t.

        Args:
            x1: Clean data samples, shape (B, *).
            t:  Continuous time in [0, 1], shape (B,).

        Returns:
            (x_t, x0, vel): interpolated point, noise used, and regression
                            target velocity (x1 - x0), all shape (B, *).
        """
        # TODO (6.A) — sample x0 ~ N(0,I), form x_t, compute vel
        # Hint: broadcast t to match x1's spatial dimensions before multiplying.
        raise NotImplementedError

    def loss(self, v_theta: nn.Module, x1: Tensor) -> Tensor:
        """Rectified Flow training loss (RF objective).

        L_RF(θ) = E_{t,X_0,X_1} [ ‖(X_1 - X_0) - v_θ(X_t, t)‖² ]

        Args:
            v_theta: Velocity network; called as v_theta(x_t, t).
                     t is a float tensor of shape (B,) in [0, 1].
            x1:      Clean data batch, shape (B, C, H, W).

        Returns:
            Scalar loss.
        """
        # TODO (6.A)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 6.B  Euler ODE sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_sample(
        self,
        v_theta: nn.Module,
        shape: tuple[int, ...],
        num_steps: int = 100,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Euler ODE sampler for rectified flow (Problem 6.B).

        Integrates  dX/dt = v_θ(X_t, t)  from t=0 to t=1 using
        uniform step size Δt = 1 / num_steps.

        Args:
            v_theta:   Trained velocity network.
            shape:     Output shape (B, C, H, W).
            num_steps: Number of Euler integration steps.
                       After reflow, a single step (num_steps=1) should
                       produce reasonable samples.
            device:    Target device.

        Returns:
            Generated samples X_1, shape (B, C, H, W).
        """
        # TODO (6.B)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 6.C  Reflow  (data generation only — retraining uses loss() above)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        v_theta: nn.Module,
        n_pairs: int,
        image_shape: tuple[int, ...],
        num_steps: int = 100,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Generate (X̂_0, X̂_1) pairs for the reflow procedure (Problem 6.C).

        For each fresh noise sample X̂_0 ~ N(0, I), run the Euler ODE to
        obtain X̂_1 = Φ_1(X̂_0).  The resulting pairs are used to retrain
        the velocity network, producing straighter trajectories.

        Args:
            v_theta:     Trained velocity network (first-round).
            n_pairs:     Total number of pairs to generate (e.g. 50 000).
            image_shape: Spatial shape of one image (C, H, W).
            num_steps:   Euler steps used for the ODE integration.
            batch_size:  Number of pairs to generate per forward pass.
            device:      Target device.

        Returns:
            (x0_all, x1_all): tensors of shape (n_pairs, C, H, W) on CPU.
        """
        # TODO (6.C)
        raise NotImplementedError
