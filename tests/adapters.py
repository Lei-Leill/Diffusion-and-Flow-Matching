"""
tests/adapters.py  —  Bind student implementations to the test harness
======================================================================
Fill in each function so the test suite can call your code.
Do not change function signatures.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


# ------------------------------------------------------------------
# VP SDE (Part 5.A)
# ------------------------------------------------------------------

def make_vpsde(beta_min: float = 0.01, beta_max: float = 5.0, T: int = 1000) -> VPSDE:
    """Return a VPSDE instance with the given hyperparameters."""
    # TODO: return VPSDE(...)
    raise NotImplementedError


def run_beta(sde: VPSDE, t: Tensor) -> Tensor:
    """Return β(t)."""
    # TODO
    raise NotImplementedError


def run_c(sde: VPSDE, t: Tensor) -> Tensor:
    """Return c(t)."""
    # TODO
    raise NotImplementedError


def run_sigma(sde: VPSDE, t: Tensor) -> Tensor:
    """Return σ(t)."""
    # TODO
    raise NotImplementedError


def run_marginal(sde: VPSDE, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Return (x_t, eps) from the VP marginal q(x_t | x_0)."""
    # TODO
    raise NotImplementedError


# ------------------------------------------------------------------
# Rectified Flow (Part 6.A)
# ------------------------------------------------------------------

def make_rectflow() -> RectifiedFlow:
    """Return a RectifiedFlow instance."""
    # TODO
    raise NotImplementedError


def run_rf_forward(flow: RectifiedFlow, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, x0, vel) from the rectified flow forward process."""
    # TODO
    raise NotImplementedError
