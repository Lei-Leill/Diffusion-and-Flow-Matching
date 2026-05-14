"""tests/test_vp.py — Autograder tests for the VP SDE (Part 5.A)."""

import math
import pytest
import torch
from tests.adapters import make_vpsde, run_beta, run_c, run_sigma, run_marginal


@pytest.fixture
def sde():
    return make_vpsde(beta_min=0.01, beta_max=5.0, T=1000)


class TestBeta:
    def test_endpoints(self, sde):
        t = torch.tensor([0.0, 1.0])
        b = run_beta(sde, t)
        assert b.shape == (2,)
        assert torch.allclose(b[0], torch.tensor(0.01), atol=1e-5)
        assert torch.allclose(b[1], torch.tensor(5.0),  atol=1e-5)

    def test_monotone(self, sde):
        t = torch.linspace(0, 1, 50)
        b = run_beta(sde, t)
        assert (b[1:] >= b[:-1]).all(), "β(t) should be non-decreasing"


class TestC:
    def test_at_zero(self, sde):
        t = torch.tensor([0.0])
        c = run_c(sde, t)
        assert torch.allclose(c, torch.ones(1), atol=1e-5), "c(0) should be 1"

    def test_decreasing(self, sde):
        t = torch.linspace(0, 1, 50)
        c = run_c(sde, t)
        assert (c[1:] <= c[:-1]).all(), "c(t) should be non-increasing"

    def test_range(self, sde):
        t = torch.linspace(0, 1, 100)
        c = run_c(sde, t)
        assert (c >= 0).all() and (c <= 1).all()


class TestSigma:
    def test_at_zero(self, sde):
        t = torch.tensor([0.0])
        s = run_sigma(sde, t)
        assert torch.allclose(s, torch.zeros(1), atol=1e-5), "σ(0) should be 0"

    def test_variance_sum(self, sde):
        """c(t)² + σ(t)² should equal 1 (variance preserving)."""
        t = torch.linspace(0.01, 0.99, 50)
        c = run_c(sde, t)
        s = run_sigma(sde, t)
        assert torch.allclose(c**2 + s**2, torch.ones_like(t), atol=1e-5)


class TestMarginal:
    def test_shape(self, sde):
        B, C, H, W = 4, 1, 28, 28
        x0 = torch.randn(B, C, H, W)
        t  = torch.rand(B)
        xt, eps = run_marginal(sde, x0, t)
        assert xt.shape  == x0.shape
        assert eps.shape == x0.shape

    def test_mean_at_t0(self, sde):
        """At t≈0, x_t should be approximately x0."""
        x0 = torch.randn(32, 1, 28, 28)
        t  = torch.full((32,), 1e-4)
        xt, _ = run_marginal(sde, x0, t)
        assert torch.allclose(xt, x0, atol=0.05)
