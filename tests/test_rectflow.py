"""tests/test_rectflow.py — Autograder tests for Rectified Flow (Part 6.A)."""

import pytest
import torch
from tests.adapters import make_rectflow, run_rf_forward


@pytest.fixture
def flow():
    return make_rectflow()


class TestForwardProcess:
    def test_shape(self, flow):
        B = 8
        x1 = torch.randn(B, 1, 28, 28)
        t  = torch.rand(B)
        xt, x0, vel = run_rf_forward(flow, x1, t)
        assert xt.shape  == x1.shape
        assert x0.shape  == x1.shape
        assert vel.shape == x1.shape

    def test_velocity_target(self, flow):
        """vel should equal x1 - x0."""
        x1 = torch.randn(16, 1, 28, 28)
        t  = torch.rand(16)
        xt, x0, vel = run_rf_forward(flow, x1, t)
        assert torch.allclose(vel, x1 - x0, atol=1e-6)

    def test_interpolation_at_t0(self, flow):
        """At t=0, x_t should equal x0."""
        x1 = torch.randn(8, 1, 28, 28)
        t  = torch.zeros(8)
        xt, x0, _ = run_rf_forward(flow, x1, t)
        assert torch.allclose(xt, x0, atol=1e-6)

    def test_interpolation_at_t1(self, flow):
        """At t=1, x_t should equal x1."""
        x1 = torch.randn(8, 1, 28, 28)
        t  = torch.ones(8)
        xt, x0, _ = run_rf_forward(flow, x1, t)
        assert torch.allclose(xt, x1, atol=1e-6)

    def test_linearity(self, flow):
        """x_t = (1-t)*x0 + t*x1 for t in (0,1)."""
        x1 = torch.randn(4, 1, 28, 28)
        t  = torch.full((4,), 0.5)
        xt, x0, _ = run_rf_forward(flow, x1, t)
        expected = 0.5 * x0 + 0.5 * x1
        assert torch.allclose(xt, expected, atol=1e-6)
