"""
scripts/plot_coefficient.py  —  Part 1.8
=========================================
Plot the DDPM loss coefficient
    β_t² / (2 σ_t² α_t (1 - ᾱ_t))
vs. t on a log-scale y-axis.

Usage::
    python scripts/plot_coefficient.py --T 1000 --beta_start 1e-4 --beta_end 0.02
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


def linear_schedule(T: int, beta_start: float, beta_end: float):
    return np.linspace(beta_start, beta_end, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",          type=int,   default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end",   type=float, default=0.02)
    parser.add_argument("--out",        type=str,   default="coefficient_plot.png")
    args = parser.parse_args()

    # TODO (1.8) — compute betas, alphas, alpha_bars, sigma^2, and the coefficient,
    # then plot with a log-scale y-axis and save to args.out.
    raise NotImplementedError


if __name__ == "__main__":
    main()
