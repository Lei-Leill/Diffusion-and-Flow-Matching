"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda

The script prints a markdown table with KID mean ± std for each
(method, num_steps) combination.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--batch_size",type=int,   default=128)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)  # [0,1]
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


@torch.no_grad()
def ddim_sample(sde: VPSDE, score_model: UNet, shape, num_steps: int, device) -> torch.Tensor:
    batch = shape[0]
    dt = 1.0 / num_steps
    t1 = torch.ones(batch, device=device)
    x = sde.sigma(t1).reshape(batch, *([1] * (len(shape) - 1))) * torch.randn(
        shape, device=device
    )
    for i in range(num_steps):
        t_now = 1.0 - i * dt
        t_next = max(1.0 - (i + 1) * dt, 0.0)
        t = torch.full((batch,), t_now, device=device)
        s = torch.full((batch,), t_next, device=device)
        score = score_model(x, t)
        view = (batch,) + (1,) * (x.ndim - 1)
        c_t = sde.c(t).reshape(view).clamp_min(1e-6)
        sigma_t = sde.sigma(t).reshape(view)
        c_s = sde.c(s).reshape(view)
        sigma_s = sde.sigma(s).reshape(view)
        eps_hat = -sigma_t * score
        x0_hat = (x + sigma_t.square() * score) / c_t
        x = c_s * x0_hat + sigma_s * eps_hat
    return x.clamp(-1, 1)


@torch.no_grad()
def generate_in_batches(method, n_samples: int, batch_size: int, device, image_shape=(1, 28, 28)):
    chunks = []
    produced = 0
    while produced < n_samples:
        bsz = min(batch_size, n_samples - produced)
        chunks.append(method((bsz, *image_shape), device).cpu())
        produced += bsz
    return torch.cat(chunks, dim=0)


def main():
    args = get_args()
    device = torch.device(args.device)

    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()

    flow = RectifiedFlow()
    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    real_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    real_ds = Subset(real_ds, range(min(args.n_samples, len(real_ds))))
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, shuffle=False)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        real_dir = os.path.join(tmp, "real")
        os.makedirs(real_dir, exist_ok=True)
        index = 0
        for x, _ in real_dl:
            for img in x:
                save_image(img * 0.5 + 0.5, os.path.join(real_dir, f"{index:05d}.png"))
                index += 1

        for method in METHODS:
            for steps in STEP_COUNTS:
                gen_dir = os.path.join(tmp, f"{method}_{steps}")
                if method == "rectflow":
                    samples = generate_in_batches(
                        lambda shape, dev: flow.euler_sample(
                            rf_model, shape, num_steps=steps, device=dev
                        ),
                        args.n_samples,
                        args.batch_size,
                        device,
                    )
                elif method == "ddim":
                    samples = generate_in_batches(
                        lambda shape, dev: ddim_sample(sde, vp_model, shape, steps, dev),
                        args.n_samples,
                        args.batch_size,
                        device,
                    )
                elif method == "em":
                    samples = generate_in_batches(
                        lambda shape, dev: sde.euler_maruyama(
                            vp_model, shape, num_steps=steps, device=dev
                        ),
                        args.n_samples,
                        args.batch_size,
                        device,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                save_samples_to_dir(samples, gen_dir)
                metrics = compute_kid(gen_dir, real_dir)
                mean = metrics["kernel_inception_distance_mean"]
                std = metrics["kernel_inception_distance_std"]
                results[(method, steps)] = (mean, std)
                print(f"{method:8s} {steps:4d}: {mean:.6f} ± {std:.6f}")

    print("\n| Method | Steps | KID mean | KID std |")
    print("|---|---:|---:|---:|")
    for method in METHODS:
        for steps in STEP_COUNTS:
            mean, std = results[(method, steps)]
            print(f"| {method} | {steps} | {mean:.6f} | {std:.6f} |")


if __name__ == "__main__":
    main()
