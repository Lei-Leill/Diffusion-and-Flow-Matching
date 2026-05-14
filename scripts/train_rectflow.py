"""
scripts/train_rectflow.py  —  Part 6: Train a Rectified Flow model on FashionMNIST
====================================================================================

Usage::
    # First-round training
    python scripts/train_rectflow.py --epochs 50 --save_dir runs/rectflow

    # Reflow (Problem 6.C): generate pairs then retrain
    python scripts/train_rectflow.py --reflow --checkpoint runs/rectflow/best.pt \
        --n_reflow_pairs 50000 --epochs 20 --save_dir runs/rectflow_reflow
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from diffusion.unet import UNet
from diffusion.rectflow import RectifiedFlow


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=128)
    p.add_argument("--save_dir",       type=str,   default="runs/rectflow")
    p.add_argument("--device",         type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    # Reflow options
    p.add_argument("--reflow",         action="store_true",
                   help="Run the reflow procedure using a pretrained checkpoint.")
    p.add_argument("--checkpoint",     type=str,   default=None,
                   help="Path to pretrained .pt file (required for --reflow).")
    p.add_argument("--n_reflow_pairs", type=int,   default=50_000)
    p.add_argument("--reflow_steps",   type=int,   default=100,
                   help="Euler steps used to generate reflow pairs.")
    return p.parse_args()


def build_dataloader(batch_size: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def train_one_epoch(model, flow, dataloader, optimizer, device):
    model.train()
    running = 0.0
    for x1, _ in dataloader:
        x1 = x1.to(device)
        loss = flow.loss(model, x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * x1.size(0)
    return running / len(dataloader.dataset)


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    flow  = RectifiedFlow()
    model = UNet(in_channels=1, base_channels=64).to(device)

    if args.reflow:
        # ---------------------------------------------------------------
        # Reflow: generate paired data from the pre-trained model, then
        # retrain on those pairs.
        # ---------------------------------------------------------------
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --reflow")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")

        print(f"Generating {args.n_reflow_pairs} reflow pairs ...")
        x0_all, x1_all = flow.generate_reflow_pairs(
            model,
            n_pairs=args.n_reflow_pairs,
            image_shape=(1, 28, 28),
            num_steps=args.reflow_steps,
            device=device,
        )
        reflow_ds = TensorDataset(x0_all, x1_all)
        dataloader = DataLoader(reflow_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

        # Re-initialise a fresh model for retraining
        model = UNet(in_channels=1, base_channels=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_losses = []
        best_loss = math.inf
        for epoch in range(1, args.epochs + 1):
            # Reflow training uses pre-paired data; loss just does MSE on vel
            model.train()
            running = 0.0
            for x0_b, x1_b in dataloader:
                x0_b, x1_b = x0_b.to(device), x1_b.to(device)
                t = torch.rand(x0_b.size(0), device=device)
                # broadcast t: (B,) -> (B,1,1,1)
                t4 = t[:, None, None, None]
                x_t = (1 - t4) * x0_b + t4 * x1_b
                vel = x1_b - x0_b
                import torch.nn.functional as F
                loss = F.mse_loss(model(x_t, t), vel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item() * x0_b.size(0)
            epoch_loss = running / len(dataloader.dataset)
            train_losses.append(epoch_loss)
            print(f"Reflow epoch {epoch:3d}/{args.epochs} | loss {epoch_loss:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")

    else:
        # ---------------------------------------------------------------
        # Standard first-round training
        # ---------------------------------------------------------------
        dataloader = build_dataloader(args.batch_size)
        optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_losses = []
        best_loss = math.inf
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, flow, dataloader, optimizer, device)
            train_losses.append(loss)
            print(f"Epoch {epoch:3d}/{args.epochs} | loss {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")

    import numpy as np
    np.save(Path(args.save_dir) / "train_losses.npy", np.array(train_losses))
    print(f"Done. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
