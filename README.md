# EE/CS 148B HW 4 — Diffusion Models

Starter code for HW 4 of EE/CS 148B (Spring 2026).  See `hw4.pdf` (in the
assignment release) for the full writeup.

## Repository Layout

```
hw4/
├── diffusion/                      # Core library
│   ├── __init__.py
│   ├── unet.py                     # PROVIDED: time-conditioned U-Net (shared by Parts 5 & 6)
│   ├── vp.py                       # TODO: VP SDE, EM sampler, PC sampler  (Parts 5.A & 5.B)
│   └── rectflow.py                 # TODO: Rectified Flow forward, loss, Euler sampler  (Part 6)
│
├── scripts/                        # CLI entry points
│   ├── plot_coefficient.py         # TODO: Part 1.8 — coefficient plot
│   ├── train_vp.py                 # Part 5.C — train VP score model on FashionMNIST
│   ├── train_rectflow.py           # Part 6.A/C — train Rectified Flow (+ reflow)
│   ├── sample.py                   # TODO: Parts 5.C, 6.B, 6.D — generate & compare samples
│   ├── eval_kid.py                 # TODO: Part 6.B — KID evaluation table
│   └── guided_diffusion_experiments.py  # TODO: Part 7 — plotting helpers
│
├── tests/                          # Test infrastructure
│   ├── adapters.py                 # TODO: bind your implementations to test hooks
│   ├── test_vp.py                  # Autograder tests for VP SDE
│   └── test_rectflow.py            # Autograder tests for Rectified Flow
│
├── configs/
│   ├── vp_fashionmnist.yaml        # Hyperparameters for Part 5
│   └── rectflow_fashionmnist.yaml  # Hyperparameters for Part 6
│
├── data/                           # Dataset cache (gitignored)
│   └── README.md
│
├── pyproject.toml
├── README.md
└── .gitignore
```

`PROVIDED` files contain working code and should not be modified.
`TODO` files contain skeleton methods with `raise NotImplementedError` and
clear docstrings — these are what you implement.

## Assignment Map

| Part | Topic | Files to edit |
|------|-------|---------------|
| 1.8  | Coefficient plot | `scripts/plot_coefficient.py` |
| 5.A  | VP SDE definition | `diffusion/vp.py` |
| 5.B  | EM & PC samplers | `diffusion/vp.py` |
| 5.C  | Generate samples & plots | `scripts/sample.py` |
| 5.D (EC) | Inpainting | `diffusion/vp.py` |
| 6.A  | Rectified Flow forward + loss | `diffusion/rectflow.py`, `scripts/train_rectflow.py` |
| 6.B  | Euler sampler + KID table | `diffusion/rectflow.py`, `scripts/eval_kid.py` |
| 6.C  | Reflow pairs + retrain | `diffusion/rectflow.py` |
| 6.D  | Side-by-side grid | `scripts/sample.py` |
| 7    | Guided diffusion (256×256) | `scripts/guided_diffusion_experiments.py` |

Parts 1–4 are written problems only (no code to submit).

## Setup

We recommend [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install core dependencies
uv sync

# Install torch-fidelity for KID evaluation (Part 6.B)
uv sync --extra fidelity

# Install test dependencies
uv sync --extra test

# Verify
uv run python -c "import diffusion; print('OK')"
```

## Running Tests

After filling in `tests/adapters.py`, run:

```bash
# VP SDE tests
uv run pytest -k test_vp

# Rectified Flow tests
uv run pytest -k test_rectflow

# All tests
uv run pytest
```

The Gradescope autograder calls the same test suite.

## Training (Part 5 — VP Score Model)

```bash
# Train (∼10 min on A100)
uv run python scripts/train_vp.py --config configs/vp_fashionmnist.yaml

# Generate EM samples
uv run python scripts/sample.py --method em \
    --checkpoint runs/vp/best.pt --beta_min 0.01 --beta_max 5.0

# Generate PC samples (vary --n_corrector)
uv run python scripts/sample.py --method pc \
    --checkpoint runs/vp/best.pt --beta_min 0.01 --beta_max 5.0 --n_corrector 3
```

## Training (Part 6 — Rectified Flow)

```bash
# First-round training (∼10 min on A100)
uv run python scripts/train_rectflow.py --config configs/rectflow_fashionmnist.yaml

# Reflow (generate 50k pairs then retrain for 20 epochs)
uv run python scripts/train_rectflow.py --reflow \
    --checkpoint runs/rectflow/best.pt \
    --save_dir runs/rectflow_reflow

# One-step sample after reflow
uv run python scripts/sample.py --method rectflow \
    --checkpoint runs/rectflow_reflow/best.pt --num_steps 1

# KID evaluation table
uv run python scripts/eval_kid.py \
    --vp_checkpoint runs/vp/best.pt \
    --rf_checkpoint runs/rectflow/best.pt
```

## Part 7 — Guided Diffusion (256×256)

```bash
# Clone the OpenAI codebase
git clone https://github.com/openai/guided-diffusion
cd guided-diffusion && pip install -e .

# Download model weights (see data/README.md for links)
mkdir -p models

# Unconditional generation (Problem 7.1)
OPENAI_LOGDIR=./out python scripts/image_sample.py \
    --model_path models/256x256_diffusion_uncond.pt \
    <MODEL_FLAGS> <SAMPLE_FLAGS>

# Then plot with our helper
python ../scripts/guided_diffusion_experiments.py --task 7_1 --npz out/samples_*.npz
```

See `scripts/guided_diffusion_experiments.py` for plotting helpers for all
sub-problems (7.1–7.5).

## Submission

Submit two items to Gradescope:

- `assignment4.pdf` — typed answers, plots, and a link to your Colab notebook.
- **HW4 — Code**: your `VP.py` file (renamed from `diffusion/vp.py`).

Both your Colab notebook and submitted code must run without error.

## Compute

| Part | GPU recommendation | Estimated time |
|------|--------------------|----------------|
| 5 — VP training | A100 | ~10 min |
| 6 — RF training | A100 | ~10 min |
| 6.C — Reflow pair generation | A100 | ~15 min |
| 7 — Guided diffusion | A100 | 5–10 min per run |

Use Google Colab Pro+ and switch to the **Premium GPU** runtime only when
actively running experiments to conserve compute units.

## Acknowledgments

Course staff: Aadarsh Sahoo, Ziqi Ma.
