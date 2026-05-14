# data/

This directory is **gitignored** and used as a local cache for datasets.

FashionMNIST will be downloaded here automatically the first time you run
`scripts/train_vp.py` or `scripts/train_rectflow.py`.

For Part 7, download the guided-diffusion model weights and place them under
`models/` (one level up from this directory, inside the guided-diffusion repo):

```
guided-diffusion/
  models/
    256x256_diffusion_uncond.pt   # unconditional model (Problem 7.1–7.3)
    256x256_classifier.pt         # classifier (Problem 7.4–7.5)
```

Follow the download instructions on the
[guided-diffusion model card](https://github.com/openai/guided-diffusion#model-flags).
