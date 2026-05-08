# raicom

PyTorch / [timm](https://github.com/huggingface/pytorch-image-models) notebooks for weather image classification: single-model training and a four-model ensemble with stacking (XGBoost).

## Layout

| Path | Purpose |
|------|--------|
| `notebooks/` | Jupyter notebooks (training, ensemble, checks) |
| `src/raicom/` | Small Python package (`paths` for dataset root) |
| `data/raw/` | Place ImageFolder dataset here as `dataset/` (see `data/README.md`) |
| `outputs/checkpoints/` | Saved weights (gitignored) |
| `outputs/figures/` | Plots (gitignored) |

## Setup

1. Python 3.10+ recommended.
2. Install PyTorch for your OS/CUDA from [pytorch.org](https://pytorch.org/) if you need a specific build.
3. From the repository root:

```bash
pip install -e ".[train]"
```

Or:

```bash
pip install -r requirements.txt
pip install -e .
```

4. Put your dataset at `data/raw/dataset/` (ImageFolder: one subfolder per class), **or** set:

```bash
set RAICOM_DATA_ROOT=D:\path\to\your\dataset
```

(On Linux/macOS use `export RAICOM_DATA_ROOT=...`.)

5. Open notebooks under `notebooks/` (start Jupyter from repo root or any folder; notebooks add `src` to `sys.path` when needed).

## License

MIT — see `LICENSE`.
