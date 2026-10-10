# Stationary GSM extraction for periodic metal pixel metasurfaces

Python implementation and numerical data accompanying *A Stationary GSM Extraction Method for 2D Periodic Metal Pixel Metasurfaces*. The model is an 8×8 zero-thickness PEC pixel sheet with a 20 mm period in free space. Three extraction operators act on the same surface currents.

## Installation

Python 3.11 or later is recommended. No MATLAB installation, MATLAB licence, or proprietary numerical toolbox is required.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m stationary_gsm verify
python -m stationary_gsm demo
```

Activate the virtual environment before installing the requirements and running the commands. On Windows use `.venv\Scripts\activate`. On Linux or macOS use `source .venv/bin/activate`.

`verify` recomputes the paper statistics from the original 90,000 CSV rows. `demo` solves an oblique-incidence EFIE example and checks output-error scaling, the quadratic identity, and power balance. Both commands work with the repository files alone.

## Data and full checks

Download `stationary-gsm-data-v2.0.0.zip` from the [fixed release](https://github.com/zhjeffrey123-eng/stationary-gsm-awpl/releases/tag/v2.0.0) and extract it at the repository root, merging the `output` directory.

```bash
python -m stationary_gsm audit
python -m stationary_gsm evaluate --count 1000 --out results/evaluation
python -m stationary_gsm stack --full --out results/stack
python -m stationary_gsm fine --N 48 64 80 96 --frequencies 5 7 9 --angles 0 20 40 --ratios 6 8 --out results/fine
python -m stationary_gsm fields --out results/fields
python -m stationary_gsm figures --out results/figures
```

All calculation commands require a new output directory. Original data are read-only inputs. Fine-grid calculations are substantially more expensive than the small demonstration.

## Training and label generation

```bash
python -m stationary_gsm labels --count 900 --out results/labels
python -m stationary_gsm novel-labels --count 1000 --out results/novel-labels
python -m stationary_gsm train --out results/training
python -m stationary_gsm evaluate --model results/training/model.npz --count 1000 --out results/retrained-evaluation
```

Training uses NumPy GELU layers, analytic gradients, and AdamW, with validation-based model selection. Random number generators and floating-point kernels differ between languages, so retraining does not promise the original weights. Use the supplied `model.npz` for the frozen-model checks. The neural model covers normal incidence with M=1. Oblique-incidence theory and reference calculations use the paired-incidence EFIE.

## File formats and provenance

Large current labels use HDF5 (`.h5`). Model parameters and reference results use NumPy archives (`.npz`), with no pickle. The public package contains no `.m` or `.mat` files. `docs/data_conversion.json` records the original and converted file checksums and exact value checks for 323 source data files. Nested reference archives store numeric arrays and a JSON schema, read by `stationary_gsm.data.load_bundle`.

The manuscript CSV records describe the original numerical experiments. Conversion preserves their values, including failed cases and recorded timings. Python recomputation writes separate results and timings. See [README_zh.md](README_zh.md) for detailed Chinese instructions, [docs/METHODS_zh.md](docs/METHODS_zh.md) for conventions, and [docs/PAPER_MAP.md](docs/PAPER_MAP.md) for the figure-to-file mapping.
