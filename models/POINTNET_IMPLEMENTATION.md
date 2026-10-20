# PointNet implementation for 3DGrocery100

This is an additive implementation layer for the existing repository. It deliberately does not change `common/dataset.py`, `common/preprocessing.py`, `common/metrics.py`, the frozen split files, or the shared preprocessing contract.

## Files included

- `models/pointnet.py`: PointNet classification model, 3x3 XYZ T-Net, 64x64 feature T-Net, global max-pooling, classification head, and feature-transform orthogonality regularizer.
- `scripts/train_pointnet.py`: training with a deterministic held-out validation subset taken from the official training split. It saves the best validation checkpoint, latest checkpoint, and CSV history.
- `scripts/evaluate_pointnet.py`: evaluates a saved checkpoint once on the official test split and writes overall metrics, per-class accuracy, and a confusion matrix.
- `tests/test_pointnet_model.py`: fast model-only checks that do not require the raw dataset.

## Put the files into your repository

Copy each folder/file from this bundle into the matching path at your repository root. The files are new; when merging into your working tree, keep existing common pipeline files unchanged.

## Environment

Run commands from the repository root and install the repository's existing `requirements.txt`. The tests require PyTorch and pytest.

## Run model-only tests

```bash
python -m pytest tests/test_pointnet_model.py -v
```

## Train baseline (XYZ)

The raw data directory must directly contain `modelnet100_train_1024pts_fps.dat` and `modelnet100_test_1024pts_fps.dat`. Since the current common Dataset code defaults to a Windows-specific raw path, pass your Linux path explicitly:

```bash
python scripts/train_pointnet.py \
  --raw-root "/home/terminator18ops/Documents/Aditya/Coding/Grocery" \
  --processed-root "data/processed" \
  --mode xyz \
  --epochs 50 \
  --batch-size 8
```

Start with 1–2 epochs and a small batch to check your paths and runtime before launching a long job. If your dataset files are in a different directory, point `--raw-root` at the folder containing the two `.dat` files.

The script automatically prefers CUDA, then PyTorch XPU, then CPU when `--device auto` is used. Intel Arc acceleration is used only if the installed PyTorch build actually reports an available XPU device; otherwise training will run on CPU.

## Evaluate final test results

```bash
python scripts/evaluate_pointnet.py \
  --raw-root "/absolute/path/to/3DGrocery100" \
  --processed-root "data/processed" \
  --checkpoint "checkpoints/pointnet/best_pointnet.pt"
```

Outputs are saved to `results/pointnet/` by default.

## Design notes

- Input arrays are consumed in the existing `(B, N, C)` layout. The model transposes internally for `Conv1d`.
- The 3x3 input transform is applied to XYZ only. RGB, when used in `xyzrgb` mode, is not treated as spatial coordinates and is concatenated unchanged.
- The classifier returns raw logits and a feature-transform matrix. Use raw logits with `CrossEntropyLoss` (no softmax before the loss).
- Training uses a reproducible 90/10 random split of the official training subset by default. The official test split is not read by the training loop and should be used only for final evaluation.
- The feature transform orthogonality penalty is added to classification loss with a configurable default weight of `1e-3`.
- This implementation is a baseline, not a claim about achieved accuracy. Metrics must be measured on your dataset and compute environment.
