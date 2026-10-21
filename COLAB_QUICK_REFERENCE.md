# PointNet 3DGrocery100 - Colab Quick Reference Card
# ================================================
# Copy-paste these exact commands into Colab cells

## 🔧 CELL 1: Setup (Run Once)
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install tqdm matplotlib numpy pyyaml -q

from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/Grocery.git /content/Grocery
# OR: %cp -r /content/drive/MyDrive/Grocery /content/Grocery

import os, sys
PROJECT_ROOT = "/content/Grocery"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

RAW_ROOT = "/content/Grocery/data"
PROCESSED_ROOT = "/content/Grocery/data/processed"
```

## 📊 CELL 2: Experiment A - Baseline (XYZ, 50ep, no aug)
```bash
!python scripts/train_pointnet.py \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --mode xyz \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 1e-4 \
  --feature-reg-weight 1e-3 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --seed 42 \
  --device auto \
  --output-dir "/content/drive/MyDrive/pointnet_experiments/exp_A_baseline" \
  --log-interval 100 \
  --plot-interval 5 \
  --no-plot \
  --no-aug \
  --scheduler step \
  --scheduler-step-size 20 \
  --scheduler-gamma 0.7 \
  --early-stopping-patience 15 \
  --early-stopping-min-delta 1e-4
```

## 🎨 CELL 3: Experiment B - Augmentation (XYZ, 50ep, conservative)
```bash
!python scripts/train_pointnet.py \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --mode xyz \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 1e-4 \
  --feature-reg-weight 1e-3 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --seed 42 \
  --device auto \
  --output-dir "/content/drive/MyDrive/pointnet_experiments/exp_B_augmentation" \
  --log-interval 100 \
  --plot-interval 5 \
  --no-plot \
  --aug-rotation 0.0 \
  --aug-scale 0.05 \
  --aug-translate 0.02 \
  --aug-jitter-std 0.01 \
  --aug-jitter-clip 0.05 \
  --scheduler step \
  --scheduler-step-size 20 \
  --scheduler-gamma 0.7 \
  --early-stopping-patience 15 \
  --early-stopping-min-delta 1e-4
```

## ⏱️ CELL 4: Experiment C - Longer + Cosine (100ep)
```bash
!python scripts/train_pointnet.py \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --mode xyz \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 1e-4 \
  --feature-reg-weight 1e-3 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --seed 42 \
  --device auto \
  --output-dir "/content/drive/MyDrive/pointnet_experiments/exp_C_longer_cosine" \
  --log-interval 100 \
  --plot-interval 5 \
  --no-plot \
  --aug-rotation 0.0 \
  --aug-scale 0.05 \
  --aug-translate 0.02 \
  --aug-jitter-std 0.01 \
  --aug-jitter-clip 0.05 \
  --scheduler cosine \
  --cosine-t-max 100 \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 1e-4
```

## 🌈 CELL 5: Experiment D - XYZRGB (100ep, cosine)
```bash
!python scripts/train_pointnet.py \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --mode xyzrgb \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 1e-4 \
  --feature-reg-weight 1e-3 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --seed 42 \
  --device auto \
  --output-dir "/content/drive/MyDrive/pointnet_experiments/exp_D_xyzrgb" \
  --log-interval 100 \
  --plot-interval 5 \
  --no-plot \
  --aug-rotation 0.0 \
  --aug-scale 0.05 \
  --aug-translate 0.02 \
  --aug-jitter-std 0.01 \
  --aug-jitter-clip 0.05 \
  --scheduler cosine \
  --cosine-t-max 100 \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 1e-4
```

## ✅ CELL 6: Evaluate on Official Test Set
```bash
BEST_CHECKPOINT="/content/drive/MyDrive/pointnet_experiments/exp_C_longer_cosine/best_pointnet.pt"

!python scripts/evaluate_pointnet.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --batch-size 32 \
  --num-workers 4 \
  --device auto \
  --output-dir "/content/drive/MyDrive/pointnet_experiments/test_evaluation"
```

## 🔄 CELL 7: Resume Training (re-run same command)
```bash
# Just re-run the EXACT same command from Cell 2/3/4/5
# Script auto-resumes from last_pointnet.pt in --output-dir
```

## 📈 CELL 8: View Training Dashboard
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

exp_dir = "/content/drive/MyDrive/pointnet_experiments/exp_C_longer_cosine"
img = mpimg.imread(os.path.join(exp_dir, "training_dashboard_final.png"))
plt.figure(figsize=(14, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## 📋 CELL 9: Compare All Experiments
```python
import json, os, pandas as pd
import matplotlib.pyplot as plt

exp_dirs = {
    "A_baseline": "/content/drive/MyDrive/pointnet_experiments/exp_A_baseline",
    "B_augmentation": "/content/drive/MyDrive/pointnet_experiments/exp_B_augmentation",
    "C_longer_cosine": "/content/drive/MyDrive/pointnet_experiments/exp_C_longer_cosine",
    "D_xyzrgb": "/content/drive/MyDrive/pointnet_experiments/exp_D_xyzrgb",
}

histories = {}
for name, path in exp_dirs.items():
    hist_path = os.path.join(path, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            histories[name] = pd.DataFrame(json.load(f))
        print(f"{name}: {len(histories[name])} epochs, best val OA: {histories[name]['val_overall_accuracy'].max():.4f}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for name, df in histories.items():
    axes[0,0].plot(df['epoch'], df['val_overall_accuracy'], label=name, linewidth=2)
    axes[0,1].plot(df['epoch'], df['val_mean_class_accuracy'], label=name, linewidth=2)
    axes[1,0].plot(df['epoch'], df['train_loss'], label=f"{name}_train", linestyle='--', alpha=0.7)
    axes[1,0].plot(df['epoch'], df['val_loss'], label=f"{name}_val", linewidth=2)
    axes[1,1].plot(df['epoch'], df['learning_rate'], label=name, linewidth=2)

for ax in axes.flat:
    ax.legend(); ax.grid(True, alpha=0.3)
axes[0,0].set_title("Val OA"); axes[0,1].set_title("Val mAcc")
axes[1,0].set_title("Loss"); axes[1,1].set_title("LR"); axes[1,1].set_yscale('log')
plt.tight_layout(); plt.show()

# Summary table
summary = []
for name, df in histories.items():
    best_idx = df['val_overall_accuracy'].idxmax()
    summary.append({
        "Experiment": name,
        "Best Epoch": int(df.loc[best_idx, 'epoch']),
        "Best Val OA": f"{df.loc[best_idx, 'val_overall_accuracy']:.4f}",
        "Best Val mAcc": f"{df.loc[best_idx, 'val_mean_class_accuracy']:.4f}",
        "Final Train OA": f"{df.iloc[-1]['train_overall_accuracy']:.4f}",
        "Final Val OA": f"{df.iloc[-1]['val_overall_accuracy']:.4f}",
        "Epochs": len(df),
    })
display(pd.DataFrame(summary))
```

## 🎯 Key Configurations Summary

| Experiment | Mode | Epochs | Scheduler | Augmentation | Purpose |
|------------|------|--------|-----------|--------------|---------|
| A | XYZ | 50 | StepLR | None | Baseline |
| B | XYZ | 50 | StepLR | Conservative | Aug effect |
| C | XYZ | 100 | Cosine | Conservative | LR schedule + longer |
| D | XYZRGB | 100 | Cosine | Conservative | RGB benefit |

## 📁 Output Structure (per experiment)
```
exp_X_name/
├── best_pointnet.pt          # Best val OA checkpoint
├── last_pointnet.pt          # Last epoch checkpoint (for resume)
├── training_history.csv      # Per-epoch metrics (CSV)
├── training_history.json     # Per-epoch metrics (JSON)
├── training_summary.json     # Final summary
├── training_dashboard_final.png  # Final plots
└── training_dashboard_epoch_XXX.png  # Intermediate plots (every 5 epochs)
```

## ⚡ Pro Tips
- Use `--no-plot` to disable matplotlib (faster, no GUI needed)
- Checkpoints auto-resume - just re-run same command
- All results saved to Google Drive via `--output-dir`
- Test set ONLY evaluated via `evaluate_pointnet.py` after training
- Seed=42 for reproducibility across experiments