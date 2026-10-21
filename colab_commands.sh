#!/bin/bash
# ============================================================
# Google Colab Training Commands for PointNet on 3DGrocery100
# ============================================================
# Copy-paste each section into separate Colab cells
# ============================================================

# ------------------------------------------------------------
# CELL 1: Environment Setup (run once)
# ------------------------------------------------------------
cat << 'EOF'
# Install PyTorch with CUDA 12.1
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install tqdm matplotlib numpy pyyaml -q

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/YOUR_USERNAME/Grocery.git /content/Grocery
# OR copy from Drive:
# %cp -r /content/drive/MyDrive/Grocery /content/Grocery

import os, sys
PROJECT_ROOT = "/content/Grocery"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Verify data
RAW_ROOT = "/content/Grocery/data"
PROCESSED_ROOT = "/content/Grocery/data/processed"
print("Train DAT:", os.path.exists(os.path.join(RAW_ROOT, "modelnet100_train_1024pts_fps.dat")))
print("Test DAT:", os.path.exists(os.path.join(RAW_ROOT, "modelnet100_test_1024pts_fps.dat")))
EOF

# ------------------------------------------------------------
# CELL 2: Experiment A - Baseline (XYZ, 50 epochs, no augmentation)
# ------------------------------------------------------------
cat << 'EOF'
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
EOF

# ------------------------------------------------------------
# CELL 3: Experiment B - With Conservative Augmentation (XYZ, 50 epochs)
# ------------------------------------------------------------
cat << 'EOF'
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
EOF

# ------------------------------------------------------------
# CELL 4: Experiment C - Longer Training + Cosine Annealing (100 epochs)
# ------------------------------------------------------------
cat << 'EOF'
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
EOF

# ------------------------------------------------------------
# CELL 5: Experiment D - XYZRGB Mode (100 epochs, cosine)
# ------------------------------------------------------------
cat << 'EOF'
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
EOF

# ------------------------------------------------------------
# CELL 6: Evaluate Best Model on Official Test Set
# ------------------------------------------------------------
cat << 'EOF'
# Update this path to your best checkpoint
BEST_CHECKPOINT="/content/drive/MyDrive/pointnet_experiments/exp_C_longer_cosine/best_pointnet.pt"

!python scripts/evaluate_pointnet.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --batch-size 32 \
  --num-workers 4 \
  --device auto \
  --output-dir "/content/drive/MyDrive/pointnet_experiments/test_evaluation"
EOF

# ------------------------------------------------------------
# CELL 7: Resume Interrupted Training
# ------------------------------------------------------------
cat << 'EOF'
# To resume, just re-run the SAME command with the SAME --output-dir
# The script auto-detects last_pointnet.pt and resumes from there
# Example - resume Experiment C:
!python scripts/train_pointnet.py \
  --raw-root "$RAW_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --mode xyz \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output-dir "/content/drive/MyDrive/pointnet_experiments/exp_C_longer_cosine" \
  --no-plot \
  --aug-rotation 0.0 \
  --aug-scale 0.05 \
  --aug-translate 0.02 \
  --aug-jitter-std 0.01 \
  --scheduler cosine \
  --cosine-t-max 100 \
  --early-stopping-patience 20
EOF

# ------------------------------------------------------------
# CELL 8: Visualize Training Curves
# ------------------------------------------------------------
cat << 'EOF'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# View final dashboard from any experiment
exp_dir = "/content/drive/MyDrive/pointnet_experiments/exp_C_longer_cosine"
img = mpimg.imread(os.path.join(exp_dir, "training_dashboard_final.png"))
plt.figure(figsize=(14, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
EOF

# ------------------------------------------------------------
# CELL 9: Compare All Experiments
# ------------------------------------------------------------
cat << 'EOF'
import json, os
import pandas as pd
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
        best = histories[name]['val_overall_accuracy'].max()
        print(f"{name}: {len(histories[name])} epochs, best val OA: {best:.4f}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for name, df in histories.items():
    axes[0,0].plot(df['epoch'], df['val_overall_accuracy'], label=name, linewidth=2)
    axes[0,1].plot(df['epoch'], df['val_mean_class_accuracy'], label=name, linewidth=2)
    axes[1,0].plot(df['epoch'], df['train_loss'], label=f"{name}_train", linestyle='--', alpha=0.7)
    axes[1,0].plot(df['epoch'], df['val_loss'], label=f"{name}_val", linewidth=2)
    axes[1,1].plot(df['epoch'], df['learning_rate'], label=name, linewidth=2)

axes[0,0].set_title("Validation Overall Accuracy"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
axes[0,1].set_title("Validation Mean Class Accuracy"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
axes[1,0].set_title("Loss Curves"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
axes[1,1].set_title("Learning Rate"); axes[1,1].set_yscale('log'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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
pd.DataFrame(summary).to_markdown(index=False)
EOF

echo "============================================================"
echo "Copy each CELL block above into separate Colab cells"
echo "============================================================"