"""Train PointNet on 3DGrocery100 using the shared common data pipeline.

Example (run from the repository root):
    python scripts/train_pointnet.py --raw-root /absolute/path/to/3DGrocery100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Subset
from tqdm.auto import tqdm

# Make ``python scripts/train_pointnet.py`` work when launched from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import Grocery3DDataset, make_dataloader  # noqa: E402
from common.metrics import accuracy, mean_class_accuracy  # noqa: E402
from models.pointnet import PointNet, feature_transform_regularizer  # noqa: E402

# Optional plotting - graceful degradation if matplotlib unavailable
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet on 3DGrocery100")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Directory containing modelnet100_train_1024pts_fps.dat and test .dat")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--mode", choices=("xyz", "xyzrgb"), default="xyz",
                        help="Use XYZ (recommended baseline) or XYZRGB input")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-reg-weight", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "pointnet")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Print training progress every N batches")
    # Phase 1: Training history and visualization
    parser.add_argument("--plot-interval", type=int, default=5,
                        help="Update and save training graphs every N epochs (0 to disable)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting even if matplotlib is available")
    # Phase 3: Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Early stopping patience in epochs (0 to disable)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4,
                        help="Minimum improvement in validation OA for early stopping")
    # Phase 3: LR Scheduler options
    parser.add_argument("--scheduler", choices=("step", "cosine", "plateau"), default="step",
                        help="Learning rate scheduler type")
    parser.add_argument("--scheduler-step-size", type=int, default=20,
                        help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=0.7,
                        help="Gamma for StepLR scheduler")
    parser.add_argument("--cosine-t-max", type=int, default=100,
                        help="T_max for CosineAnnealingLR scheduler")
    parser.add_argument("--plateau-patience", type=int, default=10,
                        help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--plateau-factor", type=float, default=0.5,
                        help="Factor for ReduceLROnPlateau scheduler")
    # Phase 2: Data Augmentation
    parser.add_argument("--aug-rotation", type=float, default=0.0,
                        help="Max rotation angle in radians around Z-axis (0 to disable)")
    parser.add_argument("--aug-scale", type=float, default=0.05,
                        help="Scale variation range (0 to disable)")
    parser.add_argument("--aug-translate", type=float, default=0.02,
                        help="Translation range (0 to disable)")
    parser.add_argument("--aug-jitter-std", type=float, default=0.01,
                        help="Gaussian jitter std (0 to disable)")
    parser.add_argument("--aug-jitter-clip", type=float, default=0.05,
                        help="Jitter clipping value")
    parser.add_argument("--no-aug", action="store_true",
                        help="Disable all augmentations (baseline compatible)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be >= 1 and --batch-size must be >= 2")
    if not 0.0 < args.val_ratio < 0.5:
        parser.error("--val-ratio must be between 0 and 0.5")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.feature_reg_weight < 0:
        parser.error("Learning rate must be positive; regularization weights cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative")
    if args.early_stopping_min_delta < 0:
        parser.error("--early-stopping-min-delta cannot be negative")
    if not args.raw_root.is_dir():
        parser.error(f"--raw-root is not an existing directory: {args.raw_root}")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but this PyTorch installation has no available XPU device")
    return torch.device(requested)


def save_training_plots(
    history: List[Dict],
    output_dir: Path,
    epoch: int,
    final: bool = False,
) -> None:
    """Generate and save training dashboard plots."""
    if not PLOTTING_AVAILABLE:
        return
    
    if not history:
        return
    
    epochs = [row["epoch"] for row in history]
    
    # Extract metrics
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_oa = [row["train_overall_accuracy"] for row in history]
    val_oa = [row["val_overall_accuracy"] for row in history]
    val_macc = [row["val_mean_class_accuracy"] for row in history]
    learning_rates = [row["learning_rate"] for row in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"PointNet Training Dashboard (Epoch {epoch})", fontsize=14)
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train Loss", color="tab:blue", linewidth=1.5)
    ax.plot(epochs, val_loss, label="Val Loss", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Overall Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, train_oa, label="Train OA", color="tab:blue", linewidth=1.5)
    ax.plot(epochs, val_oa, label="Val OA", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Overall Accuracy (OA)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Mean Class Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, val_macc, label="Val mAcc", color="tab:green", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Class Accuracy")
    ax.set_title("Validation Mean Class Accuracy (mAcc)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, learning_rates, label="Learning Rate", color="tab:red", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if final:
        plot_path = output_dir / "training_dashboard_final.png"
    else:
        plot_path = output_dir / f"training_dashboard_epoch_{epoch:03d}.png"
    
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Also save a compact version for quick viewing
    if final:
        compact_path = output_dir / "training_dashboard_latest.png"
        plt.savefig(compact_path, dpi=150, bbox_inches="tight")


def save_history_json(history: List[Dict], output_dir: Path) -> None:
    """Save training history as JSON for easy loading."""
    json_path = output_dir / "training_history.json"
    with json_path.open("w") as f:
        json.dump(history, f, indent=2)


def run_epoch(
    model: PointNet,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    feature_reg_weight: float,
    epoch: int,
    log_interval: int,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    description = "train" if training else "val"
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [{description}]", leave=False)

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_idx, (points, labels) in enumerate(progress, start=1):
            points = points.to(device, non_blocking=(device.type in ("cuda", "xpu")))
            labels = labels.to(device, dtype=torch.long, non_blocking=(device.type in ("cuda", "xpu")))
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits, feature_transform = model(points)
            classification_loss = criterion(logits, labels)
            reg_loss = feature_transform_regularizer(feature_transform).to(device)
            loss = classification_loss + feature_reg_weight * reg_loss

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            predictions = logits.detach().argmax(dim=1)
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.detach().cpu())
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
            if training and log_interval > 0 and batch_idx % log_interval == 0:
                progress.write(f"  batch {batch_idx}/{len(loader)} | loss={loss.detach().item():.4f}")

    predictions_all = torch.cat(all_predictions)
    targets_all = torch.cat(all_targets)
    return {
        "loss": total_loss / max(total_samples, 1),
        "overall_accuracy": accuracy(predictions_all, targets_all),
        "mean_class_accuracy": mean_class_accuracy(
            predictions_all, targets_all, n_classes=model.num_classes
        ),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw root     : {args.raw_root.resolve()}")
    print(f"Processed    : {args.processed_root.resolve()}")
    print(f"Mode         : {args.mode}")
    print(f"Device       : {device}")
    if device.type == "cpu":
        print("Note: training on CPU; start with fewer epochs to verify the pipeline.")

    # Use the existing dataset implementation. Hold out a reproducible part of
    # train for validation; the official test split remains untouched until final evaluation.
    
    # Create augmentation config for training
    from common.augmentation import AugmentationConfig
    if args.no_aug:
        aug_config = None
    else:
        aug_config = AugmentationConfig(
            rotation_range=args.aug_rotation,
            scale_range=args.aug_scale,
            translate_range=args.aug_translate,
            jitter_std=args.aug_jitter_std,
            jitter_clip=args.aug_jitter_clip,
            enabled=not args.no_aug,
        )
    
    full_train = Grocery3DDataset(
        split="train", mode=args.mode, raw_root=args.raw_root,
        processed_root=args.processed_root,
        augmentation=aug_config,
    )
    n_val = max(1, int(round(len(full_train) * args.val_ratio)))
    n_train = len(full_train) - n_val
    split_generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=split_generator
    )
    pin_memory = device.type in ("cuda", "xpu")
    train_loader = make_dataloader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        seed=args.seed, drop_last=True,
    )
    val_loader = make_dataloader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory, seed=args.seed,
    )

    model = PointNet(
        num_classes=full_train.n_classes,
        input_channels=3 if args.mode == "xyz" else 6,
        feature_transform=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    
    # Create scheduler based on args
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.cosine_t_max
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.plateau_factor,
            patience=args.plateau_patience, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    history_path = args.output_dir / "training_history.csv"
    history = []
    best_val_acc = -1.0
    best_val_macc = -1.0
    best_path = args.output_dir / "best_pointnet.pt"
    last_path = args.output_dir / "last_pointnet.pt"
    
    # Early stopping state
    early_stop_counter = 0
    early_stop_triggered = False
    
    # Resume support: check for existing checkpoints
    start_epoch = 1
    if last_path.exists():
        print(f"Found existing checkpoint: {last_path}")
        print("Resuming training...")
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_accuracy", -1.0)
        best_val_macc = checkpoint.get("best_val_macc", -1.0)
        
        # Load history if exists
        if history_path.exists():
            with history_path.open("r") as f:
                reader = csv.DictReader(f)
                history = [row for row in reader]
                # Convert string values to float
                for row in history:
                    for key in row:
                        if key != "epoch":
                            row[key] = float(row[key])
                        else:
                            row[key] = int(row[key])
            print(f"Loaded history with {len(history)} epochs")

    epoch_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer,
            args.feature_reg_weight, epoch, args.log_interval,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, None,
            args.feature_reg_weight, epoch, args.log_interval,
        )
        
        # Step scheduler - handle ReduceLROnPlateau differently
        if args.scheduler == "plateau":
            scheduler.step(val_metrics["overall_accuracy"])
        else:
            scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_duration_sec": epoch_duration,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        
        # Save CSV history
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        
        # Save JSON history
        save_history_json(history, args.output_dir)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, OA {train_metrics['overall_accuracy']:.4f}, "
            f"mAcc {train_metrics['mean_class_accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, OA {val_metrics['overall_accuracy']:.4f}, "
            f"mAcc {val_metrics['mean_class_accuracy']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"time {epoch_duration:.1f}s"
        )
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_accuracy": max(best_val_acc, val_metrics["overall_accuracy"]),
            "best_val_macc": max(best_val_macc, val_metrics["mean_class_accuracy"]),
            "model_config": {
                "num_classes": full_train.n_classes,
                "input_channels": 3 if args.mode == "xyz" else 6,
                "feature_transform": True,
            },
            "training_config": {
                "mode": args.mode,
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "feature_reg_weight": args.feature_reg_weight,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "augmentation": {
                    "enabled": not args.no_aug,
                    "rotation_range": args.aug_rotation,
                    "scale_range": args.aug_scale,
                    "translate_range": args.aug_translate,
                    "jitter_std": args.aug_jitter_std,
                    "jitter_clip": args.aug_jitter_clip,
                } if not args.no_aug else None,
            },
        }
        torch.save(checkpoint, last_path)
        
        # Check for best validation accuracy
        is_best = False
        if val_metrics["overall_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["overall_accuracy"]
            best_val_macc = val_metrics["mean_class_accuracy"]
            is_best = True
            checkpoint["best_val_accuracy"] = best_val_acc
            checkpoint["best_val_macc"] = best_val_macc
            torch.save(checkpoint, best_path)
            print(f"  Saved new best checkpoint: {best_path} (val OA={best_val_acc:.4f}, mAcc={best_val_macc:.4f})")
        
        # Early stopping check
        if args.early_stopping_patience > 0:
            if val_metrics["overall_accuracy"] > best_val_acc + args.early_stopping_min_delta:
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs "
                          f"(no improvement in val OA for {args.early_stopping_patience} epochs)")
                    early_stop_triggered = True
        
        # Generate plots at intervals
        if (args.plot_interval > 0 and not args.no_plot and PLOTTING_AVAILABLE and
            (epoch % args.plot_interval == 0 or is_best or epoch == args.epochs)):
            save_training_plots(history, args.output_dir, epoch)
            print(f"  Saved training dashboard plots")
        
        if early_stop_triggered:
            break
    
    # Save final plots
    if not args.no_plot and PLOTTING_AVAILABLE:
        save_training_plots(history, args.output_dir, epoch, final=True)
        print(f"  Saved final training dashboard plots")
    elif not PLOTTING_AVAILABLE and not args.no_plot:
        print("  Note: matplotlib not available, skipping plot generation")

    total_duration = time.time() - epoch_start_time
    summary = {
        "best_validation_overall_accuracy": best_val_acc,
        "best_validation_mean_class_accuracy": best_val_macc,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_csv": str(history_path),
        "history_json": str(args.output_dir / "training_history.json"),
        "epochs_completed": epoch,
        "total_training_time_sec": total_duration,
        "early_stopped": early_stop_triggered,
        "device": str(device),
        "mode": args.mode,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))
    print("Official test data was not used during training/validation; run evaluate_pointnet.py for final metrics.")


if __name__ == "__main__":
    main()
