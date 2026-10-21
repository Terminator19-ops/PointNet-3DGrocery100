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
from pathlib import Path
from typing import Dict, Tuple

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
from common.device import choose_device, seed_device  # noqa: E402
from common.metrics import accuracy, mean_class_accuracy  # noqa: E402
from models.pointnet import PointNet, feature_transform_regularizer  # noqa: E402


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
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be >= 1 and --batch-size must be >= 2")
    if not 0.0 < args.val_ratio < 0.5:
        parser.error("--val-ratio must be between 0 and 0.5")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.feature_reg_weight < 0:
        parser.error("Learning rate must be positive; regularization weights cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not args.raw_root.is_dir():
        parser.error(f"--raw-root is not an existing directory: {args.raw_root}")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_device(seed)


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
    full_train = Grocery3DDataset(
        split="train", mode=args.mode, raw_root=args.raw_root,
        processed_root=args.processed_root,
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    history_path = args.output_dir / "training_history.csv"
    history = []
    best_val_acc = -1.0
    best_path = args.output_dir / "best_pointnet.pt"
    last_path = args.output_dir / "last_pointnet.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer,
            args.feature_reg_weight, epoch, args.log_interval,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, None,
            args.feature_reg_weight, epoch, args.log_interval,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, OA {train_metrics['overall_accuracy']:.4f}, "
            f"mAcc {train_metrics['mean_class_accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, OA {val_metrics['overall_accuracy']:.4f}, "
            f"mAcc {val_metrics['mean_class_accuracy']:.4f}"
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_accuracy": max(best_val_acc, val_metrics["overall_accuracy"]),
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
            },
        }
        torch.save(checkpoint, last_path)
        if val_metrics["overall_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["overall_accuracy"]
            checkpoint["best_val_accuracy"] = best_val_acc
            torch.save(checkpoint, best_path)
            print(f"  Saved new best checkpoint: {best_path} (val OA={best_val_acc:.4f})")

    summary = {
        "best_validation_overall_accuracy": best_val_acc,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_csv": str(history_path),
        "epochs_completed": args.epochs,
        "device": str(device),
        "mode": args.mode,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))
    print("Official test data was not used during training/validation; run evaluate_pointnet.py for final metrics.")


if __name__ == "__main__":
    main()
