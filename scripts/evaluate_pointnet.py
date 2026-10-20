"""Evaluate a trained PointNet checkpoint on the official 3DGrocery100 test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import Grocery3DDataset, make_dataloader  # noqa: E402
from common.metrics import compute_all_metrics, confusion_matrix_np, per_class_accuracy  # noqa: E402
from models.pointnet import PointNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PointNet on the official test split")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "pointnet" / "best_pointnet.pt")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "pointnet")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.raw_root.is_dir():
        parser.error(f"--raw-root is not an existing directory: {args.raw_root}")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be >= 1 and --num-workers >= 0")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU requested but unavailable in this PyTorch installation")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = checkpoint["model_config"]
    training_cfg = checkpoint.get("training_config", {})
    mode = training_cfg.get("mode", "xyz" if model_cfg["input_channels"] == 3 else "xyzrgb")

    test_dataset = Grocery3DDataset(
        split="test", mode=mode, raw_root=args.raw_root,
        processed_root=args.processed_root,
    )
    test_loader = make_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type in ("cuda", "xpu"),
    )

    model = PointNet(
        num_classes=int(model_cfg["num_classes"]),
        input_channels=int(model_cfg["input_channels"]),
        feature_transform=bool(model_cfg.get("feature_transform", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for points, labels in tqdm(test_loader, desc="Evaluating official test split"):
            points = points.to(device, non_blocking=device.type in ("cuda", "xpu"))
            logits, _ = model(points)
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_targets.append(labels.to(dtype=torch.long).cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    metrics = compute_all_metrics(predictions, targets, n_classes=model.num_classes)
    metrics["test_samples"] = int(targets.numel())
    metrics["checkpoint"] = str(args.checkpoint.resolve())
    metrics["mode"] = mode
    metrics["device"] = str(device)
    class_acc = per_class_accuracy(predictions, targets, n_classes=model.num_classes)
    class_names = {int(k): v for k, v in test_dataset.id_to_label.items()}
    confusion = confusion_matrix_np(predictions, targets, n_classes=model.num_classes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.output_dir / "per_class_accuracy.csv").open("w", newline="") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "accuracy"])
        for class_id in range(model.num_classes):
            value = class_acc[class_id]
            writer.writerow([class_id, class_names.get(class_id, str(class_id)), value])
    np.savetxt(args.output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")

    print(json.dumps(metrics, indent=2))
    print(f"Per-class accuracy: {args.output_dir / 'per_class_accuracy.csv'}")
    print(f"Confusion matrix : {args.output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
