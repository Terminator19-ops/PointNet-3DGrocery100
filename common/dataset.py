"""
common/dataset.py
=================
PyTorch Dataset and DataLoader factory for 3DGrocery100.

All four model tracks (PointNet, PointNet++, DGCNN, PointMLP) must import
from this module.  No model-specific logic lives here.

Typical usage
-------------
>>> from common.dataset import Grocery3DDataset, make_dataloader
>>> train_ds = Grocery3DDataset(split="train", mode="xyz")
>>> points, label = train_ds[0]
>>> # points.shape == torch.Size([1024, 3])
>>> # label is an int in [0, 99]

>>> loader = make_dataloader(train_ds, batch_size=32, shuffle=True)
>>> batch_pts, batch_lbl = next(iter(loader))
>>> # batch_pts.shape == torch.Size([32, 1024, 3])
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.preprocessing import (
    RANDOM_SEED,
    N_POINTS,
    normalize_xyz,
    sample_points,
    select_channels,
    validate_points,
    clean_points,
    set_global_seed,
)
from common.augmentation import (
    AugmentationConfig,
    augment_point_cloud,
    BASELINE_CONFIG,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default resolved paths  (relative to project root, or absolute from config)
# ---------------------------------------------------------------------------
_DEFAULT_PROCESSED_ROOT = Path("data/processed")
_DEFAULT_SPLITS_DIR = _DEFAULT_PROCESSED_ROOT / "splits"
_DEFAULT_LABELS_JSON = _DEFAULT_PROCESSED_ROOT / "labels.json"

# Raw .dat file locations (model-agnostic default — read from config ideally)
_DEFAULT_RAW_ROOT = Path(
    "C:/Users/anees/Desktop/Clubs/MISC/misc projects/"
    "Groceries Project Datasets/3DGrocery100"
)


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def load_label_map(labels_json: Union[str, Path]) -> Dict[str, dict]:
    """
    Load the label map from data/processed/labels.json.

    Returns
    -------
    label_map : dict  {str(class_id): {"name": str, "category": str}}
    """
    path = Path(labels_json)
    if not path.exists():
        raise FileNotFoundError(
            f"labels.json not found at {path}. "
            "Run `python scripts/prepare_dataset.py` first."
        )
    with open(path, "r") as fh:
        return json.load(fh)


def build_id_name_maps(
    labels_json: Union[str, Path],
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Build bidirectional class-ID ↔ class-name lookup tables.

    Returns
    -------
    id_to_label : dict  {int class_id: str class_name}
    label_to_id : dict  {str class_name: int class_id}
    """
    label_map = load_label_map(labels_json)
    id_to_label: Dict[int, str] = {
        int(k): v["name"] for k, v in label_map.items()
    }
    label_to_id: Dict[str, int] = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class Grocery3DDataset(Dataset):
    """
    PyTorch Dataset for the 3DGrocery100 grocery point-cloud dataset.

    Parameters
    ----------
    split : {"train", "test"}
        Which split to load.
    mode : {"xyz", "xyzrgb"}
        Channel mode.  "xyz" returns (1024, 3); "xyzrgb" returns (1024, 6).
    raw_root : str or Path, optional
        Directory containing the raw .dat files.
        Defaults to the path in configs/data_config.yaml or the hardcoded
        dataset path.
    processed_root : str or Path, optional
        Directory containing processed labels.json and splits/*.json.
    apply_normalization : bool
        Whether to apply per-object unit-sphere normalization at load time.
        Default True (re-applies normalization even though .dat data is
        already normalized, to guarantee consistency).
    n_points : int
        Target number of points.  Default 1024.
    seed : int
        RNG seed for any re-sampling needed.
    metadata : bool
        If True, __getitem__ returns (points, label, sample_id).
        If False (default), returns (points, label).

    Notes
    -----
    - Data is loaded from pre-sampled .dat files; no raw mesh processing.
    - Labels are integers in [0, 99] matching labels.json.
    - The split is fixed from data/processed/splits/{split}.json.
    """

    def __init__(
        self,
        split: str = "train",
        mode: str = "xyzrgb",
        raw_root: Optional[Union[str, Path]] = None,
        processed_root: Optional[Union[str, Path]] = None,
        apply_normalization: bool = True,
        n_points: int = N_POINTS,
        seed: int = RANDOM_SEED,
        metadata: bool = False,
        augmentation: Optional[AugmentationConfig] = None,
    ) -> None:
        super().__init__()

        # --- validate arguments -------------------------------------------
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        mode = mode.lower().strip()
        if mode not in ("xyz", "xyzrgb"):
            raise ValueError(f"mode must be 'xyz' or 'xyzrgb', got '{mode}'")

        self.split = split
        self.mode = mode
        self.apply_normalization = apply_normalization
        self.n_points = n_points
        self.seed = seed
        self.metadata = metadata
        
        # Augmentation: only apply to training split, never to validation/test
        self.augmentation = augmentation if split == "train" else None
        if self.augmentation is not None and not self.augmentation.enabled:
            self.augmentation = None
        
        # RNG for augmentations (seeded for reproducibility)
        self._aug_rng = np.random.default_rng(seed + 12345) if self.augmentation else None

        # --- resolve paths ------------------------------------------------
        self.raw_root = Path(raw_root) if raw_root else _DEFAULT_RAW_ROOT
        self.processed_root = (
            Path(processed_root) if processed_root else _DEFAULT_PROCESSED_ROOT
        )
        self.splits_dir = self.processed_root / "splits"
        self.labels_json = self.processed_root / "labels.json"

        # --- load label map -----------------------------------------------
        self.id_to_label, self.label_to_id = build_id_name_maps(
            self.labels_json
        )
        self.n_classes = len(self.id_to_label)

        # --- load split index ---------------------------------------------
        split_json = self.splits_dir / f"{split}.json"
        if not split_json.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_json}. "
                "Run `python scripts/prepare_dataset.py` first."
            )
        with open(split_json, "r") as fh:
            self._index: List[Dict] = json.load(fh)

        logger.info(
            "Grocery3DDataset | split=%s | mode=%s | %d samples | %d classes | aug=%s",
            split,
            mode,
            len(self._index),
            self.n_classes,
            "enabled" if self.augmentation else "disabled",
        )

        # --- load .dat file into memory (large but fast for training) ------
        dat_name = f"modelnet100_{split}_1024pts_fps.dat"
        dat_path = self.raw_root / dat_name
        if not dat_path.exists():
            raise FileNotFoundError(
                f"DAT file not found: {dat_path}. "
                "Check raw_root in configs/data_config.yaml."
            )

        logger.info("Loading %s …", dat_path)
        with open(dat_path, "rb") as fh:
            raw = pickle.load(fh)

        self._point_clouds: List[np.ndarray] = raw[0]
        self._labels_raw: List[np.ndarray] = raw[1]

        if len(self._point_clouds) != len(self._index):
            raise RuntimeError(
                f"DAT file has {len(self._point_clouds)} samples but "
                f"split JSON has {len(self._index)} entries. "
                "Data integrity problem — re-run prepare_dataset.py."
            )

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str]]:
        """
        Returns
        -------
        points : torch.Tensor, shape [1024, 3] or [1024, 6], dtype float32
        label  : int  in [0, 99]
        sample_id : str  (only when metadata=True)
        """
        entry = self._index[idx]
        sample_id: str = entry["sample_id"]
        label: int = int(entry["class_id"])

        # Fetch raw array from memory-mapped .dat payload
        points_raw: np.ndarray = np.asarray(
            self._point_clouds[idx], dtype=np.float32
        )

        # --- clean ---------------------------------------------------------
        try:
            points = clean_points(points_raw)
        except ValueError:
            # Degenerate sample: fall back to returning zeros with a warning
            logger.warning("Degenerate sample %s — returning zeros", sample_id)
            n_ch = 6 if self.mode == "xyzrgb" else 3
            points_tensor = torch.zeros(self.n_points, n_ch, dtype=torch.float32)
            if self.metadata:
                return points_tensor, label, sample_id
            return points_tensor, label

        # --- resample if needed -------------------------------------------
        if points.shape[0] != self.n_points:
            points = sample_points(points, n=self.n_points, seed=self.seed)

        # --- normalize XYZ ------------------------------------------------
        if self.apply_normalization:
            points = normalize_xyz(points)

        # --- augmentation (training only) ---------------------------------
        if self.augmentation is not None:
            points = augment_point_cloud(points, self.augmentation, self._aug_rng)

        # --- select channels ----------------------------------------------
        points = select_channels(points, mode=self.mode)

        # --- to tensor ----------------------------------------------------
        points_tensor = torch.from_numpy(points)  # already float32

        if self.metadata:
            return points_tensor, label, sample_id
        return points_tensor, label

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_class_name(self, class_id: int) -> str:
        """Return human-readable class name for a given class ID."""
        return self.id_to_label.get(class_id, f"<unknown:{class_id}>")

    def get_class_id(self, class_name: str) -> int:
        """Return integer class ID for a given class name."""
        if class_name not in self.label_to_id:
            raise KeyError(f"Class '{class_name}' not in label map.")
        return self.label_to_id[class_name]

    def get_sample_info(self, idx: int) -> Dict:
        """Return metadata dict for sample at position idx."""
        return dict(self._index[idx])

    def class_counts(self) -> Dict[int, int]:
        """Return {class_id: count} for this split."""
        from collections import Counter
        return dict(Counter(entry["class_id"] for entry in self._index))


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: Grocery3DDataset,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = RANDOM_SEED,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader from a Grocery3DDataset.

    Parameters
    ----------
    dataset      : Grocery3DDataset instance
    batch_size   : samples per batch
    shuffle      : if None, True for train split, False for test split
    num_workers  : parallel data-loading workers (set to 0 on Windows if issues)
    pin_memory   : pin host memory for faster GPU transfer
    seed         : worker seed for reproducibility
    **kwargs     : forwarded to DataLoader

    Returns
    -------
    loader : DataLoader
    """
    if shuffle is None:
        shuffle = (dataset.split == "train")

    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python -m common.dataset)
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    set_global_seed(42)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    print("\n=== Smoke test: Grocery3DDataset ===")
    for split in ("train", "test"):
        for mode in ("xyz", "xyzrgb"):
            ds = Grocery3DDataset(split=split, mode=mode)
            pts, lbl = ds[0]
            expected_ch = 3 if mode == "xyz" else 6
            assert pts.shape == torch.Size([N_POINTS, expected_ch]), (
                f"Shape mismatch for split={split} mode={mode}: {pts.shape}"
            )
            assert isinstance(lbl, int), f"label is not int: {type(lbl)}"
            assert 0 <= lbl <= 99, f"label out of range: {lbl}"
            assert pts.dtype == torch.float32

            loader = make_dataloader(ds, batch_size=4, num_workers=0)
            batch_pts, batch_lbl = next(iter(loader))
            assert batch_pts.shape == torch.Size([4, N_POINTS, expected_ch])
            assert batch_lbl.shape == torch.Size([4])

            print(
                f"  [{split}|{mode}] len={len(ds)} "
                f"pts={pts.shape} lbl={lbl} ({ds.get_class_name(lbl)}) "
                f"batch={batch_pts.shape} ✓"
            )

    print("\nAll smoke tests passed ✓")


if __name__ == "__main__":
    _smoke_test()
