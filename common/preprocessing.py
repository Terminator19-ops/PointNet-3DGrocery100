"""
common/preprocessing.py
=======================
Reusable, model-agnostic preprocessing functions for the 3DGrocery100
point-cloud dataset.

Dataset facts (verified from raw files):
  - Each .dat file is a pickle of [list_of_arrays, list_of_label_arrays]
  - Every array is (1024, 6) float32  (columns: X Y Z R G B)
  - XYZ is already unit-sphere normalized per object in the .dat files
  - RGB values are in [0, 1] float range
  - Farthest Point Sampling (FPS) was used during dataset construction
  - The official train/test split is encoded in modelnet100_train.txt /
    modelnet100_test.txt (66 032 train, 21 866 test samples)

All functions are pure (no side-effects) and accept/return NumPy arrays
so they can be tested independently of PyTorch.
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_POINTS: int = 1024
N_CHANNELS_XYZ: int = 3
N_CHANNELS_XYZRGB: int = 6
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set fixed seeds for Python, NumPy, and PyTorch.

    Call once at the start of every script / training run to guarantee
    reproducibility.  The seed value is read from configs/data_config.yaml;
    this function is the single place that applies it.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not required for pure preprocessing
    logger.debug("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Raw-file loading
# ---------------------------------------------------------------------------

def load_dat_file(dat_path: str | Path) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load a 3DGrocery100 / ModelNet-style pre-sampled .dat file.

    The file format is a pickle of::

        [
            [array(1024, 6), ...],   # point clouds
            [array([label]), ...],   # integer class labels
        ]

    Parameters
    ----------
    dat_path : str or Path
        Path to the .dat file.

    Returns
    -------
    point_clouds : list of ndarray, each shape (1024, 6) float32
    labels       : list of int
    """
    dat_path = Path(dat_path)
    if not dat_path.exists():
        raise FileNotFoundError(f"DAT file not found: {dat_path}")

    logger.info("Loading %s (%.1f MB) …", dat_path.name,
                dat_path.stat().st_size / 1e6)

    with open(dat_path, "rb") as fh:
        raw = pickle.load(fh)

    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise ValueError(
            f"Unexpected DAT structure in {dat_path}. "
            "Expected [list_of_clouds, list_of_labels]."
        )

    point_clouds: List[np.ndarray] = []
    labels: List[int] = []

    for pc, lbl in zip(raw[0], raw[1]):
        arr = np.asarray(pc, dtype=np.float32)
        point_clouds.append(arr)
        labels.append(int(np.asarray(lbl).flatten()[0]))

    logger.info("Loaded %d samples from %s", len(point_clouds), dat_path.name)
    return point_clouds, labels


def load_split_ids(txt_path: str | Path) -> List[str]:
    """
    Load a ModelNet-style split .txt file.

    Each line has the format::

        <class_name>=<UUID>_<class_name>_<frame_id>

    Returns a list of sample ID strings (order preserved).
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Split file not found: {txt_path}")

    ids: List[str] = []
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_shape_names(names_path: str | Path) -> List[str]:
    """
    Load the ordered list of class names from a *_shape_names.txt file.

    The position in the list equals the integer class ID (0-indexed).
    """
    names_path = Path(names_path)
    with open(names_path, "r") as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    return names


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_points(
    points: np.ndarray,
    label: int,
    n_classes: int = 100,
    require_rgb: bool = False,
    sample_id: str = "<unknown>",
) -> Dict[str, object]:
    """
    Validate a single point-cloud sample.

    Parameters
    ----------
    points      : array to validate, expected (N, 3) or (N, 6)
    label       : integer class ID
    n_classes   : total number of valid classes
    require_rgb : if True, XYZRGB must be present
    sample_id   : string identifier used in error messages

    Returns
    -------
    result : dict with keys
        status  : "VALID" | "WARNING" | "REJECTED"
        issues  : list[str]  — human-readable problem descriptions
    """
    issues: List[str] = []

    # --- shape checks -------------------------------------------------------
    if not isinstance(points, np.ndarray):
        issues.append("NOT_NDARRAY: points is not a numpy array")
        return {"status": "REJECTED", "issues": issues}

    if points.ndim != 2:
        issues.append(f"INVALID_NDIM: expected 2D array, got {points.ndim}D")
        return {"status": "REJECTED", "issues": issues}

    n_pts, n_ch = points.shape

    if n_pts == 0:
        issues.append("EMPTY: point cloud has zero points")
        return {"status": "REJECTED", "issues": issues}

    if n_ch < 3:
        issues.append(
            f"TOO_FEW_CHANNELS: expected at least 3 (XYZ), got {n_ch}"
        )
        return {"status": "REJECTED", "issues": issues}

    if require_rgb and n_ch < 6:
        issues.append(
            f"NO_RGB: mode=xyzrgb requires 6 channels, got {n_ch}"
        )
        return {"status": "REJECTED", "issues": issues}

    # --- numeric checks -----------------------------------------------------
    xyz = points[:, :3]

    if not np.issubdtype(points.dtype, np.floating):
        issues.append(
            f"BAD_DTYPE: expected float, got {points.dtype}; will cast"
        )

    if np.any(np.isnan(xyz)):
        issues.append("NAN_XYZ: NaN values in XYZ coordinates")
        return {"status": "REJECTED", "issues": issues}

    if np.any(np.isinf(xyz)):
        issues.append("INF_XYZ: Inf values in XYZ coordinates")
        return {"status": "REJECTED", "issues": issues}

    # --- RGB range check (non-fatal) ----------------------------------------
    if n_ch >= 6:
        rgb = points[:, 3:6]
        if np.any(np.isnan(rgb)) or np.any(np.isinf(rgb)):
            issues.append("NAN_INF_RGB: NaN/Inf in RGB channels")
        if rgb.min() < -0.01 or rgb.max() > 1.01:
            issues.append(
                f"RGB_OUT_OF_RANGE: expected [0,1], got "
                f"[{rgb.min():.4f}, {rgb.max():.4f}]"
            )

    # --- point count warning ------------------------------------------------
    if n_pts != N_POINTS:
        issues.append(
            f"POINT_COUNT_MISMATCH: expected {N_POINTS}, got {n_pts}; "
            "will resample"
        )

    # --- label check --------------------------------------------------------
    if not isinstance(label, (int, np.integer)):
        issues.append(f"BAD_LABEL_TYPE: label is {type(label)}, expected int")
        return {"status": "REJECTED", "issues": issues}

    if label < 0 or label >= n_classes:
        issues.append(
            f"LABEL_OUT_OF_RANGE: label={label} not in [0, {n_classes - 1}]"
        )
        return {"status": "REJECTED", "issues": issues}

    # --- final verdict ------------------------------------------------------
    if issues:
        return {"status": "WARNING", "issues": issues}
    return {"status": "VALID", "issues": []}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_points(points: np.ndarray) -> np.ndarray:
    """
    Remove or repair minor data quality issues.

    Operations performed (in order):
    1. Cast to float32
    2. Drop rows that contain NaN or Inf in XYZ (channels 0-2)
    3. Clamp RGB channels to [0, 1]

    If all rows were dropped, raises ValueError.

    Parameters
    ----------
    points : array (N, C) where C >= 3

    Returns
    -------
    cleaned : array (M, C) float32, M <= N
    """
    points = np.asarray(points, dtype=np.float32)

    # Drop rows with NaN/Inf in XYZ
    xyz = points[:, :3]
    bad_mask = np.any(np.isnan(xyz) | np.isinf(xyz), axis=1)
    n_bad = bad_mask.sum()
    if n_bad > 0:
        logger.warning("Removing %d point(s) with NaN/Inf XYZ", n_bad)
        points = points[~bad_mask]

    if points.shape[0] == 0:
        raise ValueError("All points removed during cleaning — sample unusable")

    # Clamp RGB
    if points.shape[1] >= 6:
        points[:, 3:6] = np.clip(points[:, 3:6], 0.0, 1.0)

    return points


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_xyz(points: np.ndarray) -> np.ndarray:
    """
    Apply per-object unit-sphere normalization to XYZ coordinates.

    Procedure (matches the 3DGrocery100 dataset construction convention):
    1. Compute the centroid of the point cloud: mean over all XYZ points
    2. Subtract the centroid (centre the cloud at the origin)
    3. Compute the maximum L2 distance from the origin
    4. Divide all XYZ coordinates by that max distance

    Result: every point is in [-1, 1]^3 (approximately), with the furthest
    point exactly on the unit sphere.

    RGB channels (columns 3-5) are NOT modified.

    This is the IDENTICAL normalization applied by all four model tracks.
    Changing this function changes normalization for the entire project.

    Parameters
    ----------
    points : array (N, C), C >= 3

    Returns
    -------
    normalized : array (N, C) float32 — same shape, XYZ columns modified
    """
    points = np.asarray(points, dtype=np.float32).copy()
    xyz = points[:, :3]

    centroid = xyz.mean(axis=0)          # shape (3,)
    xyz -= centroid                      # centre at origin

    max_dist = np.sqrt((xyz ** 2).sum(axis=1)).max()
    if max_dist < 1e-8:
        logger.warning("Near-zero bounding sphere — normalization skipped")
    else:
        xyz /= max_dist                  # scale to unit sphere

    points[:, :3] = xyz
    return points


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_points(
    points: np.ndarray,
    n: int = N_POINTS,
    method: str = "random",
    seed: Optional[int] = RANDOM_SEED,
) -> np.ndarray:
    """
    Resample a point cloud to exactly *n* points.

    Strategy
    --------
    - If len(points) == n   : return as-is (no copy unless dtype conversion)
    - If len(points) >  n   : random downsample without replacement
    - If len(points) <  n   : repeat-pad (deterministic), then random shuffle

    The repeat-pad strategy: tile the cloud until >= n points exist, then
    randomly sample n without replacement.  This is reproducible given a
    fixed seed and avoids information loss while maintaining the required
    output shape.

    Parameters
    ----------
    points : array (N, C)
    n      : target number of points
    method : "random" (only option currently; "fps" requires Open3D)
    seed   : random seed for reproducibility; None = use global state

    Returns
    -------
    sampled : array (n, C) float32
    """
    points = np.asarray(points, dtype=np.float32)
    current_n = points.shape[0]

    if current_n == n:
        return points

    rng = np.random.default_rng(seed)

    if current_n > n:
        # Downsample: random without replacement
        idx = rng.choice(current_n, size=n, replace=False)
        idx.sort()
        return points[idx]

    # Upsample: tile then randomly select n (deterministic)
    # Tile the array until we have at least n points
    repeats = int(np.ceil(n / current_n))
    tiled = np.tile(points, (repeats, 1))
    idx = rng.choice(tiled.shape[0], size=n, replace=False)
    idx.sort()
    return tiled[idx]


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

def select_channels(points: np.ndarray, mode: str = "xyzrgb") -> np.ndarray:
    """
    Return the requested channel subset of a (N, 6) array.

    Parameters
    ----------
    points : array (N, C), C >= 3
    mode   : "xyz" → columns 0:3; "xyzrgb" → columns 0:6

    Returns
    -------
    subset : array (N, 3) or (N, 6)
    """
    mode = mode.lower().strip()
    if mode == "xyz":
        return points[:, :3].astype(np.float32)
    elif mode == "xyzrgb":
        if points.shape[1] < 6:
            raise ValueError(
                f"mode='xyzrgb' requires 6 channels, but array has "
                f"{points.shape[1]} columns. "
                "Use mode='xyz' for this sample."
            )
        return points[:, :6].astype(np.float32)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'xyz' or 'xyzrgb'.")


# ---------------------------------------------------------------------------
# End-to-end single-sample processing
# ---------------------------------------------------------------------------

def process_sample(
    points: np.ndarray,
    label: int,
    mode: str = "xyzrgb",
    n_points: int = N_POINTS,
    apply_normalization: bool = True,
    seed: Optional[int] = RANDOM_SEED,
    n_classes: int = 100,
    sample_id: str = "<unknown>",
) -> Tuple[Optional[np.ndarray], Optional[int], Dict]:
    """
    Full preprocessing pipeline for a single sample.

    Steps:
    1. validate_points
    2. clean_points (if not REJECTED)
    3. sample_points to n_points
    4. normalize_xyz (optional)
    5. select_channels

    Parameters
    ----------
    points            : raw point array
    label             : integer class ID
    mode              : "xyz" or "xyzrgb"
    n_points          : target point count
    apply_normalization : whether to run normalize_xyz
    seed              : RNG seed for sampling
    n_classes         : number of valid classes
    sample_id         : for logging/reports

    Returns
    -------
    processed_points : array (n_points, 3 or 6) or None if rejected
    label            : int or None if rejected
    report           : dict with status / issues
    """
    require_rgb = (mode == "xyzrgb")
    report = validate_points(
        points, label,
        n_classes=n_classes,
        require_rgb=require_rgb,
        sample_id=sample_id,
    )

    if report["status"] == "REJECTED":
        return None, None, report

    # Clean
    try:
        points = clean_points(points)
    except ValueError as exc:
        report["status"] = "REJECTED"
        report["issues"].append(f"CLEAN_FAILED: {exc}")
        return None, None, report

    # Ensure we have enough channels for xyzrgb mode
    if require_rgb and points.shape[1] < 6:
        report["status"] = "REJECTED"
        report["issues"].append(
            "INSUFFICIENT_CHANNELS: need 6 for xyzrgb after cleaning"
        )
        return None, None, report

    # Sample to n_points
    points = sample_points(points, n=n_points, seed=seed)

    # Normalize XYZ
    if apply_normalization:
        points = normalize_xyz(points)

    # Select channels
    points = select_channels(points, mode=mode)

    return points, int(label), report
