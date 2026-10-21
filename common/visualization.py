"""
common/visualization.py
=======================
Visualization utilities for 3DGrocery100 point clouds.

All plots are saved to files (no interactive display required) so they
work in headless/server environments as well.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logger = logging.getLogger(__name__)

_VIZ_DIR = Path("results/visualizations")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _scatter3d(
    ax,
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    title: str = "",
    point_size: float = 0.5,
) -> None:
    """Plot a single point cloud on a 3D axes."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = rgb if rgb is not None else z  # colour by height if no RGB
    cmap = None if rgb is not None else "viridis"
    ax.scatter(x, y, z, c=c, s=point_size, cmap=cmap, linewidths=0)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])


# ---------------------------------------------------------------------------
# Single sample
# ---------------------------------------------------------------------------

def plot_point_cloud(
    points: np.ndarray,
    class_name: str = "",
    mode: str = "xyzrgb",
    out_path: Optional[str | Path] = None,
    point_size: float = 0.5,
) -> Path:
    """
    Plot a single point cloud and save to file.

    Parameters
    ----------
    points     : array (N, 3) or (N, 6)
    class_name : used in the title
    mode       : "xyz" or "xyzrgb"
    out_path   : save path; auto-generated if None
    point_size : scatter marker size

    Returns
    -------
    Path to the saved figure
    """
    xyz = points[:, :3]
    rgb = None
    if mode == "xyzrgb" and points.shape[1] >= 6:
        rgb = np.clip(points[:, 3:6], 0.0, 1.0)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    _scatter3d(ax, xyz, rgb, title=class_name, point_size=point_size)
    plt.tight_layout()

    if out_path is None:
        _VIZ_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = class_name.replace(" ", "_").replace("/", "_")
        out_path = _VIZ_DIR / f"cloud_{safe_name}.png"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Contact sheet (multiple classes)
# ---------------------------------------------------------------------------

def plot_contact_sheet(
    samples: List[Tuple[np.ndarray, str]],   # [(points, class_name), …]
    out_path: Optional[str | Path] = None,
    cols: int = 5,
    point_size: float = 0.4,
    title: str = "3DGrocery100 — Sample Point Clouds",
) -> Path:
    """
    Plot a grid of point clouds from different classes.

    Parameters
    ----------
    samples   : list of (points_array, class_name) tuples
    out_path  : save path; auto-generated if None
    cols      : number of columns in the grid
    point_size: scatter marker size
    title     : figure super-title

    Returns
    -------
    Path to saved figure
    """
    n = len(samples)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 3, rows * 2.8))
    fig.suptitle(title, fontsize=11, y=1.01)

    for i, (pts, cls_name) in enumerate(samples):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        xyz = pts[:, :3]
        rgb = None
        if pts.shape[1] >= 6:
            rgb = np.clip(pts[:, 3:6], 0.0, 1.0)
        _scatter3d(ax, xyz, rgb, title=cls_name, point_size=point_size)

    plt.tight_layout()

    if out_path is None:
        _VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _VIZ_DIR / "contact_sheet.png"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Point-count distribution
# ---------------------------------------------------------------------------

def plot_point_count_distribution(
    counts: Sequence[int],
    out_path: Optional[str | Path] = None,
    title: str = "Point-Count Distribution",
    n_bins: int = 30,
) -> Path:
    """
    Histogram of per-sample point counts.

    Parameters
    ----------
    counts   : iterable of integer point counts
    out_path : save path; auto-generated if None
    title    : plot title
    n_bins   : number of histogram bins

    Returns
    -------
    Path to saved figure
    """
    counts_arr = np.array(list(counts))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts_arr, bins=n_bins, color="#4C72B0", edgecolor="white",
            linewidth=0.5)
    ax.set_xlabel("Number of points per sample")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.axvline(counts_arr.mean(), color="red", linestyle="--",
               label=f"Mean = {counts_arr.mean():.0f}")
    ax.axvline(np.median(counts_arr), color="orange", linestyle="--",
               label=f"Median = {np.median(counts_arr):.0f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if out_path is None:
        _VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _VIZ_DIR / "point_count_distribution.png"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_class_distribution(
    class_names: List[str],
    counts: List[int],
    out_path: Optional[str | Path] = None,
    title: str = "Samples per Class",
) -> Path:
    """
    Horizontal bar chart of per-class sample counts.

    Parameters
    ----------
    class_names : list of class name strings
    counts      : corresponding sample counts
    out_path    : save path; auto-generated if None
    title       : plot title

    Returns
    -------
    Path to saved figure
    """
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.2)))
    y_pos = np.arange(n)
    ax.barh(y_pos, counts, color="#4C72B0", edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Number of samples")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if out_path is None:
        _VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _VIZ_DIR / "class_distribution.png"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path
