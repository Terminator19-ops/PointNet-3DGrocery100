"""
common/augmentation.py
======================
Point-cloud data augmentation for 3DGrocery100 training.

All augmentations are applied to training data only. Validation and test
data are never augmented to ensure deterministic evaluation.

Augmentations are applied in the following order:
1. Random rotation around vertical axis (Z-axis)
2. Random scaling (uniform)
3. Random translation (jitter)
4. Gaussian jitter on XYZ coordinates

Each augmentation is independently configurable with conservative defaults
suitable for grocery object classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for point-cloud augmentations.
    
    All parameters are conservative defaults for grocery object classification.
    Set any parameter to 0 or False to disable that augmentation.
    """
    # Rotation around vertical axis (Z-axis) in radians
    # Set to 0 to disable. Typical range: 0 to 2*pi for full rotation,
    # or smaller for limited rotation (e.g., pi/4 for ±45°)
    rotation_range: float = 0.0  # Disabled by default for orientation-sensitive objects
    
    # Uniform scaling factor range (multiplicative)
    # Scale is sampled uniformly from [1-scale_range, 1+scale_range]
    # Set to 0 to disable. Typical: 0.05 for ±5%
    scale_range: float = 0.05
    
    # Translation jitter range (additive)
    # Translation sampled uniformly from [-translate_range, translate_range] per axis
    # Set to 0 to disable. Typical: 0.02 for ±2% of unit sphere
    translate_range: float = 0.02
    
    # Gaussian jitter on XYZ coordinates
    # Standard deviation of Gaussian noise added to each XYZ coordinate
    # Set to 0 to disable. Typical: 0.01 for 1% noise
    jitter_std: float = 0.01
    
    # Jitter clip value (prevents extreme outliers)
    # Jitter values are clipped to [-jitter_clip, jitter_clip]
    jitter_clip: float = 0.05
    
    # Whether to apply augmentations (master switch)
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.rotation_range < 0:
            raise ValueError("rotation_range must be non-negative")
        if self.scale_range < 0:
            raise ValueError("scale_range must be non-negative")
        if self.translate_range < 0:
            raise ValueError("translate_range must be non-negative")
        if self.jitter_std < 0:
            raise ValueError("jitter_std must be non-negative")
        if self.jitter_clip <= 0:
            raise ValueError("jitter_clip must be positive")


def rotate_point_cloud_z(
    points: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate point cloud around Z-axis (vertical axis).
    
    Args:
        points: (N, C) array where first 3 columns are XYZ
        angle: Rotation angle in radians
        
    Returns:
        Rotated points (same shape as input)
    """
    if angle == 0:
        return points
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Rotation matrix for Z-axis
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    points = points.copy()
    points[:, :3] = points[:, :3] @ rot_matrix.T
    return points


def scale_point_cloud(
    points: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Apply uniform scaling to XYZ coordinates.
    
    Args:
        points: (N, C) array where first 3 columns are XYZ
        scale: Multiplicative scale factor
        
    Returns:
        Scaled points (same shape as input)
    """
    if scale == 1.0:
        return points
    
    points = points.copy()
    points[:, :3] *= scale
    return points


def translate_point_cloud(
    points: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply translation to XYZ coordinates.
    
    Args:
        points: (N, C) array where first 3 columns are XYZ
        translation: (3,) array of translation offsets
        
    Returns:
        Translated points (same shape as input)
    """
    if np.allclose(translation, 0):
        return points
    
    points = points.copy()
    points[:, :3] += translation
    return points


def jitter_point_cloud(
    points: np.ndarray,
    std: float,
    clip: float,
) -> np.ndarray:
    """Add Gaussian jitter to XYZ coordinates.
    
    Args:
        points: (N, C) array where first 3 columns are XYZ
        std: Standard deviation of Gaussian noise
        clip: Maximum absolute jitter value (clipping)
        
    Returns:
        Jittered points (same shape as input)
    """
    if std == 0:
        return points
    
    points = points.copy()
    n_points = points.shape[0]
    jitter = np.random.normal(0, std, (n_points, 3)).astype(np.float32)
    jitter = np.clip(jitter, -clip, clip)
    points[:, :3] += jitter
    return points


def augment_point_cloud(
    points: np.ndarray,
    config: AugmentationConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply all configured augmentations to a point cloud.
    
    Augmentations are applied in order:
    1. Rotation around Z-axis
    2. Uniform scaling
    3. Translation
    4. Gaussian jitter
    
    Only XYZ coordinates (first 3 channels) are modified.
    RGB channels (if present) are preserved unchanged.
    
    Args:
        points: (N, C) array, C >= 3
        config: AugmentationConfig with parameters
        rng: Optional random generator for reproducibility
        
    Returns:
        Augmented points (same shape as input)
    """
    if not config.enabled:
        return points
    
    if rng is None:
        rng = np.random.default_rng()
    
    points = points.copy()
    
    # 1. Rotation around Z-axis
    if config.rotation_range > 0:
        angle = rng.uniform(-config.rotation_range, config.rotation_range)
        points = rotate_point_cloud_z(points, angle)
    
    # 2. Uniform scaling
    if config.scale_range > 0:
        scale = rng.uniform(1 - config.scale_range, 1 + config.scale_range)
        points = scale_point_cloud(points, scale)
    
    # 3. Translation
    if config.translate_range > 0:
        translation = rng.uniform(
            -config.translate_range, 
            config.translate_range, 
            size=3
        ).astype(np.float32)
        points = translate_point_cloud(points, translation)
    
    # 4. Gaussian jitter
    if config.jitter_std > 0:
        points = jitter_point_cloud(points, config.jitter_std, config.jitter_clip)
    
    return points


def create_augmentation_config(
    rotation_range: float = 0.0,
    scale_range: float = 0.05,
    translate_range: float = 0.02,
    jitter_std: float = 0.01,
    jitter_clip: float = 0.05,
    enabled: bool = True,
) -> AugmentationConfig:
    """Factory function to create AugmentationConfig with custom values.
    
    Args:
        rotation_range: Max rotation angle in radians (0 to disable)
        scale_range: Scale variation range (0 to disable)
        translate_range: Translation range (0 to disable)
        jitter_std: Gaussian jitter std (0 to disable)
        jitter_clip: Jitter clipping value
        enabled: Master enable switch
        
    Returns:
        AugmentationConfig instance
    """
    return AugmentationConfig(
        rotation_range=rotation_range,
        scale_range=scale_range,
        translate_range=translate_range,
        jitter_std=jitter_std,
        jitter_clip=jitter_clip,
        enabled=enabled,
    )


# Baseline-compatible configuration (no augmentation)
BASELINE_CONFIG = AugmentationConfig(
    rotation_range=0.0,
    scale_range=0.0,
    translate_range=0.0,
    jitter_std=0.0,
    jitter_clip=0.05,
    enabled=False,
)

# Conservative augmentation for grocery objects
CONSERVATIVE_CONFIG = AugmentationConfig(
    rotation_range=0.0,  # No rotation - grocery objects have canonical orientation
    scale_range=0.05,    # ±5% scaling
    translate_range=0.02, # ±2% translation
    jitter_std=0.01,     # 1% Gaussian jitter
    jitter_clip=0.05,
    enabled=True,
)

# Moderate augmentation (includes limited rotation)
MODERATE_CONFIG = AugmentationConfig(
    rotation_range=np.pi / 8,  # ±22.5° rotation
    scale_range=0.08,          # ±8% scaling
    translate_range=0.03,      # ±3% translation
    jitter_std=0.015,          # 1.5% Gaussian jitter
    jitter_clip=0.05,
    enabled=True,
)

# Full augmentation (for experimentation)
FULL_CONFIG = AugmentationConfig(
    rotation_range=2 * np.pi,  # Full 360° rotation
    scale_range=0.1,           # ±10% scaling
    translate_range=0.05,      # ±5% translation
    jitter_std=0.02,           # 2% Gaussian jitter
    jitter_clip=0.05,
    enabled=True,
)