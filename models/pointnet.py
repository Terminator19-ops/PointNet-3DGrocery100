"""PointNet classification model for the 3DGrocery100 point-cloud dataset.

Expected input format: (batch_size, num_points, channels), matching
``common.dataset.Grocery3DDataset``. The first three channels are XYZ;
optional channels 3: are preserved and are not treated as coordinates.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Learn a k-by-k feature transform from a set of point features."""

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

        # Start as the identity transform; the network can learn adjustments.
        nn.init.zeros_(self.fc3.weight)
        with torch.no_grad():
            self.fc3.bias.copy_(torch.eye(k).reshape(-1))

    def forward(self, features: Tensor) -> Tensor:
        """Return a transformation matrix of shape (B, k, k).

        Args:
            features: (B, k, N) point or feature tensor.
        """
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2).values
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        return self.fc3(x).reshape(-1, self.k, self.k)


def feature_transform_regularizer(transform: Optional[Tensor]) -> Tensor:
    """PointNet orthogonality penalty for a learned feature transform.

    Computes mean Frobenius norm of ``I - A A^T`` over the batch.
    If ``transform`` is None, returns a scalar zero tensor on the CPU.
    """
    if transform is None:
        return torch.tensor(0.0)
    if transform.ndim != 3 or transform.shape[-1] != transform.shape[-2]:
        raise ValueError("transform must have shape (B, K, K)")
    k = transform.shape[-1]
    identity = torch.eye(k, dtype=transform.dtype, device=transform.device)
    gram = torch.bmm(transform, transform.transpose(1, 2))
    return torch.linalg.matrix_norm(identity.unsqueeze(0) - gram, ord="fro", dim=(-2, -1)).mean()


class PointNet(nn.Module):
    """Vanilla PointNet for whole-object classification.

    Args:
        num_classes: Number of output classes (100 for 3DGrocery100).
        input_channels: 3 for XYZ or 6 for XYZRGB.
        feature_transform: Whether to use the learned 64x64 feature transform.

    Returns:
        ``(logits, feature_transform_matrix)`` where logits has shape
        ``(B, num_classes)``. The transform is ``(B, 64, 64)`` or ``None``.
        Pass logits directly to ``torch.nn.CrossEntropyLoss``; do not apply
        softmax first.
    """

    def __init__(
        self,
        num_classes: int = 100,
        input_channels: int = 3,
        feature_transform: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.use_feature_transform = feature_transform

        # Spatial transform is always 3x3 and is applied to XYZ only.
        # For XYZRGB, RGB channels are concatenated unchanged after transform.
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.feature_transform = TNet(k=64) if feature_transform else None
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Classify input point clouds in (B, N, C) layout."""
        if points.ndim != 3:
            raise ValueError(
                f"Expected points shaped (B, N, C), got {tuple(points.shape)}"
            )
        if points.shape[-1] != self.input_channels:
            raise ValueError(
                f"Model expects {self.input_channels} channels, got {points.shape[-1]}"
            )
        if points.shape[1] < 1:
            raise ValueError("Each point cloud must contain at least one point")

        # Pointwise Conv1d layers expect (B, C, N).
        x = points.transpose(1, 2).contiguous()

        input_transform = self.input_transform(x[:, :3, :])
        xyz = torch.bmm(x[:, :3, :].transpose(1, 2), input_transform).transpose(1, 2)
        if self.input_channels == 3:
            x = xyz
        else:
            x = torch.cat((xyz, x[:, 3:, :]), dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        transform = self.feature_transform(x) if self.feature_transform is not None else None
        if transform is not None:
            x = torch.bmm(x.transpose(1, 2), transform).transpose(1, 2).contiguous()

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Symmetric aggregation makes the global feature invariant to point order.
        x = torch.max(x, dim=2).values
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout2(x)
        logits = self.fc3(x)
        return logits, transform
