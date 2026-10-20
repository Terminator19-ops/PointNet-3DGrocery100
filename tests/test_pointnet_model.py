"""Fast model-only tests; these do not require the raw 3DGrocery100 files."""

import pytest
import torch

from models.pointnet import PointNet, feature_transform_regularizer


@pytest.mark.parametrize("channels", [3, 6])
def test_pointnet_output_shapes_and_finite_loss(channels: int) -> None:
    torch.manual_seed(7)
    model = PointNet(num_classes=100, input_channels=channels)
    model.train()
    points = torch.randn(2, 1024, channels)
    logits, transform = model(points)

    assert logits.shape == (2, 100)
    assert transform is not None and transform.shape == (2, 64, 64)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([4, 91]))
    loss = loss + 1e-3 * feature_transform_regularizer(transform)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fc3.weight.grad is not None
    assert torch.isfinite(model.fc3.weight.grad).all()


def test_pointnet_is_permutation_invariant_in_eval_mode() -> None:
    torch.manual_seed(11)
    model = PointNet(num_classes=100, input_channels=3).eval()
    points = torch.randn(2, 128, 3)
    permutation = torch.randperm(points.shape[1])
    with torch.no_grad():
        logits_a, _ = model(points)
        logits_b, _ = model(points[:, permutation, :])
    torch.testing.assert_close(logits_a, logits_b, rtol=1e-4, atol=1e-5)


def test_rejects_wrong_channel_count() -> None:
    model = PointNet(num_classes=100, input_channels=3)
    with pytest.raises(ValueError, match="expects 3 channels"):
        model(torch.randn(2, 32, 6))
