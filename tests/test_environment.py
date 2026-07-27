from __future__ import annotations

import cv2
import numpy as np
import torch
import ultralytics


def test_core_packages_are_importable() -> None:
    """Core computer-vision packages must expose version information."""
    assert torch.__version__
    assert ultralytics.__version__
    assert cv2.__version__
    assert np.__version__


def test_torch_cpu_tensor_operation() -> None:
    """The Day 1 baseline must support a deterministic CPU tensor operation."""
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = left @ left
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]])

    assert result.device.type == "cpu"
    assert torch.equal(result, expected)


def test_opencv_image_operation() -> None:
    """OpenCV must be able to modify an in-memory image."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (180, 80), (255, 255, 255), 2)

    assert image.shape == (100, 200, 3)
    assert image.dtype == np.uint8
    assert int(image.sum()) > 0