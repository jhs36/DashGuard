"""Public single-frame object detection interfaces."""

from dashguard.detection.types import Detection
from dashguard.detection.ultralytics_detector import (
    TARGET_CLASS_NAMES,
    UltralyticsDetector,
    convert_ultralytics_result,
)

__all__ = [
    "Detection",
    "TARGET_CLASS_NAMES",
    "UltralyticsDetector",
    "convert_ultralytics_result",
]
