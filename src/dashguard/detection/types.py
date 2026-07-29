"""Data structures shared by object detection implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One target-class detection in pixel coordinates."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
