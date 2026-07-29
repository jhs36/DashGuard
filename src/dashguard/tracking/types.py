"""Data structures shared by object tracking implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    """One target-class track observation in a video frame."""

    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str
    frame_index: int
    timestamp_sec: float
