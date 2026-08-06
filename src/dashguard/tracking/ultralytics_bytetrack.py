"""Ultralytics ByteTrack adapter for consecutive BGR video frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dashguard.detection import TARGET_CLASS_NAMES
from dashguard.tracking.types import Track


def _to_list(values: Any, field_name: str) -> list[Any]:
    """Move tensor-like values to CPU and return a Python list."""

    converted = values
    for method_name in ("detach", "cpu", "tolist"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()

    if isinstance(converted, list):
        return converted

    try:
        return list(converted)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be convertible to a list") from exc


def _class_name(
    names: Mapping[int, str] | Sequence[str],
    class_id: int,
) -> str:
    """Resolve a class name from either supported model names shape."""

    if isinstance(names, Mapping):
        try:
            return names[class_id]
        except KeyError as exc:
            raise ValueError(
                f"No class name exists for class ID {class_id}"
            ) from exc

    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        try:
            return names[class_id]
        except IndexError as exc:
            raise ValueError(
                f"No class name exists for class ID {class_id}"
            ) from exc

    raise TypeError("names must be a mapping or a sequence of class names")


def _target_class_ids(
    names: Mapping[int, str] | Sequence[str],
) -> tuple[int, ...]:
    """Calculate target IDs by matching names supplied by the model."""

    if isinstance(names, Mapping):
        matching_ids = (
            int(class_id)
            for class_id, class_name in names.items()
            if class_name in TARGET_CLASS_NAMES
        )
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        matching_ids = (
            class_id
            for class_id, class_name in enumerate(names)
            if class_name in TARGET_CLASS_NAMES
        )
    else:
        raise TypeError("names must be a mapping or a sequence of class names")

    return tuple(sorted(matching_ids))


def convert_ultralytics_result(
    result: Any,
    frame_index: int,
    timestamp_sec: float,
) -> list[Track]:
    """Convert one Ultralytics-like tracking result into plain tracks."""

    boxes = result.boxes
    if boxes is None:
        return []

    coordinates = _to_list(boxes.xyxy, "boxes.xyxy")
    confidences = _to_list(boxes.conf, "boxes.conf")
    class_ids = _to_list(boxes.cls, "boxes.cls")
    raw_track_ids = boxes.id

    if not coordinates and not confidences and not class_ids:
        if raw_track_ids is None:
            return []

    if raw_track_ids is None:
        raise RuntimeError(
            "Tracking IDs are unavailable: boxes.id is None despite detections"
        )

    track_ids = _to_list(raw_track_ids, "boxes.id")
    lengths = (
        len(coordinates),
        len(confidences),
        len(class_ids),
        len(track_ids),
    )
    if len(set(lengths)) != 1:
        raise ValueError(
            "Tracking array lengths differ: "
            f"xyxy={lengths[0]}, conf={lengths[1]}, "
            f"cls={lengths[2]}, id={lengths[3]}"
        )
    if not coordinates:
        return []

    names = result.names
    converted_timestamp = float(timestamp_sec)
    tracks: list[Track] = []
    for raw_xyxy, raw_confidence, raw_class_id, raw_track_id in zip(
        coordinates,
        confidences,
        class_ids,
        track_ids,
        strict=True,
    ):
        xyxy_values = _to_list(raw_xyxy, "one boxes.xyxy row")
        if len(xyxy_values) != 4:
            raise ValueError(
                "Each boxes.xyxy row must contain exactly four coordinates"
            )

        class_id = int(raw_class_id)
        track_id = int(raw_track_id)
        if track_id <= 0:
            raise ValueError(
                f"Track ID must be greater than zero; received {track_id}"
            )

        class_name = _class_name(names, class_id)
        if class_name not in TARGET_CLASS_NAMES:
            continue

        tracks.append(
            Track(
                track_id=track_id,
                xyxy=(
                    float(xyxy_values[0]),
                    float(xyxy_values[1]),
                    float(xyxy_values[2]),
                    float(xyxy_values[3]),
                ),
                confidence=float(raw_confidence),
                class_name=class_name,
                frame_index=frame_index,
                timestamp_sec=converted_timestamp,
            )
        )

    return tracks


convert_ultralytics_tracking_result = convert_ultralytics_result


class UltralyticsByteTracker:
    """Track target objects in consecutive NumPy BGR frames."""

    def __init__(
        self,
        model: Any,
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        """Configure tracking and calculate target IDs from ``model.names``."""

        self._model = model
        self._confidence = confidence
        self._imgsz = imgsz
        self._device = device
        self._tracker_config = tracker_config
        self._target_class_ids = _target_class_ids(model.names)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
        tracker_config: str = "bytetrack.yaml",
    ) -> UltralyticsByteTracker:
        """Explicitly load an Ultralytics YOLO model and create a tracker."""

        from ultralytics import YOLO

        return cls(
            model=YOLO(str(model_path)),
            confidence=confidence,
            imgsz=imgsz,
            device=device,
            tracker_config=tracker_config,
        )

    def track(
        self,
        image: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Track]:
        """Return target-class tracks for one frame."""

        results = self._model.track(
            source=image,
            persist=True,
            tracker=self._tracker_config,
            conf=self._confidence,
            imgsz=self._imgsz,
            device=self._device,
            classes=list(self._target_class_ids),
            verbose=False,
        )
        result_count = len(results)
        if result_count != 1:
            raise RuntimeError(
                "Single-image tracking must return exactly one result; "
                f"received {result_count}"
            )

        result = results[0]
        if result.boxes is not None and result.boxes.id is None:
            return []

        return convert_ultralytics_result(
            result,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
