"""Ultralytics adapter for single-frame target object detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dashguard.detection.types import Detection


TARGET_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    }
)


def _to_list(values: Any, field_name: str) -> list[Any]:
    """Move tensor-like values to CPU and return a regular Python list."""

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
    """Resolve a model class name from either supported names shape."""

    try:
        return names[class_id]
    except (IndexError, KeyError) as exc:
        raise ValueError(f"No class name exists for class ID {class_id}") from exc


def _target_class_ids(
    names: Mapping[int, str] | Sequence[str],
) -> tuple[int, ...]:
    """Calculate target IDs from model-provided class names."""

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


def convert_ultralytics_result(result: Any) -> list[Detection]:
    """Convert one Ultralytics-like result into target-class detections."""

    boxes = result.boxes
    if boxes is None:
        return []

    coordinates = _to_list(boxes.xyxy, "boxes.xyxy")
    confidences = _to_list(boxes.conf, "boxes.conf")
    class_ids = _to_list(boxes.cls, "boxes.cls")

    lengths = (len(coordinates), len(confidences), len(class_ids))
    if len(set(lengths)) != 1:
        raise ValueError(
            "Detection array lengths differ: "
            f"xyxy={lengths[0]}, conf={lengths[1]}, cls={lengths[2]}"
        )
    if not coordinates:
        return []

    names = result.names
    detections: list[Detection] = []
    for raw_xyxy, raw_confidence, raw_class_id in zip(
        coordinates,
        confidences,
        class_ids,
        strict=True,
    ):
        xyxy_values = _to_list(raw_xyxy, "one boxes.xyxy row")
        if len(xyxy_values) != 4:
            raise ValueError(
                "Each boxes.xyxy row must contain exactly four coordinates"
            )

        class_id = int(raw_class_id)
        class_name = _class_name(names, class_id)
        if class_name not in TARGET_CLASS_NAMES:
            continue

        detections.append(
            Detection(
                xyxy=(
                    float(xyxy_values[0]),
                    float(xyxy_values[1]),
                    float(xyxy_values[2]),
                    float(xyxy_values[3]),
                ),
                confidence=float(raw_confidence),
                class_id=class_id,
                class_name=class_name,
            )
        )

    return detections


class UltralyticsDetector:
    """Run an injected Ultralytics model on one BGR image at a time."""

    def __init__(
        self,
        model: Any,
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        """Configure inference and calculate target IDs from ``model.names``."""

        self._model = model
        self._confidence = confidence
        self._imgsz = imgsz
        self._device = device
        self._target_class_ids = _target_class_ids(model.names)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> UltralyticsDetector:
        """Load an Ultralytics YOLO model explicitly and create a detector."""

        from ultralytics import YOLO

        return cls(
            model=YOLO(str(model_path)),
            confidence=confidence,
            imgsz=imgsz,
            device=device,
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return target-class detections for one NumPy BGR frame."""

        results = self._model.predict(
            source=image,
            conf=self._confidence,
            imgsz=self._imgsz,
            device=self._device,
            classes=list(self._target_class_ids),
            verbose=False,
        )
        result_count = len(results)
        if result_count != 1:
            raise RuntimeError(
                "Single-image prediction must return exactly one result; "
                f"received {result_count}"
            )

        return convert_ultralytics_result(results[0])
