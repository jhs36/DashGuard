"""Convert BDD100K Detection 2020 frame annotations to YOLO labels."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import cv2


YOLO_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)

BDD100K_CATEGORY_TO_YOLO_CLASS_ID: dict[str, int] = {
    "pedestrian": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 4,
    "truck": 5,
}

ConversionStats: TypeAlias = dict[str, int]

_STAT_KEYS = (
    "frames_seen",
    "images_written",
    "label_files_written",
    "target_boxes_written",
    "empty_label_files",
    "excluded_category_boxes",
    "clipped_boxes",
    "invalid_boxes",
)


@dataclass(frozen=True)
class _BoxConversion:
    coordinates: tuple[float, float, float, float] | None
    clipped: bool


@dataclass(frozen=True)
class _PreparedFrame:
    source_path: Path
    image_name: str
    label_name: str
    label_text: str


def load_bdd100k_frames(
    annotations_path: str | Path,
) -> list[Mapping[str, object]]:
    """Read and validate the top level of a BDD100K frame-list JSON file."""

    path = Path(annotations_path)
    if not path.is_file():
        raise FileNotFoundError(f"BDD100K annotations file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read BDD100K frame-list JSON {path}: {exc}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"BDD100K frame-list JSON must contain a list: {path}"
        )

    frames: list[Mapping[str, object]] = []
    for index, frame in enumerate(payload):
        if not isinstance(frame, Mapping):
            raise ValueError(
                f"BDD100K frame at index {index} must be an object"
            )
        frames.append(cast(Mapping[str, object], frame))
    return frames


def box2d_to_yolo(
    box2d: Mapping[str, object],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Convert a BDD100K ``box2d`` xyxy mapping to normalized YOLO xywh.

    Coordinates are clipped to the image. Invalid, non-finite, reversed, or
    zero-area boxes return ``None``.
    """

    return _convert_box2d(box2d, image_width, image_height).coordinates


def convert_bdd100k_to_yolo(
    annotations_path: str | Path,
    images_dir: str | Path,
    output_root: str | Path,
    split: str,
    *,
    limit: int | None = None,
) -> ConversionStats:
    """Convert one official BDD100K split and return conversion statistics."""

    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'")
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")

    root = Path(output_root)
    output_images_dir = root / "images" / split
    output_labels_dir = root / "labels" / split
    _ensure_output_directories_are_empty(
        output_images_dir,
        output_labels_dir,
        split,
    )

    frames = load_bdd100k_frames(annotations_path)
    if limit is not None:
        frames = frames[:limit]

    stats = _new_stats()
    stats["frames_seen"] = len(frames)
    prepared_frames = _prepare_frames(frames, Path(images_dir), stats)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for frame in prepared_frames:
        image_destination = output_images_dir / frame.image_name
        label_destination = output_labels_dir / frame.label_name
        if image_destination.exists() or label_destination.exists():
            raise FileExistsError(
                "Refusing to overwrite split output file: "
                f"{image_destination if image_destination.exists() else label_destination}"
            )

        shutil.copy2(frame.source_path, image_destination)
        stats["images_written"] += 1
        label_destination.write_text(frame.label_text, encoding="utf-8")
        stats["label_files_written"] += 1

    return stats


def _new_stats() -> ConversionStats:
    return {key: 0 for key in _STAT_KEYS}


def _prepare_frames(
    frames: list[Mapping[str, object]],
    images_dir: Path,
    stats: ConversionStats,
) -> list[_PreparedFrame]:
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"BDD100K images directory not found: {images_dir}"
        )

    images_root = images_dir.resolve()
    prepared: list[_PreparedFrame] = []
    image_names: set[str] = set()
    label_names: set[str] = set()

    for index, frame in enumerate(frames):
        frame_name = frame.get("name")
        if not isinstance(frame_name, str) or not frame_name.strip():
            raise ValueError(
                f"BDD100K frame at index {index} has no valid name"
            )

        source_path = (images_root / frame_name).resolve()
        try:
            source_path.relative_to(images_root)
        except ValueError as exc:
            raise ValueError(
                f"BDD100K frame name escapes images directory: {frame_name}"
            ) from exc

        image_name = Path(frame_name).name
        label_name = f"{Path(image_name).stem}.txt"
        image_key = image_name.casefold()
        label_key = label_name.casefold()
        if image_key in image_names or label_key in label_names:
            raise ValueError(
                f"BDD100K frame output basename is not unique: {frame_name}"
            )
        image_names.add(image_key)
        label_names.add(label_key)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"BDD100K image for frame '{frame_name}' not found: "
                f"{source_path}"
            )

        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2:
            raise OSError(
                f"BDD100K image for frame '{frame_name}' is unreadable: "
                f"{source_path}"
            )
        image_height, image_width = image.shape[:2]
        if image_width <= 0 or image_height <= 0:
            raise OSError(
                f"BDD100K image for frame '{frame_name}' has invalid "
                f"dimensions: {source_path}"
            )

        label_text = _frame_label_text(
            frame,
            image_width,
            image_height,
            index,
            stats,
        )
        if not label_text:
            stats["empty_label_files"] += 1

        prepared.append(
            _PreparedFrame(
                source_path=source_path,
                image_name=image_name,
                label_name=label_name,
                label_text=label_text,
            )
        )

    return prepared


def _frame_label_text(
    frame: Mapping[str, object],
    image_width: int,
    image_height: int,
    frame_index: int,
    stats: ConversionStats,
) -> str:
    labels = frame.get("labels", [])
    if labels is None:
        labels = []
    if not isinstance(labels, list):
        raise ValueError(
            f"BDD100K frame at index {frame_index} has non-list labels"
        )

    lines: list[str] = []
    for label in labels:
        if not isinstance(label, Mapping):
            stats["excluded_category_boxes"] += 1
            continue

        category = label.get("category")
        class_id = (
            BDD100K_CATEGORY_TO_YOLO_CLASS_ID.get(category)
            if isinstance(category, str)
            else None
        )
        if class_id is None:
            stats["excluded_category_boxes"] += 1
            continue

        box2d = label.get("box2d")
        if not isinstance(box2d, Mapping):
            stats["invalid_boxes"] += 1
            continue

        converted = _convert_box2d(
            cast(Mapping[str, object], box2d),
            image_width,
            image_height,
        )
        if converted.clipped:
            stats["clipped_boxes"] += 1
        if converted.coordinates is None:
            stats["invalid_boxes"] += 1
            continue

        center_x, center_y, width, height = converted.coordinates
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )
        stats["target_boxes_written"] += 1

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _convert_box2d(
    box2d: Mapping[str, object],
    image_width: int,
    image_height: int,
) -> _BoxConversion:
    if (
        isinstance(image_width, bool)
        or isinstance(image_height, bool)
        or not isinstance(image_width, int)
        or not isinstance(image_height, int)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("image width and height must be positive integers")

    values: list[float] = []
    for key in ("x1", "y1", "x2", "y2"):
        raw_value = box2d.get(key)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
        ):
            return _BoxConversion(None, False)
        try:
            value = float(raw_value)
        except (OverflowError, ValueError):
            return _BoxConversion(None, False)
        if not math.isfinite(value):
            return _BoxConversion(None, False)
        values.append(value)

    x1, y1, x2, y2 = values
    clipped_x1 = min(max(x1, 0.0), float(image_width))
    clipped_y1 = min(max(y1, 0.0), float(image_height))
    clipped_x2 = min(max(x2, 0.0), float(image_width))
    clipped_y2 = min(max(y2, 0.0), float(image_height))
    clipped = (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (
        x1,
        y1,
        x2,
        y2,
    )

    width = clipped_x2 - clipped_x1
    height = clipped_y2 - clipped_y1
    if width <= 0.0 or height <= 0.0:
        return _BoxConversion(None, clipped)

    return _BoxConversion(
        (
            ((clipped_x1 + clipped_x2) / 2.0) / image_width,
            ((clipped_y1 + clipped_y2) / 2.0) / image_height,
            width / image_width,
            height / image_height,
        ),
        clipped,
    )


def _ensure_output_directories_are_empty(
    images_dir: Path,
    labels_dir: Path,
    split: str,
) -> None:
    for directory in (images_dir, labels_dir):
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(
                f"Refusing to overwrite non-directory split output: {directory}"
            )
        if directory.is_dir() and any(
            path.is_file() for path in directory.rglob("*")
        ):
            raise FileExistsError(
                f"Refusing to overwrite existing '{split}' split files in "
                f"{directory}"
            )
