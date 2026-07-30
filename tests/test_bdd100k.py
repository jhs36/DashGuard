from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dashguard.data import (
    BDD100K_CATEGORY_TO_YOLO_CLASS_ID,
    YOLO_CLASS_NAMES,
    box2d_to_yolo,
    convert_bdd100k_to_yolo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_SCRIPT = PROJECT_ROOT / "scripts" / "convert_bdd100k_to_yolo.py"


def _write_image(path: Path, width: int = 1280, height: int = 720) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_annotations(path: Path, frames: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(frames), encoding="utf-8")


def _label(
    category: str,
    box2d: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "category": category,
        "box2d": box2d
        or {"x1": 128.0, "y1": 72.0, "x2": 384.0, "y2": 216.0},
    }


def _run_cli(
    annotations: Path,
    images_dir: Path,
    output_root: Path,
    split: str,
    *,
    limit: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(CONVERTER_SCRIPT),
        "--annotations",
        str(annotations),
        "--images-dir",
        str(images_dir),
        "--output-root",
        str(output_root),
        "--split",
        split,
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_class_mapping_and_id_order_are_exact() -> None:
    assert YOLO_CLASS_NAMES == (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    )
    assert BDD100K_CATEGORY_TO_YOLO_CLASS_ID == {
        "pedestrian": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "bus": 4,
        "truck": 5,
    }


def test_known_1280_by_720_box_converts_to_normalized_xywh() -> None:
    converted = box2d_to_yolo(
        {"x1": 128.0, "y1": 72.0, "x2": 384.0, "y2": 216.0},
        1280,
        720,
    )

    assert converted == pytest.approx((0.2, 0.2, 0.2, 0.2))


def test_pedestrian_is_written_as_person_zero(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    output_root = tmp_path / "output"
    _write_image(images_dir / "frame.jpg")
    _write_annotations(
        annotations,
        [{"name": "frame.jpg", "labels": [_label("pedestrian")]}],
    )

    stats = convert_bdd100k_to_yolo(
        annotations,
        images_dir,
        output_root,
        "train",
    )

    assert (output_root / "labels/train/frame.txt").read_text() == (
        "0 0.200000 0.200000 0.200000 0.200000\n"
    )
    assert stats["target_boxes_written"] == 1


def test_rider_excluded_and_other_excluded_categories_never_written(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    output_root = tmp_path / "output"
    _write_image(images_dir / "frame.jpg")
    excluded = [
        "rider",
        "other person",
        "other vehicle",
        "traffic light",
        "traffic sign",
        "trailer",
        "train",
        "unknown future category",
    ]
    _write_annotations(
        annotations,
        [
            {
                "name": "frame.jpg",
                "labels": [_label(category) for category in excluded]
                + [_label("car")],
            }
        ],
    )

    stats = convert_bdd100k_to_yolo(
        annotations,
        images_dir,
        output_root,
        "train",
    )

    assert (output_root / "labels/train/frame.txt").read_text().startswith(
        "2 "
    )
    assert stats["target_boxes_written"] == 1
    assert stats["excluded_category_boxes"] == len(excluded)


def test_out_of_bounds_box_is_clipped(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    output_root = tmp_path / "output"
    _write_image(images_dir / "frame.jpg")
    _write_annotations(
        annotations,
        [
            {
                "name": "frame.jpg",
                "labels": [
                    _label(
                        "bus",
                        {
                            "x1": -10.0,
                            "y1": -20.0,
                            "x2": 1290.0,
                            "y2": 740.0,
                        },
                    )
                ],
            }
        ],
    )

    stats = convert_bdd100k_to_yolo(
        annotations,
        images_dir,
        output_root,
        "val",
    )

    assert (output_root / "labels/val/frame.txt").read_text() == (
        "4 0.500000 0.500000 1.000000 1.000000\n"
    )
    assert stats["clipped_boxes"] == 1


@pytest.mark.parametrize(
    "box2d",
    [
        {"x1": 20.0, "y1": 10.0, "x2": 5.0, "y2": 30.0},
        {"x1": 1280.0, "y1": 0.0, "x2": 1400.0, "y2": 20.0},
        {"x1": float("nan"), "y1": 0.0, "x2": 10.0, "y2": 20.0},
        {"x1": 0.0, "y1": 0.0, "x2": float("inf"), "y2": 20.0},
    ],
)
def test_invalid_reversed_zero_area_and_nonfinite_boxes_are_excluded(
    tmp_path: Path,
    box2d: dict[str, float],
) -> None:
    images_dir = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    output_root = tmp_path / "output"
    _write_image(images_dir / "frame.jpg")
    _write_annotations(
        annotations,
        [
            {
                "name": "frame.jpg",
                "labels": [_label("truck", box2d)],
            }
        ],
    )

    stats = convert_bdd100k_to_yolo(
        annotations,
        images_dir,
        output_root,
        "train",
    )

    assert (output_root / "labels/train/frame.txt").read_text() == ""
    assert stats["invalid_boxes"] == 1
    assert stats["target_boxes_written"] == 0


def test_frame_without_targets_creates_empty_label_file(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    output_root = tmp_path / "output"
    _write_image(images_dir / "empty.jpg")
    _write_annotations(
        annotations,
        [{"name": "empty.jpg", "labels": [_label("rider")]}],
    )

    stats = convert_bdd100k_to_yolo(
        annotations,
        images_dir,
        output_root,
        "train",
    )

    assert (output_root / "images/train/empty.jpg").is_file()
    assert (output_root / "labels/train/empty.txt").read_bytes() == b""
    assert stats["empty_label_files"] == 1


def test_missing_image_raises_clear_exception(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [{"name": "missing.jpg", "labels": []}],
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"image for frame 'missing\.jpg' not found",
    ):
        convert_bdd100k_to_yolo(
            annotations,
            images_dir,
            tmp_path / "output",
            "train",
        )


def test_cli_limit_only_writes_requested_number_of_frames(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    output_root = tmp_path / "output"
    _write_image(images_dir / "first.jpg")
    _write_image(images_dir / "second.jpg")
    _write_annotations(
        annotations,
        [
            {"name": "first.jpg", "labels": []},
            {"name": "second.jpg", "labels": []},
        ],
    )

    result = _run_cli(
        annotations,
        images_dir,
        output_root,
        "train",
        limit=1,
    )

    assert result.returncode == 0, result.stderr
    assert "frames_seen=1" in result.stdout
    assert [path.name for path in (output_root / "images/train").iterdir()] == [
        "first.jpg"
    ]


def test_cli_allows_train_then_val_rejects_repeat_and_writes_yaml(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    train_annotations = tmp_path / "train.json"
    val_annotations = tmp_path / "val.json"
    output_root = tmp_path / "output"
    _write_image(images_dir / "train.jpg")
    _write_image(images_dir / "val.jpg")
    _write_annotations(
        train_annotations,
        [{"name": "train.jpg", "labels": [_label("car")]}],
    )
    _write_annotations(
        val_annotations,
        [{"name": "val.jpg", "labels": [_label("bus")]}],
    )

    train_result = _run_cli(
        train_annotations,
        images_dir,
        output_root,
        "train",
    )
    val_result = _run_cli(
        val_annotations,
        images_dir,
        output_root,
        "val",
    )
    repeat_result = _run_cli(
        train_annotations,
        images_dir,
        output_root,
        "train",
    )

    assert train_result.returncode == 0, train_result.stderr
    assert val_result.returncode == 0, val_result.stderr
    assert repeat_result.returncode != 0
    assert "Refusing to overwrite" in repeat_result.stderr
    assert (output_root / "images/train/train.jpg").is_file()
    assert (output_root / "images/val/val.jpg").is_file()
    assert (output_root / "data.yaml").read_text() == (
        f"path: {json.dumps(output_root.resolve().as_posix())}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
        "  1: bicycle\n"
        "  2: car\n"
        "  3: motorcycle\n"
        "  4: bus\n"
        "  5: truck\n"
    )
