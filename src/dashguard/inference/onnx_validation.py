"""Compare PyTorch and ONNX Runtime raw outputs from one shared input."""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashguard.detection import Detection, TARGET_CLASS_NAMES


SCHEMA_VERSION = "dashguard.onnx_backend_consistency.v2"
CPU_PROVIDER = "CPUExecutionProvider"
DIFFERENCE_THRESHOLDS: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_MAX_DETECTIONS = 300
DEFAULT_AGNOSTIC_NMS = False
RAW_ALLCLOSE_RTOL = 1e-4
RAW_ALLCLOSE_ATOL = 1e-4
MIN_MATCHED_BBOX_IOU = 0.995
MAX_BBOX_COORDINATE_ABS_DIFF = 0.25
MAX_CONFIDENCE_ABS_DIFF = 0.001


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the deterministic SHA-256 hex digest of one file."""

    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    stat = file_path.stat()
    if not file_path.is_file():
        raise FileNotFoundError(f"Expected a regular file: {file_path}")
    return {
        "path": str(file_path),
        "resolved_path": str(file_path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": file_sha256(file_path),
    }


def load_bgr_image(path: str | Path) -> np.ndarray:
    """Load one three-channel BGR image with OpenCV."""

    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV failed to read image: {image_path}")
    return image


def preprocess_bgr_image(image: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """Create one fixed-shape FP32 NCHW input using Ultralytics ordering."""

    from ultralytics.data.augment import LetterBox

    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a three-channel HWC BGR image; "
            f"received shape {list(image.shape)}"
        )
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 BGR input; received {image.dtype}")

    letterboxed = LetterBox(
        new_shape=(imgsz, imgsz),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        padding_value=114,
        interpolation=cv2.INTER_LINEAR,
    )(image=image)
    if letterboxed.shape != (imgsz, imgsz, 3):
        raise RuntimeError(
            "Fixed letterbox returned an unexpected shape: "
            f"{list(letterboxed.shape)}"
        )

    rgb_chw = letterboxed[..., ::-1].transpose((2, 0, 1))
    batched = np.ascontiguousarray(rgb_chw[np.newaxis, ...])
    normalized = batched.astype(np.float32, copy=False)
    normalized /= np.float32(255.0)

    expected_shape = (1, 3, imgsz, imgsz)
    if normalized.shape != expected_shape:
        raise RuntimeError(
            f"Preprocessing produced {list(normalized.shape)}, "
            f"expected {list(expected_shape)}"
        )
    if normalized.dtype != np.float32 or not normalized.flags.c_contiguous:
        raise RuntimeError("Preprocessing did not produce contiguous float32 data")
    return normalized


def letterbox_geometry(
    original_shape: Sequence[int],
    imgsz: int,
) -> dict[str, Any]:
    """Return the exact fixed LetterBox ratio and applied integer padding."""

    original_height, original_width = (int(original_shape[0]), int(original_shape[1]))
    ratio = min(imgsz / original_height, imgsz / original_width)
    resized_width = round(original_width * ratio)
    resized_height = round(original_height * ratio)
    horizontal_padding = (imgsz - resized_width) / 2
    vertical_padding = (imgsz - resized_height) / 2
    left = round(horizontal_padding - 0.1)
    right = round(horizontal_padding + 0.1)
    top = round(vertical_padding - 0.1)
    bottom = round(vertical_padding + 0.1)
    return {
        "ratio": [ratio, ratio],
        "resized_shape": [resized_height, resized_width],
        "padding_ltrb": [left, top, right, bottom],
    }


def pytorch_output_to_numpy(raw_output: Any) -> np.ndarray:
    """Extract the export-corresponding first tensor and return FP32 NumPy."""

    import torch

    selected = raw_output
    if isinstance(raw_output, (tuple, list)):
        if not raw_output:
            raise ValueError("PyTorch forward returned an empty tuple/list")
        selected = raw_output[0]
    if not isinstance(selected, torch.Tensor):
        raise TypeError(
            "PyTorch forward output (or its first tuple/list item) must be a tensor"
        )
    return np.ascontiguousarray(
        selected.detach().cpu().numpy(),
        dtype=np.float32,
    )


class PytorchRunner:
    """Run a direct CPU forward on the internal model from ``YOLO(pt_path)``."""

    def __init__(self, model_path: str | Path) -> None:
        import torch
        from ultralytics import YOLO

        wrapper = YOLO(str(model_path))
        model = wrapper.model
        if model is None:
            raise RuntimeError("YOLO did not expose an internal PyTorch model")
        self._torch = torch
        self._model = model.to("cpu").float().eval()
        self.class_names = model.names
        self.end2end = bool(getattr(model, "end2end", False))

    def run(self, input_array: np.ndarray) -> np.ndarray:
        """Run direct inference from the caller-owned shared NumPy array."""

        tensor = self._torch.from_numpy(input_array)
        if tensor.device.type != "cpu" or tensor.dtype != self._torch.float32:
            raise RuntimeError("PyTorch input must be a CPU float32 tensor")
        with self._torch.inference_mode():
            raw_output = self._model(tensor)
        return pytorch_output_to_numpy(raw_output)


def _metadata_shape(shape: Sequence[Any]) -> list[int | str | None]:
    converted: list[int | str | None] = []
    for dimension in shape:
        if dimension is None or isinstance(dimension, str):
            converted.append(dimension)
        else:
            converted.append(int(dimension))
    return converted


def _node_metadata(node: Any) -> dict[str, Any]:
    return {
        "name": str(node.name),
        "shape": _metadata_shape(node.shape),
        "dtype": str(node.type),
    }


def _validate_metadata_shape(
    actual_shape: Sequence[int],
    metadata: Mapping[str, Any],
    label: str,
) -> None:
    expected_shape = metadata["shape"]
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{label} rank does not match metadata: actual={list(actual_shape)}, "
            f"metadata={list(expected_shape)}"
        )
    for index, (actual, expected) in enumerate(
        zip(actual_shape, expected_shape, strict=True)
    ):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(
                f"{label} shape does not match metadata at dimension {index}: "
                f"actual={list(actual_shape)}, metadata={list(expected_shape)}"
            )


def _validate_float_tensor_metadata(
    array: np.ndarray,
    metadata: Mapping[str, Any],
    label: str,
) -> None:
    _validate_metadata_shape(array.shape, metadata, label)
    if metadata["dtype"] != "tensor(float)":
        raise TypeError(
            f"{label} metadata dtype must be tensor(float); "
            f"received {metadata['dtype']}"
        )
    if array.dtype != np.float32:
        raise TypeError(f"{label} array must be float32; received {array.dtype}")


class OnnxRuntimeRunner:
    """Run one-output ONNX inference with the CPU execution provider only."""

    def __init__(self, model_path: str | Path, session: Any = None) -> None:
        if session is None:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(model_path),
                providers=[CPU_PROVIDER],
            )

        providers = list(session.get_providers())
        if providers != [CPU_PROVIDER]:
            raise RuntimeError(
                "ONNX Runtime must use only CPUExecutionProvider; "
                f"actual providers: {providers}"
            )
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if len(inputs) != 1:
            raise ValueError(
                f"ONNX model must expose exactly one input; received {len(inputs)}"
            )
        if len(outputs) != 1:
            raise ValueError(
                f"ONNX model must expose exactly one output; received {len(outputs)}"
            )

        self._session = session
        self.providers = providers
        self.input_metadata = _node_metadata(inputs[0])
        self.output_metadata = _node_metadata(outputs[0])

    def validate_input(self, input_array: np.ndarray) -> None:
        """Require the actual input to match the session input metadata."""

        _validate_float_tensor_metadata(
            input_array,
            self.input_metadata,
            "ONNX input",
        )

    def run(self, input_array: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime using its metadata-discovered I/O names."""

        self.validate_input(input_array)
        outputs = self._session.run(
            [self.output_metadata["name"]],
            {self.input_metadata["name"]: input_array},
        )
        if len(outputs) != 1:
            raise RuntimeError(
                "ONNX Runtime returned an unexpected output count: "
                f"{len(outputs)}"
            )
        output = np.asarray(outputs[0])
        _validate_float_tensor_metadata(
            output,
            self.output_metadata,
            "ONNX output",
        )
        return np.ascontiguousarray(output, dtype=np.float32)


def _validated_numeric_output(output: Any, label: str) -> np.ndarray:
    array = np.asarray(output)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{label} output must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{label} output contains NaN or infinity")
    return normalized


def calculate_difference_statistics(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
) -> dict[str, Any]:
    """Calculate raw difference measurements without an acceptance threshold."""

    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(
            "Output shapes differ: "
            f"PyTorch={list(pytorch_output.shape)}, "
            f"ONNX={list(onnx_output.shape)}"
        )
    if pytorch_output.size == 0:
        raise ValueError("Raw outputs must contain at least one element")
    if not np.isfinite(pytorch_output).all():
        raise ValueError("PyTorch output contains NaN or infinity")
    if not np.isfinite(onnx_output).all():
        raise ValueError("ONNX output contains NaN or infinity")

    delta = np.subtract(
        pytorch_output,
        onnx_output,
        dtype=np.float64,
    )
    absolute = np.abs(delta)
    percentiles = np.percentile(absolute, [50, 95, 99])
    max_flat_index = int(np.argmax(absolute))
    max_index = tuple(
        int(value)
        for value in np.unravel_index(max_flat_index, absolute.shape)
    )
    pytorch_value = float(pytorch_output[max_index])
    onnx_value = float(onnx_output[max_index])

    statistics: dict[str, Any] = {
        "max_abs_diff": float(absolute.flat[max_flat_index]),
        "mean_abs_diff": float(np.mean(absolute, dtype=np.float64)),
        "rmse": float(
            math.sqrt(np.mean(np.square(delta), dtype=np.float64))
        ),
        "p50_abs_diff": float(percentiles[0]),
        "p95_abs_diff": float(percentiles[1]),
        "p99_abs_diff": float(percentiles[2]),
    }
    for threshold in DIFFERENCE_THRESHOLDS:
        threshold_label = format(threshold, ".0e").replace("e-0", "e-")
        statistics[f"fraction_abs_diff_gt_{threshold_label}"] = float(
            np.mean(absolute > threshold, dtype=np.float64)
        )
    statistics["max_difference"] = {
        "flat_index": max_flat_index,
        "index": list(max_index),
        "tensor_index": list(max_index),
        "pytorch_value": pytorch_value,
        "onnx_value": onnx_value,
        "signed_pytorch_minus_onnx": pytorch_value - onnx_value,
    }
    return statistics


def compare_backend_outputs(
    input_array: np.ndarray,
    pytorch_forward: Callable[[np.ndarray], Any],
    onnx_forward: Callable[[np.ndarray], Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pass the exact same array object to both backends and compare outputs."""

    pytorch_output = _validated_numeric_output(
        pytorch_forward(input_array),
        "PyTorch",
    )
    onnx_output = _validated_numeric_output(
        onnx_forward(input_array),
        "ONNX",
    )

    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(
            "Output shapes differ: "
            f"PyTorch={list(pytorch_output.shape)}, "
            f"ONNX={list(onnx_output.shape)}"
        )
    if pytorch_output.ndim == 0 or pytorch_output.shape[0] != 1:
        raise ValueError(
            "PyTorch output batch dimension must be 1; "
            f"received shape {list(pytorch_output.shape)}"
        )
    if onnx_output.ndim == 0 or onnx_output.shape[0] != 1:
        raise ValueError(
            "ONNX output batch dimension must be 1; "
            f"received shape {list(onnx_output.shape)}"
        )
    if pytorch_output.size != onnx_output.size:
        raise ValueError(
            "Raw output element counts differ: "
            f"PyTorch={pytorch_output.size}, ONNX={onnx_output.size}"
        )

    return (
        pytorch_output,
        onnx_output,
        calculate_difference_statistics(pytorch_output, onnx_output),
    )


def raw_allclose_diagnostic(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
) -> dict[str, Any]:
    """Record the required raw allclose diagnostic without deciding acceptance."""

    return {
        "passed": bool(
            np.allclose(
                pytorch_output,
                onnx_output,
                rtol=RAW_ALLCLOSE_RTOL,
                atol=RAW_ALLCLOSE_ATOL,
                equal_nan=False,
            )
        ),
        "rtol": RAW_ALLCLOSE_RTOL,
        "atol": RAW_ALLCLOSE_ATOL,
        "equal_nan": False,
        "role": "diagnostic_only",
    }


def _class_name_mapping(
    class_names: Mapping[int, str] | Sequence[str],
) -> dict[int, str]:
    if isinstance(class_names, Mapping):
        mapping = {int(class_id): str(name) for class_id, name in class_names.items()}
    elif isinstance(class_names, Sequence) and not isinstance(
        class_names, (str, bytes)
    ):
        mapping = {index: str(name) for index, name in enumerate(class_names)}
    else:
        raise TypeError("class_names must be a mapping or sequence")
    discovered_targets = TARGET_CLASS_NAMES.intersection(mapping.values())
    if discovered_targets != TARGET_CLASS_NAMES:
        missing = sorted(TARGET_CLASS_NAMES - discovered_targets)
        raise ValueError(f"Model class names are missing target classes: {missing}")
    return mapping


def _target_class_ids(class_names: Mapping[int, str]) -> list[int]:
    return sorted(
        class_id
        for class_id, class_name in class_names.items()
        if class_name in TARGET_CLASS_NAMES
    )


def postprocess_raw_detections(
    raw_output: np.ndarray,
    *,
    input_shape: Sequence[int],
    original_shape: Sequence[int],
    geometry: Mapping[str, Any],
    class_names: Mapping[int, str] | Sequence[str],
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_det: int = DEFAULT_MAX_DETECTIONS,
    agnostic_nms: bool = DEFAULT_AGNOSTIC_NMS,
    end2end: bool = False,
) -> list[Detection]:
    """Apply one shared Ultralytics detect postprocess to either raw output."""

    import torch
    from ultralytics.utils import ops
    from ultralytics.utils.nms import non_max_suppression

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not isinstance(max_det, int) or isinstance(max_det, bool) or max_det <= 0:
        raise ValueError("max_det must be a positive integer")
    if raw_output.ndim != 3 or raw_output.shape[0] != 1:
        raise ValueError(
            "Detect postprocessing requires one BCN raw output; "
            f"received shape {list(raw_output.shape)}"
        )

    names = _class_name_mapping(class_names)
    # Ultralytics NMS converts xywh to xyxy in-place through a transposed view.
    # Keep the measurement arrays pristine by giving NMS a working copy.
    working_output = np.array(
        raw_output,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    prediction = torch.from_numpy(working_output)
    nms_outputs = non_max_suppression(
        prediction,
        conf_thres=confidence,
        iou_thres=iou_threshold,
        classes=_target_class_ids(names),
        agnostic=agnostic_nms,
        multi_label=False,
        max_det=max_det,
        nc=0,
        rotated=False,
        end2end=end2end,
        return_idxs=False,
    )
    if len(nms_outputs) != 1:
        raise RuntimeError(
            "Single-image NMS must return exactly one result; "
            f"received {len(nms_outputs)}"
        )

    detections_tensor = nms_outputs[0]
    ratio = geometry["ratio"]
    left, top = geometry["padding_ltrb"][:2]
    if detections_tensor.numel():
        ops.scale_boxes(
            (int(input_shape[-2]), int(input_shape[-1])),
            detections_tensor[:, :4],
            (int(original_shape[0]), int(original_shape[1])),
            ratio_pad=((float(ratio[0]), float(ratio[1])), (int(left), int(top))),
        )

    rows = detections_tensor.detach().cpu().numpy()
    detections: list[Detection] = []
    for row in rows:
        class_id = int(row[5])
        if class_id not in names:
            raise ValueError(f"No class name exists for class ID {class_id}")
        detections.append(
            Detection(
                xyxy=tuple(float(value) for value in row[:4]),
                confidence=float(row[4]),
                class_id=class_id,
                class_name=names[class_id],
            )
        )
    return detections


def bbox_iou(left: Detection, right: Detection) -> float:
    """Calculate continuous-coordinate IoU for two detection boxes."""

    left_x1, left_y1, left_x2, left_y2 = left.xyxy
    right_x1, right_y1, right_x2, right_y2 = right.xyxy
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_coordinate_max_abs_diff(left: Detection, right: Detection) -> float:
    """Return the largest corresponding xyxy coordinate difference."""

    return max(
        abs(left_value - right_value)
        for left_value, right_value in zip(left.xyxy, right.xyxy, strict=True)
    )


def confidence_abs_diff(left: Detection, right: Detection) -> float:
    """Return the absolute confidence difference."""

    return abs(left.confidence - right.confidence)


def _detection_json(detection: Detection) -> dict[str, Any]:
    return {
        "xyxy": list(detection.xyxy),
        "confidence": detection.confidence,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
    }


def match_detections(
    pytorch_detections: Sequence[Detection],
    onnx_detections: Sequence[Detection],
) -> dict[str, Any]:
    """Greedily match same-class detections by IoU with deterministic ties."""

    candidates: list[tuple[float, int, int]] = []
    for pytorch_index, pytorch_detection in enumerate(pytorch_detections):
        for onnx_index, onnx_detection in enumerate(onnx_detections):
            if pytorch_detection.class_id == onnx_detection.class_id:
                candidates.append(
                    (
                        bbox_iou(pytorch_detection, onnx_detection),
                        pytorch_index,
                        onnx_index,
                    )
                )
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))

    used_pytorch: set[int] = set()
    used_onnx: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, pytorch_index, onnx_index in candidates:
        if pytorch_index in used_pytorch or onnx_index in used_onnx:
            continue
        used_pytorch.add(pytorch_index)
        used_onnx.add(onnx_index)
        pytorch_detection = pytorch_detections[pytorch_index]
        onnx_detection = onnx_detections[onnx_index]
        coordinate_difference = bbox_coordinate_max_abs_diff(
            pytorch_detection,
            onnx_detection,
        )
        confidence_difference = confidence_abs_diff(
            pytorch_detection,
            onnx_detection,
        )
        criteria = {
            "class_id_equal": (
                pytorch_detection.class_id == onnx_detection.class_id
            ),
            "class_name_equal": (
                pytorch_detection.class_name == onnx_detection.class_name
            ),
            "bbox_iou_at_least_0_995": iou >= MIN_MATCHED_BBOX_IOU,
            "bbox_coordinate_max_abs_diff_at_most_0_25": (
                coordinate_difference <= MAX_BBOX_COORDINATE_ABS_DIFF
            ),
            "confidence_abs_diff_at_most_0_001": (
                confidence_difference <= MAX_CONFIDENCE_ABS_DIFF
            ),
        }
        matches.append(
            {
                "pytorch_index": pytorch_index,
                "onnx_index": onnx_index,
                "class_id": pytorch_detection.class_id,
                "class_name": pytorch_detection.class_name,
                "pytorch_class_name": pytorch_detection.class_name,
                "onnx_class_name": onnx_detection.class_name,
                "pytorch_bbox": list(pytorch_detection.xyxy),
                "onnx_bbox": list(onnx_detection.xyxy),
                "bbox_coordinate_max_abs_diff": coordinate_difference,
                "bbox_iou": iou,
                "pytorch_confidence": pytorch_detection.confidence,
                "onnx_confidence": onnx_detection.confidence,
                "confidence_abs_diff": confidence_difference,
                "criteria": criteria,
                "passed": all(criteria.values()),
            }
        )

    unmatched_pytorch = sorted(set(range(len(pytorch_detections))) - used_pytorch)
    unmatched_onnx = sorted(set(range(len(onnx_detections))) - used_onnx)
    count_equal = len(pytorch_detections) == len(onnx_detections)
    accepted = (
        count_equal
        and not unmatched_pytorch
        and not unmatched_onnx
        and all(match["passed"] for match in matches)
    )
    matched_ious = [match["bbox_iou"] for match in matches]
    bbox_differences = [
        match["bbox_coordinate_max_abs_diff"] for match in matches
    ]
    confidence_differences = [match["confidence_abs_diff"] for match in matches]
    return {
        "pytorch_detection_count": len(pytorch_detections),
        "onnx_detection_count": len(onnx_detections),
        "detection_count_equal": count_equal,
        "pytorch_detections": [
            _detection_json(detection) for detection in pytorch_detections
        ],
        "onnx_detections": [
            _detection_json(detection) for detection in onnx_detections
        ],
        "matched_count": len(matches),
        "unmatched_pytorch_indices": unmatched_pytorch,
        "unmatched_onnx_indices": unmatched_onnx,
        "matches": matches,
        "minimum_matched_iou": min(matched_ious) if matched_ious else None,
        "maximum_bbox_coordinate_abs_diff": (
            max(bbox_differences) if bbox_differences else 0.0
        ),
        "maximum_confidence_abs_diff": (
            max(confidence_differences) if confidence_differences else 0.0
        ),
        "detection_acceptance": "passed" if accepted else "failed",
    }


def _acceptance_criteria() -> dict[str, Any]:
    return {
        "scope": (
            "DashGuard project consistency criteria for this model, preprocessing, "
            "and FP32 CPU backends; not a universal ONNX standard"
        ),
        "raw_diagnostic": {
            "operation": "numpy.allclose",
            "rtol": RAW_ALLCLOSE_RTOL,
            "atol": RAW_ALLCLOSE_ATOL,
            "equal_nan": False,
            "determines_detection_acceptance": False,
        },
        "detection": {
            "detection_count_equal": True,
            "unmatched_pytorch_count": 0,
            "unmatched_onnx_count": 0,
            "class_id_exact": True,
            "class_name_exact": True,
            "minimum_bbox_iou": MIN_MATCHED_BBOX_IOU,
            "maximum_bbox_coordinate_abs_diff_pixels": (
                MAX_BBOX_COORDINATE_ABS_DIFF
            ),
            "maximum_confidence_abs_diff": MAX_CONFIDENCE_ABS_DIFF,
            "all_images_must_pass": True,
        },
    }


def _aggregate_detection_comparisons(
    image_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons = [result["detection_comparison"] for result in image_results]
    matches = [match for comparison in comparisons for match in comparison["matches"]]
    accepted = all(
        comparison["detection_acceptance"] == "passed"
        for comparison in comparisons
    )
    return {
        "total_pytorch_detections": sum(
            comparison["pytorch_detection_count"] for comparison in comparisons
        ),
        "total_onnx_detections": sum(
            comparison["onnx_detection_count"] for comparison in comparisons
        ),
        "total_matched": sum(
            comparison["matched_count"] for comparison in comparisons
        ),
        "total_unmatched_pytorch": sum(
            len(comparison["unmatched_pytorch_indices"])
            for comparison in comparisons
        ),
        "total_unmatched_onnx": sum(
            len(comparison["unmatched_onnx_indices"])
            for comparison in comparisons
        ),
        "minimum_matched_iou": (
            min(match["bbox_iou"] for match in matches) if matches else None
        ),
        "maximum_bbox_coordinate_abs_diff": (
            max(match["bbox_coordinate_max_abs_diff"] for match in matches)
            if matches
            else 0.0
        ),
        "maximum_confidence_abs_diff": (
            max(match["confidence_abs_diff"] for match in matches)
            if matches
            else 0.0
        ),
        "raw_allclose_passed_image_count": sum(
            bool(result["raw_allclose"]["passed"]) for result in image_results
        ),
        "detection_acceptance": "passed" if accepted else "failed",
    }


def _add_aggregate_max_difference_provenance(
    aggregate_difference: dict[str, Any],
    pytorch_outputs: Sequence[np.ndarray],
) -> None:
    maximum = aggregate_difference["max_difference"]
    concatenated_flat_index = int(maximum["flat_index"])
    offset = 0
    for image_index, output in enumerate(pytorch_outputs):
        next_offset = offset + output.size
        if concatenated_flat_index < next_offset:
            local_flat_index = concatenated_flat_index - offset
            tensor_index = [
                int(value)
                for value in np.unravel_index(local_flat_index, output.shape)
            ]
            maximum.update(
                {
                    "image_index": image_index,
                    "tensor_index": tensor_index,
                    "concatenated_flat_index": concatenated_flat_index,
                }
            )
            return
        offset = next_offset
    raise RuntimeError("Aggregate maximum index did not map to an input image")


def _runtime_versions() -> dict[str, str]:
    import onnxruntime
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def _preprocessing_metadata(imgsz: int) -> dict[str, Any]:
    return {
        "description": (
            "OpenCV BGR load; fixed centered Ultralytics LetterBox; "
            "BGR to RGB; HWC to CHW; batch; C-contiguous; float32; /255.0"
        ),
        "steps": [
            "opencv_imread_bgr",
            "letterbox_fixed_shape",
            "bgr_to_rgb",
            "hwc_to_chw",
            "add_batch_dimension",
            "c_contiguous",
            "float32",
            "divide_by_255",
        ],
        "letterbox": {
            "new_shape": [imgsz, imgsz],
            "auto": False,
            "scale_fill": False,
            "scaleup": True,
            "center": True,
            "padding_value": 114,
            "interpolation": "cv2.INTER_LINEAR",
        },
        "backend_input_contract": "one shared NumPy array object per image",
    }


def _postprocessing_metadata(
    class_names: Mapping[int, str],
    *,
    confidence: float,
    iou_threshold: float,
    max_det: int,
    end2end: bool,
) -> dict[str, Any]:
    target_ids = _target_class_ids(class_names)
    return {
        "implementation": {
            "nms": "ultralytics.utils.nms.non_max_suppression",
            "coordinate_scaling": "ultralytics.utils.ops.scale_boxes",
            "same_function_for_both_backends": True,
        },
        "confidence_threshold": confidence,
        "iou_threshold": iou_threshold,
        "iou_threshold_source": (
            "Ultralytics 8.4.107 default detect configuration and existing "
            "UltralyticsDetector behavior"
        ),
        "target_class_ids": target_ids,
        "target_class_names": [class_names[class_id] for class_id in target_ids],
        "maximum_detections": max_det,
        "agnostic_nms": DEFAULT_AGNOSTIC_NMS,
        "multi_label": False,
        "nc_argument": 0,
        "end2end": end2end,
        "raw_output_layout": "BCN [batch, 4 + class_count, candidates]",
        "coordinate_scaling_uses_recorded_letterbox_ratio_and_padding": True,
        "coordinates_rounded_before_comparison": False,
    }


def _node_metadata_copy(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(metadata["name"]),
        "shape": _metadata_shape(metadata["shape"]),
        "dtype": str(metadata["dtype"]),
    }


def _assert_output_available(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output JSON already exists (use --overwrite to replace it): "
            f"{output_path}"
        )


def write_json_atomic(
    payload: Mapping[str, Any],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write strict UTF-8 JSON atomically, preserving existing output by default."""

    destination = Path(output_path)
    _assert_output_available(destination, overwrite)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    "Output JSON appeared while validation was running: "
                    f"{destination}"
                ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def validate_onnx_outputs(
    image_paths: Sequence[str | Path],
    pytorch_model_path: str | Path,
    onnx_model_path: str | Path,
    imgsz: int,
    output_json: str | Path,
    overwrite: bool = False,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_det: int = DEFAULT_MAX_DETECTIONS,
    pytorch_runner: Any = None,
    onnx_runner: Any = None,
) -> dict[str, Any]:
    """Validate raw and detection consistency, then atomically publish a report."""

    if not image_paths:
        raise ValueError("At least one image path is required")
    destination = Path(output_json)
    _assert_output_available(destination, overwrite)

    pytorch_model = _file_metadata(pytorch_model_path)
    onnx_model = _file_metadata(onnx_model_path)
    if pytorch_runner is None:
        pytorch_runner = PytorchRunner(pytorch_model_path)
    if onnx_runner is None:
        onnx_runner = OnnxRuntimeRunner(onnx_model_path)

    class_names = _class_name_mapping(pytorch_runner.class_names)
    end2end = bool(getattr(pytorch_runner, "end2end", False))

    providers = list(onnx_runner.providers)
    if providers != [CPU_PROVIDER]:
        raise RuntimeError(
            "ONNX Runtime must use only CPUExecutionProvider; "
            f"actual providers: {providers}"
        )
    input_metadata = _node_metadata_copy(onnx_runner.input_metadata)
    output_metadata = _node_metadata_copy(onnx_runner.output_metadata)

    image_results: list[dict[str, Any]] = []
    pytorch_outputs: list[np.ndarray] = []
    onnx_outputs: list[np.ndarray] = []
    for image_path_value in image_paths:
        image_path = Path(image_path_value)
        image_metadata = _file_metadata(image_path)
        bgr_image = load_bgr_image(image_path)
        input_array = preprocess_bgr_image(bgr_image, imgsz)
        geometry = letterbox_geometry(bgr_image.shape, imgsz)
        _validate_float_tensor_metadata(
            input_array,
            input_metadata,
            "ONNX input",
        )

        pytorch_output, onnx_output, difference = compare_backend_outputs(
            input_array,
            pytorch_runner.run,
            onnx_runner.run,
        )
        _validate_float_tensor_metadata(
            onnx_output,
            output_metadata,
            "ONNX output",
        )
        raw_allclose = raw_allclose_diagnostic(pytorch_output, onnx_output)
        postprocess_arguments = {
            "input_shape": input_array.shape,
            "original_shape": bgr_image.shape,
            "geometry": geometry,
            "class_names": class_names,
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "max_det": max_det,
            "agnostic_nms": DEFAULT_AGNOSTIC_NMS,
            "end2end": end2end,
        }
        pytorch_detections = postprocess_raw_detections(
            pytorch_output,
            **postprocess_arguments,
        )
        onnx_detections = postprocess_raw_detections(
            onnx_output,
            **postprocess_arguments,
        )
        detection_comparison = match_detections(
            pytorch_detections,
            onnx_detections,
        )
        pytorch_outputs.append(pytorch_output)
        onnx_outputs.append(onnx_output)
        image_results.append(
            {
                **image_metadata,
                "original_shape": list(bgr_image.shape),
                "preprocessed": {
                    "shape": list(input_array.shape),
                    "dtype": str(input_array.dtype),
                    "min": float(np.min(input_array)),
                    "max": float(np.max(input_array)),
                    "c_contiguous": bool(input_array.flags.c_contiguous),
                    "letterbox_geometry": geometry,
                },
                "pytorch_output": {
                    "shape": list(pytorch_output.shape),
                    "dtype": str(pytorch_output.dtype),
                },
                "onnx_output": {
                    "shape": list(onnx_output.shape),
                    "dtype": str(onnx_output.dtype),
                },
                "difference": difference,
                "raw_allclose": raw_allclose,
                "detection_comparison": detection_comparison,
                "detection_acceptance": detection_comparison[
                    "detection_acceptance"
                ],
            }
        )

    aggregate_pytorch = np.concatenate(
        [output.reshape(-1) for output in pytorch_outputs]
    ).reshape(1, -1)
    aggregate_onnx = np.concatenate(
        [output.reshape(-1) for output in onnx_outputs]
    ).reshape(1, -1)
    aggregate_difference = calculate_difference_statistics(
        aggregate_pytorch,
        aggregate_onnx,
    )
    _add_aggregate_max_difference_provenance(
        aggregate_difference,
        pytorch_outputs,
    )
    aggregate_detection = _aggregate_detection_comparisons(image_results)
    numerical_acceptance = aggregate_detection["detection_acceptance"]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "versions": _runtime_versions(),
        "structural_validation": "passed",
        "numerical_acceptance": numerical_acceptance,
        "models": {
            "pytorch": pytorch_model,
            "onnx": onnx_model,
        },
        "onnx_runtime": {
            "requested_providers": [CPU_PROVIDER],
            "actual_providers": providers,
        },
        "onnx_io_metadata": {
            "inputs": [input_metadata],
            "outputs": [output_metadata],
        },
        "imgsz": imgsz,
        "preprocessing": _preprocessing_metadata(imgsz),
        "postprocessing": _postprocessing_metadata(
            class_names,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_det=max_det,
            end2end=end2end,
        ),
        "acceptance_criteria": _acceptance_criteria(),
        "images": image_results,
        "aggregate": {
            "image_count": len(image_results),
            "element_count": int(aggregate_pytorch.size),
            "difference": aggregate_difference,
            **aggregate_detection,
        },
    }
    write_json_atomic(payload, destination, overwrite=overwrite)
    return payload
