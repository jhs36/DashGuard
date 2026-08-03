"""Auditable CPU latency benchmarks for equivalent PyTorch and ONNX models."""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashguard.detection import Detection
from dashguard.inference.onnx_validation import (
    CPU_PROVIDER,
    DEFAULT_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    file_sha256,
    letterbox_geometry,
    postprocess_raw_detections,
    preprocess_bgr_image,
    pytorch_output_to_numpy,
)


SCHEMA_VERSION = "dashguard.cpu_backend_benchmark.v2"
EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_OUTPUT_SHAPE = [1, 84, 8400]
PERCENTILE_METHOD = "linear interpolation (NumPy method='linear')"
STARTUP_COMPARISON_NOTE = (
    "application startup cost; informative comparison, not an identical "
    "computational boundary"
)
STARTUP_SINGLE_MEASUREMENT_NOTE = (
    "count=1; percentile fields equal the only sample and do not represent a "
    "latency distribution"
)
BACKEND_PYTORCH = "pytorch"
BACKEND_ONNX = "onnxruntime_cpu"
CSV_COLUMNS = [
    "stage",
    "backend",
    "count",
    "minimum_ms",
    "mean_ms",
    "standard_deviation_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "maximum_ms",
    "estimated_single_stream_fps_from_p50",
    "estimated_single_stream_fps_from_p95",
    "notes",
]


def file_metadata(path: str | Path) -> dict[str, Any]:
    """Return stable file metadata used to connect benchmark evidence."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Expected a regular file: {file_path}")
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "resolved_path": str(file_path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": file_sha256(file_path),
    }


def latency_statistics(samples_ns: Sequence[int | float]) -> dict[str, Any]:
    """Validate nanosecond samples and return JSON-safe millisecond statistics."""

    if not samples_ns:
        raise ValueError("At least one latency sample is required")
    normalized_ns: list[float] = []
    for index, sample in enumerate(samples_ns):
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise TypeError(f"Latency sample {index} must be numeric")
        value = float(sample)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Latency sample {index} must be finite and nonnegative"
            )
        normalized_ns.append(value)

    samples_ms = np.asarray(normalized_ns, dtype=np.float64) / 1_000_000.0
    percentiles = np.percentile(samples_ms, [50, 95, 99], method="linear")
    result = {
        "count": len(normalized_ns),
        "minimum_ms": float(np.min(samples_ms)),
        "mean_ms": float(statistics.fmean(float(value) for value in samples_ms)),
        "standard_deviation_ms": float(
            statistics.pstdev(float(value) for value in samples_ms)
        ),
        "p50_ms": float(percentiles[0]),
        "p95_ms": float(percentiles[1]),
        "p99_ms": float(percentiles[2]),
        "maximum_ms": float(np.max(samples_ms)),
        "latency_samples_ms": [float(value) for value in samples_ms],
    }
    _assert_json_finite(result)
    return result


def estimated_single_stream_fps(statistics_result: Mapping[str, Any]) -> dict[str, float]:
    """Estimate sequential single-stream FPS from P50 and P95 latency."""

    derived: dict[str, float] = {}
    for percentile in ("p50", "p95"):
        latency = float(statistics_result[f"{percentile}_ms"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(
                f"{percentile}_ms must be finite and greater than zero for FPS"
            )
        derived[f"estimated_single_stream_fps_from_{percentile}"] = 1000.0 / latency
    return derived


def onnx_speedup_over_pytorch(
    pytorch_statistics: Mapping[str, Any],
    onnx_statistics: Mapping[str, Any],
) -> dict[str, float]:
    """Return ONNX-over-PyTorch sequential latency ratios."""

    speedup: dict[str, float] = {}
    for percentile in ("p50", "p95"):
        pytorch_latency = float(pytorch_statistics[f"{percentile}_ms"])
        onnx_latency = float(onnx_statistics[f"{percentile}_ms"])
        if not math.isfinite(pytorch_latency) or pytorch_latency < 0:
            raise ValueError(f"PyTorch {percentile} latency is invalid")
        if not math.isfinite(onnx_latency) or onnx_latency <= 0:
            raise ValueError(f"ONNX {percentile} latency must be greater than zero")
        speedup[f"onnx_speedup_over_pytorch_{percentile}"] = (
            pytorch_latency / onnx_latency
        )
    return speedup


def backend_order(iteration: int) -> tuple[str, str]:
    """Alternate AB/BA ordering within the balanced four-step schedule."""

    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    if iteration % 2 == 0:
        return BACKEND_PYTORCH, BACKEND_ONNX
    return BACKEND_ONNX, BACKEND_PYTORCH


def image_index_for_iteration(iteration: int, image_count: int) -> int:
    """Choose paired images independently from backend order in a four-step cycle."""

    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    if image_count == 2:
        return (iteration % 4) // 2
    return iteration % image_count


def balanced_schedule_entry(iteration: int) -> dict[str, Any]:
    """Describe one entry in the required two-image, two-order schedule."""

    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    return {
        "iteration_mod_4": iteration % 4,
        "image_index": image_index_for_iteration(iteration, 2),
        "backend_order": list(backend_order(iteration)),
    }


def validate_balanced_iteration_count(value: int, label: str) -> None:
    """Require complete four-step cycles so every image/order cell is equal."""

    if value < 0 or (label == "iterations" and value == 0):
        raise ValueError(
            "warmup must be nonnegative and iterations must be positive"
        )
    if value % 4 != 0:
        raise ValueError(f"{label} must be a multiple of 4 for balanced scheduling")


def schedule_balance_summary(iterations: int) -> dict[str, Any]:
    """Count each image/backend-position cell for a complete schedule."""

    if iterations < 0 or iterations % 4 != 0:
        raise ValueError("schedule count must be a nonnegative multiple of 4")
    by_backend = {
        backend: {
            f"image_{image_index}": {"first": 0, "second": 0}
            for image_index in (0, 1)
        }
        for backend in (BACKEND_PYTORCH, BACKEND_ONNX)
    }
    for iteration in range(iterations):
        entry = balanced_schedule_entry(iteration)
        for position, backend in zip(
            ("first", "second"), entry["backend_order"], strict=True
        ):
            by_backend[backend][f"image_{entry['image_index']}"][position] += 1
    return {
        "iteration_count": iterations,
        "complete_four_step_cycles": iterations // 4,
        "by_backend_image_and_position": by_backend,
        "exactly_balanced": all(
            len(
                {
                    count
                    for image_counts in backend_counts.values()
                    for count in image_counts.values()
                }
            )
            == 1
            for backend_counts in by_backend.values()
        ),
    }


@contextmanager
def _garbage_collection_disabled(gc_controller: Any = gc) -> Any:
    originally_enabled = bool(gc_controller.isenabled())
    if originally_enabled:
        gc_controller.disable()
    try:
        yield
    finally:
        currently_enabled = bool(gc_controller.isenabled())
        if originally_enabled and not currently_enabled:
            gc_controller.enable()
        elif not originally_enabled and currently_enabled:
            gc_controller.disable()


def measure_startup_load(
    loader: Callable[[], Any],
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    gc_controller: Any = gc,
) -> tuple[Any, int]:
    """Time one loader without including any later warmup or inference."""

    with _garbage_collection_disabled(gc_controller):
        start_ns = clock_ns()
        loaded = loader()
        elapsed_ns = clock_ns() - start_ns
    if elapsed_ns < 0:
        raise ValueError("Clock returned a negative startup duration")
    return loaded, elapsed_ns


def measure_latency_samples(
    action: Callable[[int], Any],
    *,
    image_count: int,
    warmup: int,
    iterations: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    gc_controller: Any = gc,
    result_callback: Callable[[int, Any], None] | None = None,
) -> list[int]:
    """Warm up, then time a single action; statistics remain a caller concern."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    for iteration in range(warmup):
        action(image_index_for_iteration(iteration, image_count))

    samples: list[int] = []
    with _garbage_collection_disabled(gc_controller):
        for iteration in range(iterations):
            image_index = image_index_for_iteration(iteration, image_count)
            start_ns = clock_ns()
            result = action(image_index)
            elapsed_ns = clock_ns() - start_ns
            if elapsed_ns < 0:
                raise ValueError("Clock returned a negative latency duration")
            samples.append(elapsed_ns)
            if result_callback is not None:
                result_callback(image_index, result)
    return samples


def measure_paired_backend_samples(
    actions: Mapping[str, Callable[[int], Any]],
    *,
    image_count: int,
    warmup: int,
    iterations: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    gc_controller: Any = gc,
    result_callback: Callable[[str, int, Any], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Measure paired backends and preserve schedule provenance for every sample."""

    expected_backends = {BACKEND_PYTORCH, BACKEND_ONNX}
    if set(actions) != expected_backends:
        raise ValueError(f"actions must contain exactly {sorted(expected_backends)}")
    if image_count != 2:
        raise ValueError("Balanced paired measurement requires exactly two images")
    validate_balanced_iteration_count(warmup, "warmup")
    validate_balanced_iteration_count(iterations, "iterations")

    for iteration in range(warmup):
        image_index = image_index_for_iteration(iteration, image_count)
        for backend in backend_order(iteration):
            actions[backend](image_index)

    samples: dict[str, list[dict[str, Any]]] = {
        BACKEND_PYTORCH: [],
        BACKEND_ONNX: [],
    }
    with _garbage_collection_disabled(gc_controller):
        for iteration in range(iterations):
            image_index = image_index_for_iteration(iteration, image_count)
            for execution_position, backend in zip(
                ("first", "second"), backend_order(iteration), strict=True
            ):
                start_ns = clock_ns()
                result = actions[backend](image_index)
                elapsed_ns = clock_ns() - start_ns
                if elapsed_ns < 0:
                    raise ValueError("Clock returned a negative latency duration")
                samples[backend].append(
                    {
                        "sample_index": len(samples[backend]),
                        "iteration": iteration,
                        "image_index": image_index,
                        "execution_position": execution_position,
                        "latency_ns": elapsed_ns,
                    }
                )
                if result_callback is not None:
                    result_callback(backend, image_index, result)
    return samples


def sample_latencies_ns(sample_records: Sequence[Mapping[str, Any]]) -> list[int]:
    """Extract validated raw latency values from provenance-bearing samples."""

    return [int(record["latency_ns"]) for record in sample_records]


def _audit_group_statistics(
    sample_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statistics_result = latency_statistics(sample_latencies_ns(sample_records))
    return {
        "count": statistics_result["count"],
        "mean_ms": statistics_result["mean_ms"],
        "p50_ms": statistics_result["p50_ms"],
        "p95_ms": statistics_result["p95_ms"],
    }


def subgroup_audit_statistics(
    sample_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate image, execution-position, and joint audit subgroups."""

    if not sample_records:
        raise ValueError("At least one provenance-bearing sample is required")
    sample_indices = [int(record["sample_index"]) for record in sample_records]
    expected_indices = list(range(len(sample_records)))
    if sorted(sample_indices) != expected_indices:
        raise ValueError("Subgroup samples contain duplicate or missing sample indices")
    for record in sample_records:
        if record["image_index"] not in (0, 1):
            raise ValueError("Subgroup sample image_index must be 0 or 1")
        if record["execution_position"] not in ("first", "second"):
            raise ValueError("Subgroup execution_position must be first or second")

    by_image = {
        f"image_{image_index}": _audit_group_statistics(
            [
                record
                for record in sample_records
                if record["image_index"] == image_index
            ]
        )
        for image_index in (0, 1)
    }
    by_execution_order = {
        position: _audit_group_statistics(
            [
                record
                for record in sample_records
                if record["execution_position"] == position
            ]
        )
        for position in ("first", "second")
    }
    by_image_and_execution_order = {
        f"image_{image_index}": {
            position: _audit_group_statistics(
                [
                    record
                    for record in sample_records
                    if record["image_index"] == image_index
                    and record["execution_position"] == position
                ]
            )
            for position in ("first", "second")
        }
        for image_index in (0, 1)
    }
    joint_count_sum = sum(
        group["count"]
        for image_groups in by_image_and_execution_order.values()
        for group in image_groups.values()
    )
    integrity = {
        "overall_count": len(sample_records),
        "unique_sample_index_count": len(set(sample_indices)),
        "by_image_count_sum": sum(group["count"] for group in by_image.values()),
        "by_execution_order_count_sum": sum(
            group["count"] for group in by_execution_order.values()
        ),
        "by_image_and_execution_order_count_sum": joint_count_sum,
        "duplicate_sample_count": len(sample_indices) - len(set(sample_indices)),
        "missing_sample_count": len(set(expected_indices) - set(sample_indices)),
        "status": "passed",
    }
    if any(
        integrity[key] != len(sample_records)
        for key in (
            "unique_sample_index_count",
            "by_image_count_sum",
            "by_execution_order_count_sum",
            "by_image_and_execution_order_count_sum",
        )
    ):
        raise RuntimeError("Subgroup audit did not cover every sample exactly once")
    return {
        "role": "audit_only; does not replace overall benchmark statistics",
        "by_image": by_image,
        "by_execution_order": by_execution_order,
        "by_image_and_execution_order": by_image_and_execution_order,
        "integrity": integrity,
    }


def _load_json_strict(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains forbidden non-finite value: {value}")

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("Consistency JSON root must be an object")
    return payload


def _canonical_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def validate_consistency_preflight(
    consistency_json_path: str | Path,
    pytorch_model_path: str | Path,
    onnx_model_path: str | Path,
    image_paths: Sequence[str | Path],
    imgsz: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Connect all benchmark inputs to a previously accepted consistency report."""

    if not image_paths:
        raise ValueError("At least one image path is required")
    consistency_path = Path(consistency_json_path)
    consistency_metadata = file_metadata(consistency_path)
    report = _load_json_strict(consistency_path)

    _require_equal(
        report.get("structural_validation"),
        "passed",
        f"structural_validation ({consistency_metadata['resolved_path']})",
    )
    _require_equal(
        report.get("numerical_acceptance"),
        "passed",
        f"numerical_acceptance ({consistency_metadata['resolved_path']})",
    )

    pytorch_metadata = file_metadata(pytorch_model_path)
    onnx_metadata = file_metadata(onnx_model_path)
    _require_equal(
        pytorch_metadata["sha256"],
        report["models"]["pytorch"]["sha256"],
        f"PyTorch model SHA-256 ({pytorch_metadata['resolved_path']})",
    )
    _require_equal(
        onnx_metadata["sha256"],
        report["models"]["onnx"]["sha256"],
        f"ONNX model SHA-256 ({onnx_metadata['resolved_path']})",
    )

    runtime = report["onnx_runtime"]
    _require_equal(
        runtime.get("requested_providers"),
        [CPU_PROVIDER],
        "consistency requested ONNX providers",
    )
    _require_equal(
        runtime.get("actual_providers"),
        [CPU_PROVIDER],
        "consistency actual ONNX providers",
    )
    input_nodes = report["onnx_io_metadata"]["inputs"]
    output_nodes = report["onnx_io_metadata"]["outputs"]
    if len(input_nodes) != 1 or len(output_nodes) != 1:
        raise ValueError("Consistency JSON must describe exactly one ONNX input and output")
    _require_equal(input_nodes[0].get("shape"), EXPECTED_INPUT_SHAPE, "ONNX input shape")
    _require_equal(output_nodes[0].get("shape"), EXPECTED_OUTPUT_SHAPE, "ONNX output shape")
    _require_equal(imgsz, 640, "benchmark image size")
    _require_equal(report.get("imgsz"), imgsz, "consistency image size")

    postprocessing = report["postprocessing"]
    expected_postprocessing = {
        "confidence_threshold": DEFAULT_CONFIDENCE,
        "iou_threshold": DEFAULT_IOU_THRESHOLD,
        "maximum_detections": DEFAULT_MAX_DETECTIONS,
        "agnostic_nms": DEFAULT_AGNOSTIC_NMS,
        "multi_label": False,
        "target_class_ids": [0, 1, 2, 3, 5, 7],
    }
    for key, expected in expected_postprocessing.items():
        _require_equal(postprocessing.get(key), expected, f"postprocessing {key}")

    consistency_images = report.get("images", [])
    matched_entries: list[dict[str, Any]] = []
    actual_images: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for image_path_value in image_paths:
        image_metadata = file_metadata(image_path_value)
        actual_images.append(image_metadata)
        canonical = _canonical_path(image_metadata["resolved_path"])
        matches = [
            (index, entry)
            for index, entry in enumerate(consistency_images)
            if _canonical_path(entry["resolved_path"]) == canonical
        ]
        if len(matches) != 1:
            raise ValueError(
                "Each benchmark image must match exactly one consistency JSON path: "
                f"{image_metadata['resolved_path']}"
            )
        index, entry = matches[0]
        if index in used_indices:
            raise ValueError("Benchmark image paths must be unique")
        used_indices.add(index)
        _require_equal(
            image_metadata["sha256"],
            entry["sha256"],
            f"input image SHA-256 ({image_metadata['path']})",
        )
        matched_entries.append(entry)

    summary = {
        "status": "passed",
        "checks": {
            "structural_validation_passed": True,
            "numerical_acceptance_passed": True,
            "pytorch_model_sha256_matches": True,
            "onnx_model_sha256_matches": True,
            "all_input_sha256_match": True,
            "consistency_provider_is_cpu_only": True,
            "input_shape_is_1x3x640x640": True,
            "output_shape_is_1x84x8400": True,
            "postprocessing_configuration_matches": True,
        },
        "consistency_json": consistency_metadata,
        "models": {"pytorch": pytorch_metadata, "onnx": onnx_metadata},
        "inputs": actual_images,
    }
    return report, summary, matched_entries


def _node_metadata(node: Any) -> dict[str, Any]:
    shape: list[int | str | None] = []
    for dimension in node.shape:
        if dimension is None or isinstance(dimension, str):
            shape.append(dimension)
        else:
            shape.append(int(dimension))
    return {"name": str(node.name), "shape": shape, "dtype": str(node.type)}


def validate_runtime_provider(providers: Sequence[str]) -> None:
    """Require an ONNX Runtime session backed only by the CPU provider."""

    if list(providers) != [CPU_PROVIDER]:
        raise RuntimeError(
            "ONNX Runtime must use exactly CPUExecutionProvider; "
            f"actual providers: {list(providers)}"
        )


def validate_raw_output(output: Any, label: str) -> np.ndarray:
    """Normalize a raw result only outside inference-only timers and validate it."""

    array = np.asarray(output)
    if list(array.shape) != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(
            f"{label} output shape mismatch: actual={list(array.shape)}, "
            f"expected={EXPECTED_OUTPUT_SHAPE}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{label} output must have a floating-point dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} output contains NaN or infinity")
    return np.ascontiguousarray(array, dtype=np.float32)


def detection_class_signature(
    detections: Sequence[Detection | Mapping[str, Any]],
) -> list[list[Any]]:
    """Return a stable, coordinate-independent detection class signature."""

    signature: list[tuple[int, str]] = []
    for detection in detections:
        if isinstance(detection, Mapping):
            signature.append((int(detection["class_id"]), str(detection["class_name"])))
        else:
            signature.append((int(detection.class_id), str(detection.class_name)))
    return [[class_id, class_name] for class_id, class_name in sorted(signature)]


def validate_detection_classes(
    detections: Sequence[Detection],
    expected: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    """Require actual detection counts and classes to match accepted evidence."""

    actual_signature = detection_class_signature(detections)
    expected_signature = detection_class_signature(expected)
    if Counter(map(tuple, actual_signature)) != Counter(map(tuple, expected_signature)):
        raise ValueError(
            f"{label} detection count/classes mismatch: "
            f"actual={actual_signature}, expected={expected_signature}"
        )


def configure_threads(threads: int, opencv_threads: int) -> tuple[Any, dict[str, Any]]:
    """Apply and verify every requested CPU thread setting before model execution."""

    if threads <= 0 or opencv_threads <= 0:
        raise ValueError("Thread counts must be positive")
    import torch

    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise RuntimeError(
            f"torch.set_num_threads did not apply: {torch.get_num_threads()}"
        )

    interop_status = "applied"
    interop_error: str | None = None
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        interop_error = str(exc)
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError(
                "Unable to set PyTorch inter-op threads to 1; "
                f"actual={torch.get_num_interop_threads()}: {exc}"
            ) from exc
        interop_status = "already_1; setter reported RuntimeError"

    cv2.setNumThreads(opencv_threads)
    actual_opencv_threads = cv2.getNumThreads()
    if actual_opencv_threads != opencv_threads:
        raise RuntimeError(
            "cv2.setNumThreads did not apply: "
            f"requested={opencv_threads}, actual={actual_opencv_threads}"
        )

    settings = {
        "requested": {
            "pytorch_intra_op_threads": threads,
            "pytorch_inter_op_threads": 1,
            "onnx_intra_op_threads": threads,
            "onnx_inter_op_threads": 1,
            "opencv_threads": opencv_threads,
        },
        "pytorch": {
            "device": "cpu",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "interop_setting_status": interop_status,
            "interop_setting_error": interop_error,
            "eval_mode": True,
            "inference_mode": True,
        },
        "opencv": {
            "requested_threads": opencv_threads,
            "actual_threads": actual_opencv_threads,
            "setting_status": "applied",
        },
    }
    return torch, settings


def _enum_name(value: Any) -> str:
    text = str(value)
    return text.split(".")[-1]


def _runtime_versions(torch: Any, ort: Any) -> dict[str, str]:
    def package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "ultralytics": package_version("ultralytics"),
        "onnxruntime": str(ort.__version__),
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
    }


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _environment_metadata(torch: Any, ort: Any) -> dict[str, Any]:
    cpu_name = ""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            cpu_name = ""
    cpu_name = cpu_name or (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.uname().processor
        or "unknown"
    )
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "cpu": cpu_name,
        "logical_cpu_count": os.cpu_count(),
        "versions": _runtime_versions(torch, ort),
    }


def _postprocess(
    raw_output: np.ndarray,
    input_array: np.ndarray,
    image: np.ndarray,
    geometry: Mapping[str, Any],
    class_names: Mapping[int, str] | Sequence[str],
    end2end: bool,
) -> list[Detection]:
    return postprocess_raw_detections(
        raw_output,
        input_shape=input_array.shape,
        original_shape=image.shape,
        geometry=geometry,
        class_names=class_names,
        confidence=DEFAULT_CONFIDENCE,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        max_det=DEFAULT_MAX_DETECTIONS,
        agnostic_nms=DEFAULT_AGNOSTIC_NMS,
        end2end=end2end,
    )


def _measurement_boundaries() -> dict[str, Any]:
    return {
        "startup_load": {
            "measurement_type": "single_measurement",
            "pytorch": (
                "YOLO(pt_path), internal model acquisition, CPU float32 transfer, "
                "and eval preparation"
            ),
            "onnxruntime_cpu": (
                "SessionOptions creation/configuration, InferenceSession creation, "
                "and actual provider verification"
            ),
            "comparison_note": STARTUP_COMPARISON_NOTE,
            "interpretation_note": STARTUP_SINGLE_MEASUREMENT_NOTE,
            "excluded": ["warmup", "repeated inference"],
        },
        "preprocess": {
            "included": (
                "shared fixed LetterBox and BGR uint8 HWC to contiguous normalized "
                "FP32 [1,3,640,640] NumPy conversion"
            ),
            "excluded": ["disk read", "image decode", "backend inference"],
        },
        "inference_only": {
            "pytorch": "direct internal model forward on a pre-created CPU tensor",
            "onnxruntime_cpu": "InferenceSession.run() on a pre-created NumPy input",
            "excluded": [
                "file read",
                "input array creation",
                "torch tensor creation",
                "NumPy output adapter",
                "postprocess",
                "statistics",
            ],
            "synchronization": (
                "CPU direct forward and CPU InferenceSession.run return only after "
                "the measured inference completes"
            ),
        },
        "postprocess": {
            "included": (
                "backend-specific precomputed raw output copy required by mutating "
                "NMS, shared NMS, coordinate scaling, and Detection creation"
            ),
            "excluded": ["backend inference"],
        },
        "image_to_detections": {
            "included": [
                "shared preprocess from an already decoded in-memory BGR image",
                "backend inference",
                "required backend input/output adapters",
                "shared NMS",
                "original-coordinate scaling",
                "final Detection creation",
            ],
            "excluded": [
                "disk read and image decode",
                "model/session load",
                "consistency validation",
                "statistics",
                "JSON/CSV output",
                "stdout",
            ],
        },
    }


def _assert_json_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"JSON contains non-finite float at {path}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")


def ensure_output_pair_available(
    output_json: str | Path,
    output_csv: str | Path,
    overwrite: bool,
) -> None:
    """Fail before any benchmark work if either destination already exists."""

    destinations = [Path(output_json), Path(output_csv)]
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError("JSON and CSV destinations must be different files")
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Benchmark output already exists; pass --overwrite to replace both: "
            + ", ".join(existing)
        )


def _write_json_temp(payload: Mapping[str, Any], destination: Path) -> Path:
    _assert_json_finite(payload)
    serialized = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_csv_temp(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=CSV_COLUMNS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                if list(row.keys()) != CSV_COLUMNS:
                    raise ValueError("CSV row keys must exactly match CSV_COLUMNS order")
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def publish_report_pair_atomic(
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_json: str | Path,
    output_csv: str | Path,
    *,
    overwrite: bool,
    replace: Callable[[str | Path, str | Path], None] = os.replace,
) -> None:
    """Publish JSON and CSV together, restoring originals after any failure."""

    json_destination = Path(output_json)
    csv_destination = Path(output_csv)
    ensure_output_pair_available(json_destination, csv_destination, overwrite)
    json_temporary: Path | None = None
    csv_temporary: Path | None = None
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    rollback_incomplete = False
    try:
        json_temporary = _write_json_temp(payload, json_destination)
        csv_temporary = _write_csv_temp(rows, csv_destination)

        for destination in (json_destination, csv_destination):
            if destination.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".bak",
                    dir=destination.parent,
                )
                os.close(descriptor)
                backup = Path(backup_name)
                backup.unlink()
                replace(destination, backup)
                backups[destination] = backup

        replace(json_temporary, json_destination)
        published.append(json_destination)
        json_temporary = None
        replace(csv_temporary, csv_destination)
        published.append(csv_destination)
        csv_temporary = None
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        for destination in reversed(published):
            try:
                destination.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"remove {destination}: {exc}")
        for destination, backup in backups.items():
            if backup.exists():
                try:
                    replace(backup, destination)
                except OSError as exc:
                    rollback_errors.append(f"restore {destination}: {exc}")
        if rollback_errors:
            rollback_incomplete = True
            raise RuntimeError(
                "Benchmark publish failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from publish_error
        raise
    finally:
        if json_temporary is not None:
            json_temporary.unlink(missing_ok=True)
        if csv_temporary is not None:
            csv_temporary.unlink(missing_ok=True)
        if not rollback_incomplete:
            for backup in backups.values():
                backup.unlink(missing_ok=True)


def _csv_row(
    stage: str,
    backend: str,
    stats: Mapping[str, Any],
    notes: str,
    *,
    include_fps: bool = False,
) -> dict[str, Any]:
    fps = estimated_single_stream_fps(stats) if include_fps else {}
    return {
        "stage": stage,
        "backend": backend,
        "count": stats["count"],
        "minimum_ms": stats["minimum_ms"],
        "mean_ms": stats["mean_ms"],
        "standard_deviation_ms": stats["standard_deviation_ms"],
        "p50_ms": stats["p50_ms"],
        "p95_ms": stats["p95_ms"],
        "p99_ms": stats["p99_ms"],
        "maximum_ms": stats["maximum_ms"],
        "estimated_single_stream_fps_from_p50": fps.get(
            "estimated_single_stream_fps_from_p50", ""
        ),
        "estimated_single_stream_fps_from_p95": fps.get(
            "estimated_single_stream_fps_from_p95", ""
        ),
        "notes": notes,
    }


def build_csv_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nine required stage/backend statistics into CSV rows."""

    sequential_note = (
        "FPS is estimated from sequential single-stream latency; it is not "
        "concurrent throughput."
    )
    return [
        _csv_row(
            "startup_load",
            BACKEND_PYTORCH,
            results["startup_load"][BACKEND_PYTORCH],
            STARTUP_COMPARISON_NOTE + "; " + STARTUP_SINGLE_MEASUREMENT_NOTE,
        ),
        _csv_row(
            "startup_load",
            BACKEND_ONNX,
            results["startup_load"][BACKEND_ONNX],
            STARTUP_COMPARISON_NOTE + "; " + STARTUP_SINGLE_MEASUREMENT_NOTE,
        ),
        _csv_row("preprocess", "shared", results["preprocess"]["shared"], "Disk decode excluded."),
        _csv_row(
            "inference_only",
            BACKEND_PYTORCH,
            results["inference_only"][BACKEND_PYTORCH],
            sequential_note,
            include_fps=True,
        ),
        _csv_row(
            "inference_only",
            BACKEND_ONNX,
            results["inference_only"][BACKEND_ONNX],
            sequential_note,
            include_fps=True,
        ),
        _csv_row(
            "postprocess",
            "pytorch_output",
            results["postprocess"]["pytorch_output"],
            "Inference excluded; required NMS input copy included.",
        ),
        _csv_row(
            "postprocess",
            "onnx_output",
            results["postprocess"]["onnx_output"],
            "Inference excluded; required NMS input copy included.",
        ),
        _csv_row(
            "image_to_detections",
            BACKEND_PYTORCH,
            results["image_to_detections"][BACKEND_PYTORCH],
            sequential_note + " Disk decode and model load excluded.",
            include_fps=True,
        ),
        _csv_row(
            "image_to_detections",
            BACKEND_ONNX,
            results["image_to_detections"][BACKEND_ONNX],
            sequential_note + " Disk decode and model load excluded.",
            include_fps=True,
        ),
    ]


def _validate_files_unchanged(
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    for metadata_before in [
        preflight["models"]["pytorch"],
        preflight["models"]["onnx"],
        *preflight["inputs"],
    ]:
        metadata_after = file_metadata(metadata_before["resolved_path"])
        _require_equal(
            metadata_after["sha256"],
            metadata_before["sha256"],
            f"postflight SHA-256 ({metadata_before['resolved_path']})",
        )
        _require_equal(
            metadata_after["size_bytes"],
            metadata_before["size_bytes"],
            f"postflight size ({metadata_before['resolved_path']})",
        )
        checked.append(
            {
                "resolved_path": metadata_before["resolved_path"],
                "size_bytes_unchanged": True,
                "sha256_unchanged": True,
            }
        )
    return {"files": checked}


def startup_single_measurement_result(sample_ns: int | float) -> dict[str, Any]:
    """Represent startup as one observation, not as a latency distribution."""

    statistics_result = latency_statistics([sample_ns])
    return {
        **statistics_result,
        "measurement_type": "single_measurement",
        "single_measurement_ms": statistics_result["latency_samples_ms"][0],
        "interpretation_note": STARTUP_SINGLE_MEASUREMENT_NOTE,
    }


def run_cpu_backend_benchmark(
    *,
    image_paths: Sequence[str | Path],
    pytorch_model_path: str | Path,
    onnx_model_path: str | Path,
    consistency_json_path: str | Path,
    imgsz: int,
    threads: int,
    opencv_threads: int,
    warmup: int,
    iterations: int,
    output_json: str | Path,
    output_csv: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run all five boundaries and atomically publish the benchmark evidence."""

    if len(image_paths) != 2:
        raise ValueError("Balanced CPU benchmark requires exactly two image paths")
    validate_balanced_iteration_count(warmup, "warmup")
    validate_balanced_iteration_count(iterations, "iterations")
    ensure_output_pair_available(output_json, output_csv, overwrite)
    original_gc_enabled = gc.isenabled()
    torch, thread_settings = configure_threads(threads, opencv_threads)

    consistency, preflight, consistency_image_entries = validate_consistency_preflight(
        consistency_json_path,
        pytorch_model_path,
        onnx_model_path,
        image_paths,
        imgsz,
    )

    images: list[np.ndarray] = []
    for image_path_value in image_paths:
        image = cv2.imread(str(Path(image_path_value)), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV failed to read image: {image_path_value}")
        images.append(image)

    import onnxruntime as ort
    from ultralytics import YOLO

    def load_pytorch() -> dict[str, Any]:
        wrapper = YOLO(str(pytorch_model_path))
        model = wrapper.model
        if model is None:
            raise RuntimeError("YOLO did not expose an internal PyTorch model")
        model = model.to("cpu").float().eval()
        if model.training:
            raise RuntimeError("PyTorch model did not enter eval mode")
        return {
            "wrapper": wrapper,
            "model": model,
            "class_names": model.names,
            "end2end": bool(getattr(model, "end2end", False)),
        }

    pytorch_backend, pytorch_startup_ns = measure_startup_load(load_pytorch)

    def load_onnx() -> dict[str, Any]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=options,
            providers=[CPU_PROVIDER],
        )
        validate_runtime_provider(session.get_providers())
        return {"session": session, "requested_options": options}

    onnx_backend, onnx_startup_ns = measure_startup_load(load_onnx)
    session = onnx_backend["session"]
    validate_runtime_provider(session.get_providers())
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ONNX session must expose exactly one input and output")
    input_metadata = _node_metadata(inputs[0])
    output_metadata = _node_metadata(outputs[0])
    _require_equal(input_metadata["shape"], EXPECTED_INPUT_SHAPE, "runtime ONNX input shape")
    _require_equal(output_metadata["shape"], EXPECTED_OUTPUT_SHAPE, "runtime ONNX output shape")

    actual_options = session.get_session_options()
    onnx_options = {
        "intra_op_num_threads": int(actual_options.intra_op_num_threads),
        "inter_op_num_threads": int(actual_options.inter_op_num_threads),
        "execution_mode": _enum_name(actual_options.execution_mode),
        "graph_optimization_level": _enum_name(actual_options.graph_optimization_level),
    }
    _require_equal(onnx_options["intra_op_num_threads"], threads, "ORT intra-op threads")
    _require_equal(onnx_options["inter_op_num_threads"], 1, "ORT inter-op threads")
    _require_equal(onnx_options["execution_mode"], "ORT_SEQUENTIAL", "ORT execution mode")
    _require_equal(
        onnx_options["graph_optimization_level"],
        "ORT_ENABLE_ALL",
        "ORT graph optimization level",
    )
    thread_settings["onnxruntime"] = {
        **onnx_options,
        "requested_providers": [CPU_PROVIDER],
        "actual_providers": list(session.get_providers()),
        "provider_options": session.get_provider_options(),
        "setting_status": "applied_and_verified",
    }

    input_name = input_metadata["name"]
    output_name = output_metadata["name"]
    model = pytorch_backend["model"]
    class_names = pytorch_backend["class_names"]
    end2end = pytorch_backend["end2end"]

    preprocessed_inputs = [preprocess_bgr_image(image, imgsz) for image in images]
    geometries = [letterbox_geometry(image.shape, imgsz) for image in images]
    torch_inputs = [torch.from_numpy(array) for array in preprocessed_inputs]
    for tensor in torch_inputs:
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise RuntimeError("Pre-created PyTorch input is not CPU float32")

    def pytorch_raw_numpy(index: int) -> np.ndarray:
        with torch.inference_mode():
            native = model(torch_inputs[index])
        return validate_raw_output(
            pytorch_output_to_numpy(native),
            f"PyTorch image {index}",
        )

    def onnx_raw_numpy(index: int) -> np.ndarray:
        native = session.run(
            [output_name],
            {input_name: preprocessed_inputs[index]},
        )
        if len(native) != 1:
            raise RuntimeError("ONNX Runtime returned an unexpected output count")
        return validate_raw_output(native[0], f"ONNX image {index}")

    premeasurement_raw = {
        BACKEND_PYTORCH: [pytorch_raw_numpy(index) for index in range(len(images))],
        BACKEND_ONNX: [onnx_raw_numpy(index) for index in range(len(images))],
    }
    for index in range(len(images)):
        expected_comparison = consistency_image_entries[index]["detection_comparison"]
        validate_detection_classes(
            _postprocess(
                premeasurement_raw[BACKEND_PYTORCH][index],
                preprocessed_inputs[index],
                images[index],
                geometries[index],
                class_names,
                end2end,
            ),
            expected_comparison["pytorch_detections"],
            f"premeasurement PyTorch image {index}",
        )
        validate_detection_classes(
            _postprocess(
                premeasurement_raw[BACKEND_ONNX][index],
                preprocessed_inputs[index],
                images[index],
                geometries[index],
                class_names,
                end2end,
            ),
            expected_comparison["onnx_detections"],
            f"premeasurement ONNX image {index}",
        )

    preprocess_samples = measure_latency_samples(
        lambda index: preprocess_bgr_image(images[index], imgsz),
        image_count=len(images),
        warmup=warmup,
        iterations=iterations,
    )

    inference_actions = {
        BACKEND_PYTORCH: lambda index: model(torch_inputs[index]),
        BACKEND_ONNX: lambda index: session.run(
            [output_name], {input_name: preprocessed_inputs[index]}
        ),
    }
    with torch.inference_mode():
        inference_samples = measure_paired_backend_samples(
            inference_actions,
            image_count=len(images),
            warmup=warmup,
            iterations=iterations,
        )

    postprocess_actions = {
        BACKEND_PYTORCH: lambda index: _postprocess(
            premeasurement_raw[BACKEND_PYTORCH][index],
            preprocessed_inputs[index],
            images[index],
            geometries[index],
            class_names,
            end2end,
        ),
        BACKEND_ONNX: lambda index: _postprocess(
            premeasurement_raw[BACKEND_ONNX][index],
            preprocessed_inputs[index],
            images[index],
            geometries[index],
            class_names,
            end2end,
        ),
    }
    postprocess_samples = measure_paired_backend_samples(
        postprocess_actions,
        image_count=len(images),
        warmup=warmup,
        iterations=iterations,
    )

    def pytorch_end_to_end(index: int) -> list[Detection]:
        input_array = preprocess_bgr_image(images[index], imgsz)
        geometry = letterbox_geometry(images[index].shape, imgsz)
        tensor = torch.from_numpy(input_array)
        native_output = model(tensor)
        raw_output = pytorch_output_to_numpy(native_output)
        return _postprocess(
            raw_output,
            input_array,
            images[index],
            geometry,
            class_names,
            end2end,
        )

    def onnx_end_to_end(index: int) -> list[Detection]:
        input_array = preprocess_bgr_image(images[index], imgsz)
        geometry = letterbox_geometry(images[index].shape, imgsz)
        native_outputs = session.run([output_name], {input_name: input_array})
        if len(native_outputs) != 1:
            raise RuntimeError("ONNX Runtime returned an unexpected output count")
        raw_output = np.ascontiguousarray(
            native_outputs[0],
            dtype=np.float32,
        )
        return _postprocess(
            raw_output,
            input_array,
            images[index],
            geometry,
            class_names,
            end2end,
        )

    timed_detection_results: dict[str, list[tuple[int, Sequence[Detection]]]] = {
        BACKEND_PYTORCH: [],
        BACKEND_ONNX: [],
    }

    def preserve_timed_result(
        backend: str,
        index: int,
        detections: Sequence[Detection],
    ) -> None:
        timed_detection_results[backend].append((index, detections))

    with torch.inference_mode():
        end_to_end_samples = measure_paired_backend_samples(
            {
                BACKEND_PYTORCH: pytorch_end_to_end,
                BACKEND_ONNX: onnx_end_to_end,
            },
            image_count=len(images),
            warmup=warmup,
            iterations=iterations,
            result_callback=preserve_timed_result,
        )

    timed_detection_validations = {BACKEND_PYTORCH: 0, BACKEND_ONNX: 0}
    for backend, preserved_results in timed_detection_results.items():
        expected_key = (
            "pytorch_detections"
            if backend == BACKEND_PYTORCH
            else "onnx_detections"
        )
        for index, detections in preserved_results:
            comparison = consistency_image_entries[index]["detection_comparison"]
            validate_detection_classes(
                detections,
                comparison[expected_key],
                f"timed {backend} image {index}",
            )
            timed_detection_validations[backend] += 1

    postmeasurement_raw = {
        BACKEND_PYTORCH: [pytorch_raw_numpy(index) for index in range(len(images))],
        BACKEND_ONNX: [onnx_raw_numpy(index) for index in range(len(images))],
    }
    for backend, outputs_for_backend in postmeasurement_raw.items():
        for index, output in enumerate(outputs_for_backend):
            validate_raw_output(output, f"postmeasurement {backend} image {index}")
    validate_runtime_provider(session.get_providers())
    unchanged_validation = _validate_files_unchanged(preflight)

    if gc.isenabled() != original_gc_enabled:
        raise RuntimeError("Garbage collection state was not restored after measurement")

    results = {
        "startup_load": {
            BACKEND_PYTORCH: startup_single_measurement_result(
                pytorch_startup_ns
            ),
            BACKEND_ONNX: startup_single_measurement_result(onnx_startup_ns),
        },
        "preprocess": {"shared": latency_statistics(preprocess_samples)},
        "inference_only": {
            backend: latency_statistics(sample_latencies_ns(sample_records))
            for backend, sample_records in inference_samples.items()
        },
        "postprocess": {
            "pytorch_output": latency_statistics(
                sample_latencies_ns(postprocess_samples[BACKEND_PYTORCH])
            ),
            "onnx_output": latency_statistics(
                sample_latencies_ns(postprocess_samples[BACKEND_ONNX])
            ),
        },
        "image_to_detections": {
            backend: latency_statistics(sample_latencies_ns(sample_records))
            for backend, sample_records in end_to_end_samples.items()
        },
    }
    subgroup_audit = {
        "inference_only": {
            backend: subgroup_audit_statistics(sample_records)
            for backend, sample_records in inference_samples.items()
        },
        "postprocess": {
            backend: subgroup_audit_statistics(sample_records)
            for backend, sample_records in postprocess_samples.items()
        },
        "image_to_detections": {
            backend: subgroup_audit_statistics(sample_records)
            for backend, sample_records in end_to_end_samples.items()
        },
    }
    derived_metrics: dict[str, Any] = {
        "interpretation": (
            "All FPS values are estimates from sequential single-stream latency, "
            "not measured concurrent throughput."
        )
    }
    for stage in ("inference_only", "image_to_detections"):
        derived_metrics[stage] = {
            BACKEND_PYTORCH: estimated_single_stream_fps(
                results[stage][BACKEND_PYTORCH]
            ),
            BACKEND_ONNX: estimated_single_stream_fps(
                results[stage][BACKEND_ONNX]
            ),
            **onnx_speedup_over_pytorch(
                results[stage][BACKEND_PYTORCH],
                results[stage][BACKEND_ONNX],
            ),
        }

    actual_input_records = []
    for metadata_value, image, consistency_entry in zip(
        preflight["inputs"], images, consistency_image_entries, strict=True
    ):
        actual_input_records.append(
            {
                **metadata_value,
                "decoded_bgr_shape": list(image.shape),
                "consistency_image_path": consistency_entry["path"],
                "expected_pytorch_detection_classes": detection_class_signature(
                    consistency_entry["detection_comparison"]["pytorch_detections"]
                ),
                "expected_onnx_detection_classes": detection_class_signature(
                    consistency_entry["detection_comparison"]["onnx_detections"]
                ),
            }
        )

    postflight = {
        "status": "passed",
        "checks": {
            "runtime_provider_is_cpu_only": True,
            "premeasurement_outputs_finite_and_shape_1x84x8400": True,
            "postmeasurement_outputs_finite_and_shape_1x84x8400": True,
            "all_timed_end_to_end_detection_counts_and_classes_match": True,
            "model_and_input_files_unchanged": True,
            "garbage_collection_state_restored": True,
        },
        "timed_end_to_end_results_validated": timed_detection_validations,
        **unchanged_validation,
    }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "benchmark_purpose": (
            "Separate startup, shared preprocess, inference-only, shared "
            "postprocess, and in-memory image-to-detections CPU latency for "
            "an output-consistent PyTorch/ONNX model pair."
        ),
        "benchmark_limit": (
            "Sequential single-stream latency benchmark on one host; it does not "
            "measure concurrent throughput or generalize to other hardware."
        ),
        "git": {"head": _git_head()},
        "environment": _environment_metadata(torch, ort),
        "thread_settings": thread_settings,
        "clock": {
            "function": "time.perf_counter_ns",
            "internal_unit": "nanosecond",
            "output_unit": "millisecond",
            "cpu_execution_synchronous_at_call_return": True,
        },
        "garbage_collection_policy": {
            "disabled_during_timed_regions": True,
            "warmup_samples_timed": False,
            "state_before_benchmark": "enabled" if original_gc_enabled else "disabled",
            "state_after_benchmark": "enabled" if gc.isenabled() else "disabled",
            "original_state_restored_after_each_timed_region": True,
        },
        "models": preflight["models"],
        "consistency_evidence": {
            **preflight["consistency_json"],
            "schema_version": consistency.get("schema_version"),
            "structural_validation": consistency["structural_validation"],
            "numerical_acceptance": consistency["numerical_acceptance"],
        },
        "inputs": actual_input_records,
        "onnx_io_metadata": {"inputs": [input_metadata], "outputs": [output_metadata]},
        "onnx_runtime": {
            "provider": CPU_PROVIDER,
            "actual_providers": list(session.get_providers()),
            "session_options": onnx_options,
        },
        "benchmark_configuration": {
            "imgsz": imgsz,
            "warmup_iterations_per_stage": warmup,
            "timed_iterations_per_stage": iterations,
            "balanced_schedule": {
                "cycle_length": 4,
                "incomplete_cycle_policy": (
                    "warmup and timed iteration counts must be multiples of 4"
                ),
                "cycle": [balanced_schedule_entry(index) for index in range(4)],
                "description": (
                    "Two images and AB/BA backend execution order are independently "
                    "balanced over each four-step cycle for warmup and timed stages."
                ),
                "warmup_balance": schedule_balance_summary(warmup),
                "timed_balance": schedule_balance_summary(iterations),
            },
            "cpu_affinity_changed": False,
            "process_priority_changed": False,
        },
        "statistics_policy": {
            "percentile_method": PERCENTILE_METHOD,
            "standard_deviation": "population standard deviation (ddof=0)",
            "validation_before_statistics": "all samples finite and nonnegative",
            "individual_samples_preserved": True,
        },
        "measurement_boundaries": _measurement_boundaries(),
        "results": results,
        "subgroup_audit": subgroup_audit,
        "derived_metrics": derived_metrics,
        "preflight_validation": preflight,
        "postflight_validation": postflight,
        "known_limitations": [
            "Startup costs are informative application costs, not identical computational boundaries.",
            "Required preflight SHA-256 reads may warm operating-system file caches before startup timing.",
            "Only two balanced images are represented; workload diversity is limited.",
            "The four-step schedule balances image/order cells, but the first backend still executes immediately before the second within each iteration.",
            "OS scheduling, background activity, CPU frequency, and thermals were observed rather than controlled.",
            "No CPU affinity or process-priority changes were made.",
            "Derived FPS is the reciprocal of sequential latency and is not measured concurrent throughput.",
        ],
    }
    _assert_json_finite(payload)
    rows = build_csv_rows(results)
    publish_report_pair_atomic(
        payload,
        rows,
        output_json,
        output_csv,
        overwrite=overwrite,
    )
    return payload
