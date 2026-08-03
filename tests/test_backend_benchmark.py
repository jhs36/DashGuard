from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import dashguard.inference.benchmark as benchmark
from dashguard.detection import Detection
from dashguard.inference.benchmark import (
    BACKEND_ONNX,
    BACKEND_PYTORCH,
    CSV_COLUMNS,
    STARTUP_SINGLE_MEASUREMENT_NOTE,
    balanced_schedule_entry,
    backend_order,
    build_csv_rows,
    ensure_output_pair_available,
    estimated_single_stream_fps,
    file_metadata,
    image_index_for_iteration,
    latency_statistics,
    measure_latency_samples,
    measure_paired_backend_samples,
    measure_startup_load,
    onnx_speedup_over_pytorch,
    publish_report_pair_atomic,
    sample_latencies_ns,
    schedule_balance_summary,
    startup_single_measurement_result,
    subgroup_audit_statistics,
    validate_balanced_iteration_count,
    validate_consistency_preflight,
    validate_detection_classes,
    validate_raw_output,
    validate_runtime_provider,
)
from dashguard.inference.onnx_validation import CPU_PROVIDER


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


class FakeGc:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def isenabled(self) -> bool:
        return self.enabled

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True


def _write_preflight_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pytorch_model = tmp_path / "model.pt"
    onnx_model = tmp_path / "model.onnx"
    image = tmp_path / "image.jpg"
    pytorch_model.write_bytes(b"pytorch")
    onnx_model.write_bytes(b"onnx")
    image.write_bytes(b"image")
    report = {
        "schema_version": "test",
        "structural_validation": "passed",
        "numerical_acceptance": "passed",
        "models": {
            "pytorch": file_metadata(pytorch_model),
            "onnx": file_metadata(onnx_model),
        },
        "onnx_runtime": {
            "requested_providers": [CPU_PROVIDER],
            "actual_providers": [CPU_PROVIDER],
        },
        "onnx_io_metadata": {
            "inputs": [
                {"name": "images", "shape": [1, 3, 640, 640], "dtype": "tensor(float)"}
            ],
            "outputs": [
                {"name": "output0", "shape": [1, 84, 8400], "dtype": "tensor(float)"}
            ],
        },
        "imgsz": 640,
        "postprocessing": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "maximum_detections": 300,
            "agnostic_nms": False,
            "multi_label": False,
            "target_class_ids": [0, 1, 2, 3, 5, 7],
        },
        "images": [
            {
                **file_metadata(image),
                "detection_comparison": {
                    "pytorch_detections": [],
                    "onnx_detections": [],
                },
            }
        ],
    }
    consistency = tmp_path / "consistency.json"
    consistency.write_text(json.dumps(report), encoding="utf-8")
    return consistency, pytorch_model, onnx_model, image


def _mutate_json(path: Path, mutation: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_csv_row() -> dict[str, Any]:
    return {
        "stage": "preprocess",
        "backend": "shared",
        "count": 1,
        "minimum_ms": 1.0,
        "mean_ms": 1.0,
        "standard_deviation_ms": 0.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
        "maximum_ms": 1.0,
        "estimated_single_stream_fps_from_p50": "",
        "estimated_single_stream_fps_from_p95": "",
        "notes": "test",
    }


def _stage_results() -> dict[str, Any]:
    stats = latency_statistics([1_000_000])
    return {
        "startup_load": {BACKEND_PYTORCH: stats, BACKEND_ONNX: stats},
        "preprocess": {"shared": stats},
        "inference_only": {BACKEND_PYTORCH: stats, BACKEND_ONNX: stats},
        "postprocess": {"pytorch_output": stats, "onnx_output": stats},
        "image_to_detections": {BACKEND_PYTORCH: stats, BACKEND_ONNX: stats},
    }


def test_latency_statistics_p50_p95_and_p99_use_linear_interpolation() -> None:
    result = latency_statistics([1_000_000, 2_000_000, 3_000_000, 4_000_000])

    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["p99_ms"] == pytest.approx(3.97)
    assert result["latency_samples_ms"] == [1.0, 2.0, 3.0, 4.0]


def test_latency_statistics_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="At least one"):
        latency_statistics([])


@pytest.mark.parametrize("sample", [-1, float("nan"), float("inf"), -float("inf")])
def test_latency_statistics_rejects_negative_or_nonfinite_samples(sample: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        latency_statistics([sample])


def test_estimated_single_stream_fps_uses_latency_reciprocal() -> None:
    result = estimated_single_stream_fps({"p50_ms": 20.0, "p95_ms": 40.0})

    assert result == {
        "estimated_single_stream_fps_from_p50": 50.0,
        "estimated_single_stream_fps_from_p95": 25.0,
    }


def test_onnx_speedup_uses_pytorch_latency_divided_by_onnx_latency() -> None:
    result = onnx_speedup_over_pytorch(
        {"p50_ms": 20.0, "p95_ms": 30.0},
        {"p50_ms": 10.0, "p95_ms": 12.0},
    )

    assert result == {
        "onnx_speedup_over_pytorch_p50": 2.0,
        "onnx_speedup_over_pytorch_p95": 2.5,
    }


def test_four_step_schedule_is_exact() -> None:
    assert [balanced_schedule_entry(index) for index in range(4)] == [
        {
            "iteration_mod_4": 0,
            "image_index": 0,
            "backend_order": [BACKEND_PYTORCH, BACKEND_ONNX],
        },
        {
            "iteration_mod_4": 1,
            "image_index": 0,
            "backend_order": [BACKEND_ONNX, BACKEND_PYTORCH],
        },
        {
            "iteration_mod_4": 2,
            "image_index": 1,
            "backend_order": [BACKEND_PYTORCH, BACKEND_ONNX],
        },
        {
            "iteration_mod_4": 3,
            "image_index": 1,
            "backend_order": [BACKEND_ONNX, BACKEND_PYTORCH],
        },
    ]


def test_image_selection_is_independent_from_backend_order() -> None:
    assert [image_index_for_iteration(index, 2) for index in range(8)] == [
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
    ]
    assert [backend_order(index) for index in range(4)] == [
        (BACKEND_PYTORCH, BACKEND_ONNX),
        (BACKEND_ONNX, BACKEND_PYTORCH),
        (BACKEND_PYTORCH, BACKEND_ONNX),
        (BACKEND_ONNX, BACKEND_PYTORCH),
    ]


def test_warmup_calls_are_not_included_in_latency_samples() -> None:
    calls: list[int] = []
    samples = measure_latency_samples(
        lambda index: calls.append(index),
        image_count=2,
        warmup=2,
        iterations=3,
        clock_ns=FakeClock([0, 10, 20, 31, 40, 52]),
    )

    assert calls == [0, 0, 0, 0, 1]
    assert samples == [10, 11, 12]


def test_statistics_are_not_calculated_inside_timed_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "latency_statistics",
        lambda samples: pytest.fail("statistics entered timed measurement"),
    )

    samples = measure_latency_samples(
        lambda index: index,
        image_count=1,
        warmup=0,
        iterations=1,
        clock_ns=FakeClock([100, 107]),
    )

    assert samples == [7]


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_gc_state_is_restored_even_when_timed_action_fails(initially_enabled: bool) -> None:
    fake_gc = FakeGc(initially_enabled)

    with pytest.raises(RuntimeError, match="injected"):
        measure_latency_samples(
            lambda index: (_ for _ in ()).throw(RuntimeError("injected")),
            image_count=1,
            warmup=0,
            iterations=1,
            clock_ns=FakeClock([0]),
            gc_controller=fake_gc,
        )

    assert fake_gc.enabled is initially_enabled


@pytest.mark.parametrize("field", ["structural_validation", "numerical_acceptance"])
def test_consistency_failed_status_stops_preflight(tmp_path: Path, field: str) -> None:
    consistency, pytorch_model, onnx_model, image = _write_preflight_fixture(tmp_path)
    _mutate_json(consistency, lambda payload: payload.__setitem__(field, "failed"))

    with pytest.raises(ValueError, match=field):
        validate_consistency_preflight(consistency, pytorch_model, onnx_model, [image], 640)


def test_pytorch_hash_mismatch_stops_preflight(tmp_path: Path) -> None:
    consistency, pytorch_model, onnx_model, image = _write_preflight_fixture(tmp_path)
    pytorch_model.write_bytes(b"changed")

    with pytest.raises(ValueError, match="PyTorch model SHA-256"):
        validate_consistency_preflight(consistency, pytorch_model, onnx_model, [image], 640)


def test_onnx_hash_mismatch_stops_preflight(tmp_path: Path) -> None:
    consistency, pytorch_model, onnx_model, image = _write_preflight_fixture(tmp_path)
    onnx_model.write_bytes(b"changed")

    with pytest.raises(ValueError, match="ONNX model SHA-256"):
        validate_consistency_preflight(consistency, pytorch_model, onnx_model, [image], 640)


def test_input_hash_mismatch_stops_preflight(tmp_path: Path) -> None:
    consistency, pytorch_model, onnx_model, image = _write_preflight_fixture(tmp_path)
    image.write_bytes(b"changed")

    with pytest.raises(ValueError, match="input image SHA-256"):
        validate_consistency_preflight(consistency, pytorch_model, onnx_model, [image], 640)


def test_non_cpu_runtime_provider_stops_validation() -> None:
    with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
        validate_runtime_provider(["CUDAExecutionProvider", CPU_PROVIDER])


def test_output_shape_mismatch_stops_validation() -> None:
    with pytest.raises(ValueError, match="output shape mismatch"):
        validate_raw_output(np.zeros((1, 84, 1), dtype=np.float32), "fake")


def test_detection_count_or_class_postflight_mismatch_stops_validation() -> None:
    actual = [Detection((0.0, 0.0, 1.0, 1.0), 0.9, 2, "car")]
    expected = [{"class_id": 7, "class_name": "truck"}]

    with pytest.raises(ValueError, match="count/classes mismatch"):
        validate_detection_classes(actual, expected, "postflight")


def test_startup_load_is_separate_from_inference_measurement() -> None:
    events: list[str] = []
    backend_object, startup_ns = measure_startup_load(
        lambda: events.append("load") or object(),
        clock_ns=FakeClock([0, 5]),
    )
    samples = measure_latency_samples(
        lambda index: events.append("infer"),
        image_count=1,
        warmup=1,
        iterations=1,
        clock_ns=FakeClock([10, 17]),
    )

    assert backend_object is not None
    assert startup_ns == 5
    assert samples == [7]
    assert events == ["load", "infer", "infer"]


def test_existing_either_output_prevents_default_overwrite(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "result.csv"
    output_csv.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        ensure_output_pair_available(output_json, output_csv, overwrite=False)


def test_overwrite_atomically_replaces_both_outputs(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "result.csv"
    output_json.write_text("old-json", encoding="utf-8")
    output_csv.write_text("old-csv", encoding="utf-8")

    publish_report_pair_atomic(
        {"value": 1.0},
        [_valid_csv_row()],
        output_json,
        output_csv,
        overwrite=True,
    )

    assert json.loads(output_json.read_text(encoding="utf-8")) == {"value": 1.0}
    with output_csv.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream))[0]["stage"] == "preprocess"


def test_json_and_csv_publish_roll_back_together(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "result.csv"
    output_json.write_text("old-json", encoding="utf-8")
    output_csv.write_text("old-csv", encoding="utf-8")
    replace_calls = 0

    def fail_second_publish(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("injected CSV publish failure")
        benchmark.os.replace(source, destination)

    with pytest.raises(OSError, match="injected CSV"):
        publish_report_pair_atomic(
            {"value": 1.0},
            [_valid_csv_row()],
            output_json,
            output_csv,
            overwrite=True,
            replace=fail_second_publish,
        )

    assert output_json.read_text(encoding="utf-8") == "old-json"
    assert output_csv.read_text(encoding="utf-8") == "old-csv"
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.bak"))


def test_incomplete_rollback_preserves_recoverable_backup(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "result.csv"
    output_json.write_text("old-json", encoding="utf-8")
    output_csv.write_text("old-csv", encoding="utf-8")
    replace_calls = 0

    def fail_publish_and_one_restore(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls in {4, 5}:
            raise OSError(f"injected replace failure {replace_calls}")
        benchmark.os.replace(source, destination)

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        publish_report_pair_atomic(
            {"value": 1.0},
            [_valid_csv_row()],
            output_json,
            output_csv,
            overwrite=True,
            replace=fail_publish_and_one_restore,
        )

    backups = list(tmp_path.glob("*.bak"))
    assert not output_json.exists()
    assert output_csv.read_text(encoding="utf-8") == "old-csv"
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old-json"


def test_json_publish_rejects_nonfinite_numbers(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "result.csv"

    with pytest.raises(ValueError, match="non-finite"):
        publish_report_pair_atomic(
            {"bad": float("nan")},
            [_valid_csv_row()],
            output_json,
            output_csv,
            overwrite=False,
        )

    assert not output_json.exists()
    assert not output_csv.exists()


def test_csv_rows_have_exact_required_schema() -> None:
    rows = build_csv_rows(_stage_results())

    assert len(rows) == 9
    assert all(list(row.keys()) == CSV_COLUMNS for row in rows)
    assert [(row["stage"], row["backend"]) for row in rows] == [
        ("startup_load", BACKEND_PYTORCH),
        ("startup_load", BACKEND_ONNX),
        ("preprocess", "shared"),
        ("inference_only", BACKEND_PYTORCH),
        ("inference_only", BACKEND_ONNX),
        ("postprocess", "pytorch_output"),
        ("postprocess", "onnx_output"),
        ("image_to_detections", BACKEND_PYTORCH),
        ("image_to_detections", BACKEND_ONNX),
    ]


def test_derived_fps_is_named_as_estimate_not_throughput() -> None:
    derived = estimated_single_stream_fps({"p50_ms": 10.0, "p95_ms": 20.0})

    assert all("estimated_single_stream_fps" in key for key in derived)
    assert all("throughput" not in key for key in derived)


def test_paired_measurement_uses_balanced_cycle_for_warmup_and_timed() -> None:
    calls: list[tuple[str, int]] = []

    def action(backend_name: str) -> Any:
        return lambda image_index: calls.append((backend_name, image_index))

    samples = measure_paired_backend_samples(
        {BACKEND_PYTORCH: action(BACKEND_PYTORCH), BACKEND_ONNX: action(BACKEND_ONNX)},
        image_count=2,
        warmup=4,
        iterations=4,
        clock_ns=FakeClock(list(range(16))),
    )

    expected_cycle = [
        (BACKEND_PYTORCH, 0),
        (BACKEND_ONNX, 0),
        (BACKEND_ONNX, 0),
        (BACKEND_PYTORCH, 0),
        (BACKEND_PYTORCH, 1),
        (BACKEND_ONNX, 1),
        (BACKEND_ONNX, 1),
        (BACKEND_PYTORCH, 1),
    ]
    assert calls == expected_cycle + expected_cycle
    assert sample_latencies_ns(samples[BACKEND_PYTORCH]) == [1, 1, 1, 1]
    assert sample_latencies_ns(samples[BACKEND_ONNX]) == [1, 1, 1, 1]
    assert [record["execution_position"] for record in samples[BACKEND_PYTORCH]] == [
        "first",
        "second",
        "first",
        "second",
    ]


@pytest.mark.parametrize("iteration_count, cell_count", [(20, 5), (100, 25)])
def test_warmup_and_timed_schedules_are_exactly_balanced(
    iteration_count: int,
    cell_count: int,
) -> None:
    summary = schedule_balance_summary(iteration_count)

    assert summary["exactly_balanced"] is True
    assert summary["complete_four_step_cycles"] == cell_count
    for backend_counts in summary["by_backend_image_and_position"].values():
        for image_counts in backend_counts.values():
            assert image_counts == {"first": cell_count, "second": cell_count}


@pytest.mark.parametrize(
    "value,label",
    [(2, "warmup"), (22, "warmup"), (1, "iterations"), (101, "iterations")],
)
def test_incomplete_four_step_cycle_is_rejected(value: int, label: str) -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        validate_balanced_iteration_count(value, label)


def _audit_sample_records() -> list[dict[str, Any]]:
    cells = [
        (0, "first"),
        (0, "second"),
        (1, "first"),
        (1, "second"),
    ] * 2
    return [
        {
            "sample_index": index,
            "iteration": index,
            "image_index": image_index,
            "execution_position": position,
            "latency_ns": (index + 1) * 1_000_000,
        }
        for index, (image_index, position) in enumerate(cells)
    ]


def test_subgroup_audit_aggregates_images_and_execution_order() -> None:
    audit = subgroup_audit_statistics(_audit_sample_records())

    assert audit["by_image"]["image_0"] == {
        "count": 4,
        "mean_ms": 3.5,
        "p50_ms": 3.5,
        "p95_ms": pytest.approx(5.85),
    }
    assert audit["by_image"]["image_1"]["p50_ms"] == 5.5
    assert audit["by_execution_order"]["first"]["p50_ms"] == 4.0
    assert audit["by_execution_order"]["second"]["p50_ms"] == 5.0


def test_subgroup_audit_aggregates_joint_image_and_order_cells() -> None:
    joint = subgroup_audit_statistics(_audit_sample_records())[
        "by_image_and_execution_order"
    ]

    assert joint["image_0"]["first"] == {
        "count": 2,
        "mean_ms": 3.0,
        "p50_ms": 3.0,
        "p95_ms": pytest.approx(4.8),
    }
    assert joint["image_0"]["second"]["mean_ms"] == 4.0
    assert joint["image_1"]["first"]["mean_ms"] == 5.0
    assert joint["image_1"]["second"]["mean_ms"] == 6.0


def test_subgroup_counts_cover_every_sample_without_duplication() -> None:
    audit = subgroup_audit_statistics(_audit_sample_records())

    assert audit["integrity"] == {
        "overall_count": 8,
        "unique_sample_index_count": 8,
        "by_image_count_sum": 8,
        "by_execution_order_count_sum": 8,
        "by_image_and_execution_order_count_sum": 8,
        "duplicate_sample_count": 0,
        "missing_sample_count": 0,
        "status": "passed",
    }


def test_subgroup_duplicate_or_missing_sample_index_is_rejected() -> None:
    records = _audit_sample_records()
    records[-1]["sample_index"] = 0

    with pytest.raises(ValueError, match="duplicate or missing"):
        subgroup_audit_statistics(records)


def test_startup_result_is_explicitly_one_measurement() -> None:
    result = startup_single_measurement_result(12_500_000)

    assert result["measurement_type"] == "single_measurement"
    assert result["single_measurement_ms"] == 12.5
    assert result["count"] == 1
    assert result["interpretation_note"] == STARTUP_SINGLE_MEASUREMENT_NOTE


def test_startup_csv_note_restricts_percentile_interpretation() -> None:
    rows = build_csv_rows(_stage_results())
    startup_rows = [row for row in rows if row["stage"] == "startup_load"]

    assert len(startup_rows) == 2
    assert all(STARTUP_SINGLE_MEASUREMENT_NOTE in row["notes"] for row in startup_rows)


def test_publishing_balanced_outputs_does_not_modify_existing_outputs(
    tmp_path: Path,
) -> None:
    existing_json = tmp_path / "day7_cpu_backend_benchmark.json"
    existing_csv = tmp_path / "day7_cpu_backend_benchmark.csv"
    balanced_json = tmp_path / "day7_cpu_backend_benchmark_balanced.json"
    balanced_csv = tmp_path / "day7_cpu_backend_benchmark_balanced.csv"
    existing_json.write_bytes(b"existing-json")
    existing_csv.write_bytes(b"existing-csv")

    publish_report_pair_atomic(
        {"balanced": True},
        [_valid_csv_row()],
        balanced_json,
        balanced_csv,
        overwrite=False,
    )

    assert existing_json.read_bytes() == b"existing-json"
    assert existing_csv.read_bytes() == b"existing-csv"
    assert balanced_json.exists() and balanced_csv.exists()
