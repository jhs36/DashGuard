from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from dashguard.detection import Detection
from dashguard.inference.onnx_validation import (
    CPU_PROVIDER,
    MAX_BBOX_COORDINATE_ABS_DIFF,
    MAX_CONFIDENCE_ABS_DIFF,
    MIN_MATCHED_BBOX_IOU,
    OnnxRuntimeRunner,
    bbox_coordinate_max_abs_diff,
    bbox_iou,
    calculate_difference_statistics,
    compare_backend_outputs,
    confidence_abs_diff,
    file_sha256,
    match_detections,
    preprocess_bgr_image,
    pytorch_output_to_numpy,
    raw_allclose_diagnostic,
    validate_onnx_outputs,
    write_json_atomic,
)


class FakeNode:
    def __init__(self, name: str, shape: list[int], dtype: str) -> None:
        self.name = name
        self.shape = shape
        self.type = dtype


class FakeSession:
    def __init__(
        self,
        output: np.ndarray,
        *,
        providers: list[str] | None = None,
        inputs: list[FakeNode] | None = None,
        outputs: list[FakeNode] | None = None,
    ) -> None:
        self.output = output
        self.providers = providers or [CPU_PROVIDER]
        self.inputs = inputs or [
            FakeNode("discovered_input", [1, 3, 4, 4], "tensor(float)")
        ]
        self.outputs = outputs or [
            FakeNode("discovered_output", list(output.shape), "tensor(float)")
        ]
        self.run_calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_providers(self) -> list[str]:
        return self.providers

    def get_inputs(self) -> list[FakeNode]:
        return self.inputs

    def get_outputs(self) -> list[FakeNode]:
        return self.outputs

    def run(
        self,
        output_names: list[str],
        feeds: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.run_calls.append((output_names, feeds))
        return [self.output]


class FakePytorchRunner:
    class_names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
    end2end = False

    def __init__(self, output: np.ndarray, fail: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.input_ids: list[int] = []

    def run(self, input_array: np.ndarray) -> np.ndarray:
        self.input_ids.append(id(input_array))
        if self.fail:
            raise RuntimeError("injected inference failure")
        return self.output


class FakeOnnxRunner:
    providers = [CPU_PROVIDER]

    def __init__(self, output: np.ndarray, imgsz: int) -> None:
        self.output = output
        self.input_ids: list[int] = []
        self.input_metadata = {
            "name": "images",
            "shape": [1, 3, imgsz, imgsz],
            "dtype": "tensor(float)",
        }
        self.output_metadata = {
            "name": "output0",
            "shape": list(output.shape),
            "dtype": "tensor(float)",
        }

    def run(self, input_array: np.ndarray) -> np.ndarray:
        self.input_ids.append(id(input_array))
        return self.output


def _write_validation_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "image.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((5, 9, 3), 127, dtype=np.uint8),
    )
    pytorch_model = tmp_path / "model.pt"
    onnx_model = tmp_path / "model.onnx"
    pytorch_model.write_bytes(b"pytorch-model")
    onnx_model.write_bytes(b"onnx-model")
    return image_path, pytorch_model, onnx_model


def _one_person_raw_output() -> np.ndarray:
    output = np.zeros((1, 84, 1), dtype=np.float32)
    output[0, :4, 0] = [4.0, 4.0, 2.0, 2.0]
    output[0, 4, 0] = 0.9
    return output


def _detection(
    xyxy: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    confidence: float = 0.9,
    class_id: int = 2,
    class_name: str = "car",
) -> Detection:
    return Detection(xyxy, confidence, class_id, class_name)


def test_preprocessing_produces_fixed_contiguous_fp32_nchw_range() -> None:
    image = np.zeros((5, 9, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    result = preprocess_bgr_image(image, imgsz=640)

    assert result.shape == (1, 3, 640, 640)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
    assert float(result[0, 2].max()) == 1.0  # BGR blue becomes RGB channel 2.


def test_exact_same_preprocessed_array_object_reaches_both_backends() -> None:
    input_array = np.zeros((1, 3, 4, 4), dtype=np.float32)
    output = np.zeros((1, 2, 3), dtype=np.float32)
    received_ids: list[int] = []

    def forward(shared: np.ndarray) -> np.ndarray:
        received_ids.append(id(shared))
        return output

    compare_backend_outputs(input_array, forward, forward)

    assert received_ids == [id(input_array), id(input_array)]


@pytest.mark.parametrize("container", [tuple, list])
def test_pytorch_tuple_or_list_uses_first_tensor(container: Any) -> None:
    first = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    second = torch.tensor([[[99.0]]], dtype=torch.float32)

    result = pytorch_output_to_numpy(container((first, second)))

    assert result.dtype == np.float32
    assert np.array_equal(result, first.numpy())


def test_identical_arrays_have_zero_difference_statistics() -> None:
    output = np.arange(12, dtype=np.float32).reshape(1, 3, 4)

    statistics = calculate_difference_statistics(output, output.copy())

    scalar_statistics = {
        key: value
        for key, value in statistics.items()
        if key != "max_difference"
    }
    assert scalar_statistics
    assert all(value == 0.0 for value in scalar_statistics.values())
    assert statistics["max_difference"]["flat_index"] == 0


def test_small_perturbation_statistics_are_measured_exactly() -> None:
    pytorch_output = np.zeros((1, 1, 4), dtype=np.float32)
    onnx_output = np.array([[[0.0, 1.0, 2.0, 3.0]]], dtype=np.float32)

    statistics = calculate_difference_statistics(pytorch_output, onnx_output)

    assert statistics["max_abs_diff"] == 3.0
    assert statistics["mean_abs_diff"] == 1.5
    assert statistics["rmse"] == pytest.approx(math.sqrt(3.5))
    assert statistics["p50_abs_diff"] == 1.5
    assert statistics["p95_abs_diff"] == pytest.approx(2.85)
    assert statistics["p99_abs_diff"] == pytest.approx(2.97)
    assert statistics["fraction_abs_diff_gt_1e-5"] == 0.75
    assert statistics["max_difference"] == {
        "flat_index": 3,
        "index": [0, 0, 3],
        "tensor_index": [0, 0, 3],
        "pytorch_value": 0.0,
        "onnx_value": 3.0,
        "signed_pytorch_minus_onnx": -3.0,
    }


def test_shape_mismatch_raises_before_difference_calculation() -> None:
    input_array = np.zeros((1, 3, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Output shapes differ"):
        compare_backend_outputs(
            input_array,
            lambda _: np.zeros((1, 2, 3), dtype=np.float32),
            lambda _: np.zeros((1, 3, 2), dtype=np.float32),
        )


@pytest.mark.parametrize("backend", ["pytorch", "onnx"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_backend_output_is_rejected(
    backend: str,
    bad_value: float,
) -> None:
    input_array = np.zeros((1, 3, 4, 4), dtype=np.float32)
    finite = np.zeros((1, 2, 3), dtype=np.float32)
    invalid = finite.copy()
    invalid[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN or infinity"):
        compare_backend_outputs(
            input_array,
            lambda _: invalid if backend == "pytorch" else finite,
            lambda _: invalid if backend == "onnx" else finite,
        )


def test_onnx_runner_uses_discovered_names_and_same_input_array() -> None:
    expected_output = np.zeros((1, 2, 3), dtype=np.float32)
    session = FakeSession(expected_output)
    runner = OnnxRuntimeRunner("unused.onnx", session=session)
    input_array = np.zeros((1, 3, 4, 4), dtype=np.float32)

    result = runner.run(input_array)

    assert result is expected_output
    assert runner.providers == [CPU_PROVIDER]
    assert runner.input_metadata == {
        "name": "discovered_input",
        "shape": [1, 3, 4, 4],
        "dtype": "tensor(float)",
    }
    output_names, feeds = session.run_calls[0]
    assert output_names == ["discovered_output"]
    assert feeds["discovered_input"] is input_array


def test_onnx_runner_rejects_non_cpu_provider_or_multiple_outputs() -> None:
    output = np.zeros((1, 2, 3), dtype=np.float32)

    with pytest.raises(RuntimeError, match="only CPUExecutionProvider"):
        OnnxRuntimeRunner(
            "unused.onnx",
            session=FakeSession(output, providers=["OtherProvider", CPU_PROVIDER]),
        )
    with pytest.raises(ValueError, match="exactly one output"):
        OnnxRuntimeRunner(
            "unused.onnx",
            session=FakeSession(
                output,
                outputs=[
                    FakeNode("first", [1, 2, 3], "tensor(float)"),
                    FakeNode("second", [1, 2, 3], "tensor(float)"),
                ],
            ),
        )


def test_existing_json_is_not_overwritten_by_default(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    output_path.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_json_atomic({"new": True}, output_path)

    assert output_path.read_text(encoding="utf-8") == "original"


def test_overwrite_atomically_replaces_existing_json(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    output_path.write_text("original", encoding="utf-8")

    write_json_atomic({"new": True}, output_path, overwrite=True)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"new": True}


def test_serialization_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"

    with pytest.raises(TypeError):
        write_json_atomic({"not_serializable": object()}, output_path)

    assert not output_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_validation_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    image, pytorch_model, onnx_model = _write_validation_inputs(tmp_path)
    output_path = tmp_path / "report.json"
    output = np.zeros((1, 2, 3), dtype=np.float32)

    with pytest.raises(RuntimeError, match="injected inference failure"):
        validate_onnx_outputs(
            [image],
            pytorch_model,
            onnx_model,
            8,
            output_path,
            pytorch_runner=FakePytorchRunner(output, fail=True),
            onnx_runner=FakeOnnxRunner(output, 8),
        )

    assert not output_path.exists()


def test_file_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "content.bin"
    path.write_bytes(b"DashGuard\x00Day7")

    first = file_sha256(path, chunk_size=1)
    second = file_sha256(path, chunk_size=1024)

    assert first == second
    assert first == "4839282a34cc572df6a6a8fe3d42a700ee7e02ad86fa573ffb1db6819bada611"


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_json_values_are_never_written(
    tmp_path: Path,
    bad_value: float,
) -> None:
    output_path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_atomic({"bad": bad_value}, output_path)

    assert not output_path.exists()


def test_validation_report_records_structure_and_shared_input(tmp_path: Path) -> None:
    image, pytorch_model, onnx_model = _write_validation_inputs(tmp_path)
    output_path = tmp_path / "report.json"
    output = _one_person_raw_output()
    pytorch_runner = FakePytorchRunner(output)
    onnx_runner = FakeOnnxRunner(output.copy(), 8)

    report = validate_onnx_outputs(
        [image],
        pytorch_model,
        onnx_model,
        8,
        output_path,
        pytorch_runner=pytorch_runner,
        onnx_runner=onnx_runner,
    )

    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed == report
    assert report["structural_validation"] == "passed"
    assert report["numerical_acceptance"] == "passed"
    assert "passed" not in report
    assert report["images"][0]["preprocessed"]["shape"] == [1, 3, 8, 8]
    assert report["images"][0]["pytorch_output"]["shape"] == [1, 84, 1]
    assert report["aggregate"]["element_count"] == 84
    assert pytorch_runner.input_ids == onnx_runner.input_ids


def test_identical_detection_lists_match_completely() -> None:
    detections = [_detection(), _detection(class_id=0, class_name="person")]

    result = match_detections(detections, detections)

    assert result["matched_count"] == 2
    assert result["unmatched_pytorch_indices"] == []
    assert result["unmatched_onnx_indices"] == []
    assert all(match["passed"] for match in result["matches"])
    assert result["detection_acceptance"] == "passed"


def test_detection_order_does_not_affect_matching() -> None:
    car = _detection()
    person = _detection(
        xyxy=(20.0, 20.0, 30.0, 30.0),
        class_id=0,
        class_name="person",
    )

    result = match_detections([car, person], [person, car])

    assert [
        (match["pytorch_index"], match["onnx_index"])
        for match in result["matches"]
    ] == [(0, 1), (1, 0)]
    assert result["detection_acceptance"] == "passed"


def test_different_classes_are_not_matching_candidates() -> None:
    result = match_detections(
        [_detection(class_id=2, class_name="car")],
        [_detection(class_id=7, class_name="truck")],
    )

    assert result["matched_count"] == 0
    assert result["unmatched_pytorch_indices"] == [0]
    assert result["unmatched_onnx_indices"] == [0]
    assert result["detection_acceptance"] == "failed"


def test_one_detection_cannot_be_matched_more_than_once() -> None:
    result = match_detections(
        [_detection()],
        [_detection(), _detection(xyxy=(1.0, 1.0, 11.0, 11.0))],
    )

    assert result["matched_count"] == 1
    assert result["matches"][0]["onnx_index"] == 0
    assert result["unmatched_onnx_indices"] == [1]


def test_equal_iou_ties_use_pytorch_then_onnx_indices() -> None:
    duplicated = [_detection(), _detection()]

    result = match_detections(duplicated, duplicated)

    assert [
        (match["pytorch_index"], match["onnx_index"])
        for match in result["matches"]
    ] == [(0, 0), (1, 1)]


def test_bbox_iou_uses_continuous_xyxy_geometry() -> None:
    left = _detection(xyxy=(0.0, 0.0, 10.0, 10.0))
    right = _detection(xyxy=(5.0, 0.0, 15.0, 10.0))

    assert bbox_iou(left, right) == pytest.approx(1.0 / 3.0)


def test_bbox_coordinate_difference_is_maximum_corresponding_delta() -> None:
    left = _detection(xyxy=(1.0, 2.0, 3.0, 4.0))
    right = _detection(xyxy=(1.1, 1.5, 3.25, 4.0))

    assert bbox_coordinate_max_abs_diff(left, right) == 0.5


def test_confidence_difference_is_absolute() -> None:
    assert confidence_abs_diff(
        _detection(confidence=0.8),
        _detection(confidence=0.75),
    ) == pytest.approx(0.05)


def test_unmatched_pytorch_indices_are_recorded() -> None:
    result = match_detections([_detection()], [])

    assert result["unmatched_pytorch_indices"] == [0]
    assert result["unmatched_onnx_indices"] == []


def test_unmatched_onnx_indices_are_recorded() -> None:
    result = match_detections([], [_detection()])

    assert result["unmatched_pytorch_indices"] == []
    assert result["unmatched_onnx_indices"] == [0]


def test_detection_count_difference_fails_acceptance() -> None:
    result = match_detections([_detection()], [_detection(), _detection()])

    assert result["detection_count_equal"] is False
    assert result["detection_acceptance"] == "failed"


def test_iou_below_acceptance_threshold_fails() -> None:
    result = match_detections(
        [_detection()],
        [_detection(xyxy=(0.1, 0.0, 10.1, 10.0))],
    )

    match = result["matches"][0]
    assert match["bbox_iou"] < MIN_MATCHED_BBOX_IOU
    assert match["criteria"]["bbox_iou_at_least_0_995"] is False
    assert result["detection_acceptance"] == "failed"


def test_bbox_difference_above_acceptance_threshold_fails() -> None:
    result = match_detections(
        [_detection(xyxy=(0.0, 0.0, 1000.0, 1000.0))],
        [_detection(xyxy=(0.3, 0.0, 1000.3, 1000.0))],
    )

    match = result["matches"][0]
    assert match["bbox_iou"] >= MIN_MATCHED_BBOX_IOU
    assert (
        match["bbox_coordinate_max_abs_diff"]
        > MAX_BBOX_COORDINATE_ABS_DIFF
    )
    assert result["detection_acceptance"] == "failed"


def test_confidence_difference_above_acceptance_threshold_fails() -> None:
    result = match_detections(
        [_detection(confidence=0.9)],
        [_detection(confidence=0.9 - 0.0011)],
    )

    match = result["matches"][0]
    assert match["confidence_abs_diff"] > MAX_CONFIDENCE_ABS_DIFF
    assert result["detection_acceptance"] == "failed"


def test_all_detection_acceptance_boundary_values_pass() -> None:
    result = match_detections(
        [
            _detection(
                xyxy=(0.0, 0.0, 1000.0, 1000.0),
                confidence=MAX_CONFIDENCE_ABS_DIFF,
            )
        ],
        [
            _detection(
                xyxy=(
                    MAX_BBOX_COORDINATE_ABS_DIFF,
                    0.0,
                    1000.0 + MAX_BBOX_COORDINATE_ABS_DIFF,
                    1000.0,
                ),
                confidence=0.0,
            )
        ],
    )

    match = result["matches"][0]
    assert match["bbox_coordinate_max_abs_diff"] == 0.25
    assert match["confidence_abs_diff"] == 0.001
    assert match["bbox_iou"] >= 0.995
    assert match["passed"] is True
    assert result["detection_acceptance"] == "passed"


def test_raw_allclose_result_records_required_numpy_settings() -> None:
    output = np.zeros((1, 2, 3), dtype=np.float32)

    passed = raw_allclose_diagnostic(output, output.copy())
    failed_output = output.copy()
    failed_output[0, 0, 0] = 0.001
    failed = raw_allclose_diagnostic(output, failed_output)

    assert passed == {
        "passed": True,
        "rtol": 1e-4,
        "atol": 1e-4,
        "equal_nan": False,
        "role": "diagnostic_only",
    }
    assert failed["passed"] is False


def test_raw_allclose_failure_does_not_fail_detection_acceptance(
    tmp_path: Path,
) -> None:
    image, pytorch_model, onnx_model = _write_validation_inputs(tmp_path)
    pytorch_output = _one_person_raw_output()
    onnx_output = pytorch_output.copy()
    onnx_output[0, 10, 0] = 0.001

    report = validate_onnx_outputs(
        [image],
        pytorch_model,
        onnx_model,
        8,
        tmp_path / "report.json",
        pytorch_runner=FakePytorchRunner(pytorch_output),
        onnx_runner=FakeOnnxRunner(onnx_output, 8),
    )

    assert report["images"][0]["raw_allclose"]["passed"] is False
    assert report["images"][0]["detection_acceptance"] == "passed"
    assert report["aggregate"]["raw_allclose_passed_image_count"] == 0
    assert report["numerical_acceptance"] == "passed"


def test_aggregate_detection_failure_sets_top_level_acceptance(
    tmp_path: Path,
) -> None:
    image, pytorch_model, onnx_model = _write_validation_inputs(tmp_path)
    pytorch_output = _one_person_raw_output()
    onnx_output = pytorch_output.copy()
    onnx_output[0, 4, 0] = 0.89

    report = validate_onnx_outputs(
        [image],
        pytorch_model,
        onnx_model,
        8,
        tmp_path / "report.json",
        pytorch_runner=FakePytorchRunner(pytorch_output),
        onnx_runner=FakeOnnxRunner(onnx_output, 8),
    )

    assert report["images"][0]["detection_acceptance"] == "failed"
    assert report["aggregate"]["detection_acceptance"] == "failed"
    assert report["numerical_acceptance"] == "failed"


def test_json_schema_records_fixed_criteria_and_postprocessing(
    tmp_path: Path,
) -> None:
    image, pytorch_model, onnx_model = _write_validation_inputs(tmp_path)
    output = _one_person_raw_output()

    report = validate_onnx_outputs(
        [image],
        pytorch_model,
        onnx_model,
        8,
        tmp_path / "report.json",
        pytorch_runner=FakePytorchRunner(output),
        onnx_runner=FakeOnnxRunner(output.copy(), 8),
    )

    criteria = report["acceptance_criteria"]
    assert report["schema_version"] == "dashguard.onnx_backend_consistency.v2"
    assert criteria["raw_diagnostic"]["determines_detection_acceptance"] is False
    assert criteria["detection"]["minimum_bbox_iou"] == 0.995
    assert criteria["detection"]["maximum_bbox_coordinate_abs_diff_pixels"] == 0.25
    assert criteria["detection"]["maximum_confidence_abs_diff"] == 0.001
    assert report["postprocessing"]["iou_threshold"] == 0.7
    assert report["postprocessing"]["maximum_detections"] == 300
    assert report["postprocessing"]["agnostic_nms"] is False
    assert report["postprocessing"]["target_class_names"] == [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    ]


def test_aggregate_max_difference_has_image_and_tensor_provenance(
    tmp_path: Path,
) -> None:
    image, pytorch_model, onnx_model = _write_validation_inputs(tmp_path)
    pytorch_output = _one_person_raw_output()
    onnx_output = pytorch_output.copy()
    onnx_output[0, 10, 0] = 0.001

    report = validate_onnx_outputs(
        [image],
        pytorch_model,
        onnx_model,
        8,
        tmp_path / "report.json",
        pytorch_runner=FakePytorchRunner(pytorch_output),
        onnx_runner=FakeOnnxRunner(onnx_output, 8),
    )

    per_image_maximum = report["images"][0]["difference"]["max_difference"]
    aggregate_maximum = report["aggregate"]["difference"]["max_difference"]
    assert per_image_maximum["tensor_index"] == [0, 10, 0]
    assert aggregate_maximum["image_index"] == 0
    assert aggregate_maximum["tensor_index"] == [0, 10, 0]
    assert aggregate_maximum["concatenated_flat_index"] == 10
