from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from dashguard.detection import (
    Detection,
    TARGET_CLASS_NAMES,
    UltralyticsDetector,
    convert_ultralytics_result,
)


class FakeTensor:
    """Small tensor stand-in that records conversion method order."""

    def __init__(self, values: Any) -> None:
        self.values = values
        self.calls: list[str] = []

    def detach(self) -> FakeTensor:
        self.calls.append("detach")
        return self

    def cpu(self) -> FakeTensor:
        self.calls.append("cpu")
        return self

    def tolist(self) -> list[Any]:
        self.calls.append("tolist")
        return list(self.values)


class FakeBoxes:
    def __init__(self, xyxy: Any, conf: Any, cls: Any) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None, names: Any) -> None:
        self.boxes = boxes
        self.names = names


def make_result(
    *,
    xyxy: Any = ((1, 2, 3, 4),),
    conf: Any = (0.8,),
    cls: Any = (0,),
    names: Any = None,
) -> FakeResult:
    if names is None:
        names = {0: "person"}
    return FakeResult(FakeBoxes(xyxy, conf, cls), names)


def test_detection_is_frozen_dataclass_with_required_field_order() -> None:
    detection = Detection((1.0, 2.0, 3.0, 4.0), 0.9, 2, "car")

    assert is_dataclass(Detection)
    assert [field.name for field in fields(Detection)] == [
        "xyxy",
        "confidence",
        "class_id",
        "class_name",
    ]
    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.5  # type: ignore[misc]


def test_target_class_names_are_exactly_the_six_required_names() -> None:
    assert isinstance(TARGET_CLASS_NAMES, frozenset)
    assert TARGET_CLASS_NAMES == {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    }


def test_conversion_supports_dict_names_mapping() -> None:
    result = make_result(cls=(17,), names={17: "truck"})

    assert convert_ultralytics_result(result) == [
        Detection((1.0, 2.0, 3.0, 4.0), 0.8, 17, "truck")
    ]


def test_conversion_supports_list_names_mapping() -> None:
    result = make_result(cls=(2,), names=["person", "cat", "bicycle"])

    assert convert_ultralytics_result(result) == [
        Detection((1.0, 2.0, 3.0, 4.0), 0.8, 2, "bicycle")
    ]


def test_conversion_filters_non_targets_and_preserves_input_order() -> None:
    result = make_result(
        xyxy=((1, 1, 2, 2), (3, 3, 4, 4), (5, 5, 6, 6)),
        conf=(0.91, 0.82, 0.73),
        cls=(4, 9, 1),
        names={1: "bus", 4: "car", 9: "dog"},
    )

    assert convert_ultralytics_result(result) == [
        Detection((1.0, 1.0, 2.0, 2.0), 0.91, 4, "car"),
        Detection((5.0, 5.0, 6.0, 6.0), 0.73, 1, "bus"),
    ]


def test_conversion_returns_empty_list_when_boxes_is_none() -> None:
    assert convert_ultralytics_result(FakeResult(None, {0: "person"})) == []


def test_conversion_returns_empty_list_for_empty_detection_arrays() -> None:
    assert convert_ultralytics_result(
        make_result(xyxy=(), conf=(), cls=())
    ) == []


def test_conversion_returns_python_numeric_types_and_uses_tensor_order() -> None:
    result = make_result(
        xyxy=((np.float32(1.25), np.float64(2.5), 3, 4),),
        conf=(np.float32(0.75),),
        cls=(np.int64(8),),
        names={8: "motorcycle"},
    )

    detection = convert_ultralytics_result(result)[0]

    assert all(type(value) is float for value in detection.xyxy)
    assert type(detection.confidence) is float
    assert type(detection.class_id) is int
    assert result.boxes is not None
    assert result.boxes.xyxy.calls == ["detach", "cpu", "tolist"]
    assert result.boxes.conf.calls == ["detach", "cpu", "tolist"]
    assert result.boxes.cls.calls == ["detach", "cpu", "tolist"]


@pytest.mark.parametrize(
    ("xyxy", "conf", "cls"),
    [
        (((1, 2, 3, 4),), (), (0,)),
        ((), (0.8,), (0,)),
        (((1, 2, 3, 4),), (0.8,), ()),
    ],
)
def test_conversion_rejects_mismatched_array_lengths(
    xyxy: Any,
    conf: Any,
    cls: Any,
) -> None:
    with pytest.raises(ValueError, match="array lengths differ"):
        convert_ultralytics_result(
            make_result(xyxy=xyxy, conf=conf, cls=cls)
        )


def test_detector_passes_expected_predict_arguments_and_returns_detections() -> None:
    result = make_result(cls=(11,), names={11: "car"})
    model = Mock()
    model.names = {99: "cat", 11: "car", 7: "person"}
    model.predict.return_value = [result]
    detector = UltralyticsDetector(
        model,
        confidence=0.4,
        imgsz=320,
        device="cpu",
    )
    image = np.zeros((8, 12, 3), dtype=np.uint8)

    detections = detector.detect(image)

    model.predict.assert_called_once_with(
        source=image,
        conf=0.4,
        imgsz=320,
        device="cpu",
        classes=[7, 11],
        verbose=False,
    )
    assert detections == [
        Detection((1.0, 2.0, 3.0, 4.0), 0.8, 11, "car")
    ]
    assert isinstance(detections, list)
    assert all(isinstance(item, Detection) for item in detections)


def test_detector_calculates_target_ids_from_list_model_names() -> None:
    model = Mock()
    model.names = ["dog", "truck", "cat", "person"]
    model.predict.return_value = [
        make_result(cls=(1,), names=model.names)
    ]
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    UltralyticsDetector(model).detect(image)

    assert model.predict.call_args.kwargs["classes"] == [1, 3]


@pytest.mark.parametrize("results", [[], [Mock(), Mock()]])
def test_detector_rejects_result_counts_other_than_one(
    results: list[Any],
) -> None:
    model = Mock()
    model.names = {0: "person"}
    model.predict.return_value = results

    with pytest.raises(RuntimeError, match="exactly one result"):
        UltralyticsDetector(model).detect(
            np.zeros((2, 2, 3), dtype=np.uint8)
        )


def test_injected_model_construction_does_not_import_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise AssertionError("Ultralytics must not be imported for injection")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    model = Mock()
    model.names = {0: "person"}

    UltralyticsDetector(model)
