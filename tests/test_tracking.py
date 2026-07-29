from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from dashguard.tracking import (
    Track,
    UltralyticsByteTracker,
    convert_ultralytics_result,
)


class FakeTensor:
    """Tensor stand-in that records conversion method order."""

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
    def __init__(
        self,
        xyxy: Any,
        conf: Any,
        cls: Any,
        track_ids: Any,
    ) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)
        self.id = (
            None if track_ids is None else FakeTensor(track_ids)
        )


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None, names: Any) -> None:
        self.boxes = boxes
        self.names = names


def make_result(
    *,
    xyxy: Any = ((1, 2, 3, 4),),
    conf: Any = (0.8,),
    cls: Any = (0,),
    track_ids: Any = (1,),
    names: Any = None,
) -> FakeResult:
    if names is None:
        names = {0: "person"}
    return FakeResult(
        FakeBoxes(xyxy, conf, cls, track_ids),
        names,
    )


def convert(
    result: FakeResult,
    *,
    frame_index: int = 4,
    timestamp_sec: float = 0.5,
) -> list[Track]:
    return convert_ultralytics_result(
        result,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
    )


def test_track_is_frozen_dataclass_with_required_field_order() -> None:
    track = Track(7, (1.0, 2.0, 3.0, 4.0), 0.9, "car", 3, 0.1)

    assert is_dataclass(Track)
    assert [field.name for field in fields(Track)] == [
        "track_id",
        "xyxy",
        "confidence",
        "class_name",
        "frame_index",
        "timestamp_sec",
    ]
    with pytest.raises(FrozenInstanceError):
        track.confidence = 0.5  # type: ignore[misc]


def test_conversion_returns_empty_list_when_boxes_is_none() -> None:
    assert convert(FakeResult(None, {0: "person"})) == []


def test_empty_detection_arrays_with_none_id_return_empty_list() -> None:
    assert convert(
        make_result(xyxy=(), conf=(), cls=(), track_ids=None)
    ) == []


def test_detections_without_tracking_ids_raise_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="boxes.id is None"):
        convert(make_result(track_ids=None))


def test_conversion_supports_dict_names() -> None:
    result = make_result(cls=(17,), track_ids=(8,), names={17: "truck"})

    assert convert(result) == [
        Track(8, (1.0, 2.0, 3.0, 4.0), 0.8, "truck", 4, 0.5)
    ]


def test_conversion_supports_list_names() -> None:
    result = make_result(
        cls=(2,),
        track_ids=(9,),
        names=["person", "cat", "bicycle"],
    )

    assert convert(result) == [
        Track(9, (1.0, 2.0, 3.0, 4.0), 0.8, "bicycle", 4, 0.5)
    ]


def test_conversion_filters_non_targets_and_preserves_order() -> None:
    result = make_result(
        xyxy=((1, 1, 2, 2), (3, 3, 4, 4), (5, 5, 6, 6)),
        conf=(0.91, 0.82, 0.73),
        cls=(4, 9, 1),
        track_ids=(12, 13, 14),
        names={1: "bus", 4: "car", 9: "dog"},
    )

    assert convert(result) == [
        Track(12, (1.0, 1.0, 2.0, 2.0), 0.91, "car", 4, 0.5),
        Track(14, (5.0, 5.0, 6.0, 6.0), 0.73, "bus", 4, 0.5),
    ]


def test_conversion_returns_python_numbers_and_uses_tensor_order() -> None:
    result = make_result(
        xyxy=((np.float32(1.25), np.float64(2.5), 3, 4),),
        conf=(np.float32(0.75),),
        cls=(np.int64(8),),
        track_ids=(np.int64(23),),
        names={8: "motorcycle"},
    )

    track = convert(result, timestamp_sec=np.float32(1.25))[0]

    assert all(type(value) is float for value in track.xyxy)
    assert type(track.confidence) is float
    assert type(track.track_id) is int
    assert type(track.timestamp_sec) is float
    assert result.boxes is not None
    assert result.boxes.xyxy.calls == ["detach", "cpu", "tolist"]
    assert result.boxes.conf.calls == ["detach", "cpu", "tolist"]
    assert result.boxes.cls.calls == ["detach", "cpu", "tolist"]
    assert result.boxes.id is not None
    assert result.boxes.id.calls == ["detach", "cpu", "tolist"]


@pytest.mark.parametrize(
    ("xyxy", "conf", "cls", "track_ids"),
    [
        (((1, 2, 3, 4),), (), (0,), (1,)),
        ((), (0.8,), (0,), (1,)),
        (((1, 2, 3, 4),), (0.8,), (), (1,)),
        (((1, 2, 3, 4),), (0.8,), (0,), ()),
    ],
)
def test_conversion_rejects_mismatched_array_lengths(
    xyxy: Any,
    conf: Any,
    cls: Any,
    track_ids: Any,
) -> None:
    with pytest.raises(ValueError, match="array lengths differ"):
        convert(
            make_result(
                xyxy=xyxy,
                conf=conf,
                cls=cls,
                track_ids=track_ids,
            )
        )


def test_conversion_rejects_bbox_without_four_coordinates() -> None:
    with pytest.raises(ValueError, match="exactly four coordinates"):
        convert(make_result(xyxy=((1, 2, 3),)))


@pytest.mark.parametrize("track_id", [0, -3])
def test_conversion_rejects_non_positive_track_id(track_id: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        convert(make_result(track_ids=(track_id,)))


def test_conversion_preserves_frame_index_and_timestamp() -> None:
    track = convert(
        make_result(track_ids=(11,)),
        frame_index=321,
        timestamp_sec=10.75,
    )[0]

    assert track.frame_index == 321
    assert track.timestamp_sec == 10.75


def test_tracker_passes_exact_track_arguments_and_returns_tracks() -> None:
    result = make_result(cls=(11,), track_ids=(42,), names={11: "car"})
    model = Mock()
    model.names = {99: "cat", 11: "car", 7: "person"}
    model.track.return_value = [result]
    tracker = UltralyticsByteTracker(
        model,
        confidence=0.4,
        imgsz=320,
        device="cpu",
    )
    image = np.zeros((8, 12, 3), dtype=np.uint8)

    tracks = tracker.track(image, frame_index=9, timestamp_sec=0.3)

    model.track.assert_called_once_with(
        source=image,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        imgsz=320,
        device="cpu",
        classes=[7, 11],
        verbose=False,
    )
    assert tracks == [
        Track(42, (1.0, 2.0, 3.0, 4.0), 0.8, "car", 9, 0.3)
    ]


def test_tracker_calculates_target_ids_from_list_model_names() -> None:
    model = Mock()
    model.names = ["dog", "truck", "cat", "person"]
    model.track.return_value = [
        make_result(cls=(1,), track_ids=(2,), names=model.names)
    ]

    UltralyticsByteTracker(model).track(
        np.zeros((2, 2, 3), dtype=np.uint8),
        frame_index=0,
        timestamp_sec=0.0,
    )

    assert model.track.call_args.kwargs["classes"] == [1, 3]


@pytest.mark.parametrize("results", [[], [Mock(), Mock()]])
def test_tracker_rejects_result_counts_other_than_one(
    results: list[Any],
) -> None:
    model = Mock()
    model.names = {0: "person"}
    model.track.return_value = results

    with pytest.raises(RuntimeError, match="exactly one result"):
        UltralyticsByteTracker(model).track(
            np.zeros((2, 2, 3), dtype=np.uint8),
            frame_index=0,
            timestamp_sec=0.0,
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

    UltralyticsByteTracker(model)
