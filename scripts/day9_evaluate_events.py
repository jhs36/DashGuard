"""Evaluate Day 6/8 event predictions against Day 9 human labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dashguard.evaluation import (
    PRIMARY_IOU_THRESHOLD,
    SENSITIVITY_IOU_THRESHOLD,
    EvaluationValidationError,
    build_evaluation_report,
    load_event_label_scope,
    load_event_labels,
    load_event_predictions,
    validate_label_scope_predictions,
    write_evaluation_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the Day 9 temporal evaluation command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Match fixed-schema human event labels to Day 6/8 events.json "
            "intervals and atomically write primary and sensitivity metrics."
        )
    )
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--label-scope-csv", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--prediction",
        action="append",
        nargs=2,
        metavar=("VIDEO_ID", "EVENTS_JSON"),
        help=(
            "Explicit video ID and its events.json; repeat for multiple videos."
        ),
    )
    inputs.add_argument(
        "--manifest",
        type=Path,
        help=(
            "JSON manifest with a predictions array of video_id/events_json objects."
        ),
    )
    parser.add_argument(
        "--primary-threshold",
        default=PRIMARY_IOU_THRESHOLD,
        type=float,
    )
    parser.add_argument(
        "--sensitivity-threshold",
        default=SENSITIVITY_IOU_THRESHOLD,
        type=float,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output report.",
    )
    return parser


def _manifest_predictions(path: Path) -> list[tuple[str, Path]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload: Any = json.load(stream)
    except json.JSONDecodeError as exc:
        raise EvaluationValidationError(
            f"invalid prediction manifest JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("predictions"), list
    ):
        raise EvaluationValidationError(
            "prediction manifest must contain a predictions array"
        )
    specifications: list[tuple[str, Path]] = []
    for index, item in enumerate(payload["predictions"]):
        if not isinstance(item, dict):
            raise EvaluationValidationError(
                f"manifest predictions[{index}] must be an object"
            )
        unexpected = set(item) - {"video_id", "events_json"}
        missing = {"video_id", "events_json"} - set(item)
        if unexpected or missing:
            raise EvaluationValidationError(
                f"manifest predictions[{index}] must contain exactly "
                "video_id and events_json"
            )
        video_id = item["video_id"]
        events_json = item["events_json"]
        if not isinstance(video_id, str) or not video_id.strip():
            raise EvaluationValidationError(
                f"manifest predictions[{index}].video_id must be non-empty"
            )
        if not isinstance(events_json, str) or not events_json.strip():
            raise EvaluationValidationError(
                f"manifest predictions[{index}].events_json must be non-empty"
            )
        events_path = Path(events_json)
        if not events_path.is_absolute():
            events_path = path.parent / events_path
        specifications.append((video_id, events_path))
    if not specifications:
        raise EvaluationValidationError(
            "prediction manifest must declare at least one events.json"
        )
    return specifications


def _prediction_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.manifest is not None:
        return _manifest_predictions(args.manifest)
    return [(video_id, Path(path)) for video_id, path in args.prediction]


def _protect_inputs(
    output_path: Path,
    labels_path: Path,
    label_scope_path: Path,
    prediction_paths: Sequence[Path],
    manifest_path: Path | None,
) -> None:
    protected = {labels_path.resolve(), label_scope_path.resolve()}
    protected.update(path.resolve() for path in prediction_paths)
    if manifest_path is not None:
        protected.add(manifest_path.resolve())
    if output_path.resolve() in protected:
        raise EvaluationValidationError("output JSON must not replace an input file")


def run(args: argparse.Namespace) -> int:
    """Load, evaluate, write, and report a concise summary."""

    try:
        specifications = _prediction_specs(args)
        prediction_paths = [path for _, path in specifications]
        _protect_inputs(
            args.output_json,
            args.labels_csv,
            args.label_scope_csv,
            prediction_paths,
            args.manifest,
        )
        label_scope = load_event_label_scope(args.label_scope_csv)
        predictions = tuple(
            load_event_predictions(path, video_id=video_id)
            for video_id, path in specifications
        )
        validate_label_scope_predictions(label_scope, predictions)
        labels = load_event_labels(
            args.labels_csv,
            video_durations_sec=dict(
                label_scope.video_durations_sec_by_video
            ),
        )
        report = build_evaluation_report(
            labels,
            predictions,
            label_scope=label_scope,
            primary_threshold=args.primary_threshold,
            sensitivity_threshold=args.sensitivity_threshold,
        )
        written_path = write_evaluation_report(
            report,
            args.output_json,
            overwrite=args.overwrite,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"Event evaluation failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "output_json": str(written_path.resolve()),
                "thresholds": report["thresholds"],
                "micro": report["micro"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run Day 9 temporal event evaluation."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
