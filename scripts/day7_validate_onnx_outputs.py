"""Compare Day 7 PyTorch and ONNX raw outputs on shared inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dashguard.inference.onnx_validation import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    validate_onnx_outputs,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the Day 7 raw-output validation CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure PyTorch and ONNX Runtime raw-output differences from "
            "one shared FP32 NCHW tensor per image."
        )
    )
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        type=Path,
        help="Input image path; repeat for multiple images.",
    )
    parser.add_argument("--pytorch-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("outputs/day7/models/yolo11n_fp32_opset20_640.onnx"),
    )
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument(
        "--confidence",
        type=_unit_float,
        default=DEFAULT_CONFIDENCE,
    )
    parser.add_argument(
        "--iou-threshold",
        type=_unit_float,
        default=DEFAULT_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--max-det",
        type=_positive_int,
        default=DEFAULT_MAX_DETECTIONS,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Run validation and print a compact audit summary."""

    try:
        report = validate_onnx_outputs(
            image_paths=args.image,
            pytorch_model_path=args.pytorch_model,
            onnx_model_path=args.onnx_model,
            imgsz=args.imgsz,
            output_json=args.output_json,
            overwrite=args.overwrite,
            confidence=args.confidence,
            iou_threshold=args.iou_threshold,
            max_det=args.max_det,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Validation failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    for image in report["images"]:
        raw_allclose = image["raw_allclose"]["passed"]
        comparison = image["detection_comparison"]
        minimum_iou = comparison["minimum_matched_iou"]
        minimum_iou_text = "none" if minimum_iou is None else f"{minimum_iou:.9g}"
        print(
            f"image={image['path']} "
            f"raw_allclose={str(raw_allclose).lower()} "
            f"detections_pt={comparison['pytorch_detection_count']} "
            f"detections_onnx={comparison['onnx_detection_count']} "
            f"matched={comparison['matched_count']} "
            f"unmatched_pt={len(comparison['unmatched_pytorch_indices'])} "
            f"unmatched_onnx={len(comparison['unmatched_onnx_indices'])} "
            f"min_iou={minimum_iou_text} "
            "max_bbox_diff="
            f"{comparison['maximum_bbox_coordinate_abs_diff']:.9g} "
            "max_conf_diff="
            f"{comparison['maximum_confidence_abs_diff']:.9g} "
            f"acceptance={comparison['detection_acceptance']}"
        )
    print(f"aggregate_acceptance={report['numerical_acceptance']}")
    print(f"json={args.output_json.resolve()}")
    return 0


def main() -> int:
    """Parse command-line arguments and run the consistency validation."""

    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
