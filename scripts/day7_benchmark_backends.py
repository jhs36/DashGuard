"""Benchmark Day 7 PyTorch CPU and ONNX Runtime CPU latency boundaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dashguard.inference.benchmark import (
    BACKEND_ONNX,
    BACKEND_PYTORCH,
    run_cpu_backend_benchmark,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the balanced Day 7 CPU benchmark command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark output-consistent PyTorch CPU and ONNX Runtime CPU "
            "startup, preprocess, inference, postprocess, and end-to-end latency."
        )
    )
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        type=Path,
        help="Input image path; provide exactly twice for the balanced schedule.",
    )
    parser.add_argument("--pytorch-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("outputs/day7/models/yolo11n_fp32_opset20_640.onnx"),
    )
    parser.add_argument(
        "--consistency-json",
        type=Path,
        default=Path("outputs/day7/validation/day7_backend_consistency.json"),
    )
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--opencv-threads", type=_positive_int, default=1)
    parser.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=20,
        help="Warmup count per stage; must be a multiple of 4.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=100,
        help="Timed count per stage/backend; must be a multiple of 4.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/day7/benchmarks/day7_cpu_backend_benchmark_balanced.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/day7/benchmarks/day7_cpu_backend_benchmark_balanced.csv"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace both balanced output files if they exist.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the benchmark and print only the primary latency summary."""

    try:
        report = run_cpu_backend_benchmark(
            image_paths=args.image,
            pytorch_model_path=args.pytorch_model,
            onnx_model_path=args.onnx_model,
            consistency_json_path=args.consistency_json,
            imgsz=args.imgsz,
            threads=args.threads,
            opencv_threads=args.opencv_threads,
            warmup=args.warmup,
            iterations=args.iterations,
            output_json=args.output_json,
            output_csv=args.output_csv,
            overwrite=args.overwrite,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Benchmark failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    results = report["results"]
    print(
        "startup_load_single_measurement_ms "
        "pytorch="
        f"{results['startup_load'][BACKEND_PYTORCH]['single_measurement_ms']:.3f} "
        "onnxruntime_cpu="
        f"{results['startup_load'][BACKEND_ONNX]['single_measurement_ms']:.3f}"
    )
    print(
        "preprocess_ms "
        f"p50={results['preprocess']['shared']['p50_ms']:.3f} "
        f"p95={results['preprocess']['shared']['p95_ms']:.3f}"
    )
    for stage in ("inference_only", "image_to_detections"):
        derived = report["derived_metrics"][stage]
        print(
            f"{stage}_ms "
            f"pytorch_p50={results[stage][BACKEND_PYTORCH]['p50_ms']:.3f} "
            f"pytorch_p95={results[stage][BACKEND_PYTORCH]['p95_ms']:.3f} "
            f"onnx_p50={results[stage][BACKEND_ONNX]['p50_ms']:.3f} "
            f"onnx_p95={results[stage][BACKEND_ONNX]['p95_ms']:.3f} "
            f"onnx_speedup_p50={derived['onnx_speedup_over_pytorch_p50']:.3f}x "
            f"onnx_speedup_p95={derived['onnx_speedup_over_pytorch_p95']:.3f}x"
        )
    print(f"json={args.output_json.resolve()}")
    print(f"csv={args.output_csv.resolve()}")
    return 0


def main() -> int:
    """Parse command-line arguments and run the balanced CPU benchmark."""

    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
