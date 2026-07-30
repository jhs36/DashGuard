"""Convert one BDD100K Detection 2020 split to YOLO format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dashguard.data import (
    YOLO_CLASS_NAMES,
    ConversionStats,
    convert_bdd100k_to_yolo,
)


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the BDD100K conversion command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert a BDD100K Detection 2020 frame-list JSON split to YOLO."
        )
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--limit", type=_positive_int)
    return parser


def data_yaml_text(output_root: str | Path) -> str:
    """Return the fixed DashGuard YOLO dataset configuration."""

    dataset_path = Path(output_root).resolve().as_posix()
    names = "\n".join(
        f"  {class_id}: {name}"
        for class_id, name in enumerate(YOLO_CLASS_NAMES)
    )
    return (
        f"path: {json.dumps(dataset_path, ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )


def run(args: argparse.Namespace) -> int:
    """Run one split conversion and return a process exit code."""

    try:
        stats = convert_bdd100k_to_yolo(
            annotations_path=args.annotations,
            images_dir=args.images_dir,
            output_root=args.output_root,
            split=args.split,
            limit=args.limit,
        )
        data_yaml_path = args.output_root / "data.yaml"
        data_yaml_path.write_text(
            data_yaml_text(args.output_root),
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"BDD100K conversion failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    _print_stats(stats)
    return 0


def _print_stats(stats: ConversionStats) -> None:
    print("BDD100K conversion complete")
    for key, value in stats.items():
        print(f"{key}={value}")


def main() -> int:
    """Parse command-line arguments and run the conversion."""

    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
