"""Launch the local-only DashGuard Gradio demo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dashguard.demo import create_gradio_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the local DashGuard PyTorch CPU video demo."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/day8/runs"),
        help="isolated Day 8 run directory",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("yolo11n.pt"),
        help="PyTorch yolo11n.pt path",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="localhost TCP port (default: 7860)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = create_gradio_app(
        output_root=args.output_root,
        model_path=args.model_path,
    )
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        max_file_size="1gb",
        allowed_paths=[str(args.output_root.resolve())],
    )


if __name__ == "__main__":
    main()
