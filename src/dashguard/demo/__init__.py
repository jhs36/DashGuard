"""Public Day 8 demo workflow interfaces."""

from dashguard.demo.gradio_app import create_gradio_app
from dashguard.demo.workflow import (
    BACKEND,
    DemoRunResult,
    SUCCESS_STATUS,
    SUPPORTED_VIDEO_EXTENSIONS,
    run_dashguard_demo,
)

__all__ = [
    "BACKEND",
    "DemoRunResult",
    "SUCCESS_STATUS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "create_gradio_app",
    "run_dashguard_demo",
]
