"""Inference validation helpers."""

from dashguard.inference.onnx_validation import (
    OnnxRuntimeRunner,
    PytorchRunner,
    bbox_coordinate_max_abs_diff,
    bbox_iou,
    calculate_difference_statistics,
    compare_backend_outputs,
    confidence_abs_diff,
    file_sha256,
    letterbox_geometry,
    match_detections,
    postprocess_raw_detections,
    preprocess_bgr_image,
    pytorch_output_to_numpy,
    raw_allclose_diagnostic,
    validate_onnx_outputs,
    write_json_atomic,
)

__all__ = [
    "OnnxRuntimeRunner",
    "PytorchRunner",
    "bbox_coordinate_max_abs_diff",
    "bbox_iou",
    "calculate_difference_statistics",
    "compare_backend_outputs",
    "confidence_abs_diff",
    "file_sha256",
    "letterbox_geometry",
    "match_detections",
    "postprocess_raw_detections",
    "preprocess_bgr_image",
    "pytorch_output_to_numpy",
    "raw_allclose_diagnostic",
    "validate_onnx_outputs",
    "write_json_atomic",
]
