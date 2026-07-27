from __future__ import annotations

from collections import Counter
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "bus.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_smoke"
OUTPUT_IMAGE = OUTPUT_DIR / "bus_result.jpg"


def validate_input_files() -> None:
    """Fail early with a clear message when required files are missing."""
    missing = [
        str(path)
        for path in (MODEL_PATH, IMAGE_PATH)
        if not path.is_file()
    ]

    if missing:
        joined = "\n- ".join(missing)
        raise FileNotFoundError(f"Required files are missing:\n- {joined}")


def main() -> None:
    validate_input_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        device="cpu",
        conf=0.25,
        imgsz=640,
        verbose=True,
    )

    if len(results) != 1:
        raise RuntimeError(f"Expected one result, received {len(results)}.")

    result = results[0]
    detected_classes = [
        result.names[int(class_id)]
        for class_id in result.boxes.cls.tolist()
    ]
    class_counts = Counter(detected_classes)

    plotted_image = result.plot()
    success = result.save(filename=str(OUTPUT_IMAGE))

    if not success or not OUTPUT_IMAGE.is_file():
        raise RuntimeError(f"Failed to save result image: {OUTPUT_IMAGE}")

    print()
    print("=== YOLO SMOKE TEST ===")
    print(f"model: {MODEL_PATH}")
    print(f"source: {IMAGE_PATH}")
    print(f"device: cpu")
    print(f"detections: {len(detected_classes)}")
    print(f"class_counts: {dict(class_counts)}")
    print(f"result_shape: {plotted_image.shape}")
    print(f"output: {OUTPUT_IMAGE}")
    print(f"output_size_bytes: {OUTPUT_IMAGE.stat().st_size}")
    print("yolo_smoke: OK")


if __name__ == "__main__":
    main()