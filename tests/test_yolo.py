#!/usr/bin/env python3
"""
tests/test_yolo.py – Phase 1 visual test for YOLOv8 detection.

Captures a single frame from the webcam (or loads a provided image),
runs YOLOv8 inference via ONNX Runtime, draws red bounding boxes
around detected objects, and saves the result.

Usage
-----
    # Webcam (default camera 0)
    python tests/test_yolo.py

    # Specific camera index
    python tests/test_yolo.py --camera 1

    # Use an existing image instead of webcam
    python tests/test_yolo.py --image path/to/photo.jpg

    # Filter to specific COCO classes (0 = person)
    python tests/test_yolo.py --classes 0

Output
------
    yolo_test_output.jpg   — saved in the repo root for review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.yolo_detector import (
    Detection,
    YOLODetector,
    draw_detections,
    ensure_model,
)

OUTPUT_FILE = "yolo_test_output.jpg"
MODEL_DIR = "models"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: YOLOv8 visual test")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--image", type=str, default=None, help="Path to a test image (skip webcam)")
    p.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="COCO class IDs to keep (e.g. 0 for person). Default: all.",
    )
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--model", type=str, default=None, help="Path to ONNX model (auto-downloads if omitted)")
    return p.parse_args(argv)


def grab_frame(camera_idx: int) -> cv2.typing.MatLike:
    """Capture a single frame from the webcam."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[test_yolo] Cannot open camera {camera_idx}", file=sys.stderr)
        sys.exit(1)

    # Warm up — discard first few frames for auto-exposure
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[test_yolo] Failed to capture frame", file=sys.stderr)
        sys.exit(1)

    print(f"[test_yolo] Captured frame: {frame.shape[1]}x{frame.shape[0]}")
    return frame


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Get input frame ───────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[test_yolo] Cannot read image: {args.image}", file=sys.stderr)
            sys.exit(1)
        print(f"[test_yolo] Loaded image: {args.image} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        frame = grab_frame(args.camera)

    # ── Ensure ONNX model exists ──────────────────────────────────────
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = ensure_model(MODEL_DIR)

    # ── Run detection ─────────────────────────────────────────────────
    detector = YOLODetector(
        model_path,
        conf_threshold=args.conf,
        target_classes=args.classes,
    )
    detections = detector.detect(frame)

    # ── Report results ────────────────────────────────────────────────
    print(f"\n[test_yolo] Found {len(detections)} detection(s):")
    for i, det in enumerate(detections):
        print(
            f"  [{i}] {det.class_name:>12s}  conf={det.confidence:.3f}  "
            f"bbox=({det.x1}, {det.y1}, {det.x2}, {det.y2})"
        )

    # ── Draw and save ─────────────────────────────────────────────────
    output = frame.copy()
    draw_detections(output, detections, color=(0, 0, 255), thickness=2)

    # Add summary text
    summary = f"YOLOv8n | {len(detections)} detections | conf>={args.conf}"
    cv2.putText(
        output, summary, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    cv2.imwrite(OUTPUT_FILE, output)
    print(f"\n[test_yolo] Output saved to {OUTPUT_FILE}")
    print("[test_yolo] Review the image to verify bounding boxes are correct.")


if __name__ == "__main__":
    main()
