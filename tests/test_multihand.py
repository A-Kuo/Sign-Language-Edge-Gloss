#!/usr/bin/env python3
"""
tests/test_multihand.py – Phase 1b visual test for multi-hand detection.

Verifies that the HandTracker can detect and stably track two hands
simultaneously, without flickering between them.

Three test modes:
  1. Webcam live capture (default)
  2. Static image (--image)
  3. Synthetic / dummy backend (--dummy)

Outputs
-------
  multihand_test_output.jpg  — annotated frame saved in repo root.
  stdout                     — per-hand slot status and wrist positions.

Usage
-----
    python tests/test_multihand.py                   # webcam
    python tests/test_multihand.py --image photo.jpg # static image
    python tests/test_multihand.py --dummy           # no QAI needed
    python tests/test_multihand.py --use-yolo        # YOLO cascading crop path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision_tracker import (
    HandTracker,
    draw_hands,
    hands_to_feature_vector,
    FEATURE_DIM,
    NUM_HAND_JOINTS,
)

OUTPUT_FILE = "multihand_test_output.jpg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1b: Multi-hand detection test")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--image", type=str, default=None, help="Path to a test image")
    p.add_argument("--dummy", action="store_true", help="Force dummy backend (no QAI)")
    p.add_argument("--use-yolo", action="store_true", help="Use YOLO cascading crop")
    p.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO confidence")
    p.add_argument("--frames", type=int, default=10,
                    help="Number of frames to capture for stability check (webcam mode)")
    return p.parse_args(argv)


def grab_frames(camera_idx: int, n: int) -> list[np.ndarray]:
    """Capture *n* frames from the webcam."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[test_multihand] Cannot open camera {camera_idx}", file=sys.stderr)
        sys.exit(1)

    for _ in range(5):
        cap.read()

    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("[test_multihand] Failed to capture any frames", file=sys.stderr)
        sys.exit(1)

    print(f"[test_multihand] Captured {len(frames)} frame(s): "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Build tracker ─────────────────────────────────────────────────
    tracker = HandTracker(score_threshold=0.5)
    if args.dummy:
        tracker._mode = "dummy"
        tracker.app = None

    yolo_detector = None
    if args.use_yolo:
        from src.yolo_detector import YOLODetector, ensure_model
        onnx_path = ensure_model()
        yolo_detector = YOLODetector(
            onnx_path,
            conf_threshold=args.yolo_conf,
            target_classes=[0],
        )

    print(f"[test_multihand] Backend: {tracker.mode}")
    print(f"[test_multihand] YOLO crop: {'yes' if yolo_detector else 'no'}")

    # ── Get frames ────────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[test_multihand] Cannot read image: {args.image}", file=sys.stderr)
            sys.exit(1)
        frames = [frame]
    elif args.dummy:
        print("[test_multihand] Dummy mode – using synthetic 640x480 frames")
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(args.frames)]
    else:
        frames = grab_frames(args.camera, args.frames)

    # ── Run tracking on every frame ───────────────────────────────────
    history: list[dict[str, np.ndarray | None]] = []
    vectors: list[np.ndarray] = []

    for i, frame in enumerate(frames):
        crops = None
        if yolo_detector is not None:
            from src.yolo_detector import draw_detections
            dets = yolo_detector.detect(frame)
            crops = [
                (d.x1, d.y1, d.x2, d.y2)
                for d in sorted(
                    dets,
                    key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1),
                    reverse=True,
                )[:2]
            ]
            if not crops:
                crops = None

        hands = tracker.track(frame, crops=crops)
        vec = hands_to_feature_vector(hands)
        history.append(hands)
        vectors.append(vec)

        n_hands = sum(1 for v in hands.values() if v is not None)
        sides = [s for s, v in hands.items() if v is not None]
        wrists = {s: hands[s][0].tolist() for s in sides}
        print(f"  Frame {i:>2d}: {n_hands} hand(s) active  sides={sides}  wrists={wrists}")

    # ── Stability analysis ────────────────────────────────────────────
    print("\n--- Stability Report ---")
    hand_counts = [sum(1 for v in h.values() if v is not None) for h in history]
    print(f"  Hand counts per frame : {hand_counts}")
    print(f"  Min / Max hands       : {min(hand_counts)} / {max(hand_counts)}")

    if len(vectors) >= 2:
        diffs = [
            float(np.linalg.norm(vectors[i] - vectors[i - 1]))
            for i in range(1, len(vectors))
        ]
        print(f"  Frame-to-frame diffs  : {[f'{d:.4f}' for d in diffs]}")
        mean_diff = np.mean(diffs)
        print(f"  Mean diff             : {mean_diff:.4f}")
        if mean_diff > 0.5:
            print("  ⚠ HIGH VARIANCE — possible flickering detected")
        else:
            print("  ✓ Tracking appears stable")

    # ── Feature vector sanity ─────────────────────────────────────────
    print(f"\n  Feature vector dim    : {vectors[-1].shape} (expected ({FEATURE_DIM},))")
    right_nonzero = np.any(vectors[-1][:42] != 0)
    left_nonzero = np.any(vectors[-1][42:] != 0)
    print(f"  Right hand data       : {'present' if right_nonzero else 'MISSING'}")
    print(f"  Left hand data        : {'present' if left_nonzero else 'MISSING'}")

    # ── Draw and save last frame ──────────────────────────────────────
    output = frames[-1].copy()
    if yolo_detector is not None:
        from src.yolo_detector import draw_detections
        dets = yolo_detector.detect(output)
        draw_detections(output, dets)

    draw_hands(output, history[-1])

    n_hands = sum(1 for v in history[-1].values() if v is not None)
    summary = f"MultiHand Test | {n_hands} hand(s) | backend={tracker.mode}"
    cv2.putText(
        output, summary, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    cv2.imwrite(OUTPUT_FILE, output)
    print(f"\n[test_multihand] Output saved to {OUTPUT_FILE}")
    print("[test_multihand] Review the image to verify BOTH hands are drawn.")


if __name__ == "__main__":
    main()
