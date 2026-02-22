#!/usr/bin/env python3
"""
main.py – EdgeGloss orchestrator.

Pipeline (full frame):
  Webcam  ──►  MediaPipe Hand  ──►  GlossClassifier  ──►  stdout / overlay

Pipeline (cascading crop, with --use-yolo):
  Webcam  ──►  YOLOv8 (person boxes)  ──►  MediaPipe on crops  ──►  GlossClassifier  ──►  stdout / overlay

Usage
-----
    python main.py                       # default webcam, heuristic classifier
    python main.py --use-yolo            # YOLO person detection + MediaPipe on crops (demo)
    python main.py --camera 1            # different camera index
    python main.py --lstm models/lstm.pt # use trained LSTM checkpoint
    python main.py --no-display          # headless (e.g. SSH / CI)
"""

from __future__ import annotations

import argparse
import collections
import sys

import cv2
import numpy as np

from src.gloss_classifier import BUFFER_LEN, GlossClassifier
from src.vision_tracker import (
    FEATURE_DIM,
    HandTracker,
    draw_hands,
    hands_to_feature_vector,
)
from src.yolo_detector import YOLODetector, draw_detections, ensure_model

# ── Stillness detection ──────────────────────────────────────────────────────
# A gloss is emitted only after the hand has been held roughly still for a
# minimum number of consecutive frames.  The threshold is normalised by the
# frame diagonal so it is resolution-independent.

STILLNESS_THRESHOLD = 0.012  # fraction of frame diagonal
STILLNESS_FRAMES = 8         # consecutive frames below threshold to trigger


def _frame_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Normalised L2 between two feature vectors."""
    return float(np.linalg.norm(vec_a - vec_b))


# ── Main loop ────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeGloss – sign language gloss detector")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--lstm", type=str, default=None, help="Path to LSTM checkpoint")
    p.add_argument(
        "--use-yolo",
        action="store_true",
        help="Run YOLOv8 first (person boxes), then MediaPipe on crops (cascading)",
    )
    p.add_argument(
        "--yolo-conf",
        type=float,
        default=0.5,
        help="YOLO detection confidence threshold (default 0.5)",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode – skip OpenCV window",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum hand-detection confidence",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Initialise pipeline components ───────────────────────────────────
    tracker = HandTracker(score_threshold=args.score_threshold)
    classifier = GlossClassifier(checkpoint=args.lstm)

    yolo_detector = None
    if args.use_yolo:
        onnx_path = ensure_model()
        yolo_detector = YOLODetector(
            onnx_path,
            conf_threshold=args.yolo_conf,
            target_classes=[0],  # COCO person – crop for MediaPipe
        )

    vision_backend = "YOLO+" + tracker.mode if yolo_detector else tracker.mode
    print(f"[EdgeGloss] Vision backend : {vision_backend}")
    print(f"[EdgeGloss] Classifier     : {classifier.backend}")
    print("[EdgeGloss] Press 'q' to quit.\n")

    # ── State ────────────────────────────────────────────────────────────
    buffer: collections.deque[np.ndarray] = collections.deque(maxlen=BUFFER_LEN)
    still_count = 0
    last_gloss: str | None = None
    prev_vec: np.ndarray | None = None

    # ── Webcam loop ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[EdgeGloss] Cannot open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detect hands (optionally via YOLO crops)
            if yolo_detector is not None:
                detections = yolo_detector.detect(frame)
                # Use up to 2 largest person boxes (by area) as crops
                crops = [
                    (d.x1, d.y1, d.x2, d.y2)
                    for d in sorted(
                        detections,
                        key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1),
                        reverse=True,
                    )[:2]
                ]
                hands = tracker.track(frame, crops=crops if crops else None)
            else:
                hands = tracker.track(frame)

            # 2. Build feature vector and push to buffer
            vec = hands_to_feature_vector(hands)
            buffer.append(vec)

            # 3. Stillness detection
            if prev_vec is not None:
                dist = _frame_distance(vec, prev_vec)
                if dist < STILLNESS_THRESHOLD:
                    still_count += 1
                else:
                    still_count = 0
            prev_vec = vec

            # 4. Classify when hand held still and buffer full
            if still_count >= STILLNESS_FRAMES and len(buffer) == BUFFER_LEN:
                buf_arr = np.stack(list(buffer))  # (T, 84)
                gloss = classifier.classify(buf_arr)

                if gloss is not None and gloss != last_gloss:
                    last_gloss = gloss
                    print(f"  >> GLOSS: {gloss}")

                # Reset stillness counter so we don't re-fire every frame
                still_count = 0

            # 5. Draw overlay (Zoom-style: subtle, muted)
            if not args.no_display:
                if yolo_detector is not None:
                    draw_detections(frame, detections)
                draw_hands(frame, hands)

                h, w = frame.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.48
                thickness = 1
                color_text = (220, 220, 220)  # soft white (BGR)
                color_bar = (32, 32, 32)      # dark bar

                # Top bar: semi-transparent strip for status
                bar_h = 56
                roi = frame[0:bar_h, 0:w].copy()
                cv2.rectangle(roi, (0, 0), (w, bar_h), color_bar, -1)
                frame[0:bar_h, 0:w] = cv2.addWeighted(roi, 0.5, frame[0:bar_h, 0:w], 0.5, 0)

                n_hands = sum(1 for v in hands.values() if v is not None)
                info_lines = [
                    f"{vision_backend}  ·  {classifier.backend}  ·  Hands: {n_hands}",
                    f"Buffer {len(buffer)}/{BUFFER_LEN}  ·  Still {still_count}",
                ]
                if last_gloss:
                    info_lines.append(f"Gloss: {last_gloss}")

                for i, line in enumerate(info_lines):
                    cv2.putText(
                        frame, line,
                        (14, 22 + i * 20),
                        font, scale, color_text, thickness,
                        cv2.LINE_AA,
                    )

                # Bottom-right: "Press Q to quit" (Zoom-style corner hint)
                quit_text = "Press Q to quit"
                (tw, th), _ = cv2.getTextSize(quit_text, font, scale, thickness)
                pad_x, pad_y = 10, 6
                qx1 = w - tw - pad_x * 2 - 8
                qy1 = h - th - pad_y * 2 - 8
                qx2, qy2 = w - 6, h - 6
                roi_q = frame[qy1:qy2, qx1:qx2].copy()
                cv2.rectangle(roi_q, (0, 0), (qx2 - qx1, qy2 - qy1), color_bar, -1)
                frame[qy1:qy2, qx1:qx2] = cv2.addWeighted(roi_q, 0.5, frame[qy1:qy2, qx1:qx2], 0.5, 0)
                cv2.putText(
                    frame, quit_text,
                    (qx1 + pad_x, qy2 - pad_y),
                    font, scale, color_text, thickness,
                    cv2.LINE_AA,
                )

                cv2.imshow("EdgeGloss", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or Escape
                    break

    except KeyboardInterrupt:
        print("\n[EdgeGloss] Interrupted.")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[EdgeGloss] Done.")


if __name__ == "__main__":
    main()
