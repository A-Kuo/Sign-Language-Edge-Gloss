#!/usr/bin/env python3
"""
main.py – EdgeGloss orchestrator.

Pipeline (cascading crop):
  Webcam  ──►  YOLOv8 (person boxes)  ──►  MediaPipe on crops  ──►  GlossClassifier  ──►  stdout / overlay

Runs with sensible defaults - just execute: python main.py
"""

from __future__ import annotations

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


def main() -> None:
    # Default configuration - runs like a C executable, no flags needed
    camera_index = 0
    lstm_checkpoint = None
    use_yolo = True  # Use YOLO by default for better hand detection
    yolo_conf = 0.5
    no_display = False
    score_threshold = 0.5

    # ── Initialise pipeline components ───────────────────────────────────
    tracker = HandTracker(score_threshold=score_threshold)
    classifier = GlossClassifier(checkpoint=lstm_checkpoint)

    yolo_detector = None
    if use_yolo:
        onnx_path = ensure_model()
        yolo_detector = YOLODetector(
            onnx_path,
            conf_threshold=yolo_conf,
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
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[EdgeGloss] Cannot open camera {camera_index}", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detect hands (optionally via YOLO crops)
            detections = []
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
            if not no_display:
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

                info_lines = [
                    f"{vision_backend}  ·  {classifier.backend}",
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
        if not no_display:
            cv2.destroyAllWindows()
        print("[EdgeGloss] Done.")


if __name__ == "__main__":
    main()
