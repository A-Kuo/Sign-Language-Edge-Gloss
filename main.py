#!/usr/bin/env python3
"""
main.py – EdgeGloss orchestrator.

Webcam  ──►  MediaPipe Hand  ──►  GlossClassifier  ──►  stdout / overlay
             (21 × 2 hands)       (heuristic|LSTM)

Usage
-----
    python main.py                       # default webcam, heuristic classifier
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

    print(f"[EdgeGloss] Vision backend : {tracker.mode}")
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

            # 1. Detect hands
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

            # 5. Draw overlay
            if not args.no_display:
                draw_hands(frame, hands)

                # HUD text
                info_lines = [
                    f"Mode: {tracker.mode} | Cls: {classifier.backend}",
                    f"Buffer: {len(buffer)}/{BUFFER_LEN}  Still: {still_count}",
                ]
                if last_gloss:
                    info_lines.append(f"Last gloss: {last_gloss}")

                for i, line in enumerate(info_lines):
                    cv2.putText(
                        frame,
                        line,
                        (10, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("EdgeGloss", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
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
