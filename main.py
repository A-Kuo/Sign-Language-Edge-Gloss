"""
EdgeGloss — Real-time, 100% on-device sign language translation.
Pipeline: Webcam → RTMPose keypoints → Gloss classifier → Local LLM → English

Run: python main.py [--no-llm] [--debug]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path so "import src" finds the local src package.
# Do NOT run "pip install src" — that installs a different PyPI package.
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import time
from collections import deque

import cv2
import numpy as np

from src.vision_tracker import VisionTracker
from src.gloss_classifier import GlossClassifier
from src.llm_translator import LLMTranslator

# ─────────────────────────── Config ────────────────────────────────────────
FRAME_BUFFER_SIZE   = 30          # frames held in sliding window
STATIC_PAUSE_FRAMES = 45          # frames of stillness before LLM trigger
STILLNESS_THRESHOLD = 0.01        # mean keypoint delta in [0,1] space to count as "still"
DISPLAY_WIDTH       = 1280
DISPLAY_HEIGHT      = 720
# ───────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="EdgeGloss pipeline")
    p.add_argument("--no-llm",  action="store_true", help="Skip LLM stage (gloss only)")
    p.add_argument("--debug",   action="store_true", help="Print raw keypoint deltas")
    p.add_argument("--camera",  type=int, default=0,  help="Webcam device index")
    return p.parse_args()


def is_static(kp_buffer: deque, threshold: float) -> bool:
    """Return True when the last two frames in the buffer are nearly identical.

    Keypoints are already in [0,1] normalised space. We use the mean L2
    distance across joints; threshold is in that same space (resolution-independent).
    """
    if len(kp_buffer) < 2:
        return False
    kp_curr = np.array(kp_buffer[-1])   # (133, 2)
    kp_prev = np.array(kp_buffer[-2])   # (133, 2)
    delta = np.linalg.norm(kp_curr - kp_prev, axis=-1)
    mean_delta = float(delta.mean())
    return mean_delta < threshold


def draw_hud(frame: np.ndarray, gloss_history: list[str],
             translation: str, static_counter: int) -> np.ndarray:
    """Overlay gloss history, translation, and status bar on the frame."""
    h, w = frame.shape[:2]

    # --- semi-transparent bottom bar ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # --- gloss tokens ---
    gloss_str = "  ›  ".join(gloss_history[-8:]) if gloss_history else "—"
    cv2.putText(frame, f"GLOSS: {gloss_str}",
                (12, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 220, 180), 1, cv2.LINE_AA)

    # --- translated sentence ---
    cv2.putText(frame, f"EN:    {translation}",
                (12, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (240, 240, 240), 2, cv2.LINE_AA)

    # --- pause meter ---
    bar_pct = min(static_counter / STATIC_PAUSE_FRAMES, 1.0)
    bar_w   = int((w - 24) * bar_pct)
    color   = (0, int(200 * bar_pct), int(220 * (1 - bar_pct)))
    cv2.rectangle(frame, (12, h - 18), (12 + bar_w, h - 8), color, -1)
    cv2.rectangle(frame, (12, h - 18), (w - 12, h - 8), (80, 80, 80), 1)
    cv2.putText(frame, "PAUSE METER", (14, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    return frame


def main():
    args = parse_args()

    print("[EdgeGloss] Initialising pipeline …")
    tracker    = VisionTracker()
    classifier = GlossClassifier()
    translator = LLMTranslator(disabled=args.no_llm)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    if not cap.isOpened():
        sys.exit(f"[EdgeGloss] ERROR: Cannot open camera {args.camera}")

    try:
        print("[EdgeGloss] Camera open. Press Q to quit.\n")

        kp_buffer:      deque      = deque(maxlen=FRAME_BUFFER_SIZE)
        gloss_history:  list[str]  = []
        translation:    str        = ""
        static_counter: int        = 0
        last_gloss:     str        = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[EdgeGloss] Frame grab failed — retrying …")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # ── Stage 1: keypoint extraction ──────────────────────────────────
            keypoints, frame = tracker.process(frame)   # returns (133,2) + annotated frame

            if keypoints is not None:
                kp_buffer.append(keypoints)

                # ── Stage 2: gloss classification ─────────────────────────────
                if len(kp_buffer) == FRAME_BUFFER_SIZE:
                    gloss = classifier.classify(list(kp_buffer))
                    if gloss and gloss != last_gloss:
                        gloss_history.append(gloss)
                        last_gloss = gloss
                        static_counter = 0          # reset pause meter on new sign

                # ── Pause detection → LLM trigger ─────────────────────────────
                if is_static(kp_buffer, STILLNESS_THRESHOLD):
                    static_counter += 1
                    if args.debug:
                        print(f"[debug] static_counter={static_counter}")
                else:
                    static_counter = max(0, static_counter - 2)

                if static_counter >= STATIC_PAUSE_FRAMES and gloss_history:
                    print(f"[EdgeGloss] Pause detected. Sending to LLM: {gloss_history}")
                    translation    = translator.translate(gloss_history)
                    gloss_history  = []            # clear after translation
                    static_counter = 0
                    last_gloss     = ""
                    print(f"[EdgeGloss] Translation: {translation}\n")

            # ── Render HUD ─────────────────────────────────────────────────────
            frame = draw_hud(frame, gloss_history, translation, static_counter)
            cv2.imshow("EdgeGloss — On-Device Sign Language Translator", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("[EdgeGloss] Shutdown complete.")


if __name__ == "__main__":
    main()
