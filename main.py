"""main.py — EdgeGloss orchestrator loop.

Wires the three pipeline stages together and drives the frame loop:

    Camera → VisionTracker → GlossClassifier → LLMTranslator → Output

Run with:
    python main.py [--camera 0] [--config config.yaml] [--no-display]
"""

from __future__ import annotations

import argparse
import sys
from typing import List

# ---------------------------------------------------------------------------
# Optional display dependency — gracefully degrade when not available.
# ---------------------------------------------------------------------------
try:
    import cv2  # type: ignore[import]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from src.vision_tracker import VisionTracker
from src.gloss_classifier import GlossClassifier
from src.llm_translator import LLMTranslator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="EdgeGloss",
        description="Real-time on-device sign-language translation.",
    )
    parser.add_argument(
        "--camera",
        default=0,
        help="Camera device index or path to a video file (default: 0).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless — do not open an OpenCV window.",
    )
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    """Load YAML config; return an empty dict when the file is missing."""
    try:
        import yaml  # type: ignore[import]

        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def run(args: argparse.Namespace) -> None:
    """Main pipeline loop."""
    config = load_config(args.config)

    # -- Instantiate pipeline modules --
    vt_cfg = config.get("vision_tracker", {})
    tracker = VisionTracker(
        model_name=vt_cfg.get("model", "rtmpose-m"),
        confidence=float(vt_cfg.get("confidence", 0.3)),
    )

    gc_cfg = config.get("gloss_classifier", {})
    classifier = GlossClassifier(
        window_size=int(gc_cfg.get("window_size", 30)),
        lstm_model_path=gc_cfg.get("lstm_model"),
        confidence=float(gc_cfg.get("confidence", 0.7)),
    )

    lt_cfg = config.get("llm_translator", {})
    translator = LLMTranslator(
        backend=lt_cfg.get("backend", "stub"),
        model_path=lt_cfg.get("model_path"),
        max_tokens=int(lt_cfg.get("max_tokens", 64)),
    )

    # -- Open camera / video --
    if not _CV2_AVAILABLE:
        print("OpenCV not available — exiting.", file=sys.stderr)
        return

    cap = cv2.VideoCapture(int(args.camera) if isinstance(args.camera, int) else args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera/video: {args.camera}", file=sys.stderr)
        return

    accumulated_glosses: List[str] = []
    current_sentence = ""

    print("EdgeGloss running — press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = tracker.process(frame)
            gloss = classifier.update(keypoints)

            if gloss is not None:
                accumulated_glosses.append(gloss)
                current_sentence = translator.translate(accumulated_glosses)
                print(f"Gloss: {gloss!r}  →  Sentence: {current_sentence!r}")

            if not args.no_display:
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    current_sentence,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("EdgeGloss", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.no_display and _CV2_AVAILABLE:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
