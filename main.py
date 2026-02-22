"""
main.py – Orchestrator loop for Sign-Language-Edge-Gloss.

Wires together VisionTracker → GlossClassifier → LLMTranslator and drives
the real-time pipeline.

Usage::

    python main.py [--camera 0] [--width 640] [--height 480]
                   [--window 30] [--confidence 0.6]
                   [--llm-model models/llm] [--llm-backend transformers]
                   [--lstm-checkpoint models/gloss_lstm.pt]
"""

from __future__ import annotations

import argparse
import signal
import sys

from src.vision_tracker import VisionTracker
from src.gloss_classifier import GlossClassifier
from src.llm_translator import LLMTranslator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign-Language-Edge-Gloss real-time pipeline"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                        help="Capture width in pixels (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Capture height in pixels (default: 480)")
    parser.add_argument("--window", type=int, default=30,
                        help="Gloss classifier sliding-window size (default: 30)")
    parser.add_argument("--confidence", type=float, default=0.6,
                        help="Minimum gloss confidence threshold (default: 0.6)")
    parser.add_argument("--llm-model", default=None,
                        help="Path to local LLM weights (default: stub mode)")
    parser.add_argument("--llm-backend", default="transformers",
                        choices=["transformers", "llama_cpp", "stub"],
                        help="LLM inference backend (default: transformers)")
    parser.add_argument("--lstm-checkpoint", default=None,
                        help="Path to GlossClassifier LSTM checkpoint (optional)")
    return parser.parse_args(argv)


def build_pipeline(args: argparse.Namespace):
    """Instantiate and return (tracker, classifier, translator)."""
    tracker = VisionTracker(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
    )
    classifier = GlossClassifier(
        window_size=args.window,
        confidence_threshold=args.confidence,
        lstm_checkpoint=args.lstm_checkpoint,
    )
    translator = LLMTranslator(
        model_path=args.llm_model,
        backend=args.llm_backend if args.llm_model else "stub",
    )
    return tracker, classifier, translator


def run(tracker: VisionTracker,
        classifier: GlossClassifier,
        translator: LLMTranslator) -> None:
    """Main capture-classify-translate loop."""
    gloss_buffer: list[str] = []

    print("Pipeline running. Press Ctrl-C to stop.")
    tracker.start()
    try:
        while True:
            frame_data = tracker.get_keypoints()
            if frame_data is None:
                continue

            keypoints = frame_data["keypoints"]
            gloss = classifier.update(keypoints)

            if gloss is not None:
                gloss_buffer.append(gloss)
                print(f"[GLOSS] {gloss}")

                # Translate when a pause is detected (stub: translate every 5
                # glosses; replace with a silence/boundary detector later).
                if len(gloss_buffer) >= 5:
                    sentence = translator.translate(gloss_buffer)
                    print(f"[SENTENCE] {sentence}")
                    gloss_buffer.clear()
                    classifier.reset()

    except KeyboardInterrupt:
        # Flush any remaining glosses on exit
        if gloss_buffer:
            sentence = translator.translate(gloss_buffer)
            print(f"[SENTENCE] {sentence}")
    finally:
        tracker.stop()
        print("Pipeline stopped.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tracker, classifier, translator = build_pipeline(args)

    # Handle SIGTERM gracefully (e.g. systemd / container shutdown)
    def _sigterm_handler(signum, frame):  # noqa: ARG001
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    run(tracker, classifier, translator)


if __name__ == "__main__":
    main()
