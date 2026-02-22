#!/usr/bin/env python3
"""
tests/test_multihand.py – Phase 1b visual test for multi-hand detection.

Verifies that the HandTracker can detect and stably track two hands
simultaneously, without flickering between them.

Three test modes:
  1. Webcam live capture (default)
  2. Static image (--image)
  3. Synthetic / dummy backend (--dummy)

Also tests FIR landmark smoothing, rotation normalisation, and the
clear-on-loss behaviour ported from Multi-HandTrackingGPU.

Outputs
-------
  multihand_test_output.jpg  — annotated frame saved in repo root.
  stdout                     — per-hand slot status, stability report,
                               and smoothing effectiveness metrics.

Usage
-----
    python tests/test_multihand.py                   # webcam
    python tests/test_multihand.py --image photo.jpg # static image
    python tests/test_multihand.py --dummy           # no QAI needed
    python tests/test_multihand.py --use-yolo        # YOLO cascading crop path
    python tests/test_multihand.py --no-smooth       # disable FIR smoothing
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
    hands_to_rotation_invariant_vector,
    normalize_hand_rotation,
    _LandmarkSmoother,
    FEATURE_DIM,
    NUM_HAND_JOINTS,
    WRIST_IDX,
)

OUTPUT_FILE = "multihand_test_output.jpg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1b: Multi-hand detection test")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--image", type=str, default=None, help="Path to a test image")
    p.add_argument("--dummy", action="store_true", help="Force dummy backend (no QAI)")
    p.add_argument("--use-yolo", action="store_true", help="Use YOLO cascading crop")
    p.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO confidence")
    p.add_argument("--no-smooth", action="store_true", help="Disable FIR smoothing")
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


def test_smoother_unit() -> None:
    """Unit-test the FIR landmark smoother in isolation."""
    print("\n--- FIR Smoother Unit Test ---")
    sm = _LandmarkSmoother()
    base = np.random.RandomState(0).randn(21, 2).astype(np.float32) * 0.1 + 0.5

    raw_diffs = []
    smooth_diffs = []
    for i in range(15):
        jitter = np.random.RandomState(i + 100).randn(21, 2).astype(np.float32) * 0.02
        noisy = base + jitter
        smoothed = sm.push(noisy)

        if i >= 7:
            raw_diffs.append(np.linalg.norm(noisy - base))
            smooth_diffs.append(np.linalg.norm(smoothed - base))

    raw_mean = float(np.mean(raw_diffs))
    smooth_mean = float(np.mean(smooth_diffs))
    reduction = (1 - smooth_mean / raw_mean) * 100
    print(f"  Raw mean L2 from base:      {raw_mean:.5f}")
    print(f"  Smoothed mean L2 from base: {smooth_mean:.5f}")
    print(f"  Jitter reduction:            {reduction:.1f}%")

    if reduction > 0:
        print("  PASS — smoothing reduces jitter")
    else:
        print("  WARN — smoothing did not reduce jitter")


def test_rotation_normalisation() -> None:
    """Verify rotation normalisation aligns middle MCP direction."""
    print("\n--- Rotation Normalisation Test ---")
    rng = np.random.RandomState(42)

    n_pass = 0
    for trial in range(5):
        lm = rng.randn(21, 2).astype(np.float32) * 0.15 + 0.5
        normed = normalize_hand_rotation(lm)

        wrist_dist = float(np.linalg.norm(normed[WRIST_IDX]))
        mcp_dir = normed[9] / (float(np.linalg.norm(normed[9])) + 1e-8)
        angle_err = float(np.abs(np.arctan2(mcp_dir[0], -mcp_dir[1])) * 180 / np.pi)

        ok = wrist_dist < 1e-5 and angle_err < 1.0
        n_pass += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"  Trial {trial}: wrist_dist={wrist_dist:.6f}  "
              f"angle_err={angle_err:.2f}deg  [{status}]")

    print(f"  {n_pass}/5 passed")


def test_clear_on_loss() -> None:
    """Verify smoother buffers clear when a hand disappears."""
    print("\n--- Clear-on-Loss Test ---")
    tracker = HandTracker(smoothing=True)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        tracker.track(frame)

    r_len_before = len(tracker._smoother_right)
    l_len_before = len(tracker._smoother_left)

    tracker._smooth_result({"left": None, "right": None})

    r_len_after = len(tracker._smoother_right)
    l_len_after = len(tracker._smoother_left)

    print(f"  Before loss: right={r_len_before}, left={l_len_before}")
    print(f"  After loss:  right={r_len_after}, left={l_len_after}")

    if r_len_after == 0 and l_len_after == 0:
        print("  PASS — buffers cleared on hand loss")
    else:
        print("  FAIL — buffers not cleared")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Build tracker ─────────────────────────────────────────────────
    tracker = HandTracker(
        score_threshold=0.5,
        smoothing=not args.no_smooth,
    )
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

    print(f"[test_multihand] Backend:   {tracker.mode}")
    print(f"[test_multihand] YOLO crop: {'yes' if yolo_detector else 'no'}")
    print(f"[test_multihand] Smoothing: {tracker.smoothing}")

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
    ri_vectors: list[np.ndarray] = []

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
        ri_vec = hands_to_rotation_invariant_vector(hands)
        history.append(hands)
        vectors.append(vec)
        ri_vectors.append(ri_vec)

        n_hands = sum(1 for v in hands.values() if v is not None)
        sides = [s for s, v in hands.items() if v is not None]
        wrists = {s: [f"{hands[s][0][0]:.3f}", f"{hands[s][0][1]:.3f}"] for s in sides}
        print(f"  Frame {i:>2d}: {n_hands} hand(s)  sides={sides}  wrists={wrists}")

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
        mean_diff = float(np.mean(diffs))
        print(f"  Mean diff             : {mean_diff:.4f}")
        if mean_diff > 0.5:
            print("  WARNING — HIGH VARIANCE — possible flickering detected")
        else:
            print("  OK — tracking appears stable")

    # ── Feature vector sanity ─────────────────────────────────────────
    print(f"\n  Feature vector dim    : {vectors[-1].shape} (expected ({FEATURE_DIM},))")
    right_nonzero = np.any(vectors[-1][:42] != 0)
    left_nonzero = np.any(vectors[-1][42:] != 0)
    print(f"  Right hand data       : {'present' if right_nonzero else 'MISSING'}")
    print(f"  Left hand data        : {'present' if left_nonzero else 'MISSING'}")
    print(f"  Rot-invariant dim     : {ri_vectors[-1].shape} (expected ({FEATURE_DIM},))")

    # ── Sub-tests ─────────────────────────────────────────────────────
    test_smoother_unit()
    test_rotation_normalisation()
    test_clear_on_loss()

    # ── Draw and save last frame ──────────────────────────────────────
    output = frames[-1].copy()
    if yolo_detector is not None:
        from src.yolo_detector import draw_detections
        dets = yolo_detector.detect(output)
        draw_detections(output, dets)

    draw_hands(output, history[-1])

    n_hands = sum(1 for v in history[-1].values() if v is not None)
    summary = f"MultiHand | {n_hands} hand(s) | {tracker.mode} | smooth={'on' if tracker.smoothing else 'off'}"
    cv2.putText(
        output, summary, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    cv2.imwrite(OUTPUT_FILE, output)
    print(f"\n[test_multihand] Output saved to {OUTPUT_FILE}")
    print("[test_multihand] Review the image to verify BOTH hands are drawn.")


if __name__ == "__main__":
    main()
