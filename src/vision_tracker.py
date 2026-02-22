"""
vision_tracker.py – Qualcomm AI Hub MediaPipe Hand wrapper.

Extracts 21-joint hand landmarks per detected hand from BGR webcam frames.
Falls back to deterministic dummy keypoints when ``qai_hub_models`` is not
installed so the rest of the pipeline can be tested end-to-end.

Multi-hand architecture (inspired by teevee112/Multi-HandTrackingGPU):
  1. Collect ALL detected hands before assignment (NMS-style, no early discard).
  2. Assign to left/right slots via temporal cost-matrix matching.
  3. Per-hand FIR smoothing on the full 21-joint landmark buffer to
     reduce jitter (ported from Multi-HandTrackingGPU's smooth_keypoints).
  4. Clear history when a hand is lost to avoid stale-data contamination.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque

# ── Constants ────────────────────────────────────────────────────────────────

NUM_HAND_JOINTS = 21
NUM_HANDS = 2  # left + right
FEATURE_DIM_PER_HAND = NUM_HAND_JOINTS * 2  # (x, y) per joint
FEATURE_DIM = FEATURE_DIM_PER_HAND * NUM_HANDS  # 84

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9  # used for palm-length normalisation

# 21-point hand skeleton connectivity (MediaPipe convention)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (5, 6), (6, 7), (7, 8),                 # index
    (9, 10), (10, 11), (11, 12),            # middle
    (13, 14), (14, 15), (15, 16),           # ring
    (17, 18), (18, 19), (19, 20),           # pinky
    (0, 5), (5, 9), (9, 13), (13, 17),      # palm
    (0, 17),
]

# Colours (BGR) for drawing
_RIGHT_COLOUR = (255, 200, 0)   # cyan-ish
_LEFT_COLOUR = (0, 100, 255)    # orange-ish
_POINT_COLOUR = (0, 255, 0)     # green

# ── Temporal tracking constants ──────────────────────────────────────────────

# When de-duplicating hands from overlapping crops, two wrists closer than
# this normalised distance are considered the same physical hand.
_DEDUP_WRIST_DIST = 0.08

# FIR smoothing filter (ported from Multi-HandTrackingGPU config.py).
# 7-tap causal low-pass filter: older frames get lower (even negative)
# weights, recent frames get higher weights.  Sum ≈ 1.0.
_FIR_COEFFS = np.array(
    [-0.17857, -0.07143, 0.03571, 0.14286, 0.25, 0.35714, 0.46429],
    dtype=np.float32,
)
_FIR_LEN = len(_FIR_COEFFS)

# ── Try importing Qualcomm AI Hub ────────────────────────────────────────────

try:
    import torch
    from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
    from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand

    _HAS_QAI = True
except ImportError:
    _HAS_QAI = False


# ── FIR Landmark Smoother ────────────────────────────────────────────────────


class _LandmarkSmoother:
    """Per-hand FIR low-pass filter on the full (21, 2) landmark array.

    Maintains a fixed-length deque of recent landmark frames and returns
    the weighted sum using ``_FIR_COEFFS``.  When the buffer has fewer
    entries than the filter length, a truncated (re-normalised) version
    of the newest coefficients is used so output is always valid.
    """

    def __init__(self) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=_FIR_LEN)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, landmarks: np.ndarray) -> np.ndarray:
        """Add a new (21, 2) frame and return the smoothed output."""
        self._buf.append(landmarks.copy())
        n = len(self._buf)

        if n == 1:
            return landmarks.copy()

        # Use the *newest* n coefficients, re-normalised to sum to 1
        coeffs = _FIR_COEFFS[-n:].copy()
        coeffs /= coeffs.sum()

        stacked = np.stack(list(self._buf))          # (n, 21, 2)
        smoothed = np.einsum("i,ijk->jk", coeffs, stacked)  # (21, 2)
        return smoothed.astype(np.float32)


# ── Public API ───────────────────────────────────────────────────────────────


class HandTracker:
    """Detect hands and return normalised 21-joint landmarks per frame.

    Attributes
    ----------
    mode : str
        ``"qai"`` when running through the Qualcomm pipeline,
        ``"dummy"`` when falling back to synthetic keypoints.
    smoothing : bool
        Whether FIR landmark smoothing is active.
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        smoothing: bool = True,
    ) -> None:
        self.score_threshold = score_threshold
        self.smoothing = smoothing

        # Temporal tracking state (wrist positions from previous frame)
        self._prev_right_wrist: np.ndarray | None = None  # (2,) normalised
        self._prev_left_wrist: np.ndarray | None = None

        # Per-hand FIR smoothers
        self._smoother_right = _LandmarkSmoother()
        self._smoother_left = _LandmarkSmoother()

        if _HAS_QAI:
            model = MediaPipeHand.from_pretrained()
            self.app = MediaPipeHandApp.from_pretrained(model)
            self._mode = "qai"
        else:
            self.app = None
            self._mode = "dummy"
            print("[HandTracker] qai_hub_models not found – using dummy keypoints")

    @property
    def mode(self) -> str:
        return self._mode

    # ── Core tracking ────────────────────────────────────────────────────

    def track(
        self,
        bgr_frame: np.ndarray,
        crops: list[tuple[int, int, int, int]] | None = None,
    ) -> dict[str, np.ndarray | None]:
        """Return ``{'left': (21,2)|None, 'right': (21,2)|None}`` in [0, 1] (full frame).

        Parameters
        ----------
        bgr_frame : np.ndarray
            Full BGR image (e.g. webcam frame).
        crops : list of (x1, y1, x2, y2) or None
            If provided, run hand detection only on these ROIs (cascading crop).
            Landmarks are mapped back to full-frame normalised [0, 1].
            If None or empty, runs on the full frame.
        """
        if self._mode == "dummy":
            raw = self._dummy_track(bgr_frame)
        elif crops:
            raw = self._qai_track_crops(bgr_frame, crops)
        else:
            raw = self._qai_track(bgr_frame)

        return self._smooth_result(raw)

    # ── FIR smoothing wrapper ─────────────────────────────────────────────

    def _smooth_result(
        self, raw: dict[str, np.ndarray | None]
    ) -> dict[str, np.ndarray | None]:
        """Apply per-hand FIR smoothing.  Clear a hand's history when lost."""
        result: dict[str, np.ndarray | None] = {"left": None, "right": None}

        for side, smoother in [
            ("right", self._smoother_right),
            ("left", self._smoother_left),
        ]:
            lm = raw[side]
            if lm is not None:
                result[side] = smoother.push(lm) if self.smoothing else lm
            else:
                smoother.clear()

        return result

    # ── Low-level MediaPipe extraction ────────────────────────────────────

    def _detect_hands_raw(
        self, bgr_image: np.ndarray
    ) -> list[dict[str, np.ndarray | bool]]:
        """Run MediaPipe on *bgr_image* and return ALL detected hands.

        Each entry is ``{'landmarks': (21,2) float32 in [0,1], 'is_right': bool}``.
        Coordinates are normalised to the input image dimensions.
        """
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        raw = self.app.predict_landmarks_from_image(rgb, raw_output=True)
        batched_landmarks = raw[3]
        batched_is_right = raw[4]

        hands: list[dict] = []
        if len(batched_landmarks) == 0:
            return hands

        landmarks_t = batched_landmarks[0]
        if (
            not isinstance(landmarks_t, torch.Tensor)
            or landmarks_t.nelement() == 0
        ):
            return hands

        is_right_list = batched_is_right[0]

        for i, is_right in enumerate(is_right_list):
            lm = landmarks_t[i].cpu().numpy()  # (21, 3)  x_px, y_px, conf
            xy = lm[:, :2].copy()
            xy[:, 0] /= w
            xy[:, 1] /= h
            hands.append({
                "landmarks": xy.astype(np.float32),
                "is_right": bool(is_right),
            })

        return hands

    # ── Qualcomm path (full frame) ────────────────────────────────────────

    def _qai_track(self, bgr_frame: np.ndarray) -> dict[str, np.ndarray | None]:
        all_hands = self._detect_hands_raw(bgr_frame)
        return self._assign_to_slots(all_hands)

    # ── Qualcomm path with YOLO crops ─────────────────────────────────────

    def _qai_track_crops(
        self,
        bgr_frame: np.ndarray,
        crops: list[tuple[int, int, int, int]],
    ) -> dict[str, np.ndarray | None]:
        """Run MediaPipe on each crop and merge landmarks into full-frame [0,1]."""
        fh, fw = bgr_frame.shape[:2]
        all_hands: list[dict] = []

        for (x1, y1, x2, y2) in crops:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = bgr_frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]

            crop_hands = self._detect_hands_raw(crop)

            for h in crop_hands:
                lm = h["landmarks"]  # (21, 2) normalised in crop space
                full_x = (x1 + lm[:, 0] * crop_w) / fw
                full_y = (y1 + lm[:, 1] * crop_h) / fh
                all_hands.append({
                    "landmarks": np.stack([full_x, full_y], axis=1).astype(np.float32),
                    "is_right": h["is_right"],
                })

        all_hands = self._deduplicate_hands(all_hands)
        return self._assign_to_slots(all_hands)

    # ── Hand assignment logic ─────────────────────────────────────────────

    def _assign_to_slots(
        self, all_hands: list[dict]
    ) -> dict[str, np.ndarray | None]:
        """Robustly assign detected hands to ``left`` / ``right`` slots.

        Strategy (ordered by priority):
        1. When tracking history exists for both hands, use temporal matching
           (wrist-position cost matrix) to maintain frame-to-frame consistency
           and prevent flickering — this overrides chirality labels.
        2. Without full history, trust MediaPipe chirality when left and right
           are both represented.
        3. Final fallback: spatial position (lower wrist-x → right hand,
           which matches the typical non-mirrored webcam view).
        """
        result: dict[str, np.ndarray | None] = {"left": None, "right": None}

        if not all_hands:
            return result

        if len(all_hands) == 1:
            h = all_hands[0]
            if self._has_tracking_history():
                key = self._nearest_slot(h["landmarks"][WRIST_IDX])
            else:
                key = "right" if h["is_right"] else "left"
            result[key] = h["landmarks"]

        else:
            hands = all_hands[:2]

            if self._has_both_tracking():
                result = self._match_pair_to_previous(hands[0], hands[1])
            else:
                rights = [h for h in hands if h["is_right"]]
                lefts = [h for h in hands if not h["is_right"]]

                if rights and lefts:
                    result["right"] = rights[0]["landmarks"]
                    result["left"] = lefts[0]["landmarks"]
                elif self._has_tracking_history():
                    result = self._match_pair_to_previous(hands[0], hands[1])
                else:
                    sorted_h = sorted(
                        hands, key=lambda h: h["landmarks"][WRIST_IDX, 0]
                    )
                    result["right"] = sorted_h[0]["landmarks"]
                    result["left"] = sorted_h[1]["landmarks"]

        self._update_tracking(result)
        return result

    # ── Temporal tracking helpers ─────────────────────────────────────────

    def _has_tracking_history(self) -> bool:
        return (
            self._prev_right_wrist is not None
            or self._prev_left_wrist is not None
        )

    def _has_both_tracking(self) -> bool:
        """True when we have wrist history for both slots."""
        return (
            self._prev_right_wrist is not None
            and self._prev_left_wrist is not None
        )

    def _nearest_slot(self, wrist: np.ndarray) -> str:
        """Return the slot whose previous wrist position is closest."""
        d_right = float("inf")
        d_left = float("inf")
        if self._prev_right_wrist is not None:
            d_right = float(np.linalg.norm(wrist - self._prev_right_wrist))
        if self._prev_left_wrist is not None:
            d_left = float(np.linalg.norm(wrist - self._prev_left_wrist))
        return "right" if d_right <= d_left else "left"

    def _match_pair_to_previous(
        self, hand_a: dict, hand_b: dict
    ) -> dict[str, np.ndarray | None]:
        """Assign two hands to left/right using a 2x2 cost matrix against
        previous wrist positions (optimal assignment by exhaustive check)."""
        result: dict[str, np.ndarray | None] = {"left": None, "right": None}

        wa = hand_a["landmarks"][WRIST_IDX]
        wb = hand_b["landmarks"][WRIST_IDX]

        prev_r = self._prev_right_wrist
        prev_l = self._prev_left_wrist

        cost_ar_bl = 0.0
        cost_al_br = 0.0
        if prev_r is not None:
            cost_ar_bl += float(np.linalg.norm(wa - prev_r))
            cost_al_br += float(np.linalg.norm(wb - prev_r))
        if prev_l is not None:
            cost_ar_bl += float(np.linalg.norm(wb - prev_l))
            cost_al_br += float(np.linalg.norm(wa - prev_l))

        if cost_ar_bl <= cost_al_br:
            result["right"] = hand_a["landmarks"]
            result["left"] = hand_b["landmarks"]
        else:
            result["left"] = hand_a["landmarks"]
            result["right"] = hand_b["landmarks"]

        return result

    def _update_tracking(self, result: dict[str, np.ndarray | None]) -> None:
        """Persist wrist positions for the next frame's matching."""
        if result["right"] is not None:
            self._prev_right_wrist = result["right"][WRIST_IDX].copy()
        else:
            self._prev_right_wrist = None
        if result["left"] is not None:
            self._prev_left_wrist = result["left"][WRIST_IDX].copy()
        else:
            self._prev_left_wrist = None

    # ── De-duplication (overlapping crops) ────────────────────────────────

    @staticmethod
    def _deduplicate_hands(hands: list[dict]) -> list[dict]:
        """Remove duplicate detections whose wrists are very close."""
        if len(hands) <= 2:
            return hands

        unique: list[dict] = [hands[0]]
        for h in hands[1:]:
            wrist = h["landmarks"][WRIST_IDX]
            is_dup = any(
                np.linalg.norm(wrist - u["landmarks"][WRIST_IDX]) < _DEDUP_WRIST_DIST
                for u in unique
            )
            if not is_dup:
                unique.append(h)

        return unique[:NUM_HANDS]

    # ── Dummy fallback ───────────────────────────────────────────────────

    @staticmethod
    def _dummy_track(bgr_frame: np.ndarray) -> dict[str, np.ndarray | None]:
        """Deterministic dummy landmarks – returns both hands for testing."""
        rng = np.random.RandomState(42)
        right = 0.35 + 0.08 * rng.randn(NUM_HAND_JOINTS, 2).astype(np.float32)
        left = 0.65 + 0.08 * rng.randn(NUM_HAND_JOINTS, 2).astype(np.float32)
        right = np.clip(right, 0.0, 1.0)
        left = np.clip(left, 0.0, 1.0)
        return {"left": left, "right": right}


# ── Drawing utilities ────────────────────────────────────────────────────────


def draw_hands(
    bgr_frame: np.ndarray,
    hands: dict[str, np.ndarray | None],
    point_radius: int = 3,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw detected hand skeletons onto *bgr_frame* (mutates in-place)."""
    h, w = bgr_frame.shape[:2]

    for side, landmarks in hands.items():
        if landmarks is None:
            continue
        colour = _RIGHT_COLOUR if side == "right" else _LEFT_COLOUR
        pts = (landmarks * [w, h]).astype(int)

        for a, b in HAND_CONNECTIONS:
            cv2.line(bgr_frame, tuple(pts[a]), tuple(pts[b]), colour, line_thickness)

        for x, y in pts:
            cv2.circle(bgr_frame, (x, y), point_radius, _POINT_COLOUR, -1)

        label = f"{side.upper()}"
        wrist = tuple(pts[WRIST_IDX])
        cv2.putText(
            bgr_frame, label,
            (wrist[0] + 5, wrist[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )

    return bgr_frame


# ── Feature extraction for the classifier ────────────────────────────────────


def hands_to_feature_vector(
    hands: dict[str, np.ndarray | None],
) -> np.ndarray:
    """Flatten both hands into a single (84,) vector centred on each wrist.

    Missing hands are zero-filled.  Output order: [right_42, left_42].
    """
    parts: list[np.ndarray] = []
    for side in ("right", "left"):
        lm = hands[side]
        if lm is not None:
            centred = lm - lm[0:1]  # subtract wrist
            parts.append(centred.flatten())  # 42 values
        else:
            parts.append(np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32))
    return np.concatenate(parts)  # (84,)


def normalize_hand_rotation(
    landmarks: np.ndarray,
) -> np.ndarray:
    """Rotate-normalise a (21, 2) hand so the middle finger points upward.

    Ported from Multi-HandTrackingGPU's ``process_keypoints.normalize_keypoints``.
    Centres on wrist, scales by palm length (wrist→middle-MCP), rotates so
    the wrist→middle-MCP vector aligns with (0, -1).

    Useful for building rotation-invariant feature vectors for the classifier.
    """
    wrist = landmarks[WRIST_IDX]
    middle_mcp = landmarks[MIDDLE_MCP_IDX]

    centred = landmarks - wrist
    palm_len = float(np.linalg.norm(middle_mcp - wrist))
    if palm_len < 1e-6:
        return centred

    normed = centred / palm_len

    direction = normed[MIDDLE_MCP_IDX]
    target = np.array([0.0, -1.0])
    cos_a = float(np.dot(direction, target))
    sin_a = float(direction[0] * target[1] - direction[1] * target[0])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    return (normed @ rot.T).astype(np.float32)


def hands_to_rotation_invariant_vector(
    hands: dict[str, np.ndarray | None],
) -> np.ndarray:
    """Like :func:`hands_to_feature_vector` but rotation-normalised.

    Output shape: ``(84,)``  — same dimension, so it is a drop-in
    replacement for the LSTM or heuristic classifier.
    """
    parts: list[np.ndarray] = []
    for side in ("right", "left"):
        lm = hands[side]
        if lm is not None:
            parts.append(normalize_hand_rotation(lm).flatten())
        else:
            parts.append(np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32))
    return np.concatenate(parts)
