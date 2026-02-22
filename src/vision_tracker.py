"""
vision_tracker.py – Qualcomm AI Hub MediaPipe Hand wrapper.

Extracts 21-joint hand landmarks per detected hand from BGR webcam frames.
Falls back to deterministic dummy keypoints when ``qai_hub_models`` is not
installed so the rest of the pipeline can be tested end-to-end.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

NUM_HAND_JOINTS = 21
NUM_HANDS = 2  # left + right
FEATURE_DIM_PER_HAND = NUM_HAND_JOINTS * 2  # (x, y) per joint
FEATURE_DIM = FEATURE_DIM_PER_HAND * NUM_HANDS  # 84

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

# ── Try importing Qualcomm AI Hub ────────────────────────────────────────────

try:
    import torch
    from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
    from qai_hub_models.models.mediapipe_hand.model import MediaPipeHand

    _HAS_QAI = True
except ImportError:
    _HAS_QAI = False


# ── Public API ───────────────────────────────────────────────────────────────


class HandTracker:
    """Detect hands and return normalised 21-joint landmarks per frame.

    Attributes
    ----------
    mode : str
        ``"qai"`` when running through the Qualcomm pipeline,
        ``"dummy"`` when falling back to synthetic keypoints.
    """

    def __init__(self, score_threshold: float = 0.5) -> None:
        self.score_threshold = score_threshold

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
            return self._dummy_track(bgr_frame)
        if crops:
            return self._qai_track_crops(bgr_frame, crops)
        return self._qai_track(bgr_frame)

    # ── Qualcomm path ────────────────────────────────────────────────────

    def _qai_track(self, bgr_frame: np.ndarray) -> dict[str, np.ndarray | None]:
        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        raw = self.app.predict_landmarks_from_image(rgb_frame, raw_output=True)
        # raw layout (from MediaPipeHandApp):
        #   [0] batched_selected_boxes        list[Tensor]
        #   [1] batched_selected_keypoints    list[Tensor]
        #   [2] batched_roi_4corners          list[Tensor]
        #   [3] batched_selected_landmarks    list[Tensor]  shape (N, 21, 3)
        #   [4] batched_is_right_hand         list[list[bool]]

        batched_landmarks = raw[3]
        batched_is_right = raw[4]

        result: dict[str, np.ndarray | None] = {"left": None, "right": None}

        if len(batched_landmarks) == 0:
            return result

        landmarks_t = batched_landmarks[0]  # first (only) batch element
        if (
            not isinstance(landmarks_t, torch.Tensor)
            or landmarks_t.nelement() == 0
        ):
            return result

        is_right_list = batched_is_right[0]

        for i, is_right in enumerate(is_right_list):
            lm = landmarks_t[i].cpu().numpy()  # (21, 3)  x_px, y_px, conf
            xy = lm[:, :2].copy()
            xy[:, 0] /= w
            xy[:, 1] /= h

            key = "right" if is_right else "left"
            if result[key] is None:  # keep first detection per side
                result[key] = xy.astype(np.float32)

        return result

    # ── Qualcomm path with YOLO crops ─────────────────────────────────────

    def _qai_track_crops(
        self,
        bgr_frame: np.ndarray,
        crops: list[tuple[int, int, int, int]],
    ) -> dict[str, np.ndarray | None]:
        """Run MediaPipe on each crop and merge landmarks into full-frame [0,1]."""
        fh, fw = bgr_frame.shape[:2]
        result: dict[str, np.ndarray | None] = {"left": None, "right": None}

        for (x1, y1, x2, y2) in crops:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = bgr_frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]
            hands_crop = self._qai_track(crop)  # [0,1] in crop space

            for side in ("left", "right"):
                if result[side] is not None:
                    continue
                lm = hands_crop.get(side)
                if lm is None:
                    continue
                # Map from crop [0,1] to full-frame [0,1]
                full_x = (x1 + lm[:, 0] * crop_w) / fw
                full_y = (y1 + lm[:, 1] * crop_h) / fh
                result[side] = np.stack([full_x, full_y], axis=1).astype(np.float32)

        return result

    # ── Dummy fallback ───────────────────────────────────────────────────

    @staticmethod
    def _dummy_track(bgr_frame: np.ndarray) -> dict[str, np.ndarray | None]:
        """Deterministic dummy landmarks centred in the frame."""
        rng = np.random.RandomState(42)
        right = 0.5 + 0.08 * rng.randn(NUM_HAND_JOINTS, 2).astype(np.float32)
        right = np.clip(right, 0.0, 1.0)
        return {"left": None, "right": right}


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
