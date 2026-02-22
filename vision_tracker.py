"""
src/vision_tracker.py
─────────────────────
Wraps Qualcomm AI Hub's RTMPose-Body2d model.

On AMD64 dev machines  → runs in PyTorch FP32 mode (CPU/CUDA).
On ARM/NPU demo machines → swap MODEL_BACKEND to "onnx" and point
                           MODEL_PATH at your compiled .onnx binary.

RTMPose-Body2d outputs 133 keypoints (WholeBody):
  0–16  : body  (COCO-17)
  17–22 : feet
  23–90 : face
  91–132: hands (21 left + 21 right)

Each keypoint is (x, y) in pixel space.  We normalise to [0,1] before
feeding downstream so the classifier is resolution-agnostic.
"""

from __future__ import annotations

import numpy as np
import cv2

# ─── Try to import Qualcomm AI Hub model; fall back to dummy for dev ────────
try:
    from qai_hub_models.models.rtmpose_body2d import Model as RTMPoseModel
    _QAI_AVAILABLE = True
except ImportError:
    _QAI_AVAILABLE = False

# ─── Colour palette for skeleton drawing (BGR) ──────────────────────────────
_SKELETON_COLOR = (0, 220, 180)      # teal
_JOINT_COLOR    = (255, 80,  0)      # orange
_JOINT_RADIUS   = 4
_SKELETON_THICKNESS = 2

# COCO-17 body skeleton connections (joint index pairs)
_COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),            # head
    (5,6),(5,7),(7,9),(6,8),(8,10),     # shoulders / arms
    (5,11),(6,12),(11,12),              # torso
    (11,13),(13,15),(12,14),(14,16),    # legs
]

# ─── WholeBody 133-joint hand connections (relative to joint offset) ─────────
_LEFT_HAND_OFFSET  = 91
_RIGHT_HAND_OFFSET = 112
_HAND_SKELETON = [
    (0,1),(1,2),(2,3),(3,4),             # thumb
    (0,5),(5,6),(6,7),(7,8),             # index
    (0,9),(9,10),(10,11),(11,12),        # middle
    (0,13),(13,14),(14,15),(15,16),      # ring
    (0,17),(17,18),(18,19),(19,20),      # pinky
]

NUM_KEYPOINTS = 133


class VisionTracker:
    """
    Extracts 133 WholeBody keypoints from a single BGR frame.

    Returns
    -------
    keypoints : np.ndarray, shape (133, 2), dtype float32
        (x, y) normalised to [0,1] relative to frame dimensions.
    annotated_frame : np.ndarray
        Input frame with skeleton overlay drawn on it.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

        if _QAI_AVAILABLE:
            print("[VisionTracker] Loading RTMPose-Body2d from qai_hub_models …")
            self._model = RTMPoseModel.from_pretrained()
            self._model.eval()
            print("[VisionTracker] Model loaded.")
        else:
            print("[VisionTracker] WARNING: qai_hub_models not found.")
            print("                Running in DUMMY mode — random keypoints.")
            print("                Install with: pip install 'qai-hub-models[rtmpose-body2d]'")

    # ──────────────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Parameters
        ----------
        frame : np.ndarray  BGR image (H, W, 3)

        Returns
        -------
        (keypoints, annotated_frame)
        """
        h, w = frame.shape[:2]

        if self._model is not None:
            keypoints = self._run_rtmpose(frame, w, h)
        else:
            keypoints = self._dummy_keypoints()

        annotated = self._draw_skeleton(frame.copy(), keypoints, w, h)
        return keypoints, annotated

    # ──────────────────────────────────────────────────────────────────────
    def _run_rtmpose(self, frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Run the actual RTMPose inference.

        qai_hub_models expects a PIL Image or torch Tensor.
        We convert BGR→RGB, resize to model's expected input (192×256),
        normalise, run inference, then map back to original pixel space.
        """
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Model handles its own pre/post-processing internally
        with torch.no_grad():
            result = self._model(pil)

        # result is typically (keypoints, scores) — shape (1, 133, 2) / (1, 133)
        kp = result[0]  # (1, 133, 2) — pixel coords in model's output space
        if hasattr(kp, "numpy"):
            kp = kp.numpy()
        kp = kp.squeeze(0)  # (133, 2)

        # Normalise to [0,1]
        kp[:, 0] /= w
        kp[:, 1] /= h
        return kp.astype(np.float32)

    def _dummy_keypoints(self) -> np.ndarray:
        """
        Generates plausible random keypoints for unit-testing the pipeline
        without real hardware.  Keypoints cluster around the upper body region.
        """
        kp = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        # Body (0-16): centre upper half
        kp[:17, 0] = np.random.normal(0.5, 0.08, 17)
        kp[:17, 1] = np.random.normal(0.4, 0.12, 17)
        # Hands (91-132): around mid-frame
        kp[91:, 0]  = np.random.normal(0.5, 0.15, 42)
        kp[91:, 1]  = np.random.normal(0.6, 0.10, 42)
        return np.clip(kp, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────────
    def _draw_skeleton(self, frame: np.ndarray, kp: np.ndarray,
                       w: int, h: int) -> np.ndarray:
        """Draw joints and skeleton connections on the frame."""
        # De-normalise back to pixel space for drawing
        pts = (kp * np.array([w, h])).astype(int)   # (133, 2)

        # Body skeleton
        for i, j in _COCO_SKELETON:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]),
                         _SKELETON_COLOR, _SKELETON_THICKNESS, cv2.LINE_AA)

        # Both hands
        for offset in (_LEFT_HAND_OFFSET, _RIGHT_HAND_OFFSET):
            for i, j in _HAND_SKELETON:
                pi, pj = offset + i, offset + j
                if pi < len(pts) and pj < len(pts):
                    cv2.line(frame, tuple(pts[pi]), tuple(pts[pj]),
                             (180, 80, 240), 1, cv2.LINE_AA)

        # All joints
        for pt in pts:
            cv2.circle(frame, tuple(pt), _JOINT_RADIUS, _JOINT_COLOR, -1, cv2.LINE_AA)

        return frame
