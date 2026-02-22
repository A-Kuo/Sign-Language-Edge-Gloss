"""
vision_tracker.py – RTMPose wrapper for real-time keypoint extraction.

Wraps a camera capture loop and an RTMPose inference backend to produce
per-frame skeleton keypoints consumed by GlossClassifier.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np


class VisionTracker:
    """Capture video frames and extract body/hand keypoints via RTMPose.

    Usage::

        tracker = VisionTracker(camera_index=0)
        tracker.start()
        try:
            while True:
                data = tracker.get_keypoints()
                if data is not None:
                    process(data)
        finally:
            tracker.stop()

    The keypoints dict has the shape::

        {
            "timestamp": float,
            "keypoints": {
                "body":       List[List[float]],   # 17 COCO joints, each [x, y, score]
                "left_hand":  List[List[float]],   # 21 hand joints, each [x, y, score]
                "right_hand": List[List[float]],
            }
        }
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        model_config: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[Dict] = None
        self._lock = threading.Lock()

        # TODO: initialise RTMPose model
        # from mmpose.apis import init_model
        # self._model = init_model(model_config, model_checkpoint, device="cpu")
        self._model = None  # placeholder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera and begin the capture/inference loop in a thread."""
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}"
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        # daemon=True so the thread does not block process exit;
        # stop() should be called explicitly to release camera resources.
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture loop and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def get_keypoints(self, timeout: float = 1.0) -> Optional[Dict]:
        """Return the most recent keypoints dict, or *None* if not yet ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            time.sleep(0.005)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            keypoints = self._infer(frame)
            with self._lock:
                self._latest = {
                    "timestamp": time.time(),
                    "keypoints": keypoints,
                }

    def _infer(self, frame: np.ndarray) -> Dict[str, List]:
        """Run RTMPose inference on *frame* and return raw keypoints.

        TODO: replace stub with real MMPose inference call::

            from mmpose.apis import inference_topdown
            results = inference_topdown(self._model, frame)
            # parse results into body / left_hand / right_hand lists

        Returns placeholder zeros until the model is wired up.
        """
        # Placeholder: return zero keypoints
        return {
            "body": [[0.0, 0.0, 0.0]] * 17,
            "left_hand": [[0.0, 0.0, 0.0]] * 21,
            "right_hand": [[0.0, 0.0, 0.0]] * 21,
        }
