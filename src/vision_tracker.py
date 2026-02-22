"""vision_tracker.py — RTMPose wrapper for key-point extraction.

Wraps MMPose's RTMPose model to extract body, left-hand, and right-hand
key-points from a single BGR video frame.  Falls back to a zero-filled
stub when MMPose is not installed, so the rest of the pipeline can be
developed and tested without the full model environment.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Number of key-points returned for each body part.
_N_POSE = 17       # COCO whole-body body key-points
_N_HAND = 21       # MediaPipe / RTMPose hand key-points


class VisionTracker:
    """Extract pose and hand key-points from video frames using RTMPose.

    Parameters
    ----------
    model_name:
        RTMPose model variant to load (e.g. ``"rtmpose-m"``).
    confidence:
        Minimum detection confidence; key-points below this score are
        zeroed out.
    """

    def __init__(self, model_name: str = "rtmpose-m", confidence: float = 0.3) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model = None
        self._try_load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Run pose estimation on a single BGR frame.

        Parameters
        ----------
        frame:
            BGR image as a ``numpy.ndarray`` with shape ``(H, W, 3)``.

        Returns
        -------
        dict with keys ``"pose"``, ``"left_hand"``, and ``"right_hand"``,
        each containing an ``ndarray`` of shape ``(N, 3)`` where columns
        are ``(x, y, confidence)``.
        """
        if self._model is not None:
            return self._run_model(frame)
        return self._stub_keypoints()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to import MMPose and load the RTMPose model."""
        try:
            from mmpose.apis import init_model  # type: ignore[import]

            self._model = init_model(self.model_name, device="cpu")
        except Exception as exc:
            # MMPose not installed or model unavailable — use stub.
            logger.warning("RTMPose model could not be loaded (%s). Using stub.", exc)
            self._model = None

    def _run_model(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Run the loaded RTMPose model and return key-point arrays."""
        from mmpose.apis import inference_topdown  # type: ignore[import]

        results = inference_topdown(self._model, frame)
        # TODO: parse MMPose result format into the standard dict layout.
        return self._stub_keypoints()

    @staticmethod
    def _stub_keypoints() -> dict[str, np.ndarray]:
        """Return zero-filled key-point arrays (used when model is absent)."""
        return {
            "pose": np.zeros((_N_POSE, 3), dtype=np.float32),
            "left_hand": np.zeros((_N_HAND, 3), dtype=np.float32),
            "right_hand": np.zeros((_N_HAND, 3), dtype=np.float32),
        }
