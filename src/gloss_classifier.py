"""
gloss_classifier.py – Map hand-landmark sequences to sign-language glosses.

Two backends in one file:

* **HeuristicMatcher** (MVP, no training)
    Collapses a T-frame buffer to a mean pose, computes per-finger extension
    ratios, and does nearest-neighbour against prototype vectors for the
    ASL / ESL manual alphabet.

* **LSTMClassifier** (upgrade path)
    ``X ∈ ℝ^{T×D}`` where *T* = 30 frames and *D* = 84 (both hands centred
    on their wrist anchors).  The final hidden state ``h_T`` is the gesture
    embedding, projected to class logits.

The ``GlossClassifier`` façade picks the right backend automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.vision_tracker import FEATURE_DIM, NUM_HAND_JOINTS

# ── Constants ────────────────────────────────────────────────────────────────

BUFFER_LEN = 30  # frames (~1 s @ 30 fps)

# Fingertip and MCP indices (MediaPipe 21-point convention)
_FINGERTIP_IDX = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
_FINGER_MCP_IDX = [2, 5, 9, 13, 17]   # corresponding MCP joints
WRIST_IDX = 0

# ── ASL static-alphabet prototypes ──────────────────────────────────────────
# Each entry maps a gloss string to a 5-element binary vector:
#   [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
# 1 = finger extended,  0 = finger curled.
# These capture the dominant posture for the 26 ASL manual-alphabet letters
# plus a few common glosses.  Only the *right hand* is checked in the MVP.

ASL_ALPHABET: dict[str, list[int]] = {
    "A": [0, 0, 0, 0, 0],  # fist, thumb across fingers
    "B": [0, 1, 1, 1, 1],  # open palm, thumb tucked
    "C": [1, 1, 1, 1, 1],  # curved hand (all partly extended)
    "D": [0, 1, 0, 0, 0],  # index up, rest curled
    "E": [0, 0, 0, 0, 0],  # curled fingers, thumb across  (same as A in this coarse encoding)
    "F": [1, 0, 1, 1, 1],  # OK-ring: thumb+index circle, rest up
    "G": [1, 1, 0, 0, 0],  # thumb + index point sideways
    "H": [0, 1, 1, 0, 0],  # index + middle horizontal
    "I": [0, 0, 0, 0, 1],  # pinky up
    "K": [1, 1, 1, 0, 0],  # index + middle + thumb
    "L": [1, 1, 0, 0, 0],  # L-shape: thumb + index
    "O": [1, 1, 1, 1, 1],  # all tips touching (similar to C)
    "R": [0, 1, 1, 0, 0],  # index + middle crossed
    "U": [0, 1, 1, 0, 0],  # index + middle together
    "V": [0, 1, 1, 0, 0],  # peace sign
    "W": [0, 1, 1, 1, 0],  # three fingers
    "Y": [1, 0, 0, 0, 1],  # hang-loose: thumb + pinky
}

# Numpy prototype matrix (num_signs × 5)
_PROTO_NAMES = list(ASL_ALPHABET.keys())
_PROTO_MATRIX = np.array(list(ASL_ALPHABET.values()), dtype=np.float32)

# ── Heuristic helpers ────────────────────────────────────────────────────────


def _finger_extension_ratios(landmarks: np.ndarray) -> np.ndarray:
    """Compute per-finger extension ratio for a (21, 2) hand landmark array.

    Extension ratio = dist(wrist, tip) / dist(wrist, mcp).
    Returns a (5,) array in [0, ∞).  A ratio > ~1.5 generally means extended.
    """
    wrist = landmarks[WRIST_IDX]
    tip_dists = np.linalg.norm(landmarks[_FINGERTIP_IDX] - wrist, axis=1)
    mcp_dists = np.linalg.norm(landmarks[_FINGER_MCP_IDX] - wrist, axis=1)
    # Avoid division by zero
    mcp_dists = np.maximum(mcp_dists, 1e-6)
    return tip_dists / mcp_dists


def _ratios_to_binary(ratios: np.ndarray, threshold: float = 1.4) -> np.ndarray:
    """Threshold extension ratios to binary extended / curled."""
    return (ratios > threshold).astype(np.float32)


# ── HeuristicMatcher ─────────────────────────────────────────────────────────


class HeuristicMatcher:
    """Nearest-neighbour against ASL alphabet prototypes.

    Works on static (held) signs only.  Collapse the frame buffer to a
    mean pose, extract the finger-extension binary vector, and compare
    against :data:`_PROTO_MATRIX` using L2 distance.
    """

    def __init__(self, threshold: float = 1.4, max_dist: float = 2.0) -> None:
        self.threshold = threshold
        self.max_dist = max_dist  # reject if nearest neighbour is too far

    def classify(self, buffer: np.ndarray) -> str | None:
        """Classify a landmark buffer.

        Parameters
        ----------
        buffer : np.ndarray
            Shape ``(T, 84)`` – the recent frame feature vectors.

        Returns
        -------
        str or None
            The matched gloss string, or ``None`` if no confident match.
        """
        # Extract right-hand portion (first 42 values → reshape to 21×2)
        right_buf = buffer[:, :42].reshape(-1, NUM_HAND_JOINTS, 2)

        # Mean pose across time (ignore frames where hand was missing = all zeros)
        nonzero_mask = np.any(right_buf.reshape(right_buf.shape[0], -1) != 0, axis=1)
        if not nonzero_mask.any():
            return None
        mean_pose = right_buf[nonzero_mask].mean(axis=0)  # (21, 2)

        ratios = _finger_extension_ratios(mean_pose)
        binary = _ratios_to_binary(ratios, self.threshold)

        dists = np.linalg.norm(_PROTO_MATRIX - binary, axis=1)
        best_idx = int(np.argmin(dists))

        if dists[best_idx] > self.max_dist:
            return None
        return _PROTO_NAMES[best_idx]


# ── LSTMClassifier ───────────────────────────────────────────────────────────


class LSTMClassifier(nn.Module):
    """Sequence classifier: ``(batch, T, 84) → (batch, num_classes)``.

    The final hidden state of the top LSTM layer is projected to class logits.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = len(_PROTO_NAMES),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.class_names = list(_PROTO_NAMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, T, D)`` where ``T`` = sequence length and
            ``D`` = 84 (both hands, wrist-centred, xy).

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        # h_n shape: (num_layers, batch, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # last layer hidden → logits


# ── GlossClassifier façade ───────────────────────────────────────────────────


class GlossClassifier:
    """Unified interface that selects the backend automatically.

    Parameters
    ----------
    checkpoint : str or Path or None
        Path to a saved ``LSTMClassifier`` state-dict.  If provided the LSTM
        backend is used; otherwise the heuristic matcher runs.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        if checkpoint is not None:
            ckpt = Path(checkpoint)
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            self.lstm = LSTMClassifier()
            self.lstm.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.lstm.to(self.device)
            self.lstm.eval()
            self._backend = "lstm"
            print(f"[GlossClassifier] LSTM loaded from {ckpt}")
        else:
            self.lstm = None
            self.heuristic = HeuristicMatcher()
            self._backend = "heuristic"

    @property
    def backend(self) -> str:
        return self._backend

    def classify(self, buffer: np.ndarray) -> str | None:
        """Classify a sequence of feature vectors.

        Parameters
        ----------
        buffer : np.ndarray
            Shape ``(T, 84)`` – the recent wrist-centred feature vectors.

        Returns
        -------
        str or None
            Predicted gloss, or ``None`` if confidence is too low.
        """
        if self._backend == "lstm":
            return self._lstm_classify(buffer)
        return self.heuristic.classify(buffer)

    def _lstm_classify(self, buffer: np.ndarray) -> str | None:
        assert self.lstm is not None
        x = torch.from_numpy(buffer).unsqueeze(0).to(self.device)  # (1, T, 84)
        with torch.no_grad():
            logits = self.lstm(x)  # (1, C)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)

        if conf.item() < 0.3:  # low-confidence rejection
            return None
        return self.lstm.class_names[idx.item()]
