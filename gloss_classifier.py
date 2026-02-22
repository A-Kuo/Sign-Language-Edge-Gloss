"""
src/gloss_classifier.py
────────────────────────
Stage 2: Keypoint buffer → Gloss token

Architecture
─────────────
MVP (default): Distance-based heuristic matcher.
  Computes hand-relative joint angles and wrist-to-landmark distances
  from the mean frame of the 30-frame buffer, then nearest-neighbour
  matches against hand-crafted prototype vectors.

Upgrade path (post-hackathon): Swap to the LSTMClassifier below,
  which treats the (T=30, D=42) hand-keypoint sequence as a time-series
  classification problem — a natural fit for a 1-D temporal model.
  The LSTM hidden state h_T is the "compressed gesture embedding" that
  feeds a linear head for gloss prediction.

Maths note
───────────
Input tensor per call: X ∈ ℝ^{T×D}  where T=30 frames, D=42 (21 left-hand
  joints × 2 + 21 right-hand joints × 2).  We normalise relative to the
  wrist anchor so the classifier is translation-invariant.

  x_norm[t, j] = x[t, j] - x[t, wrist]   (centering)

The LSTM computes:
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    forget gate
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    input  gate
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c · [h_{t-1}, x_t] + b_c)
  h_t = o_t ⊙ tanh(c_t)

Final gloss = argmax(W_out · h_T + b_out)

For the hackathon we skip training the LSTM (no labelled data) and use the
heuristic matcher instead.  The LSTM class is included so you can drop in
a trained checkpoint on day two.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# ─── Joint index constants (in the 133-joint WholeBody schema) ──────────────
WRIST_LEFT     = 9    # COCO body — left  wrist
WRIST_RIGHT    = 10   # COCO body — right wrist
LH_START       = 91   # left  hand joints 0-20  → indices 91-111
RH_START       = 112  # right hand joints 0-20  → indices 112-132

HAND_DIM       = 42   # 21 joints × 2 coords, per hand → 2 hands = 84; we use 42 for one
SEQUENCE_LEN   = 30
NUM_GLOSSES    = 5    # expand as you add more signs

# ─── MVP gloss vocabulary ────────────────────────────────────────────────────
GLOSS_LABELS = ["HELLO", "WORLD", "HACKATHON", "HELP", "FINISH"]

# ─── Heuristic prototype vectors ─────────────────────────────────────────────
# Each prototype is a (42,) vector: 21 right-hand joints (x, y) centred on wrist.
# These are rough unit-normalised shapes — replace with real measured averages
# for higher accuracy.

def _flat(pairs: list[tuple[float, float]]) -> np.ndarray:
    return np.array(pairs, dtype=np.float32).ravel()

# Right-hand open flat (HELLO wave)
_HELLO = _flat([
    (0.00, 0.00),   # wrist anchor
    (-0.05, 0.15),  # thumb base ... (simplified — 20 more joints below)
    *[(0.0 + i*0.03, 0.20 + i*0.01) for i in range(19)]
])

# Right-hand fist (WORLD / default closed)
_WORLD = _flat([
    (0.00, 0.00),
    (0.04, 0.08),
    *[(0.02 + i*0.01, 0.05) for i in range(19)]
])

# Two-handed spread (HACKATHON — both hands wide)
_HACKATHON = _flat([
    (0.00,  0.00),
    (-0.12, 0.18),
    *[(-0.08 + i*0.02, 0.12) for i in range(19)]
])

# Palm out push (HELP)
_HELP = _flat([
    (0.00, 0.00),
    (0.02, 0.20),
    *[(0.01, 0.18 + i*0.02) for i in range(19)]
])

# Downward swipe (FINISH)
_FINISH = _flat([
    (0.00, 0.00),
    (0.05, -0.10),
    *[(0.03 + i*0.01, -0.08 - i*0.02) for i in range(19)]
])

PROTOTYPES: dict[str, np.ndarray] = {
    "HELLO":     _HELLO,
    "WORLD":     _WORLD,
    "HACKATHON": _HACKATHON,
    "HELP":      _HELP,
    "FINISH":    _FINISH,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Heuristic Matcher (MVP — no training required)
# ═══════════════════════════════════════════════════════════════════════════

class HeuristicMatcher:
    """Nearest-neighbour classifier in prototype space."""

    def __init__(self, threshold: float = 0.18):
        self.threshold = threshold

    def _extract_feature(self, kp_buffer: list[np.ndarray]) -> np.ndarray:
        """
        Collapse (T, 133, 2) buffer → (42,) feature vector.

        1. Average across time  → (133, 2) mean pose.
        2. Isolate right-hand   → (21, 2).
        3. Centre on wrist      → translation invariance.
        4. Scale by hand span   → scale invariance.
        5. Flatten              → (42,).
        """
        mean_pose = np.mean(kp_buffer, axis=0)           # (133, 2)
        rh = mean_pose[RH_START: RH_START + 21]          # (21, 2)

        wrist = rh[0]                                     # wrist at index 0
        rh_centred = rh - wrist                           # centre

        span = np.linalg.norm(rh_centred, axis=-1).max() + 1e-6
        rh_norm = rh_centred / span                       # scale-invariant

        return rh_norm.ravel().astype(np.float32)         # (42,)

    def predict(self, kp_buffer: list[np.ndarray]) -> str | None:
        feat = self._extract_feature(kp_buffer)

        best_gloss = None
        best_dist  = float("inf")

        for gloss, proto in PROTOTYPES.items():
            dist = float(np.linalg.norm(feat - proto))
            if dist < best_dist:
                best_dist  = dist
                best_gloss = gloss

        return best_gloss if best_dist < self.threshold else None


# ═══════════════════════════════════════════════════════════════════════════
#  LSTM Classifier (upgrade path — requires training data)
# ═══════════════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    """
    Temporal gesture classifier.

    Input : (batch, T=30, input_dim=84)   — both hands concatenated
    Output: (batch, num_classes)          — gloss logits

    Hidden state h_T encodes the full signing motion as a fixed-size
    embedding, analogous to a "sentence embedding" from NLP.
    """

    def __init__(
        self,
        input_dim:   int = 84,
        hidden_dim:  int = 128,
        num_layers:  int = 2,
        num_classes: int = NUM_GLOSSES,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, H)
        h_T = h_n[-1]                # take top-layer final hidden state
        return self.head(h_T)        # (B, num_classes)

    @torch.no_grad()
    def predict_gloss(self, kp_buffer: list[np.ndarray]) -> str | None:
        """
        Extract both-hand feature sequence → predict gloss.
        Returns None if below confidence threshold.
        """
        seq = self._build_sequence(kp_buffer)          # (T, 84)
        x   = torch.tensor(seq).unsqueeze(0)           # (1, T, 84)
        logits = self(x)                               # (1, C)
        probs  = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        if conf.item() < 0.60:
            return None
        return GLOSS_LABELS[idx.item()]

    def _build_sequence(self, kp_buffer: list[np.ndarray]) -> np.ndarray:
        seq = []
        for kp in kp_buffer:                          # kp: (133, 2)
            lh = (kp[LH_START: LH_START + 21] - kp[WRIST_LEFT]).ravel()
            rh = (kp[RH_START: RH_START + 21] - kp[WRIST_RIGHT]).ravel()
            seq.append(np.concatenate([lh, rh]))      # (84,)
        return np.array(seq, dtype=np.float32)        # (T, 84)


# ═══════════════════════════════════════════════════════════════════════════
#  Public interface used by main.py
# ═══════════════════════════════════════════════════════════════════════════

class GlossClassifier:
    """
    Façade over either the heuristic or LSTM backend.

    Usage
    ─────
        clf = GlossClassifier()                        # heuristic MVP
        clf = GlossClassifier(checkpoint="model.pt")   # trained LSTM
    """

    def __init__(self, checkpoint: str | None = None, threshold: float = 0.18):
        if checkpoint:
            print(f"[GlossClassifier] Loading LSTM from {checkpoint}")
            self._backend = LSTMClassifier()
            state = torch.load(checkpoint, map_location="cpu")
            self._backend.load_state_dict(state)
            self._backend.eval()
            self._mode = "lstm"
        else:
            print("[GlossClassifier] Using heuristic matcher (MVP mode).")
            self._backend = HeuristicMatcher(threshold=threshold)
            self._mode    = "heuristic"

    def classify(self, kp_buffer: list[np.ndarray]) -> str | None:
        """
        Parameters
        ----------
        kp_buffer : list of (133, 2) float32 arrays, length == SEQUENCE_LEN

        Returns
        -------
        Gloss string (e.g. "HELLO") or None if no confident match.
        """
        if len(kp_buffer) < SEQUENCE_LEN:
            return None

        if self._mode == "lstm":
            return self._backend.predict_gloss(kp_buffer)
        else:
            return self._backend.predict(kp_buffer)
