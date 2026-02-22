# Sign-Language-Edge-Gloss

All six files pass syntax check and are ready. Here's a map of every decision made:

---

## EdgeGloss — What Was Built

### File structure
```
EdgeGloss/
├── main.py                   ← orchestrator loop
├── requirements.txt
├── docs/SPEC.md              ← paste this to GitHub first
└── src/
    ├── __init__.py
    ├── vision_tracker.py     ← RTMPose wrapper
    ├── gloss_classifier.py   ← heuristic + LSTM classifier
    └── llm_translator.py     ← llama.cpp / transformers / stub
```

---

### Key design decisions

EdgeGloss stays 100% on-device: MediaPipe / Qualcomm RTMPose-Body2d runs locally to read hand and body keypoints, we map ESL alphabet or phrase gestures via a pre-trained/existing gesture dictionary, and the output is translated to natural language without any cloud calls.

**`vision_tracker.py`** — dual-mode from day one. If `qai_hub_models` isn't installed it drops to a dummy keypoint generator so the rest of the pipeline can be tested end-to-end without the Qualcomm package. Draws both the COCO-17 body skeleton and 21-joint hand skeletons in different colours so you can visually confirm detection.

**`gloss_classifier.py`** — two backends in one file, which maps directly to how you'd do this in production, both backed by the ESL alphabet/phrase dictionary:
- **HeuristicMatcher** (MVP): Collapses the 30-frame buffer to a single mean pose, centres and normalises the right-hand joints, then does nearest-neighbour against prototype vectors. No training needed.
- **LSTMClassifier** (upgrade path): Input tensor is X ∈ ℝ^{T×D} where T=30 frames and D=84 (both hands centred on their wrist anchors). The final hidden state h_T is the gesture embedding — identical to how you'd encode a sentence in an NLP classifier. Swap in by passing `checkpoint="models/lstm.pt"` to `GlossClassifier`.

**`llm_translator.py`** — priority-ordered backend detection: llama.cpp GGUF → HuggingFace transformers → stub passthrough. The GGUF Q4_K_M format is the same NF4-style 4-bit quantisation you used in your QLoRA financial project, so the mental model is identical — 16-element codebook per block, per-block scale factor, ~50% weight compression with <1% perplexity hit. Future step: use the local LLM to forecast likely glosses/words a few beats ahead of visible hand signs to keep latency low while staying offline.

**`main.py`** — the pause detection uses normalised L2 distance across joints divided by frame diagonal so the `STILLNESS_THRESHOLD=0.012` constant is resolution-independent, not pixel-dependent.

---

### First command to run
```bash
python main.py --no-llm   # verify webcam + skeleton before loading the LLM
```

Then for the LLM once you have the GGUF downloaded:
```bash
export LLAMA_GGUF_PATH=models/Meta-Llama-3.2-3B-Instruct-Q4_K_M.gguf
python main.py
```
