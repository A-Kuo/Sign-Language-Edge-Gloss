# Sign-Language-Edge-Gloss — System Specification

## Overview

**Sign-Language-Edge-Gloss** is a real-time, edge-deployed pipeline that:

1. Captures video frames from a camera.
2. Runs pose/hand-landmark estimation (RTMPose) to extract skeleton keypoints.
3. Classifies sequences of keypoints into *glosses* — discrete sign-language tokens.
4. Translates gloss sequences into fluent natural-language sentences using a local LLM.

All inference runs on-device (Qualcomm SoC target) without cloud connectivity.

---

## Architecture

```
Camera → VisionTracker → GlossClassifier → LLMTranslator → Text Output
              ↓                  ↓                 ↓
        Keypoints (JSON)   Gloss tokens      Natural sentence
```

### Components

| Module | File | Responsibility |
|--------|------|----------------|
| Orchestrator | `main.py` | Frame loop, wires components together |
| Vision Tracker | `src/vision_tracker.py` | RTMPose wrapper; outputs per-frame keypoints |
| Gloss Classifier | `src/gloss_classifier.py` | Heuristic rules + optional LSTM; maps keypoint windows → gloss tokens |
| LLM Translator | `src/llm_translator.py` | Converts gloss sequence → natural language via local LLM |

---

## Data Flow

### 1. Keypoint Schema (VisionTracker output)

```json
{
  "timestamp": 1700000000.123,
  "keypoints": {
    "body": [[x, y, score], ...],   // 17 COCO keypoints
    "left_hand": [[x, y, score], ...],  // 21 MediaPipe hand keypoints
    "right_hand": [[x, y, score], ...]
  }
}
```

### 2. Gloss Token (GlossClassifier output)

A string label, e.g. `"HELLO"`, `"THANK_YOU"`, `"I"`, `"WANT"`, `"WATER"`.

### 3. LLM Input / Output

- **Input**: space-separated gloss sequence, e.g. `"I WANT WATER"`
- **Output**: fluent sentence, e.g. `"I would like some water, please."`

---

## Configuration

All runtime settings are passed as keyword arguments or via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `FRAME_WIDTH` | `640` | Capture width in pixels |
| `FRAME_HEIGHT` | `480` | Capture height in pixels |
| `GLOSS_WINDOW` | `30` | Sliding window size (frames) for classifier |
| `GLOSS_CONFIDENCE` | `0.6` | Minimum classifier confidence to emit a gloss |
| `LLM_MODEL_PATH` | `models/llm` | Path to local LLM weights |
| `LLM_BACKEND` | `transformers` | Backend: `transformers` or `llama_cpp` |

---

## Interfaces

### VisionTracker

```python
tracker = VisionTracker(camera_index=0, width=640, height=480)
tracker.start()
frame_data = tracker.get_keypoints()  # blocking; returns dict or None
tracker.stop()
```

### GlossClassifier

```python
classifier = GlossClassifier(window_size=30, confidence_threshold=0.6)
gloss = classifier.update(keypoints_dict)  # returns str or None
```

### LLMTranslator

```python
translator = LLMTranslator(model_path="models/llm", backend="transformers")
sentence = translator.translate(["I", "WANT", "WATER"])  # returns str
```

---

## Deployment Target

- **Hardware**: Qualcomm Snapdragon (QNN SDK compatible)
- **OS**: Android / Linux on ARM
- **Inference**: SNPE / QNN delegate for RTMPose and LSTM; llama.cpp Q4 for LLM

---

## Future Work

- Replace heuristic gloss rules with a trained LSTM/Transformer classifier.
- Add SNPE/QNN quantization pass for vision and classifier models.
- Build a thin Android UI wrapper.
- Expand gloss vocabulary beyond the initial prototype set.
