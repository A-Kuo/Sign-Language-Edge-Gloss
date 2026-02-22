# EdgeGloss — Sign Language Edge Translator: Specification

## 1. Overview

EdgeGloss is a real-time, on-device sign language translation pipeline that:

1. **Captures** video frames from a camera.
2. **Tracks** body / hand / face key-points using RTMPose.
3. **Classifies** sequences of key-points into sign-language *glosses* (isolated signs) using a heuristic rule-engine and / or an LSTM model.
4. **Translates** gloss sequences into fluent natural-language sentences using a quantized LLM (llama.cpp or a HuggingFace transformers model).

The full pipeline runs on edge hardware (e.g., a Qualcomm SoC) with no cloud dependency.

---

## 2. System Architecture

```
Camera → VisionTracker → GlossClassifier → LLMTranslator → Output
```

| Module              | File                        | Responsibility                              |
|---------------------|-----------------------------|---------------------------------------------|
| Orchestrator        | `main.py`                   | Frame loop, module wiring, CLI entry-point  |
| Vision Tracker      | `src/vision_tracker.py`     | RTMPose wrapper, key-point extraction       |
| Gloss Classifier    | `src/gloss_classifier.py`   | Heuristic rules + LSTM sequence classifier  |
| LLM Translator      | `src/llm_translator.py`     | Gloss → sentence via llama.cpp / transformers|

---

## 3. Module Specifications

### 3.1 VisionTracker (`src/vision_tracker.py`)

- **Input:** BGR video frame (`numpy.ndarray`, shape `[H, W, 3]`).
- **Output:** Dictionary of key-point arrays:
  - `pose` — 17 body key-points (COCO format)
  - `left_hand` — 21 hand key-points
  - `right_hand` — 21 hand key-points
- **Backend:** RTMPose via MMPose; falls back to MediaPipe stub when MMPose is unavailable.
- **Configuration:** Model paths and confidence thresholds read from `config.yaml`.

### 3.2 GlossClassifier (`src/gloss_classifier.py`)

- **Input:** Sliding window of key-point dictionaries (length configurable, default 30 frames).
- **Output:** Gloss label string (e.g., `"HELLO"`) or `None` when confidence is below threshold.
- **Stage 1 — Heuristic engine:** Fast hand-shape / motion rules (no model required at startup).
- **Stage 2 — LSTM model:** Trained on key-point sequences; loaded lazily on first use.
- **Model format:** PyTorch `state_dict` (`.pt` file), path configurable.

### 3.3 LLMTranslator (`src/llm_translator.py`)

- **Input:** List of gloss strings (e.g., `["I", "WANT", "COFFEE"]`).
- **Output:** Natural-language sentence string (e.g., `"I want a coffee."`).
- **Backends (priority order):**
  1. `llama.cpp` via `llama-cpp-python` — preferred for edge deployment.
  2. HuggingFace `transformers` — fallback / development stub.
  3. Echo stub — returns glosses joined by spaces (used when no model is present).
- **Prompt template:** `"Translate the following ASL gloss sequence to English: {glosses}"`

---

## 4. Orchestrator Loop (`main.py`)

```
while camera is open:
    frame  = camera.read()
    kpts   = tracker.process(frame)
    gloss  = classifier.update(kpts)
    if gloss:
        sentence = translator.translate(accumulated_glosses)
        display(sentence)
```

CLI flags:

| Flag           | Default | Description                          |
|----------------|---------|--------------------------------------|
| `--camera`     | `0`     | Camera device index or video path    |
| `--config`     | `config.yaml` | Path to YAML configuration file |
| `--no-display` | `False` | Run headless (no OpenCV window)      |

---

## 5. Configuration (`config.yaml`)

```yaml
vision_tracker:
  model: rtmpose-m
  confidence: 0.3

gloss_classifier:
  window_size: 30
  lstm_model: models/gloss_lstm.pt
  confidence: 0.7

llm_translator:
  backend: llama_cpp          # llama_cpp | transformers | stub
  model_path: models/llm.gguf
  max_tokens: 64
```

---

## 6. Non-Functional Requirements

- Minimum throughput: **15 FPS** on Qualcomm Snapdragon 8 Gen 2 (or equivalent).
- Peak RAM: **< 2 GB** (LSTM + quantized LLM Q4).
- No internet access required at runtime.
- Python ≥ 3.10.
