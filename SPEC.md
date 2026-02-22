# EdgeGloss — Project Specification

> 100% on-device, real-time sign language translation.
> Offline, private, low-latency.  No cloud calls.

---

## Pipeline

```
Webcam  ──►  MediaPipe Hand  ──►  GlossClassifier  ──►  Gloss string
             (Qualcomm AI Hub)    (heuristic / LSTM)
             21 joints × 2 hands
```

## Stages

| # | Module | Input | Output |
|---|--------|-------|--------|
| 1 | `src/vision_tracker.py` | BGR frame | `{'left': (21,2), 'right': (21,2)}` normalised |
| 2 | `src/gloss_classifier.py` | 30-frame (T×84) feature buffer | Gloss string ("A", "B", …) |

## Model Details

**Hand Detection** uses the Qualcomm AI Hub `mediapipe_hand` pipeline:

1. **BlazePalm** detector → bounding boxes + 7 palm keypoints
2. **BlazeHandLandmark** regressor → 21 hand joints (x, y, confidence)
3. Handedness classification (left / right)

Landmarks are normalised to `[0, 1]` relative to frame size, then centred
on the wrist joint before being fed to the classifier.

**Feature vector per frame**: `84 = 21 joints × 2 (x,y) × 2 hands`

## Classifier Backends

| Backend | When | Training |
|---------|------|----------|
| `HeuristicMatcher` | No checkpoint | None — finger-extension ratios + nearest-neighbour |
| `LSTMClassifier` | `--lstm path/to/lstm.pt` | Supervised on `(T×84, label)` pairs |

### LSTM architecture

```
Input  ──►  LSTM(84 → 128, 2 layers, dropout=0.3)  ──►  FC(128 → C)  ──►  logits
            h_T (final hidden state)
```

## Cross-Platform Strategy

| Machine | Mode | Model format |
|---------|------|--------------|
| AMD64 dev laptop | PyTorch FP32 | `qai_hub_models` standard |
| ARM / Snapdragon | NPU accelerated | Compiled `.tflite` / `.onnx` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run (heuristic classifier, no training needed)
python main.py

# 3. Run headless (no OpenCV window)
python main.py --no-display

# 4. With a trained LSTM checkpoint
python main.py --lstm models/lstm.pt
```

## Upgrading to LSTM

1. Collect labelled `(keypoint_sequence, gloss)` pairs.
2. Train `LSTMClassifier` from `src/gloss_classifier.py`.
3. Save: `torch.save(model.state_dict(), "models/lstm.pt")`
4. Run: `python main.py --lstm models/lstm.pt`
