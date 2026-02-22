# EdgeGloss — Project Specification

> 100% on-device, real-time sign language translation.
> Offline, private, low-latency.  No cloud calls.

---

## Pipeline (Cascading Crop Architecture)

```
Webcam  ──►  YOLOv8           ──►  (Depth Anything V2)  ──►  MediaPipe Hands    ──►  Dictionary / LSTM  ──►  Gloss string
             (hand/face boxes)      (Z-depth per box)         (21 joints on crop)     (heuristic / LSTM)
```

### Key Insight — Cascading Crop

MediaPipe is **not** run on the full frame.  YOLO first finds tight
bounding boxes around hands (and optionally faces).  Only those crops
are passed to MediaPipe, which eliminates false positives and improves
landmark accuracy on small or distant hands.

## Stages

| # | Module | Input | Output |
|---|--------|-------|--------|
| 1 | `src/yolo_detector.py` | BGR frame | Bounding boxes `[(x1,y1,x2,y2, conf, class)]` |
| 2 | `src/depth_estimator.py` | BGR frame + boxes | Z-depth per box (future) |
| 3 | `src/mediapipe_tracker.py` | Cropped hand images | 21-joint landmarks (global coords) |
| 4 | `src/dictionary_logic.py` | Landmarks + Z-depth | Gloss string |
| 5 | `src/llm_translator.py` | Gloss buffer | English sentence (future) |

## Model Details

### YOLOv8 (Phase 1 — The Spotter)

- **Model**: YOLOv8n exported to ONNX
- **Runtime**: ONNX Runtime (CUDA provider → CPU fallback)
- **Input**: 640 x 640, normalised [0, 1]
- **Output**: Bounding boxes with class scores
- **Classes**: Configurable — default COCO-80 (`person` class 0);
  swap to a hand-specific model for tighter boxes.

### Depth Anything V2 (Phase 2 — The Scaler)

- Monocular depth estimation for Z-scaling
- Average depth inside each YOLO bounding box

### MediaPipe Hands (Phase 3 — The Surgeon)

- 21-joint hand skeleton (x, y, confidence)
- Run **only** on YOLO crops, not the full frame
- Crop-to-global coordinate remapping

### Classifier (Phase 4)

| Backend | When | Training |
|---------|------|----------|
| `HeuristicMatcher` | No checkpoint | Finger-extension ratios + nearest-neighbour |
| `LSTMClassifier` | `--lstm path/to/lstm.pt` | Supervised on `(T x 84, label)` pairs |

#### LSTM architecture

```
Input  ──►  LSTM(84 → 128, 2 layers, dropout=0.3)  ──►  FC(128 → C)  ──►  logits
            h_T (final hidden state)
```

**Feature vector per frame**: `84 = 21 joints x 2 (x,y) x 2 hands`

## Cross-Platform Strategy

| Machine | Mode | Model format |
|---------|------|--------------|
| AMD64 dev laptop | PyTorch / ONNX Runtime (CUDA) | `.onnx` |
| ARM / Snapdragon | NPU accelerated | `.tflite` / `.onnx` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download / export YOLOv8n ONNX model
python -c "from src.yolo_detector import ensure_model; ensure_model()"

# 3. Run Phase 1 visual test
python tests/test_yolo.py

# 4. Run full pipeline (heuristic classifier)
python main.py

# 5. Headless mode
python main.py --no-display

# 6. With a trained LSTM checkpoint
python main.py --lstm models/lstm.pt
```
