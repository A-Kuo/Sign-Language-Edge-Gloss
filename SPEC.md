# EdgeGloss — Project Specification

> 100% on-device, real-time sign language translation.  
> Offline, private, low-latency.  No cloud calls.

---

## Pipeline

```
Webcam  ──►  RTMPose-Body2d  ──►  GlossClassifier  ──►  LLMTranslator  ──►  English
             (133 keypoints)       (LSTM / heuristic)    (Llama 3.2 3B)
```

## Stages

| # | Module | Input | Output |
|---|--------|-------|--------|
| 1 | `vision_tracker.py` | BGR frame | (133, 2) normalised keypoints |
| 2 | `gloss_classifier.py` | 30-frame keypoint buffer | Gloss string ("HELLO") |
| 3 | `llm_translator.py` | Gloss token list | English sentence |

## Cross-Platform Strategy

| Machine | Mode | Model format |
|---------|------|--------------|
| AMD64 dev laptop | PyTorch FP32 | `qai_hub_models` standard |
| ARM / Snapdragon demo | NPU accelerated | Compiled `.tflite` / `.onnx` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download quantised LLM
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Meta-Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/

# 3. Set path
export LLAMA_GGUF_PATH=models/Meta-Llama-3.2-3B-Instruct-Q4_K_M.gguf

# 4. Run
python main.py

# 5. Run without LLM (gloss-only, fastest)
python main.py --no-llm
```

## Upgrading Gloss Classifier to LSTM

1. Collect labelled (keypoint_sequence, gloss) pairs.
2. Train `LSTMClassifier` in `gloss_classifier.py`.
3. Save checkpoint: `torch.save(model.state_dict(), "models/lstm.pt")`
4. Load: `GlossClassifier(checkpoint="models/lstm.pt")`
