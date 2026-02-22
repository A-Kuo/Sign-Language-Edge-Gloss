# CLAUDE.md — EdgeGloss Project Rules

## Core Rules

1. **ONNX Runtime for all models** — Every neural network runs through
   ONNX Runtime with `CUDAExecutionProvider` (falls back to CPU when no
   GPU is available).  No framework-specific inference in production paths.

2. **Cascading Crop architecture** — YOLO detects bounding boxes first.
   Downstream models (Depth Anything, MediaPipe) operate only on those
   crops, never on the full frame.

3. **Visual test before moving on** — Every phase must have a standalone
   test script in `tests/` that produces a saved image (e.g.
   `yolo_test_output.jpg`) for human review before the next phase begins.

## Project Layout

```
docs/SPEC.md          — Full project specification
src/                  — Source modules
tests/                — Per-phase visual test scripts
models/               — ONNX model files (gitignored)
main.py               — Orchestrator / entry point
```

## Conventions

- All inference uses ONNX Runtime (`onnxruntime` / `onnxruntime-gpu`).
- Model files live in `models/` and are **not** checked into git.
- Test outputs (`*_test_output.jpg`) are saved in the repo root for review.
- BGR colour order throughout (OpenCV convention); convert to RGB only at
  model input boundaries.
