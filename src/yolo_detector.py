"""
yolo_detector.py – YOLOv8 hand & face detector via ONNX Runtime.

Loads a YOLOv8 ONNX model and runs inference with ONNX Runtime
(CUDAExecutionProvider when available, otherwise CPU).  Produces
bounding boxes with class IDs and confidence scores.

Usage
-----
    from src.yolo_detector import YOLODetector, ensure_model

    onnx_path = ensure_model()              # download + export if needed
    detector = YOLODetector(onnx_path)
    detections = detector.detect(bgr_frame)  # list[Detection]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_INPUT_SIZE = 640
_LETTERBOX_COLOR = (114, 114, 114)

# COCO 80 class names (used by YOLOv8n default model)
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class Detection:
    """Single object detection result."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


# ── Letterbox preprocessing ──────────────────────────────────────────────────


def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = _LETTERBOX_COLOR,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with padding to *new_shape* keeping aspect ratio.

    Returns
    -------
    padded : np.ndarray
        Resized + padded image.
    ratio : float
        Scale factor applied (min of h/w ratios).
    pad : (int, int)
        (pad_left, pad_top) in pixels.
    """
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    dw = (new_shape[1] - new_unpad_w) / 2
    dh = (new_shape[0] - new_unpad_h) / 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        image = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, r, (left, top)


# ── NMS ──────────────────────────────────────────────────────────────────────


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    """Greedy Non-Maximum Suppression on (N, 4) xyxy boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


# ── YOLODetector ─────────────────────────────────────────────────────────────


class YOLODetector:
    """YOLOv8 object detector backed by ONNX Runtime.

    Parameters
    ----------
    model_path : str or Path
        Path to a YOLOv8 ``.onnx`` file.
    conf_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        IoU threshold for NMS.
    target_classes : list[int] or None
        If set, only keep detections whose class ID is in this list.
        For COCO: ``[0]`` = person only.
    class_names : list[str] or None
        Human-readable names per class.  Defaults to COCO-80.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_classes: list[int] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.class_names = class_names or COCO_CLASSES

        # Pick the best available provider
        providers: list[str] = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        inp_shape = self.session.get_inputs()[0].shape
        self.input_h = inp_shape[2] if isinstance(inp_shape[2], int) else _DEFAULT_INPUT_SIZE
        self.input_w = inp_shape[3] if isinstance(inp_shape[3], int) else _DEFAULT_INPUT_SIZE

        active = self.session.get_providers()
        print(f"[YOLODetector] Loaded {model_path}")
        print(f"[YOLODetector] Providers: {active}")
        print(f"[YOLODetector] Input: {self.input_h}x{self.input_w}")

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        """Run YOLOv8 detection on a BGR image.

        Returns a list of :class:`Detection` objects in original-image
        pixel coordinates.
        """
        blob, ratio, pad = self._preprocess(bgr_image)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs[0], ratio, pad, bgr_image.shape[:2])

    # ── Internals ─────────────────────────────────────────────────────────

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        letterboxed, ratio, pad = _letterbox(
            image, (self.input_h, self.input_w)
        )
        # BGR → RGB, HWC → CHW, normalise to [0, 1], add batch dim
        blob = (
            letterboxed[:, :, ::-1]
            .transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
        )
        blob = np.ascontiguousarray(blob[np.newaxis])
        return blob, ratio, pad

    def _postprocess(
        self,
        raw: np.ndarray,
        ratio: float,
        pad: tuple[int, int],
        orig_hw: tuple[int, int],
    ) -> list[Detection]:
        # raw shape: [1, 4+C, N]  (e.g. [1, 84, 8400] for COCO-80)
        preds = raw[0].T  # [N, 4+C]

        boxes_cxcywh = preds[:, :4]
        scores_all = preds[:, 4:]

        class_ids = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(len(scores_all)), class_ids]

        # Confidence filter
        mask = confidences >= self.conf_threshold
        if self.target_classes is not None:
            mask &= np.isin(class_ids, self.target_classes)

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2
        boxes = np.empty_like(boxes_cxcywh)
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Map back to original image coordinates
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio

        # Clip to image bounds
        orig_h, orig_w = orig_hw
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        keep = _nms(boxes, confidences, self.iou_threshold)

        detections: list[Detection] = []
        for i in keep:
            cid = int(class_ids[i])
            detections.append(
                Detection(
                    x1=int(boxes[i, 0]),
                    y1=int(boxes[i, 1]),
                    x2=int(boxes[i, 2]),
                    y2=int(boxes[i, 3]),
                    confidence=float(confidences[i]),
                    class_id=cid,
                    class_name=(
                        self.class_names[cid]
                        if cid < len(self.class_names)
                        else f"class_{cid}"
                    ),
                )
            )
        return detections


# ── Drawing helper ───────────────────────────────────────────────────────────


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on *image* (mutates in-place)."""
    for det in detections:
        cv2.rectangle(image, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            image, (det.x1, det.y1 - th - 8), (det.x1 + tw + 4, det.y1), color, -1
        )
        cv2.putText(
            image, label, (det.x1 + 2, det.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    return image


# ── Model download / export helper ──────────────────────────────────────────


def ensure_model(model_dir: str | Path = "models") -> Path:
    """Download YOLOv8n and export to ONNX if not already present.

    Requires the ``ultralytics`` package (install-time only, not needed
    for inference).

    Returns the path to the ``.onnx`` file.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    onnx_path = model_dir / "yolov8n.onnx"

    if onnx_path.exists():
        print(f"[ensure_model] Already present: {onnx_path}")
        return onnx_path

    print("[ensure_model] Downloading YOLOv8n and exporting to ONNX ...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    exported = model.export(format="onnx", imgsz=640, simplify=True)

    # ultralytics returns the export path as a string
    src = Path(exported) if exported else Path("yolov8n.onnx")
    if src.exists() and src != onnx_path:
        src.rename(onnx_path)
    # Clean up the .pt if it was downloaded to cwd
    pt_file = Path("yolov8n.pt")
    if pt_file.exists():
        pt_file.unlink()

    print(f"[ensure_model] Model ready: {onnx_path}")
    return onnx_path
