"""Shared utilities for YOLOX post-training quantization and inference."""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Config & Path Loader
# ----------------------------------------------------------------------
def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        # Try finding in parent directories or next to repo root
        config_path = REPO_ROOT / config_path.name
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Extract model name from weights path or use a fallback name
    weights_path = Path(cfg.get("weights", "best_ckpt.pth"))
    if len(weights_path.parts) >= 3:
        model_name = weights_path.parent.parent.name
    else:
        model_name = "model"
    cfg["model_name"] = model_name

    # Resolve paths relative to REPO_ROOT
    base = REPO_ROOT
    cfg["exp_file"] = str((base / cfg["exp_file"]).resolve())
    cfg["weights"] = str((base / cfg["weights"]).resolve())
    cfg["classes"] = str((base / cfg["classes"]).resolve())
    cfg["calibration_dir"] = str((base / cfg["calibration_dir"]).resolve())

    # Set and resolve output directories and files
    output_dir = cfg.get("output_dir")
    if output_dir:
        output_dir = (base / output_dir).resolve()
    else:
        output_dir = base / "artifacts" / cfg["model_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)

    cfg["fp32_onnx"] = str((output_dir / "fp32.onnx").resolve())
    cfg["int8_onnx"] = str((output_dir / "int8.onnx").resolve())
    cfg["int8_pytorch"] = str((output_dir / "int8_pytorch.pt").resolve())

    # Set defaults
    cfg.setdefault("input_size", 640)
    cfg.setdefault("opset", 11)
    cfg.setdefault("max_calib_images", 100)
    cfg.setdefault("conf_thres", 0.25)
    cfg.setdefault("nms_thres", 0.45)
    cfg.setdefault("calib_method", "MinMax")
    cfg.setdefault("per_channel", True)

    return cfg




def load_class_names(classes_path: str) -> list[str]:
    with open(classes_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def letterbox_blob(img_path: str, input_size: int = 640) -> np.ndarray | None:
    img = cv2.imread(img_path)
    if img is None:
        return None
    return letterbox_blob_from_bgr(img, input_size)


def letterbox_blob_from_bgr(img: np.ndarray, input_size: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    ratio = min(input_size / h, input_size / w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    blob = padded.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return np.ascontiguousarray(blob, dtype=np.float32)


def letterbox_with_ratio(img: np.ndarray, input_size: int = 640) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    ratio = min(input_size / h, input_size / w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    blob = padded.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0).astype(np.float32)
    return blob, ratio


# ----------------------------------------------------------------------
# Postprocessing (Grid decoding + Multiclass NMS)
# ----------------------------------------------------------------------
def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, conf_thres: float, nms_thr: float):
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        keep = cls_scores > conf_thres
        if keep.sum() == 0:
            continue

        valid_scores = cls_scores[keep]
        valid_boxes = boxes[keep]

        indices = cv2.dnn.NMSBoxes(
            valid_boxes.tolist(),
            valid_scores.tolist(),
            conf_thres,
            nms_thr,
        )

        if len(indices) > 0:
            for i in indices.flatten():
                final_dets.append([*valid_boxes[i], valid_scores[i], cls_ind])

    return final_dets


def postprocess(outputs: Any, ratio: float, input_size: int, conf_thres: float, nms_thr: float):
    # Handle list outputs or raw tensors/numpy arrays
    if isinstance(outputs, (list, tuple)):
        predictions = outputs[0]
    else:
        predictions = outputs

    if hasattr(predictions, "cpu"):
        predictions = predictions.cpu().detach().numpy()

    if len(predictions.shape) == 3:
        predictions = predictions[0]

    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    hsizes = [input_size // s for s in strides]
    wsizes = [input_size // s for s in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)[0]
    expanded_strides = np.concatenate(expanded_strides, 1)[0]

    # Decode predictions
    decoded_xy = (predictions[..., :2] + grids) * expanded_strides
    decoded_wh = np.exp(predictions[..., 2:4]) * expanded_strides

    boxes = np.zeros_like(predictions[..., :4])
    boxes[..., 0] = decoded_xy[..., 0] - decoded_wh[..., 0] / 2
    boxes[..., 1] = decoded_xy[..., 1] - decoded_wh[..., 1] / 2
    boxes[..., 2] = decoded_xy[..., 0] + decoded_wh[..., 0] / 2
    boxes[..., 3] = decoded_xy[..., 1] + decoded_wh[..., 1] / 2
    boxes /= ratio

    scores = predictions[..., 4:5] * predictions[..., 5:]

    return multiclass_nms(boxes, scores, conf_thres, nms_thr)


# ----------------------------------------------------------------------
# Calibration Data Reader for ORT
# ----------------------------------------------------------------------
try:
    from onnxruntime.quantization import CalibrationDataReader

    class YOLOXCalibReader(CalibrationDataReader):
        def __init__(
            self,
            image_dir: str,
            input_name: str,
            input_size: int = 640,
            max_images: int = 100,
        ):
            self.input_name = input_name
            self.input_size = input_size
            self.image_paths: list[str] = []

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

            self.image_paths = sorted(self.image_paths)[:max_images]
            print(f"Loaded {len(self.image_paths)} calibration images.")
            self.idx = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            while self.idx < len(self.image_paths):
                path = self.image_paths[self.idx]
                self.idx += 1
                blob = letterbox_blob(path, self.input_size)
                if blob is None:
                    continue
                return {self.input_name: blob}
            return None
except ImportError:
    # If onnxruntime is not available in some context, define dummy class
    class YOLOXCalibReader:
        def __init__(self, *args, **kwargs):
            raise ImportError("onnxruntime is required to use YOLOXCalibReader.")
