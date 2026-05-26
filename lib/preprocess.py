"""YOLOX letterbox preprocess (shared by calib, export check, inference)."""

from __future__ import annotations

import cv2
import numpy as np


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

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    blob = padded.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return np.ascontiguousarray(blob, dtype=np.float32)


def letterbox_with_ratio(img: np.ndarray, input_size: int = 640):
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
