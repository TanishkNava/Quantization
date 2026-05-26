"""ONNX Runtime calibration data reader for YOLOX static quantization."""

from __future__ import annotations

import glob
import os

from onnxruntime.quantization import CalibrationDataReader

from lib.preprocess import letterbox_blob


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
        print(f"Calibration images: {len(self.image_paths)}")
        self.idx = 0

    def get_next(self):
        while self.idx < len(self.image_paths):
            path = self.image_paths[self.idx]
            self.idx += 1
            blob = letterbox_blob(path, self.input_size)
            if blob is None:
                continue
            return {self.input_name: blob}
        return None
