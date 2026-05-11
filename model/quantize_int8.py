import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

MODEL_FP32 = "best_ckpt.onnx"
MODEL_INT8 = "best_int8.onnx"

CALIB_DIR = "../calibration_images"

INPUT_SIZE = 640

MAX_CALIB_IMAGES = 100


def preprocess(img_path):

    img = cv2.imread(img_path)

    if img is None:
        return None

    h, w = img.shape[:2]

    ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)

    new_h = int(h * ratio)
    new_w = int(w * ratio)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8,
    )

    padded[:new_h, :new_w] = resized

    blob = padded.transpose(2, 0, 1)

    blob = np.expand_dims(blob, axis=0)

    blob = np.ascontiguousarray(
        blob,
        dtype=np.float32,
    )

    return blob


class YOLOXDataReader(CalibrationDataReader):

    def __init__(self, image_dir, input_name):

        self.input_name = input_name

        self.image_paths = []

        for ext in ("*.jpg", "*.jpeg", "*.png"):

            self.image_paths.extend(
                glob.glob(
                    os.path.join(image_dir, ext)
                )
            )

        self.image_paths = sorted(
            self.image_paths
        )[:MAX_CALIB_IMAGES]

        print(
            "Calibration images:",
            len(self.image_paths)
        )

        self.idx = 0

    def get_next(self):

        while self.idx < len(self.image_paths):

            path = self.image_paths[self.idx]

            self.idx += 1

            blob = preprocess(path)

            if blob is None:
                continue

            return {
                self.input_name: blob
            }

        return None


print("Loading FP32 model...")

session = ort.InferenceSession(
    MODEL_FP32,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

print("Input name:", input_name)

reader = YOLOXDataReader(
    CALIB_DIR,
    input_name,
)

print("Starting INT8 quantization...")

quantize_static(
    model_input=MODEL_FP32,
    model_output=MODEL_INT8,

    calibration_data_reader=reader,

    quant_format=QuantFormat.QDQ,

    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,

    per_channel=False,

    calibrate_method=CalibrationMethod.MinMax,
)

print("INT8 model saved:")
print(MODEL_INT8)