#!/usr/bin/env python3
"""Run ONNX inference (FP32 or INT8) with classes from config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.config import load_class_names, load_config  # noqa: E402
from lib.preprocess import letterbox_with_ratio  # noqa: E402
from lib.yolox_postprocess import postprocess  # noqa: E402


def infer(cfg: dict, image: str, model_kind: str, output: str | None) -> str:
    if model_kind == "fp32":
        model_path = cfg["fp32_onnx"]
    elif model_kind == "int8":
        model_path = cfg["int8_onnx"]
    else:
        raise ValueError(f"model_kind must be fp32 or int8, got {model_kind!r}")

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    class_names = load_class_names(cfg["classes"])
    img = cv2.imread(image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image}")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    blob, ratio = letterbox_with_ratio(img, cfg["input_size"])
    outputs = session.run(None, {input_name: blob})

    dets = postprocess(
        outputs,
        ratio,
        cfg["input_size"],
        cfg["conf_thres"],
        cfg["nms_thres"],
    )

    print(f"TOTAL DETECTIONS: {len(dets)}")
    vis = img.copy()
    for det in dets:
        x1, y1, x2, y2, score, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = class_names[int(cls_id)]
        print(f"{label} {score:.3f} ({x1},{y1},{x2},{y2})")

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if output is None:
        output = str(
            Path(cfg["output_dir"]) / f"infer_{model_kind}_{Path(image).stem}.jpg"
        )
    cv2.imwrite(output, vis)
    print(f"Saved {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for a quantized model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model",
        choices=("fp32", "int8"),
        default="int8",
        help="Which ONNX artifact to run",
    )
    parser.add_argument("--output", default=None, help="Output visualization path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer(cfg, args.image, args.model, args.output)


if __name__ == "__main__":
    main()
