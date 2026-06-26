#!/usr/bin/env python3
"""Run ONNX Runtime inference (FP32, static INT8, or dynamic INT8) on an image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import onnxruntime as ort

# Add parent directory to python path to import from utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import load_config, load_class_names, letterbox_with_ratio, postprocess


def infer_onnx(cfg: dict, image_path: str, model_type: str, output_path: str | None) -> str:
    # Resolve model path based on type
    if model_type == "fp32":
        model_path = cfg["fp32_onnx"]
    elif model_type == "int8":
        model_path = cfg["int8_onnx"]
    elif model_type == "dynamic":
        model_path = str(Path(cfg["output_dir"]) / "int8_dynamic.onnx")
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choices are 'fp32', 'int8', 'dynamic'")

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}\nPlease run quantization/export first.")

    print(f"Loading {model_type.upper()} ONNX model from: {model_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # Load labels
    class_names = load_class_names(cfg["classes"])

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Test image not found at: {image_path}")

    # Preprocess
    blob, ratio = letterbox_with_ratio(img, cfg["input_size"])

    # Forward inference
    outputs = session.run(None, {input_name: blob})

    # Decode predictions and run NMS
    dets = postprocess(
        outputs=outputs,
        ratio=ratio,
        input_size=cfg["input_size"],
        conf_thres=cfg["conf_thres"],
        nms_thr=cfg["nms_thres"],
    )

    print(f"Total detections: {len(dets)}")
    vis_img = img.copy()

    for det in dets:
        x1, y1, x2, y2, score, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = class_names[int(cls_id)]
        print(f" - {label}: {score:.3f} at ({x1}, {y1}, {x2}, {y2})")

        # Draw box and label
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            f"{label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if output_path is None:
        output_path = str(
            Path(cfg["output_dir"]) / f"infer_ort_{model_type}_{Path(image_path).stem}.jpg"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Saved visualization to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="ONNX model inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--type",
        choices=("fp32", "int8", "dynamic"),
        default="int8",
        help="ONNX model type to run (default: int8)",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Custom output image path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_onnx(cfg, args.image, args.type, args.output)


if __name__ == "__main__":
    main()

