#!/usr/bin/env python3
"""ONNX Runtime (ORT) post-training static quantization script."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import torch

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

# Insert current working directory to python path for YOLOX exp imports
sys.path.insert(0, str(Path.cwd()))

from utils import load_config, YOLOXCalibReader
from yolox.exp import get_exp


def export_fp32(cfg: dict) -> str:
    fp32_path = Path(cfg["fp32_onnx"])
    if fp32_path.is_file():
        print(f"FP32 ONNX model already exists at: {fp32_path}")
        return str(fp32_path)

    print("--- Exporting FP32 ONNX model ---")
    weights = Path(cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights}")

    exp = get_exp(cfg["exp_file"], None)
    model = exp.get_model()

    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    if hasattr(model, "head"):
        model.head.decode_in_inference = False

    size = getattr(exp, "test_size", (cfg["input_size"], cfg["input_size"]))
    if isinstance(size, int):
        size = (size, size)

    dummy = torch.randn(1, 3, size[0], size[1])
    fp32_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting PyTorch model to ONNX: {fp32_path}")
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["images"],
        output_names=["output"],
        opset_version=cfg["opset"],
        do_constant_folding=True,
    )
    print("FP32 ONNX Export Completed.")
    return str(fp32_path)


def quantize_ort(cfg: dict) -> str:
    fp32_path = export_fp32(cfg)

    calib_dir = Path(cfg["calibration_dir"])
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    print(f"Loading FP32 model: {fp32_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(fp32_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX model input name: {input_name}")

    reader = YOLOXCalibReader(
        image_dir=str(calib_dir),
        input_name=input_name,
        input_size=cfg["input_size"],
        max_images=cfg["max_calib_images"],
    )

    out_path = Path(cfg["int8_onnx"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run quantization preprocessing
    preproc_path = fp32_path.replace(".onnx", "_preproc.onnx")
    print("Running ONNX quantization pre-processing...")
    try:
        quant_pre_process(fp32_path, preproc_path)
        model_input = preproc_path
        print(f"Pre-processed model saved to: {preproc_path}")
    except Exception as e:
        print(f"Pre-processing failed: {e}. Using raw FP32 model for quantization.")
        model_input = fp32_path

    # Calibration method
    calib_method_str = cfg.get("calib_method", "MinMax")
    print(f"Calibration method: {calib_method_str}")
    if calib_method_str.lower() == "entropy":
        calib_method = CalibrationMethod.Entropy
    elif calib_method_str.lower() == "percentile":
        calib_method = CalibrationMethod.Percentile
    else:
        calib_method = CalibrationMethod.MinMax

    try:
        print(f"Starting static INT8 quantization -> {out_path}")
        quantize_static(
            model_input=model_input,
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=cfg.get("per_channel", True),
            calibrate_method=calib_method,
        )

        print(f"Static quantization finished successfully. Saved quantized model: {out_path}")
    finally:
        if model_input == preproc_path and os.path.exists(preproc_path):
            print(f"Cleaning up temporary file: {preproc_path}")
            try:
                os.remove(preproc_path)
            except Exception as e:
                print(f"Failed to remove temporary file {preproc_path}: {e}")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="ORT post-training static INT8 quantization")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    quantize_ort(cfg)


if __name__ == "__main__":
    main()

