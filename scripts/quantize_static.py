#!/usr/bin/env python3
"""Static INT8 quantization: FP32 ONNX + calibration images -> INT8 ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.calib_reader import YOLOXCalibReader  # noqa: E402
from lib.config import load_config  # noqa: E402


def quantize(cfg: dict) -> str:
    fp32 = Path(cfg["fp32_onnx"])
    if not fp32.is_file():
        raise FileNotFoundError(
            f"FP32 ONNX not found: {fp32}\nRun export first: "
            f"python scripts/run.py --config <cfg> export"
        )

    calib_dir = Path(cfg["calibration_dir"])
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    print(f"Loading FP32 model: {fp32}")
    session = ort.InferenceSession(
        str(fp32),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    print("Input name:", input_name)

    reader = YOLOXCalibReader(
        str(calib_dir),
        input_name,
        input_size=cfg["input_size"],
        max_images=cfg["max_calib_images"],
    )

    out_path = Path(cfg["int8_onnx"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting static INT8 quantization -> {out_path}")
    quantize_static(
        model_input=str(fp32),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print("Done.")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Static INT8 ONNX quantization")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML (e.g. configs/dock.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    quantize(cfg)


if __name__ == "__main__":
    main()
