#!/usr/bin/env python3
"""ONNX Runtime (ORT) post-training dynamic quantization script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType

# Insert current working directory to python path for YOLOX exp imports
sys.path.insert(0, str(Path.cwd()))

from utils import load_config
from quantize_ort import export_fp32


def quantize_ort_dynamic(cfg: dict) -> str:
    fp32_path = export_fp32(cfg)

    out_path = Path(cfg["output_dir"]) / "int8_dynamic.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting dynamic INT8 quantization: {fp32_path} -> {out_path}")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
        reduce_range=False,
    )

    print(f"Dynamic quantization finished successfully. Saved quantized model: {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="ORT post-training dynamic INT8 quantization")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    quantize_ort_dynamic(cfg)


if __name__ == "__main__":
    main()

