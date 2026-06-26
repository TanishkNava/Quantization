#!/usr/bin/env python3
"""PyTorch native post-training static quantization script using FX Graph Mode."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
import torch
import torch.ao.quantization as quantization
from torch.ao.quantization import quantize_fx

# Insert current working directory to python path for YOLOX exp imports
sys.path.insert(0, str(Path.cwd()))

from utils import load_config, letterbox_blob
from yolox.exp import get_exp


def quantize_pytorch(cfg: dict) -> str:
    print("--- PyTorch Native Static Quantization ---")
    weights = Path(cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights}")

    exp = get_exp(cfg["exp_file"], None)
    model = exp.get_model()

    # Load weights
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Disable decode in inference to allow symbolic tracing by FX
    if hasattr(model, "head"):
        model.head.decode_in_inference = False
        print("Disabled decode_in_inference on YOLOX head.")

    input_size = cfg["input_size"]
    example_inputs = torch.randn(1, 3, input_size, input_size)

    # 1. Prepare for FX static quantization
    print("Preparing model for FX static quantization...")
    qconfig_mapping = quantization.get_default_qconfig_mapping("fbgemm")
    model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

    # 2. Calibration
    calib_dir = Path(cfg["calibration_dir"])
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(glob.glob(str(calib_dir / ext)))
    image_paths = sorted(image_paths)[:cfg["max_calib_images"]]

    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in: {calib_dir}")

    print(f"Running calibration on {len(image_paths)} images...")
    model_prepared.eval()
    with torch.no_grad():
        for path in image_paths:
            blob = letterbox_blob(path, input_size)
            if blob is None:
                continue
            input_tensor = torch.from_numpy(blob)
            model_prepared(input_tensor)

    # 3. Convert to INT8
    print("Converting model to quantized INT8...")
    model_quantized = quantize_fx.convert_fx(model_prepared)

    # 4. Trace and save TorchScript serialized model
    print("Tracing quantized model to TorchScript...")
    model_quantized.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model_quantized, example_inputs)

    out_path = Path(cfg["int8_pytorch"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced_model, str(out_path))
    print(f"Successfully saved PyTorch quantized model to: {out_path}")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="PyTorch native static INT8 quantization")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    quantize_pytorch(cfg)


if __name__ == "__main__":
    main()

