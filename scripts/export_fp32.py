#!/usr/bin/env python3
"""Export a YOLOX .pth checkpoint to FP32 ONNX using the model exp file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.config import load_config  # noqa: E402
from yolox.exp import get_exp  # noqa: E402


def export_fp32(cfg: dict) -> str:
    weights = Path(cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    exp = get_exp(cfg["exp_file"], None)
    model = exp.get_model()

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
    out_path = Path(cfg["fp32_onnx"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting FP32 ONNX -> {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["output"],
        opset_version=cfg["opset"],
        do_constant_folding=True,
    )
    print("Done.")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Export YOLOX .pth to FP32 ONNX")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML (e.g. configs/dock.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    export_fp32(cfg)


if __name__ == "__main__":
    main()
