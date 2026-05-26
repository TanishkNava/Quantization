#!/usr/bin/env python3
"""
Unified CLI for the static quantization pipeline.

  export   — .pth + exp -> FP32 ONNX
  quantize — FP32 ONNX + cal images -> INT8 ONNX
  infer    — run FP32 or INT8 ONNX on one image
  all      — export then quantize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.config import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="YOLOX static quantization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run.py --config configs/dock.yaml export
  python scripts/run.py --config configs/dock.yaml quantize
  python scripts/run.py --config configs/dock.yaml all
  python scripts/run.py --config configs/dock.yaml infer --image test.jpg --model int8
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Model config YAML (configs/dock.yaml, configs/gate.yaml, ...)",
    )
    parser.add_argument(
        "command",
        choices=("export", "quantize", "infer", "all"),
        help="Pipeline step to run",
    )
    parser.add_argument("--image", help="Image path (required for infer)")
    parser.add_argument(
        "--model",
        choices=("fp32", "int8"),
        default="int8",
        help="ONNX variant for infer",
    )
    parser.add_argument("--output", help="Output image path for infer")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "export":
        from scripts.export_fp32 import export_fp32

        export_fp32(cfg)
    elif args.command == "quantize":
        from scripts.quantize_static import quantize

        quantize(cfg)
    elif args.command == "all":
        from scripts.export_fp32 import export_fp32
        from scripts.quantize_static import quantize

        export_fp32(cfg)
        quantize(cfg)
    elif args.command == "infer":
        if not args.image:
            parser.error("infer requires --image")
        from scripts.infer import infer

        infer(cfg, args.image, args.model, args.output)


if __name__ == "__main__":
    main()
