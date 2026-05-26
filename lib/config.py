"""Load per-model YAML configs for export / quantize / infer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    name = cfg.get("model_name") or config_path.stem
    cfg["model_name"] = name

    base = REPO_ROOT
    cfg["exp_file"] = str(_resolve(cfg["exp_file"], base))
    cfg["weights"] = str(_resolve(cfg["weights"], base))
    cfg["classes"] = str(_resolve(cfg["classes"], base))
    cfg["calibration_dir"] = str(_resolve(cfg["calibration_dir"], base))

    output_dir = cfg.get("output_dir")
    if output_dir:
        output_dir = _resolve(output_dir, base)
    else:
        output_dir = REPO_ROOT / "artifacts" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)

    cfg["fp32_onnx"] = str(
        _resolve(cfg.get("fp32_onnx", output_dir / "fp32.onnx"), base)
    )
    cfg["int8_onnx"] = str(
        _resolve(cfg.get("int8_onnx", output_dir / "int8.onnx"), base)
    )

    cfg.setdefault("input_size", 640)
    cfg.setdefault("opset", 11)
    cfg.setdefault("max_calib_images", 100)
    cfg.setdefault("conf_thres", 0.25)
    cfg.setdefault("nms_thres", 0.45)

    return cfg


def load_class_names(classes_path: str) -> list[str]:
    with open(classes_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
