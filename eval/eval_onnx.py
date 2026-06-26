#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOX ONNX Evaluation Script (VOC mAP)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import torch

# Add parent directory to python path to import from utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import load_config, load_class_names
from yolox.utils import postprocess as yolox_postprocess


def preprocess(img, input_size):
    h, w = img.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh = int(h * scale)
    nw = int(w * scale)

    resized = cv2.resize(img, (nw, nh))
    padded = np.ones((input_size, input_size, 3), dtype=np.uint8) * 114
    padded[:nh, :nw] = resized

    padded = padded.astype(np.float32)
    padded = padded[:, :, ::-1]  # BGR to RGB
    padded /= 255.0

    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)

    return padded, scale


def load_gt(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x1 = float(parts[1])
        y1 = float(parts[2])
        x2 = float(parts[3])
        y2 = float(parts[4])
        boxes.append([cls, x1, y1, x2, y2])

    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0
    return inter / union


def evaluate_onnx(cfg: dict, model_type: str, valid_txt: str, data_dir: str, conf_thres: float, nms_thres: float, iou_eval_thres: float):
    # Resolve model path based on type
    if model_type == "fp32":
        model_path = cfg["fp32_onnx"]
    elif model_type == "int8":
        model_path = cfg["int8_onnx"]
    elif model_type == "dynamic":
        model_path = str(Path(cfg["output_dir"]) / "int8_dynamic.onnx")
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    class_names = load_class_names(cfg["classes"])
    num_classes = len(class_names)

    print("\n==============================")
    print("ONNX Evaluation Settings")
    print("==============================")
    print(f"Model Type         : {model_type.upper()}")
    print(f"Model Path         : {model_path}")
    print(f"Confidence Thresh  : {conf_thres}")
    print(f"NMS Thresh         : {nms_thres}")
    print(f"IoU AP Thresh      : {iou_eval_thres}")
    print(f"Input Size         : {cfg['input_size']}")
    print("==============================\n")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        model_path,
        providers=providers
    )
    input_name = session.get_inputs()[0].name

    with open(valid_txt, "r") as f:
        image_ids = [x.strip() for x in f.readlines() if x.strip()]


    gt_count = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)

    for image_id in tqdm(image_ids, desc="Evaluating"):
        img_path = os.path.join(data_dir, "JPEGImages", image_id + ".jpg")
        label_path = os.path.join(data_dir, "labels", image_id + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        inp, scale = preprocess(img, cfg["input_size"])
        outputs = session.run(None, {input_name: inp})
        outputs = torch.tensor(outputs[0])

        outputs = yolox_postprocess(
            outputs,
            num_classes=num_classes,
            conf_thre=conf_thres,
            nms_thre=nms_thres,
            class_agnostic=True
        )

        preds = outputs[0]
        gt_boxes = load_gt(label_path)
        used_gt = set()

        for gt in gt_boxes:
            gt_count[gt[0]] += 1

        if preds is None:
            continue

        preds = preds.cpu().numpy()

        for pred in preds:
            x1, y1, x2, y2 = pred[:4]
            score = pred[4] * pred[5]
            cls = int(pred[6])

            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale

            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(gt_boxes):
                gt_cls, gx1, gy1, gx2, gy2 = gt
                if gt_cls != cls:
                    continue

                iou = compute_iou([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_eval_thres and best_gt_idx not in used_gt:
                tp[cls] += 1
                used_gt.add(best_gt_idx)
            else:
                fp[cls] += 1

    print("\n==============================")
    print("Per Class AP")
    print("==============================")
    aps = []
    for cls_id, cls_name in enumerate(class_names):
        total_gt = gt_count[cls_id]
        true_pos = tp[cls_id]
        false_pos = fp[cls_id]

        if total_gt == 0:
            ap = 0
        else:
            precision = true_pos / max(true_pos + false_pos, 1)
            recall = true_pos / total_gt
            ap = precision * recall

        aps.append(ap)
        print(f"{cls_name:20s} AP50: {ap:.4f} TP={true_pos} FP={false_pos} GT={total_gt}")

    map50 = np.mean(aps)
    print("\n==============================")
    print(f"mAP@0.5 = {map50:.4f}")
    print("==============================\n")
    return map50


def main():
    parser = argparse.ArgumentParser("YOLOX ONNX Evaluation")
    parser.add_argument(
        "--type",
        choices=("fp32", "int8", "dynamic"),
        default="int8",
        help="ONNX model type to run (default: int8)",
    )
    parser.add_argument(
        "--valid_txt",
        required=True,
        help="Path to VOC valid.txt containing image IDs",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to VOC dataset directory containing JPEGImages/ and labels/",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--conf", type=float, default=0.01, help="confidence threshold")
    parser.add_argument("--nms", type=float, default=0.65, help="NMS threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU evaluation threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_onnx(cfg, args.type, args.valid_txt, args.data_dir, args.conf, args.nms, args.iou)


if __name__ == "__main__":
    main()

