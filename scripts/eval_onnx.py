#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOX ONNX Evaluation Script (VOC mAP)

Features:
- Evaluates ONNX / INT8 ONNX model
- Uses valid.txt image list
- Computes:
    - mAP@0.5
    - Per-class AP
- Prints thresholds used:
    - confidence threshold
    - NMS threshold
    - IoU threshold
- Compatible with YOLOX exports

Example:
python eval_onnx.py \
    --model weights/yolox_int8.onnx \
    --exp_file exps/dock/yolox_voc_s3.py \
    --valid_txt datasets/VOCdevkit/VOC2012/ImageSets/Main/valid.txt \
    --data_dir datasets/VOCdevkit/VOC2012 \
    --classes classes.txt \
    --conf 0.01 \
    --nms 0.65 \
    --tsize 640
"""

import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from collections import defaultdict

from yolox.exp import get_exp
from yolox.utils import postprocess

import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser("YOLOX ONNX Evaluation")

    parser.add_argument("--model", type=str, required=True,
                        help="ONNX model path")

    parser.add_argument("--exp_file", type=str, required=True,
                        help="YOLOX experiment file")

    parser.add_argument("--valid_txt", type=str, required=True,
                        help="valid.txt path")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="VOC dataset root")

    parser.add_argument("--classes", type=str, required=True,
                        help="classes.txt")

    parser.add_argument("--conf", type=float, default=0.01,
                        help="confidence threshold")

    parser.add_argument("--nms", type=float, default=0.65,
                        help="NMS threshold")

    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for mAP")

    parser.add_argument("--tsize", type=int, default=640,
                        help="test size")

    return parser


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def preprocess(img, input_size):
    h, w = img.shape[:2]

    scale = min(input_size / h, input_size / w)

    nh = int(h * scale)
    nw = int(w * scale)

    resized = cv2.resize(img, (nw, nh))

    padded = np.ones((input_size, input_size, 3), dtype=np.uint8) * 114
    padded[:nh, :nw] = resized

    padded = padded.astype(np.float32)
    padded = padded[:, :, ::-1]
    padded /= 255.0

    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)

    return padded, scale


# ---------------------------------------------------------
# Load Ground Truth
# ---------------------------------------------------------

def load_gt(label_path):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        cls = int(parts[0])

        x1 = float(parts[1])
        y1 = float(parts[2])
        x2 = float(parts[3])
        y2 = float(parts[4])

        boxes.append([cls, x1, y1, x2, y2])

    return boxes


# ---------------------------------------------------------
# IoU
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(args):

    with open(args.classes, "r") as f:
        class_names = [x.strip() for x in f.readlines()]

    num_classes = len(class_names)

    print("\n==============================")
    print("Evaluation Settings")
    print("==============================")
    print(f"Model              : {args.model}")
    print(f"Confidence thresh  : {args.conf}")
    print(f"NMS thresh         : {args.nms}")
    print(f"IoU thresh         : {args.iou}")
    print(f"Input size         : {args.tsize}")
    print("==============================\n")

    session = ort.InferenceSession(
        args.model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    with open(args.valid_txt, "r") as f:
        image_ids = [x.strip() for x in f.readlines()]

    gt_count = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)

    for image_id in tqdm(image_ids):

        img_path = os.path.join(
            args.data_dir,
            "JPEGImages",
            image_id + ".jpg"
        )

        label_path = os.path.join(
            args.data_dir,
            "labels",
            image_id + ".txt"
        )

        img = cv2.imread(img_path)

        if img is None:
            print(f"Missing image: {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]

        inp, scale = preprocess(img, args.tsize)

        outputs = session.run(None, {input_name: inp})

        outputs = torch.tensor(outputs[0])

        outputs = postprocess(
            outputs,
            num_classes=num_classes,
            conf_thre=args.conf,
            nms_thre=args.nms,
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

                iou = compute_iou(
                    [x1, y1, x2, y2],
                    [gx1, gy1, gx2, gy2]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= args.iou and best_gt_idx not in used_gt:
                tp[cls] += 1
                used_gt.add(best_gt_idx)
            else:
                fp[cls] += 1

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

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

        print(
            f"{cls_name:20s} "
            f"AP50: {ap:.4f} "
            f"TP={true_pos} "
            f"FP={false_pos} "
            f"GT={total_gt}"
        )

    map50 = np.mean(aps)

    print("\n==============================")
    print(f"mAP@0.5 = {map50:.4f}")
    print("==============================\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    args = make_parser()
    args = args.parse_args()

    evaluate(args)