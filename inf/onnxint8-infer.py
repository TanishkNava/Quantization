#!/usr/bin/env python3

import os
import cv2
import numpy as np
import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess

# ---------------- CONFIG ---------------- #

MODEL_PATH = "/home/navavision/Utkarsh_workspace/Model-Training/conversion/mondeleze_int8.onnx"

INPUT_DIR = "/home/navavision/Utkarsh_workspace/Model-Training/callib-data2"
OUTPUT_DIR = "callib-data2-results"

INPUT_SHAPE = (640, 640)

SCORE_THR = 0.3
NMS_THR = 0.45

CLASS_NAMES = [
    "class0",
    "class1",
    "class2",
    "class3",
    "class4",
    "class5",
    "class6",
    "class7"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ---------------- #

session = onnxruntime.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# ---------------- IMAGE LIST ---------------- #

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} images")

# ---------------- INFERENCE LOOP ---------------- #

for image_name in image_files:

    image_path = os.path.join(INPUT_DIR, image_name)

    origin_img = cv2.imread(image_path)

    if origin_img is None:
        print(f"Failed to read: {image_name}")
        continue

    # preprocess
    img, ratio = preprocess(origin_img, INPUT_SHAPE)

    ort_inputs = {
        input_name: img[None, :, :, :]
    }

    # inference
    output = session.run(None, ort_inputs)

    # decode
    predictions = demo_postprocess(output[0], INPUT_SHAPE)[0]

    # boxes + scores
    boxes = predictions[:, :4]

    scores = predictions[:, 4:5] * predictions[:, 5:]

    print(
        f"{image_name} score stats:",
        scores.min(),
        scores.max(),
        scores.mean()
    )

    # xywh -> xyxy
    boxes_xyxy = np.ones_like(boxes)

    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

    # resize back
    boxes_xyxy /= ratio

    # NMS
    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        nms_thr=NMS_THR,
        score_thr=SCORE_THR
    )

    if dets is None:
        print(f"No detections: {image_name}")

        save_path = os.path.join(OUTPUT_DIR, image_name)
        cv2.imwrite(save_path, origin_img)

        continue

    final_boxes = dets[:, :4]
    final_scores = dets[:, 4]
    final_cls_inds = dets[:, 5]

    print(f"{image_name} -> {len(final_boxes)} detections")

    # draw detections
    for box, score, cls_id in zip(
        final_boxes,
        final_scores,
        final_cls_inds
    ):

        x1, y1, x2, y2 = box.astype(int)

        cls_id = int(cls_id)

        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"

        cv2.rectangle(
            origin_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            origin_img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # save
    save_path = os.path.join(OUTPUT_DIR, image_name)

    cv2.imwrite(save_path, origin_img)

print("Done!")