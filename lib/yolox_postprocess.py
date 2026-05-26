"""YOLOX decode + multiclass NMS for ONNX inference."""

from __future__ import annotations

import cv2
import numpy as np


def multiclass_nms(boxes, scores, conf_thres: float, nms_thr: float):
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        keep = cls_scores > conf_thres
        if keep.sum() == 0:
            continue

        valid_scores = cls_scores[keep]
        valid_boxes = boxes[keep]

        indices = cv2.dnn.NMSBoxes(
            valid_boxes.tolist(),
            valid_scores.tolist(),
            conf_thres,
            nms_thr,
        )

        if len(indices) > 0:
            for i in indices.flatten():
                final_dets.append([*valid_boxes[i], valid_scores[i], cls_ind])

    return final_dets


def postprocess(outputs, ratio: float, input_size: int, conf_thres: float, nms_thr: float):
    predictions = outputs[0][0]

    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    hsizes = [input_size // s for s in strides]
    wsizes = [input_size // s for s in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    predictions[..., :2] = (predictions[..., :2] + grids) * expanded_strides
    predictions[..., 2:4] = np.exp(predictions[..., 2:4]) * expanded_strides

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy /= ratio

    return multiclass_nms(boxes_xyxy, scores, conf_thres, nms_thr)
