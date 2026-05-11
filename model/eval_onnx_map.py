import os
import time
import cv2
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from xml.etree import ElementTree as ET

CLASSES = [
    "person",
    "dock_open",
    "truck",
    "dock_closed",
    "forklift",
    "box_in_hand",
    "pallet_load",
    "no_truck",
]

MODEL = "best_ckpt.pth"

VOC_ROOT = "/home/navavision/Utkarsh-Workspace/Quant/YOLOX/datasets/VOCdevkit/VOC2012"

VALID_TXT = os.path.join(
    VOC_ROOT,
    "ImageSets/Main/valid.txt"
)

INPUT_SIZE = 640

CONF_THRES = 0.001
NMS_THRES = 0.65

IOU_THRESHOLDS = [0.5, 0.7]

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name


def preprocess(img):

    h, w = img.shape[:2]

    ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)

    new_h = int(h * ratio)
    new_w = int(w * ratio)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8,
    )

    padded[:new_h, :new_w] = resized

    blob = padded.transpose(2, 0, 1)

    blob = np.expand_dims(blob, axis=0)

    blob = np.ascontiguousarray(
        blob,
        dtype=np.float32,
    )

    return blob, ratio


def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def nms(boxes, scores, thresh):

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONF_THRES,
        thresh,
    )

    if len(indices) == 0:
        return []

    return indices.flatten()


def postprocess(outputs, ratio):

    predictions = outputs[0][0]

    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    hsizes = [INPUT_SIZE // s for s in strides]
    wsizes = [INPUT_SIZE // s for s in strides]

    for hsize, wsize, stride in zip(
        hsizes,
        wsizes,
        strides,
    ):

        xv, yv = np.meshgrid(
            np.arange(wsize),
            np.arange(hsize),
        )

        grid = np.stack(
            (xv, yv),
            axis=2
        ).reshape(1, -1, 2)

        grids.append(grid)

        shape = grid.shape[:2]

        expanded_strides.append(
            np.full((*shape, 1), stride)
        )

    grids = np.concatenate(grids, axis=1)

    expanded_strides = np.concatenate(
        expanded_strides,
        axis=1,
    )

    predictions[..., :2] = (
        predictions[..., :2] + grids
    ) * expanded_strides

    predictions[..., 2:4] = np.exp(
        predictions[..., 2:4]
    ) * expanded_strides

    boxes = predictions[:, :4]

    scores = (
        predictions[:, 4:5] *
        predictions[:, 5:]
    )

    boxes_xyxy = np.zeros_like(boxes)

    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy /= ratio

    detections = []

    for cls_id in range(len(CLASSES)):

        cls_scores = scores[:, cls_id]

        keep = cls_scores > CONF_THRES

        if keep.sum() == 0:
            continue

        valid_boxes = boxes_xyxy[keep]

        valid_scores = cls_scores[keep]

        keep_idx = nms(
            valid_boxes,
            valid_scores,
            NMS_THRES,
        )

        for idx in keep_idx:

            detections.append(
                [
                    *valid_boxes[idx],
                    float(valid_scores[idx]),
                    cls_id,
                ]
            )

    return detections


def parse_xml(xml_path):

    tree = ET.parse(xml_path)

    root = tree.getroot()

    gts = []

    for obj in root.findall("object"):

        cls_name = obj.find("name").text

        if cls_name not in CLASSES:
            continue

        cls_id = CLASSES.index(cls_name)

        box = obj.find("bndbox")

        xmin = int(float(box.find("xmin").text))
        ymin = int(float(box.find("ymin").text))
        xmax = int(float(box.find("xmax").text))
        ymax = int(float(box.find("ymax").text))

        gts.append(
            [
                xmin,
                ymin,
                xmax,
                ymax,
                cls_id,
            ]
        )

    return gts


all_predictions = {
    i: [] for i in range(len(CLASSES))
}

gt_counter = {
    i: 0 for i in range(len(CLASSES))
}


with open(VALID_TXT, "r") as f:

    ids = [
        x.strip()
        for x in f.readlines()
    ]


all_times = []

print("Running inference...")

for img_id in tqdm(ids):

    img_path = os.path.join(
        VOC_ROOT,
        "JPEGImages",
        img_id + ".jpg"
    )

    xml_path = os.path.join(
        VOC_ROOT,
        "Annotations",
        img_id + ".xml"
    )

    img = cv2.imread(img_path)

    if img is None:
        continue

    blob, ratio = preprocess(img)

    t0 = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_name: blob
        },
    )

    infer_time = (
        time.perf_counter() - t0
    ) * 1000

    all_times.append(infer_time)

    detections = postprocess(
        outputs,
        ratio,
    )

    gts = parse_xml(xml_path)

    for gt in gts:

        gt_counter[gt[4]] += 1

    for det in detections:

        x1, y1, x2, y2, score, cls_id = det

        all_predictions[cls_id].append(
            {
                "image_id": img_id,
                "score": score,
                "bbox": [x1, y1, x2, y2],
            }
        )


results = {}

for iou_thr in IOU_THRESHOLDS:

    aps = []

    print()
    print("=" * 50)

    print(f"mAP@{int(iou_thr * 100)}")

    print("=" * 50)

    for cls_id, cls_name in enumerate(CLASSES):

        preds = sorted(
            all_predictions[cls_id],
            key=lambda x: x["score"],
            reverse=True,
        )

        TP = np.zeros(len(preds))
        FP = np.zeros(len(preds))

        gt_used = {}

        for idx, pred in enumerate(preds):

            img_id = pred["image_id"]

            xml_path = os.path.join(
                VOC_ROOT,
                "Annotations",
                img_id + ".xml"
            )

            gts = parse_xml(xml_path)

            gts = [
                g for g in gts
                if g[4] == cls_id
            ]

            best_iou = 0
            best_gt = -1

            for i, gt in enumerate(gts):

                iou = compute_iou(
                    pred["bbox"],
                    gt[:4],
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt = i

            if best_iou >= iou_thr:

                key = (img_id, best_gt)

                if key not in gt_used:

                    TP[idx] = 1
                    gt_used[key] = True

                else:

                    FP[idx] = 1

            else:

                FP[idx] = 1

        TP = np.cumsum(TP)
        FP = np.cumsum(FP)

        total_gt = gt_counter[cls_id]

        if total_gt == 0:
            continue

        recalls = TP / total_gt

        precisions = TP / (
            TP + FP + 1e-6
        )

        ap = 0

        for t in np.arange(0, 1.1, 0.1):

            if np.sum(recalls >= t) == 0:

                p = 0

            else:

                p = np.max(
                    precisions[recalls >= t]
                )

            ap += p / 11

        aps.append(ap)

        print(
            f"{cls_name:<15} AP: {ap:.4f}"
        )

    mean_ap = np.mean(aps)

    results[iou_thr] = mean_ap

    print()

    print(
        f"mAP@{int(iou_thr * 100)} = "
        f"{mean_ap:.4f}"
    )


print()
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

for k, v in results.items():

    print(
        f"mAP@{int(k * 100)} : "
        f"{v:.4f}"
    )

print()

print(
    f"Average inference: "
    f"{np.mean(all_times):.2f} ms/img"
)