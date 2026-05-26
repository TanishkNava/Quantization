import cv2
import numpy as np
import onnxruntime as ort

# =========================
# CONFIG
# =========================

MODEL_PATH = "/home/navavision/Utkarsh-Workspace/Quant/model/yolox_s_int8_1905.onnx"
IMAGE_PATH = "/home/navavision/Utkarsh-Workspace/Quant/model/test.jpg"
CLASS_FILE = "/home/navavision/Utkarsh-Workspace/Quant/model/classes.txt"

INPUT_SIZE = 640

CONF_THRES = 0.25
NMS_THRES = 0.45

# =========================
# LOAD CLASSES
# =========================

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [x.strip() for x in f.readlines()]

# =========================
# ONNX SESSION
# =========================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("INPUT NAME :", input_name)
print("INPUT SHAPE:", session.get_inputs()[0].shape)

# =========================
# PREPROCESS
# =========================

def preprocess(img, input_size=640):

    h, w = img.shape[:2]

    ratio = min(input_size / h, input_size / w)

    resized_h = int(h * ratio)
    resized_w = int(w * ratio)

    resized = cv2.resize(img, (resized_w, resized_h))

    padded = np.full(
        (input_size, input_size, 3),
        114,
        dtype=np.uint8,
    )

    padded[:resized_h, :resized_w] = resized

    padded = padded.astype(np.float32)

    # CHW
    padded = padded.transpose(2, 0, 1)

    # NCHW
    padded = np.expand_dims(padded, axis=0)

    return padded, ratio

# =========================
# NMS
# =========================

def multiclass_nms(boxes, scores, nms_thr):

    final_dets = []

    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):

        cls_scores = scores[:, cls_ind]

        keep = cls_scores > CONF_THRES

        if keep.sum() == 0:
            continue

        valid_scores = cls_scores[keep]
        valid_boxes = boxes[keep]

        # xyxy -> xywh
        bboxes = []

        for b in valid_boxes:

            x1, y1, x2, y2 = b

            bboxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1),
            ])

        indices = cv2.dnn.NMSBoxes(
            bboxes,
            valid_scores.tolist(),
            CONF_THRES,
            nms_thr,
        )

        if len(indices) > 0:

            for i in indices.flatten():

                final_dets.append([
                    *valid_boxes[i],
                    valid_scores[i],
                    cls_ind,
                ])

    return final_dets

# =========================
# POSTPROCESS
# =========================
def postprocess(outputs, ratio):

    predictions = outputs[0]

    print("\n===== OUTPUT INFO =====")
    print("Shape :", predictions.shape)
    print("Dtype :", predictions.dtype)
    print("Min   :", predictions.min())
    print("Max   :", predictions.max())

    if len(predictions.shape) == 3:
        predictions = predictions[0]

    # NO GRID DECODE
    # MODEL ALREADY DECODED

    boxes = predictions[:, :4]

    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.zeros_like(boxes)

    # xywh -> xyxy
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy /= ratio

    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        NMS_THRES
    )

    return dets


# =========================
# MAIN
# =========================

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError(f"Cannot read image: {IMAGE_PATH}")

orig = img.copy()

blob, ratio = preprocess(img)

print("INPUT BLOB SHAPE:", blob.shape)

outputs = session.run(
    None,
    {
        input_name: blob
    }
)

dets = postprocess(outputs, ratio)

print("\nTOTAL DETECTIONS:", len(dets))

for det in dets:

    x1, y1, x2, y2, score, cls_id = det

    # skip invalid
    if not np.isfinite([x1, y1, x2, y2]).all():
        continue

    x1 = max(0, min(int(x1), orig.shape[1] - 1))
    y1 = max(0, min(int(y1), orig.shape[0] - 1))
    x2 = max(0, min(int(x2), orig.shape[1] - 1))
    y2 = max(0, min(int(y2), orig.shape[0] - 1))

    if x2 <= x1 or y2 <= y1:
        continue

    cls_id = int(cls_id)

    label = CLASS_NAMES[cls_id]

    print(
        f"{label} {score:.3f} "
        f"({x1},{y1},{x2},{y2})"
    )

    cv2.rectangle(
        orig,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        orig,
        f"{label} {score:.2f}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# =========================
# SAVE
# =========================

output_file = "qat_onnx_output.jpg"

cv2.imwrite(output_file, orig)

print(f"\nSaved: {output_file}")