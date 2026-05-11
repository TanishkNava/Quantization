import cv2
import numpy as np
import onnxruntime as ort

MODEL = "best_ckpt.onnx"

session = ort.InferenceSession(
    MODEL,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

print("INPUT:", input_name)

img = np.random.uniform(
    0,
    255,
    (1, 3, 640, 640),
).astype(np.float32)

outputs = session.run(
    None,
    {
        input_name: img
    },
)

print("OUTPUTS:", len(outputs))

for i, out in enumerate(outputs):
    print(i, out.shape)
