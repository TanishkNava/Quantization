from onnxruntime.quantization import quantize_dynamic, QuantType

FP32_MODEL = "best_ckpt.onnx"
DYNAMIC_MODEL = "best_dynamic_int8.onnx"

quantize_dynamic(
    model_input=FP32_MODEL,
    model_output=DYNAMIC_MODEL,

    # weights quantized dynamically
    weight_type=QuantType.QInt8,

    # Conv + MatMul + Gemm
    per_channel=True,

    # reduce size
    reduce_range=False,
)

print("Saved:", DYNAMIC_MODEL)