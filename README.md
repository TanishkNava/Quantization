
# Post Training Quantization(Static)

Static quantization means calculating the quantization parameters (Scale and Zero-point) before inference, using a calibration dataset. 

Regardless of whether you use Eager, FX, or PT2E, the underlying math and process of converting float32 to int8 remain the same

First clone the offical repo from the official [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) for the full environment setup
## User Guide

1. Exporting to ONNX
First, ensure your pre-trained YOLOX model is exported to ONNX format using the provided experiment file.

```python
# Standard YOLOX ONNX export (adjust based on your actual export script location)
python tools/export_onnx.py --output-name best_ckpt.onnx -f model/yolox_voc_s3.py -c best_ckpt.pth
```
2. INT8 Quantization
Navigate to the model/ directory and run the quantization script. This step typically requires a calibration dataset to compute the quantization parameters.
create a folder and add sample data (200-500) from Training data.

```python
cd model
# Example command - adjust arguments based on your script
python quantize_int8.py --input best_ckpt.onnx --output yolox_int8.onnx
```
3. Evaluation
To ensure the quantized INT8 model maintains acceptable accuracy and bounding box precision, run the evaluation script:

```python
Bash
python eval_onnx_map.py -f yolox_voc_s3.py -m yolox_int8.onnx
```

4. Inference
Use the scripts in the inf/ folder to run accelerated inference using the quantized model.

```python
cd ../inf
# Example command
python inference.py --model ../model/yolox_int8.onnx --image path/to/test_image.jpg --classes ../model/classes.txt
```
