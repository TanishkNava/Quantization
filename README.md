# YOLOX Quantization & Evaluation Workspace

This repository provides a clean, unified workspace for applying **Post-Training Quantization (PTQ)** to YOLOX object detection models. It supports both **ONNX Runtime (ORT)** and **PyTorch Native (FX Graph Mode)** quantization, alongside dedicated scripts for inference and VOC mAP evaluation.

---

## 📁 Directory Layout

The repository has been streamlined to contain only the necessary configuration, calibration data, and scripts:

```text
Quantization/
├── config.yaml          # Unified configuration file for all models (dock, gate, ppe_nano, etc.)
├── utils.py             # Shared helpers for preprocessing, postprocessing (NMS), and path resolution
├── quantize_ort.py      # ONNX Runtime static INT8 quantization script
├── quantize_pytorch.py  # PyTorch native static INT8 quantization script
├── quantize_dynamic.py  # ONNX Runtime dynamic INT8 quantization script
├── eval/                # Evaluation and inference scripts
│   ├── infer_onnx.py    # Run inference on an ONNX model (FP32, static INT8, dynamic INT8)
│   ├── eval_onnx.py     # Compute VOC mAP for an ONNX model
│   ├── infer_pytorch.py # Run inference on a serialized PyTorch quantized model (.pt)
│   └── eval_pytorch.py  # Compute VOC mAP for a PyTorch quantized model
├── calibration/         # Calibration image directories per model (e.g. calibration/ppe_nano/)
├── models/              # Tracked weights (.pth) and classes.txt per model (e.g. models/ppe_nano/)
└── exps/                # Custom YOLOX Exp files (e.g. exps/ppe_nano/yolox_voc_nano.py)
```

---

## ⚙️ Unified Configuration (`config.yaml`)

All model configurations are kept in a single `config.yaml` file at the root. Path resolution handles absolute directories and resolves relative paths automatically against the repository root.

Example configuration block:
```yaml
ppe_nano:
  exp_file: exps/ppe_nano/yolox_voc_nano.py
  weights: models/ppe_nano/weights/best_ckpt.pth
  classes: models/ppe_nano/classes.txt
  calibration_dir: calibration/ppe_nano
  input_size: 416
  opset: 18
  max_calib_images: 30
  calib_method: entropy   # options: MinMax, Entropy, Percentile
  output_dir: artifacts/ppe_nano
  conf_thres: 0.25
  nms_thres: 0.45
```

---

## 🚀 Execution Guide

Modify the active configuration in `config.yaml` to point to the desired model paths and settings.

### 1. ONNX Runtime (ORT) Static Quantization
This script exports the `.pth` weights to FP32 ONNX (if not already exported) and performs static quantization:
```bash
python quantize_ort.py
```
*Outputs: `artifacts/<model>/fp32.onnx`, `artifacts/<model>/int8.onnx`*

### 2. ONNX Runtime (ORT) Dynamic Quantization
Quantizes only weights dynamically (compatible with standard CPU platforms):
```bash
python quantize_dynamic.py
```
*Outputs: `artifacts/<model>/int8_dynamic.onnx`*

### 3. PyTorch Native Static Quantization
Performs native PyTorch PTQ in FX Graph Mode. Runs calibration using calibration dataset, converts the model, traces it, and serializes it to a TorchScript file:
```bash
python quantize_pytorch.py
```
*Outputs: `artifacts/<model>/int8_pytorch.pt`*

---

## 📊 Inference and Evaluation Guide

### 1. Run Inference

#### ONNX Runtime Inference
To run inference on an image using the static INT8 ONNX model:
```bash
python eval/infer_onnx.py --image calibration/ppe_nano/60826.jpg --type int8
```
To run using the dynamically quantized model, specify `--type dynamic`.
*Outputs: `artifacts/<model>/infer_ort_<type>_<image_name>.jpg`*

#### PyTorch Native Quantized Inference
To run inference on an image using the TorchScript serialized model:
```bash
python eval/infer_pytorch.py --image calibration/ppe_nano/60826.jpg
```
*Outputs: `artifacts/<model>/infer_pytorch_int8_<image_name>.jpg`*

---

### 2. Run Dataset Evaluation (mAP)

#### ONNX Runtime Evaluation
Evaluate mAP@0.5 on a validation dataset:
```bash
python eval/eval_onnx.py \
    --type int8 \
    --valid_txt path/to/valid.txt \
    --data_dir path/to/voc_dataset_dir
```

#### PyTorch Native Evaluation
Evaluate mAP@0.5 of the PyTorch native quantized model on a validation dataset:
```bash
python eval/eval_pytorch.py \
    --valid_txt path/to/valid.txt \
    --data_dir path/to/voc_dataset_dir
```
