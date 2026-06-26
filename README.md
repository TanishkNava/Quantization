
Readme · MD
# YOLOX Quantization
 
A clean, unified workspace for applying **Post-Training Quantization (PTQ)** to YOLOX object detection models. Supports three quantization strategies — ORT static INT8, ORT dynamic INT8, and PyTorch native FX Graph Mode static INT8 — with shared inference and VOC mAP evaluation scripts.
 
---
 
## Contents
 
- [Directory Layout](#directory-layout)
- [Configuration](#configuration)
- [Quantization](#quantization)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Choosing a Strategy](#choosing-a-strategy)
---
 
## Directory Layout
 
```
Quantization/
├── config.yaml             # Unified config for all models
├── utils.py                # Preprocessing, NMS, path resolution helpers
│
├── quantize_ort.py         # ORT static INT8 quantization
├── quantize_dynamic.py     # ORT dynamic INT8 quantization
├── quantize_pytorch.py     # PyTorch native FX Graph Mode static INT8
│
├── eval/
│   ├── infer_onnx.py       # ONNX inference (fp32 / static / dynamic)
│   ├── eval_onnx.py        # VOC mAP evaluation for ONNX models
│   ├── infer_pytorch.py    # TorchScript inference
│   └── eval_pytorch.py     # VOC mAP evaluation for PyTorch models
│
├── calibration/            # Per-model calibration image dirs
│   └── <model>/
├── models/                 # Per-model weights (.pth) and classes.txt
│   └── <model>/
└── exps/                   # Custom YOLOX Exp files
    └── <model>/
```
 
Artifacts (exported ONNX, quantized models) are written to `artifacts/<model>/` at runtime and are not tracked in version control.
 
---
 
## Configuration
 
All models are configured in a single `config.yaml` at the repo root. Paths can be absolute or relative to the repo root — both are resolved automatically.
 
```yaml
ppe_nano:
  exp_file:        exps/ppe_nano/yolox_voc_nano.py
  weights:         models/ppe_nano/weights/best_ckpt.pth
  classes:         models/ppe_nano/classes.txt
  calibration_dir: calibration/ppe_nano
  input_size:      416
  opset:           18
  max_calib_images: 30
  calib_method:    entropy        # MinMax | Entropy | Percentile
  output_dir:      artifacts/ppe_nano
  conf_thres:      0.25
  nms_thres:       0.45
```
 
Set the active model key at the top of each script, or pass it as a CLI argument where supported.
 
---
 
## Quantization
 
All three scripts read from `config.yaml` and write outputs to `artifacts/<model>/`.
 
### ORT Static INT8
 
Exports `.pth` → FP32 ONNX (if not already present), then runs calibration and static quantization.
 
```bash
python quantize_ort.py
```
 
**Outputs:** `artifacts/<model>/fp32.onnx`, `artifacts/<model>/int8.onnx`
 
> Best for edge CPU targets with AVX2/AVX-512 (VNNI) support where activations matter for accuracy. Calibration dataset quality directly affects output fidelity.
 
---
 
### ORT Dynamic INT8
 
Quantizes weights only; activations remain in FP32 and are quantized at runtime. No calibration dataset required.
 
```bash
python quantize_dynamic.py
```
 
**Outputs:** `artifacts/<model>/int8_dynamic.onnx`
 
> More portable across CPU platforms. Trades some accuracy for broader compatibility. Good first step when static quantization shows large accuracy drops.
 
---
 
### PyTorch Native Static INT8 (FX Graph Mode)
 
Runs calibration through the FX-traced model, converts to a quantized representation, and serializes to TorchScript.
 
```bash
python quantize_pytorch.py
```
 
**Outputs:** `artifacts/<model>/int8_pytorch.pt`
 
> Useful for comparing against ORT results or for PyTorch-native deployment pipelines. Note: YOLOX's custom decoupled head can affect operator fusion — validate outputs carefully.
 
---
 
## Inference
 
### ONNX
 
```bash
# Static INT8
python eval/infer_onnx.py --image calibration/ppe_nano/60826.jpg --type int8
 
# Dynamic INT8
python eval/infer_onnx.py --image calibration/ppe_nano/60826.jpg --type dynamic
 
# FP32 baseline
python eval/infer_onnx.py --image calibration/ppe_nano/60826.jpg --type fp32
```
 
**Output:** `artifacts/<model>/infer_ort_<type>_<image_name>.jpg`
 
### PyTorch Native
 
```bash
python eval/infer_pytorch.py --image calibration/ppe_nano/60826.jpg
```
 
**Output:** `artifacts/<model>/infer_pytorch_int8_<image_name>.jpg`
 
---
 
## Evaluation
 
Run mAP@0.5 on a validation split to measure quantization accuracy loss.
 
### ONNX
 
```bash
python eval/eval_onnx.py \
  --type int8 \
  --valid_txt /path/to/valid.txt \
  --data_dir  /path/to/voc_dataset/
```
 
`--type` accepts `fp32`, `int8`, or `dynamic`.
 
### PyTorch Native
 
```bash
python eval/eval_pytorch.py \
  --valid_txt /path/to/valid.txt \
  --data_dir  /path/to/voc_dataset/
```
 
---