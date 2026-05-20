# DL Classifier Python Library

Python deep learning library for pixel classification, used by the QuPath DL Pixel Classifier extension via [Appose](https://github.com/apposed/appose) shared-memory IPC.

## Overview

This library provides the training and inference engines that power the DL Pixel Classifier extension. It is **not** used as a standalone server -- the QuPath extension communicates with it directly through Appose's embedded Python environment and shared-memory tile transfer.

## Features

- **Multi-device support**: CUDA, Apple Silicon (MPS), and CPU fallback
- **GPU memory management**: Automatic cache clearing and memory monitoring
- **Multiple architectures**: UNet, UNet++, DeepLabV3+, FPN, PSPNet, MANet, LinkNet, PAN
- **Pretrained encoders**: ResNet, EfficientNet, MobileNet, DenseNet, VGG, and more
- **Histology-pretrained encoders**: ResNet-50 models pretrained on TCGA/Lunit/Kather100K tissue patches via timm
- **Transfer learning**: Layer-level freeze/unfreeze control with encoder-aware recommendations
- **ONNX export**: Automatic export for deployment
- **Sparse annotation support**: Works with line/brush annotations (UNLABELED_INDEX=255)

## Requirements

- Python 3.11+
- PyTorch 2.1+
- CUDA-capable GPU (recommended, but not required)

## Project Structure

```
python_server/
  dlclassifier_server/
    services/
      training_service.py      # PyTorch training loop
      inference_service.py     # ONNX + PyTorch inference
      pretrained_models.py     # Encoder/architecture catalog
      model_registry.py        # Trained-model registry
      muvit_model.py           # MuViT model factory
      job_manager.py           # Async job tracking
      gpu_manager.py           # CUDA/MPS/CPU detection
    utils/                     # Shared utilities (normalization, etc.)
      batchrenorm.py           # BatchRenorm layer replacement
  tests/
    conftest.py                # Shared fixtures
    generate_test_data.py      # Synthetic data generator
    test_gpu_manager.py        # GPU manager tests
    test_training_service.py   # Training service tests
    test_inference_service.py  # Inference service tests
```

## Installation (Development Only)

> **End-users do not need to install this package manually.** The QuPath extension automatically manages the Python environment via Appose. This section is for developers only.

```bash
cd python_server
pip install -e ".[dev]"
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Running with Coverage

```bash
pytest tests/ --cov=dlclassifier_server --cov-report=html
```

## Training Data Format

### Directory Structure

Training data must be organized in the following structure:

```
training_data/
  config.json                   # Configuration file (see below)
  train/
    images/
      patch_0000.tiff           # RGB or multi-channel TIFF images
      patch_0001.tiff
      ...
    masks/
      patch_0000.png            # Single-channel masks
      patch_0001.png
      ...
  validation/
    images/
      patch_0004.tiff
      ...
    masks/
      patch_0004.png
      ...
```

### config.json Format

```json
{
  "patch_size": 512,
  "context_padding": 128,
  "classes": ["Background", "Foreground"],
  "class_weights": [1.0, 2.0],
  "unlabeled_index": 255
}
```

| Field | Description |
|-------|-------------|
| `patch_size` | Tile size in pixels (center content region) |
| `context_padding` | Pixels of real-data border per side (0 = disabled). When >0, tiles on disk are `(patch_size + 2*context_padding)` pixels wide and masks have a 255 border. |
| `classes` | List of class names (order = index) |
| `class_weights` | Optional inverse-frequency weights for class balancing |
| `unlabeled_index` | Pixel value for unlabeled regions (typically 255) |

### Image Requirements

| Property | Specification |
|----------|---------------|
| **Format** | TIFF (preferred), PNG, or JPEG |
| **Size** | Typically 256x256 or 512x512 pixels |
| **Channels** | 1-N (grayscale, RGB, or multi-channel) |
| **Bit depth** | 8-bit or 16-bit |

### Mask Requirements

| Property | Specification |
|----------|---------------|
| **Format** | PNG (single-channel) |
| **Size** | Must match corresponding image |
| **Values** | 0 = class 0, 1 = class 1, ..., 255 = unlabeled |
| **Type** | uint8 |

### Sparse Annotation Support

The training system supports sparse annotations where only part of each image is labeled:

- **Unlabeled pixels**: Set to 255 (or value specified in `unlabeled_index`)
- **Loss computation**: Both CE+Dice and CrossEntropy losses use `ignore_index=255` to skip unlabeled pixels
- **Class weights**: Calculated from labeled pixels only

This allows training from line/brush annotations without requiring full image segmentation.

## Available Options

**Architectures**:
- `unet`, `unet++`, `deeplabv3`, `deeplabv3+`, `fpn`, `pspnet`, `manet`, `linknet`, `pan`

**Encoders**:
- ImageNet-pretrained: `resnet34`, `resnet50`, `efficientnet-b0`, `efficientnet-b4`, `mobilenet_v2`, `densenet121`, `vgg16`, `dpn68`, `resnext50_32x4d`, `se_resnet50`, `timm-efficientnet-b3`, `mit_b2`
- Histology-pretrained (ResNet-50, downloaded from HuggingFace via timm on first use, ~100MB each):
  - `resnet50_lunit-swav` -- Lunit SwAV, 19M TCGA patches (non-commercial)
  - `resnet50_lunit-bt` -- Lunit Barlow Twins, 19M TCGA patches (non-commercial)
  - `resnet50_kather100k` -- Kather100K colorectal tissue (CC-BY-4.0)
  - `resnet50_tcga-brca` -- TCGA-BRCA breast cancer SimCLR (GPLv3)

**Normalization Strategies**:
- `min_max` - Scale to [0, 1] using min/max values
- `percentile_99` - Clip at 99th percentile, then normalize
- `z_score` - Standard normalization (x - mean) / std
- `fixed_range` - User-specified min/max values

**Learning Rate Schedulers**:
- `onecycle` - One-cycle policy: smooth ramp-up then decay (default, recommended)
- `cosine` - Cosine annealing with warm restarts
- `step` - Step decay (reduce LR by factor every N epochs)
- `none` - Constant learning rate

**Loss Functions**:
- `ce_dice` - Combined Cross-Entropy + Dice loss (default, recommended for segmentation)
- `cross_entropy` - Standard Cross-Entropy loss

**Early Stopping Metrics**:
- `mean_iou` - Stop when mean IoU plateaus (default, recommended)
- `val_loss` - Stop when validation loss stops decreasing

**Mixed Precision**:
- `true` - Enable automatic mixed precision on CUDA GPUs (~2x speedup, default)
- `false` - Use full FP32 precision

## Model Output Structure

After training, models are saved with:

```
model_dir/
  model.pt              # PyTorch state dict
  model.onnx            # ONNX export (auto-generated)
  metadata.json         # Model metadata
```

### metadata.json Format

```json
{
  "id": "unet_20260123_141500",
  "name": "UNET Classifier",
  "architecture": {
    "type": "unet",
    "backbone": "resnet34",
    "use_pretrained": true
  },
  "input_config": {
    "num_channels": 3,
    "normalization": {
      "strategy": "percentile_99",
      "per_channel": true,
      "clip_percentile": 99.0
    }
  },
  "classes": [
    {"index": 0, "name": "Background"},
    {"index": 1, "name": "Foreground"}
  ]
}
```

## GPU Support

### Device Priority

1. **CUDA** (NVIDIA GPUs) - Highest priority
2. **MPS** (Apple Silicon) - Second priority
3. **CPU** - Fallback

### Memory Management

The library includes automatic GPU memory management:

- **Cache clearing**: Automatically clears GPU cache between epochs
- **Memory monitoring**: Logs memory usage during training (CUDA)
- **Memory estimation**: Reports model memory requirements
- **Training cleanup**: `try/finally` guarantees GPU memory is freed after training completes, fails, or is cancelled

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch first (see Requirements) |
| CUDA out of memory | Reduce batch size in training params, or use a smaller encoder (e.g., `mobilenet_v2`) |
| MPS not detected on Mac | Requires macOS 12.3+ and PyTorch 2.0+ |

### Checking Your Environment

**Verify PyTorch and GPU:**

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()}')"
```

## License

Apache License 2.0
