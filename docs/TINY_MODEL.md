# Tiny UNet: lightweight fast-to-train pixel classifier

Tiny UNet is a hand-rolled depthwise-separable U-Net architecture added in
version 0.6.1. It targets simple 2-5 class microscopy segmentation tasks
where the standard ResNet-UNet (~24M params) is 50-100x more capacity than
the task needs.

## When to use Tiny UNet

Pick Tiny UNet when:
- Your task has 2-5 classes (tissue foreground/background, nuclei,
  simple tissue types, etc.).
- Your data is non-RGB (fluorescence, multichannel, EM) and ImageNet
  pretraining does not transfer.
- You have >=500 annotated tiles and do not need the extra capacity of a
  pretrained encoder.
- Training time matters and you want to iterate quickly.

Pick standard UNet (with a pretrained ResNet/EfficientNet encoder) when:
- Your task is on RGB H&E and you have <1000 annotated tiles. ImageNet
  priors are worth 10-15 points of Dice score in this regime.
- Your task involves fine texture discrimination (cell subtypes, rare
  morphology) where capacity helps.
- You are not memory- or time-constrained.

## Size presets

The "Backbone" combo doubles as a size preset when Tiny UNet is selected:

| Preset         | Base | Depth | Params    | Best for                    |
| -------------- | ---- | ----- | --------- | --------------------------- |
| `tiny-16x4`    | 16   | 4     | ~138k     | Default, balanced           |
| `nano-8x3`     | 8    | 3     | ~10k      | 2-class tasks, fastest      |
| `compact-16x3` | 16   | 3     | ~36k      | Small shallow tasks         |
| `small-24x4`   | 24   | 4     | ~305k     | Extra capacity when needed  |

All presets require tile size divisible by `2^depth` (16 for depth=4,
8 for depth=3). Supported tile sizes: 128, 192, 256, 384, 512.

## Normalization choice

Tiny UNet supports three normalization options. The choice controls
how the network handles channel statistics during training and eval.

### BatchRenorm (default)

BatchRenorm is the default because it delivers the best empirical
accuracy on this codebase's typical microscopy tasks and avoids the
train/eval statistics drift that plagues standard BatchNorm for sliding
window inference. The rmax/dmax warmup loop in `training_service.py`
applies unchanged.

Trade-offs to be aware of:
- `torch.compile` has known graph-break risk on BRN's in-place buffer
  updates. If you plan to enable the experimental `torch.compile` path
  (Phase 3+), switch to GroupNorm.
- The `channels_last` memory format can be silently undone by BRN's
  internal reshape ops. `channels_last` is not auto-enabled for BRN
  models.
- For INT8 post-training quantization (Phase 4), BRN must be folded to
  plain BatchNorm at export time. Handled automatically by the export
  helper.

Reference: Ioffe, "Batch Renormalization: Towards Reducing Minibatch
Dependence in Batch-Normalized Models", NeurIPS 2017. See also
Buglakova et al., "Tiling artifacts and trade-offs of feature
normalization in the segmentation of large biological images",
ICCV 2025 (arXiv:2503.19545).

### GroupNorm

Choose GroupNorm when you plan to enable `torch.compile` or when running
on very small batch sizes (1-2) where even BatchRenorm's running stats
may not have enough samples.

Trade-offs:
- Typically ~0.5-1.5 mIoU below BatchRenorm on RGB H&E.
- Composes cleanly with `torch.compile` and `channels_last`.
- No running-stat drift -- same behaviour in train and eval mode.

Reference: Wu and He, "Group Normalization", ECCV 2018.

### BatchNorm (not recommended)

Plain BatchNorm is kept for compatibility and experimentation only.
Running-statistics drift between train and eval causes visible tiling
artifacts during sliding-window overlay rendering on large images. Use
BatchRenorm or GroupNorm instead.

## Other defaults

Tiny UNet ships with these defaults, chosen for fast training on small
datasets:

| Setting            | Default    | Why                                      |
| ------------------ | ---------- | ---------------------------------------- |
| Epochs             | 30         | Tiny models converge fast                |
| Batch size         | 16         | Fits easily in memory, good gradient     |
| Learning rate      | 3e-3       | Matches OneCycleLR + small model heuristic |
| Tile size          | 256        | Good balance of context and speed        |
| Pretrained weights | Off        | No ImageNet encoder to load              |
| Augmentation       | On (basic) | Flip + rotate + color jitter             |

## Training speed notes

Several checkboxes in the Advanced section of the Training dialog
affect throughput. Defaults are safe; experimental flags are clearly
labeled and off by default.

- **Fused optimizer (CUDA only)**: Enables PyTorch's fused AdamW
  implementation. Saves 2-5 ms/step on tiny models. Safe to leave on;
  silently ignored on CPU.
- **Auto-find learning rate (LR Finder)**: Runs a 100-iteration
  presweep before training to pick a good OneCycleLR peak. For Tiny
  UNet where total training can be <60s, consider disabling to save
  ~10s. When disabled, `max_lr = base_lr * sqrt(batch_size / 8)` is
  used instead.
- **GPU augmentation (experimental, CUDA only)**: Moves data
  augmentation from the CPU (albumentations) to the GPU (kornia). With
  the in-memory dataset preload active, CPU augmentation is typically
  the per-batch bottleneck; kornia can collapse that 10-20x on small
  models. Covers flip/rotate90/color-jitter/noise/blur; arbitrary-angle
  rotation and elastic transforms stay on the CPU path. Falls back
  silently if kornia is not installed or the device is not CUDA.
- **torch.compile (experimental, Linux + CUDA)**: Wraps the model with
  `torch.compile(mode="reduce-overhead")` for kernel fusion and CUDA
  graph capture. Expected 1.4-2x steady-state speedup on tiny models
  after a ~15-40 s one-time compile cost. Gated on Linux, CUDA
  capability >= 7.0, and triton availability; silently falls back to
  eager on any failure. For maximum benefit pair with `norm=gn`
  (BatchRenorm's in-place buffer updates cause graph breaks).

At inference time the extension also enables `channels_last` memory
format on CUDA when the architecture benefits (typical 1.1-1.3x on
conv-heavy models). This is automatic; no UI toggle.

## Experimental inference acceleration (Phase 4)

Two opt-in preferences under **Edit > Preferences > DL Pixel Classifier**
sit behind the label "Experimental" because they depend on GPU driver
stack quirks and can fail silently on atypical setups.

- **TensorRT Inference**: routes ONNX inference through ORT's
  `TensorrtExecutionProvider` on CUDA. Requires `onnxruntime-gpu`
  built with TRT support and a compatible CUDA toolkit. On first
  inference TRT builds an engine from `model_static.onnx` (the static
  variant exported in Phase 3b) and caches it under
  `~/.dlclassifier/tensorrt_cache`. Typical speedup: 2-4x over plain
  `CUDAExecutionProvider`. If TRT is not available, the extension
  silently falls back to the standard provider and logs a warning.
- **INT8 Quantization**: adds `trt_int8_enable=True` to the TRT
  provider options. Only takes effect when "TensorRT Inference" is
  also on. The training pipeline exports an extra ONNX variant
  `model_static_bn.onnx` with BatchRenorm folded to BatchNorm
  (safe at eval time when BRN has converged), because TRT's INT8
  calibrator cannot fuse BRN's r/d correction ops. Expect roughly
  another 2x over FP16 TRT with ~1-2 IoU point drop on simple tasks.

Windows caveats: `onnxruntime-gpu` wheels with TRT support exist but
lag upstream, and cuDNN/CUDA/TRT version combinations are fussy. If
the TRT provider fails to load, check the server logs for the
"TensorrtExecutionProvider not in [...] -- using
CUDAExecutionProvider instead" line and verify
`onnxruntime.get_available_providers()`. Report issues if you hit an
incompatibility so we can add it to a known-bad list.

## References

The design of Tiny UNet draws on:
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
  Image Segmentation", MICCAI 2015.
- Howard et al., "MobileNets: Efficient Convolutional Neural Networks
  for Mobile Vision Applications", 2017 (depthwise separable convs).
- Smith and Topin, "Super-Convergence: Very Fast Training of Neural
  Networks Using Large Learning Rates", 2019 (OneCycleLR).

Agent reports backing the design decisions:
- A1 (minimalist TinyUNet design)
- A2 (counterpoint: SMP + lightweight pretrained encoder)
- B2 (safe-default training speedups)
