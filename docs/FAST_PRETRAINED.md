# Fast Pretrained: small U-Net with mobile ImageNet encoders

Fast Pretrained is a compact pixel classifier architecture added in
version 0.6.1. It pairs a small ImageNet-pretrained mobile encoder
(EfficientNet-Lite0 or MobileNetV3-Small) with a scaled-down U-Net
decoder for fast training on small RGB datasets.

## When to use Fast Pretrained

Pick Fast Pretrained when:
- Your task is RGB H&E histopathology with under ~1000 annotated tiles.
- You want the accuracy boost that ImageNet priors provide on natural-
  image morphology (edges, textures, color structure).
- You have a few minutes to spare for training rather than a few
  seconds.

Pick [Tiny UNet](TINY_MODEL.md) instead when:
- Your data is fluorescence, multichannel, or otherwise non-RGB.
  ImageNet priors do not transfer and the added parameters hurt rather
  than help.
- You have >=1000 annotated tiles and want to iterate quickly.
- You want the smallest possible model for edge inference.

Pick the full [UNet](TRAINING_GUIDE.md) when:
- You need the largest encoders (ResNet-50, foundation models).
- Fine-grained feature discrimination is required.
- Training time is not a concern.

## Encoders

Fast Pretrained ships with two encoder choices via the "Backbone" combo:

| Encoder                          | Params | ImageNet top-1 | When to pick |
| -------------------------------- | ------ | -------------- | ------------ |
| `timm-tf_efficientnet_lite0`     | ~4.2M  | 75.1%          | Default -- best balance. No SE blocks or hard-swish means it compiles and exports cleanly. |
| `timm-mobilenetv3_small_100`     | ~2.0M  | 67.7%          | Smallest, fastest. Pick when VRAM or inference latency is tight. |

Decoder channels are fixed at `[128, 64, 32, 16, 8]` for both encoders
-- see decoder sizing note below.

## Decoder sizing

SMP's default U-Net decoder uses `[256, 128, 64, 32, 16]` channels. That
decoder alone has ~5M parameters, which dwarfs any of the mobile encoders
above. Rule of thumb: keep decoder parameters under ~1.5x encoder
parameters.

Our chosen `[128, 64, 32, 16, 8]` yields a ~1.2M-parameter decoder. Total
model size:
- EfficientNet-Lite0 + decoder: ~4.2M + ~1.2M = ~5.4M params
- MobileNetV3-Small + decoder: ~2.0M + ~1.2M = ~3.2M params

This is still small compared to the default UNet + ResNet-34 (~24M params).

## Training defaults

| Setting               | Default | Why                                      |
| --------------------- | ------- | ---------------------------------------- |
| Epochs                | 30      | Fine-tuning converges faster than scratch |
| Batch size            | 16      | Fits easily in memory                    |
| Learning rate         | 1e-3    | Lower than Tiny UNet -- we are fine-tuning |
| Tile size             | 256     | Good context / speed balance             |
| Weight initialization | ImageNet| Default; scratch available               |
| Augmentation          | On      | Flip + rotate + intensity                |

### Discriminative learning rates

Fast Pretrained uses a 1/5 encoder-to-decoder LR ratio instead of the
default 1/10 used for UNet + ResNet. Rationale (per agent report A2):
small mobile encoders have less overspecialized ImageNet features and
benefit from more aggressive adaptation. Concretely:

- Decoder LR: `learning_rate` (default 1e-3)
- Encoder LR: `learning_rate * 0.2` (default 2e-4)

This ratio is emitted by the handler via
`architecture.discriminative_lr_ratio` and picked up by
`training_service.py` when building the optimizer parameter groups.

## Multi-channel inputs

SMP adapts the first convolutional layer automatically when
`in_channels` is set. The behavior depends on input channel count:

- **1 channel** (grayscale): SMP sums the pretrained RGB weights along
  the input dimension. This preserves edge filters better than
  averaging.
- **2 channels**: keeps first two of three RGB weights and rescales to
  preserve activation magnitude.
- **3 channels**: pretrained weights used as-is (the common case).
- **4-7 channels**: SMP tiles the RGB weights to fill extra channels.
  Works out of the box but may slightly underperform an explicit
  mean-of-RGB initialization for fluorescence data; a future
  optimization can add per-domain init strategies.

For heavily non-RGB data (dense fluorescence panels, phase contrast,
EM) the Tiny UNet from-scratch path is usually a better choice than
adapting ImageNet weights.

## Training speed notes

Fast Pretrained benefits from the same training-speed options as
other architectures:
- **Fused optimizer**: enabled by default, saves 2-5 ms/step on CUDA.
- **In-memory dataset**: `auto` by default, preloads all patches to
  RAM after pre-flight RAM check.
- **Auto-find learning rate**: runs LR Finder presweep before
  OneCycleLR; can be disabled to save ~10 s per training run.

## References

Agent reports backing the design:
- A2 (counterpoint: SMP + lightweight pretrained encoder)

External references:
- Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks", ICML 2019 (original EfficientNet, lite0 drops SE
  blocks and hard-swish for mobile-friendliness).
- Howard et al., "Searching for MobileNetV3", ICCV 2019.
- Raghu et al., "Transfusion: Understanding Transfer Learning for
  Medical Imaging", NeurIPS 2019 (ImageNet priors help most with
  scarce data and few classes).
