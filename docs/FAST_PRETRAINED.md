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

Fast Pretrained ships with six encoder choices via the "Backbone" combo.
All six carry ImageNet weights.

Both columns are measured, not quoted: each encoder was built through SMP
with `encoder_weights="imagenet"` and run at every tile size the handler
offers (128 to 512), and the parameters counted from the resulting module.
The whole-U-Net figure is the one that costs VRAM and time, and it is what
the dropdown shows.

| Encoder                          | Encoder | Whole U-Net | When to pick |
| -------------------------------- | ------- | ----------- | ------------ |
| `timm-tf_efficientnet_lite0`     | 3.37M   | 5.62M       | Default, best balance. No SE blocks or hard-swish, so it compiles and exports cleanly. |
| `tu-repghostnet_050`             | 0.15M   | 1.64M       | Smallest on offer. Reach for this before Tiny UNet when the data is RGB. |
| `tu-efficientvit_b0`             | 0.68M   | 2.35M       | Very small, with attention in the deeper stages. |
| `timm-mobilenetv3_small_100`     | 0.93M   | 3.59M       | Small and widely used, a safe pick when VRAM or latency is tight. |
| `tu-mobilenetv4_conv_small`      | 1.26M   | 4.98M       | Newer design; worth a try when lite0 underfits. |
| `timm-mobilenetv3_large_100`     | 2.97M   | 6.69M       | Largest here, still well under ResNet-18's 14.3M. |

An earlier version of this table gave 4.2M and 2.0M for the first two. Those
were encoder-ish figures that matched neither the encoder nor the U-Net.

Several appealing candidates did not make the list, and the reason is worth
recording so nobody re-adds them: `ghostnet_050` and `lcnet_035` have no
published ImageNet weights, and `efficientvit_m1` and `xcit_nano_12_p16_224`
reject the downsampling pattern a U-Net decoder needs.

Decoder channels are fixed at `[128, 64, 32, 16, 8]` for every encoder
-- see decoder sizing note below.

## Where the weights come from, and working offline

Every pretrained encoder is fetched from the HuggingFace Hub on first use and
cached in `~/.cache/huggingface/hub`. That cache sits **outside the Appose
environment**, so rebuilding the environment does not carry the weights with
it, and the next training run needs the network to fetch them again. At a
workshop venue, or on a machine behind a proxy, that is a live failure mode --
and it presents as the extension being broken rather than as a download.

**Extensions > DL Pixel Classifier > Utilities > Download Pretrained
Encoders...** fetches all six ahead of time. It is safe to re-run: anything
already present is left alone. It also reports how many cached repositories
have newer weights published.

The same thing from a shell, inside the environment's Python:

```bash
python -m dlclassifier_server.services.encoder_cache prewarm   # download
python -m dlclassifier_server.services.encoder_cache status    # what is cached, what is stale
python -m dlclassifier_server.services.encoder_cache refresh   # replace stale weights
```

`status` exits non-zero when something is stale, so it can gate a build.

### How staleness is decided

On the **weight file**, not the repository. A repository's commit sha moves
when its README changes, and treating that as stale reports encoders as out of
date whenever an author edits a model card -- on one real cache that was three
repositories whose weights had not changed at all. The LFS sha256 of the
weights is the thing that matters, and the cache stores each blob under that
same sha256, so the two compare directly without downloading anything.

`status` checks **every** repository in the cache rather than a recorded
encoder-to-repository mapping, which also covers the histology and foundation
encoders. A mapping was tried first and abandoned: encoder names do not
resolve reliably to repositories (`timm-mobilenetv3_large_100` resolves by
name to `timm/mobilenetv3_large_100.ra_in1k`, but actually loads
`timm/tf_mobilenetv3_large_100.in1k`), watching the cache grow learns nothing
about an encoder that is already cached, and the cache's recorded access time
does not advance on a cache hit.

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
| Learning rate         | 1e-3    | Lower than Tiny UNet; we are fine-tuning |
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
- **GPU augmentation (experimental)**: kornia-based augmentation on
  the GPU. See [TINY_MODEL.md](TINY_MODEL.md#training-speed-notes)
  for the shared notes.
- **torch.compile (experimental, Linux + CUDA)**: wraps the model
  for kernel fusion. See [TINY_MODEL.md](TINY_MODEL.md#training-speed-notes)
  for requirements and caveats. Fast Pretrained's mobile encoders
  (EfficientNet-Lite0 without SE/hard-swish) compile cleanly.

At inference time, `channels_last` memory format is enabled
automatically on CUDA for both encoders here (no SE blocks or BRN
reshape paths to undo the layout propagation).

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
