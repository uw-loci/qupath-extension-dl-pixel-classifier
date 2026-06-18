# Normalization round-trip contract

**Read this before adding or changing any preprocessing field** (normalization
strategy, per-channel flag, clip percentile, channel selection, bit depth,
context scale, downsample, resampling, ...).

## Why this exists

A trained segmentation model only works at inference if it receives pixels
preprocessed **exactly** as they were during training. If any preprocessing
parameter used at training is not reproduced at inference, the model sees a
distribution it never learned and the output can be badly wrong -- including a
whole output class silently disappearing.

This bit us on 2026-06-17: a 3-class brightfield model (background / blue tissue
/ DAB-brown tissue) trained to validation mIoU 0.977 with **Tumor IoU 0.998**,
yet at inference **no Tumor pixel was ever predicted**. Root cause: training
exported `per_channel=false` (single shared normalization range), but the
inference path built its `ChannelConfiguration` from a live UI default of
`per_channel=true` and never read the value back from the model. Per-channel
stretching shifted the color balance enough that the brown class never won the
argmax. The model was fine; the preprocessing was desynced.

## The contract

Every preprocessing parameter that affects the pixels fed to the model MUST
travel this full loop. A parameter that stops at any stage is a latent
train/inference desync.

1. **Training export** -- `AnnotationExtractor` writes the value into the export
   `config.json` (`channel_config.normalization` and `input_config.normalization`).
   *Today `per_channel` is hardcoded `false` here -- that is the source of truth.*
2. **Model save** -- the Python `TrainingService._save_model` persists
   `input_config.normalization` (and `channel_config`) into the model's
   `metadata.json`. It must not drop fields.
3. **Metadata parse** -- `ModelManager` reads the value back out of
   `metadata.json` into `ClassifierMetadata`. Missing value -> training-safe
   default + a `WARN` so degraded older models are visible in the log.
4. **Metadata carrier** -- `ClassifierMetadata` exposes a getter, and `toMap()`
   re-emits the field so a re-saved model keeps it.
5. **Apply-path build** -- every place that builds a `ChannelConfiguration` for
   inference (`SetupDLClassifier`, `DLClassifierScripts`, overlay/objects paths)
   sets the field from the metadata getter -- never from a live UI default.
6. **Handler build** -- the per-architecture handlers
   (`UNetHandler`, `TinyUNetHandler`, `FastPretrainedHandler`, `MuViTHandler`,
   `CustomONNXHandler`) that construct the trained `ClassifierMetadata` copy the
   field from the training `ChannelConfiguration`.
7. **Inference config** -- `ApposeClassifierBackend.buildInputConfig` emits the
   field into the `input_config` sent to Python, which the inference scripts and
   `InferenceService._normalize` consume.

`ChannelConfiguration`'s builder defaults are set to the **training-safe**
values (`perChannelNormalization=false`, `clipPercentile=99.0`) so that any
apply-path builder which forgets to set a field inherits the value training
actually used, not an arbitrary UI default.

## Checklist for adding a new preprocessing field

- [ ] Persist it in `AnnotationExtractor`'s export `config.json`.
- [ ] Confirm `TrainingService._save_model` carries it into `metadata.json`.
- [ ] Parse it in `ModelManager` with a training-safe default + `WARN` when absent.
- [ ] Add field + getter + builder + `toMap()` entry in `ClassifierMetadata`.
- [ ] Add field + builder default (training-safe) in `ChannelConfiguration`, and
      copy it in `withPrecomputedStats`.
- [ ] Set it from metadata in every apply-path `ChannelConfiguration.builder()`.
- [ ] Copy it from the training `ChannelConfiguration` in every handler.
- [ ] Emit it in `ApposeClassifierBackend.buildInputConfig`.
- [ ] Consume it in `InferenceService._normalize` / the inference scripts.

## How to verify the loop is intact

Train a tiny model, then inspect the saved `metadata.json` -- the field must be
present under `channel_config.normalization` / `input_config.normalization`.
Apply the model and confirm the `input_config.normalization` in the Appose
inference request (visible in the Python console / log) matches the trained
value. A mismatch between the training config log line and the inference request
is the signature of a broken round-trip.
