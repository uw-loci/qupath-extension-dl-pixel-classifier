# Inference Guide

Step-by-step guide to applying a trained classifier to images.

## Overview

Inference applies a trained pixel classifier to new images, producing results as measurements, detection objects, or a color overlay.

## Step 1: Open an Image

Open the image you want to classify in QuPath. The image type (brightfield, fluorescence) should match what the classifier was trained on.

## Step 2: Open the Inference Dialog

Go to **Extensions > DL Pixel Classifier > Apply DL Pixel Classifier...**

## Step 3: Select a Classifier

The classifier table shows all available trained models with columns for:
- **Name** -- classifier identifier
- **Type** -- architecture (e.g., unet, muvit)
- **Channels** -- input channel count (with context scale info, e.g., "3 +2x ctx")
- **Classes** -- number of output classes
- **Trained** -- training date

Click a row to select it. The info panel below shows the architecture + backbone, input channels + context scale + tile dimensions, downsample level, and class names.

### Channel Mapping

The **CHANNEL MAPPING** section shows how the classifier's expected channels map to the current image. Each row displays:
- **Expected** channel name from the classifier
- **Mapped To** the image channel it will use
- **Status**: [OK] (exact match), [?] (fuzzy/substring match), or [X] (unmapped)

For unmatched channels, use the dropdown to manually remap to the correct image channel. For brightfield images, channels are auto-configured and this section collapses automatically.

## Step 4: Configure Output Type

| Output Type | Description | Best for |
|-------------|-------------|----------|
| **MEASUREMENTS** | Adds per-class probability values as annotation measurements | Quantification (% area per class) |
| **OBJECTS** | Creates detection or annotation objects from the classification map. Uses the same unified inference pipeline as the overlay -- predictions are identical. | Spatial analysis, counting structures |

### Object output options (OBJECTS only)

| Option | Description | Typical value |
|--------|-------------|---------------|
| **Object Type** | DETECTION (lightweight) or ANNOTATION (editable) | DETECTION for quantification |
| **Min Object Size** | Discard objects smaller than this area (um^2) | 10-100 um^2 |
| **Hole Filling** | Fill holes smaller than this area (um^2) | 5-50 um^2 |
| **Boundary Smoothing** | Simplify jagged boundaries (microns tolerance) | 0.5-2.0 um |

## Step 5: Configure Processing Options

These options are collapsed by default. Expand **PROCESSING OPTIONS** to adjust.

| Option | Default | Description |
|--------|---------|-------------|
| **Tile Size** | Auto-set from classifier | Should match training tile size. Range: 64-8192. |
| **Tile Overlap (%)** | 12.5% | Higher = better blending but slower (max 50%). See below. |
| **Blend Mode** | CENTER_CROP | How overlapping tiles merge. See blend mode details below. CENTER_CROP recommended. |
| **Use GPU** | Yes | 10-50x faster than CPU |
| **Test-Time Augmentation (TTA)** | No | Apply D4 transforms (flips + 90-degree rotations) and average predictions. ~8x slower but typically 1-3% better quality. Best for final production runs. |

### Tile overlap and context

The overlap setting controls how much context the model sees around each tile's visible region. Both the overlay and Apply Classifier use the same computation (`InferenceConfig.computeEffectivePadding`) with guardrails:

- **Minimum**: 25% of tile size per side (e.g., 128px for 512 tiles)
- **Maximum**: 3/8 of tile size per side (ensures stride >= 25% of tile size)
- **Floor**: 64px absolute minimum

The effective stride (visible pixels per tile) is `tileSize - 2 * effectivePadding`. For a 512px tile with default settings, this produces stride=256, meaning 50% overlap between adjacent tiles and each pixel's prediction comes from a tile's center region where the model has full context on all sides.

### Expanded reads (real context)

Each tile is read from the image as a tileSize-sized region (not just the stride portion). This provides the model with **real neighboring pixel data** at every tile boundary -- no artificial reflection padding. The output is center-cropped to the stride region, discarding edge predictions where the model has less context. This follows the recommendation from Buglakova et al. (ICCV 2025): "Completely remove halo region during stitching."

### Unified pipeline (overlay = objects)

The **OBJECTS** output and the **overlay** use the exact same inference pipeline (`DLPixelClassifier.applyClassification()`). This guarantees identical predictions: same expanded reads, same normalization, same model input, same center-crop, same Gaussian smoothing. The only difference is output format: the overlay renders pixels directly, while OBJECTS vectorizes the classification map into PathObjects via contour tracing.

### Image-level normalization

The extension automatically computes normalization statistics across the entire image before starting inference. This ensures all tiles receive identical input normalization, eliminating the "blocky" tile boundary artifacts that occur when each tile independently computes its own statistics.

**Priority order:**
1. **Training dataset statistics** (best) -- stored in model metadata for newly trained models
2. **Image-level sampling** -- samples ~16 tiles in a 4x4 grid across the image (~1-3s one-time cost)
3. **Per-tile normalization** -- fallback if sampling fails

This is fully automatic and requires no configuration.

### BatchRenorm (model-internal normalization)

Newly trained models use **BatchRenorm** instead of standard BatchNorm for the network's internal normalization layers. Standard BatchNorm causes a train/eval disparity: running statistics accumulated during training diverge from actual tile statistics during inference, creating artifacts that no amount of overlap or blending can fix. BatchRenorm uses consistent global statistics in both modes, producing seamless tiled predictions. Older models trained with standard BatchNorm will continue to work but may exhibit more visible tile boundaries.

## Step 6: Set Application Scope

| Scope | Description |
|-------|-------------|
| **Apply to whole image** | Classify the entire image (no annotations needed) |
| **Apply to all annotations** | Classify within all annotations (default) |
| **Apply to selected annotations only** | Classify only within selected annotations |

> **Tip**: Use "Apply to selected annotations only" to test on a small region before processing the entire image.

### Backup option

Check **Create backup of annotation measurements** to save existing measurements before overwriting. Recommended when re-running inference on previously classified images.

## Step 7: Apply

Click **Apply** to start inference. Progress is shown in the QuPath log.

> **Note:** All inference dialog settings (output type, blend mode, smoothing, tile size, overlap, GPU, object type, min object size, hole filling, application scope, backup) are remembered across sessions.

## Live Overlay Mode

For quick visual inspection without the full inference dialog:

1. **Extensions > DL Pixel Classifier > Select Overlay Model...** -- choose a trained classifier
2. **Extensions > DL Pixel Classifier > Toggle Prediction Overlay** -- check to enable, uncheck to remove
3. The overlay renders as you pan and zoom using CENTER_CROP tile handling (artifact-free boundaries)
4. Toggle off and on again without re-selecting the model -- the selection persists

If you toggle the overlay on without selecting a model first, you will be prompted to choose one.

### Overlay Settings

Overlay behavior is configured in **Edit > Preferences > DL Pixel Classifier**:

- **Overlay Prediction Smoothing** (sigma 0-10, default 2.0): Gaussian smoothing of probability maps before classification. Higher values reduce noisy per-pixel predictions, especially for ResNet/CNN models. MuViT/ViT models produce smoother predictions and may need less smoothing.
- **Overlay High-Quality Tile Averaging** (default off): Run each tile at 4 spatial offsets and average the predictions to eliminate tile boundary seams. Applies to both the overlay and Apply Classifier. ~4x slower.

Changes are applied immediately to the active overlay. The overlay always uses CENTER_CROP tile handling.

## Copy as Script

Click the **"Copy as Script"** button (left side of the button bar) to generate a Groovy script matching your current settings. See [SCRIPTING.md](SCRIPTING.md) for batch processing.
