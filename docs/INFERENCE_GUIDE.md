# Inference Guide

Step-by-step guide to applying a trained classifier to images.

## Overview

Inference applies a trained pixel classifier to new images, producing results as measurements, detection objects, or a color overlay.

## Step 1: Open an Image

Open the image you want to classify in QuPath. The image type (brightfield, fluorescence) should match what the classifier was trained on.

## Step 2: Open the Inference Dialog

Go to **Extensions > DL Pixel Classifier > Apply Classifier...**

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
| **RENDERED_OVERLAY** | Batch inference with tile blending, producing a seamless overlay. **Default and recommended.** Best for validating classifier quality -- accurately represents what OBJECTS output would look like. | Quality validation, visual comparison |
| **MEASUREMENTS** | Adds per-class probability values as annotation measurements | Quantification (% area per class) |
| **OBJECTS** | Creates detection or annotation objects from the classification map | Spatial analysis, counting structures |
| **OVERLAY** | Renders a live on-demand color overlay as you pan and zoom. Uses CENTER_CROP tile handling (artifact-free) with configurable Gaussian smoothing. | Quick visual inspection |

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

### Tile overlap and blending

Overlap determines how much adjacent tiles share:

| Overlap | Quality | Speed | Notes |
|---------|---------|-------|-------|
| 0% | Seams visible | Fastest | Objects may split at tile boundaries |
| 5-10% | Moderate | Fast | Some seam reduction |
| 10-15% | Good | Moderate | Recommended for seamless results |
| 15-25% | Best | Slower | ~2x processing time vs 0% |
| 25-50% | Diminishing returns | Much slower | Only needed for very large receptive fields |

### Real-data context padding

During inference, QuPath provides real surrounding image data around each tile via `inputPadding`. QuPath's `inputPadding` is **per-side** -- the visible stride equals `tileSize - 2 * inputPadding`. The padding amount is computed automatically: for CENTER_CROP mode, padding is `tileSize/4` per side (center 50% visible); for other blend modes, padding is `max(64, min(max(overlap, tileSize/4), tileSize * 3/8))` per side. This provides the CNN with real context at every tile boundary, eliminating the need for artificial reflection padding -- the model always sees real image data, matching how training tiles are extracted with real surrounding context.

The **blend mode** controls how overlapping predictions merge:

| Blend Mode | Description | Recommended for |
|------------|-------------|-----------------|
| **CENTER_CROP** | Keep only center predictions, discard overlap margins. **Default and recommended.** Zero boundary artifacts. Each pixel comes from a single tile's center where predictions are most reliable. | All models. Required for overlay mode. |
| **LINEAR** | Weighted average favoring tile centers. Good balance of quality and speed. | Batch inference (RENDERED_OVERLAY, OBJECTS) with CNN models |
| **GAUSSIAN** | Cosine-bell blending for smoother transitions. | Batch inference with ViT/MuViT models |
| **NONE** | No blending; last tile wins. Fastest but may show visible tile seams. | Debugging, or with 0% overlap |

> **Note:** For the live **OVERLAY** output type, CENTER_CROP is always used (the blend mode selector is disabled). This follows the recommendation from Buglakova et al. (ICCV 2025): "Completely remove halo region during stitching (don't blend it)." For batch inference (RENDERED_OVERLAY, OBJECTS), blend modes remain selectable since Python-side blending operates on the full tile batch.

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

> **Note:** All inference dialog settings (output type, blend mode, smoothing, application scope, backup) are remembered across sessions.

## Live Overlay Mode

For quick visual inspection without the full inference dialog:

1. **Extensions > DL Pixel Classifier > Select Overlay Model...** -- choose a trained classifier
2. **Extensions > DL Pixel Classifier > Toggle Prediction Overlay** -- check to enable, uncheck to remove
3. The overlay renders as you pan and zoom using CENTER_CROP tile handling (artifact-free boundaries)
4. Toggle off and on again without re-selecting the model -- the selection persists

If you toggle the overlay on without selecting a model first, you will be prompted to choose one.

### Overlay Settings

**Extensions > DL Pixel Classifier > Utilities > Overlay Settings...** lets you configure:

- **Prediction Smoothing** (sigma 0-10, default 2.0): Gaussian smoothing of probability maps before classification. Higher values reduce noisy per-pixel predictions, especially for ResNet/CNN models. MuViT/ViT models produce smoother predictions and may need less smoothing.

Changes are applied immediately to the active overlay. The overlay always uses CENTER_CROP tile handling.

## Copy as Script

Click the **"Copy as Script"** button (left side of the button bar) to generate a Groovy script matching your current settings. See [SCRIPTING.md](SCRIPTING.md) for batch processing.
