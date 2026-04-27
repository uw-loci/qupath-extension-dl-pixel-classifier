# Training Guide

Step-by-step guide to training a deep learning pixel classifier.

## Overview

Training a classifier involves:
1. Preparing annotated training data in QuPath
2. Configuring the model and training parameters
3. Running training with progress monitoring
4. Reviewing training areas to identify annotation issues (optional)
5. Continuing training if needed (optional)
6. Verifying the trained model

## Step 1: Prepare Annotations

### Create annotation classes

1. Open an image in QuPath
2. Right-click in the annotation class list (left panel) to add classes
3. Create at least two classes (e.g., "Tumor", "Stroma", "Background")

### Draw annotations

1. Use the brush, polygon, or polyline tools to annotate regions
2. Assign each annotation to a class (right-click > Set class)
3. Annotate representative examples of each class throughout the image

### Annotation best practices

- **Quality over quantity**: A few well-placed annotations are better than many sloppy ones
- **Cover variability**: Include examples from different regions, staining intensities, and tissue morphologies
- **Balance classes**: Try to annotate roughly similar areas for each class
- **Include boundaries**: Annotate along class boundaries where the model needs to make decisions
- **Use brush annotations**: The brush tool is efficient for painting regions. Line annotations also work but require setting the line stroke width parameter

## Step 2: Open the Training Dialog

Go to **Extensions > DL Pixel Classifier > Train Classifier...**

The dialog opens in **Basic mode** by default, showing only the essentials: Training Data Source, Annotation Classes, and Classifier Name. Click **"Show All Settings"** in the header to reveal all configuration options. The mode is remembered across sessions.

In advanced mode, the dialog has collapsible titled pane sections:

1. **TRAINING DATA SOURCE** -- select images and load classes
2. **MODEL ARCHITECTURE** -- architecture and encoder selection
3. **WEIGHT INITIALIZATION** -- how to initialize model weights
4. **TRAINING PARAMETERS** -- epochs, batch size, tile size, etc.
5. **TRAINING STRATEGY** -- scheduler, loss, early stopping (collapsed)
6. **CHANNEL CONFIGURATION** -- input channel selection and normalization
7. **ANNOTATION CLASSES** -- class selection and weight balancing
8. **DATA AUGMENTATION** -- flip, rotation, intensity augmentation (collapsed)
9. **NAME YOUR CLASSIFIER** -- name and description

## Loading Settings from a Previous Model

To pre-populate dialog settings from a previously trained classifier:

1. In the **WEIGHT INITIALIZATION** section, select the **"Continue training from saved model"** radio button
2. Click **"Select model..."** to open the model picker
3. Select a model from the table (sorted by date, newest first)
4. Click **OK**

This populates:
- **Architecture, backbone, tile size, downsample, context scale, epochs** from the model metadata
- **Learning rate, batch size, augmentation, scheduler, loss function, early stopping, and all other hyperparameters** from the model's saved training settings
- **Classifier name** auto-generated as `Retrain_OriginalName_YYYYMMDD`
- **Class auto-matching** after you load classes from images -- classes matching the source model are auto-selected

All fields can be adjusted before training. Older models (trained before this feature) will only populate the architecture-level settings; hyperparameters will keep their preference defaults.

### UI Locking When Continuing Training

When **Continue training from saved model** is selected and a model is loaded, the following controls are **locked** because the saved weights require an exact architecture match:

- Architecture (UNet, MuViT, Custom ONNX)
- Backbone/Encoder (resnet34, resnet50, etc.)
- MuViT handler parameters (model size, patch size, level scales, position encoding)
- Tile Size
- Resolution (Downsample)
- Context Scale

You **can** still adjust: learning rate, epochs, batch size, augmentation, training strategy, channel configuration, and class selection/weights.

## Step 3: Configure Basic Settings

### Training Data Source

Check the project images to include in training. Only images with classified annotations are shown.

| Step | Description |
|------|-------------|
| **1. Select images** | Check images in the list. Use "Select All" / "Select None" for bulk selection. |
| **2. Load Classes** | Click **"Load Classes from Selected Images"** to populate the class list and initialize channels from the first image. |

Multi-image training combines patches from all selected images into one training set, improving generalization. If you previously loaded settings from a model, classes matching the source model are auto-selected after loading.

### Classifier Info

| Setting | Description |
|---------|-------------|
| **Classifier Name** | Unique identifier (letters, numbers, underscore, hyphen). Used as filename. |
| **Description** | Optional free-text description for documentation. |

## Step 4: Configure Model Architecture

### Architecture

| Architecture | Best for | Reference |
|-------------|----------|-----------|
| **UNet** | General-purpose segmentation. Good default. | [Paper](https://arxiv.org/abs/1505.04597) |
| **MuViT (Transformer)** | Multi-scale feature fusion with Vision Transformer encoder. Supports optional MAE pretraining. | - |
| **Custom ONNX Model** | Importing externally trained models. Advanced users. | - |

### Encoder (UNet)

When UNet is selected, choose a backbone encoder:

**Standard backbones (ImageNet-pretrained):**

| Backbone | Speed | Accuracy | VRAM | Notes |
|----------|-------|----------|------|-------|
| resnet18 | Very fast | Good | Very low | Lightweight option |
| resnet34 | Fast | Good | Low | Best default |
| resnet50 | Medium | Better | Medium | For complex tasks |
| efficientnet-b0 | Very fast | Good | Very low | Lightweight |
| efficientnet-b1 | Fast | Good | Low | Slightly larger than b0 |
| efficientnet-b2 | Fast | Better | Low | Good accuracy/speed balance |
| mobilenet_v2 | Very fast | Good | Very low | Smallest model |

**Histology-pretrained backbones (ResNet-50 based, ~100 MB download on first use):**

| Backbone | Pretraining | Notes |
|----------|-------------|-------|
| resnet50_lunit-swav (Histology) | Lunit SwAV self-supervised on 19M TCGA patches | Best for general tissue classification |
| resnet50_lunit-bt (Histology) | Lunit Barlow Twins self-supervised on 19M TCGA patches | Alternative self-supervised approach |
| resnet50_kather100k (Histology) | Supervised on 100K colorectal tissue patches | Trained on colorectal tissue at 20x |
| resnet50_tcga-brca (Histology) | SimCLR self-supervised on TCGA breast cancer | Trained on breast cancer tissue at 20x |

Histology-pretrained backbones (marked "Histology" in the dropdown) use weights learned from millions of **H&E-stained brightfield patches at approximately 20x magnification (3-channel RGB)**. They typically produce better results for H&E histopathology with less training data.

> **Important:** Histology backbones are designed for H&E brightfield images. For **fluorescence, multiplex IF, or multi-channel (>3 channel) images**, use a standard ImageNet backbone (resnet34 or resnet50) instead. The histology-pretrained first conv layer encodes H&E color responses that do not transfer to fluorescence intensity patterns. See [Backbone Selection](BEST_PRACTICES.md#backbone-selection) for detailed guidance.

**Foundation model encoders (downloaded on-demand from HuggingFace):**

Foundation models are large-scale vision transformers pretrained on massive histopathology datasets. They provide richer tissue representations than ResNet-based encoders and often achieve better results with fewer training examples. Foundation model integration inspired by LazySlide (Zheng et al. 2026, Nature Methods). Only commercially-permissive licenses (Apache 2.0, MIT) are included.

| Encoder | Parameters | License | VRAM (batch=4, 512px) |
|---------|-----------|---------|----------------------|
| hibou-b | 86M | Apache 2.0 | ~6 GB |
| hibou-l | 300M | Apache 2.0 | ~10 GB |
| dinov2-large | 300M | Apache 2.0 | ~10 GB |
| virchow | 632M | Apache 2.0 | ~12 GB |
| h-optimus-0 | 1.1B | Apache 2.0 | ~16 GB |
| midnight | 1.1B | MIT | ~16 GB |

Foundation model weights are **downloaded on-demand** (~100 MB to ~2 GB depending on model size) and cached locally after the first download. They are not bundled with the extension.

**Key considerations for foundation models:**

- Foundation models default to **encoder-frozen training**. Their pretrained features are already very strong, so freezing the encoder and training only the decoder is recommended for most datasets. Unfreeze selectively only with large datasets (>5000 tiles).
- **Gated models** on HuggingFace require authentication. If a model requires a HuggingFace token, the extension will prompt you. Obtain a token at https://huggingface.co/settings/tokens and accept the model's license agreement on its HuggingFace page.
- Like histology-pretrained ResNet-50 backbones, foundation models were trained on H&E RGB images. For fluorescence or multi-channel images, use ImageNet backbones instead.

See [Backbone Selection](BEST_PRACTICES.md#backbone-selection) for detailed guidance on choosing between foundation models, histology ResNet-50, and ImageNet backbones.

### MuViT Configuration

When MuViT (Transformer) is selected, the encoder combo is hidden and model-specific controls appear:

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Model size** | muvit-small, muvit-base, muvit-large | Transformer capacity. Larger models need more data and VRAM. |
| **Patch size** | 8, 16 | ViT patch size. 16 recommended. Smaller = finer detail, more compute. |
| **Level scales** | Text (e.g., "1,4") | Multi-resolution scale factors. |
| **Position encoding** | per_layer, shared, fixed, none | Rotary position encoding mode. per_layer recommended. |

## Step 5: Configure Weight Initialization

The **WEIGHT INITIALIZATION** section controls how model weights are initialized. Choose one of five strategies:

| Strategy | When to use |
|----------|-------------|
| **Train from scratch** | Very large datasets or custom architectures. Rarely needed. |
| **Use pretrained backbone weights** | **Recommended for most cases.** Uses ImageNet or histology-pretrained encoder weights. Shows a layer freeze panel for fine-grained control. |
| **Use MAE pretrained encoder** | MuViT only. Load encoder weights from a self-supervised MAE pretrained .pt file. Click "Browse..." to select. Architecture locks to match the encoder. |
| **Use SSL pretrained encoder** | UNet only. Load encoder weights from a SimCLR/BYOL self-supervised pretrained .pt file. Use when you have domain-specific unlabeled images (different microscope, staining, tissue type) and want the encoder to recognize those visual patterns before supervised training. Run **Utilities > SSL Pretrain Encoder** first. See [Domain Adaptation Guide](DOMAIN_ADAPTATION_GUIDE.md). |
| **Continue training from saved model** | Resume from a previously trained classifier. Click "Select model..." to pick the model. All settings populate from the saved model. |

When **Use pretrained backbone weights** is selected, a layer freeze panel appears:
- Per-layer checkboxes to freeze/unfreeze individual encoder layers
- **Freeze All** / **Unfreeze All** / **Use Recommended** buttons
- Small datasets (<500 tiles): freeze most layers; large datasets (>5000): unfreeze nearly all

## Step 6: Configure Training Parameters

### Core parameters

| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Epochs** | 50 | 50-200 for small datasets, 20-100 for large. Early stopping prevents overfitting. |
| **Batch Size** | 8 | 4-8 for 8GB VRAM with 512px tiles. Reduce if out-of-memory. |
| **Learning Rate** | 0.0001 | Safe default for AdamW. Reduce further if loss oscillates. When using OneCycleLR, an LR finder auto-runs to suggest the optimal max learning rate. |
| **Validation Split** | 20% | 15-25% typical. Uses stratified sampling for balanced splits. |
| **Tile Size** | 512 | Must be divisible by 32. 256 for cell-level, 512 for tissue-level. |
| **Whole image** | Off | Checkbox. Uses entire image as one tile (small images only). Disables tile size, overlap, and context scale -- but downsample stays unlocked so you can adjust resolution to fit within the max tile size. A dynamic warning shows orange (images fit) or red (images will be tiled) based on your actual image dimensions at the current downsample. |
| **Resolution** | 1x | 1x, 2x, 4x, 8x, 16x. Higher = more context, less detail. Locked when continuing from a saved model. "Preview" button shows the image at selected resolution. |
| **Context Scale** | 4x (Recommended) | None, 2x, 4x, 8x, 16x. Adds surrounding context at lower resolution alongside the main tile. Hidden for MuViT (handles multi-scale internally). |
| **Tile Overlap** | 0% | 10-25% generates more patches from limited annotations. |
| **Line Stroke Width** | QuPath's stroke | Width for polyline annotation masks (minimum 1px). Increase for sparse lines. |

### Training Strategy (collapsed by default)

| Parameter | Default | Guidance |
|-----------|---------|----------|
| **LR Scheduler** | One Cycle | Best default. "Reduce on Plateau" is a good alternative when training is noisy. |
| **Loss Function** | Cross Entropy + Dice | Recommended. Dice optimizes IoU directly. See "Loss function options" below for alternatives including Focal, Boundary-softened CE, and Lovasz-Softmax. |
| **Early Stop Metric** | Mean IoU | More reliable than validation loss. |
| **Early Stop Patience** | 15 | Epochs without improvement before stopping. |
| **Focus Class** | (none) | Select a class whose IoU overrides mean IoU for best-model selection. |
| **Min Focus IoU** | 0.0 | Minimum IoU threshold for focus class before early stopping kicks in. |
| **Mixed Precision** | Enabled | FP16/BF16 mixed precision. ~2x speedup on NVIDIA GPUs. |
| **Gradient Accumulation** | 1 | Accumulate over N batches. Set 2-4 to simulate larger batches on limited VRAM. |
| **Progressive Resizing** | Off | Train at half resolution first (40% of epochs), then full resolution. |

### Loss function options

The Loss Function combo exposes eight variants. OHEM composes with
most of them (focal modulation and boundary weighting applied
BEFORE the top-K hard-pixel sort); the two Lovasz variants are
the only exceptions (OHEM silently disabled -- Lovasz is a
sorted-errors Jaccard surrogate, not a per-pixel loss).

| Option | When to pick it |
|---|---|
| Cross Entropy + Dice | Default. Safe starting point for any task. |
| Cross Entropy | Per-pixel only. Use when Dice is undesirable (e.g. rare class + mostly-correct background where Dice washes out). |
| Focal + Dice | Down-weights easy pixels via `(1-p_t)^gamma`. Use when classes have very different difficulty. With OHEM, `focal_gamma` is preserved inside the hard set via `OHEMFocalLoss`. |
| Focal | Focal alone, no Dice. |
| Boundary-softened CE | CE weighted by Euclidean distance to the nearest annotation boundary. Down-weights noisy edge pixels. Use when manual annotations have imprecise boundaries. Parameters: sigma (falloff, default 3px), w_min (floor at exact boundary, default 0.1). |
| Boundary-softened CE + Dice | The recommended pairing for edge-noisy annotations -- boundary CE handles the edges, Dice optimizes region overlap. With OHEM, boundary weight is applied before top-K so OHEM capacity focuses on interior errors. |
| Lovasz-Softmax | Directly optimizes mean IoU. Best after a CE warmup or combined with CE (see next row). No hyperparameters. Per-class weights apply. |
| CE + Lovasz-Softmax | CE provides stable early gradient, Lovasz pushes directly toward IoU. Safer than bare Lovasz from init. |

Per-class weights (from the class multipliers in the class list)
apply to CE, Focal, Boundary-softened CE, and every Lovasz
variant. See also the "Option interaction" runtime watcher that
shows a popup when a known-risky composition is detected (e.g.
tile overlap > 0 combined with no per-image split role).

### Automatic Optimizations

The following optimizations are applied automatically -- no configuration needed:

- **Context padding**: Training tiles are automatically extracted with a border of real surrounding image data, matching the geometry used during inference (where QuPath provides real context via `inputPadding`). The padding amount is computed as `max(64, min(max(overlap, tileSize/4), tileSize * 3/8))` pixels per side. The mask border is filled with 255 (ignore_index) so the loss only computes on the annotated center region. This eliminates train/inference geometry mismatch and tile edge artifacts. Disabled for whole-image mode.
- **AdamW optimizer** with fast.ai-tuned hyperparameters (betas=0.9/0.99, eps=1e-5, weight_decay=0.01). AdamW decouples weight decay from the gradient update, producing better generalization than Adam.
- **Discriminative learning rates**: When using pretrained weights, the encoder automatically receives 1/10th of the base learning rate while the decoder and segmentation head train at the full rate. This prevents catastrophic forgetting of pretrained features.
- **LR Finder**: When using the One Cycle scheduler on a new training run (not resuming from checkpoint), an automatic learning rate range test runs before training to find the optimal max learning rate. The suggested LR is logged and used as the OneCycleLR max_lr.
- **BF16 auto-detection**: On Ampere+ GPUs (RTX 3000 series and newer), training and inference automatically use BF16 mixed precision instead of FP16. BF16 has a larger dynamic range and does not require gradient scaling, improving stability.

## Step 7: Select Channels and Classes

### Channel Configuration

- For RGB brightfield images, channels are auto-configured
- For fluorescence/spectral images, select and order channels manually
- Set per-channel normalization strategy: PERCENTILE_99 (recommended), MIN_MAX, Z_SCORE, or FIXED_RANGE
- Channel order must match at inference time

### Annotation Classes

- Select at least 2 annotation classes
- A pie chart shows per-class annotation area distribution
- Adjust weight multipliers for imbalanced classes (>1.0 boosts rare classes)
- **Rebalance Classes** button auto-sets weights inversely proportional to class area
- **Rebalance by default** checkbox auto-rebalances whenever classes are loaded

### Data Augmentation (collapsed by default)

| Augmentation | Default | Notes |
|-------------|---------|-------|
| Horizontal flip | On | Almost always beneficial |
| Vertical flip | On | Safe for most histopathology |
| Rotation (90 deg) | On | Combines with flips for 8x augmentation |
| Intensity augmentation | Auto-selected | Three modes: **None**, **Brightfield (color jitter)** for H&E, **Fluorescence (per-channel)** for IF. Auto-selected based on image type. |
| Elastic deformation | Off | Effective but ~30% slower. [Albumentations](https://albumentations.ai/docs/) |

## Step 8: Start Training

Click **Start Training**. A progress window shows:
- Patch extraction progress
- Epoch-by-epoch training and validation loss
- Live loss chart with separate colors for train (blue) and validation (red) loss
- Per-class IoU metrics
- Early stopping status

> **Settings are remembered:** When you train a model and later load it to continue training, all training dialog settings (architecture, tile size, learning rate, loss function, gradient accumulation, progressive resize, augmentation, etc.) are restored from the model's saved metadata. This means you can iterate quickly without re-entering parameters.

### Cancelling training

Click **Cancel** at any time. A dialog offers three choices:

| Option | What it saves |
|--------|--------------|
| **Best Epoch** | The model from the epoch with the highest mean IoU |
| **Last Epoch** | The model from the most recently completed epoch |
| **Do Not Save** | Discards all progress |

After choosing, the dialog becomes closeable immediately -- you do not need to wait for background cleanup. The saved model is fully usable for inference or for continuing training later.

### If training is interrupted unexpectedly

If training is interrupted by a crash, power outage, or accidental close, your progress is automatically protected. The extension saves a full checkpoint to disk every time validation metrics improve, so you can recover the best model or resume training from the last best epoch.

See [Troubleshooting: What can I do if training is interrupted?](TROUBLESHOOTING.md#what-can-i-do-if-training-is-interrupted) for detailed recovery instructions.

### What to watch for

- **Training loss should decrease** over epochs
- **Validation loss should follow** training loss down
- **Diverging losses** (val goes up, train goes down) = overfitting
- **Both losses plateau** = model has converged

## Step 9: Review Training Areas (Optional)

When training completes successfully, a **"Review Training Areas..."** button appears in the progress dialog. This runs the trained model over all training tiles and ranks them by loss to help you identify annotation errors, hard cases, and model failures.

> **Important:** Training tiles are cleaned up when you close the progress dialog. Review your training areas *before* closing -- or save the session (see below) to reopen later without re-running evaluation.

### How it works

1. Click **"Review Training Areas..."** in the completed training progress dialog
2. The model evaluates every training tile (train + val splits) and computes per-tile metrics
3. A results dialog opens showing tiles sorted by loss (highest first)

### The Training Area Issues dialog

| Column | Description |
|--------|-------------|
| **Image** | Source image name (for multi-image training) |
| **Split** | Whether the tile was in the train or val split |
| **Loss** | Cross-entropy loss -- higher = model disagrees more with annotation |
| **Disagree%** | Percentage of pixels where prediction differs from annotation |
| **mIoU** | Mean Intersection-over-Union across classes present in the tile |
| **Classes** | Which classes are present in the tile |

### Filtering and navigation

- **Filter by split**: Use the dropdown to show only train or val tiles
- **Loss threshold**: Use the slider to show only tiles above a minimum loss
- **Click a row** to navigate the QuPath viewer to that tile's location. The selected tile's loss heatmap or disagreement map appears as an overlay directly in the viewer, aligned to the tile bounds. Clicking does **not** create any annotations in your project.
- For multi-image projects, clicking automatically switches to the correct image.

### Viewer overlay

While a row is selected, the tile's diagnostic image is overlaid in the QuPath viewer:

- **Loss Heatmap** (default): per-pixel loss intensity, colored blue (low) -> yellow -> red (high)
- **Disagreement**: pixels where the prediction differs from the ground truth, colored by the predicted class
- Switch modes with the **Overlay** dropdown in the preview panel
- The overlay respects QuPath's **View > Overlay opacity** slider. If opacity drops below 10%, or if **View > Show pixel classification** is off, a yellow warning banner appears at the top of the dialog explaining why the overlay may be invisible.
- Closing the dialog removes the overlay. If the production prediction overlay was running when you opened the dialog, it is restored.

### Saving and reloading sessions

The **Save Session...** and **Load Session...** buttons persist a Training Area Issues session so you can revisit it later without re-running evaluation.

- **Save Session...** -- confirms the classifier identity, tile count, estimated disk usage, and on-disk location before writing. Sessions are stored under `<classifier_dir>/training_issues_sessions/<yyyyMMdd_HHmmss>/` alongside the model files, so they travel with the classifier.
- **Load Session...** -- lists saved sessions for the current classifier and reopens the one you pick. If the classifier has been retrained since the session was saved (different model file size or modification time), a warning confirms whether to open the (now-stale) results.
- Sessions duplicate the PNG assets into the session folder, so they survive even after the transient `disagreement/` folder is cleared.

You can also reopen saved sessions from outside the training workflow via **Extensions > DL Pixel Classifier > Utilities > Load Saved Training Area Issues...** -- useful after restarting QuPath or opening a different project that shares the same classifier.

### What to look for

| Pattern | Likely cause | Action |
|---------|-------------|--------|
| Very high loss on a few tiles | Annotation error (wrong class) | Fix the annotation and retrain |
| High loss cluster in one region | Inconsistent annotation criteria | Re-annotate the region consistently |
| High loss on val but not train | Model memorizing, not generalizing | Add more diverse annotations |
| High disagreement on boundaries | Normal -- boundaries are hardest | Consider annotating boundaries more carefully |
| Entire class has high loss | Class may be poorly defined | Check if the class has consistent visual features |

See [BEST_PRACTICES.md](BEST_PRACTICES.md#interpreting-tile-evaluation-results) for detailed guidance on interpreting results and improving annotations.

## Step 10: Continue Training (Optional)

When training completes successfully, a **"Continue Training..."** button appears alongside "Review Training Areas..." in the progress dialog. This allows you to extend training with adjusted parameters:

1. Click **"Continue Training..."**
2. A resume dialog lets you adjust epochs, learning rate, and batch size
3. Training continues from the best checkpoint with the new settings

This is useful when training stopped early but the model could benefit from more epochs or a different learning rate.

## MAE Pretraining (MuViT Encoder)

Before training a MuViT-based classifier, you can optionally pretrain the encoder using Masked Autoencoder (MAE) self-supervised learning on unlabeled image tiles. This is a separate workflow accessible from the Utilities menu.

### When to use MAE pretraining

- You plan to train a **MuViT** classifier and have a large collection of unlabeled images from your domain
- Your target tissue or staining is not well-represented by standard pretrained weights
- You want the encoder to learn domain-specific features before supervised fine-tuning

### How to pretrain

1. **Prepare image tiles**: Export unlabeled tiles (PNG, TIFF, JPEG, or BMP) from your images into a directory. These do not need annotations -- any representative image patches will work.
2. Go to **Extensions > DL Pixel Classifier > Utilities > MAE Pretrain Encoder...**
3. Configure the pretraining parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Model Configuration** | muvit-small | Model size (small/base/large) -- should match what you plan to use for training |
| **Patch Size** | 16 | Vision transformer patch size (8 or 16) |
| **Level Scales** | 1,4 | Multi-resolution scale factors |
| **Epochs** | 100 | Pretraining epochs (auto-suggested based on dataset size) |
| **Mask Ratio** | 0.75 | Fraction of patches masked during pretraining (0.5-0.9) |
| **Batch Size** | 8 | Tiles per training step |
| **Learning Rate** | 0.00015 | AdamW learning rate |
| **Warmup Epochs** | 5 | Linear warmup epochs |

4. Select the directory containing your image tiles (the dialog scans and reports the count)
5. Choose an output directory (defaults to `{project}/mae_pretrained/`)
6. Click **Start Pretraining** -- a progress monitor shows reconstruction loss over epochs

### Using the pretrained encoder

After pretraining completes, load the encoder weights when training a MuViT classifier:

1. Open **Extensions > DL Pixel Classifier > Train Classifier...**
2. Select the **MuViT (Transformer)** architecture
3. In **WEIGHT INITIALIZATION**, select **"Use MAE pretrained encoder"**
4. Click **"Browse..."** and select the pretrained .pt file
5. Architecture settings will auto-lock to match the encoder's metadata

#### What happens under the hood

Selecting "Use MAE pretrained encoder" (or "Continue training from saved
model") does three things automatically:

1. Loads the saved weights into the model.
2. Flags the run as `use_pretrained=true` so the optimizer setup splits
   parameters into **encoder** and **head** groups.
3. Applies a **discriminative learning rate**: the encoder group trains at
   `learning_rate x discriminative_lr_ratio` (default 0.1x) while the head
   trains at the full learning rate. This preserves the MAE features
   instead of letting the aggressive supervised LR dismantle them.

If you see an epoch-by-epoch loss that decreases for ~20-30 epochs and then
*suddenly* spikes back up with `acc` near 1% and per-class IoUs collapsing
to ~0, that's the classic "discriminative LRs weren't applied, MAE features
got wrecked" signature. Verify the training-config log line on the next run
includes `discriminative LRs (ratio=0.1, ...)` rather than just
`AdamW (lr=0.0001, ...)`.

### Dataset size guidance

The dialog auto-suggests epoch counts based on your dataset:

| Dataset size | Suggested epochs | Notes |
|-------------|-----------------|-------|
| < 50 tiles | 500 | Very small -- consider gathering more data |
| 50-200 tiles | 300 | Small dataset, needs many passes |
| 200-1000 tiles | 100 | Good balance |
| > 1000 tiles | 50 | Large dataset, fewer passes needed |

See [PARAMETERS.md](PARAMETERS.md#mae-pretraining-parameters) for detailed parameter reference and [BEST_PRACTICES.md](BEST_PRACTICES.md#mae-pretraining) for tuning guidance.

## Step 11: Verify the Result

When training completes, the classifier is saved to your QuPath project under `classifiers/`. View it via **Extensions > DL Pixel Classifier > Manage Models...**

### Quick verification

1. Apply the classifier to a test annotation (see [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md))
2. Check results visually using the RENDERED_OVERLAY output type (recommended) or OVERLAY for live preview
3. If results are poor, see [BEST_PRACTICES.md](BEST_PRACTICES.md) for improvement strategies

## Copy as Script

Click the **"Copy as Script"** button (left side of the button bar) in the training dialog to generate a Groovy script matching your current settings. Paste into QuPath's Script Editor for reproducible and batch training workflows. See [SCRIPTING.md](SCRIPTING.md) for details.
