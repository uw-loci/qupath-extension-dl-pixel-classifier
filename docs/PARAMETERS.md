# Parameter Reference

Complete reference for every parameter in the training and inference dialogs. This mirrors the tooltip content in the extension UI.

## Training Parameters

The training dialog is accessed via **Extensions > DL Pixel Classifier > Train Classifier...**

Dialog sections appear in this order:

1. TRAINING DATA SOURCE
2. CLASSIFIER INFO
3. MODEL ARCHITECTURE
4. WEIGHT INITIALIZATION
5. TRAINING PARAMETERS
6. TRAINING STRATEGY
7. CHANNEL CONFIGURATION
8. ANNOTATION CLASSES
9. DATA AUGMENTATION

### Training Data Source

| Parameter | Type | Description |
|-----------|------|-------------|
| **Image selection list** | Checkbox list | Check project images to include in training. Only images with classified annotations are shown. |
| **Load Classes from Selected Images** | Button | Reads annotations from selected images, populates the class list, and initializes channel configuration. If a model was loaded, auto-matches classes. |

### Classifier Info

| Parameter | Type | Description |
|-----------|------|-------------|
| **Classifier Name** | Text | Unique identifier for the classifier. Used as filename. Only letters, numbers, underscore, and hyphen allowed. |
| **Description** | Text | Optional free-text description stored in classifier metadata. |

### Model Architecture

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Architecture** | UNet, MuViT (Transformer), Custom ONNX Model | Segmentation architecture. UNet is the best general-purpose choice. MuViT uses multi-resolution Vision Transformer feature fusion for multi-scale context. Custom ONNX allows importing externally trained models. See [UNet paper](https://arxiv.org/abs/1505.04597). |
| **Encoder** | resnet18, resnet34, resnet50, efficientnet-b0/b1/b2, mobilenet_v2, plus 4 histology-pretrained variants (resnet50_lunit-swav, resnet50_lunit-bt, resnet50_kather100k, resnet50_tcga-brca) | Pretrained encoder network (shown for UNet only). Histology backbones use H&E tissue-pretrained weights (20x, 3-channel RGB) instead of ImageNet -- best for H&E brightfield. For fluorescence or multi-channel images, use ImageNet backbones. See [Backbone Selection](BEST_PRACTICES.md#backbone-selection). |

When **MuViT (Transformer)** is selected, the Encoder combo is hidden and a handler-specific UI appears instead:

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Model size** | muvit-small, muvit-base, muvit-large | Transformer model capacity. Larger models learn richer features but need more data, training time, and VRAM. |
| **Patch size** | 8, 16 | Vision transformer patch size. Smaller patches capture finer detail but increase compute quadratically. 16 is recommended for most cases. |
| **Level scales** | Text (e.g., "1,4") | Comma-separated multi-resolution scale factors for multi-scale feature fusion. |
| **Position encoding** | per_layer, shared, fixed, none | Rotary position encoding mode. per_layer is recommended. |

### Weight Initialization

Controls how model weights are initialized before training. This section has four mutually exclusive radio buttons:

| Strategy | Description |
|----------|-------------|
| **Train from scratch** | Random initialization. Only recommended for very large datasets. |
| **Use pretrained backbone weights** | Initialize the encoder with pretrained weights (ImageNet or histology). Almost always recommended. Shows a layer freeze panel for fine-grained control over which layers to train vs. freeze. |
| **Use MAE pretrained encoder** | Load encoder weights from a self-supervised MAE pretrained model (.pt file). Click "Browse..." to select the file. Architecture settings auto-lock to match the encoder metadata. MuViT only. |
| **Continue training from saved model** | Resume training from a previously trained classifier. Click "Select model..." to pick the model. All dialog fields populate from the saved model's metadata and training settings. |

When **Use pretrained backbone weights** is selected, a **Layer Freeze Panel** appears with:
- Per-layer checkboxes to freeze/unfreeze individual encoder layers
- **Freeze All** / **Unfreeze All** / **Use Recommended** buttons
- Freezing early layers preserves general features; unfreezing later layers adapts to your data

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Epochs** | 50 | 1-1000 | Complete passes through training data. Early stopping prevents overfitting. 50-200 for small datasets, 20-100 for large. |
| **Batch Size** | 8 | 1-128 | Tiles per training step. Larger = more stable gradients, more VRAM. 4-8 for 8GB VRAM with 512px tiles. |
| **Learning Rate** | 0.001 | 0.00001-1.0 | Step size for gradient descent. 1e-3 for AdamW default, 1e-4 if oscillating, 1e-5 for full fine-tuning. When using OneCycleLR, an automatic LR finder runs before training to suggest an optimal max learning rate. |
| **Validation Split** | 20% | 5-50% | Percentage held out for validation. Uses stratified sampling to ensure all classes appear in both train and validation sets. 15-25% typical. |
| **Tile Size** | 512 | 64-1024 | Patch size in pixels. Must be divisible by 32 (encoder downsampling requirement). 256 for cell-level, 512 for tissue-level. |
| **Whole image** | Off | Checkbox | Use the entire image as a single training tile (for small images only). Disables tile size, overlap, and context scale controls. For ViT models (MuViT), tile size is capped at the architecture's max supported size (512px) since self-attention is O(n^2) in patch count. If the image exceeds this limit at the selected downsample, it will be tiled automatically. |
| **Resolution** | 1x | 1x, 2x, 4x, 8x, 16x | Image downsample level. Higher = more context per tile, less detail. A "Preview" button opens a mini-viewer showing the image at the selected resolution. |
| **Context Scale** | 4x (Recommended) | None, 2x, 4x, 8x, 16x | Provides additional surrounding context at a lower resolution alongside the main tile. Helps the model understand broader tissue architecture. Hidden for MuViT (which handles multi-scale internally via level_scales). |
| **Tile Overlap** | 0% | 0-50% | Overlap between training tiles. 10-25% generates more patches from limited annotations. |
| **Line Stroke Width** | QuPath's stroke | 1-50 | Pixel width for polyline annotation masks. Default is QuPath's annotation stroke thickness (typically 5). Increase for sparse lines. |

### Training Strategy

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **LR Scheduler** | One Cycle | One Cycle, Cosine Annealing, Reduce on Plateau, Step Decay, None | Learning rate schedule. One Cycle is recommended for most cases. Reduce on Plateau automatically lowers the LR when the monitored metric stops improving (factor=0.5, patience=10). See [PyTorch schedulers](https://pytorch.org/docs/stable/optim.html). |
| **Loss Function** | Cross Entropy + Dice | Cross Entropy + Dice, Cross Entropy | Cross Entropy + Dice is recommended. Dice directly optimizes IoU. See [smp losses](https://smp.readthedocs.io/en/latest/losses.html). |
| **Early Stop Metric** | Mean IoU | Mean IoU, Validation Loss | Mean IoU is more reliable than loss for stopping. |
| **Early Stop Patience** | 15 | 3-50 | Epochs without improvement before stopping. 10-15 default, 20-30 for noisy curves. |
| **Focus Class** | (none) | (none), or any class | Select a class whose per-class IoU overrides the early stop metric for best-model selection. Useful when one class is more important than overall mean IoU. |
| **Min Focus IoU** | 0.0 | 0.0-1.0 | Minimum IoU threshold for the focus class that must be reached before early stopping kicks in. Only shown when a focus class is selected. |
| **Mixed Precision** | Enabled | On/Off | Automatic mixed precision (FP16/FP32) on CUDA GPUs. Typically provides ~2x speedup with no accuracy loss. Only active when training on NVIDIA GPUs; ignored on CPU/MPS. See [PyTorch AMP](https://pytorch.org/docs/stable/amp.html). |
| **Gradient Accumulation** | 1 | 1-8 | Number of batches to accumulate before updating weights. Effectively multiplies batch size without increasing VRAM. Set to 2-4 to simulate larger batches on limited GPU memory. |
| **Progressive Resizing** | Off | On/Off | Train at half resolution for the first 40% of epochs, then switch to full resolution. Speeds up early training and acts as implicit regularization. Inspired by fast.ai. |

### Channel Configuration

| Parameter | Description |
|-----------|-------------|
| **Available Channels** | Image channels available for selection. Multi-select with Ctrl+click. |
| **Selected Channels** | Channels used as model input. Order must match at inference time. |
| **Normalization** | Per-channel intensity normalization: PERCENTILE_99 (recommended), MIN_MAX, Z_SCORE, FIXED_RANGE. |

### Annotation Classes

| Parameter | Description |
|-----------|-------------|
| **Class list** | Annotation classes found in the image. At least 2 must be selected. A pie chart shows the per-class annotation area distribution. |
| **Weight multiplier** | Per-class weight multiplier. >1.0 boosts underrepresented classes. |
| **Rebalance Classes** | Button that auto-sets weight multipliers inversely proportional to class area, compensating for class imbalance. |
| **Rebalance by default** | Checkbox (default: on). When enabled, weight multipliers are automatically rebalanced whenever classes are loaded. |

### Data Augmentation

| Augmentation | Default | Description |
|-------------|---------|-------------|
| **Horizontal flip** | On | Mirror tiles left-right. Almost always beneficial. |
| **Vertical flip** | On | Mirror tiles top-bottom. Safe for most histopathology. |
| **Rotation (90 deg)** | On | Rotate by 0/90/180/270. Combines with flips for 8x augmentation. |
| **Intensity augmentation** | Auto-selected | Three modes: **None** (no intensity transforms), **Brightfield (color jitter)** (correlated RGB brightness/contrast/saturation -- good for H&E), **Fluorescence (per-channel)** (independent per-channel intensity scaling -- good for IF). Auto-selected based on image type when classes are loaded. |
| **Elastic deformation** | Off | Smooth spatial deformations. Effective but ~30% slower. See [Albumentations](https://albumentations.ai/docs/). |

---

## MAE Pretraining Parameters

These parameters are available in the **MAE Pretrain Encoder** dialog (**Extensions > DL Pixel Classifier > Utilities > MAE Pretrain Encoder...**). MAE pretraining is a standalone workflow for self-supervised pretraining of MuViT encoder weights on unlabeled image tiles.

### Model Architecture

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Model Configuration** | muvit-small, muvit-base, muvit-large | MuViT model size. Must match the model you plan to use for supervised training. Larger models learn richer features but need more data and training time. |
| **Patch Size** | 8, 16 | Vision transformer patch size. Smaller patches capture finer detail but increase compute. 16 is recommended for most cases. |
| **Level Scales** | Text (e.g., "1,4") | Comma-separated multi-resolution scale factors for multi-scale feature fusion. |

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Epochs** | 100 | 10-2000 | Number of pretraining epochs. Auto-suggested based on dataset size: <50 tiles -> 500, <200 -> 300, <1000 -> 100, 1000+ -> 50. |
| **Mask Ratio** | 0.75 | 0.5-0.9 | Fraction of image patches masked during pretraining. Higher = harder reconstruction task. 0.75 is the standard MAE default. |
| **Batch Size** | 8 | 1-64 | Tiles per training step. Reduce if out-of-memory. |
| **Learning Rate** | 0.00015 | 0.00001-0.01 | AdamW learning rate. The default (1.5e-4) follows the original MAE paper recommendation. Displayed with 5 decimal places. |
| **Warmup Epochs** | 5 | 0-50 | Number of epochs for linear learning rate warmup from 0 to the target learning rate. |

### Data

| Parameter | Description |
|-----------|-------------|
| **Data Directory** | Directory containing unlabeled image tiles (.png, .tif, .tiff, .jpg, .jpeg, .bmp, .raw). The dialog scans the directory and reports the number of images found with an auto-suggested epoch count. |
| **Dataset Info** | Auto-populated label showing image count and epoch recommendation after selecting a directory. |

### Output

| Parameter | Description |
|-----------|-------------|
| **Output Directory** | Where to save the pretrained encoder weights. Defaults to `{project}/mae_pretrained/`. The directory is created automatically if it does not exist. |

---

## Inference Parameters

The inference dialog is accessed via **Extensions > DL Pixel Classifier > Apply Classifier...**

### Classifier Selection

| Parameter | Description |
|-----------|-------------|
| **Classifier table** | Select a trained classifier. Shows name, type (architecture), channels (with context scale info), class count, and training date. |
| **Info panel** | Below the table, shows architecture + backbone, input channels + context scale + tile dimensions, downsample level, and class names for the selected classifier. |

### Output Options

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **Output Type** | RENDERED_OVERLAY | MEASUREMENTS, OBJECTS, OVERLAY, RENDERED_OVERLAY | How results are represented. **RENDERED_OVERLAY** (recommended) runs batch inference with tile blending, producing a seamless overlay that accurately represents what OBJECTS output would look like. Best for validating classifier quality. **OVERLAY** enables live on-demand classification as you pan and zoom. **OBJECTS** creates detection/annotation objects. **MEASUREMENTS** adds per-class probabilities to existing annotations. |
| **Object Type** | DETECTION | DETECTION, ANNOTATION | QuPath object type (OBJECTS output only). |
| **Min Object Size** | 10 um^2 | 0-10000 | Discard objects below this area. |
| **Hole Filling** | 5 um^2 | 0-1000 | Fill interior holes below this area. |
| **Boundary Smoothing** | 1.0 um | 0-10 | Simplification tolerance in microns. |

### Channel Mapping

| Parameter | Description |
|-----------|-------------|
| **Channel mapping grid** | Shows expected channels from the classifier and how they map to the current image's channels. Each row shows Expected -> Mapped To -> Status. Status is [OK] (exact match), [?] (fuzzy/substring match), or [X] (unmapped). Channels can be manually remapped using dropdown overrides. |
| **Summary label** | Shows the number of matched/unmatched channels. For brightfield images, the channel section auto-collapses when all channels match. |

### Processing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Tile Size** | Auto | Auto-set from classifier. Must be divisible by 32. Range: 64-8192. |
| **Tile Overlap (%)** | 12.5% | 0-50%. Higher = better blending, slower processing. |
| **Blend Mode** | CENTER_CROP | **CENTER_CROP** (default, recommended): Keep only center predictions, discard overlap margins. Zero boundary artifacts. Forced for OVERLAY output type. **LINEAR**: Weighted average favoring tile centers. Available for batch inference (RENDERED_OVERLAY, OBJECTS). **GAUSSIAN**: Cosine-bell blending for smoother transitions. **NONE**: No blending; last tile wins. Fastest but may show visible tile seams. |
| **Use GPU** | On | 10-50x faster than CPU. Falls back automatically. |
| **Test-Time Augmentation (TTA)** | Off | Applies D4 transforms (flips + 90-degree rotations) during inference and averages the predictions. Typically improves segmentation quality by 1-3% at the cost of ~8x slower inference. Best for final production runs where quality matters most. |

### Normalization

| Parameter | Description |
|-----------|-------------|
| **Image-level normalization** | Automatically enabled. Computes per-channel normalization statistics once across the entire image (sampling ~16 tiles in a 4x4 grid), then applies the same statistics to every tile. Eliminates tile boundary artifacts caused by per-tile normalization. |
| **Training dataset stats** | When available in the model metadata (models trained after this update), normalization uses statistics from the training dataset for the best consistency. Falls back to image-level sampling for older models. |

### Application Scope

| Parameter | Description |
|-----------|-------------|
| **Apply to whole image** | Classify entire image without annotations. |
| **Apply to all annotations** | Classify within all annotations (default). |
| **Apply to selected annotations only** | Classify only within selected annotations. |
| **Create backup** | Save existing measurements before overwriting (default: off). |

> **Note:** All inference dialog settings (output type, blend mode, smoothing, scope, backup) are remembered across sessions.
