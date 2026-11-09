# Parameter Reference

Reference for every parameter in the training and inference dialogs, with defaults and ranges. It follows the tooltips in the extension, which carry more detail.

Terms such as epoch, batch, loss, IoU and learning rate are explained in [Words used in these guides](TRAINING_GUIDE.md#words-used-in-these-guides).

## Training Parameters

The training dialog is opened from **Extensions > DL Pixel Classifier > Train DL Pixel Classifier...**

The dialog opens in its basic view. **Show All Settings**, in the title row, shows every control. Rows marked *(All Settings)* below are hidden in the basic view.

Dialog sections appear in this order:

1. Training Data Source
2. Model Architecture
3. Weight Initialization
4. Tiles & Resolution
5. Duration & Stopping
6. Batch Size & Memory
7. Learning Rate & Optimizer (All Settings only)
8. Loss Function (All Settings only)
9. Performance (All Settings only)
10. Channel Configuration
11. Annotation Classes
12. Data Augmentation (All Settings only)
13. Name Your Classifier

### Training Data Source

| Parameter | Type | Description |
|-----------|------|-------------|
| **Image selection list** | Checkbox list | Check project images to include in training. Only images with classified annotations are shown. |
| **Load Classes from Selected Images** | Button | Reads annotations from selected images, populates the class list, and initializes channel configuration. If a model was loaded, auto-matches classes. |

### Model Architecture

> **Basic view** hides the Architecture and Encoder controls and instead offers a single **Model size** selector: **Fast**, **Small**, **Medium**, **Large**. Small/Medium/Large are UNet with pretrained ResNet18/34/50. **Fast** is a tiny network for simple, visually-obvious classes (background/tissue/artifact): the dialog auto-picks **Fast Pretrained** (pretrained mobile encoder) for 3-channel RGB images, or **Tiny UNet** (from scratch) for multi-channel data. If the image has more than 7 channels, Fast falls back to Small UNet. Click **Show All Settings** for the full Architecture/Encoder controls below.

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Architecture** | UNet, Tiny UNet, Fast Pretrained, MuViT (Transformer) | Segmentation architecture. UNet is the best general-purpose choice. **Tiny UNet** is a lightweight depthwise-separable UNet (no pretrained weights) for simple 2-5 class tasks and non-RGB data, see [Tiny UNet](TINY_MODEL.md). **Fast Pretrained** is a compact UNet with mobile ImageNet-pretrained encoders for small RGB H&E datasets, see [Fast Pretrained](FAST_PRETRAINED.md). MuViT uses multi-resolution Vision Transformer feature fusion for multi-scale context. The Custom ONNX Model handler is not offered here: an imported ONNX model can only be applied, not trained. See [UNet paper](https://arxiv.org/abs/1505.04597). |
| **Encoder** | resnet18, resnet34, resnet50, efficientnet-b0/b1/b2, mobilenet_v2, 4 histology-pretrained variants (resnet50_lunit-swav, resnet50_lunit-bt, resnet50_kather100k, resnet50_tcga-brca), and 6 foundation model encoders (see below) | Pretrained encoder network (shown for UNet only). Histology backbones use H&E tissue-pretrained weights (20x, 3-channel RGB) instead of ImageNet, best for H&E brightfield. Foundation model encoders provide large-scale pretrained representations for tissue analysis. For fluorescence or multi-channel images, use ImageNet backbones. See [Backbone Selection](BEST_PRACTICES.md#backbone-selection). |

**Foundation model encoders** (downloaded on-demand from HuggingFace, not bundled). Foundation model integration inspired by LazySlide (Zheng et al. 2026, Nature Methods). Only commercially-permissive licenses are included:

| Encoder | Parameters | License | Notes |
|---------|-----------|---------|-------|
| h-optimus-0 | 1.1B | Apache 2.0 | Largest histopathology foundation model. Requires 16+ GB VRAM. |
| virchow | 632M | Apache 2.0 | Strong general-purpose histopathology encoder. |
| hibou-l | 300M | Apache 2.0 | Good capacity/efficiency balance for histopathology. |
| hibou-b | 86M | Apache 2.0 | Lightweight foundation model. Suitable for 8 GB VRAM. |
| midnight | 1.1B | MIT | Large histopathology model with permissive license. |
| dinov2-large | 300M | Apache 2.0 | General-purpose (not histology-specific). Good for diverse image types. |

Gated models on HuggingFace need a HuggingFace access token, set as the `HF_TOKEN` environment variable before QuPath starts. The help next to the Encoder list in the Train dialog gives the steps for Windows, macOS and Linux.

When **MuViT (Transformer)** is selected, the Encoder combo is hidden and a handler-specific UI appears instead:

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Model size** | muvit-small, muvit-base, muvit-large | Transformer model capacity. Larger models learn richer features but need more data, training time, and VRAM. Locked when continuing training from a saved MuViT model. |
| **Patch size** | 8, 16 | Vision transformer patch size. Smaller patches capture finer detail but increase compute quadratically. 16 is recommended for most cases. Locked when continuing training. |
| **Level scales** | Text (e.g., "1,4") | Comma-separated multi-resolution scale factors for multi-scale feature fusion. Locked when continuing training. |
| **Position encoding** | per_layer, shared, fixed, none | Rotary position encoding mode. per_layer is recommended. Locked when continuing training. |

### Weight Initialization

Controls how model weights are initialized before training. This section has five mutually exclusive radio buttons:

| Strategy | Description |
|----------|-------------|
| **Train from scratch** | Random initialization. Only recommended for very large datasets. |
| **Continue from provided weights** | Start from the weights that come with the selected encoder (ImageNet, histology or a foundation model) and train them further. Almost always recommended. Shows a layer freeze panel for fine-grained control over which layers to train vs. freeze. |
| **Use MAE pretrained encoder** | Load encoder weights from a self-supervised MAE pretrained model (.pt file). Click "Browse..." to select the file. Architecture settings auto-lock to match the encoder metadata. MuViT only. |
| **Use SSL pretrained encoder** | Load encoder weights from a self-supervised SSL pretrained model (.pt file). Click "Browse..." to select the file. For CNN encoders (ResNet, EfficientNet, MobileNet) produced by the SSL Pretrain Encoder workflow. UNet only. The **SSL Pretrain Encoder...** menu item that makes these files is hidden until you turn on **Edit > Preferences > DL Pixel Classifier > Show Developer Pretraining Options** and restart QuPath. See [Domain Adaptation Guide](DOMAIN_ADAPTATION_GUIDE.md). |
| **Continue training one of your saved models** | Resume training from a previously trained classifier. Click "Select model..." to pick the model. All dialog fields populate from the saved model's metadata and training settings. Architecture, backbone, tile size, downsample, context scale, and handler-specific parameters (MuViT model size, patch size, level scales) are locked to match the saved model weights. |

When **Continue from provided weights** is selected, a **Layer Freeze Panel** appears with:
- Per-layer checkboxes to freeze/unfreeze individual encoder layers
- **Keep All Pretrained** / **Retrain Everything** / **Use Recommended** buttons, plus a **Retraining:** preset list with **Apply**
- Freezing early layers preserves general features; unfreezing later layers adapts to your data

### Tiles & Resolution

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Tile Size** | 512 | 64-1024 | Patch size in pixels, at the chosen resolution. Must be divisible by 32 (encoder downsampling requirement). 256 for cell-level, 512 for tissue-level. |
| **Whole image** *(All Settings)* | Off | Checkbox | Use the entire image as a single training tile (for small images only). Disables tile size, overlap, and context scale controls. Downsample remains unlocked so you can adjust resolution to fit within the architecture's max tile size. A dynamic warning appears: orange when images fit, red bold when images will be tiled at the current downsample (checks actual selected image dimensions). For ViT models (MuViT), tile size is capped at 512px since self-attention is O(n^2) in patch count. |
| **Resolution** | 1x (Full resolution) | 1x, 2x, 4x, 8x, 16x | Downsample from the image's full resolution. The model sees pixels of (image pixel size x downsample): at 0.25 um/px, 4x means 1 um/px, so a 512-px tile covers 512 um. Higher = more context per tile, less detail. Applying the classifier later reads the image at the same downsample. Locked when continuing training from a saved model. A **Preview** button opens a mini-viewer showing the image at the selected resolution. |
| **Surrounding context** | None (single scale) | None (single scale), 2x context, 4x context (Recommended), 8x context, 16x context | Provides additional surrounding context at a lower resolution alongside the main tile. Helps the model understand broader tissue architecture. Adds a little GPU memory, and doubles the RAM the in-memory patch cache needs. Hidden for MuViT (which handles multi-scale internally via level_scales). For how Resolution + Surrounding context + context padding shape the actual tile geometry across training, the per-tile preview, and inference, see [Context and Downsample Design](CONTEXT_AND_DOWNSAMPLE_DESIGN.md). |
| **Tile Overlap (%)** *(All Settings)* | 13% | 0-50% | Overlap between training tiles. 10-25% generates more patches from limited annotations. The starting value comes from the tile-overlap preference (12.5%) that the Train dialog, the Apply dialog and the overlay share, so changing it here also changes the overlay's overlap. An overlap above 0 with no image set to Train or Val raises a warning, because overlapping tiles can then land on both sides of the train/validation split. |
| **Line Stroke Width** | QuPath's stroke | 1-50 | Pixel width for polyline annotation masks. Default is QuPath's annotation stroke thickness (typically 5). Increase for sparse lines. In the basic view it appears only when the selected images have line annotations. |
| **Min Annotation Coverage (%)** *(All Settings)* | 5% | 0-25% | Filters "corner sliver" tiles where only a tiny fraction of any source annotation lies inside the tile. A tile is dropped only when this **and** Min Tile Label Fraction both fail (hybrid OR rule), so tiles deep inside huge polygons are still kept. Set to 0 to disable this side of the rule; raise it if the disagreement heatmap still shows sliver tiles after training. |
| **Min Tile Label Fraction (%)** *(All Settings)* | 25% | 0-100% | OR escape for the sliver filter: tiles whose labeled pixels cover at least this fraction of the tile are always kept, even when their per-annotation coverage is small. Prevents over-rejecting tiles that sit fully inside large annotations. Set to 0 to disable this side of the rule and rely purely on Min Annotation Coverage (stricter). |

### Duration & Stopping

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Epochs** | 50 | 1-1000 | Complete passes through the training data. Early stopping ends the run sooner when the validation score stops improving. 50-200 for small datasets, 20-100 for large. |
| **Validation Split (%)** *(All Settings)* | 20% | 5-50% | Percentage held out for validation. Uses stratified sampling to ensure all classes appear in both train and validation sets. 15-25% typical. |
| **Enable early stopping** *(All Settings)* | On | On/Off | When off, training runs for the full epoch count. The best model is still saved either way. |
| **Early Stop Metric** *(All Settings)* | Mean IoU | Mean IoU, Validation Loss | Mean IoU is more reliable than loss for stopping. |
| **Early Stop Patience** *(All Settings)* | 15 | 3-50 | Epochs without improvement before stopping. 10-15 default, 20-30 for noisy curves. |
| **Stop on marginal improvement** (`marginalStopEnabled`) *(All Settings)* | Off | On/Off | Ends a run that is still improving, but only barely. Early Stop Patience asks whether the model has stopped setting records; a converged run keeps setting them in the fourth decimal, which resets patience indefinitely. This asks instead whether the **best** score has gained more than **Min Improvement** over the last **Over Epochs** epochs. Off by default. Watches the Early Stop Metric, and works whether or not early stopping itself is enabled. The best model is still saved. |
| **Min Improvement** (`marginalStopMinImprovement`) | 0.01 | 0.0001-0.5 | How much the Early Stop Metric's best score must gain across the window for training to continue. Units are the metric's own: mean IoU when the Early Stop Metric is Mean IoU, loss when it is Validation Loss. 0.005 keeps going for smaller gains; 0.001 nearly always reaches the epoch limit. Only shown when the exit is enabled. |
| **Over Epochs** (`marginalStopWindow`) | 20 | 2-100 | How many epochs the improvement is measured across. No stopping decision is made until this many epochs have elapsed. 10 stops sooner and reacts more to noise; 30-50 suits metrics that move in long slow steps. Only shown when the exit is enabled. |
| **Focus Class** *(All Settings)* | None (use Mean IoU) | None, or any class | Select a class whose per-class IoU overrides the early stop metric for best-model selection. Useful when one class is more important than overall mean IoU. |
| **Min Focus IoU** | 0.50 | 0.0-1.0 | Minimum IoU threshold for the focus class that must be reached before early stopping kicks in. Only shown when a focus class is selected. A saved value of 0 comes back as 0.50 the next time the dialog opens. |
| **Random Seed** *(All Settings)* | 0 | 0-999999 | 0 gives a new random start every run, and no seed is saved with the model. Any positive value seeds Python, NumPy and PyTorch and switches cuDNN to deterministic algorithms (the tooltip estimates 10-20% slower), so the same data, settings and software give the same run on the same machine. Set one for any run you may need to repeat or report. The train/validation split does not depend on the seed. |

### Batch Size & Memory

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Batch Size** | 8 | 1-128 | Tiles per training step. Larger = more stable gradients, more VRAM, and fewer weight updates per epoch. 4-8 for 8GB VRAM with 512px tiles. |
| **Gradient Accumulation** *(All Settings)* | 1 | 1-8 | Number of batches to accumulate before updating weights. Effectively multiplies batch size without increasing VRAM. Set to 2-4 to simulate larger batches on limited GPU memory. |
| **Enable mixed precision (AMP)** *(All Settings)* | On | On/Off | Does part of the arithmetic in 16-bit numbers on NVIDIA GPUs: BF16 when the GPU supports it, otherwise FP16. Usually faster and uses less GPU memory. Ignored on CPU and Apple Silicon. See [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html). |

### Learning Rate & Optimizer

This section appears only after **Show All Settings**.

| Parameter | Default | Range / Options | Description |
|-----------|---------|-------|-------------|
| **Learning Rate** | 0.0001 | 0.00001-1.0 | How large each weight update is. 1e-4 (0.0001) is the recommended default, stable for both fresh training and continue-training. Raise it to 1e-3 only with the One Cycle schedule, whose learning-rate finder picks the peak for you. |
| **Encoder LR Factor** | 0.1 | 0.01-1.0 | Multiplies the learning rate for the encoder layers. 0.1 trains the encoder at a tenth of the decoder's rate, which suits ImageNet-pretrained backbones on new image types. 0.3-0.5 suits histology-pretrained backbones on histology. 1.0 uses one rate for every layer. Has no effect when training from scratch. |
| **Weight Decay** | 0.01 | 0.0-0.5 | AdamW weight decay; penalizes large weights to reduce overfitting. 0.05-0.1 for very small datasets, 0.001 for large datasets or from-scratch training, 0 turns it off. |
| **LR Scheduler** | One Cycle | One Cycle, Cosine Annealing, Reduce on Plateau, Step Decay, None | How the learning rate changes during training. One Cycle is recommended for most cases. Reduce on Plateau halves the learning rate when the monitored metric stops improving. See [PyTorch schedulers](https://docs.pytorch.org/docs/stable/optim.html). |
| **Auto-find learning rate (LR Finder)** | On | On/Off | Runs a 100-step sweep before training to pick the One Cycle peak learning rate. Only used with the One Cycle schedule. When off, the peak is the base learning rate x sqrt(batch size / 8). Turn it off for a tiny model where the sweep is a large share of the run, or when you already know a good rate. |

### Loss Function

This section appears only after **Show All Settings**.

| Parameter | Default | Range / Options | Description |
|-----------|---------|---------|-------------|
| **Loss Function** | Cross Entropy + Dice | Cross Entropy + Dice, Cross Entropy, Focal + Dice, Focal, Boundary-softened CE, Boundary-softened CE + Dice, Lovasz-Softmax, CE + Lovasz-Softmax | Cross Entropy + Dice is recommended for most cases. **Focal + Dice** down-weights well-classified pixels via (1-p)^gamma, helping the model focus on hard regions (e.g., small structures in large images). **Focal** is focal loss without Dice. **Boundary-softened CE** weights CE by distance to the nearest annotation boundary, down-weighting noisy edge pixels (set with **Boundary Sigma (px)** and **Boundary Floor Weight**); **Boundary-softened CE + Dice** pairs it with Dice. **Lovasz-Softmax** directly optimizes mean IoU and has no hyperparameters; **CE + Lovasz-Softmax** adds a CE term for stable early gradient. Hard-pixel mining (**Hard Pixel End %**) composes with all variants except the two Lovasz options. See [smp losses](https://smp.readthedocs.io/en/latest/losses.html) and [Training Guide: Loss function options](TRAINING_GUIDE.md#loss-function-options). |
| **Focal Gamma** | 2.0 | 0.5-5.0 | Focal loss focusing parameter. Higher = stronger focus on hard pixels. Only visible when a Focal loss variant is selected. gamma=2 is standard. |
| **Boundary Sigma (px)** | 3.0 | 0.5-32 | How far from an annotation boundary the down-weighting reaches, in pixels. Larger = wider soft band. 5-10 px for noisy, polygon-traced edges. Only visible with a Boundary-softened CE variant. |
| **Boundary Floor Weight** | 0.1 | 0.0-1.0 | The weight a pixel exactly on an annotation boundary gets. 0 ignores boundary pixels entirely; 1.0 is plain CE. Only visible with a Boundary-softened CE variant. |
| **Hard Pixel End %** | 100 | 5-100 | Online Hard Example Mining (OHEM): each batch, keep only the hardest N% of pixels for the pixel loss. 100% = all pixels (no mining). 25% = aggressive mining. Ignores well-classified pixels entirely instead of down-weighting them as focal loss does. Dice loss still uses all pixels. |
| **Hard Pixel Start %** | 100 | 5-100 | Used only when Hard Pixel End % is below 100. Start above End: the ratio moves linearly from Start to End over the first 75% of epochs. Start equal to End: the ratio is fixed. Must be at least Hard Pixel End %. |
| **Adaptive per-class floor** | Off | On/Off | Used only when Hard Pixel End % is below 100. Off: each class keeps its own hardest share of pixels, in proportion to its size. On: the hardest pixels are chosen across all classes, then every class present is topped up to a minimum, so mining can concentrate on the class the model is failing without any class dropping to zero. |

### Performance

This section appears only after **Show All Settings**.

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **Fused optimizer (CUDA only)** | On | On/Off | PyTorch's fused AdamW update on NVIDIA GPUs: same math, a few milliseconds faster per step. Ignored on CPU and Apple Silicon. |
| **Progressive resizing** | Off | On/Off | Train at half resolution for the first 40% of epochs, then switch to full resolution. Speeds up early training and acts as implicit regularization. Inspired by fast.ai. |
| **GPU augmentation (experimental, CUDA only)** | Off | On/Off | Runs flips, 90-degree rotations, color or brightness changes, noise and blur on the GPU (kornia) instead of the CPU. Elastic deformation stays on the CPU path. Falls back to the CPU when kornia is missing or the device is not CUDA. |
| **torch.compile (experimental, Linux+CUDA)** | Off | On/Off | Compiles the model for faster steps after a 15-40 s start-up cost. Linux with an NVIDIA GPU only; the box is disabled on other systems. Falls back to normal execution on any failure. |
| **In-memory dataset** | auto | auto, on, off, bounded | Preloads training patches into RAM to skip reading them from disk every batch. **auto**: preload when the dataset fits in about half of the free RAM, otherwise read from disk. **on**: always preload (may run out of memory). **off**: always read from disk. **bounded**: trains on a fixed random subset that fits in the **Training: Bounded Cache Fraction** of RAM (default 40%). |
| **DataLoader workers** | 0 | 0-8 | Background processes that load and augment patches while the GPU works. Ignored (set to 0) while **In-memory dataset** is auto or on; set it to off or bounded to use workers. See [Preferences: DataLoader workers](PREFERENCES.md#training-dataloader-workers-experimental). |

### Channel Configuration

| Parameter | Description |
|-----------|-------------|
| **Available Channels** | Image channels available for selection. Multi-select with Ctrl+click. |
| **Selected Channels** | Channels used as model input. Order must match at inference time. |
| **Normalization** | Strategy: PERCENTILE_99 (recommended; maps the 1st-99th percentile range to 0-1 and clips the rest), MIN_MAX, Z_SCORE (mean 0, standard deviation 1), FIXED_RANGE. Saved with the model and reused when the classifier is applied. |
| **Per-channel normalization (normalize each channel independently)** | Shown and on for images with more than 3 channels that are not brightfield; hidden and off otherwise. On: each channel gets its own range (fluorescence channels differ in brightness). Off: one range shared by all channels (keeps RGB color balance). Saved with the model and reused when the classifier is applied. |

### Annotation Classes

| Parameter | Description |
|-----------|-------------|
| **Class list** | Annotation classes found in the image. At least 2 must be selected. A pie chart shows the per-class annotation area distribution. |
| **Weight multiplier** | Per-class weight multiplier. >1.0 boosts underrepresented classes. Hidden in the basic view. |
| **Rebalance Classes** | Button that auto-sets weight multipliers inversely proportional to class area, compensating for class imbalance. |
| **Rebalance by default** | Checkbox (default: on). When enabled, weight multipliers are automatically rebalanced whenever classes are loaded. |

### Data Augmentation

This section appears only after **Show All Settings**.

| Augmentation | Default | Description |
|-------------|---------|-------------|
| **Horizontal flip** | On | Mirror tiles left-right. Almost always beneficial. |
| **Vertical flip** | On | Mirror tiles top-bottom. Safe for most histopathology. |
| **Random rotation (90 deg)** | On | Rotate by 0/90/180/270. Combines with flips for 8x augmentation. |
| **Intensity augmentation** | Auto-selected | Three modes: **None** (no intensity transforms), **Brightfield (color jitter)** (correlated RGB brightness/contrast/saturation, good for H&E), **Fluorescence (per-channel)** (independent per-channel intensity scaling, good for IF). Auto-selected based on image type when classes are loaded. |
| **Elastic deformation** | Off | Smooth spatial deformations. Effective but ~30% slower. See [Albumentations](https://albumentations.ai/docs/). |

### Name Your Classifier

| Parameter | Type | Description |
|-----------|------|-------------|
| **Classifier Name** | Text | The name shown in menus and saved with the model. The model folder is derived from it (lower-cased, other characters replaced by `_`, a timestamp appended), so only `< > : " / \ \| ? *` are refused; spaces, periods and parentheses are fine. |
| **Description** | Text | Optional free-text description stored in classifier metadata. |

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

## SSL Pretraining Parameters

These parameters are available in the **SSL Pretrain Encoder** dialog (**Extensions > DL Pixel Classifier > Utilities > SSL Pretrain Encoder...**). That menu item is hidden until you turn on **Edit > Preferences > DL Pixel Classifier > Show Developer Pretraining Options** and restart QuPath. SSL pretraining trains CNN encoder backbones (ResNet, EfficientNet, MobileNet) using self-supervised learning on unlabeled image tiles.

### Method & Architecture

| Parameter | Options | Description |
|-----------|---------|-------------|
| **SSL Method** | SimCLR, BYOL | Self-supervised method. SimCLR uses contrastive learning (benefits from larger batches). BYOL uses self-distillation (works better with small datasets). |
| **Backbone** | resnet18, resnet34, resnet50, efficientnet-b0/b1/b2, mobilenet_v2, histology-pretrained variants | CNN encoder backbone to pretrain. Must match the backbone you will use for supervised training. Foundation/ViT encoders are excluded (not compatible with SMP SSL). |
| **Temperature** | 0.05-1.0 (default 0.5) | SimCLR only. Temperature scaling for contrastive loss. Lower = sharper similarity distribution. Try 0.1-0.2 for small datasets. |
| **Projection dim** | 64-512 (default 256) | Output dimension of the projection head MLP. 256 is the standard for both methods. |
| **Initialize from trained model** | File path (optional) | For domain-adaptive pretraining: select an existing trained model (.pt file) to initialize the encoder. The encoder starts with the learned weights and adapts them during SSL pretraining. Leave empty to train from scratch. |

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Epochs** | 100 | 10-2000 | Number of pretraining epochs. Auto-suggested based on dataset size in folder mode. |
| **Batch Size** | 64 (SimCLR) / 32 (BYOL) | 2-256 | Tiles per training step. SimCLR benefits from larger batches. Gradient accumulation auto-applied for SimCLR when effective batch < 256. |
| **Learning Rate** | 0.0003 | 0.00001-0.01 | AdamW learning rate with cosine annealing and linear warmup. |
| **Warmup Epochs** | 10 | 0-50 | Linear learning rate warmup. Prevents early instability. |

### Data

| Parameter | Description |
|-----------|-------------|
| **Source mode** | **Project images**: extract tiles from annotated regions of selected QuPath project images. **Pre-extracted folder**: use existing image tiles from a directory. |
| **Images** | (Project mode) Select which project images to include. Shows annotation count per image. |
| **Annotation classes** | (Project mode) Multi-select: choose which annotation classes define tile extraction regions. Only areas covered by checked classes contribute tiles. |
| **Tile size** | 128-1024 (default 256). Size of extracted tiles. Should be close to the tile size used for supervised training. |
| **Downsample** | 1x, 2x, 4x, 8x. Resolution factor when reading tiles. Match the downsample used in supervised training. |
| **Max tiles (total)** | 100-200000 (default 5000). Cap on total tiles after extraction. Prevents one large image from dominating. |

### Output

| Parameter | Description |
|-----------|-------------|
| **Output directory** | Where to save `model.pt` (encoder weights) and `metadata.json`. Defaults to `{project}/ssl_pretrained/{backbone}_{method}_{timestamp}/`. |

---

## Inference Parameters

The inference dialog is opened from **Extensions > DL Pixel Classifier > Apply DL Pixel Classifier...**

### Classifier Selection

| Parameter | Description |
|-----------|-------------|
| **Classifier table** | Select a trained classifier. Shows name, type (architecture), channels (with context scale info), class count, and training date. |
| **Info panel** | Below the table, shows architecture + backbone, input channels + context scale + tile dimensions, downsample level, and class names for the selected classifier. |

### Output Options

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **Output Type** | OBJECTS | OBJECTS, MEASUREMENTS | How results are represented. **OBJECTS** creates detection or annotation objects. It builds a class map of each annotation's whole bounding box, and an annotation whose map would need more than half of QuPath's free memory is refused with a message giving the numbers; split it, use MEASUREMENTS (which needs no map), or raise QuPath's memory limit. **MEASUREMENTS** adds `DL: <class> area (um^2)` and `DL: <class> %` for each class, plus `DL: Total pixels`, counted from the same per-pixel classification as OBJECTS (see [Inference Guide](INFERENCE_GUIDE.md#step-4-configure-output-type)). For a live preview, use **Toggle Prediction Overlay** from the menu. |
| **Object Type** | DETECTION | DETECTION, ANNOTATION | QuPath object type (OBJECTS output only). |
| **Min Object Size (um2)** | 10 um^2 | 0-10000 | Discard objects below this area. |
| **Hole Filling (um2)** | 5 um^2 | 0-1000 | Fill interior holes below this area. |
| **Boundary Smoothing** | 1.0 um | 0-10 | Simplification tolerance in micrometers (um). 0 keeps pixel-exact boundaries. |
| **Max Object Size (um2)** | 0 (no limit) | 0-10,000,000 | Discard objects above this area. |
| **Add shape measurements (circularity, solidity)** | Off | On/Off | Attach circularity and solidity to each object, for filtering by shape. |

Min/Max Object Size, Hole Filling and Boundary Smoothing apply to OBJECTS only. On an image with no pixel calibration, the um and um^2 values are applied as if 1 pixel = 1 um.

### Channel Mapping

| Parameter | Description |
|-----------|-------------|
| **Channel mapping grid** | Shows expected channels from the classifier and how they map to the current image's channels. Each row shows Expected -> Mapped To -> Status. Status is [OK] (exact match), [?] (fuzzy/substring match), or [X] (unmapped). Channels can be manually remapped using dropdown overrides. |
| **Summary label** | Shows the number of matched/unmatched channels. For brightfield images, the channel section auto-collapses when all channels match. |

### Processing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Tile Size** | Auto | Auto-set from classifier. Must be divisible by 32. Range: 64-8192. |
| **Tile Overlap (%)** | 12.5% | 0-50%. The margin of real image read on each side of a tile, as a percentage of the tile. Only the tile's center is kept, so the stride is tile x (1 - 2 x overlap%): 12.5% of a 512 px tile is 64 px on each side, stride 384 px. Higher = fewer seams, more tiles, slower. Shared with the Train dialog and the overlay. |
| **Blend Mode** | GAUSSIAN | GAUSSIAN, LINEAR, CENTER_CROP, NONE. Every mode keeps only the center of each tile and drops its edges. **CENTER_CROP** also raises each tile's margin to at least 25% of the tile, which removes rectangular blocks of the wrong class that line up with the tile grid, at the cost of about twice as many tiles at the default overlap. **GAUSSIAN**, **LINEAR** and **NONE** use the Tile Overlap as set and give the same OBJECTS and MEASUREMENTS results. The overlay uses the mode last chosen here. See [Rectangular blocks of the wrong class](TROUBLESHOOTING.md#rectangular-blocks-of-the-wrong-class-aligned-to-a-grid). |
| **Use GPU if available** | On | Saved with the dialog settings. In this release inference uses the GPU whenever the environment can (NVIDIA CUDA, then Apple Silicon, then CPU), whatever this box says. |
| **Test-Time Augmentation (TTA)** | Off | Runs each tile 8 times (flipped and rotated) and averages the answers. About 8x slower; the tooltip estimates 1-3% better segmentation. Best for final runs. |

### Normalization

| Parameter | Description |
|-----------|-------------|
| **Normalization method** | Always the one the model was trained with (strategy, per-channel setting and clip percentile), whatever the channel panel shows. See [Normalization round trip](NORMALIZATION_ROUNDTRIP.md). |
| **Image-level normalization** | Automatically enabled. Computes per-channel normalization statistics once across the entire image (sampling ~16 tiles in a 4x4 grid), then applies the same statistics to every tile. Eliminates tile boundary artifacts caused by per-tile normalization. |
| **Training dataset stats** | When the model's metadata carries training-set statistics, those are used; otherwise the extension samples the image as above. |

### Application Scope

| Parameter | Description |
|-----------|-------------|
| **Apply to whole image** | Classify entire image without annotations. |
| **Apply to all annotations** | Classify within all annotations (default). |
| **Apply to selected annotations only** | Classify only within selected annotations. |

With **Apply to whole image** and MEASUREMENTS, the extension adds a "Full Image" annotation to the image and puts the measurements on it.

> **Note:** The Apply dialog remembers output type, object type, object size limits, hole filling, boundary smoothing, shape measurements, tile size, overlap, blend mode, the GPU box and application scope across sessions.
