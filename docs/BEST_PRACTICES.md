# Best Practices

Guidance for project organization, backbone selection, annotation strategy, hyperparameter tuning, and improving classifier results.

## Training vs. Production Projects

**The single most important organizational practice: keep your training annotations in a dedicated project, separate from where you apply the classifier for analysis.**

### Why separate projects?

A training project is a curated dataset -- your annotations are the ground truth that the model learns from. When you run inference in the same project (especially with OBJECTS output), the classifier generates thousands of detection or annotation objects on top of your hand-drawn training annotations. This creates several problems:

- **Data corruption risk**: Generated objects mix with your training annotations, making it hard to distinguish human labels from model predictions. If you retrain from this project, the model may inadvertently learn from its own (potentially incorrect) outputs.
- **Difficult iteration**: When you want to refine annotations and retrain, you must first clean out all the generated objects -- a tedious and error-prone process, especially across many images.
- **No clean baseline**: You lose the ability to compare different classifiers on the same images, since each run's objects accumulate.

### The recommended workflow

```
TRAINING PROJECT                         PRODUCTION PROJECT
+----------------------------------+     +----------------------------------+
| 1. Create annotation classes     |     | 1. Create a new QuPath project   |
| 2. Draw training annotations     |     |    with the images to analyze    |
| 3. Train classifier              |     | 2. Copy the trained classifier   |
| 4. Toggle overlay to check       |     |    into this project             |
|    quality visually              |     | 3. Apply classifier to images    |
| 5. Fix annotations, retrain     |     |    (OBJECTS, MEASUREMENTS, etc.) |
| 6. Repeat until satisfied        |     | 4. Run downstream analysis       |
+----------------------------------+     +----------------------------------+
```

**Step by step:**

1. **Training project**: Create a QuPath project with representative images. Draw annotations, train classifiers, and iterate. Use the **overlay** and **Review Training Areas** to evaluate quality -- these do not create persistent objects.
2. **Export the model**: When satisfied, locate the classifier directory at `{project}/classifiers/dl/{model_name}/`. You need `model.pt` and `metadata.json`. (Checkpoint files like `best_in_progress_*.pt` are not needed for inference.)
3. **Production project**: Create a separate QuPath project containing the images you want to analyze. Create a `classifiers/dl/{model_name}/` directory and copy `model.pt` and `metadata.json` into it. The model now appears in the inference dialog.
4. **Apply the classifier**: Use **Apply Classifier** or the [Scripting API](SCRIPTING.md) to run the model across all images. Generate OBJECTS, MEASUREMENTS, or RENDERED_OVERLAY as needed for your analysis.
5. **Iterate safely**: If the classifier needs improvement, go back to the training project, refine annotations, retrain, and copy the updated model to the production project. The training project remains clean.

### What is safe to do in the training project?

| Action | Safe? | Why |
|--------|-------|-----|
| **Toggle Prediction Overlay** | Yes | Renders on-the-fly, creates no persistent objects |
| **Review Training Areas** | Yes | Read-only evaluation of training tiles |
| **Apply Classifier (MEASUREMENTS)** | Use caution | Adds measurement columns to existing annotations -- won't create new objects, but modifies your training annotations' measurement tables |
| **Apply Classifier (OBJECTS)** | No | Creates detection/annotation objects that mix with training annotations |
| **Apply Classifier (RENDERED_OVERLAY)** | Yes | Creates a static image overlay, no objects |

### Quick project setup for production

The fastest way to set up a production project:

1. In QuPath, create a new project (**File > Project > Create project**)
2. Add your images to the project
3. In a file browser, navigate to `{training_project}/classifiers/dl/`
4. Copy the entire model folder(s) into `{production_project}/classifiers/dl/`
5. Open the production project in QuPath -- the classifiers appear in the Apply Classifier dialog

For scripted batch processing across an entire production project, see [Scripting: Batch process a project](SCRIPTING.md#batch-process-a-project).

## Backbone Selection

### Quick reference by image type

| Image type | Best backbone | Why |
|-----------|--------------|-----|
| H&E brightfield (20x) | Histology backbone | Pretrained on millions of H&E patches at 20x -- features transfer directly |
| H&E brightfield (other mag) | Histology backbone or resnet34 | Histology backbones still help but were trained at 20x; ImageNet is a safe fallback |
| Fluorescence (1-3 channels) | resnet34 (ImageNet) | ImageNet edge/texture features transfer well; histology H&E colors do not match IF |
| Multiplex IF (4+ channels) | resnet34 or resnet50 (ImageNet) | The first conv layer is automatically adapted to N input channels; ImageNet is the best starting point |
| Spectral / hyperspectral | resnet34 (ImageNet) | Same channel adaptation; more channels need more training data |

### Understanding histology-pretrained backbones

The histology-pretrained encoders were all trained on **3-channel H&E-stained brightfield** images at approximately **20x magnification** (0.5 um/px). Their learned features are specific to H&E color distributions and tissue morphology at that resolution:

| Encoder | Training data | Magnification | Channels | Method |
|---------|-------------|---------------|----------|--------|
| ResNet-50 Lunit SwAV | 19M TCGA H&E patches | ~20x | 3 (RGB) | SwAV self-supervised |
| ResNet-50 Lunit Barlow Twins | 19M TCGA H&E patches | ~20x | 3 (RGB) | Barlow Twins self-supervised |
| ResNet-50 Kather100K | 100K colorectal H&E patches | 20x (0.5 um/px) | 3 (RGB) | Supervised classification |
| ResNet-50 TCGA-BRCA | TCGA breast cancer H&E | ~20x | 3 (RGB) | SimCLR self-supervised |

**When histology backbones help most:**
- H&E brightfield at 20x -- this is exactly the domain they were trained on
- H&E at other magnifications -- features still partially transfer, especially tissue texture patterns
- Any 3-channel brightfield stain with eosin-like color distributions

**When to use ImageNet backbones instead:**
- **Fluorescence / IF images**: Histology backbones learned H&E-specific color features (pink eosin, blue hematoxylin). These do not transfer to fluorescence intensity patterns. ImageNet backbones provide better generic edge and texture features.
- **Multi-channel images (>3 channels)**: When the model has more than 3 input channels, the first convolutional layer must be adapted regardless of pretraining. ImageNet weights for the first conv are replicated across the extra channels. Histology first-conv weights encode H&E color responses that would be meaningless for IF channel combinations.
- **Non-tissue images**: Bright-field stains other than H&E where color patterns differ significantly.
- **Training instability with histology backbones**: If you see erratic loss curves, oscillating class IoU, or slow convergence with a histology-pretrained encoder, try switching to a standard ResNet (e.g., resnet50 with ImageNet weights) with limited frozen layers. ImageNet's generic edge and texture features sometimes provide a more stable training starting point, especially for non-H&E or multi-channel images.

### Foundation model encoders

Foundation model encoders are large-scale vision transformers pretrained on massive histopathology datasets (typically millions of whole-slide images). They provide rich, general-purpose tissue representations that transfer well to a wide range of downstream tasks with minimal fine-tuning. Foundation model integration inspired by LazySlide (Zheng et al. 2026, Nature Methods).

All foundation model encoders are **downloaded on-demand** from HuggingFace (~100 MB to ~2 GB depending on model size) and cached locally. They are not bundled with the extension. Only models with **commercially-permissive licenses** (Apache 2.0 or MIT) are included.

| Encoder | Parameters | License | Training data | Best for |
|---------|-----------|---------|---------------|----------|
| h-optimus-0 | 1.1B | Apache 2.0 | Large-scale histopathology | Highest capacity; best when VRAM and data are sufficient |
| virchow | 632M | Apache 2.0 | Large-scale histopathology | Strong general-purpose tissue encoder |
| hibou-l | 300M | Apache 2.0 | Histopathology | Good balance of capacity and efficiency |
| hibou-b | 86M | Apache 2.0 | Histopathology | Lightweight foundation model; good for limited VRAM |
| midnight | 1.1B | MIT | Histopathology | Large capacity with permissive license |
| dinov2-large | 300M | Apache 2.0 | General images (ImageNet-22k+) | General-purpose; not histology-specific but strong on diverse image types |

**When to use foundation models vs histology-pretrained ResNet-50:**

- **Use foundation models when:** You have 12+ GB VRAM, your task involves H&E or general tissue classification, and you want the best possible feature quality with less annotation effort. Foundation models learn richer representations than ResNet-based encoders and often achieve better results with fewer training examples.
- **Use histology-pretrained ResNet-50 when:** You have limited VRAM (8 GB or less), need fast training/inference, or your dataset is large enough that a simpler encoder works well. ResNet-50 histology backbones are much smaller and faster while still outperforming ImageNet-only weights on H&E.
- **Use ImageNet ResNet-34/50 when:** Your images are fluorescence, multiplex IF, or multi-channel (>3 channels). Foundation models, like histology ResNet-50, were trained on H&E RGB images and their features do not transfer to fluorescence intensity patterns.

**Important notes:**

- Foundation model encoders default to **encoder-frozen training** because their pretrained representations are already very strong. Freezing the encoder and only training the decoder is recommended for small-to-medium datasets. Unfreeze selectively only with large datasets (>5000 tiles).
- **Gated models** on HuggingFace require a HuggingFace authentication token. If a model requires authentication, the extension will prompt you to enter your HuggingFace token. You can obtain a token at https://huggingface.co/settings/tokens and accept the model's license agreement on its HuggingFace page.
- Foundation model encoders are significantly larger than ResNet backbones. Plan for higher VRAM usage and longer per-epoch training times.

| Encoder | Approximate VRAM (batch=4, 512px) | Download size |
|---------|-----------------------------------|---------------|
| hibou-b | ~6 GB | ~350 MB |
| hibou-l | ~10 GB | ~1.2 GB |
| virchow | ~12 GB | ~2.5 GB |
| h-optimus-0 | ~16 GB | ~4.4 GB |
| midnight | ~16 GB | ~4.4 GB |
| dinov2-large | ~10 GB | ~1.2 GB |

### By dataset size

| Dataset size | Recommended backbone | Freeze strategy |
|-------------|---------------------|-----------------|
| <200 tiles | resnet34 or efficientnet-b0 | Freeze all encoder layers |
| 200-1000 tiles | resnet34 | Freeze early layers (first 2 blocks) |
| 1000-5000 tiles | resnet34 or resnet50 | Freeze first 1-2 blocks |
| >5000 tiles | resnet50 (or histology for H&E) | Unfreeze most or all layers |

### By computational resources

| VRAM | Recommended backbone | Max tile size at batch=8 |
|------|-------------|--------------------------|
| 4 GB | efficientnet-b0 | 256px |
| 8 GB | resnet34 | 512px |
| 12+ GB | resnet50 | 512-1024px |
| 24+ GB | resnet50 | 1024px |

### Multi-channel fluorescence tips

When working with multiplex IF or multi-channel images:

- **Use ImageNet backbones** (resnet34 or resnet50), not histology backbones. The model automatically adapts the first convolutional layer to your channel count.
- **More channels need more training data.** With 7+ channels, each channel adds parameters to the first conv layer that start without meaningful pretrained values. Plan for at least 500-1000 tiles.
- **Consider per-channel normalization** (`per_channel: true`). Different fluorescence channels often have very different intensity ranges (e.g., DAPI vs. weak markers).
- **Select only relevant channels** in the channel selection panel rather than using all available channels. Fewer input channels means faster training and less data needed.
- **resnet34 is sufficient for most IF tasks.** Only move to resnet50 if you have a large dataset (>5000 tiles) and complex tissue patterns. The extra capacity of resnet50 is more likely to overfit on small IF datasets.

## Annotation Strategy

### Quality principles

1. **Annotate boundaries carefully**: The model learns most from class transition zones
2. **Cover variability**: Include examples from different staining intensities, tissue regions, and preparation qualities
3. **Be consistent**: Apply the same classification criteria throughout
4. **Include "hard" cases**: Annotate areas where classification is ambiguous -- these are most informative
5. **Use multiple images**: Multi-image training produces more robust classifiers

### Minimum annotation requirements

| Task complexity | Minimum annotations per class | Recommended |
|----------------|------------------------------|-------------|
| 2-class (foreground/background) | 10 brush strokes | 30+ brush strokes |
| 3-4 class | 15 annotations per class | 40+ per class |
| 5+ classes | 20 annotations per class | 50+ per class |

### Common mistakes

- **Only annotating "easy" regions**: The model needs hard examples to learn boundaries
- **Unbalanced annotations**: One class has 10x more area than another -- use weight multipliers to compensate
- **Inconsistent labeling**: Different annotators applying different criteria to the same tissue
- **Annotating only one image**: Single-image classifiers often fail on new images

## Hyperparameter Tuning

### First training run

Use these conservative settings for your first attempt:

```
Architecture: UNet
Backbone: resnet34
Epochs: 50 (early stopping will handle the rest)
Batch Size: 8
Learning Rate: 0.0001  (auto-tuned by LR finder when using One Cycle)
Tile Size: 512
Pretrained: Yes
LR Scheduler: One Cycle
Loss: CE + Dice
Gradient Accumulation: 1  (increase to 2-4 if VRAM is limited)
Progressive Resizing: Off  (try enabling for large tile sizes)
```

> **Note:** The optimizer is AdamW with fast.ai-tuned defaults (weight_decay=0.01). Discriminative learning rates are automatically applied when using pretrained weights -- the encoder trains at 1/10th the base LR.

### If results are poor

#### Model underfits (low accuracy on both train and validation)

- Increase epochs
- Increase model capacity (resnet50 instead of resnet34)
- Reduce tile overlap if generating too many similar patches
- Unfreeze more encoder layers
- Check annotation quality

#### Model overfits (high train accuracy, low validation accuracy)

- Freeze more encoder layers
- Reduce learning rate
- Enable more augmentation (color jitter, elastic deformation)
- Add more training data (more annotations, more images)
- Reduce model capacity (resnet34 instead of resnet50)
- Increase validation split to detect overfitting earlier

#### Training is unstable (loss oscillates)

- Reduce learning rate (try 1e-4 or 1e-5)
- Switch to One Cycle scheduler (the auto LR finder helps find the right max LR)
- Try "Reduce on Plateau" scheduler -- it automatically lowers the LR when progress stalls
- Reduce batch size, or use gradient accumulation (set accumulation=2-4 with a smaller batch)
- Check for annotation errors (mislabeled regions)

#### Specific classes perform poorly

- Increase weight multiplier for that class
- Add more annotations for that class, especially at boundaries
- Check if the class has consistent visual features
- Consider merging visually similar classes
- **Try Focal + Dice loss** if a region is consistently hard (see below)

### Focal loss and OHEM for hard regions

When most of the image is easy but a small region is hard (e.g., vein classification at the wing hinge vs. easy intervein areas), standard CE gives every pixel equal weight. This drowns the hard-region gradient in easy-pixel signal.

**Focal loss** (`Focal + Dice`) modulates CE by `(1 - p_t)^gamma`, where `p_t` is the model's predicted probability for the true class. Well-classified pixels (high `p_t`) get down-weighted; misclassified pixels (low `p_t`) contribute full gradient.

| Gamma | Effect |
|-------|--------|
| 0 | Standard CE (no focusing) |
| 1 | Mild focusing |
| 2 | Standard focal (recommended starting point) |
| 3-5 | Aggressive -- use when the hard region is very small |

**When to use focal loss:**
- One class or region consistently underperforms despite good annotations
- Large images where the hard region is a small fraction of total area
- You have already tried class weight multipliers and want more focusing

**OHEM (Online Hard Example Mining)** is more aggressive: it keeps only the hardest K% of pixels per batch and completely ignores the rest. Set "Hard Pixel %" to 25% to keep only the hardest quarter.

| Approach | Softness | Best for |
|----------|----------|----------|
| Class weights | Softest | Class imbalance (one class has less area) |
| Focal loss (gamma=2) | Medium | Hard regions within a class |
| OHEM (25%) | Aggressive | Very hard regions, model plateauing on easy data |

**Tip:** You can combine OHEM with any loss function. When combined with Dice variants, OHEM applies to the pixel-loss component only (Dice is unchanged since it operates on region overlap).

**No-code alternative:** Instead of focal loss, you can add small annotations around just the hard regions and use tiled mode. The tile exporter generates tiles from each annotation, so focused annotations produce extra tiles from the difficult area.

### Advanced tuning

| Scenario | Adjustment |
|----------|-----------|
| Very small dataset (<200 tiles) | efficientnet-b0, freeze all encoder, augmentation on, epochs=200, batch=4 |
| Large dataset (>10000 tiles) | resnet50, unfreeze all, epochs=50, batch=16, lr=5e-4 |
| Multi-scale features needed | Try a larger backbone (resnet50) or import a custom ONNX model |
| Staining variation between slides | Enable color jitter augmentation |
| Tissue distortion artifacts | Enable elastic deformation augmentation |
| Limited VRAM but want large effective batch | Set batch=4 with gradient accumulation=4 (effective batch 16) |
| Large tile size (512-1024px) | Enable progressive resizing to speed up early epochs |
| Noisy training curves | Use "Reduce on Plateau" scheduler instead of One Cycle |

## Monitoring Training

### What to watch in the loss chart

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both losses decrease smoothly | Good training | Continue |
| Train loss decreases, val loss increases | Overfitting | More data, more augmentation, more freezing |
| Both losses plateau early | Underfitting | More capacity, unfreeze layers, check data |
| Loss oscillates wildly | LR too high | Reduce learning rate |
| Loss barely decreases | LR too low or data issue | Increase LR, check annotations |

### When to stop manually

Early stopping handles this automatically, but you can also cancel manually. When you click **Cancel**, a dialog offers three options:

- **Best Epoch** -- save the model from the epoch with the highest mean IoU
- **Last Epoch** -- save the model from the most recently completed epoch
- **Do Not Save** -- discard all progress

After choosing, the dialog becomes closeable immediately -- you do not need to wait for the background cleanup to finish. The saved model is fully usable for inference.

## Normalization Strategy

The extension supports four normalization strategies. The choice affects how pixel intensities are scaled before the model sees them.

| Strategy | Best for | Notes |
|----------|----------|-------|
| **PERCENTILE_99** (default) | Most images | Clips to 1st/99th percentile, robust to outliers |
| **MIN_MAX** | Uniform-intensity images | Uses full dynamic range, sensitive to outliers |
| **Z_SCORE** | Images with consistent intensity distributions | Mean/std normalization, good for fluorescence |
| **FIXED_RANGE** | When you know the exact intensity range | Specify min/max values explicitly (e.g., 0-4095 for 12-bit) |

**Image-level normalization** (enabled by default) computes statistics once across the entire image, then applies them consistently to every tile. This eliminates input-level tile boundary artifacts. Newly trained models also save training dataset statistics in their metadata for even better consistency across different images.

**BatchRenorm** -- All newly trained models use BatchRenorm instead of standard BatchNorm for the network's internal normalization layers. This eliminates a second source of tiling artifacts: standard BatchNorm accumulates running statistics during training that can diverge from actual tile statistics at inference time, causing inconsistent predictions at tile boundaries. BatchRenorm uses consistent global statistics in both training and inference, producing seamless tiled predictions. See [Buglakova et al., ICCV 2025](https://arxiv.org/abs/2503.19545).

**Context padding** -- Training tiles are automatically extracted with a border of real surrounding image data. During inference, QuPath's `inputPadding` provides real context around each tile. Context padding ensures training geometry matches inference geometry: the model always sees real data at tile edges, never artificial reflection-padded data. The padding amount (`max(64, min(max(overlap, tileSize/4), tileSize * 3/8))` pixels per side) is computed automatically. The mask border is filled with 255 (ignore_index) so the loss function ignores the padding region. This is disabled for whole-image mode where no surrounding data is available.

## Improving Results

### Quick wins

1. **Add more annotations** at class boundaries
2. **Use multi-image training** if available
3. **Enable augmentation** (especially flips and rotation)
4. **Use a histology backbone** for H&E images
5. **Increase epochs** with early stopping (it is safe to overshoot)
6. **Retrain from a previous model** using "Continue training from saved model" in the Weight Initialization section to iterate quickly with the same hyperparameters
7. **Re-train models** to save normalization statistics -- new models automatically store training dataset stats for improved inference consistency

### Medium effort

1. **Adjust class weights** for imbalanced datasets
2. **Try layer freezing strategies** (freeze most layers for small datasets, unfreeze for large)
3. **Experiment with tile size** (256 vs 512)
4. **Try different downsample levels** for tissue-level features

### Medium-high effort

1. **Enable Test-Time Augmentation (TTA)** during inference for 1-3% quality improvement (~8x slower)
2. **Use gradient accumulation** (set to 2-4) to simulate larger batches without more VRAM
3. **Enable progressive resizing** to speed up training on large tile sizes and reduce overfitting

### High effort

1. **Annotate more images** for diversity
2. **Try a custom ONNX model** trained externally with a different architecture
3. **Manual layer freeze tuning** for your specific dataset
4. **Cross-validation** using multiple train/test splits

## MAE Pretraining

Masked Autoencoder (MAE) pretraining teaches a MuViT encoder to understand tissue structure from unlabeled images before supervised training. This section covers when and how to use it effectively.

### When MAE pretraining helps

- **Domain-specific tissue**: Your images contain tissue types or staining patterns not well-represented by generic pretrained weights
- **Large unlabeled datasets**: You have many images but limited annotations -- MAE leverages the unlabeled data
- **Small labeled datasets**: A pretrained encoder needs fewer labeled examples to fine-tune effectively

### When to skip MAE pretraining

- **Standard H&E histopathology**: The built-in histology-pretrained backbones (Lunit, Kather100K) are already trained on millions of H&E patches
- **Very few images**: With fewer than ~30 tiles, the encoder cannot learn meaningful representations
- **Quick experiments**: Standard pretrained weights are sufficient for initial experiments; MAE pretraining adds hours to the workflow

### Mask ratio selection

| Mask ratio | Best for | Notes |
|-----------|----------|-------|
| 0.75 (default) | Most cases | Good balance between reconstruction difficulty and feature learning |
| 0.5-0.6 | Simple tissue patterns | Easier reconstruction task, learns coarser features |
| 0.8-0.9 | Complex, high-detail tissue | Harder task forces finer-grained feature learning; may need more epochs |

### Tips for effective pretraining

1. **Match model configuration**: Use the same model size (small/base/large) and patch size for pretraining and downstream training
2. **Use representative tiles**: Include tiles from various regions, staining intensities, and tissue types present in your target dataset
3. **Monitor reconstruction loss**: The loss should decrease steadily. If it plateaus very early, consider increasing mask ratio or epochs
4. **Pretraining on CPU is extremely slow**: GPU (CUDA) is strongly recommended. A typical pretraining run takes 1-6 hours on GPU depending on dataset size and epochs
5. **Save encoder weights**: The output directory contains the encoder weights file that can be loaded during classifier training

## SSL Pretraining (SimCLR / BYOL)

SSL (Self-Supervised Learning) pretraining teaches a CNN encoder backbone (ResNet, EfficientNet, MobileNet) to understand your images without labels. Unlike MAE which works with the MuViT transformer, SSL pretraining works with the standard UNet backbones.

### When SSL pretraining helps

- **Domain shift**: Your classifier works on your images but fails on images from a different microscope, staining protocol, or compression. SSL pretraining adapts the encoder to the new domain's visual characteristics. See the [Domain Adaptation Guide](DOMAIN_ADAPTATION_GUIDE.md) for a step-by-step walkthrough.
- **New tissue types or sample preparation**: You are working with a tissue or specimen type that looks very different from ImageNet natural images (the default pretraining). SSL lets the encoder learn the visual vocabulary of your specific data.
- **Small labeled datasets**: You have a large collection of unannotated images and a limited annotation budget. Pretrain on the unannotated images, then fine-tune with few labels.
- **Multi-site studies**: Images from different sites or scanners have systematic appearance differences. SSL pretraining on the pooled unlabeled data creates an encoder that generalizes across sites.
- **Rare classes**: When the class of interest is rare and hard to annotate, SSL pretraining on the broader tissue context helps the encoder understand the background and surrounding structures, which improves detection of the rare class during supervised training.

### When to skip SSL pretraining

- **Standard H&E histopathology at 20x**: The built-in histology-pretrained backbones (Lunit SwAV, Barlow Twins) are already trained on millions of H&E patches at 20x. They are likely better than a small SSL pretraining run.
- **Very few tiles (< 30)**: The encoder needs enough diversity to learn meaningful patterns. With fewer than ~30 tiles, SSL pretraining may not improve over ImageNet weights.
- **Quick experiments**: If you are still iterating on annotation strategy, skip pretraining until you have a stable workflow.

### SimCLR vs. BYOL

| Criterion | SimCLR | BYOL |
|-----------|--------|------|
| Batch size sensitivity | Needs large effective batch (64+) | Works with small batches (16-32) |
| Small datasets | May underperform | **Better choice** |
| Large datasets | **Better choice** | Good |
| Training speed | Faster per epoch | Slightly slower (two networks) |
| Typical use | General pretraining | Domain adaptation with limited data |

**Recommendation:** Use **BYOL** for domain adaptation (you typically have tens to hundreds of images, not thousands). Use **SimCLR** when you have a large pool of unlabeled images (500+).

### Domain-adaptive pretraining

The most powerful use of SSL pretraining is **domain adaptation** -- taking an encoder that already works well on one set of images and adapting it to a different set:

1. Train a supervised classifier on your original images (the "source domain")
2. Collect unlabeled images from the new domain (different microscope, different staining, etc.)
3. Run **SSL Pretrain Encoder** with the original model as the "source model"
4. The encoder preserves its learned features while adapting to the new image characteristics
5. Fine-tune with a few annotations from the new domain

This approach requires far fewer annotations than training from scratch because the encoder already knows what biological structures look like -- it just needs to learn how they appear in the new imaging conditions.

### Tips

1. **Match the backbone**: Use the same backbone for SSL pretraining and supervised training (e.g., both ResNet-34)
2. **100-200 epochs is usually sufficient**: The loss should decrease steadily. Diminishing returns past 200 epochs for most datasets.
3. **Use representative tiles**: Include tiles from various regions and staining intensities
4. **Check metadata.json**: After pretraining, open `metadata.json` next to the model.pt to verify the method, backbone, and epoch count

## Interpreting Tile Evaluation Results

After training, the **Review Training Areas** feature evaluates every training tile and ranks them by loss. This section explains how to use those results to improve your classifier.

### Understanding the metrics

| Metric | What it means | Healthy range |
|--------|--------------|---------------|
| **Loss** | Cross-entropy loss for the tile. Higher = model's prediction differs more from the annotation. | < 0.5 for well-learned tiles |
| **Disagree%** | Percentage of labeled pixels where the model's prediction differs from the annotation mask. | < 10% for clear regions |
| **mIoU** | Mean Intersection-over-Union across all classes present in the tile. 1.0 = perfect agreement. | > 0.7 for good predictions |

### Prioritizing which tiles to review

1. **Start with the highest-loss tiles** -- these are the most likely annotation errors
2. **Focus on training split first** -- high loss on training tiles almost always indicates an annotation problem, since the model had the chance to learn from these tiles
3. **High-loss validation tiles** may indicate areas where the model hasn't generalized, which is expected for unusual tissue patterns
4. **Tiles with very high disagreement (>50%)** are likely mislabeled or contain mixed classes

### Common patterns and actions

**Annotation errors (most common):**
- A tile labeled "Tumor" is actually stroma, or vice versa
- Fix: Navigate to the tile, correct the annotation class, and retrain

**Boundary ambiguity:**
- Tiles at class boundaries naturally have higher loss
- Fix: If boundary annotations are inconsistent, re-annotate with a consistent policy. Some boundary loss is normal and expected.

**Hard cases:**
- Tiles with unusual morphology that the model hasn't learned well
- Fix: Add more annotations of similar tissue patterns to help the model learn

**Staining variation:**
- Tiles from regions with different staining intensity
- Fix: Enable color jitter augmentation, or add annotations from diverse staining regions

### Iterative improvement workflow

1. Train your initial model
2. Click **"Review Training Areas..."** before closing the progress dialog
3. Sort by loss and review the top 10-20 problematic tiles
4. Navigate to each tile, assess whether it's an annotation error or a hard case
5. Fix annotation errors directly in QuPath
6. Retrain using **"Continue training from saved model"** in the Weight Initialization section with the corrected annotations
7. Repeat until the high-loss tiles are genuine hard cases rather than annotation errors
