# Tips and Tricks

Practical workflow tips for getting the best results from the DL Pixel Classifier.

## Iterative Training: Quick Runs to Fix Annotations

**Don't commit to a long training run before checking your annotations.**

1. **Run a short 5-10 epoch training** with your initial annotations
2. **Toggle the overlay** to see where the model makes mistakes
3. **Look for the largest errors** -- these usually indicate annotation problems:
   - A region classified as the wrong class may have been annotated incorrectly
   - Boundaries between classes may be drawn too loosely or tightly
   - An entire class may be underrepresented (too few annotations)
4. **Fix the annotations** in QuPath based on what you see
5. **Re-train** with the corrected annotations for a longer run (50-200 epochs)

This iterative approach saves hours compared to training for 200 epochs only to discover that an annotation was wrong. Even a few epochs is enough for the model to reveal gross labeling errors.

**Pro tip:** After the short run, use **Review Training Areas** to see which tiles the model classified correctly and which it struggled with. The worst-performing tiles often point directly to annotation issues.

## Hard Pixel % (OHEM): Large Homogeneous Regions

If your training images have **large uniform regions** (e.g., background glass, tissue interiors, empty space), most pixels in each batch are "easy" -- the model classifies them correctly very early in training. Without intervention, the model spends most of its training time on these easy pixels, gaining almost nothing.

**When to use Hard Pixel %:**
- Your annotations include large area annotations where most of the interior is homogeneous
- The model quickly reaches high accuracy but stops improving on difficult boundary regions
- You see the model correctly classifying tissue interiors but struggling at class boundaries

**Recommended workflow:**
1. Start training WITHOUT Hard Pixel % (100%, the default) for the first run
2. If the model plateaus on easy regions but boundary accuracy is poor, reduce to 25-50%
3. For very large homogeneous regions, try 5-10% to focus almost entirely on boundaries

**Why not always use it?** For small annotations or images with few easy pixels, OHEM can be too aggressive -- it may discard pixels the model genuinely needs to learn. If your annotations are primarily lines or narrow polygons along class boundaries, OHEM provides little benefit because most pixels are already "hard."

**Alternative: Focal Loss** is a softer approach that down-weights easy pixels rather than completely ignoring them. Try Focal Loss first if you're unsure; switch to OHEM if Focal Loss doesn't focus enough on the hard cases.

## Line Annotations vs. Area Annotations

**Line annotations along class boundaries are often more effective than large area fills:**

- Lines focus training on the pixels that matter most -- the boundaries between classes
- Area annotations overrepresent the easy interior pixels (see Hard Pixel % above)
- Lines are faster to draw and easier to correct
- The model learns "this is where class A meets class B" rather than "here are 50,000 pixels of class A interior"

**When to use area annotations:**
- The class has no clear boundary (e.g., "tissue vs. background")
- The interior texture varies and needs to be learned (e.g., heterogeneous tumor)
- You have very few annotations and need more training pixels

**When to use line annotations:**
- Classes are defined by their boundaries (e.g., "vein margin", "epithelial border")
- The interior of each class is relatively uniform
- You want to train faster with less annotation effort

You can mix both: use area annotations for background/tissue and line annotations for fine structures.

## Using the Overlay for Quality Control

The **Toggle Prediction Overlay** is your primary tool for evaluating classifier quality:

1. Train a model (even a short run)
2. Toggle the overlay on
3. Pan across the entire image, looking for:
   - **Systematic errors**: entire regions consistently misclassified = annotation problem
   - **Noisy boundaries**: ragged, speckled edges = model needs more epochs or smoother annotations
   - **Missing classes**: a class never appears in the overlay = check that it's in the training data
4. The overlay and Apply Classifier (OBJECTS) use the exact same inference pipeline, so what the overlay shows is what you'll get as objects

## Choosing Tile Size

| Tile Size | Best for | Trade-offs |
|-----------|----------|------------|
| 256 | Cell-level features, small structures | Fast training, more tiles needed for coverage |
| 512 | General purpose (recommended default) | Good balance of context and memory |
| 1024 | Large tissue structures, architecture patterns | Slower training, requires more VRAM |

**Rule of thumb:** the tile should be large enough that a human could classify the center pixel by looking at the tile. If you need to see more surrounding context to make the classification, increase the tile size.

## When Training Stalls

If validation loss or mIoU plateaus early:

1. **Check class balance**: use "Rebalance Classes" to auto-weight underrepresented classes
2. **Add more annotations**: especially for the worst-performing class
3. **Try a different backbone**: resnet50 has more capacity than resnet34
4. **Increase tile size**: the model may need more spatial context
5. **Enable Hard Pixel %**: if easy pixels are dominating the loss (see above)
6. **Lower learning rate**: if loss oscillates wildly, check that LR is 1e-4 (the default)

## Learning Rate: Leave It at 0.0001

The default learning rate of **0.0001 (1e-4)** works well for nearly all cases. With discriminative learning rates enabled (automatic when using pretrained or frozen layers), the encoder gets 1/10th the rate (1e-5) while the decoder and head get the full 1e-4. This is a proven balance that:

- Prevents catastrophic forgetting of pretrained encoder features
- Gives the decoder enough gradient to learn your specific task
- Works for both fresh training and continue-training from a saved model

**When you might increase it:**
- You're using **OneCycleLR scheduler** (not ReduceOnPlateau) -- OneCycleLR runs an automatic LR finder that safely determines the optimal peak rate
- Training is converging **very slowly** after 50+ epochs with no improvement

**When you might decrease it:**
- **Continue-training** from a model that already achieved good results -- try 1e-5 for gentle fine-tuning
- Training with **all encoder layers unfrozen** -- the full network is more sensitive to large updates

**Never go above 1e-3 with ReduceOnPlateau** -- this causes wild oscillation where the model swings between excellent and terrible validation results epoch-to-epoch, especially with high class weights.

## Sharing Models

To share a trained classifier with someone else:

1. Navigate to `{project}/classifiers/dl/{model_name}/`
2. Share **only** `model.pt` and `metadata.json`
3. The recipient places both files in a folder under their project's `classifiers/dl/` directory
4. Files named `best_in_progress_*.pt` and `checkpoint_*.pt` are training artifacts (5x larger) and are NOT needed for inference -- safe to delete

## Multi-Scale Context

Enable **Context Scale** (2x-16x) when classification depends on what surrounds a region, not just the region itself:

- **Tumor vs. stroma**: the tissue architecture around a cell cluster helps distinguish tumor from benign
- **Tissue type classification**: knowing you're in liver vs. kidney helps classify individual structures
- **Anatomical regions**: large-scale spatial patterns that a single tile can't capture

Context scale adds minimal memory overhead (~5-10%) by interleaving a downsampled wide-view with the detail tile. Both the overlay and Apply Classifier handle context tiles automatically.

## Feed Training Logs to an LLM for Analysis

After a training run, copy the training log from the QuPath log window and paste it into an LLM (ChatGPT, Claude, Gemini, etc.) for analysis. The model can identify patterns you might miss: validation instability, class-specific bottlenecks, learning rate issues, and annotation problems.

**Example prompt:**

> I am training a deep learning pixel classifier for H&E histology images using the [DL Pixel Classifier extension for QuPath](https://github.com/uw-loci/qupath-extension-dl-pixel-classifier). The training configuration and parameter documentation is here: [PARAMETERS.md](https://github.com/uw-loci/qupath-extension-dl-pixel-classifier/blob/main/docs/PARAMETERS.md) and [TRAINING_GUIDE.md](https://github.com/uw-loci/qupath-extension-dl-pixel-classifier/blob/main/docs/TRAINING_GUIDE.md).
>
> I currently have patches across about 10 whole slide H&E images. Please analyze the following training results and provide suggestions for what could be improved when I continue training.

Then paste the full training log output (everything from "Starting training for classifier" through the final epoch).

<details>
<summary><b>Example: training log output (click to expand)</b></summary>

```
Starting training for classifier: Classifier_20260321
Export directory: C:\Users\MICHAE~1\AppData\Local\Temp\dl-training6088532274938927794
Context padding: 128px per side (tiles will be 768x768)
Exporting from 12 project images...
Exported 3638 training patches
Connected to classification backend
Transfer learning: 3 layer groups frozen: encoder.conv1, encoder.layer1, encoder.layer2
Training on NVIDIA GeForce RTX 3090 (CUDA)
--- Training Configuration ---
  Architecture: unet (resnet50_kather100k)
  Optimizer: AdamW (wd=0.01, betas=0.9/0.99) [discriminative LRs]
  Scheduler: ReduceOnPlateau (factor=0.5, patience=10)
  Loss: ce_dice + OHEM (keep 50%)
  Batch Size: 8 (accumulation=2, effective=16)
  Tile Size: 512x512
  Mixed Precision: BF16
  Classes: 3 (Gland, Ignore*, Normal)
  Training Patches: 2910 train / 728 val
  Normalization: percentile_99 (per_channel=False)
  Channels: 6
  Early Stopping: mean_iou (patience=30), focus=Gland (min IoU=0.80)
  Augmentation: p_rotate, p_flip, p_elastic, intensity_mode
  Progressive Resize: Off
  Discriminative LRs: encoder=0.000010, decoder=0.000100, head=0.000100
Epoch 1: train_loss=1.2748, val_loss=0.8495, acc=75.0%, mIoU=0.6134
  IoU: Gland=0.340 Ignore*=0.814 Normal=0.687
Epoch 2: train_loss=0.8013, val_loss=0.9228, acc=81.7%, mIoU=0.6820
  IoU: Gland=0.403 Ignore*=0.863 Normal=0.780
...
Epoch 92: train_loss=0.2497, val_loss=0.3644, acc=96.0%, mIoU=0.8975
  IoU: Gland=0.763 Ignore*=0.980 Normal=0.950
...
Epoch 96: train_loss=0.2859, val_loss=0.2190, acc=94.5%, mIoU=0.8752
  IoU: Gland=0.719 Ignore*=0.979 Normal=0.927
Training paused at epoch 96
Best model: epoch 92
Loss: 0.3644 | Accuracy: 96.04% | mIoU: 0.8975
```

</details>

<details>
<summary><b>Example: LLM analysis response (click to expand)</b></summary>

**Overall Performance:**

| Metric | Early (ep 1-10) | Mid (ep 30-50) | Best (ep 92) |
|--------|----------------|----------------|--------------|
| mIoU | 0.61 - 0.80 | 0.80 - 0.89 | **0.8975** |
| Accuracy | 75 - 91% | 92 - 96% | **96.0%** |
| Gland IoU | 0.34 - 0.61 | 0.64 - 0.75 | **0.763** |

**Key findings:**

1. **High validation variance** -- loss spikes on individual epochs (e.g., epoch 3 val_loss=4.93, epoch 49 val_loss=2.09) concentrated on the Ignore\* class, suggesting inconsistent annotation boundaries.

2. **Gland is the bottleneck** -- never reached the 0.80 IoU focus target. Ranges 0.48-0.76 even in late epochs while Normal hits 0.90+ consistently.

3. **No overfitting** -- train loss (~0.25) and val loss (~0.22-0.36) gap is healthy, model has capacity to learn more.

**Recommendations:**

- **Review Ignore\* annotations** for inconsistent boundaries (the loss spike source)
- **Add more Gland annotations**, focusing on boundaries, small/fragmented glands, and morphological variety across slides
- **Unfreeze layer2** when continuing (keep conv1 + layer1 frozen) to allow more encoder fine-tuning
- **Lower decoder/head LR to ~0.00005** since you're continuing from a good checkpoint
- **Another 50-100 epochs** should be sufficient with improved annotations

</details>

**Tips for getting better LLM analysis:**

- Include the **full log**, not just a summary -- the per-epoch per-class breakdown is where the insights are
- Mention how many images and what tissue type you're working with
- Include the links to the documentation so the LLM understands the available parameters
- Ask specifically about what to change for the **next** training run
