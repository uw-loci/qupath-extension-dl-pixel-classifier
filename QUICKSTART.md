# Quickstart Guide

Get from zero to your first trained pixel classifier in about 10 minutes.

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.6.0 or later (including 0.7.0) |
| Platform | Windows 10+ (64-bit), Linux (64-bit), or macOS Apple Silicon (M1+) |
| GPU | NVIDIA GPU with CUDA recommended; Apple Silicon (MPS) also works; CPU fallback available |
| Internet | Required for first-time environment setup (~2-4 GB download) |

> **Not supported:** Intel Macs and 32-bit systems. See [Installation Guide](docs/INSTALLATION.md) for full platform details and NVIDIA driver requirements.

> **Note:** A separate Python installation is **not** required. The extension manages its own embedded Python environment via [Appose](https://github.com/apposed/appose).

---

## Step 1: Install the Extension

1. Download the latest JAR from the [GitHub Releases](https://github.com/uw-loci/qupath-extension-dl-pixel-classifier/releases) page
2. **Drag and drop** the JAR directly onto the open QuPath window (easiest), or copy it manually to your QuPath extensions directory:

| OS | Typical extensions path |
|----|------------------------|
| Windows | `C:\Users\<you>\QuPath\v0.6\extensions\` (or `v0.7` for QuPath 0.7.0) |
| macOS | `~/Library/Application Support/QuPath/v0.6/extensions/` (or `v0.7`) |
| Linux | `~/.local/share/QuPath/v0.6/extensions/` (or `v0.7`) |

3. Restart QuPath

> **Tip:** In QuPath, **Edit > Preferences > Extensions** shows the extensions directory path. You can drag and drop the JAR there.

---

## Step 2: Set Up the Python Environment

On first launch after installing the extension, only **Setup DL Environment...** and the **Utilities** submenu will be visible in the menu.

1. Go to **Extensions > DL Pixel Classifier > Setup DL Environment...**
2. Review the download size warning (~2-4 GB)
3. Optionally uncheck **ONNX export support** (~200 MB savings) if you don't need it
4. Click **Begin Setup**
5. Wait for the download and configuration to complete (the dialog shows progress)
6. When complete, the dialog reports which GPU backend was detected (CUDA, MPS, or CPU)
7. Click **Close** when done

The training and inference menu items now appear automatically. On subsequent launches, the environment is detected on disk and everything is ready immediately.

> **After updating the extension:** If you install a new version, you **must** rebuild the Python environment. The extension enforces version matching -- an error notification will appear and training/inference will be blocked until the environment is rebuilt. Go to **Utilities > Rebuild DL Environment...** to update.

> **GPU not detected?** If the setup reports CPU-only but you have an NVIDIA GPU, make sure your NVIDIA drivers are installed and try **Utilities > Rebuild DL Environment...** See [Troubleshooting](docs/TROUBLESHOOTING.md) for details.

---

## Step 3: Train Your First Classifier

### 3a. Prepare annotations in QuPath

1. Open an image in QuPath
2. Create at least two annotation classes (e.g., right-click in the annotation class list to add "Foreground" and "Background")
3. Draw annotations on the image using the brush or polygon tools
4. Assign each annotation to a class (right-click the annotation > Set class)

> **Minimum requirement:** At least one annotation per class. More annotations = better results. Line/brush annotations work well -- you don't need to label every pixel.

### 3b. Open the training dialog

**Extensions > DL Pixel Classifier > Train Classifier...**

### 3c. Configure training

The dialog has collapsible sections. For a quick first test:

| Setting | Recommended first-run value |
|---------|-----------------------------|
| **Classifier Name** | `test_classifier_v1` |
| **Training Data Source** | Current image only (default) |
| **Architecture** | `unet` |
| **Backbone** | `resnet34` (or a histology backbone -- see below) |
| **Epochs** | `3` (just to verify it works) |
| **Tile Size** | `256` or `512` |
| **Weight Initialization** | Use pretrained backbone weights |

Leave everything else at defaults.

### 3d. Start training

Click **Start Training**. A progress window appears showing:
- Export progress (extracting patches from annotations)
- Training progress with epoch-by-epoch train loss and validation loss
- A live loss chart

A 3-epoch test run should complete in under a minute on GPU.

### 3e. Verify the result

When training completes, the classifier is saved to your QuPath project under `classifiers/`. You can see it via **Extensions > DL Pixel Classifier > Manage Models...**

---

## Step 4: Apply the Classifier

1. Open an image (same one or a different one)
2. Create annotation(s) around the region(s) you want to classify
3. **Extensions > DL Pixel Classifier > Apply Classifier...**
4. Select your trained classifier
5. Choose an output type:
   - **Rendered Overlay** -- batch inference with seamless blending (recommended for quality validation)
   - **Measurements** -- adds class probabilities as annotation measurements
   - **Objects** -- creates detection objects from the classification map
   - **Overlay** -- renders a live on-demand color overlay as you pan and zoom
6. Click **Apply**

---

## Retraining from a Previous Model

To iterate on a model with updated annotations or adjusted settings:

1. Open the training dialog (**Train Classifier...**)
2. In the **WEIGHT INITIALIZATION** section, select **"Continue training from saved model"**
3. Click **"Select model..."** and choose the model you want to build on
4. All parameters (architecture, learning rate, augmentation, etc.) are pre-filled
5. Adjust any settings as needed
6. Select images and load classes -- classes matching the source model are auto-selected
7. Click **Start Training**

The new classifier is saved separately; the original model is not modified.

---

## Multi-Image Training

To train from annotations across multiple project images:

1. Open a QuPath project with multiple annotated images
2. Open the training dialog (**Train Classifier...**)
3. Under **TRAINING DATA SOURCE**, check images to include (only images with classified annotations appear)
4. Click **"Load Classes from Selected Images"**
5. Train as usual -- patches from all selected images are combined into one training set

---

## Viewing Python Logs

The **Python Console** shows all Python-side output, including GPU initialization, model loading, and error messages. This is the primary diagnostic tool for troubleshooting.

1. Go to **Extensions > DL Pixel Classifier > Utilities > Python Console**
2. The console displays all Python stderr output in real time
3. Use **Copy to Clipboard** to capture logs for bug reports

The Python Console is especially useful for:
- Confirming GPU detection ("CUDA available: True" or "MPS available: True")
- Diagnosing model loading errors
- Viewing training/inference progress details from the Python side

> **See also:** **Extensions > DL Pixel Classifier > Utilities > System Info** provides a full diagnostic dump (PyTorch version, CUDA version, GPU info, package versions) with a **Copy to Clipboard** button for sharing in bug reports.

---

## Troubleshooting Quick Reference

| Problem | Quick fix |
|---------|-----------|
| Setup fails or stalls | Check internet; try **Utilities > Rebuild DL Environment...** |
| GPU not detected | Install NVIDIA drivers first, then rebuild environment |
| Training fails immediately | Verify annotations have assigned classes; check QuPath log |
| Out of memory | Reduce batch size (4 or 2) or tile size (256) |
| Menu items don't appear | Complete Setup DL Environment; restart QuPath |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for the full troubleshooting guide.

---

## Next Steps

- **Use separate projects for training and production** -- keep your annotated training images in one project and apply the finished classifier in a separate project for analysis. This protects your training data from being cluttered with generated objects. See [Best Practices: Training vs. Production Projects](docs/BEST_PRACTICES.md#training-vs-production-projects) for details.
- **Increase epochs** once you've verified the pipeline works (50-100 for real training)
- **Try histology-pretrained backbones** -- select a backbone ending in "(Histology)" for weights pretrained on tissue patches instead of ImageNet. These produce better features for tissue classification and need less layer freezing. ~100MB download on first use (cached afterward).
- **Try foundation model encoders** -- select h-optimus-0, virchow, hibou-l, hibou-b, midnight, or dinov2-large in the encoder dropdown for large-scale pretrained tissue representations. Downloaded on-demand (~100 MB to ~2 GB, cached after first use). Requires 10-16 GB VRAM for most models. Gated models need a HuggingFace token. See the [Training Guide](docs/TRAINING_GUIDE.md) for details.
- **Try transfer learning** -- freeze early encoder layers for faster convergence on small datasets
- **Experiment with backbones** -- try a larger backbone (resnet50) or a histology-pretrained backbone for tissue classification, or import a custom ONNX model
- **Multi-image training** -- combine annotations from several images for a more robust classifier
- **Tune training strategy** -- expand the "TRAINING STRATEGY" section in the training dialog to adjust the LR scheduler, loss function, early stopping metric/patience, and mixed precision
- **MAE pretraining** -- if using the MuViT architecture, pretrain the encoder on unlabeled tiles via **Utilities > MAE Pretrain Encoder...** for better domain-specific features. See the [Training Guide](docs/TRAINING_GUIDE.md#mae-pretraining-muvit-encoder) for details.
- See the [Training Guide](docs/TRAINING_GUIDE.md) and [Inference Guide](docs/INFERENCE_GUIDE.md) for detailed parameter explanations

---

## Building from Source

> This section is for **developers** contributing to the extension. End-users should download the pre-built JAR from GitHub Releases (see Step 1 above).

```bash
git clone https://github.com/uw-loci/qupath-extension-dl-pixel-classifier.git
cd qupath-extension-dl-pixel-classifier
./gradlew build
```

This produces a JAR file in `build/libs/`. Copy it to your QuPath extensions directory and restart QuPath.

See [docs/INSTALLATION.md](docs/INSTALLATION.md) Part 5 for full build instructions and testing details.
