# QuPath
# Deep Learning
# Pixel Classifier
# Extension

A QuPath extension for deep learning-based pixel classification, supporting both brightfield (RGB) and multi-channel fluorescence/spectral images.

## Features

- **Train custom classifiers** from annotated regions using sparse annotations
- **Multi-image training** from multiple project images in a single training run
- **Multi-channel support** with per-channel normalization
- **Real-time progress** with separate train/val loss charting
- **Multiple output types**: Measurements, detection objects, or classification overlays
- **Pixel-level inference** for OBJECTS and OVERLAY output types with full per-pixel probability maps
- **Image-level normalization** eliminates tile boundary artifacts by computing consistent statistics across the entire image
- **Load settings from a previous model** for quick retraining iterations with class auto-matching
- **Dialog preference persistence** -- training and inference settings are remembered across sessions
- **Combined CE + Dice loss** for improved segmentation quality (default)
- **IoU-based early stopping** monitors mean IoU instead of validation loss
- **Mixed precision training** (AMP) for ~2x speedup on CUDA GPUs
- **Configurable training strategy** via collapsed "Training Strategy" section in the training dialog (scheduler, loss function, early stopping metric/patience, mixed precision)
- **Histology-pretrained encoders** from TCGA/Lunit/Kather100K for better tissue feature extraction
- **MuViT (Multi-scale Vision Transformer)** architecture with multi-resolution feature fusion
- **MAE self-supervised pretraining** for MuViT encoders using unlabeled image tiles (standalone workflow via Utilities menu)
- **Pluggable architecture** supporting UNet, MuViT, and custom ONNX models
- **Appose shared-memory IPC** for embedded Python inference with zero-copy tile transfer
- **Groovy scripting API** for batch processing
- **Headless builder API** for running workflows without GUI
- **"Copy as Script" buttons** in dialogs for reproducible workflows
- **Post-training tile evaluation** identifies annotation errors and hard cases by running the model over all training tiles and ranking by loss
- **Hierarchical geometry union** for efficient ROI merging

## Installation

1. **Download** the latest JAR from the [GitHub Releases](https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier/releases) page
2. **Install** the JAR using one of these methods:
   - **Drag and drop** the JAR file directly onto the open QuPath window (easiest), or
   - **Copy** the JAR manually to your QuPath extensions directory:

| OS | Extensions path |
|----|----------------|
| Windows | `C:\Users\<you>\QuPath\v0.6\extensions\` (or `v0.7` for QuPath 0.7.0) |
| macOS | `~/Library/Application Support/QuPath/v0.6/extensions/` (or `v0.7`) |
| Linux | `~/.local/share/QuPath/v0.6/extensions/` (or `v0.7`) |

3. **Restart QuPath** -- the extension appears under **Extensions > DL Pixel Classifier**

> **Tip:** In QuPath, **Edit > Preferences > Extensions** shows the extensions directory path.

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed instructions and GPU configuration.

## Getting Started

1. **Set up the Python environment** -- go to **Extensions > DL Pixel Classifier > Setup DL Environment...** in the QuPath menu bar. This downloads and configures PyTorch, CUDA, and all dependencies automatically (~2-4 GB, first time only).

   > **After updating the extension:** If you install a new version of this extension, you **must** rebuild the Python environment to match. The extension enforces version matching and will block training/inference if the environment is out of date. An error notification will appear with instructions. Go to **Extensions > DL Pixel Classifier > Rebuild Python Environment** to update.

2. **Train a classifier** -- create annotations, open **Extensions > DL Pixel Classifier > Train Classifier...**, and click Start Training
3. **Apply the classifier** -- open **Extensions > DL Pixel Classifier > Apply Classifier...**, select a model, choose an output type, and click Apply

See [QUICKSTART.md](QUICKSTART.md) for a complete walkthrough (zero to classifier in ~10 minutes).

## Requirements

- QuPath 0.6.0+ (including 0.7.0)
- NVIDIA GPU with CUDA recommended (CPU and Apple Silicon MPS also supported)
- Internet connection for first-time environment setup (~2-4 GB download)

> **Note:** A separate Python installation is **not** required. The extension manages its own embedded Python environment via [Appose](https://github.com/apposed/appose).

## Documentation

| Guide | Description |
|-------|-------------|
| [Quickstart](QUICKSTART.md) | Zero-to-classifier in 10 minutes |
| [Installation](docs/INSTALLATION.md) | Full setup: extension install + Python environment + GPU configuration |
| [Training Guide](docs/TRAINING_GUIDE.md) | Step-by-step training workflow how-to |
| [Inference Guide](docs/INFERENCE_GUIDE.md) | Step-by-step inference workflow how-to |
| [Parameters](docs/PARAMETERS.md) | Every parameter with defaults, ranges, and ML guidance |
| [Scripting](docs/SCRIPTING.md) | Groovy API, builder pattern, batch processing, Copy as Script |
| [Best Practices](docs/BEST_PRACTICES.md) | Backbone selection, annotation strategy, hyperparameter tuning |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Environment issues, GPU issues, training/inference issues, diagnostics |
| [Preferences](docs/PREFERENCES.md) | All persistent preferences with defaults and keys |
| [Appose Dev Guide](docs/APPOSE_DEV_GUIDE.md) | Constraints and patterns for embedded Python scripts (developer reference) |
| [Python Library](python_server/README.md) | Python deep learning library (training, inference, models) |

## GPU Support

The extension automatically detects and uses available GPU hardware:

- **NVIDIA GPUs (CUDA)** -- auto-detected on Windows and Linux. Requires NVIDIA drivers to be installed.
- **Apple Silicon (MPS)** -- auto-detected on macOS with M-series chips.
- **CPU fallback** -- automatic when no GPU is available. Training will be slower but functional.

The setup wizard reports which GPU backend was detected at completion. See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) if GPU is not detected.

## Supported Image Types

| Image Type | Channels | Bit Depth | Strategy |
|------------|----------|-----------|----------|
| Brightfield RGB | 3 | 8-bit | Direct input |
| Immunofluorescence | 2-8+ | 8/12/16-bit | Channel selection + normalization |
| Spectral/Hyperspectral | 10-100+ | 16-bit | Channel grouping |

## Architecture

```
qupath-extension-DL-pixel-classifier/
├── src/main/java/qupath/ext/dlclassifier/
│   ├── SetupDLClassifier.java        # Extension entry point & menu management
│   ├── DLClassifierChecks.java       # Startup validation
│   ├── classifier/                    # Classifier type system
│   │   ├── ClassifierHandler.java
│   │   ├── ClassifierRegistry.java
│   │   └── handlers/
│   │       ├── UNetHandler.java
│   │       ├── MuViTHandler.java
│   │       └── CustomONNXHandler.java
│   ├── controller/                    # Workflow orchestration
│   │   ├── DLClassifierController.java
│   │   ├── TrainingWorkflow.java
│   │   ├── InferenceWorkflow.java
│   │   └── ModelManagementWorkflow.java
│   ├── service/                       # Backend services
│   │   ├── ApposeService.java        # Appose embedded Python management
│   │   ├── ApposeClassifierBackend.java  # Appose backend implementation
│   │   ├── ClassifierBackend.java    # Backend interface
│   │   ├── ClassifierClient.java     # Data records and utilities
│   │   ├── BackendFactory.java       # Backend initialization
│   │   ├── DLPixelClassifier.java    # QuPath PixelClassifier integration
│   │   ├── ModelManager.java
│   │   └── OverlayService.java
│   ├── model/                         # Data objects
│   │   ├── TrainingConfig.java
│   │   ├── InferenceConfig.java
│   │   ├── ChannelConfiguration.java
│   │   └── ClassifierMetadata.java
│   ├── utilities/                     # Processing utilities
│   │   ├── AnnotationExtractor.java  # Training data export (single + multi-image)
│   │   ├── TileProcessor.java
│   │   ├── ChannelNormalizer.java
│   │   ├── BitDepthConverter.java
│   │   └── OutputGenerator.java
│   ├── ui/                            # UI components
│   │   ├── TrainingDialog.java
│   │   ├── InferenceDialog.java
│   │   ├── SetupEnvironmentDialog.java   # First-time setup wizard
│   │   ├── ChannelSelectionPanel.java
│   │   ├── LayerFreezePanel.java
│   │   ├── ProgressMonitorController.java
│   │   ├── PythonConsoleWindow.java
│   │   ├── TrainingAreaIssuesDialog.java  # Post-training tile evaluation results
│   │   ├── MAEPretrainingDialog.java     # Standalone MAE pretraining config
│   │   └── TooltipHelper.java
│   ├── scripting/
│   │   ├── DLClassifierScripts.java   # Groovy API
│   │   └── ScriptGenerator.java       # Dialog-to-script generation
│   └── preferences/
│       └── DLClassifierPreferences.java
│
├── python_server/                     # Python DL library (used by Appose)
│   └── dlclassifier_server/
│       ├── services/                  # Training/inference services
│       └── utils/                     # Shared utilities (normalization, etc.)
```

## For Developers

### Building from source

```bash
git clone https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier.git
cd qupath-extension-DL-pixel-classifier
./gradlew build
```

This produces a JAR file in `build/libs/`. Copy it to your QuPath extensions directory and restart QuPath.

For a shadow JAR that bundles all dependencies:

```bash
./gradlew shadowJar
```

### Running Python tests

```bash
cd python_server
pip install -e ".[dev]"
pytest tests/ -v
```

Current status: **78 tests passing, 5 skipped**

## Acknowledgements

This extension builds on many excellent open-source projects and pretrained models.

### Segmentation Framework

- **[segmentation-models-pytorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch)** (MIT) -- Pavel Iakubovskii. Provides all segmentation architectures (U-Net, U-Net++, DeepLab V3/V3+, FPN, PSPNet, MA-Net, LinkNet) and encoder backbone integration.
- **[PyTorch](https://pytorch.org/)** (BSD) -- Meta AI. Deep learning framework used for training and inference.
- **[TorchVision](https://pytorch.org/vision/)** (BSD) -- PyTorch team. Provides ImageNet-pretrained backbone weights (ResNet, EfficientNet, DenseNet, MobileNet, VGG, SE-ResNet).

### Segmentation Architectures (via SMP)

| Architecture | Original Paper |
|---|---|
| U-Net | Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015) |
| U-Net++ | Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (DLMIA 2018) |
| DeepLab V3 | Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017) |
| DeepLab V3+ | Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (ECCV 2018) |
| FPN | Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017) |
| PSPNet | Zhao et al., "Pyramid Scene Parsing Network" (CVPR 2017) |
| MA-Net | Fan et al., "MA-Net: A Multi-Scale Attention Network for Liver and Tumor Segmentation" (IEEE Access 2020) |
| LinkNet | Chaurasia & Culurciello, "LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation" (VCIP 2017) |

### Encoder Backbones (via SMP + TorchVision)

| Backbone | Original Paper |
|---|---|
| ResNet | He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016) |
| EfficientNet | Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019) |
| SE-ResNet | Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018) |
| DenseNet | Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017) |
| MobileNet V2 | Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018) |
| VGG | Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (ICLR 2015) |

### Histology-Pretrained Encoders

Histology-specific encoder weights are loaded via **[timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)** (Apache 2.0) from the **[HuggingFace Model Hub](https://huggingface.co/)**, using weights curated by [1aurent](https://huggingface.co/1aurent):

| Encoder | Training Data | Method | License | Reference |
|---|---|---|---|---|
| ResNet-50 Lunit SwAV | 19M TCGA patches | SwAV self-supervised | Non-commercial | Kang et al., "Benchmarking Self-Supervised Learning on Diverse Pathology Datasets" (2023) |
| ResNet-50 Lunit Barlow Twins | 19M TCGA patches | Barlow Twins self-supervised | Non-commercial | Kang et al. (2023) |
| ResNet-50 Kather100K | Kather100K colorectal tissue | Supervised classification | CC-BY-4.0 | Kather et al., "Predicting survival from colorectal cancer histology slides using deep learning" (PLOS Medicine 2019) |
| ResNet-50 TCGA-BRCA | TCGA breast cancer | SimCLR self-supervised | GPLv3 | Ciga et al., "Self supervised contrastive learning for digital histopathology" (Machine Learning with Applications 2022) |

### Data Augmentation

- **[Albumentations](https://github.com/albumentations-team/albumentations)** (MIT) -- Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations" (Information 2020). Used for training-time augmentations (flips, rotations, elastic deformations, color jitter, noise).

### Model Export and Inference

- **[ONNX](https://onnx.ai/)** (Apache 2.0) -- Open Neural Network Exchange format for model serialization.
- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** (MIT) -- Microsoft. High-performance inference engine with CUDA, CoreML, and CPU backends.

### Java-Python Integration

- **[Appose](https://github.com/apposed/appose)** (Apache 2.0) -- SciJava. Shared-memory IPC framework for embedded Python execution with zero-copy tile transfer.

### Image I/O

- **[tifffile](https://github.com/cgohlke/tifffile)** (BSD) -- Christoph Gohlke. Multi-channel TIFF reader/writer for microscopy images.
- **[imagecodecs](https://github.com/cgohlke/imagecodecs)** (BSD) -- Christoph Gohlke. Image compression codec library.
- **[Pillow](https://python-pillow.org/)** (HPND) -- PIL fork for standard image format support.
- **[scikit-image](https://scikit-image.org/)** (BSD) -- Image processing algorithms.

### Host Application

- **[QuPath](https://qupath.github.io/)** (GPL-3.0) -- Bankhead et al., "QuPath: Open source software for digital pathology image analysis" (Scientific Reports 2017). The digital pathology platform this extension integrates with.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please open issues or pull requests on GitHub.

## AI-Assisted Development

This project was developed with assistance from [Claude](https://claude.ai) (Anthropic). Claude was used as a development tool for code generation, architecture design, debugging, and documentation throughout the project.
