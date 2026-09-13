# Installation Guide

Complete setup instructions for the DL Pixel Classifier extension.

## Supported Platforms

| Platform | GPU Backend | Training | Inference | Status |
|----------|-------------|----------|-----------|--------|
| **Windows 10/11 (64-bit)** | NVIDIA CUDA | Fast | Fast | **Recommended** |
| **Linux (64-bit)** | NVIDIA CUDA | Fast | Fast | Supported |
| **macOS Apple Silicon (M1/M2/M3/M4)** | MPS | Slow | Moderate | Supported |
| **Windows / Linux (no GPU)** | CPU | Very slow | Slow | Functional but impractical for training |
| ~~macOS Intel~~ | -- | -- | -- | **Not supported** |
| ~~32-bit systems~~ | -- | -- | -- | **Not supported** |

> **Intel Mac users:** This extension does not support Intel-based Macs (x86_64 macOS). The Python environment cannot be built on this platform. You will need a machine with an Apple Silicon Mac, Windows PC, or Linux system.

### NVIDIA Driver Requirements

The extension bundles **PyTorch with CUDA 12**. You do **not** need to install CUDA separately, PyTorch includes its own CUDA runtime. You **do** need NVIDIA drivers new enough to support CUDA 12:

| Requirement | Version |
|-------------|---------|
| Minimum NVIDIA driver (Linux) | >= 525.60.13 |
| Minimum NVIDIA driver (Windows) | >= 528.33 |

Run `nvidia-smi` in a terminal to check your driver version. Update from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) if needed.

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.6.0 or later (including 0.7.0) |
| GPU | NVIDIA GPU with CUDA recommended; Apple Silicon (MPS) also works; CPU fallback available |
| Internet | Required for first-time environment setup (~2-4 GB download) |

> **Note:** A separate Python installation is **not** required. The extension manages its own embedded Python environment via [Appose](https://github.com/apposed/appose).

## Part 1: Install the Extension

### Download the JAR

Download the latest release JAR from the [GitHub Releases](https://github.com/uw-loci/qupath-extension-dl-pixel-classifier/releases) page.

### Install the JAR

**Drag and drop** the JAR file directly onto the open QuPath window (easiest), or copy it manually to your QuPath extensions directory:

| OS | Typical extensions path |
|----|------------------------|
| Windows | `C:\Users\<you>\QuPath\v0.6\extensions\` (or `v0.7` for QuPath 0.7.0) |
| macOS | `~/Library/Application Support/QuPath/v0.6/extensions/` (or `v0.7`) |
| Linux | `~/.local/share/QuPath/v0.6/extensions/` (or `v0.7`) |

> **Tip:** In QuPath, **Edit > Preferences > Extensions** shows the extensions directory path. You can drag and drop the JAR there.

### Verify installation

Restart QuPath. You should see **Extensions > DL Pixel Classifier** in the menu bar. On first launch, only **Setup DL Environment...** and the **Utilities** submenu will be visible. This is normal. The training and inference menu items appear after the environment is set up.

## Part 2: Python Environment Setup (Appose, the default)

The extension uses [Appose](https://github.com/apposed/appose) to automatically manage an embedded Python environment. No manual Python setup is needed.

### First-time setup

1. Open QuPath with the extension installed
2. Go to **Extensions > DL Pixel Classifier**
3. Click **Setup DL Environment...**
4. Review the download size warning (~2-4 GB)
5. Optionally uncheck **ONNX export support** to reduce download size (~200 MB savings)
6. Click **Begin Setup**
7. Wait for the environment to download and configure (may take several minutes depending on connection speed)
8. When complete, click **Close**, the training and inference menu items will appear automatically

### What gets downloaded

The setup wizard uses [pixi](https://pixi.sh/) (via Appose) to create an isolated Python environment containing:

- Python 3.11
- PyTorch 2.1+ (with CUDA support on Windows/Linux)
- segmentation-models-pytorch
- NumPy, Pillow, scikit-image
- ONNX and ONNX Runtime (optional)

### Environment location

The environment is stored at:

| OS | Path |
|----|------|
| Windows | `C:\Users\<you>\.local\share\appose\dl-pixel-classifier\` |
| macOS | `~/.local/share/appose/dl-pixel-classifier/` |
| Linux | `~/.local/share/appose/dl-pixel-classifier/` |

### Rebuilding the environment

If the environment becomes corrupted, you want a fresh install, or you have installed a new version of the extension that requires updated Python packages:

1. Go to **Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment...**
2. Confirm the rebuild (this deletes the existing environment)
3. Complete the setup wizard again

## Part 3: GPU Configuration

### NVIDIA GPU (CUDA)

1. Install the latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
   - **Windows:** Install "Game Ready" or "Studio" drivers
   - **Linux:** Install via your distribution's package manager or NVIDIA's `.run` installer
2. **Important:** NVIDIA drivers must be installed **before** running the environment setup. If you installed drivers after setup, use **Utilities > Rebuild DL Environment...** to reinstall.

### Verifying GPU detection (Appose mode)

After completing the setup wizard, verify that the GPU was detected:

1. **Setup dialog completion message**: the dialog reports which GPU backend was found (CUDA, MPS, or CPU)
2. **Python Console**: go to **Extensions > DL Pixel Classifier > Utilities > Python Console** and look for:
   - `CUDA available: True` (NVIDIA GPU)
   - `MPS available: True` (Apple Silicon)
3. **System Info**: go to **Extensions > DL Pixel Classifier > Utilities > System Info** for a full diagnostic dump including PyTorch version, CUDA version, and GPU details

### Apple Silicon (MPS)

MPS support is automatic on macOS with Apple Silicon (M-series chips). No additional configuration needed.

### CPU fallback

If no GPU is detected, the backend automatically falls back to CPU. Training will be slower but functional.

## Part 4: Verifying the Complete Setup

1. Open QuPath with the extension installed
2. You should see **Extensions > DL Pixel Classifier** in the menu bar
3. If this is first time: only **Setup DL Environment...** and the **Utilities** submenu are visible
4. After running setup: all workflow items (Train, Apply, Toggle Prediction Overlay, etc.) appear
5. Open the **Python Console** (Utilities menu) to verify GPU status

If issues occur, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Part 5: Building from Source (for Developers)

> This section is for **developers** who want to build the extension from source. End-users should download the pre-built JAR from GitHub Releases (see Part 1).

### Build the extension

```bash
git clone https://github.com/uw-loci/qupath-extension-dl-pixel-classifier.git
cd qupath-extension-dl-pixel-classifier
./gradlew build
```

This produces a JAR file in `build/libs/`. Copy it to your QuPath extensions directory and restart QuPath.

### Shadow JAR (bundled dependencies)

For a self-contained JAR that includes all dependencies:

```bash
./gradlew shadowJar
```

### Running Python tests

```bash
cd python_server
pip install -e ".[dev]"
pytest tests/ -v
```

### Java build requirements

- Java JDK 21+
- Gradle (wrapper included in the repository)

