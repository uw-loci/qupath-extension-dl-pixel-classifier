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

The GPU environment includes **PyTorch with CUDA 12**. You do **not** need to install CUDA separately, PyTorch includes its own CUDA runtime. You **do** need NVIDIA drivers new enough to support CUDA 12:

| Requirement | Version |
|-------------|---------|
| Minimum NVIDIA driver (Linux) | >= 525.60.13 |
| Minimum NVIDIA driver (Windows) | >= 528.33 |

To check your driver version:

1. **Open a terminal** (on Windows, **Command Prompt**).
2. **Type `nvidia-smi` and press Enter.**
3. **Read Driver Version** in the top line of the output.

If it is too old, update it from [NVIDIA Driver Downloads](https://www.nvidia.com/en-us/drivers/).

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.7.0 or later |
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
| Windows | `C:\Users\<you>\QuPath\v0.7\extensions\` |
| macOS | `~/Library/Application Support/QuPath/v0.7/extensions/` |
| Linux | `~/.local/share/QuPath/v0.7/extensions/` |

> **Tip:** In QuPath, **Edit > Preferences > Extensions** shows the extensions directory path. You can drag and drop the JAR there.

### Verify installation

Restart QuPath. You should see **Extensions > DL Pixel Classifier** in the menu bar. On first launch the menu shows **Setup DL Environment...** at the top; the training and inference items are visible but grayed out until the environment is set up.

## Part 2: Set up the Python environment

The extension uses [Appose](https://github.com/apposed/appose) to automatically manage an embedded Python environment. No manual Python setup is needed.

### First-time setup

1. Open QuPath with the extension installed.
2. **Click Extensions > DL Pixel Classifier > Setup DL Environment...**
3. Read the download note (about 2-4 GB).
4. Optional: under **Environment location**, click **Change...** if your home folder has a storage quota (see [Environment location](#environment-location)).
5. Under **Compute environment**, choose **GPU / CUDA (requires an NVIDIA GPU)** or **CPU (works everywhere)**. GPU is preselected when an NVIDIA card is detected. On a Mac, choose CPU.
6. **Click Begin Setup.**
7. Wait while the environment downloads and installs; the status line shows progress. This may take several minutes depending on connection speed.
8. When it reads **Environment setup complete!**, **click Close**. The training and inference menu items become available.

If the GPU environment cannot be installed, setup installs the CPU environment instead and says so.

### What gets downloaded

The setup wizard uses [pixi](https://pixi.prefix.dev/) (via Appose) to create an isolated Python environment containing:

- Python 3.11
- PyTorch 2.1+ (the CUDA build in the GPU environment, the CPU build in the CPU environment)
- segmentation-models-pytorch
- NumPy, Pillow, scikit-image
- ONNX and ONNX Runtime

### Environment location

The setup wizard shows the full path it will build to before anything is downloaded, and marks it **[already built]** when an environment is already there. **Change...** picks a different folder and **Use default** puts it back. Choose another location when your home directory is quota-limited, as it often is on HPC systems and managed desktops, where an environment this size fails with "Quota exceeded". To change it later, use the **Python environment location** preference.

By default the environment is stored at:

| OS | Path |
|----|------|
| Windows | `C:\Users\<you>\.local\share\appose\dl-pixel-classifier\` |
| macOS | `~/.local/share/appose/dl-pixel-classifier/` |
| Linux | `~/.local/share/appose/dl-pixel-classifier/` |

The GPU environment is named `dl-pixel-classifier-gpu` instead, so the CPU and GPU environments can be installed side by side.

### Rebuilding the environment

After you update the extension, the environment has to match the new version. By default the extension rebuilds it in the background on the first start (the workflow menu items come back when it finishes). If you turned off **Auto-Rebuild Environment on Update**, if the automatic rebuild fails, or if the environment is corrupted:

1. **Click Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment...**
2. Confirm the rebuild (this deletes the existing environment).
3. Complete the setup wizard again.

## Part 3: GPU Configuration

### NVIDIA GPU (CUDA)

1. Install the latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/)
   - **Windows:** Install "Game Ready" or "Studio" drivers
   - **Linux:** Install via your distribution's package manager or NVIDIA's `.run` installer
2. **Important:** install the NVIDIA drivers **before** running the environment setup, so the GPU environment can be chosen. If you installed drivers after setup, use **Utilities > Compute Environment (CPU / GPU)...** and switch to GPU. Your trained classifiers and settings are not affected.

### Check that your GPU was found

After completing the setup wizard:

1. **Setup dialog completion message**: if the CPU environment was installed, the message says so and offers **Switch to GPU and rebuild...**; if the GPU environment was installed but cannot use CUDA, it warns you and offers **Switch to CPU and rebuild...**.
2. **System Info**: go to **Extensions > DL Pixel Classifier > Utilities > System Info...** for a full diagnostic dump including PyTorch version, CUDA version, and GPU details. Look for:
   - `CUDA available: True` (NVIDIA GPU)
   - `MPS available: True` (Apple Silicon)
3. **QuPath log** (**View > Show log**): when the Python environment starts, the extension logs a `DL Pixel Classifier Environment` block with the same `CUDA available` and `MPS available` lines.

### Apple Silicon (MPS)

Choose the CPU environment in setup; the GPU option is CUDA and needs an NVIDIA card. PyTorch then uses the Apple GPU (MPS) when it is available. No additional configuration needed.

### CPU fallback

If no GPU is detected, the backend automatically falls back to CPU. Training will be slower but functional.

## Part 4: Verifying the Complete Setup

1. Open QuPath with the extension installed
2. You should see **Extensions > DL Pixel Classifier** in the menu bar
3. If this is first time: **Setup DL Environment...** is at the top and the workflow items are grayed out
4. After running setup: the workflow items (Train, Apply, Toggle Prediction Overlay, etc.) are enabled. Train needs an open project and Apply needs an open image.
5. Open **Utilities > System Info...** to check GPU status

If issues occur, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Part 5: Building from Source (for Developers)

> This section is for **developers** who want to build the extension from source. End-users should download the pre-built JAR from GitHub Releases (see Part 1).

### Build the extension

```bash
git clone https://github.com/uw-loci/qupath-extension-dl-pixel-classifier.git
cd qupath-extension-dl-pixel-classifier
./gradlew shadowJar
```

This produces `build/libs/qupath-extension-dl-pixel-classifier-<version>-all.jar`, with all dependencies bundled. Copy **the `-all.jar`** to your QuPath extensions directory and restart QuPath. It is also the only jar published as a release asset.

Do not use the plain `.jar` that `./gradlew build` also leaves in `build/libs/`: it lacks the bundled dependencies (Appose) and will not load in QuPath.

### Running Python tests

```bash
cd python_server
pip install -e ".[dev]"
pytest tests/ -v
```

### Java build requirements

- Java JDK 21+
- Gradle (wrapper included in the repository)
