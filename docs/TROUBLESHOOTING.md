# Troubleshooting

Common issues and solutions for the DL Pixel Classifier extension.

## Viewing Logs and Diagnostics

Before diving into specific issues, know where to find diagnostic information:

### QuPath log

**View > Show log** in QuPath displays Java-side messages from the extension, including backend initialization, error messages, and workflow progress.

### Python Console

**Extensions > DL Pixel Classifier > Utilities > Python Console** shows all Python-side output (stderr) in real time:

- GPU initialization messages (e.g., "CUDA available: True")
- Model loading and inference progress
- Python errors and stack traces
- Normalization statistics and processing details

Use **Copy to Clipboard** to capture the full log for bug reports.

> **Tip:** Keep the Python Console open during training and inference to monitor the Python backend's activity.

### System Info

**Extensions > DL Pixel Classifier > Utilities > System Info** provides a complete diagnostic dump:

- PyTorch version and build info
- CUDA version and GPU details (name, VRAM, driver version)
- Installed Python package versions
- Backend mode and environment status

Use **Copy to Clipboard** to share the full output when reporting bugs.

## Environment Setup Issues (Appose Mode)

### Setup dialog shows an error

| Symptom | Cause | Fix |
|---------|-------|-----|
| Network error during setup | No internet connection | Connect to the internet and click **Retry** |
| Download stalls or times out | Slow/unstable connection | Cancel and retry; the ~2-4 GB download may take several minutes |
| "Resource not found" | JAR may be corrupted | Re-download the JAR from GitHub Releases and reinstall |

### Setup completes but GPU not detected

If the setup wizard reports CPU-only but you have an NVIDIA GPU:

1. **Check the Python Console** -- look for "CUDA available: False"
2. **Verify NVIDIA drivers are installed** -- open a terminal and run `nvidia-smi`. If this fails, install drivers first.
3. **Drivers must be installed before environment setup** -- the Appose environment installs PyTorch with CUDA support, but it needs to detect your GPU drivers during setup
4. **If drivers were installed after setup**: Go to **Utilities > Rebuild DL Environment...** to delete and re-create the environment with GPU support
5. **Windows-specific**: Install "Game Ready" or "Studio" drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)

### Menu items don't appear after setup

1. Verify the environment directory exists:
   - Windows: `C:\Users\<you>\.local\share\appose\dl-pixel-classifier\.pixi\`
   - macOS/Linux: `~/.local/share/appose/dl-pixel-classifier/.pixi/`
2. Close and reopen QuPath
3. Check the QuPath log (**View > Show log**) for errors

### Version mismatch error dialog

If you see an error dialog saying the Python package version does not match the extension version:

1. This means the Java JAR is a different version than the installed Python environment
2. Training and inference are blocked until the versions match -- this prevents silent incompatibilities
3. **Fix:** Go to **Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment...** to delete and reinstall with the correct Python package version
4. This typically happens when you update the extension JAR without rebuilding the Python environment

### Environment not updating after installing a new JAR

When you install a new version of the extension, the Python environment should update automatically on the next **Setup DL Environment...** run. The extension compares the bundled `pixi.toml` against the on-disk copy and forces a full rebuild if they differ.

If the environment does not update (e.g., still using an old Python version):

1. **Re-run setup**: Go to **Extensions > DL Pixel Classifier > Setup DL Environment...** and click **Begin Setup**. This triggers the automatic update check.
2. **If that does not work**: Use **Utilities > Rebuild DL Environment...** to force a full delete and reinstall.
3. **If rebuild fails** (e.g., Windows file-locking): Close QuPath completely, then manually delete the environment directory:

| OS | Environment path |
|----|-----------------|
| Windows | `C:\Users\<you>\.local\share\appose\dl-pixel-classifier\` |
| macOS / Linux | `~/.local/share/appose/dl-pixel-classifier/` |

> **Tip:** This is the same path shown in the "Advanced Diagnostics" section at the bottom of this page.

Then reopen QuPath and run **Setup DL Environment...** again.

> **Why this happens:** The extension manages its own embedded Python environment via pixi. When the bundled `pixi.toml` changes (e.g., Python version bumped from 3.10 to 3.11, new packages added), the extension detects the change and deletes `pixi.lock` and `.pixi/` to force pixi to re-resolve everything. On Windows, if the old Python process is still running, file locks can prevent deletion -- the extension attempts a rename-fallback, but a full QuPath restart is sometimes needed.

### Environment seems corrupted

Use **Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment...** to delete the environment and re-run setup.

### Only "Setup DL Environment..." is visible

This is normal on first launch before the environment has been set up. Click it to begin the setup wizard.

## GPU Issues

### GPU not detected

1. Open the **Python Console** (Utilities menu) and look for "CUDA available" or "MPS available"
2. Run **System Info** (Utilities menu) and check the CUDA/GPU section
3. Verify NVIDIA drivers are installed by running `nvidia-smi` in a terminal
4. If drivers were installed **after** environment setup: use **Utilities > Rebuild DL Environment...** to reinstall
5. If still not detected: check that your NVIDIA driver's CUDA version is >= the PyTorch CUDA version shown in System Info
6. **Apple Silicon:** MPS should be auto-detected. If not, verify you are running macOS 12.3+ and check System Info output

### CUDA out of memory

| Fix | Description |
|-----|-------------|
| Reduce batch size | Try 4, then 2, then 1 |
| Reduce tile size | 256 instead of 512 |
| Use smaller backbone | efficientnet-b0 instead of resnet50 |
| Free GPU memory | Use the "Free GPU Memory" utility menu item |
| Close other GPU programs | Other applications may be using VRAM |
| Disable mixed precision | Rarely helps, but try if nothing else works |

### GPU memory not freed after training

Use **Extensions > DL Pixel Classifier > Utilities > Free GPU Memory** to force-clear all GPU state. This cancels running jobs, clears cached models, and frees VRAM.

## Training Issues

### Training fails immediately

| Symptom | Cause | Fix |
|---------|-------|-----|
| "No annotations found" | No classified annotations | Create annotations and assign them to classes |
| "At least 2 classes required" | Only one class annotated | Add annotations for a second class |
| "Server error" | Server connection failed | Check server is running and accessible |
| Dialog won't open | No image loaded | Open an image first |

### Training is very slow

| Cause | Fix |
|-------|-----|
| Running on CPU | Open the Python Console and look for "CUDA available: True". If False, install NVIDIA drivers and rebuild the environment. |
| Mixed precision disabled | Enable in Training Strategy section |
| Very large tile size | Reduce tile size (256 instead of 512) |
| Very large batch size | Reduce batch size |
| Too much augmentation | Disable elastic deformation (slowest augmentation) |

### Training appears to stall or hang

If training starts but stops making progress (no new epoch updates):

1. **Check the Python Console** for errors -- a GPU out-of-memory error may have occurred silently
2. **Check GPU utilization** -- run `nvidia-smi` in a terminal to see if GPU memory is fully consumed
3. **Kill QuPath and restart** if the process is deadlocked
4. Try reducing batch size or tile size to lower GPU memory usage

> **Note:** In versions before 0.3.5, a transient "thread death" error could cause the extension to retry training, creating two concurrent training processes on the same GPU that deadlock. This has been fixed -- training no longer retries on thread death errors.

### Training produces poor results

See [BEST_PRACTICES.md](BEST_PRACTICES.md) for detailed guidance on improving results.

Quick checklist:
- [ ] Annotations are accurate and consistent
- [ ] At least 2 classes with sufficient annotations
- [ ] Pretrained weights enabled
- [ ] Appropriate backbone for dataset size
- [ ] Augmentation enabled (at least flips and rotation)

## Inference Issues

### Inference produces blank/uniform results

- The classifier may not have trained well -- check training loss curves
- Channel configuration may not match training -- verify channel order and count
- Resolution (downsample) may differ from training

### Tile seams visible in output

Newly trained models use **BatchRenorm** normalization layers, which eliminate tiling artifacts caused by the neural network's internal normalization. Combined with image-level input normalization (enabled by default), this addresses both sources of tile boundary artifacts.

If seams are still visible:

- **Re-train the model** -- new models use BatchRenorm and save dataset normalization statistics, giving the best cross-tile consistency. Older models trained with standard BatchNorm are more susceptible to tiling artifacts.
- Increase tile overlap percentage (10-15% recommended)
- Use LINEAR or GAUSSIAN blend mode instead of NONE
- Verify overlap is not 0%
- For overlays, adjust the **Overlay Overlap (um)** preference (Edit > Preferences) to increase physical overlap distance

### Objects are fragmented or too small

- Increase hole filling threshold
- Decrease min object size threshold
- Increase boundary smoothing
- Consider using a larger tile size for more context

### Inference is very slow

- Enable GPU in processing options
- Reduce tile overlap (but quality may suffer)
- Use NONE blend mode for fastest processing (quality trade-off)
- Process selected annotations first to estimate total time

## Tile Evaluation Issues

### "Review Training Areas..." button doesn't appear

The button only appears when training completes successfully. It will not appear if:
- Training was cancelled
- Training failed with an error
- The training data path is no longer available

### Evaluation is slow

Tile evaluation runs the model over every training tile. Time depends on:
- Number of tiles (more annotations = more tiles)
- GPU vs CPU (GPU is much faster)
- Model complexity (larger backbones take longer)

The evaluation progress bar shows which tile is being processed. You can cancel the evaluation at any time.

### "Training tiles are cleaned up when this dialog closes"

This is expected behavior. Training tiles are stored in a temporary directory and are deleted when you close the progress dialog. To review training areas, click the button **before** closing the dialog.

If you need to re-evaluate, retrain the model -- the evaluation can only run while the training tiles still exist.

### Double-click navigation doesn't work

- Verify the image is still in the project
- For multi-image training, the dialog attempts to switch to the correct image automatically. If the image was renamed or removed from the project, navigation will fail.
- Check the QuPath log (**View > Show log**) for error messages

### High loss on most tiles

If nearly all tiles show high loss, the model likely did not train well:
- Check training loss curves -- did the model converge?
- Verify annotations are correct and consistent
- See [BEST_PRACTICES.md](BEST_PRACTICES.md#interpreting-tile-evaluation-results) for improvement strategies

## Extension Issues

### Menu items are hidden (not visible)

- **First launch:** Only **Setup DL Environment...** and **Utilities** are visible until you complete the environment setup
- **After setup completes:** All workflow items should appear. If not, restart QuPath

### Menu items are grayed out (visible but disabled)

Menu items like Train and Apply require an open image/project. Open an image first, then the menu items will become active.

### "No classifiers available"

- Train a classifier first, or check that classifiers are saved in the project's `classifiers/` directory
- Verify the backend is running and the model storage path is accessible

### Preferences not saving

Preferences are saved automatically when you click "Start Training" or "Apply" in the dialogs. They persist across QuPath sessions via QuPath's preference system. If preferences are not saving:
- Check QuPath's preferences directory is writable
- Verify no other QuPath instance is locking the preferences file

## Advanced Diagnostics

### Environment details

The Python environment is a self-contained installation managed by [pixi](https://pixi.sh/) via [Appose](https://github.com/apposed/appose):

| Item | Location |
|------|----------|
| Environment root | `~/.local/share/appose/dl-pixel-classifier/` (see OS-specific paths above) |
| pixi.toml | `<env root>/pixi.toml` -- defines all Python dependencies |
| pixi.lock | `<env root>/pixi.lock` -- resolved dependency versions |
| Python installation | `<env root>/.pixi/envs/default/` |

**When to rebuild vs fresh install:**
- **Rebuild** (Utilities > Rebuild DL Environment): Deletes the existing environment and runs setup again. Use when the environment is corrupted or after driver changes.
- **Fresh install**: Manually delete the entire environment directory, then run Setup DL Environment. Use as a last resort if Rebuild does not work.

### Known transient behaviors

- **Brief pause on first inference**: The model loads into GPU memory on the first inference call. Subsequent calls are faster.
- **Occasional "thread death" during overlay rendering**: This is a transient error that auto-retries. If the overlay eventually renders correctly, no action is needed.
- **Large initial memory spike during training**: PyTorch allocates GPU memory aggressively on the first batch, then stabilizes. Monitor GPU memory over several epochs before reducing batch size.

### Reporting bugs

When filing a bug report, please include:

1. **System Info output** -- Utilities > System Info > Copy to Clipboard
2. **Python Console log** -- Utilities > Python Console > Copy to Clipboard
3. **QuPath log** -- View > Show log (copy relevant error messages)
4. **Steps to reproduce** -- what you did, what you expected, what happened
5. **Image details** -- image type (brightfield/fluorescence), channel count, bit depth

File issues at the [GitHub repository](https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier/issues).

## Diagnostic Commands

### Built-in diagnostics

Use the built-in tools (no terminal required):

| Tool | Menu path | What it shows |
|------|-----------|---------------|
| **Python Console** | Utilities > Python Console | Real-time Python output, GPU init, errors |
| **System Info** | Utilities > System Info | PyTorch/CUDA versions, GPU details, packages |
| **QuPath log** | View > Show log | Java-side extension messages |

### Terminal GPU verification

If you suspect GPU driver issues, verify from a terminal:

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Check PyTorch GPU access (if Python is available)
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

> **Note:** The Python Console and System Info utilities provide the same information without requiring terminal access.
