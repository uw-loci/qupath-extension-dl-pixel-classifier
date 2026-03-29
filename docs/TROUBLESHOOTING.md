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

### Saved log files

Python worker output is automatically saved to your project directory during training:

```
{project}/logs/dl-pixel-classifier/session_20260328_143052.log
```

Each training session creates a new timestamped log file. These persist after QuPath is closed, so you can review them later for debugging or share them in bug reports. The log contains all Python-side output: model creation, checkpoint loading, per-epoch progress, errors, and warnings.

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

### What can I do if training is interrupted?

If training is interrupted -- whether by a crash, power outage, or accidental QuPath close -- your progress is automatically saved. The extension writes a full checkpoint to disk every time a new best epoch is found during training.

#### Automatic crash-recovery checkpoint

During training, the file `best_in_progress_{model_type}.pt` is saved to the checkpoints directory whenever validation metrics improve:

| OS | Checkpoint path |
|----|----------------|
| Windows | `C:\Users\<you>\.dlclassifier\checkpoints\` |
| macOS / Linux | `~/.dlclassifier/checkpoints/` |

This file contains everything needed to either **recover the trained model** or **resume training**:

- Best model weights (from the epoch with the highest validation score)
- Optimizer state (learning rate, momentum, adaptive parameters)
- Scheduler state (learning rate schedule position)
- Early stopping state (patience counter, best score)
- Full training history (per-epoch metrics)
- Model configuration (architecture, backbone, classes, input config)

The checkpoint is saved in the same format as the pause/resume checkpoint, so all recovery methods work identically.

#### Option 1: Recover the best model (no further training needed)

If the model had already converged before the interruption, you can finalize it into a usable classifier:

1. Open QuPath with the DL Pixel Classifier extension
2. Open the **Python Console** (Utilities menu)
3. Use the `finalize_training.py` script with the checkpoint path:

The finalize script loads the checkpoint, extracts the best model weights, and saves a complete classifier (model.pt + metadata.json + ONNX export) that you can use for inference.

> **Tip:** Check the QuPath log from your interrupted session -- it logs the checkpoint path each time a new best is saved:
> `Best checkpoint saved to disk (epoch 42): C:\Users\you\.dlclassifier\checkpoints\best_in_progress_unet.pt`

#### Option 2: Resume training from the checkpoint

If you want to continue training from where it left off:

1. Open the training dialog (**Extensions > DL Pixel Classifier > Train DL Pixel Classifier**)
2. Select **Continue from checkpoint** in the Weight Initialization section
3. Browse to the `best_in_progress_{model_type}.pt` file
4. Training resumes from the last best epoch with all optimizer state intact

> **Note:** Training resumes from the epoch where the best model was saved, not from the epoch where the crash occurred. For example, if the best model was at epoch 30 and the crash happened at epoch 45, training resumes from epoch 30. Epochs 31-45 are re-trained, but the model's optimizer and scheduler state are preserved, so results should be similar.

#### Option 3: Use the Pause feature to save progress intentionally

If you anticipate needing to stop training (e.g., end of day, shared workstation):

1. Click **Pause** in the training progress dialog
2. Training saves a full checkpoint and stops
3. You can then:
   - **Resume** to continue training immediately
   - **Complete Training** to finalize the best model without further training
   - **Close** to stop and come back later (checkpoint persists on disk and in memory)

The pause checkpoint is stored both in memory (for immediate resume within the same QuPath session) and on disk (for recovery after restart).

#### What is NOT recoverable

| Scenario | Model recovery | Resume training |
|----------|---------------|-----------------|
| Crash after at least one best epoch saved | Yes | Yes (from last best epoch) |
| Crash before any validation epoch completes | No | No |
| Power outage during checkpoint write | Possibly corrupted | Possibly corrupted |
| Pause, then cancel, same QuPath session | Yes | Yes |
| Pause, then cancel, after QuPath restart | Yes (from .pt file) | Yes (load .pt manually) |

> **Tip:** The best-in-progress checkpoint is automatically cleaned up when training completes normally or is cancelled with a save. If you see a `best_in_progress_*.pt` file in the checkpoints directory, it means a previous training was interrupted and can be recovered.

### Training produces poor results

See [BEST_PRACTICES.md](BEST_PRACTICES.md) for detailed guidance on improving results.

Quick checklist:
- [ ] Annotations are accurate and consistent
- [ ] At least 2 classes with sufficient annotations
- [ ] Pretrained weights enabled
- [ ] Appropriate backbone for dataset size
- [ ] Augmentation enabled (at least flips and rotation)

### One class has wildly inconsistent IoU or loss across epochs

**Symptom:** A class oscillates between good IoU (e.g., 0.85) and terrible IoU (0.00 or 0.06) from one epoch to the next, while other classes are stable. The per-class loss for that class may spike to a consistent high value (e.g., always ~6.2) on bad epochs.

**Cause: Rare or localized annotations landing entirely in the validation set.** The train/validation split assigns whole annotations to one side or the other. If a class has only one annotation, or one annotation that looks visually distinct from the others, all tiles from that annotation may end up in validation. The model never trains on those tiles but gets penalized for them at every epoch.

This is especially common with:
- **Background/ignore classes** that include both typical regions (e.g., white slide) and atypical regions (e.g., dark debris, ink marks, tissue folds)
- **Rare structures** that only appear on one or two slides
- **Edge cases** where the class looks very different in one region vs. another

**How to identify it:** Look for a suspiciously consistent per-class loss value appearing at roughly half the epochs. A loss that's always ~6.2 (or any specific repeating value) on "bad" epochs points to one specific validation tile or region that the model consistently misclassifies.

**Fixes:**

1. **Break large annotations into smaller pieces.** The stratified splitter assigns whole annotations to train or val. One large annotation = all-or-nothing. Three smaller annotations = the splitter is more likely to place some in train and some in val, so the model learns the pattern.

2. **Add annotations of the rare pattern to multiple images.** If dark debris exists on one slide, annotate it on several images. Even small annotations ensure the pattern appears in training tiles across the split.

3. **Consider a separate class for visually distinct patterns.** If "Ignore" includes both white background and black debris, the model must learn that visually opposite things are the same class. A 4th class ("Artifact" or "Debris") lets the model learn each pattern independently. You can merge classes at analysis time.

4. **Check what's in the validation set.** After a short training run (5-10 epochs), use **Review Training Areas** to see which tiles have the highest loss. Double-click to navigate to them. If they're all from one annotation, that annotation is likely in the validation set and underrepresented in training.

> **Note:** The random train/val split changes each time you start training or resume with re-exported data. A class that's problematic in one run may be fine in the next (different split), which makes this issue intermittent and hard to diagnose without the per-class loss breakdown.

### Validation loss spikes to extreme values

**Symptom:** `val_loss` shows values in the hundreds or thousands, while the individual per-class losses sum to much less than the reported total.

**Cause:** Extreme logit magnitudes from pretrained or continued-training models. When model outputs become very large, `cross_entropy(-log(softmax(x)))` produces huge per-pixel losses even in FP32 precision. This was fixed in v0.4.2+ by clamping logits to the range [-50, 50] before loss computation.

**Fix:** Update to v0.4.7+ and use **Utilities > Rebuild DL Environment...** to ensure the Python environment matches the latest version.

### Continue-training produces worse results than original

**Symptom:** Continuing training from a saved model oscillates wildly or diverges instead of improving.

**Common causes:**

- **Learning rate too high for continue-training.** Use `0.0001`, not `0.001`. The model is already near a minimum, so a large learning rate pushes it out of the basin.
- **`loadSettingsFromModel` may restore the OLD learning rate from the saved model.** Always verify that the LR spinner shows `0.00010` before clicking Train. If the spinner shows `0.00100`, the old model's LR was loaded -- change it manually.
- **Frozen layers, discriminative LRs, and scheduler settings are now preserved** (v0.4.0+). If you are on an older version, these settings may be lost when loading a model.
- **Version mismatch between the pip package and the JAR.** Check for the version mismatch notification dialog on startup. If the Python environment is out of date, use **Utilities > Rebuild DL Environment...** to update it.

## Inference Issues

### Inference produces blank/uniform results

- The classifier may not have trained well -- check training loss curves
- Channel configuration may not match training -- verify channel order and count
- Resolution (downsample) may differ from training

### Tile seams visible in output

Both the overlay and Apply Classifier (OBJECTS) use the same unified inference pipeline, so they should produce identical results. The pipeline reads expanded tile regions from the actual image (real context, not reflection padding), center-crops to the stride region, and applies Gaussian smoothing. This eliminates most tile boundary artifacts.

If seams are still visible:

- **Re-train the model** -- new models use BatchRenorm and save dataset normalization statistics, giving the best cross-tile consistency. Older models trained with standard BatchNorm are more susceptible to tiling artifacts.
- Increase the **Overlay Smoothing** sigma (Edit > Preferences or Overlay Settings) -- higher values smooth noisy per-pixel predictions
- The tile overlap is enforced automatically (minimum 25% per side) -- manually increasing it beyond the default has diminishing returns

### Objects don't match the overlay

The overlay and Apply Classifier (OBJECTS) use the exact same `DLPixelClassifier.applyClassification()` method. If objects appear different from the overlay:

- **Check application scope**: The overlay covers the entire image. Apply Classifier processes only within the selected annotation(s). Make sure the annotation covers the area you expect.
- **Annotation shape clipping**: Objects are clipped to the parent annotation's geometric shape (not just its bounding box). An elliptical annotation will produce elliptically-bounded objects.
- Objects are vectorized from the pixel classification via contour tracing, so very thin features or single-pixel classifications may not produce visible objects. Adjust **Min Object Size** if small features are missing.

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

File issues at the [GitHub repository](https://github.com/uw-loci/qupath-extension-dl-pixel-classifier/issues).

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
