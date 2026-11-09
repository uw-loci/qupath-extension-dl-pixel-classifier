# Preferences Reference

All persistent preferences for the DL Pixel Classifier extension, their defaults, and where they appear.

## How Preferences Work

QuPath saves these for your user account, and they survive restarts.

- **Dialog preferences** are saved automatically when you click "Start Training" or "Apply"
- **Extension preferences** are editable in Edit > Preferences under "DL Pixel Classifier"
- Keys are prefixed with `dlclassifier.`, except the GPU device choice, `appose.gpuDevice`, which every Appose extension shares

## Extension Preferences

These appear in **Edit > Preferences > DL Pixel Classifier**.

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Python environment location | `dlclassifier.env.baseDir` | `""` | Folder the Python environment is built under. Blank means the default (`~/.local/share/appose`). Set it when your home folder has a storage quota. Takes effect on the next environment build; use **Utilities > Rebuild DL Environment...** to move an existing environment. |
| Python environment: compute variant | `dlclassifier.env.variant` | `CPU` | Which environment to install: `CPU` (installs anywhere) or `GPU` (CUDA; needs an NVIDIA GPU). Switching builds a separate environment, a download of several GB. |
| Python environment: GPU device (all Appose extensions) | `appose.gpuDevice` | `auto` | Which NVIDIA card the Python worker uses on a machine with more than one. `auto` picks the card with the most free memory when the first worker starts; a number is the index `nvidia-smi` shows. Shared with the other Appose extensions; restart QuPath to apply it to a running worker. |
| Use GPU for Inference | `dlclassifier.useGPU` | `true` | Also set by the **Use GPU if available** box in the Apply dialog. In this release inference uses the GPU whenever the environment can, whatever this says. |
| Training Data Export Directory | `dlclassifier.trainingExportDir` | `""` | Directory for exporting training data patches. Empty = use temp directory. |
| Auto-Rebuild Environment on Update | `dlclassifier.autoRebuildEnvironment` | `true` | Automatically rebuild the Python environment when the JAR version changes. Menu items are temporarily disabled during rebuild. |
| Overlay Prediction Smoothing | `dlclassifier.overlaySmoothing` | `2.0` | Gaussian sigma, in the model's output pixels, applied to the class probabilities before each pixel takes its class. Despite the name it affects the overlay **and** Apply DL Pixel Classifier (OBJECTS and MEASUREMENTS), so changing it changes measured areas; use one value for images you intend to compare. 0 = off, 1-2 = light, 3-5 = moderate. |
| Overlay High-Quality Tile Averaging | `dlclassifier.multiPassAveraging` | `false` | Runs each tile at 4 offsets and averages the predictions. Applies to the overlay and to Apply DL Pixel Classifier. About 4x slower. |
| Overlay Fast Argmax (uint8) | `dlclassifier.useCompactArgmaxOutput` | `false` | See [Experimental inference providers](#experimental-inference-providers). |
| Experimental: TensorRT Inference | `dlclassifier.experimentalTensorRT` | `false` | See [Experimental inference providers](#experimental-inference-providers). |
| Experimental: INT8 Quantization | `dlclassifier.experimentalInt8` | `false` | See [Experimental inference providers](#experimental-inference-providers). |
| Training: Pre-Load Patches Into RAM | `dlclassifier.defaultInMemoryDataset` | `auto` | Starting value for the Train dialog's **In-memory dataset** option: `auto` (preload when the patches fit in about half of the free RAM), `on`, `off` or `bounded` (train on a random subset that fits in RAM). |
| Training: Bounded Cache Fraction | `dlclassifier.cacheBoundedFraction` | `0.40` | With `bounded`, the share of available RAM the cache may use. |
| Out-of-Distribution Warning Threshold (sigma) | `dlclassifier.oodSigmaThreshold` | `3.0` | Before inference, the image's per-channel mean, 1st and 99th percentiles are compared with the training images recorded in the model. A channel further than this many standard deviations away raises a warning. Lower = more warnings. |
| Training: DataLoader Workers (experimental) | `dlclassifier.defaultDataLoaderWorkers` | `0` | Starting value for the training dialog's "DataLoader workers" spinner, which is what a run uses. See below. |
| Show Menu Indicator Dot | `dlclassifier.showMenuDot` | `true` | Show a colored dot next to the extension name in the Extensions menu. Takes effect after restart. |
| Menu Indicator Dot Color | `dlclassifier.menuDotColor` | Magenta | Color of the menu indicator dot. Uses QuPath's color picker. Takes effect after restart. |
| Show Developer Pretraining Options | `dlclassifier.showDeveloperPretrainingOptions` | `false` | Adds **Utilities > SSL Pretrain Encoder...** (SimCLR/BYOL) to the menu. Takes effect after restarting QuPath. |

## Tile Settings

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Tile Size | `dlclassifier.tileSize` | `512` | Tile size in pixels. Saved by the Apply dialog; used by the simple scripting API. |
| Tile Overlap | `dlclassifier.tileOverlap` | `64` | Tile overlap in pixels, per side. No dialog sets it; the simple scripting API uses it. |
| Tile Overlap Percent | `dlclassifier.tileOverlapPercent` | `12.5` | Tile overlap as a percentage. Shared by the Train dialog, the Apply dialog and the overlay. |

## Object Output Settings

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Default Object Type | `dlclassifier.defaultObjectType` | `DETECTION` | QuPath object type for OBJECTS output |
| Min Object Size | `dlclassifier.minObjectSizeMicrons` | `10.0` | Minimum object area in um^2 |
| Hole Filling | `dlclassifier.holeFillingMicrons` | `5.0` | Hole filling threshold in um^2 |

## Training Defaults

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Default Epochs | `dlclassifier.defaultEpochs` | `50` | Initial epoch count |
| Default Batch Size | `dlclassifier.defaultBatchSize` | `8` | Initial batch size |
| Default Learning Rate | `dlclassifier.defaultLearningRate` | `0.0001` | Initial learning rate |
| Use Augmentation | `dlclassifier.useAugmentation` | `true` | Enable data augmentation |
| Use Pretrained Weights | `dlclassifier.usePretrainedWeights` | `true` | Use pretrained encoder weights |
| Default Normalization | `dlclassifier.defaultNormalization` | `PERCENTILE_99` | Channel normalization strategy |
| Training: DataLoader Workers (experimental) | `dlclassifier.defaultDataLoaderWorkers` | `0` | Starting value for the training dialog's "DataLoader workers" spinner, which is what a run uses. See below. |

### Training: DataLoader Workers (experimental)

At the default of `0`, batches are prepared on the main thread, so the GPU waits
while each batch is read, decoded and augmented. Raising it to 2 or more overlaps
that work with GPU compute, which is the main lever on per-epoch time when the GPU
is otherwise sitting idle between bursts.

Two conditions apply, and both will silently do nothing if you miss them.

**It depends on a fix that is not merged upstream.** Worker processes inherit the
Python worker's stdin and stdout, which are Appose's JSON protocol pipes, and
deadlock before the first batch arrives. That is
[apposed/appose#31](https://github.com/apposed/appose/issues/31), still open. The
extension applies the fix itself: the first run that asks for workers restarts
the Python worker with Appose's protocol moved off those two streams, which costs
one restart and nothing after that. A run that asks for `0` workers leaves the
streams exactly as Appose ships them. If training stops before the first batch,
set the spinner back to `0` and please say so in an issue.

**It is ignored while the in-memory dataset cache is `auto` (the default) or `on`.**
Each worker process would copy the whole cache, so the cache wins and workers are
forced back to `0` (you get a popup saying so). To use workers, set the training
dialog's **In-memory dataset** option to `off` or `bounded`. Which of the two is
faster depends on your disk and augmentation settings, so it is worth timing one
epoch each way.

## Training Dialog Preferences (remembered across sessions)

These are saved when you click "Start Training" and restored next time you open the dialog.

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Last Architecture | `dlclassifier.lastArchitecture` | `unet` | Last used model architecture |
| Last Backbone | `dlclassifier.lastBackbone` | `resnet34` | Last used encoder backbone |
| Validation Split | `dlclassifier.validationSplit` | `20` | Validation split percentage |
| Horizontal Flip | `dlclassifier.augFlipHorizontal` | `true` | Augmentation: horizontal flip |
| Vertical Flip | `dlclassifier.augFlipVertical` | `true` | Augmentation: vertical flip |
| Rotation | `dlclassifier.augRotation` | `true` | Augmentation: 90-degree rotation |
| Elastic Deformation | `dlclassifier.augElasticDeform` | `false` | Augmentation: elastic distortion |
| Intensity Aug Mode | `dlclassifier.augIntensityMode` | `generic` (shown as None) | Intensity augmentation mode: `brightfield`, `fluorescence`, or anything else, which the dialog shows as None |
| Resolution Downsample | `dlclassifier.defaultDownsample` | `1.0` | Resolution downsample factor (1x, 2x, 4x, 8x, 16x) |
| Context Scale | `dlclassifier.defaultContextScale` | `1` | Surrounding context level (1 = None, 2, 4, 8, 16) |
| Line Stroke Width | `dlclassifier.lastLineStrokeWidth` | `0` | Line annotation mask width (0 = use QuPath's annotation stroke thickness) |
| Rebalance by Default | `dlclassifier.rebalanceByDefault` | `true` | Auto-rebalance class weights when classes are loaded |
| Random Seed | `dlclassifier.lastSeed` | `0` | 0 = a new random start every run; any positive value makes the run repeatable |

## Training Recipe Preferences (remembered across sessions)

These are in the Learning Rate & Optimizer, Loss Function, Duration & Stopping and Batch Size & Memory sections of the training dialog.

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| LR Scheduler | `dlclassifier.defaultScheduler` | `onecycle` | Learning rate schedule type |
| Loss Function | `dlclassifier.defaultLossFunction` | `ce_dice` | Training loss function |
| Early Stop Metric | `dlclassifier.defaultEarlyStoppingMetric` | `mean_iou` | Metric for early stopping (`disabled` turns early stopping off) |
| Early Stop Patience | `dlclassifier.defaultEarlyStoppingPatience` | `15` | Epochs without improvement |
| Mixed Precision | `dlclassifier.defaultMixedPrecision` | `true` | 16-bit mixed precision on NVIDIA GPUs |

## Inference Dialog Preferences (remembered across sessions)

These are saved when you click "Apply" and restored next time you open the dialog.

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Output Type | `dlclassifier.lastOutputType` | `OBJECTS` | Last used output type. The Apply dialog offers only `OBJECTS` and `MEASUREMENTS`; any other stored value (e.g. a legacy `RENDERED_OVERLAY`) is coerced to `OBJECTS` when the dialog loads. |
| Blend Mode | `dlclassifier.lastBlendMode` | `GAUSSIAN` | Last used blend mode. The prediction overlay uses it too. |
| Smoothing | `dlclassifier.smoothing` | `1.0` | Boundary smoothing, in um |
| Application Scope | `dlclassifier.applicationScope` | `ALL_ANNOTATIONS` | Last used scope |

The probability smoothing sigma, `dlclassifier.overlaySmoothing`, is an extension preference; see **Overlay Prediction Smoothing** above.

## Loss function parameters

Only used when the corresponding Loss Function option is picked in
the training dialog; not exposed in the top-level preference pane.

| Setting | Key | Default | Notes |
|---------|-----|---------|-------|
| Default Loss Function | `dlclassifier.defaultLossFunction` | `ce_dice` | Preset for new dialogs. Values: `ce_dice`, `cross_entropy`, `focal_dice`, `focal`, `boundary_ce`, `boundary_ce_dice`, `lovasz`, `ce_lovasz`. |
| Default Focal Gamma | `dlclassifier.defaultFocalGamma` | `2.0` | Only used with focal variants. Preserved through OHEM (see `OHEMFocalLoss`). |
| Default Boundary Sigma | `dlclassifier.defaultBoundarySigma` | `3.0` | Distance-transform falloff length in pixels. Only used with `boundary_ce` / `boundary_ce_dice`. |
| Default Boundary w_min | `dlclassifier.defaultBoundaryWMin` | `0.1` | Floor weight applied at the exact class boundary (**Boundary Floor Weight** in the dialog). |

## Experimental inference providers

These flags route ONNX inference through different ORT execution
providers. Changing either toggles a runtime popup the first time
the flag is turned ON (explains that cached models are auto-
reloaded so the change takes effect immediately).

| Setting | Key | Default | Notes |
|---------|-----|---------|-------|
| Experimental: TensorRT Inference | `dlclassifier.experimentalTensorRT` | `false` | Routes CUDA inference through `TensorrtExecutionProvider`. Silently falls back to `CUDAExecutionProvider` when TRT is not installed. Per-model engine cache under `~/.dlclassifier/tensorrt_cache` keyed by SHA1(model_path + metadata.json mtime). |
| Experimental: INT8 Quantization | `dlclassifier.experimentalInt8` | `false` | Only takes effect when TensorRT is also on. Training must have run for >= 20 epochs for the BN-folded INT8 variant to be emitted. |
| Overlay Fast Argmax (uint8) | `dlclassifier.useCompactArgmaxOutput` | `false` | Returns `uint8` argmax instead of float32 probabilities. 20x smaller wire payload; disables overlay smoothing, multi-pass averaging, and tile blending. Honored by both the overlay and Apply DL Pixel Classifier. |

## Option-interaction warnings

Some risky combinations of options open a warning popup. Each popup has a
**Don't show again** box; to bring a hidden warning back, reset its key
(see [Resetting Preferences](#resetting-preferences)).

<details>
<summary>Watcher keys (for developers)</summary>

The runtime `InteractionWarningService` pops a GUI dialog when a
risky option combination is detected (see `TRAINING_GUIDE.md`).
Each watcher has a "Don't show again" checkbox in the popup; the
state is persisted per-watcher.

| Watcher | Key (boolean) | Default | Fires when... |
|---------|---------------|---------|----------------|
| TileOverlapSplit (BLOCKING) | `dlclassifier.warning.overlap-split-leakage.suppressed` | `false` | tile overlap > 0 AND no image has an explicit TRAIN_ONLY/VAL_ONLY role (pixel leakage across stratified split). Cannot be suppressed. |
| InMemoryCacheWorkers | `dlclassifier.warning.cache-workers-downgrade.suppressed` | `false` | in-memory cache active AND data_loader_workers > 0 (auto-downgraded to 0). |
| ChannelsLastBrn | `dlclassifier.warning.channels-last-brn.suppressed` | `false` | Loaded model uses BatchRenorm (channels_last inference layout auto-skipped). |
| BrnFoldConvergence | `dlclassifier.warning.int8-brn-fold-convergence.suppressed` | `false` | INT8 preference enabled (reminder that BRN must be trained >= 20 epochs). |
| ExperimentalProvidersToggle | `dlclassifier.warning.experimental-providers-toggle.suppressed` | `false` | TRT or INT8 toggled ON (reminder that ORT sessions are auto-reloaded). |
| OhemFocal | `dlclassifier.warning.ohem-focal-gamma.suppressed` | `true` (default-suppressed) | Tripwire; fires when OHEM + Focal are selected together to confirm OHEMFocalLoss composition is active. |
| PlateauValLoss | `dlclassifier.warning.plateau-val-loss-mode.suppressed` | `true` (default-suppressed) | Tripwire; fires when plateau scheduler + `val_loss` ES are selected to confirm the auto-derived plateau mode is active. |

To re-enable a default-suppressed tripwire, set the corresponding
key to `false` in QuPath's preference store.

</details>

## Resetting Preferences

Change individual preferences in **Edit > Preferences**, or by adjusting dialog settings and clicking Apply or Start Training.

QuPath keeps preferences in Java's per-user preference store (on Windows, the registry), not in a file meant for editing. To reset every QuPath preference, including this extension's, start QuPath once with the `-r` (reset) option. The GPU device choice (`appose.gpuDevice`) is kept in a separate store shared by the Appose extensions; set it back to `auto` in **Edit > Preferences**.
