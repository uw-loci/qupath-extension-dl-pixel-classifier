# Preferences Reference

All persistent preferences for the DL Pixel Classifier extension, their defaults, and where they appear.

## How Preferences Work

Preferences are stored using QuPath's persistent preference system (`PathPrefs`). They survive QuPath restarts and are scoped per user.

- **Dialog preferences** are saved automatically when you click "Start Training" or "Apply"
- **Extension preferences** are editable in Edit > Preferences under "DL Pixel Classifier"
- All preference keys are prefixed with `dlclassifier.`

## Extension Preferences

These appear in **Edit > Preferences > DL Pixel Classifier**.

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Overlay Overlap (um) | `dlclassifier.overlayOverlapUm` | `25.0` | Target tile overlap distance in microns for overlays. Converted to pixels using the image's pixel calibration. Ensures consistent overlap regardless of objective/resolution. Minimum 64 pixels. |
| Use GPU for Inference | `dlclassifier.useGPU` | `true` | Use GPU acceleration when available |
| Training Data Export Directory | `dlclassifier.trainingExportDir` | `""` | Directory for exporting training data patches. Empty = use temp directory. |

## Tile Settings

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Tile Size | `dlclassifier.tileSize` | `512` | Default tile size in pixels |
| Tile Overlap | `dlclassifier.tileOverlap` | `64` | Tile overlap in pixels |
| Tile Overlap Percent | `dlclassifier.tileOverlapPercent` | `12.5` | Tile overlap as percentage |

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
| Default Learning Rate | `dlclassifier.defaultLearningRate` | `0.001` | Initial learning rate |
| Use Augmentation | `dlclassifier.useAugmentation` | `true` | Enable data augmentation |
| Use Pretrained Weights | `dlclassifier.usePretrainedWeights` | `true` | Use pretrained encoder weights |
| Default Normalization | `dlclassifier.defaultNormalization` | `PERCENTILE_99` | Channel normalization strategy |

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
| Intensity Aug Mode | `dlclassifier.augIntensityMode` | `none` | Intensity augmentation mode: none, brightfield, fluorescence |
| Resolution Downsample | `dlclassifier.defaultDownsample` | `1.0` | Resolution downsample factor (1x, 2x, 4x, 8x, 16x) |
| Context Scale | `dlclassifier.defaultContextScale` | `1` | Multi-scale context level (1=none, 2, 4, 8, 16) |
| Line Stroke Width | `dlclassifier.lastLineStrokeWidth` | `0` | Line annotation mask width (0 = use QuPath's annotation stroke thickness) |
| Rebalance by Default | `dlclassifier.rebalanceByDefault` | `true` | Auto-rebalance class weights when classes are loaded |

## Training Strategy Preferences (remembered across sessions)

These are in the collapsed "TRAINING STRATEGY" section of the training dialog.

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| LR Scheduler | `dlclassifier.defaultScheduler` | `onecycle` | Learning rate schedule type |
| Loss Function | `dlclassifier.defaultLossFunction` | `ce_dice` | Training loss function |
| Early Stop Metric | `dlclassifier.defaultEarlyStoppingMetric` | `mean_iou` | Metric for early stopping |
| Early Stop Patience | `dlclassifier.defaultEarlyStoppingPatience` | `15` | Epochs without improvement |
| Mixed Precision | `dlclassifier.defaultMixedPrecision` | `true` | FP16/FP32 AMP |

## Inference Dialog Preferences (remembered across sessions)

These are saved when you click "Apply" and restored next time you open the dialog.

| Preference | Key | Default | Description |
|-----------|-----|---------|-------------|
| Output Type | `dlclassifier.lastOutputType` | `RENDERED_OVERLAY` | Last used output type |
| Blend Mode | `dlclassifier.lastBlendMode` | `CENTER_CROP` | Last used blend mode (CENTER_CROP recommended; forced for OVERLAY output) |
| Smoothing | `dlclassifier.smoothing` | `1.0` | Boundary smoothing amount |
| Overlay Smoothing | `dlclassifier.overlaySmoothing` | `2.0` | Gaussian sigma for probability map smoothing in overlay mode (0 = off, 1-2 = light, 3-5 = moderate) |
| Application Scope | `dlclassifier.applicationScope` | `ALL_ANNOTATIONS` | Last used scope |
| Create Backup | `dlclassifier.createBackup` | `false` | Back up measurements before overwriting |

## Resetting Preferences

To reset all DL Pixel Classifier preferences to defaults:

1. Close QuPath
2. Open QuPath's preferences file (location varies by OS)
3. Delete all entries starting with `dlclassifier.`
4. Restart QuPath

Alternatively, manually change individual preferences in **Edit > Preferences** or by adjusting dialog settings and clicking Apply/Train.
