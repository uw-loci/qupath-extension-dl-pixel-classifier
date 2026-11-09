# Scripting Guide

> **Audience:** This guide is for power users and developers who want to automate workflows via Groovy scripting in QuPath's script editor.

How to use the Groovy scripting API for batch processing and reproducible workflows.

## Overview

The DL Pixel Classifier provides two scripting interfaces:
1. **Simple API** (`DLClassifierScripts`), convenience methods for common operations
2. **Builder API** (`InferenceWorkflow.builder()`, `TrainingWorkflow.builder()`), full control over every parameter

Both dialogs also have a button that generates a runnable Groovy script from the current settings: **Copy as Groovy Script** in the Train dialog and **Copy as Script** in the Apply dialog.

## Copy as Script

The fastest way to create a script:

1. Configure settings in the Train or Apply dialog
2. Click **Copy as Groovy Script** (Train dialog) or **Copy as Script** (Apply dialog), bottom left
3. Paste into QuPath's Script Editor (**Automate > Show script editor**)
4. Modify as needed for batch use

The generated script carries the dialog's settings, including the image list and per-image Train/Val roles for training. One exception: the Apply dialog's script does not carry **Overlay High-Quality Tile Averaging**, so a script made with that preference on runs without it. If you have a script generated before 0.7.9, regenerate it from the dialog: older scripts dropped some settings and fell back to defaults.

## Simple API

### Load a classifier

```groovy
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

println DLClassifierScripts.listClassifiers()
def classifier = DLClassifierScripts.loadClassifier("collagen_v1_1762183353123")
```

The ID is the classifier's folder name under the project's `classifiers/dl/` folder (the lower-cased name plus a millisecond timestamp), not the display name. `listClassifiers()` prints the IDs of every classifier in the project and in your user folder. An unknown ID throws `Classifier not found: <id>`.

### Apply to annotations

```groovy
// Apply with default settings (measurements output)
def annotations = getAnnotationObjects()
DLClassifierScripts.classifyRegions(classifier, annotations)

// Apply with specific output type
DLClassifierScripts.classifyRegions(classifier, annotations, "objects")
DLClassifierScripts.classifyRegions(classifier, annotations, "measurements")
```

The output type is `"measurements"` (the default) or `"objects"`. `"overlay"` and `"rendered_overlay"` are also accepted, but from a script they create objects exactly like `"objects"` and draw no overlay.

MEASUREMENTS adds, to each annotation and for each class, `DL: <class> area (um^2)` and `DL: <class> %`, plus `DL: Total pixels`.

The simple API reads the channels and normalization from the model. It uses the tile size last used in the Apply dialog (`dlclassifier.tileSize`), the `dlclassifier.tileOverlap` preference (64 pixels per side unless changed), the **Use GPU** preference, and fixed defaults for everything else: blend mode GAUSSIAN, min object size 10 um^2, hole filling 5 um^2, boundary smoothing 2.0 um, probability smoothing sigma 2.0. To match an Apply dialog run exactly, use **Copy as Script** from the Apply dialog.

### Choose channels by name

For a multi-channel image, pass the channel names the model expects:

```groovy
DLClassifierScripts.classifyRegions(classifier, getAnnotationObjects(), ["DAPI", "CD3", "CD8"])
```

Output is always MEASUREMENTS. The names must match the image's channel names exactly, and their number must match the model's input channels; otherwise the call throws `Channel not found: <name>` or `Channel count mismatch: ...`.

### Batch process a project

```groovy
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

def classifier = DLClassifierScripts.loadClassifier("collagen_v1_1762183353123")

def r = DLClassifierScripts.classifyProject(classifier)
println r
```

This prints a summary such as `BatchResult[processed=3, skipped=1, errors=0, total=4]`. Images with no annotations, and images whose annotations already carry `DL:` measurements, are skipped. Each processed image is saved.

### Retrain a classifier with a tweak

Re-runs training using a previously trained classifier's settings, with
optional per-field overrides. Useful for seed sweeps and "same recipe,
one knob changed" runs:

```groovy
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

// Same config, just a different seed
def result = DLClassifierScripts.retrainClassifier(
    "collagen_v1_1762183353123",
    [seed: 43])

// Or sweep three seeds in one script
for (seed in [42, 43, 44]) {
    DLClassifierScripts.retrainClassifier(
        "collagen_v1_1762183353123",
        [seed: seed, name: "collagen_seed" + seed])
}
```

Overrides accept the same key names that the trained model's
`metadata.json` stores under `training_settings` (for example
`learning_rate`, `epochs`, `weight_decay`, `ohem_hard_ratio`, `seed`). The
top-level `name` and `description` are also overridable; without a `name`,
the new classifier is called `<id>_retrain_<timestamp>`.

A retrain reuses the original settings, tile size, epoch count, classes,
channels and normalization. It does not repeat the original run:

- It trains on the image open in the viewer only. The original image list
  and per-image Train/Val roles are not restored. Open one of the training
  images before calling it, or, for a multi-image retrain, use
  **Copy as Groovy Script** from the Train dialog (which writes the image
  list and roles) or the training builder below.
- The original seed is reused only if that run had a nonzero **Random Seed**.
  A run with the default seed of 0 recorded none, so its retrain starts from
  a different random state.

## Builder API

For full control over inference parameters:

```groovy
import qupath.ext.dlclassifier.controller.InferenceWorkflow
import qupath.ext.dlclassifier.model.*
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

def classifier = DLClassifierScripts.loadClassifier("collagen_v1_1762183353123")

def inferenceConfig = InferenceConfig.builder()
        .tileSize(512)
        .overlap(64)
        .blendMode(InferenceConfig.BlendMode.CENTER_CROP)
        .outputType(InferenceConfig.OutputType.MEASUREMENTS)
        .useGPU(true)
        .build()

// Channels and normalization from the model, as the Apply dialog does
def channelConfig = ChannelConfiguration.builder()
        .selectedChannels(classifier.getSelectedChannelsOrRange())
        .channelNames(classifier.getExpectedChannelNames())
        .bitDepth(classifier.getBitDepthTrained())
        .build()
        .withNormalizationOf(classifier)

def result = InferenceWorkflow.builder()
        .classifier(classifier)
        .config(inferenceConfig)
        .channels(channelConfig)
        .annotations(getAnnotationObjects())
        .build()
        .run()

println "Processed ${result.processedAnnotations()} annotations, ${result.processedTiles()} tiles"
```

`withNormalizationOf(classifier)` sets the normalization strategy, the per-channel setting and the clip percentile to the ones the model was trained with. Leave it out and a model trained with different preprocessing is applied with the wrong one, without any error. See [NORMALIZATION_ROUNDTRIP.md](NORMALIZATION_ROUNDTRIP.md).

### Training builder

```groovy
import qupath.ext.dlclassifier.controller.TrainingWorkflow
import qupath.ext.dlclassifier.model.*

def trainingConfig = TrainingConfig.builder()
        .classifierType("unet")
        .backbone("resnet34")
        .epochs(100)
        .batchSize(8)
        .learningRate(0.001)
        .validationSplit(0.2)
        .tileSize(512)
        .overlap(0)
        .downsample(1.0)
        .usePretrainedWeights(true)
        .schedulerType("onecycle")
        .lossFunction("ce_dice")
        .earlyStoppingMetric("mean_iou")
        .earlyStoppingPatience(15)
        .mixedPrecision(true)
        .seed(42)
        .build()

// The channels the model will see (here: an 8-bit RGB image)
def channelConfig = ChannelConfiguration.builder()
        .selectedChannels([0, 1, 2])
        .channelNames(["Red", "Green", "Blue"])
        .bitDepth(8)
        .normalizationStrategy(ChannelConfiguration.NormalizationStrategy.PERCENTILE_99)
        .build()

def images = getProject().getImageList().findAll {
    it.getImageName() in ["Tile 1.ome.tif", "Tile 4.ome.tif"]
}

def result = TrainingWorkflow.builder()
        .name("Collagen_v2")
        .config(trainingConfig)
        .channels(channelConfig)
        .classes(["Background", "Collagen"])
        .projectImages(images)
        .trainOnlyImageNames(["Tile 1.ome.tif"] as Set)
        .valOnlyImageNames(["Tile 4.ome.tif"] as Set)
        .build()
        .run()

println result.success() ? "Saved ${result.classifierId()}" : "Failed: ${result.message()}"
```

`.overlap(...)` is in pixels here. Without `.projectImages(...)`, training uses the image open in the viewer.

To end a run once it is only improving marginally, add the opt-in exit (off by
default, so it is absent from the example above):

```groovy
        .marginalStopEnabled(true)
        .marginalStopMinImprovement(0.01)   // in the early-stop metric's own units
        .marginalStopWindow(20)             // epochs the improvement is measured across
```

It watches whichever metric `.earlyStoppingMetric(...)` names, and is independent
of `.earlyStoppingPatience(...)`. See [PARAMETERS.md](PARAMETERS.md) for the full
description.

## Builder API Parameters

### InferenceConfig.builder()

| Method | Type | Description |
|--------|------|-------------|
| `.tileSize(int)` | int | Tile size in pixels, at the model's resolution (must be divisible by 32) |
| `.overlap(int)` | int | Margin read on each side of a tile, in pixels at the model's resolution |
| `.overlapPercent(double)` | double | Sets the overlap as a percentage (0-50) of the tile size already set, so call `.tileSize()` first |
| `.blendMode(BlendMode)` | enum | GAUSSIAN (default), LINEAR, CENTER_CROP, or NONE. Every mode keeps only each tile's center; CENTER_CROP also raises the margin to at least 25% of the tile |
| `.outputType(OutputType)` | enum | MEASUREMENTS (default) or OBJECTS. OVERLAY and RENDERED_OVERLAY are accepted but produce the same objects as OBJECTS |
| `.objectType(OutputObjectType)` | enum | DETECTION or ANNOTATION |
| `.minObjectSize(double)` | double | Min object area in um^2 |
| `.maxObjectSize(double)` | double | Max object area in um^2 (0 = no limit) |
| `.holeFilling(double)` | double | Hole filling threshold in um^2 |
| `.smoothing(double)` | double | Boundary smoothing in micrometers (default 2.0; the Apply dialog's default is 1.0) |
| `.addShapeMeasurements(boolean)` | boolean | Attach circularity and solidity to each object |
| `.useGPU(boolean)` | boolean | Recorded in the config; inference uses the GPU whenever the environment can |
| `.useTTA(boolean)` | boolean | Test-time augmentation (each tile run 8 times, flipped and rotated, and averaged) |
| `.multiPassAveraging(boolean)` | boolean | Run each tile at multiple spatial offsets and average |
| `.overlaySmoothingSigma(double)` | double | Gaussian sigma for probability-map smoothing, in model pixels (0 = off) |
| `.useCompactArgmaxOutput(boolean)` | boolean | Return uint8 argmax instead of float32 probabilities |
| `.maxTilesInMemory(int)` | int | Cap on tiles buffered in memory during batch inference |

### ChannelConfiguration.builder()

| Method | Type | Description |
|--------|------|-------------|
| `.selectedChannels(List<Integer>)` | list | Channel indices |
| `.channelNames(List<String>)` | list | Channel names |
| `.bitDepth(int)` | int | Image bit depth |
| `.normalizationStrategy(NormalizationStrategy)` | enum | PERCENTILE_99, MIN_MAX, Z_SCORE, FIXED_RANGE |
| `.perChannelNormalization(boolean)` | boolean | Normalize each channel separately (default false) |
| `.clipPercentile(double)` | double | Percentile for PERCENTILE_99 (default 99) |
| `.fixedRange(double, double)` | double, double | Minimum and maximum for FIXED_RANGE |

On a built configuration, `.withNormalizationOf(classifier)` returns a copy with the model's normalization.

### Additional Simple API Methods

```groovy
// List all available classifiers
def classifierIds = DLClassifierScripts.listClassifiers()

// Batch process entire project
DLClassifierScripts.classifyProject(classifier)

// Batch with options: output type "measurements" or "objects", skip images
// that already have DL measurements, no progress callback
DLClassifierScripts.classifyProject(classifier, "objects", true, null)

// Check/clear measurements
DLClassifierScripts.hasClassificationMeasurements(annotations)
DLClassifierScripts.clearClassificationMeasurements(annotations)
DLClassifierScripts.clearCurrentImageMeasurements()

// Get classification summary
def summary = DLClassifierScripts.getClassificationSummary()

// Backend status
DLClassifierScripts.isServerAvailable()
DLClassifierScripts.getGPUInfo()
```

`getClassificationSummary()` returns a map from each `DL: ... area (um^2)` column to its sum over every annotation in the current image. Nested or overlapping annotations are counted more than once.

### Script errors

| Message | Cause | Fix |
|---------|-------|-----|
| `Classifier not found: <id>` | The ID is not a classifier folder name (often the display name was passed) | Use an ID from `listClassifiers()` |
| `No image is open` | `classifyRegions` needs a current image | Open an image, or call `setBatchProjectAndImage(...)` first |
| `No project is open` | `classifyProject` needs a project | Open the project |
| `Classifier '<id>' has no training_settings recorded -- cannot retrain from it.` | The model's `metadata.json` has no training settings | Retrain from the Train dialog instead |
| `At least 2 class names are required` | The training builder got fewer than two classes | Pass two or more names to `.classes(...)` |
| `Channel not found: <name>` / `Channel count mismatch: ...` | The channel-name overload got a name the image lacks, or the wrong number of names | Use the image's exact channel names, as many as the model expects |

## Common Patterns

### Process only specific annotation classes

```groovy
def annotations = getAnnotationObjects().findAll {
    it.getPathClass()?.getName() in ["Tumor", "ROI"]
}
DLClassifierScripts.classifyRegions(classifier, annotations)
```

### Apply different classifiers to different images

```groovy
def classifierMap = [
    "liver_images": DLClassifierScripts.loadClassifier("liver_v2_1760955300000"),
    "kidney_images": DLClassifierScripts.loadClassifier("kidney_v1_1761039000000")
]

for (entry in getProject().getImageList()) {
    def imageName = entry.getImageName()
    def classifier = classifierMap.find { imageName.contains(it.key) }?.value
    if (classifier) {
        def imageData = entry.readImageData()
        def annotations = imageData.getHierarchy().getAnnotationObjects()
        if (!annotations.isEmpty()) {
            setBatchProjectAndImage(getProject(), imageData)
            DLClassifierScripts.classifyRegions(classifier, annotations)
            entry.saveImageData(imageData)
        }
    }
}
```

### Export results after classification

QuPath's **Measure > Export measurements** writes the same numbers without a script. To do it from a script:

```groovy
// After classification, export measurements to CSV in the project folder
def annotations = getAnnotationObjects()
def keys = annotations.collectMany { it.getMeasurements().keySet() as List }.unique()
def cal = getCurrentServer().getPixelCalibration()

def lines = [(["Name", "Class", "Area (um^2)"] + keys).join(",")]
for (ann in annotations) {
    def m = ann.getMeasurements()
    def area = ann.getROI().getScaledArea(cal.getPixelWidthMicrons(), cal.getPixelHeightMicrons())
    def row = [ann.getName() ?: "", ann.getPathClass() ?: "", area]
    row += keys.collect { m.containsKey(it) ? m.get(it) : "" }
    lines << row.join(",")
}

def file = new File(buildFilePath(PROJECT_BASE_DIR, "classification_results.csv"))
file.text = lines.join("\n")
println "Exported ${annotations.size()} annotations to ${file}"
```

The header lists every measurement found on any annotation, so rows stay aligned when some annotations have no `DL:` columns.
