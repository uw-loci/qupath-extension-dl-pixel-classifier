# Scripting Guide

> **Audience:** This guide is for power users and developers who want to automate workflows via Groovy scripting in QuPath's script editor.

How to use the Groovy scripting API for batch processing and reproducible workflows.

## Overview

The DL Pixel Classifier provides two scripting interfaces:
1. **Simple API** (`DLClassifierScripts`) -- convenience methods for common operations
2. **Builder API** (`InferenceWorkflow.builder()`, `TrainingWorkflow.builder()`) -- full control over every parameter

Both dialogs also include a **"Copy as Script"** button that generates a runnable Groovy script from the current settings.

## Copy as Script

The fastest way to create a script:

1. Configure settings in the Training or Inference dialog
2. Click **"Copy as Script"** (bottom-left button)
3. Paste into QuPath's Script Editor (**Automate > Show script editor**)
4. Modify as needed for batch use

As of 0.7.9 the generated script round-trips every dialog setting that the
training and inference workflows accept, including loss-function-specific
parameters (focal gamma, boundary sigma / floor weight), OHEM ratios and
schedule, gradient accumulation, mixed precision, fused optimizer, LR
Finder, GPU augmentation, torch.compile, in-memory dataset mode, the
pretrained model path, and inference-side TTA / multi-pass averaging /
overlay smoothing. Earlier versions silently dropped some of these and
fell back to defaults when the script was run later -- if you have a
script generated before 0.7.9, regenerate it from the dialog so the
emitted settings match what the GUI ran.

## Simple API

### Load a classifier

```groovy
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

def classifier = DLClassifierScripts.loadClassifier("my_classifier_id")
```

### Apply to annotations

```groovy
// Apply with default settings (measurements output)
def annotations = getAnnotationObjects()
DLClassifierScripts.classifyRegions(classifier, annotations)

// Apply with specific output type
DLClassifierScripts.classifyRegions(classifier, annotations, "objects")
DLClassifierScripts.classifyRegions(classifier, annotations, "overlay")
DLClassifierScripts.classifyRegions(classifier, annotations, "measurements")
```

### Batch process a project

```groovy
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

def classifier = DLClassifierScripts.loadClassifier("my_classifier_id")

// Use the built-in project batch method
DLClassifierScripts.classifyProject(classifier)
println "Done"
```

### Retrain a classifier with a tweak

Re-runs training using a previously trained classifier's settings, with
optional per-field overrides. Useful for seed sweeps and "same recipe,
one knob changed" runs:

```groovy
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

// Same config, just a different seed
def result = DLClassifierScripts.retrainClassifier(
    "my_classifier_id",
    [seed: 43])

// Or sweep three seeds in one script
for (seed in [42, 43, 44]) {
    DLClassifierScripts.retrainClassifier(
        "my_classifier_id",
        [seed: seed, name: "my_classifier_seed${seed}"])
}
```

Overrides accept the same key names that the trained model's
`metadata.json` stores under `training_settings` (for example
`learning_rate`, `epochs`, `weight_decay`, `ohem_hard_ratio`). The
top-level `name` and `description` are also overridable. If the
override map is omitted, the run reproduces the original training.

## Builder API

For full control over inference parameters:

```groovy
import qupath.ext.dlclassifier.controller.InferenceWorkflow
import qupath.ext.dlclassifier.model.*
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

def classifier = DLClassifierScripts.loadClassifier("my_classifier_id")

def inferenceConfig = InferenceConfig.builder()
        .tileSize(512)
        .overlap(64)
        .blendMode(InferenceConfig.BlendMode.CENTER_CROP)
        .outputType(InferenceConfig.OutputType.MEASUREMENTS)
        .useGPU(true)
        .build()

def channelConfig = ChannelConfiguration.builder()
        .selectedChannels([0, 1, 2])
        .channelNames(["Red", "Green", "Blue"])
        .bitDepth(8)
        .normalizationStrategy(ChannelConfiguration.NormalizationStrategy.PERCENTILE_99)
        .build()

def result = InferenceWorkflow.builder()
        .classifier(classifier)
        .config(inferenceConfig)
        .channels(channelConfig)
        .annotations(getAnnotationObjects())
        .build()
        .run()

println "Processed ${result.processedAnnotations()} annotations, ${result.processedTiles()} tiles"
```

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
        .build()

// Training follows the same builder pattern
```

## Builder API Parameters

### InferenceConfig.builder()

| Method | Type | Description |
|--------|------|-------------|
| `.tileSize(int)` | int | Tile size in pixels (must be divisible by 32) |
| `.overlap(int)` | int | Overlap in pixels |
| `.overlapPercent(double)` | double | Overlap as percentage (0-50) |
| `.blendMode(BlendMode)` | enum | LINEAR, GAUSSIAN, CENTER_CROP, or NONE |
| `.outputType(OutputType)` | enum | MEASUREMENTS, OBJECTS, OVERLAY, or RENDERED_OVERLAY |
| `.objectType(OutputObjectType)` | enum | DETECTION or ANNOTATION |
| `.minObjectSize(double)` | double | Min object area in um^2 |
| `.holeFilling(double)` | double | Hole filling threshold in um^2 |
| `.smoothing(double)` | double | Boundary smoothing in microns |
| `.useGPU(boolean)` | boolean | Use GPU if available |
| `.useTTA(boolean)` | boolean | Test-time augmentation (D4 transforms, averaged) |
| `.multiPassAveraging(boolean)` | boolean | Run each tile at multiple spatial offsets and average |
| `.overlaySmoothingSigma(double)` | double | Gaussian sigma for probability-map smoothing (0 = off) |
| `.useCompactArgmaxOutput(boolean)` | boolean | Return uint8 argmax instead of float32 probabilities |
| `.maxTilesInMemory(int)` | int | Cap on tiles buffered in memory during batch inference |

### ChannelConfiguration.builder()

| Method | Type | Description |
|--------|------|-------------|
| `.selectedChannels(List<Integer>)` | list | Channel indices |
| `.channelNames(List<String>)` | list | Channel names |
| `.bitDepth(int)` | int | Image bit depth |
| `.normalizationStrategy(NormalizationStrategy)` | enum | PERCENTILE_99, MIN_MAX, Z_SCORE, FIXED_RANGE |

### Additional Simple API Methods

```groovy
// List all available classifiers
def classifierIds = DLClassifierScripts.listClassifiers()

// Batch process entire project
DLClassifierScripts.classifyProject(classifier)

// Batch with options (outputType is a String: "measurements", "objects", "overlay")
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
    "liver_images": DLClassifierScripts.loadClassifier("liver_v2"),
    "kidney_images": DLClassifierScripts.loadClassifier("kidney_v1")
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

```groovy
// After classification, export measurements to CSV
def annotations = getAnnotationObjects()
def sb = new StringBuilder()
sb.append("Name,Class,Area")
for (ann in annotations) {
    for (key in ann.getMeasurements().keySet()) {
        sb.append(",").append(key)
    }
    break
}
sb.append("\n")

for (ann in annotations) {
    sb.append(ann.getName()).append(",")
    sb.append(ann.getPathClass()).append(",")
    sb.append(ann.getROI().getArea())
    for (key in ann.getMeasurements().keySet()) {
        sb.append(",").append(ann.getMeasurements().get(key))
    }
    sb.append("\n")
}

new File("classification_results.csv").text = sb.toString()
println "Exported results"
```
