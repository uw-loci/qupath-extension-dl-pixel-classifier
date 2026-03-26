/**
 * Basic DL Pixel Classification Example
 *
 * This script demonstrates how to apply a trained deep learning classifier
 * to annotations in the current image.
 *
 * Prerequisites:
 * - DL Pixel Classifier extension installed with Python environment set up
 * - At least one trained classifier must exist
 * - Image must have annotation(s) to classify
 *
 * Usage:
 * 1. Open an image in QuPath
 * 2. Create annotation(s) around regions to classify
 * 3. Run this script
 */

import qupath.ext.dlclassifier.scripting.DLClassifierScripts

// ============ Configuration ============

// Classifier ID - change this to match your trained classifier
def classifierId = "unet_resnet34_20260120_143000"

// Output type: "measurements" (area stats), "objects" (detection objects), or "overlay"
def outputType = "measurements"

// ============ Main Script ============

// Check backend availability
if (!DLClassifierScripts.isServerAvailable()) {
    println "ERROR: DL backend is not available!"
    println "Go to Extensions > DL Pixel Classifier > Setup DL Environment"
    return
}

println "Backend available. GPU: " + DLClassifierScripts.getGPUInfo()

// List available classifiers
println "\nAvailable classifiers:"
def available = DLClassifierScripts.listClassifiers()
if (available.isEmpty()) {
    println "  No classifiers found. Please train a classifier first."
    return
}
available.each { println "  - " + it }

// Load the classifier
println "\nLoading classifier: " + classifierId
def classifier
try {
    classifier = DLClassifierScripts.loadClassifier(classifierId)
    println "Loaded: " + classifier.getName()
    println "  Architecture: " + classifier.getModelType() + " / " + classifier.getBackbone()
    println "  Classes: " + classifier.getClassNames().join(", ")
    println "  Input: " + classifier.getInputChannels() + " channels, " +
            classifier.getInputWidth() + "x" + classifier.getInputHeight()
} catch (Exception e) {
    println "ERROR: Could not load classifier - " + e.getMessage()
    println "Available classifiers: " + available.join(", ")
    return
}

// Get annotations to classify
def annotations = getAnnotationObjects()
if (annotations.isEmpty()) {
    println "\nNo annotations found. Please create annotation(s) to classify."
    return
}

println "\nClassifying " + annotations.size() + " annotation(s)..."

// Run classification
try {
    DLClassifierScripts.classifyRegions(classifier, annotations, outputType)
    println "Classification complete!"

    // Show results
    if (outputType == "measurements") {
        println "\nResults:"
        annotations.each { annotation ->
            def ml = annotation.getMeasurements()
            println "  " + (annotation.getName() ?: "Unnamed") + ":"
            ml.getNames().findAll { it.startsWith("DL:") }.each { name ->
                println "    " + name + ": " + ml.get(name)
            }
        }
    }

} catch (Exception e) {
    println "ERROR: Classification failed - " + e.getMessage()
    e.printStackTrace()
}

println "\nDone."
