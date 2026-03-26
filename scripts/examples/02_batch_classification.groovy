/**
 * Batch DL Pixel Classification
 *
 * This script applies a trained classifier to all images in the current project
 * that have annotations. Results are saved back to the project.
 *
 * Prerequisites:
 * - DL Pixel Classifier extension installed with Python environment set up
 * - A trained classifier must exist
 * - Project must be open with images containing annotations
 *
 * Usage:
 * 1. Open a QuPath project
 * 2. Update the classifierId below to match your trained classifier
 * 3. Run this script
 */

import qupath.ext.dlclassifier.scripting.DLClassifierScripts

// ============ Configuration ============

// Classifier ID - change this to match your trained classifier
def classifierId = "unet_resnet34_20260120_143000"

// Output type: "measurements" or "objects"
def outputType = "measurements"

// Skip images that already have DL measurements
def skipAlreadyProcessed = true

// ============ Main Script ============

// Check backend
if (!DLClassifierScripts.isServerAvailable()) {
    println "ERROR: DL backend is not available!"
    println "Go to Extensions > DL Pixel Classifier > Setup DL Environment"
    return
}

// Check project
def project = getProject()
if (project == null) {
    println "ERROR: No project is open."
    return
}

// Load classifier
def classifier
try {
    classifier = DLClassifierScripts.loadClassifier(classifierId)
    println "Using classifier: " + classifier.getName()
} catch (Exception e) {
    println "ERROR: Could not load classifier - " + e.getMessage()
    println "Available: " + DLClassifierScripts.listClassifiers().join(", ")
    return
}

// Get project info
def entries = project.getImageList()
println "Project has " + entries.size() + " image(s)"

// ============ Option 1: Simple One-Line Batch Processing ============
// This uses the built-in batch utilities for the simplest approach

println "\n--- Running batch classification ---"

def result = DLClassifierScripts.classifyProject(
    classifier,
    outputType,
    skipAlreadyProcessed,
    { imageName, index -> println "  Processing: $imageName (${index + 1}/${entries.size()})" }
)

// Summary
println "\n========== Summary =========="
println "Processed: " + result.processed()
println "Skipped:   " + result.skipped()
println "Errors:    " + result.errors()
println "Total:     " + result.total()
println "Success:   " + (result.isSuccess() ? "YES" : "NO")
println "============================="

println "\nBatch processing complete."


// ============ Option 2: Manual Loop (if you need more control) ============
// Uncomment this section if you need custom logic for each image

/*
def processed = 0
def skipped = 0
def errors = 0

entries.each { entry ->
    println "\n--- Processing: " + entry.getImageName() + " ---"

    try {
        // Read image data
        def imageData = entry.readImageData()
        def hierarchy = imageData.getHierarchy()
        def annotations = hierarchy.getAnnotationObjects()

        if (annotations.isEmpty()) {
            println "  No annotations - skipping"
            skipped++
            return
        }

        // Check if already processed
        if (skipAlreadyProcessed && DLClassifierScripts.hasClassificationMeasurements(annotations)) {
            println "  Already processed - skipping"
            skipped++
            return
        }

        println "  Found " + annotations.size() + " annotation(s)"

        // Set as current image data (required for classification)
        setBatchProjectAndImage(project, imageData)

        // Run classification
        DLClassifierScripts.classifyRegions(classifier, annotations, outputType)

        // Save results
        entry.saveImageData(imageData)
        println "  Classification complete and saved"
        processed++

    } catch (Exception e) {
        println "  ERROR: " + e.getMessage()
        errors++
    }
}

println "\nManual batch processing complete."
println "Processed: $processed, Skipped: $skipped, Errors: $errors"
*/
