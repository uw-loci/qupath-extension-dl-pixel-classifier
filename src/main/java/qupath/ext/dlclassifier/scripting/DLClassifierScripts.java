package qupath.ext.dlclassifier.scripting;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import java.util.function.BiConsumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;
import qupath.ext.dlclassifier.controller.InferenceWorkflow;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.scripting.QP;

/**
 * Scripting API for deep learning pixel classification.
 * <p>
 * This class provides a Groovy-friendly interface for common classification
 * operations in QuPath scripts.
 *
 * <h3>Usage Examples</h3>
 * <pre>{@code
 * // Load and apply classifier
 * def classifier = DLClassifierScripts.loadClassifier("my_classifier")
 * def annotations = getAnnotationObjects()
 * DLClassifierScripts.classifyRegions(classifier, annotations)
 *
 * // Batch process entire project
 * DLClassifierScripts.classifyProject(classifier)
 * }</pre>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class DLClassifierScripts {

    private static final Logger logger = LoggerFactory.getLogger(DLClassifierScripts.class);
    private static final ModelManager modelManager = new ModelManager();

    private DLClassifierScripts() {
        // Utility class
    }

    /**
     * Loads a classifier by ID.
     *
     * @param classifierId the classifier ID
     * @return the classifier metadata
     * @throws IllegalArgumentException if classifier not found
     */
    public static ClassifierMetadata loadClassifier(String classifierId) {
        ClassifierMetadata metadata = modelManager.loadClassifier(classifierId);
        if (metadata == null) {
            throw new IllegalArgumentException("Classifier not found: " + classifierId);
        }
        logger.info("Loaded classifier: {}", metadata.getName());
        return metadata;
    }

    /**
     * Lists all available classifiers.
     *
     * @return list of classifier IDs
     */
    public static List<String> listClassifiers() {
        return modelManager.listClassifiers().stream()
                .map(ClassifierMetadata::getId)
                .toList();
    }

    /**
     * Re-runs training using the saved settings of an existing classifier
     * with optional per-field overrides. Convenience for the "train at
     * seeds 42, 43, 44 with otherwise identical config" workflow that
     * previously required dropping into {@link
     * qupath.ext.dlclassifier.controller.TrainingWorkflow#builder()} and
     * re-entering every parameter.
     *
     * <p>The new classifier name defaults to
     * {@code <existingId>_retrain_<timestamp>}; override the name (and any
     * other field) by passing it through {@code overrides}. The overrides
     * map accepts the same key names that
     * {@link qupath.ext.dlclassifier.model.TrainingConfig.Builder#fromTrainingSettings(java.util.Map)}
     * consumes (e.g. {@code "seed"}, {@code "epochs"}, {@code "learning_rate"}).
     *
     * <p>Training runs against {@code QP.getCurrentImageData()} unless the
     * project images and per-image roles are populated separately. For a
     * multi-image script, use the lower-level
     * {@link qupath.ext.dlclassifier.controller.TrainingWorkflow#builder()}
     * directly.
     *
     * @param existingClassifierId the saved classifier to copy settings from
     * @param overrides            field overrides keyed by the metadata
     *                             {@code training_settings} key names; may be
     *                             null or empty for a pure re-run
     * @return the training result
     * @throws IllegalArgumentException if the classifier does not exist or
     *                                  has no usable training settings
     */
    public static qupath.ext.dlclassifier.controller.TrainingWorkflow.TrainingResult retrainClassifier(
            String existingClassifierId, Map<String, Object> overrides) {
        ClassifierMetadata source = loadClassifier(existingClassifierId);
        Map<String, Object> ts = source.getTrainingSettings();
        if (ts == null || ts.isEmpty()) {
            throw new IllegalArgumentException("Classifier '" + existingClassifierId
                    + "' has no training_settings recorded -- cannot retrain from it.");
        }
        Map<String, Object> merged = new LinkedHashMap<>(ts);
        if (overrides != null) merged.putAll(overrides);
        qupath.ext.dlclassifier.model.TrainingConfig.Builder cb =
                qupath.ext.dlclassifier.model.TrainingConfig.Builder.fromTrainingSettings(merged);
        // Top-level metadata fields aren't in training_settings; copy from the
        // source classifier so the architecture matches.
        cb.modelType(source.getModelType())
                .backbone(source.getBackbone())
                .contextScale(source.getContextScale())
                .downsample(source.getDownsample());
        qupath.ext.dlclassifier.model.TrainingConfig config = cb.build();

        ChannelConfiguration channels = ChannelConfiguration.builder()
                .channelNames(source.getExpectedChannelNames())
                .normalizationStrategy(source.getNormalizationStrategy())
                // Carry the trained normalization flags forward -- see
                // NORMALIZATION_ROUNDTRIP.md.
                .perChannelNormalization(source.isPerChannelNormalization())
                .clipPercentile(source.getClipPercentile())
                .bitDepth(source.getBitDepthTrained())
                .build();
        List<String> classes = source.getClasses().stream()
                .map(ClassifierMetadata.ClassInfo::name)
                .toList();
        String name = (overrides != null && overrides.get("name") instanceof String n && !n.isEmpty())
                ? n
                : existingClassifierId + "_retrain_" + System.currentTimeMillis();
        String description =
                (overrides != null && overrides.get("description") instanceof String d) ? d : source.getDescription();
        logger.info(
                "Retraining from '{}' as '{}', {} override(s)",
                existingClassifierId,
                name,
                overrides == null ? 0 : overrides.size());
        return qupath.ext.dlclassifier.controller.TrainingWorkflow.builder()
                .name(name)
                .description(description)
                .config(config)
                .channels(channels)
                .classes(classes)
                .build()
                .run();
    }

    /**
     * Convenience overload: re-trains with no overrides (true "duplicate the
     * config and run again"). Most useful as a smoke test that the metadata
     * round-trips cleanly.
     */
    public static qupath.ext.dlclassifier.controller.TrainingWorkflow.TrainingResult retrainClassifier(
            String existingClassifierId) {
        return retrainClassifier(existingClassifierId, Collections.emptyMap());
    }

    /**
     * Classifies regions using a loaded classifier.
     *
     * @param classifier  the classifier metadata
     * @param annotations annotations to classify
     */
    public static void classifyRegions(ClassifierMetadata classifier, Collection<PathObject> annotations) {
        classifyRegions(classifier, annotations, "measurements");
    }

    /**
     * Classifies regions using a loaded classifier with specified output type.
     *
     * @param classifier  the classifier metadata
     * @param annotations annotations to classify
     * @param outputType  output type: "measurements", "objects", or "overlay"
     */
    public static void classifyRegions(
            ClassifierMetadata classifier, Collection<PathObject> annotations, String outputType) {
        ImageData<BufferedImage> imageData = QP.getCurrentImageData();
        if (imageData == null) {
            throw new IllegalStateException("No image is open");
        }

        // Build configuration
        InferenceConfig.OutputType outType =
                switch (outputType.toLowerCase()) {
                    case "objects" -> InferenceConfig.OutputType.OBJECTS;
                    case "overlay" -> InferenceConfig.OutputType.OVERLAY;
                    case "rendered_overlay" -> InferenceConfig.OutputType.RENDERED_OVERLAY;
                    default -> InferenceConfig.OutputType.MEASUREMENTS;
                };

        InferenceConfig config = InferenceConfig.builder()
                .tileSize(DLClassifierPreferences.getTileSize())
                .overlap(DLClassifierPreferences.getTileOverlap())
                .outputType(outType)
                .useGPU(DLClassifierPreferences.isUseGPU())
                .build();

        // Build channel configuration
        ChannelConfiguration channelConfig = ChannelConfiguration.builder()
                .selectedChannels(
                        classifier.getExpectedChannelNames().isEmpty()
                                ? List.of(0, 1, 2)
                                : java.util.stream.IntStream.range(0, classifier.getInputChannels())
                                        .boxed()
                                        .toList())
                .channelNames(
                        classifier.getExpectedChannelNames().isEmpty()
                                ? List.of("Red", "Green", "Blue")
                                : classifier.getExpectedChannelNames())
                .bitDepth(classifier.getBitDepthTrained())
                .normalizationStrategy(classifier.getNormalizationStrategy())
                // Carry the trained normalization flags so inference matches
                // training -- see NORMALIZATION_ROUNDTRIP.md.
                .perChannelNormalization(classifier.isPerChannelNormalization())
                .clipPercentile(classifier.getClipPercentile())
                .build();

        // Run classification
        try {
            runClassification(classifier, annotations, config, channelConfig, imageData);
        } catch (IOException e) {
            logger.error("Classification failed: {}", e.getMessage());
            throw new RuntimeException("Classification failed: " + e.getMessage(), e);
        }
    }

    /**
     * Classifies regions with explicit channel selection.
     *
     * @param classifier    the classifier metadata
     * @param annotations   annotations to classify
     * @param channelNames  names of channels to use
     */
    public static void classifyRegions(
            ClassifierMetadata classifier, Collection<PathObject> annotations, List<String> channelNames) {
        // Validate channel count
        if (channelNames.size() != classifier.getInputChannels()) {
            throw new IllegalArgumentException(String.format(
                    "Channel count mismatch: classifier expects %d channels but %d provided",
                    classifier.getInputChannels(), channelNames.size()));
        }

        ImageData<BufferedImage> imageData = QP.getCurrentImageData();
        if (imageData == null) {
            throw new IllegalStateException("No image is open");
        }

        // Build channel configuration with specified channels
        List<Integer> channelIndices = new ArrayList<>();
        var serverChannels = imageData.getServer().getMetadata().getChannels();

        for (String name : channelNames) {
            boolean found = false;
            for (int i = 0; i < serverChannels.size(); i++) {
                if (serverChannels.get(i).getName().equals(name)) {
                    channelIndices.add(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw new IllegalArgumentException("Channel not found: " + name);
            }
        }

        ChannelConfiguration channelConfig = ChannelConfiguration.builder()
                .selectedChannels(channelIndices)
                .channelNames(channelNames)
                .bitDepth(classifier.getBitDepthTrained())
                .normalizationStrategy(classifier.getNormalizationStrategy())
                // Carry the trained normalization flags so inference matches
                // training -- see NORMALIZATION_ROUNDTRIP.md.
                .perChannelNormalization(classifier.isPerChannelNormalization())
                .clipPercentile(classifier.getClipPercentile())
                .build();

        InferenceConfig config = InferenceConfig.builder()
                .tileSize(DLClassifierPreferences.getTileSize())
                .overlap(DLClassifierPreferences.getTileOverlap())
                .outputType(InferenceConfig.OutputType.MEASUREMENTS)
                .useGPU(DLClassifierPreferences.isUseGPU())
                .build();

        try {
            runClassification(classifier, annotations, config, channelConfig, imageData);
        } catch (IOException e) {
            throw new RuntimeException("Classification failed: " + e.getMessage(), e);
        }
    }

    /**
     * Runs classification on annotations using the InferenceWorkflow builder.
     */
    private static void runClassification(
            ClassifierMetadata classifier,
            Collection<PathObject> annotations,
            InferenceConfig config,
            ChannelConfiguration channelConfig,
            ImageData<BufferedImage> imageData)
            throws IOException {
        logger.info("Running classification on {} annotations", annotations.size());

        InferenceWorkflow.InferenceResult result = InferenceWorkflow.builder()
                .classifier(classifier)
                .config(config)
                .channels(channelConfig)
                .annotations(new ArrayList<>(annotations))
                .imageData(imageData)
                .build()
                .run();

        if (!result.success()) {
            throw new IOException("Classification failed: " + result.message());
        }

        logger.info("Classification complete: {}", result.message());
    }

    /**
     * Gets the default classifier handler (UNet).
     *
     * @return the default handler
     */
    public static ClassifierHandler getDefaultHandler() {
        return ClassifierRegistry.getDefaultHandler();
    }

    /**
     * Gets a classifier handler by type.
     *
     * @param type the classifier type (e.g., "unet")
     * @return the handler, or default if not found
     */
    public static ClassifierHandler getHandler(String type) {
        return ClassifierRegistry.getHandler(type).orElse(ClassifierRegistry.getDefaultHandler());
    }

    /**
     * Checks if the classification backend (Appose Python environment) is available.
     *
     * @return true if backend is healthy and ready for inference/training
     */
    public static boolean isServerAvailable() {
        try {
            ClassifierBackend backend = BackendFactory.getBackend();
            return backend.checkHealth();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Gets GPU information from the backend.
     *
     * @return GPU info string
     */
    public static String getGPUInfo() {
        try {
            ClassifierBackend backend = BackendFactory.getBackend();
            return backend.getGPUInfo();
        } catch (Exception e) {
            return "Unknown (backend unavailable)";
        }
    }

    // ==================== Batch Processing Utilities ====================

    /**
     * Result of a batch classification operation.
     *
     * @param processed number of images successfully processed
     * @param skipped   number of images skipped (no annotations or already processed)
     * @param errors    number of images that had errors
     * @param total     total number of images in the batch
     */
    public record BatchResult(int processed, int skipped, int errors, int total) {
        /**
         * Returns true if all images were processed without errors.
         */
        public boolean isSuccess() {
            return errors == 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "BatchResult[processed=%d, skipped=%d, errors=%d, total=%d]", processed, skipped, errors, total);
        }
    }

    /**
     * Classifies all images in the current project that have annotations.
     * <p>
     * This is a convenience method that processes all images, skipping those
     * that have already been processed (have DL measurements).
     *
     * <h3>Example</h3>
     * <pre>{@code
     * def classifier = DLClassifierScripts.loadClassifier("my_classifier")
     * def result = DLClassifierScripts.classifyProject(classifier)
     * println "Processed: ${result.processed()}, Skipped: ${result.skipped()}"
     * }</pre>
     *
     * @param classifier the classifier to use
     * @return result summary with processed/skipped/error counts
     */
    public static BatchResult classifyProject(ClassifierMetadata classifier) {
        return classifyProject(classifier, "measurements", true, null);
    }

    /**
     * Classifies all images in the current project.
     *
     * @param classifier      the classifier to use
     * @param outputType      output type: "measurements", "objects", or "overlay"
     * @param skipProcessed   if true, skip images that already have DL measurements
     * @param progressHandler optional callback for progress updates (imageName, index)
     * @return result summary
     */
    public static BatchResult classifyProject(
            ClassifierMetadata classifier,
            String outputType,
            boolean skipProcessed,
            BiConsumer<String, Integer> progressHandler) {
        Project<BufferedImage> project = QP.getProject();
        if (project == null) {
            throw new IllegalStateException("No project is open");
        }

        List<ProjectImageEntry<BufferedImage>> entries = project.getImageList();
        return classifyProjectImages(classifier, entries, outputType, skipProcessed, progressHandler);
    }

    /**
     * Classifies a specific list of project images.
     *
     * @param classifier      the classifier to use
     * @param entries         list of project image entries to process
     * @param outputType      output type: "measurements", "objects", or "overlay"
     * @param skipProcessed   if true, skip images that already have DL measurements
     * @param progressHandler optional callback for progress updates (imageName, index)
     * @return result summary
     */
    public static BatchResult classifyProjectImages(
            ClassifierMetadata classifier,
            List<ProjectImageEntry<BufferedImage>> entries,
            String outputType,
            boolean skipProcessed,
            BiConsumer<String, Integer> progressHandler) {
        int processed = 0;
        int skipped = 0;
        int errors = 0;
        int total = entries.size();

        logger.info("Starting batch classification of {} images", total);

        for (int i = 0; i < entries.size(); i++) {
            ProjectImageEntry<BufferedImage> entry = entries.get(i);
            String imageName = entry.getImageName();

            if (progressHandler != null) {
                progressHandler.accept(imageName, i);
            }

            try {
                ImageData<BufferedImage> imageData = entry.readImageData();
                var hierarchy = imageData.getHierarchy();
                var annotations = hierarchy.getAnnotationObjects();

                // Skip if no annotations
                if (annotations.isEmpty()) {
                    logger.debug("Skipping {} - no annotations", imageName);
                    skipped++;
                    continue;
                }

                // Skip if already processed
                if (skipProcessed && hasClassificationMeasurements(annotations)) {
                    logger.debug("Skipping {} - already processed", imageName);
                    skipped++;
                    continue;
                }

                logger.info("Processing: {} ({}/{})", imageName, i + 1, total);

                // Set as current image data for classification
                QP.setBatchProjectAndImage(QP.getProject(), imageData);

                // Run classification
                classifyRegions(classifier, annotations, outputType);

                // Save results
                entry.saveImageData(imageData);
                processed++;

            } catch (Exception e) {
                logger.error("Error processing {}: {}", imageName, e.getMessage());
                errors++;
            }
        }

        BatchResult result = new BatchResult(processed, skipped, errors, total);
        logger.info("Batch classification complete: {}", result);
        return result;
    }

    /**
     * Checks if any annotation in the collection has DL classification measurements.
     *
     * @param annotations annotations to check
     * @return true if at least one annotation has DL measurements
     */
    public static boolean hasClassificationMeasurements(Collection<PathObject> annotations) {
        return annotations.stream()
                .anyMatch(ann -> ann.getMeasurements().keySet().stream().anyMatch(name -> name.startsWith("DL:")));
    }

    /**
     * Removes all DL classification measurements from annotations.
     * <p>
     * This is useful for re-running classification on previously processed images.
     *
     * @param annotations annotations to clear
     * @return number of annotations that had measurements removed
     */
    public static int clearClassificationMeasurements(Collection<PathObject> annotations) {
        int cleared = 0;
        for (PathObject ann : annotations) {
            var measurements = ann.getMeasurements();
            List<String> toRemove = measurements.keySet().stream()
                    .filter(name -> name.startsWith("DL:"))
                    .toList();

            if (!toRemove.isEmpty()) {
                for (String name : toRemove) {
                    measurements.remove(name);
                }
                cleared++;
            }
        }
        return cleared;
    }

    /**
     * Clears all DL classification measurements from the current image.
     *
     * @return number of annotations that had measurements removed
     */
    public static int clearCurrentImageMeasurements() {
        ImageData<BufferedImage> imageData = QP.getCurrentImageData();
        if (imageData == null) {
            throw new IllegalStateException("No image is open");
        }

        var annotations = imageData.getHierarchy().getAnnotationObjects();
        int cleared = clearClassificationMeasurements(annotations);

        if (cleared > 0) {
            imageData
                    .getHierarchy()
                    .fireHierarchyChangedEvent(imageData.getHierarchy().getRootObject());
        }

        return cleared;
    }

    /**
     * Gets a summary of classification results for the current image.
     *
     * @return map of class name to total area (if measurements exist)
     */
    public static Map<String, Double> getClassificationSummary() {
        ImageData<BufferedImage> imageData = QP.getCurrentImageData();
        if (imageData == null) {
            throw new IllegalStateException("No image is open");
        }

        Map<String, Double> summary = new LinkedHashMap<>();
        var annotations = imageData.getHierarchy().getAnnotationObjects();

        for (PathObject ann : annotations) {
            var measurements = ann.getMeasurements();
            for (var entry : measurements.entrySet()) {
                String name = entry.getKey();
                if (name.startsWith("DL:") && name.contains("area")) {
                    double value = entry.getValue().doubleValue();
                    summary.merge(name, value, Double::sum);
                }
            }
        }

        return summary;
    }
}
