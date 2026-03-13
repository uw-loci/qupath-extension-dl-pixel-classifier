package qupath.ext.dlclassifier.controller;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.DLClassifierChecks;
import qupath.ext.dlclassifier.service.ApposeClassifierBackend;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.DLPixelClassifier;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.service.OverlayService;
import qupath.ext.dlclassifier.service.PrecomputedPixelClassifier;
import qupath.ext.dlclassifier.ui.InferenceDialog;
import qupath.ext.dlclassifier.ui.ProgressMonitorController;
import qupath.ext.dlclassifier.utilities.BitDepthConverter;
import qupath.ext.dlclassifier.utilities.ChannelNormalizer;
import qupath.ext.dlclassifier.utilities.OutputGenerator;
import qupath.ext.dlclassifier.utilities.TileEncoder;
import qupath.ext.dlclassifier.utilities.TileProcessor;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.classifiers.pixel.PixelClassifierMetadata;
import qupath.lib.common.ColorTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.scripting.QP;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.images.servers.PixelType;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

import qupath.lib.plugins.workflow.DefaultScriptableWorkflowStep;

import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Workflow for applying a trained classifier to images.
 * <p>
 * This workflow:
 * <ol>
 *   <li>Loads a trained classifier</li>
 *   <li>Generates tiles from the image or annotations</li>
 *   <li>Runs inference on the server</li>
 *   <li>Merges results and generates output (measurements, objects, or overlay)</li>
 * </ol>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class InferenceWorkflow {

    private static final Logger logger = LoggerFactory.getLogger(InferenceWorkflow.class);

    /** Warn-once flag for context tiles resized due to small images. */
    private static final AtomicBoolean contextResizeWarned = new AtomicBoolean(false);

    private QuPathGUI qupath;

    public InferenceWorkflow() {
        this.qupath = QuPathGUI.getInstance();
    }

    // ==================== Headless Result Record ====================

    /**
     * Result of a headless inference run.
     *
     * @param processedAnnotations number of annotations processed
     * @param processedTiles       total number of tiles processed
     * @param objectsCreated       number of objects created (for OBJECTS output)
     * @param success              whether the inference completed successfully
     * @param message              summary or error message
     */
    public record InferenceResult(
            int processedAnnotations,
            int processedTiles,
            int objectsCreated,
            boolean success,
            String message
    ) {}

    // ==================== Builder API ====================

    /**
     * Creates a new builder for configuring and running inference without GUI.
     * <p>
     * Example usage:
     * <pre>{@code
     * InferenceResult result = InferenceWorkflow.builder()
     *     .classifier(metadata)
     *     .config(inferenceConfig)
     *     .channels(channelConfig)
     *     .annotations(annotationList)
     *     .build()
     *     .run();
     * }</pre>
     *
     * @return a new InferenceBuilder
     */
    public static InferenceBuilder builder() {
        return new InferenceBuilder();
    }

    /**
     * Builder for configuring headless inference runs.
     */
    public static class InferenceBuilder {
        private ClassifierMetadata classifier;
        private InferenceConfig config;
        private ChannelConfiguration channels;
        private List<PathObject> annotations;
        private ImageData<BufferedImage> imageData;

        private InferenceBuilder() {}

        /** Sets the classifier metadata (required). */
        public InferenceBuilder classifier(ClassifierMetadata classifier) {
            this.classifier = classifier;
            return this;
        }

        /** Sets the inference configuration (required). */
        public InferenceBuilder config(InferenceConfig config) {
            this.config = config;
            return this;
        }

        /** Sets the channel configuration (required). */
        public InferenceBuilder channels(ChannelConfiguration channels) {
            this.channels = channels;
            return this;
        }

        /** Sets the annotations to classify (required). */
        public InferenceBuilder annotations(List<PathObject> annotations) {
            this.annotations = annotations;
            return this;
        }

        /**
         * Sets the image data to use. If not provided, falls back to
         * {@code QP.getCurrentImageData()} at run time.
         */
        public InferenceBuilder imageData(ImageData<BufferedImage> imageData) {
            this.imageData = imageData;
            return this;
        }

        /**
         * Validates parameters and builds an {@link InferenceRunner}.
         *
         * @return a runner ready to execute inference
         * @throws IllegalStateException if required parameters are missing
         */
        public InferenceRunner build() {
            Objects.requireNonNull(classifier, "Classifier metadata is required");
            Objects.requireNonNull(config, "InferenceConfig is required");
            Objects.requireNonNull(channels, "ChannelConfiguration is required");
            if (annotations == null || annotations.isEmpty()) {
                throw new IllegalStateException("At least one annotation is required");
            }
            return new InferenceRunner(classifier, config, channels, annotations, imageData);
        }
    }

    /**
     * Executes inference synchronously without GUI dependencies.
     */
    public static class InferenceRunner {
        private final ClassifierMetadata classifier;
        private final InferenceConfig config;
        private final ChannelConfiguration channels;
        private final List<PathObject> annotations;
        private final ImageData<BufferedImage> imageData;

        private InferenceRunner(ClassifierMetadata classifier,
                                InferenceConfig config,
                                ChannelConfiguration channels,
                                List<PathObject> annotations,
                                ImageData<BufferedImage> imageData) {
            this.classifier = classifier;
            this.config = config;
            this.channels = channels;
            this.annotations = new ArrayList<>(annotations);
            this.imageData = imageData;
        }

        /**
         * Runs inference synchronously and returns the result.
         *
         * @return the inference result
         */
        public InferenceResult run() {
            ImageData<BufferedImage> imgData = this.imageData;
            if (imgData == null) {
                imgData = QP.getCurrentImageData();
            }
            if (imgData == null) {
                logger.warn("No image data available for inference");
                return new InferenceResult(0, 0, 0, false, "No image data available");
            }

            try {
                // Validate classifier-image compatibility
                String compatError = validateCompatibility(classifier, imgData);
                if (compatError != null) {
                    return new InferenceResult(0, 0, 0, false, compatError);
                }

                ImageServer<BufferedImage> server = imgData.getServer();
                TileProcessor tileProcessor = new TileProcessor(config);

                ClassifierBackend backend = BackendFactory.getBackend();

                int processedAnnotations = 0;
                int processedTiles = 0;
                int objectsCreated = 0;

                for (PathObject annotation : annotations) {
                    ROI region = annotation.getROI();
                    int tilesForRegion = processRegionCore(
                            region, annotation, tileProcessor, backend,
                            classifier, channels, config, server, imgData,
                            null // no progress monitor
                    );
                    processedTiles += tilesForRegion;
                    processedAnnotations++;
                }

                imgData.getHierarchy().fireHierarchyChangedEvent(
                        imgData.getHierarchy().getRootObject());

                // Add workflow step for reproducibility (not for visual-only overlay modes)
                if (config.getOutputType() != InferenceConfig.OutputType.OVERLAY
                        && config.getOutputType() != InferenceConfig.OutputType.RENDERED_OVERLAY) {
                    addWorkflowStep(imgData, classifier, config);
                }

                String message = String.format(
                        "Classification completed: %d annotation(s), %d tile(s)",
                        processedAnnotations, processedTiles);
                return new InferenceResult(processedAnnotations, processedTiles,
                        objectsCreated, true, message);

            } catch (Exception e) {
                logger.error("Headless inference failed", e);
                return new InferenceResult(0, 0, 0, false,
                        "Inference failed: " + e.getMessage());
            }
        }
    }

    /**
     * Starts the inference workflow.
     */
    /**
     * Starts the inference workflow.
     * <p>
     * Quick prerequisites (image check) run on the FX thread. The backend
     * health check runs asynchronously because Appose initialization may
     * take time on first launch.
     */
    public void start() {
        logger.info("Starting inference workflow");

        // Quick prerequisite: image must be open (instant check on FX thread)
        if (qupath.getImageData() == null) {
            showError("No Image", "Please open an image before applying a classifier.");
            return;
        }

        // Backend health check may block while Appose initializes.
        Dialogs.showInfoNotification("DL Pixel Classifier",
                "Connecting to classification backend...");

        CompletableFuture.supplyAsync(() -> DLClassifierChecks.checkServerHealth())
                .thenAcceptAsync(healthy -> {
                    if (healthy) {
                        showInferenceDialog();
                    } else {
                        String versionWarning = ApposeClassifierBackend.getVersionWarning();
                        if (versionWarning != null && !versionWarning.isEmpty()) {
                            showError("Python Environment Update Required",
                                    "The Python environment is out of date and must be rebuilt.\n\n" +
                                    "Go to Extensions > DL Pixel Classifier > Rebuild Python Environment\n" +
                                    "to update. Inference is disabled until the environment matches\n" +
                                    "the installed extension version.");
                        } else {
                            showError("Server Unavailable",
                                    "Cannot connect to classification backend.\n\n" +
                                    "If this is the first launch, the Python environment\n" +
                                    "may still be downloading (~2-4 GB). Check the QuPath\n" +
                                    "log for progress and try again in a few minutes.\n\n" +
                                    "Alternatively, start the Python server manually and\n" +
                                    "disable 'Use Appose' in Edit > Preferences.");
                        }
                    }
                }, Platform::runLater);
    }

    /**
     * Shows the inference configuration dialog.
     */
    private void showInferenceDialog() {
        InferenceDialog.showDialog()
                .thenAccept(result -> {
                    if (result != null) {
                        logger.info("Inference dialog completed. Classifier: {}",
                                result.classifier().getName());

                        // Get target objects based on application scope
                        ImageData<BufferedImage> imageData = qupath.getImageData();
                        List<PathObject> targetObjects;

                        switch (result.applicationScope()) {
                            case WHOLE_IMAGE:
                                // Create a temporary annotation covering the entire image
                                ImageServer<BufferedImage> server = imageData.getServer();
                                ROI fullImageROI = ROIs.createRectangleROI(
                                        0, 0, server.getWidth(), server.getHeight(),
                                        ImagePlane.getDefaultPlane());
                                PathObject fullImageAnnotation = PathObjects.createAnnotationObject(fullImageROI);
                                fullImageAnnotation.setName("Full Image");
                                targetObjects = List.of(fullImageAnnotation);
                                break;

                            case SELECTED_ANNOTATIONS:
                                Collection<PathObject> selected = imageData.getHierarchy().getSelectionModel()
                                        .getSelectedObjects();
                                targetObjects = selected.stream()
                                        .filter(o -> o.isAnnotation() || o.isTMACore())
                                        .toList();

                                if (targetObjects.isEmpty()) {
                                    showError("No Selection",
                                            "No annotations selected. Please select annotations to classify.");
                                    return;
                                }
                                break;

                            case ALL_ANNOTATIONS:
                            default:
                                targetObjects = new ArrayList<>(imageData.getHierarchy().getAnnotationObjects());
                                if (targetObjects.isEmpty()) {
                                    showError("No Annotations",
                                            "No annotations found. Please create annotations to classify.");
                                    return;
                                }
                                break;
                        }

                        // Run inference with progress
                        runInferenceWithProgress(
                                result.classifier(),
                                result.inferenceConfig(),
                                result.channelConfig(),
                                targetObjects
                        );
                    }
                })
                .exceptionally(ex -> {
                    logger.error("Inference dialog failed", ex);
                    showError("Error", "Failed to show inference dialog: " + ex.getMessage());
                    return null;
                });
    }

    /**
     * Runs inference on the specified region with progress monitoring.
     *
     * @param metadata        classifier metadata
     * @param inferenceConfig inference configuration
     * @param channelConfig   channel configuration
     * @param targetObjects   objects to classify
     */
    public void runInferenceWithProgress(ClassifierMetadata metadata,
                                         InferenceConfig inferenceConfig,
                                         ChannelConfiguration channelConfig,
                                         List<PathObject> targetObjects) {
        // Create progress monitor
        ProgressMonitorController progress = ProgressMonitorController.forInference();
        progress.setOnCancel(v -> {
            logger.info("Inference cancellation requested");
        });
        progress.show();

        CompletableFuture.runAsync(() -> {
            try {
                progress.setStatus("Preparing inference...");
                progress.log("Classifier: " + metadata.getName());
                progress.log("Processing " + targetObjects.size() + " annotation(s)");

                ImageData<BufferedImage> imageData = qupath.getImageData();

                // Validate classifier-image compatibility
                String compatError = validateCompatibility(metadata, imageData);
                if (compatError != null) {
                    progress.log("ERROR: " + compatError);
                    progress.complete(false, compatError);
                    return;
                }

                ImageServer<BufferedImage> server = imageData.getServer();

                // Create tile processor
                TileProcessor tileProcessor = new TileProcessor(inferenceConfig);

                // Get appropriate backend (Appose or HTTP)
                progress.setStatus("Connecting to backend...");
                ClassifierBackend backend = BackendFactory.getBackend();
                progress.log("Connected to classification backend");

                // Detect rendered overlay mode
                boolean isRenderedOverlay = inferenceConfig.getOutputType()
                        == InferenceConfig.OutputType.RENDERED_OVERLAY;
                List<PrecomputedPixelClassifier.ClassifiedRegion> classifiedRegions =
                        isRenderedOverlay ? new ArrayList<>() : null;

                // Memory check for rendered overlay
                if (isRenderedOverlay) {
                    long totalPixels = 0;
                    for (PathObject target : targetObjects) {
                        ROI roi = target.getROI();
                        totalPixels += (long) roi.getBoundsWidth() * (long) roi.getBoundsHeight();
                    }
                    int numClasses = metadata.getClasses().size();
                    // During merge: float[H][W][C] + float[H][W] weights + byte[H][W] classMap
                    long estimatedBytes = totalPixels * (4L * numClasses + 4L + 1L);
                    long availableMemory = Runtime.getRuntime().maxMemory()
                            - Runtime.getRuntime().totalMemory()
                            + Runtime.getRuntime().freeMemory();

                    if (estimatedBytes > availableMemory * 0.5) {
                        logger.warn("Rendered overlay would require ~{}MB but only ~{}MB available. "
                                + "Falling back to on-demand overlay.",
                                estimatedBytes / (1024 * 1024), availableMemory / (1024 * 1024));
                        progress.log("Region too large for rendered overlay -- using fast overlay instead");
                        DLPixelClassifier pixelClassifier = new DLPixelClassifier(
                                metadata, channelConfig, inferenceConfig, imageData);
                        OverlayService.getInstance().applyClassifierOverlay(
                                imageData, pixelClassifier, metadata, channelConfig);
                        progress.complete(true, "Applied fast overlay (region too large for rendered overlay)");
                        return;
                    }
                }

                // Count total tiles
                int totalTiles = 0;
                for (PathObject obj : targetObjects) {
                    List<TileProcessor.TileSpec> specs = tileProcessor.generateTiles(obj.getROI(), server);
                    totalTiles += specs.size();
                }
                progress.log("Total tiles to process: " + totalTiles);

                // Process each annotation
                int processedAnnotations = 0;
                int processedTiles = 0;

                for (PathObject annotation : targetObjects) {
                    if (progress.isCancelled()) {
                        progress.complete(false, "Inference cancelled by user");
                        return;
                    }

                    String annotationName = annotation.getName() != null ?
                            annotation.getName() : "Annotation " + (processedAnnotations + 1);
                    progress.setStatus("Processing: " + annotationName);
                    progress.log("Processing annotation: " + annotationName);

                    ROI region = annotation.getROI();

                    if (isRenderedOverlay) {
                        PrecomputedPixelClassifier.ClassifiedRegion classified =
                                processRegionForRenderedOverlay(
                                        region, annotation, tileProcessor, backend, metadata,
                                        channelConfig, inferenceConfig, server, imageData, progress);
                        if (classified != null) {
                            classifiedRegions.add(classified);
                        }
                    } else {
                        int tilesForRegion = processRegionWithProgress(
                                region, annotation, tileProcessor, backend, metadata,
                                channelConfig, inferenceConfig, server, imageData, progress
                        );
                        processedTiles += tilesForRegion;
                    }

                    processedAnnotations++;

                    progress.setOverallProgress((double) processedAnnotations / targetObjects.size());
                    progress.log("Completed " + processedAnnotations + "/" + targetObjects.size() + " annotations");
                }

                // Apply rendered overlay if applicable
                if (isRenderedOverlay && classifiedRegions != null && !classifiedRegions.isEmpty()) {
                    PixelClassifierMetadata pixelMeta = buildPrecomputedMetadata(metadata, imageData);
                    IndexColorModel colorModel = buildColorModelForPrecomputed(metadata);
                    PrecomputedPixelClassifier precomputed = new PrecomputedPixelClassifier(
                            classifiedRegions, pixelMeta, colorModel);
                    OverlayService.getInstance().applyClassifierOverlay(imageData, precomputed);
                    progress.log("Rendered overlay applied (" + classifiedRegions.size() + " region(s), "
                            + processedTiles + " tiles blended)");
                }

                // Fire hierarchy update
                imageData.getHierarchy().fireHierarchyChangedEvent(
                        imageData.getHierarchy().getRootObject());

                // Add workflow step for reproducibility (not for visual-only overlay modes)
                if (inferenceConfig.getOutputType() != InferenceConfig.OutputType.OVERLAY
                        && inferenceConfig.getOutputType() != InferenceConfig.OutputType.RENDERED_OVERLAY) {
                    addWorkflowStep(imageData, metadata, inferenceConfig);
                }

                progress.complete(true, String.format(
                        "Classification completed!\nProcessed %d annotation(s), %d tile(s)",
                        processedAnnotations, processedTiles));

            } catch (Exception e) {
                logger.error("Inference failed", e);
                progress.log("ERROR: " + e.getMessage());
                progress.complete(false, "Inference failed: " + e.getMessage());
            }
        });
    }

    /**
     * Runs inference on the specified region.
     *
     * @param metadata        classifier metadata
     * @param inferenceConfig inference configuration
     * @param channelConfig   channel configuration
     * @param targetObjects   objects to classify (null for whole image)
     * @deprecated Use {@link #runInferenceWithProgress} instead
     */
    @Deprecated
    public void runInference(ClassifierMetadata metadata,
                             InferenceConfig inferenceConfig,
                             ChannelConfiguration channelConfig,
                             List<PathObject> targetObjects) {
        runInferenceWithProgress(metadata, inferenceConfig, channelConfig, targetObjects);
    }

    /**
     * Processes a single region with progress updates.
     *
     * @return the number of tiles processed
     */
    private int processRegionWithProgress(ROI region,
                                          PathObject parentObject,
                                          TileProcessor tileProcessor,
                                          ClassifierBackend backend,
                                          ClassifierMetadata metadata,
                                          ChannelConfiguration channelConfig,
                                          InferenceConfig inferenceConfig,
                                          ImageServer<BufferedImage> server,
                                          ImageData<BufferedImage> imageData,
                                          ProgressMonitorController progress) throws IOException {
        return processRegionCore(region, parentObject, tileProcessor, backend,
                metadata, channelConfig, inferenceConfig, server, imageData, progress);
    }

    /**
     * Core region processing logic shared by GUI and headless paths.
     * <p>
     * When {@code progress} is {@code null}, progress updates and cancellation
     * checks are skipped, enabling headless execution.
     *
     * @param region          the ROI to process
     * @param parentObject    the parent annotation
     * @param tileProcessor   tile processor for generating tiles
     * @param backend         classifier backend (Appose or HTTP)
     * @param metadata        classifier metadata
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @param server          image server
     * @param imageData       image data
     * @param progress        progress monitor (nullable for headless execution)
     * @return the number of tiles processed
     * @throws IOException if tile processing or server communication fails
     */
    static int processRegionCore(ROI region,
                                 PathObject parentObject,
                                 TileProcessor tileProcessor,
                                 ClassifierBackend backend,
                                 ClassifierMetadata metadata,
                                 ChannelConfiguration channelConfig,
                                 InferenceConfig inferenceConfig,
                                 ImageServer<BufferedImage> server,
                                 ImageData<BufferedImage> imageData,
                                 ProgressMonitorController progress) throws IOException {
        // Resolve classifier ID to filesystem path for the Python server
        ModelManager modelManager = new ModelManager();
        String modelDirPath = modelManager.getModelPath(metadata.getId())
                .map(p -> p.getParent().toString())
                .orElse(metadata.getId());
        logger.info("Resolved model path: {}", modelDirPath);

        // Generate tiles
        List<TileProcessor.TileSpec> tileSpecs = tileProcessor.generateTiles(region, server);
        logger.info("Generated {} tiles for region", tileSpecs.size());
        if (progress != null) progress.log("Generated " + tileSpecs.size() + " tiles");

        // OVERLAY mode uses on-demand tile rendering via QuPath's PixelClassifier
        // interface, so skip batch tile processing entirely
        if (inferenceConfig.getOutputType() == InferenceConfig.OutputType.OVERLAY) {
            DLPixelClassifier pixelClassifier = new DLPixelClassifier(
                    metadata, channelConfig, inferenceConfig, imageData);
            OverlayService.getInstance().applyClassifierOverlay(
                    imageData, pixelClassifier, metadata, channelConfig);
            if (progress != null) {
                progress.log("Classification overlay applied - tiles rendered on demand");
            }
            return 0;
        }

        // Branch on output type: MEASUREMENTS uses aggregated tile-level inference,
        // OBJECTS needs full pixel-level probability maps
        boolean usePixelInference = inferenceConfig.getOutputType() != InferenceConfig.OutputType.MEASUREMENTS;

        // Multi-scale context: when contextScale > 1, detail + context tiles are concatenated
        int contextScale = metadata.getContextScale();

        // Process in batches with parallel tile preparation
        int batchSize = tileProcessor.getMaxTilesInMemory();
        List<float[][][]> allResults = new ArrayList<>();
        Path tempDir = null;
        int tileSize = inferenceConfig.getTileSize();

        // Thread pool for parallel tile reading -- overlaps I/O with GPU inference
        int poolSize = Math.min(4, Runtime.getRuntime().availableProcessors());
        ExecutorService tilePool = Executors.newFixedThreadPool(poolSize);

        try {
            if (usePixelInference) {
                tempDir = Files.createTempDirectory("dl-pixel-inference-");
                logger.info("Using pixel-level inference, temp dir: {}", tempDir);
            }

            // Double-buffered pipeline: prepare next batch while current batch infers
            PreparedBatch nextPrepared = null;

            for (int i = 0; i < tileSpecs.size(); i += batchSize) {
                if (progress != null && progress.isCancelled()) {
                    return i;
                }

                int end = Math.min(i + batchSize, tileSpecs.size());

                if (progress != null) {
                    progress.setDetail(String.format("Processing tiles %d-%d of %d",
                            i + 1, end, tileSpecs.size()));
                    progress.setCurrentProgress((double) i / tileSpecs.size());
                }

                // Get current batch -- either from pre-prepared or prepare now
                PreparedBatch currentBatch;
                if (nextPrepared != null) {
                    currentBatch = nextPrepared;
                    nextPrepared = null;
                } else {
                    List<TileProcessor.TileSpec> batch = tileSpecs.subList(i, end);
                    currentBatch = prepareBatch(batch, tileProcessor, server, tilePool, contextScale);
                }

                // Start preparing NEXT batch in parallel while we send current to server
                int nextStart = i + batchSize;
                Future<PreparedBatch> nextBatchFuture = null;
                if (nextStart < tileSpecs.size()) {
                    int nextEnd = Math.min(nextStart + batchSize, tileSpecs.size());
                    List<TileProcessor.TileSpec> nextBatchSpecs = tileSpecs.subList(nextStart, nextEnd);
                    nextBatchFuture = tilePool.submit(() ->
                            prepareBatch(nextBatchSpecs, tileProcessor, server, tilePool, contextScale));
                }

                // Send current batch to server for inference
                if (usePixelInference) {
                    ClassifierClient.PixelInferenceResult pixelResult =
                            backend.runPixelInferenceBinary(
                                    modelDirPath, currentBatch.rawBytes(), currentBatch.tileIds(),
                                    tileSize, tileSize, currentBatch.numChannels(),
                                    currentBatch.dtype(),
                                    channelConfig, inferenceConfig, tempDir, 0);

                    if (pixelResult == null) {
                        if (contextScale > 1) {
                            logger.warn("Binary pixel inference failed for context_scale={} " +
                                    "model; JSON fallback does not support multi-scale " +
                                    "context tiles", contextScale);
                        }
                        pixelResult = backend.runPixelInference(
                                modelDirPath, currentBatch.tileDataList(), channelConfig,
                                inferenceConfig, tempDir, 0);
                    }

                    if (pixelResult != null && pixelResult.outputPaths() != null) {
                        for (ClassifierClient.TileData tile : currentBatch.tileDataList()) {
                            String outputPath = pixelResult.outputPaths().get(tile.id());
                            if (outputPath != null) {
                                float[][][] probMap = ClassifierClient.readProbabilityMap(
                                        Path.of(outputPath),
                                        pixelResult.numClasses(),
                                        tileSize, tileSize);
                                allResults.add(probMap);
                            }
                        }
                    }
                } else {
                    ClassifierClient.InferenceResult result =
                            backend.runInferenceBinary(
                                    modelDirPath, currentBatch.rawBytes(), currentBatch.tileIds(),
                                    tileSize, tileSize, currentBatch.numChannels(),
                                    currentBatch.dtype(),
                                    channelConfig, inferenceConfig);

                    if (result == null) {
                        if (contextScale > 1) {
                            logger.warn("Binary inference failed for context_scale={} " +
                                    "model; JSON fallback does not support multi-scale " +
                                    "context tiles", contextScale);
                        }
                        result = backend.runInference(
                                modelDirPath, currentBatch.tileDataList(), channelConfig,
                                inferenceConfig);
                    }

                    if (result != null && result.predictions() != null) {
                        for (float[] probs : result.predictions().values()) {
                            float[][][] tileResult = new float[1][1][probs.length];
                            tileResult[0][0] = probs;
                            allResults.add(tileResult);
                        }
                    }
                }

                // Collect the pre-prepared next batch (blocks until ready)
                if (nextBatchFuture != null) {
                    try {
                        nextPrepared = nextBatchFuture.get();
                    } catch (Exception e) {
                        logger.warn("Parallel tile prep failed, will prepare sequentially", e);
                        nextPrepared = null;
                    }
                }
            }

            if (progress != null) progress.setCurrentProgress(1.0);

            // Create output generator
            OutputGenerator outputGenerator = new OutputGenerator(imageData, metadata, inferenceConfig);

            // Generate output based on type
            switch (inferenceConfig.getOutputType()) {
                case MEASUREMENTS:
                    outputGenerator.addMeasurements(parentObject, allResults, tileSpecs);
                    if (progress != null) progress.log("Added measurements to annotation");
                    break;

                case OBJECTS:
                    // Use the new merged-map approach for proper cross-tile object handling
                    int numClasses = metadata.getClasses().size();
                    List<PathObject> objects = outputGenerator.createObjectsFromTiles(
                            tileProcessor,
                            allResults,
                            tileSpecs,
                            region,
                            numClasses,
                            inferenceConfig.getObjectType()
                    );
                    imageData.getHierarchy().addObjects(objects);
                    if (progress != null) {
                        progress.log("Created " + objects.size() + " " +
                                inferenceConfig.getObjectType().name().toLowerCase() + " objects");
                    }
                    break;

                case OVERLAY:
                    // Should not reach here - OVERLAY exits early above
                    logger.warn("OVERLAY case reached in switch - this should not happen");
                    break;

                case RENDERED_OVERLAY:
                    // Handled by caller (runInferenceWithProgress collects ClassifiedRegions)
                    // Nothing to do here per-region; the caller assembles the overlay
                    break;
            }

            return tileSpecs.size();
        } finally {
            // Shut down tile preparation thread pool
            tilePool.shutdownNow();

            // Clean up temp directory for pixel inference
            if (tempDir != null) {
                try {
                    Files.walk(tempDir)
                            .sorted(Comparator.reverseOrder())
                            .forEach(path -> {
                                try {
                                    Files.deleteIfExists(path);
                                } catch (IOException e) {
                                    logger.warn("Failed to delete temp file: {}", path, e);
                                }
                            });
                } catch (IOException e) {
                    logger.warn("Failed to clean up temp directory: {}", tempDir, e);
                }
            }
        }
    }

    /**
     * Processes a region for rendered overlay: runs batch inference with blending
     * and returns the merged classification map as a ClassifiedRegion.
     * <p>
     * Reuses the same tile generation, batch inference, and probability blending
     * pipeline as OBJECTS mode, but returns the blended classification map instead
     * of creating PathObjects.
     *
     * @return the classified region, or null if no tiles were processed
     */
    static PrecomputedPixelClassifier.ClassifiedRegion processRegionForRenderedOverlay(
            ROI region,
            PathObject parentObject,
            TileProcessor tileProcessor,
            ClassifierBackend backend,
            ClassifierMetadata metadata,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            ImageServer<BufferedImage> server,
            ImageData<BufferedImage> imageData,
            ProgressMonitorController progress) throws IOException {

        // Resolve classifier ID to filesystem path
        ModelManager modelManager = new ModelManager();
        String modelDirPath = modelManager.getModelPath(metadata.getId())
                .map(p -> p.getParent().toString())
                .orElse(metadata.getId());

        // Generate tiles
        List<TileProcessor.TileSpec> tileSpecs = tileProcessor.generateTiles(region, server);
        if (tileSpecs.isEmpty()) return null;
        if (progress != null) progress.log("Generated " + tileSpecs.size() + " tiles for rendered overlay");

        int contextScale = metadata.getContextScale();
        int batchSize = tileProcessor.getMaxTilesInMemory();
        List<float[][][]> allResults = new ArrayList<>();
        Path tempDir = null;
        int tileSize = inferenceConfig.getTileSize();

        int poolSize = Math.min(4, Runtime.getRuntime().availableProcessors());
        ExecutorService tilePool = Executors.newFixedThreadPool(poolSize);

        try {
            tempDir = Files.createTempDirectory("dl-rendered-overlay-");

            PreparedBatch nextPrepared = null;

            for (int i = 0; i < tileSpecs.size(); i += batchSize) {
                if (progress != null && progress.isCancelled()) return null;

                int end = Math.min(i + batchSize, tileSpecs.size());

                if (progress != null) {
                    progress.setDetail(String.format("Rendered overlay: tiles %d-%d of %d",
                            i + 1, end, tileSpecs.size()));
                    progress.setCurrentProgress((double) i / tileSpecs.size());
                }

                PreparedBatch currentBatch;
                if (nextPrepared != null) {
                    currentBatch = nextPrepared;
                    nextPrepared = null;
                } else {
                    List<TileProcessor.TileSpec> batch = tileSpecs.subList(i, end);
                    currentBatch = prepareBatch(batch, tileProcessor, server, tilePool, contextScale);
                }

                // Pre-prepare next batch
                int nextStart = i + batchSize;
                Future<PreparedBatch> nextBatchFuture = null;
                if (nextStart < tileSpecs.size()) {
                    int nextEnd = Math.min(nextStart + batchSize, tileSpecs.size());
                    List<TileProcessor.TileSpec> nextBatchSpecs = tileSpecs.subList(nextStart, nextEnd);
                    nextBatchFuture = tilePool.submit(() ->
                            prepareBatch(nextBatchSpecs, tileProcessor, server, tilePool, contextScale));
                }

                // Run pixel inference (same as processRegionCore)
                ClassifierClient.PixelInferenceResult pixelResult =
                        backend.runPixelInferenceBinary(
                                modelDirPath, currentBatch.rawBytes(), currentBatch.tileIds(),
                                tileSize, tileSize, currentBatch.numChannels(),
                                currentBatch.dtype(),
                                channelConfig, inferenceConfig, tempDir, 0);

                if (pixelResult == null) {
                    if (contextScale > 1) {
                        logger.warn("Binary pixel inference failed for context_scale={} "
                                + "model; JSON fallback does not support multi-scale "
                                + "context tiles", contextScale);
                    }
                    pixelResult = backend.runPixelInference(
                            modelDirPath, currentBatch.tileDataList(), channelConfig,
                            inferenceConfig, tempDir, 0);
                }

                if (pixelResult != null && pixelResult.outputPaths() != null) {
                    for (ClassifierClient.TileData tile : currentBatch.tileDataList()) {
                        String outputPath = pixelResult.outputPaths().get(tile.id());
                        if (outputPath != null) {
                            float[][][] probMap = ClassifierClient.readProbabilityMap(
                                    Path.of(outputPath),
                                    pixelResult.numClasses(),
                                    tileSize, tileSize);
                            allResults.add(probMap);
                        }
                    }
                }

                if (nextBatchFuture != null) {
                    try {
                        nextPrepared = nextBatchFuture.get();
                    } catch (Exception e) {
                        logger.warn("Parallel tile prep failed, will prepare sequentially", e);
                        nextPrepared = null;
                    }
                }
            }

            if (allResults.isEmpty()) return null;

            // Merge with blending (same as OBJECTS pipeline)
            int numClasses = metadata.getClasses().size();
            int regionX = (int) region.getBoundsX();
            int regionY = (int) region.getBoundsY();
            int regionWidth = (int) Math.ceil(region.getBoundsWidth());
            int regionHeight = (int) Math.ceil(region.getBoundsHeight());

            TileProcessor.MergedResult merged = tileProcessor.mergeTileResultsWithEdgeHandling(
                    tileSpecs, allResults,
                    regionX, regionY, regionWidth, regionHeight,
                    numClasses);

            // Convert int[][] to byte[][] for memory efficiency
            int[][] intMap = merged.classificationMap();
            byte[][] byteMap = new byte[regionHeight][regionWidth];
            for (int y = 0; y < regionHeight; y++) {
                for (int x = 0; x < regionWidth; x++) {
                    byteMap[y][x] = (byte) intMap[y][x];
                }
            }

            if (progress != null) {
                progress.log("Blended " + allResults.size() + " tiles for region");
            }

            return new PrecomputedPixelClassifier.ClassifiedRegion(
                    byteMap, regionX, regionY, regionWidth, regionHeight);

        } finally {
            tilePool.shutdownNow();
            if (tempDir != null) {
                try {
                    Files.walk(tempDir)
                            .sorted(Comparator.reverseOrder())
                            .forEach(path -> {
                                try {
                                    Files.deleteIfExists(path);
                                } catch (IOException e) {
                                    logger.warn("Failed to delete temp file: {}", path, e);
                                }
                            });
                } catch (IOException e) {
                    logger.warn("Failed to clean up temp directory: {}", tempDir, e);
                }
            }
        }
    }

    /**
     * Builds PixelClassifierMetadata for a PrecomputedPixelClassifier.
     * No padding needed since data is already blended.
     */
    private static PixelClassifierMetadata buildPrecomputedMetadata(
            ClassifierMetadata metadata, ImageData<BufferedImage> imageData) {
        PixelCalibration cal = imageData.getServer().getPixelCalibration();

        double downsample = metadata.getDownsample();
        if (downsample > 1.0) {
            cal = cal.createScaledInstance(downsample, downsample);
        }

        Map<Integer, PathClass> labels = new LinkedHashMap<>();
        List<ImageChannel> channels = new ArrayList<>();
        for (ClassifierMetadata.ClassInfo classInfo : metadata.getClasses()) {
            int color = parseClassColorStatic(classInfo.color(), classInfo.index());
            PathClass pathClass = PathClass.fromString(classInfo.name(), color);
            int resolvedColor = pathClass.getColor();
            labels.put(classInfo.index(), pathClass);
            channels.add(ImageChannel.getInstance(classInfo.name(), resolvedColor));
        }

        return new PixelClassifierMetadata.Builder()
                .inputResolution(cal)
                .inputShape(256, 256)   // arbitrary -- data is pre-computed
                .inputPadding(0)         // no padding needed
                .setChannelType(ImageServerMetadata.ChannelType.CLASSIFICATION)
                .outputPixelType(PixelType.UINT8)
                .classificationLabels(labels)
                .outputChannels(channels)
                .build();
    }

    /**
     * Builds IndexColorModel from classifier metadata for the PrecomputedPixelClassifier.
     */
    private static IndexColorModel buildColorModelForPrecomputed(ClassifierMetadata metadata) {
        byte[] r = new byte[256];
        byte[] g = new byte[256];
        byte[] b = new byte[256];
        byte[] a = new byte[256];

        for (ClassifierMetadata.ClassInfo classInfo : metadata.getClasses()) {
            int idx = classInfo.index();
            if (idx < 0 || idx >= 256) continue;
            int color = parseClassColorStatic(classInfo.color(), classInfo.index());
            PathClass pathClass = PathClass.fromString(classInfo.name(), color);
            int resolvedColor = pathClass.getColor();
            r[idx] = (byte) ColorTools.red(resolvedColor);
            g[idx] = (byte) ColorTools.green(resolvedColor);
            b[idx] = (byte) ColorTools.blue(resolvedColor);
            a[idx] = (byte) 255;
        }

        return new IndexColorModel(8, 256, r, g, b, a);
    }

    /** Distinct color palette for fallback when class metadata lacks colors. */
    private static final int[][] FALLBACK_PALETTE = {
            {255, 0, 0}, {0, 170, 0}, {0, 0, 255}, {255, 255, 0},
            {255, 0, 255}, {0, 255, 255}, {255, 136, 0}, {136, 0, 255}
    };

    /**
     * Parses a hex color string to a packed RGB integer (QuPath format).
     * Falls back to a distinct palette color for the given class index.
     * <p>
     * Static version of the same logic in DLPixelClassifier for use
     * by the rendered overlay helper methods.
     */
    private static int parseClassColorStatic(String colorStr, int classIndex) {
        if (colorStr == null || colorStr.isEmpty() || "#808080".equals(colorStr)) {
            int[] c = FALLBACK_PALETTE[classIndex % FALLBACK_PALETTE.length];
            return ColorTools.packRGB(c[0], c[1], c[2]);
        }
        try {
            String hex = colorStr.startsWith("#") ? colorStr.substring(1) : colorStr;
            int rgb = Integer.parseInt(hex, 16);
            return ColorTools.packRGB(
                    (rgb >> 16) & 0xFF,
                    (rgb >> 8) & 0xFF,
                    rgb & 0xFF);
        } catch (NumberFormatException e) {
            int[] c = FALLBACK_PALETTE[classIndex % FALLBACK_PALETTE.length];
            return ColorTools.packRGB(c[0], c[1], c[2]);
        }
    }

    /**
     * Processes a single region.
     * @deprecated Use {@link #processRegionWithProgress} instead
     */
    @Deprecated
    private void processRegion(ROI region,
                               TileProcessor tileProcessor,
                               ClassifierBackend backend,
                               ClassifierMetadata metadata,
                               ChannelConfiguration channelConfig,
                               InferenceConfig inferenceConfig,
                               ImageServer<BufferedImage> server) throws IOException {
        // Generate tiles
        List<TileProcessor.TileSpec> tileSpecs = tileProcessor.generateTiles(region, server);
        logger.info("Generated {} tiles for region", tileSpecs.size());

        // Process in batches
        int batchSize = tileProcessor.getMaxTilesInMemory();

        for (int i = 0; i < tileSpecs.size(); i += batchSize) {
            int end = Math.min(i + batchSize, tileSpecs.size());
            List<TileProcessor.TileSpec> batch = tileSpecs.subList(i, end);

            // Read and encode tiles
            List<ClassifierClient.TileData> tileDataList = new ArrayList<>();
            for (TileProcessor.TileSpec spec : batch) {
                BufferedImage tileImage = tileProcessor.readTile(spec, server);
                String encoded = TileEncoder.encodeTileBase64Png(tileImage);
                tileDataList.add(new ClassifierClient.TileData(
                        String.valueOf(spec.index()),
                        encoded,
                        spec.x(),
                        spec.y()
                ));
            }

            // Run inference
            backend.runInference(
                    metadata.getId(),
                    tileDataList,
                    channelConfig,
                    inferenceConfig
            );
        }
    }

    /**
     * Pre-prepared batch containing both raw binary and base64-encoded tile data.
     *
     * @param rawBytes     concatenated tile pixels (uint8 or float32 depending on dtype)
     * @param tileIds      ordered tile IDs matching byte order
     * @param tileDataList base64 tile data for JSON fallback path
     * @param dtype        "uint8" for 8-bit RGB fast path, "float32" for N-channel path
     * @param numChannels  number of channels per tile in the raw bytes
     */
    private record PreparedBatch(
            byte[] rawBytes,
            List<String> tileIds,
            List<ClassifierClient.TileData> tileDataList,
            String dtype,
            int numChannels
    ) {}

    /**
     * Prepares a batch of tiles for inference by reading images and encoding
     * them in both raw binary and base64 formats.
     * <p>
     * Tile reading is parallelized across the provided thread pool for I/O overlap.
     * When {@code contextScale > 1}, context tiles are also read and concatenated
     * with detail tiles along the channel axis.
     *
     * @param batch         tile specs for this batch
     * @param tileProcessor tile processor for reading tiles
     * @param server        image server
     * @param tilePool      thread pool for parallel reading
     * @param contextScale  context scale factor (1 = no context, >1 = multi-scale)
     * @return prepared batch data
     * @throws IOException if tile reading fails
     */
    private static PreparedBatch prepareBatch(
            List<TileProcessor.TileSpec> batch,
            TileProcessor tileProcessor,
            ImageServer<BufferedImage> server,
            ExecutorService tilePool,
            int contextScale) throws IOException {

        // Read all detail tiles in parallel
        List<Future<BufferedImage>> detailFutures = new ArrayList<>();
        List<Future<BufferedImage>> contextFutures = new ArrayList<>();
        for (TileProcessor.TileSpec spec : batch) {
            detailFutures.add(tilePool.submit(() -> tileProcessor.readTile(spec, server)));
            if (contextScale > 1) {
                contextFutures.add(tilePool.submit(() ->
                        readContextTileBatch(spec, tileProcessor, server, contextScale)));
            }
        }

        List<String> tileIds = new ArrayList<>();
        ByteArrayOutputStream rawBuffer = new ByteArrayOutputStream();
        List<ClassifierClient.TileData> tileDataList = new ArrayList<>();

        // Determine encoding path from first tile (all tiles in a batch share
        // the same image type since they come from the same server)
        String dtype = null;
        int detailChannels = 0;

        for (int j = 0; j < batch.size(); j++) {
            TileProcessor.TileSpec spec = batch.get(j);
            String tileId = String.valueOf(spec.index());
            tileIds.add(tileId);

            BufferedImage tileImage;
            try {
                tileImage = detailFutures.get(j).get();
            } catch (Exception e) {
                throw new IOException("Failed to read tile " + tileId, e);
            }

            // Decide encoding on first tile
            if (dtype == null) {
                if (TileEncoder.isSimpleRgb(tileImage)) {
                    dtype = "uint8";
                    detailChannels = 3;
                } else {
                    dtype = "float32";
                    detailChannels = tileImage.getRaster().getNumBands();
                }
            }

            // Encode detail tile
            byte[] detailBytes;
            if ("uint8".equals(dtype)) {
                detailBytes = TileEncoder.encodeTileRaw(tileImage);
            } else {
                detailBytes = TileEncoder.encodeTileRawFloat(tileImage, null);
            }

            if (contextScale > 1) {
                // Read and encode context tile, then interleave channels per pixel
                BufferedImage contextImage;
                try {
                    contextImage = contextFutures.get(j).get();
                } catch (Exception e) {
                    throw new IOException("Failed to read context tile for " + tileId, e);
                }
                byte[] contextBytes;
                if ("uint8".equals(dtype)) {
                    contextBytes = TileEncoder.encodeTileRaw(contextImage);
                } else {
                    contextBytes = TileEncoder.encodeTileRawFloat(contextImage, null);
                }
                int numPixels = tileImage.getWidth() * tileImage.getHeight();
                int bytesPerChannel = "uint8".equals(dtype) ? 1 : Float.BYTES;
                byte[] interleaved = TileEncoder.interleaveContextChannels(
                        detailBytes, contextBytes, numPixels, detailChannels, bytesPerChannel);
                rawBuffer.write(interleaved);
            } else {
                rawBuffer.write(detailBytes);
            }

            // Base64 for fallback (detail tile only -- context not supported in JSON path)
            String encoded = TileEncoder.encodeTileBase64Png(tileImage);
            tileDataList.add(new ClassifierClient.TileData(
                    tileId, encoded, spec.x(), spec.y()));
        }

        int numChannels = contextScale > 1 ? detailChannels * 2 : detailChannels;
        return new PreparedBatch(rawBuffer.toByteArray(), tileIds, tileDataList,
                dtype != null ? dtype : "uint8", numChannels);
    }

    /**
     * Reads a context tile for batch inference. The context tile covers
     * {@code contextScale} times the area of the detail tile in each dimension,
     * centered on the same location, and downsampled to the same pixel dimensions.
     * <p>
     * Three-tier strategy for handling image edges:
     * <ol>
     *   <li>Ideal: context region fits entirely within the image</li>
     *   <li>Clamped: context region is shifted to fit (image >= context size)</li>
     *   <li>Resized: image smaller than context region -- read entire image, resize to match</li>
     * </ol>
     */
    private static BufferedImage readContextTileBatch(
            TileProcessor.TileSpec spec,
            TileProcessor tileProcessor,
            ImageServer<BufferedImage> server,
            int contextScale) throws IOException {
        double downsample = tileProcessor.getDownsample();

        // Convert tile coordinates from downsampled space to full-res
        int fullResX = (int) (spec.x() * downsample);
        int fullResY = (int) (spec.y() * downsample);
        int fullResW = (int) (spec.width() * downsample);
        int fullResH = (int) (spec.height() * downsample);

        // Context region covers contextScale times the area in each dimension
        int contextW = fullResW * contextScale;
        int contextH = fullResH * contextScale;

        int imgW = server.getWidth();
        int imgH = server.getHeight();

        // Expected output dimensions: same as detail tile pixel dimensions
        int expectedW = spec.width();
        int expectedH = spec.height();

        int cx, cy, readW, readH;

        if (imgW >= contextW && imgH >= contextH) {
            // Tier 1/2: Image large enough -- clamp position to keep context within bounds
            int centerX = fullResX + fullResW / 2;
            int centerY = fullResY + fullResH / 2;
            cx = centerX - contextW / 2;
            cy = centerY - contextH / 2;
            cx = Math.max(0, Math.min(cx, imgW - contextW));
            cy = Math.max(0, Math.min(cy, imgH - contextH));
            readW = contextW;
            readH = contextH;
        } else {
            // Tier 3: Image smaller than context region -- read entire image
            cx = 0;
            cy = 0;
            readW = imgW;
            readH = imgH;
            if (contextResizeWarned.compareAndSet(false, true)) {
                logger.warn("Image ({}x{}) smaller than context region ({}x{}) -- " +
                        "reading entire image and resizing to match detail tile dimensions",
                        imgW, imgH, contextW, contextH);
            }
        }

        // Read at higher downsample so output pixel dimensions match detail tile
        double contextDownsample = downsample * contextScale;
        RegionRequest contextRequest = RegionRequest.createInstance(
                server.getPath(), contextDownsample,
                cx, cy, readW, readH);

        BufferedImage contextImage = server.readRegion(contextRequest);
        if (contextImage == null) {
            throw new IOException("Failed to read context tile for spec " + spec.index());
        }

        // Resize to expected dimensions if the read region was smaller than expected
        if (contextImage.getWidth() != expectedW || contextImage.getHeight() != expectedH) {
            BufferedImage resized = new BufferedImage(expectedW, expectedH, contextImage.getType());
            java.awt.Graphics2D g = resized.createGraphics();
            g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                    java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(contextImage, 0, 0, expectedW, expectedH, null);
            g.dispose();
            contextImage = resized;
        }

        return contextImage;
    }

    /**
     * Validates that the classifier is compatible with the current image.
     *
     * @param metadata  classifier metadata
     * @param imageData image data
     * @return null if compatible, or an error message string if not
     */
    static String validateCompatibility(ClassifierMetadata metadata,
                                        ImageData<BufferedImage> imageData) {
        if (imageData == null || imageData.getServer() == null) {
            return "No image data available";
        }

        ImageServer<BufferedImage> server = imageData.getServer();
        int imageChannels = server.nChannels();
        int classifierChannels = metadata.getInputChannels();

        if (imageChannels < classifierChannels) {
            return String.format(
                    "Channel mismatch: classifier expects %d channel(s) but image has %d.\n" +
                    "The classifier was trained on %d-channel images and cannot be applied to this image.",
                    classifierChannels, imageChannels, classifierChannels);
        }

        return null;
    }

    /**
     * Adds a workflow step to the image history for "Run for project" support.
     * The generated script uses DLClassifierScripts to reload and re-run the
     * classifier, so it works when QuPath sets each image as current.
     */
    private static void addWorkflowStep(ImageData<?> imageData,
                                         ClassifierMetadata metadata,
                                         InferenceConfig config) {
        String outputType = config.getOutputType().name().toLowerCase();
        String script = String.format(
                "import qupath.ext.dlclassifier.scripting.DLClassifierScripts%n" +
                "DLClassifierScripts.classifyRegions(" +
                    "DLClassifierScripts.loadClassifier(\"%s\"), " +
                    "getAnnotationObjects(), \"%s\")",
                metadata.getId().replace("\\", "\\\\").replace("\"", "\\\""),
                outputType);

        imageData.getHistoryWorkflow().addStep(
                new DefaultScriptableWorkflowStep(
                        "DL Pixel Classification (" + outputType + ")",
                        script));
        logger.info("Added workflow step for DL classification ({})", outputType);
    }

    /**
     * Shows an error dialog.
     */
    private void showError(String title, String message) {
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle(title);
            alert.setHeaderText(null);
            alert.setContentText(message);
            alert.showAndWait();
        });
    }
}
