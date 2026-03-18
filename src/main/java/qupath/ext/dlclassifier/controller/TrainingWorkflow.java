package qupath.ext.dlclassifier.controller;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.DLClassifierChecks;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.service.ApposeClassifierBackend;
import qupath.ext.dlclassifier.service.ApposeService;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.service.OverlayService;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.ui.ProgressMonitorController;
import qupath.ext.dlclassifier.ui.TrainingAreaIssuesDialog;
import qupath.ext.dlclassifier.ui.TrainingDialog;
import qupath.ext.dlclassifier.utilities.AnnotationExtractor;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.scripting.QP;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;

import javafx.geometry.Insets;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;

import qupath.lib.common.ColorTools;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Workflow for training a new deep learning pixel classifier.
 * <p>
 * This workflow guides the user through:
 * <ol>
 *   <li>Selecting classification classes from annotations</li>
 *   <li>Configuring channel selection and normalization</li>
 *   <li>Setting training hyperparameters</li>
 *   <li>Exporting training data</li>
 *   <li>Training the model on the server</li>
 *   <li>Saving the trained classifier</li>
 * </ol>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TrainingWorkflow {

    private static final Logger logger = LoggerFactory.getLogger(TrainingWorkflow.class);

    private QuPathGUI qupath;

    public TrainingWorkflow() {
        this.qupath = QuPathGUI.getInstance();
    }

    // ==================== Headless Result Record ====================

    /**
     * Result of a headless training run.
     *
     * @param classifierId    the saved classifier ID
     * @param classifierName  the classifier display name
     * @param finalLoss       best model validation loss
     * @param finalAccuracy   best model accuracy
     * @param bestEpoch       epoch that produced the best model
     * @param bestMeanIoU     best model mean IoU
     * @param epochsCompleted number of epochs completed
     * @param success         whether training completed successfully
     * @param message         summary or error message
     */
    public record TrainingResult(
            String classifierId,
            String classifierName,
            double finalLoss,
            double finalAccuracy,
            int bestEpoch,
            double bestMeanIoU,
            int epochsCompleted,
            boolean success,
            String message
    ) {}

    /**
     * Parameters for resuming training.
     *
     * @param totalEpochs new total epoch count
     * @param learningRate new learning rate
     * @param batchSize new batch size
     */
    public record ResumeParams(int totalEpochs, double learningRate, int batchSize) {}

    // ==================== Builder API ====================

    /**
     * Creates a new builder for configuring and running training without GUI.
     * <p>
     * Example usage:
     * <pre>{@code
     * TrainingResult result = TrainingWorkflow.builder()
     *     .name("Collagen_Classifier")
     *     .config(trainingConfig)
     *     .channels(channelConfig)
     *     .classes(List.of("Background", "Collagen"))
     *     .build()
     *     .run();
     * }</pre>
     *
     * @return a new TrainingBuilder
     */
    public static TrainingBuilder builder() {
        return new TrainingBuilder();
    }

    /**
     * Builder for configuring headless training runs.
     */
    public static class TrainingBuilder {
        private String name;
        private String description = "";
        private TrainingConfig config;
        private ChannelConfiguration channels;
        private List<String> classes;
        private ImageData<BufferedImage> imageData;

        private TrainingBuilder() {}

        /** Sets the classifier name (required). */
        public TrainingBuilder name(String name) {
            this.name = name;
            return this;
        }

        /** Sets the classifier description (optional, defaults to empty). */
        public TrainingBuilder description(String description) {
            this.description = description;
            return this;
        }

        /** Sets the training configuration (required). */
        public TrainingBuilder config(TrainingConfig config) {
            this.config = config;
            return this;
        }

        /** Sets the channel configuration (required). */
        public TrainingBuilder channels(ChannelConfiguration channels) {
            this.channels = channels;
            return this;
        }

        /** Sets the class names for training (required, minimum 2). */
        public TrainingBuilder classes(List<String> classes) {
            this.classes = classes;
            return this;
        }

        /**
         * Sets the image data to use. If not provided, falls back to
         * {@code QP.getCurrentImageData()} at run time.
         */
        public TrainingBuilder imageData(ImageData<BufferedImage> imageData) {
            this.imageData = imageData;
            return this;
        }

        /**
         * Validates parameters and builds a {@link TrainingRunner}.
         *
         * @return a runner ready to execute training
         * @throws IllegalStateException if required parameters are missing
         */
        public TrainingRunner build() {
            Objects.requireNonNull(name, "Classifier name is required");
            Objects.requireNonNull(config, "TrainingConfig is required");
            Objects.requireNonNull(channels, "ChannelConfiguration is required");
            if (classes == null || classes.size() < 2) {
                throw new IllegalStateException("At least 2 class names are required");
            }
            return new TrainingRunner(name, description, config, channels, classes, imageData);
        }
    }

    /**
     * Executes training synchronously without GUI dependencies.
     */
    public static class TrainingRunner {
        private final String name;
        private final String description;
        private final TrainingConfig config;
        private final ChannelConfiguration channels;
        private final List<String> classes;
        private final ImageData<BufferedImage> imageData;

        private TrainingRunner(String name, String description,
                               TrainingConfig config, ChannelConfiguration channels,
                               List<String> classes, ImageData<BufferedImage> imageData) {
            this.name = name;
            this.description = description;
            this.config = config;
            this.channels = channels;
            this.classes = new ArrayList<>(classes);
            this.imageData = imageData;
        }

        /**
         * Runs training synchronously and returns the result.
         *
         * @return the training result
         */
        public TrainingResult run() {
            ImageData<BufferedImage> imgData = this.imageData;
            if (imgData == null) {
                imgData = QP.getCurrentImageData();
            }
            if (imgData == null) {
                logger.warn("No image data available for training");
                return new TrainingResult(null, name, 0.0, 0.0, 0, 0.0, 0, false,
                        "No image data available");
            }

            ClassifierHandler handler = ClassifierRegistry.getHandler(config.getModelType())
                    .orElse(ClassifierRegistry.getDefaultHandler());

            return trainCore(name, description, handler, config, channels, classes,
                    imgData, null, null);
        }
    }

    /**
     * Starts the training workflow.
     * <p>
     * Quick prerequisites (project check) run on the FX thread. The backend
     * health check runs asynchronously because Appose initialization may
     * take time on first launch (building the pixi environment, importing
     * PyTorch, etc.).
     */
    public void start() {
        logger.info("Starting training workflow");

        // Quick prerequisite: project must be open (instant check on FX thread)
        if (qupath.getProject() == null) {
            showError("No Project",
                    "A QuPath project must be open to train a classifier.\n\n" +
                    "Classifiers are saved within the project, and training data\n" +
                    "is exported from project images.\n\n" +
                    "Please create or open a project first.");
            return;
        }

        // Backend health check may block while Appose initializes.
        // Run it on a background thread with a status notification.
        Dialogs.showInfoNotification("DL Pixel Classifier",
                "Connecting to classification backend...");

        CompletableFuture.supplyAsync(() -> DLClassifierChecks.checkServerHealth())
                .thenAcceptAsync(healthy -> {
                    if (healthy) {
                        showTrainingDialog();
                    } else {
                        String versionWarning = ApposeClassifierBackend.getVersionWarning();
                        if (versionWarning != null && !versionWarning.isEmpty()) {
                            showError("Python Environment Update Required",
                                    "The Python environment is out of date and must be rebuilt.\n\n" +
                                    "Go to Extensions > DL Pixel Classifier > Rebuild Python Environment\n" +
                                    "to update. Training is disabled until the environment matches\n" +
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
     * Shows the training configuration dialog.
     */
    private void showTrainingDialog() {
        TrainingDialog.showDialog()
                .thenAccept(result -> {
                    if (result != null) {
                        logger.info("Training dialog completed. Classifier: {}", result.classifierName());

                        // Get the classifier handler
                        ClassifierHandler handler = ClassifierRegistry.getHandler(
                                result.trainingConfig().getModelType())
                                .orElse(ClassifierRegistry.getDefaultHandler());

                        // Start training with progress monitor
                        trainClassifierWithProgress(
                                result.classifierName(),
                                result.description(),
                                handler,
                                result.trainingConfig(),
                                result.channelConfig(),
                                result.selectedClasses(),
                                result.selectedImages(),
                                result.classColors(),
                                result.handlerParameters()
                        );
                    }
                })
                .exceptionally(ex -> {
                    // User cancelling the dialog produces a CancellationException - not an error
                    Throwable cause = ex instanceof java.util.concurrent.CompletionException ? ex.getCause() : ex;
                    if (cause instanceof java.util.concurrent.CancellationException) {
                        logger.info("Training dialog cancelled by user");
                    } else {
                        logger.error("Training dialog failed", ex);
                        showError("Error", "Failed to show training dialog: " + ex.getMessage());
                    }
                    return null;
                });
    }

    /**
     * Executes the training process with progress monitoring.
     *
     * @param classifierName name for the classifier
     * @param description    classifier description
     * @param handler        the classifier handler
     * @param trainingConfig training configuration
     * @param channelConfig  channel configuration
     * @param classNames     list of class names
     * @param selectedImages project images to train from, or null for current image only
     * @param classColors    map of class name to packed RGB color for chart styling, or null
     */
    public void trainClassifierWithProgress(String classifierName,
                                            String description,
                                            ClassifierHandler handler,
                                            TrainingConfig trainingConfig,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames,
                                            List<ProjectImageEntry<BufferedImage>> selectedImages,
                                            Map<String, Integer> classColors,
                                            Map<String, Object> handlerParameters) {
        // Check for unsaved changes before training
        if (!checkUnsavedChanges(selectedImages)) {
            return;
        }

        // Warn user if training will run on CPU (very slow)
        try {
            ApposeService appose = ApposeService.getInstance();
            if (appose.isAvailable() && !appose.isGpuAvailable()) {
                boolean isMac = System.getProperty("os.name", "").toLowerCase().contains("mac");
                String message;
                if (isMac) {
                    message = "No GPU acceleration detected. Training on CPU will be very slow "
                            + "(potentially hours instead of minutes).\n\n"
                            + "If you have Apple Silicon, try rebuilding the DL environment\n"
                            + "to enable MPS acceleration.\n\n"
                            + "Continue training on CPU?";
                } else {
                    message = "No GPU (CUDA) detected. Training on CPU will be very slow "
                            + "(potentially hours instead of minutes).\n\n"
                            + "To enable GPU acceleration:\n"
                            + "  1. Install or update NVIDIA GPU drivers\n"
                            + "  2. Use Extensions > DL Pixel Classifier > Utilities >\n"
                            + "     Rebuild DL Environment\n\n"
                            + "Continue training on CPU?";
                }
                boolean proceed = Dialogs.showConfirmDialog("CPU Training Warning", message);
                if (!proceed) {
                    return;
                }
            }
        } catch (Exception e) {
            // Don't block training if we can't check GPU status
            logger.debug("Could not check GPU status: {}", e.getMessage());
        }

        // Generate classifierId early so model files can be saved directly
        // to the project directory during training (not just at the end).
        String classifierId = classifierName.toLowerCase().replaceAll("[^a-z0-9_-]", "_")
                + "_" + System.currentTimeMillis();

        // Create project-local model output directory
        Path modelOutputDir = null;
        try {
            var project = qupath.getProject();
            if (project != null) {
                modelOutputDir = project.getPath().getParent()
                        .resolve("classifiers/dl")
                        .resolve(classifierId);
                Files.createDirectories(modelOutputDir);
                trainingConfig.setModelOutputDir(modelOutputDir.toString());
                logger.info("Model output directory: {}", modelOutputDir);
            }
        } catch (IOException e) {
            logger.warn("Could not create project model directory, using default: {}", e.getMessage());
        }
        // Capture for use in closures
        final Path finalModelOutputDir = modelOutputDir;
        final String finalClassifierId = classifierId;

        // Create progress monitor
        ProgressMonitorController progress = ProgressMonitorController.forTraining();
        if (classColors != null && !classColors.isEmpty()) {
            progress.setClassColors(classColors);
        }
        progress.setOnCancel(v -> logger.info("Training cancellation requested"));
        progress.show();

        // Shared state for pause/resume and review
        final String[] currentJobId = {null};
        final Path[] trainingDataPathHolder = {null};
        final String[] modelPathHolder = {null};

        // Wire review training areas callback
        progress.setOnReviewTrainingAreas(v -> {
            Path dataPath = trainingDataPathHolder[0];
            String modelPath = modelPathHolder[0];
            if (dataPath == null || modelPath == null) {
                progress.log("Cannot review: training data or model path not available");
                return;
            }
            // Build input config map for evaluation
            Map<String, Object> inputConfig = ApposeClassifierBackend.buildInputConfig(channelConfig);
            CompletableFuture.runAsync(() -> {
                try {
                    progress.setStatus("Evaluating training tiles...");
                    progress.log("Starting post-training evaluation...");

                    ClassifierBackend backend = BackendFactory.getBackend();
                    List<ClassifierClient.TileEvaluationResult> results = backend.evaluateTiles(
                            Path.of(modelPath),
                            dataPath,
                            trainingConfig.getModelType(),
                            trainingConfig.getBackbone(),
                            inputConfig,
                            classNames,
                            classColors,
                            evalProgress -> {
                                progress.setDetail(evalProgress.message());
                                double pct = evalProgress.totalTiles() > 0
                                        ? (double) evalProgress.currentTile() / evalProgress.totalTiles()
                                        : -1;
                                progress.setOverallProgress(pct);
                            },
                            () -> false
                    );

                    progress.setStatus("Complete");
                    progress.setOverallProgress(1.0);
                    progress.log("Evaluation complete: " + results.size() + " tiles analyzed");

                    // Open the results dialog on the FX thread
                    Platform.runLater(() -> {
                        TrainingAreaIssuesDialog dialog = new TrainingAreaIssuesDialog(
                                classifierName, results, trainingConfig.getDownsample(),
                                trainingConfig.getTileSize(),
                                classColors);
                        dialog.show();
                    });
                } catch (Exception e) {
                    logger.error("Tile evaluation failed", e);
                    progress.log("ERROR: Evaluation failed: " + e.getMessage());
                    progress.setStatus("Complete");
                }
            });
        });

        // Wire pause callback
        progress.setOnPause(v -> {
            if (currentJobId[0] != null) {
                try {
                    ClassifierBackend backend = BackendFactory.getBackend();
                    backend.pauseTraining(currentJobId[0]);
                } catch (Exception e) {
                    logger.error("Failed to pause training", e);
                    progress.log("ERROR: Failed to pause: " + e.getMessage());
                }
            }
        });

        // Wire resume callback -- uses the SAME classifierId/modelOutputDir
        progress.setOnResume(v -> {
            CompletableFuture.runAsync(() -> handleResume(
                    currentJobId[0], classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    selectedImages, progress, currentJobId,
                    finalClassifierId, finalModelOutputDir));
        });

        // Wire complete-early callback -- uses the SAME classifierId/modelOutputDir
        progress.setOnCompleteEarly(v -> {
            CompletableFuture.runAsync(() -> handleCompleteEarly(
                    currentJobId[0], classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    selectedImages, progress, classColors,
                    finalClassifierId, finalModelOutputDir));
        });

        // Wire continue-training callback -- resumes from completion checkpoint
        progress.setOnContinueTraining(v -> {
            CompletableFuture.runAsync(() -> handleResume(
                    currentJobId[0], classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    selectedImages, progress, currentJobId,
                    finalClassifierId, finalModelOutputDir));
        });

        // Suspend overlay during training to prevent Appose "thread death" races
        // from concurrent tile requests and to free GPU memory for training.
        OverlayService overlayService = OverlayService.getInstance();
        overlayService.suspendForTraining();

        CompletableFuture.runAsync(() -> {
            try {
                TrainingResult result = trainCore(classifierName, description, handler,
                        trainingConfig, channelConfig, classNames,
                        null, selectedImages, progress, currentJobId,
                        classColors, finalClassifierId, finalModelOutputDir,
                        trainingDataPathHolder, modelPathHolder, handlerParameters);

                if (result.success()) {
                    String completionMsg;
                    if (result.message() != null && result.message().contains("cancelled")) {
                        completionMsg = result.message();
                    } else {
                        completionMsg = String.format(
                                "Classifier trained successfully!\nBest model: epoch %d\n"
                                + "Loss: %.4f | Accuracy: %.2f%% | mIoU: %.4f",
                                result.bestEpoch(), result.finalLoss(),
                                result.finalAccuracy() * 100, result.bestMeanIoU());
                    }
                    progress.complete(true, completionMsg);
                } else if (result.message() != null && result.message().contains("paused")) {
                    // Paused state is handled by showPausedState - don't close
                    logger.info("Training paused, waiting for user action");
                } else {
                    progress.complete(false, result.message());
                    // Clean up training data on failure (not needed for review)
                    cleanupTempDir(trainingDataPathHolder[0]);
                    trainingDataPathHolder[0] = null;
                }
            } finally {
                overlayService.resumeAfterTraining();
            }
        });

        // Clean up training data when the progress dialog is closed
        progress.getStage().setOnHidden(e -> {
            Path dataPath = trainingDataPathHolder[0];
            if (dataPath != null) {
                cleanupTempDir(dataPath);
                trainingDataPathHolder[0] = null;
                logger.info("Cleaned up training data on dialog close");
            }
        });
    }

    /**
     * Core training logic shared by GUI and headless paths.
     * <p>
     * When {@code progress} is {@code null}, progress updates and cancellation
     * checks are skipped, enabling headless execution.
     *
     * @param classifierName name for the classifier
     * @param description    classifier description
     * @param handler        the classifier handler
     * @param trainingConfig training configuration
     * @param channelConfig  channel configuration
     * @param classNames     list of class names
     * @param imageData      image data for extracting training patches (single-image mode)
     * @param selectedImages project images for multi-image training, or null for single-image
     * @param progress       progress monitor (nullable for headless execution)
     * @return the training result
     */
    static TrainingResult trainCore(String classifierName,
                                    String description,
                                    ClassifierHandler handler,
                                    TrainingConfig trainingConfig,
                                    ChannelConfiguration channelConfig,
                                    List<String> classNames,
                                    ImageData<BufferedImage> imageData,
                                    List<ProjectImageEntry<BufferedImage>> selectedImages,
                                    ProgressMonitorController progress) {
        return trainCore(classifierName, description, handler, trainingConfig,
                channelConfig, classNames, imageData, selectedImages, progress, null, null);
    }

    /**
     * Core training logic shared by GUI and headless paths.
     *
     * @param classifierName name for the classifier
     * @param description    classifier description
     * @param handler        the classifier handler
     * @param trainingConfig training configuration
     * @param channelConfig  channel configuration
     * @param classNames     list of class names
     * @param imageData      image data for extracting training patches (single-image mode)
     * @param selectedImages project images for multi-image training, or null for single-image
     * @param progress       progress monitor (nullable for headless execution)
     * @param jobIdHolder    optional array to receive the job ID (element 0 is set)
     * @param classColors    map of class name to packed RGB color, or null
     * @return the training result
     */
    static TrainingResult trainCore(String classifierName,
                                    String description,
                                    ClassifierHandler handler,
                                    TrainingConfig trainingConfig,
                                    ChannelConfiguration channelConfig,
                                    List<String> classNames,
                                    ImageData<BufferedImage> imageData,
                                    List<ProjectImageEntry<BufferedImage>> selectedImages,
                                    ProgressMonitorController progress,
                                    String[] jobIdHolder,
                                    Map<String, Integer> classColors) {
        return trainCore(classifierName, description, handler, trainingConfig,
                channelConfig, classNames, imageData, selectedImages, progress,
                jobIdHolder, classColors, null, null);
    }

    /**
     * Core training logic shared by GUI and headless paths.
     *
     * @param classifierName  name for the classifier
     * @param description     classifier description
     * @param handler         the classifier handler
     * @param trainingConfig  training configuration
     * @param channelConfig   channel configuration
     * @param classNames      list of class names
     * @param imageData       image data for extracting training patches (single-image mode)
     * @param selectedImages  project images for multi-image training, or null for single-image
     * @param progress        progress monitor (nullable for headless execution)
     * @param jobIdHolder     optional array to receive the job ID (element 0 is set)
     * @param classColors     map of class name to packed RGB color, or null
     * @param classifierId    pre-generated classifier ID, or null to generate at save time
     * @param modelOutputDir  project-local model output directory, or null for default
     * @return the training result
     */
    static TrainingResult trainCore(String classifierName,
                                    String description,
                                    ClassifierHandler handler,
                                    TrainingConfig trainingConfig,
                                    ChannelConfiguration channelConfig,
                                    List<String> classNames,
                                    ImageData<BufferedImage> imageData,
                                    List<ProjectImageEntry<BufferedImage>> selectedImages,
                                    ProgressMonitorController progress,
                                    String[] jobIdHolder,
                                    Map<String, Integer> classColors,
                                    String classifierId,
                                    Path modelOutputDir) {
        return trainCore(classifierName, description, handler, trainingConfig,
                channelConfig, classNames, imageData, selectedImages, progress,
                jobIdHolder, classColors, classifierId, modelOutputDir, null, null, null);
    }

    /**
     * Core training logic shared by GUI and headless paths.
     *
     * @param classifierName        name for the classifier
     * @param description           classifier description
     * @param handler               the classifier handler
     * @param trainingConfig        training configuration
     * @param channelConfig         channel configuration
     * @param classNames            list of class names
     * @param imageData             image data for single-image mode
     * @param selectedImages        project images for multi-image training, or null
     * @param progress              progress monitor (nullable for headless execution)
     * @param jobIdHolder           optional array to receive the job ID
     * @param classColors           map of class name to packed RGB color, or null
     * @param classifierId          pre-generated classifier ID, or null
     * @param modelOutputDir        project-local model output directory, or null
     * @param trainingDataPathHolder if non-null, element 0 is set to the training data temp dir;
     *                               caller is responsible for cleanup. If null, tempDir is cleaned up.
     * @param modelPathHolder       if non-null, element 0 is set to the saved model path on success
     * @param handlerParameters     handler-specific parameters (e.g. MAE pretraining config), or null
     * @return the training result
     */
    static TrainingResult trainCore(String classifierName,
                                    String description,
                                    ClassifierHandler handler,
                                    TrainingConfig trainingConfig,
                                    ChannelConfiguration channelConfig,
                                    List<String> classNames,
                                    ImageData<BufferedImage> imageData,
                                    List<ProjectImageEntry<BufferedImage>> selectedImages,
                                    ProgressMonitorController progress,
                                    String[] jobIdHolder,
                                    Map<String, Integer> classColors,
                                    String classifierId,
                                    Path modelOutputDir,
                                    Path[] trainingDataPathHolder,
                                    String[] modelPathHolder,
                                    Map<String, Object> handlerParameters) {
        Path tempDir = null;
        try {
            if (progress != null) {
                progress.setStatus("Exporting training patches...");
                progress.setCurrentProgress(-1);
            }
            logger.info("Starting training for classifier: {}", classifierName);
            if (progress != null) progress.log("Starting training for classifier: " + classifierName);

            // Export training data (use configured export directory if set)
            String exportDirPref = DLClassifierPreferences.getTrainingExportDir();
            if (exportDirPref != null && !exportDirPref.isEmpty()) {
                Path exportBase = Path.of(exportDirPref);
                Files.createDirectories(exportBase);
                tempDir = Files.createTempDirectory(exportBase, "dl-training");
            } else {
                tempDir = Files.createTempDirectory("dl-training");
            }
            logger.info("Exporting training data to: {}", tempDir);
            if (progress != null) progress.log("Export directory: " + tempDir);
            if (trainingDataPathHolder != null && trainingDataPathHolder.length > 0) {
                trainingDataPathHolder[0] = tempDir;
            }

            Map<String, Double> weightMultipliers = trainingConfig.getClassWeightMultipliers();

            // Compute effective tile size (whole-image mode uses actual image dimensions)
            // For ViT models, cap at the handler's max supported tile size since
            // global self-attention is O(n^2) in patch count and huge tiles
            // produce too many patches for the model to learn from effectively.
            List<Integer> supportedSizes = handler.getSupportedTileSizes();
            int maxHandlerTileSize = supportedSizes.isEmpty()
                    ? Integer.MAX_VALUE
                    : supportedSizes.stream().mapToInt(Integer::intValue).max().orElse(Integer.MAX_VALUE);

            int effectiveTileSize;
            if (trainingConfig.isWholeImage()) {
                int maxW = 0, maxH = 0;
                if (selectedImages != null && !selectedImages.isEmpty()) {
                    for (ProjectImageEntry<BufferedImage> entry : selectedImages) {
                        try {
                            ImageData<BufferedImage> entryData = entry.readImageData();
                            maxW = Math.max(maxW, entryData.getServer().getWidth());
                            maxH = Math.max(maxH, entryData.getServer().getHeight());
                            entryData.getServer().close();
                        } catch (Exception e) {
                            logger.warn("Could not read dimensions for {}: {}",
                                    entry.getImageName(), e.getMessage());
                        }
                    }
                } else if (imageData != null) {
                    maxW = imageData.getServer().getWidth();
                    maxH = imageData.getServer().getHeight();
                }

                int uncappedSize = trainingConfig.computeEffectiveTileSize(maxW, maxH);
                effectiveTileSize = trainingConfig.computeEffectiveTileSize(
                        maxW, maxH, maxHandlerTileSize);

                if (effectiveTileSize < uncappedSize) {
                    logger.warn("Whole-image tile size capped from {}px to {}px "
                            + "(max for {} architecture). The image will be tiled "
                            + "instead of processed whole.",
                            uncappedSize, effectiveTileSize, handler.getType());
                    if (progress != null) {
                        progress.log(String.format(
                                "Whole-image mode: tile size capped at %dpx (max for %s). "
                                + "Image (%dx%d) will be tiled -- ViT self-attention is O(n^2) "
                                + "and %dpx would create too many patches to learn from.",
                                effectiveTileSize, handler.getDisplayName(),
                                maxW, maxH, uncappedSize));
                    }
                } else {
                    logger.info("Whole-image mode: max dimensions {}x{}, effective tile size {}",
                            maxW, maxH, effectiveTileSize);
                    if (progress != null) {
                        progress.log("Whole-image mode: effective tile size " + effectiveTileSize
                                + "px (from " + maxW + "x" + maxH + " image)");
                    }
                }
            } else {
                effectiveTileSize = trainingConfig.getTileSize();
            }

            // Set the effective tile size on the config so the Python backend receives it
            trainingConfig.setEffectiveTileSize(effectiveTileSize);

            // Auto-reduce batch size for large tiles to prevent GPU OOM
            trainingConfig.adjustBatchForTileSize(effectiveTileSize);
            if (trainingConfig.getBatchSize() == 1 && progress != null
                    && trainingConfig.getGradientAccumulationSteps() > 1) {
                progress.log(String.format(
                        "Large tile size (%dpx): batch reduced to 1, gradient accumulation %d "
                        + "(effective batch %d)",
                        effectiveTileSize, trainingConfig.getGradientAccumulationSteps(),
                        trainingConfig.getGradientAccumulationSteps()));
            }

            // Compute context padding: real image data around each tile to match inference geometry.
            // Disabled for whole-image mode (no surrounding data available).
            int contextPadding = trainingConfig.isWholeImage() ? 0
                    : computeTrainingContextPadding(effectiveTileSize, trainingConfig);
            if (contextPadding > 0) {
                int paddedSize = effectiveTileSize + 2 * contextPadding;
                logger.info("Context padding: {}px per side (tiles will be {}x{})",
                        contextPadding, paddedSize, paddedSize);
                if (progress != null) {
                    progress.log("Context padding: " + contextPadding + "px per side (tiles will be "
                            + paddedSize + "x" + paddedSize + ")");
                }
            }

            int patchCount;
            if (selectedImages != null && !selectedImages.isEmpty()) {
                // Multi-image project export
                if (progress != null) {
                    progress.setStatus("Exporting patches from " + selectedImages.size() + " images...");
                    progress.log("Exporting from " + selectedImages.size() + " project images...");
                }
                AnnotationExtractor.ExportResult exportResult = AnnotationExtractor.exportFromProject(
                        selectedImages,
                        effectiveTileSize,
                        channelConfig,
                        classNames,
                        tempDir,
                        trainingConfig.getValidationSplit(),
                        trainingConfig.getLineStrokeWidth(),
                        weightMultipliers,
                        trainingConfig.getDownsample(),
                        trainingConfig.getContextScale(),
                        contextPadding
                );
                patchCount = exportResult.totalPatches();
            } else {
                // Single-image export
                AnnotationExtractor extractor = new AnnotationExtractor(
                        imageData,
                        effectiveTileSize,
                        channelConfig,
                        trainingConfig.getLineStrokeWidth(),
                        trainingConfig.getDownsample(),
                        trainingConfig.getContextScale(),
                        contextPadding
                );
                AnnotationExtractor.ExportResult exportResult = extractor.exportTrainingData(
                        tempDir, classNames, trainingConfig.getValidationSplit(), weightMultipliers);
                patchCount = exportResult.totalPatches();
            }
            if (progress != null) {
                progress.log("Exported " + patchCount + " training patches");
                progress.setStatus("Exported " + patchCount + " patches. Connecting to backend...");
            }

            if (progress != null && progress.isCancelled()) {
                return new TrainingResult(null, classifierName, 0, 0, 0, 0.0, 0, false,
                        "Training cancelled by user");
            }

            // Get appropriate backend (Appose or HTTP) and start training
            ClassifierBackend backend = BackendFactory.getBackend();
            if (progress != null) {
                progress.log("Connected to classification backend");
            }

            if (progress != null) {
                // Log frozen layer configuration
                List<String> frozen = trainingConfig.getFrozenLayers();
                if (frozen != null && !frozen.isEmpty()) {
                    progress.log("Transfer learning: " + frozen.size() + " layer groups frozen: "
                            + String.join(", ", frozen));
                } else if (trainingConfig.isUsePretrainedWeights()) {
                    progress.log("Transfer learning: pretrained weights loaded, all layers trainable");
                }

                progress.setStatus("Building model and starting first epoch...");
                progress.setOverallProgress(0);
            }

            final AtomicInteger lastLoggedEpoch = new AtomicInteger(-1);

            ClassifierClient.TrainingResult serverResult = backend.startTraining(
                    trainingConfig,
                    channelConfig,
                    classNames,
                    tempDir,
                    trainingProgress -> {
                        if (progress != null && progress.isCancelled()) {
                            return;
                        }
                        if (progress != null) {
                            // Handle setup phase updates (before training loop starts)
                            if (trainingProgress.isSetupPhase()) {
                                // "initializing" = first update with device info
                                if ("initializing".equals(trainingProgress.status())) {
                                    String deviceMsg = formatDeviceMessage(
                                            trainingProgress.device(), trainingProgress.deviceInfo());
                                    progress.log(deviceMsg);
                                    progress.setStatus("Initializing model for "
                                            + trainingProgress.totalEpochs() + " epoch run...");
                                } else if ("training_config".equals(trainingProgress.setupPhase())) {
                                    // Training configuration summary
                                    progress.log("--- Training Configuration ---");
                                    var config = trainingProgress.configSummary();
                                    if (config != null) {
                                        for (var entry : config.entrySet()) {
                                            progress.log("  " + entry.getKey() + ": " + entry.getValue());
                                        }
                                    }
                                } else if ("training_batch".equals(trainingProgress.setupPhase())) {
                                    // Batch-level progress within an epoch (for long epochs on slow devices)
                                    var cfg = trainingProgress.configSummary();
                                    if (cfg != null) {
                                        String batch = cfg.getOrDefault("batch", "?");
                                        String totalBatches = cfg.getOrDefault("total_batches", "?");
                                        String batchEpoch = cfg.getOrDefault("epoch", "?");
                                        String totalEp = cfg.getOrDefault("total_epochs", String.valueOf(trainingProgress.totalEpochs()));
                                        String elapsed = cfg.getOrDefault("elapsed_seconds", "");
                                        String elapsedStr = elapsed.isEmpty() ? "" : " (" + elapsed + "s)";
                                        progress.setStatus(String.format("Epoch %s/%s - batch %s/%s%s",
                                                batchEpoch, totalEp, batch, totalBatches, elapsedStr));
                                        progress.setDetail(String.format("Batch %s/%s - loss: %s",
                                                batch, totalBatches, cfg.getOrDefault("batch_loss", "?")));
                                    }
                                } else {
                                    // "setup" = granular phase updates during setup
                                    progress.setStatus(formatSetupPhase(trainingProgress.setupPhase()));
                                }
                                return;
                            }

                            // Update status on first real epoch
                            if (lastLoggedEpoch.get() < 0) {
                                progress.setStatus("Training (" + trainingProgress.totalEpochs() + " epochs)...");
                            }

                            // Always update progress bar and detail text (lightweight, keeps UI responsive)
                            double progressValue = (double) trainingProgress.epoch() / trainingProgress.totalEpochs();
                            progress.setOverallProgress(progressValue);
                            progress.setDetail(String.format("Epoch %d/%d - Loss: %.4f - mIoU: %.4f",
                                    trainingProgress.epoch(), trainingProgress.totalEpochs(),
                                    trainingProgress.loss(), trainingProgress.meanIoU()));

                            // Only log and update charts once per epoch
                            int currentEpoch = trainingProgress.epoch();
                            if (lastLoggedEpoch.getAndSet(currentEpoch) < currentEpoch) {
                                progress.updateTrainingMetrics(
                                        trainingProgress.epoch(),
                                        trainingProgress.loss(),
                                        trainingProgress.valLoss(),
                                        trainingProgress.perClassIoU(),
                                        trainingProgress.perClassLoss()
                                );

                                // Log with per-class breakdown
                                // Use scientific notation for very small train_loss
                                // to avoid misleading 0.0000 (overfitting small dataset)
                                String tlFmt = (trainingProgress.loss() > 0 && trainingProgress.loss() < 0.0001)
                                        ? String.format("%.2e", trainingProgress.loss())
                                        : String.format("%.4f", trainingProgress.loss());
                                StringBuilder logMsg = new StringBuilder();
                                logMsg.append(String.format(
                                        "Epoch %d: train_loss=%s, val_loss=%.4f, acc=%.1f%%, mIoU=%.4f",
                                        trainingProgress.epoch(), tlFmt,
                                        trainingProgress.valLoss(), trainingProgress.accuracy() * 100,
                                        trainingProgress.meanIoU()));
                                progress.log(logMsg.toString());

                                if (trainingProgress.perClassIoU() != null && !trainingProgress.perClassIoU().isEmpty()) {
                                    StringBuilder iouLine = new StringBuilder("  IoU:");
                                    for (var entry : trainingProgress.perClassIoU().entrySet()) {
                                        iouLine.append(String.format(" %s=%.3f", entry.getKey(), entry.getValue()));
                                    }
                                    progress.log(iouLine.toString());
                                }
                                if (trainingProgress.perClassLoss() != null && !trainingProgress.perClassLoss().isEmpty()) {
                                    StringBuilder lossLine = new StringBuilder("  Loss:");
                                    for (var entry : trainingProgress.perClassLoss().entrySet()) {
                                        lossLine.append(String.format(" %s=%.4f", entry.getKey(), entry.getValue()));
                                    }
                                    progress.log(lossLine.toString());
                                }
                            }
                        }
                    },
                    () -> progress != null && progress.isCancelled(),
                    jobId -> {
                        if (jobIdHolder != null && jobIdHolder.length > 0) {
                            jobIdHolder[0] = jobId;
                        }
                    }
            );

            if (serverResult.isPaused()) {
                if (progress != null) {
                    progress.log("Training paused at epoch " + serverResult.lastEpoch());
                    progress.showPausedState(serverResult.lastEpoch(), serverResult.totalEpochs());
                }
                return new TrainingResult(null, classifierName, 0, 0, 0, 0.0, 0, false,
                        "Training paused at epoch " + serverResult.lastEpoch());
            }

            if (serverResult.isCancelled()) {
                ProgressMonitorController.CancelSaveMode saveMode = progress != null
                        ? progress.getCancelSaveMode()
                        : ProgressMonitorController.CancelSaveMode.DO_NOT_SAVE;

                if (saveMode == ProgressMonitorController.CancelSaveMode.DO_NOT_SAVE
                        || !serverResult.isCancelledWithSave()) {
                    return new TrainingResult(null, classifierName, 0, 0, 0, 0.0, 0, false,
                            "Training cancelled by user");
                }

                // User chose to save -- pick the appropriate model path
                String savedModelPath = (saveMode == ProgressMonitorController.CancelSaveMode.LAST_EPOCH
                        && serverResult.lastModelPath() != null)
                        ? serverResult.lastModelPath()
                        : serverResult.modelPath();

                String epochLabel = (saveMode == ProgressMonitorController.CancelSaveMode.LAST_EPOCH)
                        ? "last epoch" : "best epoch (" + serverResult.bestEpoch() + ")";
                logger.info("Saving cancelled training ({}) from: {}", epochLabel, savedModelPath);
                if (progress != null) {
                    progress.log("Saving " + epochLabel + " model...");
                    progress.setStatus("Saving classifier...");
                }

                String effectiveId = classifierId != null ? classifierId
                        : classifierName.toLowerCase().replaceAll("[^a-z0-9_-]", "_")
                          + "_" + System.currentTimeMillis();
                List<ClassifierMetadata.ClassInfo> classInfoList = buildClassInfoList(classNames, classColors);
                ClassifierMetadata cancelMetadata = ClassifierMetadata.builder()
                        .id(effectiveId)
                        .name(classifierName)
                        .description(description)
                        .modelType(trainingConfig.getModelType())
                        .backbone(trainingConfig.getBackbone())
                        .inputChannels(channelConfig.getSelectedChannels().size())
                        .contextScale(trainingConfig.getContextScale())
                        .downsample(trainingConfig.getDownsample())
                        .expectedChannelNames(channelConfig.getChannelNames())
                        .inputSize(effectiveTileSize, effectiveTileSize)
                        .classes(classInfoList)
                        .normalizationStrategy(channelConfig.getNormalizationStrategy())
                        .bitDepthTrained(channelConfig.getBitDepth())
                        .trainingEpochs(serverResult.lastEpoch())
                        .finalLoss(serverResult.finalLoss())
                        .finalAccuracy(serverResult.finalAccuracy())
                        .trainingSettings(buildTrainingSettingsMap(trainingConfig))
                        .build();
                boolean filesInPlace = modelOutputDir != null;
                ModelManager modelManager = new ModelManager();
                modelManager.saveClassifier(cancelMetadata, Path.of(savedModelPath),
                        true, filesInPlace);
                String msg = "Saved " + epochLabel + " model as " + effectiveId;
                logger.info(msg);
                if (progress != null) progress.log(msg);
                return new TrainingResult(
                        effectiveId, classifierName,
                        serverResult.finalLoss(), serverResult.finalAccuracy(),
                        serverResult.bestEpoch(), serverResult.bestMeanIoU(),
                        serverResult.lastEpoch(), true,
                        "Training cancelled -- " + epochLabel + " model saved"
                );
            }

            logger.info("Training completed. Model saved to: {}", serverResult.modelPath());
            if (progress != null) progress.log("Training completed. Model path: " + serverResult.modelPath());
            if (modelPathHolder != null && modelPathHolder.length > 0) {
                modelPathHolder[0] = serverResult.modelPath();
            }

            // Build and save metadata
            if (progress != null) progress.setStatus("Saving classifier...");

            // Use pre-generated classifierId if provided (GUI path), otherwise generate
            String effectiveId = classifierId != null ? classifierId
                    : classifierName.toLowerCase().replaceAll("[^a-z0-9_-]", "_")
                      + "_" + System.currentTimeMillis();

            List<ClassifierMetadata.ClassInfo> classInfoList = buildClassInfoList(classNames, classColors);

            ClassifierMetadata metadata = ClassifierMetadata.builder()
                    .id(effectiveId)
                    .name(classifierName)
                    .description(description)
                    .modelType(trainingConfig.getModelType())
                    .backbone(trainingConfig.getBackbone())
                    .inputChannels(channelConfig.getSelectedChannels().size())
                    .contextScale(trainingConfig.getContextScale())
                    .downsample(trainingConfig.getDownsample())
                    .expectedChannelNames(channelConfig.getChannelNames())
                    .inputSize(effectiveTileSize, effectiveTileSize)
                    .classes(classInfoList)
                    .normalizationStrategy(channelConfig.getNormalizationStrategy())
                    .bitDepthTrained(channelConfig.getBitDepth())
                    .trainingEpochs(trainingConfig.getEpochs())
                    .finalLoss(serverResult.finalLoss())
                    .finalAccuracy(serverResult.finalAccuracy())
                    .trainingSettings(buildTrainingSettingsMap(trainingConfig))
                    .build();

            // Save the classifier. When modelOutputDir is set, files are already
            // in the project directory -- skip the copy step.
            boolean filesInPlace = modelOutputDir != null;
            ModelManager modelManager = new ModelManager();
            modelManager.saveClassifier(metadata, Path.of(serverResult.modelPath()),
                    true, filesInPlace);
            if (progress != null) progress.log("Classifier saved: " + metadata.getId());

            return new TrainingResult(
                    effectiveId,
                    classifierName,
                    serverResult.finalLoss(),
                    serverResult.finalAccuracy(),
                    serverResult.bestEpoch(),
                    serverResult.bestMeanIoU(),
                    trainingConfig.getEpochs(),
                    true,
                    "Training completed successfully"
            );

        } catch (Exception e) {
            logger.error("Training failed", e);
            if (progress != null) progress.log("ERROR: " + e.getMessage());
            // Clean up partial model directory on failure
            if (modelOutputDir != null) {
                cleanupTempDir(modelOutputDir);
            }
            return new TrainingResult(null, classifierName, 0, 0, 0, 0.0, 0, false,
                    "Training failed: " + e.getMessage());
        } finally {
            // Skip cleanup if caller wants to keep training data for post-training review
            if (trainingDataPathHolder == null) {
                cleanupTempDir(tempDir);
            }
        }
    }

    /**
     * Executes the training process.
     *
     * @param handler       the classifier handler
     * @param trainingConfig training configuration
     * @param channelConfig channel configuration
     * @param classNames    list of class names
     * @deprecated Use {@link #trainClassifierWithProgress} instead
     */
    @Deprecated
    public void trainClassifier(ClassifierHandler handler,
                                TrainingConfig trainingConfig,
                                ChannelConfiguration channelConfig,
                                List<String> classNames) {
        trainClassifierWithProgress("Untitled", "", handler, trainingConfig, channelConfig, classNames, null, null, null);
    }

    /**
     * Handles the resume flow after training has been paused.
     * <p>
     * This method runs on a background thread and uses Platform.runLater
     * for any UI interactions (dialogs, unsaved changes check).
     */
    private void handleResume(String jobId,
                              String classifierName,
                              String description,
                              ClassifierHandler handler,
                              TrainingConfig trainingConfig,
                              ChannelConfiguration channelConfig,
                              List<String> classNames,
                              List<ProjectImageEntry<BufferedImage>> selectedImages,
                              ProgressMonitorController progress,
                              String[] currentJobId,
                              String classifierId,
                              Path modelOutputDir) {
        Path tempDir = null;
        try {
            // 1. Check for unsaved changes (on FX thread)
            CompletableFuture<Boolean> unsavedCheck = new CompletableFuture<>();
            Platform.runLater(() -> {
                boolean proceed = checkUnsavedChanges(selectedImages);
                unsavedCheck.complete(proceed);
            });
            if (!unsavedCheck.get()) {
                // User cancelled -- stay in paused state
                return;
            }

            // 2. Look up how many epochs have already been trained
            int epochsCompleted = 0;
            ClassifierBackend resumeBackend = BackendFactory.getBackend();
            if (resumeBackend instanceof ApposeClassifierBackend apposeBackend) {
                var checkpointInfo = apposeBackend.getCheckpointInfo(jobId);
                if (checkpointInfo != null) {
                    epochsCompleted = checkpointInfo.lastEpoch();
                }
            }

            // 3. Show resume param dialog (on FX thread)
            final int completedEpochs = epochsCompleted;
            CompletableFuture<Optional<ResumeParams>> paramsFuture = new CompletableFuture<>();
            Platform.runLater(() -> {
                Optional<ResumeParams> params = showResumeParamDialog(
                        completedEpochs,
                        trainingConfig.getEpochs(),
                        trainingConfig.getLearningRate(),
                        trainingConfig.getBatchSize());
                paramsFuture.complete(params);
            });
            Optional<ResumeParams> paramsOpt = paramsFuture.get();
            if (paramsOpt.isEmpty()) {
                // User cancelled dialog -- stay in paused state
                return;
            }
            ResumeParams params = paramsOpt.get();

            // 4. Re-export training data
            progress.showResumedState();
            progress.setStatus("Re-exporting training data...");
            progress.log("Re-exporting annotations (includes any new/modified annotations)...");

            String exportDirPref = DLClassifierPreferences.getTrainingExportDir();
            if (exportDirPref != null && !exportDirPref.isEmpty()) {
                Path exportBase = Path.of(exportDirPref);
                Files.createDirectories(exportBase);
                tempDir = Files.createTempDirectory(exportBase, "dl-training-resume");
            } else {
                tempDir = Files.createTempDirectory("dl-training-resume");
            }
            // Compute effective tile size for resume (same logic as trainCore)
            // Cap at handler's max supported tile size for ViT models
            List<Integer> resumeSizes = handler.getSupportedTileSizes();
            int resumeMaxTile = resumeSizes.isEmpty()
                    ? Integer.MAX_VALUE
                    : resumeSizes.stream().mapToInt(Integer::intValue).max().orElse(Integer.MAX_VALUE);

            int effectiveTileSize;
            if (trainingConfig.isWholeImage()) {
                int maxW = 0, maxH = 0;
                if (selectedImages != null && !selectedImages.isEmpty()) {
                    for (ProjectImageEntry<BufferedImage> entry : selectedImages) {
                        try {
                            ImageData<BufferedImage> entryData = entry.readImageData();
                            maxW = Math.max(maxW, entryData.getServer().getWidth());
                            maxH = Math.max(maxH, entryData.getServer().getHeight());
                            entryData.getServer().close();
                        } catch (Exception e) {
                            logger.warn("Could not read dimensions for {}: {}",
                                    entry.getImageName(), e.getMessage());
                        }
                    }
                } else {
                    ImageData<BufferedImage> currentImageData = qupath.getImageData();
                    if (currentImageData != null) {
                        maxW = currentImageData.getServer().getWidth();
                        maxH = currentImageData.getServer().getHeight();
                    }
                }
                effectiveTileSize = trainingConfig.computeEffectiveTileSize(
                        maxW, maxH, resumeMaxTile);
                progress.log("Whole-image mode: effective tile size " + effectiveTileSize + "px");
            } else {
                effectiveTileSize = trainingConfig.getTileSize();
            }

            // Set effective tile size and auto-reduce batch for large tiles
            trainingConfig.setEffectiveTileSize(effectiveTileSize);
            trainingConfig.adjustBatchForTileSize(effectiveTileSize);

            Map<String, Double> resumeMultipliers = trainingConfig.getClassWeightMultipliers();
            int contextPadding = trainingConfig.isWholeImage() ? 0
                    : computeTrainingContextPadding(effectiveTileSize, trainingConfig);
            int patchCount;
            if (selectedImages != null && !selectedImages.isEmpty()) {
                AnnotationExtractor.ExportResult exportResult = AnnotationExtractor.exportFromProject(
                        selectedImages,
                        effectiveTileSize,
                        channelConfig,
                        classNames,
                        tempDir,
                        trainingConfig.getValidationSplit(),
                        trainingConfig.getLineStrokeWidth(),
                        resumeMultipliers,
                        trainingConfig.getDownsample(),
                        trainingConfig.getContextScale(),
                        contextPadding
                );
                patchCount = exportResult.totalPatches();
            } else {
                ImageData<BufferedImage> imageData = qupath.getImageData();
                AnnotationExtractor extractor = new AnnotationExtractor(
                        imageData, effectiveTileSize, channelConfig,
                        trainingConfig.getLineStrokeWidth(), trainingConfig.getDownsample(),
                        trainingConfig.getContextScale(), contextPadding);
                AnnotationExtractor.ExportResult exportResult = extractor.exportTrainingData(
                        tempDir, classNames, trainingConfig.getValidationSplit(), resumeMultipliers);
                patchCount = exportResult.totalPatches();
            }
            progress.log("Re-exported " + patchCount + " training patches");
            progress.setStatus("Resuming training...");

            // Log frozen layer configuration
            List<String> frozen = trainingConfig.getFrozenLayers();
            if (frozen != null && !frozen.isEmpty()) {
                progress.log("Transfer learning: " + frozen.size() + " layer groups frozen: "
                        + String.join(", ", frozen));
            } else if (trainingConfig.isUsePretrainedWeights()) {
                progress.log("Transfer learning: pretrained weights loaded, all layers trainable");
            }

            // 5. Suspend overlay during resumed training to free GPU memory
            OverlayService.getInstance().suspendForTraining();

            // Resume training via backend
            ClassifierBackend backend = BackendFactory.getBackend();

            final AtomicInteger lastLoggedEpochResume = new AtomicInteger(-1);

            ClassifierClient.TrainingResult serverResult = backend.resumeTraining(
                    jobId,
                    tempDir,
                    params.totalEpochs(),
                    params.learningRate(),
                    params.batchSize(),
                    trainingProgress -> {
                        if (progress.isCancelled()) return;

                        // Always update progress bar and detail text (lightweight, keeps UI responsive)
                        double progressValue = (double) trainingProgress.epoch() / trainingProgress.totalEpochs();
                        progress.setOverallProgress(progressValue);
                        progress.setDetail(String.format("Epoch %d/%d - Loss: %.4f - mIoU: %.4f",
                                trainingProgress.epoch(), trainingProgress.totalEpochs(),
                                trainingProgress.loss(), trainingProgress.meanIoU()));

                        // Only log and update charts once per epoch
                        int currentEpoch = trainingProgress.epoch();
                        if (lastLoggedEpochResume.getAndSet(currentEpoch) < currentEpoch) {
                            progress.updateTrainingMetrics(
                                    trainingProgress.epoch(),
                                    trainingProgress.loss(),
                                    trainingProgress.valLoss(),
                                    trainingProgress.perClassIoU(),
                                    trainingProgress.perClassLoss());

                            String tlFmt = (trainingProgress.loss() > 0 && trainingProgress.loss() < 0.0001)
                                    ? String.format("%.2e", trainingProgress.loss())
                                    : String.format("%.4f", trainingProgress.loss());
                            StringBuilder logMsg = new StringBuilder();
                            logMsg.append(String.format(
                                    "Epoch %d: train_loss=%s, val_loss=%.4f, acc=%.1f%%, mIoU=%.4f",
                                    trainingProgress.epoch(), tlFmt,
                                    trainingProgress.valLoss(), trainingProgress.accuracy() * 100,
                                    trainingProgress.meanIoU()));
                            progress.log(logMsg.toString());

                            if (trainingProgress.perClassIoU() != null && !trainingProgress.perClassIoU().isEmpty()) {
                                StringBuilder iouLine = new StringBuilder("  IoU:");
                                for (var entry : trainingProgress.perClassIoU().entrySet()) {
                                    iouLine.append(String.format(" %s=%.3f", entry.getKey(), entry.getValue()));
                                }
                                progress.log(iouLine.toString());
                            }
                            if (trainingProgress.perClassLoss() != null && !trainingProgress.perClassLoss().isEmpty()) {
                                StringBuilder lossLine = new StringBuilder("  Loss:");
                                for (var entry : trainingProgress.perClassLoss().entrySet()) {
                                    lossLine.append(String.format(" %s=%.4f", entry.getKey(), entry.getValue()));
                                }
                                progress.log(lossLine.toString());
                            }
                        }
                    },
                    progress::isCancelled
            );

            // 6. Handle result -- may be paused again, completed, or cancelled
            if (serverResult.isPaused()) {
                progress.log("Training paused again at epoch " + serverResult.lastEpoch());
                progress.showPausedState(serverResult.lastEpoch(), serverResult.totalEpochs());
                currentJobId[0] = serverResult.jobId();
            } else if (serverResult.isCancelled()) {
                ProgressMonitorController.CancelSaveMode resumeSaveMode =
                        progress.getCancelSaveMode();
                if (resumeSaveMode != ProgressMonitorController.CancelSaveMode.DO_NOT_SAVE
                        && serverResult.isCancelledWithSave()) {
                    // Save the model from the resumed training
                    String savedPath = (resumeSaveMode == ProgressMonitorController.CancelSaveMode.LAST_EPOCH
                            && serverResult.lastModelPath() != null)
                            ? serverResult.lastModelPath()
                            : serverResult.modelPath();
                    progress.log("Saving cancelled model from resumed training...");
                    progress.setStatus("Saving classifier...");
                    String rId = classifierId != null ? classifierId
                            : classifierName.toLowerCase().replaceAll("[^a-z0-9_-]", "_")
                              + "_" + System.currentTimeMillis();
                    List<ClassifierMetadata.ClassInfo> rClassInfo = buildClassInfoList(classNames, null);
                    ClassifierMetadata rMeta = ClassifierMetadata.builder()
                            .id(rId).name(classifierName).description(description)
                            .modelType(trainingConfig.getModelType())
                            .backbone(trainingConfig.getBackbone())
                            .inputChannels(channelConfig.getSelectedChannels().size())
                            .contextScale(trainingConfig.getContextScale())
                            .downsample(trainingConfig.getDownsample())
                            .expectedChannelNames(channelConfig.getChannelNames())
                            .inputSize(effectiveTileSize, effectiveTileSize)
                            .classes(rClassInfo)
                            .normalizationStrategy(channelConfig.getNormalizationStrategy())
                            .bitDepthTrained(channelConfig.getBitDepth())
                            .trainingEpochs(serverResult.lastEpoch())
                            .finalLoss(serverResult.finalLoss())
                            .finalAccuracy(serverResult.finalAccuracy())
                            .trainingSettings(buildTrainingSettingsMap(trainingConfig))
                            .build();
                    ModelManager rManager = new ModelManager();
                    rManager.saveClassifier(rMeta, Path.of(savedPath), true, false);
                    progress.complete(true, "Saved model as " + rId);
                } else {
                    progress.complete(false, "Training cancelled by user");
                }
            } else {
                // Completed -- save the classifier
                progress.setStatus("Saving classifier...");
                // Use the same classifierId from the initial training
                String effectiveId = classifierId != null ? classifierId
                        : classifierName.toLowerCase().replaceAll("[^a-z0-9_-]", "_")
                          + "_" + System.currentTimeMillis();

                List<ClassifierMetadata.ClassInfo> classInfoList = buildClassInfoList(classNames, null);

                ClassifierMetadata metadata = ClassifierMetadata.builder()
                        .id(effectiveId)
                        .name(classifierName)
                        .description(description)
                        .modelType(trainingConfig.getModelType())
                        .backbone(trainingConfig.getBackbone())
                        .inputChannels(channelConfig.getSelectedChannels().size())
                        .contextScale(trainingConfig.getContextScale())
                        .downsample(trainingConfig.getDownsample())
                        .expectedChannelNames(channelConfig.getChannelNames())
                        .inputSize(effectiveTileSize, effectiveTileSize)
                        .classes(classInfoList)
                        .normalizationStrategy(channelConfig.getNormalizationStrategy())
                        .bitDepthTrained(channelConfig.getBitDepth())
                        .trainingEpochs(params.totalEpochs())
                        .finalLoss(serverResult.finalLoss())
                        .finalAccuracy(serverResult.finalAccuracy())
                        .trainingSettings(buildTrainingSettingsMap(trainingConfig))
                        .build();

                boolean filesInPlace = modelOutputDir != null;
                ModelManager modelManager = new ModelManager();
                modelManager.saveClassifier(metadata, Path.of(serverResult.modelPath()),
                        true, filesInPlace);
                progress.log("Classifier saved: " + metadata.getId());

                // Update jobId so "Continue Training" can find the new checkpoint
                currentJobId[0] = serverResult.jobId();

                progress.complete(true, String.format(
                        "Classifier trained successfully!\nBest model: epoch %d\n" +
                        "Loss: %.4f | Accuracy: %.2f%% | mIoU: %.4f",
                        serverResult.bestEpoch(), serverResult.finalLoss(),
                        serverResult.finalAccuracy() * 100, serverResult.bestMeanIoU()));
            }

        } catch (Exception e) {
            logger.error("Resume failed", e);
            progress.log("ERROR: Resume failed: " + e.getMessage());
            progress.complete(false, "Resume failed: " + e.getMessage());
        } finally {
            OverlayService.getInstance().resumeAfterTraining();
            cleanupTempDir(tempDir);
        }
    }

    /**
     * Handles the "Complete Training" action from the paused state.
     * Loads the best model from the training checkpoint and saves it as
     * the final classifier.
     */
    private void handleCompleteEarly(String jobId,
                                     String classifierName,
                                     String description,
                                     ClassifierHandler handler,
                                     TrainingConfig trainingConfig,
                                     ChannelConfiguration channelConfig,
                                     List<String> classNames,
                                     List<ProjectImageEntry<BufferedImage>> selectedImages,
                                     ProgressMonitorController progress,
                                     Map<String, Integer> classColors,
                                     String classifierId,
                                     Path modelOutputDir) {
        try {
            ClassifierBackend backend = BackendFactory.getBackend();
            if (!(backend instanceof ApposeClassifierBackend apposeBackend)) {
                progress.complete(false, "Complete-early requires Appose backend");
                return;
            }

            ApposeClassifierBackend.CheckpointInfo checkpoint = apposeBackend.getCheckpointInfo(jobId);
            if (checkpoint == null || checkpoint.path() == null || checkpoint.path().isEmpty()) {
                progress.complete(false, "No checkpoint available for this training job");
                return;
            }

            progress.setStatus("Finalizing model from checkpoint...");
            progress.log("Loading best model from checkpoint...");

            String modelOutputDirStr = modelOutputDir != null ? modelOutputDir.toString() : null;
            ClassifierClient.TrainingResult serverResult =
                    apposeBackend.finalizeTraining(checkpoint.path(), modelOutputDirStr);

            if (serverResult.modelPath() == null || serverResult.modelPath().isEmpty()) {
                progress.complete(false, "Failed to save model from checkpoint");
                return;
            }

            // Save metadata (same pattern as normal completion)
            progress.setStatus("Saving classifier...");
            String effectiveId = classifierId != null ? classifierId
                    : classifierName.toLowerCase().replaceAll("[^a-z0-9_-]", "_")
                      + "_" + System.currentTimeMillis();

            // Compute effective tile size for metadata
            int effectiveTileSize = trainingConfig.getTileSize();
            if (trainingConfig.isWholeImage()) {
                int maxW = 0, maxH = 0;
                if (selectedImages != null && !selectedImages.isEmpty()) {
                    for (ProjectImageEntry<BufferedImage> entry : selectedImages) {
                        try {
                            ImageData<BufferedImage> entryData = entry.readImageData();
                            maxW = Math.max(maxW, entryData.getServer().getWidth());
                            maxH = Math.max(maxH, entryData.getServer().getHeight());
                            entryData.getServer().close();
                        } catch (Exception e) {
                            logger.warn("Could not read dimensions for {}: {}",
                                    entry.getImageName(), e.getMessage());
                        }
                    }
                } else {
                    ImageData<BufferedImage> currentImageData = qupath.getImageData();
                    if (currentImageData != null) {
                        maxW = currentImageData.getServer().getWidth();
                        maxH = currentImageData.getServer().getHeight();
                    }
                }
                effectiveTileSize = trainingConfig.computeEffectiveTileSize(maxW, maxH);
            }

            List<ClassifierMetadata.ClassInfo> classInfoList = buildClassInfoList(classNames, classColors);

            ClassifierMetadata metadata = ClassifierMetadata.builder()
                    .id(effectiveId)
                    .name(classifierName)
                    .description(description)
                    .modelType(trainingConfig.getModelType())
                    .backbone(trainingConfig.getBackbone())
                    .inputChannels(channelConfig.getSelectedChannels().size())
                    .contextScale(trainingConfig.getContextScale())
                    .downsample(trainingConfig.getDownsample())
                    .expectedChannelNames(channelConfig.getChannelNames())
                    .inputSize(effectiveTileSize, effectiveTileSize)
                    .classes(classInfoList)
                    .normalizationStrategy(channelConfig.getNormalizationStrategy())
                    .bitDepthTrained(channelConfig.getBitDepth())
                    .trainingEpochs(checkpoint.lastEpoch())
                    .finalLoss(serverResult.finalLoss())
                    .finalAccuracy(serverResult.finalAccuracy())
                    .trainingSettings(buildTrainingSettingsMap(trainingConfig))
                    .build();

            boolean filesInPlace = modelOutputDir != null;
            ModelManager modelManager = new ModelManager();
            modelManager.saveClassifier(metadata, Path.of(serverResult.modelPath()),
                    true, filesInPlace);
            progress.log("Classifier saved: " + metadata.getId());

            progress.complete(true, String.format(
                    "Training completed early!\nBest model: epoch %d\n"
                    + "Loss: %.4f | Accuracy: %.2f%% | mIoU: %.4f",
                    serverResult.bestEpoch(), serverResult.finalLoss(),
                    serverResult.finalAccuracy() * 100, serverResult.bestMeanIoU()));

        } catch (Exception e) {
            logger.error("Complete early failed", e);
            progress.log("ERROR: " + e.getMessage());
            progress.complete(false, "Failed to complete training: " + e.getMessage());
        }
    }

    /**
     * Shows a dialog to configure resume parameters.
     *
     * @param currentEpochs current total epochs setting
     * @param currentLR     current learning rate
     * @param currentBatch  current batch size
     * @return optional resume params, or empty if user cancelled
     */
    private Optional<ResumeParams> showResumeParamDialog(int epochsCompleted,
                                                          int originalEpochs,
                                                          double currentLR,
                                                          int currentBatch) {
        Dialog<ResumeParams> dialog = new Dialog<>();
        dialog.setTitle("Resume Training");
        dialog.setHeaderText("Adjust parameters for resumed training");
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20));

        int row = 0;

        // Show how many epochs have been completed
        if (epochsCompleted > 0) {
            Label completedLabel = new Label(epochsCompleted + " epochs completed so far");
            completedLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");
            grid.add(completedLabel, 0, row, 2, 1);
            row++;
        }

        // Default additional epochs to the original configured count
        int defaultAdditional = Math.max(originalEpochs, 10);
        Spinner<Integer> epochSpinner = new Spinner<>(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 1000, defaultAdditional));
        epochSpinner.setEditable(true);

        TextField lrField = new TextField(String.valueOf(currentLR));
        lrField.setPrefWidth(120);

        Spinner<Integer> batchSpinner = new Spinner<>(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 128, currentBatch));
        batchSpinner.setEditable(true);

        grid.add(new Label("Additional Epochs:"), 0, row);
        grid.add(epochSpinner, 1, row);
        row++;
        grid.add(new Label("Learning Rate:"), 0, row);
        grid.add(lrField, 1, row);
        row++;
        grid.add(new Label("Batch Size:"), 0, row);
        grid.add(batchSpinner, 1, row);

        dialog.getDialogPane().setContent(grid);

        final int completed = epochsCompleted;
        dialog.setResultConverter(buttonType -> {
            if (buttonType == ButtonType.OK) {
                try {
                    double lr = Double.parseDouble(lrField.getText().trim());
                    int totalEpochs = completed + epochSpinner.getValue();
                    return new ResumeParams(totalEpochs, lr, batchSpinner.getValue());
                } catch (NumberFormatException e) {
                    logger.warn("Invalid learning rate value: {}", lrField.getText());
                    return null;
                }
            }
            return null;
        });

        return dialog.showAndWait();
    }

    /**
     * Checks for unsaved changes and saves them before training.
     * <p>
     * In single-image mode, training uses the live in-memory ImageData, so
     * unsaved annotations are included automatically.
     * <p>
     * In multi-image mode, training reads from saved .qpdata files on disk,
     * so this method auto-saves the current image to ensure the newest
     * annotations are included.
     * <p>
     * This method must be called on the JavaFX Application Thread.
     *
     * @param selectedImages the selected project images, or null for single-image mode
     * @return true if training should proceed, false if user cancelled
     */
    private boolean checkUnsavedChanges(List<ProjectImageEntry<BufferedImage>> selectedImages) {
        ImageData<BufferedImage> currentImageData = qupath.getImageData();
        if (currentImageData == null) {
            return true;
        }

        boolean isMultiImage = selectedImages != null && !selectedImages.isEmpty();
        boolean hasUnsavedChanges = currentImageData.isChanged();

        if (!hasUnsavedChanges) {
            return true;
        }

        if (isMultiImage) {
            // Multi-image mode reads from saved .qpdata files -- auto-save to include latest changes
            var project = qupath.getProject();
            if (project != null) {
                var currentEntry = project.getEntry(currentImageData);
                if (currentEntry != null) {
                    try {
                        logger.info("Auto-saving current image data before multi-image training...");
                        currentEntry.saveImageData(currentImageData);
                        logger.info("Saved current image: {}", currentEntry.getImageName());
                    } catch (Exception e) {
                        logger.error("Failed to auto-save current image data", e);
                        return Dialogs.showConfirmDialog(
                                "Save Failed",
                                "Could not auto-save the current image:\n" +
                                e.getMessage() + "\n\n" +
                                "Unsaved annotation changes will NOT be included\n" +
                                "in multi-image training.\n\n" +
                                "Continue anyway?"
                        );
                    }
                }
            }
            return true;
        } else {
            // Single-image mode uses live in-memory data -- unsaved annotations are included
            logger.info("Current image has unsaved changes - these will be included in single-image training");
            return true;
        }
    }

    /**
     * Builds a map of all training hyperparameters for metadata persistence.
     * This allows loading settings from a previously trained model.
     *
     * @param config the training configuration
     * @return map of parameter names to values
     */
    static Map<String, Object> buildTrainingSettingsMap(TrainingConfig config) {
        Map<String, Object> settings = new LinkedHashMap<>();
        settings.put("learning_rate", config.getLearningRate());
        settings.put("batch_size", config.getBatchSize());
        settings.put("weight_decay", config.getWeightDecay());
        settings.put("validation_split", config.getValidationSplit());
        settings.put("overlap", config.getOverlap());
        settings.put("line_stroke_width", config.getLineStrokeWidth());
        settings.put("use_pretrained_weights", config.isUsePretrainedWeights());
        settings.put("frozen_layers", config.getFrozenLayers());
        settings.put("scheduler_type", config.getSchedulerType());
        settings.put("loss_function", config.getLossFunction());
        String lf = config.getLossFunction();
        if ("focal_dice".equals(lf) || "focal".equals(lf)) {
            settings.put("focal_gamma", config.getFocalGamma());
        }
        if (config.getOhemHardRatio() < 1.0) {
            settings.put("ohem_hard_ratio", config.getOhemHardRatio());
        }
        settings.put("early_stopping_metric", config.getEarlyStoppingMetric());
        settings.put("early_stopping_patience", config.getEarlyStoppingPatience());
        settings.put("mixed_precision", config.isMixedPrecision());
        settings.put("augmentation_config", config.getAugmentationConfig());
        if (config.getIntensityAugMode() != null) {
            settings.put("intensity_aug_mode", config.getIntensityAugMode());
        }
        if (!config.getClassWeightMultipliers().isEmpty()) {
            settings.put("class_weight_multipliers", config.getClassWeightMultipliers());
        }
        if (config.getFocusClass() != null) {
            settings.put("focus_class", config.getFocusClass());
            settings.put("focus_class_min_iou", config.getFocusClassMinIoU());
        }
        settings.put("whole_image", config.isWholeImage());
        settings.put("gradient_accumulation_steps", config.getGradientAccumulationSteps());
        settings.put("progressive_resize", config.isProgressiveResize());
        // Persist handler-specific params (e.g., MuViT model_config, patch_size,
        // level_scales, rope_mode) so they can be restored when loading from model.
        Map<String, Object> handlerParams = config.getHandlerParameters();
        if (handlerParams != null && !handlerParams.isEmpty()) {
            settings.put("handler_parameters", handlerParams);
        }
        return settings;
    }

    /**
     * Builds a list of ClassInfo objects with proper hex color strings.
     * Uses the class colors from the training dialog when available,
     * falling back to a distinct color palette.
     */
    static List<ClassifierMetadata.ClassInfo> buildClassInfoList(List<String> classNames,
                                                                  Map<String, Integer> classColors) {
        List<ClassifierMetadata.ClassInfo> classInfoList = new ArrayList<>();
        for (int i = 0; i < classNames.size(); i++) {
            String name = classNames.get(i);
            String hexColor;
            if (classColors != null && classColors.containsKey(name)) {
                int packed = classColors.get(name);
                hexColor = String.format("#%02X%02X%02X",
                        ColorTools.red(packed), ColorTools.green(packed), ColorTools.blue(packed));
            } else {
                hexColor = getDefaultClassColor(i);
            }
            classInfoList.add(new ClassifierMetadata.ClassInfo(i, name, hexColor));
        }
        return classInfoList;
    }

    /**
     * Returns a distinct default color for a class index.
     * Used when class colors are not available (e.g. headless training).
     */
    private static String getDefaultClassColor(int classIndex) {
        String[] palette = {
                "#FF0000", "#00AA00", "#0000FF", "#FFFF00",
                "#FF00FF", "#00FFFF", "#FF8800", "#8800FF"
        };
        return palette[classIndex % palette.length];
    }

    /**
     * Computes context padding for training tiles to match inference geometry.
     * Mirrors {@code DLPixelClassifier.computeOverlayPadding()} so that training
     * tiles include the same amount of surrounding real data as inference tiles.
     *
     * @param tileSize tile size in pixels
     * @param config   training configuration (provides overlap setting)
     * @return padding in pixels (at least 64, at most tileSize*3/8)
     */
    private static int computeTrainingContextPadding(int tileSize, TrainingConfig config) {
        int configOverlap = config.getOverlap();
        int minPadding = tileSize / 4;
        int padding = Math.max(configOverlap, minPadding);
        // Max 3/8 of tileSize ensures visible stride >= 25% of tileSize
        int maxPadding = Math.max(64, tileSize * 3 / 8);
        return Math.max(64, Math.min(padding, maxPadding));
    }

    /**
     * Formats a user-friendly device message from the epoch-0 training progress.
     */
    private static String formatDeviceMessage(String device, String deviceInfo) {
        if (device == null || device.isEmpty()) {
            return "Python backend ready, building model...";
        }
        switch (device) {
            case "cuda":
                String gpuName = (deviceInfo != null && !deviceInfo.isEmpty() && !"CPU".equals(deviceInfo))
                        ? deviceInfo : "NVIDIA GPU";
                return "Training on " + gpuName + " (CUDA)";
            case "mps":
                return "Training on Apple Silicon (MPS)";
            case "cpu":
                return "Training on CPU (no GPU detected -- this will be slow)";
            default:
                return "Training on device: " + device;
        }
    }

    /**
     * Converts a setup phase identifier to a user-friendly status message.
     */
    private static String formatSetupPhase(String phase) {
        if (phase == null) return "Setting up...";
        switch (phase) {
            case "creating_model":
                return "Creating model architecture...";
            case "loading_data":
                return "Loading training data...";
            case "computing_stats":
                return "Computing normalization statistics...";
            case "configuring_optimizer":
                return "Configuring optimizer and scheduler...";
            case "finding_learning_rate":
                return "Running learning rate finder...";
            case "loading_pretrained_weights":
                return "Loading pretrained weights...";
            case "loading_checkpoint":
                return "Loading checkpoint...";
            case "starting_training":
                return "Starting first epoch...";
            default:
                return "Setting up (" + phase + ")...";
        }
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

    /**
     * Deletes a temporary training data directory and all its contents.
     * Logs warnings on failure but does not throw.
     */
    private static void cleanupTempDir(Path tempDir) {
        if (tempDir == null || !Files.exists(tempDir)) return;
        try (java.util.stream.Stream<Path> paths = Files.walk(tempDir)) {
            paths.sorted(java.util.Comparator.reverseOrder())
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException e) {
                            logger.warn("Failed to delete temp file: {}", path);
                        }
                    });
            logger.info("Cleaned up training temp directory: {}", tempDir);
        } catch (IOException e) {
            logger.warn("Failed to clean up training temp directory: {}", tempDir, e);
        }
    }
}
