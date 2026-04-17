package qupath.ext.dlclassifier.controller;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.BuildInfo;
import qupath.ext.dlclassifier.DLClassifierChecks;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.time.LocalDateTime;
import qupath.ext.dlclassifier.ui.PythonConsoleWindow;
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
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

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

    /** Name of the marker file written inside active training temp directories. */
    public static final String ACTIVE_MARKER = ".dl-training-active";

    /**
     * Global lockfile preventing concurrent training across QuPath instances.
     * Uses OS-level {@link FileLock} so the lock is automatically released if
     * the process crashes.
     */
    private static final Path TRAINING_LOCK_PATH = Path.of(
            System.getProperty("java.io.tmpdir"), "dl-classifier-training.lock");

    private static RandomAccessFile lockRaf;
    private static FileChannel lockChannel;
    private static FileLock trainingLock;
    private static volatile boolean trainingInProgress = false;

    private QuPathGUI qupath;

    public TrainingWorkflow() {
        this.qupath = QuPathGUI.getInstance();
    }

    /**
     * Attempts to acquire the global training lock.  Returns true if acquired,
     * false if another instance already holds it.
     */
    private static synchronized boolean acquireTrainingLock() {
        if (trainingInProgress) {
            // Same JVM is already training
            return false;
        }
        if (trainingLock != null) {
            // This JVM already holds the file lock (shouldn't happen, but be safe)
            return true;
        }
        try {
            lockRaf = new RandomAccessFile(TRAINING_LOCK_PATH.toFile(), "rw");
            lockChannel = lockRaf.getChannel();
            trainingLock = lockChannel.tryLock();
            if (trainingLock == null) {
                // Another process holds the lock
                closeLockResources();
                return false;
            }
            trainingInProgress = true;
            return true;
        } catch (OverlappingFileLockException e) {
            // This JVM already has a lock on this channel
            closeLockResources();
            return false;
        } catch (IOException e) {
            logger.warn("Could not acquire training lock: {}", e.getMessage());
            closeLockResources();
            // Allow training to proceed if lock file is inaccessible
            trainingInProgress = true;
            return true;
        }
    }

    /**
     * Releases the global training lock.
     */
    private static synchronized void releaseTrainingLock() {
        trainingInProgress = false;
        try {
            if (trainingLock != null) {
                trainingLock.release();
                trainingLock = null;
            }
        } catch (IOException e) {
            logger.debug("Error releasing training lock", e);
        }
        closeLockResources();
        try {
            Files.deleteIfExists(TRAINING_LOCK_PATH);
        } catch (IOException ignored) {}
    }

    private static void closeLockResources() {
        try { if (lockChannel != null) lockChannel.close(); } catch (IOException ignored) {}
        try { if (lockRaf != null) lockRaf.close(); } catch (IOException ignored) {}
        lockChannel = null;
        lockRaf = null;
    }

    /**
     * Writes a marker file inside a training temp directory to indicate it is
     * actively in use.  The cleanup code skips directories containing this file.
     */
    static void writeActiveMarker(Path tempDir) {
        try {
            Files.writeString(tempDir.resolve(ACTIVE_MARKER),
                    "Training in progress - do not delete\n"
                    + "PID: " + ProcessHandle.current().pid() + "\n"
                    + "Started: " + java.time.Instant.now() + "\n");
        } catch (IOException e) {
            logger.debug("Could not write active marker in {}", tempDir, e);
        }
    }

    /**
     * Removes the active marker file from a training temp directory.
     */
    static void removeActiveMarker(Path tempDir) {
        if (tempDir == null) return;
        try {
            Files.deleteIfExists(tempDir.resolve(ACTIVE_MARKER));
        } catch (IOException e) {
            logger.debug("Could not remove active marker from {}", tempDir, e);
        }
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
            String message,
            String focusClassName,
            double focusClassIoU,
            boolean focusClassTargetMet
    ) {
        /** Compact constructor without focus class info. */
        public TrainingResult(String classifierId, String classifierName,
                              double finalLoss, double finalAccuracy,
                              int bestEpoch, double bestMeanIoU,
                              int epochsCompleted, boolean success, String message) {
            this(classifierId, classifierName, finalLoss, finalAccuracy,
                    bestEpoch, bestMeanIoU, epochsCompleted, success, message,
                    null, 0.0, true);
        }
    }

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
     * Shows the welcome/getting-started message on first use.
     * Returns true if the user dismissed or already dismissed it,
     * false if the user closed the dialog (cancel).
     */
    private boolean showWelcomeMessageIfNeeded() {
        if (!DLClassifierPreferences.isShowWelcomeMessage()) {
            return true;
        }

        javafx.scene.control.CheckBox neverAgain = new javafx.scene.control.CheckBox(
                "Don't show this message again");

        javafx.scene.control.Label message = new javafx.scene.control.Label(
                "Deep learning pixel classifiers are powerful but significantly slower "
                + "than QuPath's built-in pixel classifiers and most other segmentation "
                + "methods. They require training on annotated examples before they can "
                + "be used.\n\n"
                + "Performance considerations:\n"
                + "  - A modern NVIDIA GPU with CUDA support is strongly recommended.\n"
                + "    Without GPU acceleration, training can be extremely slow.\n"
                + "  - Fast storage (SSD/NVMe) and a modern CPU also help, but training\n"
                + "    will always take time -- minutes to hours depending on dataset size.\n\n"
                + "Tips for creating training annotations efficiently:\n"
                + "  - Use QuPath's built-in Pixel Classifier to generate initial annotations,\n"
                + "    then correct mistakes manually.\n"
                + "  - Use the Segment Anything Model (SAM) extension for targeted regions,\n"
                + "    then refine the results.\n"
                + "  - Editing existing annotations is much faster than drawing from scratch.\n"
                + "  - High-quality annotations give high-quality results, and more is better.\n"
                + "    Beware of getting sloppy at the end of a long annotation session --\n"
                + "    consistency in labeling is important.");
        message.setWrapText(true);
        message.setMaxWidth(520);

        javafx.scene.control.Hyperlink tipsLink = new javafx.scene.control.Hyperlink(
                "Tips & Tricks Guide");
        tipsLink.setOnAction(ev -> {
            try {
                java.awt.Desktop.getDesktop().browse(java.net.URI.create(
                        "https://github.com/uw-loci/qupath-extension-dl-pixel-classifier"
                        + "/blob/main/docs/TIPS_AND_TRICKS.md"));
            } catch (Exception ex) {
                logger.debug("Could not open Tips & Tricks URL: {}", ex.getMessage());
            }
        });

        javafx.scene.layout.VBox content = new javafx.scene.layout.VBox(12, message, tipsLink, neverAgain);
        content.setPadding(new javafx.geometry.Insets(10));

        javafx.scene.control.Dialog<javafx.scene.control.ButtonType> dialog = new javafx.scene.control.Dialog<>();
        dialog.setTitle("Getting Started with DL Pixel Classification");
        dialog.setHeaderText("Before you begin");
        dialog.getDialogPane().setContent(content);
        dialog.getDialogPane().getButtonTypes().addAll(
                javafx.scene.control.ButtonType.OK, javafx.scene.control.ButtonType.CANCEL);
        dialog.getDialogPane().setPrefWidth(560);

        var result = dialog.showAndWait();
        if (result.isPresent() && result.get() == javafx.scene.control.ButtonType.OK) {
            if (neverAgain.isSelected()) {
                DLClassifierPreferences.setShowWelcomeMessage(false);
            }
            return true;
        }
        return false;
    }

    /**
     * Shows the training configuration dialog.
     */
    private void showTrainingDialog() {
        if (!showWelcomeMessageIfNeeded()) {
            return;
        }

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
                                result.handlerParameters(),
                                result.trainOnlyImages(),
                                result.valOnlyImages()
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
     * @param classifierName  name for the classifier
     * @param description     classifier description
     * @param handler         the classifier handler
     * @param trainingConfig  training configuration
     * @param channelConfig   channel configuration
     * @param classNames      list of class names
     * @param selectedImages  project images to train from, or null for current image only
     * @param classColors     map of class name to packed RGB color for chart styling, or null
     * @param trainOnlyImages image names assigned exclusively to training (may be null/empty)
     * @param valOnlyImages   image names assigned exclusively to validation (may be null/empty)
     */
    public void trainClassifierWithProgress(String classifierName,
                                            String description,
                                            ClassifierHandler handler,
                                            TrainingConfig trainingConfig,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames,
                                            List<ProjectImageEntry<BufferedImage>> selectedImages,
                                            Map<String, Integer> classColors,
                                            Map<String, Object> handlerParameters,
                                            Set<String> trainOnlyImages,
                                            Set<String> valOnlyImages) {
        // Check for unsaved changes before training
        if (!checkUnsavedChanges(selectedImages)) {
            return;
        }

        // Start Python log file in project directory
        PythonConsoleWindow.startFileLogging();

        // Prevent concurrent training across QuPath instances
        if (!acquireTrainingLock()) {
            Dialogs.showErrorMessage("Training In Progress",
                    "Another QuPath instance is already training a DL classifier.\n\n"
                    + "Only one training session can run at a time.\n"
                    + "Wait for the other training to finish, or close the other "
                    + "QuPath instance before starting a new training run.");
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

        // Store classifier name in config so it survives in checkpoints
        trainingConfig.setClassifierName(classifierName);

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

                    // Open the results dialog on the FX thread. Pass the
                    // model directory and metadata so the dialog can offer
                    // Save/Load session functionality.
                    //
                    // Python's model_path IS the classifier directory (contains
                    // model.pt, metadata.json, disagreement/). If the path points
                    // to a file (legacy/fallback), fall back to its parent.
                    Path modelPathCandidate = Path.of(modelPath);
                    Path modelDir = Files.isDirectory(modelPathCandidate)
                            ? modelPathCandidate
                            : modelPathCandidate.getParent();
                    qupath.ext.dlclassifier.model.ClassifierMetadata metadata = null;
                    try {
                        metadata = new ModelManager().loadMetadata(modelDir);
                    } catch (Exception metaErr) {
                        logger.debug("Could not load classifier metadata for session support: {}",
                                metaErr.getMessage());
                    }
                    if (metadata == null) {
                        logger.warn("No classifier metadata found at {} -- "
                                + "Save/Load Session will be disabled", modelDir);
                    }
                    final qupath.ext.dlclassifier.model.ClassifierMetadata finalMetadata = metadata;
                    final Path finalModelDir = modelDir;
                    Platform.runLater(() -> {
                        TrainingAreaIssuesDialog dialog = new TrainingAreaIssuesDialog(
                                classifierName,
                                finalMetadata,
                                finalModelDir,
                                results,
                                trainingConfig.getDownsample(),
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
            } else {
                // Defensive: the pause button is disabled until jobIdCallback
                // fires (ProgressMonitorController.onTrainingJobStarted), so
                // this branch should be unreachable. If it fires, a new
                // regression has reintroduced the race.
                logger.warn("Pause clicked but currentJobId is null -- "
                        + "button-disable fix regressed?");
                progress.log("ERROR: Pause clicked before training job id was assigned. "
                        + "Nothing will happen -- please report this.");
            }
        });

        // Wire resume callback -- uses the SAME classifierId/modelOutputDir
        progress.setOnResume(v -> {
            CompletableFuture.runAsync(() -> handleResume(
                    currentJobId[0], classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    selectedImages, progress, currentJobId,
                    finalClassifierId, finalModelOutputDir,
                    trainingDataPathHolder, modelPathHolder));
        });

        // Wire complete-early callback -- uses the SAME classifierId/modelOutputDir
        progress.setOnCompleteEarly(v -> {
            CompletableFuture.runAsync(() -> handleCompleteEarly(
                    currentJobId[0], classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    selectedImages, progress, classColors,
                    finalClassifierId, finalModelOutputDir, modelPathHolder));
        });

        // Wire continue-training callback -- resumes from completion checkpoint
        progress.setOnContinueTraining(v -> {
            CompletableFuture.runAsync(() -> handleResume(
                    currentJobId[0], classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    selectedImages, progress, currentJobId,
                    finalClassifierId, finalModelOutputDir,
                    trainingDataPathHolder, modelPathHolder));
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
                        trainingDataPathHolder, modelPathHolder, handlerParameters,
                        trainOnlyImages != null ? trainOnlyImages : Collections.emptySet(),
                        valOnlyImages != null ? valOnlyImages : Collections.emptySet());

                if (result.success()) {
                    String completionMsg;
                    if (result.message() != null && result.message().contains("cancelled")) {
                        completionMsg = result.message();
                    } else {
                        StringBuilder sb = new StringBuilder();
                        sb.append(String.format(
                                "Classifier trained successfully!\nBest model: epoch %d\n"
                                + "Loss: %.4f | Accuracy: %.2f%% | mIoU: %.4f",
                                result.bestEpoch(), result.finalLoss(),
                                result.finalAccuracy() * 100, result.bestMeanIoU()));
                        if (result.focusClassName() != null) {
                            sb.append(String.format("\nFocus class '%s' IoU: %.4f",
                                    result.focusClassName(), result.focusClassIoU()));
                            if (!result.focusClassTargetMet()) {
                                sb.append(" [TARGET NOT MET]");
                            }
                        }
                        // Append diagnostic hints from trainCore if present
                        if (result.message() != null && result.message().contains("Diagnostic hints:")) {
                            int idx = result.message().indexOf("\n\nDiagnostic hints:");
                            if (idx >= 0) {
                                sb.append(result.message().substring(idx));
                            }
                        }
                        completionMsg = sb.toString();
                    }
                    // Warn with a dialog if focus class target was not met
                    if (result.focusClassName() != null && !result.focusClassTargetMet()) {
                        String warningMsg = completionMsg;
                        progress.complete(true, completionMsg);
                        Platform.runLater(() -> {
                            var alert = new javafx.scene.control.Alert(
                                    javafx.scene.control.Alert.AlertType.WARNING);
                            alert.setTitle("Focus Class Target Not Met");
                            alert.setHeaderText(String.format(
                                    "Focus class '%s' did not reach the target IoU",
                                    result.focusClassName()));
                            alert.setContentText(String.format(
                                    "Best IoU: %.4f\n\n"
                                    + "The model was saved but may not perform well "
                                    + "for class '%s'. Consider:\n"
                                    + "- Adding more training annotations for this class\n"
                                    + "- Training for more epochs\n"
                                    + "- Checking that the class appears in the validation split",
                                    result.focusClassIoU(), result.focusClassName()));
                            alert.show();
                        });
                    } else {
                        progress.complete(true, completionMsg);
                    }
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
                releaseTrainingLock();
                PythonConsoleWindow.flushLogFile();
            }
        });

        // Clean up training data when the progress dialog is closed
        progress.getStage().setOnHidden(e -> {
            Path dataPath = trainingDataPathHolder[0];
            if (dataPath != null) {
                removeActiveMarker(dataPath);
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
                jobIdHolder, classColors, classifierId, modelOutputDir, null, null, null,
                Collections.emptySet(), Collections.emptySet());
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
     * @param trainOnlyImages       image names assigned exclusively to training (may be null/empty)
     * @param valOnlyImages         image names assigned exclusively to validation (may be null/empty)
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
                                    Map<String, Object> handlerParameters,
                                    Set<String> trainOnlyImages,
                                    Set<String> valOnlyImages) {
        Path tempDir = null;
        try {
            if (progress != null) {
                progress.setStatus("Exporting training patches...");
                progress.setCurrentProgress(-1);
            }
            logger.info("Starting training for classifier: {}", classifierName);
            if (progress != null) {
                progress.log("DL Pixel Classifier " + BuildInfo.getSummary());
                progress.log("Starting training for classifier: " + classifierName);
            }

            // Export training data (use configured export directory if set)
            String exportDirPref = DLClassifierPreferences.getTrainingExportDir();
            if (exportDirPref != null && !exportDirPref.isEmpty()) {
                Path exportBase = Path.of(exportDirPref);
                Files.createDirectories(exportBase);
                tempDir = Files.createTempDirectory(exportBase, "dl-training");
            } else {
                tempDir = Files.createTempDirectory("dl-training");
            }
            writeActiveMarker(tempDir);
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
                        contextPadding,
                        trainOnlyImages,
                        valOnlyImages
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
                logInMemoryCacheEstimate(progress, trainingConfig, channelConfig, patchCount);
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
            final AtomicReference<Map<String, Double>> lastPerClassIoU = new AtomicReference<>(Map.of());
            final AtomicReference<Double> lastTrainLoss = new AtomicReference<>(0.0);
            final AtomicReference<Double> lastValLoss = new AtomicReference<>(0.0);

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
                                        trainingProgress.totalEpochs(),
                                        trainingProgress.loss(),
                                        trainingProgress.valLoss(),
                                        trainingProgress.perClassIoU(),
                                        trainingProgress.perClassLoss()
                                );

                                // Store latest metrics for post-training diagnostic hints
                                if (trainingProgress.perClassIoU() != null) {
                                    lastPerClassIoU.set(trainingProgress.perClassIoU());
                                }
                                lastTrainLoss.set(trainingProgress.loss());
                                lastValLoss.set(trainingProgress.valLoss());

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
                        // Python worker is now ready to receive pause signals.
                        if (progress != null) {
                            progress.onTrainingJobStarted();
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
                        .createdAt(LocalDateTime.now())
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
                    .createdAt(LocalDateTime.now())
                    .build();

            // Save the classifier. When modelOutputDir is set, files are already
            // in the project directory -- skip the copy step.
            boolean filesInPlace = modelOutputDir != null;
            ModelManager modelManager = new ModelManager();
            modelManager.saveClassifier(metadata, Path.of(serverResult.modelPath()),
                    true, filesInPlace);
            if (progress != null) progress.log("Classifier saved: " + metadata.getId());

            // Build diagnostic hints from final training metrics
            String completionMessage = "Training completed successfully";
            List<String> hints = new ArrayList<>();
            if (serverResult.bestMeanIoU() < 0.5) {
                hints.add("Low accuracy. Try: more annotations, different downsample, longer training.");
            }
            double tl = lastTrainLoss.get(), vl = lastValLoss.get();
            if (tl > 0 && vl > 0 && vl > tl * 2.0) {
                hints.add("Possible overfitting. Try: more augmentation, smaller model, more data.");
            }
            for (var iouEntry : lastPerClassIoU.get().entrySet()) {
                if (iouEntry.getValue() != null && iouEntry.getValue() == 0.0) {
                    hints.add(String.format(
                            "Class '%s' was never learned. Check annotation quality/quantity.",
                            iouEntry.getKey()));
                }
            }
            if (!hints.isEmpty()) {
                completionMessage += "\n\nDiagnostic hints:\n- " + String.join("\n- ", hints);
                if (progress != null) {
                    progress.log("--- Diagnostic hints ---");
                    for (String hint : hints) {
                        progress.log("  - " + hint);
                    }
                }
            }

            return new TrainingResult(
                    effectiveId,
                    classifierName,
                    serverResult.finalLoss(),
                    serverResult.finalAccuracy(),
                    serverResult.bestEpoch(),
                    serverResult.bestMeanIoU(),
                    trainingConfig.getEpochs(),
                    true,
                    completionMessage,
                    serverResult.focusClassName(),
                    serverResult.focusClassIoU(),
                    serverResult.focusClassTargetMet()
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
                removeActiveMarker(tempDir);
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
        trainClassifierWithProgress("Untitled", "", handler, trainingConfig, channelConfig, classNames, null, null, null, Collections.emptySet(), Collections.emptySet());
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
                              Path modelOutputDir,
                              Path[] trainingDataPathHolder,
                              String[] modelPathHolder) {
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
            // Update the outer holder so the dialog close handler can clean up
            // and Review Training Areas can find the data
            if (trainingDataPathHolder != null && trainingDataPathHolder.length > 0) {
                // Clean up old temp dir from previous run
                Path oldTempDir = trainingDataPathHolder[0];
                if (oldTempDir != null && !oldTempDir.equals(tempDir)) {
                    cleanupTempDir(oldTempDir);
                }
                trainingDataPathHolder[0] = tempDir;
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

                        // Handle setup phase updates (same filter as initial training)
                        if (trainingProgress.isSetupPhase()) {
                            if ("initializing".equals(trainingProgress.status())) {
                                String deviceMsg = formatDeviceMessage(
                                        trainingProgress.device(), trainingProgress.deviceInfo());
                                progress.log(deviceMsg);
                            } else if ("training_config".equals(trainingProgress.setupPhase())) {
                                progress.log("--- Resumed Training Configuration ---");
                                var config = trainingProgress.configSummary();
                                if (config != null) {
                                    for (var entry : config.entrySet()) {
                                        progress.log("  " + entry.getKey() + ": " + entry.getValue());
                                    }
                                }
                            } else if ("training_batch".equals(trainingProgress.setupPhase())) {
                                var cfg = trainingProgress.configSummary();
                                if (cfg != null) {
                                    String batch = cfg.getOrDefault("batch", "?");
                                    String totalBatches = cfg.getOrDefault("total_batches", "?");
                                    String batchEpoch = cfg.getOrDefault("epoch", "?");
                                    String totalEp = cfg.getOrDefault("total_epochs",
                                            String.valueOf(trainingProgress.totalEpochs()));
                                    String elapsed = cfg.getOrDefault("elapsed_seconds", "");
                                    String elapsedStr = elapsed.isEmpty() ? "" : " (" + elapsed + "s)";
                                    progress.setStatus(String.format("Epoch %s/%s - batch %s/%s%s",
                                            batchEpoch, totalEp, batch, totalBatches, elapsedStr));
                                    progress.setDetail(String.format("Batch %s/%s - loss: %s",
                                            batch, totalBatches, cfg.getOrDefault("batch_loss", "?")));
                                }
                            } else {
                                progress.setStatus(formatSetupPhase(trainingProgress.setupPhase()));
                            }
                            return;
                        }

                        // Update status on first real epoch
                        if (lastLoggedEpochResume.get() < 0) {
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
                        if (lastLoggedEpochResume.getAndSet(currentEpoch) < currentEpoch) {
                            progress.updateTrainingMetrics(
                                    trainingProgress.epoch(),
                                    trainingProgress.totalEpochs(),
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
                    progress::isCancelled,
                    newJobId -> {
                        if (currentJobId != null && currentJobId.length > 0) {
                            currentJobId[0] = newJobId;
                        }
                        // Python worker is now ready to receive pause signals.
                        progress.onTrainingJobStarted();
                    }
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
                            .createdAt(LocalDateTime.now())
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
                        .createdAt(LocalDateTime.now())
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
            releaseTrainingLock();
            // Only clean up if the outer holder isn't managing this temp dir.
            // When the holder IS set, the dialog close handler cleans up
            // (allowing Review Training Areas to access the data first).
            if (trainingDataPathHolder == null
                    || trainingDataPathHolder.length == 0
                    || trainingDataPathHolder[0] != tempDir) {
                removeActiveMarker(tempDir);
                cleanupTempDir(tempDir);
            }
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
                                     Path modelOutputDir,
                                     String[] modelPathHolder) {
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
                    .createdAt(LocalDateTime.now())
                    .build();

            boolean filesInPlace = modelOutputDir != null;
            ModelManager modelManager = new ModelManager();
            modelManager.saveClassifier(metadata, Path.of(serverResult.modelPath()),
                    true, filesInPlace);
            progress.log("Classifier saved: " + metadata.getId());

            // Set model path so Review Training Areas can find the model
            if (modelPathHolder != null && modelPathHolder.length > 0) {
                modelPathHolder[0] = serverResult.modelPath();
            }

            // Clean up training checkpoint files from the model directory.
            // These are only needed for crash recovery and resume -- the final
            // model.pt is the only file needed for inference.
            Path modelDir = modelOutputDir != null ? modelOutputDir
                    : Path.of(serverResult.modelPath());
            if (Files.isDirectory(modelDir)) {
                try (var files = Files.list(modelDir)) {
                    files.filter(p -> {
                        String name = p.getFileName().toString();
                        return name.startsWith("best_in_progress_")
                                || name.startsWith("checkpoint_");
                    }).forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                            logger.debug("Cleaned up training artifact: {}", p.getFileName());
                        } catch (IOException ignored) {}
                    });
                } catch (IOException e) {
                    logger.debug("Could not clean up training artifacts: {}", e.getMessage());
                }
            }

            // Disable continue-training -- checkpoints were cleaned up above
            // and there is nothing to resume from.
            progress.setOnContinueTraining(null);
            progress.setOnResume(null);

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
        dialog.setHeaderText("How many additional epochs?");
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

        // Default additional epochs to the remaining portion of the original
        // configured count (or at least 10 if we have already hit the target).
        int defaultAdditional = Math.max(originalEpochs - epochsCompleted, 10);
        Spinner<Integer> epochSpinner = new Spinner<>(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 1000, defaultAdditional));
        epochSpinner.setEditable(true);
        epochSpinner.setPrefWidth(120);

        grid.add(new Label("Additional Epochs:"), 0, row);
        grid.add(epochSpinner, 1, row);
        row++;

        // Reassure the user that other hyperparameters are unchanged on resume
        // (they still take effect through currentLR / currentBatch below).
        Label hint = new Label(
                "Learning rate and batch size are unchanged from the original run.");
        hint.setStyle("-fx-text-fill: #666; -fx-font-style: italic; -fx-font-size: 11px;");
        hint.setWrapText(true);
        grid.add(hint, 0, row, 2, 1);

        dialog.getDialogPane().setContent(grid);

        final int completed = epochsCompleted;
        dialog.setResultConverter(buttonType -> {
            if (buttonType == ButtonType.OK) {
                int totalEpochs = completed + epochSpinner.getValue();
                return new ResumeParams(totalEpochs, currentLR, currentBatch);
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
            settings.put("ohem_hard_ratio_start", config.getOhemHardRatioStart());
            settings.put("ohem_schedule", config.getOhemSchedule());
            settings.put("ohem_adaptive_floor", config.isOhemAdaptiveFloor());
        }
        settings.put("data_loader_workers", config.getDataLoaderWorkers());
        settings.put("in_memory_dataset", config.getInMemoryDataset());
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
     * Delegates to the shared {@link InferenceConfig#computeEffectivePadding}
     * so training, overlay, and Apply Classifier all use the same computation.
     *
     * @param tileSize tile size in pixels
     * @param config   training configuration (provides overlap setting)
     * @return padding in pixels (at least 64, at most tileSize*3/8)
     */
    private static int computeTrainingContextPadding(int tileSize, TrainingConfig config) {
        return InferenceConfig.computeEffectivePadding(tileSize, config.getOverlap());
    }

    /**
     * Logs a pre-flight estimate of the in-memory dataset cache size versus
     * available system RAM so the user sees it before training starts. The
     * actual decision still happens in Python (the Java estimate may differ
     * slightly because Python measures a real tile after export), but the
     * two agree to within a factor of two and that is good enough for the
     * user to decide whether to cancel and flip the preference off.
     */
    private static void logInMemoryCacheEstimate(ProgressMonitorController progress,
                                                 TrainingConfig config,
                                                 ChannelConfiguration channels,
                                                 int patchCount) {
        if (progress == null) return;
        String mode = config.getInMemoryDataset();
        if ("off".equals(mode)) {
            progress.log("In-memory cache: off (streaming from disk). "
                    + "Set preference to 'auto' if you want to try caching.");
            return;
        }
        int tile = config.getTileSize();
        int chs = Math.max(1, channels.getSelectedChannels().size());
        boolean hasContext = config.getContextScale() > 1;
        long perImage = (long) tile * tile * chs; // uint8 bytes
        if (hasContext) perImage *= 2L;
        long perMask = (long) tile * tile; // uint8
        long totalBytes = (long) patchCount * (perImage + perMask);

        long availableBytes = -1L;
        try {
            java.lang.management.OperatingSystemMXBean osBean =
                    java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            if (osBean instanceof com.sun.management.OperatingSystemMXBean sun) {
                availableBytes = sun.getFreeMemorySize();
            }
        } catch (Throwable t) {
            // com.sun.* not available on some JVMs -- fall through
        }

        double estGb = totalBytes / 1e9;
        String header = String.format(
                "In-memory cache: ~%.2f GB needed for %d patches (mode=%s)",
                estGb, patchCount, mode);
        if (availableBytes <= 0) {
            progress.log(header + ". Free RAM unknown -- Python will make the "
                    + "final decision based on psutil.");
            return;
        }
        double availGb = availableBytes / 1e9;
        double usedPct = 100.0 * totalBytes / availableBytes;
        String body = String.format(
                " / %.2f GB OS free RAM (would use %.0f%% of free).",
                availGb, usedPct);
        progress.log(header + body);

        if ("auto".equals(mode)) {
            if (totalBytes < 0.25 * availableBytes) {
                progress.log("  auto -> ENABLED (fits under the 25% ceiling).");
            } else {
                progress.log("  auto -> DISABLED (would exceed 25% ceiling). "
                        + "Training will stream from disk. Set preference "
                        + "to 'on' to override.");
            }
        } else if ("on".equals(mode)) {
            if (totalBytes > availableBytes) {
                progress.log("  on -> FORCED despite ESTIMATE EXCEEDING FREE RAM. "
                        + "Preload may fail with MemoryError. "
                        + "Consider Cancel -> set preference to 'auto' or 'off'.");
            } else if (totalBytes > 0.5 * availableBytes) {
                progress.log("  on -> forced. Warning: cache will use more than "
                        + "50% of free RAM -- other processes may be squeezed.");
            } else {
                progress.log("  on -> forced (comfortably within free RAM).");
            }
        }
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
