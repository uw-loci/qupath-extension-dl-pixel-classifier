package qupath.ext.dlclassifier;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.binding.BooleanBinding;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.geometry.Insets;
import javafx.scene.control.Alert;
import javafx.scene.control.CheckBox;
import javafx.scene.control.CheckMenuItem;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.controller.DLClassifierController;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.ApposeClassifierBackend;
import qupath.ext.dlclassifier.service.ApposeService;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.DLPixelClassifier;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.service.OverlayService;
import qupath.ext.dlclassifier.service.warnings.InteractionWarningRegistration;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.utilities.CheckpointScanner;
import qupath.ext.dlclassifier.utilities.CheckpointScanner.OrphanedCheckpoint;
import qupath.ext.dlclassifier.ui.MAEPretrainingDialog;
import qupath.ext.dlclassifier.ui.SSLPretrainingDialog;
import qupath.ext.dlclassifier.ui.ProgressMonitorController;
import qupath.ext.dlclassifier.ui.PythonConsoleWindow;
import qupath.ext.dlclassifier.ui.SetupEnvironmentDialog;
import qupath.ext.dlclassifier.ui.TooltipHelper;
import qupath.ext.dlclassifier.ui.TrainingAreaIssuesDialog;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.common.GeneralTools;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.images.ImageData;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Entry point for the Deep Learning Pixel Classifier extension.
 * <p>
 * This extension provides deep learning-based pixel classification capabilities for QuPath,
 * supporting both brightfield RGB and multi-channel fluorescence/spectral images.
 * <p>
 * Key features:
 * <ul>
 *   <li>Train custom pixel classifiers using sparse annotations</li>
 *   <li>Support for multi-channel images with per-channel normalization</li>
 *   <li>Pluggable model architecture system (UNet, SegFormer, etc.)</li>
 *   <li>REST API communication with Python deep learning server</li>
 *   <li>Output as measurements, objects, or classification overlays</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class SetupDLClassifier implements QuPathExtension, GitHubProject {

    private static final Logger logger = LoggerFactory.getLogger(SetupDLClassifier.class);

    // Load extension metadata
    private static final ResourceBundle res = ResourceBundle.getBundle("qupath.ext.dlclassifier.ui.strings");
    private static final String EXTENSION_NAME = res.getString("name");
    private static final String EXTENSION_DESCRIPTION = res.getString("description");
    private static final Version EXTENSION_QUPATH_VERSION = Version.parse("v0.6.0");
    private static final GitHubRepo EXTENSION_REPOSITORY =
            GitHubRepo.create(EXTENSION_NAME, "uw-loci", "qupath-extension-dl-pixel-classifier");

    /**
     * Observable property tracking whether the DL environment is ready for use.
     * When true, workflow menu items are enabled. When false, they are disabled
     * and the Setup item is shown at the top of the menu.
     */
    private final BooleanProperty environmentReady = new SimpleBooleanProperty(false);

    /** True if the backend (Appose or HTTP) has been initialized and is available. */
    private boolean serverAvailable;

    @Override
    public String getName() {
        return EXTENSION_NAME;
    }

    @Override
    public String getDescription() {
        return EXTENSION_DESCRIPTION;
    }

    @Override
    public Version getQuPathVersion() {
        return EXTENSION_QUPATH_VERSION;
    }

    @Override
    public GitHubRepo getRepository() {
        return EXTENSION_REPOSITORY;
    }

    @Override
    public void installExtension(QuPathGUI qupath) {
        logger.info("Installing extension: {}", EXTENSION_NAME);

        // Register persistent preferences
        DLClassifierPreferences.installPreferences(qupath);

        // Register interaction-warning watchers so pre-training,
        // pre-inference and preference-toggle checks know what to
        // evaluate. Safe to call before anything else uses the
        // service -- registration is idempotent by watcher id.
        InteractionWarningRegistration.registerAll();

        // Fast filesystem check to determine environment state (no downloads)
        updateEnvironmentState();

        // Build menu on the FX thread
        Platform.runLater(() -> addMenuItem(qupath));

        // If environment is already built, start background initialization of Python service
        if (environmentReady.get()) {
            startBackgroundInitialization();
        }

        // Surface orphaned training checkpoints after project open, so users
        // notice they can recover from an interrupted run without having to
        // remember the Utilities menu.
        installCheckpointRecoveryWatcher(qupath);
    }

    /**
     * Session-scoped record of orphaned checkpoints we have already toasted
     * about, so the user is not re-notified every time they switch projects.
     */
    private final java.util.Set<Path> toastedCheckpointsThisSession =
            java.util.concurrent.ConcurrentHashMap.newKeySet();

    /**
     * Installs a listener on the project property that scans the central
     * checkpoint registry for orphaned best-in-progress files whenever a
     * project is opened. If any are found (that we haven't already toasted
     * about this session), shows a non-modal info notification.
     */
    private void installCheckpointRecoveryWatcher(QuPathGUI qupath) {
        qupath.projectProperty().addListener((obs, oldProj, newProj) -> {
            if (newProj == null) return;
            Thread scanThread = new Thread(() -> {
                List<OrphanedCheckpoint> orphans =
                        CheckpointScanner.scanCentralRegistry(
                                java.util.Collections.emptySet());
                orphans.removeIf(o -> toastedCheckpointsThisSession.contains(o.file()));
                if (orphans.isEmpty()) return;
                orphans.forEach(o -> toastedCheckpointsThisSession.add(o.file()));

                int count = orphans.size();
                String first = orphans.get(0).classifierName();
                String detail;
                if (count == 1) {
                    detail = String.format(
                            "Unfinished training checkpoint found: %s.\n"
                            + "Open Extensions > DL Pixel Classifier to recover.",
                            first);
                } else {
                    detail = String.format(
                            "%d unfinished training checkpoints found (including %s).\n"
                            + "Open Extensions > DL Pixel Classifier to recover.",
                            count, first);
                }
                Platform.runLater(() ->
                        Dialogs.showInfoNotification(EXTENSION_NAME, detail));
            }, "DLClassifier-CheckpointScan");
            scanThread.setDaemon(true);
            scanThread.start();
        });
    }

    /**
     * Determines the current environment state based on filesystem.
     */
    private void updateEnvironmentState() {
        if (ApposeService.isEnvironmentBuilt()) {
            environmentReady.set(true);
            logger.debug("Appose environment found on disk");
        } else {
            environmentReady.set(false);
            logger.info("Appose environment not found - setup required");
        }
    }

    /**
     * Starts background initialization of the Appose service when the environment
     * is already built. This pre-warms the Python subprocess so it is ready when
     * the user clicks a workflow item.
     * <p>
     * If initialization fails (e.g., packages not installed), sets
     * environmentReady to false so the setup item reappears.
     */
    private void startBackgroundInitialization() {
        Thread initThread = new Thread(() -> {
            // Clean up orphaned temp directories from previous sessions before
            // doing anything else.  These accumulate when QuPath is force-killed,
            // training is paused without cleanup, or the dialog is closed mid-run.
            cleanupOrphanedTempDirs();

            try {
                Platform.runLater(() -> Dialogs.showInfoNotification(
                        EXTENSION_NAME,
                        "Setting up Python environment (first time may take a minute)..."));

                ApposeService appose = ApposeService.getInstance();
                appose.initialize();

                // Use cached health/version info from the combined verification
                // task inside initialize().  This avoids a separate health check
                // task that can crash when the Appose worker exits between tasks.
                boolean healthy = appose.isLastHealthy();
                String versionWarn = appose.getLastVersionWarning();

                if (!healthy || (versionWarn != null && !versionWarn.isEmpty())) {
                    serverAvailable = false;

                    // Distinguish version mismatch from generic health failure.
                    // A crash (e.g. worker killed by concurrent rebuild) should
                    // NOT trigger auto-rebuild -- only a real version warning should.
                    boolean isVersionMismatch = versionWarn != null && !versionWarn.isEmpty();

                    if (!isVersionMismatch) {
                        // Generic health check failure (crash, timeout, etc.)
                        logger.warn("Health check failed but no version mismatch detected. "
                                + "The Python worker may have crashed or been shut down.");
                        Platform.runLater(() -> {
                            // Don't disable menu -- the service initialized OK,
                            // this may be a transient issue
                            Dialogs.showWarningNotification(
                                    EXTENSION_NAME,
                                    "Python health check failed (worker may have crashed).\n" +
                                            "Try using the extension normally -- it will reconnect.");
                        });
                    } else if (DLClassifierPreferences.isAutoRebuildEnvironment()) {
                        logger.info("Version mismatch detected -- auto-rebuilding environment");
                        Platform.runLater(() -> {
                            environmentReady.set(false);
                            Dialogs.showInfoNotification(
                                    EXTENSION_NAME,
                                    "Updating Python environment to match extension.\n" +
                                            "Workflow menu items will re-appear when complete.");
                        });
                        try {
                            ApposeService.getInstance().upgradeServerPackage(
                                    msg -> logger.info("[Auto-rebuild] {}", msg));

                            // Re-check health after upgrade
                            boolean healthyNow = appose.isLastHealthy();
                            String warnNow = appose.getLastVersionWarning();
                            if (healthyNow && (warnNow == null || warnNow.isEmpty())) {
                                serverAvailable = true;
                                Platform.runLater(() -> {
                                    environmentReady.set(true);
                                    Dialogs.showInfoNotification(
                                            EXTENSION_NAME,
                                            "Python environment updated successfully.");
                                });
                                logger.info("Auto-rebuild completed successfully");
                            } else {
                                logger.error("Auto-rebuild completed but health check still fails");
                                Platform.runLater(() -> {
                                    environmentReady.set(false);
                                    Dialogs.showErrorNotification(
                                            EXTENSION_NAME,
                                            "Auto-rebuild failed. Use Utilities >\n" +
                                                    "Rebuild DL Environment manually.");
                                });
                            }
                        } catch (Exception rebuildEx) {
                            logger.error("Auto-rebuild failed: {}", rebuildEx.getMessage());
                            Platform.runLater(() -> {
                                environmentReady.set(false);
                                Dialogs.showErrorNotification(
                                        EXTENSION_NAME,
                                        "Auto-rebuild failed: " + rebuildEx.getMessage() +
                                                "\nUse Utilities > Rebuild DL Environment manually.");
                            });
                        }
                    } else {
                        // Manual rebuild required
                        String message = versionWarn != null && !versionWarn.isEmpty()
                                ? versionWarn
                                : "Python environment health check failed.";
                        logger.error("Environment version mismatch: {}", message);
                        Platform.runLater(() -> {
                            environmentReady.set(false);
                            Dialogs.showErrorNotification(
                                    EXTENSION_NAME,
                                    "Python package version mismatch.\n" +
                                            "Go to Extensions > " + EXTENSION_NAME +
                                            " > Utilities >\nRebuild DL Environment to update.");
                        });
                    }
                } else {
                    serverAvailable = true;
                    logger.info("Appose backend initialized successfully (background)");
                }
            } catch (Exception e) {
                logger.warn("Background Appose init failed: {}", e.getMessage());
                serverAvailable = false;
                // Environment exists on disk but isn't functional --
                // revert to "needs setup" state so the user can fix it
                Platform.runLater(() -> {
                    environmentReady.set(false);
                    Dialogs.showWarningNotification(
                            EXTENSION_NAME,
                            "Python environment exists but failed to initialize.\n" +
                                    "Use Setup DL Environment or Rebuild to fix.");
                });
            }
        }, "DLClassifier-BackgroundInit");
        initThread.setDaemon(true);
        initThread.start();
    }

    private void addMenuItem(QuPathGUI qupath) {
        // Create the top level Extensions > DL Pixel Classifier menu
        var extensionMenu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);

        // Add colored indicator dot for quick identification in the menu
        if (DLClassifierPreferences.isShowMenuDot()) {
            int argb = DLClassifierPreferences.getMenuDotColor();
            int a = (argb >> 24) & 0xFF;
            int r = (argb >> 16) & 0xFF;
            int g = (argb >> 8) & 0xFF;
            int b = argb & 0xFF;
            Color dotColor = Color.rgb(r, g, b, a / 255.0);
            extensionMenu.setGraphic(new Circle(4, dotColor));
        }

        // === SETUP MENU ITEM (visible only when environment not ready) ===
        MenuItem setupItem = new MenuItem(res.getString("menu.setupEnvironment"));
        TooltipHelper.installOnMenuItem(setupItem,
                "Download and configure the Python deep learning environment.\n" +
                        "Required for first-time use. Downloads approximately 2-4 GB.");
        setupItem.setOnAction(e -> showSetupDialog(qupath));

        // Binding: visible when environment is NOT ready
        BooleanBinding showSetup = environmentReady.not();
        setupItem.visibleProperty().bind(showSetup);

        // Setup separator - visible only with setup item
        SeparatorMenuItem setupSeparator = new SeparatorMenuItem();
        setupSeparator.visibleProperty().bind(showSetup);

        // === MAIN WORKFLOW MENU ITEMS (visible when environmentReady) ===

        // 1) Train Classifier - create a new classifier from annotations
        MenuItem trainOption = new MenuItem(res.getString("menu.training"));
        TooltipHelper.installOnMenuItem(trainOption,
                "Train a new deep learning pixel classifier from annotated regions.\n" +
                        "Requires at least 2 annotation classes (e.g. Foreground/Background).\n" +
                        "Supports single-image and multi-image training from project images.");
        trainOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> qupath.getProject() == null,
                        qupath.projectProperty()
                ).or(environmentReady.not())
        );
        trainOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("training"));

        // 2) Apply Classifier - run inference on current image
        MenuItem inferenceOption = new MenuItem(res.getString("menu.inference"));
        TooltipHelper.installOnMenuItem(inferenceOption,
                "Apply a trained classifier to the current image or selected annotations.\n" +
                        "Results can be added as measurements, detection/annotation objects,\n" +
                        "or live classification overlays.");
        inferenceOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> qupath.getImageData() == null,
                        qupath.imageDataProperty()
                ).or(environmentReady.not())
        );
        inferenceOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("inference"));

        // Separator between train/inference and overlay controls
        SeparatorMenuItem sep1 = new SeparatorMenuItem();

        OverlayService overlayService = OverlayService.getInstance();
        BooleanBinding noImage = Bindings.createBooleanBinding(
                () -> qupath.getImageData() == null,
                qupath.imageDataProperty()
        );

        // When the user changes overlay smoothing or tile-averaging in
        // Preferences, rebuild the active overlay so the change is visible
        // immediately -- matching the old "Overlay Settings..." dialog.
        DLClassifierPreferences.overlaySmoothingProperty().addListener((obs, oldV, newV) -> {
            if (overlayService.hasOverlay()) overlayService.recreateOverlay();
        });
        DLClassifierPreferences.multiPassAveragingProperty().addListener((obs, oldV, newV) -> {
            if (overlayService.hasOverlay()) overlayService.recreateOverlay();
        });

        // 3) Select Overlay Model - choose which classifier to use for the overlay
        MenuItem selectModelOption = new MenuItem("Select Overlay Model...");
        TooltipHelper.installOnMenuItem(selectModelOption,
                "Choose a trained classifier for the prediction overlay.\n" +
                        "The selected model is used when toggling the overlay on/off.");
        selectModelOption.setOnAction(e -> selectOverlayModel(qupath, overlayService));
        selectModelOption.disableProperty().bind(noImage.or(environmentReady.not()));

        // 4) Toggle Prediction Overlay - simple on/off toggle
        CheckMenuItem livePredictionOption = new CheckMenuItem(res.getString("menu.toggleOverlay"));
        TooltipHelper.installOnMenuItem(livePredictionOption,
                "Toggle the DL classification overlay on/off.\n" +
                        "Uses the model chosen via 'Select Overlay Model'.\n" +
                        "If no model is selected, you will be prompted to choose one.");
        // Sync CheckMenuItem state from the property (for programmatic changes)
        overlayService.livePredictionProperty().addListener((obs, wasLive, isLive) ->
                livePredictionOption.setSelected(isLive));
        // Toggle overlay on/off
        livePredictionOption.setOnAction(e -> {
            if (livePredictionOption.isSelected()) {
                if (overlayService.hasOverlay()) {
                    overlayService.setLivePrediction(true);
                } else if (overlayService.hasSelectedModel()) {
                    // Model already selected - show notice then create overlay
                    showOverlayNoticeIfNeeded();
                    ImageData<BufferedImage> imageData = qupath.getImageData();
                    if (imageData != null) {
                        overlayService.createOverlayFromSelection(imageData);
                    } else {
                        livePredictionOption.setSelected(false);
                    }
                } else {
                    // No model selected - prompt user (notice shown after selection)
                    selectOverlayModel(qupath, overlayService);
                    if (!overlayService.hasOverlay()) {
                        livePredictionOption.setSelected(false);
                    }
                }
            } else {
                overlayService.removeOverlay();
            }
        });
        livePredictionOption.disableProperty().bind(
                noImage.or(overlayService.trainingActiveProperty()).or(environmentReady.not()));

        // Separator before models
        SeparatorMenuItem sep2 = new SeparatorMenuItem();

        // 5) Manage Models - browse and manage saved classifiers
        MenuItem modelsOption = new MenuItem(res.getString("menu.manageModels"));
        TooltipHelper.installOnMenuItem(modelsOption,
                "Browse, import, export, and delete saved classifiers.\n" +
                        "View model metadata, training configuration, and class mappings.");
        modelsOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("modelManagement"));
        modelsOption.disableProperty().bind(environmentReady.not());

        // Separator before utilities
        SeparatorMenuItem sep3 = new SeparatorMenuItem();

        // === UTILITIES SUBMENU ===
        Menu utilitiesMenu = new Menu("Utilities");

        // Free GPU Memory - disabled when environment not ready
        MenuItem freeGpuOption = new MenuItem("Free GPU Memory");
        TooltipHelper.installOnMenuItem(freeGpuOption,
                "Force-clear all GPU memory held by the classification server.\n" +
                        "Cancels running training jobs, clears cached models, and\n" +
                        "frees GPU VRAM. Use after a crash or failed training.");
        BooleanProperty freeGpuRunning = new SimpleBooleanProperty(false);
        freeGpuOption.disableProperty().bind(freeGpuRunning.or(environmentReady.not()));
        freeGpuOption.setOnAction(e -> {
            freeGpuRunning.set(true);
            Thread clearThread = new Thread(() -> {
                try {
                    ClassifierBackend backend = BackendFactory.getBackend();
                    String result = backend.clearGPUMemory();
                    Platform.runLater(() -> {
                        freeGpuRunning.set(false);
                        if (result != null) {
                            Dialogs.showInfoNotification(EXTENSION_NAME, result);
                        } else {
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "Failed to clear GPU memory. Is the backend available?");
                        }
                    });
                } catch (Exception ex) {
                    logger.error("GPU memory clear failed", ex);
                    Platform.runLater(() -> {
                        freeGpuRunning.set(false);
                        Dialogs.showErrorNotification(EXTENSION_NAME,
                                "Error clearing GPU memory: " + ex.getMessage());
                    });
                }
            }, "DLClassifier-FreeGPU");
            clearThread.setDaemon(true);
            clearThread.start();
        });

        // MAE Pretrain Encoder - visible when environment ready
        MenuItem maePretrainOption = new MenuItem("MAE Pretrain Encoder...");
        TooltipHelper.installOnMenuItem(maePretrainOption,
                "Self-supervised pretraining for MuViT encoder.\n" +
                "Train on unlabeled image tiles using masked autoencoder.\n" +
                "The resulting encoder can be loaded via 'Continue from model'.");
        maePretrainOption.setOnAction(e -> startMAEPretraining());
        maePretrainOption.disableProperty().bind(environmentReady.not());

        MenuItem sslPretrainOption = new MenuItem("SSL Pretrain Encoder...");
        TooltipHelper.installOnMenuItem(sslPretrainOption,
                "Self-supervised pretraining for CNN encoders (ResNet, EfficientNet).\n" +
                "Uses SimCLR or BYOL on tiles from annotated regions.\n" +
                "The resulting encoder can be loaded via 'Use SSL pretrained encoder'.");
        sslPretrainOption.setOnAction(e -> startSSLPretraining());
        sslPretrainOption.disableProperty().bind(environmentReady.not());

        // Rebuild DL Environment - always visible so users can fix broken environments
        MenuItem rebuildItem = new MenuItem(res.getString("menu.rebuildEnvironment"));
        TooltipHelper.installOnMenuItem(rebuildItem,
                "Delete and re-download the Python deep learning environment.\n" +
                        "Use this if the environment becomes corrupted or you want a fresh install.");
        rebuildItem.setOnAction(e -> rebuildEnvironment(qupath));

        // System Info - visible when environment ready
        MenuItem systemInfoOption = new MenuItem("System Info...");
        TooltipHelper.installOnMenuItem(systemInfoOption,
                "Show detailed system information including GPU status,\n" +
                        "Python package versions, and platform details.\n" +
                        "All information is copyable for bug reports.");
        systemInfoOption.setOnAction(e -> showSystemInfo());
        systemInfoOption.disableProperty().bind(environmentReady.not());

        // Python Console - visible when environment ready
        MenuItem pythonConsoleOption = new MenuItem(res.getString("menu.pythonConsole"));
        TooltipHelper.installOnMenuItem(pythonConsoleOption,
                "Show a live console window displaying Python process output.\n" +
                "Useful for monitoring model loading, inference, and debugging.");
        pythonConsoleOption.setOnAction(e -> PythonConsoleWindow.getInstance().show());
        pythonConsoleOption.disableProperty().bind(environmentReady.not());

        // Where Are My Files? - always visible
        MenuItem whereFilesOption = new MenuItem("Where Are My Files?");
        TooltipHelper.installOnMenuItem(whereFilesOption,
                "Show all directories where this extension stores files:\n" +
                        "trained classifiers, Python environment, temp data, etc.\n" +
                        "Includes disk usage and links to open each folder.");
        whereFilesOption.setOnAction(e -> showWhereAreMyFiles());

        // Clean Up Storage
        MenuItem cleanUpOption = new MenuItem("Clean Up Storage...");
        TooltipHelper.installOnMenuItem(cleanUpOption,
                "Remove orphaned files from previous training sessions:\n" +
                        "temp directories, old checkpoints, and model leftovers.\n" +
                        "Shows what will be deleted before proceeding.");
        cleanUpOption.setOnAction(e -> cleanUpStorage());

        // Load Saved Training Area Issues - standalone reload entry point
        MenuItem loadIssuesOption = new MenuItem("Load Saved Training Area Issues...");
        TooltipHelper.installOnMenuItem(loadIssuesOption,
                "Reopen a previously saved Training Area Issues session\n"
                        + "for a specific classifier, without re-running evaluation.");
        loadIssuesOption.setOnAction(e -> openSavedTrainingIssuesDialog());
        loadIssuesOption.disableProperty().bind(environmentReady.not());

        utilitiesMenu.getItems().addAll(pythonConsoleOption, whereFilesOption,
                systemInfoOption, new SeparatorMenuItem(),
                freeGpuOption, maePretrainOption, sslPretrainOption, cleanUpOption,
                loadIssuesOption,
                new SeparatorMenuItem(), rebuildItem);

        // === BUILD FINAL MENU ===
        extensionMenu.getItems().addAll(
                setupItem,
                setupSeparator,
                trainOption,
                inferenceOption,
                sep1,
                selectModelOption,
                livePredictionOption,
                sep2,
                modelsOption,
                sep3,
                utilitiesMenu
        );

        logger.info("Menu items added for extension: {}", EXTENSION_NAME);
    }

    /**
     * Recovers a trained model from a specific checkpoint {@code .pt} file.
     * Runs {@code finalize_training.py} in the background and reports the
     * finalized model via an info notification. If no project is open, the
     * user is asked to confirm saving to the default fallback location.
     * <p>
     * Called from the project-open toast and the TrainingDialog banner,
     * both of which discover orphaned checkpoints automatically.
     */
    public static void recoverFromCheckpoint(QuPathGUI qupath, Path checkpointPath) {
        if (checkpointPath == null || !Files.isRegularFile(checkpointPath)) {
            Dialogs.showErrorNotification(EXTENSION_NAME,
                    "Checkpoint file not found: " + checkpointPath);
            return;
        }

        // If no project is open, confirm with the user that recovery will
        // still proceed (finalize_training.py saves to its default location).
        if (qupath.getProject() == null) {
            Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
            confirm.setTitle("Recover Without Project");
            confirm.setHeaderText("No project is currently open");
            confirm.setContentText(
                    "The recovered model will be saved to the Python server's\n"
                    + "default model directory instead of a project folder.\n\n"
                    + "Continue?");
            if (confirm.showAndWait().orElse(null) != javafx.scene.control.ButtonType.OK) {
                return;
            }
        }

        String checkpointPathStr = checkpointPath.toAbsolutePath().toString();
        logger.info("Recovering model from checkpoint: {}", checkpointPathStr);

        // Determine output directory: project classifiers dir if available
        String modelOutputDir = null;
        if (qupath.getProject() != null) {
            Path classifiersDir = qupath.getProject().getPath().getParent()
                    .resolve("classifiers").resolve("dl");
            String baseName = checkpointPath.getFileName().toString().replace(".pt", "");
            Path outputDir = classifiersDir.resolve("recovered_" + baseName);
            try {
                Files.createDirectories(outputDir);
                modelOutputDir = outputDir.toString();
            } catch (IOException ex) {
                logger.warn("Could not create output dir, using default location", ex);
            }
        }

        String finalOutputDir = modelOutputDir;
        Dialogs.showInfoNotification(EXTENSION_NAME,
                "Recovering model from checkpoint...\nThis may take a moment.");

        Thread recoverThread = new Thread(() -> {
            try {
                ClassifierBackend backend = BackendFactory.getBackend();
                ClassifierClient.TrainingResult result =
                        backend.finalizeTraining(checkpointPathStr, finalOutputDir);
                Platform.runLater(() -> {
                    String msg = String.format(
                            "Model recovered successfully!\n\n" +
                            "Best epoch: %d\nMean IoU: %.4f\nSaved to: %s",
                            result.bestEpoch(), result.bestMeanIoU(), result.modelPath());
                    Dialogs.showInfoNotification(EXTENSION_NAME, msg);
                    logger.info("Model recovered: {} (best epoch {}, mIoU {})",
                            result.modelPath(), result.bestEpoch(), result.bestMeanIoU());
                    // Remove the source checkpoint from the central registry
                    // now that we've finalized it to a proper model. Leave
                    // any project-dir copy alone -- TrainingWorkflow cleans
                    // those up after its own runs.
                    try {
                        Path registry = CheckpointScanner.getRegistryDir();
                        if (checkpointPath.startsWith(registry)) {
                            Files.deleteIfExists(checkpointPath);
                        }
                    } catch (IOException ignored) {}
                });
            } catch (Exception ex) {
                logger.error("Failed to recover from checkpoint", ex);
                Platform.runLater(() ->
                        Dialogs.showErrorNotification(EXTENSION_NAME,
                                "Failed to recover model: " + ex.getMessage()));
            }
        }, "DLClassifier-RecoverCheckpoint");
        recoverThread.setDaemon(true);
        recoverThread.start();
    }

    private void showSystemInfo() {
        // Collect Java-side info immediately
        StringBuilder sb = new StringBuilder();
        sb.append("=== QuPath / Extension ===\n");
        sb.append("QuPath version: ").append(GeneralTools.getVersion()).append("\n");
        String extVersion = GeneralTools.getPackageVersion(SetupDLClassifier.class);
        sb.append("Extension: ").append(EXTENSION_NAME)
                .append(extVersion != null ? " v" + extVersion : "").append("\n");
        sb.append("Backend mode: Appose (embedded Python)\n");

        ApposeService appose = ApposeService.getInstance();
        if (appose.isAvailable()) {
            sb.append("Appose status: initialized\n");
            sb.append("GPU type: ").append(appose.getGpuType()).append("\n");
        } else {
            String err = appose.getInitError();
            sb.append("Appose status: NOT available");
            if (err != null) sb.append(" (").append(err).append(")");
            sb.append("\n");
        }
        sb.append("Environment path: ").append(ApposeService.getEnvironmentPath()).append("\n");
        sb.append("\n");

        sb.append("=== Java / OS ===\n");
        sb.append("OS: ").append(System.getProperty("os.name")).append(" ")
                .append(System.getProperty("os.version")).append(" (")
                .append(System.getProperty("os.arch")).append(")\n");
        sb.append("JVM: ").append(System.getProperty("java.vm.name")).append(" ")
                .append(System.getProperty("java.version")).append("\n");
        sb.append("Max heap: ").append(Runtime.getRuntime().maxMemory() / (1024 * 1024)).append(" MB\n");
        sb.append("Available processors: ").append(Runtime.getRuntime().availableProcessors()).append("\n");
        sb.append("\n");

        String javaInfo = sb.toString();

        // Now run the Python system_info script for package versions & GPU details
        if (appose.isAvailable()) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "Collecting system information...");
            Thread infoThread = new Thread(() -> {
                String pythonInfo;
                try {
                    var task = appose.runTask("system_info", java.util.Map.of());
                    pythonInfo = String.valueOf(task.outputs.get("info_text"));
                } catch (Exception ex) {
                    logger.warn("Failed to collect Python system info: {}", ex.getMessage());
                    pythonInfo = "=== Python ===\nFailed to collect: " + ex.getMessage() + "\n";
                }

                String fullInfo = javaInfo + pythonInfo;
                Platform.runLater(() -> showSystemInfoDialog(fullInfo));
            }, "DLClassifier-SystemInfo");
            infoThread.setDaemon(true);
            infoThread.start();
        } else {
            String fullInfo = javaInfo + "=== Python ===\nAppose service not available.\n"
                    + "Python system info requires a working Appose environment.\n";
            showSystemInfoDialog(fullInfo);
        }
    }

    /**
     * Shows the system info in a dialog with a copyable text area.
     */
    private void showSystemInfoDialog(String infoText) {
        javafx.scene.control.TextArea textArea = new javafx.scene.control.TextArea(infoText);
        textArea.setEditable(false);
        textArea.setWrapText(false);
        textArea.setFont(javafx.scene.text.Font.font("monospace", 12));
        textArea.setPrefWidth(600);
        textArea.setPrefHeight(500);

        javafx.scene.control.Dialog<Void> dialog = new javafx.scene.control.Dialog<>();
        dialog.setTitle(EXTENSION_NAME + " - System Info");
        dialog.setHeaderText("System configuration and package versions");
        dialog.getDialogPane().setContent(textArea);
        dialog.getDialogPane().getButtonTypes().add(javafx.scene.control.ButtonType.CLOSE);
        dialog.setResizable(true);

        // Add a Copy button
        javafx.scene.control.ButtonType copyType = new javafx.scene.control.ButtonType(
                "Copy to Clipboard", javafx.scene.control.ButtonBar.ButtonData.LEFT);
        dialog.getDialogPane().getButtonTypes().add(copyType);

        // Prevent the Copy button from closing the dialog
        dialog.getDialogPane().lookupButton(copyType).addEventFilter(
                javafx.event.ActionEvent.ACTION, event -> {
                    javafx.scene.input.Clipboard clipboard = javafx.scene.input.Clipboard.getSystemClipboard();
                    javafx.scene.input.ClipboardContent content = new javafx.scene.input.ClipboardContent();
                    content.putString(infoText);
                    clipboard.setContent(content);
                    Dialogs.showInfoNotification(EXTENSION_NAME, "System info copied to clipboard.");
                    event.consume();
                });

        dialog.showAndWait();
    }

    /**
     * Shows a dialog listing all directories where this extension stores files,
     * with disk usage for each location and buttons to open them.
     */
    private void showWhereAreMyFiles() {
        String userHome = System.getProperty("user.home");
        String tempDir = System.getProperty("java.io.tmpdir");

        // Collect all file locations
        StringBuilder sb = new StringBuilder();

        // 1. Python environment
        Path envPath = ApposeService.getEnvironmentPath();
        sb.append("=== Python Environment (PyTorch, CUDA, dependencies) ===\n");
        sb.append("  ").append(envPath).append("\n");
        sb.append("  Size: ").append(formatDirSize(envPath)).append("\n\n");

        // 2. Project classifiers
        var project = QuPathGUI.getInstance().getProject();
        if (project != null) {
            Path projectClassifiers = project.getPath().getParent().resolve("classifiers/dl");
            sb.append("=== Project Classifiers (this project's trained models) ===\n");
            sb.append("  ").append(projectClassifiers).append("\n");
            sb.append("  Size: ").append(formatDirSize(projectClassifiers)).append("\n");
            sb.append("  Models: ").append(countSubdirs(projectClassifiers)).append("\n\n");
        } else {
            sb.append("=== Project Classifiers ===\n");
            sb.append("  No project open.\n\n");
        }

        // 3. User-level classifiers (shared across projects)
        Path userClassifiers = Path.of(userHome, ".qupath", "classifiers", "dl");
        sb.append("=== Shared Classifiers (available to all projects) ===\n");
        sb.append("  ").append(userClassifiers).append("\n");
        sb.append("  Size: ").append(formatDirSize(userClassifiers)).append("\n");
        sb.append("  Models: ").append(countSubdirs(userClassifiers)).append("\n\n");

        // 4. Default model storage (Python side)
        Path defaultModels = Path.of(userHome, ".dlclassifier", "models");
        sb.append("=== Default Model Storage (Python training output) ===\n");
        sb.append("  ").append(defaultModels).append("\n");
        sb.append("  Size: ").append(formatDirSize(defaultModels)).append("\n\n");

        // 5. Checkpoints
        Path checkpoints = Path.of(userHome, ".dlclassifier", "checkpoints");
        sb.append("=== Training Checkpoints (pause/resume state) ===\n");
        sb.append("  ").append(checkpoints).append("\n");
        sb.append("  Size: ").append(formatDirSize(checkpoints)).append("\n\n");

        // 6. Training data export
        String exportDir = DLClassifierPreferences.getTrainingExportDir();
        sb.append("=== Training Data Export (temporary tile images) ===\n");
        if (exportDir != null && !exportDir.isEmpty()) {
            sb.append("  Configured: ").append(exportDir).append("\n");
            sb.append("  Size: ").append(formatDirSize(Path.of(exportDir))).append("\n");
        } else {
            sb.append("  Using system temp: ").append(tempDir).append("\n");
        }
        sb.append("  (Normally cleaned up after training; orphaned on crash)\n\n");

        // 7. System temp - count dl-* orphans
        sb.append("=== System Temp Directory ===\n");
        sb.append("  ").append(tempDir).append("\n");
        long[] orphanInfo = countDLOrphans(Path.of(tempDir));
        sb.append("  DL classifier temp items: ").append(orphanInfo[0]).append("\n");
        if (orphanInfo[1] > 0) {
            sb.append("  Total size of DL temp items: ").append(formatBytes(orphanInfo[1])).append("\n");
        }
        if (orphanInfo[0] > 0) {
            sb.append("  (These may be orphaned from crashed/cancelled training runs)\n");
        }
        sb.append("\n");

        // Summary
        sb.append("=== Notes ===\n");
        sb.append("- Trained classifiers are saved in your QuPath project's classifiers/dl/ folder.\n");
        sb.append("- The Python environment is shared across all projects (~2-4 GB).\n");
        sb.append("- Temp files (dl-training-*, dl-pause-*, dl-overlay-*) in the system temp\n");
        sb.append("  directory are safe to delete when QuPath is not training.\n");
        sb.append("- To change where training tiles are exported, go to:\n");
        sb.append("  Edit > Preferences > DL Pixel Classifier > Training Data Export Directory\n");

        // Show dialog
        showWhereAreMyFilesDialog(sb.toString());
    }

    /**
     * Shows the file locations dialog with open-folder and copy buttons.
     */
    private void showWhereAreMyFilesDialog(String infoText) {
        javafx.scene.control.TextArea textArea = new javafx.scene.control.TextArea(infoText);
        textArea.setEditable(false);
        textArea.setWrapText(false);
        textArea.setFont(javafx.scene.text.Font.font("monospace", 12));
        textArea.setPrefWidth(700);
        textArea.setPrefHeight(550);

        javafx.scene.control.Dialog<Void> dialog = new javafx.scene.control.Dialog<>();
        dialog.setTitle(EXTENSION_NAME + " - Where Are My Files?");
        dialog.setHeaderText("All directories used by the DL Pixel Classifier extension");
        dialog.getDialogPane().setContent(textArea);
        dialog.getDialogPane().getButtonTypes().add(javafx.scene.control.ButtonType.CLOSE);
        dialog.setResizable(true);

        // Copy button
        javafx.scene.control.ButtonType copyType = new javafx.scene.control.ButtonType(
                "Copy to Clipboard", javafx.scene.control.ButtonBar.ButtonData.LEFT);
        dialog.getDialogPane().getButtonTypes().add(copyType);
        dialog.getDialogPane().lookupButton(copyType).addEventFilter(
                javafx.event.ActionEvent.ACTION, event -> {
                    javafx.scene.input.Clipboard clipboard = javafx.scene.input.Clipboard.getSystemClipboard();
                    javafx.scene.input.ClipboardContent content = new javafx.scene.input.ClipboardContent();
                    content.putString(infoText);
                    clipboard.setContent(content);
                    Dialogs.showInfoNotification(EXTENSION_NAME, "File locations copied to clipboard.");
                    event.consume();
                });

        // Open Temp Folder button
        javafx.scene.control.ButtonType openTempType = new javafx.scene.control.ButtonType(
                "Open Temp Folder", javafx.scene.control.ButtonBar.ButtonData.LEFT);
        dialog.getDialogPane().getButtonTypes().add(openTempType);
        dialog.getDialogPane().lookupButton(openTempType).addEventFilter(
                javafx.event.ActionEvent.ACTION, event -> {
                    try {
                        java.awt.Desktop.getDesktop().open(
                                new java.io.File(System.getProperty("java.io.tmpdir")));
                    } catch (Exception ex) {
                        logger.warn("Could not open temp folder", ex);
                    }
                    event.consume();
                });

        dialog.showAndWait();
    }

    /**
     * Formats the total size of a directory, or returns "does not exist" / "empty".
     */
    private static String formatDirSize(Path dir) {
        if (!Files.isDirectory(dir)) {
            return "(does not exist)";
        }
        try (var stream = Files.walk(dir)) {
            long totalBytes = stream
                    .filter(Files::isRegularFile)
                    .mapToLong(p -> {
                        try { return Files.size(p); }
                        catch (IOException e) { return 0; }
                    })
                    .sum();
            return totalBytes == 0 ? "(empty)" : formatBytes(totalBytes);
        } catch (IOException e) {
            return "(error reading)";
        }
    }

    /**
     * Counts immediate subdirectories in a directory.
     */
    private static int countSubdirs(Path dir) {
        if (!Files.isDirectory(dir)) return 0;
        try (var stream = Files.list(dir)) {
            return (int) stream.filter(Files::isDirectory).count();
        } catch (IOException e) {
            return 0;
        }
    }

    /**
     * Interactive cleanup utility: scans for orphaned files and shows the
     * user what will be deleted before proceeding.
     */
    /**
     * Presents a picker of available classifiers, then (via
     * {@link TrainingAreaIssuesDialog#openSavedSessionFromDisk}) a picker of
     * saved sessions, and reopens the dialog without re-running evaluation.
     */
    private void openSavedTrainingIssuesDialog() {
        try {
            ModelManager manager = new ModelManager();
            List<ClassifierMetadata> classifiers = manager.listClassifiers();
            if (classifiers == null || classifiers.isEmpty()) {
                Dialogs.showInfoNotification(EXTENSION_NAME,
                        "No classifiers found in project or user classifier directories.");
                return;
            }
            ClassifierMetadata metadata = Dialogs.showChoiceDialog(
                    "Load Saved Training Area Issues",
                    "Choose the classifier whose sessions you want to browse:",
                    classifiers,
                    classifiers.get(0));
            if (metadata == null) {
                return;
            }
            Path modelDir = manager.getModelPath(metadata.getId())
                    .map(Path::getParent)
                    .orElse(null);
            if (modelDir == null) {
                Dialogs.showErrorMessage(EXTENSION_NAME,
                        "Could not locate model directory for classifier '"
                                + metadata.getName() + "'.");
                return;
            }
            TrainingAreaIssuesDialog.openSavedSessionFromDisk(metadata, modelDir);
        } catch (Exception e) {
            logger.error("Failed to open saved Training Area Issues session", e);
            Dialogs.showErrorMessage(EXTENSION_NAME,
                    "Failed to open saved session: " + e.getMessage());
        }
    }

    private void cleanUpStorage() {
        try {
            String userHome = System.getProperty("user.home");
            Path modelsDir = Path.of(userHome, ".dlclassifier", "models");
            Path checkpointsDir = Path.of(userHome, ".dlclassifier", "checkpoints");
            Path tempDir = Path.of(System.getProperty("java.io.tmpdir"));

            StringBuilder report = new StringBuilder();
            long totalBytes = 0;
            int totalItems = 0;

            // 1. Orphaned models in ~/.dlclassifier/models/
            if (Files.isDirectory(modelsDir)) {
                try (var entries = Files.list(modelsDir)) {
                    var orphans = entries.filter(Files::isDirectory).toList();
                    if (!orphans.isEmpty()) {
                        long size = orphans.stream().mapToLong(this::dirSize).sum();
                        report.append(String.format(
                                "Orphaned model directories: %d (%.1f MB)\n  %s\n\n",
                                orphans.size(), size / (1024.0 * 1024.0), modelsDir));
                        totalBytes += size;
                        totalItems += orphans.size();
                    }
                }
            }

            // 2. Old checkpoints in ~/.dlclassifier/checkpoints/
            if (Files.isDirectory(checkpointsDir)) {
                try (var entries = Files.list(checkpointsDir)) {
                    var files = entries.filter(p -> {
                        String name = p.getFileName().toString();
                        return name.endsWith(".pt") || name.endsWith(".json");
                    }).toList();
                    if (!files.isEmpty()) {
                        long size = files.stream().mapToLong(p -> {
                            try { return Files.size(p); }
                            catch (IOException e) { return 0; }
                        }).sum();
                        report.append(String.format(
                                "Old checkpoint files: %d (%.1f MB)\n  %s\n\n",
                                files.size(), size / (1024.0 * 1024.0), checkpointsDir));
                        totalBytes += size;
                        totalItems += files.size();
                    }
                }
            }

            // 3. Orphaned temp directories (not active, any age)
            if (Files.isDirectory(tempDir)) {
                try (var entries = Files.list(tempDir)) {
                    var orphans = entries.filter(p -> {
                        String name = p.getFileName().toString();
                        return (name.startsWith("dl-training")
                                || name.startsWith("dl-pause")
                                || name.startsWith("dl-overlay")
                                || name.startsWith("dl-pixel-inference")
                                || name.startsWith("dl-rendered-overlay")
                                || name.startsWith("dl-classifier-import"))
                                && !Files.exists(p.resolve(
                                    qupath.ext.dlclassifier.controller.TrainingWorkflow.ACTIVE_MARKER));
                    }).toList();
                    if (!orphans.isEmpty()) {
                        long size = orphans.stream().mapToLong(this::dirSize).sum();
                        report.append(String.format(
                                "Orphaned temp directories: %d (%.1f GB)\n  %s\n\n",
                                orphans.size(), size / (1024.0 * 1024.0 * 1024.0), tempDir));
                        totalBytes += size;
                        totalItems += orphans.size();
                    }
                }
            }

            if (totalItems == 0) {
                Dialogs.showInfoNotification(EXTENSION_NAME, "No orphaned files found. Storage is clean.");
                return;
            }

            report.insert(0, String.format(
                    "Found %d items totaling %.1f GB that can be cleaned up:\n\n",
                    totalItems, totalBytes / (1024.0 * 1024.0 * 1024.0)));
            report.append("Delete all of the above?");

            boolean proceed = Dialogs.showConfirmDialog("Clean Up Storage", report.toString());
            if (!proceed) return;

            long deletedBytes = 0;
            int deletedItems = 0;

            // Delete orphaned models
            if (Files.isDirectory(modelsDir)) {
                try (var entries = Files.list(modelsDir)) {
                    for (Path orphan : entries.filter(Files::isDirectory).toList()) {
                        long size = dirSize(orphan);
                        deleteDirectory(orphan);
                        deletedBytes += size;
                        deletedItems++;
                    }
                }
            }

            // Delete old checkpoints
            if (Files.isDirectory(checkpointsDir)) {
                try (var entries = Files.list(checkpointsDir)) {
                    for (Path file : entries.toList()) {
                        long size = Files.size(file);
                        Files.deleteIfExists(file);
                        deletedBytes += size;
                        deletedItems++;
                    }
                }
            }

            // Delete orphaned temp dirs
            if (Files.isDirectory(tempDir)) {
                try (var entries = Files.list(tempDir)) {
                    for (Path orphan : entries.filter(p -> {
                        String name = p.getFileName().toString();
                        return (name.startsWith("dl-training")
                                || name.startsWith("dl-pause")
                                || name.startsWith("dl-overlay")
                                || name.startsWith("dl-pixel-inference")
                                || name.startsWith("dl-rendered-overlay")
                                || name.startsWith("dl-classifier-import"))
                                && !Files.exists(p.resolve(
                                    qupath.ext.dlclassifier.controller.TrainingWorkflow.ACTIVE_MARKER));
                    }).toList()) {
                        long size = dirSize(orphan);
                        deleteDirectory(orphan);
                        deletedBytes += size;
                        deletedItems++;
                    }
                }
            }

            Dialogs.showInfoNotification(EXTENSION_NAME,
                    String.format("Cleaned up %d items, freed %.1f GB",
                            deletedItems, deletedBytes / (1024.0 * 1024.0 * 1024.0)));
            logger.info("Storage cleanup: deleted {} items, freed {:.0f} MB",
                    deletedItems, deletedBytes / (1024.0 * 1024.0));

        } catch (Exception e) {
            logger.error("Storage cleanup failed", e);
            Dialogs.showErrorMessage("Clean Up Storage", "Cleanup failed: " + e.getMessage());
        }
    }

    private long dirSize(Path dir) {
        try (var walk = Files.walk(dir)) {
            return walk.filter(Files::isRegularFile)
                    .mapToLong(p -> {
                        try { return Files.size(p); }
                        catch (IOException e) { return 0; }
                    })
                    .sum();
        } catch (IOException e) {
            return 0;
        }
    }

    private void deleteDirectory(Path dir) {
        try (var walk = Files.walk(dir)) {
            walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> {
                        try { Files.deleteIfExists(p); }
                        catch (IOException ignored) {}
                    });
        } catch (IOException e) {
            logger.debug("Could not delete directory: {}", dir, e);
        }
    }

    /**
     * Deletes orphaned dl-training-*, dl-pause-*, dl-overlay-* directories
     * from the system temp directory.  Only deletes directories older than
     * 1 hour to avoid interfering with an active training run in another
     * QuPath instance.
     */
    private static void cleanupOrphanedTempDirs() {
        try {
            Path tempDir = Path.of(System.getProperty("java.io.tmpdir"));

            // Also clean the configured export directory if set
            String exportPref = DLClassifierPreferences.getTrainingExportDir();
            java.util.List<Path> dirsToScan = new java.util.ArrayList<>();
            if (Files.isDirectory(tempDir)) dirsToScan.add(tempDir);
            if (exportPref != null && !exportPref.isEmpty()) {
                Path exportDir = Path.of(exportPref);
                if (Files.isDirectory(exportDir) && !exportDir.equals(tempDir)) {
                    dirsToScan.add(exportDir);
                }
            }
            if (dirsToScan.isEmpty()) return;

            long cutoffMs = System.currentTimeMillis() - 3_600_000; // 1 hour ago
            long deletedCount = 0;
            long deletedBytes = 0;

            for (Path scanDir : dirsToScan)
            try (var stream = Files.list(scanDir)) {
                var orphans = stream
                        .filter(p -> {
                            String name = p.getFileName().toString();
                            return name.startsWith("dl-training")
                                    || name.startsWith("dl-pause")
                                    || name.startsWith("dl-overlay")
                                    || name.startsWith("dl-pixel-inference")
                                    || name.startsWith("dl-rendered-overlay")
                                    || name.startsWith("dl-classifier-import");
                        })
                        .filter(p -> {
                            // Never delete directories with an active training marker.
                            // The marker is written by TrainingWorkflow when training
                            // starts and removed when it finishes.  This prevents a
                            // second QuPath instance from deleting files that a
                            // running training session needs.
                            if (Files.exists(p.resolve(
                                    qupath.ext.dlclassifier.controller.TrainingWorkflow.ACTIVE_MARKER))) {
                                return false;
                            }
                            try {
                                return Files.getLastModifiedTime(p).toMillis() < cutoffMs;
                            } catch (IOException e) {
                                return false;
                            }
                        })
                        .toList();

                for (Path orphan : orphans) {
                    try {
                        long size = 0;
                        if (Files.isDirectory(orphan)) {
                            try (var walk = Files.walk(orphan)) {
                                size = walk.filter(Files::isRegularFile)
                                        .mapToLong(p -> {
                                            try { return Files.size(p); }
                                            catch (IOException e) { return 0; }
                                        })
                                        .sum();
                            }
                            try (var walk = Files.walk(orphan)) {
                                walk.sorted(java.util.Comparator.reverseOrder())
                                        .forEach(p -> {
                                            try { Files.deleteIfExists(p); }
                                            catch (IOException ignored) {}
                                        });
                            }
                        } else {
                            size = Files.size(orphan);
                            Files.deleteIfExists(orphan);
                        }
                        deletedCount++;
                        deletedBytes += size;
                    } catch (IOException e) {
                        logger.debug("Could not delete orphan: {}", orphan, e);
                    }
                }
            }

            if (deletedCount > 0) {
                double mb = deletedBytes / (1024.0 * 1024.0);
                logger.info("Cleaned up {} orphaned temp directories ({} MB)",
                        deletedCount, String.format("%.0f", mb));
                double finalMb = mb;
                Platform.runLater(() -> Dialogs.showInfoNotification(
                        EXTENSION_NAME,
                        String.format("Cleaned up %.0f MB of leftover training data "
                                + "from previous sessions.", finalMb)));
            }
        } catch (Exception e) {
            logger.debug("Orphan cleanup failed: {}", e.getMessage());
        }
    }

    /**
     * Counts dl-* prefixed items in the temp directory and their total size.
     *
     * @return [count, totalBytes]
     */
    private static long[] countDLOrphans(Path tempDir) {
        if (!Files.isDirectory(tempDir)) return new long[]{0, 0};
        long count = 0;
        long totalSize = 0;
        try (var stream = Files.list(tempDir)) {
            var items = stream
                    .filter(p -> p.getFileName().toString().startsWith("dl-"))
                    .toList();
            count = items.size();
            for (Path item : items) {
                if (Files.isDirectory(item)) {
                    try (var walk = Files.walk(item)) {
                        totalSize += walk
                                .filter(Files::isRegularFile)
                                .mapToLong(p -> {
                                    try { return Files.size(p); }
                                    catch (IOException e) { return 0; }
                                })
                                .sum();
                    }
                } else {
                    totalSize += Files.size(item);
                }
            }
        } catch (IOException e) {
            // ignore
        }
        return new long[]{count, totalSize};
    }

    /**
     * Formats a byte count as a human-readable string.
     */
    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    /**
     * Shows the setup environment dialog for first-time installation.
     */
    private void showSetupDialog(QuPathGUI qupath) {
        SetupEnvironmentDialog dialog = new SetupEnvironmentDialog(
                qupath.getStage(),
                () -> {
                    environmentReady.set(true);
                    serverAvailable = true;
                    logger.info("Environment setup completed via dialog");
                }
        );
        dialog.show();
    }

    /**
     * Rebuilds the DL environment by shutting down, deleting, and re-setting up.
     */
    private void rebuildEnvironment(QuPathGUI qupath) {
        boolean confirm = Dialogs.showConfirmDialog(
                res.getString("menu.rebuildEnvironment"),
                "This will shut down the Python service, delete the current environment,\n" +
                        "and re-download all dependencies (~2-4 GB).\n\n" +
                        "Continue?");
        if (!confirm) {
            return;
        }

        // Shut down the running service and delete the environment
        try {
            ApposeService.getInstance().shutdown();
            ApposeService.getInstance().deleteEnvironment();
        } catch (Exception e) {
            logger.error("Failed to delete environment", e);
            Dialogs.showErrorNotification(EXTENSION_NAME,
                    "Failed to delete environment: " + e.getMessage());
            return;
        }

        // Update state and show setup dialog
        environmentReady.set(false);
        serverAvailable = false;
        showSetupDialog(qupath);
    }

    /**
     * Launches the MAE pretraining workflow: health check, config dialog, progress monitor.
     */
    private void startMAEPretraining() {
        Dialogs.showInfoNotification(EXTENSION_NAME,
                "Connecting to classification backend...");

        CompletableFuture.supplyAsync(() -> DLClassifierChecks.checkServerHealth())
                .thenAcceptAsync(healthy -> {
                    if (healthy) {
                        showMAEPretrainingDialog();
                    } else {
                        String versionWarning = ApposeClassifierBackend.getVersionWarning();
                        if (versionWarning != null && !versionWarning.isEmpty()) {
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "Python environment is out of date.\n" +
                                    "Go to Rebuild Python Environment to update.");
                        } else {
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "Cannot connect to classification backend.\n\n" +
                                    "If this is the first launch, the Python environment\n" +
                                    "may still be downloading (~2-4 GB). Check the QuPath\n" +
                                    "log for progress and try again in a few minutes.");
                        }
                    }
                }, Platform::runLater);
    }

    /**
     * Shows the MAE pretraining config dialog and launches pretraining on success.
     */
    private void showMAEPretrainingDialog() {
        var configOpt = MAEPretrainingDialog.showDialog();
        if (configOpt.isEmpty()) return;

        var config = configOpt.get();

        // Ensure output directory exists
        try {
            Files.createDirectories(config.outputDir());
        } catch (IOException e) {
            Dialogs.showErrorNotification(EXTENSION_NAME,
                    "Cannot create output directory: " + e.getMessage());
            return;
        }

        // Project mode: make sure a temp tile dir can be created before opening progress
        final Path projectTempTileDir;
        if (config.sourceMode() == MAEPretrainingDialog.SourceMode.PROJECT_IMAGES) {
            try {
                // Use the configured training data export directory if set,
                // otherwise fall back to system temp
                String exportDirPref = DLClassifierPreferences.getTrainingExportDir();
                if (exportDirPref != null && !exportDirPref.isEmpty()) {
                    Path exportBase = Path.of(exportDirPref);
                    Files.createDirectories(exportBase);
                    projectTempTileDir = Files.createTempDirectory(exportBase, "mae-patches-");
                } else {
                    projectTempTileDir = Files.createTempDirectory("mae-patches-");
                }
            } catch (IOException e) {
                Dialogs.showErrorNotification(EXTENSION_NAME,
                        "Cannot create temp tile directory: " + e.getMessage());
                return;
            }
        } else {
            projectTempTileDir = null;
        }

        // Get Appose backend (MAE pretraining is Appose-only)
        ClassifierBackend backend = BackendFactory.getBackend();
        if (!(backend instanceof ApposeClassifierBackend apposeBackend)) {
            Dialogs.showErrorNotification(EXTENSION_NAME,
                    "MAE pretraining requires the Appose backend.");
            return;
        }

        // Create progress monitor for pretraining (loss chart + pause/stop)
        ProgressMonitorController progress = ProgressMonitorController.forPretraining();
        progress.show();

        // Tracks the currently-active pretraining job ID. Pause writes a signal
        // file keyed on this ID; the field is updated on resume so subsequent
        // pause/finalize calls hit the live worker.
        final String[] currentJobId = {null};
        final String runName = String.valueOf(config.config().getOrDefault("run_name", ""));
        final int totalEpochs = ((Number) config.config().getOrDefault("epochs", 100)).intValue();

        progress.setOnPause(v -> {
            if (currentJobId[0] != null) {
                try {
                    apposeBackend.pauseTraining(currentJobId[0]);
                    progress.log("Pause requested -- will pause after current epoch");
                } catch (IOException ex) {
                    logger.error("Failed to write pause signal", ex);
                    progress.log("ERROR: failed to write pause signal: " + ex.getMessage());
                }
            }
        });

        Thread pretrainThread = new Thread(() -> {
            try {
                Path effectiveDataPath = config.dataPath();
                if (config.sourceMode() == MAEPretrainingDialog.SourceMode.PROJECT_IMAGES) {
                    progress.log("Extracting MAE tiles from " + config.projectImages().size()
                            + " project image(s) into " + projectTempTileDir);
                    progress.setStatus("Extracting MAE tiles...");
                    extractMAETilesFromProject(config, projectTempTileDir, progress);
                    effectiveDataPath = projectTempTileDir.resolve("train").resolve("images");
                    if (!Files.isDirectory(effectiveDataPath)) {
                        throw new IOException(
                                "MAE tile extraction produced no tiles under "
                                        + effectiveDataPath + ". Ensure selected images have "
                                        + "Tissue-classed annotations.");
                    }
                    progress.log("MAE tile extraction complete; using tiles from "
                            + effectiveDataPath);
                }

                logger.info("Starting MAE pretraining: run={}, model={}, epochs={}, data={}",
                        runName,
                        config.config().get("model_config"),
                        config.config().get("epochs"),
                        effectiveDataPath);
                progress.log("MAE pretraining starting"
                        + (runName.isEmpty() ? "" : " (run: " + runName + ")") + "...");
                progress.log("Model: " + config.config().get("model_config")
                        + ", patch=" + config.config().get("patch_size")
                        + ", scales=" + config.config().get("level_scales"));
                progress.log("Training: " + config.config().get("epochs")
                        + " epochs, batch=" + config.config().get("batch_size")
                        + ", lr=" + config.config().get("learning_rate")
                        + ", mask=" + config.config().get("mask_ratio"));
                progress.log("Data: " + effectiveDataPath);
                progress.log("Output: " + config.outputDir());
                final Path dataPathForRun = effectiveDataPath;

                Consumer<ClassifierClient.TrainingProgress> progressCb =
                        buildPretrainProgressCallback(progress, "Pretraining", runName,
                                totalEpochs, new int[]{-1});

                ClassifierClient.TrainingResult result = apposeBackend.startPretraining(
                        config.config(),
                        dataPathForRun,
                        config.outputDir(),
                        progressCb,
                        progress::isCancelled,
                        newJobId -> {
                            currentJobId[0] = newJobId;
                            progress.onTrainingJobStarted();
                        }
                );

                handlePretrainResult("MAE", "pretrain_mae", apposeBackend, result, progress,
                        config.outputDir(), runName, currentJobId, dataPathForRun);

            } catch (IOException e) {
                if (progress.isCancelled()) {
                    progress.complete(false, "MAE pretraining cancelled.");
                    logger.info("MAE pretraining cancelled by user");
                } else {
                    logger.error("MAE pretraining failed", e);
                    progress.log("ERROR: " + e.getMessage());
                    progress.complete(false,
                            "MAE pretraining failed: " + e.getMessage());
                    Platform.runLater(() ->
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "MAE pretraining failed: " + e.getMessage()));
                }
            } finally {
                if (projectTempTileDir != null) {
                    try {
                        deleteRecursive(projectTempTileDir);
                        logger.info("Deleted MAE temp tile dir {}", projectTempTileDir);
                    } catch (IOException ex) {
                        logger.warn("Failed to delete MAE temp tile dir {}: {}",
                                projectTempTileDir, ex.toString());
                    }
                }
            }
        }, "DLClassifier-MAEPretrain");
        pretrainThread.setDaemon(true);
        pretrainThread.start();
    }

    /**
     * Runs AnnotationExtractor on the MAE config's project images, writing
     * tiles gated by the "Tissue" class to {@code tempDir/train/images}.
     * Then enforces the per-run total tile cap by shuffling and deleting
     * surplus tiles.
     */
    private static void extractMAETilesFromProject(
            MAEPretrainingDialog.MAEPretrainingConfig config,
            Path tempDir,
            ProgressMonitorController progress) throws IOException {

        List<qupath.lib.projects.ProjectImageEntry<java.awt.image.BufferedImage>> entries =
                config.projectImages();
        if (entries == null || entries.isEmpty()) {
            throw new IOException("No project images selected for MAE extraction");
        }

        // Derive a ChannelConfiguration from the first image. MAE doesn't care
        // about channel names; we just need a shape + bit depth consistent with
        // what the extractor will write.
        qupath.ext.dlclassifier.model.ChannelConfiguration channelConfig;
        try (var imageData = entries.get(0).readImageData().getServer()) {
            int bitDepth = imageData.getPixelType().getBitsPerPixel();
            int nCh = imageData.nChannels();
            java.util.List<Integer> idx = new java.util.ArrayList<>();
            java.util.List<String> names = new java.util.ArrayList<>();
            for (int i = 0; i < nCh; i++) {
                idx.add(i);
                names.add(imageData.getChannel(i).getName());
            }
            channelConfig = qupath.ext.dlclassifier.model.ChannelConfiguration.builder()
                    .selectedChannels(idx)
                    .channelNames(names)
                    .bitDepth(bitDepth)
                    .normalizationStrategy(
                            qupath.ext.dlclassifier.model.ChannelConfiguration
                                    .NormalizationStrategy.PERCENTILE_99)
                    .build();
        } catch (Exception ex) {
            throw new IOException("Failed to derive channel config: " + ex.getMessage(), ex);
        }

        // Delegate to AnnotationExtractor using "Tissue" as the only class. The
        // existing extractor already handles context padding, uint8/raw split,
        // and manifest writing. MAE will ignore masks.
        qupath.ext.dlclassifier.utilities.AnnotationExtractor.ExportResult result =
                qupath.ext.dlclassifier.utilities.AnnotationExtractor.exportFromProject(
                        entries,
                        config.extractionTileSize(),
                        channelConfig,
                        java.util.List.of(
                                qupath.ext.dlclassifier.utilities.TissueDetectionUtility.TISSUE_CLASS_NAME),
                        tempDir,
                        0.0, // validation split -- all to train/
                        3,   // line stroke width (irrelevant for polygon tissue annotations)
                        java.util.Collections.emptyMap(),
                        config.extractionDownsample(),
                        1, 0,
                        java.util.Collections.emptySet(),
                        java.util.Collections.emptySet());

        int totalTiles = result.totalPatches();
        progress.log("Extracted " + totalTiles + " tiles total");

        int cap = config.maxTilesTotal();
        if (cap > 0 && totalTiles > cap) {
            capTilesTotal(tempDir, cap, progress);
        }
    }

    /**
     * Keeps {@code cap} randomly chosen tiles in {@code tempDir/train/images}
     * (with their corresponding masks and context tiles, if present), deleting
     * the rest.
     */
    private static void capTilesTotal(Path tempDir, int cap,
                                      ProgressMonitorController progress) throws IOException {
        Path imagesDir = tempDir.resolve("train").resolve("images");
        Path masksDir = tempDir.resolve("train").resolve("masks");
        Path contextDir = tempDir.resolve("train").resolve("context");
        if (!Files.isDirectory(imagesDir)) return;

        java.util.List<Path> imageFiles = new java.util.ArrayList<>();
        try (var stream = Files.list(imagesDir)) {
            stream.filter(Files::isRegularFile).forEach(imageFiles::add);
        }
        if (imageFiles.size() <= cap) return;

        java.util.Collections.shuffle(imageFiles, new java.util.Random(0xC0FFEE));
        int removed = 0;
        for (int i = cap; i < imageFiles.size(); i++) {
            Path img = imageFiles.get(i);
            try { Files.deleteIfExists(img); } catch (IOException ignored) {}
            String base = img.getFileName().toString().replaceFirst("\\.(tiff?|raw)$", "");
            Path mask = masksDir.resolve(base + ".png");
            try { Files.deleteIfExists(mask); } catch (IOException ignored) {}
            if (Files.isDirectory(contextDir)) {
                for (String ext : new String[]{".tiff", ".tif", ".raw"}) {
                    try { Files.deleteIfExists(contextDir.resolve(base + ext)); }
                    catch (IOException ignored) {}
                }
            }
            removed++;
        }
        progress.log("Capped MAE tiles to " + cap + " (removed " + removed + ")");
    }

    // ==================== SSL Pretraining ====================

    private void startSSLPretraining() {
        Dialogs.showInfoNotification(EXTENSION_NAME,
                "Connecting to classification backend...");

        CompletableFuture.supplyAsync(() -> DLClassifierChecks.checkServerHealth())
                .thenAcceptAsync(healthy -> {
                    if (healthy) {
                        showSSLPretrainingDialog();
                    } else {
                        String versionWarning = ApposeClassifierBackend.getVersionWarning();
                        if (versionWarning != null && !versionWarning.isEmpty()) {
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "Python environment is out of date.\n" +
                                    "Go to Rebuild Python Environment to update.");
                        } else {
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "Cannot connect to classification backend.\n\n" +
                                    "If this is the first launch, the Python environment\n" +
                                    "may still be downloading. Check the QuPath log\n" +
                                    "for progress and try again in a few minutes.");
                        }
                    }
                }, Platform::runLater);
    }

    private void showSSLPretrainingDialog() {
        var configOpt = SSLPretrainingDialog.showDialog();
        if (configOpt.isEmpty()) return;

        var config = configOpt.get();

        try {
            Files.createDirectories(config.outputDir());
        } catch (IOException e) {
            Dialogs.showErrorNotification(EXTENSION_NAME,
                    "Cannot create output directory: " + e.getMessage());
            return;
        }

        final Path projectTempTileDir;
        if (config.sourceMode() == SSLPretrainingDialog.SourceMode.PROJECT_IMAGES) {
            try {
                // Use the configured training data export directory if set,
                // otherwise fall back to system temp
                String exportDirPref = DLClassifierPreferences.getTrainingExportDir();
                if (exportDirPref != null && !exportDirPref.isEmpty()) {
                    Path exportBase = Path.of(exportDirPref);
                    Files.createDirectories(exportBase);
                    projectTempTileDir = Files.createTempDirectory(exportBase, "ssl-patches-");
                } else {
                    projectTempTileDir = Files.createTempDirectory("ssl-patches-");
                }
            } catch (IOException e) {
                Dialogs.showErrorNotification(EXTENSION_NAME,
                        "Cannot create temp tile directory: " + e.getMessage());
                return;
            }
        } else {
            projectTempTileDir = null;
        }

        ClassifierBackend backend = BackendFactory.getBackend();
        if (!(backend instanceof ApposeClassifierBackend apposeBackend)) {
            Dialogs.showErrorNotification(EXTENSION_NAME,
                    "SSL pretraining requires the Appose backend.");
            return;
        }

        ProgressMonitorController progress = ProgressMonitorController.forSSLPretraining();
        progress.show();

        final String[] currentJobId = {null};
        final String runName = String.valueOf(config.config().getOrDefault("run_name", ""));
        final int totalEpochs = ((Number) config.config().getOrDefault("epochs", 100)).intValue();
        progress.setOnPause(v -> {
            if (currentJobId[0] != null) {
                try {
                    apposeBackend.pauseTraining(currentJobId[0]);
                    progress.log("Pause requested -- will pause after current epoch");
                } catch (IOException ex) {
                    logger.error("Failed to write pause signal", ex);
                    progress.log("ERROR: failed to write pause signal: " + ex.getMessage());
                }
            }
        });

        Thread pretrainThread = new Thread(() -> {
            try {
                Path effectiveDataPath = config.dataPath();
                if (config.sourceMode() == SSLPretrainingDialog.SourceMode.PROJECT_IMAGES) {
                    int nImages = config.projectImages().size();
                    progress.log("Extracting SSL tiles from " + nImages
                            + " project image(s) into " + projectTempTileDir);
                    progress.setStatus("Extracting tiles (" + nImages + " images)...");

                    // Disk space pre-check
                    long estimatedBytes = (long) config.extractionTileSize()
                            * config.extractionTileSize() * 3 * 4
                            * Math.min(config.maxTilesTotal(), 200000);
                    long freeBytes = projectTempTileDir.toFile().getUsableSpace();
                    if (freeBytes > 0 && estimatedBytes > freeBytes * 0.8) {
                        progress.log(String.format(
                                "WARNING: Estimated %.1f GB needed, %.1f GB free on %s",
                                estimatedBytes / 1e9, freeBytes / 1e9,
                                projectTempTileDir.getRoot()));
                    }

                    extractSSLTilesFromProject(config, projectTempTileDir, progress);
                    effectiveDataPath = projectTempTileDir.resolve("train").resolve("images");
                    if (!Files.isDirectory(effectiveDataPath)) {
                        throw new IOException(
                                "SSL tile extraction produced no tiles under "
                                        + effectiveDataPath + ". Ensure selected images have "
                                        + "annotations of the selected classes.");
                    }
                    progress.log("SSL tile extraction complete; using tiles from "
                            + effectiveDataPath);
                }

                String method = (String) config.config().get("method");
                String encoder = (String) config.config().get("encoder_name");
                logger.info("Starting SSL pretraining: method={}, encoder={}, epochs={}, data={}",
                        method, encoder, config.config().get("epochs"), effectiveDataPath);
                progress.log("SSL pretraining starting...");
                progress.log("Method: " + method + ", Backbone: " + encoder);
                progress.log("Training: " + config.config().get("epochs")
                        + " epochs, batch=" + config.config().get("batch_size")
                        + ", lr=" + config.config().get("learning_rate"));
                progress.log("Data: " + effectiveDataPath);
                progress.log("Output: " + config.outputDir());
                final Path dataPathForRun = effectiveDataPath;

                Consumer<ClassifierClient.TrainingProgress> sslProgressCb =
                        buildSSLPretrainProgressCallback(progress, runName,
                                totalEpochs, new int[]{-1});

                ClassifierClient.TrainingResult result = apposeBackend.startSSLPretraining(
                        config.config(),
                        dataPathForRun,
                        config.outputDir(),
                        sslProgressCb,
                        progress::isCancelled,
                        newJobId -> {
                            currentJobId[0] = newJobId;
                            progress.onTrainingJobStarted();
                        }
                );

                handlePretrainResult("SSL", "pretrain_ssl", apposeBackend, result, progress,
                        config.outputDir(), runName, currentJobId, dataPathForRun);

            } catch (IOException e) {
                if (progress.isCancelled()) {
                    progress.complete(false, "SSL pretraining cancelled.");
                    logger.info("SSL pretraining cancelled by user");
                } else {
                    logger.error("SSL pretraining failed", e);
                    progress.log("ERROR: " + e.getMessage());
                    progress.complete(false,
                            "SSL pretraining failed: " + e.getMessage());
                    Platform.runLater(() ->
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "SSL pretraining failed: " + e.getMessage()));
                }
            } finally {
                if (projectTempTileDir != null) {
                    try {
                        deleteRecursive(projectTempTileDir);
                        logger.info("Deleted SSL temp tile dir {}", projectTempTileDir);
                    } catch (IOException ex) {
                        logger.warn("Failed to delete SSL temp tile dir {}: {}",
                                projectTempTileDir, ex.toString());
                    }
                }
            }
        }, "DLClassifier-SSLPretrain");
        pretrainThread.setDaemon(true);
        pretrainThread.start();
    }

    /**
     * Extracts tiles from user-selected annotation classes for SSL pretraining.
     * Same approach as MAE extraction but with configurable annotation classes.
     */
    private static void extractSSLTilesFromProject(
            SSLPretrainingDialog.SSLPretrainingConfig config,
            Path tempDir,
            ProgressMonitorController progress) throws IOException {

        List<qupath.lib.projects.ProjectImageEntry<java.awt.image.BufferedImage>> entries =
                config.projectImages();
        if (entries == null || entries.isEmpty()) {
            throw new IOException("No project images selected for SSL extraction");
        }

        List<String> annotationClasses = config.annotationClasses();
        if (annotationClasses == null || annotationClasses.isEmpty()) {
            throw new IOException("No annotation classes selected for SSL extraction");
        }

        // Derive ChannelConfiguration from first image
        qupath.ext.dlclassifier.model.ChannelConfiguration channelConfig;
        try (var imageData = entries.get(0).readImageData().getServer()) {
            int bitDepth = imageData.getPixelType().getBitsPerPixel();
            int nCh = imageData.nChannels();
            java.util.List<Integer> idx = new java.util.ArrayList<>();
            java.util.List<String> names = new java.util.ArrayList<>();
            for (int i = 0; i < nCh; i++) {
                idx.add(i);
                names.add(imageData.getChannel(i).getName());
            }
            channelConfig = qupath.ext.dlclassifier.model.ChannelConfiguration.builder()
                    .selectedChannels(idx)
                    .channelNames(names)
                    .bitDepth(bitDepth)
                    .normalizationStrategy(
                            qupath.ext.dlclassifier.model.ChannelConfiguration
                                    .NormalizationStrategy.PERCENTILE_99)
                    .build();
        } catch (Exception ex) {
            throw new IOException("Failed to derive channel config: " + ex.getMessage(), ex);
        }

        qupath.ext.dlclassifier.utilities.AnnotationExtractor.ExportResult result =
                qupath.ext.dlclassifier.utilities.AnnotationExtractor.exportFromProject(
                        entries,
                        config.extractionTileSize(),
                        channelConfig,
                        annotationClasses,
                        tempDir,
                        0.0,  // validation split -- all to train/
                        3,    // line stroke width
                        java.util.Collections.emptyMap(),
                        config.extractionDownsample(),
                        1, 0,
                        java.util.Collections.emptySet(),
                        java.util.Collections.emptySet());

        int totalTiles = result.totalPatches();
        progress.log("Extracted " + totalTiles + " tiles total from "
                + annotationClasses.size() + " annotation class(es): " + annotationClasses);

        int cap = config.maxTilesTotal();
        if (cap > 0 && totalTiles > cap) {
            capTilesTotal(tempDir, cap, progress);
        }
    }

    private static void deleteRecursive(Path root) throws IOException {
        if (!Files.exists(root)) return;
        try (var stream = Files.walk(root)) {
            stream.sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> {
                        try { Files.deleteIfExists(p); } catch (IOException ignored) {}
                    });
        }
    }

    /**
     * Shows a one-time informational notice about overlay generation time.
     * <p>
     * Returns true if the user acknowledged (or already dismissed it before),
     * false if cancelled.
     */
    private boolean showOverlayNoticeIfNeeded() {
        if (DLClassifierPreferences.isOverlayNoticeDismissed()) {
            return true;
        }

        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(EXTENSION_NAME);
        alert.setHeaderText("Overlay Generation");

        Label message = new Label(
                "Depending on your hardware and model size, the prediction overlay " +
                "may take a moment to appear as tiles are classified on demand.\n\n" +
                "The overlay will fill in progressively as you pan and zoom.");
        message.setWrapText(true);
        message.setMaxWidth(400);

        CheckBox dontShowAgain = new CheckBox("Do not show this message again");

        VBox content = new VBox(10, message, dontShowAgain);
        content.setPadding(new Insets(10, 0, 0, 0));
        alert.getDialogPane().setContent(content);

        alert.showAndWait();
        if (dontShowAgain.isSelected()) {
            DLClassifierPreferences.setOverlayNoticeDismissed(true);
        }
        return true;
    }

    /**
     * Prompts the user to select a classifier for overlay use.
     * Stores the selection in OverlayService and immediately creates the overlay
     * if an image is open.
     */
    private void selectOverlayModel(QuPathGUI qupath, OverlayService overlayService) {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showWarningNotification(EXTENSION_NAME, "No image is open.");
            return;
        }

        // List available classifiers
        ModelManager modelManager = new ModelManager();
        List<ClassifierMetadata> classifiers = modelManager.listClassifiers();
        if (classifiers.isEmpty()) {
            Dialogs.showWarningNotification(EXTENSION_NAME,
                    "No classifiers available. Train a classifier first.");
            return;
        }

        // Sort by creation date (newest first)
        classifiers.sort((a, b) -> {
            if (a.getCreatedAt() == null && b.getCreatedAt() == null) return 0;
            if (a.getCreatedAt() == null) return 1;
            if (b.getCreatedAt() == null) return -1;
            return b.getCreatedAt().compareTo(a.getCreatedAt());
        });

        // Show a choice dialog with informative labels:
        // Name | architecture/backbone | classes | date
        List<String> names = classifiers.stream()
                .map(c -> {
                    String arch = c.getModelType() != null ? c.getModelType() : "?";
                    String backbone = c.getBackbone() != null ? c.getBackbone() : "";
                    String classCount = c.getClasses() != null
                            ? c.getClasses().size() + " classes" : "";
                    String date = "";
                    if (c.getCreatedAt() != null) {
                        var dt = c.getCreatedAt();
                        date = String.format("%d-%02d-%02d %02d:%02d",
                                dt.getYear(), dt.getMonthValue(), dt.getDayOfMonth(),
                                dt.getHour(), dt.getMinute());
                    }
                    return String.format("%s  [%s%s, %s]  %s",
                            c.getName(),
                            arch,
                            backbone.isEmpty() ? "" : "/" + backbone,
                            classCount,
                            date);
                })
                .toList();
        String defaultChoice = names.get(0);
        if (overlayService.getSelectedMetadata() != null) {
            String currentId = overlayService.getSelectedMetadata().getId();
            for (int i = 0; i < classifiers.size(); i++) {
                if (classifiers.get(i).getId().equals(currentId)) {
                    defaultChoice = names.get(i);
                    break;
                }
            }
        }
        String choice = Dialogs.showChoiceDialog("Select Overlay Model",
                "Choose a classifier for the prediction overlay:", names, defaultChoice);
        if (choice == null) {
            return;
        }

        // Find the selected classifier
        int selectedIdx = names.indexOf(choice);
        ClassifierMetadata metadata = classifiers.get(selectedIdx);

        // Build channel config from metadata
        List<String> expectedChannels = metadata.getExpectedChannelNames();
        List<Integer> selectedChannels = new java.util.ArrayList<>();
        for (int i = 0; i < Math.max(expectedChannels.size(), metadata.getInputChannels()); i++) {
            selectedChannels.add(i);
        }
        ChannelConfiguration channelConfig = ChannelConfiguration.builder()
                .selectedChannels(selectedChannels)
                .channelNames(expectedChannels)
                .bitDepth(metadata.getBitDepthTrained())
                .normalizationStrategy(metadata.getNormalizationStrategy())
                .build();

        // Store selection and create overlay
        overlayService.selectModel(metadata, channelConfig);
        showOverlayNoticeIfNeeded();
        overlayService.createOverlayFromSelection(imageData);
        Dialogs.showInfoNotification(EXTENSION_NAME,
                "Overlay model: " + metadata.getName());
    }

    /**
     * Formats a user-friendly device message from progress info.
     */
    /** Format a duration in milliseconds to a short human-readable string (e.g., "5m 30s"). */
    private static String formatSSLDuration(long millis) {
        long totalSec = millis / 1000;
        if (totalSec < 60) return totalSec + "s";
        long min = totalSec / 60;
        long sec = totalSec % 60;
        if (min < 60) return min + "m " + sec + "s";
        long hr = min / 60;
        min = min % 60;
        return hr + "h " + min + "m";
    }

    private static String formatDeviceMessage(String device, String deviceInfo) {
        if (device == null || device.isEmpty()) {
            return "Python backend ready, building model...";
        }
        return switch (device) {
            case "cuda" -> {
                String gpuName = (deviceInfo != null && !deviceInfo.isEmpty()
                        && !"CPU".equals(deviceInfo)) ? deviceInfo : "NVIDIA GPU";
                yield "Pretraining on " + gpuName + " (CUDA)";
            }
            case "mps" -> "Pretraining on Apple Silicon (MPS)";
            case "cpu" -> "Pretraining on CPU (no GPU detected -- this will be slow)";
            default -> "Pretraining on device: " + device;
        };
    }

    /**
     * Converts a setup phase identifier to a user-friendly status message.
     */
    private static String formatSetupPhase(String phase) {
        if (phase == null) return "Setting up...";
        return switch (phase) {
            case "creating_model" -> "Creating model architecture...";
            case "loading_data" -> "Loading image tiles...";
            case "computing_stats" -> "Computing normalization statistics...";
            case "starting_training" -> "Starting first epoch...";
            case "saving_model" -> "Saving encoder weights...";
            case "training_batch" -> "Training...";
            default -> "Setting up (" + phase + ")...";
        };
    }

    /**
     * Builds a progress callback for MAE pretraining. Tracks first-epoch logging
     * and prefixes the status line with the run name when one was provided.
     */
    private Consumer<ClassifierClient.TrainingProgress> buildPretrainProgressCallback(
            ProgressMonitorController progress, String verb, String runName,
            int totalEpochs, int[] lastLoggedEpoch) {
        String runLabel = (runName != null && !runName.isEmpty()) ? " " + runName : "";
        return maeProgress -> {
            if (maeProgress.isSetupPhase()) {
                if ("initializing".equals(maeProgress.status())) {
                    String deviceMsg = formatDeviceMessage(
                            maeProgress.device(), maeProgress.deviceInfo());
                    progress.log(deviceMsg);
                    progress.setStatus("Initializing for "
                            + maeProgress.totalEpochs() + " epoch run...");
                } else {
                    String phaseMsg = formatSetupPhase(maeProgress.setupPhase());
                    progress.setStatus(phaseMsg);
                    progress.log(phaseMsg);
                }
                return;
            }
            if (lastLoggedEpoch[0] < 0) {
                progress.log("Training loop started");
                progress.setStatus(verb + runLabel
                        + " (" + maeProgress.totalEpochs() + " epochs)...");
            }
            if (maeProgress.totalEpochs() > 0) {
                progress.setOverallProgress(
                        (double) maeProgress.epoch() / maeProgress.totalEpochs());
            }
            progress.setDetail(String.format(
                    "Epoch %d/%d  |  Loss: %.4f",
                    maeProgress.epoch(), maeProgress.totalEpochs(), maeProgress.loss()));
            progress.updateTrainingMetrics(
                    maeProgress.epoch(), maeProgress.totalEpochs(),
                    maeProgress.loss(), Double.NaN, null, null);
            int epoch = maeProgress.epoch();
            if (epoch == 1 || epoch % 10 == 0 || epoch == maeProgress.totalEpochs()) {
                progress.log(String.format("Epoch %d/%d: loss=%.6f",
                        epoch, maeProgress.totalEpochs(), maeProgress.loss()));
                lastLoggedEpoch[0] = epoch;
            }
        };
    }

    /**
     * Builds a progress callback for SSL pretraining. Same shape as MAE but
     * forwards SSL-specific timing fields (images/sec, ETA) into the detail line.
     */
    private Consumer<ClassifierClient.TrainingProgress> buildSSLPretrainProgressCallback(
            ProgressMonitorController progress, String runName,
            int totalEpochs, int[] lastLoggedEpoch) {
        String runLabel = (runName != null && !runName.isEmpty()) ? " " + runName : "";
        return sslProgress -> {
            if (sslProgress.isSetupPhase()) {
                if ("initializing".equals(sslProgress.status())) {
                    progress.log(formatDeviceMessage(
                            sslProgress.device(), sslProgress.deviceInfo()));
                    progress.setStatus("Initializing for "
                            + sslProgress.totalEpochs() + " epoch run...");
                } else {
                    String phase = sslProgress.setupPhase();
                    String dataMsg = sslProgress.configSummary() != null
                            ? sslProgress.configSummary().get("message") : null;
                    if (dataMsg != null && !dataMsg.isEmpty()) {
                        progress.log(dataMsg);
                    }
                    String phaseMsg = formatSetupPhase(phase);
                    progress.setStatus(phaseMsg);
                    if (dataMsg == null || dataMsg.isEmpty()) {
                        progress.log(phaseMsg);
                    }
                }
                return;
            }
            if (lastLoggedEpoch[0] < 0) {
                progress.log("Training loop started");
                progress.setStatus("Pretraining" + runLabel
                        + " (" + sslProgress.totalEpochs() + " epochs)...");
            }
            if (sslProgress.totalEpochs() > 0) {
                progress.setOverallProgress(
                        (double) sslProgress.epoch() / sslProgress.totalEpochs());
            }
            Map<String, String> timing = sslProgress.configSummary();
            double imgPerSec = 0, remainingSec = 0;
            try {
                if (timing != null && timing.containsKey("images_per_sec"))
                    imgPerSec = Double.parseDouble(timing.get("images_per_sec"));
                if (timing != null && timing.containsKey("remaining_sec"))
                    remainingSec = Double.parseDouble(timing.get("remaining_sec"));
            } catch (NumberFormatException ignored) {}
            if (imgPerSec > 0 && remainingSec > 0) {
                progress.setDetail(String.format(
                        "Epoch %d/%d  |  Loss: %.4f  |  %.0f img/s  |  ~%s left",
                        sslProgress.epoch(), sslProgress.totalEpochs(),
                        sslProgress.loss(), imgPerSec,
                        formatSSLDuration((long)(remainingSec * 1000))));
            } else {
                progress.setDetail(String.format(
                        "Epoch %d/%d  |  Loss: %.4f",
                        sslProgress.epoch(), sslProgress.totalEpochs(), sslProgress.loss()));
            }
            progress.updateTrainingMetrics(
                    sslProgress.epoch(), sslProgress.totalEpochs(),
                    sslProgress.loss(), Double.NaN, null, null);
            int epoch = sslProgress.epoch();
            progress.log(String.format("Epoch %d/%d: loss=%.6f",
                    epoch, sslProgress.totalEpochs(), sslProgress.loss()));
            lastLoggedEpoch[0] = epoch;
        };
    }

    /**
     * Handles the result of a pretraining (or resumed pretraining) run.
     * On pause, shows the paused state and wires Resume/Complete callbacks.
     * On normal completion, shows the success message and offers a notification.
     *
     * @param label        "MAE" or "SSL" -- used in user-visible messages
     * @param taskName     Appose task name for resume ("pretrain_mae" or "pretrain_ssl")
     * @param backend      Appose backend
     * @param result       result of the just-finished run
     * @param progress     progress monitor
     * @param outputDir    pretraining output directory (where pause_checkpoint.pt lives)
     * @param runName      run name to show in resumed/pretraining headers
     * @param currentJobId mutable holder for the active job ID
     * @param dataPath     data path used for the original run (re-used on resume)
     */
    private void handlePretrainResult(String label, String taskName,
                                      ApposeClassifierBackend backend,
                                      ClassifierClient.TrainingResult result,
                                      ProgressMonitorController progress,
                                      Path outputDir, String runName,
                                      String[] currentJobId, Path dataPath) {
        if (result.isPaused()) {
            int last = result.lastEpoch();
            int total = result.totalEpochs();
            progress.showPausedState(last, total);
            progress.log(label + " pretraining paused at epoch " + last + "/" + total
                    + ". Resume to continue or Complete Training to save best encoder.");

            progress.setOnResume(v -> {
                Thread resumeThread = new Thread(() -> {
                    try {
                        progress.showResumedState();
                        Consumer<ClassifierClient.TrainingProgress> cb =
                                "pretrain_ssl".equals(taskName)
                                        ? buildSSLPretrainProgressCallback(
                                                progress, runName, total, new int[]{-1})
                                        : buildPretrainProgressCallback(
                                                progress, "Pretraining", runName,
                                                total, new int[]{-1});
                        ClassifierClient.TrainingResult resumeResult =
                                backend.resumePretraining(taskName, currentJobId[0], outputDir,
                                        cb, progress::isCancelled,
                                        newJobId -> {
                                            currentJobId[0] = newJobId;
                                            progress.onTrainingJobStarted();
                                        });
                        handlePretrainResult(label, taskName, backend, resumeResult,
                                progress, outputDir, runName, currentJobId, dataPath);
                    } catch (IOException ex) {
                        logger.error("{} resume failed", label, ex);
                        progress.log("ERROR: resume failed: " + ex.getMessage());
                        progress.complete(false, label + " resume failed: " + ex.getMessage());
                    }
                }, "DLClassifier-" + label + "Resume");
                resumeThread.setDaemon(true);
                resumeThread.start();
            });

            progress.setOnCompleteEarly(v -> {
                Thread finalizeThread = new Thread(() -> {
                    try {
                        progress.setStatus("Saving encoder...");
                        progress.log("Finalizing: extracting best encoder from "
                                + result.checkpointPath());
                        ClassifierClient.TrainingResult finalResult =
                                backend.finalizePretraining(result.checkpointPath(), outputDir);
                        String encoderPath = finalResult.modelPath();
                        progress.complete(true, String.format(
                                "Encoder saved to:%n%s%n%nBest loss: %.4f",
                                encoderPath, finalResult.finalLoss()));
                        Platform.runLater(() ->
                                Dialogs.showInfoNotification(EXTENSION_NAME,
                                        label + " pretraining complete (early stop). Encoder saved to:\n"
                                                + encoderPath));
                    } catch (IOException ex) {
                        logger.error("{} finalize failed", label, ex);
                        progress.log("ERROR: finalize failed: " + ex.getMessage());
                        progress.complete(false,
                                label + " finalize failed: " + ex.getMessage());
                    }
                }, "DLClassifier-" + label + "Finalize");
                finalizeThread.setDaemon(true);
                finalizeThread.start();
            });
            return;
        }

        String encoderPath = result.modelPath();
        boolean hasSavedModel = encoderPath != null && !encoderPath.isEmpty();
        if (hasSavedModel) {
            // Quality flags from Python: "ok", "warn", "likely_collapse",
            // "aborted_collapse", "cancelled". The collapse-probe variants
            // mean the encoder is almost certainly unusable -- show that
            // prominently rather than burying it under a "complete!"
            // headline. Cancelled runs get their own banner so the user
            // doesn't mistake an early-stop encoder for a finished one.
            String quality = result.quality();
            java.util.List<String> warnings = result.warnings();
            boolean collapsed = "likely_collapse".equals(quality)
                    || "aborted_collapse".equals(quality);
            boolean cancelled = result.cancelled()
                    || "cancelled".equals(quality);
            boolean hasWarnings = result.hasQualityWarnings();

            StringBuilder messageBuilder = new StringBuilder();
            if (collapsed) {
                messageBuilder.append("[REVIEW WARNINGS] The collapse probe flagged this run.\n")
                        .append("The saved encoder is almost certainly unusable for downstream\n")
                        .append("supervised training. See details below.\n\n");
            } else if (cancelled) {
                messageBuilder.append("[CANCELLED] Training was stopped before normal completion.\n")
                        .append("The encoder reflects the best epoch's weights up to the\n")
                        .append("point of cancellation, NOT a fully trained model.\n\n");
            } else if (hasWarnings) {
                messageBuilder.append("[REVIEW WARNINGS] This run completed with quality warnings.\n\n");
            }
            if ("MAE".equals(label)) {
                messageBuilder.append(String.format(
                        "Encoder saved to:%n%s%n%nFinal reconstruction loss: %.4f%n%n"
                                + "To use: In the training dialog, select MuViT and%n"
                                + "choose 'Continue from model' to load this encoder.",
                        encoderPath, result.finalLoss()));
            } else {
                messageBuilder.append(String.format(
                        "Encoder saved to:%n%s%n%nFinal loss: %.4f%n%n"
                                + "To use: In the training dialog, select UNet and%n"
                                + "choose 'Use SSL pretrained encoder' to load this backbone.",
                        encoderPath, result.finalLoss()));
            }
            if (hasWarnings) {
                messageBuilder.append("\n\n--- Warnings ---");
                for (String w : warnings) {
                    messageBuilder.append("\n* ").append(w);
                }
            }
            String message = messageBuilder.toString();
            progress.complete(true, message);
            logger.info("{} pretraining {}: loss={}, path={}, quality={}, warnings={}",
                    label, cancelled ? "cancelled (saved)" : "complete",
                    result.finalLoss(), encoderPath, quality, warnings.size());
            for (String w : warnings) {
                logger.warn("{} pretraining warning: {}", label, w);
            }
            Platform.runLater(() -> {
                if (collapsed) {
                    // Use a blocking warning dialog so the user can't miss it,
                    // then a notification as a follow-up reminder.
                    Dialogs.showWarningNotification(EXTENSION_NAME,
                            label + " pretraining flagged: encoder may be unusable. See dialog for details.");
                    Alert alert = new Alert(Alert.AlertType.WARNING);
                    alert.setTitle(label + " pretraining: encoder collapsed");
                    alert.setHeaderText("Collapse probe aborted training");
                    alert.setContentText(message);
                    alert.getDialogPane().setPrefWidth(560);
                    alert.show();
                } else if (cancelled) {
                    Dialogs.showWarningNotification(EXTENSION_NAME,
                            label + " pretraining cancelled. Partial encoder saved -- see dialog.");
                    Alert alert = new Alert(Alert.AlertType.WARNING);
                    alert.setTitle(label + " pretraining: cancelled");
                    alert.setHeaderText("Training stopped early; partial encoder saved");
                    alert.setContentText(message);
                    alert.getDialogPane().setPrefWidth(560);
                    alert.show();
                } else if (hasWarnings) {
                    Dialogs.showWarningNotification(EXTENSION_NAME,
                            label + " pretraining complete with warnings. See dialog for details.");
                } else {
                    Dialogs.showInfoNotification(EXTENSION_NAME,
                            label + " pretraining complete! Encoder saved to:\n" + encoderPath);
                }
            });
        } else {
            progress.complete(false, label + " pretraining cancelled (no model saved).");
            logger.info("{} pretraining cancelled, no model saved", label);
        }
    }

}
