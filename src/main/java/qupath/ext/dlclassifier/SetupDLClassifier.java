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
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.ui.MAEPretrainingDialog;
import qupath.ext.dlclassifier.ui.ProgressMonitorController;
import qupath.ext.dlclassifier.ui.PythonConsoleWindow;
import qupath.ext.dlclassifier.ui.SetupEnvironmentDialog;
import qupath.ext.dlclassifier.ui.TooltipHelper;
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
            GitHubRepo.create(EXTENSION_NAME, "uw-loci", "qupath-extension-DL-pixel-classifier");

    /**
     * Observable property tracking whether the DL environment is ready for use.
     * When true, workflow menu items are visible. When false, only the setup item is shown.
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

        // Fast filesystem check to determine environment state (no downloads)
        updateEnvironmentState();

        // Build menu on the FX thread
        Platform.runLater(() -> addMenuItem(qupath));

        // If environment is already built, start background initialization of Python service
        if (environmentReady.get()) {
            startBackgroundInitialization();
        }
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
            try {
                ApposeService.getInstance().initialize();
                serverAvailable = true;
                logger.info("Appose backend initialized successfully (background)");
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
                )
        );
        trainOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("training"));
        trainOption.visibleProperty().bind(environmentReady);

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
                )
        );
        inferenceOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("inference"));
        inferenceOption.visibleProperty().bind(environmentReady);

        // Separator between train/inference and overlay controls
        SeparatorMenuItem sep1 = new SeparatorMenuItem();
        sep1.visibleProperty().bind(environmentReady);

        OverlayService overlayService = OverlayService.getInstance();
        BooleanBinding noImage = Bindings.createBooleanBinding(
                () -> qupath.getImageData() == null,
                qupath.imageDataProperty()
        );

        // 3) Select Overlay Model - choose which classifier to use for the overlay
        MenuItem selectModelOption = new MenuItem("Select Overlay Model...");
        TooltipHelper.installOnMenuItem(selectModelOption,
                "Choose a trained classifier for the prediction overlay.\n" +
                        "The selected model is used when toggling the overlay on/off.");
        selectModelOption.setOnAction(e -> selectOverlayModel(qupath, overlayService));
        selectModelOption.disableProperty().bind(noImage);
        selectModelOption.visibleProperty().bind(environmentReady);

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
                noImage.or(overlayService.trainingActiveProperty()));
        livePredictionOption.visibleProperty().bind(environmentReady);

        // Separator before models
        SeparatorMenuItem sep2 = new SeparatorMenuItem();
        sep2.visibleProperty().bind(environmentReady);

        // 5) Manage Models - browse and manage saved classifiers
        MenuItem modelsOption = new MenuItem(res.getString("menu.manageModels"));
        TooltipHelper.installOnMenuItem(modelsOption,
                "Browse, import, export, and delete saved classifiers.\n" +
                        "View model metadata, training configuration, and class mappings.");
        modelsOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("modelManagement"));
        modelsOption.visibleProperty().bind(environmentReady);

        // Separator before utilities
        SeparatorMenuItem sep3 = new SeparatorMenuItem();
        sep3.visibleProperty().bind(environmentReady);

        // === UTILITIES SUBMENU ===
        Menu utilitiesMenu = new Menu("Utilities");

        // Free GPU Memory - visible when environment ready
        MenuItem freeGpuOption = new MenuItem("Free GPU Memory");
        TooltipHelper.installOnMenuItem(freeGpuOption,
                "Force-clear all GPU memory held by the classification server.\n" +
                        "Cancels running training jobs, clears cached models, and\n" +
                        "frees GPU VRAM. Use after a crash or failed training.");
        freeGpuOption.setOnAction(e -> {
            freeGpuOption.setDisable(true);
            Thread clearThread = new Thread(() -> {
                try {
                    ClassifierBackend backend = BackendFactory.getBackend();
                    String result = backend.clearGPUMemory();
                    Platform.runLater(() -> {
                        freeGpuOption.setDisable(false);
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
                        freeGpuOption.setDisable(false);
                        Dialogs.showErrorNotification(EXTENSION_NAME,
                                "Error clearing GPU memory: " + ex.getMessage());
                    });
                }
            }, "DLClassifier-FreeGPU");
            clearThread.setDaemon(true);
            clearThread.start();
        });
        freeGpuOption.visibleProperty().bind(environmentReady);

        // MAE Pretrain Encoder - visible when environment ready
        MenuItem maePretrainOption = new MenuItem("MAE Pretrain Encoder...");
        TooltipHelper.installOnMenuItem(maePretrainOption,
                "Self-supervised pretraining for MuViT encoder.\n" +
                "Train on unlabeled image tiles using masked autoencoder.\n" +
                "The resulting encoder can be loaded via 'Continue from model'.");
        maePretrainOption.setOnAction(e -> startMAEPretraining());
        maePretrainOption.visibleProperty().bind(environmentReady);

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
        systemInfoOption.visibleProperty().bind(environmentReady);

        // Python Console - visible when environment ready
        MenuItem pythonConsoleOption = new MenuItem(res.getString("menu.pythonConsole"));
        TooltipHelper.installOnMenuItem(pythonConsoleOption,
                "Show a live console window displaying Python process output.\n" +
                "Useful for monitoring model loading, inference, and debugging.");
        pythonConsoleOption.setOnAction(e -> PythonConsoleWindow.getInstance().show());
        pythonConsoleOption.visibleProperty().bind(environmentReady);

        // Overlay Settings - configure prediction smoothing
        MenuItem overlaySettingsOption = new MenuItem("Overlay Settings...");
        TooltipHelper.installOnMenuItem(overlaySettingsOption,
                "Configure prediction smoothing for the overlay.\n" +
                        "Changes apply immediately if an overlay is active.");
        overlaySettingsOption.setOnAction(e ->
                new qupath.ext.dlclassifier.ui.OverlaySettingsDialog(overlayService).show());
        overlaySettingsOption.visibleProperty().bind(environmentReady);

        utilitiesMenu.getItems().addAll(overlaySettingsOption, freeGpuOption,
                maePretrainOption, new SeparatorMenuItem(), systemInfoOption,
                pythonConsoleOption, new SeparatorMenuItem(), rebuildItem);

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
     * Collects Java-side and Python-side system information and shows it
     * in a copyable text dialog.
     */
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

        // Get Appose backend (MAE pretraining is Appose-only)
        ClassifierBackend backend = BackendFactory.getBackend();
        if (!(backend instanceof ApposeClassifierBackend apposeBackend)) {
            Dialogs.showErrorNotification(EXTENSION_NAME,
                    "MAE pretraining requires the Appose backend.");
            return;
        }

        // Create progress monitor for pretraining (loss chart only, no class metrics)
        ProgressMonitorController progress = ProgressMonitorController.forPretraining();
        progress.show();

        // Launch pretraining on daemon thread
        final int[] lastLoggedEpoch = {-1};

        Thread pretrainThread = new Thread(() -> {
            try {
                logger.info("Starting MAE pretraining: model={}, epochs={}, data={}",
                        config.config().get("model_config"),
                        config.config().get("epochs"),
                        config.dataPath());
                progress.log("MAE pretraining starting...");
                progress.log("Model: " + config.config().get("model_config")
                        + ", patch=" + config.config().get("patch_size")
                        + ", scales=" + config.config().get("level_scales"));
                progress.log("Training: " + config.config().get("epochs")
                        + " epochs, batch=" + config.config().get("batch_size")
                        + ", lr=" + config.config().get("learning_rate")
                        + ", mask=" + config.config().get("mask_ratio"));
                progress.log("Data: " + config.dataPath());
                progress.log("Output: " + config.outputDir());

                ClassifierClient.TrainingResult result = apposeBackend.startPretraining(
                        config.config(),
                        config.dataPath(),
                        config.outputDir(),
                        maeProgress -> {
                            // Setup phase updates (before training loop)
                            if (maeProgress.isSetupPhase()) {
                                if ("initializing".equals(maeProgress.status())) {
                                    String deviceMsg = formatDeviceMessage(
                                            maeProgress.device(), maeProgress.deviceInfo());
                                    progress.log(deviceMsg);
                                    progress.setStatus("Initializing for "
                                            + maeProgress.totalEpochs() + " epoch run...");
                                } else {
                                    String phaseMsg = formatSetupPhase(
                                            maeProgress.setupPhase());
                                    progress.setStatus(phaseMsg);
                                    progress.log(phaseMsg);
                                }
                                return;
                            }

                            // First real epoch - log start
                            if (lastLoggedEpoch[0] < 0) {
                                progress.log("Training loop started");
                                progress.setStatus("Pretraining ("
                                        + maeProgress.totalEpochs() + " epochs)...");
                            }

                            // Always update progress bar and status
                            if (maeProgress.totalEpochs() > 0) {
                                progress.setOverallProgress(
                                        (double) maeProgress.epoch() / maeProgress.totalEpochs());
                            }
                            progress.setDetail(String.format(
                                    "Epoch %d/%d  |  Loss: %.4f",
                                    maeProgress.epoch(), maeProgress.totalEpochs(),
                                    maeProgress.loss()));

                            // Update loss chart
                            progress.updateTrainingMetrics(
                                    maeProgress.epoch(),
                                    maeProgress.loss(),
                                    Double.NaN,
                                    null, null);

                            // Log every 10 epochs or first epoch
                            int epoch = maeProgress.epoch();
                            if (epoch == 1 || epoch % 10 == 0
                                    || epoch == maeProgress.totalEpochs()) {
                                progress.log(String.format(
                                        "Epoch %d/%d: loss=%.6f",
                                        epoch, maeProgress.totalEpochs(),
                                        maeProgress.loss()));
                                lastLoggedEpoch[0] = epoch;
                            }
                        },
                        progress::isCancelled
                );

                // Success
                String encoderPath = result.modelPath();
                String message = String.format(
                        "Encoder saved to:\n%s\n\n" +
                        "Final reconstruction loss: %.4f\n\n" +
                        "To use: In the training dialog, select MuViT and\n" +
                        "choose 'Continue from model' to load this encoder.",
                        encoderPath, result.finalLoss());
                progress.complete(true, message);
                logger.info("MAE pretraining complete: loss={}, path={}",
                        result.finalLoss(), encoderPath);
                Platform.runLater(() ->
                        Dialogs.showInfoNotification(EXTENSION_NAME,
                                "MAE pretraining complete! Encoder saved to:\n" + encoderPath));

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
            }
        }, "DLClassifier-MAEPretrain");
        pretrainThread.setDaemon(true);
        pretrainThread.start();
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

        // Show a choice dialog, pre-select current model if one is selected
        List<String> names = classifiers.stream()
                .map(c -> c.getName() + " (" + c.getId() + ")")
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
            default -> "Setting up (" + phase + ")...";
        };
    }

}
