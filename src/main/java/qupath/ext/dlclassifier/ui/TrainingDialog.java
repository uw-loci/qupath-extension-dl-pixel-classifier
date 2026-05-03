package qupath.ext.dlclassifier.ui;

import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.chart.PieChart;
import javafx.scene.layout.*;
import javafx.scene.Scene;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import javafx.util.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.scripting.ScriptGenerator;
import qupath.ext.dlclassifier.service.ApposeService;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.SetupDLClassifier;
import qupath.ext.dlclassifier.utilities.CheckpointScanner;
import qupath.ext.dlclassifier.utilities.CheckpointScanner.OrphanedCheckpoint;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.commands.MiniViewers;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.scripting.QP;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.awt.image.BufferedImage;
import java.io.FileReader;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Dialog for configuring deep learning classifier training.
 * <p>
 * This dialog provides a comprehensive interface for:
 * <ul>
 *   <li>Classifier naming and description</li>
 *   <li>Model architecture selection</li>
 *   <li>Training hyperparameter configuration</li>
 *   <li>Channel selection for multi-channel images</li>
 *   <li>Annotation class selection for training data</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TrainingDialog {

    private static final Logger logger = LoggerFactory.getLogger(TrainingDialog.class);

    /**
     * Result of the training dialog.
     *
     * @param classifierName   name for the classifier
     * @param description      classifier description
     * @param trainingConfig   training configuration
     * @param channelConfig    channel configuration
     * @param selectedClasses  selected class names
     * @param selectedImages   project images to train from, or null for current image only
     * @param classColors      map of class name to packed RGB color (from QuPath PathClass)
     * @param trainOnlyImages  image names assigned exclusively to the training set (advanced mode)
     * @param valOnlyImages    image names assigned exclusively to the validation set (advanced mode)
     */
    public record TrainingDialogResult(
            String classifierName,
            String description,
            TrainingConfig trainingConfig,
            ChannelConfiguration channelConfig,
            List<String> selectedClasses,
            List<ProjectImageEntry<BufferedImage>> selectedImages,
            Map<String, Integer> classColors,
            Map<String, Object> handlerParameters,
            Set<String> trainOnlyImages,
            Set<String> valOnlyImages
    ) {
        /** Returns true if training should use multiple project images. */
        public boolean isMultiImage() {
            return selectedImages != null && !selectedImages.isEmpty();
        }

        /** Returns true if MAE pretraining should run before fine-tuning. */
        public boolean isMaePretrainEnabled() {
            return handlerParameters != null
                    && Boolean.TRUE.equals(handlerParameters.get("mae_pretrain_enabled"));
        }
    }

    private TrainingDialog() {
        // Utility class
    }

    /**
     * Singleton guard: at most one training dialog may be open at a time.
     * Prevents confusion from a second configuration window being opened
     * mid-training or while another is being configured. Access on FX thread.
     */
    private static Stage activeDialog;

    /**
     * Shows the training configuration dialog.
     * <p>
     * If a training dialog is already open, it is brought to the front and
     * the returned future is cancelled (no new dialog is created).
     *
     * @return CompletableFuture with the result, or cancelled if user cancels
     *         or a dialog is already open
     */
    public static CompletableFuture<TrainingDialogResult> showDialog() {
        CompletableFuture<TrainingDialogResult> future = new CompletableFuture<>();

        Platform.runLater(() -> {
            try {
                if (activeDialog != null && activeDialog.isShowing()) {
                    logger.info("Training dialog already open -- focusing existing window");
                    activeDialog.toFront();
                    activeDialog.requestFocus();
                    Dialogs.showInfoNotification("Training Dialog",
                            "A training configuration window is already open.");
                    future.cancel(true);
                    return;
                }
                TrainingDialogBuilder builder = new TrainingDialogBuilder();
                builder.buildAndShow(result -> {
                    if (result != null) {
                        future.complete(result);
                    } else {
                        future.cancel(true);
                    }
                });
            } catch (Exception e) {
                logger.error("Error showing training dialog", e);
                future.completeExceptionally(e);
            }
        });

        return future;
    }

    /**
     * Inner builder class for constructing the dialog.
     */
    private static class TrainingDialogBuilder {

        private Stage dialog;
        private Consumer<TrainingDialogResult> onResult;
        private boolean resultDelivered;
        private VBox checkpointRecoveryBanner;
        private final Map<String, String> validationErrors = new LinkedHashMap<>();

        /** Controls basic/advanced mode visibility. Persisted across sessions. */
        private final javafx.beans.property.BooleanProperty advancedMode =
                new javafx.beans.property.SimpleBooleanProperty(
                        DLClassifierPreferences.isAdvancedMode());

        // Basic info fields
        private TextField classifierNameField;
        private TextArea descriptionField;

        // Model architecture
        private ComboBox<String> architectureCombo;
        private ComboBox<String> backboneCombo;

        // Handler-specific UI (populated dynamically from ClassifierHandler.createTrainingUI())
        private javafx.scene.layout.VBox handlerUIContainer;
        private ClassifierHandler.TrainingUI currentHandlerUI;

        // Training parameters
        private Spinner<Integer> epochsSpinner;
        private Spinner<Integer> batchSizeSpinner;
        private Spinner<Double> learningRateSpinner;
        private Label lrInfoLabel;
        private Spinner<Integer> validationSplitSpinner;
        // Shows derived train/val counts when users have imposed per-image roles.
        private Label validationSplitObservedLabel;
        // Stashed user-entered spinner value, restored when we leave fully-manual mode.
        private Integer lastUserValidationSplitPct;
        // Guard so our own programmatic spinner updates don't overwrite lastUserValidationSplitPct.
        private boolean updatingSpinnerProgrammatically;

        // Tiling parameters
        private Spinner<Integer> tileSizeSpinner;
        private Spinner<Integer> overlapSpinner;
        private CheckBox wholeImageCheck;
        private Label wholeImageInfoLabel;
        private ComboBox<String> downsampleCombo;
        private ComboBox<String> contextScaleCombo;
        private Spinner<Integer> lineStrokeWidthSpinner;
        private Label lineStrokeLabel;

        // Channel selection
        private ChannelSelectionPanel channelPanel;

        // Class selection
        private ListView<ClassItem> classListView;
        private PieChart classDistributionChart;
        private CheckBox rebalanceByDefaultCheck;

        // Augmentation
        private CheckBox flipHorizontalCheck;
        private CheckBox flipVerticalCheck;
        private CheckBox rotationCheck;
        private ComboBox<String> intensityAugCombo;
        private boolean intensityAugUserModified = false;
        private CheckBox elasticCheck;

        // Training strategy
        private ComboBox<String> schedulerCombo;
        private ComboBox<String> lossFunctionCombo;
        private Spinner<Double> focalGammaSpinner;
        private Label focalGammaLabel;
        private Spinner<Double> boundarySigmaSpinner;
        private Label boundarySigmaLabel;
        private Spinner<Double> boundaryWMinSpinner;
        private Label boundaryWMinLabel;
        private Spinner<Integer> ohemSpinner;
        private Spinner<Integer> ohemStartSpinner;
        private CheckBox ohemAdaptiveFloorCheck;
        private ComboBox<String> ohemScheduleCombo;
        private Label ohemScheduleLabel;
        private ComboBox<String> earlyStoppingMetricCombo;
        private Spinner<Integer> earlyStoppingPatienceSpinner;
        private Button autoDistributeBtn;
        private boolean inAutoDistribute = false;
        private CheckBox mixedPrecisionCheck;
        private Spinner<Integer> gradientAccumulationSpinner;
        private CheckBox progressiveResizeCheck;
        private CheckBox fusedOptimizerCheck;
        private CheckBox useLrFinderCheck;
        private CheckBox gpuAugmentationCheck;
        private CheckBox useTorchCompileCheck;

        // Focus class
        private ComboBox<String> focusClassCombo;
        private Spinner<Double> focusClassMinIoUSpinner;
        private Label focusClassMinIoULabel;

        // Weight initialization (unified radio group)
        private ToggleGroup weightInitGroup;
        private RadioButton scratchRadio;
        private RadioButton backbonePretrainedRadio;
        private RadioButton maeEncoderRadio;
        private RadioButton sslEncoderRadio;
        private RadioButton continueTrainingRadio;
        private VBox backbonePretrainedContent;
        private VBox maeEncoderContent;
        private VBox sslEncoderContent;
        private VBox continueTrainingContent;
        private TextField maeEncoderPathField;
        private Label maeEncoderInfoLabel;
        private int maeEncoderInputChannels = -1;  // -1 = no MAE loaded
        private int maeEncoderTileSize = -1;        // -1 = unknown
        private TextField sslEncoderPathField;
        private Label sslEncoderInfoLabel;
        private LayerFreezePanel layerFreezePanel;
        private VBox layerFreezeContainer;
        private Label layerFreezeInfoLabel;
        private ClassifierBackend backend;

        // Image source selection
        private ListView<ImageSelectionItem> imageSelectionList;
        private List<TitledPane> gatedSections = new ArrayList<>();
        private boolean classesLoaded = false;
        private boolean hasLineAnnotations = false;
        private Button loadClassesButton;

        // Architecture-dependent sections
        private Label backboneLabel;
        private Label contextScaleLabel;

        // Load settings from model
        private Label loadedModelLabel;
        private List<String> sourceModelClassNames;
        private String pretrainedModelPtPath;
        private String pretrainedModelArchitecture;
        private String pretrainedModelBackbone;

        // Spatial info labels
        private Label resolutionInfoLabel;
        private Label contextInfoLabel;
        private double nativePixelSizeMicrons = Double.NaN;

        // Mini viewer preview (nullable, set when preview window is open)
        private MiniViewers.MiniViewerManager previewManager;
        private Stage previewStage;

        // Context scale preview (shows what the context view sees)
        private MiniViewers.MiniViewerManager contextPreviewManager;
        private Stage contextPreviewStage;

        // Preview window linking: when both are open, they move together
        private boolean syncingPreviews = false;
        private javafx.beans.value.ChangeListener<Number> resPosXListener;
        private javafx.beans.value.ChangeListener<Number> resPosYListener;
        private javafx.beans.value.ChangeListener<Number> ctxPosXListener;
        private javafx.beans.value.ChangeListener<Number> ctxPosYListener;

        // Live VRAM estimation
        private Label vramEstimateLabel;

        // Live tile-settings advisory (small tile, low/high overlap, zero stride).
        // Purely informational -- the values are not rewritten silently.
        private Label tileAdvisoryLabel;
        private Label earlyStoppingStatusLabel;
        private Spinner<Double> discriminativeLrSpinner;
        private Label effectiveLrLabel;
        private Spinner<Double> weightDecaySpinner;
        private Spinner<Integer> seedSpinner;
        private Label advancedSettingsWarning;
        private Label backboneCompatWarning;
        private boolean lastImageIsBrightfield = true;
        private int gpuTotalMb = 0;  // cached GPU memory (0 = unknown/CPU)

        // Pre-training tile/time estimate
        private Label tileEstimateLabel;
        private double cachedTotalAnnotationArea = 0;
        private int lastLoadedClassCount = 0;
        private int lastLoadedImageCount = 0;

        // Error display
        private VBox errorSummaryPanel;
        private VBox errorListBox;
        private Button okButton;

        public void buildAndShow(Consumer<TrainingDialogResult> resultCallback) {
            this.onResult = resultCallback;

            dialog = new Stage();
            dialog.initOwner(QuPathGUI.getInstance().getStage());
            dialog.initModality(Modality.NONE);
            dialog.setTitle("Train DL Pixel Classifier");
            dialog.setResizable(true);
            dialog.setAlwaysOnTop(true);

            // Create buttons
            Button copyScriptButton = new Button("Copy as QuPath Script");
            TooltipHelper.install(copyScriptButton,
                    "Copy current settings as a Groovy script for QuPath's script editor.\n" +
                    "The script captures all training parameters so you can\n" +
                    "reproduce or share the configuration.");
            copyScriptButton.setOnAction(e -> copyTrainingScript(copyScriptButton));

            okButton = new Button("Start Training");
            okButton.setDisable(true);
            // Intentionally NOT setDefaultButton(true): the dialog has many
            // text fields and Enter used to kick off training while the user
            // was still typing a value. Training is a long-running, not-
            // trivially-reversible action, so require an explicit click.
            okButton.setOnAction(e -> {
                var result = buildResult();
                if (result == null) {
                    // buildResult() returned null (e.g. user chose "go back"
                    // from VRAM warning or channel mismatch) -- keep dialog open
                    return;
                }
                resultDelivered = true;
                onResult.accept(result);
                dialog.close();
            });

            Button cancelButton = new Button("Cancel");
            cancelButton.setCancelButton(true);
            cancelButton.setOnAction(e -> dialog.close());

            // When window closes via Cancel or X, signal cancellation
            dialog.setOnHidden(e -> {
                if (!resultDelivered) {
                    onResult.accept(null);
                }
                // Close any open preview windows
                if (previewStage != null) {
                    previewStage.close();
                    previewManager = null;
                    previewStage = null;
                }
                if (contextPreviewStage != null) {
                    contextPreviewStage.close();
                    contextPreviewManager = null;
                    contextPreviewStage = null;
                }
                if (activeDialog == dialog) {
                    activeDialog = null;
                }
            });

            Region spacer = new Region();
            HBox.setHgrow(spacer, Priority.ALWAYS);
            HBox buttonBar = new HBox(8, copyScriptButton, spacer, okButton, cancelButton);
            buttonBar.setPadding(new Insets(10, 0, 0, 0));

            // Create content
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Checkpoint recovery banner -- populated asynchronously below.
            // Lives at the very top of the dialog so users cannot miss it
            // after an interrupted training run.
            VBox checkpointRecoveryBanner = new VBox(6);
            checkpointRecoveryBanner.setVisible(false);
            checkpointRecoveryBanner.setManaged(false);
            this.checkpointRecoveryBanner = checkpointRecoveryBanner;

            // Initialize backend for server communication
            try {
                backend = BackendFactory.getBackend();
            } catch (Exception e) {
                logger.warn("Could not initialize classifier backend: {}", e.getMessage());
            }

            // Create channel and class sections first so their fields exist
            // before the model section's backbone listener fires
            TitledPane channelSection = createChannelSection();
            TitledPane classSection = createClassSection();

            // Image source section is always enabled
            TitledPane imageSourceSection = createImageSourceSection();

            // All other sections are gated behind "Load Classes"
            TitledPane basicInfoSection = createBasicInfoSection();
            TitledPane modelSection = createModelSection();
            TitledPane weightInitSection = createWeightInitializationSection();
            TitledPane trainingSection = createTrainingSection();
            TitledPane strategySection = createTrainingStrategySection();
            TitledPane augmentationSection = createAugmentationSection();

            gatedSections.addAll(List.of(
                    basicInfoSection, modelSection, weightInitSection,
                    trainingSection, strategySection,
                    channelSection, classSection, augmentationSection
            ));

            // Build layout: header, image source, gated sections, error panel, button bar
            // Classifier info (naming) comes after model/training params so the user
            // can choose a name informed by those settings.
            content.getChildren().addAll(
                    checkpointRecoveryBanner,
                    createHeaderBox(),
                    imageSourceSection,
                    modelSection,
                    weightInitSection,
                    trainingSection,
                    strategySection,
                    channelSection,
                    classSection,
                    augmentationSection,
                    basicInfoSection,
                    createErrorSummaryPanel(),
                    buttonBar
            );

            // Advanced-only sections: fully hidden in basic mode
            // (Model, weight init, training params, and channel sections are now
            //  visible in basic mode with per-control visibility bindings instead)
            for (TitledPane advSection : List.of(strategySection, augmentationSection)) {
                advSection.visibleProperty().bind(advancedMode);
                advSection.managedProperty().bind(advancedMode);
            }

            // Disable gated sections until classes are loaded
            setGatedSectionsEnabled(false);

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setMaxHeight(Double.MAX_VALUE);
            scrollPane.setPrefHeight(advancedMode.get() ? 700 : 650);
            scrollPane.setPrefWidth(550);

            // Resize dialog when toggling modes
            advancedMode.addListener((obs, wasAdvanced, isAdvanced) -> {
                if (isAdvanced) {
                    double maxH = javafx.stage.Screen.getPrimary().getVisualBounds().getHeight() - 100;
                    dialog.setHeight(Math.min(900, maxH));
                } else {
                    dialog.setHeight(700);
                }
            });

            dialog.setScene(new Scene(scrollPane));

            // Generate default classifier name
            String timestamp = java.time.LocalDate.now().toString().replace("-", "");
            classifierNameField.setText("Classifier_" + timestamp);

            // Trigger initial layer load now that all UI components exist
            updateLayerFreezePanel();

            // Show LR info based on initial scheduler selection
            updateLrInfoLabel();

            // Apply architecture-specific section visibility
            updateSectionsForArchitecture(architectureCombo.getValue());

            // Cache GPU memory for live VRAM estimation
            cacheGpuMemory();

            // Initial validation and VRAM estimate
            updateValidation();
            updateVramEstimate();
            updateTileAdvisory();

            // Scan for orphaned best-in-progress checkpoints so we can offer
            // one-click recovery for any interrupted training.
            refreshCheckpointRecoveryBanner();

            activeDialog = dialog;
            dialog.show();
        }

        /**
         * Scans the central checkpoint registry on a background thread and
         * populates the banner with any orphaned best-in-progress files.
         * Called once when the dialog opens and again after the user clicks
         * Delete/Recover on a row, to remove handled entries.
         */
        private void refreshCheckpointRecoveryBanner() {
            Thread scanThread = new Thread(() -> {
                List<OrphanedCheckpoint> orphans =
                        CheckpointScanner.scanCentralRegistry(Collections.emptySet());
                Platform.runLater(() -> populateCheckpointBanner(orphans));
            }, "DLClassifier-DialogCheckpointScan");
            scanThread.setDaemon(true);
            scanThread.start();
        }

        private void populateCheckpointBanner(List<OrphanedCheckpoint> orphans) {
            checkpointRecoveryBanner.getChildren().clear();
            if (orphans.isEmpty()) {
                checkpointRecoveryBanner.setVisible(false);
                checkpointRecoveryBanner.setManaged(false);
                return;
            }

            Label header = new Label("Unfinished training detected");
            header.setStyle("-fx-font-weight: bold; -fx-text-fill: #8a5a00;");

            Label subtitle = new Label(
                    "These runs were interrupted before they could finish. "
                    + "Click Recover to finalize the best model so far.");
            subtitle.setStyle("-fx-text-fill: #555; -fx-font-size: 11px;");
            subtitle.setWrapText(true);

            VBox rows = new VBox(4);
            for (OrphanedCheckpoint orphan : orphans) {
                rows.getChildren().add(createCheckpointRow(orphan));
            }

            VBox bannerContent = new VBox(4, header, subtitle, rows);
            bannerContent.setPadding(new Insets(8, 10, 8, 10));
            bannerContent.setStyle(
                    "-fx-background-color: #fff7d6; "
                    + "-fx-border-color: #e0b84a; "
                    + "-fx-border-width: 1; "
                    + "-fx-background-radius: 4; "
                    + "-fx-border-radius: 4;");

            checkpointRecoveryBanner.getChildren().add(bannerContent);
            checkpointRecoveryBanner.setVisible(true);
            checkpointRecoveryBanner.setManaged(true);
        }

        private HBox createCheckpointRow(OrphanedCheckpoint orphan) {
            double sizeMb = orphan.sizeBytes() / (1024.0 * 1024.0);
            long ageSeconds = java.time.Duration.between(
                    orphan.modified(), java.time.Instant.now()).getSeconds();
            String ageLabel = formatAge(ageSeconds);

            Label info = new Label(String.format(
                    "%s  -  %.0f MB, last saved %s ago",
                    orphan.classifierName(), sizeMb, ageLabel));
            info.setStyle("-fx-font-size: 11px;");
            HBox.setHgrow(info, Priority.ALWAYS);
            info.setMaxWidth(Double.MAX_VALUE);

            Button recoverButton = new Button("Recover");
            TooltipHelper.install(recoverButton,
                    "Run finalize_training.py on this checkpoint and save the\n"
                    + "best model as a usable classifier in this project.");
            recoverButton.setOnAction(e -> {
                SetupDLClassifier.recoverFromCheckpoint(
                        QuPathGUI.getInstance(), orphan.file());
                // Recovery runs on a background thread; optimistically remove
                // the row so the user sees feedback. The file itself is deleted
                // by the recovery code once it succeeds.
                refreshCheckpointRecoveryBanner();
            });

            Button deleteButton = new Button("Delete");
            TooltipHelper.install(deleteButton,
                    "Permanently delete this checkpoint file without recovering.");
            deleteButton.setOnAction(e -> {
                Alert confirm = new Alert(Alert.AlertType.CONFIRMATION,
                        "Delete checkpoint for '" + orphan.classifierName() + "'?\n\n"
                        + "This cannot be undone.",
                        ButtonType.OK, ButtonType.CANCEL);
                confirm.setHeaderText("Delete Checkpoint");
                // The Train dialog is alwaysOnTop, so child dialogs must own up
                // to it AND flip their own alwaysOnTop flag, otherwise the
                // confirmation opens behind the dialog and becomes unclickable.
                if (dialog != null) {
                    confirm.initOwner(dialog);
                }
                Stage confirmStage = (Stage) confirm.getDialogPane().getScene().getWindow();
                confirmStage.setAlwaysOnTop(true);
                if (confirm.showAndWait().orElse(null) == ButtonType.OK) {
                    try {
                        java.nio.file.Files.deleteIfExists(orphan.file());
                    } catch (java.io.IOException ex) {
                        Dialogs.showErrorNotification("DL Pixel Classifier",
                                "Could not delete checkpoint: " + ex.getMessage());
                    }
                    refreshCheckpointRecoveryBanner();
                }
            });

            Button dismissButton = new Button("Dismiss");
            TooltipHelper.install(dismissButton,
                    "Hide this row for the current session. The checkpoint\n"
                    + "file is preserved and will reappear next time.");
            dismissButton.setOnAction(e -> {
                HBox row = (HBox) ((Button) e.getSource()).getParent();
                VBox parent = (VBox) row.getParent();
                parent.getChildren().remove(row);
                if (parent.getChildren().isEmpty()) {
                    checkpointRecoveryBanner.setVisible(false);
                    checkpointRecoveryBanner.setManaged(false);
                }
            });

            HBox row = new HBox(6, info, recoverButton, deleteButton, dismissButton);
            row.setAlignment(Pos.CENTER_LEFT);
            return row;
        }

        private static String formatAge(long seconds) {
            if (seconds < 60) return seconds + "s";
            long minutes = seconds / 60;
            if (minutes < 60) return minutes + "m";
            long hours = minutes / 60;
            if (hours < 24) return hours + "h";
            long days = hours / 24;
            return days + "d";
        }

        private VBox createHeaderBox() {
            VBox headerBox = new VBox(5);
            headerBox.setPadding(new Insets(0, 0, 5, 0));

            Label titleLabel = new Label("Configure Classifier Training");
            titleLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");

            // Toggle button for basic/advanced mode
            javafx.scene.control.ToggleButton advancedToggle = new javafx.scene.control.ToggleButton();
            advancedToggle.selectedProperty().bindBidirectional(advancedMode);
            advancedToggle.textProperty().bind(
                    javafx.beans.binding.Bindings.when(advancedMode)
                            .then("Show Basic View")
                            .otherwise("Show All Settings"));
            advancedToggle.setStyle("-fx-font-size: 11px;");
            TooltipHelper.install(advancedToggle,
                    "Toggle between a simplified view for quick training\n" +
                    "and the full settings for fine-tuning all parameters.\n\n" +
                    "Basic: Select images, classes, name, then train.\n" +
                    "Advanced: All architecture, training, and augmentation options.");

            Region headerSpacer = new Region();
            javafx.scene.layout.HBox.setHgrow(headerSpacer, Priority.ALWAYS);
            javafx.scene.layout.HBox titleRow = new javafx.scene.layout.HBox(10,
                    titleLabel, headerSpacer, advancedToggle);
            titleRow.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

            Label subtitleLabel = new Label("Train a deep learning model to classify pixels in your images");
            subtitleLabel.setStyle("-fx-text-fill: #666;");

            // Basic mode hint (hidden in advanced)
            Label basicHint = new Label(
                    "Select images, load classes, choose an encoder, pick classes, " +
                    "name your classifier, and click Start Training. " +
                    "Start with ResNet18 or ResNet34 -- larger models need more data and time.");
            basicHint.setWrapText(true);
            basicHint.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
            basicHint.visibleProperty().bind(advancedMode.not());
            basicHint.managedProperty().bind(advancedMode.not());

            // Warning label when advanced settings differ from defaults (shown in basic mode)
            advancedSettingsWarning = new Label("Note: Some advanced settings are still active. Switch to All Settings to review.");
            advancedSettingsWarning.setWrapText(true);
            advancedSettingsWarning.setStyle("-fx-text-fill: #856404; -fx-background-color: #fff3cd; " +
                    "-fx-padding: 4 8; -fx-background-radius: 3; -fx-font-size: 11px;");
            advancedSettingsWarning.setVisible(false);
            advancedSettingsWarning.setManaged(false);

            // Persist mode preference and refresh conditional visibility
            advancedMode.addListener((obs, old, newVal) -> {
                    DLClassifierPreferences.setAdvancedMode(newVal);
                    updateLineStrokeVisibility();
                    if (!newVal) {
                        // Switching to basic -- check if advanced settings differ from defaults
                        boolean hasNonDefaults = checkAdvancedSettingsDiffer();
                        advancedSettingsWarning.setVisible(hasNonDefaults);
                        advancedSettingsWarning.setManaged(hasNonDefaults);
                    } else {
                        advancedSettingsWarning.setVisible(false);
                        advancedSettingsWarning.setManaged(false);
                    }
            });

            headerBox.getChildren().addAll(titleRow, subtitleLabel, basicHint,
                    advancedSettingsWarning, new Separator());
            return headerBox;
        }

        private TitledPane createWeightInitializationSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            weightInitGroup = new ToggleGroup();

            // --- Train from scratch ---
            scratchRadio = new RadioButton(
                    ClassifierHandler.WeightInitStrategy.SCRATCH.getDisplayName());
            scratchRadio.setToggleGroup(weightInitGroup);
            scratchRadio.setUserData(ClassifierHandler.WeightInitStrategy.SCRATCH);
            TooltipHelper.install(scratchRadio,
                    "Initialize all model weights randomly.\n" +
                    "Requires more data and epochs to converge.");

            Label scratchInfo = new Label("All model weights are randomly initialized.");
            scratchInfo.setWrapText(true);
            scratchInfo.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
            scratchInfo.setPadding(new Insets(0, 0, 0, 20));

            // --- Pretrained backbone (CNN) ---
            backbonePretrainedRadio = new RadioButton(
                    ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED.getDisplayName());
            backbonePretrainedRadio.setToggleGroup(weightInitGroup);
            backbonePretrainedRadio.setUserData(ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);
            TooltipHelper.installWithLink(backbonePretrainedRadio,
                    "Initialize encoder with pretrained weights.\n" +
                    "Dramatically improves convergence and final accuracy,\n" +
                    "especially with limited training data.",
                    "https://cs231n.github.io/transfer-learning/");

            Label backboneInfo = new Label(
                    "Transfer learning uses pretrained weights from ImageNet. " +
                    "Freeze early layers to preserve general features, train later layers to adapt to your data.");
            backboneInfo.setWrapText(true);
            backboneInfo.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");

            // Update info text when backbone or mode changes (ImageNet vs histology vs foundation)
            backboneCombo.valueProperty().addListener((obs, old, newVal) -> {
                updateBackboneInfoText(newVal, backboneInfo);
                checkBackboneImageCompatibility();
            });
            advancedMode.addListener((obs, old, isAdv) -> {
                updateBackboneInfoText(backboneCombo.getValue(), backboneInfo);
            });

            layerFreezePanel = new LayerFreezePanel();
            layerFreezePanel.setBackend(backend);

            // Mode-aware info/warning label that sits above the freeze panel.
            // Text is updated by updateLayerFreezeInfoLabel() depending on whether
            // the user is starting from a backbone or continuing from a saved model.
            layerFreezeInfoLabel = new Label();
            layerFreezeInfoLabel.setWrapText(true);
            layerFreezeInfoLabel.setStyle("-fx-text-fill: #555; -fx-font-size: 11px;");

            layerFreezeContainer = new VBox(5, layerFreezeInfoLabel, layerFreezePanel);
            layerFreezeContainer.setPadding(new Insets(0, 0, 0, 20));

            // Backbone-image type compatibility warning (non-blocking)
            backboneCompatWarning = new Label();
            backboneCompatWarning.setWrapText(true);
            backboneCompatWarning.setStyle("-fx-text-fill: #856404; -fx-background-color: #fff3cd; " +
                    "-fx-padding: 4 8; -fx-background-radius: 3; -fx-font-size: 11px;");
            backboneCompatWarning.setVisible(false);
            backboneCompatWarning.setManaged(false);

            backbonePretrainedContent = new VBox(5, backboneInfo, backboneCompatWarning);
            backbonePretrainedContent.setPadding(new Insets(0, 0, 0, 20));

            // --- MAE pretrained encoder (MuViT) ---
            maeEncoderRadio = new RadioButton(
                    ClassifierHandler.WeightInitStrategy.MAE_ENCODER.getDisplayName());
            maeEncoderRadio.setToggleGroup(weightInitGroup);
            maeEncoderRadio.setUserData(ClassifierHandler.WeightInitStrategy.MAE_ENCODER);
            TooltipHelper.install(maeEncoderRadio,
                    "Load encoder weights from a self-supervised MAE pretrained model.\n" +
                    "Matching layers are loaded; non-matching layers (decoder) are skipped.");

            maeEncoderPathField = new TextField();
            maeEncoderPathField.setEditable(false);
            maeEncoderPathField.setPromptText("Select MAE encoder .pt file...");
            maeEncoderPathField.setMaxWidth(Double.MAX_VALUE);
            HBox.setHgrow(maeEncoderPathField, Priority.ALWAYS);

            Button maeBrowseButton = new Button("Browse...");
            maeBrowseButton.setOnAction(e -> {
                FileChooser chooser = new FileChooser();
                chooser.setTitle("Select MAE Pretrained Encoder (.pt)");
                chooser.getExtensionFilters().add(
                        new FileChooser.ExtensionFilter("PyTorch model", "*.pt"));
                // Default to previously selected path, then project's mae_pretrained dir
                String currentPath = maeEncoderPathField.getText();
                if (currentPath != null && !currentPath.isEmpty()) {
                    java.io.File parent = new java.io.File(currentPath).getParentFile();
                    if (parent != null && parent.isDirectory()) {
                        chooser.setInitialDirectory(parent);
                    }
                } else {
                    Project<?> project = QuPathGUI.getInstance().getProject();
                    if (project != null) {
                        try {
                            java.nio.file.Path maeDir = project.getPath().getParent()
                                    .resolve("mae_pretrained");
                            if (java.nio.file.Files.isDirectory(maeDir)) {
                                chooser.setInitialDirectory(maeDir.toFile());
                            } else {
                                chooser.setInitialDirectory(
                                        project.getPath().getParent().toFile());
                            }
                        } catch (Exception ex) {
                            logger.debug("Could not resolve project MAE directory: {}",
                                    ex.getMessage());
                        }
                    }
                }
                java.io.File file = chooser.showOpenDialog(dialog);
                if (file != null) {
                    maeEncoderPathField.setText(file.getAbsolutePath());
                    loadMaeEncoderMetadata(file);
                    updateValidation();
                }
            });
            maeBrowseButton.setTooltip(new Tooltip("Browse for a .pt encoder file"));

            Button maeClearButton = new Button("Clear");
            maeClearButton.setOnAction(e -> {
                maeEncoderPathField.setText("");
                maeEncoderInputChannels = -1;
                maeEncoderTileSize = -1;
                if (currentHandlerUI != null) {
                    currentHandlerUI.setLocked(false);
                }
                maeEncoderInfoLabel.setText("");
                maeEncoderInfoLabel.setVisible(false);
                maeEncoderInfoLabel.setManaged(false);
                updateValidation();
            });

            HBox maeFileRow = new HBox(5, maeEncoderPathField, maeBrowseButton, maeClearButton);
            maeFileRow.setAlignment(Pos.CENTER_LEFT);
            HBox.setHgrow(maeEncoderPathField, Priority.ALWAYS);

            maeEncoderInfoLabel = new Label();
            maeEncoderInfoLabel.setWrapText(true);
            maeEncoderInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #336699;");
            maeEncoderInfoLabel.setVisible(false);
            maeEncoderInfoLabel.setManaged(false);

            maeEncoderContent = new VBox(5, maeFileRow, maeEncoderInfoLabel);
            maeEncoderContent.setPadding(new Insets(0, 0, 0, 20));

            // --- SSL pretrained encoder (SimCLR/BYOL) ---
            sslEncoderRadio = new RadioButton(
                    ClassifierHandler.WeightInitStrategy.SSL_ENCODER.getDisplayName());
            sslEncoderRadio.setToggleGroup(weightInitGroup);
            sslEncoderRadio.setUserData(ClassifierHandler.WeightInitStrategy.SSL_ENCODER);
            TooltipHelper.install(sslEncoderRadio,
                    "Load encoder weights from a self-supervised SimCLR/BYOL pretrained model.\n" +
                    "Matching encoder layers are loaded; decoder/head are randomly initialized.\n" +
                    "Use the 'SSL Pretrain Encoder' utility to create these weights.");

            sslEncoderPathField = new TextField();
            sslEncoderPathField.setEditable(false);
            sslEncoderPathField.setPromptText("Select SSL encoder .pt file...");
            sslEncoderPathField.setMaxWidth(Double.MAX_VALUE);
            HBox.setHgrow(sslEncoderPathField, Priority.ALWAYS);

            Button sslBrowseButton = new Button("Browse...");
            sslBrowseButton.setOnAction(e -> {
                FileChooser chooser = new FileChooser();
                chooser.setTitle("Select SSL Pretrained Encoder (.pt)");
                chooser.getExtensionFilters().add(
                        new FileChooser.ExtensionFilter("PyTorch model", "*.pt"));
                String currentPath = sslEncoderPathField.getText();
                if (currentPath != null && !currentPath.isEmpty()) {
                    java.io.File parent = new java.io.File(currentPath).getParentFile();
                    if (parent != null && parent.isDirectory()) {
                        chooser.setInitialDirectory(parent);
                    }
                } else {
                    Project<?> project = QuPathGUI.getInstance().getProject();
                    if (project != null) {
                        try {
                            java.nio.file.Path sslDir = project.getPath().getParent()
                                    .resolve("ssl_pretrained");
                            if (java.nio.file.Files.isDirectory(sslDir)) {
                                chooser.setInitialDirectory(sslDir.toFile());
                            } else {
                                chooser.setInitialDirectory(
                                        project.getPath().getParent().toFile());
                            }
                        } catch (Exception ex) {
                            logger.debug("Could not resolve project SSL directory: {}",
                                    ex.getMessage());
                        }
                    }
                }
                java.io.File file = chooser.showOpenDialog(dialog);
                if (file != null) {
                    sslEncoderPathField.setText(file.getAbsolutePath());
                    loadSSLEncoderMetadata(file);
                    updateValidation();
                }
            });

            Button sslClearButton = new Button("Clear");
            sslClearButton.setOnAction(e -> {
                sslEncoderPathField.setText("");
                sslEncoderInfoLabel.setText("");
                sslEncoderInfoLabel.setVisible(false);
                sslEncoderInfoLabel.setManaged(false);
                updateValidation();
            });

            HBox sslFileRow = new HBox(5, sslEncoderPathField, sslBrowseButton, sslClearButton);
            sslFileRow.setAlignment(Pos.CENTER_LEFT);
            HBox.setHgrow(sslEncoderPathField, Priority.ALWAYS);

            sslEncoderInfoLabel = new Label();
            sslEncoderInfoLabel.setWrapText(true);
            sslEncoderInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #336699;");
            sslEncoderInfoLabel.setVisible(false);
            sslEncoderInfoLabel.setManaged(false);

            sslEncoderContent = new VBox(5, sslFileRow, sslEncoderInfoLabel);
            sslEncoderContent.setPadding(new Insets(0, 0, 0, 20));

            // --- Continue training from saved model ---
            continueTrainingRadio = new RadioButton(
                    ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING.getDisplayName());
            continueTrainingRadio.setToggleGroup(weightInitGroup);
            continueTrainingRadio.setUserData(ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING);
            TooltipHelper.install(continueTrainingRadio,
                    "Load all weights from a previously trained model as the starting point.\n" +
                    "The optimizer and learning rate schedule start fresh.\n" +
                    "Architecture, backbone, tile size, downsample, and context scale\n" +
                    "are locked to match the saved model.\n" +
                    "Useful for fine-tuning on additional data or adjusted classes.");
            // Radio is always selectable so the user can see the "Select model..." button.
            // The Train button validates that a model has actually been loaded.

            Button selectModelButton = new Button("Select model...");
            selectModelButton.setOnAction(e -> {
                ModelManager modelManager = new ModelManager();
                List<ClassifierMetadata> classifiers = modelManager.listClassifiers();
                if (classifiers.isEmpty()) {
                    Dialogs.showWarningNotification("Load Settings",
                            "No trained classifiers found in the project or user directory.");
                    return;
                }
                String currentArch = (architectureCombo != null)
                        ? architectureCombo.getValue() : null;
                Optional<ClassifierMetadata> selected =
                        ModelPickerDialog.show(dialog, classifiers, currentArch);
                selected.ifPresent(this::loadSettingsFromModel);
            });
            TooltipHelper.install(selectModelButton,
                    "Load settings from a previously trained model to retrain or refine it.\n" +
                    "Populates all dialog fields from the selected model's configuration.");

            Button loadCheckpointButton = new Button("Load checkpoint...");
            loadCheckpointButton.setOnAction(e -> loadCheckpointFile());
            TooltipHelper.install(loadCheckpointButton,
                    "Recover from an interrupted training checkpoint (.pt file).\n" +
                    "Finalizes the checkpoint into a usable model, then loads its\n" +
                    "settings so you can continue training from where it left off.\n\n" +
                    "Look for checkpoint files in:\n" +
                    "  - Your project's classifiers/dl/ directory (opened by default)\n" +
                    "  - ~/.dlclassifier/checkpoints/ (pause/resume + best_in_progress files)\n" +
                    "  - Files named checkpoint_*.pt or best_in_progress_*.pt");

            loadedModelLabel = new Label();
            loadedModelLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");

            continueTrainingContent = new VBox(5,
                    new HBox(10, selectModelButton, loadCheckpointButton),
                    loadedModelLabel);
            continueTrainingContent.setPadding(new Insets(0, 0, 0, 20));

            // Toggle group listener: show/hide sub-content + re-validate
            weightInitGroup.selectedToggleProperty().addListener((obs, old, newVal) -> {
                    updateWeightInitSubContent();
                    updateValidation();
            });

            // Basic mode: hide scratch and MAE options, hide layer freeze panel
            scratchRadio.visibleProperty().bind(advancedMode);
            scratchRadio.managedProperty().bind(advancedMode);
            scratchInfo.visibleProperty().bind(advancedMode);
            scratchInfo.managedProperty().bind(advancedMode);
            maeEncoderRadio.visibleProperty().bind(advancedMode);
            maeEncoderRadio.managedProperty().bind(advancedMode);
            maeEncoderContent.visibleProperty().bind(advancedMode);
            maeEncoderContent.managedProperty().bind(advancedMode);
            sslEncoderRadio.visibleProperty().bind(advancedMode);
            sslEncoderRadio.managedProperty().bind(advancedMode);
            sslEncoderContent.visibleProperty().bind(advancedMode);
            sslEncoderContent.managedProperty().bind(advancedMode);
            // The layer freeze panel is advanced-mode-only AND only relevant for
            // strategies that load pretrained weights (backbone pretrained or
            // continue training). updateWeightInitSubContent() ANDs both conditions.
            // Re-run the visibility check whenever advanced mode toggles.
            advancedMode.addListener((obs, wasAdvanced, isAdvanced) -> updateWeightInitSubContent());

            // Assemble all options. The freeze container sits at the bottom and is
            // shared by BACKBONE_PRETRAINED and CONTINUE_TRAINING -- it stays hidden
            // for SCRATCH/MAE/SSL via updateWeightInitSubContent().
            content.getChildren().addAll(
                    scratchRadio, scratchInfo,
                    backbonePretrainedRadio, backbonePretrainedContent,
                    maeEncoderRadio, maeEncoderContent,
                    sslEncoderRadio, sslEncoderContent,
                    continueTrainingRadio, continueTrainingContent,
                    layerFreezeContainer
            );

            // Set initial selection based on handler + preferences
            ClassifierHandler handler = ClassifierRegistry.getHandler(architectureCombo.getValue())
                    .orElse(ClassifierRegistry.getDefaultHandler());
            updateWeightInitOptions(architectureCombo.getValue());

            TitledPane pane = new TitledPane("WEIGHT INITIALIZATION", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            TooltipHelper.install(pane,
                    "Choose how to initialize model weights.\n" +
                    "Pretrained weights (default) transfer learned features and train faster.\n" +
                    "Continue training picks up from a previously saved model.");
            return pane;
        }

        /**
         * Shows/hides sub-content panels based on the selected weight initialization radio.
         */
        /**
         * Sets visible/managed on a node only if the property is not already bound.
         */
        private void setVisibleIfUnbound(javafx.scene.Node node, boolean visible) {
            if (!node.visibleProperty().isBound()) node.setVisible(visible);
            if (!node.managedProperty().isBound()) node.setManaged(visible);
        }

        private void updateWeightInitSubContent() {
            ClassifierHandler.WeightInitStrategy selected = getSelectedWeightInitStrategy();

            setVisibleIfUnbound(backbonePretrainedContent,
                    selected == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);

            setVisibleIfUnbound(maeEncoderContent,
                    selected == ClassifierHandler.WeightInitStrategy.MAE_ENCODER);

            setVisibleIfUnbound(sslEncoderContent,
                    selected == ClassifierHandler.WeightInitStrategy.SSL_ENCODER);

            setVisibleIfUnbound(continueTrainingContent,
                    selected == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING);

            // The freeze panel applies to strategies that load pretrained weights
            // (backbone pretrained AND continue-training). Hide it for SCRATCH/MAE/SSL,
            // and also hide it entirely in basic mode.
            boolean freezeApplicable =
                    selected == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED
                    || selected == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING;
            boolean showFreeze = freezeApplicable && advancedMode.get();
            setVisibleIfUnbound(layerFreezeContainer, showFreeze);
            updateLayerFreezeInfoLabel(selected);

            // Refresh the layer list whenever a transfer-learning path is selected.
            if (freezeApplicable) {
                updateLayerFreezePanel();
            }

            // Lock/unlock handler UI (e.g., MuViT model size, patch size, level scales).
            // Must stay locked for MAE_ENCODER, SSL_ENCODER (encoder weights require
            // matching arch), and CONTINUE_TRAINING (saved model weights require matching arch).
            boolean continuing = selected == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING;
            boolean sslSelected = selected == ClassifierHandler.WeightInitStrategy.SSL_ENCODER;
            if (currentHandlerUI != null) {
                if (selected == ClassifierHandler.WeightInitStrategy.MAE_ENCODER
                        || sslSelected || continuing) {
                    currentHandlerUI.setLocked(true);
                } else {
                    currentHandlerUI.setLocked(false);
                }
            }
            if (selected != ClassifierHandler.WeightInitStrategy.MAE_ENCODER
                    && maeEncoderInfoLabel != null) {
                maeEncoderInfoLabel.setText("");
                maeEncoderInfoLabel.setVisible(false);
                maeEncoderInfoLabel.setManaged(false);
            }
            if (selected != ClassifierHandler.WeightInitStrategy.SSL_ENCODER
                    && sslEncoderInfoLabel != null) {
                sslEncoderInfoLabel.setText("");
                sslEncoderInfoLabel.setVisible(false);
                sslEncoderInfoLabel.setManaged(false);
            }

            // Lock architecture, resolution, and context scale when continuing from
            // a saved model or using an MAE/SSL encoder. The saved/pretrained weights
            // are tied to the exact architecture type.
            // Guard: these controls may not exist yet during initial construction.
            boolean maeSelected = selected == ClassifierHandler.WeightInitStrategy.MAE_ENCODER;
            if (architectureCombo != null) architectureCombo.setDisable(continuing || maeSelected || sslSelected);
            if (backboneCombo != null) backboneCombo.setDisable(continuing || maeSelected || sslSelected);
            boolean wholeImage = wholeImageCheck != null && wholeImageCheck.isSelected();
            if (tileSizeSpinner != null) tileSizeSpinner.setDisable(continuing || wholeImage);
            // Downsample stays enabled in whole-image mode so the user can adjust
            // it to fit within the architecture's max tile size.
            if (downsampleCombo != null) downsampleCombo.setDisable(continuing);
            if (contextScaleCombo != null) contextScaleCombo.setDisable(continuing || wholeImage);
        }

        /**
         * Updates the visible radio buttons based on the handler's supported strategies.
         * Falls back to the handler's default if the current selection is no longer supported.
         */
        private void updateWeightInitOptions(String architecture) {
            ClassifierHandler handler = ClassifierRegistry.getHandler(architecture)
                    .orElse(ClassifierRegistry.getDefaultHandler());
            java.util.Set<ClassifierHandler.WeightInitStrategy> supported =
                    handler.getSupportedWeightInitStrategies();

            // Show/hide each radio based on handler support
            setRadioAvailable(scratchRadio,
                    supported.contains(ClassifierHandler.WeightInitStrategy.SCRATCH));
            setRadioAvailable(backbonePretrainedRadio,
                    supported.contains(ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED));
            setRadioAvailable(maeEncoderRadio,
                    supported.contains(ClassifierHandler.WeightInitStrategy.MAE_ENCODER));
            setRadioAvailable(sslEncoderRadio,
                    supported.contains(ClassifierHandler.WeightInitStrategy.SSL_ENCODER));
            // Continue training is always shown but may be disabled
            setRadioAvailable(continueTrainingRadio,
                    supported.contains(ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING));

            // If current selection is no longer supported, fall back to handler default
            ClassifierHandler.WeightInitStrategy current = getSelectedWeightInitStrategy();
            if (current == null || !supported.contains(current)) {
                ClassifierHandler.WeightInitStrategy defaultStrategy = handler.getDefaultWeightInitStrategy();
                // Check preferences for CNN backbone pretrained
                if (supported.contains(ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED)
                        && DLClassifierPreferences.isUsePretrainedWeights()) {
                    selectWeightInitStrategy(ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);
                } else {
                    selectWeightInitStrategy(defaultStrategy);
                }
            }

            updateWeightInitSubContent();
        }

        private void setRadioAvailable(RadioButton radio, boolean available) {
            // Skip nodes whose visibility is already bound (e.g., to advancedMode)
            if (!radio.visibleProperty().isBound()) {
                radio.setVisible(available);
            }
            if (!radio.managedProperty().isBound()) {
                radio.setManaged(available);
            }
            if (!available && radio.isSelected()) {
                radio.setSelected(false);
            }
        }

        /**
         * Returns the currently selected weight initialization strategy.
         */
        private ClassifierHandler.WeightInitStrategy getSelectedWeightInitStrategy() {
            Toggle selected = weightInitGroup.getSelectedToggle();
            if (selected == null) return null;
            return (ClassifierHandler.WeightInitStrategy) selected.getUserData();
        }

        /**
         * Programmatically selects a weight initialization strategy radio button.
         */
        private void selectWeightInitStrategy(ClassifierHandler.WeightInitStrategy strategy) {
            if (strategy == null) return;
            for (Toggle toggle : weightInitGroup.getToggles()) {
                if (toggle.getUserData() == strategy) {
                    toggle.setSelected(true);
                    return;
                }
            }
        }

        /**
         * Reads metadata.json alongside an MAE encoder .pt file and locks
         * architecture controls to match the encoder configuration.
         */
        private void loadMaeEncoderMetadata(java.io.File ptFile) {
            maeEncoderInputChannels = -1;
            maeEncoderTileSize = -1;

            java.io.File metadataFile = new java.io.File(ptFile.getParentFile(), "metadata.json");
            if (!metadataFile.exists()) {
                logger.warn("No metadata.json found alongside {}. "
                        + "Architecture settings will not be locked.", ptFile.getName());
                maeEncoderInfoLabel.setText("");
                maeEncoderInfoLabel.setVisible(false);
                maeEncoderInfoLabel.setManaged(false);
                return;
            }

            try (java.io.Reader reader = java.nio.file.Files.newBufferedReader(
                    metadataFile.toPath(), java.nio.charset.StandardCharsets.UTF_8)) {
                JsonObject root = new Gson().fromJson(reader, JsonObject.class);
                JsonObject arch = root.has("architecture")
                        ? root.getAsJsonObject("architecture") : null;
                if (arch == null) {
                    logger.warn("metadata.json has no 'architecture' key; cannot lock settings.");
                    return;
                }

                Map<String, Object> archParams = new HashMap<>();
                if (arch.has("model_config"))
                    archParams.put("model_config", arch.get("model_config").getAsString());
                if (arch.has("patch_size"))
                    archParams.put("patch_size", arch.get("patch_size").getAsInt());
                if (arch.has("level_scales"))
                    archParams.put("level_scales", arch.get("level_scales").getAsString());
                if (arch.has("rope_mode"))
                    archParams.put("rope_mode", arch.get("rope_mode").getAsString());

                // Store input_channels for validation at build time
                if (arch.has("input_channels"))
                    maeEncoderInputChannels = arch.get("input_channels").getAsInt();

                // Store tile_size for the info label (not locked -- tile size
                // can differ, but showing the pretraining value helps the user
                // choose a compatible setting)
                if (arch.has("tile_size"))
                    maeEncoderTileSize = arch.get("tile_size").getAsInt();

                if (currentHandlerUI != null) {
                    currentHandlerUI.applyParameters(archParams);
                    currentHandlerUI.setLocked(true);
                }

                // Set tile size to match MAE pretraining (editable -- user can
                // change it, but the pretrained resolution is a good default)
                if (maeEncoderTileSize > 0 && tileSizeSpinner != null) {
                    tileSizeSpinner.getValueFactory().setValue(maeEncoderTileSize);
                }

                // Sync the hidden backboneCombo so TrainingConfig.backbone matches
                if (archParams.containsKey("model_config")) {
                    String config = (String) archParams.get("model_config");
                    if (backboneCombo.getItems().contains(config)) {
                        backboneCombo.setValue(config);
                    }
                }

                // Build a summary string for the info label
                StringBuilder info = new StringBuilder("Locked to encoder:");
                if (archParams.containsKey("model_config"))
                    info.append(" ").append(archParams.get("model_config"));
                if (archParams.containsKey("patch_size"))
                    info.append(", patch ").append(archParams.get("patch_size"));
                if (archParams.containsKey("level_scales"))
                    info.append(", scales ").append(archParams.get("level_scales"));
                if (maeEncoderInputChannels > 0)
                    info.append(", ").append(maeEncoderInputChannels).append("ch");
                if (maeEncoderTileSize > 0)
                    info.append(" (pretrained at ").append(maeEncoderTileSize).append("px)");

                maeEncoderInfoLabel.setText(info.toString());
                maeEncoderInfoLabel.setVisible(true);
                maeEncoderInfoLabel.setManaged(true);

                logger.info("Loaded MAE encoder metadata: {}", archParams);
            } catch (Exception e) {
                logger.warn("Failed to read metadata.json: {}", e.getMessage());
                maeEncoderInfoLabel.setText("");
                maeEncoderInfoLabel.setVisible(false);
                maeEncoderInfoLabel.setManaged(false);
            }
        }

        /**
         * Reads metadata.json alongside the SSL encoder .pt file to detect the
         * backbone and lock the architecture/backbone combos to match.
         */
        private void loadSSLEncoderMetadata(java.io.File ptFile) {
            java.io.File metadataFile = new java.io.File(ptFile.getParentFile(), "metadata.json");
            if (!metadataFile.exists()) {
                logger.warn("No metadata.json found alongside {}. "
                        + "Architecture settings will not be locked.", ptFile.getName());
                sslEncoderInfoLabel.setText("");
                sslEncoderInfoLabel.setVisible(false);
                sslEncoderInfoLabel.setManaged(false);
                return;
            }

            try (java.io.Reader reader = java.nio.file.Files.newBufferedReader(
                    metadataFile.toPath(), java.nio.charset.StandardCharsets.UTF_8)) {
                JsonObject root = new Gson().fromJson(reader, JsonObject.class);
                JsonObject arch = root.has("architecture")
                        ? root.getAsJsonObject("architecture") : null;
                if (arch == null) {
                    logger.warn("SSL metadata.json has no 'architecture' key.");
                    return;
                }

                String encoderName = arch.has("encoder_name")
                        ? arch.get("encoder_name").getAsString() : null;
                int inputChannels = arch.has("input_channels")
                        ? arch.get("input_channels").getAsInt() : -1;
                int tileSize = arch.has("tile_size")
                        ? arch.get("tile_size").getAsInt() : -1;

                String sslMethod = root.has("ssl_method")
                        ? root.get("ssl_method").getAsString() : "unknown";

                // Lock architecture to UNet and backbone to the pretrained encoder
                if (architectureCombo != null && architectureCombo.getItems().contains("unet")) {
                    architectureCombo.setValue("unet");
                    architectureCombo.setDisable(true);
                }
                if (encoderName != null && backboneCombo != null
                        && backboneCombo.getItems().contains(encoderName)) {
                    backboneCombo.setValue(encoderName);
                    backboneCombo.setDisable(true);
                }
                if (tileSize > 0 && tileSizeSpinner != null) {
                    tileSizeSpinner.getValueFactory().setValue(tileSize);
                }

                // Build info label
                StringBuilder info = new StringBuilder("Locked to SSL encoder:");
                info.append(" ").append(sslMethod.toUpperCase());
                if (encoderName != null) info.append(" ").append(encoderName);
                if (inputChannels > 0)
                    info.append(", ").append(inputChannels).append("ch");
                if (tileSize > 0)
                    info.append(" (pretrained at ").append(tileSize).append("px)");

                sslEncoderInfoLabel.setText(info.toString());
                sslEncoderInfoLabel.setVisible(true);
                sslEncoderInfoLabel.setManaged(true);

                logger.info("Loaded SSL encoder metadata: method={}, backbone={}, ch={}, tile={}",
                        sslMethod, encoderName, inputChannels, tileSize);
            } catch (Exception e) {
                logger.warn("Failed to read SSL metadata.json: {}", e.getMessage());
                sslEncoderInfoLabel.setText("");
                sslEncoderInfoLabel.setVisible(false);
                sslEncoderInfoLabel.setManaged(false);
            }
        }

        @SuppressWarnings("unchecked")
        /**
         * Opens a file chooser for checkpoint .pt files, finalizes the checkpoint
         * into a usable model, then loads its settings into the dialog.
         */
        private void loadCheckpointFile() {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Select Checkpoint File");
            fileChooser.getExtensionFilters().addAll(
                    new FileChooser.ExtensionFilter("PyTorch Checkpoint", "*.pt"),
                    new FileChooser.ExtensionFilter("All Files", "*.*"));

            // Default to the project's classifiers/dl directory when available
            // (where completed models live, plus where in-progress .pt files land
            // when modelOutputDir is set). Fall back to the per-user checkpoints
            // dir where the Python backend writes best_in_progress_*.pt files.
            java.nio.file.Path initialDir = null;
            QuPathGUI qupathForInitDir = QuPathGUI.getInstance();
            if (qupathForInitDir != null && qupathForInitDir.getProject() != null
                    && qupathForInitDir.getProject().getPath() != null) {
                java.nio.file.Path projectClassifiersDl = qupathForInitDir.getProject()
                        .getPath().getParent()
                        .resolve("classifiers").resolve("dl");
                if (java.nio.file.Files.isDirectory(projectClassifiersDl)) {
                    initialDir = projectClassifiersDl;
                }
            }
            if (initialDir == null) {
                java.nio.file.Path userCheckpointDir = java.nio.file.Path.of(
                        System.getProperty("user.home"), ".dlclassifier", "checkpoints");
                if (java.nio.file.Files.isDirectory(userCheckpointDir)) {
                    initialDir = userCheckpointDir;
                }
            }
            if (initialDir != null) {
                fileChooser.setInitialDirectory(initialDir.toFile());
            }

            java.io.File selected = fileChooser.showOpenDialog(dialog.getOwner());
            if (selected == null) return;

            String checkpointPath = selected.getAbsolutePath();
            loadedModelLabel.setText("Recovering checkpoint...");
            loadedModelLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");

            // Determine output directory for the recovered model
            String modelOutputDir = null;
            QuPathGUI qupath = QuPathGUI.getInstance();
            if (qupath != null && qupath.getProject() != null) {
                java.nio.file.Path classifiersDir = qupath.getProject().getPath().getParent()
                        .resolve("classifiers").resolve("dl");
                String baseName = selected.getName().replace(".pt", "");
                java.nio.file.Path outputDir = classifiersDir.resolve("recovered_" + baseName);
                try {
                    java.nio.file.Files.createDirectories(outputDir);
                    modelOutputDir = outputDir.toString();
                } catch (java.io.IOException ex) {
                    logger.warn("Could not create output dir, using default", ex);
                }
            }

            String finalOutputDir = modelOutputDir;

            // Run finalization in background to keep UI responsive
            Thread recoverThread = new Thread(() -> {
                try {
                    ClassifierBackend backend = BackendFactory.getBackend();
                    ClassifierClient.TrainingResult result =
                            backend.finalizeTraining(checkpointPath, finalOutputDir);

                    // Load metadata using ModelManager's parser (handles the
                    // custom JSON structure that Gson can't auto-map)
                    ModelManager modelManager = new ModelManager();
                    java.nio.file.Path modelDir = java.nio.file.Path.of(result.modelPath());
                    ClassifierMetadata recovered = modelManager.loadMetadata(modelDir);

                    // Store the .pt path directly (getModelPath can't find it
                    // because the directory name doesn't match the model ID)
                    java.nio.file.Path recoveredPt = java.nio.file.Path.of(
                            result.modelPath(), "model.pt");
                    String recoveredPtPath = java.nio.file.Files.exists(recoveredPt)
                            ? recoveredPt.toString() : null;

                    ClassifierMetadata finalRecovered = recovered;
                    String finalPtPath = recoveredPtPath;
                    Platform.runLater(() -> {
                        if (finalRecovered != null) {
                            loadSettingsFromModel(finalRecovered);
                            // Override the .pt path and arch/backbone since
                            // getModelPath can't find the recovered model by ID
                            // (directory name doesn't match model ID).
                            // Must set AFTER loadSettingsFromModel which clears these.
                            if (finalPtPath != null) {
                                pretrainedModelPtPath = finalPtPath;
                                pretrainedModelArchitecture = finalRecovered.getModelType();
                                pretrainedModelBackbone = finalRecovered.getBackbone();
                                selectWeightInitStrategy(
                                        ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING);
                                updateValidation();
                            }
                            loadedModelLabel.setText("Recovered from: " + selected.getName());
                            loadedModelLabel.setStyle(
                                    "-fx-text-fill: #666; -fx-font-style: italic;");
                            logger.info("Loaded settings from recovered checkpoint: {} "
                                    + "(ptPath={}, arch={}, backbone={})",
                                    selected.getName(), finalPtPath,
                                    finalRecovered.getModelType(),
                                    finalRecovered.getBackbone());
                        } else {
                            loadedModelLabel.setText("Recovered but could not load metadata");
                            loadedModelLabel.setStyle(
                                    "-fx-text-fill: #cc6600; -fx-font-style: italic;");
                            logger.warn("Checkpoint finalized to {} but metadata not found",
                                    result.modelPath());
                        }
                    });
                } catch (Exception ex) {
                    logger.error("Failed to recover checkpoint: {}", ex.getMessage(), ex);
                    Platform.runLater(() -> {
                        loadedModelLabel.setText("Recovery failed: " + ex.getMessage());
                        loadedModelLabel.setStyle(
                                "-fx-text-fill: #cc0000; -fx-font-style: italic;");
                    });
                }
            }, "DLClassifier-RecoverCheckpoint");
            recoverThread.setDaemon(true);
            recoverThread.start();
        }

        private void loadSettingsFromModel(ClassifierMetadata metadata) {
            logger.info("Loading settings from model: {} ({})", metadata.getName(), metadata.getId());

            // --- Always available from ClassifierMetadata ---

            // Architecture and backbone
            String modelType = metadata.getModelType();
            if (modelType != null && architectureCombo.getItems().contains(modelType)) {
                architectureCombo.setValue(modelType);
            } else if (modelType != null) {
                logger.warn("Architecture '{}' from model not available in current registry", modelType);
                Dialogs.showWarningNotification("Load Settings",
                        "Architecture '" + modelType + "' is not available.\nKeeping current selection.");
            }

            String backbone = metadata.getBackbone();
            if (backbone != null && !backbone.isEmpty()) {
                // Backbone options may have changed after architecture was set; check after a brief delay
                Platform.runLater(() -> {
                    if (backboneCombo.getItems().contains(backbone)) {
                        backboneCombo.setValue(backbone);
                    } else {
                        logger.warn("Backbone '{}' from model not available for architecture '{}'",
                                backbone, architectureCombo.getValue());
                    }
                    // Sync handler UI (e.g., MuViT modelConfigCombo) to match the
                    // loaded backbone. Without this, the hidden backboneCombo has
                    // "muvit-large" but the visible handler UI defaults to "muvit-base".
                    if (currentHandlerUI != null) {
                        currentHandlerUI.applyParameters(Map.of("model_config", backbone));
                    }
                });
            }

            // Tile size
            if (metadata.getInputWidth() > 0) {
                tileSizeSpinner.getValueFactory().setValue(metadata.getInputWidth());
            }

            // Downsample
            if (metadata.getDownsample() > 0) {
                downsampleCombo.setValue(mapDownsampleToDisplay(metadata.getDownsample()));
            }

            // Context scale
            if (metadata.getContextScale() > 0) {
                contextScaleCombo.setValue(mapContextScaleToDisplay(metadata.getContextScale()));
            }

            // Epochs
            if (metadata.getTrainingEpochs() > 0) {
                epochsSpinner.getValueFactory().setValue(metadata.getTrainingEpochs());
            }

            // Normalization strategy
            if (metadata.getNormalizationStrategy() != null && channelPanel != null) {
                channelPanel.setNormalizationStrategy(metadata.getNormalizationStrategy());
            }

            // --- From training_settings map (null for older models) ---
            Map<String, Object> ts = metadata.getTrainingSettings();
            if (ts != null) {
                // Learning rate
                if (ts.containsKey("learning_rate")) {
                    learningRateSpinner.getValueFactory().setValue(
                            ((Number) ts.get("learning_rate")).doubleValue());
                }

                // Batch size
                if (ts.containsKey("batch_size")) {
                    batchSizeSpinner.getValueFactory().setValue(
                            ((Number) ts.get("batch_size")).intValue());
                }

                // Validation split (stored as 0.0-1.0, display as %)
                if (ts.containsKey("validation_split")) {
                    double vs = ((Number) ts.get("validation_split")).doubleValue();
                    validationSplitSpinner.getValueFactory().setValue(
                            (int) Math.round(vs * 100));
                }

                // Overlap
                if (ts.containsKey("overlap")) {
                    overlapSpinner.getValueFactory().setValue(
                            ((Number) ts.get("overlap")).intValue());
                }

                // Line stroke width
                if (ts.containsKey("line_stroke_width")) {
                    lineStrokeWidthSpinner.getValueFactory().setValue(
                            ((Number) ts.get("line_stroke_width")).intValue());
                }

                // Pretrained weights -> weight init radio
                if (ts.containsKey("use_pretrained_weights")
                        && Boolean.TRUE.equals(ts.get("use_pretrained_weights"))) {
                    selectWeightInitStrategy(ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);
                }

                // Frozen layers -> restore freeze panel state
                if (ts.containsKey("frozen_layers") && layerFreezePanel != null) {
                    Object fl = ts.get("frozen_layers");
                    if (fl instanceof List<?> frozenList) {
                        List<String> frozenNames = new ArrayList<>();
                        for (Object item : frozenList) {
                            frozenNames.add(String.valueOf(item));
                        }
                        // Defer until layer panel is populated (may be loading async)
                        Platform.runLater(() -> {
                            if (layerFreezePanel != null) {
                                layerFreezePanel.setFrozenLayerNames(frozenNames);
                            }
                        });
                    }
                }

                // Scheduler
                if (ts.containsKey("scheduler_type")) {
                    schedulerCombo.setValue(mapSchedulerToDisplay((String) ts.get("scheduler_type")));
                }

                // Loss function
                if (ts.containsKey("loss_function")) {
                    lossFunctionCombo.setValue(mapLossFunctionToDisplay((String) ts.get("loss_function")));
                }
                if (ts.containsKey("focal_gamma")) {
                    focalGammaSpinner.getValueFactory().setValue(
                            ((Number) ts.get("focal_gamma")).doubleValue());
                }
                if (ts.containsKey("boundary_sigma")) {
                    boundarySigmaSpinner.getValueFactory().setValue(
                            ((Number) ts.get("boundary_sigma")).doubleValue());
                }
                if (ts.containsKey("boundary_w_min")) {
                    boundaryWMinSpinner.getValueFactory().setValue(
                            ((Number) ts.get("boundary_w_min")).doubleValue());
                }
                if (ts.containsKey("ohem_hard_ratio")) {
                    ohemSpinner.getValueFactory().setValue(
                            (int) Math.round(((Number) ts.get("ohem_hard_ratio")).doubleValue() * 100));
                }
                if (ts.containsKey("ohem_hard_ratio_start")) {
                    ohemStartSpinner.getValueFactory().setValue(
                            (int) Math.round(((Number) ts.get("ohem_hard_ratio_start")).doubleValue() * 100));
                } else if (ts.containsKey("ohem_schedule")) {
                    // Back-compat: derive start from the old schedule field.
                    // "anneal" used to mean start at 100%, "fixed" meant start == end.
                    String sched = String.valueOf(ts.get("ohem_schedule"));
                    ohemStartSpinner.getValueFactory().setValue(
                            "anneal".equals(sched) ? 100 : ohemSpinner.getValue());
                }
                if (ts.containsKey("ohem_adaptive_floor")) {
                    ohemAdaptiveFloorCheck.setSelected(
                            Boolean.TRUE.equals(ts.get("ohem_adaptive_floor")));
                }

                // Early stopping
                if (ts.containsKey("early_stopping_metric")) {
                    earlyStoppingMetricCombo.setValue(
                            mapEarlyStoppingMetricToDisplay((String) ts.get("early_stopping_metric")));
                }
                if (ts.containsKey("early_stopping_patience")) {
                    earlyStoppingPatienceSpinner.getValueFactory().setValue(
                            ((Number) ts.get("early_stopping_patience")).intValue());
                }

                // Mixed precision
                if (ts.containsKey("mixed_precision")) {
                    mixedPrecisionCheck.setSelected((Boolean) ts.get("mixed_precision"));
                }

                // Focus class
                if (ts.containsKey("focus_class")) {
                    Object fc = ts.get("focus_class");
                    if (fc != null) {
                        String focusClassName = String.valueOf(fc);
                        // Defer to after classes are loaded if combo not yet populated
                        if (focusClassCombo.getItems().contains(focusClassName)) {
                            focusClassCombo.setValue(focusClassName);
                        } else {
                            // Will be set when classes are loaded via updateFocusClassCombo
                            // Store temporarily for post-load matching
                            Platform.runLater(() -> {
                                if (focusClassCombo.getItems().contains(focusClassName)) {
                                    focusClassCombo.setValue(focusClassName);
                                }
                            });
                        }
                    }
                }
                if (ts.containsKey("focus_class_min_iou")) {
                    focusClassMinIoUSpinner.getValueFactory().setValue(
                            ((Number) ts.get("focus_class_min_iou")).doubleValue());
                }

                // Augmentation config
                if (ts.containsKey("augmentation_config")) {
                    Object augObj = ts.get("augmentation_config");
                    if (augObj instanceof Map<?, ?> rawAugMap) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> augConfig = (Map<String, Object>) rawAugMap;
                        if (augConfig.containsKey("flip_horizontal"))
                            flipHorizontalCheck.setSelected(Boolean.TRUE.equals(augConfig.get("flip_horizontal")));
                        if (augConfig.containsKey("flip_vertical"))
                            flipVerticalCheck.setSelected(Boolean.TRUE.equals(augConfig.get("flip_vertical")));
                        if (augConfig.containsKey("rotation_90"))
                            rotationCheck.setSelected(Boolean.TRUE.equals(augConfig.get("rotation_90")));
                        // Legacy color_jitter -> map to brightfield intensity mode
                        if (augConfig.containsKey("color_jitter")
                                && Boolean.TRUE.equals(augConfig.get("color_jitter"))) {
                            intensityAugCombo.setValue("Brightfield (color jitter)");
                        }
                        if (augConfig.containsKey("elastic_deformation"))
                            elasticCheck.setSelected(Boolean.TRUE.equals(augConfig.get("elastic_deformation")));
                    }
                }

                // New intensity_aug_mode field (overrides legacy color_jitter if present)
                if (ts.containsKey("intensity_aug_mode")) {
                    String mode = String.valueOf(ts.get("intensity_aug_mode"));
                    intensityAugCombo.setValue(mapIntensityModeToDisplay(mode));
                }

                // Whole-image mode
                if (ts.containsKey("whole_image")) {
                    wholeImageCheck.setSelected(Boolean.TRUE.equals(ts.get("whole_image")));
                }

                // Gradient accumulation
                if (ts.containsKey("gradient_accumulation_steps")) {
                    gradientAccumulationSpinner.getValueFactory().setValue(
                            ((Number) ts.get("gradient_accumulation_steps")).intValue());
                }

                // Progressive resize
                if (ts.containsKey("progressive_resize")) {
                    progressiveResizeCheck.setSelected(
                            Boolean.TRUE.equals(ts.get("progressive_resize")));
                }

                // Handler-specific parameters (e.g., MuViT model_config, patch_size,
                // level_scales, rope_mode). Apply to the handler UI after architecture
                // is set so the correct handler UI exists.
                if (ts.containsKey("handler_parameters")) {
                    Object hp = ts.get("handler_parameters");
                    if (hp instanceof Map<?, ?> rawHpMap && currentHandlerUI != null) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> handlerParams = (Map<String, Object>) rawHpMap;
                        Platform.runLater(() -> {
                            if (currentHandlerUI != null) {
                                currentHandlerUI.applyParameters(handlerParams);
                            }
                        });
                    }
                }
            }

            // --- Classifier name and description ---
            // Use the original classifier name if it looks like a user-specified name
            // (not the generic "UNET Classifier" etc.). Otherwise prefix with "Retrain_".
            String originalName = metadata.getName();
            String timestamp = java.time.LocalDate.now().toString().replace("-", "");
            if (originalName != null && !originalName.isEmpty()
                    && !originalName.toUpperCase().endsWith(" CLASSIFIER")) {
                classifierNameField.setText(originalName);
            } else {
                classifierNameField.setText("Retrain_" + originalName + "_" + timestamp);
            }
            descriptionField.setText("Retrained from: " + originalName);

            // Store source model's class list for auto-matching after class loading
            sourceModelClassNames = metadata.getClassNames();

            // Resolve the model's .pt file path for continue-training
            ModelManager modelManager2 = new ModelManager();
            pretrainedModelPtPath = null;
            pretrainedModelArchitecture = metadata.getModelType();
            pretrainedModelBackbone = metadata.getBackbone();
            modelManager2.getModelPath(metadata.getId()).ifPresent(modelPath -> {
                // getModelPath prefers ONNX; we need the .pt file specifically
                java.nio.file.Path ptPath = modelPath.getParent().resolve("model.pt");
                if (java.nio.file.Files.exists(ptPath)) {
                    pretrainedModelPtPath = ptPath.toString();
                }
            });
            if (pretrainedModelPtPath == null) {
                logger.warn("No model.pt found for '{}' -- continue training not available",
                        metadata.getName());
                loadedModelLabel.setText("No model.pt found for: " + metadata.getName());
                loadedModelLabel.setStyle("-fx-text-fill: #cc6600; -fx-font-style: italic;");
            } else {
                // Auto-select continue training since user picked a model
                selectWeightInitStrategy(ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING);
                loadedModelLabel.setText("Loaded from: " + metadata.getName());
                loadedModelLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");
                // Defer compatibility check -- backbone combo is set via
                // Platform.runLater (line ~994) and hasn't updated yet.
                Platform.runLater(() -> {
                    validatePretrainedModelCompatibility();
                    updateValidation();
                });
            }

            logger.info("Settings loaded from model '{}'. {} training settings fields applied.",
                    metadata.getName(), ts != null ? ts.size() : 0);
        }

        /**
         * Checks if the current architecture/backbone selection matches the pretrained model.
         * Disables and unchecks the "Initialize weights" checkbox if they don't match,
         * since loading weights from an incompatible architecture would fail or produce
         * garbage results.
         */
        private void validatePretrainedModelCompatibility() {
            if (pretrainedModelPtPath == null || continueTrainingRadio == null) {
                return;
            }

            String currentArch = architectureCombo.getValue();
            String currentBackbone = backboneCombo.getValue();

            boolean archMatch = Objects.equals(currentArch, pretrainedModelArchitecture);
            boolean backboneMatch = Objects.equals(currentBackbone, pretrainedModelBackbone);

            if (!archMatch || !backboneMatch) {
                // Warn but don't disable -- let updateValidation() handle blocking
                loadedModelLabel.setText(String.format(
                        "Loaded: %s (architecture mismatch: %s/%s)",
                        loadedModelLabel.getText().replace("Loaded from: ", ""),
                        currentArch, currentBackbone));
                loadedModelLabel.setStyle("-fx-text-fill: #cc6600; -fx-font-style: italic;");
                // Clear the path so validation blocks training
                pretrainedModelPtPath = null;
                if (getSelectedWeightInitStrategy() ==
                        ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING) {
                    logger.info("Architecture/backbone changed from {}/{} to {}/{} " +
                            "-- continue training model invalidated",
                            pretrainedModelArchitecture, pretrainedModelBackbone,
                            currentArch, currentBackbone);
                }
            }
            updateValidation();
        }

        private TitledPane createBasicInfoSection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // Classifier name
            classifierNameField = new TextField();
            classifierNameField.setPromptText("e.g., Collagen_Classifier_v1");
            classifierNameField.setPrefWidth(300);
            classifierNameField.textProperty().addListener((obs, old, newVal) -> validateClassifierName(newVal));

            Label nameLabel = new Label("Classifier Name:");
            TooltipHelper.install(
                    "Unique identifier for this classifier.\n" +
                    "Used as the filename when saving.\n" +
                    "Only letters, numbers, underscore, and hyphen allowed.",
                    nameLabel, classifierNameField);

            grid.add(nameLabel, 0, row);
            grid.add(classifierNameField, 1, row);
            row++;

            // Description
            descriptionField = new TextArea();
            descriptionField.setPromptText("Optional description of what this classifier detects...");
            descriptionField.setPrefRowCount(2);
            descriptionField.setWrapText(true);

            Label descLabel = new Label("Description:");
            TooltipHelper.install(
                    "Optional free-text description of what this classifier detects.\n" +
                    "Stored in classifier metadata for documentation.\n" +
                    "Example: 'Collagen vs. epithelium in H&E stained liver sections'",
                    descLabel, descriptionField);

            // Description: advanced-only
            descLabel.visibleProperty().bind(advancedMode);
            descLabel.managedProperty().bind(advancedMode);
            descriptionField.visibleProperty().bind(advancedMode);
            descriptionField.managedProperty().bind(advancedMode);

            grid.add(descLabel, 0, row);
            grid.add(descriptionField, 1, row);

            TitledPane pane = new TitledPane("NAME YOUR CLASSIFIER", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create(
                    "Name and describe your classifier.\n" +
                    "Positioned after training settings so you can name based on your chosen parameters."));
            return pane;
        }

        private TitledPane createImageSourceSection() {
            VBox content = new VBox(8);
            content.setPadding(new Insets(10));

            Label info = new Label("Select project images to include in training:");
            info.setStyle("-fx-text-fill: #666;");

            imageSelectionList = new ListView<>();
            imageSelectionList.setCellFactory(lv -> new ImageSelectionCell(advancedMode));
            imageSelectionList.setPrefHeight(150);
            TooltipHelper.install(imageSelectionList,
                    "Check the project images to include in training.\n" +
                    "Only images with classified annotations are shown.\n" +
                    "Patches from all selected images are combined into one training set.");

            // Populate project images that have classified annotations
            Project<BufferedImage> project = QuPathGUI.getInstance().getProject();
            if (project != null) {
                for (ProjectImageEntry<BufferedImage> entry : project.getImageList()) {
                    try {
                        ImageData<BufferedImage> data = entry.readImageData();
                        long annotationCount = data.getHierarchy().getAnnotationObjects().stream()
                                .filter(a -> a.getPathClass() != null)
                                .count();
                        int imgW = data.getServer().getWidth();
                        int imgH = data.getServer().getHeight();
                        data.getServer().close();
                        if (annotationCount > 0) {
                            ImageSelectionItem item = new ImageSelectionItem(
                                    entry, annotationCount, imgW, imgH);
                            // When image selection changes, update button state, mark classes
                            // stale, and recheck whole-image size warning
                            item.selected.addListener((obs, old, newVal) -> {
                                updateLoadClassesButtonState();
                                if (classesLoaded) {
                                    markClassesStale();
                                }
                                updateWholeImageInfoLabel();
                                updateValidationSplitSpinnerState();
                            });
                            item.splitRole.addListener((obs, old, newVal) -> {
                                updateValidationSplitSpinnerState();
                                // Any manual override clears the auto-distribute
                                // highlight; auto-distribute itself runs inside
                                // its own guard so it doesn't trip this.
                                if (!inAutoDistribute) {
                                    applyAutoDistributeHighlight(false);
                                }
                            });
                            imageSelectionList.getItems().add(item);
                        }
                    } catch (Exception e) {
                        logger.debug("Could not read image '{}': {}",
                                entry.getImageName(), e.getMessage());
                    }
                }
            }

            // Select All / Select None buttons
            Button selectAllImagesBtn = new Button("Select All");
            TooltipHelper.install(selectAllImagesBtn, "Select all project images for training");
            selectAllImagesBtn.setOnAction(e ->
                    imageSelectionList.getItems().forEach(item -> item.selected.set(true)));

            Button selectNoneImagesBtn = new Button("Select None");
            TooltipHelper.install(selectNoneImagesBtn, "Deselect all project images");
            selectNoneImagesBtn.setOnAction(e ->
                    imageSelectionList.getItems().forEach(item -> item.selected.set(false)));

            autoDistributeBtn = new Button("Auto-Distribute");
            TooltipHelper.install(autoDistributeBtn,
                    "Assign selected images to Training or Validation using the current\n" +
                    "split percentage. Similar images -- same dimensions, channel count,\n" +
                    "and filename prefix -- are bucketed together so related images\n" +
                    "aren't clumped on one side of the split.\n\n" +
                    "Auto-distribute runs once when the dialog opens (highlighted while\n" +
                    "active). Click again to reshuffle with a new random seed, or use the\n" +
                    "per-image Train/Val dropdowns to override individual assignments.");
            autoDistributeBtn.setOnAction(e -> autoDistributeSelectedImages());
            // Per-image Train/Val dropdowns are only rendered in advanced mode,
            // so the button that manipulates them follows the same visibility.
            autoDistributeBtn.visibleProperty().bind(advancedMode);
            autoDistributeBtn.managedProperty().bind(advancedMode);
            // Apply highlight style now -- the initial auto-distribute happens
            // shortly after image-list population, so showing the highlight
            // from the start matches the active state the user will see.
            applyAutoDistributeHighlight(true);

            // Load Classes button
            loadClassesButton = new Button("Load Classes from Selected Images");
            loadClassesButton.setStyle("-fx-font-weight: bold;");
            loadClassesButton.setMaxWidth(Double.MAX_VALUE);
            TooltipHelper.install(loadClassesButton,
                    "Read annotations from the selected images and populate\n" +
                    "the class list with the union of all classes found.\n" +
                    "Also initializes channel configuration from the first image.");
            loadClassesButton.setOnAction(e -> loadClassesFromSelectedImages());

            HBox imageButtonBox = new HBox(10, selectAllImagesBtn, selectNoneImagesBtn, autoDistributeBtn);

            tileEstimateLabel = new Label();
            tileEstimateLabel.setWrapText(true);
            tileEstimateLabel.setStyle("-fx-text-fill: #2a7a2a; -fx-font-size: 11px;");
            tileEstimateLabel.setVisible(false);
            tileEstimateLabel.setManaged(false);

            content.getChildren().addAll(info, imageSelectionList, imageButtonBox,
                    loadClassesButton, tileEstimateLabel);

            // Show a message if no annotated images found
            if (imageSelectionList.getItems().isEmpty()) {
                Label noImagesLabel = new Label("No project images with classified annotations found.");
                noImagesLabel.setStyle("-fx-text-fill: #cc6600; -fx-font-style: italic;");
                content.getChildren().add(1, noImagesLabel);
            }

            // Initialize button state
            updateLoadClassesButtonState();

            // Run Auto-Distribute once on open so a sensible per-image train/val
            // split is in place before the user touches anything. New items
            // default to selected=true and splitRole=BOTH, so this never
            // overwrites a user choice -- it only seeds the initial state.
            // Deferred to runLater so listeners and validation-split spinner
            // are fully wired before splitRole writes fire their callbacks.
            Platform.runLater(() -> {
                if (imageSelectionList.getItems().size() >= 2) {
                    autoDistributeSelectedImages();
                }
            });

            TitledPane pane = new TitledPane("TRAINING DATA SOURCE", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select project images and load classes for training"));
            return pane;
        }

        /**
         * Returns a lowercased grouping key for image "similarity", used to
         * avoid putting all related images on one side of the train/val split.
         * Takes the filename stem up to the first '_', '-', or ' '.
         */
        private String filenamePrefix(String imageName) {
            if (imageName == null || imageName.isEmpty()) return "";
            int dot = imageName.lastIndexOf('.');
            String stem = (dot > 0) ? imageName.substring(0, dot) : imageName;
            int cut = stem.length();
            for (char sep : new char[]{'_', '-', ' '}) {
                int idx = stem.indexOf(sep);
                if (idx > 0 && idx < cut) cut = idx;
            }
            return stem.substring(0, cut).toLowerCase(Locale.ROOT);
        }

        /**
         * Auto-Distribute button handler. Bucket selected images by
         * (dimensions, channel count, filename prefix), shuffle each bucket,
         * and assign the first {@code ceil(bucket.size * valPct)} items to
         * VAL_ONLY and the rest to TRAIN_ONLY. Unselected items are never
         * touched. If any selected item already has a non-BOTH role, a
         * confirmation dialog is shown before anything is overwritten.
         */
        private void autoDistributeSelectedImages() {
            List<ImageSelectionItem> selected = imageSelectionList.getItems().stream()
                    .filter(item -> item.selected.get())
                    .collect(Collectors.toList());
            if (selected.size() < 2) {
                Dialogs.showInfoNotification("Auto-Distribute",
                        "Select at least two images before auto-distributing.");
                return;
            }
            // Guard against the splitRole listener clearing the highlight
            // while we're mid-distribution.
            inAutoDistribute = true;
            try {

            long fixedCount = selected.stream()
                    .filter(item -> item.splitRole.get() != SplitRole.BOTH)
                    .count();
            if (fixedCount > 0) {
                boolean proceed = Dialogs.showConfirmDialog("Auto-Distribute",
                        fixedCount + " of " + selected.size() + " selected images "
                                + "already have a fixed Train/Val role. "
                                + "Auto-Distribute will overwrite them. Continue?");
                if (!proceed) return;
            }

            int numChannels = -1;
            try {
                if (channelPanel != null && channelPanel.isValid()) {
                    numChannels = channelPanel.getChannelConfiguration().getNumChannels();
                }
            } catch (Exception ignored) {
                // channel config not loaded yet -- grouping key just won't include channel count
            }

            int splitPct = (validationSplitSpinner != null)
                    ? validationSplitSpinner.getValue()
                    : DLClassifierPreferences.getValidationSplit();
            double valFraction = splitPct / 100.0;

            Map<String, List<ImageSelectionItem>> buckets = new LinkedHashMap<>();
            for (ImageSelectionItem item : selected) {
                String key = item.imageWidth + "x" + item.imageHeight
                        + "|" + (numChannels > 0 ? String.valueOf(numChannels) : "?")
                        + "|" + filenamePrefix(item.entry.getImageName());
                buckets.computeIfAbsent(key, k -> new ArrayList<>()).add(item);
            }

            long seed = System.currentTimeMillis();
            Random rng = new Random(seed);
            int nTrain = 0;
            int nVal = 0;
            StringBuilder summary = new StringBuilder();
            for (Map.Entry<String, List<ImageSelectionItem>> e : buckets.entrySet()) {
                List<ImageSelectionItem> bucket = e.getValue();
                Collections.shuffle(bucket, rng);
                // Buckets of size 1 always go to training so every bucket
                // contributes at least one training image; validation gets
                // items only when the bucket has at least 2 images.
                int bucketVal = (bucket.size() == 1)
                        ? 0
                        : Math.max(1, Math.min(bucket.size() - 1,
                                (int) Math.round(bucket.size() * valFraction)));
                for (int i = 0; i < bucket.size(); i++) {
                    ImageSelectionItem item = bucket.get(i);
                    if (i < bucketVal) {
                        item.splitRole.set(SplitRole.VAL_ONLY);
                        nVal++;
                    } else {
                        item.splitRole.set(SplitRole.TRAIN_ONLY);
                        nTrain++;
                    }
                }
                if (summary.length() > 0) summary.append(", ");
                summary.append(e.getKey()).append(": ")
                        .append(bucket.size() - bucketVal).append("t/")
                        .append(bucketVal).append("v");
            }

            logger.info("Auto-distribute (seed={}): {} train / {} val across {} bucket(s); {}",
                    seed, nTrain, nVal, buckets.size(), summary);
            // The splitRole listeners installed when items were added will
            // fire updateValidationSplitSpinnerState() for us.

            // Highlight the button so the user can see the assignment came
            // from auto-distribute (vs. a manual or saved choice).
            applyAutoDistributeHighlight(true);
            } finally {
                inAutoDistribute = false;
            }
        }

        /**
         * Toggle the visual highlight on the Auto-Distribute button. Highlighted
         * = "this distribution is from auto-distribute, click to reshuffle".
         * Cleared once the user makes any manual per-image role change.
         */
        private void applyAutoDistributeHighlight(boolean active) {
            if (autoDistributeBtn == null) return;
            if (active) {
                autoDistributeBtn.setStyle(
                        "-fx-background-color: #fff3cd; "
                        + "-fx-border-color: #d39e00; "
                        + "-fx-border-width: 2; "
                        + "-fx-border-radius: 3; "
                        + "-fx-background-radius: 3; "
                        + "-fx-font-weight: bold;");
            } else {
                autoDistributeBtn.setStyle("");
            }
        }

        /**
         * Sync the Validation Split (%) spinner with the current per-image
         * role assignments. Three states:
         * <ul>
         *   <li>No fixed roles (all BOTH): spinner editable, observed label hidden.</li>
         *   <li>Some fixed roles, some BOTH: spinner editable, observed label shows
         *       how many images are fixed train/val.</li>
         *   <li>All selected images have fixed roles (no BOTH): spinner is disabled,
         *       shows the observed percentage, and the user's last manually-typed
         *       value is stashed so it can be restored when a BOTH image reappears.</li>
         * </ul>
         */
        private void updateValidationSplitSpinnerState() {
            if (validationSplitSpinner == null || imageSelectionList == null) return;

            int nTrain = 0;
            int nVal = 0;
            int nBoth = 0;
            for (ImageSelectionItem item : imageSelectionList.getItems()) {
                if (!item.selected.get()) continue;
                SplitRole r = item.splitRole.get();
                if (r == SplitRole.TRAIN_ONLY) nTrain++;
                else if (r == SplitRole.VAL_ONLY) nVal++;
                else nBoth++;
            }
            int fixedTotal = nTrain + nVal;

            if (nBoth == 0 && fixedTotal >= 2) {
                // Fully-manual mode: disable spinner, show observed.
                if (lastUserValidationSplitPct == null) {
                    lastUserValidationSplitPct = validationSplitSpinner.getValue();
                }
                int observedPct = (int) Math.round(100.0 * nVal / fixedTotal);
                updatingSpinnerProgrammatically = true;
                try {
                    validationSplitSpinner.getValueFactory().setValue(observedPct);
                } finally {
                    updatingSpinnerProgrammatically = false;
                }
                validationSplitSpinner.setDisable(true);
                if (validationSplitObservedLabel != null) {
                    validationSplitObservedLabel.setText(String.format(
                            "Disabled: all selected images have fixed roles "
                                    + "(%d train / %d val = %d%%). Change a "
                                    + "dropdown to 'Both' to re-enable.",
                            nTrain, nVal, observedPct));
                }
            } else {
                // Mixed or no fixed roles: spinner editable.
                if (validationSplitSpinner.isDisable() && lastUserValidationSplitPct != null) {
                    updatingSpinnerProgrammatically = true;
                    try {
                        validationSplitSpinner.getValueFactory()
                                .setValue(lastUserValidationSplitPct);
                    } finally {
                        updatingSpinnerProgrammatically = false;
                    }
                    lastUserValidationSplitPct = null;
                }
                validationSplitSpinner.setDisable(false);
                if (validationSplitObservedLabel != null) {
                    if (fixedTotal > 0) {
                        validationSplitObservedLabel.setText(String.format(
                                "Plus %d image(s) fixed to Train, %d fixed to Val "
                                        + "from manual assignments.",
                                nTrain, nVal));
                    } else {
                        validationSplitObservedLabel.setText("");
                    }
                }
            }
        }

        private TitledPane createModelSection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // Architecture selection
            List<String> architectures = new ArrayList<>(ClassifierRegistry.getAllTypes());
            architectureCombo = new ComboBox<>(FXCollections.observableArrayList(architectures));
            // Restore last used architecture from preferences, falling back to first in list
            String savedArchitecture = DLClassifierPreferences.getLastArchitecture();
            if (architectures.contains(savedArchitecture)) {
                architectureCombo.setValue(savedArchitecture);
            } else {
                architectureCombo.setValue(architectures.isEmpty() ? "unet" : architectures.get(0));
            }
            Label archLabel = new Label("Architecture:");
            TooltipHelper.installWithLink(
                    "Segmentation architecture:\n\n" +
                    "UNet: Encoder-decoder with skip connections.\n" +
                    "  Best general-purpose choice. Good default for most tasks.\n\n" +
                    "Tiny UNet: Minimal UNet, no pretrained weights.\n" +
                    "  Fast experiments, any channel count, low VRAM.\n\n" +
                    "Fast Pretrained: UNet with mobile encoders.\n" +
                    "  Quick training with lightweight ImageNet weights.\n\n" +
                    "MuViT (Transformer): Multi-resolution Vision Transformer.\n" +
                    "  Multi-scale fusion. Supports MAE pretraining.\n\n" +
                    "Custom ONNX: Import externally trained models.\n" +
                    "  Inference only. UNTESTED -- expect rough edges.\n\n" +
                    "Click '?' for detailed guide with references.",
                    "https://arxiv.org/abs/1505.04597",
                    archLabel, architectureCombo);
            architectureCombo.valueProperty().addListener((obs, old, newVal) -> updateBackboneOptions(newVal));

            Button archHelpBtn = new Button("?");
            archHelpBtn.setStyle("-fx-font-size: 10; -fx-padding: 1 6 1 6; -fx-min-width: 22;");
            archHelpBtn.setTooltip(new Tooltip("Model architecture guide"));
            archHelpBtn.setOnAction(e -> showArchitectureGuide());

            // Architecture row: hidden in basic mode (fixed to unet)
            archLabel.visibleProperty().bind(advancedMode);
            archLabel.managedProperty().bind(advancedMode);
            architectureCombo.visibleProperty().bind(advancedMode);
            architectureCombo.managedProperty().bind(advancedMode);
            archHelpBtn.visibleProperty().bind(advancedMode);
            archHelpBtn.managedProperty().bind(advancedMode);
            grid.add(archLabel, 0, row);
            grid.add(architectureCombo, 1, row);
            grid.add(archHelpBtn, 2, row);
            row++;

            // Backbone selection
            backboneCombo = new ComboBox<>();
            // Tooltip installed after backboneLabel is created below
            updateBackboneOptions(architectureCombo.getValue());

            // When architecture or backbone changes, invalidate pretrained weight loading
            // if the new selection doesn't match the source model
            architectureCombo.valueProperty().addListener((obs, old, newVal) ->
                    validatePretrainedModelCompatibility());
            backboneCombo.valueProperty().addListener((obs, old, newVal) ->
                    validatePretrainedModelCompatibility());
            backboneCombo.valueProperty().addListener((obs, old, newVal) -> updateLayerFreezePanel());

            backboneLabel = new Label("Encoder:");
            TooltipHelper.installWithLink(
                    "Encoder (backbone) that extracts image features.\n" +
                    "Options depend on the selected architecture.\n\n" +
                    "--- UNet: Standard (ImageNet) ---\n" +
                    "resnet34: Best default. Good balance of speed and accuracy.\n" +
                    "resnet50: More capacity for larger datasets.\n" +
                    "efficientnet-b0: Lightweight, fast inference.\n" +
                    "mobilenet_v2: Fastest inference, smallest model.\n\n" +
                    "--- UNet: Histology-pretrained ---\n" +
                    "Lunit, Kather100K, TCGA-BRCA: Trained on H&E tissue\n" +
                    "at 20x. Best for H&E brightfield. ~100MB download.\n\n" +
                    "--- UNet: Foundation Models ---\n" +
                    "H-optimus-0, Virchow, Hibou, Midnight, DINOv2:\n" +
                    "Large-scale models (86M-1.1B params). ~200MB-2GB download.\n" +
                    "Gated models need HF_TOKEN env var.\n\n" +
                    "--- Tiny UNet ---\n" +
                    "Size presets (Nano/Tiny/Compact/Small). No pretrained\n" +
                    "weights -- trains from scratch, any channel count.\n\n" +
                    "--- Fast Pretrained ---\n" +
                    "EfficientNet-Lite0 or MobileNetV3-Small. Lightweight\n" +
                    "ImageNet encoders for fast training.\n\n" +
                    "Click '?' on the Architecture row for a detailed guide.",
                    "https://github.com/uw-loci/qupath-extension-dl-pixel-classifier/blob/main/docs/BEST_PRACTICES.md#backbone-selection",
                    backboneLabel, backboneCombo);
            grid.add(backboneLabel, 0, row);
            grid.add(backboneCombo, 1, row);
            row++;

            // Basic mode guidance (hidden in advanced)
            Label basicModelHint = new Label(
                    "Small: Fast training, low VRAM. Good starting point.\n" +
                    "Medium: Best balance of speed and accuracy (recommended).\n" +
                    "Large: Most capacity, needs more data and VRAM.");
            basicModelHint.setWrapText(true);
            basicModelHint.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
            basicModelHint.visibleProperty().bind(advancedMode.not());
            basicModelHint.managedProperty().bind(advancedMode.not());
            grid.add(basicModelHint, 0, row, 2, 1);

            // Help button for basic mode (hidden in advanced)
            Button basicHelpBtn = new Button("?");
            basicHelpBtn.setStyle("-fx-font-size: 10; -fx-padding: 1 6 1 6; -fx-min-width: 22;");
            basicHelpBtn.setTooltip(new Tooltip("Learn about model sizes"));
            basicHelpBtn.setOnAction(e -> showBasicArchitectureGuide());
            basicHelpBtn.visibleProperty().bind(advancedMode.not());
            basicHelpBtn.managedProperty().bind(advancedMode.not());
            grid.add(basicHelpBtn, 2, row);

            // Dynamic handler-specific UI (e.g., MuViT transformer parameters)
            handlerUIContainer = new javafx.scene.layout.VBox();
            handlerUIContainer.visibleProperty().bind(advancedMode);
            handlerUIContainer.managedProperty().bind(advancedMode);
            architectureCombo.valueProperty().addListener((obs, old, newVal) -> {
                updateHandlerUI(newVal);
                updateSectionsForArchitecture(newVal);
            });
            updateHandlerUI(architectureCombo.getValue());

            // In basic mode, force unet architecture and re-filter backbone options
            advancedMode.addListener((obs, old, isAdv) -> {
                if (!isAdv && !"unet".equals(architectureCombo.getValue())) {
                    String oldArch = architectureCombo.getValue();
                    // Setting the value triggers the combo's listener which calls updateBackboneOptions
                    architectureCombo.setValue("unet");
                    showTemporaryNotification(String.format(
                            "Architecture changed from %s to UNet (basic mode only supports UNet).", oldArch));
                } else {
                    updateBackboneOptions(architectureCombo.getValue());
                }
            });

            javafx.scene.layout.VBox modelContent = new javafx.scene.layout.VBox(5, grid, handlerUIContainer);
            TitledPane pane = new TitledPane("MODEL ARCHITECTURE", modelContent);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            TooltipHelper.install(pane,
                    "Select the neural network encoder.\n" +
                    "In basic mode, only ResNet encoders are shown.\n" +
                    "Switch to advanced mode for all architectures and encoders.");
            return pane;
        }

        /**
         * Builds a TrainingConfig from the current dialog state.
         * Used by both buildResult() and copyTrainingScript() to avoid duplication.
         */
        private TrainingConfig buildTrainingConfig() {
            ClassifierHandler.WeightInitStrategy strategy = getSelectedWeightInitStrategy();
            if (strategy == null) {
                strategy = ClassifierHandler.WeightInitStrategy.SCRATCH;
            }

            boolean usePretrained = (strategy == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);
            String pretrainedPath = null;
            List<String> frozenLayers = new ArrayList<>();

            switch (strategy) {
                case BACKBONE_PRETRAINED:
                    if (layerFreezePanel != null) {
                        frozenLayers = layerFreezePanel.getFrozenLayerNames();
                    }
                    break;
                case MAE_ENCODER:
                    String maePath = maeEncoderPathField.getText();
                    if (maePath != null && !maePath.isEmpty()) {
                        pretrainedPath = maePath;
                    }
                    // Treat an MAE-pretrained encoder as "pretrained" so the Python
                    // optimizer setup applies discriminative LRs (encoder gets
                    // 0.1x lr, head gets full lr). Without this, AdamW trains every
                    // parameter at the same aggressive supervised LR and the MAE
                    // features are destroyed in a handful of epochs -- the classic
                    // "fine-tuning catastrophic forgetting" failure.
                    usePretrained = true;
                    break;
                case SSL_ENCODER:
                    String sslPath = sslEncoderPathField.getText();
                    if (sslPath != null && !sslPath.isEmpty()) {
                        pretrainedPath = sslPath;
                    }
                    // Same discriminative LR reasoning as MAE_ENCODER above.
                    usePretrained = true;
                    break;
                case CONTINUE_TRAINING:
                    pretrainedPath = pretrainedModelPtPath;
                    // Continue training should use discriminative LRs (encoder gets
                    // lower LR to preserve learned features, decoder/head get full LR
                    // to adapt to new/changed annotations). Also apply frozen layers
                    // if the user selected any.
                    usePretrained = true;
                    if (layerFreezePanel != null) {
                        frozenLayers = layerFreezePanel.getFrozenLayerNames();
                    }
                    break;
                default:
                    break;
            }

            return TrainingConfig.builder()
                    .classifierType(architectureCombo.getValue())
                    .backbone(backboneCombo.getValue())
                    .epochs(epochsSpinner.getValue())
                    .batchSize(batchSizeSpinner.getValue())
                    .learningRate(learningRateSpinner.getValue())
                    .weightDecay(weightDecaySpinner.getValue())
                    .discriminativeLrRatio(discriminativeLrSpinner.getValue())
                    .seed(seedSpinner.getValue() == 0 ? null : seedSpinner.getValue())
                    .validationSplit(validationSplitSpinner.getValue() / 100.0)
                    .tileSize(tileSizeSpinner.getValue())
                    .trainingPixelSizeMicrons(nativePixelSizeMicrons)
                    .overlap(overlapSpinner.getValue())
                    .downsample(parseDownsample(downsampleCombo.getValue()))
                    .contextScale(parseContextScale(contextScaleCombo.getValue()))
                    .augmentation(buildAugmentationConfig())
                    .augmentationParams(AdvancedAugmentationDialog.buildParamsFromPreferences())
                    .intensityAugMode(mapIntensityModeFromDisplay(intensityAugCombo.getValue()))
                    .usePretrainedWeights(usePretrained)
                    .frozenLayers(frozenLayers)
                    .lineStrokeWidth(lineStrokeWidthSpinner.getValue())
                    .classWeightMultipliers(getClassWeightMultipliers())
                    .schedulerType(mapSchedulerFromDisplay(schedulerCombo.getValue()))
                    .lossFunction(mapLossFunctionFromDisplay(lossFunctionCombo.getValue()))
                    .focalGamma(focalGammaSpinner.getValue())
                    .boundarySigma(boundarySigmaSpinner.getValue())
                    .boundaryWMin(boundaryWMinSpinner.getValue())
                    .hasPerImageSplitRoles(computeHasPerImageSplitRoles())
                    .ohemHardRatio(ohemSpinner.getValue() / 100.0)
                    .ohemHardRatioStart(ohemStartSpinner.getValue() / 100.0)
                    .ohemSchedule(ohemStartSpinner.getValue() > ohemSpinner.getValue()
                            ? "anneal" : "fixed")
                    .ohemAdaptiveFloor(ohemAdaptiveFloorCheck.isSelected())
                    .inMemoryDataset(DLClassifierPreferences.getDefaultInMemoryDataset())
                    .earlyStoppingMetric(mapEarlyStoppingMetricFromDisplay(earlyStoppingMetricCombo.getValue()))
                    .earlyStoppingPatience(earlyStoppingPatienceSpinner.getValue())
                    .mixedPrecision(mixedPrecisionCheck.isSelected())
                    .fusedOptimizer(fusedOptimizerCheck != null
                            ? fusedOptimizerCheck.isSelected() : true)
                    .useLrFinder(useLrFinderCheck != null
                            ? useLrFinderCheck.isSelected() : true)
                    .gpuAugmentation(gpuAugmentationCheck != null
                            && gpuAugmentationCheck.isSelected())
                    .useTorchCompile(useTorchCompileCheck != null
                            && useTorchCompileCheck.isSelected())
                    .gradientAccumulationSteps(gradientAccumulationSpinner.getValue())
                    .progressiveResize(progressiveResizeCheck.isSelected())
                    .focusClass(mapFocusClassFromDisplay(focusClassCombo.getValue()))
                    .focusClassMinIoU(focusClassMinIoUSpinner.getValue())
                    .pretrainedModelPath(pretrainedPath)
                    .wholeImage(wholeImageCheck.isSelected())
                    .handlerParameters(currentHandlerUI != null
                            ? currentHandlerUI.getParameters() : Map.of())
                    .build();
        }

        /**
         * @return true iff at least one selected image has an explicit
         *     TRAIN_ONLY or VAL_ONLY role. Used by
         *     TileOverlapSplitWatcher to detect the overlap+leakage
         *     situation (only per-image roles prevent leakage).
         */
        private boolean computeHasPerImageSplitRoles() {
            if (imageSelectionList == null) return false;
            for (ImageSelectionItem item : imageSelectionList.getItems()) {
                if (!item.selected.get()) continue;
                SplitRole role = item.splitRole.get();
                if (role == SplitRole.TRAIN_ONLY
                        || role == SplitRole.VAL_ONLY) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Extracts class weight multipliers from the class list view.
         */
        private Map<String, Double> getClassWeightMultipliers() {
            Map<String, Double> multipliers = new LinkedHashMap<>();
            for (ClassItem item : classListView.getItems()) {
                if (item.selected().get()) {
                    double multiplier = item.weightMultiplier().get();
                    if (multiplier != 1.0) {
                        multipliers.put(item.name(), multiplier);
                    }
                }
            }
            return multipliers;
        }

        /**
         * Updates the LR info label based on the selected scheduler.
         * When OneCycleLR is selected, warns that the LR finder will
         * override the user's learning rate.
         */
        private void updateLrInfoLabel() {
            if (lrInfoLabel == null || schedulerCombo == null) return;
            String sched = schedulerCombo.getValue();
            if ("One Cycle".equals(sched)) {
                lrInfoLabel.setText(
                        "OneCycleLR: LR Finder will auto-adjust peak lr " +
                        "(capped at 10x this value). This value sets the " +
                        "base for discriminative LR ratios.");
                lrInfoLabel.setStyle(
                        "-fx-text-fill: #856404; -fx-font-size: 0.85em; " +
                        "-fx-background-color: #FFF3CD; -fx-padding: 4 6; " +
                        "-fx-background-radius: 3;");
            } else if ("None".equals(sched)) {
                lrInfoLabel.setText("Constant lr throughout training.");
                lrInfoLabel.setStyle("-fx-text-fill: #666666; -fx-font-size: 0.85em;");
            } else {
                lrInfoLabel.setText("Starting lr -- scheduler adjusts during training.");
                lrInfoLabel.setStyle("-fx-text-fill: #666666; -fx-font-size: 0.85em;");
            }
            // Visibility controlled by advancedMode binding -- do not set here
        }

        private void updateEffectiveLrLabel() {
            if (effectiveLrLabel == null || learningRateSpinner == null
                    || discriminativeLrSpinner == null) return;

            ClassifierHandler.WeightInitStrategy strategy = getSelectedWeightInitStrategy();
            boolean isPretrained = strategy != null
                    && strategy != ClassifierHandler.WeightInitStrategy.SCRATCH;

            if (!isPretrained) {
                effectiveLrLabel.setText("");
                return;
            }

            double lr = learningRateSpinner.getValue();
            double ratio = discriminativeLrSpinner.getValue();
            double encoderLr = lr * ratio;

            String text = String.format("Encoder: %.6f | Decoder: %.6f | Head: %.6f",
                    encoderLr, lr, lr);

            if (schedulerCombo != null && "One Cycle".equals(schedulerCombo.getValue())) {
                text += " (peak LRs auto-found by LR finder)";
            }

            effectiveLrLabel.setText(text);
        }

        private void updateBackboneInfoText(String backbone, Label backboneInfo) {
            if (backbone == null || backboneInfo == null) return;

            // Basic mode: simplified language for non-ML users
            if (!advancedMode.get()) {
                backbonePretrainedRadio.setText("Start from a pre-trained model (recommended)");
                backboneInfo.setText(
                        "The model begins with knowledge learned from millions of images, " +
                        "so it needs less of your data to learn well.");
                return;
            }

            // Advanced mode: full technical detail
            if (isFoundationModel(backbone)) {
                backboneInfo.setText(
                        "Foundation model encoder (downloaded on-demand from HuggingFace). " +
                        "Large-scale vision model with rich feature representations. " +
                        "Gated models need HF_TOKEN env var. " +
                        "Inspired by LazySlide (Zheng et al. 2026, Nature Methods).");
                backbonePretrainedRadio.setText("Use foundation model pretrained weights");
            } else if (backbone.contains("_")) {
                backboneInfo.setText(
                        "Histology-pretrained on H&E tissue patches at 20x (3-channel RGB). " +
                        "Best for H&E brightfield. Less freezing needed. " +
                        "For fluorescence or multi-channel images, use an ImageNet backbone instead. " +
                        "~100MB download on first use (cached).");
                backbonePretrainedRadio.setText("Use histology pretrained weights");
            } else {
                backboneInfo.setText(
                        "ImageNet-pretrained weights provide general edge/texture features " +
                        "that transfer to most image types including fluorescence and multi-channel. " +
                        "Freeze early layers to preserve these features.");
                backbonePretrainedRadio.setText("Use pretrained backbone weights");
            }
        }

        private void updateEarlyStoppingStatusLabel() {
            if (earlyStoppingStatusLabel == null) return;
            String metric = earlyStoppingMetricCombo != null
                    ? mapEarlyStoppingMetricFromDisplay(earlyStoppingMetricCombo.getValue())
                    : "mean_iou";
            if ("disabled".equals(metric)) {
                earlyStoppingStatusLabel.setText(
                        "Early stopping disabled (will train all configured epochs)");
                return;
            }
            int patience = earlyStoppingPatienceSpinner != null
                    ? earlyStoppingPatienceSpinner.getValue() : 15;
            String metricDisplay = "mean_iou".equals(metric) ? "Mean IoU" : "Validation Loss";
            earlyStoppingStatusLabel.setText(String.format(
                    "Early stopping enabled (patience: %d, metric: %s)", patience, metricDisplay));
        }

        private void updateLayerFreezeInfoLabel(ClassifierHandler.WeightInitStrategy selected) {
            if (layerFreezeInfoLabel == null) return;
            if (selected == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING) {
                // Strong warning: changing freeze settings here can either help the
                // model adapt to a new data distribution or destroy features that
                // took the original training run to learn. Spell out both directions.
                layerFreezeInfoLabel.setStyle("-fx-text-fill: #856404; "
                        + "-fx-background-color: #fff3cd; -fx-padding: 6 8; "
                        + "-fx-background-radius: 3; -fx-font-size: 11px;");
                layerFreezeInfoLabel.setText(
                        "Continuing training inherits the freeze list from the loaded "
                        + "model. Changing it has real consequences:\n"
                        + "  - UNFREEZING early encoder layers (conv1, layer1, layer2) "
                        + "lets the model adapt low-level features (color, scale, "
                        + "texture) to a new data distribution, but with too high a "
                        + "learning rate it can wash out features the original run "
                        + "learned. Pair with a low encoder LR (discriminative LRs).\n"
                        + "  - FREEZING more layers preserves the existing model but "
                        + "prevents it from learning anything new in those layers -- "
                        + "useful only if your new data is very similar to the "
                        + "original training set.\n"
                        + "If your new data comes from a different microscope, "
                        + "stain, or downsample, prefer unfreezing the encoder.");
            } else if (selected == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED) {
                layerFreezeInfoLabel.setStyle("-fx-text-fill: #555; -fx-font-size: 11px;");
                layerFreezeInfoLabel.setText(
                        "Freezing early encoder layers preserves ImageNet/MAE features "
                        + "and trains faster, but limits how much the model can adapt "
                        + "to your data. Unfreeze more layers if your images differ "
                        + "strongly from the pretrained domain.");
            } else {
                layerFreezeInfoLabel.setText("");
            }
        }

        private void updateLayerFreezePanel() {
            if (layerFreezePanel == null) return;
            ClassifierHandler.WeightInitStrategy strat = getSelectedWeightInitStrategy();
            if (strat != ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED
                    && strat != ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING) {
                return;
            }

            String architecture = architectureCombo.getValue();
            String encoder = backboneCombo.getValue();

            if (architecture != null && encoder != null && channelPanel != null) {
                // Guard: channel panel may not have channels selected yet during init
                if (!channelPanel.isValid()) return;

                try {
                    int numChannels = channelPanel.getChannelConfiguration().getNumChannels();
                    int numClasses = (int) classListView.getItems().stream()
                            .filter(item -> item.selected().get())
                            .count();
                    if (numClasses < 2) numClasses = 2;

                    // Load layers asynchronously to avoid blocking the FX thread on HTTP call
                    final int ch = numChannels;
                    final int cls = numClasses;
                    final int ctxScale = contextScaleCombo != null
                            ? parseContextScale(contextScaleCombo.getValue()) : 1;
                    CompletableFuture.runAsync(() -> {
                        layerFreezePanel.loadLayers(architecture, encoder, ch, cls, ctxScale);
                    });
                } catch (Exception e) {
                    logger.warn("Could not update layer freeze panel: {}", e.getMessage());
                }
            }
        }

        private TitledPane createTrainingSection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // Epochs
            epochsSpinner = new Spinner<>(1, 1000, DLClassifierPreferences.getDefaultEpochs(), 10);
            epochsSpinner.setEditable(true);
            epochsSpinner.setPrefWidth(100);

            Label epochsLabel = new Label("Epochs:");
            TooltipHelper.install(
                    "Number of complete passes through the training data.\n" +
                    "More epochs allow the model to learn more but risk overfitting.\n" +
                    "Watch validation loss to determine when to stop.\n\n" +
                    "Typical range: 50-200 for small datasets, 20-100 for large.\n" +
                    "Early stopping will halt training automatically if the model\n" +
                    "stops improving, so it is safe to set a high value.",
                    epochsLabel, epochsSpinner);

            grid.add(epochsLabel, 0, row);
            grid.add(epochsSpinner, 1, row);
            row++;

            // Epochs hint (basic mode only)
            Label epochsHint = new Label("Training passes over all data. Early stopping halts if no improvement.");
            epochsHint.setWrapText(true);
            epochsHint.setStyle("-fx-text-fill: #888; -fx-font-size: 11px;");
            epochsHint.visibleProperty().bind(advancedMode.not());
            epochsHint.managedProperty().bind(advancedMode.not());
            grid.add(epochsHint, 0, row, 2, 1);
            row++;

            // Early stopping status (basic mode only)
            earlyStoppingStatusLabel = new Label("Early stopping enabled (patience: 15, metric: Mean IoU)");
            earlyStoppingStatusLabel.setWrapText(true);
            earlyStoppingStatusLabel.setStyle("-fx-text-fill: #2a7a2a; -fx-font-size: 11px;");
            earlyStoppingStatusLabel.visibleProperty().bind(advancedMode.not());
            earlyStoppingStatusLabel.managedProperty().bind(advancedMode.not());
            grid.add(earlyStoppingStatusLabel, 0, row, 2, 1);
            row++;

            // Batch size
            batchSizeSpinner = new Spinner<>(1, 128, DLClassifierPreferences.getDefaultBatchSize(), 4);
            batchSizeSpinner.setEditable(true);
            batchSizeSpinner.setPrefWidth(100);

            Label batchLabel = new Label("Batch Size:");
            TooltipHelper.install(
                    "Number of tiles processed together in each training step.\n" +
                    "Larger batches give more stable gradients but use more GPU memory.\n" +
                    "Reduce if you get CUDA out-of-memory errors.\n\n" +
                    "Typical: 4-16 depending on tile size and GPU VRAM.\n" +
                    "With 512px tiles: 4-8 for 8GB VRAM, 8-16 for 12+ GB VRAM.\n" +
                    "With 256px tiles: double the above batch sizes.",
                    batchLabel, batchSizeSpinner);

            grid.add(batchLabel, 0, row);
            grid.add(batchSizeSpinner, 1, row);
            row++;

            // Batch size hint (basic mode only)
            Label batchHint = new Label("Images processed at once. Larger = faster but more GPU memory.");
            batchHint.setWrapText(true);
            batchHint.setStyle("-fx-text-fill: #888; -fx-font-size: 11px;");
            batchHint.visibleProperty().bind(advancedMode.not());
            batchHint.managedProperty().bind(advancedMode.not());
            grid.add(batchHint, 0, row, 2, 1);
            row++;

            // Live VRAM estimate (updated when architecture/batch/tile/etc. change)
            vramEstimateLabel = new Label();
            vramEstimateLabel.setWrapText(true);
            vramEstimateLabel.setStyle("-fx-font-size: 11px;");
            grid.add(vramEstimateLabel, 0, row, 3, 1);
            row++;

            // Learning rate (advanced only)
            learningRateSpinner = new Spinner<>(0.00001, 1.0, DLClassifierPreferences.getDefaultLearningRate(), 0.0001);
            learningRateSpinner.setEditable(true);
            learningRateSpinner.setPrefWidth(100);
            // Default StringConverter rounds 0.001 to "0.0" - use enough decimal places
            var lrFactory = (SpinnerValueFactory.DoubleSpinnerValueFactory) learningRateSpinner.getValueFactory();
            lrFactory.setConverter(new javafx.util.StringConverter<Double>() {
                @Override
                public String toString(Double value) {
                    return value == null ? "" : String.format("%.5f", value);
                }
                @Override
                public Double fromString(String string) {
                    try {
                        return Double.parseDouble(string.trim());
                    } catch (NumberFormatException e) {
                        return lrFactory.getValue();
                    }
                }
            });
            Label lrLabel = new Label("Learning Rate:");
            TooltipHelper.install(
                    "Controls the step size during gradient descent.\n" +
                    "Too high: training oscillates or diverges. Too low: training stalls.\n\n" +
                    "Recommended: 1e-4 (0.0001) -- stable for both fresh training and\n" +
                    "continue-training. With discriminative LRs, the encoder gets 1/10th\n" +
                    "this rate (1e-5) and decoder/head get the full rate.\n\n" +
                    "Only increase to 1e-3 if training is very slow to converge and you\n" +
                    "are using OneCycleLR (which auto-finds the optimal max LR).\n\n" +
                    "The LR scheduler will further adjust the rate during training.",
                    lrLabel, learningRateSpinner);
            lrLabel.visibleProperty().bind(advancedMode);
            lrLabel.managedProperty().bind(advancedMode);
            learningRateSpinner.visibleProperty().bind(advancedMode);
            learningRateSpinner.managedProperty().bind(advancedMode);

            grid.add(lrLabel, 0, row);
            grid.add(learningRateSpinner, 1, row);
            row++;

            // LR info label -- updates based on scheduler selection (advanced only)
            lrInfoLabel = new Label();
            lrInfoLabel.setWrapText(true);
            lrInfoLabel.setMaxWidth(Double.MAX_VALUE);
            lrInfoLabel.setStyle("-fx-text-fill: #666666; -fx-font-size: 0.85em;");
            lrInfoLabel.visibleProperty().bind(advancedMode);
            lrInfoLabel.managedProperty().bind(advancedMode);
            grid.add(lrInfoLabel, 0, row, 2, 1);
            row++;

            // Discriminative LR ratio (advanced only)
            discriminativeLrSpinner = new Spinner<>(0.01, 1.0,
                    DLClassifierPreferences.getDefaultDiscriminativeLrRatio(), 0.05);
            discriminativeLrSpinner.setEditable(true);
            discriminativeLrSpinner.setPrefWidth(100);
            var discLrFactory = (SpinnerValueFactory.DoubleSpinnerValueFactory)
                    discriminativeLrSpinner.getValueFactory();
            discLrFactory.setConverter(new javafx.util.StringConverter<Double>() {
                @Override public String toString(Double value) {
                    return value == null ? "" : String.format("%.2f", value);
                }
                @Override public Double fromString(String string) {
                    try { return Double.parseDouble(string.trim()); }
                    catch (NumberFormatException e) { return discLrFactory.getValue(); }
                }
            });
            Label discLrLabel = new Label("Encoder LR Factor:");
            TooltipHelper.install(
                    "Ratio applied to the base learning rate for encoder layers.\n\n" +
                    "0.1 (default): Encoder trains at 1/10th the decoder LR.\n" +
                    "  Good for ImageNet-pretrained backbones on new domains.\n\n" +
                    "0.3 - 0.5: For histology-pretrained backbones on histology data.\n" +
                    "  The pretrained features are already domain-relevant.\n\n" +
                    "1.0: Same LR for all layers (no discriminative LRs).\n\n" +
                    "Lower values preserve pretrained features more aggressively.\n" +
                    "Only applies when using pretrained weights (not from scratch).",
                    discLrLabel, discriminativeLrSpinner);
            discLrLabel.visibleProperty().bind(advancedMode);
            discLrLabel.managedProperty().bind(advancedMode);
            discriminativeLrSpinner.visibleProperty().bind(advancedMode);
            discriminativeLrSpinner.managedProperty().bind(advancedMode);
            grid.add(discLrLabel, 0, row);
            grid.add(discriminativeLrSpinner, 1, row);
            row++;

            // Effective per-group LR display (advanced only)
            effectiveLrLabel = new Label();
            effectiveLrLabel.setWrapText(true);
            effectiveLrLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
            effectiveLrLabel.visibleProperty().bind(advancedMode);
            effectiveLrLabel.managedProperty().bind(advancedMode);
            grid.add(effectiveLrLabel, 0, row, 2, 1);
            row++;

            // Update effective LR display when inputs change
            learningRateSpinner.valueProperty().addListener((obs, o, n) -> updateEffectiveLrLabel());
            discriminativeLrSpinner.valueProperty().addListener((obs, o, n) -> updateEffectiveLrLabel());
            updateEffectiveLrLabel();

            // Weight decay (advanced only)
            weightDecaySpinner = new Spinner<>(0.0, 0.5,
                    DLClassifierPreferences.getDefaultWeightDecay(), 0.005);
            weightDecaySpinner.setEditable(true);
            weightDecaySpinner.setPrefWidth(100);
            var wdFactory = (SpinnerValueFactory.DoubleSpinnerValueFactory)
                    weightDecaySpinner.getValueFactory();
            wdFactory.setConverter(new javafx.util.StringConverter<Double>() {
                @Override public String toString(Double value) {
                    return value == null ? "" : String.format("%.3f", value);
                }
                @Override public Double fromString(String string) {
                    try { return Double.parseDouble(string.trim()); }
                    catch (NumberFormatException e) { return wdFactory.getValue(); }
                }
            });
            Label wdLabel = new Label("Weight Decay:");
            TooltipHelper.install(
                    "L2 regularization strength (AdamW). Penalizes large weights\n" +
                    "to prevent overfitting.\n\n" +
                    "0.01 (default): Good for most training runs.\n" +
                    "0.05 - 0.1: Increase for very small datasets.\n" +
                    "0.001: Decrease for large datasets or from-scratch training.\n" +
                    "0: Disable weight decay entirely.",
                    wdLabel, weightDecaySpinner);
            wdLabel.visibleProperty().bind(advancedMode);
            wdLabel.managedProperty().bind(advancedMode);
            weightDecaySpinner.visibleProperty().bind(advancedMode);
            weightDecaySpinner.managedProperty().bind(advancedMode);
            grid.add(wdLabel, 0, row);
            grid.add(weightDecaySpinner, 1, row);
            row++;

            // Reproducibility seed (advanced only)
            seedSpinner = new Spinner<>(0, 999999, DLClassifierPreferences.getLastSeed(), 1);
            seedSpinner.setEditable(true);
            seedSpinner.setPrefWidth(100);
            Label seedLabel = new Label("Random Seed:");
            TooltipHelper.install(
                    "Set a fixed seed for reproducible training results.\n\n" +
                    "0 (default): Non-deterministic -- results vary between runs.\n" +
                    "Any positive value: Deterministic -- same results given same\n" +
                    "  data and settings. Useful for ablation studies.\n\n" +
                    "Note: Deterministic mode may reduce training speed by 10-20%\n" +
                    "due to cuDNN deterministic algorithms.",
                    seedLabel, seedSpinner);
            seedLabel.visibleProperty().bind(advancedMode);
            seedLabel.managedProperty().bind(advancedMode);
            seedSpinner.visibleProperty().bind(advancedMode);
            seedSpinner.managedProperty().bind(advancedMode);
            grid.add(seedLabel, 0, row);
            grid.add(seedSpinner, 1, row);
            row++;

            // Validation split (advanced only)
            validationSplitSpinner = new Spinner<>(5, 50, DLClassifierPreferences.getValidationSplit(), 5);
            validationSplitSpinner.setEditable(true);
            validationSplitSpinner.setPrefWidth(100);
            Label valSplitLabel = new Label("Validation Split (%):");
            TooltipHelper.install(
                    "Percentage of annotated tiles held out for validation.\n" +
                    "Used to monitor overfitting and select the best model.\n" +
                    "Higher values give more reliable validation metrics\n" +
                    "but leave less data for training.\n\n" +
                    "15-25% is typical. Use 10% for very small datasets\n" +
                    "(fewer than ~50 annotations). Use 25-30% for large\n" +
                    "datasets where you can afford it.\n\n" +
                    "Important: if you use a Focus Class, ensure it has\n" +
                    "enough annotations that some end up in the validation\n" +
                    "split. A focus class with 0 validation samples will\n" +
                    "always show 0.0 IoU, preventing meaningful model selection.",
                    valSplitLabel, validationSplitSpinner);
            valSplitLabel.visibleProperty().bind(advancedMode);
            valSplitLabel.managedProperty().bind(advancedMode);
            validationSplitSpinner.visibleProperty().bind(advancedMode);
            validationSplitSpinner.managedProperty().bind(advancedMode);

            grid.add(valSplitLabel, 0, row);
            grid.add(validationSplitSpinner, 1, row);
            row++;

            // Record user-typed values so we can restore them if an
            // Auto-Distribute or manual role assignment later forces the
            // spinner to show an observed value and disable it.
            validationSplitSpinner.valueProperty().addListener((obs, oldVal, newVal) -> {
                if (!updatingSpinnerProgrammatically && newVal != null) {
                    lastUserValidationSplitPct = newVal;
                }
            });

            // Companion label that shows train/val counts derived from
            // per-image roles, and explains why the spinner is disabled
            // when all selected images have fixed roles.
            validationSplitObservedLabel = new Label();
            validationSplitObservedLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 10px;");
            validationSplitObservedLabel.setWrapText(true);
            validationSplitObservedLabel.setMaxWidth(280);
            validationSplitObservedLabel.setVisible(false);
            validationSplitObservedLabel.setManaged(false);
            validationSplitObservedLabel.visibleProperty().bind(advancedMode.and(
                    validationSplitObservedLabel.textProperty().isNotEmpty()));
            validationSplitObservedLabel.managedProperty().bind(advancedMode.and(
                    validationSplitObservedLabel.textProperty().isNotEmpty()));
            grid.add(validationSplitObservedLabel, 1, row, 2, 1);
            row++;

            // Tile size
            tileSizeSpinner = new Spinner<>(64, 1024, DLClassifierPreferences.getTileSize(), 64);
            tileSizeSpinner.setEditable(true);
            tileSizeSpinner.setPrefWidth(100);
            Label tileSizeLabel = new Label("Tile Size:");
            TooltipHelper.install(
                    "Size of square patches extracted from annotations for training.\n" +
                    "Must be divisible by 32 (encoder downsampling requirement).\n" +
                    "Larger tiles capture more context but use more memory.\n\n" +
                    "256: Good for cell-level features. Faster training.\n" +
                    "512: Good balance of context and memory. Recommended default.\n" +
                    "1024: Maximum context but requires large GPU VRAM.",
                    tileSizeLabel, tileSizeSpinner);

            // Whole-image checkbox
            wholeImageCheck = new CheckBox("Whole image\n(small images only)");
            wholeImageCheck.setWrapText(true);
            wholeImageCheck.setStyle("-fx-text-fill: #CC7A00; -fx-font-weight: bold;");
            TooltipHelper.install(wholeImageCheck,
                    "Use the entire image as a single training tile.\n" +
                    "Disables tile size, overlap, and context scale controls --\n" +
                    "each image becomes one training sample.\n" +
                    "Downsample remains unlocked so you can adjust resolution\n" +
                    "to fit within the architecture's max tile size.\n\n" +
                    "Use only for small images where tiling is unnecessary.\n" +
                    "The effective tile size is computed from image dimensions\n" +
                    "at export time and rounded to a multiple of 32.\n\n" +
                    "For multi-image training, the largest image dimensions\n" +
                    "across all selected images are used (smaller images are\n" +
                    "padded with unlabeled=255).");
            wholeImageCheck.selectedProperty().addListener((obs, old, checked) -> {
                // Keep controls locked if continuing from saved model
                boolean continuing = getSelectedWeightInitStrategy() ==
                        ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING;
                tileSizeSpinner.setDisable(checked || continuing);
                overlapSpinner.setDisable(checked);
                // Downsample stays enabled in whole-image mode -- the user may
                // need to increase it so the downsampled image fits within the
                // architecture's max tile size.
                downsampleCombo.setDisable(continuing);
                contextScaleCombo.setDisable(checked || continuing);
                if (checked) {
                    contextScaleCombo.setValue("None (single scale)");
                }
                updateWholeImageInfoLabel();
            });

            // Info label shown when whole-image mode is checked and architecture
            // has a tile size cap (ViT models)
            wholeImageInfoLabel = new Label();
            wholeImageInfoLabel.setStyle("-fx-text-fill: #CC7A00; -fx-font-size: 11px;");
            wholeImageInfoLabel.setWrapText(true);
            wholeImageInfoLabel.setVisible(false);
            wholeImageInfoLabel.setManaged(false);

            // Whole image checkbox: advanced only
            wholeImageCheck.visibleProperty().bind(advancedMode);
            wholeImageCheck.managedProperty().bind(advancedMode);

            grid.add(tileSizeLabel, 0, row);
            grid.add(tileSizeSpinner, 1, row);
            grid.add(wholeImageCheck, 2, row);
            row++;
            grid.add(wholeImageInfoLabel, 0, row, 3, 1);
            row++;

            // Tile size hint (basic mode only)
            Label tileHint = new Label("Pixel size of patches the model learns from.");
            tileHint.setWrapText(true);
            tileHint.setStyle("-fx-text-fill: #888; -fx-font-size: 11px;");
            tileHint.visibleProperty().bind(advancedMode.not());
            tileHint.managedProperty().bind(advancedMode.not());
            grid.add(tileHint, 0, row, 2, 1);
            row++;

            // Downsample
            downsampleCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "1x (Full resolution)",
                    "2x (Half resolution)",
                    "4x (Quarter resolution)",
                    "8x (1/8 resolution)",
                    "16x (1/16 resolution)"
            ));
            downsampleCombo.setValue(mapDownsampleToDisplay(DLClassifierPreferences.getDefaultDownsample()));
            Label resLabel = new Label("Resolution:");
            TooltipHelper.install(
                    "Controls image resolution for training.\n" +
                    "Higher downsample = more spatial context per tile but less detail.\n\n" +
                    "1x: Full resolution -- best for cell-level features.\n" +
                    "2x: Half resolution -- good for tissue structures.\n" +
                    "4x: Quarter resolution -- each 512px tile covers 2048px of tissue.\n" +
                    "8x: Low resolution -- for large-scale region classification.\n" +
                    "16x: Very low resolution -- for whole-slide macro features.\n\n" +
                    "Locked when continuing training from a saved model.\n\n" +
                    "Tip: If your data is higher resolution than the model was\n" +
                    "trained on, consider downsampling to match. For example,\n" +
                    "use 2x for 40x data when the model was trained on 20x data,\n" +
                    "so the model sees features at the expected physical scale.\n\n" +
                    "Use the Preview button to see what the model will see\n" +
                    "at the selected downsample level.\n\n" +
                    "Must match at inference time for consistent results.",
                    resLabel, downsampleCombo);

            Button previewBtn = new Button("Preview");
            previewBtn.setMinWidth(Region.USE_PREF_SIZE);
            previewBtn.setOnAction(e -> {
                QuPathViewer viewer = QuPathGUI.getInstance().getViewer();
                if (viewer == null || viewer.getImageData() == null) {
                    Dialogs.showWarningNotification("Preview",
                            "No image is currently open.");
                    return;
                }
                double ds = parseDownsample(downsampleCombo.getValue());
                previewManager = MiniViewers.createManager(viewer);
                previewManager.setDownsample(ds);

                previewStage = new Stage();
                previewStage.initOwner(QuPathGUI.getInstance().getStage());
                previewStage.setAlwaysOnTop(true);
                previewStage.setTitle(String.format("Resolution Preview (%.0fx downsample)", ds));
                Scene scene = new Scene(previewManager.getPane(), 400, 400);
                previewStage.setScene(scene);
                previewStage.setOnHiding(ev -> {
                    unlinkPreviewStages();
                    previewManager = null;
                    this.previewStage = null;
                });
                previewStage.show();
                // Position next to context preview if it's already open
                if (contextPreviewStage != null && contextPreviewStage.isShowing()) {
                    previewStage.setX(contextPreviewStage.getX() - previewStage.getWidth() - 5);
                    previewStage.setY(contextPreviewStage.getY());
                }
                linkPreviewStages();
            });
            TooltipHelper.install(previewBtn,
                    "Open a preview window showing the image at the\n" +
                    "selected downsample level. This is what the model\n" +
                    "will see during training.");

            grid.add(resLabel, 0, row);
            HBox dsBox = new HBox(8, downsampleCombo, previewBtn);
            dsBox.setAlignment(Pos.CENTER_LEFT);
            grid.add(dsBox, 1, row);
            row++;

            // Resolution info label
            resolutionInfoLabel = new Label();
            resolutionInfoLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
            resolutionInfoLabel.setWrapText(true);
            grid.add(resolutionInfoLabel, 0, row, 2, 1);
            row++;

            // Context scale
            contextScaleCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "None (single scale)",
                    "2x context",
                    "4x context (Recommended)",
                    "8x context",
                    "16x context"
            ));
            contextScaleCombo.setValue(mapContextScaleToDisplay(DLClassifierPreferences.getDefaultContextScale()));
            contextScaleLabel = new Label("Surrounding context:");
            TooltipHelper.install(
                    "Not sure? Leave at 1x for your first run. Try 4x if classification\n" +
                    "depends on surrounding tissue structure (e.g., tumor vs. stroma).\n\n" +
                    "Multi-scale context feeds the model two views of each location:\n" +
                    "the full-resolution tile for detail, plus a larger surrounding\n" +
                    "region (downsampled to the same pixel size) for spatial context.\n\n" +
                    "None: Single-scale input (current behavior).\n" +
                    "2x: Context covers 2x the area. Moderate additional context.\n" +
                    "4x: Context covers 4x the area. Good for tissue-level patterns.\n" +
                    "8x: Context covers 8x the area. For large-scale classification.\n" +
                    "16x: Context covers 16x the area. Maximum spatial context.\n\n" +
                    "When to use: classification depends on what surrounds a region,\n" +
                    "not just the region itself. For example, distinguishing tumor\n" +
                    "from stroma may require seeing the tissue architecture around\n" +
                    "each tile. Not needed when local texture is sufficient (e.g.,\n" +
                    "collagen vs. epithelium in H&E).\n\n" +
                    "Adds C extra input channels (e.g., 3ch RGB -> 6ch with context).\n" +
                    "Modest memory increase (~5-10%). Compatible with all architectures.",
                    contextScaleLabel, contextScaleCombo);

            Button contextPreviewBtn = new Button("Preview");
            contextPreviewBtn.setMinWidth(Region.USE_PREF_SIZE);
            contextPreviewBtn.setOnAction(e -> {
                QuPathViewer viewer = QuPathGUI.getInstance().getViewer();
                if (viewer == null || viewer.getImageData() == null) {
                    Dialogs.showWarningNotification("Context Preview",
                            "No image is currently open.");
                    return;
                }
                int ctxScale = parseContextScale(contextScaleCombo.getValue());
                if (ctxScale <= 1) {
                    Dialogs.showWarningNotification("Context Preview",
                            "Context scale is set to None. Select a context scale first.");
                    return;
                }
                double ds = parseDownsample(downsampleCombo.getValue()) * ctxScale;
                contextPreviewManager = MiniViewers.createManager(viewer);
                contextPreviewManager.setDownsample(ds);

                contextPreviewStage = new Stage();
                contextPreviewStage.initOwner(QuPathGUI.getInstance().getStage());
                contextPreviewStage.setAlwaysOnTop(true);
                contextPreviewStage.setTitle(String.format(
                        "Context Preview (%dx context at %.0fx downsample)", ctxScale, ds));
                Scene ctxScene = new Scene(contextPreviewManager.getPane(), 400, 400);
                contextPreviewStage.setScene(ctxScene);
                contextPreviewStage.setOnHiding(ev -> {
                    unlinkPreviewStages();
                    contextPreviewManager = null;
                    contextPreviewStage = null;
                });
                contextPreviewStage.show();
                // Position next to resolution preview if it's already open
                if (previewStage != null && previewStage.isShowing()) {
                    contextPreviewStage.setX(previewStage.getX() + previewStage.getWidth() + 5);
                    contextPreviewStage.setY(previewStage.getY());
                }
                linkPreviewStages();
            });
            TooltipHelper.install(contextPreviewBtn,
                    "Open a preview window showing what the context\n" +
                    "channel sees -- the same location but covering a\n" +
                    "wider area, downsampled to the same tile size.\n" +
                    "Compare with the Resolution Preview to see how\n" +
                    "the model receives both detail and context.");

            grid.add(contextScaleLabel, 0, row);
            HBox ctxBox = new HBox(8, contextScaleCombo, contextPreviewBtn);
            ctxBox.setAlignment(Pos.CENTER_LEFT);
            grid.add(ctxBox, 1, row);
            row++;

            // Context info label
            contextInfoLabel = new Label();
            contextInfoLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
            contextInfoLabel.setWrapText(true);
            grid.add(contextInfoLabel, 0, row, 2, 1);
            row++;

            // Wire up listeners to update spatial info when any relevant value changes
            tileSizeSpinner.valueProperty().addListener((obs, old, newVal) -> updateSpatialInfoLabels());
            downsampleCombo.valueProperty().addListener((obs, old, newVal) -> {
                updateSpatialInfoLabels();
                updateWholeImageInfoLabel();
                if (previewManager != null) {
                    double ds = parseDownsample(newVal);
                    previewManager.setDownsample(ds);
                    if (previewStage != null) {
                        previewStage.setTitle(String.format("Resolution Preview (%.0fx downsample)", ds));
                    }
                }
                updateContextPreview();
            });
            contextScaleCombo.valueProperty().addListener((obs, old, newVal) -> {
                updateSpatialInfoLabels();
                updateContextPreview();
                updateLayerFreezePanel();
            });
            // Initial update (will show pixel-only info until image is loaded)
            updateSpatialInfoLabels();

            // Wire VRAM estimation listeners to all parameters that affect GPU memory
            tileSizeSpinner.valueProperty().addListener((obs, old, newVal) -> {
                updateVramEstimate();
                updateTileAdvisory();
            });
            batchSizeSpinner.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
            contextScaleCombo.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
            architectureCombo.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
            backboneCombo.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
            // overlapSpinner listener wired after construction below

            // Wire tile/time estimate listeners -- refresh when relevant parameters change
            tileSizeSpinner.valueProperty().addListener((obs, o, n) -> {
                if (lastLoadedClassCount > 0) updateTileEstimateLabel(lastLoadedClassCount, lastLoadedImageCount);
            });
            downsampleCombo.valueProperty().addListener((obs, o, n) -> {
                if (lastLoadedClassCount > 0) updateTileEstimateLabel(lastLoadedClassCount, lastLoadedImageCount);
            });
            epochsSpinner.valueProperty().addListener((obs, o, n) -> {
                if (lastLoadedClassCount > 0) updateTileEstimateLabel(lastLoadedClassCount, lastLoadedImageCount);
            });
            batchSizeSpinner.valueProperty().addListener((obs, o, n) -> {
                if (lastLoadedClassCount > 0) updateTileEstimateLabel(lastLoadedClassCount, lastLoadedImageCount);
            });
            backboneCombo.valueProperty().addListener((obs, o, n) -> {
                if (lastLoadedClassCount > 0) updateTileEstimateLabel(lastLoadedClassCount, lastLoadedImageCount);
            });
            // Context scale affects the in-memory cache estimate (2x storage
            // when > 1) -- refresh the tile/cache label when it changes.
            contextScaleCombo.valueProperty().addListener((obs, o, n) -> {
                if (lastLoadedClassCount > 0) updateTileEstimateLabel(lastLoadedClassCount, lastLoadedImageCount);
            });
            DLClassifierPreferences.defaultInMemoryDatasetProperty().addListener((obs, o, n) -> {
                if (lastLoadedClassCount > 0) updateTileEstimateLabel(lastLoadedClassCount, lastLoadedImageCount);
            });

            // Overlap (advanced only)
            overlapSpinner = new Spinner<>(0, 50, DLClassifierPreferences.getTileOverlap(), 5);
            overlapSpinner.setEditable(true);
            overlapSpinner.setPrefWidth(100);
            Label overlapLabel = new Label("Tile Overlap (%):");
            TooltipHelper.install(
                    "Overlap between adjacent training tiles as a percentage.\n" +
                    "Higher overlap generates more training patches from\n" +
                    "the same annotations but increases extraction time.\n\n" +
                    "0%: No overlap -- fastest extraction, fewer tiles.\n" +
                    "10-25%: Typical range -- good balance of diversity and speed.\n" +
                    "Higher overlap is most beneficial with limited annotations.",
                    overlapLabel, overlapSpinner);
            overlapLabel.visibleProperty().bind(advancedMode);
            overlapLabel.managedProperty().bind(advancedMode);
            overlapSpinner.visibleProperty().bind(advancedMode);
            overlapSpinner.managedProperty().bind(advancedMode);

            grid.add(overlapLabel, 0, row);
            grid.add(overlapSpinner, 1, row);
            row++;

            overlapSpinner.valueProperty().addListener((obs, old, newVal) -> updateTileAdvisory());

            // Tile settings advisory -- updated on tileSize / overlap change.
            // Green when in the sensible range, orange when likely suboptimal,
            // red when the requested overlap would force stride<=0. This is
            // advice; the dialog still allows the user to proceed.
            tileAdvisoryLabel = new Label();
            tileAdvisoryLabel.setWrapText(true);
            tileAdvisoryLabel.setStyle("-fx-font-size: 11px;");
            tileAdvisoryLabel.setVisible(false);
            tileAdvisoryLabel.setManaged(false);
            grid.add(tileAdvisoryLabel, 0, row, 3, 1);
            row++;

            // Line stroke width - restore from preferences, or fall back to QuPath's stroke thickness
            // In basic mode: only shown when line annotations are detected in selected images.
            // In advanced mode: always shown.
            int savedStroke = DLClassifierPreferences.getLastLineStrokeWidth();
            if (savedStroke <= 0) {
                try {
                    savedStroke = (int) Math.max(1, PathPrefs.annotationStrokeThicknessProperty().get());
                } catch (Exception e) {
                    savedStroke = 5;
                    logger.debug("Could not read QuPath annotation stroke thickness, using default");
                }
            }
            lineStrokeWidthSpinner = new Spinner<>(1, 50, savedStroke, 1);
            lineStrokeWidthSpinner.setEditable(true);
            lineStrokeWidthSpinner.setPrefWidth(100);
            lineStrokeLabel = new Label("Line Stroke Width:");
            TooltipHelper.install(
                    "Width in pixels for rendering line/polyline annotations as training masks.\n" +
                    "Pre-filled from QuPath's annotation stroke thickness.\n\n" +
                    "Thin strokes (<5px) produce sparse training signal from polyline\n" +
                    "annotations -- consider increasing for better training.\n" +
                    "Only affects line/polyline annotations; area annotations are\n" +
                    "filled completely regardless of this setting.",
                    lineStrokeLabel, lineStrokeWidthSpinner);

            grid.add(lineStrokeLabel, 0, row);
            grid.add(lineStrokeWidthSpinner, 1, row);

            TitledPane pane = new TitledPane("TRAINING PARAMETERS", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Configure training hyperparameters and tile extraction settings"));
            return pane;
        }

        private TitledPane createTrainingStrategySection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // LR Scheduler
            schedulerCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "One Cycle", "Cosine Annealing", "Reduce on Plateau", "Step Decay", "None"));
            schedulerCombo.setValue(mapSchedulerToDisplay(DLClassifierPreferences.getDefaultScheduler()));
            Label schedulerLabel = new Label("LR Scheduler:");
            TooltipHelper.installWithLink(
                    "Learning rate schedule during training:\n\n" +
                    "One Cycle (recommended): Smooth ramp-up then decay.\n" +
                    "  Auto-runs LR finder to choose max learning rate.\n\n" +
                    "Cosine Annealing: Periodic warm restarts.\n" +
                    "  Can escape local minima but may cause LR spikes.\n\n" +
                    "Reduce on Plateau: Halves LR when metric stops improving.\n" +
                    "  Adapts to training dynamics. Good for long runs.\n\n" +
                    "Step Decay: Reduce LR by factor every N epochs.\n" +
                    "  Predictable but requires manual tuning of step schedule.\n\n" +
                    "None: Constant learning rate throughout training.",
                    "https://pytorch.org/docs/stable/optim.html",
                    schedulerLabel, schedulerCombo);

            grid.add(schedulerLabel, 0, row);
            grid.add(schedulerCombo, 1, row);
            row++;

            // Scheduler behavior description -- updates when selection changes
            Label schedulerDesc = new Label();
            schedulerDesc.setWrapText(true);
            schedulerDesc.setStyle("-fx-text-fill: #666666; -fx-font-size: 0.9em;");
            schedulerDesc.setMaxWidth(Double.MAX_VALUE);
            Runnable updateSchedulerDesc = () -> {
                String sel = schedulerCombo.getValue();
                if (sel == null) sel = "";
                int epochs = epochsSpinner.getValue();
                int peakEpoch = Math.max(1, (int) (epochs * 0.3));
                switch (sel) {
                    case "One Cycle" -> schedulerDesc.setText(String.format(
                            "LR ramps up to a peak (auto-found) at ~epoch %d of %d, "
                            + "then cosine-decays to near zero. Expect validation "
                            + "volatility during ramp-up (epochs 1-%d). Best "
                            + "results emerge during the decay phase (epochs %d-%d).",
                            peakEpoch, epochs, peakEpoch, peakEpoch, epochs));
                    case "Reduce on Plateau" -> schedulerDesc.setText(String.format(
                            "Starts at your set learning rate and halves it when "
                            + "validation stops improving. Stable from epoch 1 -- "
                            + "no ramp-up volatility. May converge to a slightly "
                            + "worse optimum than One Cycle. With %d epochs, "
                            + "expect 3-5 LR reductions.",
                            epochs));
                    case "Cosine Annealing" -> schedulerDesc.setText(
                            "Periodically decays LR to zero then warm-restarts. "
                            + "Can help escape local minima but causes periodic "
                            + "LR spikes that may destabilize minority classes.");
                    case "Step Decay" -> schedulerDesc.setText(
                            "Reduces LR by a fixed factor every N epochs. "
                            + "Predictable but requires manual tuning of the "
                            + "step schedule.");
                    case "None" -> schedulerDesc.setText(
                            "Constant learning rate throughout training. Only "
                            + "useful for short experiments or debugging.");
                    default -> schedulerDesc.setText("");
                }
            };
            schedulerCombo.valueProperty().addListener((obs, old, val) -> {
                updateSchedulerDesc.run();
                updateLrInfoLabel();
            });
            epochsSpinner.valueProperty().addListener((obs, old, val) -> updateSchedulerDesc.run());
            updateSchedulerDesc.run();
            grid.add(schedulerDesc, 0, row, 2, 1);
            row++;

            // Loss function
            lossFunctionCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "Cross Entropy + Dice", "Cross Entropy",
                    "Focal + Dice", "Focal",
                    "Boundary-softened CE", "Boundary-softened CE + Dice",
                    "Lovasz-Softmax", "CE + Lovasz-Softmax"));
            lossFunctionCombo.setValue(mapLossFunctionToDisplay(DLClassifierPreferences.getDefaultLossFunction()));
            Label lossLabel = new Label("Loss Function:");
            TooltipHelper.installWithLink(
                    "Loss function for training:\n\n" +
                    "CE + Dice (recommended): Combines per-pixel Cross Entropy with\n" +
                    "  region overlap Dice loss. Modern standard for segmentation.\n\n" +
                    "Cross Entropy: Per-pixel classification loss only.\n\n" +
                    "Focal + Dice: Focal loss down-weights easy pixels via\n" +
                    "  (1-p)^gamma, combined with Dice. Best when some regions\n" +
                    "  are much harder than others (e.g., small structures).\n\n" +
                    "Focal: Focal loss only (no Dice component).\n\n" +
                    "Boundary-softened CE: CE with per-pixel distance-transform\n" +
                    "  weighting. Down-weights pixels near class boundaries in\n" +
                    "  the annotation. Use when manual annotations have noisy /\n" +
                    "  imprecise edges. Reference: inverse of Ronneberger 2015\n" +
                    "  boundary weighting.\n\n" +
                    "Boundary-softened CE + Dice: combines the above with Dice,\n" +
                    "  recommended when edge noise is the main error source.\n\n" +
                    "Lovasz-Softmax: directly optimises mean IoU. Best used\n" +
                    "  after a CE warmup (or as CE + Lovasz). No hyperparameters.\n" +
                    "  Berman et al., CVPR 2018 (arXiv:1705.08790).\n\n" +
                    "CE + Lovasz-Softmax: CE provides stable early gradient,\n" +
                    "  Lovasz pushes directly toward IoU. OHEM is disabled\n" +
                    "  for the two Lovasz variants (Lovasz is a sorted-\n" +
                    "  errors surrogate, not a per-pixel loss).\n\n" +
                    "OHEM composes with all other variants:\n" +
                    "- With CE / CE+Dice: top-K of the per-pixel CE.\n" +
                    "- With Focal / Focal+Dice: the focal modulation\n" +
                    "  (1-p_t)^gamma is applied BEFORE the top-K sort\n" +
                    "  (OHEMFocalLoss), so focal_gamma is preserved.\n" +
                    "- With Boundary-softened CE: the boundary weight\n" +
                    "  map is applied BEFORE top-K, pushing edge\n" +
                    "  annotation noise out of the hard set and\n" +
                    "  focusing OHEM capacity on interior errors.\n\n" +
                    "Class weights apply everywhere: CE, Focal, Boundary-\n" +
                    "softened CE, and Lovasz-Softmax (all variants).",
                    "https://smp.readthedocs.io/en/latest/losses.html",
                    lossLabel, lossFunctionCombo);

            grid.add(lossLabel, 0, row);
            grid.add(lossFunctionCombo, 1, row);
            row++;

            // Focal gamma (visible only when focal variant selected)
            focalGammaLabel = new Label("Focal Gamma:");
            focalGammaSpinner = new Spinner<>(
                    new SpinnerValueFactory.DoubleSpinnerValueFactory(
                            0.5, 5.0, DLClassifierPreferences.getDefaultFocalGamma(), 0.5));
            focalGammaSpinner.setEditable(true);
            TooltipHelper.install(
                    "Focal loss focusing parameter (gamma). Down-weights\n" +
                    "easy pixels so the model pays more attention to mistakes.\n\n" +
                    "  gamma=0: equivalent to standard Cross Entropy\n" +
                    "  gamma=1: mild focusing\n" +
                    "  gamma=2: standard (recommended)\n" +
                    "  gamma=3-5: aggressive focusing for very hard regions\n\n" +
                    "When to use: your classes have very different difficulty\n" +
                    "levels (e.g., small structures surrounded by large easy\n" +
                    "regions). Unlike OHEM, focal loss keeps ALL pixels but\n" +
                    "gradually reduces the weight of confident predictions.\n" +
                    "Good first step before trying the more aggressive OHEM.",
                    focalGammaLabel, focalGammaSpinner);
            boolean focalSelected = isFocalLossSelected(lossFunctionCombo.getValue());
            focalGammaLabel.setVisible(focalSelected);
            focalGammaLabel.setManaged(focalSelected);
            focalGammaSpinner.setVisible(focalSelected);
            focalGammaSpinner.setManaged(focalSelected);
            grid.add(focalGammaLabel, 0, row);
            grid.add(focalGammaSpinner, 1, row);
            row++;

            // Boundary-softening params (visible only when a boundary_ce*
            // variant is selected). sigma controls the EDT falloff length;
            // w_min is the floor weight applied at exact boundaries.
            boundarySigmaLabel = new Label("Boundary Sigma (px):");
            boundarySigmaSpinner = new Spinner<>(
                    new SpinnerValueFactory.DoubleSpinnerValueFactory(
                            0.5, 32.0, DLClassifierPreferences.getDefaultBoundarySigma(), 0.5));
            boundarySigmaSpinner.setEditable(true);
            TooltipHelper.install(
                    "Boundary-softening falloff length in pixels.\n\n" +
                    "Each pixel's CE loss is weighted by\n" +
                    "  w = w_min + (1 - w_min) * (1 - exp(-d / sigma))\n" +
                    "where d is the Euclidean distance to the nearest\n" +
                    "annotation boundary. Larger sigma = wider soft band.\n\n" +
                    "  sigma=1-2 px: very tight band (strict).\n" +
                    "  sigma=3 px (default): matches typical annotator\n" +
                    "    jitter on 256-px tiles.\n" +
                    "  sigma=5-10 px: aggressive softening; try when edge\n" +
                    "    noise is severe (e.g. polygon-traced annotations).\n\n" +
                    "Tile size divided by 2^depth (for Tiny UNet depth=4, that's\n" +
                    "16 px) is a reasonable upper bound -- beyond that most\n" +
                    "pixels are 'near' a boundary and the weighting does nothing.",
                    boundarySigmaLabel, boundarySigmaSpinner);

            boundaryWMinLabel = new Label("Boundary Floor Weight:");
            boundaryWMinSpinner = new Spinner<>(
                    new SpinnerValueFactory.DoubleSpinnerValueFactory(
                            0.0, 1.0, DLClassifierPreferences.getDefaultBoundaryWMin(), 0.05));
            boundaryWMinSpinner.setEditable(true);
            TooltipHelper.install(
                    "Minimum weight applied at an exact annotation boundary.\n\n" +
                    "  w_min=0.0: edges contribute no gradient at all\n" +
                    "    (equivalent to setting the annotation boundary\n" +
                    "    as an ignore band).\n" +
                    "  w_min=0.1 (default): edges contribute 10% of\n" +
                    "    interior gradient. Safe starting point.\n" +
                    "  w_min=1.0: no softening (equivalent to plain CE).\n\n" +
                    "Pair with sigma to shape the curve: (sigma=3, w_min=0.1)\n" +
                    "means boundary pixels get 10% weight, recovering to ~63%\n" +
                    "at distance = sigma and ~95% at distance = 3 * sigma.",
                    boundaryWMinLabel, boundaryWMinSpinner);

            boolean boundarySelected = isBoundaryLossSelected(lossFunctionCombo.getValue());
            boundarySigmaLabel.setVisible(boundarySelected);
            boundarySigmaLabel.setManaged(boundarySelected);
            boundarySigmaSpinner.setVisible(boundarySelected);
            boundarySigmaSpinner.setManaged(boundarySelected);
            boundaryWMinLabel.setVisible(boundarySelected);
            boundaryWMinLabel.setManaged(boundarySelected);
            boundaryWMinSpinner.setVisible(boundarySelected);
            boundaryWMinSpinner.setManaged(boundarySelected);
            grid.add(boundarySigmaLabel, 0, row);
            grid.add(boundarySigmaSpinner, 1, row);
            row++;
            grid.add(boundaryWMinLabel, 0, row);
            grid.add(boundaryWMinSpinner, 1, row);
            row++;

            // Single listener handles both conditional parameter groups.
            lossFunctionCombo.valueProperty().addListener((obs, oldVal, newVal) -> {
                boolean showFocal = isFocalLossSelected(newVal);
                focalGammaLabel.setVisible(showFocal);
                focalGammaLabel.setManaged(showFocal);
                focalGammaSpinner.setVisible(showFocal);
                focalGammaSpinner.setManaged(showFocal);
                boolean showBoundary = isBoundaryLossSelected(newVal);
                boundarySigmaLabel.setVisible(showBoundary);
                boundarySigmaLabel.setManaged(showBoundary);
                boundarySigmaSpinner.setVisible(showBoundary);
                boundarySigmaSpinner.setManaged(showBoundary);
                boundaryWMinLabel.setVisible(showBoundary);
                boundaryWMinLabel.setManaged(showBoundary);
                boundaryWMinSpinner.setVisible(showBoundary);
                boundaryWMinSpinner.setManaged(showBoundary);
            });

            // OHEM hard pixel % (END / target value).
            // Minimum is 5% to match the tooltip's "5% = very aggressive"
            // guidance; the previous 10% floor blocked the tooltip's own
            // recommendation.
            ohemSpinner = new Spinner<>(
                    new SpinnerValueFactory.IntegerSpinnerValueFactory(
                            5, 100, DLClassifierPreferences.getDefaultOhemHardPixelPct(), 5));
            ohemSpinner.setEditable(true);
            Label ohemLabel = new Label("Hard Pixel End %:");
            TooltipHelper.install(
                    "Online Hard Example Mining (OHEM): each batch, keep only the\n" +
                    "hardest N% of pixels and ignore the rest. Resets every batch\n" +
                    "(not cumulative across epochs).\n\n" +
                    "This value is the END / target ratio training converges to.\n" +
                    "Combined with Hard Pixel Start %, you can anneal from a\n" +
                    "wide start to a narrow end, or just hold a fixed ratio.\n\n" +
                    "100% = all pixels (no OHEM).\n" +
                    "25% = keep only the hardest quarter -- aggressive.\n" +
                    "5% = very aggressive, focuses almost entirely on mistakes.\n\n" +
                    "When to use: your images have large uniform regions (e.g.,\n" +
                    "background, empty glass) that the model learns quickly. Without\n" +
                    "OHEM, easy pixels dominate training and the model spends most\n" +
                    "of its time on regions it already classifies correctly. With\n" +
                    "OHEM, training focuses on boundaries and confusing regions\n" +
                    "where accuracy actually needs improvement.\n\n" +
                    "Tip: try Focal loss first as a softer alternative -- it\n" +
                    "down-weights easy pixels rather than completely ignoring them.\n" +
                    "Note: only affects the pixel-loss component; Dice loss still\n" +
                    "uses all pixels so the model doesn't forget easy classes.",
                    ohemLabel, ohemSpinner);
            grid.add(ohemLabel, 0, row);
            grid.add(ohemSpinner, 1, row);
            row++;

            // OHEM hard pixel % (START value). Visible only when OHEM is active.
            // When start > end, the ratio anneals linearly from start to end
            // over the first 75% of epochs. When start == end, the ratio is
            // constant.
            int defaultStartPct = DLClassifierPreferences.getDefaultOhemHardPixelStartPct();
            // Guard: start must be >= end. If the saved preference is below
            // the current end, clamp it up so the spinner is valid.
            if (defaultStartPct < ohemSpinner.getValue()) {
                defaultStartPct = ohemSpinner.getValue();
            }
            ohemStartSpinner = new Spinner<>(
                    new SpinnerValueFactory.IntegerSpinnerValueFactory(
                            5, 100, defaultStartPct, 5));
            ohemStartSpinner.setEditable(true);
            Label ohemStartLabel = new Label("Hard Pixel Start %:");
            TooltipHelper.install(
                    "Hard Pixel % used at the start of training. Combined with\n" +
                    "Hard Pixel End %, controls whether OHEM anneals or stays fixed.\n\n" +
                    "Start > End: Linearly anneal from Start to End over the first\n" +
                    "  75% of epochs. Lets the model learn basic class distributions\n" +
                    "  from all pixels early, then gradually shift focus to hard\n" +
                    "  cases (boundaries, confusing regions).\n\n" +
                    "Start == End: Fixed at that value throughout training.\n" +
                    "  Use this when continuing training from an existing model\n" +
                    "  that has already learned the easy pixels -- skip the warm-up\n" +
                    "  and go straight to mining hard examples.\n\n" +
                    "Must be >= Hard Pixel End %.",
                    ohemStartLabel, ohemStartSpinner);
            // Hide the schedule combo -- behavior is now derived from start vs end.
            ohemScheduleLabel = new Label("Hard Pixel Schedule:");
            ohemScheduleCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "Fixed", "Anneal"));
            ohemScheduleLabel.setVisible(false);
            ohemScheduleLabel.setManaged(false);
            ohemScheduleCombo.setVisible(false);
            ohemScheduleCombo.setManaged(false);

            boolean ohemActive = ohemSpinner.getValue() < 100;
            ohemStartLabel.setVisible(ohemActive);
            ohemStartLabel.setManaged(ohemActive);
            ohemStartSpinner.setVisible(ohemActive);
            ohemStartSpinner.setManaged(ohemActive);
            ohemSpinner.valueProperty().addListener((obs, old, val) -> {
                boolean active = val != null && val < 100;
                ohemStartLabel.setVisible(active);
                ohemStartLabel.setManaged(active);
                ohemStartSpinner.setVisible(active);
                ohemStartSpinner.setManaged(active);
                // Enforce start >= end whenever end changes.
                if (val != null && ohemStartSpinner.getValue() < val) {
                    ohemStartSpinner.getValueFactory().setValue(val);
                }
            });
            ohemStartSpinner.valueProperty().addListener((obs, old, val) -> {
                // Enforce start >= end whenever start changes.
                if (val != null && val < ohemSpinner.getValue()) {
                    ohemStartSpinner.getValueFactory().setValue(ohemSpinner.getValue());
                }
            });
            grid.add(ohemStartLabel, 0, row);
            grid.add(ohemStartSpinner, 1, row);
            row++;

            // OHEM pixel-selection strategy. Visible only when OHEM is active.
            ohemAdaptiveFloorCheck = new CheckBox("Adaptive per-class floor");
            ohemAdaptiveFloorCheck.setSelected(
                    DLClassifierPreferences.isDefaultOhemAdaptiveFloor());
            TooltipHelper.install(
                    "Which pixels OHEM keeps when selecting the hardest N%.\n\n" +
                    "UNCHECKED (legacy, per-class proportional):\n" +
                    "  For each class, keep the hardest (hard_ratio * class_count)\n" +
                    "  pixels. The rare class's share of the selected loss matches\n" +
                    "  its natural fraction of the batch -- so if rare is 5% of\n" +
                    "  pixels it stays 5% of the loss. Simple and stable, but\n" +
                    "  weak when class imbalance is extreme.\n\n" +
                    "CHECKED (adaptive, global topk + per-class floor):\n" +
                    "  1. Select the hardest N% globally, ignoring class.\n" +
                    "  2. For every class present in the batch, top it up to at\n" +
                    "     least floor = min(class_count, max(32, k / (2 * num_classes)))\n" +
                    "     with that class's hardest remaining pixels.\n" +
                    "  Lets hard pixels naturally concentrate on whichever class\n" +
                    "  the model is currently failing (usually the minority),\n" +
                    "  while guaranteeing no present class drops to zero pixels.\n\n" +
                    "Try this when rare classes stagnate while majority classes\n" +
                    "are already near IoU 1.0 and OHEM doesn't seem to help.",
                    ohemAdaptiveFloorCheck);

            boolean ohemActiveNow = ohemSpinner.getValue() < 100;
            ohemAdaptiveFloorCheck.setVisible(ohemActiveNow);
            ohemAdaptiveFloorCheck.setManaged(ohemActiveNow);
            ohemSpinner.valueProperty().addListener((obs, old, val) -> {
                boolean active = val != null && val < 100;
                ohemAdaptiveFloorCheck.setVisible(active);
                ohemAdaptiveFloorCheck.setManaged(active);
            });
            grid.add(ohemAdaptiveFloorCheck, 0, row, 2, 1);
            row++;

            // Early stopping metric
            earlyStoppingMetricCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "Mean IoU", "Validation Loss", "Disabled"));
            earlyStoppingMetricCombo.setValue(
                    mapEarlyStoppingMetricToDisplay(DLClassifierPreferences.getDefaultEarlyStoppingMetric()));
            Label esMetricLabel = new Label("Early Stop Metric:");
            TooltipHelper.install(
                    "Which metric decides WHEN TO STOP training.\n" +
                    "Independent of Focus Class: Focus Class drives 'best\n" +
                    "model' checkpoint selection, while this metric drives\n" +
                    "early stopping and the plateau LR scheduler.\n\n" +
                    "Mean IoU (recommended): Intersection-over-union averaged\n" +
                    "  across all classes. Directly measures segmentation\n" +
                    "  quality. Stops when overall accuracy plateaus.\n\n" +
                    "Validation Loss: Combined loss on held-out data.\n" +
                    "  Can oscillate while IoU still improves, so Mean IoU\n" +
                    "  is generally more reliable.\n\n" +
                    "Disabled: Train for the full epoch count regardless of\n" +
                    "  metric progress. Best-model selection still applies.",
                    esMetricLabel, earlyStoppingMetricCombo);

            grid.add(esMetricLabel, 0, row);
            grid.add(earlyStoppingMetricCombo, 1, row);
            row++;

            // Early stopping patience
            earlyStoppingPatienceSpinner = new Spinner<>(3, 50,
                    DLClassifierPreferences.getDefaultEarlyStoppingPatience(), 1);
            earlyStoppingPatienceSpinner.setEditable(true);
            earlyStoppingPatienceSpinner.setPrefWidth(100);
            Label esPatienceLabel = new Label("Early Stop Patience:");
            TooltipHelper.install(
                    "How many consecutive epochs without improvement before stopping.\n\n" +
                    "After each epoch, if the early stop metric hasn't improved in\n" +
                    "this many epochs, training stops and the best model is saved.\n\n" +
                    "10-15: Good default for most runs.\n" +
                    "20-30: Use with cosine annealing or noisy loss curves.\n" +
                    "3-5: For quick experiments.\n\n" +
                    "It is safe to set a high epoch count (e.g. 200) and rely on\n" +
                    "patience to stop training -- you won't waste GPU time.",
                    esPatienceLabel, earlyStoppingPatienceSpinner);

            grid.add(esPatienceLabel, 0, row);
            grid.add(earlyStoppingPatienceSpinner, 1, row);
            row++;

            // Focus class
            focusClassCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "None (use Mean IoU)"));
            focusClassCombo.setValue("None (use Mean IoU)");
            Label focusClassLabel = new Label("Focus Class:");
            TooltipHelper.install(
                    "Optionally select a class to focus on for BEST-MODEL\n" +
                    "checkpoint selection.\n\n" +
                    "When set, the focus class's per-class IoU decides which\n" +
                    "epoch's weights are saved as the final model. The Early\n" +
                    "Stop Metric (above) is INDEPENDENT -- it controls when\n" +
                    "training stops based on its own metric (Mean IoU or\n" +
                    "Validation Loss).\n\n" +
                    "Combine them: e.g., Focus Class = 'vein' + Early Stop\n" +
                    "Metric = 'Mean IoU' saves the best vein model but lets\n" +
                    "training run until overall mean IoU plateaus.\n\n" +
                    "Use this when you care more about one class than the\n" +
                    "others. For example, if detecting 'Hinge' is critical,\n" +
                    "set it as the focus class so the best model is the one\n" +
                    "with the best Hinge IoU, not the best average across\n" +
                    "all classes.",
                    focusClassLabel, focusClassCombo);

            focusClassCombo.valueProperty().addListener((obs, old, newVal) -> {
                boolean hasFocusClass = newVal != null && !newVal.startsWith("None");
                // Show/hide min IoU row
                focusClassMinIoUSpinner.setVisible(hasFocusClass);
                focusClassMinIoUSpinner.setManaged(hasFocusClass);
                focusClassMinIoULabel.setVisible(hasFocusClass);
                focusClassMinIoULabel.setManaged(hasFocusClass);
                // Focus Class and Early Stop Metric are independent: focus
                // class drives best-model selection; ES metric drives when
                // training stops. Both can be set without conflict.
            });

            grid.add(focusClassLabel, 0, row);
            grid.add(focusClassCombo, 1, row);
            row++;

            // Focus class min IoU threshold (hidden by default)
            focusClassMinIoULabel = new Label("Min Focus IoU:");
            double savedMinIoU = DLClassifierPreferences.getDefaultFocusClassMinIoU();
            focusClassMinIoUSpinner = new Spinner<>(0.0, 1.0,
                    savedMinIoU > 0 ? savedMinIoU : 0.5, 0.05);
            focusClassMinIoUSpinner.setEditable(true);
            focusClassMinIoUSpinner.setPrefWidth(100);
            var minIoUFactory = (SpinnerValueFactory.DoubleSpinnerValueFactory) focusClassMinIoUSpinner.getValueFactory();
            minIoUFactory.setConverter(new javafx.util.StringConverter<Double>() {
                @Override
                public String toString(Double value) {
                    return value == null ? "" : String.format("%.2f", value);
                }
                @Override
                public Double fromString(String string) {
                    try {
                        return Double.parseDouble(string.trim());
                    } catch (NumberFormatException e) {
                        return minIoUFactory.getValue();
                    }
                }
            });
            TooltipHelper.install(
                    "Minimum IoU threshold for the focus class.\n\n" +
                    "Training will not stop early until the focus class\n" +
                    "reaches this IoU, regardless of patience.\n\n" +
                    "0.00: No minimum -- early stopping works normally.\n" +
                    "0.30: Training continues until focus class IoU >= 0.30,\n" +
                    "  then patience-based stopping resumes.\n\n" +
                    "Set this to prevent the model from stopping before the\n" +
                    "focus class has had a chance to learn.",
                    focusClassMinIoULabel, focusClassMinIoUSpinner);

            // Hidden by default
            focusClassMinIoUSpinner.setVisible(false);
            focusClassMinIoUSpinner.setManaged(false);
            focusClassMinIoULabel.setVisible(false);
            focusClassMinIoULabel.setManaged(false);

            grid.add(focusClassMinIoULabel, 0, row);
            grid.add(focusClassMinIoUSpinner, 1, row);
            row++;

            // Mixed precision
            mixedPrecisionCheck = new CheckBox("Enable mixed precision (AMP)");
            mixedPrecisionCheck.setSelected(DLClassifierPreferences.isDefaultMixedPrecision());
            TooltipHelper.installWithLink(mixedPrecisionCheck,
                    "Use automatic mixed precision (FP16/FP32) on CUDA GPUs.\n" +
                    "Typically provides ~2x speedup with no accuracy loss.\n" +
                    "Only active when training on NVIDIA GPUs; ignored on CPU/MPS.\n\n" +
                    "Safe to leave enabled -- PyTorch automatically manages which\n" +
                    "operations use FP16 vs FP32 for numerical stability.",
                    "https://pytorch.org/docs/stable/amp.html");

            mixedPrecisionCheck.selectedProperty().addListener((obs, old, newVal) -> updateVramEstimate());
            grid.add(mixedPrecisionCheck, 0, row, 2, 1);
            row++;

            // Gradient accumulation
            gradientAccumulationSpinner = new Spinner<>(1, 8,
                    DLClassifierPreferences.getDefaultGradientAccumulation(), 1);
            gradientAccumulationSpinner.setEditable(true);
            gradientAccumulationSpinner.setPrefWidth(100);
            Label gradAccLabel = new Label("Gradient Accumulation:");
            TooltipHelper.install(
                    "Accumulate gradients over N batches before updating weights.\n\n" +
                    "Effective batch size = Batch Size x Accumulation Steps.\n" +
                    "Use this when GPU memory is too limited for large batches.\n\n" +
                    "1: Normal training (no accumulation).\n" +
                    "2-4: Simulates 2-4x larger batch without extra memory.\n" +
                    "8: Maximum accumulation; very stable but slower per epoch.",
                    gradAccLabel, gradientAccumulationSpinner);

            grid.add(gradAccLabel, 0, row);
            grid.add(gradientAccumulationSpinner, 1, row);
            row++;

            // Progressive resizing
            progressiveResizeCheck = new CheckBox("Progressive resizing");
            progressiveResizeCheck.setSelected(DLClassifierPreferences.isDefaultProgressiveResize());
            TooltipHelper.install(progressiveResizeCheck,
                    "Train at half tile resolution for the first 40% of epochs,\n" +
                    "then switch to full resolution.\n\n" +
                    "Benefits:\n" +
                    "- Faster early training (4x fewer pixels)\n" +
                    "- Acts as regularization (prevents overfitting)\n" +
                    "- Helps model learn coarse features first\n\n" +
                    "When to use: you have limited training data and the model\n" +
                    "overfits before learning good features. Also helpful for\n" +
                    "long training runs where you want faster initial epochs.\n" +
                    "Leave unchecked for standard training or when you have\n" +
                    "plenty of annotated data.");

            grid.add(progressiveResizeCheck, 0, row, 2, 1);
            row++;

            // Fused optimizer: one-kernel AdamW update, saves 2-5 ms/step on tiny models.
            fusedOptimizerCheck = new CheckBox("Fused optimizer (CUDA only)");
            fusedOptimizerCheck.setSelected(true);
            TooltipHelper.install(fusedOptimizerCheck,
                    "Use PyTorch's fused AdamW implementation on NVIDIA GPUs.\n\n" +
                    "Benefits:\n" +
                    "- Single CUDA kernel for the full param update (2-5 ms/step saved)\n" +
                    "- No accuracy change; same math as the standard implementation\n\n" +
                    "Safe to leave enabled. Ignored on CPU, MPS, and older PyTorch (<2.0).\n" +
                    "Disable only if you hit a CUDA kernel error that mentions 'fused'.");
            grid.add(fusedOptimizerCheck, 0, row, 2, 1);
            row++;

            // Auto-find learning rate (LR Finder presweep toggle)
            useLrFinderCheck = new CheckBox("Auto-find learning rate (LR Finder)");
            useLrFinderCheck.setSelected(true);
            TooltipHelper.install(useLrFinderCheck,
                    "Run a 100-iteration LR Finder presweep before training to pick\n" +
                    "a good OneCycleLR peak learning rate.\n\n" +
                    "When to disable:\n" +
                    "- Training a tiny model where the presweep is a big fraction of\n" +
                    "  total training time\n" +
                    "- You already know a good learning rate for this task\n\n" +
                    "When disabled, max_lr = base_lr * sqrt(batch_size / 8) is used.\n" +
                    "Only affects training when the scheduler is OneCycleLR.");
            grid.add(useLrFinderCheck, 0, row, 2, 1);
            row++;

            // GPU augmentation via kornia (experimental, opt-in).
            gpuAugmentationCheck = new CheckBox("GPU augmentation (experimental, CUDA only)");
            gpuAugmentationCheck.setSelected(false);
            TooltipHelper.installWithLink(gpuAugmentationCheck,
                    "Run data augmentation on the GPU via kornia instead of on\n" +
                    "the CPU via albumentations. When the in-memory dataset\n" +
                    "preload is active, CPU augmentation is the dominant cost\n" +
                    "per batch; moving it to the GPU can speed up an epoch\n" +
                    "10-20x on small models.\n\n" +
                    "Covered augmentations:\n" +
                    "- Horizontal / vertical flip, 90 deg rotation\n" +
                    "- Color jitter (brightfield) or brightness/contrast (other)\n" +
                    "- Low-probability Gaussian noise and blur\n\n" +
                    "Skipped on the GPU path (still available via CPU):\n" +
                    "- Elastic transform, grid distortion, arbitrary-angle rotation\n\n" +
                    "Safe to leave off. Silently falls back to CPU albumentations\n" +
                    "when kornia is not installed or the device is not CUDA.",
                    "https://kornia.readthedocs.io/en/latest/augmentation.html");
            grid.add(gpuAugmentationCheck, 0, row, 2, 1);
            row++;

            // torch.compile at training (experimental, Linux+CUDA only).
            // Disable the checkbox on non-Linux hosts: triton is the Inductor
            // backend for GPU kernel generation and does not install cleanly on
            // Windows or macOS today, so even if the user toggled it the Python
            // side would log "torch.compile is Linux-gated" and fall back. Grey
            // the box + append a hint so that gate is visible in the UI.
            boolean isLinux = System.getProperty("os.name", "")
                    .toLowerCase().contains("linux");
            String compileLabel = isLinux
                    ? "torch.compile (experimental, Linux+CUDA)"
                    : "torch.compile (Linux-only, disabled on this OS)";
            useTorchCompileCheck = new CheckBox(compileLabel);
            useTorchCompileCheck.setSelected(false);
            useTorchCompileCheck.setDisable(!isLinux);
            TooltipHelper.installWithLink(useTorchCompileCheck,
                    "Wrap the training model with torch.compile(mode=\"reduce-overhead\")\n" +
                    "for kernel fusion and CUDA graph capture. On tiny models this is\n" +
                    "usually a 1.4-2x steady-state speedup after the compile cost\n" +
                    "(~15-40 s on the first iteration).\n\n" +
                    "Requirements auto-checked at runtime:\n" +
                    "- Linux host (triton does not install cleanly on Windows yet)\n" +
                    "- CUDA GPU with compute capability 7.0+ (Volta or newer)\n" +
                    "- triton importable in the Python environment\n\n" +
                    "Known caveat: BatchRenorm's in-place buffer updates can trigger\n" +
                    "graph breaks that limit the speedup. For maximum benefit, pair\n" +
                    "with Tiny UNet + norm=gn instead of the default norm=brn.\n\n" +
                    "Safe to leave off. Silently falls back to eager mode on any\n" +
                    "failure; check the training log for the actual outcome.",
                    "https://pytorch.org/docs/stable/generated/torch.compile.html");
            grid.add(useTorchCompileCheck, 0, row, 2, 1);

            // Update the basic-mode early stopping status label when these controls change,
            // and grey out the patience spinner when early stopping is disabled.
            Runnable applyEarlyStoppingDisableState = () -> {
                boolean disabled = "Disabled".equals(earlyStoppingMetricCombo.getValue());
                earlyStoppingPatienceSpinner.setDisable(disabled);
                esPatienceLabel.setDisable(disabled);
            };
            earlyStoppingMetricCombo.valueProperty().addListener((obs, o, n) -> {
                updateEarlyStoppingStatusLabel();
                applyEarlyStoppingDisableState.run();
            });
            earlyStoppingPatienceSpinner.valueProperty().addListener(
                    (obs, o, n) -> updateEarlyStoppingStatusLabel());
            applyEarlyStoppingDisableState.run();

            TitledPane pane = new TitledPane("TRAINING STRATEGY", grid);
            pane.setExpanded(false); // Collapsed by default - advanced settings
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create(
                    "Advanced training strategy: scheduler, loss function, early stopping, and mixed precision"));
            return pane;
        }

        private TitledPane createChannelSection() {
            channelPanel = new ChannelSelectionPanel();
            channelPanel.validProperty().addListener((obs, old, valid) -> {
                updateValidation();
                // Channel count affects model layer structure - reload layers when channels become valid
                if (valid) {
                    updateLayerFreezePanel();
                }
            });

            TitledPane pane = new TitledPane("CHANNEL CONFIGURATION", channelPanel);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select and order image channels for model input"));
            return pane;
        }

        private TitledPane createClassSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            Label infoLabel = new Label("Select annotation classes to use for training:");
            infoLabel.setStyle("-fx-text-fill: #666;");

            // Pie chart showing per-class annotation area distribution
            classDistributionChart = new PieChart();
            classDistributionChart.setLegendVisible(false);
            classDistributionChart.setLabelsVisible(true);
            classDistributionChart.setLabelLineLength(10);
            classDistributionChart.setPrefHeight(180);
            classDistributionChart.setMaxHeight(180);
            classDistributionChart.setVisible(false);
            classDistributionChart.setManaged(false);

            classListView = new ListView<>();
            classListView.setCellFactory(lv -> new ClassListCell(advancedMode));
            classListView.setPrefHeight(120);
            TooltipHelper.install(classListView,
                    "Annotation classes found in the selected images.\n" +
                    "At least 2 classes must be selected for training.\n" +
                    "Each class should have representative annotations.\n\n" +
                    "Tip: Line/polyline annotations are recommended over area\n" +
                    "annotations. Lines focus training on class boundaries\n" +
                    "where accuracy matters most, and avoid overtraining on\n" +
                    "easy central regions.\n\n" +
                    "Use the weight spinner (right) to boost underrepresented\n" +
                    "classes. For example, set weight=2.0 for a rare class.");

            // Add select all / none / rebalance buttons
            Button selectAllBtn = new Button("Select All");
            TooltipHelper.install(selectAllBtn, "Select all annotation classes for training");
            selectAllBtn.setOnAction(e -> classListView.getItems().forEach(item -> item.selected().set(true)));

            Button selectNoneBtn = new Button("Select None");
            TooltipHelper.install(selectNoneBtn, "Deselect all annotation classes");
            selectNoneBtn.setOnAction(e -> classListView.getItems().forEach(item -> item.selected().set(false)));

            Button rebalanceBtn = new Button("Rebalance Classes");
            TooltipHelper.install(rebalanceBtn,
                    "Auto-set weight multipliers to compensate for class imbalance.\n\n" +
                    "Classes with fewer annotated pixels receive higher weights so the\n" +
                    "model pays equal attention to all classes during training.\n\n" +
                    "Works with both area and line annotations. For lines, pixel\n" +
                    "coverage is estimated from line length x stroke width.\n\n" +
                    "Note: Rebalancing weights helps but does NOT replace having\n" +
                    "sufficient training data. Adding more annotations for under-\n" +
                    "represented classes will produce better results than relying\n" +
                    "on weight compensation alone.");
            rebalanceBtn.setOnAction(e -> rebalanceClassWeights());

            rebalanceByDefaultCheck = new CheckBox("Rebalance by default");
            rebalanceByDefaultCheck.setSelected(DLClassifierPreferences.isRebalanceByDefault());
            rebalanceByDefaultCheck.selectedProperty().addListener((obs, old, newVal) ->
                    DLClassifierPreferences.setRebalanceByDefault(newVal));
            TooltipHelper.install(rebalanceByDefaultCheck,
                    "Automatically rebalance class weights when classes are loaded.\n\n" +
                    "When checked, class weights are auto-set each time you load\n" +
                    "classes, so underrepresented classes receive higher weights.\n" +
                    "You can still manually adjust weights after loading.");

            // Rebalance controls: advanced-only
            rebalanceBtn.visibleProperty().bind(advancedMode);
            rebalanceBtn.managedProperty().bind(advancedMode);
            rebalanceByDefaultCheck.visibleProperty().bind(advancedMode);
            rebalanceByDefaultCheck.managedProperty().bind(advancedMode);

            // Refresh chart visibility when toggling modes
            advancedMode.addListener((obs, old, newVal) -> refreshPieChart());

            HBox buttonBox = new HBox(10, selectAllBtn, selectNoneBtn, rebalanceBtn);

            content.getChildren().addAll(infoLabel, classDistributionChart, classListView,
                    rebalanceByDefaultCheck, buttonBox);

            TitledPane pane = new TitledPane("ANNOTATION CLASSES", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select which annotation classes to include in training"));
            return pane;
        }

        private TitledPane createAugmentationSection() {
            VBox content = new VBox(8);
            content.setPadding(new Insets(10));

            flipHorizontalCheck = new CheckBox("Horizontal flip");
            flipHorizontalCheck.setSelected(DLClassifierPreferences.isAugFlipHorizontal());
            TooltipHelper.install(flipHorizontalCheck,
                    "Randomly mirror tiles left-right during training.\n" +
                    "Effective for tissue where orientation is arbitrary.\n" +
                    "Almost always beneficial; disable only if horizontal\n" +
                    "orientation carries meaning in your images.");

            flipVerticalCheck = new CheckBox("Vertical flip");
            flipVerticalCheck.setSelected(DLClassifierPreferences.isAugFlipVertical());
            TooltipHelper.install(flipVerticalCheck,
                    "Randomly mirror tiles top-bottom during training.\n" +
                    "Same rationale as horizontal flip.\n" +
                    "Safe to enable for most histopathology images.");

            rotationCheck = new CheckBox("Random rotation (90 deg)");
            rotationCheck.setSelected(DLClassifierPreferences.isAugRotation());
            TooltipHelper.install(rotationCheck,
                    "Randomly rotate tiles by 0/90/180/270 degrees.\n" +
                    "Preserves pixel alignment (no interpolation artifacts).\n" +
                    "Beneficial when tissue structures have no preferred\n" +
                    "orientation. Combines well with flips for 8x augmentation.");

            // Intensity augmentation mode (replaces color jitter checkbox)
            Label intensityLabel = new Label("Intensity augmentation:");
            intensityAugCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "None", "Brightfield (color jitter)", "Fluorescence (per-channel)"));
            intensityAugCombo.setValue(mapIntensityModeToDisplay(DLClassifierPreferences.getAugIntensityMode()));
            intensityAugCombo.setPrefWidth(220);
            // Track manual user changes so auto-detection doesn't override
            intensityAugCombo.setOnAction(e -> intensityAugUserModified = true);
            TooltipHelper.install(
                    "Intensity augmentation mode.\n\n" +
                    "None: No intensity/color transforms.\n\n" +
                    "Brightfield: Correlated brightness, contrast, and gamma\n" +
                    "  across all channels. Recommended for H&E stained images.\n\n" +
                    "Fluorescence: Independent random intensity scaling per channel.\n" +
                    "  Recommended for fluorescence or multi-spectral images where\n" +
                    "  each channel is an independent signal.",
                    intensityLabel, intensityAugCombo);
            HBox intensityRow = new HBox(10, intensityLabel, intensityAugCombo);
            intensityRow.setAlignment(Pos.CENTER_LEFT);

            elasticCheck = new CheckBox("Elastic deformation");
            elasticCheck.setSelected(DLClassifierPreferences.isAugElasticDeform());
            TooltipHelper.installWithLink(elasticCheck,
                    "Apply smooth random spatial deformations to tiles.\n" +
                    "Simulates tissue distortion and cutting artifacts.\n" +
                    "Computationally expensive but effective for histopathology.\n" +
                    "May reduce training speed by ~30%.\n\n" +
                    "Most beneficial when training data is limited and the\n" +
                    "model needs to handle shape variations in the tissue.",
                    "https://albumentations.ai/docs/");

            // Advanced augmentation button -- opens a popup for strength/probability tuning.
            // Visible only in advanced mode; bound below via advancedMode binding.
            Button advancedAugButton = new Button("Advanced augmentation settings...");
            TooltipHelper.install(advancedAugButton,
                    "Fine-tune augmentation strengths and probabilities:\n" +
                    "brightness/contrast/gamma limits, elastic alpha/sigma,\n" +
                    "noise std, and per-augmentation probabilities.\n\n" +
                    "Defaults match the built-in augmentation pipeline --\n" +
                    "changes here override those defaults for all future training runs.");
            advancedAugButton.setOnAction(e -> {
                AdvancedAugmentationDialog dialog = new AdvancedAugmentationDialog(
                        advancedAugButton.getScene() != null
                                ? advancedAugButton.getScene().getWindow()
                                : null);
                dialog.showAndWait();
            });
            advancedAugButton.visibleProperty().bind(advancedMode);
            advancedAugButton.managedProperty().bind(advancedMode);

            content.getChildren().addAll(
                    flipHorizontalCheck,
                    flipVerticalCheck,
                    rotationCheck,
                    intensityRow,
                    elasticCheck,
                    advancedAugButton
            );

            TitledPane pane = new TitledPane("DATA AUGMENTATION", content);
            pane.setExpanded(false); // Collapsed by default
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Configure data augmentation to improve model generalization"));
            return pane;
        }

        private VBox createErrorSummaryPanel() {
            errorSummaryPanel = new VBox(5);
            errorSummaryPanel.setStyle(
                    "-fx-background-color: #fff3cd; " +
                    "-fx-border-color: #ffc107; " +
                    "-fx-border-width: 1px; " +
                    "-fx-padding: 10px;"
            );
            errorSummaryPanel.setVisible(false);
            errorSummaryPanel.setManaged(false);

            Label errorTitle = new Label("Please fix the following errors:");
            errorTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #856404;");

            errorListBox = new VBox(3);

            errorSummaryPanel.getChildren().addAll(errorTitle, errorListBox);
            return errorSummaryPanel;
        }

        /**
         * Loads classes from all selected project images as a union.
         * Runs image reading on a background thread to avoid blocking the FX thread.
         */
        private void loadClassesFromSelectedImages() {
            List<ImageSelectionItem> selectedItems = imageSelectionList.getItems().stream()
                    .filter(item -> item.selected.get())
                    .collect(Collectors.toList());

            if (selectedItems.isEmpty()) {
                return;
            }

            loadClassesButton.setDisable(true);
            loadClassesButton.setText("Loading classes...");

            // Capture stroke width on FX thread before async work -- used to
            // estimate pixel coverage for line/polyline annotations
            final int strokeWidth = lineStrokeWidthSpinner.getValue();

            CompletableFuture.runAsync(() -> {
                // Accumulate classes and effective pixel coverage across all selected images.
                // For area annotations, coverage = ROI area.
                // For line annotations, coverage = path length * stroke width.
                // This allows rebalancing to work with line annotations, which are
                // the recommended annotation style for boundary-focused training.
                Map<String, PathClass> classMap = new TreeMap<>();
                Map<String, Double> classAreas = new LinkedHashMap<>();
                ImageData<BufferedImage> firstImageData = null;
                boolean foundLineAnnotations = false;

                // Collect pixel sizes from all selected images for spatial info display
                Set<Double> pixelSizes = new LinkedHashSet<>();

                for (ImageSelectionItem selItem : selectedItems) {
                    try {
                        ImageData<BufferedImage> data = selItem.entry.readImageData();

                        // Keep the first image's data for channel initialization
                        if (firstImageData == null) {
                            firstImageData = data;
                        }

                        // Collect pixel size from each image
                        try {
                            double ps = data.getServer().getPixelCalibration()
                                    .getAveragedPixelSizeMicrons();
                            if (!Double.isNaN(ps) && ps > 0) {
                                // Round to 4 decimal places for comparison
                                pixelSizes.add(Math.round(ps * 10000.0) / 10000.0);
                            }
                        } catch (Exception e) {
                            logger.debug("Could not read pixel size from '{}': {}",
                                    selItem.entry.getImageName(), e.getMessage());
                        }

                        for (PathObject annotation : data.getHierarchy().getAnnotationObjects()) {
                            PathClass pathClass = annotation.getPathClass();
                            if (pathClass != null && !pathClass.isDerivedClass()) {
                                classMap.putIfAbsent(pathClass.getName(), pathClass);

                                // Estimate effective pixel coverage:
                                // - Line/polyline ROIs have getArea()=0, so use
                                //   path length * stroke width as the pixel count
                                // - Area ROIs use geometric area directly
                                // - Mixed annotations (some lines, some areas) sum naturally
                                var roi = annotation.getROI();
                                double coverage;
                                if (roi.isLine()) {
                                    coverage = roi.getLength() * strokeWidth;
                                    foundLineAnnotations = true;
                                } else {
                                    coverage = roi.getArea();
                                }
                                classAreas.merge(pathClass.getName(), coverage, Double::sum);
                            }
                        }

                        // Close server for all images except the first
                        // (ChannelSelectionPanel stores currentServer for lazy bit depth lookup)
                        if (data != firstImageData) {
                            try {
                                data.getServer().close();
                            } catch (Exception e) {
                                logger.debug("Error closing image server: {}", e.getMessage());
                            }
                        }
                    } catch (Exception e) {
                        logger.warn("Could not read image '{}': {}",
                                selItem.entry.getImageName(), e.getMessage());
                    }
                }

                final ImageData<BufferedImage> channelImageData = firstImageData;
                final Map<String, PathClass> finalClassMap = classMap;
                final Map<String, Double> finalClassAreas = classAreas;
                final Set<Double> finalPixelSizes = new LinkedHashSet<>(pixelSizes);
                final boolean finalHasLineAnnotations = foundLineAnnotations;

                Platform.runLater(() -> {
                    // Track whether selected images contain line annotations
                    // (used to conditionally show line stroke width in basic mode)
                    hasLineAnnotations = finalHasLineAnnotations;
                    updateLineStrokeVisibility();

                    // Update native pixel size for spatial info labels
                    if (finalPixelSizes.size() == 1) {
                        nativePixelSizeMicrons = finalPixelSizes.iterator().next();
                    } else {
                        // Mixed or missing pixel sizes -- can't show physical units
                        nativePixelSizeMicrons = Double.NaN;
                    }
                    updateSpatialInfoLabels();

                    // Initialize channel panel from first image
                    if (channelImageData != null) {
                        channelPanel.setImageData(channelImageData);
                        channelPanel.autoConfigureForImageType(
                                channelImageData.getImageType(),
                                channelImageData.getServer().nChannels());

                        // Auto-select appropriate intensity mode based on image type,
                        // but only if the user hasn't manually changed it
                        if (!intensityAugUserModified) {
                            if (!isBrightfield(channelImageData)) {
                                intensityAugCombo.setValue("Fluorescence (per-channel)");
                            } else {
                                intensityAugCombo.setValue("Brightfield (color jitter)");
                            }
                            // Reset flag so programmatic change isn't treated as manual
                            intensityAugUserModified = false;
                        }
                        // Store image type for backbone compatibility check
                        lastImageIsBrightfield = isBrightfield(channelImageData);
                        checkBackboneImageCompatibility();
                    }

                    // Populate class list with union of all classes
                    classListView.getItems().clear();
                    for (Map.Entry<String, PathClass> entry : finalClassMap.entrySet()) {
                        PathClass pathClass = entry.getValue();
                        double area = finalClassAreas.getOrDefault(entry.getKey(), 0.0);
                        ClassItem classItem = new ClassItem(
                                pathClass.getName(), pathClass.getColor(), true, area);
                        classItem.selected().addListener((obs, old, newVal) -> {
                            refreshPieChart();
                            updateValidation();
                        });
                        classListView.getItems().add(classItem);
                    }

                    // Compute tile estimate from annotation areas
                    double totalArea = finalClassAreas.values().stream()
                            .mapToDouble(d -> d).sum();
                    cachedTotalAnnotationArea = totalArea;
                    lastLoadedClassCount = finalClassMap.size();
                    lastLoadedImageCount = selectedItems.size();
                    updateTileEstimateLabel(finalClassMap.size(), selectedItems.size());

                    refreshPieChart();

                    // Populate focus class combo with loaded class names
                    updateFocusClassCombo();

                    // Auto-rebalance class weights if preference is set
                    if (rebalanceByDefaultCheck.isSelected()) {
                        rebalanceClassWeights();
                    }

                    // Enable gated sections
                    classesLoaded = true;
                    setGatedSectionsEnabled(true);

                    // Auto-match classes from source model if one was loaded
                    if (sourceModelClassNames != null && !sourceModelClassNames.isEmpty()) {
                        autoMatchModelClasses();
                    }

                    // Reset button state
                    loadClassesButton.setText("Load Classes from Selected Images");
                    updateLoadClassesButtonState();
                    updateValidation();

                    logger.info("Loaded {} classes from {} images",
                            finalClassMap.size(), selectedItems.size());
                });
            });
        }

        /** Enables/disables the Load Classes button based on whether any images are checked. */
        private void updateLoadClassesButtonState() {
            boolean anySelected = imageSelectionList.getItems().stream()
                    .anyMatch(item -> item.selected.get());
            loadClassesButton.setDisable(!anySelected);
        }

        /** Updates the tile estimate label shown after classes are loaded. */
        private void updateTileEstimateLabel(int classCount, int imageCount) {
            if (tileEstimateLabel == null || cachedTotalAnnotationArea <= 0) return;
            int est = estimateTileCount();
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(
                    "Loaded %d classes from %d images. Estimated ~%,d training tiles.",
                    classCount, imageCount, est));
            appendInMemoryCacheEstimate(sb, est);
            tileEstimateLabel.setText(sb.toString());
            tileEstimateLabel.setVisible(true);
            tileEstimateLabel.setManaged(true);
        }

        /**
         * Appends an in-memory dataset cache estimate to the tile-estimate
         * label so the user sees RAM requirements at selection time (not
         * post-mortem during training). Mirrors the Python-side decision
         * (25% ceiling for "auto"; warns if "on" exceeds free RAM).
         */
        private void appendInMemoryCacheEstimate(StringBuilder sb, int tileEst) {
            String mode = DLClassifierPreferences.getDefaultInMemoryDataset();
            if ("off".equals(mode)) {
                sb.append(" In-memory cache: off.");
                tileEstimateLabel.setStyle("-fx-text-fill: #2a7a2a; -fx-font-size: 11px;");
                return;
            }
            int tile = tileSizeSpinner != null ? tileSizeSpinner.getValue() : 512;
            int chs = 3;
            int bitDepth = 8;
            try {
                if (channelPanel != null && channelPanel.isValid()) {
                    ChannelConfiguration cc = channelPanel.getChannelConfiguration();
                    chs = Math.max(1, cc.getNumChannels());
                    bitDepth = cc.getBitDepth();
                }
            } catch (Exception ignored) {
                // Pre-selection: fall back to RGB 8-bit defaults
            }
            // AnnotationExtractor.savePatch exports as uint8 TIFF only when
            // numBands <= 4 AND dataType == TYPE_BYTE; otherwise float32 .raw
            // at 4 bytes per channel per pixel.
            int bytesPerPixel = (chs <= 4 && bitDepth == 8) ? 1 : 4;
            int ctxScale = contextScaleCombo != null
                    ? parseContextScale(contextScaleCombo.getValue()) : 1;
            boolean hasContext = ctxScale > 1;
            long perImage = (long) tile * tile * chs * bytesPerPixel;
            if (hasContext) perImage *= 2L;
            // Masks are always int indexed PNGs on disk but cached as uint8
            // in RAM (class index fits in a byte for <=256 classes).
            long perMask = (long) tile * tile;
            long totalBytes = (long) tileEst * (perImage + perMask);
            double estGb = totalBytes / 1e9;

            long availableBytes = -1L;
            try {
                java.lang.management.OperatingSystemMXBean osBean =
                        java.lang.management.ManagementFactory.getOperatingSystemMXBean();
                if (osBean instanceof com.sun.management.OperatingSystemMXBean sun) {
                    availableBytes = sun.getFreeMemorySize();
                }
            } catch (Throwable ignored) {
                // com.sun.* unavailable on some JVMs
            }

            if (availableBytes <= 0) {
                sb.append(String.format(" In-memory cache: ~%.2f GB (mode=%s).", estGb, mode));
                tileEstimateLabel.setStyle("-fx-text-fill: #2a7a2a; -fx-font-size: 11px;");
                return;
            }
            double availGb = availableBytes / 1e9;

            String color = "#2a7a2a"; // green
            String verdict;
            if ("auto".equals(mode)) {
                if (totalBytes < 0.25 * availableBytes) {
                    verdict = "auto -> will enable";
                } else {
                    verdict = "auto -> will decline (>25% of free); set to 'on' to force";
                    color = "#cc6600"; // orange
                }
            } else { // "on"
                if (totalBytes > availableBytes) {
                    verdict = "on -> FORCED over free RAM; preload may OOM";
                    color = "#cc0000"; // red
                } else if (totalBytes > 0.5 * availableBytes) {
                    verdict = "on -> forced (>50% of free RAM; tight)";
                    color = "#cc6600";
                } else {
                    verdict = "on -> forced";
                }
            }
            sb.append(String.format(
                    " In-memory cache: ~%.2f GB of %.2f GB free RAM (%s).",
                    estGb, availGb, verdict));
            tileEstimateLabel.setStyle(
                    "-fx-text-fill: " + color + "; -fx-font-size: 11px;");
        }

        /** Estimates tile count from cached annotation area and current settings. */
        private int estimateTileCount() {
            if (cachedTotalAnnotationArea <= 0) return 0;
            double tileSize = tileSizeSpinner.getValue();
            double downsample = parseDownsample(downsampleCombo.getValue());
            double coveragePerTile = tileSize * downsample;
            // PatchSampler uses ~50% overlap between patches
            double stepSize = coveragePerTile * 0.75;
            return Math.max(1, (int) (cachedTotalAnnotationArea / (stepSize * stepSize)));
        }

        /** Visual indicator when images change after classes were already loaded. */
        private void markClassesStale() {
            loadClassesButton.setText("Reload Classes (images changed)");
            loadClassesButton.setStyle(
                    "-fx-font-weight: bold; -fx-text-fill: #cc6600;");
        }

        /**
         * Auto-selects annotation classes that match the source model's class names.
         * Deselects classes not in the source model.
         * Shows an info notification if some model classes were not found.
         */
        private void autoMatchModelClasses() {
            Set<String> modelClasses = new HashSet<>(sourceModelClassNames);
            List<String> matched = new ArrayList<>();
            List<String> notFound = new ArrayList<>();

            for (ClassItem item : classListView.getItems()) {
                if (modelClasses.contains(item.name())) {
                    item.selected().set(true);
                    matched.add(item.name());
                } else {
                    item.selected().set(false);
                }
            }

            // Check which model classes were not found in annotations
            Set<String> annotationClassNames = classListView.getItems().stream()
                    .map(ClassItem::name)
                    .collect(Collectors.toSet());
            for (String modelClass : sourceModelClassNames) {
                if (!annotationClassNames.contains(modelClass)) {
                    notFound.add(modelClass);
                }
            }

            refreshPieChart();
            updateValidation();

            if (!notFound.isEmpty()) {
                Dialogs.showInfoNotification("Class Auto-Match",
                        "Matched " + matched.size() + " of " + sourceModelClassNames.size()
                        + " model classes.\nNot found in annotations: "
                        + String.join(", ", notFound));
                logger.info("Class auto-match: {} matched, {} not found: {}",
                        matched.size(), notFound.size(), notFound);
            } else {
                logger.info("Class auto-match: all {} model classes matched", matched.size());
            }
        }

        /** Enables or disables all gated sections (everything except image source). */
        private void setGatedSectionsEnabled(boolean enabled) {
            for (TitledPane pane : gatedSections) {
                pane.setDisable(!enabled);
                if (!enabled) {
                    pane.setExpanded(false);
                }
            }
        }

        /** Updates the focus class combo with current class names from classListView. */
        private void updateFocusClassCombo() {
            String currentSelection = focusClassCombo.getValue();
            List<String> items = new ArrayList<>();
            items.add("None (use Mean IoU)");
            for (ClassItem item : classListView.getItems()) {
                items.add(item.name());
            }
            focusClassCombo.setItems(FXCollections.observableArrayList(items));
            // Restore previous selection if still valid, otherwise try saved preference
            if (currentSelection != null && items.contains(currentSelection)) {
                focusClassCombo.setValue(currentSelection);
            } else {
                String savedFocus = DLClassifierPreferences.getDefaultFocusClass();
                if (savedFocus != null && !savedFocus.isEmpty()
                        && items.contains(savedFocus)) {
                    focusClassCombo.setValue(savedFocus);
                    double savedMinIoU = DLClassifierPreferences.getDefaultFocusClassMinIoU();
                    if (savedMinIoU > 0) {
                        focusClassMinIoUSpinner.getValueFactory().setValue(savedMinIoU);
                    }
                } else {
                    focusClassCombo.setValue("None (use Mean IoU)");
                }
            }
        }

        /** Maps focus class combo display value to config value (null for "None"). */
        private static String mapFocusClassFromDisplay(String display) {
            if (display == null || display.startsWith("None")) return null;
            return display;
        }

        private static String mapOhemScheduleFromDisplay(String display) {
            if (display != null && display.startsWith("Anneal")) return "anneal";
            return "fixed";
        }

        /**
         * Updates the resolution and context info labels based on current
         * tile size, downsample, context scale, and native pixel size.
         * <p>
         * Handles three pixel calibration states:
         * - Single consistent pixel size across all images -> show physical units (um)
         * - Mixed pixel sizes across images -> warn user, show pixels only
         * - No pixel calibration -> note unavailability, show pixels only
         */
        /**
         * Shows/hides line stroke width controls based on mode and annotation type.
         * In advanced mode: always visible. In basic mode: only when line annotations detected.
         */
        private void updateLineStrokeVisibility() {
            boolean show = advancedMode.get() || hasLineAnnotations;
            if (lineStrokeLabel != null) {
                lineStrokeLabel.setVisible(show);
                lineStrokeLabel.setManaged(show);
            }
            if (lineStrokeWidthSpinner != null) {
                lineStrokeWidthSpinner.setVisible(show);
                lineStrokeWidthSpinner.setManaged(show);
            }
        }

        private void updateSpatialInfoLabels() {
            int tileSize = tileSizeSpinner.getValue();
            double downsample = parseDownsample(downsampleCombo.getValue());
            int contextScale = parseContextScale(contextScaleCombo.getValue());

            // Detail tile covers tileSize * downsample native pixels
            int detailCoveragePixels = (int) (tileSize * downsample);
            boolean hasCalibration = !Double.isNaN(nativePixelSizeMicrons) && nativePixelSizeMicrons > 0;

            if (hasCalibration) {
                double effectivePixelSize = nativePixelSizeMicrons * downsample;
                double detailCoverageUm = detailCoveragePixels * nativePixelSizeMicrons;
                resolutionInfoLabel.setText(String.format(
                        "Detail tile: %dpx x %dpx = %.0f x %.0fum (%.3f um/px effective)",
                        tileSize, tileSize, detailCoverageUm, detailCoverageUm, effectivePixelSize));
            } else if (classesLoaded) {
                // Classes loaded but no calibration -- tell the user why
                resolutionInfoLabel.setText(String.format(
                        "Detail tile: %dpx x %dpx covers %d x %d native pixels " +
                        "(pixel size not available -- images lack calibration or have mixed pixel sizes)",
                        tileSize, tileSize, detailCoveragePixels, detailCoveragePixels));
            } else {
                resolutionInfoLabel.setText(String.format(
                        "Detail tile: %dpx x %dpx covers %d x %d native pixels " +
                        "(load classes to show physical dimensions)",
                        tileSize, tileSize, detailCoveragePixels, detailCoveragePixels));
            }

            if (contextScale <= 1) {
                contextInfoLabel.setText("No context -- model sees only the detail tile.");
            } else {
                int contextCoveragePixels = detailCoveragePixels * contextScale;
                if (hasCalibration) {
                    double contextCoverageUm = contextCoveragePixels * nativePixelSizeMicrons;
                    contextInfoLabel.setText(String.format(
                            "Context window: %.0f x %.0fum (%dx the detail tile area, " +
                            "downsampled to %dpx x %dpx)",
                            contextCoverageUm, contextCoverageUm, contextScale * contextScale,
                            tileSize, tileSize));
                } else {
                    contextInfoLabel.setText(String.format(
                            "Context window: %d x %d native pixels (%dx the detail tile area, " +
                            "downsampled to %dpx x %dpx)",
                            contextCoveragePixels, contextCoveragePixels, contextScale * contextScale,
                            tileSize, tileSize));
                }
            }
        }

        /**
         * Updates the context preview window downsample when the context scale
         * or downsample changes. No-op if the context preview window is not open.
         */
        private void updateContextPreview() {
            if (contextPreviewManager != null) {
                int ctxScale = parseContextScale(contextScaleCombo.getValue());
                double ds = parseDownsample(downsampleCombo.getValue()) * ctxScale;
                contextPreviewManager.setDownsample(ds);
                if (contextPreviewStage != null) {
                    contextPreviewStage.setTitle(String.format(
                            "Context Preview (%dx context at %.0fx downsample)",
                            ctxScale, ds));
                }
            }
        }

        /**
         * Links the resolution and context preview stages so they move together.
         * Called after either preview is shown when both are open.
         */
        private void linkPreviewStages() {
            // Only link when both stages exist and are showing
            if (previewStage == null || contextPreviewStage == null
                    || !previewStage.isShowing() || !contextPreviewStage.isShowing()) {
                return;
            }

            // Remove any existing listeners first
            unlinkPreviewStages();

            // Compute the initial offset (context relative to resolution)
            final double[] offsetX = { contextPreviewStage.getX() - previewStage.getX() };
            final double[] offsetY = { contextPreviewStage.getY() - previewStage.getY() };

            // Resolution stage drives context stage
            resPosXListener = (obs, old, newX) -> {
                if (!syncingPreviews && contextPreviewStage != null) {
                    syncingPreviews = true;
                    contextPreviewStage.setX(newX.doubleValue() + offsetX[0]);
                    syncingPreviews = false;
                }
            };
            resPosYListener = (obs, old, newY) -> {
                if (!syncingPreviews && contextPreviewStage != null) {
                    syncingPreviews = true;
                    contextPreviewStage.setY(newY.doubleValue() + offsetY[0]);
                    syncingPreviews = false;
                }
            };

            // Context stage drives resolution stage
            ctxPosXListener = (obs, old, newX) -> {
                if (!syncingPreviews && previewStage != null) {
                    syncingPreviews = true;
                    previewStage.setX(newX.doubleValue() - offsetX[0]);
                    syncingPreviews = false;
                }
            };
            ctxPosYListener = (obs, old, newY) -> {
                if (!syncingPreviews && previewStage != null) {
                    syncingPreviews = true;
                    previewStage.setY(newY.doubleValue() - offsetY[0]);
                    syncingPreviews = false;
                }
            };

            previewStage.xProperty().addListener(resPosXListener);
            previewStage.yProperty().addListener(resPosYListener);
            contextPreviewStage.xProperty().addListener(ctxPosXListener);
            contextPreviewStage.yProperty().addListener(ctxPosYListener);
        }

        /**
         * Removes position-sync listeners from preview stages.
         */
        private void unlinkPreviewStages() {
            if (previewStage != null && resPosXListener != null) {
                previewStage.xProperty().removeListener(resPosXListener);
                previewStage.yProperty().removeListener(resPosYListener);
            }
            if (contextPreviewStage != null && ctxPosXListener != null) {
                contextPreviewStage.xProperty().removeListener(ctxPosXListener);
                contextPreviewStage.yProperty().removeListener(ctxPosYListener);
            }
            resPosXListener = null;
            resPosYListener = null;
            ctxPosXListener = null;
            ctxPosYListener = null;
        }

        private boolean isBrightfield(ImageData<BufferedImage> imageData) {
            ImageData.ImageType type = imageData.getImageType();
            return type == ImageData.ImageType.BRIGHTFIELD_H_E
                    || type == ImageData.ImageType.BRIGHTFIELD_H_DAB
                    || type == ImageData.ImageType.BRIGHTFIELD_OTHER;
        }

        /**
         * Checks if a backbone name refers to a foundation model encoder.
         * Foundation models are downloaded on-demand from HuggingFace.
         */
        private static boolean isFoundationModel(String backbone) {
            return backbone != null && (
                    backbone.equals("h-optimus-0") ||
                    backbone.equals("virchow") ||
                    backbone.startsWith("hibou-") ||
                    backbone.equals("midnight") ||
                    backbone.startsWith("dinov2-"));
        }

        /**
         * Updates the handler-specific UI section when the architecture changes.
         * Handlers can provide custom UI controls (e.g., MuViT patch size, levels, MAE pretraining).
         */
        private void updateHandlerUI(String architecture) {
            handlerUIContainer.getChildren().clear();
            currentHandlerUI = null;

            ClassifierHandler handler = ClassifierRegistry.getHandler(architecture)
                    .orElse(null);
            if (handler == null) return;

            handler.createTrainingUI().ifPresent(ui -> {
                currentHandlerUI = ui;
                handlerUIContainer.getChildren().add(ui.getNode());
            });
        }

        /**
         * Hides or shows dialog sections that don't apply to certain architectures.
         * MuViT handles multi-scale internally and doesn't use ImageNet backbones,
         * so the Transfer Learning and Context Scale sections are hidden.
         */
        private void updateSectionsForArchitecture(String architecture) {
            boolean isMuViT = "muvit".equals(architecture);

            // Update weight initialization radio buttons for this architecture
            updateWeightInitOptions(architecture);

            // Backbone/Encoder row: MuViT shows model size in its handler UI instead
            if (backboneLabel != null) {
                boolean showBackbone = !isMuViT;
                backboneLabel.setVisible(showBackbone);
                backboneLabel.setManaged(showBackbone);
                backboneCombo.setVisible(showBackbone);
                backboneCombo.setManaged(showBackbone);
            }

            // Context Scale: MuViT handles multi-resolution internally via level_scales
            if (contextScaleCombo != null) {
                boolean showContextScale = !isMuViT;
                contextScaleCombo.setVisible(showContextScale);
                contextScaleCombo.setManaged(showContextScale);
                if (contextScaleLabel != null) {
                    contextScaleLabel.setVisible(showContextScale);
                    contextScaleLabel.setManaged(showContextScale);
                }
                if (contextInfoLabel != null) {
                    contextInfoLabel.setVisible(showContextScale);
                    contextInfoLabel.setManaged(showContextScale);
                }
                if (isMuViT) {
                    contextScaleCombo.setValue("None (single scale)");
                }
            }

            // Update whole-image info label for ViT tile size cap
            updateWholeImageInfoLabel();
        }

        /**
         * Shows or hides the whole-image info label based on architecture, downsample,
         * and actual selected image dimensions.
         * <p>
         * Orange text = informational warning (images fit at current downsample).
         * Red text = images WILL be tiled instead of processed whole at this downsample.
         */
        private void updateWholeImageInfoLabel() {
            if (wholeImageInfoLabel == null) return;

            if (wholeImageCheck == null || !wholeImageCheck.isSelected()) {
                wholeImageInfoLabel.setVisible(false);
                wholeImageInfoLabel.setManaged(false);
                return;
            }

            String arch = architectureCombo.getValue();
            ClassifierHandler handler = ClassifierRegistry.getHandler(arch)
                    .orElse(ClassifierRegistry.getDefaultHandler());
            List<Integer> sizes = handler.getSupportedTileSizes();
            int maxTile = sizes.isEmpty()
                    ? Integer.MAX_VALUE
                    : sizes.stream().mapToInt(Integer::intValue).max().orElse(Integer.MAX_VALUE);

            if (maxTile > 1024) {
                wholeImageInfoLabel.setVisible(false);
                wholeImageInfoLabel.setManaged(false);
                return;
            }

            // Compute the largest selected image dimension at the current downsample
            double downsample = parseDownsample(downsampleCombo.getValue());
            int maxDimAtDs = 0;
            String largestImageName = null;
            if (imageSelectionList != null) {
                for (ImageSelectionItem item : imageSelectionList.getItems()) {
                    if (!item.selected.get()) continue;
                    int dim = (int) Math.ceil(
                            Math.max(item.imageWidth, item.imageHeight) / downsample);
                    if (dim > maxDimAtDs) {
                        maxDimAtDs = dim;
                        largestImageName = item.entry.getImageName();
                    }
                }
            }

            boolean willBeTiled = maxDimAtDs > maxTile;

            if (willBeTiled) {
                // RED: images exceed max tile size -- whole-image mode will be ignored
                wholeImageInfoLabel.setStyle(
                        "-fx-text-fill: #CC0000; -fx-font-size: 11px; -fx-font-weight: bold;");
                wholeImageInfoLabel.setText(String.format(
                        "WARNING: \"%s\" is %dpx at %.0fx downsample, exceeding the %dpx max. "
                        + "This image WILL BE TILED, not processed whole. "
                        + "Increase downsample to fit within %dpx.",
                        largestImageName, maxDimAtDs, downsample, maxTile, maxTile));
            } else {
                // ORANGE: informational -- images fit, but there is a cap
                wholeImageInfoLabel.setStyle(
                        "-fx-text-fill: #CC7A00; -fx-font-size: 11px; -fx-font-weight: normal;");
                if (maxDimAtDs > 0) {
                    wholeImageInfoLabel.setText(String.format(
                            "%s limits tiles to %dpx max. Your largest selected image is "
                            + "%dpx at %.0fx downsample -- fits within the limit.",
                            handler.getDisplayName(), maxTile, maxDimAtDs, downsample));
                } else {
                    wholeImageInfoLabel.setText(String.format(
                            "%s limits tiles to %dpx max. If your image exceeds %dpx "
                            + "at the selected downsample, it will be tiled automatically.",
                            handler.getDisplayName(), maxTile, maxTile));
                }
            }
            wholeImageInfoLabel.setVisible(true);
            wholeImageInfoLabel.setManaged(true);
        }

        /** Backbones shown in basic mode (simple ResNets only). */
        private static final List<String> BASIC_BACKBONES =
                List.of("resnet18", "resnet34", "resnet50");

        /** User-friendly labels for basic mode backbone selection. */
        private static String getBasicModeBackboneLabel(String backbone) {
            return switch (backbone) {
                case "resnet18" -> "Small -- ResNet18 (fast, good starting point)";
                case "resnet34" -> "Medium -- ResNet34 (recommended)";
                case "resnet50" -> "Large -- ResNet50 (best accuracy, needs more data)";
                default -> backbone;
            };
        }

        private void updateBackboneOptions(String architecture) {
            ClassifierHandler handler = ClassifierRegistry.getHandler(architecture)
                    .orElse(ClassifierRegistry.getDefaultHandler());

            Map<String, Object> params = handler.getArchitectureParams(null);
            Object backbones = params.get("available_backbones");

            List<String> backboneList = new ArrayList<>();
            if (backbones instanceof List<?>) {
                for (Object b : (List<?>) backbones) {
                    backboneList.add(b.toString());
                }
            } else {
                backboneList.addAll(List.of("resnet34", "resnet50", "efficientnet-b0"));
            }

            // In basic mode, show only the simple ResNets
            if (!advancedMode.get()) {
                backboneList.retainAll(BASIC_BACKBONES);
                if (backboneList.isEmpty()) {
                    // Fallback if handler has none of the basic backbones
                    backboneList.addAll(BASIC_BACKBONES);
                }
            }

            backboneCombo.setItems(FXCollections.observableArrayList(backboneList));

            // Show display names via custom cell factory (handler-specific lookup).
            // In basic mode, show user-friendly size labels instead of raw model names.
            backboneCombo.setCellFactory(lv -> new ListCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty);
                    if (empty || item == null) {
                        setText(null);
                    } else if (!advancedMode.get()) {
                        setText(getBasicModeBackboneLabel(item));
                    } else {
                        setText(handler.getBackboneDisplayName(item));
                    }
                }
            });
            backboneCombo.setButtonCell(new ListCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty);
                    if (empty || item == null) {
                        setText(null);
                    } else if (!advancedMode.get()) {
                        setText(getBasicModeBackboneLabel(item));
                    } else {
                        setText(handler.getBackboneDisplayName(item));
                    }
                }
            });

            // Restore last used backbone from preferences if it's available for this architecture
            String savedBackbone = DLClassifierPreferences.getLastBackbone();
            if (backboneList.contains(savedBackbone)) {
                backboneCombo.setValue(savedBackbone);
            } else if (!backboneList.isEmpty()) {
                backboneCombo.setValue(backboneList.get(0));
            }
        }

        /** Checks backbone-image type compatibility and shows/hides a non-blocking warning. */
        private void checkBackboneImageCompatibility() {
            if (backboneCompatWarning == null) return;
            String backbone = backboneCombo.getValue();
            if (backbone == null) {
                backboneCompatWarning.setVisible(false);
                backboneCompatWarning.setManaged(false);
                return;
            }
            boolean isHistologyBackbone = backbone.contains("_lunit")
                    || backbone.contains("_kather") || backbone.contains("_tcga")
                    || backbone.contains("_pathology");
            if (isHistologyBackbone && !lastImageIsBrightfield) {
                backboneCompatWarning.setText(
                        "Histology backbone selected but images appear to be fluorescence. " +
                        "Consider a standard backbone (resnet34, resnet50) for best results.");
                backboneCompatWarning.setVisible(true);
                backboneCompatWarning.setManaged(true);
            } else {
                backboneCompatWarning.setVisible(false);
                backboneCompatWarning.setManaged(false);
            }
        }

        /** Checks if any Training Strategy or Augmentation settings differ from defaults. */
        private boolean checkAdvancedSettingsDiffer() {
            // Training strategy
            if (schedulerCombo != null && !"One Cycle".equals(schedulerCombo.getValue())) return true;
            if (lossFunctionCombo != null && !"Cross Entropy + Dice".equals(lossFunctionCombo.getValue())) return true;
            if (earlyStoppingPatienceSpinner != null && earlyStoppingPatienceSpinner.getValue() != 15) return true;
            if (earlyStoppingMetricCombo != null && "Disabled".equals(earlyStoppingMetricCombo.getValue())) return true;
            if (mixedPrecisionCheck != null && !mixedPrecisionCheck.isSelected()) return true;
            if (gradientAccumulationSpinner != null && gradientAccumulationSpinner.getValue() != 1) return true;
            if (progressiveResizeCheck != null && progressiveResizeCheck.isSelected()) return true;
            // Augmentation
            if (flipHorizontalCheck != null && !flipHorizontalCheck.isSelected()) return true;
            if (flipVerticalCheck != null && !flipVerticalCheck.isSelected()) return true;
            if (rotationCheck != null && !rotationCheck.isSelected()) return true;
            if (elasticCheck != null && elasticCheck.isSelected()) return true;
            return false;
        }

        /** Shows a temporary amber notification that auto-dismisses after 5 seconds. */
        private void showTemporaryNotification(String message) {
            Label note = new Label(message);
            note.setWrapText(true);
            note.setStyle("-fx-text-fill: #856404; -fx-background-color: #fff3cd; " +
                    "-fx-padding: 4 8; -fx-background-radius: 3; -fx-font-size: 11px;");
            // Insert before the error summary panel (which is near the bottom)
            VBox dialogContent = (VBox) errorSummaryPanel.getParent();
            if (dialogContent != null) {
                int idx = dialogContent.getChildren().indexOf(errorSummaryPanel);
                if (idx >= 0) {
                    dialogContent.getChildren().add(idx, note);
                }
            }
            PauseTransition pause = new PauseTransition(Duration.seconds(5));
            pause.setOnFinished(e -> {
                if (dialogContent != null) {
                    dialogContent.getChildren().remove(note);
                }
            });
            pause.play();
        }

        private void validateClassifierName(String name) {
            if (name == null || name.trim().isEmpty()) {
                validationErrors.put("name", "Classifier name is required");
            } else if (!name.matches("[a-zA-Z0-9_-]+")) {
                validationErrors.put("name", "Classifier name can only contain letters, numbers, underscore, and hyphen");
            } else {
                validationErrors.remove("name");
            }
            updateErrorSummary();
        }

        private void updateValidation() {
            // Check that classes have been loaded
            if (!classesLoaded) {
                validationErrors.put("classesLoaded",
                        "Select images and click 'Load Classes from Selected Images'");
                // Clear channel/class errors since they are not relevant yet
                validationErrors.remove("channels");
                validationErrors.remove("classes");
                validationErrors.remove("images");
                updateErrorSummary();
                return;
            }
            validationErrors.remove("classesLoaded");

            // Check at least 1 image is selected
            long selectedImageCount = imageSelectionList.getItems().stream()
                    .filter(item -> item.selected.get())
                    .count();
            if (selectedImageCount < 1) {
                validationErrors.put("images", "At least 1 image must be selected for training");
            } else {
                validationErrors.remove("images");
            }

            // Check channels
            if (!channelPanel.isValid()) {
                validationErrors.put("channels", "Invalid channel configuration");
            } else {
                validationErrors.remove("channels");
            }

            // Check classes
            long selectedClasses = classListView.getItems().stream()
                    .filter(item -> item.selected().get())
                    .count();
            if (selectedClasses < 2) {
                validationErrors.put("classes", "At least 2 classes must be selected for training");
            } else {
                validationErrors.remove("classes");
            }

            // Check weight init: CONTINUE_TRAINING requires a loaded model
            ClassifierHandler.WeightInitStrategy weightStrategy = getSelectedWeightInitStrategy();
            if (weightStrategy == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING
                    && (pretrainedModelPtPath == null || pretrainedModelPtPath.isEmpty())) {
                validationErrors.put("weightInit",
                        "Continue training requires a model -- click 'Select model...' or 'Load checkpoint...'");
            } else if (weightStrategy == ClassifierHandler.WeightInitStrategy.MAE_ENCODER
                    && (maeEncoderPathField.getText() == null || maeEncoderPathField.getText().isEmpty())) {
                validationErrors.put("weightInit",
                        "MAE encoder requires a .pt file -- click 'Browse...' to select one");
            } else {
                validationErrors.remove("weightInit");
            }

            updateErrorSummary();
        }

        private void updateErrorSummary() {
            if (errorListBox == null || errorSummaryPanel == null) return;
            if (validationErrors.isEmpty()) {
                errorSummaryPanel.setVisible(false);
                errorSummaryPanel.setManaged(false);
                okButton.setDisable(false);
            } else {
                errorListBox.getChildren().clear();
                validationErrors.forEach((fieldId, errorMsg) -> {
                    Label errorLabel = new Label("- " + errorMsg);
                    errorLabel.setStyle("-fx-text-fill: #856404;");
                    errorListBox.getChildren().add(errorLabel);
                });

                errorSummaryPanel.setVisible(true);
                errorSummaryPanel.setManaged(true);
                okButton.setDisable(true);
            }
        }

        private void refreshPieChart() {
            if (classDistributionChart == null) return;
            classDistributionChart.getData().clear();

            double totalArea = 0;
            List<ClassItem> selectedItems = new ArrayList<>();
            for (ClassItem item : classListView.getItems()) {
                if (item.selected().get() && item.annotationArea() > 0) {
                    selectedItems.add(item);
                    totalArea += item.annotationArea();
                }
            }

            if (totalArea == 0 || selectedItems.isEmpty() || !advancedMode.get()) {
                classDistributionChart.setVisible(false);
                classDistributionChart.setManaged(false);
                return;
            }

            classDistributionChart.setVisible(true);
            classDistributionChart.setManaged(true);

            for (ClassItem item : selectedItems) {
                double pct = (item.annotationArea() / totalArea) * 100.0;
                String label = String.format("%s (%.1f%%)", item.name(), pct);
                PieChart.Data data = new PieChart.Data(label, item.annotationArea());
                classDistributionChart.getData().add(data);
            }

            // Apply QuPath class colors to pie slices
            for (int i = 0; i < selectedItems.size(); i++) {
                ClassItem item = selectedItems.get(i);
                PieChart.Data data = classDistributionChart.getData().get(i);
                if (item.color() != null) {
                    int r = (item.color() >> 16) & 0xFF;
                    int g = (item.color() >> 8) & 0xFF;
                    int b = item.color() & 0xFF;
                    String style = "-fx-pie-color: rgb(" + r + "," + g + "," + b + ");";
                    // Node may not exist yet if chart hasn't been laid out
                    if (data.getNode() != null) {
                        data.getNode().setStyle(style);
                    }
                    data.nodeProperty().addListener((obs, oldNode, newNode) -> {
                        if (newNode != null) newNode.setStyle(style);
                    });
                }
            }
        }

        private void rebalanceClassWeights() {
            List<ClassItem> allItems = classListView.getItems();
            logger.info("Rebalance: classListView has {} items", allItems.size());

            if (allItems.isEmpty()) {
                Dialogs.showWarningNotification("Rebalance",
                        "No classes loaded. Click 'Load Classes from Selected Images' first.");
                return;
            }

            List<ClassItem> selected = allItems.stream()
                    .filter(item -> item.selected().get())
                    .collect(Collectors.toList());

            if (selected.isEmpty()) {
                Dialogs.showWarningNotification("Rebalance",
                        "No classes are selected. Check at least 2 classes to rebalance.");
                logger.warn("Rebalance: no selected classes");
                return;
            }

            // Log per-class areas for diagnostics
            for (ClassItem item : selected) {
                logger.info("Rebalance: class '{}' annotationArea={}", item.name(), item.annotationArea());
            }

            // Collect non-zero areas and sort for median calculation
            List<Double> areas = selected.stream()
                    .map(ClassItem::annotationArea)
                    .filter(a -> a > 0)
                    .sorted()
                    .collect(Collectors.toList());

            if (areas.isEmpty()) {
                Dialogs.showWarningNotification("Rebalance",
                        "All selected classes have zero estimated pixel coverage.\n" +
                        "Try reloading classes (coverage is estimated from annotation\n" +
                        "area or line length x stroke width).");
                logger.warn("Rebalance: all selected classes have zero coverage -- cannot compute weights. " +
                        "Are annotations point ROIs? Line/area annotations are required.");
                return;
            }

            // Compute median area
            double median;
            int n = areas.size();
            if (n % 2 == 0) {
                median = (areas.get(n / 2 - 1) + areas.get(n / 2)) / 2.0;
            } else {
                median = areas.get(n / 2);
            }
            logger.info("Rebalance: median area = {}", median);

            // Set inverse-frequency weights clamped to spinner range [0.1, 10.0]
            // The property->spinner listener in ClassListCell updates spinners automatically
            for (ClassItem item : selected) {
                double area = item.annotationArea();
                double weight;
                if (area > 0) {
                    weight = Math.max(0.1, Math.min(10.0, median / area));
                } else {
                    weight = 1.0;
                }
                item.weightMultiplier().set(weight);
            }

            // Build user-visible summary
            StringBuilder sb = new StringBuilder();
            for (ClassItem item : selected) {
                if (sb.length() > 0) sb.append(", ");
                sb.append(item.name())
                  .append("=").append(String.format("%.2f", item.weightMultiplier().get()));
            }
            String summary = sb.toString();
            logger.info("Rebalanced class weights: {}", summary);
            Dialogs.showInfoNotification("Rebalance",
                    "Weights updated: " + summary);
        }

        private TrainingDialogResult buildResult() {
            // Save dialog settings to preferences for next session
            DLClassifierPreferences.setLastArchitecture(architectureCombo.getValue());
            DLClassifierPreferences.setLastBackbone(backboneCombo.getValue());
            DLClassifierPreferences.setDefaultEpochs(epochsSpinner.getValue());
            DLClassifierPreferences.setDefaultBatchSize(batchSizeSpinner.getValue());
            DLClassifierPreferences.setDefaultLearningRate(learningRateSpinner.getValue());
            DLClassifierPreferences.setDefaultWeightDecay(weightDecaySpinner.getValue());
            DLClassifierPreferences.setDefaultDiscriminativeLrRatio(discriminativeLrSpinner.getValue());
            DLClassifierPreferences.setLastSeed(seedSpinner.getValue());
            DLClassifierPreferences.setValidationSplit(validationSplitSpinner.getValue());
            DLClassifierPreferences.setTileSize(tileSizeSpinner.getValue());
            DLClassifierPreferences.setTileOverlap(overlapSpinner.getValue());
            DLClassifierPreferences.setDefaultDownsample(parseDownsample(downsampleCombo.getValue()));
            DLClassifierPreferences.setDefaultContextScale(parseContextScale(contextScaleCombo.getValue()));
            DLClassifierPreferences.setLastLineStrokeWidth(lineStrokeWidthSpinner.getValue());
            DLClassifierPreferences.setUsePretrainedWeights(
                    getSelectedWeightInitStrategy() == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);
            DLClassifierPreferences.setAugFlipHorizontal(flipHorizontalCheck.isSelected());
            DLClassifierPreferences.setAugFlipVertical(flipVerticalCheck.isSelected());
            DLClassifierPreferences.setAugRotation(rotationCheck.isSelected());
            DLClassifierPreferences.setAugIntensityMode(mapIntensityModeFromDisplay(intensityAugCombo.getValue()));
            DLClassifierPreferences.setAugElasticDeform(elasticCheck.isSelected());

            // Save training strategy preferences
            DLClassifierPreferences.setDefaultScheduler(mapSchedulerFromDisplay(schedulerCombo.getValue()));
            DLClassifierPreferences.setDefaultLossFunction(mapLossFunctionFromDisplay(lossFunctionCombo.getValue()));
            DLClassifierPreferences.setDefaultEarlyStoppingMetric(
                    mapEarlyStoppingMetricFromDisplay(earlyStoppingMetricCombo.getValue()));
            DLClassifierPreferences.setDefaultEarlyStoppingPatience(earlyStoppingPatienceSpinner.getValue());
            DLClassifierPreferences.setDefaultMixedPrecision(mixedPrecisionCheck.isSelected());
            DLClassifierPreferences.setDefaultGradientAccumulation(gradientAccumulationSpinner.getValue());
            DLClassifierPreferences.setDefaultOhemHardPixelPct(ohemSpinner.getValue());
            DLClassifierPreferences.setDefaultOhemHardPixelStartPct(ohemStartSpinner.getValue());
            DLClassifierPreferences.setDefaultOhemSchedule(
                    ohemStartSpinner.getValue() > ohemSpinner.getValue() ? "anneal" : "fixed");
            DLClassifierPreferences.setDefaultOhemAdaptiveFloor(ohemAdaptiveFloorCheck.isSelected());
            DLClassifierPreferences.setDefaultFocalGamma(focalGammaSpinner.getValue());
            DLClassifierPreferences.setDefaultBoundarySigma(boundarySigmaSpinner.getValue());
            DLClassifierPreferences.setDefaultBoundaryWMin(boundaryWMinSpinner.getValue());
            DLClassifierPreferences.setDefaultProgressiveResize(progressiveResizeCheck.isSelected());
            DLClassifierPreferences.setDefaultFocusClass(
                    mapFocusClassFromDisplay(focusClassCombo.getValue()));
            DLClassifierPreferences.setDefaultFocusClassMinIoU(focusClassMinIoUSpinner.getValue());

            // Build training config from unified weight init strategy
            TrainingConfig trainingConfig = buildTrainingConfig();

            // Run pairwise interaction checks against the built config.
            // BLOCKING watchers (e.g. overlap + no-per-image-split-roles)
            // return the user to the dialog if they pick Cancel; INFO/WARN
            // watchers show a single popup they can dismiss or suppress
            // per-warning.
            var interactionWarnings = qupath.ext.dlclassifier.service
                    .warnings.InteractionWarningService.evaluate(trainingConfig);
            var visibleInteractionWarnings = qupath.ext.dlclassifier.service
                    .warnings.InteractionWarningService.filterVisible(
                            interactionWarnings);
            if (!visibleInteractionWarnings.isEmpty()) {
                boolean proceed = qupath.ext.dlclassifier.service
                        .warnings.InteractionWarningService.showIfAny(
                                visibleInteractionWarnings, dialog);
                if (!proceed) {
                    return null;
                }
            }

            // Get channel config
            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            // Validate channel count against MAE encoder requirement
            if (maeEncoderInputChannels > 0
                    && getSelectedWeightInitStrategy() == ClassifierHandler.WeightInitStrategy.MAE_ENCODER) {
                int selectedChannels = channelConfig.getNumChannels();
                if (selectedChannels != maeEncoderInputChannels) {
                    Dialogs.showErrorMessage("Channel Mismatch",
                            String.format("The MAE encoder was pretrained with %d channels "
                                    + "but %d channels are currently selected.\n\n"
                                    + "Change the channel selection to match, or choose "
                                    + "a different weight initialization strategy.",
                                    maeEncoderInputChannels, selectedChannels));
                    return null;
                }
            }

            // Pre-flight VRAM estimation: warn INSIDE the dialog so the user can
            // adjust settings (tile size, batch, downsample) without losing their work.
            try {
                ApposeService appose = ApposeService.getInstance();
                if (appose.isAvailable() && "cuda".equals(appose.getGpuType())) {
                    var gpuTask = appose.runTask("health_check", java.util.Map.of());
                    int totalMb = ((Number) gpuTask.outputs.get("gpu_memory_mb")).intValue();
                    if (totalMb > 0) {
                        String modelType = trainingConfig.getModelType();
                        String backbone = trainingConfig.getBackbone();
                        int tileSize = trainingConfig.getTileSize();
                        int batchSize = trainingConfig.getBatchSize();
                        int gradAccum = trainingConfig.getGradientAccumulationSteps();
                        boolean mixedPrec = trainingConfig.isMixedPrecision();
                        int contextScale = trainingConfig.getContextScale();

                        // Backbone-aware model size estimates (MB of parameters)
                        double modelMb = estimateModelSizeMb(modelType, backbone);
                        double actMultiplier = "muvit".equals(modelType) ? 10.0 : 4.0;
                        // Mixed precision roughly halves activation/gradient memory
                        if (mixedPrec) actMultiplier *= 0.6;
                        // Context scale enlarges tiles via padding AND doubles channels
                        int effectiveTile = tileSize;
                        if (contextScale > 1) {
                            effectiveTile = tileSize + 2 * (tileSize / contextScale);
                            actMultiplier *= 1.1;
                        }

                        double areaScale = (double)(effectiveTile * effectiveTile) / (256.0 * 256.0);
                        double estimatedMb = modelMb * (1 + 3 + actMultiplier * areaScale * batchSize);
                        double budgetMb = totalMb * 0.85;

                        if (estimatedMb > budgetMb) {
                            // Find the max batch size that fits at current tile size
                            int maxBatchAtTile = 0;
                            for (int b = batchSize; b >= 1; b--) {
                                double est = modelMb * (1 + 3 + actMultiplier * areaScale * b);
                                if (est <= budgetMb) { maxBatchAtTile = b; break; }
                            }

                            int effectiveBatch = batchSize * gradAccum;
                            StringBuilder suggestions = new StringBuilder();
                            suggestions.append(String.format(
                                    "Estimated VRAM: %.0f MB (GPU has %d MB, ~%.0f MB usable).\n"
                                    + "Current: %s (%s), %dx%d tiles, batch %d"
                                    + (gradAccum > 1 ? " (x%d accum = %d effective)" : "")
                                    + ".\n\n",
                                    estimatedMb, totalMb, budgetMb,
                                    modelType, backbone, tileSize, tileSize,
                                    batchSize, gradAccum, effectiveBatch));

                            if (maxBatchAtTile > 0) {
                                // Can fit at current tile size with smaller batch
                                int suggestedAccum = Math.max(1,
                                        (int) Math.ceil((double) effectiveBatch / maxBatchAtTile));
                                double estFit = modelMb * (1 + 3 + actMultiplier * areaScale * maxBatchAtTile);
                                suggestions.append("Suggested settings that fit in VRAM:\n");
                                suggestions.append(String.format(
                                        "  - Batch size %d with gradient accumulation %d "
                                        + "(effective batch %d, ~%.0f MB)\n",
                                        maxBatchAtTile, suggestedAccum,
                                        maxBatchAtTile * suggestedAccum, estFit));
                            } else {
                                // Even batch=1 doesn't fit at this tile size
                                suggestions.append("Even batch size 1 exceeds VRAM at this tile size.\n");
                                suggestions.append("Suggested settings that fit in VRAM:\n");
                            }

                            // Suggest smaller tile sizes if needed
                            for (int candidate : new int[]{512, 384, 256, 128}) {
                                if (candidate >= tileSize) continue;
                                int candEffective = contextScale > 1
                                        ? candidate + 2 * (candidate / contextScale) : candidate;
                                double candArea = (double)(candEffective * candEffective) / (256.0 * 256.0);
                                // Find max batch at this tile size
                                int candMaxBatch = 0;
                                for (int b = batchSize; b >= 1; b--) {
                                    double bEst = modelMb * (1 + 3 + actMultiplier * candArea * b);
                                    if (bEst <= budgetMb) { candMaxBatch = b; break; }
                                }
                                if (candMaxBatch > 0) {
                                    int candAccum = Math.max(1,
                                            (int) Math.ceil((double) effectiveBatch / candMaxBatch));
                                    double candEst = modelMb * (1 + 3 + actMultiplier * candArea * candMaxBatch);
                                    suggestions.append(String.format(
                                            "  - %dpx tiles, batch %d x%d accum "
                                            + "(effective %d, ~%.0f MB)\n",
                                            candidate, candMaxBatch, candAccum,
                                            candMaxBatch * candAccum, candEst));
                                    break; // Show best fitting alternative
                                }
                            }

                            // Note about MAE pretraining tile size if applicable
                            if (maeEncoderTileSize > 0 && tileSize != maeEncoderTileSize) {
                                suggestions.append(String.format(
                                        "\nNote: MAE encoder was pretrained at %dpx -- "
                                        + "using the same tile size\ngives best weight transfer.\n",
                                        maeEncoderTileSize));
                            }

                            suggestions.append("\nGradient accumulation simulates larger batches "
                                    + "without extra VRAM.\n"
                                    + "Increase downsample to compensate for smaller tiles.\n\n"
                                    + "Go back and adjust settings?");

                            boolean goBack = Dialogs.showConfirmDialog("VRAM Warning",
                                    suggestions.toString());
                            if (goBack) {
                                return null;  // Stay in dialog so user can adjust
                            }
                        }
                    }
                }
            } catch (Exception e) {
                logger.debug("Could not estimate VRAM usage: {}", e.getMessage());
            }

            // Get selected classes
            List<String> selectedClasses = classListView.getItems().stream()
                    .filter(item -> item.selected().get())
                    .map(ClassItem::name)
                    .collect(Collectors.toList());

            // Always collect selected images from the list
            List<ProjectImageEntry<BufferedImage>> selectedImages = imageSelectionList.getItems().stream()
                    .filter(item -> item.selected.get())
                    .map(item -> item.entry)
                    .collect(Collectors.toList());

            // Collect image-level split role assignments (advanced mode feature).
            // Train-only and val-only images bypass the stratified splitter.
            Set<String> trainOnlyImages = new LinkedHashSet<>();
            Set<String> valOnlyImages = new LinkedHashSet<>();
            for (ImageSelectionItem item : imageSelectionList.getItems()) {
                if (!item.selected.get()) continue;
                SplitRole role = item.splitRole.get();
                if (role == SplitRole.TRAIN_ONLY) {
                    trainOnlyImages.add(item.entry.getImageName());
                } else if (role == SplitRole.VAL_ONLY) {
                    valOnlyImages.add(item.entry.getImageName());
                }
            }

            // Warn if ALL selected images are val-only (no training data)
            if (!valOnlyImages.isEmpty() && trainOnlyImages.size() + (selectedImages.size() - trainOnlyImages.size() - valOnlyImages.size()) == 0) {
                // Every non-val image is train-only or there are no "Both" images;
                // but more critically, if ALL are val-only there is nothing to train on.
                long bothCount = imageSelectionList.getItems().stream()
                        .filter(i -> i.selected.get() && i.splitRole.get() == SplitRole.BOTH)
                        .count();
                if (bothCount == 0 && trainOnlyImages.isEmpty()) {
                    Dialogs.showWarningNotification("Split Role Warning",
                            "All selected images are set to 'Val'. "
                            + "There will be no training data. "
                            + "Please set at least one image to 'Train' or 'Both'.");
                    return null;
                }
            }

            // Collect handler-specific parameters (e.g., MuViT architecture config)
            Map<String, Object> handlerParams = currentHandlerUI != null
                    ? currentHandlerUI.getParameters()
                    : Map.of();

            // Extract class colors from selected class items
            Map<String, Integer> classColors = new LinkedHashMap<>();
            for (ClassItem item : classListView.getItems()) {
                if (item.selected().get() && item.color() != null) {
                    classColors.put(item.name(), item.color());
                }
            }

            return new TrainingDialogResult(
                    classifierNameField.getText().trim(),
                    descriptionField.getText().trim(),
                    trainingConfig,
                    channelConfig,
                    selectedClasses,
                    selectedImages,
                    classColors,
                    handlerParams,
                    trainOnlyImages,
                    valOnlyImages
            );
        }

        private Map<String, Boolean> buildAugmentationConfig() {
            Map<String, Boolean> config = new LinkedHashMap<>();
            config.put("flip_horizontal", flipHorizontalCheck.isSelected());
            config.put("flip_vertical", flipVerticalCheck.isSelected());
            config.put("rotation_90", rotationCheck.isSelected());
            config.put("elastic_deformation", elasticCheck.isSelected());
            return config;
        }

        /**
         * Maps intensity augmentation mode from display string to config value.
         */
        private static String mapIntensityModeFromDisplay(String display) {
            if (display == null) return "none";
            if (display.startsWith("Brightfield")) return "brightfield";
            if (display.startsWith("Fluorescence")) return "fluorescence";
            return "none";
        }

        /**
         * Maps intensity augmentation mode from config value to display string.
         */
        private static String mapIntensityModeToDisplay(String mode) {
            if (mode == null) return "None";
            return switch (mode) {
                case "brightfield" -> "Brightfield (color jitter)";
                case "fluorescence" -> "Fluorescence (per-channel)";
                default -> "None";
            };
        }

        /**
         * Parses downsample value from ComboBox display string.
         */
        private static double parseDownsample(String displayValue) {
            if (displayValue == null) return 1.0;
            if (displayValue.startsWith("16x")) return 16.0;
            if (displayValue.startsWith("8x")) return 8.0;
            if (displayValue.startsWith("4x")) return 4.0;
            if (displayValue.startsWith("2x")) return 2.0;
            return 1.0;
        }

        /**
         * Parses context scale value from ComboBox display string.
         */
        private static int parseContextScale(String displayValue) {
            if (displayValue == null) return 1;
            if (displayValue.startsWith("16x")) return 16;
            if (displayValue.startsWith("8x")) return 8;
            if (displayValue.startsWith("4x")) return 4;
            if (displayValue.startsWith("2x")) return 2;
            return 1;
        }

        // ==================== VRAM Estimation ====================

        /**
         * Caches the GPU total memory from the Appose health check.
         * Runs on a background thread to avoid blocking the FX thread,
         * then updates the VRAM estimate label when done.
         */
        private void cacheGpuMemory() {
            try {
                ApposeService appose = ApposeService.getInstance();
                if (appose.isAvailable() && "cuda".equals(appose.getGpuType())) {
                    // Use cached GPU memory from the combined verification/health
                    // task that ran during initialize() -- no separate task needed.
                    gpuTotalMb = appose.getLastGpuMemoryMb();
                } else if (appose.isAvailable() && mixedPrecisionCheck != null) {
                    // Mixed precision only works on CUDA GPUs -- disable the
                    // checkbox so the user doesn't configure a setting that
                    // will be silently ignored at training time.
                    String device = appose.getGpuType();
                    Platform.runLater(() -> {
                        mixedPrecisionCheck.setSelected(false);
                        mixedPrecisionCheck.setDisable(true);
                        TooltipHelper.install(mixedPrecisionCheck,
                                "Mixed precision requires an NVIDIA CUDA GPU.\n" +
                                "Current device: " + device +
                                " -- this setting has no effect.");
                    });
                }
            } catch (Exception e) {
                logger.debug("Could not cache GPU memory: {}", e.getMessage());
            }
        }

        /**
         * Updates the tile-settings advisory label based on current tile
         * size and overlap. Warns the user about small tiles (slow/low
         * edge context), low overlap (seams), high overlap (wasted
         * compute), and overlap large enough to force stride=0 at
         * inference. The values are never silently rewritten -- this is
         * pure advice, mirroring the VRAM estimate directly above.
         */
        private void updateTileAdvisory() {
            if (tileAdvisoryLabel == null || tileSizeSpinner == null
                    || overlapSpinner == null) return;
            int tileSize = tileSizeSpinner.getValue();
            int overlapPct = overlapSpinner.getValue();
            // Spinner is percentage (0-50); checkTileSettings expects pixels
            int overlap = (int) Math.round(tileSize * overlapPct / 100.0);
            String advisory = qupath.ext.dlclassifier.model.InferenceConfig
                    .checkTileSettings(tileSize, overlap);
            if (advisory == null) {
                tileAdvisoryLabel.setVisible(false);
                tileAdvisoryLabel.setManaged(false);
                tileAdvisoryLabel.setText("");
                return;
            }
            String color = overlap >= tileSize / 2 ? "#CC0000"  // red: stride<=0
                    : "#CC7A00";                                 // orange: suboptimal
            tileAdvisoryLabel.setStyle(
                    "-fx-font-size: 11px; -fx-text-fill: " + color + ";"
                    + " -fx-font-weight: bold;");
            tileAdvisoryLabel.setText(advisory);
            tileAdvisoryLabel.setVisible(true);
            tileAdvisoryLabel.setManaged(true);
        }

        /**
         * Updates the live VRAM estimate label based on current dialog settings.
         * Uses the same estimation formula as the pre-flight check but runs
         * instantly from cached GPU info.
         */
        private void updateVramEstimate() {
            if (vramEstimateLabel == null) return;

            // Hide if no GPU info or controls not yet initialized
            if (gpuTotalMb <= 0 || architectureCombo == null || backboneCombo == null
                    || tileSizeSpinner == null || batchSizeSpinner == null) {
                vramEstimateLabel.setText("");
                return;
            }

            try {
                String modelType = architectureCombo.getValue();
                String backbone = backboneCombo.getValue();
                int tileSize = tileSizeSpinner.getValue();
                int batchSize = batchSizeSpinner.getValue();
                boolean mixedPrec = mixedPrecisionCheck != null && mixedPrecisionCheck.isSelected();
                int contextScale = contextScaleCombo != null
                        ? parseContextScale(contextScaleCombo.getValue()) : 1;

                double modelMb = estimateModelSizeMb(modelType, backbone);

                // Context scale enlarges tiles via padding AND doubles channels.
                int effectiveTile = tileSize;
                if (contextScale > 1) {
                    effectiveTile = tileSize + 2 * (tileSize / contextScale);
                }

                double estimatedMb;
                if ("tiny-unet".equals(modelType)) {
                    // TinyUNet activations are driven by base channels x spatial
                    // area x batch -- NOT by param count.  Every encoder/decoder
                    // stage stores its feature map for backward, and BRN adds
                    // three (x_hat, weight-broadcast, bias-broadcast) per layer,
                    // so the integer factor is large.  Empirical at B=64,
                    // tile=512, base=16, bf16: ~45 GB OOM on a 24 GB 3090 ->
                    // peak usage about 100x * base * H * W * bytes * batch.
                    // See agent B1/B2 reports and
                    // claude-reports/2026-04-17_input-size-divisibility.md.
                    int base = tinyUnetBase(backbone);
                    double bytesPerElem = mixedPrec ? 2.0 : 4.0;
                    double actBytes = (double) batchSize
                            * base
                            * effectiveTile * effectiveTile
                            * bytesPerElem
                            * 100.0;
                    // Model + gradients + Adam state: tiny (< 20 MB even for
                    // small-24x4) but include for completeness.
                    estimatedMb = actBytes / (1024.0 * 1024.0) + 5.0 * modelMb;
                } else {
                    // Pretrained encoders (UNet / Fast Pretrained / MuViT).
                    // Here model weights dominate deeper in the encoder, so a
                    // param-proportional multiplier is a reasonable first cut.
                    double actMultiplier = "muvit".equals(modelType) ? 10.0 : 4.0;
                    if (mixedPrec) actMultiplier *= 0.6;
                    if (contextScale > 1) actMultiplier *= 1.1;
                    double areaScale = (double)(effectiveTile * effectiveTile)
                            / (256.0 * 256.0);
                    estimatedMb = modelMb
                            * (1 + 3 + actMultiplier * areaScale * batchSize);
                }

                double budgetMb = gpuTotalMb * 0.85;

                double pct = (estimatedMb / gpuTotalMb) * 100;

                String tileNote = effectiveTile != tileSize
                        ? String.format(" [%dpx with context padding]", effectiveTile) : "";
                String text = String.format("Est. VRAM: ~%.0f MB / %,d MB (%.0f%%)%s",
                        estimatedMb, gpuTotalMb, pct, tileNote);

                if (estimatedMb > budgetMb) {
                    // Exceeds safe budget -- red warning
                    vramEstimateLabel.setStyle(
                            "-fx-font-size: 11px; -fx-text-fill: #CC0000; -fx-font-weight: bold;");
                    // Find max batch that fits by linearly scaling the
                    // activation term.  Both estimators above are linear in
                    // batchSize so divide the "above budget" excess out.
                    double perBatchMb = (estimatedMb - 5.0 * modelMb) / batchSize;
                    int maxBatch = 0;
                    for (int b = batchSize; b >= 1; b--) {
                        double est = 5.0 * modelMb + perBatchMb * b;
                        if (est <= budgetMb) { maxBatch = b; break; }
                    }
                    if (maxBatch > 0) {
                        text += String.format("  --  EXCEEDS GPU! Try batch %d or smaller tiles", maxBatch);
                    } else {
                        text += "  --  EXCEEDS GPU! Reduce tile size";
                    }
                } else if (pct > 75) {
                    // Tight -- orange warning
                    vramEstimateLabel.setStyle(
                            "-fx-font-size: 11px; -fx-text-fill: #CC7A00; -fx-font-weight: bold;");
                    text += "  --  tight, may OOM with large augmentations";
                } else {
                    // OK -- normal color
                    vramEstimateLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #228B22;");
                }

                vramEstimateLabel.setText(text);
            } catch (Exception e) {
                vramEstimateLabel.setText("");
                logger.debug("VRAM estimate update failed: {}", e.getMessage());
            }
        }

        /**
         * Estimates model parameter size in MB based on architecture and backbone.
         * These are approximate values for the model weights only -- optimizer state
         * and activations are accounted for by multipliers in the caller.
         */
        /**
         * Base-channel count for a Tiny UNet preset. Activation memory at
         * training time scales linearly with this, so the VRAM estimator
         * reads it directly rather than guessing via param count.
         */
        private static int tinyUnetBase(String backbone) {
            if (backbone == null) return 16;
            return switch (backbone) {
                case "nano-8x3"     -> 8;
                case "compact-16x3" -> 16;
                case "tiny-16x4"    -> 16;
                case "small-24x4"   -> 24;
                default             -> 16;
            };
        }

        private static double estimateModelSizeMb(String modelType, String backbone) {
            if ("muvit".equals(modelType)) return 140.0;
            if ("tiny-unet".equals(modelType)) {
                // Per-preset parameter footprint (weights + grads + Adam state ~4x).
                if (backbone == null) return 1.5;
                return switch (backbone) {
                    case "nano-8x3"     -> 0.4;
                    case "compact-16x3" -> 0.8;
                    case "tiny-16x4"    -> 1.5;
                    case "small-24x4"   -> 3.5;
                    default             -> 1.5;
                };
            }
            if (backbone == null) return 30.0;
            // Approximate parameter counts (MB) for common backbones in a UNet decoder.
            // Values include both encoder and decoder parameters.
            return switch (backbone.toLowerCase()) {
                // ResNet family
                case "resnet18" -> 47.0;   // ~11.7M params
                case "resnet34" -> 87.0;   // ~21.8M params
                case "resnet50" -> 100.0;  // ~25.6M params
                case "resnet101" -> 170.0; // ~44.5M params
                case "resnet152" -> 230.0; // ~60.2M params
                // EfficientNet family
                case "efficientnet-b0" -> 21.0;  // ~5.3M params
                case "efficientnet-b1" -> 31.0;  // ~7.8M params
                case "efficientnet-b2" -> 36.0;  // ~9.1M params
                case "efficientnet-b3" -> 48.0;  // ~12M params
                case "efficientnet-b4" -> 76.0;  // ~19.3M params
                case "efficientnet-b5" -> 120.0; // ~30.4M params
                // DenseNet family
                case "densenet121" -> 32.0;  // ~8M params
                case "densenet169" -> 56.0;  // ~14.1M params
                case "densenet201" -> 80.0;  // ~20M params
                // MobileNet
                case "mobilenet_v2" -> 14.0;       // ~3.5M params
                case "timm-mobilenetv3_large_100" -> 22.0; // ~5.4M params
                // Histology-pretrained ResNet-50 variants
                case "resnet50_lunit-swav", "resnet50_lunit-bt",
                     "resnet50_kather100k", "resnet50_tcga-brca" -> 100.0; // ~25.6M params
                // Foundation models (large ViT encoders + UNet decoder)
                case "h-optimus-0", "midnight" -> 4400.0;  // ~1.1B params ViT-G
                case "virchow" -> 2500.0;                   // ~632M params ViT-H
                case "hibou-l", "dinov2-large" -> 1200.0;   // ~304M params ViT-L
                case "hibou-b" -> 350.0;                     // ~86M params ViT-B
                // Other pathology encoders
                case "uni", "conch", "phikon" -> 350.0;      // ~86M+ params
                // Default for unknown backbones
                default -> {
                    // Heuristic: if the name contains "50" or "101", assume larger
                    if (backbone.contains("50")) yield 100.0;
                    if (backbone.contains("101")) yield 170.0;
                    if (backbone.contains("optimus") || backbone.contains("midnight")) yield 4400.0;
                    if (backbone.contains("virchow")) yield 2500.0;
                    if (backbone.contains("hibou") || backbone.contains("dinov2")) yield 1200.0;
                    yield 50.0; // Conservative default
                }
            };
        }

        // ==================== Display/Value Mapping Helpers ====================

        private static String mapSchedulerToDisplay(String value) {
            if (value == null) return "One Cycle";
            return switch (value) {
                case "onecycle" -> "One Cycle";
                case "cosine" -> "Cosine Annealing";
                case "plateau" -> "Reduce on Plateau";
                case "step" -> "Step Decay";
                case "none" -> "None";
                default -> "One Cycle";
            };
        }

        private static String mapSchedulerFromDisplay(String display) {
            if (display == null) return "onecycle";
            return switch (display) {
                case "One Cycle" -> "onecycle";
                case "Cosine Annealing" -> "cosine";
                case "Reduce on Plateau" -> "plateau";
                case "Step Decay" -> "step";
                case "None" -> "none";
                default -> "onecycle";
            };
        }

        private static String mapLossFunctionToDisplay(String value) {
            return switch (value) {
                case "cross_entropy" -> "Cross Entropy";
                case "focal_dice" -> "Focal + Dice";
                case "focal" -> "Focal";
                case "boundary_ce" -> "Boundary-softened CE";
                case "boundary_ce_dice" -> "Boundary-softened CE + Dice";
                case "lovasz" -> "Lovasz-Softmax";
                case "ce_lovasz" -> "CE + Lovasz-Softmax";
                default -> "Cross Entropy + Dice";
            };
        }

        private static String mapLossFunctionFromDisplay(String display) {
            return switch (display) {
                case "Cross Entropy" -> "cross_entropy";
                case "Focal + Dice" -> "focal_dice";
                case "Focal" -> "focal";
                case "Boundary-softened CE" -> "boundary_ce";
                case "Boundary-softened CE + Dice" -> "boundary_ce_dice";
                case "Lovasz-Softmax" -> "lovasz";
                case "CE + Lovasz-Softmax" -> "ce_lovasz";
                default -> "ce_dice";
            };
        }

        private static boolean isFocalLossSelected(String display) {
            return "Focal + Dice".equals(display) || "Focal".equals(display);
        }

        private static boolean isBoundaryLossSelected(String display) {
            return "Boundary-softened CE".equals(display)
                    || "Boundary-softened CE + Dice".equals(display);
        }

        private static String mapEarlyStoppingMetricToDisplay(String value) {
            if ("val_loss".equals(value)) return "Validation Loss";
            if ("disabled".equals(value)) return "Disabled";
            return "Mean IoU";
        }

        private static String mapEarlyStoppingMetricFromDisplay(String display) {
            if ("Validation Loss".equals(display)) return "val_loss";
            if ("Disabled".equals(display)) return "disabled";
            return "mean_iou";
        }

        private static String mapDownsampleToDisplay(double value) {
            if (value >= 16.0) return "16x (1/16 resolution)";
            if (value >= 8.0) return "8x (1/8 resolution)";
            if (value >= 4.0) return "4x (Quarter resolution)";
            if (value >= 2.0) return "2x (Half resolution)";
            return "1x (Full resolution)";
        }

        private static String mapContextScaleToDisplay(int value) {
            return switch (value) {
                case 16 -> "16x context";
                case 8 -> "8x context";
                case 4 -> "4x context (Recommended)";
                case 2 -> "2x context";
                default -> "None (single scale)";
            };
        }

        private void copyTrainingScript(Button sourceButton) {
            String name = classifierNameField.getText().trim();
            if (name.isEmpty()) {
                showCopyFeedback(sourceButton, "Enter a classifier name first");
                return;
            }

            TrainingConfig config = buildTrainingConfig();
            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            List<String> selectedClasses = classListView.getItems().stream()
                    .filter(item -> item.selected().get())
                    .map(ClassItem::name)
                    .collect(Collectors.toList());

            String script = ScriptGenerator.generateTrainingScript(
                    name, descriptionField.getText().trim(),
                    config, channelConfig, selectedClasses);

            Clipboard clipboard = Clipboard.getSystemClipboard();
            ClipboardContent content = new ClipboardContent();
            content.putString(script);
            clipboard.setContent(content);

            showCopyFeedback(sourceButton, "Script copied to clipboard!");
        }

        private void showCopyFeedback(Button button, String message) {
            Tooltip tooltip = new Tooltip(message);
            tooltip.setAutoHide(true);
            tooltip.show(button,
                    button.localToScreen(button.getBoundsInLocal()).getMinX(),
                    button.localToScreen(button.getBoundsInLocal()).getMinY() - 30);
            PauseTransition pause = new PauseTransition(Duration.seconds(2));
            pause.setOnFinished(e -> tooltip.hide());
            pause.play();
        }

        /**
         * Shows a simplified guide for basic mode users explaining just the
         * three ResNet size options and general guidance.
         */
        private void showBasicArchitectureGuide() {
            Dialog<Void> dialog = new Dialog<>();
            dialog.setTitle("Choosing a Model Size");
            dialog.setHeaderText("Which model size should I pick?");
            dialog.getDialogPane().getButtonTypes().add(ButtonType.CLOSE);
            dialog.setResizable(true);

            VBox content = new VBox(12);
            content.setPadding(new Insets(10));
            content.setPrefWidth(480);

            Label intro = new Label(
                    "In basic mode, you choose between three model sizes. All three "
                    + "use the same proven UNet architecture -- they differ only in "
                    + "capacity (how much the model can learn).");
            intro.setWrapText(true);
            content.getChildren().add(intro);

            content.getChildren().add(createModelEntry(
                    "Small (ResNet-18)",
                    "Fastest training and inference. Uses the least GPU memory (~2-3 GB). "
                    + "Good starting point when you have a small dataset (< 50 training "
                    + "tiles) or want quick iterations to test your annotations. "
                    + "May underperform on complex tasks with many classes.",
                    null, null));

            content.getChildren().add(createModelEntry(
                    "Medium (ResNet-34) -- Recommended",
                    "Best balance of speed and accuracy for most tasks. Moderate GPU "
                    + "memory (~3-5 GB). This is the default and works well for "
                    + "most histology segmentation tasks. Start here unless you "
                    + "have a reason to choose otherwise.",
                    null, null));

            content.getChildren().add(createModelEntry(
                    "Large (ResNet-50)",
                    "Most learning capacity. Needs more GPU memory (~5-8 GB) and "
                    + "benefits from larger datasets (100+ training tiles). "
                    + "Choose this when Medium is not capturing enough detail, "
                    + "or for complex tasks with many tissue classes.",
                    null, null));

            content.getChildren().add(createSectionHeader("Tips"));
            Label tips = new Label(
                    "- Start with Medium. Only switch if results are unsatisfactory.\n"
                    + "- More training data generally matters more than a larger model.\n"
                    + "- If training is very slow, try Small first to verify your "
                    + "annotations are correct, then retrain with Medium.\n"
                    + "- Switch to 'All Settings' mode for access to histology-pretrained "
                    + "encoders, foundation models, and additional architectures.");
            tips.setWrapText(true);
            content.getChildren().add(tips);

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setPrefHeight(400);
            dialog.getDialogPane().setContent(scrollPane);
            dialog.getDialogPane().setPrefWidth(520);
            dialog.showAndWait();
        }

        /**
         * Shows a guide dialog explaining available model architectures and
         * encoders with links to their associated papers.
         */
        private void showArchitectureGuide() {
            Dialog<Void> dialog = new Dialog<>();
            dialog.setTitle("Model Architecture Guide");
            dialog.setHeaderText("Choosing an architecture and encoder for your task");
            dialog.getDialogPane().getButtonTypes().add(ButtonType.CLOSE);
            dialog.setResizable(true);

            VBox content = new VBox(12);
            content.setPadding(new Insets(10));
            content.setPrefWidth(560);

            // --- Architectures ---
            content.getChildren().add(createSectionHeader("Segmentation Architectures"));

            content.getChildren().add(createModelEntry(
                    "UNet (Recommended Default)",
                    "Symmetric encoder-decoder with skip connections. The most widely used "
                    + "architecture for biomedical image segmentation. Works well for most "
                    + "tasks including H&E gland segmentation, cell detection, and tissue "
                    + "classification. Choose this unless you have a specific reason not to.",
                    "Ronneberger et al. 2015",
                    "https://arxiv.org/abs/1505.04597"));

            content.getChildren().add(createModelEntry(
                    "MuViT (Multi-resolution Vision Transformer)",
                    "Transformer-based architecture with multi-scale feature fusion. "
                    + "Supports self-supervised MAE pretraining on your own unlabeled data. "
                    + "May outperform UNet on tasks requiring long-range context, but "
                    + "requires more VRAM and training time.",
                    null, null));

            content.getChildren().add(createModelEntry(
                    "Tiny UNet (Lightweight, No Pretrained Weights)",
                    "A minimal UNet variant with configurable depth and width. "
                    + "Trains from scratch (no pretrained encoder) so it works with any "
                    + "number of input channels. Four size presets: Nano (~10K params, "
                    + "for simple 2-class tasks), Tiny (~138K, default), Compact (~36K), "
                    + "and Small (~305K). Good for quick experiments or when standard "
                    + "encoders are too large for your GPU.",
                    null, null));

            content.getChildren().add(createModelEntry(
                    "Fast Pretrained (Small RGB Models)",
                    "Lightweight UNet with mobile-optimized ImageNet encoders. "
                    + "Designed for fast training and inference with small GPU memory "
                    + "footprint. Two encoder options: EfficientNet-Lite0 (~4.2M params, "
                    + "recommended) and MobileNetV3-Small (~2.0M params, fastest). "
                    + "Good middle ground between Tiny UNet (no pretraining) and full "
                    + "UNet (heavier encoders).",
                    null, null));

            // --- Standard Encoders ---
            content.getChildren().add(createSectionHeader("Standard Encoders (ImageNet-pretrained)"));

            content.getChildren().add(createModelEntry(
                    "ResNet-34 (Recommended Default)",
                    "Good balance of speed, accuracy, and memory usage. The best starting "
                    + "point for most tasks. 21.8M parameters.",
                    "He et al. 2016",
                    "https://arxiv.org/abs/1512.03385"));

            content.getChildren().add(createModelEntry(
                    "ResNet-50 / ResNet-101",
                    "More capacity than ResNet-34. Use for larger datasets or complex "
                    + "tasks where ResNet-34 plateaus. 25.6M / 44.5M parameters.",
                    "He et al. 2016",
                    "https://arxiv.org/abs/1512.03385"));

            content.getChildren().add(createModelEntry(
                    "EfficientNet-B0 / B1 / B2",
                    "Compound-scaled networks, lighter than ResNets. Good for "
                    + "low-VRAM GPUs or when inference speed is critical. B0 is the "
                    + "smallest (5.3M params), B2 the largest (9.2M params).",
                    "Tan & Le 2019",
                    "https://arxiv.org/abs/1905.11946"));

            content.getChildren().add(createModelEntry(
                    "MobileNet-V2",
                    "Very lightweight encoder designed for mobile/edge deployment. "
                    + "Fastest inference of the standard encoders (~3.5M params). "
                    + "Good when inference speed is the top priority.",
                    "Sandler et al. 2018",
                    "https://arxiv.org/abs/1801.04381"));

            // --- Histology Encoders ---
            content.getChildren().add(createSectionHeader(
                    "Histology-Pretrained Encoders (Best for H&E)"));

            content.getChildren().add(new Label(
                    "These ResNet-50 encoders were self-supervised on millions of H&E "
                    + "tissue patches at 20x magnification. They already understand tissue "
                    + "morphology, which gives a significant head start over ImageNet weights "
                    + "for histopathology tasks."));

            content.getChildren().add(createModelEntry(
                    "Lunit SwAV / Lunit Barlow Twins",
                    "Trained on 30M+ H&E patches from TCGA using SwAV or Barlow Twins "
                    + "self-supervised learning. Among the best-performing histology encoders "
                    + "for downstream tasks. Non-commercial license.",
                    "Kang et al. 2023",
                    "https://doi.org/10.1038/s41591-023-02512-1"));

            content.getChildren().add(createModelEntry(
                    "Kather100K",
                    "Trained on 100K H&E patches across 9 tissue types from the NCT-CRC "
                    + "colorectal cancer dataset. Good for colorectal tissue analysis.",
                    "Kather et al. 2019",
                    "https://doi.org/10.1038/s41591-019-0462-y"));

            content.getChildren().add(createModelEntry(
                    "TCGA-BRCA",
                    "SimCLR self-supervised training on TCGA breast cancer slides. "
                    + "Specifically tuned for breast tissue morphology.",
                    null, null));

            // --- Foundation Models ---
            content.getChildren().add(createSectionHeader(
                    "Foundation Models (Large-Scale, Downloaded On-Demand)"));

            content.getChildren().add(new Label(
                    "Large vision transformers (86M-1.1B parameters) trained on millions "
                    + "of pathology or natural images. Powerful but require more VRAM and "
                    + "download 200MB-2GB on first use. Best when you have limited labeled data."));

            content.getChildren().add(createModelEntry(
                    "H-optimus-0 (Bioptimus) -- GATED",
                    "ViT-G pathology foundation model trained on 500K+ whole slide images. "
                    + "1.1B parameters, 1536-dim features. Apache 2.0 license. "
                    + "Requires HuggingFace token (see below).",
                    "Filiot et al. 2024",
                    "https://arxiv.org/abs/2309.07778"));

            content.getChildren().add(createModelEntry(
                    "Virchow (Paige AI) -- GATED",
                    "ViT-H pathology foundation model trained on 1.5M slides from "
                    + "diverse tissue types. 632M parameters. Apache 2.0 license. "
                    + "Requires HuggingFace token (see below).",
                    "Vorontsov et al. 2024",
                    "https://arxiv.org/abs/2309.07778"));

            content.getChildren().add(createModelEntry(
                    "Hibou-B / Hibou-L (HistAI) -- GATED",
                    "DINOv2-based pathology models. Hibou-B (86M params) is lighter, "
                    + "Hibou-L (304M params) is more powerful. Apache 2.0 license. "
                    + "Requires HuggingFace token (see below).",
                    "Nechaev et al. 2024",
                    "https://arxiv.org/abs/2406.09414"));

            content.getChildren().add(createModelEntry(
                    "Midnight (Kaiko AI) -- ungated",
                    "ViT-G trained on TCGA data only. 1.1B parameters. "
                    + "MIT license. No authentication required.",
                    null, null));

            content.getChildren().add(createModelEntry(
                    "DINOv2-Large (Meta) -- ungated",
                    "General-purpose vision transformer, not histology-specific. "
                    + "Can work for non-H&E or unusual staining. 304M parameters. "
                    + "Apache 2.0 license. No authentication required.",
                    "Oquab et al. 2024",
                    "https://arxiv.org/abs/2304.07193"));

            // --- Gated Model Access ---
            content.getChildren().add(createSectionHeader(
                    "Accessing Gated Models (HuggingFace Token)"));
            Label gatedExplain = new Label(
                    "Some foundation models (marked GATED above) require you to accept "
                    + "a license agreement on HuggingFace before downloading. To use them:\n\n"
                    + "1. Create a free account at huggingface.co\n"
                    + "2. Visit the model's page (e.g. bioptimus/H-optimus-0) and click\n"
                    + "   'Agree and access repository' to accept the license\n"
                    + "3. Create an access token at huggingface.co/settings/tokens\n"
                    + "   (select 'Read' permission -- 'Write' is not needed)\n"
                    + "4. Set the HF_TOKEN environment variable so QuPath can find it:\n\n"
                    + "   Windows:\n"
                    + "     Settings > System > About > Advanced system settings >\n"
                    + "     Environment Variables > New (under User variables) >\n"
                    + "     Variable name: HF_TOKEN\n"
                    + "     Variable value: hf_your_token_here\n"
                    + "     Then restart QuPath.\n\n"
                    + "   macOS:\n"
                    + "     Open Terminal and run:\n"
                    + "     echo 'export HF_TOKEN=hf_your_token_here' >> ~/.zshrc\n"
                    + "     Then restart QuPath.\n\n"
                    + "   Linux:\n"
                    + "     Add to ~/.bashrc or ~/.profile:\n"
                    + "     export HF_TOKEN=hf_your_token_here\n"
                    + "     Then restart QuPath.\n\n"
                    + "Ungated models (Midnight, DINOv2) work without any setup.\n"
                    + "Histology-pretrained encoders (Lunit, Kather) are also ungated.");
            gatedExplain.setWrapText(true);
            content.getChildren().add(gatedExplain);

            Hyperlink hfTokenLink = new Hyperlink("HuggingFace: Create Access Token");
            hfTokenLink.setOnAction(e -> {
                try {
                    java.awt.Desktop.getDesktop().browse(
                            java.net.URI.create("https://huggingface.co/settings/tokens"));
                } catch (Exception ex) {
                    logger.debug("Could not open HuggingFace URL: {}", ex.getMessage());
                }
            });
            content.getChildren().add(hfTokenLink);

            // --- Recommendation ---
            content.getChildren().add(createSectionHeader("Quick Recommendation"));
            Label recommendation = new Label(
                    "For most tasks: UNet + ResNet-34 (start here).\n"
                    + "For H&E histology: UNet + Lunit SwAV or Kather100K encoder.\n"
                    + "For fluorescence/multi-channel: UNet + ResNet-34.\n"
                    + "For limited labeled data: UNet + foundation model (H-optimus-0).\n"
                    + "For fastest training: Fast Pretrained + EfficientNet-Lite0.\n"
                    + "For fastest inference: UNet + EfficientNet-B0 or MobileNet-V2.\n"
                    + "For quick experiments: Tiny UNet (no download, any channel count).");
            recommendation.setWrapText(true);
            recommendation.setStyle("-fx-font-style: italic;");
            content.getChildren().add(recommendation);

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setPrefHeight(500);
            dialog.getDialogPane().setContent(scrollPane);
            dialog.getDialogPane().setPrefWidth(600);
            dialog.showAndWait();
        }

        private Label createSectionHeader(String text) {
            Label header = new Label(text);
            header.setStyle("-fx-font-weight: bold; -fx-font-size: 13; "
                    + "-fx-padding: 8 0 2 0;");
            return header;
        }

        private VBox createModelEntry(String name, String description,
                                       String paperRef, String paperUrl) {
            VBox entry = new VBox(2);
            entry.setPadding(new Insets(0, 0, 0, 10));

            Label nameLabel = new Label(name);
            nameLabel.setStyle("-fx-font-weight: bold;");
            entry.getChildren().add(nameLabel);

            Label descLabel = new Label(description);
            descLabel.setWrapText(true);
            entry.getChildren().add(descLabel);

            if (paperRef != null && paperUrl != null) {
                Hyperlink link = new Hyperlink("Paper: " + paperRef);
                link.setOnAction(e -> {
                    try {
                        java.awt.Desktop.getDesktop().browse(
                                java.net.URI.create(paperUrl));
                    } catch (Exception ex) {
                        logger.debug("Could not open URL: {}", ex.getMessage());
                    }
                });
                entry.getChildren().add(link);
            }

            return entry;
        }
    }

    /**
     * A lightweight modal dialog for selecting a trained classifier model.
     * Displays a table of all available classifiers with key metadata.
     */
    private static class ModelPickerDialog {

        static Optional<ClassifierMetadata> show(Window owner,
                                                   List<ClassifierMetadata> classifiers) {
            return show(owner, classifiers, null);
        }

        /**
         * Shows the picker. When {@code currentArchitecture} is non-null,
         * the list is filtered to classifiers whose modelType matches --
         * Continue Training locks architecture/backbone/tile to the saved
         * model, so showing mismatched rows would silently switch the
         * user's selected architecture. A footer label summarises any
         * hidden rows by their architecture so the user can see what was
         * filtered and change the combo if they want a different one.
         */
        static Optional<ClassifierMetadata> show(Window owner,
                                                   List<ClassifierMetadata> classifiers,
                                                   String currentArchitecture) {
            Dialog<ClassifierMetadata> dialog = new Dialog<>();
            dialog.initOwner(owner);
            dialog.setTitle("Select Model");
            String header = "Choose a previously trained model to load settings from";
            if (currentArchitecture != null && !currentArchitecture.isBlank()) {
                header += " (filtered to architecture: " + currentArchitecture + ")";
            }
            dialog.setHeaderText(header);
            dialog.setResizable(true);

            ButtonType okType = new ButtonType("OK", ButtonBar.ButtonData.OK_DONE);
            dialog.getDialogPane().getButtonTypes().addAll(okType, ButtonType.CANCEL);

            List<ClassifierMetadata> visible;
            Map<String, Long> hiddenByArch = new LinkedHashMap<>();
            if (currentArchitecture != null && !currentArchitecture.isBlank()) {
                visible = new ArrayList<>();
                for (ClassifierMetadata m : classifiers) {
                    String mt = m.getModelType();
                    if (currentArchitecture.equals(mt)) {
                        visible.add(m);
                    } else {
                        hiddenByArch.merge(mt == null ? "?" : mt, 1L, Long::sum);
                    }
                }
            } else {
                visible = new ArrayList<>(classifiers);
            }

            TableView<ClassifierMetadata> table = new TableView<>();
            table.setPrefHeight(300);
            table.setPrefWidth(600);

            TableColumn<ClassifierMetadata, String> nameCol = new TableColumn<>("Name");
            nameCol.setCellValueFactory(cd ->
                    new javafx.beans.property.SimpleStringProperty(cd.getValue().getName()));
            nameCol.setPrefWidth(180);

            TableColumn<ClassifierMetadata, String> archCol = new TableColumn<>("Architecture");
            archCol.setCellValueFactory(cd ->
                    new javafx.beans.property.SimpleStringProperty(
                            cd.getValue().getModelType() + " / " + cd.getValue().getBackbone()));
            archCol.setPrefWidth(160);

            TableColumn<ClassifierMetadata, String> classesCol = new TableColumn<>("Classes");
            classesCol.setCellValueFactory(cd ->
                    new javafx.beans.property.SimpleStringProperty(
                            String.join(", ", cd.getValue().getClassNames())));
            classesCol.setPrefWidth(150);

            TableColumn<ClassifierMetadata, String> dateCol = new TableColumn<>("Date");
            dateCol.setCellValueFactory(cd -> {
                var dt = cd.getValue().getCreatedAt();
                String text = dt != null
                        ? dt.toLocalDate().toString()
                        : "";
                return new javafx.beans.property.SimpleStringProperty(text);
            });
            dateCol.setPrefWidth(90);

            table.getColumns().addAll(List.of(nameCol, archCol, classesCol, dateCol));
            table.getItems().addAll(visible);

            // Sort by date descending (newest first)
            table.getItems().sort((a, b) -> {
                if (a.getCreatedAt() == null && b.getCreatedAt() == null) return 0;
                if (a.getCreatedAt() == null) return 1;
                if (b.getCreatedAt() == null) return -1;
                return b.getCreatedAt().compareTo(a.getCreatedAt());
            });

            // Select first row by default
            if (!table.getItems().isEmpty()) {
                table.getSelectionModel().select(0);
            }

            // Disable OK until a row is selected
            Button okButton = (Button) dialog.getDialogPane().lookupButton(okType);
            okButton.disableProperty().bind(
                    table.getSelectionModel().selectedItemProperty().isNull());

            // Double-click to confirm
            table.setOnMouseClicked(event -> {
                if (event.getClickCount() == 2 && table.getSelectionModel().getSelectedItem() != null) {
                    okButton.fire();
                }
            });

            VBox root = new VBox(6, table);
            if (!hiddenByArch.isEmpty()) {
                long hiddenTotal = hiddenByArch.values().stream().mapToLong(Long::longValue).sum();
                String breakdown = hiddenByArch.entrySet().stream()
                        .map(en -> en.getValue() + " " + en.getKey())
                        .collect(Collectors.joining(", "));
                Label note = new Label(hiddenTotal + " other model(s) hidden (" + breakdown
                        + "). Change the architecture combo to see them.");
                note.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");
                note.setWrapText(true);
                root.getChildren().add(note);
            } else if (visible.isEmpty() && currentArchitecture != null) {
                Label note = new Label("No saved models match architecture '"
                        + currentArchitecture + "'. Change the architecture combo to load "
                        + "a model from a different architecture.");
                note.setStyle("-fx-text-fill: #cc6600; -fx-font-size: 11px;");
                note.setWrapText(true);
                root.getChildren().add(note);
            }
            dialog.getDialogPane().setContent(root);

            dialog.setResultConverter(button -> {
                if (button == okType) {
                    return table.getSelectionModel().getSelectedItem();
                }
                return null;
            });

            return dialog.showAndWait();
        }
    }

    /**
     * Represents a class item in the list.
     */
    private record ClassItem(String name, Integer color,
                              javafx.beans.property.BooleanProperty selected,
                              javafx.beans.property.DoubleProperty weightMultiplier,
                              double annotationArea) {
        public ClassItem(String name, Integer color, boolean selected, double annotationArea) {
            this(name, color,
                    new javafx.beans.property.SimpleBooleanProperty(selected),
                    new javafx.beans.property.SimpleDoubleProperty(1.0),
                    annotationArea);
        }
    }

    /**
     * Custom cell renderer for class items with weight multiplier spinner.
     * Properly manages bidirectional binding between the spinner and the
     * ClassItem's weightMultiplier property, cleaning up on cell reuse.
     */
    private static class ClassListCell extends ListCell<ClassItem> {
        private final HBox content;
        private final CheckBox checkBox;
        private final javafx.scene.shape.Rectangle colorBox;
        private final Label weightLabel;
        private final Spinner<Double> weightSpinner;

        // Track current bindings for cleanup on cell reuse
        private javafx.beans.property.BooleanProperty boundSelectedProperty;
        private javafx.beans.value.ChangeListener<Boolean> itemToCheckboxListener;
        private javafx.beans.value.ChangeListener<Number> weightToSpinnerListener;
        private javafx.beans.value.ChangeListener<Double> spinnerToWeightListener;
        private javafx.beans.property.DoubleProperty boundWeightProperty;

        public ClassListCell(javafx.beans.property.BooleanProperty advancedMode) {
            checkBox = new CheckBox();
            colorBox = new javafx.scene.shape.Rectangle(16, 16);
            colorBox.setStroke(javafx.scene.paint.Color.BLACK);
            colorBox.setStrokeWidth(1);

            weightLabel = new Label("Weight:");
            weightLabel.setStyle("-fx-text-fill: #666;");

            weightSpinner = new Spinner<>(0.1, 10.0, 1.0, 0.1);
            weightSpinner.setPrefWidth(80);
            weightSpinner.setEditable(true);
            weightSpinner.setTooltip(TooltipHelper.create(
                    "Multiplier applied to auto-computed class weight.\n" +
                    "1.0 = no change. >1.0 emphasizes this class.\n" +
                    "Use to boost underperforming or rare classes.\n\n" +
                    "Example: Set to 2.0 for a class with few annotations\n" +
                    "to give it more influence during training."));

            Region spacer = new Region();
            HBox.setHgrow(spacer, Priority.ALWAYS);

            content = new HBox(8, checkBox, colorBox, spacer, weightLabel, weightSpinner);
            content.setAlignment(Pos.CENTER_LEFT);

            // Hide weight controls in basic mode
            if (advancedMode != null) {
                weightLabel.visibleProperty().bind(advancedMode);
                weightLabel.managedProperty().bind(advancedMode);
                weightSpinner.visibleProperty().bind(advancedMode);
                weightSpinner.managedProperty().bind(advancedMode);
            }
        }

        @Override
        protected void updateItem(ClassItem item, boolean empty) {
            super.updateItem(item, empty);

            // Clean up previous listeners to avoid leaks during cell reuse
            if (boundSelectedProperty != null && itemToCheckboxListener != null) {
                boundSelectedProperty.removeListener(itemToCheckboxListener);
            }
            checkBox.setOnAction(null);
            boundSelectedProperty = null;
            itemToCheckboxListener = null;
            if (boundWeightProperty != null && weightToSpinnerListener != null) {
                boundWeightProperty.removeListener(weightToSpinnerListener);
            }
            if (spinnerToWeightListener != null) {
                weightSpinner.valueProperty().removeListener(spinnerToWeightListener);
            }
            boundWeightProperty = null;
            weightToSpinnerListener = null;
            spinnerToWeightListener = null;

            if (empty || item == null) {
                setGraphic(null);
            } else {
                checkBox.setText(item.name());
                checkBox.setSelected(item.selected().get());
                checkBox.setOnAction(e -> item.selected().set(checkBox.isSelected()));
                boundSelectedProperty = item.selected();
                itemToCheckboxListener = (obs, oldVal, newVal) -> checkBox.setSelected(newVal);
                boundSelectedProperty.addListener(itemToCheckboxListener);

                if (item.color() != null) {
                    int r = (item.color() >> 16) & 0xFF;
                    int g = (item.color() >> 8) & 0xFF;
                    int b = item.color() & 0xFF;
                    colorBox.setFill(javafx.scene.paint.Color.rgb(r, g, b));
                } else {
                    colorBox.setFill(javafx.scene.paint.Color.GRAY);
                }

                // Set initial spinner value
                weightSpinner.getValueFactory().setValue(item.weightMultiplier().get());

                // Property -> spinner: update spinner when weight changes programmatically
                boundWeightProperty = item.weightMultiplier();
                weightToSpinnerListener = (obs, oldVal, newVal) -> {
                    if (newVal != null) {
                        weightSpinner.getValueFactory().setValue(newVal.doubleValue());
                    }
                };
                boundWeightProperty.addListener(weightToSpinnerListener);

                // Spinner -> property: update weight when user changes spinner
                spinnerToWeightListener = (obs, oldVal, newVal) -> {
                    if (newVal != null) {
                        item.weightMultiplier().set(newVal);
                    }
                };
                weightSpinner.valueProperty().addListener(spinnerToWeightListener);

                setGraphic(content);
            }
        }
    }

    /**
     * Designates how an image's tiles participate in the train/validation split.
     * <ul>
     *   <li>{@code BOTH} (default) -- tiles enter the stratified splitter normally</li>
     *   <li>{@code TRAIN_ONLY} -- all tiles go directly to the training set</li>
     *   <li>{@code VAL_ONLY} -- all tiles go directly to the validation set</li>
     * </ul>
     */
    public enum SplitRole {
        BOTH("Both"),
        TRAIN_ONLY("Train"),
        VAL_ONLY("Val");

        private final String displayName;

        SplitRole(String displayName) {
            this.displayName = displayName;
        }

        @Override
        public String toString() {
            return displayName;
        }
    }

    /**
     * Represents a project image in the selection list.
     */
    private static class ImageSelectionItem {
        final ProjectImageEntry<BufferedImage> entry;
        final String imageName;
        final long annotationCount;
        final int imageWidth;
        final int imageHeight;
        final javafx.beans.property.BooleanProperty selected;
        final javafx.beans.property.ObjectProperty<SplitRole> splitRole;

        ImageSelectionItem(ProjectImageEntry<BufferedImage> entry, long annotationCount,
                           int imageWidth, int imageHeight) {
            this.entry = entry;
            this.imageName = entry.getImageName() + " (" + annotationCount + " annotations)";
            this.annotationCount = annotationCount;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            this.selected = new javafx.beans.property.SimpleBooleanProperty(true);
            this.splitRole = new javafx.beans.property.SimpleObjectProperty<>(SplitRole.BOTH);
        }
    }

    /**
     * Custom list cell for the image selection list. Shows a checkbox, the image
     * name, and (in advanced mode) a compact ComboBox for Train/Val/Both assignment.
     * <p>
     * Uses explicit listeners instead of bidirectional binding to prevent
     * linked checkbox behavior during ListView cell recycling.
     */
    private static class ImageSelectionCell extends ListCell<ImageSelectionItem> {

        private final javafx.beans.property.BooleanProperty advancedMode;
        private final CheckBox checkBox = new CheckBox();
        private final ComboBox<SplitRole> roleCombo = new ComboBox<>(
                FXCollections.observableArrayList(SplitRole.values()));
        private final HBox content = new HBox(6);

        private javafx.beans.property.BooleanProperty boundSelectedProperty;
        private javafx.beans.value.ChangeListener<Boolean> itemToCheckboxListener;
        private javafx.beans.property.ObjectProperty<SplitRole> boundRoleProperty;
        private javafx.beans.value.ChangeListener<SplitRole> itemToComboListener;

        ImageSelectionCell(javafx.beans.property.BooleanProperty advancedMode) {
            this.advancedMode = advancedMode;

            roleCombo.setPrefWidth(70);
            roleCombo.setStyle("-fx-font-size: 10px;");
            roleCombo.visibleProperty().bind(advancedMode);
            roleCombo.managedProperty().bind(advancedMode);

            content.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        }

        @Override
        protected void updateItem(ImageSelectionItem item, boolean empty) {
            super.updateItem(item, empty);

            // Clean up previous listeners to avoid leaks during cell reuse
            if (boundSelectedProperty != null && itemToCheckboxListener != null) {
                boundSelectedProperty.removeListener(itemToCheckboxListener);
            }
            checkBox.setOnAction(null);
            boundSelectedProperty = null;
            itemToCheckboxListener = null;

            if (boundRoleProperty != null && itemToComboListener != null) {
                boundRoleProperty.removeListener(itemToComboListener);
            }
            roleCombo.setOnAction(null);
            boundRoleProperty = null;
            itemToComboListener = null;

            if (empty || item == null) {
                setGraphic(null);
            } else {
                // Checkbox binding
                checkBox.setText(item.imageName);
                boundSelectedProperty = item.selected;
                checkBox.setSelected(boundSelectedProperty.get());
                checkBox.setOnAction(e -> boundSelectedProperty.set(checkBox.isSelected()));
                itemToCheckboxListener = (obs, oldVal, newVal) -> checkBox.setSelected(newVal);
                boundSelectedProperty.addListener(itemToCheckboxListener);

                // SplitRole combo binding
                boundRoleProperty = item.splitRole;
                roleCombo.setValue(boundRoleProperty.get());
                roleCombo.setOnAction(e -> {
                    if (boundRoleProperty != null) {
                        boundRoleProperty.set(roleCombo.getValue());
                    }
                });
                itemToComboListener = (obs, oldVal, newVal) -> roleCombo.setValue(newVal);
                boundRoleProperty.addListener(itemToComboListener);

                content.getChildren().setAll(checkBox, roleCombo);
                setGraphic(content);
            }
        }
    }

    /**
     * Generic checkbox list cell for items with a selected property.
     * Uses explicit listeners instead of bidirectional binding to prevent
     * linked checkbox behavior during ListView cell recycling.
     */
    private static class CheckBoxListCell<T> extends ListCell<T> {
        private final java.util.function.Function<T, javafx.beans.property.BooleanProperty> selectedExtractor;
        private final java.util.function.Function<T, String> textExtractor;
        private final CheckBox checkBox = new CheckBox();

        private javafx.beans.property.BooleanProperty boundSelectedProperty;
        private javafx.beans.value.ChangeListener<Boolean> itemToCheckboxListener;

        CheckBoxListCell(java.util.function.Function<T, javafx.beans.property.BooleanProperty> selectedExtractor,
                         java.util.function.Function<T, String> textExtractor) {
            this.selectedExtractor = selectedExtractor;
            this.textExtractor = textExtractor;
        }

        @Override
        protected void updateItem(T item, boolean empty) {
            super.updateItem(item, empty);

            // Clean up previous listeners to avoid leaks during cell reuse
            if (boundSelectedProperty != null && itemToCheckboxListener != null) {
                boundSelectedProperty.removeListener(itemToCheckboxListener);
            }
            checkBox.setOnAction(null);
            boundSelectedProperty = null;
            itemToCheckboxListener = null;

            if (empty || item == null) {
                setGraphic(null);
            } else {
                checkBox.setText(textExtractor.apply(item));
                boundSelectedProperty = selectedExtractor.apply(item);
                checkBox.setSelected(boundSelectedProperty.get());
                checkBox.setOnAction(e -> boundSelectedProperty.set(checkBox.isSelected()));
                itemToCheckboxListener = (obs, oldVal, newVal) -> checkBox.setSelected(newVal);
                boundSelectedProperty.addListener(itemToCheckboxListener);
                setGraphic(checkBox);
            }
        }
    }
}
