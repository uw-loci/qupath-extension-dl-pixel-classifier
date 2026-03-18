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
import qupath.ext.dlclassifier.service.ModelManager;
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
     * @param classifierName  name for the classifier
     * @param description     classifier description
     * @param trainingConfig  training configuration
     * @param channelConfig   channel configuration
     * @param selectedClasses selected class names
     * @param selectedImages  project images to train from, or null for current image only
     * @param classColors     map of class name to packed RGB color (from QuPath PathClass)
     */
    public record TrainingDialogResult(
            String classifierName,
            String description,
            TrainingConfig trainingConfig,
            ChannelConfiguration channelConfig,
            List<String> selectedClasses,
            List<ProjectImageEntry<BufferedImage>> selectedImages,
            Map<String, Integer> classColors,
            Map<String, Object> handlerParameters
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
     * Shows the training configuration dialog.
     *
     * @return CompletableFuture with the result, or cancelled if user cancels
     */
    public static CompletableFuture<TrainingDialogResult> showDialog() {
        CompletableFuture<TrainingDialogResult> future = new CompletableFuture<>();

        Platform.runLater(() -> {
            try {
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
        private final Map<String, String> validationErrors = new LinkedHashMap<>();

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
        private Spinner<Integer> validationSplitSpinner;

        // Tiling parameters
        private Spinner<Integer> tileSizeSpinner;
        private Spinner<Integer> overlapSpinner;
        private CheckBox wholeImageCheck;
        private Label wholeImageInfoLabel;
        private ComboBox<String> downsampleCombo;
        private ComboBox<String> contextScaleCombo;
        private Spinner<Integer> lineStrokeWidthSpinner;

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
        private CheckBox elasticCheck;

        // Training strategy
        private ComboBox<String> schedulerCombo;
        private ComboBox<String> lossFunctionCombo;
        private Spinner<Double> focalGammaSpinner;
        private Label focalGammaLabel;
        private Spinner<Integer> ohemSpinner;
        private ComboBox<String> earlyStoppingMetricCombo;
        private Spinner<Integer> earlyStoppingPatienceSpinner;
        private CheckBox mixedPrecisionCheck;
        private Spinner<Integer> gradientAccumulationSpinner;
        private CheckBox progressiveResizeCheck;

        // Focus class
        private ComboBox<String> focusClassCombo;
        private Spinner<Double> focusClassMinIoUSpinner;
        private Label focusClassMinIoULabel;

        // Weight initialization (unified radio group)
        private ToggleGroup weightInitGroup;
        private RadioButton scratchRadio;
        private RadioButton backbonePretrainedRadio;
        private RadioButton maeEncoderRadio;
        private RadioButton continueTrainingRadio;
        private VBox backbonePretrainedContent;
        private VBox maeEncoderContent;
        private VBox continueTrainingContent;
        private TextField maeEncoderPathField;
        private Label maeEncoderInfoLabel;
        private int maeEncoderInputChannels = -1;  // -1 = no MAE loaded
        private int maeEncoderTileSize = -1;        // -1 = unknown
        private LayerFreezePanel layerFreezePanel;
        private ClassifierBackend backend;

        // Image source selection
        private ListView<ImageSelectionItem> imageSelectionList;
        private List<TitledPane> gatedSections = new ArrayList<>();
        private boolean classesLoaded = false;
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
            Button copyScriptButton = new Button("Copy as Script");
            copyScriptButton.setOnAction(e -> copyTrainingScript(copyScriptButton));

            okButton = new Button("Start Training");
            okButton.setDisable(true);
            okButton.setDefaultButton(true);
            okButton.setOnAction(e -> {
                resultDelivered = true;
                onResult.accept(buildResult());
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
            });

            Region spacer = new Region();
            HBox.setHgrow(spacer, Priority.ALWAYS);
            HBox buttonBar = new HBox(8, copyScriptButton, spacer, okButton, cancelButton);
            buttonBar.setPadding(new Insets(10, 0, 0, 0));

            // Create content
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

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

            // Disable gated sections until classes are loaded
            setGatedSectionsEnabled(false);

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setMaxHeight(Double.MAX_VALUE);
            scrollPane.setPrefHeight(600);
            scrollPane.setPrefWidth(550);

            dialog.setScene(new Scene(scrollPane));

            // Generate default classifier name
            String timestamp = java.time.LocalDate.now().toString().replace("-", "");
            classifierNameField.setText("Classifier_" + timestamp);

            // Trigger initial layer load now that all UI components exist
            updateLayerFreezePanel();

            // Apply architecture-specific section visibility
            updateSectionsForArchitecture(architectureCombo.getValue());

            // Initial validation
            updateValidation();

            dialog.show();
        }

        private VBox createHeaderBox() {
            VBox headerBox = new VBox(5);
            headerBox.setPadding(new Insets(0, 0, 5, 0));

            Label titleLabel = new Label("Configure Classifier Training");
            titleLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");

            Label subtitleLabel = new Label("Train a deep learning model to classify pixels in your images");
            subtitleLabel.setStyle("-fx-text-fill: #666;");

            headerBox.getChildren().addAll(titleLabel, subtitleLabel, new Separator());
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

            // Update info text when backbone changes (ImageNet vs histology)
            backboneCombo.valueProperty().addListener((obs, old, newVal) -> {
                if (newVal != null && newVal.contains("_")) {
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
            });

            layerFreezePanel = new LayerFreezePanel();
            layerFreezePanel.setBackend(backend);

            backbonePretrainedContent = new VBox(5, backboneInfo, layerFreezePanel);
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
                Optional<ClassifierMetadata> selected = ModelPickerDialog.show(dialog, classifiers);
                selected.ifPresent(this::loadSettingsFromModel);
            });
            TooltipHelper.install(selectModelButton,
                    "Load settings from a previously trained model to retrain or refine it.\n" +
                    "Populates all dialog fields from the selected model's configuration.");

            loadedModelLabel = new Label();
            loadedModelLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");

            continueTrainingContent = new VBox(5, new HBox(10, selectModelButton, loadedModelLabel));
            continueTrainingContent.setPadding(new Insets(0, 0, 0, 20));

            // Toggle group listener: show/hide sub-content + re-validate
            weightInitGroup.selectedToggleProperty().addListener((obs, old, newVal) -> {
                    updateWeightInitSubContent();
                    updateValidation();
            });

            // Assemble all options
            content.getChildren().addAll(
                    scratchRadio, scratchInfo,
                    backbonePretrainedRadio, backbonePretrainedContent,
                    maeEncoderRadio, maeEncoderContent,
                    continueTrainingRadio, continueTrainingContent
            );

            // Set initial selection based on handler + preferences
            ClassifierHandler handler = ClassifierRegistry.getHandler(architectureCombo.getValue())
                    .orElse(ClassifierRegistry.getDefaultHandler());
            updateWeightInitOptions(architectureCombo.getValue());

            TitledPane pane = new TitledPane("WEIGHT INITIALIZATION", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create(
                    "Choose how model weights are initialized before training"));
            return pane;
        }

        /**
         * Shows/hides sub-content panels based on the selected weight initialization radio.
         */
        private void updateWeightInitSubContent() {
            ClassifierHandler.WeightInitStrategy selected = getSelectedWeightInitStrategy();

            backbonePretrainedContent.setVisible(
                    selected == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);
            backbonePretrainedContent.setManaged(
                    selected == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED);

            maeEncoderContent.setVisible(
                    selected == ClassifierHandler.WeightInitStrategy.MAE_ENCODER);
            maeEncoderContent.setManaged(
                    selected == ClassifierHandler.WeightInitStrategy.MAE_ENCODER);

            continueTrainingContent.setVisible(
                    selected == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING);
            continueTrainingContent.setManaged(
                    selected == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING);

            // Refresh layer freeze panel when backbone pretrained is selected
            if (selected == ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED) {
                updateLayerFreezePanel();
            }

            // Lock/unlock handler UI (e.g., MuViT model size, patch size, level scales).
            // Must stay locked for MAE_ENCODER (encoder weights require matching arch)
            // and CONTINUE_TRAINING (saved model weights require matching arch).
            boolean continuing = selected == ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING;
            if (currentHandlerUI != null) {
                if (selected == ClassifierHandler.WeightInitStrategy.MAE_ENCODER || continuing) {
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

            // Lock architecture, resolution, and context scale when continuing from
            // a saved model or using an MAE encoder. The saved/pretrained weights
            // are tied to the exact architecture type.
            // Guard: these controls may not exist yet during initial construction.
            boolean maeSelected = selected == ClassifierHandler.WeightInitStrategy.MAE_ENCODER;
            if (architectureCombo != null) architectureCombo.setDisable(continuing || maeSelected);
            if (backboneCombo != null) backboneCombo.setDisable(continuing || maeSelected);
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
            radio.setVisible(available);
            radio.setManaged(available);
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

            try (FileReader reader = new FileReader(metadataFile)) {
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

        @SuppressWarnings("unchecked")
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
                if (ts.containsKey("ohem_hard_ratio")) {
                    ohemSpinner.getValueFactory().setValue(
                            (int) Math.round(((Number) ts.get("ohem_hard_ratio")).doubleValue() * 100));
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
                    if (augObj instanceof Map) {
                        Map<String, Object> augConfig = (Map<String, Object>) augObj;
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

                // Handler-specific parameters (e.g., MuViT model_config, patch_size,
                // level_scales, rope_mode). Apply to the handler UI after architecture
                // is set so the correct handler UI exists.
                if (ts.containsKey("handler_parameters")) {
                    Object hp = ts.get("handler_parameters");
                    if (hp instanceof Map && currentHandlerUI != null) {
                        Map<String, Object> handlerParams = (Map<String, Object>) hp;
                        Platform.runLater(() -> {
                            if (currentHandlerUI != null) {
                                currentHandlerUI.applyParameters(handlerParams);
                            }
                        });
                    }
                }
            }

            // --- Classifier name and description ---
            String timestamp = java.time.LocalDate.now().toString().replace("-", "");
            classifierNameField.setText("Retrain_" + metadata.getName() + "_" + timestamp);
            descriptionField.setText("Retrained from: " + metadata.getName());

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
                validatePretrainedModelCompatibility();
                // Auto-select continue training since user picked a model
                selectWeightInitStrategy(ClassifierHandler.WeightInitStrategy.CONTINUE_TRAINING);
                loadedModelLabel.setText("Loaded from: " + metadata.getName());
                loadedModelLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");
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
            imageSelectionList.setCellFactory(lv -> new CheckBoxListCell<>(
                    item -> item.selected,
                    item -> item.imageName
            ));
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

            // Load Classes button
            loadClassesButton = new Button("Load Classes from Selected Images");
            loadClassesButton.setStyle("-fx-font-weight: bold;");
            loadClassesButton.setMaxWidth(Double.MAX_VALUE);
            TooltipHelper.install(loadClassesButton,
                    "Read annotations from the selected images and populate\n" +
                    "the class list with the union of all classes found.\n" +
                    "Also initializes channel configuration from the first image.");
            loadClassesButton.setOnAction(e -> loadClassesFromSelectedImages());

            HBox imageButtonBox = new HBox(10, selectAllImagesBtn, selectNoneImagesBtn);

            content.getChildren().addAll(info, imageSelectionList, imageButtonBox, loadClassesButton);

            // Show a message if no annotated images found
            if (imageSelectionList.getItems().isEmpty()) {
                Label noImagesLabel = new Label("No project images with classified annotations found.");
                noImagesLabel.setStyle("-fx-text-fill: #cc6600; -fx-font-style: italic;");
                content.getChildren().add(1, noImagesLabel);
            }

            // Initialize button state
            updateLoadClassesButtonState();

            TitledPane pane = new TitledPane("TRAINING DATA SOURCE", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select project images and load classes for training"));
            return pane;
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
                    "UNet: Symmetric encoder-decoder with skip connections.\n" +
                    "  Best general-purpose choice. Good default for most tasks.\n\n" +
                    "MuViT (Transformer): Multi-resolution Vision Transformer\n" +
                    "  with multi-scale feature fusion. Supports MAE pretraining.\n\n" +
                    "Custom ONNX: Import externally trained models for inference.",
                    "https://arxiv.org/abs/1505.04597",
                    archLabel, architectureCombo);
            architectureCombo.valueProperty().addListener((obs, old, newVal) -> updateBackboneOptions(newVal));

            grid.add(archLabel, 0, row);
            grid.add(architectureCombo, 1, row);
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
                    "Pretrained encoder network that extracts features:\n\n" +
                    "resnet34: Best default. Good balance of speed and accuracy.\n" +
                    "resnet50: More capacity. For large datasets or complex tasks.\n" +
                    "efficientnet-b0: Lightweight, fast inference, low VRAM.\n\n" +
                    "Histology encoders (marked 'Histology') were pretrained on\n" +
                    "millions of H&E tissue patches at 20x. Best for H&E brightfield.\n" +
                    "NOT recommended for fluorescence or multi-channel images --\n" +
                    "use ImageNet backbones (resnet34/50) for IF instead.\n" +
                    "~100MB download on first use (cached).",
                    "https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier/blob/main/docs/BEST_PRACTICES.md#backbone-selection",
                    backboneLabel, backboneCombo);
            grid.add(backboneLabel, 0, row);
            grid.add(backboneCombo, 1, row);

            // Dynamic handler-specific UI (e.g., MuViT transformer parameters)
            handlerUIContainer = new javafx.scene.layout.VBox();
            architectureCombo.valueProperty().addListener((obs, old, newVal) -> {
                updateHandlerUI(newVal);
                updateSectionsForArchitecture(newVal);
            });
            updateHandlerUI(architectureCombo.getValue());

            javafx.scene.layout.VBox modelContent = new javafx.scene.layout.VBox(5, grid, handlerUIContainer);
            TitledPane pane = new TitledPane("MODEL ARCHITECTURE", modelContent);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select the neural network architecture and encoder"));
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
                    break;
                case CONTINUE_TRAINING:
                    pretrainedPath = pretrainedModelPtPath;
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
                    .validationSplit(validationSplitSpinner.getValue() / 100.0)
                    .tileSize(tileSizeSpinner.getValue())
                    .overlap(overlapSpinner.getValue())
                    .downsample(parseDownsample(downsampleCombo.getValue()))
                    .contextScale(parseContextScale(contextScaleCombo.getValue()))
                    .augmentation(buildAugmentationConfig())
                    .intensityAugMode(mapIntensityModeFromDisplay(intensityAugCombo.getValue()))
                    .usePretrainedWeights(usePretrained)
                    .frozenLayers(frozenLayers)
                    .lineStrokeWidth(lineStrokeWidthSpinner.getValue())
                    .classWeightMultipliers(getClassWeightMultipliers())
                    .schedulerType(mapSchedulerFromDisplay(schedulerCombo.getValue()))
                    .lossFunction(mapLossFunctionFromDisplay(lossFunctionCombo.getValue()))
                    .focalGamma(focalGammaSpinner.getValue())
                    .ohemHardRatio(ohemSpinner.getValue() / 100.0)
                    .earlyStoppingMetric(mapEarlyStoppingMetricFromDisplay(earlyStoppingMetricCombo.getValue()))
                    .earlyStoppingPatience(earlyStoppingPatienceSpinner.getValue())
                    .mixedPrecision(mixedPrecisionCheck.isSelected())
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

        private void updateLayerFreezePanel() {
            if (layerFreezePanel == null
                    || getSelectedWeightInitStrategy() != ClassifierHandler.WeightInitStrategy.BACKBONE_PRETRAINED) {
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
                    CompletableFuture.runAsync(() -> {
                        layerFreezePanel.loadLayers(architecture, encoder, ch, cls);
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

            // Learning rate
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
                    "Too high: training diverges. Too low: training stalls.\n\n" +
                    "Default 1e-3 (0.001) is a safe starting point for Adam optimizer.\n" +
                    "Reduce to 1e-4 if loss oscillates wildly.\n" +
                    "Use 1e-5 when fine-tuning all layers (no freezing).\n" +
                    "The LR scheduler will adjust the rate during training.",
                    lrLabel, learningRateSpinner);

            grid.add(lrLabel, 0, row);
            grid.add(learningRateSpinner, 1, row);
            row++;

            // Validation split
            validationSplitSpinner = new Spinner<>(5, 50, DLClassifierPreferences.getValidationSplit(), 5);
            validationSplitSpinner.setEditable(true);
            validationSplitSpinner.setPrefWidth(100);
            Label valSplitLabel = new Label("Validation Split (%):");
            TooltipHelper.install(
                    "Percentage of annotated tiles held out for validation.\n" +
                    "Used to monitor overfitting during training.\n" +
                    "Higher values give more reliable validation metrics\n" +
                    "but leave less data for training.\n\n" +
                    "15-25% is typical. Use 10% for very small datasets.\n" +
                    "Use 25-30% for large datasets where you can afford it.",
                    valSplitLabel, validationSplitSpinner);

            grid.add(valSplitLabel, 0, row);
            grid.add(validationSplitSpinner, 1, row);
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

            grid.add(tileSizeLabel, 0, row);
            grid.add(tileSizeSpinner, 1, row);
            grid.add(wholeImageCheck, 2, row);
            row++;
            grid.add(wholeImageInfoLabel, 0, row, 3, 1);
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
                previewStage.setTitle(String.format("Resolution Preview (%.0fx downsample)", ds));
                Scene scene = new Scene(previewManager.getPane(), 400, 400);
                previewStage.setScene(scene);
                previewStage.setOnHiding(ev -> {
                    previewManager = null;
                    this.previewStage = null;
                });
                previewStage.show();
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
            contextScaleLabel = new Label("Context Scale:");
            TooltipHelper.install(
                    "Multi-scale context feeds the model two views of each location:\n" +
                    "the full-resolution tile for detail, plus a larger surrounding\n" +
                    "region (downsampled to the same pixel size) for spatial context.\n\n" +
                    "None: Single-scale input (current behavior).\n" +
                    "2x: Context covers 2x the area. Moderate additional context.\n" +
                    "4x: Context covers 4x the area. Good for tissue-level patterns.\n" +
                    "8x: Context covers 8x the area. For large-scale classification.\n" +
                    "16x: Context covers 16x the area. Maximum spatial context.\n\n" +
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
                contextPreviewStage.setTitle(String.format(
                        "Context Preview (%dx context at %.0fx downsample)", ctxScale, ds));
                Scene scene = new Scene(contextPreviewManager.getPane(), 400, 400);
                contextPreviewStage.setScene(scene);
                contextPreviewStage.setOnHiding(ev -> {
                    contextPreviewManager = null;
                    contextPreviewStage = null;
                });
                contextPreviewStage.show();
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
            });
            // Initial update (will show pixel-only info until image is loaded)
            updateSpatialInfoLabels();

            // Overlap
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

            grid.add(overlapLabel, 0, row);
            grid.add(overlapSpinner, 1, row);
            row++;

            // Line stroke width - restore from preferences, or fall back to QuPath's stroke thickness
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
            Label strokeLabel = new Label("Line Stroke Width:");
            TooltipHelper.install(
                    "Width in pixels for rendering line/polyline annotations as training masks.\n" +
                    "Pre-filled from QuPath's annotation stroke thickness.\n\n" +
                    "Thin strokes (<5px) produce sparse training signal from polyline\n" +
                    "annotations -- consider increasing for better training.\n" +
                    "Only affects line/polyline annotations; area annotations are\n" +
                    "filled completely regardless of this setting.",
                    strokeLabel, lineStrokeWidthSpinner);

            grid.add(strokeLabel, 0, row);
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

            // Loss function
            lossFunctionCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "Cross Entropy + Dice", "Cross Entropy",
                    "Focal + Dice", "Focal"));
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
                    "Focal: Focal loss only (no Dice component).",
                    "https://smp.readthedocs.io/en/latest/losses.html",
                    lossLabel, lossFunctionCombo);

            grid.add(lossLabel, 0, row);
            grid.add(lossFunctionCombo, 1, row);
            row++;

            // Focal gamma (visible only when focal variant selected)
            focalGammaLabel = new Label("Focal Gamma:");
            focalGammaSpinner = new Spinner<>(
                    new SpinnerValueFactory.DoubleSpinnerValueFactory(0.5, 5.0, 2.0, 0.5));
            focalGammaSpinner.setEditable(true);
            TooltipHelper.install(
                    "Focal loss focusing parameter (gamma).\n\n" +
                    "Higher gamma = stronger focus on hard pixels.\n" +
                    "  gamma=0: equivalent to standard Cross Entropy\n" +
                    "  gamma=1: mild focusing\n" +
                    "  gamma=2: standard (recommended)\n" +
                    "  gamma=3-5: aggressive focusing for very hard regions",
                    focalGammaLabel, focalGammaSpinner);
            boolean focalSelected = isFocalLossSelected(lossFunctionCombo.getValue());
            focalGammaLabel.setVisible(focalSelected);
            focalGammaLabel.setManaged(focalSelected);
            focalGammaSpinner.setVisible(focalSelected);
            focalGammaSpinner.setManaged(focalSelected);
            lossFunctionCombo.valueProperty().addListener((obs, oldVal, newVal) -> {
                boolean show = isFocalLossSelected(newVal);
                focalGammaLabel.setVisible(show);
                focalGammaLabel.setManaged(show);
                focalGammaSpinner.setVisible(show);
                focalGammaSpinner.setManaged(show);
            });
            grid.add(focalGammaLabel, 0, row);
            grid.add(focalGammaSpinner, 1, row);
            row++;

            // OHEM hard pixel %
            ohemSpinner = new Spinner<>(
                    new SpinnerValueFactory.IntegerSpinnerValueFactory(10, 100, 100, 5));
            ohemSpinner.setEditable(true);
            Label ohemLabel = new Label("Hard Pixel %:");
            TooltipHelper.install(
                    "Online Hard Example Mining (OHEM): keep only the\n" +
                    "hardest N% of pixels per batch.\n\n" +
                    "100% = all pixels (standard, no OHEM).\n" +
                    "25% = keep only the hardest quarter -- aggressive.\n\n" +
                    "More aggressive than focal loss: completely ignores\n" +
                    "easy pixels instead of down-weighting them.\n" +
                    "Tip: try focal loss first as a softer alternative.",
                    ohemLabel, ohemSpinner);
            grid.add(ohemLabel, 0, row);
            grid.add(ohemSpinner, 1, row);
            row++;

            // Early stopping metric
            earlyStoppingMetricCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "Mean IoU", "Validation Loss"));
            earlyStoppingMetricCombo.setValue(
                    mapEarlyStoppingMetricToDisplay(DLClassifierPreferences.getDefaultEarlyStoppingMetric()));
            Label esMetricLabel = new Label("Early Stop Metric:");
            TooltipHelper.install(
                    "Which metric decides the 'best' model checkpoint.\n\n" +
                    "The best model weights are saved whenever this metric improves,\n" +
                    "and training stops if it hasn't improved for 'patience' epochs.\n" +
                    "The final saved model is always from the best epoch, not the last.\n\n" +
                    "Mean IoU (recommended): Intersection-over-union averaged across\n" +
                    "  all classes. Directly measures segmentation quality.\n\n" +
                    "Validation Loss: Combined loss on held-out data.\n" +
                    "  Can oscillate while IoU still improves, so Mean IoU is\n" +
                    "  generally more reliable.",
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
                    "Optionally select a class to focus on for best model selection.\n\n" +
                    "When set, the focus class's per-class IoU is used instead of\n" +
                    "the Early Stop Metric for determining the best model and\n" +
                    "triggering early stopping.\n\n" +
                    "Use this when you care more about one class than the others.\n" +
                    "For example, if detecting 'Hinge' is critical, set it as the\n" +
                    "focus class so the best model is the one with the best Hinge IoU,\n" +
                    "not the best average across all classes.",
                    focusClassLabel, focusClassCombo);

            focusClassCombo.valueProperty().addListener((obs, old, newVal) -> {
                boolean hasFocusClass = newVal != null && !newVal.startsWith("None");
                // Show/hide min IoU row
                focusClassMinIoUSpinner.setVisible(hasFocusClass);
                focusClassMinIoUSpinner.setManaged(hasFocusClass);
                focusClassMinIoULabel.setVisible(hasFocusClass);
                focusClassMinIoULabel.setManaged(hasFocusClass);
                // Visually disable early stopping metric combo when focus class overrides it
                earlyStoppingMetricCombo.setDisable(hasFocusClass);
                if (hasFocusClass) {
                    TooltipHelper.install(
                            "Overridden by Focus Class selection.\n" +
                            "The focus class's IoU will be used for best model\n" +
                            "selection and early stopping instead.",
                            esMetricLabel, earlyStoppingMetricCombo);
                } else {
                    TooltipHelper.install(
                            "Which metric decides the 'best' model checkpoint.\n\n" +
                            "Mean IoU (recommended): Intersection-over-union averaged across\n" +
                            "  all classes. Directly measures segmentation quality.\n\n" +
                            "Validation Loss: Combined loss on held-out data.",
                            esMetricLabel, earlyStoppingMetricCombo);
                }
            });

            grid.add(focusClassLabel, 0, row);
            grid.add(focusClassCombo, 1, row);
            row++;

            // Focus class min IoU threshold (hidden by default)
            focusClassMinIoULabel = new Label("Min Focus IoU:");
            focusClassMinIoUSpinner = new Spinner<>(0.0, 1.0, 0.5, 0.05);
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

            grid.add(mixedPrecisionCheck, 0, row, 2, 1);
            row++;

            // Gradient accumulation
            gradientAccumulationSpinner = new Spinner<>(1, 8, 1, 1);
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
            progressiveResizeCheck.setSelected(false);
            TooltipHelper.install(progressiveResizeCheck,
                    "Train at half tile resolution for the first 40% of epochs,\n" +
                    "then switch to full resolution.\n\n" +
                    "Benefits:\n" +
                    "- Faster early training (4x fewer pixels)\n" +
                    "- Acts as regularization (prevents overfitting)\n" +
                    "- Helps model learn coarse features first\n\n" +
                    "Leave unchecked for standard training.");

            grid.add(progressiveResizeCheck, 0, row, 2, 1);

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
            classListView.setCellFactory(lv -> new ClassListCell());
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

            content.getChildren().addAll(
                    flipHorizontalCheck,
                    flipVerticalCheck,
                    rotationCheck,
                    intensityRow,
                    elasticCheck
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

                Platform.runLater(() -> {
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

                        // Auto-select appropriate intensity mode based on image type
                        if (!isBrightfield(channelImageData)) {
                            // Fluorescence images default to per-channel mode
                            intensityAugCombo.setValue("Fluorescence (per-channel)");
                        } else {
                            // Brightfield images default to color jitter
                            intensityAugCombo.setValue("Brightfield (color jitter)");
                        }
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
            // Restore previous selection if still valid, otherwise reset
            if (currentSelection != null && items.contains(currentSelection)) {
                focusClassCombo.setValue(currentSelection);
            } else {
                focusClassCombo.setValue("None (use Mean IoU)");
            }
        }

        /** Maps focus class combo display value to config value (null for "None"). */
        private static String mapFocusClassFromDisplay(String display) {
            if (display == null || display.startsWith("None")) return null;
            return display;
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

        private boolean isBrightfield(ImageData<BufferedImage> imageData) {
            ImageData.ImageType type = imageData.getImageType();
            return type == ImageData.ImageType.BRIGHTFIELD_H_E
                    || type == ImageData.ImageType.BRIGHTFIELD_H_DAB
                    || type == ImageData.ImageType.BRIGHTFIELD_OTHER;
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

            backboneCombo.setItems(FXCollections.observableArrayList(backboneList));

            // Show display names via custom cell factory (handler-specific lookup)
            backboneCombo.setCellFactory(lv -> new ListCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty);
                    if (empty || item == null) {
                        setText(null);
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
                        "Continue training requires a model -- click 'Select model...' first");
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

            if (totalArea == 0 || selectedItems.isEmpty()) {
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

            // Build training config from unified weight init strategy
            TrainingConfig trainingConfig = buildTrainingConfig();

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
                        int tileSize = trainingConfig.getTileSize();
                        int batchSize = trainingConfig.getBatchSize();
                        int gradAccum = trainingConfig.getGradientAccumulationSteps();
                        double modelMb = "muvit".equals(modelType) ? 140.0 : 30.0;
                        double actMultiplier = "muvit".equals(modelType) ? 10.0 : 4.0;
                        double areaScale = (double)(tileSize * tileSize) / (256.0 * 256.0);
                        double estimatedMb = modelMb * (1 + 3 + actMultiplier * areaScale * batchSize);
                        double budgetMb = totalMb * 0.85;

                        if (estimatedMb > budgetMb) {
                            // Compute specific settings that would fit
                            StringBuilder suggestions = new StringBuilder();
                            suggestions.append(String.format(
                                    "Estimated VRAM: %.0f MB (GPU has %d MB usable).\n"
                                    + "Current settings: %s, %dx%d tiles, batch %d (x%d accum).\n\n"
                                    + "Settings that would fit in VRAM:\n",
                                    estimatedMb, totalMb, modelType, tileSize, tileSize,
                                    batchSize, gradAccum));

                            // Suggest batch=1 at current tile size
                            double estBatch1 = modelMb * (1 + 3 + actMultiplier * areaScale * 1);
                            if (estBatch1 <= budgetMb && batchSize > 1) {
                                suggestions.append(String.format(
                                        "  - Batch size 1 at %dpx (~%.0f MB)\n", tileSize, estBatch1));
                            }

                            // Suggest smaller tile sizes with current batch
                            for (int candidate : new int[]{512, 256, 128}) {
                                if (candidate >= tileSize) continue;
                                double candArea = (double)(candidate * candidate) / (256.0 * 256.0);
                                double candEst = modelMb * (1 + 3 + actMultiplier * candArea * batchSize);
                                if (candEst <= budgetMb) {
                                    int maxBatch = 1;
                                    for (int b = batchSize; b >= 1; b--) {
                                        double bEst = modelMb * (1 + 3 + actMultiplier * candArea * b);
                                        if (bEst <= budgetMb) { maxBatch = b; break; }
                                    }
                                    suggestions.append(String.format(
                                            "  - %dpx tiles, batch %d (~%.0f MB)\n",
                                            candidate, maxBatch,
                                            modelMb * (1 + 3 + actMultiplier * candArea * maxBatch)));
                                    break;  // Show best fitting option
                                }
                            }

                            // Note about MAE pretraining tile size if applicable
                            if (maeEncoderTileSize > 0 && tileSize != maeEncoderTileSize) {
                                suggestions.append(String.format(
                                        "\nNote: MAE encoder was pretrained at %dpx -- "
                                        + "using the same tile size\ngives best weight transfer.\n",
                                        maeEncoderTileSize));
                            }

                            suggestions.append("\nIncrease downsample to compensate for smaller tiles.\n"
                                    + "Gradient accumulation can simulate larger batches without extra VRAM.\n\n"
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
                    handlerParams
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
                default -> "Cross Entropy + Dice";
            };
        }

        private static String mapLossFunctionFromDisplay(String display) {
            return switch (display) {
                case "Cross Entropy" -> "cross_entropy";
                case "Focal + Dice" -> "focal_dice";
                case "Focal" -> "focal";
                default -> "ce_dice";
            };
        }

        private static boolean isFocalLossSelected(String display) {
            return "Focal + Dice".equals(display) || "Focal".equals(display);
        }

        private static String mapEarlyStoppingMetricToDisplay(String value) {
            if ("val_loss".equals(value)) return "Validation Loss";
            return "Mean IoU";
        }

        private static String mapEarlyStoppingMetricFromDisplay(String display) {
            if ("Validation Loss".equals(display)) return "val_loss";
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
    }

    /**
     * A lightweight modal dialog for selecting a trained classifier model.
     * Displays a table of all available classifiers with key metadata.
     */
    private static class ModelPickerDialog {

        static Optional<ClassifierMetadata> show(Window owner,
                                                   List<ClassifierMetadata> classifiers) {
            Dialog<ClassifierMetadata> dialog = new Dialog<>();
            dialog.initOwner(owner);
            dialog.setTitle("Select Model");
            dialog.setHeaderText("Choose a previously trained model to load settings from");
            dialog.setResizable(true);

            ButtonType okType = new ButtonType("OK", ButtonBar.ButtonData.OK_DONE);
            dialog.getDialogPane().getButtonTypes().addAll(okType, ButtonType.CANCEL);

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
            table.getItems().addAll(classifiers);

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

            dialog.getDialogPane().setContent(table);

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

        public ClassListCell() {
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
     * Represents a project image in the selection list.
     */
    private static class ImageSelectionItem {
        final ProjectImageEntry<BufferedImage> entry;
        final String imageName;
        final long annotationCount;
        final int imageWidth;
        final int imageHeight;
        final javafx.beans.property.BooleanProperty selected;

        ImageSelectionItem(ProjectImageEntry<BufferedImage> entry, long annotationCount,
                           int imageWidth, int imageHeight) {
            this.entry = entry;
            this.imageName = entry.getImageName() + " (" + annotationCount + " annotations)";
            this.annotationCount = annotationCount;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            this.selected = new javafx.beans.property.SimpleBooleanProperty(true);
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
