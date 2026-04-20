package qupath.ext.dlclassifier.ui;

import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.layout.*;
import javafx.util.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.InferenceConfig.OutputObjectType;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.scripting.ScriptGenerator;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.scripting.QP;

import java.awt.image.BufferedImage;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * Dialog for configuring deep learning classifier inference.
 * <p>
 * This dialog provides an interface for:
 * <ul>
 *   <li>Classifier selection from available models</li>
 *   <li>Output type configuration (measurements, objects, overlay)</li>
 *   <li>Channel mapping for multi-channel images</li>
 *   <li>Post-processing options</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class InferenceDialog {

    private static final Logger logger = LoggerFactory.getLogger(InferenceDialog.class);

    /**
     * Result of the inference dialog.
     */
    public record InferenceDialogResult(
            ClassifierMetadata classifier,
            InferenceConfig inferenceConfig,
            ChannelConfiguration channelConfig,
            InferenceConfig.ApplicationScope applicationScope,
            boolean createBackup
    ) {}

    private InferenceDialog() {
        // Utility class
    }

    /**
     * Shows the inference configuration dialog.
     *
     * @return CompletableFuture with the result, or cancelled if user cancels
     */
    public static CompletableFuture<InferenceDialogResult> showDialog() {
        CompletableFuture<InferenceDialogResult> future = new CompletableFuture<>();

        Platform.runLater(() -> {
            try {
                InferenceDialogBuilder builder = new InferenceDialogBuilder();
                Optional<InferenceDialogResult> result = builder.buildAndShow();
                if (result.isPresent()) {
                    future.complete(result.get());
                } else {
                    future.complete(null);
                }
            } catch (Exception e) {
                logger.error("Error showing inference dialog", e);
                future.completeExceptionally(e);
            }
        });

        return future;
    }

    /**
     * Inner builder class for constructing the dialog.
     */
    private static class InferenceDialogBuilder {

        private final ModelManager modelManager = new ModelManager();
        private Dialog<InferenceDialogResult> dialog;

        // Classifier selection
        private TableView<ClassifierMetadata> classifierTable;
        private Label classifierInfoLabel;

        // Output options
        private ComboBox<InferenceConfig.OutputType> outputTypeCombo;
        private ComboBox<OutputObjectType> objectTypeCombo;
        private Spinner<Double> minObjectSizeSpinner;
        private Spinner<Double> holeFillingSpinner;
        private Spinner<Double> smoothingSpinner;

        // Channel configuration
        private ChannelSelectionPanel channelPanel;
        private VBox channelMappingPanel;

        // Processing options
        private Spinner<Integer> tileSizeSpinner;
        private Spinner<Integer> overlapSpinner;
        private Spinner<Double> overlapPercentSpinner;
        private Label overlapWarningLabel;
        private ComboBox<InferenceConfig.BlendMode> blendModeCombo;
        private CheckBox useGPUCheck;
        private CheckBox useTTACheck;

        // Channel section (for brightfield auto-configuration)
        private TitledPane channelSectionPane;

        // Scope options
        private RadioButton applyToSelectedRadio;
        private RadioButton applyToAllRadio;
        private RadioButton applyToWholeImageRadio;
        private CheckBox createBackupCheck;

        private Button okButton;

        public Optional<InferenceDialogResult> buildAndShow() {
            dialog = new Dialog<>();
            dialog.initOwner(QuPathGUI.getInstance().getStage());
            dialog.setTitle("Apply DL Pixel Classifier");
            dialog.setResizable(true);

            // Create header
            createHeader();

            // Create button types
            ButtonType applyType = new ButtonType("Apply", ButtonBar.ButtonData.OK_DONE);
            ButtonType cancelType = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);
            ButtonType copyScriptType = new ButtonType("Copy as Script", ButtonBar.ButtonData.LEFT);
            dialog.getDialogPane().getButtonTypes().addAll(copyScriptType, applyType, cancelType);

            okButton = (Button) dialog.getDialogPane().lookupButton(applyType);
            okButton.setDisable(true);

            // Wire up the "Copy as Script" button
            Button copyScriptButton = (Button) dialog.getDialogPane().lookupButton(copyScriptType);
            copyScriptButton.addEventFilter(ActionEvent.ACTION, event -> {
                event.consume(); // Prevent dialog from closing
                copyInferenceScript(copyScriptButton);
            });

            // Create content -- all UI components are constructed first,
            // THEN listeners are installed, so cross-section references are safe.
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            content.getChildren().addAll(
                    createClassifierSection(),
                    createOutputSection(),
                    createChannelSection(),
                    createProcessingSection(),
                    createScopeSection()
            );

            // Install cross-section listeners AFTER all components exist
            installListeners();

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setMaxHeight(Double.MAX_VALUE);
            scrollPane.setPrefHeight(550);
            scrollPane.setPrefWidth(600);

            dialog.getDialogPane().setContent(scrollPane);

            // Load available classifiers (fires classifier selection listener)
            loadClassifiers();

            // Initialize with current image
            initializeFromCurrentImage();

            // Set result converter
            dialog.setResultConverter(button -> {
                if (button != applyType) {
                    return null;
                }
                return buildResult();
            });

            return dialog.showAndWait();
        }

        private void createHeader() {
            VBox headerBox = new VBox(5);
            headerBox.setPadding(new Insets(10));

            Label titleLabel = new Label("Apply Classification to Image");
            titleLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");

            Label subtitleLabel = new Label("Select a trained classifier and configure output options");
            subtitleLabel.setStyle("-fx-text-fill: #666;");

            headerBox.getChildren().addAll(titleLabel, subtitleLabel, new Separator());
            dialog.getDialogPane().setHeader(headerBox);
        }

        private TitledPane createClassifierSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Create classifier table
            classifierTable = new TableView<>();
            classifierTable.setPrefHeight(150);
            classifierTable.setPlaceholder(new Label("No classifiers available. Train a classifier first."));
            TooltipHelper.install(classifierTable,
                    "Available trained classifiers.\n" +
                    "Select one to apply to the current image.\n" +
                    "Channel count must match between classifier and image.");

            TableColumn<ClassifierMetadata, String> nameCol = new TableColumn<>("Name");
            nameCol.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().getName()));
            nameCol.setPrefWidth(150);

            TableColumn<ClassifierMetadata, String> typeCol = new TableColumn<>("Type");
            typeCol.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().getModelType()));
            typeCol.setPrefWidth(80);

            TableColumn<ClassifierMetadata, String> channelsCol = new TableColumn<>("Channels");
            channelsCol.setCellValueFactory(data -> {
                ClassifierMetadata m = data.getValue();
                String display = String.valueOf(m.getInputChannels());
                if (m.getContextScale() > 1) {
                    display += " +" + m.getContextScale() + "x ctx";
                }
                return new SimpleStringProperty(display);
            });
            channelsCol.setPrefWidth(90);

            TableColumn<ClassifierMetadata, String> classesCol = new TableColumn<>("Classes");
            classesCol.setCellValueFactory(data -> new SimpleStringProperty(
                    String.valueOf(data.getValue().getClassNames().size())));
            classesCol.setPrefWidth(60);

            TableColumn<ClassifierMetadata, String> dateCol = new TableColumn<>("Trained");
            dateCol.setCellValueFactory(data -> {
                var created = data.getValue().getCreatedAt();
                if (created != null) {
                    return new SimpleStringProperty(created.format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
                }
                return new SimpleStringProperty("-");
            });
            dateCol.setPrefWidth(90);

            classifierTable.getColumns().addAll(List.of(nameCol, typeCol, channelsCol, classesCol, dateCol));

            // Info label
            classifierInfoLabel = new Label("Select a classifier to see details");
            classifierInfoLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");
            classifierInfoLabel.setWrapText(true);

            content.getChildren().addAll(classifierTable, classifierInfoLabel);

            TitledPane pane = new TitledPane("SELECT CLASSIFIER", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Choose a trained classifier to apply to the current image"));
            return pane;
        }

        private TitledPane createOutputSection() {
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

            // Output type - only OBJECTS and MEASUREMENTS are offered here.
            // Overlays are handled via the dedicated Toggle Overlay menu item.
            outputTypeCombo = new ComboBox<>(FXCollections.observableArrayList(
                    InferenceConfig.OutputType.OBJECTS,
                    InferenceConfig.OutputType.MEASUREMENTS));
            try {
                InferenceConfig.OutputType saved = InferenceConfig.OutputType.valueOf(
                        DLClassifierPreferences.getLastOutputType());
                // Only restore if it's one of the available options
                if (saved == InferenceConfig.OutputType.OBJECTS
                        || saved == InferenceConfig.OutputType.MEASUREMENTS) {
                    outputTypeCombo.setValue(saved);
                } else {
                    outputTypeCombo.setValue(InferenceConfig.OutputType.OBJECTS);
                }
            } catch (IllegalArgumentException e) {
                outputTypeCombo.setValue(InferenceConfig.OutputType.OBJECTS);
            }
            TooltipHelper.install(outputTypeCombo,
                    "How classification results are represented:\n\n" +
                    "OBJECTS: Create detection/annotation objects for classified regions.\n" +
                    "  Best for spatial analysis and counting discrete structures.\n\n" +
                    "MEASUREMENTS: Add per-class probability measurements to annotations.\n" +
                    "  Best for quantification workflows (e.g. % area per class).\n\n" +
                    "For quick visualization, use the Toggle Overlay in the main menu\n" +
                    "(Extensions > DL Pixel Classifier > Toggle Prediction Overlay).");

            grid.add(new Label("Output Type:"), 0, row);
            grid.add(outputTypeCombo, 1, row);
            row++;

            // Object type (DETECTION vs ANNOTATION) - only for OBJECTS output
            objectTypeCombo = new ComboBox<>(FXCollections.observableArrayList(OutputObjectType.values()));
            try {
                objectTypeCombo.setValue(OutputObjectType.valueOf(
                        DLClassifierPreferences.getDefaultObjectType()));
            } catch (IllegalArgumentException e) {
                objectTypeCombo.setValue(OutputObjectType.DETECTION);
            }
            TooltipHelper.install(objectTypeCombo,
                    "QuPath object type for classified regions:\n\n" +
                    "DETECTION: Lightweight, non-editable objects for quantification.\n" +
                    "  Faster to create and render. Ideal for large numbers of objects.\n\n" +
                    "ANNOTATION: Editable objects that can be further classified\n" +
                    "  or used as parent objects for nested analysis workflows.");

            grid.add(new Label("Object Type:"), 0, row);
            grid.add(objectTypeCombo, 1, row);
            row++;

            // Min object size (for OBJECTS output)
            minObjectSizeSpinner = new Spinner<>(0.0, 10000.0,
                    DLClassifierPreferences.getMinObjectSizeMicrons(), 1.0);
            minObjectSizeSpinner.setEditable(true);
            minObjectSizeSpinner.setPrefWidth(100);
            TooltipHelper.install(minObjectSizeSpinner,
                    "Minimum area threshold in um^2 for generated objects.\n" +
                    "Objects smaller than this are discarded as noise.\n" +
                    "Set to 0 to keep all objects regardless of size.\n" +
                    "Typical values: 10-100 um^2 depending on structure size.");

            grid.add(new Label("Min Object Size (um2):"), 0, row);
            grid.add(minObjectSizeSpinner, 1, row);
            row++;

            // Hole filling
            holeFillingSpinner = new Spinner<>(0.0, 1000.0,
                    DLClassifierPreferences.getHoleFillingMicrons(), 1.0);
            holeFillingSpinner.setEditable(true);
            holeFillingSpinner.setPrefWidth(100);
            TooltipHelper.install(holeFillingSpinner,
                    "Fill interior holes in objects smaller than this area (um^2).\n" +
                    "Removes small gaps caused by misclassified pixels\n" +
                    "within otherwise solid regions. Set to 0 to disable.\n" +
                    "Typical values: 5-50 um^2.");

            grid.add(new Label("Hole Filling (um2):"), 0, row);
            grid.add(holeFillingSpinner, 1, row);
            row++;

            // Smoothing - restore from preferences
            smoothingSpinner = new Spinner<>(0.0, 10.0, DLClassifierPreferences.getSmoothing(), 0.5);
            smoothingSpinner.setEditable(true);
            smoothingSpinner.setPrefWidth(100);
            TooltipHelper.install(smoothingSpinner,
                    "Boundary simplification tolerance in microns.\n" +
                    "Smooths jagged object boundaries using topology-preserving\n" +
                    "simplification. Higher values produce simpler boundaries.\n" +
                    "Set to 0 for pixel-exact boundaries.\n" +
                    "Typical: 0.5-2.0 um for a good balance of accuracy and smoothness.");

            grid.add(new Label("Boundary Smoothing:"), 0, row);
            grid.add(smoothingSpinner, 1, row);

            TitledPane pane = new TitledPane("OUTPUT OPTIONS", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Configure how classification results are generated"));
            return pane;
        }

        private TitledPane createChannelSection() {
            channelPanel = new ChannelSelectionPanel();

            channelMappingPanel = new VBox(5);
            channelMappingPanel.setPadding(new Insets(5, 0, 0, 0));
            channelMappingPanel.setVisible(false);
            channelMappingPanel.setManaged(false);

            VBox channelContent = new VBox(5, channelPanel, channelMappingPanel);

            channelSectionPane = new TitledPane("CHANNEL MAPPING", channelContent);
            channelSectionPane.setExpanded(true);
            channelSectionPane.setStyle("-fx-font-weight: bold;");
            channelSectionPane.setTooltip(TooltipHelper.create(
                    "Map image channels to classifier input channels.\n" +
                    "For brightfield RGB images, this is auto-configured."));
            return channelSectionPane;
        }

        private TitledPane createProcessingSection() {
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

            // Tile size
            tileSizeSpinner = new Spinner<>(64, 8192, DLClassifierPreferences.getTileSize(), 64);
            tileSizeSpinner.setEditable(true);
            tileSizeSpinner.setPrefWidth(100);
            TooltipHelper.install(tileSizeSpinner,
                    "Tile size in pixels for inference processing.\n" +
                    "Auto-set to match the classifier's training tile size.\n" +
                    "Must be divisible by 32. Larger tiles may improve\n" +
                    "context but use more GPU memory.\n" +
                    "Changing this from the training tile size is not recommended.");

            grid.add(new Label("Tile Size:"), 0, row);
            grid.add(tileSizeSpinner, 1, row);
            row++;

            // Overlap percent (preferred - percentage based)
            overlapPercentSpinner = new Spinner<>(0.0, 50.0, DLClassifierPreferences.getTileOverlapPercent(), 2.5);
            overlapPercentSpinner.setEditable(true);
            overlapPercentSpinner.setPrefWidth(100);
            TooltipHelper.install(overlapPercentSpinner,
                    "Tile overlap as percentage of tile size (0-50%).\n" +
                    "Controls both batch blending AND overlay tile boundary quality.\n" +
                    "Higher values eliminate edge artifacts but increase processing time.\n\n" +
                    "0-10%: Fast but may show tile seams in overlays.\n" +
                    "10-25%: Good balance -- recommended default.\n" +
                    "25-50%: Best quality -- eliminates all edge artifacts.");
            overlapPercentSpinner.valueProperty().addListener((obs, old, newVal) -> updateOverlapWarning(newVal));

            grid.add(new Label("Tile Overlap (%):"), 0, row);
            grid.add(overlapPercentSpinner, 1, row);
            row++;

            // Overlap warning label
            overlapWarningLabel = new Label();
            overlapWarningLabel.setWrapText(true);
            overlapWarningLabel.setMaxWidth(350);
            overlapWarningLabel.setStyle("-fx-font-size: 11px;");
            grid.add(overlapWarningLabel, 0, row, 2, 1);
            row++;

            // Keep legacy overlap spinner hidden but functional
            overlapSpinner = new Spinner<>(0, 256, DLClassifierPreferences.getTileOverlap(), 8);
            // Don't add to grid - it's computed from overlapPercentSpinner

            // Blend mode - restore from preferences
            blendModeCombo = new ComboBox<>(FXCollections.observableArrayList(InferenceConfig.BlendMode.values()));
            try {
                blendModeCombo.setValue(InferenceConfig.BlendMode.valueOf(DLClassifierPreferences.getLastBlendMode()));
            } catch (IllegalArgumentException e) {
                blendModeCombo.setValue(InferenceConfig.BlendMode.GAUSSIAN);
            }
            TooltipHelper.install(blendModeCombo,
                    "Strategy for handling tile boundaries:\n\n" +
                    "GAUSSIAN (Recommended): Cosine-bell blending for smooth transitions.\n" +
                    "  Eliminates tile boundary artifacts with smooth S-curve averaging.\n" +
                    "  Forced for OVERLAY output type.\n\n" +
                    "LINEAR: Weighted average blending at tile boundaries.\n" +
                    "  Faster but may show faint grid lines, especially with BatchNorm models.\n" +
                    "  Available for batch inference only (RENDERED_OVERLAY, OBJECTS).\n\n" +
                    "CENTER_CROP: Use only center predictions from each tile.\n" +
                    "  No blending -- discards predictions near tile edges.\n" +
                    "  Available for batch inference only (RENDERED_OVERLAY, OBJECTS).\n\n" +
                    "NONE: No blending; raw tile predictions.\n" +
                    "  Fastest but will show visible tile seams.");

            grid.add(new Label("Blend Mode:"), 0, row);
            grid.add(blendModeCombo, 1, row);
            row++;

            // GPU
            useGPUCheck = new CheckBox("Use GPU if available");
            useGPUCheck.setSelected(DLClassifierPreferences.isUseGPU());
            TooltipHelper.install(useGPUCheck,
                    "Run inference on GPU (CUDA) if available.\n" +
                    "Typically 10-50x faster than CPU.\n" +
                    "Requires CUDA-enabled PyTorch on the server.\n" +
                    "Falls back to CPU automatically if GPU is unavailable.\n" +
                    "Apple Silicon (MPS) is also supported.");

            grid.add(useGPUCheck, 0, row, 2, 1);
            row++;

            // Test-Time Augmentation
            useTTACheck = new CheckBox("Test-Time Augmentation (TTA)");
            useTTACheck.setSelected(false);
            TooltipHelper.install(useTTACheck,
                    "Run inference with D4 augmentations (flips + 90-degree rotations)\n" +
                    "and average the predictions.\n\n" +
                    "Typically improves segmentation quality by 1-3%,\n" +
                    "but inference is ~8x slower.\n\n" +
                    "Recommended for final results, not for iterative testing.");

            grid.add(useTTACheck, 0, row, 2, 1);

            // Initialize warning based on default value
            updateOverlapWarning(overlapPercentSpinner.getValue());

            TitledPane pane = new TitledPane("PROCESSING OPTIONS", grid);
            pane.setExpanded(false); // Collapsed by default
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Configure tile processing, blending, and GPU settings"));
            return pane;
        }

        private void updateOverlapWarning(double overlapPercent) {
            if (overlapPercent == 0.0) {
                overlapWarningLabel.setText("WARNING: Objects will NOT be merged across tile boundaries");
                overlapWarningLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #D32F2F;");
            } else if (overlapPercent < 10.0) {
                overlapWarningLabel.setText("Note: Low overlap may result in visible seams in output");
                overlapWarningLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #F57C00;");
            } else if (overlapPercent >= 25.0) {
                overlapWarningLabel.setText("High overlap -- best quality, eliminates edge artifacts");
                overlapWarningLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #388E3C;");
            } else {
                overlapWarningLabel.setText("Good overlap for seamless blending");
                overlapWarningLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #388E3C;");
            }

            // Update the pixel-based overlap value
            int tileSize = tileSizeSpinner != null ? tileSizeSpinner.getValue() : 512;
            int overlapPixels = (int) Math.round(tileSize * overlapPercent / 100.0);
            if (overlapSpinner != null) {
                overlapSpinner.getValueFactory().setValue(overlapPixels);
            }
        }

        private TitledPane createScopeSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Application scope
            ToggleGroup scopeGroup = new ToggleGroup();

            // Restore saved scope from preferences
            InferenceConfig.ApplicationScope savedScope;
            try {
                savedScope = InferenceConfig.ApplicationScope.valueOf(
                        DLClassifierPreferences.getApplicationScope());
            } catch (IllegalArgumentException e) {
                savedScope = InferenceConfig.ApplicationScope.ALL_ANNOTATIONS;
            }

            applyToWholeImageRadio = new RadioButton("Apply to whole image");
            applyToWholeImageRadio.setToggleGroup(scopeGroup);
            applyToWholeImageRadio.setSelected(savedScope == InferenceConfig.ApplicationScope.WHOLE_IMAGE);
            TooltipHelper.install(applyToWholeImageRadio,
                    "Classify the entire image without requiring annotations.\n" +
                    "Recommended for overlay output or full-image classification.\n" +
                    "Processing time depends on image size and tile settings.");

            applyToAllRadio = new RadioButton("Apply to all annotations");
            applyToAllRadio.setToggleGroup(scopeGroup);
            applyToAllRadio.setSelected(savedScope == InferenceConfig.ApplicationScope.ALL_ANNOTATIONS);
            TooltipHelper.install(applyToAllRadio,
                    "Classify within all annotations in the image.\n" +
                    "Processes every annotation regardless of selection state.\n" +
                    "Good for batch processing after annotating regions of interest.");

            applyToSelectedRadio = new RadioButton("Apply to selected annotations only");
            applyToSelectedRadio.setToggleGroup(scopeGroup);
            applyToSelectedRadio.setSelected(savedScope == InferenceConfig.ApplicationScope.SELECTED_ANNOTATIONS);
            TooltipHelper.install(applyToSelectedRadio,
                    "Only classify within the currently selected annotations.\n" +
                    "Useful for testing on a small region before full-image inference.\n" +
                    "Select annotations in the hierarchy panel or on the viewer.");

            // Backup option - restore from preferences
            createBackupCheck = new CheckBox("Create backup of annotation measurements before applying");
            createBackupCheck.setSelected(DLClassifierPreferences.isCreateBackup());
            TooltipHelper.install(createBackupCheck,
                    "Save a copy of existing annotation measurements before\n" +
                    "overwriting with new classification results.\n" +
                    "Recommended when re-running inference on previously classified images.\n" +
                    "Backup measurements are prefixed with 'backup_'.");

            content.getChildren().addAll(
                    new Label("Application scope:"),
                    applyToWholeImageRadio,
                    applyToAllRadio,
                    applyToSelectedRadio,
                    new Separator(),
                    createBackupCheck
            );

            TitledPane pane = new TitledPane("APPLICATION SCOPE", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Control which region to classify and backup options"));
            return pane;
        }

        /**
         * Installs all cross-section listeners AFTER every UI component has been
         * created.  This guarantees that no listener can fire while a field it
         * references is still null, eliminating initialization-order NPEs.
         */
        private void installListeners() {
            // Classifier selection -> updates channel panel, tile size, validation
            classifierTable.getSelectionModel().selectedItemProperty().addListener(
                    (obs, old, selected) -> onClassifierSelected(selected));

            // Output type -> enables/disables object options, blend mode, scope
            outputTypeCombo.valueProperty().addListener(
                    (obs, old, newVal) -> updateOutputOptions(newVal));

            // Channel validity -> enables/disables OK button
            channelPanel.validProperty().addListener(
                    (obs, old, valid) -> updateValidation());

            // Apply initial state now that all components exist
            updateOutputOptions(outputTypeCombo.getValue());
        }

        private void loadClassifiers() {
            List<ClassifierMetadata> classifiers = modelManager.listClassifiers();
            classifierTable.setItems(FXCollections.observableArrayList(classifiers));

            if (!classifiers.isEmpty()) {
                classifierTable.getSelectionModel().selectFirst();
            }
        }

        private void initializeFromCurrentImage() {
            ImageData<BufferedImage> imageData = QP.getCurrentImageData();
            if (imageData != null) {
                channelPanel.setImageData(imageData);
                channelPanel.autoConfigureForImageType(imageData.getImageType(),
                        imageData.getServer().nChannels());

                // Collapse channel section for brightfield images
                if (isBrightfield(imageData)) {
                    channelSectionPane.setExpanded(false);
                }
            }
        }

        private boolean isBrightfield(ImageData<BufferedImage> imageData) {
            ImageData.ImageType type = imageData.getImageType();
            return type == ImageData.ImageType.BRIGHTFIELD_H_E
                    || type == ImageData.ImageType.BRIGHTFIELD_H_DAB
                    || type == ImageData.ImageType.BRIGHTFIELD_OTHER;
        }

        private void onClassifierSelected(ClassifierMetadata classifier) {
            if (classifier == null) {
                classifierInfoLabel.setText("Select a classifier to see details");
                classifierInfoLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");
                channelPanel.setRequiredChannelCount(-1);
                channelMappingPanel.setVisible(false);
                channelMappingPanel.setManaged(false);
                okButton.setDisable(true);
                return;
            }

            // Update info label
            StringBuilder info = new StringBuilder();
            info.append("Architecture: ").append(classifier.getModelType());
            if (classifier.getBackbone() != null) {
                info.append(" (").append(classifier.getBackbone()).append(")");
            }
            info.append("\n");
            info.append("Input: ").append(classifier.getInputChannels()).append(" channels");
            if (classifier.getContextScale() > 1) {
                info.append(" + ").append(classifier.getContextScale()).append("x context");
            }
            info.append(", ");
            info.append(classifier.getInputWidth()).append("x").append(classifier.getInputHeight()).append(" tiles\n");
            double ds = classifier.getDownsample();
            if (ds > 1.0) {
                info.append("Trained at: ").append(String.format("%.0fx downsample", ds)).append("\n");
            }
            info.append("Classes: ").append(String.join(", ", classifier.getClassNames()));

            classifierInfoLabel.setText(info.toString());
            classifierInfoLabel.setStyle("-fx-text-fill: #333;");

            // Update channel requirements
            channelPanel.setRequiredChannelCount(classifier.getInputChannels());
            channelPanel.setExpectedChannels(classifier.getExpectedChannelNames());

            // Update tile size to match classifier
            tileSizeSpinner.getValueFactory().setValue(classifier.getInputWidth());

            // Show channel mapping visualization
            updateChannelMappingDisplay(classifier);

            updateValidation();
        }

        private void updateChannelMappingDisplay(ClassifierMetadata classifier) {
            channelMappingPanel.getChildren().clear();

            List<String> expected = classifier.getExpectedChannelNames();
            if (expected == null || expected.isEmpty()) {
                channelMappingPanel.setVisible(false);
                channelMappingPanel.setManaged(false);
                return;
            }

            List<String> available = getAvailableChannelNames();

            // Build grid: [Expected] -> [Mapped To] [Status] [Override]
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(4);
            grid.setPadding(new Insets(5));

            // Header row
            Label headerExpected = new Label("Expected");
            headerExpected.setStyle("-fx-font-weight: bold; -fx-text-fill: #333;");
            Label headerMapped = new Label("Mapped To");
            headerMapped.setStyle("-fx-font-weight: bold; -fx-text-fill: #333;");
            Label headerStatus = new Label("Status");
            headerStatus.setStyle("-fx-font-weight: bold; -fx-text-fill: #333;");
            grid.addRow(0, headerExpected, new Label("->"), headerMapped, headerStatus);

            int unmatchedCount = 0;

            for (int i = 0; i < expected.size(); i++) {
                String name = expected.get(i);
                String match = findBestMatch(name, available);

                Label expectedLabel = new Label(name);
                Label arrowLabel = new Label("->");
                arrowLabel.setStyle("-fx-text-fill: #999;");

                Label mappedLabel;
                Label statusIndicator;

                if (match != null && match.equalsIgnoreCase(name)) {
                    // Exact match
                    mappedLabel = new Label(match);
                    statusIndicator = new Label("[OK]");
                    statusIndicator.setStyle("-fx-text-fill: #388E3C; -fx-font-weight: bold;");
                    grid.addRow(i + 1, expectedLabel, arrowLabel, mappedLabel, statusIndicator);
                } else if (match != null) {
                    // Fuzzy match -- show with ComboBox override
                    mappedLabel = new Label(match);
                    mappedLabel.setStyle("-fx-text-fill: #F57C00;");
                    statusIndicator = new Label("[?]");
                    statusIndicator.setStyle("-fx-text-fill: #F57C00; -fx-font-weight: bold;");

                    ComboBox<String> overrideCombo = createOverrideComboBox(available, match, i);
                    grid.addRow(i + 1, expectedLabel, arrowLabel, mappedLabel, statusIndicator, overrideCombo);
                } else {
                    // No match
                    mappedLabel = new Label("(unmapped)");
                    mappedLabel.setStyle("-fx-text-fill: #D32F2F;");
                    statusIndicator = new Label("[X]");
                    statusIndicator.setStyle("-fx-text-fill: #D32F2F; -fx-font-weight: bold;");
                    unmatchedCount++;

                    ComboBox<String> overrideCombo = createOverrideComboBox(available, null, i);
                    grid.addRow(i + 1, expectedLabel, arrowLabel, mappedLabel, statusIndicator, overrideCombo);
                }
            }

            // Summary label
            Label summaryLabel = new Label();
            summaryLabel.setPadding(new Insets(5, 0, 0, 0));
            if (unmatchedCount == 0) {
                summaryLabel.setText("All channels matched");
                summaryLabel.setStyle("-fx-text-fill: #388E3C;");
            } else {
                summaryLabel.setText(unmatchedCount + " channel(s) need manual mapping");
                summaryLabel.setStyle("-fx-text-fill: #D32F2F; -fx-font-weight: bold;");
            }

            Label titleLabel = new Label("Channel Mapping:");
            titleLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #333;");

            channelMappingPanel.getChildren().addAll(titleLabel, grid, summaryLabel);
            channelMappingPanel.setVisible(true);
            channelMappingPanel.setManaged(true);
        }

        private ComboBox<String> createOverrideComboBox(List<String> available, String currentMatch, int position) {
            ComboBox<String> combo = new ComboBox<>(FXCollections.observableArrayList(available));
            combo.setPromptText("Select channel...");
            combo.setPrefWidth(160);
            if (currentMatch != null) {
                combo.setValue(currentMatch);
            }
            combo.valueProperty().addListener((obs, old, newVal) -> {
                if (newVal != null) {
                    channelPanel.setSelectedChannelByName(position, newVal);
                }
            });
            return combo;
        }

        private List<String> getAvailableChannelNames() {
            ImageData<BufferedImage> imageData = QP.getCurrentImageData();
            if (imageData == null || imageData.getServer() == null) {
                return Collections.emptyList();
            }
            return imageData.getServer().getMetadata().getChannels().stream()
                    .map(ch -> ch.getName())
                    .collect(Collectors.toList());
        }

        private String findBestMatch(String expectedName, List<String> available) {
            // 1. Exact case-insensitive match
            for (String name : available) {
                if (name.equalsIgnoreCase(expectedName)) return name;
            }
            // 2. Substring match (e.g., "DAPI" in "Channel_DAPI_01")
            for (String name : available) {
                if (name.toLowerCase().contains(expectedName.toLowerCase())
                        || expectedName.toLowerCase().contains(name.toLowerCase())) {
                    return name;
                }
            }
            return null;
        }

        private void updateOutputOptions(InferenceConfig.OutputType outputType) {
            boolean enableObjectOptions = (outputType == InferenceConfig.OutputType.OBJECTS);
            objectTypeCombo.setDisable(!enableObjectOptions);
            minObjectSizeSpinner.setDisable(!enableObjectOptions);
            holeFillingSpinner.setDisable(!enableObjectOptions);
            smoothingSpinner.setDisable(!enableObjectOptions);

            // Blend mode is relevant for OBJECTS output (Python-side blending)
            blendModeCombo.setDisable(!enableObjectOptions);
        }

        private void updateValidation() {
            ClassifierMetadata selected = classifierTable.getSelectionModel().getSelectedItem();
            boolean valid = selected != null && channelPanel.isValid();
            okButton.setDisable(!valid);
        }

        private InferenceDialogResult buildResult() {
            // Determine selected scope
            InferenceConfig.ApplicationScope scope;
            if (applyToWholeImageRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.WHOLE_IMAGE;
            } else if (applyToSelectedRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.SELECTED_ANNOTATIONS;
            } else {
                scope = InferenceConfig.ApplicationScope.ALL_ANNOTATIONS;
            }

            // Save dialog settings to preferences for next session
            DLClassifierPreferences.setLastOutputType(outputTypeCombo.getValue().name());
            DLClassifierPreferences.setLastBlendMode(blendModeCombo.getValue().name());
            DLClassifierPreferences.setSmoothing(smoothingSpinner.getValue());
            DLClassifierPreferences.setApplicationScope(scope.name());
            DLClassifierPreferences.setCreateBackup(createBackupCheck.isSelected());
            DLClassifierPreferences.setTileSize(tileSizeSpinner.getValue());
            DLClassifierPreferences.setTileOverlapPercent(overlapPercentSpinner.getValue());
            DLClassifierPreferences.setUseGPU(useGPUCheck.isSelected());
            DLClassifierPreferences.setDefaultObjectType(objectTypeCombo.getValue().name());
            DLClassifierPreferences.setMinObjectSizeMicrons(minObjectSizeSpinner.getValue());
            DLClassifierPreferences.setHoleFillingMicrons(holeFillingSpinner.getValue());

            ClassifierMetadata classifier = classifierTable.getSelectionModel().getSelectedItem();

            // Calculate pixel overlap from percentage
            int overlapPixels = (int) Math.round(
                    tileSizeSpinner.getValue() * overlapPercentSpinner.getValue() / 100.0);

            InferenceConfig inferenceConfig = InferenceConfig.builder()
                    .tileSize(tileSizeSpinner.getValue())
                    .overlap(overlapPixels)
                    .overlapPercent(overlapPercentSpinner.getValue())
                    .blendMode(blendModeCombo.getValue())
                    .outputType(outputTypeCombo.getValue())
                    .objectType(objectTypeCombo.getValue())
                    .minObjectSize(minObjectSizeSpinner.getValue())
                    .holeFilling(holeFillingSpinner.getValue())
                    .smoothing(smoothingSpinner.getValue())
                    .useGPU(useGPUCheck.isSelected())
                    .useTTA(useTTACheck.isSelected())
                    .multiPassAveraging(DLClassifierPreferences.isMultiPassAveraging())
                    .overlaySmoothingSigma(DLClassifierPreferences.getOverlaySmoothing())
                    .useCompactArgmaxOutput(
                            DLClassifierPreferences.isUseCompactArgmaxOutput())
                    .build();

            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            return new InferenceDialogResult(
                    classifier,
                    inferenceConfig,
                    channelConfig,
                    scope,
                    createBackupCheck.isSelected()
            );
        }

        private void copyInferenceScript(Button sourceButton) {
            ClassifierMetadata classifier = classifierTable.getSelectionModel().getSelectedItem();
            if (classifier == null) {
                showCopyFeedback(sourceButton, "No classifier selected");
                return;
            }

            // Build current config from dialog state
            int overlapPixels = (int) Math.round(
                    tileSizeSpinner.getValue() * overlapPercentSpinner.getValue() / 100.0);

            InferenceConfig config = InferenceConfig.builder()
                    .tileSize(tileSizeSpinner.getValue())
                    .overlap(overlapPixels)
                    .blendMode(blendModeCombo.getValue())
                    .outputType(outputTypeCombo.getValue())
                    .objectType(objectTypeCombo.getValue())
                    .minObjectSize(minObjectSizeSpinner.getValue())
                    .holeFilling(holeFillingSpinner.getValue())
                    .smoothing(smoothingSpinner.getValue())
                    .useGPU(useGPUCheck.isSelected())
                    .useTTA(useTTACheck.isSelected())
                    .overlaySmoothingSigma(DLClassifierPreferences.getOverlaySmoothing())
                    .useCompactArgmaxOutput(
                            DLClassifierPreferences.isUseCompactArgmaxOutput())
                    .build();

            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            InferenceConfig.ApplicationScope scope;
            if (applyToWholeImageRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.WHOLE_IMAGE;
            } else if (applyToSelectedRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.SELECTED_ANNOTATIONS;
            } else {
                scope = InferenceConfig.ApplicationScope.ALL_ANNOTATIONS;
            }

            String script = ScriptGenerator.generateInferenceScript(
                    classifier.getId(), config, channelConfig, scope);

            Clipboard clipboard = Clipboard.getSystemClipboard();
            ClipboardContent content = new ClipboardContent();
            content.putString(script);
            clipboard.setContent(content);

            showCopyFeedback(sourceButton, "Script copied to clipboard!");
        }

        private void showCopyFeedback(Button button, String message) {
            Tooltip tooltip = new Tooltip(message);
            tooltip.setAutoHide(true);
            tooltip.show(button, //
                    button.localToScreen(button.getBoundsInLocal()).getMinX(),
                    button.localToScreen(button.getBoundsInLocal()).getMinY() - 30);
            PauseTransition pause = new PauseTransition(Duration.seconds(2));
            pause.setOnFinished(e -> tooltip.hide());
            pause.play();
        }
    }
}
