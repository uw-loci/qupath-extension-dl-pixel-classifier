package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.classifier.handlers.UNetHandler;
import qupath.ext.dlclassifier.service.ApposeService;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

/**
 * Dialog for configuring self-supervised pretraining (SimCLR / BYOL)
 * of SMP encoder backbones (ResNet, EfficientNet, MobileNet, etc.).
 * <p>
 * Two source modes:
 * <ul>
 *   <li><b>Project images</b> (default) -- pick project images and extract
 *       tiles from user-selected annotation classes. Only annotated regions
 *       contribute tiles, limiting background.</li>
 *   <li><b>Pre-extracted folder</b> -- point at a directory of unlabeled
 *       image tiles.</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.3.3
 */
public class SSLPretrainingDialog {

    private static final Logger logger = LoggerFactory.getLogger(SSLPretrainingDialog.class);

    private static final Set<String> IMAGE_EXTENSIONS = Set.of(
            ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".raw");

    /** Backbone names that are ViT/foundation models -- not compatible with SMP SSL pretraining. */
    private static final Set<String> EXCLUDED_BACKBONES = Set.of(
            "h-optimus-0", "virchow", "hibou-l", "hibou-b", "midnight", "dinov2-large");

    /** CNN-only backbones for SSL pretraining. */
    private static final List<String> SSL_BACKBONES = UNetHandler.BACKBONES.stream()
            .filter(b -> !EXCLUDED_BACKBONES.contains(b))
            .toList();

    public enum SourceMode { PROJECT_IMAGES, PRE_EXTRACTED_FOLDER }

    /**
     * Result record containing all configuration needed to launch SSL
     * pretraining.
     */
    public record SSLPretrainingConfig(
            Map<String, Object> config,
            Path dataPath,
            Path outputDir,
            SourceMode sourceMode,
            List<ProjectImageEntry<BufferedImage>> projectImages,
            List<String> annotationClasses,
            int extractionTileSize,
            double extractionDownsample,
            int maxTilesTotal
    ) {}

    // Method & architecture controls
    private final ComboBox<String> methodCombo;
    private final ComboBox<String> backboneCombo;
    private final Spinner<Double> temperatureSpinner;
    private final Label temperatureLabel;
    private final Spinner<Integer> projectionDimSpinner;
    private final TextField sourceModelField;
    private final Label sourceModelInfoLabel;

    // Training controls
    private final Spinner<Integer> epochsSpinner;
    private final Spinner<Integer> batchSizeSpinner;
    private final Spinner<Double> learningRateSpinner;
    private final Spinner<Integer> warmupEpochsSpinner;

    // Live VRAM estimation
    private Label vramEstimateLabel;
    private int gpuTotalMb = 0;

    // Source-mode controls
    private final ToggleGroup sourceModeGroup = new ToggleGroup();
    private final RadioButton projectModeRadio;
    private final RadioButton folderModeRadio;

    // Project mode controls
    private final ListView<SSLImageItem> projectImagesList = new ListView<>();
    private final ListView<SSLClassItem> annotationClassList = new ListView<>();
    private final Spinner<Integer> extractionTileSpinner;
    private final ComboBox<Double> extractionDownsampleCombo;
    private final Spinner<Integer> maxTilesSpinner;
    private final Label projectSummaryLabel = new Label();

    // Folder mode controls
    private final TextField dataPathField;
    private final Label datasetInfoLabel;

    // Output controls
    private final TextField outputDirField;

    private SSLPretrainingDialog() {
        // --- Method selection ---
        methodCombo = new ComboBox<>(FXCollections.observableArrayList(
                "BYOL", "SimCLR (image pairs)"));
        methodCombo.setValue("BYOL");
        methodCombo.setMaxWidth(Double.MAX_VALUE);
        TooltipHelper.install(methodCombo,
                "Self-supervised pretraining method.\n\n" +
                "SimCLR: Contrastive learning -- learns by comparing\n" +
                "augmented views of the same image. Benefits from larger\n" +
                "batch sizes.\n\n" +
                "BYOL: Self-distillation -- no negative pairs needed.\n" +
                "Works well with smaller datasets and batch sizes.");

        // --- Backbone selection ---
        backboneCombo = new ComboBox<>(FXCollections.observableArrayList(SSL_BACKBONES));
        backboneCombo.setValue("resnet34");
        backboneCombo.setMaxWidth(Double.MAX_VALUE);
        backboneCombo.setCellFactory(lv -> createBackboneCell());
        backboneCombo.setButtonCell(createBackboneCell());
        TooltipHelper.install(backboneCombo,
                "CNN encoder backbone to pretrain.\n" +
                "Must match the backbone you plan to use for supervised training.\n" +
                "ResNet-34 is a good default for medium-sized datasets.");

        // --- Temperature (SimCLR only) ---
        temperatureSpinner = new Spinner<>(0.05, 1.0, 0.5, 0.05);
        temperatureSpinner.setEditable(true);
        temperatureSpinner.setPrefWidth(100);
        temperatureLabel = new Label("Temperature:");
        TooltipHelper.install(
                "SimCLR temperature parameter.\n" +
                "Lower values make the contrastive loss sharper.\n" +
                "0.5 is a common default; try 0.1-0.2 for small datasets.",
                temperatureLabel, temperatureSpinner);

        // Show/hide temperature based on method
        methodCombo.valueProperty().addListener((obs, old, newVal) -> {
            boolean isSimCLR = newVal != null && newVal.startsWith("SimCLR");
            temperatureSpinner.setVisible(isSimCLR);
            temperatureSpinner.setManaged(isSimCLR);
            temperatureLabel.setVisible(isSimCLR);
            temperatureLabel.setManaged(isSimCLR);
            updateBatchSizeDefault(newVal);
        });

        // --- Projection dimension ---
        projectionDimSpinner = new Spinner<>(64, 512, 256, 64);
        projectionDimSpinner.setEditable(true);
        projectionDimSpinner.setPrefWidth(100);
        TooltipHelper.install(projectionDimSpinner,
                "Dimension of the projection head output.\n" +
                "256 is the standard default for both SimCLR and BYOL.");

        // --- Source model (domain-adaptive pretraining) ---
        sourceModelField = new TextField();
        sourceModelField.setEditable(false);
        sourceModelField.setPromptText("(Optional) Select trained model for domain adaptation...");
        sourceModelField.setMaxWidth(Double.MAX_VALUE);
        TooltipHelper.install(sourceModelField,
                "Optional: load encoder weights from a previously trained\n" +
                "classifier model (.pt file). The encoder will start with\n" +
                "those learned features and adapt them to the new images\n" +
                "during SSL pretraining.\n\n" +
                "Leave empty to train the encoder from scratch.");

        sourceModelInfoLabel = new Label();
        sourceModelInfoLabel.setWrapText(true);
        sourceModelInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #336699;");
        sourceModelInfoLabel.setVisible(false);
        sourceModelInfoLabel.setManaged(false);

        // --- Training parameters ---
        epochsSpinner = new Spinner<>(10, 2000, 100, 10);
        epochsSpinner.setEditable(true);
        epochsSpinner.setPrefWidth(100);
        TooltipHelper.install(epochsSpinner,
                "Number of complete passes through the training data.\n" +
                "More epochs allow the model to learn more.\n" +
                "Typical range: 50-200 depending on dataset size.");

        batchSizeSpinner = new Spinner<>(2, 256, 64, 8);
        batchSizeSpinner.setEditable(true);
        batchSizeSpinner.setPrefWidth(100);
        TooltipHelper.install(batchSizeSpinner,
                "Number of images per batch.\n" +
                "SimCLR benefits from larger batches (64+).\n" +
                "BYOL works well with smaller batches (16-32).\n" +
                "Gradient accumulation is auto-applied if needed.");

        var lrFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(1e-5, 1e-2, 3e-4, 1e-5);
        lrFactory.setConverter(new javafx.util.StringConverter<>() {
            @Override public String toString(Double value) {
                return value == null ? "" : String.format("%.5f", value);
            }
            @Override public Double fromString(String string) {
                try { return Double.parseDouble(string.trim()); }
                catch (NumberFormatException e) { return lrFactory.getValue(); }
            }
        });
        learningRateSpinner = new Spinner<>(lrFactory);
        learningRateSpinner.setEditable(true);
        learningRateSpinner.setPrefWidth(120);
        TooltipHelper.install(learningRateSpinner,
                "Step size for the optimizer.\n" +
                "3e-4 (0.00030) is a good default for SSL pretraining.\n" +
                "Cosine annealing with warmup is applied automatically.");

        warmupEpochsSpinner = new Spinner<>(0, 50, 10, 1);
        warmupEpochsSpinner.setEditable(true);
        warmupEpochsSpinner.setPrefWidth(100);
        TooltipHelper.install(warmupEpochsSpinner,
                "Number of epochs to linearly ramp up the learning rate.\n" +
                "Prevents early instability. 10 is typical for SSL.");

        // --- Source mode radios ---
        projectModeRadio = new RadioButton("Project images (extract tiles from annotations)");
        projectModeRadio.setToggleGroup(sourceModeGroup);
        projectModeRadio.setSelected(true);
        TooltipHelper.install(projectModeRadio,
                "Extract pretraining tiles from images in the currently\n" +
                "open QuPath project. Only regions covered by annotations\n" +
                "of the selected classes contribute tiles, so empty slide\n" +
                "background doesn't waste training compute.");

        folderModeRadio = new RadioButton("Pre-extracted folder");
        folderModeRadio.setToggleGroup(sourceModeGroup);
        TooltipHelper.install(folderModeRadio,
                "Use an existing directory of image tiles.\n" +
                "Supported formats: PNG, TIFF, JPEG, BMP, RAW.\n" +
                "Subdirectories are scanned recursively.");

        // --- Project-mode controls ---
        projectImagesList.setCellFactory(lv -> new SSLImageCell());
        projectImagesList.setPrefHeight(130);
        TooltipHelper.install(projectImagesList,
                "Select which project images to extract tiles from.\n" +
                "Check the images you want to include in pretraining.\n" +
                "The number of classified annotations is shown per image.");

        annotationClassList.setCellFactory(lv -> new SSLClassCell());
        annotationClassList.setPrefHeight(80);
        annotationClassList.setPlaceholder(new Label("Select images to see available classes"));
        TooltipHelper.install(annotationClassList,
                "Select which annotation classes define regions of interest.\n" +
                "Tiles are only extracted from areas covered by annotations\n" +
                "of the checked classes. Uncheck classes you want to exclude\n" +
                "(e.g., Background).");

        extractionTileSpinner = new Spinner<>(128, 1024, 256, 64);
        extractionTileSpinner.setEditable(true);
        extractionTileSpinner.setPrefWidth(100);
        TooltipHelper.install(extractionTileSpinner,
                "Size of the image tiles extracted from each slide.\n" +
                "256 is a common default for SSL pretraining.");

        extractionDownsampleCombo = new ComboBox<>(FXCollections.observableArrayList(1.0, 2.0, 4.0, 8.0));
        extractionDownsampleCombo.setValue(1.0);
        extractionDownsampleCombo.setPrefWidth(100);
        TooltipHelper.install(extractionDownsampleCombo,
                "Downsample factor applied when reading tiles.\n" +
                "1 = full resolution, 2 = half, 4 = quarter.\n" +
                "Match the downsample you use for supervised training.");

        maxTilesSpinner = new Spinner<>(100, 200000, 5000, 500);
        maxTilesSpinner.setEditable(true);
        maxTilesSpinner.setPrefWidth(100);
        TooltipHelper.install(maxTilesSpinner,
                "Maximum tiles to keep across all selected images.\n" +
                "Prevents a single large WSI from overwhelming the pool.");

        projectSummaryLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        projectSummaryLabel.setWrapText(true);

        // --- Folder-mode controls ---
        dataPathField = new TextField();
        dataPathField.setPromptText("Directory of unlabeled image tiles...");
        dataPathField.setPrefWidth(250);
        TooltipHelper.install(dataPathField,
                "Path to a directory of unlabeled image tiles.\n" +
                "Supported: PNG, TIFF, JPEG, BMP, RAW.\n" +
                "Subdirectories are scanned recursively.");

        datasetInfoLabel = new Label();
        datasetInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        datasetInfoLabel.setWrapText(true);

        // --- Output controls ---
        outputDirField = new TextField();
        outputDirField.setPromptText("Output directory for encoder weights...");
        outputDirField.setPrefWidth(250);
        TooltipHelper.install(outputDirField,
                "Directory where the pretrained encoder weights (model.pt)\n" +
                "and metadata.json will be saved.\n" +
                "Defaults to a timestamped folder in the project directory.");

        var qupath = QuPathGUI.getInstance();
        if (qupath != null && qupath.getProject() != null) {
            try {
                Path projectDir = qupath.getProject().getPath().getParent();
                String timestamp = java.time.LocalDateTime.now()
                        .format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
                outputDirField.setText(projectDir.resolve("ssl_pretrained")
                        .resolve(backboneCombo.getValue() + "_"
                                + getSelectedMethod() + "_" + timestamp)
                        .toString());
            } catch (Exception e) {
                logger.debug("Could not set default output dir: {}", e.getMessage());
            }
        }
        // Update output dir when backbone or method changes
        backboneCombo.valueProperty().addListener((obs, old, newVal) -> updateOutputDir());
        methodCombo.valueProperty().addListener((obs, old, newVal) -> updateOutputDir());

        // Populate project images list
        populateProjectImages();

        // Cache GPU memory for live VRAM estimation
        try {
            ApposeService appose = ApposeService.getInstance();
            if (appose.isAvailable() && "cuda".equals(appose.getGpuType())) {
                gpuTotalMb = appose.getLastGpuMemoryMb();
            }
        } catch (Exception e) {
            logger.debug("Could not get GPU memory info: {}", e.getMessage());
        }
    }

    public static Optional<SSLPretrainingConfig> showDialog() {
        SSLPretrainingDialog controller = new SSLPretrainingDialog();
        return controller.buildAndShow();
    }

    private Optional<SSLPretrainingConfig> buildAndShow() {
        Dialog<SSLPretrainingConfig> dialog = new Dialog<>();
        dialog.initOwner(QuPathGUI.getInstance().getStage());
        dialog.setTitle("SSL Pretrain Encoder (SimCLR / BYOL)");
        dialog.setResizable(true);

        ButtonType startType = new ButtonType("Start Pretraining", ButtonBar.ButtonData.OK_DONE);
        ButtonType cancelType = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);
        dialog.getDialogPane().getButtonTypes().addAll(startType, cancelType);

        VBox content = new VBox(10);
        content.setPadding(new Insets(10));

        Hyperlink tipsLink = new Hyperlink("SSL pretraining tips");
        tipsLink.setOnAction(e -> showSSLTipsDialog());
        tipsLink.setStyle("-fx-font-size: 11px;");
        content.getChildren().add(tipsLink);

        content.getChildren().add(buildMethodSection());
        content.getChildren().add(buildTrainingSection());
        content.getChildren().add(buildDataSection(dialog));
        content.getChildren().add(buildOutputSection(dialog));

        ScrollPane scrollPane = new ScrollPane(content);
        scrollPane.setFitToWidth(true);
        scrollPane.setPrefViewportHeight(650);
        dialog.getDialogPane().setContent(scrollPane);
        dialog.getDialogPane().setPrefWidth(580);

        // Enable Start only when valid
        Button startButton = (Button) dialog.getDialogPane().lookupButton(startType);
        startButton.disableProperty().bind(Bindings.createBooleanBinding(
                () -> !isStartValid(),
                sourceModeGroup.selectedToggleProperty(),
                dataPathField.textProperty(),
                projectImagesList.getItems(),
                annotationClassList.getItems()));

        dialog.setResultConverter(button -> button == startType ? buildConfig() : null);
        return dialog.showAndWait();
    }

    // ==================== Section Builders ====================

    private TitledPane buildMethodSection() {
        VBox root = new VBox(8);
        root.setPadding(new Insets(10));

        GridPane grid = new GridPane();
        grid.setHgap(10); grid.setVgap(8);
        // Ensure the control column stretches to fill available width
        javafx.scene.layout.ColumnConstraints labelCol = new javafx.scene.layout.ColumnConstraints();
        javafx.scene.layout.ColumnConstraints controlCol = new javafx.scene.layout.ColumnConstraints();
        controlCol.setHgrow(Priority.ALWAYS);
        grid.getColumnConstraints().addAll(labelCol, controlCol);

        Label methodLabel = new Label("SSL Method:");
        TooltipHelper.install(
                "Self-supervised pretraining method.\n\n" +
                "SimCLR: Contrastive learning. Benefits from larger batches.\n" +
                "BYOL: Self-distillation. Works well with smaller datasets.",
                methodLabel, methodCombo);
        grid.add(methodLabel, 0, 0);
        grid.add(methodCombo, 1, 0);

        Label backboneLabel = new Label("Backbone:");
        TooltipHelper.install(
                "CNN encoder backbone to pretrain.\n" +
                "Must match the backbone you plan to use for supervised training.\n" +
                "ResNet-34 is a good default for medium-sized datasets.",
                backboneLabel, backboneCombo);
        grid.add(backboneLabel, 0, 1);
        grid.add(backboneCombo, 1, 1);

        grid.add(temperatureLabel, 0, 2);
        grid.add(temperatureSpinner, 1, 2);

        Label projDimLabel = new Label("Projection dim:");
        TooltipHelper.install(
                "Dimension of the projection head output.\n" +
                "256 is the standard default for both SimCLR and BYOL.",
                projDimLabel, projectionDimSpinner);
        grid.add(projDimLabel, 0, 3);
        grid.add(projectionDimSpinner, 1, 3);

        root.getChildren().add(grid);

        // --- Domain adaptation: source model (optional) ---
        Label sourceLabel = new Label("Initialize from trained model (optional):");
        sourceLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 11px;");
        TooltipHelper.install(sourceLabel,
                "For domain adaptation: load encoder weights from a\n" +
                "previously trained model, then continue SSL pretraining\n" +
                "on new unlabeled images from the target domain.\n\n" +
                "Leave empty to train from scratch (standard SSL).");

        Button sourceModelBrowse = new Button("Browse...");
        sourceModelBrowse.setOnAction(e -> browseSourceModel());
        TooltipHelper.install(sourceModelBrowse,
                "Browse for a trained classifier model (.pt file)\n" +
                "to use as the starting point for domain adaptation.");

        Button sourceModelClear = new Button("Clear");
        TooltipHelper.install(sourceModelClear,
                "Remove the source model and train from scratch.");
        sourceModelClear.setOnAction(e -> {
            sourceModelField.setText("");
            sourceModelInfoLabel.setText("");
            sourceModelInfoLabel.setVisible(false);
            sourceModelInfoLabel.setManaged(false);
        });

        HBox sourceRow = new HBox(5, sourceModelField, sourceModelBrowse, sourceModelClear);
        sourceRow.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        HBox.setHgrow(sourceModelField, Priority.ALWAYS);

        root.getChildren().addAll(sourceLabel, sourceRow, sourceModelInfoLabel);

        TitledPane pane = new TitledPane("Method & Architecture", root);
        pane.setExpanded(true); pane.setCollapsible(false);
        return pane;
    }

    private void browseSourceModel() {
        javafx.stage.FileChooser chooser = new javafx.stage.FileChooser();
        chooser.setTitle("Select Trained Model (.pt)");
        chooser.getExtensionFilters().add(
                new javafx.stage.FileChooser.ExtensionFilter("PyTorch model", "*.pt"));

        // Default to user's classifiers directory
        java.io.File modelsDir = new java.io.File(
                System.getProperty("user.home"), ".dlclassifier/models");
        if (modelsDir.isDirectory()) {
            chooser.setInitialDirectory(modelsDir);
        }

        java.io.File file = chooser.showOpenDialog(null);
        if (file == null) return;

        sourceModelField.setText(file.getAbsolutePath());

        // Try to read companion metadata.json
        java.io.File metaFile = new java.io.File(file.getParentFile(), "metadata.json");
        if (metaFile.exists()) {
            try (java.io.Reader reader = java.nio.file.Files.newBufferedReader(
                    metaFile.toPath(), java.nio.charset.StandardCharsets.UTF_8)) {
                com.google.gson.JsonObject root =
                        new com.google.gson.Gson().fromJson(reader, com.google.gson.JsonObject.class);

                StringBuilder info = new StringBuilder("Source: ");
                com.google.gson.JsonObject arch = root.has("architecture")
                        ? root.getAsJsonObject("architecture") : null;
                if (arch != null) {
                    if (arch.has("type")) info.append(arch.get("type").getAsString());
                    if (arch.has("backbone"))
                        info.append(" (").append(arch.get("backbone").getAsString()).append(")");

                    // Auto-select matching backbone if available
                    String backbone = arch.has("backbone")
                            ? arch.get("backbone").getAsString() : null;
                    if (backbone != null && backboneCombo.getItems().contains(backbone)) {
                        backboneCombo.setValue(backbone);
                    }
                }
                if (root.has("classes")) {
                    int nClasses = root.getAsJsonArray("classes").size();
                    info.append(", ").append(nClasses).append(" classes");
                }

                sourceModelInfoLabel.setText(info.toString());
                sourceModelInfoLabel.setVisible(true);
                sourceModelInfoLabel.setManaged(true);
            } catch (Exception ex) {
                logger.debug("Could not read source model metadata: {}", ex.getMessage());
                sourceModelInfoLabel.setText("Model selected (no metadata found)");
                sourceModelInfoLabel.setVisible(true);
                sourceModelInfoLabel.setManaged(true);
            }
        } else {
            sourceModelInfoLabel.setText("Model selected (no metadata.json found)");
            sourceModelInfoLabel.setVisible(true);
            sourceModelInfoLabel.setManaged(true);
        }
    }

    private TitledPane buildTrainingSection() {
        GridPane grid = new GridPane();
        grid.setHgap(10); grid.setVgap(8); grid.setPadding(new Insets(10));

        Label epochsLabel = new Label("Epochs:");
        TooltipHelper.install(
                "Number of complete passes through the training data.\n" +
                "More epochs help, but diminishing returns past 200.\n" +
                "Typical: 50-200 depending on dataset size.",
                epochsLabel, epochsSpinner);
        grid.add(epochsLabel, 0, 0);
        grid.add(epochsSpinner, 1, 0);

        Label batchSizeLabel = new Label("Batch size:");
        TooltipHelper.install(
                "Number of images per training step.\n" +
                "SimCLR benefits from larger batches (64+).\n" +
                "BYOL works well with smaller batches (16-32).\n" +
                "Gradient accumulation is auto-applied if needed.",
                batchSizeLabel, batchSizeSpinner);
        grid.add(batchSizeLabel, 0, 1);
        grid.add(batchSizeSpinner, 1, 1);

        Label learningRateLabel = new Label("Learning rate:");
        TooltipHelper.install(
                "Step size for the optimizer.\n" +
                "3e-4 (0.00030) is a good default for SSL pretraining.\n" +
                "Cosine annealing with warmup is applied automatically.",
                learningRateLabel, learningRateSpinner);
        grid.add(learningRateLabel, 0, 2);
        grid.add(learningRateSpinner, 1, 2);

        Label warmupLabel = new Label("Warmup epochs:");
        TooltipHelper.install(
                "Number of epochs to linearly ramp up the learning rate.\n" +
                "Prevents early instability. 10 is typical for SSL.",
                warmupLabel, warmupEpochsSpinner);
        grid.add(warmupLabel, 0, 3);
        grid.add(warmupEpochsSpinner, 1, 3);

        // Live VRAM estimate
        vramEstimateLabel = new Label();
        vramEstimateLabel.setWrapText(true);
        vramEstimateLabel.setStyle("-fx-font-size: 11px;");
        grid.add(vramEstimateLabel, 0, 4, 2, 1);

        // Wire VRAM estimation listeners
        backboneCombo.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
        batchSizeSpinner.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
        extractionTileSpinner.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
        methodCombo.valueProperty().addListener((obs, old, newVal) -> updateVramEstimate());
        updateVramEstimate();

        TitledPane pane = new TitledPane("Training Parameters", grid);
        pane.setExpanded(true); pane.setCollapsible(false);
        return pane;
    }

    private TitledPane buildDataSection(Dialog<?> dialog) {
        VBox root = new VBox(6);
        root.setPadding(new Insets(10));

        HBox modeRow = new HBox(12, projectModeRadio, folderModeRadio);
        root.getChildren().add(modeRow);

        // --- Project mode subpanel ---
        VBox projectPanel = new VBox(6);

        // Image selection
        Label imagesLabel = new Label("Images:");
        imagesLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 11px;");
        TooltipHelper.install(imagesLabel,
                "Select which project images to use for pretraining.\n" +
                "Tiles are extracted from annotated regions of these images.");
        Button selectAllBtn = new Button("Select All");
        selectAllBtn.setOnAction(e -> {
            for (SSLImageItem item : projectImagesList.getItems()) item.selected = true;
            projectImagesList.refresh();
            updateAnnotationClasses();
        });
        TooltipHelper.install(selectAllBtn, "Select all images for pretraining.");
        Button selectNoneBtn = new Button("Select None");
        selectNoneBtn.setOnAction(e -> {
            for (SSLImageItem item : projectImagesList.getItems()) item.selected = false;
            projectImagesList.refresh();
            updateAnnotationClasses();
        });
        TooltipHelper.install(selectNoneBtn, "Deselect all images.");
        HBox imageButtonRow = new HBox(8, selectAllBtn, selectNoneBtn);
        projectPanel.getChildren().addAll(imagesLabel, projectImagesList, imageButtonRow);

        // Annotation class selection
        Label classesLabel = new Label("Annotation classes (tile ROI):");
        classesLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 11px;");
        TooltipHelper.install(classesLabel,
                "Only regions covered by annotations of the selected\n" +
                "classes will be used to extract tiles. Uncheck classes\n" +
                "you want to exclude (e.g., Background).");
        Button selectAllClassesBtn = new Button("All");
        selectAllClassesBtn.setOnAction(e -> {
            for (SSLClassItem item : annotationClassList.getItems()) item.selected = true;
            annotationClassList.refresh();
            updateProjectSummary();
        });
        TooltipHelper.install(selectAllClassesBtn, "Select all annotation classes.");
        Button selectNoClassesBtn = new Button("None");
        selectNoClassesBtn.setOnAction(e -> {
            for (SSLClassItem item : annotationClassList.getItems()) item.selected = false;
            annotationClassList.refresh();
            updateProjectSummary();
        });
        TooltipHelper.install(selectNoClassesBtn, "Deselect all annotation classes.");
        HBox classButtonRow = new HBox(8, selectAllClassesBtn, selectNoClassesBtn);
        projectPanel.getChildren().addAll(classesLabel, annotationClassList, classButtonRow);

        // Extraction parameters
        GridPane extractionGrid = new GridPane();
        extractionGrid.setHgap(10); extractionGrid.setVgap(8);
        extractionGrid.setPadding(new Insets(6, 0, 0, 0));
        Label extTileSizeLabel = new Label("Tile size:");
        TooltipHelper.install(
                "Size of the image tiles extracted from each slide.\n" +
                "256 is a common default for SSL pretraining.\n" +
                "Should match or be close to the tile size used in training.",
                extTileSizeLabel, extractionTileSpinner);
        extractionGrid.add(extTileSizeLabel, 0, 0);
        extractionGrid.add(extractionTileSpinner, 1, 0);
        Label extDownsampleLabel = new Label("Downsample:");
        TooltipHelper.install(
                "Downsample factor applied when reading tiles.\n" +
                "1 = full resolution, 2 = half, 4 = quarter.\n" +
                "Match the downsample you use for supervised training.",
                extDownsampleLabel, extractionDownsampleCombo);
        extractionGrid.add(extDownsampleLabel, 0, 1);
        extractionGrid.add(extractionDownsampleCombo, 1, 1);
        Label extMaxTilesLabel = new Label("Max tiles (total):");
        TooltipHelper.install(
                "Maximum tiles to keep across all selected images.\n" +
                "If more tiles are extracted, the surplus is randomly\n" +
                "discarded. Prevents one large WSI from dominating.",
                extMaxTilesLabel, maxTilesSpinner);
        extractionGrid.add(extMaxTilesLabel, 0, 2);
        extractionGrid.add(maxTilesSpinner, 1, 2);
        projectPanel.getChildren().addAll(extractionGrid, projectSummaryLabel);

        projectPanel.visibleProperty().bind(projectModeRadio.selectedProperty());
        projectPanel.managedProperty().bind(projectModeRadio.selectedProperty());

        // --- Folder mode subpanel ---
        VBox folderPanel = new VBox(6);
        GridPane folderGrid = new GridPane();
        folderGrid.setHgap(10); folderGrid.setVgap(8);
        Button browseBtn = new Button("Browse...");
        browseBtn.setOnAction(e -> {
            DirectoryChooser dc = new DirectoryChooser();
            dc.setTitle("Select Image Tile Directory");
            File dir = dc.showDialog(dialog.getDialogPane().getScene().getWindow());
            if (dir != null) {
                dataPathField.setText(dir.getAbsolutePath());
                scanDatasetAndUpdateInfo(dir);
            }
        });
        TooltipHelper.install(browseBtn, "Browse for a directory of image tiles.");
        Label imageDirLabel = new Label("Image directory:");
        TooltipHelper.install(
                "Path to a directory of unlabeled image tiles.\n" +
                "Supported: PNG, TIFF, JPEG, BMP, RAW.\n" +
                "Subdirectories are scanned recursively.",
                imageDirLabel, dataPathField);
        folderGrid.add(imageDirLabel, 0, 0);
        folderGrid.add(dataPathField, 1, 0);
        GridPane.setHgrow(dataPathField, Priority.ALWAYS);
        folderGrid.add(browseBtn, 2, 0);
        folderGrid.add(datasetInfoLabel, 0, 1, 3, 1);
        folderPanel.getChildren().add(folderGrid);
        folderPanel.visibleProperty().bind(folderModeRadio.selectedProperty());
        folderPanel.managedProperty().bind(folderModeRadio.selectedProperty());

        root.getChildren().addAll(projectPanel, folderPanel);

        TitledPane pane = new TitledPane("Data", root);
        pane.setExpanded(true); pane.setCollapsible(false);
        return pane;
    }

    private TitledPane buildOutputSection(Dialog<?> dialog) {
        GridPane grid = new GridPane();
        grid.setHgap(10); grid.setVgap(8); grid.setPadding(new Insets(10));
        Button browseBtn = new Button("Browse...");
        browseBtn.setOnAction(e -> {
            DirectoryChooser dc = new DirectoryChooser();
            dc.setTitle("Select Output Directory");
            if (!outputDirField.getText().isBlank()) {
                File current = new File(outputDirField.getText());
                if (current.isDirectory()) dc.setInitialDirectory(current);
                else if (current.getParentFile() != null && current.getParentFile().isDirectory())
                    dc.setInitialDirectory(current.getParentFile());
            }
            File dir = dc.showDialog(dialog.getDialogPane().getScene().getWindow());
            if (dir != null) outputDirField.setText(dir.getAbsolutePath());
        });
        TooltipHelper.install(browseBtn, "Browse for an output directory.");
        Label outputDirLabel = new Label("Output directory:");
        TooltipHelper.install(
                "Directory where the pretrained encoder weights (model.pt)\n" +
                "and metadata.json will be saved.\n" +
                "Defaults to a timestamped folder in the project directory.",
                outputDirLabel, outputDirField);
        grid.add(outputDirLabel, 0, 0);
        grid.add(outputDirField, 1, 0);
        GridPane.setHgrow(outputDirField, Priority.ALWAYS);
        grid.add(browseBtn, 2, 0);
        TitledPane pane = new TitledPane("Output", grid);
        pane.setExpanded(true); pane.setCollapsible(false);
        return pane;
    }

    // ==================== Config Building ====================

    private SSLPretrainingConfig buildConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put("method", getSelectedMethod());
        config.put("encoder_name", backboneCombo.getValue());
        config.put("epochs", epochsSpinner.getValue());
        config.put("batch_size", batchSizeSpinner.getValue());
        config.put("learning_rate", learningRateSpinner.getValue());
        config.put("warmup_epochs", warmupEpochsSpinner.getValue());
        config.put("temperature", temperatureSpinner.getValue());
        config.put("projection_dim", projectionDimSpinner.getValue());
        config.put("tile_size", extractionTileSpinner.getValue());

        // Domain-adaptive pretraining: pass source model path if set
        String sourceModel = sourceModelField.getText();
        if (sourceModel != null && !sourceModel.isBlank()) {
            config.put("pretrained_model_path", sourceModel);
        }

        Path outputDir;
        if (outputDirField.getText().isBlank()) {
            outputDir = Path.of(System.getProperty("java.io.tmpdir"), "ssl_pretrained");
        } else {
            outputDir = Path.of(outputDirField.getText());
        }

        if (projectModeRadio.isSelected()) {
            List<ProjectImageEntry<BufferedImage>> selected = projectImagesList.getItems().stream()
                    .filter(i -> i.selected)
                    .map(i -> i.entry)
                    .collect(Collectors.toList());
            List<String> selectedClasses = annotationClassList.getItems().stream()
                    .filter(i -> i.selected)
                    .map(i -> i.className)
                    .collect(Collectors.toList());
            return new SSLPretrainingConfig(
                    config, null, outputDir,
                    SourceMode.PROJECT_IMAGES, selected, selectedClasses,
                    extractionTileSpinner.getValue(),
                    extractionDownsampleCombo.getValue(),
                    maxTilesSpinner.getValue());
        } else {
            return new SSLPretrainingConfig(
                    config, Path.of(dataPathField.getText()), outputDir,
                    SourceMode.PRE_EXTRACTED_FOLDER, null, null,
                    0, 1.0, 0);
        }
    }

    // ==================== Validation ====================

    private boolean isStartValid() {
        if (projectModeRadio.isSelected()) {
            long nSelected = projectImagesList.getItems().stream()
                    .filter(i -> i.selected).count();
            long nSelectedClasses = annotationClassList.getItems().stream()
                    .filter(i -> i.selected).count();
            return nSelected > 0 && nSelectedClasses > 0;
        } else {
            String path = dataPathField.getText();
            return path != null && !path.isBlank() && new File(path).isDirectory();
        }
    }

    // ==================== Project Image Population ====================

    private void populateProjectImages() {
        var qupath = QuPathGUI.getInstance();
        if (qupath == null || qupath.getProject() == null) return;

        Project<BufferedImage> project = qupath.getProject();
        for (var entry : project.getImageList()) {
            String name = entry.getImageName();
            int annotationCount = 0;
            try {
                var hierarchy = entry.readHierarchy();
                annotationCount = (int) hierarchy.getAnnotationObjects().stream()
                        .filter(a -> a.getPathClass() != null)
                        .count();
            } catch (Exception e) {
                logger.debug("Could not read hierarchy for {}: {}", name, e.getMessage());
            }
            projectImagesList.getItems().add(new SSLImageItem(entry, name, annotationCount, true));
        }

        updateAnnotationClasses();
    }

    private void updateAnnotationClasses() {
        Set<String> classNames = new TreeSet<>();
        for (SSLImageItem item : projectImagesList.getItems()) {
            if (!item.selected) continue;
            try {
                var hierarchy = item.entry.readHierarchy();
                for (PathObject ann : hierarchy.getAnnotationObjects()) {
                    if (ann.getPathClass() != null) {
                        classNames.add(ann.getPathClass().getName());
                    }
                }
            } catch (Exception e) {
                logger.debug("Could not read hierarchy for {}: {}", item.name, e.getMessage());
            }
        }

        // Preserve existing selection state where possible
        Map<String, Boolean> previousState = new HashMap<>();
        for (SSLClassItem item : annotationClassList.getItems()) {
            previousState.put(item.className, item.selected);
        }

        annotationClassList.getItems().clear();
        for (String className : classNames) {
            boolean wasSelected = previousState.getOrDefault(className, true);
            annotationClassList.getItems().add(new SSLClassItem(className, wasSelected));
        }

        updateProjectSummary();
    }

    private void updateProjectSummary() {
        long nImages = projectImagesList.getItems().stream().filter(i -> i.selected).count();
        long nClasses = annotationClassList.getItems().stream().filter(i -> i.selected).count();
        projectSummaryLabel.setText(
                nImages + " image(s) selected, " + nClasses + " annotation class(es) selected");
    }

    /** Returns "simclr" or "byol" from the display combo value. */
    private String getSelectedMethod() {
        String val = methodCombo.getValue();
        if (val == null) return "byol";
        return val.startsWith("SimCLR") ? "simclr" : "byol";
    }

    private void updateVramEstimate() {
        if (vramEstimateLabel == null || gpuTotalMb <= 0) {
            if (vramEstimateLabel != null) vramEstimateLabel.setText("");
            return;
        }
        try {
            String backbone = backboneCombo.getValue();
            int tileSize = extractionTileSpinner.getValue();
            int batchSize = batchSizeSpinner.getValue();
            String method = getSelectedMethod();

            double modelMb = estimateModelSizeMb(backbone);
            // BYOL has ~2x encoder memory (online + target networks)
            double modelFactor = "byol".equals(method) ? 2.0 : 1.0;
            // SSL: model + optimizer (3x) + activations (~4x CNN, halved for AMP)
            double actMultiplier = 4.0 * 0.6; // assume mixed precision
            double areaScale = (double)(tileSize * tileSize) / (256.0 * 256.0);
            double estimatedMb = modelMb * modelFactor
                    * (1 + 3 + actMultiplier * areaScale * batchSize);

            double budgetMb = gpuTotalMb * 0.85;
            double pct = (estimatedMb / gpuTotalMb) * 100;

            String text = String.format("Est. VRAM: ~%.0f MB / %,d MB (%.0f%%)",
                    estimatedMb, gpuTotalMb, pct);

            if (estimatedMb > budgetMb) {
                vramEstimateLabel.setStyle(
                        "-fx-font-size: 11px; -fx-text-fill: #CC0000; -fx-font-weight: bold;");
                // Suggest max batch that fits
                double perBatchMb = estimatedMb / batchSize;
                int maxBatch = Math.max(1, (int)(budgetMb / perBatchMb));
                text += String.format("  --  EXCEEDS GPU! Try batch %d or smaller tiles",
                        maxBatch);
            } else if (pct > 75) {
                vramEstimateLabel.setStyle(
                        "-fx-font-size: 11px; -fx-text-fill: #CC7A00; -fx-font-weight: bold;");
                text += "  --  tight, may OOM";
            } else {
                vramEstimateLabel.setStyle(
                        "-fx-font-size: 11px; -fx-text-fill: #228B22;");
            }

            vramEstimateLabel.setText(text);
        } catch (Exception e) {
            vramEstimateLabel.setText("");
            logger.debug("VRAM estimate update failed: {}", e.getMessage());
        }
    }

    /**
     * Estimates model parameter size in MB for SMP backbones.
     * SSL wraps the encoder in a projection head, but the encoder
     * dominates the memory footprint.
     */
    private static double estimateModelSizeMb(String backbone) {
        if (backbone == null) return 30.0;
        return switch (backbone.toLowerCase()) {
            case "resnet18" -> 47.0;
            case "resnet34" -> 87.0;
            case "resnet50", "resnet50_lunit-swav", "resnet50_lunit-bt",
                 "resnet50_kather100k", "resnet50_tcga-brca" -> 100.0;
            case "efficientnet-b0" -> 21.0;
            case "efficientnet-b1" -> 31.0;
            case "efficientnet-b2" -> 36.0;
            case "mobilenet_v2" -> 14.0;
            default -> 30.0;
        };
    }

    private void updateBatchSizeDefault(String method) {
        if (method != null && method.startsWith("SimCLR")) {
            batchSizeSpinner.getValueFactory().setValue(64);
        } else {
            batchSizeSpinner.getValueFactory().setValue(32);
        }
    }

    private void updateOutputDir() {
        String current = outputDirField.getText();
        if (current.contains("ssl_pretrained" + File.separator)) {
            try {
                Path parent = Path.of(current).getParent();
                if (parent != null && parent.getFileName() != null
                        && parent.toString().contains("ssl_pretrained")) {
                    String timestamp = java.time.LocalDateTime.now()
                            .format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
                    outputDirField.setText(parent.resolve(
                            backboneCombo.getValue() + "_"
                                    + getSelectedMethod() + "_" + timestamp)
                            .toString());
                }
            } catch (Exception ignored) {}
        }
    }

    // ==================== Folder Mode ====================

    private void scanDatasetAndUpdateInfo(File dir) {
        Thread scanner = new Thread(() -> {
            try {
                long count = Files.walk(dir.toPath())
                        .filter(p -> IMAGE_EXTENSIONS.contains(
                                getExtension(p.toString()).toLowerCase()))
                        .count();
                Platform.runLater(() -> {
                    if (count == 0) {
                        datasetInfoLabel.setText("No image files found in this directory.");
                    } else {
                        datasetInfoLabel.setText(count + " image file(s) found.");
                        // Auto-suggest epochs
                        if (count < 50) epochsSpinner.getValueFactory().setValue(500);
                        else if (count < 200) epochsSpinner.getValueFactory().setValue(300);
                        else if (count < 1000) epochsSpinner.getValueFactory().setValue(100);
                        else epochsSpinner.getValueFactory().setValue(50);
                    }
                });
            } catch (Exception e) {
                Platform.runLater(() ->
                    datasetInfoLabel.setText("Error scanning directory: " + e.getMessage()));
            }
        }, "ssl-dataset-scanner");
        scanner.setDaemon(true);
        scanner.start();
    }

    private static String getExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot >= 0 ? filename.substring(dot) : "";
    }

    // ==================== Tips Dialog ====================

    private void showSSLTipsDialog() {
        Dialogs.showMessageDialog("SSL Pretraining Tips",
                "Self-supervised pretraining trains the encoder backbone\n" +
                "to learn useful visual features without labels.\n\n" +
                "SimCLR: Best with large batch sizes (64+). The encoder\n" +
                "learns to map augmented views of the same image close\n" +
                "together in feature space.\n\n" +
                "BYOL: Works well with smaller datasets and batch sizes.\n" +
                "Uses a teacher-student framework with exponential moving\n" +
                "average updates.\n\n" +
                "After pretraining, load the encoder weights in the\n" +
                "Training dialog via 'Use SSL pretrained encoder'.\n" +
                "The pretrained backbone gives better results than\n" +
                "ImageNet weights for domain-specific data.\n\n" +
                "Tips:\n" +
                " - Use 200+ tiles for meaningful pretraining\n" +
                " - Select annotation classes that cover tissue regions\n" +
                " - Match the backbone to what you'll use in training\n" +
                " - 100-200 epochs is usually sufficient");
    }

    // ==================== Cell Factories ====================

    private ListCell<String> createBackboneCell() {
        return new ListCell<>() {
            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    setText(UNetHandler.getStaticBackboneDisplayName(item));
                }
            }
        };
    }

    // ==================== Data Models ====================

    /** Image list item with selection state. */
    static class SSLImageItem {
        final ProjectImageEntry<BufferedImage> entry;
        final String name;
        final int annotationCount;
        boolean selected;

        SSLImageItem(ProjectImageEntry<BufferedImage> entry, String name,
                     int annotationCount, boolean selected) {
            this.entry = entry;
            this.name = name;
            this.annotationCount = annotationCount;
            this.selected = selected;
        }
    }

    /** Annotation class item with selection state. */
    static class SSLClassItem {
        final String className;
        boolean selected;

        SSLClassItem(String className, boolean selected) {
            this.className = className;
            this.selected = selected;
        }
    }

    /** Custom cell for image list with checkbox. */
    private class SSLImageCell extends ListCell<SSLImageItem> {
        private final CheckBox checkBox = new CheckBox();

        @Override
        protected void updateItem(SSLImageItem item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setGraphic(null);
                setText(null);
            } else {
                checkBox.setSelected(item.selected);
                checkBox.setOnAction(e -> {
                    item.selected = checkBox.isSelected();
                    updateAnnotationClasses();
                });
                String label = item.name;
                if (item.annotationCount > 0) {
                    label += " (" + item.annotationCount + " annotations)";
                } else {
                    label += " (no annotations)";
                }
                setText(label);
                setGraphic(checkBox);
            }
        }
    }

    /** Custom cell for annotation class list with checkbox. */
    private class SSLClassCell extends ListCell<SSLClassItem> {
        private final CheckBox checkBox = new CheckBox();

        @Override
        protected void updateItem(SSLClassItem item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setGraphic(null);
                setText(null);
            } else {
                checkBox.setSelected(item.selected);
                checkBox.setOnAction(e -> {
                    item.selected = checkBox.isSelected();
                    updateProjectSummary();
                });
                setText(item.className);
                setGraphic(checkBox);
            }
        }
    }
}
