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
import qupath.ext.dlclassifier.classifier.handlers.MuViTHandler;
import qupath.ext.dlclassifier.utilities.TissueDetectionUtility;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.PathObject;
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

/**
 * Dialog for configuring MAE (Masked Autoencoder) self-supervised pretraining
 * of a MuViT encoder.
 * <p>
 * Two source modes:
 * <ul>
 *   <li><b>Project images</b> (default) -- pick project images and extract
 *       tiles just-in-time from "Tissue"-classed annotations. Includes a
 *       Detect Tissue helper that runs {@link TissueDetectionUtility} on
 *       the selected images.</li>
 *   <li><b>Pre-extracted folder</b> -- point at a directory of unlabeled
 *       image tiles. Backwards-compatible with previous behaviour.</li>
 * </ul>
 * The caller (SetupDLClassifier) is responsible for extracting tiles in
 * project mode before handing the resulting {@link MAEPretrainingConfig#dataPath()}
 * to the Python backend.
 *
 * @author UW-LOCI
 * @since 0.3.0
 */
public class MAEPretrainingDialog {

    private static final Logger logger = LoggerFactory.getLogger(MAEPretrainingDialog.class);

    private static final Set<String> IMAGE_EXTENSIONS = Set.of(
            ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".raw");

    private static final Map<String, String> CONFIG_DISPLAY_NAMES = Map.of(
            "muvit-small", "MuViT-Small (6 layers, 256 dim)",
            "muvit-base", "MuViT-Base (12 layers, 512 dim)",
            "muvit-large", "MuViT-Large (16 layers, 768 dim)"
    );

    public enum SourceMode { PROJECT_IMAGES, PRE_EXTRACTED_FOLDER }

    /**
     * Result record containing all configuration needed to launch MAE
     * pretraining.
     *
     * <p>Either {@code dataPath} is populated (folder mode, legacy) OR
     * {@code projectImages} is non-null (project mode, requires the
     * caller to extract tiles to a temp directory first).
     *
     * @param config            pretraining configuration map
     * @param dataPath          directory of unlabeled tiles (folder mode), or {@code null}
     * @param outputDir         directory to save encoder weights
     * @param sourceMode        which data source was selected
     * @param projectImages     images to extract from (project mode), or {@code null}
     * @param extractionTileSize tile size to extract at (project mode)
     * @param extractionDownsample downsample to apply during extraction (project mode)
     * @param maxTilesTotal     global cap on tiles after extraction (0 = unlimited)
     */
    public record MAEPretrainingConfig(
            Map<String, Object> config,
            Path dataPath,
            Path outputDir,
            SourceMode sourceMode,
            List<ProjectImageEntry<BufferedImage>> projectImages,
            int extractionTileSize,
            double extractionDownsample,
            int maxTilesTotal
    ) {}

    // Architecture / training controls
    private final ComboBox<String> modelConfigCombo;
    private final ComboBox<Integer> patchSizeCombo;
    private final TextField levelScalesField;
    private final Spinner<Integer> epochsSpinner;
    private final Spinner<Double> maskRatioSpinner;
    private final Spinner<Integer> batchSizeSpinner;
    private final Spinner<Double> learningRateSpinner;
    private final Spinner<Integer> warmupEpochsSpinner;

    // Source-mode controls
    private final ToggleGroup sourceModeGroup = new ToggleGroup();
    private final RadioButton projectModeRadio;
    private final RadioButton folderModeRadio;

    // Project mode controls
    private final ListView<MAEImageItem> projectImagesList = new ListView<>();
    private final Spinner<Integer> extractionTileSpinner;
    private final ComboBox<Double> extractionDownsampleCombo;
    private final Spinner<Integer> maxTilesSpinner;
    private final Label projectSummaryLabel = new Label();

    // Folder mode controls
    private final TextField dataPathField;
    private final Label datasetInfoLabel;

    // Output controls
    private final TextField outputDirField;

    private MAEPretrainingDialog() {
        // Model controls
        modelConfigCombo = new ComboBox<>(FXCollections.observableArrayList(MuViTHandler.MODEL_CONFIGS));
        modelConfigCombo.setValue("muvit-base");
        modelConfigCombo.setMaxWidth(Double.MAX_VALUE);
        modelConfigCombo.setCellFactory(lv -> createConfigCell());
        modelConfigCombo.setButtonCell(createConfigCell());
        TooltipHelper.install(modelConfigCombo,
                "MuViT encoder size for pretraining.\n" +
                "Must match the model you plan to use for supervised training.");

        patchSizeCombo = new ComboBox<>(FXCollections.observableArrayList(8, 16));
        patchSizeCombo.setValue(16);
        patchSizeCombo.setMaxWidth(100);
        TooltipHelper.install(patchSizeCombo,
                "Non-overlapping ViT patch size. 16 is the standard.");

        levelScalesField = new TextField("1,4");
        levelScalesField.setMaxWidth(150);
        TooltipHelper.install(levelScalesField,
                "Comma-separated scale factors for multi-resolution input.\n" +
                "'1,4' = full resolution + 4x downsampled context.");

        epochsSpinner = new Spinner<>(10, 2000, 100, 10);
        epochsSpinner.setEditable(true);
        epochsSpinner.setPrefWidth(100);

        maskRatioSpinner = new Spinner<>(0.50, 0.90, 0.75, 0.05);
        maskRatioSpinner.setEditable(true);
        maskRatioSpinner.setPrefWidth(100);

        batchSizeSpinner = new Spinner<>(1, 64, 8, 1);
        batchSizeSpinner.setEditable(true);
        batchSizeSpinner.setPrefWidth(100);

        var lrFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(1e-5, 1e-2, 1.5e-4, 1e-5);
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

        warmupEpochsSpinner = new Spinner<>(0, 50, 5, 1);
        warmupEpochsSpinner.setEditable(true);
        warmupEpochsSpinner.setPrefWidth(100);

        // Source mode radios
        projectModeRadio = new RadioButton("Project images (auto-extract tiles from Tissue)");
        projectModeRadio.setToggleGroup(sourceModeGroup);
        projectModeRadio.setSelected(true);
        TooltipHelper.install(projectModeRadio,
                "Extract MAE pretraining tiles just-in-time from images in\n" +
                "the currently open QuPath project. Only regions covered by\n" +
                "a 'Tissue'-classed annotation contribute tiles, so empty\n" +
                "slide background doesn't waste training compute.\n\n" +
                "Use the Detect Tissue button to add Tissue annotations to\n" +
                "images that don't have them yet.");

        folderModeRadio = new RadioButton("Pre-extracted folder (legacy)");
        folderModeRadio.setToggleGroup(sourceModeGroup);
        TooltipHelper.install(folderModeRadio,
                "Use an existing directory of image tiles you have already\n" +
                "extracted (PNG / TIFF / JPG / BMP / RAW). Subdirectories\n" +
                "are scanned recursively.");

        // Project-mode controls
        projectImagesList.setCellFactory(lv -> new MAEImageCell());
        projectImagesList.setPrefHeight(160);

        extractionTileSpinner = new Spinner<>(128, 1024, 512, 64);
        extractionTileSpinner.setEditable(true);
        extractionTileSpinner.setPrefWidth(100);
        TooltipHelper.install(extractionTileSpinner,
                "Size of the image tiles extracted from each slide for\n" +
                "MAE pretraining. Should be compatible with the patch size\n" +
                "and level scales you'll use downstream (a multiple of\n" +
                "patch_size x max(level_scales)).\n\n" +
                "512 is a common default.");

        extractionDownsampleCombo = new ComboBox<>(FXCollections.observableArrayList(1.0, 2.0, 4.0, 8.0));
        extractionDownsampleCombo.setValue(1.0);
        extractionDownsampleCombo.setPrefWidth(100);
        TooltipHelper.install(extractionDownsampleCombo,
                "Downsample factor applied when reading tiles from the\n" +
                "slide. 1 = full resolution; 2/4 = successively coarser.\n" +
                "Pick a value that matches the pixel scale you expect at\n" +
                "inference time.");

        maxTilesSpinner = new Spinner<>(100, 200000, 2000, 100);
        maxTilesSpinner.setEditable(true);
        maxTilesSpinner.setPrefWidth(100);
        TooltipHelper.install(maxTilesSpinner,
                "Maximum tiles to keep across all selected images after\n" +
                "extraction. If more tiles are produced, the surplus is\n" +
                "randomly discarded. Prevents a single large WSI from\n" +
                "overwhelming the pretraining pool.");

        projectSummaryLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        projectSummaryLabel.setWrapText(true);

        // Folder-mode controls
        dataPathField = new TextField();
        dataPathField.setPromptText("Directory of unlabeled image tiles...");
        dataPathField.setPrefWidth(250);
        TooltipHelper.install(dataPathField,
                "Path to a directory containing unlabeled image tiles for\n" +
                "pretraining. Supported formats: PNG, TIFF, JPEG, BMP, RAW.\n" +
                "Subdirectories are scanned recursively.");

        datasetInfoLabel = new Label();
        datasetInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        datasetInfoLabel.setWrapText(true);

        // Output controls
        outputDirField = new TextField();
        outputDirField.setPromptText("Output directory for encoder weights...");
        outputDirField.setPrefWidth(250);

        var qupath = QuPathGUI.getInstance();
        if (qupath != null && qupath.getProject() != null) {
            try {
                Path projectDir = qupath.getProject().getPath().getParent();
                String timestamp = java.time.LocalDateTime.now()
                        .format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
                outputDirField.setText(projectDir.resolve("mae_pretrained")
                        .resolve(modelConfigCombo.getValue() + "_" + timestamp).toString());
            } catch (Exception e) {
                logger.debug("Could not set default output dir from project: {}", e.getMessage());
            }
        }
        modelConfigCombo.valueProperty().addListener((obs, old, newVal) -> {
            String current = outputDirField.getText();
            if (current.contains("mae_pretrained" + File.separator) && old != null) {
                outputDirField.setText(current.replace(old + "_", newVal + "_"));
            }
        });

        // Populate project images list
        populateProjectImages();
    }

    public static Optional<MAEPretrainingConfig> showDialog() {
        MAEPretrainingDialog controller = new MAEPretrainingDialog();
        return controller.buildAndShow();
    }

    private Optional<MAEPretrainingConfig> buildAndShow() {
        Dialog<MAEPretrainingConfig> dialog = new Dialog<>();
        dialog.initOwner(QuPathGUI.getInstance().getStage());
        dialog.setTitle("MAE Pretrain MuViT Encoder");
        dialog.setResizable(true);

        ButtonType startType = new ButtonType("Start Pretraining", ButtonBar.ButtonData.OK_DONE);
        ButtonType cancelType = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);
        dialog.getDialogPane().getButtonTypes().addAll(startType, cancelType);

        VBox content = new VBox(10);
        content.setPadding(new Insets(10));

        Hyperlink tipsLink = new Hyperlink("Pretraining tips");
        tipsLink.setOnAction(e -> showPretrainingTipsDialog());
        tipsLink.setStyle("-fx-font-size: 11px;");
        content.getChildren().add(tipsLink);

        content.getChildren().add(buildModelSection());
        content.getChildren().add(buildTrainingSection());
        content.getChildren().add(buildDataSection(dialog));
        content.getChildren().add(buildOutputSection(dialog));

        ScrollPane scrollPane = new ScrollPane(content);
        scrollPane.setFitToWidth(true);
        scrollPane.setPrefViewportHeight(600);
        dialog.getDialogPane().setContent(scrollPane);
        dialog.getDialogPane().setPrefWidth(560);

        // Enable Start only when the selected mode has valid input
        Button startButton = (Button) dialog.getDialogPane().lookupButton(startType);
        startButton.disableProperty().bind(Bindings.createBooleanBinding(
                () -> !isStartValid(),
                sourceModeGroup.selectedToggleProperty(),
                dataPathField.textProperty(),
                projectImagesList.getItems()));

        // Refresh validity when per-item selection changes
        projectImagesList.getItems().addListener((javafx.collections.ListChangeListener<MAEImageItem>) c -> {
            while (c.next()) { /* triggers binding */ }
        });

        dialog.setResultConverter(button -> button == startType ? buildConfig() : null);
        return dialog.showAndWait();
    }

    private TitledPane buildModelSection() {
        GridPane grid = new GridPane();
        grid.setHgap(10); grid.setVgap(8); grid.setPadding(new Insets(10));
        grid.add(new Label("Model size:"), 0, 0);
        grid.add(modelConfigCombo, 1, 0);
        grid.add(new Label("Patch size:"), 0, 1);
        grid.add(patchSizeCombo, 1, 1);
        grid.add(new Label("Level scales:"), 0, 2);
        grid.add(levelScalesField, 1, 2);
        TitledPane pane = new TitledPane("Model Architecture", grid);
        pane.setExpanded(true); pane.setCollapsible(false);
        return pane;
    }

    private TitledPane buildTrainingSection() {
        GridPane grid = new GridPane();
        grid.setHgap(10); grid.setVgap(8); grid.setPadding(new Insets(10));
        grid.add(new Label("Epochs:"), 0, 0);         grid.add(epochsSpinner, 1, 0);
        grid.add(new Label("Mask ratio:"), 0, 1);     grid.add(maskRatioSpinner, 1, 1);
        grid.add(new Label("Batch size:"), 0, 2);     grid.add(batchSizeSpinner, 1, 2);
        grid.add(new Label("Learning rate:"), 0, 3);  grid.add(learningRateSpinner, 1, 3);
        grid.add(new Label("Warmup epochs:"), 0, 4);  grid.add(warmupEpochsSpinner, 1, 4);
        TitledPane pane = new TitledPane("Training Parameters", grid);
        pane.setExpanded(true); pane.setCollapsible(false);
        return pane;
    }

    private TitledPane buildDataSection(Dialog<?> dialog) {
        VBox root = new VBox(6);
        root.setPadding(new Insets(10));

        // Mode radios
        HBox modeRow = new HBox(12, projectModeRadio, folderModeRadio);
        root.getChildren().add(modeRow);

        // Project mode subpanel
        VBox projectPanel = new VBox(6);
        Button selectAllBtn = new Button("Select All");
        selectAllBtn.setOnAction(e -> {
            for (MAEImageItem item : projectImagesList.getItems()) item.selected = true;
            projectImagesList.refresh();
            updateProjectSummary();
        });
        Button selectNoneBtn = new Button("Select None");
        selectNoneBtn.setOnAction(e -> {
            for (MAEImageItem item : projectImagesList.getItems()) item.selected = false;
            projectImagesList.refresh();
            updateProjectSummary();
        });
        Button detectTissueBtn = new Button("Detect Tissue...");
        detectTissueBtn.setOnAction(e -> showDetectTissueDialog(dialog));
        TooltipHelper.install(detectTissueBtn,
                "Run a simple threshold-based tissue detector on the\n" +
                "selected images. Creates one or more 'Tissue'-classed\n" +
                "annotations per image; these are what MAE extraction\n" +
                "samples tiles from.");

        HBox buttonRow = new HBox(8, selectAllBtn, selectNoneBtn, detectTissueBtn);
        projectPanel.getChildren().addAll(projectImagesList, buttonRow, projectSummaryLabel);

        GridPane extractionGrid = new GridPane();
        extractionGrid.setHgap(10); extractionGrid.setVgap(8);
        extractionGrid.setPadding(new Insets(6, 0, 0, 0));
        extractionGrid.add(new Label("Tile size:"), 0, 0);
        extractionGrid.add(extractionTileSpinner, 1, 0);
        extractionGrid.add(new Label("Downsample:"), 0, 1);
        extractionGrid.add(extractionDownsampleCombo, 1, 1);
        extractionGrid.add(new Label("Max tiles (total):"), 0, 2);
        extractionGrid.add(maxTilesSpinner, 1, 2);
        projectPanel.getChildren().add(extractionGrid);

        projectPanel.visibleProperty().bind(projectModeRadio.selectedProperty());
        projectPanel.managedProperty().bind(projectModeRadio.selectedProperty());

        // Folder mode subpanel
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
        folderGrid.add(new Label("Image directory:"), 0, 0);
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
        grid.add(new Label("Output directory:"), 0, 0);
        grid.add(outputDirField, 1, 0);
        GridPane.setHgrow(outputDirField, Priority.ALWAYS);
        grid.add(browseBtn, 2, 0);
        TitledPane pane = new TitledPane("Output", grid);
        pane.setExpanded(true); pane.setCollapsible(false);
        return pane;
    }

    private MAEPretrainingConfig buildConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put("model_config", modelConfigCombo.getValue());
        config.put("patch_size", patchSizeCombo.getValue());
        config.put("level_scales", levelScalesField.getText().trim());
        config.put("epochs", epochsSpinner.getValue());
        config.put("mask_ratio", maskRatioSpinner.getValue());
        config.put("batch_size", batchSizeSpinner.getValue());
        config.put("learning_rate", learningRateSpinner.getValue());
        config.put("warmup_epochs", warmupEpochsSpinner.getValue());

        Path outputDir;
        if (outputDirField.getText().isBlank()) {
            Path fallback = Path.of(System.getProperty("user.home"), "mae_pretrained");
            outputDir = fallback;
        } else {
            outputDir = Path.of(outputDirField.getText().trim());
        }

        if (projectModeRadio.isSelected()) {
            List<ProjectImageEntry<BufferedImage>> selected = new ArrayList<>();
            for (MAEImageItem item : projectImagesList.getItems()) {
                if (item.selected) selected.add(item.entry);
            }
            return new MAEPretrainingConfig(
                    config,
                    null,
                    outputDir,
                    SourceMode.PROJECT_IMAGES,
                    selected,
                    extractionTileSpinner.getValue(),
                    extractionDownsampleCombo.getValue(),
                    maxTilesSpinner.getValue());
        } else {
            Path dataPath = Path.of(dataPathField.getText().trim());
            return new MAEPretrainingConfig(
                    config,
                    dataPath,
                    outputDir,
                    SourceMode.PRE_EXTRACTED_FOLDER,
                    null,
                    0,
                    1.0,
                    0);
        }
    }

    private boolean isStartValid() {
        if (projectModeRadio.isSelected()) {
            long nSelected = projectImagesList.getItems().stream().filter(i -> i.selected).count();
            if (nSelected == 0) return false;
            long nWithTissue = projectImagesList.getItems().stream()
                    .filter(i -> i.selected && i.tissueRegions > 0).count();
            return nWithTissue > 0;
        } else {
            String path = dataPathField.getText();
            if (path == null || path.isBlank()) return false;
            try { return Files.isDirectory(Path.of(path)); }
            catch (Exception e) { return false; }
        }
    }

    private void populateProjectImages() {
        projectImagesList.getItems().clear();
        QuPathGUI qupath = QuPathGUI.getInstance();
        Project<BufferedImage> project = qupath != null ? qupath.getProject() : null;
        if (project == null) {
            // No project open; project mode won't be usable. The radio stays
            // selected but validation will block Start.
            projectSummaryLabel.setText("No project open. Switch to 'Pre-extracted folder'"
                    + " or open a project containing WSIs.");
            return;
        }
        for (ProjectImageEntry<BufferedImage> entry : project.getImageList()) {
            try {
                MAEImageItem item = new MAEImageItem(entry);
                // Count existing Tissue annotations without opening the full image:
                // ProjectImageEntry keeps a hierarchy snapshot we can peek at.
                try {
                    var hier = entry.readHierarchy();
                    long n = hier.getAnnotationObjects().stream()
                            .filter(a -> a.getPathClass() != null
                                    && TissueDetectionUtility.TISSUE_CLASS_NAME
                                    .equals(a.getPathClass().getName()))
                            .count();
                    item.tissueRegions = (int) n;
                } catch (Exception ignored) {
                    item.tissueRegions = 0;
                }
                projectImagesList.getItems().add(item);
            } catch (Exception ex) {
                logger.debug("Skipping project entry: {}", ex.toString());
            }
        }
        updateProjectSummary();
    }

    private void updateProjectSummary() {
        long sel = projectImagesList.getItems().stream().filter(i -> i.selected).count();
        long selWithTissue = projectImagesList.getItems().stream()
                .filter(i -> i.selected && i.tissueRegions > 0).count();
        projectSummaryLabel.setText(String.format(
                "Selected: %d images (%d have Tissue annotations)",
                sel, selWithTissue));
    }

    private void showDetectTissueDialog(Dialog<?> parent) {
        List<MAEImageItem> targets = new ArrayList<>();
        for (MAEImageItem item : projectImagesList.getItems()) {
            if (item.selected) targets.add(item);
        }
        if (targets.isEmpty()) {
            Dialogs.showWarningNotification("Detect Tissue",
                    "Select at least one image first.");
            return;
        }

        // Parameter sub-dialog
        Dialog<TissueDetectionUtility.TissueDetectionParams> sub = new Dialog<>();
        sub.initOwner(parent.getDialogPane().getScene().getWindow());
        sub.setTitle("Detect Tissue");
        sub.setHeaderText("Create Tissue annotations on " + targets.size() + " image(s)");

        Spinner<Double> dsSpin = new Spinner<>(1.0, 64.0, 16.0, 1.0);
        dsSpin.setEditable(true);
        Spinner<Double> sigmaSpin = new Spinner<>(0.0, 50.0, 5.0, 1.0);
        sigmaSpin.setEditable(true);
        ComboBox<TissueDetectionUtility.ThresholdMethod> methodCombo = new ComboBox<>(
                FXCollections.observableArrayList(TissueDetectionUtility.ThresholdMethod.values()));
        methodCombo.setValue(TissueDetectionUtility.ThresholdMethod.OTSU);
        Spinner<Integer> fixedSpin = new Spinner<>(0, 255, 200, 5);
        fixedSpin.setEditable(true);
        Spinner<Double> minAreaSpin = new Spinner<>(0.0, 1e9, 100_000.0, 10_000.0);
        minAreaSpin.setEditable(true);
        CheckBox replaceCheck = new CheckBox("Replace existing Tissue annotations");
        replaceCheck.setSelected(true);

        GridPane g = new GridPane();
        g.setHgap(10); g.setVgap(8); g.setPadding(new Insets(10));
        g.add(new Label("Downsample:"), 0, 0); g.add(dsSpin, 1, 0);
        g.add(new Label("Sigma (um):"), 0, 1); g.add(sigmaSpin, 1, 1);
        g.add(new Label("Method:"), 0, 2); g.add(methodCombo, 1, 2);
        g.add(new Label("Fixed threshold:"), 0, 3); g.add(fixedSpin, 1, 3);
        g.add(new Label("Min area (um^2):"), 0, 4); g.add(minAreaSpin, 1, 4);
        g.add(replaceCheck, 0, 5, 2, 1);

        sub.getDialogPane().setContent(g);
        ButtonType okType = new ButtonType("Detect", ButtonBar.ButtonData.OK_DONE);
        sub.getDialogPane().getButtonTypes().addAll(okType, ButtonType.CANCEL);
        sub.setResultConverter(bt -> {
            if (bt != okType) return null;
            TissueDetectionUtility.TissueDetectionParams p =
                    new TissueDetectionUtility.TissueDetectionParams();
            p.downsample = dsSpin.getValue();
            p.sigmaMicrons = sigmaSpin.getValue();
            p.method = methodCombo.getValue();
            p.fixedThreshold = fixedSpin.getValue();
            p.minAreaMicronsSq = minAreaSpin.getValue();
            p.replaceExisting = replaceCheck.isSelected();
            return p;
        });

        Optional<TissueDetectionUtility.TissueDetectionParams> opt = sub.showAndWait();
        if (opt.isEmpty()) return;
        TissueDetectionUtility.TissueDetectionParams params = opt.get();

        // Progress dialog that blocks until detection completes
        Alert progressAlert = new Alert(Alert.AlertType.INFORMATION);
        progressAlert.initOwner(parent.getDialogPane().getScene().getWindow());
        progressAlert.setTitle("Detect Tissue");
        progressAlert.setHeaderText("Running tissue detection...");
        progressAlert.setContentText("Starting...");
        progressAlert.getButtonTypes().setAll(ButtonType.CANCEL);
        progressAlert.show();

        List<ProjectImageEntry<BufferedImage>> entries = new ArrayList<>();
        for (MAEImageItem i : targets) entries.add(i.entry);

        Thread worker = new Thread(() -> {
            try {
                int total = TissueDetectionUtility.detectTissue(entries, params,
                        msg -> Platform.runLater(() -> progressAlert.setContentText(msg)));
                Platform.runLater(() -> {
                    progressAlert.setContentText(
                            String.format("Detection complete: %d annotation(s) created.", total));
                    progressAlert.getButtonTypes().setAll(ButtonType.OK);
                    populateProjectImages();
                });
            } catch (Exception ex) {
                logger.error("Tissue detection failed", ex);
                Platform.runLater(() -> {
                    progressAlert.setContentText("Detection failed: " + ex.getMessage());
                    progressAlert.getButtonTypes().setAll(ButtonType.OK);
                });
            }
        }, "DLClassifier-TissueDetect");
        worker.setDaemon(true);
        worker.start();
    }

    /** Per-row state for the project images list. */
    static class MAEImageItem {
        final ProjectImageEntry<BufferedImage> entry;
        boolean selected = true;
        int tissueRegions = 0;

        MAEImageItem(ProjectImageEntry<BufferedImage> entry) {
            this.entry = entry;
        }
    }

    private class MAEImageCell extends ListCell<MAEImageItem> {
        private final CheckBox checkBox = new CheckBox();
        private final Label info = new Label();
        private final HBox root = new HBox(8, checkBox, info);

        MAEImageCell() {
            root.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
            checkBox.setOnAction(e -> {
                MAEImageItem item = getItem();
                if (item != null) {
                    item.selected = checkBox.isSelected();
                    updateProjectSummary();
                }
            });
        }

        @Override
        protected void updateItem(MAEImageItem item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setGraphic(null);
                return;
            }
            checkBox.setSelected(item.selected);
            checkBox.setText(item.entry.getImageName());
            if (item.tissueRegions > 0) {
                info.setText("(Tissue: " + item.tissueRegions + ")");
                info.setStyle("-fx-text-fill: #2a7a2a; -fx-font-size: 11px;");
            } else {
                info.setText("(no Tissue annotations)");
                info.setStyle("-fx-text-fill: #cc6600; -fx-font-size: 11px;");
            }
            setGraphic(root);
        }
    }

    /** Scans a dataset directory and updates the info label. */
    private void scanDatasetAndUpdateInfo(File dir) {
        Path dirPath = dir.toPath();
        Path imageDir = dirPath.resolve("train").resolve("images");
        if (!Files.isDirectory(imageDir)) {
            imageDir = dirPath.resolve("images");
            if (!Files.isDirectory(imageDir)) imageDir = dirPath;
        }
        int count = 0;
        try (var stream = Files.walk(imageDir)) {
            count = (int) stream
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase();
                        return IMAGE_EXTENSIONS.stream().anyMatch(name::endsWith);
                    })
                    .count();
        } catch (Exception e) {
            logger.debug("Error scanning dataset directory: {}", e.getMessage());
        }
        if (count == 0) {
            datasetInfoLabel.setText("No supported images found in directory.");
            datasetInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #c00;");
            return;
        }
        datasetInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        int suggestedEpochs;
        String sizeCategory;
        if (count < 50) { sizeCategory = "Very small"; suggestedEpochs = 500; }
        else if (count < 200) { sizeCategory = "Small"; suggestedEpochs = 300; }
        else if (count < 1000) { sizeCategory = "Medium"; suggestedEpochs = 100; }
        else { sizeCategory = "Large"; suggestedEpochs = 50; }
        datasetInfoLabel.setText(String.format(
                "%s dataset: %,d images found. Suggested epochs: %d",
                sizeCategory, count, suggestedEpochs));
        epochsSpinner.getValueFactory().setValue(suggestedEpochs);
    }

    private static void showPretrainingTipsDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("MAE Pretraining Tips");
        alert.setHeaderText("Self-Supervised Pretraining Guide");
        alert.setResizable(true);
        alert.getDialogPane().setPrefWidth(620);
        String tips = "WHAT IS MAE PRETRAINING?\n"
                + "The encoder is taught to reconstruct randomly masked image patches from the\n"
                + "visible ones. No labels are needed.\n\n"
                + "PROJECT IMAGES MODE (recommended for WSIs)\n"
                + "Run Detect Tissue to add a 'Tissue' annotation to each slide, then\n"
                + "MAE will extract tiles just-in-time from tissue regions only -- no empty\n"
                + "slide background, and no need to pre-tile anything externally.\n\n"
                + "PRE-EXTRACTED FOLDER MODE\n"
                + "Point at a directory of unlabeled image tiles (PNG/TIFF/JPG/BMP/RAW).\n"
                + "Useful if you've already prepared tiles outside QuPath.\n\n"
                + "MASK RATIO\n"
                + "0.75 is the MAE paper default. Higher = harder reconstruction, forces more\n"
                + "semantic learning but takes longer to converge. Lower = easier, useful for\n"
                + "small or low-diversity datasets.\n\n"
                + "AFTER PRETRAINING\n"
                + "The encoder .pt file is saved to the output directory. Load it in the\n"
                + "training dialog via 'Continue from model' with MuViT architecture selected.";
        TextArea textArea = new TextArea(tips);
        textArea.setEditable(false);
        textArea.setWrapText(true);
        textArea.setPrefRowCount(22);
        textArea.setStyle("-fx-font-size: 13px;");
        VBox.setVgrow(textArea, Priority.ALWAYS);
        alert.getDialogPane().setContent(textArea);
        alert.show();
    }

    private static ListCell<String> createConfigCell() {
        return new ListCell<>() {
            @Override protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? null : CONFIG_DISPLAY_NAMES.getOrDefault(item, item));
            }
        };
    }
}
