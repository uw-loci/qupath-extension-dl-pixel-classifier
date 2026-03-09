package qupath.ext.dlclassifier.ui;

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
import qupath.lib.gui.QuPathGUI;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Dialog for configuring MAE (Masked Autoencoder) self-supervised pretraining
 * of a MuViT encoder on unlabeled image tiles.
 * <p>
 * This is a standalone workflow accessible from the Utilities menu. The resulting
 * encoder weights can be loaded via "Continue from model" in the training dialog.
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

    /**
     * Result record containing all configuration needed to launch MAE pretraining.
     *
     * @param config    pretraining configuration map (model_config, patch_size, etc.)
     * @param dataPath  directory of unlabeled image tiles
     * @param outputDir directory to save encoder weights
     */
    public record MAEPretrainingConfig(
            Map<String, Object> config,
            Path dataPath,
            Path outputDir
    ) {}

    // Controls
    private final ComboBox<String> modelConfigCombo;
    private final ComboBox<Integer> patchSizeCombo;
    private final TextField levelScalesField;
    private final Spinner<Integer> epochsSpinner;
    private final Spinner<Double> maskRatioSpinner;
    private final Spinner<Integer> batchSizeSpinner;
    private final Spinner<Double> learningRateSpinner;
    private final Spinner<Integer> warmupEpochsSpinner;
    private final TextField dataPathField;
    private final Label datasetInfoLabel;
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
                "Must match the model you plan to use for supervised training.\n\n" +
                "Small (6 layers, 256 dim): Fastest, lowest VRAM. Good for\n" +
                "small datasets or quick experiments.\n" +
                "Base (12 layers, 512 dim): Recommended default. Good balance\n" +
                "of capacity and training speed.\n" +
                "Large (16 layers, 768 dim): Highest capacity but needs more\n" +
                "data and VRAM. Best for large, diverse datasets.");

        patchSizeCombo = new ComboBox<>(FXCollections.observableArrayList(8, 16));
        patchSizeCombo.setValue(16);
        patchSizeCombo.setMaxWidth(100);
        TooltipHelper.install(patchSizeCombo,
                "Size of non-overlapping patches the image is divided into\n" +
                "before feeding to the Vision Transformer.\n\n" +
                "16 (recommended): Standard patch size. Each 256x256 tile\n" +
                "produces 16x16 = 256 tokens. Good balance of detail and speed.\n" +
                "8: Finer-grained patches. Each 256x256 tile produces\n" +
                "32x32 = 1024 tokens. Captures more detail but uses ~4x more\n" +
                "VRAM and trains significantly slower.\n\n" +
                "Must match the patch size used for downstream training.");

        levelScalesField = new TextField("1,4");
        levelScalesField.setMaxWidth(150);
        TooltipHelper.install(levelScalesField,
                "Comma-separated scale factors for multi-resolution input.\n" +
                "MuViT processes the image at multiple resolutions simultaneously,\n" +
                "fusing features across scales.\n\n" +
                "Example: '1,4' = full resolution (1x) + 4x downsampled context.\n" +
                "Example: '1,2,4' = three-scale pyramid.\n\n" +
                "The default '1,4' captures both fine detail and broader context.\n" +
                "Must match the level scales used for downstream training.");

        // Training controls
        epochsSpinner = new Spinner<>(10, 2000, 100, 10);
        epochsSpinner.setEditable(true);
        epochsSpinner.setPrefWidth(100);
        TooltipHelper.install(epochsSpinner,
                "Number of complete passes through the dataset. Range: 10-2000.\n" +
                "More epochs give the encoder more chances to learn image structure.\n" +
                "No overfitting risk in MAE -- more is generally better.\n\n" +
                "Auto-suggested when you select a data directory:\n" +
                "  < 50 images:     500 epochs (many passes compensate for low diversity)\n" +
                "  50-200 images:   300 epochs\n" +
                "  200-1000 images: 100 epochs\n" +
                "  > 1000 images:    50 epochs (enough diversity per pass)\n\n" +
                "Longer training always helps with diminishing returns.\n" +
                "No early stopping -- runs for the full epoch count.");

        maskRatioSpinner = new Spinner<>(0.50, 0.90, 0.75, 0.05);
        maskRatioSpinner.setEditable(true);
        maskRatioSpinner.setPrefWidth(100);
        TooltipHelper.install(maskRatioSpinner,
                "Fraction of image patches randomly masked during pretraining.\n" +
                "The encoder learns by reconstructing the masked patches from\n" +
                "the visible ones. Range: 0.50-0.90.\n\n" +
                "0.75 (75%): Recommended default from the MAE paper. Only 25%\n" +
                "of patches are visible, forcing the encoder to learn strong\n" +
                "semantic and structural features to fill in the rest.\n\n" +
                "Lower (0.50-0.65): Easier reconstruction task. More visible\n" +
                "patches means the model can rely on local texture. Use for\n" +
                "very small datasets (<50 images) or low-diversity images where\n" +
                "the harder task would not converge well.\n\n" +
                "Higher (0.80-0.90): Harder task -- very few visible patches.\n" +
                "Forces the model to learn global, high-level features. Use for\n" +
                "large, diverse datasets with complex tissue structures.\n" +
                "May need more epochs to converge.");

        batchSizeSpinner = new Spinner<>(1, 64, 8, 1);
        batchSizeSpinner.setEditable(true);
        batchSizeSpinner.setPrefWidth(100);
        TooltipHelper.install(batchSizeSpinner,
                "Number of image tiles processed together per training step.\n" +
                "Range: 1-64.\n\n" +
                "Larger batches give more stable gradient estimates and\n" +
                "faster training (better GPU utilization), but use more VRAM.\n" +
                "Reduce if you get CUDA out-of-memory errors.\n\n" +
                "Typical: 4-8 for 8 GB VRAM, 8-16 for 12+ GB VRAM.\n" +
                "If your dataset has fewer images than the batch size,\n" +
                "it is automatically reduced to the dataset size.");

        var lrFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(1e-5, 1e-2, 1.5e-4, 1e-5);
        lrFactory.setConverter(new javafx.util.StringConverter<>() {
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
        learningRateSpinner = new Spinner<>(lrFactory);
        learningRateSpinner.setEditable(true);
        learningRateSpinner.setPrefWidth(120);
        TooltipHelper.install(learningRateSpinner,
                "Peak learning rate for the AdamW optimizer.\n" +
                "Range: 0.00001-0.01. Default: 0.00015 (1.5e-4).\n\n" +
                "1.5e-4: Standard MAE default from the original paper.\n" +
                "Works well for most dataset sizes and model configurations.\n\n" +
                "Lower (1e-5 to 5e-5): Use for very small datasets (<50 images)\n" +
                "or if loss oscillates during training.\n" +
                "Higher (3e-4 to 1e-3): May speed up convergence on large datasets\n" +
                "but risks instability. Not usually needed.\n\n" +
                "Combined with cosine annealing: LR ramps up during warmup,\n" +
                "then decays smoothly to near zero over the remaining epochs.");

        warmupEpochsSpinner = new Spinner<>(0, 50, 5, 1);
        warmupEpochsSpinner.setEditable(true);
        warmupEpochsSpinner.setPrefWidth(100);
        TooltipHelper.install(warmupEpochsSpinner,
                "Number of epochs to linearly ramp the learning rate from 0\n" +
                "to the target value. Range: 0-50. Default: 5.\n\n" +
                "Warmup prevents early training instability by starting with\n" +
                "very small gradient steps. Vision Transformers are particularly\n" +
                "sensitive to this -- training can diverge without warmup.\n\n" +
                "2-5 epochs: Good for small datasets or short training runs.\n" +
                "5-10 epochs: Good for longer runs (200+ epochs).\n" +
                "Set to 0 to disable warmup (not recommended for transformers).");

        // Data controls
        dataPathField = new TextField();
        dataPathField.setPromptText("Directory of unlabeled image tiles...");
        dataPathField.setPrefWidth(250);
        TooltipHelper.install(dataPathField,
                "Path to a directory containing unlabeled image tiles for pretraining.\n" +
                "Supported formats: PNG, TIFF, JPEG, BMP, RAW.\n\n" +
                "These can be any crops from your whole-slide images -- no annotations\n" +
                "or labels are needed. The encoder learns from the image structure itself.\n" +
                "Subdirectories are scanned recursively. If a 'train/images/' subdirectory\n" +
                "exists, it will be used automatically.\n\n" +
                "Tip: Use representative tiles from your target tissue type and staining.\n" +
                "Include tiles from different regions and intensity ranges for best results.");

        datasetInfoLabel = new Label();
        datasetInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        datasetInfoLabel.setWrapText(true);

        // Output controls
        outputDirField = new TextField();
        outputDirField.setPromptText("Output directory for encoder weights...");
        outputDirField.setPrefWidth(250);
        TooltipHelper.install(outputDirField,
                "Directory where the pretrained encoder weights will be saved.\n" +
                "Defaults to '{project}/mae_pretrained/' if a QuPath project is open.\n\n" +
                "The output is a .pt file containing the encoder state dict.\n" +
                "To use it: open Train Classifier, select MuViT architecture,\n" +
                "and load the weights via 'Continue from model'.\n" +
                "The directory is created automatically if it does not exist.");

        // Set default output dir from project if available (unique subdirectory per run)
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

        // Update subdirectory name when model config changes (only if still auto-generated)
        modelConfigCombo.valueProperty().addListener((obs, old, newVal) -> {
            String current = outputDirField.getText();
            if (current.contains("mae_pretrained" + File.separator) && old != null) {
                outputDirField.setText(current.replace(old + "_", newVal + "_"));
            }
        });
    }

    /**
     * Shows the MAE pretraining configuration dialog.
     *
     * @return the configuration if the user clicked Start, or empty if cancelled
     */
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

        // Build content
        VBox content = new VBox(10);
        content.setPadding(new Insets(10));

        // Pretraining tips link
        Hyperlink tipsLink = new Hyperlink("Pretraining tips");
        tipsLink.setOnAction(e -> showPretrainingTipsDialog());
        tipsLink.setStyle("-fx-font-size: 11px;");
        content.getChildren().add(tipsLink);

        // Model section
        content.getChildren().add(buildModelSection());

        // Training section
        content.getChildren().add(buildTrainingSection());

        // Data section
        content.getChildren().add(buildDataSection(dialog));

        // Output section
        content.getChildren().add(buildOutputSection(dialog));

        ScrollPane scrollPane = new ScrollPane(content);
        scrollPane.setFitToWidth(true);
        scrollPane.setPrefViewportHeight(500);
        dialog.getDialogPane().setContent(scrollPane);
        dialog.getDialogPane().setPrefWidth(520);

        // Disable Start button until data dir is valid
        var startButton = dialog.getDialogPane().lookupButton(startType);
        startButton.setDisable(true);
        dataPathField.textProperty().addListener((obs, old, val) ->
                startButton.setDisable(!isValidDataDir(val)));

        // Result converter
        dialog.setResultConverter(button -> {
            if (button == startType) {
                return buildConfig();
            }
            return null;
        });

        return dialog.showAndWait();
    }

    private TitledPane buildModelSection() {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);
        grid.setPadding(new Insets(10));

        grid.add(new Label("Model size:"), 0, 0);
        grid.add(modelConfigCombo, 1, 0);
        grid.add(new Label("Patch size:"), 0, 1);
        grid.add(patchSizeCombo, 1, 1);
        grid.add(new Label("Level scales:"), 0, 2);
        grid.add(levelScalesField, 1, 2);

        TitledPane pane = new TitledPane("Model Architecture", grid);
        pane.setExpanded(true);
        pane.setCollapsible(false);
        return pane;
    }

    private TitledPane buildTrainingSection() {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);
        grid.setPadding(new Insets(10));

        grid.add(new Label("Epochs:"), 0, 0);
        grid.add(epochsSpinner, 1, 0);
        grid.add(new Label("Mask ratio:"), 0, 1);
        grid.add(maskRatioSpinner, 1, 1);
        grid.add(new Label("Batch size:"), 0, 2);
        grid.add(batchSizeSpinner, 1, 2);
        grid.add(new Label("Learning rate:"), 0, 3);
        grid.add(learningRateSpinner, 1, 3);
        grid.add(new Label("Warmup epochs:"), 0, 4);
        grid.add(warmupEpochsSpinner, 1, 4);

        TitledPane pane = new TitledPane("Training Parameters", grid);
        pane.setExpanded(true);
        pane.setCollapsible(false);
        return pane;
    }

    private TitledPane buildDataSection(Dialog<?> dialog) {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);
        grid.setPadding(new Insets(10));

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

        grid.add(new Label("Image directory:"), 0, 0);
        grid.add(dataPathField, 1, 0);
        GridPane.setHgrow(dataPathField, Priority.ALWAYS);
        grid.add(browseBtn, 2, 0);
        grid.add(datasetInfoLabel, 0, 1, 3, 1);

        TitledPane pane = new TitledPane("Data", grid);
        pane.setExpanded(true);
        pane.setCollapsible(false);
        return pane;
    }

    private TitledPane buildOutputSection(Dialog<?> dialog) {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);
        grid.setPadding(new Insets(10));

        Button browseBtn = new Button("Browse...");
        browseBtn.setOnAction(e -> {
            DirectoryChooser dc = new DirectoryChooser();
            dc.setTitle("Select Output Directory");
            if (!outputDirField.getText().isBlank()) {
                File current = new File(outputDirField.getText());
                if (current.isDirectory()) {
                    dc.setInitialDirectory(current);
                } else if (current.getParentFile() != null && current.getParentFile().isDirectory()) {
                    dc.setInitialDirectory(current.getParentFile());
                }
            }
            File dir = dc.showDialog(dialog.getDialogPane().getScene().getWindow());
            if (dir != null) {
                outputDirField.setText(dir.getAbsolutePath());
            }
        });

        grid.add(new Label("Output directory:"), 0, 0);
        grid.add(outputDirField, 1, 0);
        GridPane.setHgrow(outputDirField, Priority.ALWAYS);
        grid.add(browseBtn, 2, 0);

        TitledPane pane = new TitledPane("Output", grid);
        pane.setExpanded(true);
        pane.setCollapsible(false);
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

        Path dataPath = Path.of(dataPathField.getText().trim());
        Path outputDir;
        if (outputDirField.getText().isBlank()) {
            outputDir = dataPath.getParent().resolve("mae_pretrained");
        } else {
            outputDir = Path.of(outputDirField.getText().trim());
        }

        return new MAEPretrainingConfig(config, dataPath, outputDir);
    }

    private boolean isValidDataDir(String path) {
        if (path == null || path.isBlank()) return false;
        try {
            return Files.isDirectory(Path.of(path));
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Scans a dataset directory and updates the info label with image count
     * and recommended settings.
     */
    private void scanDatasetAndUpdateInfo(File dir) {
        Path dirPath = dir.toPath();

        // Check for train/images subdirectory (common dataset layout)
        Path imageDir = dirPath.resolve("train").resolve("images");
        if (!Files.isDirectory(imageDir)) {
            imageDir = dirPath.resolve("images");
            if (!Files.isDirectory(imageDir)) {
                imageDir = dirPath;
            }
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

        String sizeCategory;
        int suggestedEpochs;
        if (count < 50) {
            sizeCategory = "Very small";
            suggestedEpochs = 500;
        } else if (count < 200) {
            sizeCategory = "Small";
            suggestedEpochs = 300;
        } else if (count < 1000) {
            sizeCategory = "Medium";
            suggestedEpochs = 100;
        } else {
            sizeCategory = "Large";
            suggestedEpochs = 50;
        }

        datasetInfoLabel.setText(String.format(
                "%s dataset: %,d images found. Suggested epochs: %d",
                sizeCategory, count, suggestedEpochs));

        epochsSpinner.getValueFactory().setValue(suggestedEpochs);
    }

    /**
     * Shows a dialog with tips for different pretraining scenarios.
     */
    private static void showPretrainingTipsDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("MAE Pretraining Tips");
        alert.setHeaderText("Self-Supervised Pretraining Guide");
        alert.setResizable(true);
        alert.getDialogPane().setPrefWidth(600);

        String tipContent = "WHAT IS MAE PRETRAINING?\n"
                + "Masked Autoencoder (MAE) pretraining teaches the encoder to understand "
                + "image structure by masking random patches and learning to reconstruct them. "
                + "No labels are needed -- the model learns from the images themselves. "
                + "After pretraining, the encoder is fine-tuned with your labeled training data.\n\n"
                + "WHEN TO USE IT\n"
                + "- You have LIMITED LABELED DATA but plenty of unlabeled images\n"
                + "- Your images are from a specialized domain (not typical photos)\n"
                + "- Domain-specific pretraining consistently outperforms ImageNet-pretrained "
                + "models for specialized microscopy and pathology data\n\n"
                + "IMAGE DIRECTORY\n"
                + "Point to a directory of image tiles (PNG, TIFF, JPG, BMP, or RAW). "
                + "These can be unlabeled crops from your whole-slide images. "
                + "Subdirectories are scanned recursively. If the directory contains a "
                + "train/images/ subdirectory, that will be used automatically.\n\n"
                + "SMALL DATASETS (< 100 images)\n"
                + "- Use 300-500 pretraining epochs (more epochs = more passes over data)\n"
                + "- MAE is well-suited for small datasets because random masking creates "
                + "diverse reconstruction tasks even from limited images\n"
                + "- Consider lowering mask ratio to 0.60-0.65 to make reconstruction "
                + "easier when diversity is low\n\n"
                + "MEDIUM DATASETS (100 - 1000 images)\n"
                + "- Use 100-200 pretraining epochs\n"
                + "- Default mask ratio of 0.75 works well\n"
                + "- This is the sweet spot for pretraining benefit\n\n"
                + "LARGE DATASETS (> 1000 images)\n"
                + "- Use 50-100 pretraining epochs (sufficient data diversity)\n"
                + "- Default mask ratio of 0.75 is optimal\n"
                + "- Longer training always helps but with diminishing returns\n\n"
                + "MULTICHANNEL IMAGES (fluorescence, multispectral)\n"
                + "- MAE pretraining works natively with any number of channels\n"
                + "- The encoder learns per-channel and cross-channel relationships\n"
                + "- Channel count is detected automatically from your images\n"
                + "- Ensure all images in the directory have the same number of channels\n\n"
                + "MASK RATIO\n"
                + "- 0.75 (75%) is optimal for most cases per the MAE literature\n"
                + "- Higher ratios force the model to learn global/semantic features\n"
                + "- Lower ratios (0.50-0.65) are easier and may help with very small "
                + "or low-diversity datasets\n\n"
                + "AFTER PRETRAINING\n"
                + "The encoder weights (.pt file) are saved to the output directory. "
                + "To use them, open the training dialog, select MuViT as the model, "
                + "and choose \"Continue from model\" to load the pretrained encoder. "
                + "The classifier will then fine-tune from these learned representations.\n\n"
                + "REFERENCES\n"
                + "- He et al., \"Masked Autoencoders Are Scalable Vision Learners\" (CVPR 2022)\n"
                + "- Xie et al., \"SimMIM: A Simple Framework for Masked Image Modeling\" (CVPR 2022)\n"
                + "- Filiot et al., \"Self-Supervised Vision Transformers Learn Visual Concepts "
                + "in Histopathology\" (2022)";

        TextArea textArea = new TextArea(tipContent);
        textArea.setEditable(false);
        textArea.setWrapText(true);
        textArea.setPrefRowCount(25);
        textArea.setStyle("-fx-font-size: 13px;");
        VBox.setVgrow(textArea, Priority.ALWAYS);

        alert.getDialogPane().setContent(textArea);
        alert.show();
    }

    private static ListCell<String> createConfigCell() {
        return new ListCell<>() {
            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? null : CONFIG_DISPLAY_NAMES.getOrDefault(item, item));
            }
        };
    }
}
