package qupath.ext.dlclassifier.classifier.handlers;

import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.io.File;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Handler for MuViT (Multi-Resolution Vision Transformer) pixel classifiers.
 * <p>
 * MuViT processes image tiles at multiple resolution levels simultaneously
 * using cross-resolution self-attention with world-coordinate positional
 * embeddings. This enables learned multi-scale feature fusion where low-res
 * context tokens directly attend to high-res detail tokens.
 * <p>
 * Phase 1 integration: trains from scratch (no MAE pretraining).
 * The model uses a lightweight segmentation head on top of MuViT encoder features.
 *
 * @author UW-LOCI
 * @since 0.3.0
 * @see <a href="https://arxiv.org/abs/2602.24222">MuViT: Multi-Resolution Vision Transformers</a>
 */
public class MuViTHandler implements ClassifierHandler {

    /**
     * MuViT does not use traditional backbone encoders.
     * These entries represent model size configurations.
     */
    public static final List<String> MODEL_CONFIGS = List.of(
            "muvit-small",
            "muvit-base",
            "muvit-large"
    );

    /** Human-readable display names for model configurations. */
    private static final Map<String, String> CONFIG_DISPLAY_NAMES = Map.of(
            "muvit-small", "MuViT-Small (6 layers, 256 dim)",
            "muvit-base", "MuViT-Base (12 layers, 512 dim)",
            "muvit-large", "MuViT-Large (16 layers, 768 dim)"
    );

    /**
     * Returns a human-readable display name for a model configuration.
     *
     * @param config the config identifier string
     * @return display name, or the config string itself if not found
     */
    public static String getConfigDisplayName(String config) {
        return CONFIG_DISPLAY_NAMES.getOrDefault(config, config);
    }

    /** Supported tile sizes (no strict divisibility constraint for ViT). */
    public static final List<Integer> TILE_SIZES = List.of(
            128, 256, 512
    );

    @Override
    public String getType() {
        return "muvit";
    }

    @Override
    public String getDisplayName() {
        return "MuViT (Transformer)";
    }

    @Override
    public String getDescription() {
        return "Multi-resolution Vision Transformer. Processes tiles at multiple " +
                "physical scales using cross-resolution self-attention with world-coordinate " +
                "positional embeddings. Best for tasks requiring both fine detail and broad " +
                "tissue context. Requires more GPU memory than CNN architectures.";
    }

    @Override
    public TrainingConfig getDefaultTrainingConfig() {
        return TrainingConfig.builder()
                .modelType("muvit")
                .backbone("muvit-base")
                .epochs(50)
                .batchSize(4)
                .learningRate(0.0001)
                .weightDecay(0.01)
                .tileSize(256)
                .overlap(32)
                .validationSplit(0.2)
                .augmentation(true)
                .usePretrainedWeights(false)
                .freezeEncoderLayers(0)
                .build();
    }

    @Override
    public InferenceConfig getDefaultInferenceConfig() {
        return InferenceConfig.builder()
                .tileSize(256)
                .overlap(32)
                .blendMode(InferenceConfig.BlendMode.LINEAR)
                .outputType(InferenceConfig.OutputType.MEASUREMENTS)
                .minObjectSizeMicrons(10.0)
                .holeFillingMicrons(5.0)
                .boundarySmoothing(2.0)
                .maxTilesInMemory(30)
                .useGPU(true)
                .build();
    }

    @Override
    public boolean supportsVariableChannels() {
        return true;
    }

    @Override
    public int getMinChannels() {
        return 1;
    }

    @Override
    public int getMaxChannels() {
        return 16;
    }

    @Override
    public List<Integer> getSupportedTileSizes() {
        return TILE_SIZES;
    }

    @Override
    public String getBackboneDisplayName(String config) {
        return CONFIG_DISPLAY_NAMES.getOrDefault(config, config);
    }

    @Override
    public Optional<String> validateChannelConfig(ChannelConfiguration channelConfig) {
        if (channelConfig == null) {
            return Optional.of("Channel configuration is required");
        }

        int numChannels = channelConfig.getNumChannels();
        if (numChannels < getMinChannels()) {
            return Optional.of(String.format(
                    "MuViT requires at least %d channel(s), but %d selected",
                    getMinChannels(), numChannels));
        }
        if (numChannels > getMaxChannels()) {
            return Optional.of(String.format(
                    "MuViT supports at most %d channels, but %d selected",
                    getMaxChannels(), numChannels));
        }

        return Optional.empty();
    }

    @Override
    public Map<String, Object> getArchitectureParams(TrainingConfig config) {
        Map<String, Object> params = new HashMap<>();
        params.put("architecture", "muvit");
        // MuViT doesn't use backbone encoders; pass model configs as "backbones"
        // so the existing UI dropdown populates correctly
        params.put("available_backbones", MODEL_CONFIGS);

        // MuViT-specific parameters
        String modelConfig = "muvit-base";
        int patchSize = 16;
        String levelScales = "1,4";
        int numHeads = 8;
        String ropeMode = "per_layer";

        if (config != null) {
            modelConfig = config.getBackbone() != null ? config.getBackbone() : modelConfig;
            params.put("use_pretrained", false);
            params.put("freeze_encoder_layers", 0);
        }

        params.put("backbone", modelConfig);
        params.put("model_config", modelConfig);
        params.put("patch_size", patchSize);
        params.put("level_scales", levelScales);
        params.put("num_heads", numHeads);
        params.put("rope_mode", ropeMode);

        return params;
    }

    @Override
    public Optional<TrainingUI> createTrainingUI() {
        return Optional.of(new MuViTTrainingUI());
    }

    @Override
    public ClassifierMetadata buildMetadata(TrainingConfig config,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames) {
        String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String id = String.format("muvit_%s_%s", config.getBackbone(), timestamp);

        ClassifierMetadata.Builder builder = ClassifierMetadata.builder()
                .id(id)
                .name(String.format("MuViT %s Classifier", config.getBackbone()))
                .description(String.format("MuViT %s, %d channels, %d classes",
                        config.getBackbone(),
                        channelConfig.getNumChannels(),
                        classNames.size()))
                .modelType("muvit")
                .backbone(config.getBackbone())
                .inputSize(config.getTileSize(), config.getTileSize())
                .inputChannels(channelConfig.getNumChannels())
                .contextScale(config.getContextScale())
                .expectedChannelNames(channelConfig.getChannelNames())
                .normalizationStrategy(channelConfig.getNormalizationStrategy())
                .bitDepthTrained(channelConfig.getBitDepth())
                .trainingEpochs(config.getEpochs());

        for (int i = 0; i < classNames.size(); i++) {
            String color = getDefaultColor(i);
            builder.addClass(i, classNames.get(i), color);
        }

        return builder.build();
    }

    private String getDefaultColor(int index) {
        String[] colors = {
                "#808080", "#FF0000", "#00FF00", "#0000FF",
                "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500"
        };
        return colors[index % colors.length];
    }

    /**
     * UI component for MuViT-specific training parameters.
     */
    private static class MuViTTrainingUI implements TrainingUI {

        private final VBox root;
        private final ComboBox<String> modelConfigCombo;
        private final ComboBox<Integer> patchSizeCombo;
        private final TextField levelScalesField;
        private final ComboBox<String> ropeModeCombo;

        // MAE pretraining controls
        private final CheckBox pretrainCheck;
        private final TextField pretrainDataPathField;
        private final Label datasetInfoLabel;
        private final Spinner<Integer> pretrainEpochsSpinner;
        private final Spinner<Double> maskRatioSpinner;
        private final Spinner<Integer> warmupEpochsSpinner;

        public MuViTTrainingUI() {
            root = new VBox(5);

            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(10);
            grid.setPadding(new Insets(10));

            int row = 0;

            // Model size configuration
            Label configLabel = new Label("Model size:");
            modelConfigCombo = new ComboBox<>(FXCollections.observableArrayList(MODEL_CONFIGS));
            modelConfigCombo.setValue("muvit-base");
            modelConfigCombo.setMaxWidth(Double.MAX_VALUE);
            modelConfigCombo.setCellFactory(lv -> new ListCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty);
                    setText(empty || item == null ? null : getConfigDisplayName(item));
                }
            });
            modelConfigCombo.setButtonCell(new ListCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty);
                    setText(empty || item == null ? null : getConfigDisplayName(item));
                }
            });
            grid.add(configLabel, 0, row);
            grid.add(modelConfigCombo, 1, row);
            row++;

            // Patch size
            Label patchLabel = new Label("Patch size:");
            patchSizeCombo = new ComboBox<>(FXCollections.observableArrayList(8, 16));
            patchSizeCombo.setValue(16);
            patchSizeCombo.setMaxWidth(100);
            Tooltip patchTooltip = new Tooltip(
                    "Size of patches for the transformer. Smaller = more tokens = more VRAM.\n" +
                    "16 is recommended for most cases.");
            patchSizeCombo.setTooltip(patchTooltip);
            grid.add(patchLabel, 0, row);
            grid.add(patchSizeCombo, 1, row);
            row++;

            // Level scales
            Label levelsLabel = new Label("Level scales:");
            levelScalesField = new TextField("1,4");
            levelScalesField.setMaxWidth(150);
            Tooltip levelsTooltip = new Tooltip(
                    "Comma-separated scale factors for multi-resolution levels.\n" +
                    "Example: '1,4' = detail at 1x and context at 4x physical area.\n" +
                    "'1,2,8' = three levels at 1x, 2x, and 8x.");
            levelScalesField.setTooltip(levelsTooltip);
            grid.add(levelsLabel, 0, row);
            grid.add(levelScalesField, 1, row);
            row++;

            // RoPE mode
            Label ropeLabel = new Label("Position encoding:");
            ropeModeCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "per_layer", "shared", "fixed", "none"));
            ropeModeCombo.setValue("per_layer");
            ropeModeCombo.setMaxWidth(150);
            Tooltip ropeTooltip = new Tooltip(
                    "Rotary positional embedding mode for cross-resolution attention.\n" +
                    "per_layer (default): Separate learned embeddings per layer.\n" +
                    "shared: Shared learned embeddings.\n" +
                    "fixed: Fixed sinusoidal embeddings.\n" +
                    "none: No positional encoding.");
            ropeModeCombo.setTooltip(ropeTooltip);
            grid.add(ropeLabel, 0, row);
            grid.add(ropeModeCombo, 1, row);

            root.getChildren().add(grid);

            // --- MAE Pretraining section ---
            Separator sep = new Separator();
            sep.setPadding(new Insets(5, 0, 5, 0));
            root.getChildren().add(sep);

            pretrainCheck = new CheckBox("MAE pretrain encoder (self-supervised)");
            pretrainCheck.setTooltip(new Tooltip(
                    "Pretrain the MuViT encoder on unlabeled images using\n" +
                    "masked autoencoder reconstruction before fine-tuning.\n" +
                    "This improves accuracy when labeled data is limited.\n" +
                    "Requires a directory of image tiles (no labels needed)."));

            Hyperlink tipsLink = new Hyperlink("Pretraining tips");
            tipsLink.setOnAction(e -> showPretrainingTipsDialog());
            tipsLink.setStyle("-fx-font-size: 11px;");

            HBox pretrainHeader = new HBox(10, pretrainCheck, tipsLink);
            root.getChildren().add(pretrainHeader);

            GridPane pretrainGrid = new GridPane();
            pretrainGrid.setHgap(10);
            pretrainGrid.setVgap(8);
            pretrainGrid.setPadding(new Insets(5, 0, 0, 20));

            // Data path
            Label dataLabel = new Label("Image directory:");
            pretrainDataPathField = new TextField();
            pretrainDataPathField.setPromptText("Directory of unlabeled image tiles...");
            pretrainDataPathField.setPrefWidth(200);
            datasetInfoLabel = new Label();
            datasetInfoLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
            datasetInfoLabel.setWrapText(true);
            Button browseBtn = new Button("Browse...");
            browseBtn.setOnAction(e -> {
                DirectoryChooser dc = new DirectoryChooser();
                dc.setTitle("Select image directory for MAE pretraining");
                File dir = dc.showDialog(root.getScene() != null ? root.getScene().getWindow() : null);
                if (dir != null) {
                    pretrainDataPathField.setText(dir.getAbsolutePath());
                    scanDatasetAndUpdateInfo(dir);
                }
            });
            pretrainGrid.add(dataLabel, 0, 0);
            pretrainGrid.add(pretrainDataPathField, 1, 0);
            pretrainGrid.add(browseBtn, 2, 0);
            pretrainGrid.add(datasetInfoLabel, 0, 1, 3, 1);

            // Pretraining epochs
            Label epochsLabel = new Label("Pretrain epochs:");
            pretrainEpochsSpinner = new Spinner<>(10, 2000, 100, 10);
            pretrainEpochsSpinner.setEditable(true);
            pretrainEpochsSpinner.setPrefWidth(100);
            pretrainEpochsSpinner.setTooltip(new Tooltip(
                    "Number of MAE pretraining epochs.\n" +
                    "More epochs always helps -- no saturation observed up to 1600.\n" +
                    "Small datasets (<100 images): use 200-500 epochs.\n" +
                    "Medium datasets (100-1000): 100-200 epochs.\n" +
                    "Large datasets (>1000): 50-100 epochs."));
            pretrainGrid.add(epochsLabel, 0, 2);
            pretrainGrid.add(pretrainEpochsSpinner, 1, 2);

            // Mask ratio
            Label maskLabel = new Label("Mask ratio:");
            maskRatioSpinner = new Spinner<>(0.5, 0.9, 0.75, 0.05);
            maskRatioSpinner.setEditable(true);
            maskRatioSpinner.setPrefWidth(100);
            maskRatioSpinner.setTooltip(new Tooltip(
                    "Fraction of patches to mask during pretraining.\n" +
                    "0.75 (75%) is the standard default from MAE literature.\n" +
                    "For very small datasets (<50 images), try 0.60-0.70.\n" +
                    "Higher ratios force the model to learn stronger representations\n" +
                    "but need sufficient data diversity."));
            pretrainGrid.add(maskLabel, 0, 3);
            pretrainGrid.add(maskRatioSpinner, 1, 3);

            // Warmup epochs
            Label warmupLabel = new Label("Warmup epochs:");
            warmupEpochsSpinner = new Spinner<>(0, 50, 5, 1);
            warmupEpochsSpinner.setEditable(true);
            warmupEpochsSpinner.setPrefWidth(100);
            warmupEpochsSpinner.setTooltip(new Tooltip(
                    "Number of epochs for learning rate warmup.\n" +
                    "Transformers typically benefit from 2-10 warmup epochs."));
            pretrainGrid.add(warmupLabel, 0, 4);
            pretrainGrid.add(warmupEpochsSpinner, 1, 4);

            root.getChildren().add(pretrainGrid);

            // Bind visibility to checkbox
            pretrainGrid.visibleProperty().bind(pretrainCheck.selectedProperty());
            pretrainGrid.managedProperty().bind(pretrainCheck.selectedProperty());
        }

        @Override
        public Node getNode() {
            return root;
        }

        @Override
        public Map<String, Object> getParameters() {
            Map<String, Object> params = new HashMap<>();
            params.put("model_config", modelConfigCombo.getValue());
            params.put("patch_size", patchSizeCombo.getValue());
            params.put("level_scales", levelScalesField.getText().trim());
            params.put("rope_mode", ropeModeCombo.getValue());

            // MAE pretraining parameters
            params.put("mae_pretrain_enabled", pretrainCheck.isSelected());
            if (pretrainCheck.isSelected()) {
                params.put("mae_data_path", pretrainDataPathField.getText().trim());
                params.put("mae_epochs", pretrainEpochsSpinner.getValue());
                params.put("mae_mask_ratio", maskRatioSpinner.getValue());
                params.put("mae_warmup_epochs", warmupEpochsSpinner.getValue());
            }

            return params;
        }

        @Override
        public Optional<String> validate() {
            if (modelConfigCombo.getValue() == null) {
                return Optional.of("Please select a model configuration");
            }
            String scales = levelScalesField.getText().trim();
            if (scales.isEmpty()) {
                return Optional.of("Level scales cannot be empty");
            }
            try {
                String[] parts = scales.split(",");
                if (parts.length < 2) {
                    return Optional.of("At least 2 resolution levels required (e.g., '1,4')");
                }
                for (String part : parts) {
                    double val = Double.parseDouble(part.trim());
                    if (val <= 0) {
                        return Optional.of("Scale factors must be positive");
                    }
                }
            } catch (NumberFormatException e) {
                return Optional.of("Invalid level scales format. Use comma-separated numbers (e.g., '1,4')");
            }

            // Validate pretraining config if enabled
            if (pretrainCheck.isSelected()) {
                String dataPath = pretrainDataPathField.getText().trim();
                if (dataPath.isEmpty()) {
                    return Optional.of("Please select an image directory for MAE pretraining");
                }
                File dataDir = new File(dataPath);
                if (!dataDir.isDirectory()) {
                    return Optional.of("MAE pretraining data path is not a valid directory: " + dataPath);
                }
            }

            return Optional.empty();
        }

        /**
         * Scans a dataset directory and updates the info label with image count
         * and recommended settings.
         */
        private void scanDatasetAndUpdateInfo(File dir) {
            java.util.Set<String> extensions = java.util.Set.of(
                    ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".raw");
            Path dirPath = dir.toPath();

            // Check for train/images subdirectory
            Path imageDir = dirPath.resolve("train").resolve("images");
            if (!java.nio.file.Files.isDirectory(imageDir)) {
                imageDir = dirPath.resolve("images");
                if (!java.nio.file.Files.isDirectory(imageDir)) {
                    imageDir = dirPath;
                }
            }

            int count = 0;
            try (var stream = java.nio.file.Files.walk(imageDir)) {
                count = (int) stream
                        .filter(java.nio.file.Files::isRegularFile)
                        .filter(p -> {
                            String name = p.getFileName().toString().toLowerCase();
                            return extensions.stream().anyMatch(name::endsWith);
                        })
                        .count();
            } catch (Exception ignored) {
                // Fall through with count = 0
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

            // Auto-set recommended epochs
            pretrainEpochsSpinner.getValueFactory().setValue(suggestedEpochs);
        }

        /**
         * Shows a dialog with tips for different pretraining scenarios.
         */
        private void showPretrainingTipsDialog() {
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("MAE Pretraining Tips");
            alert.setHeaderText("Self-Supervised Pretraining Guide");
            alert.setResizable(true);
            alert.getDialogPane().setPrefWidth(600);

            String content = """
                    WHAT IS MAE PRETRAINING?
                    Masked Autoencoder (MAE) pretraining teaches the encoder to understand \
                    image structure by masking random patches and learning to reconstruct them. \
                    No labels are needed -- the model learns from the images themselves. \
                    After pretraining, the encoder is fine-tuned with your labeled training data.

                    WHEN TO USE IT
                    - You have LIMITED LABELED DATA but plenty of unlabeled images
                    - Your images are from a specialized domain (not typical photos)
                    - Domain-specific pretraining consistently outperforms ImageNet-pretrained \
                    models for specialized microscopy and pathology data

                    IMAGE DIRECTORY
                    Point to a directory of image tiles (PNG, TIFF, JPG, or RAW). \
                    These can be unlabeled crops from your whole-slide images. \
                    Subdirectories are scanned recursively. If the directory contains a \
                    train/images/ subdirectory, that will be used automatically.

                    SMALL DATASETS (< 100 images)
                    - Use 300-500 pretraining epochs (more epochs = more passes over data)
                    - MAE is well-suited for small datasets because random masking creates \
                    diverse reconstruction tasks even from limited images
                    - Consider lowering mask ratio to 0.60-0.65 to make reconstruction \
                    easier when diversity is low
                    - The model will see each image hundreds of times with different masks \
                    and augmentations, learning robust features

                    MEDIUM DATASETS (100 - 1000 images)
                    - Use 100-200 pretraining epochs
                    - Default mask ratio of 0.75 works well
                    - This is the sweet spot for pretraining benefit

                    LARGE DATASETS (> 1000 images)
                    - Use 50-100 pretraining epochs (sufficient data diversity)
                    - Default mask ratio of 0.75 is optimal
                    - Longer training always helps but with diminishing returns

                    MULTICHANNEL IMAGES (fluorescence, multispectral)
                    - MAE pretraining works natively with any number of channels
                    - The encoder learns per-channel and cross-channel relationships
                    - Channel count is detected automatically from your images
                    - Ensure all images in the directory have the same number of channels

                    MASK RATIO
                    - 0.75 (75%) is optimal for most cases per the MAE literature
                    - Higher ratios force the model to learn global/semantic features
                    - Lower ratios (0.50-0.65) are easier and may help with very small \
                    or low-diversity datasets
                    - The model must infer missing patches from surrounding context, \
                    learning useful representations in the process

                    REFERENCES
                    - He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
                    - Xie et al., "SimMIM: A Simple Framework for Masked Image Modeling" (CVPR 2022)
                    - Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)
                    - Filiot et al., "Self-Supervised Vision Transformers Learn Visual Concepts \
                    in Histopathology" (2022)""";

            TextArea textArea = new TextArea(content);
            textArea.setEditable(false);
            textArea.setWrapText(true);
            textArea.setPrefRowCount(25);
            textArea.setStyle("-fx-font-size: 13px;");
            VBox.setVgrow(textArea, Priority.ALWAYS);

            alert.getDialogPane().setContent(textArea);
            alert.show();
        }
    }
}
