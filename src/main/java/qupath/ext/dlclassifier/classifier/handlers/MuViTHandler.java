package qupath.ext.dlclassifier.classifier.handlers;

import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

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
                .overlap(64)
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
                .overlap(64)
                .blendMode(InferenceConfig.BlendMode.GAUSSIAN)
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
    public Set<WeightInitStrategy> getSupportedWeightInitStrategies() {
        return Set.of(WeightInitStrategy.SCRATCH,
                      WeightInitStrategy.MAE_ENCODER,
                      WeightInitStrategy.CONTINUE_TRAINING);
    }

    @Override
    public WeightInitStrategy getDefaultWeightInitStrategy() {
        return WeightInitStrategy.SCRATCH;
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
            params.put("freeze_encoder_layers", 0);
            // Do NOT set use_pretrained here. ApposeClassifierBackend.startTraining
            // sets use_pretrained from TrainingConfig before merging handler params;
            // forcing false here would silently strip the MAE/Continue-Training
            // flag whenever the merge policy changed. Leave it unset and let the
            // top-level architecture map carry the authoritative value.
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
     * <p>
     * Only contains MuViT architecture settings (model size, patch size,
     * level scales, position encoding). MAE pretraining is a separate
     * workflow accessible from the Utilities menu.
     */
    private static class MuViTTrainingUI implements TrainingUI {

        private final VBox root;
        private final ComboBox<String> modelConfigCombo;
        private final ComboBox<Integer> patchSizeCombo;
        private final TextField levelScalesField;
        private final ComboBox<String> ropeModeCombo;

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
                    "How the model encodes spatial position of patches.\n" +
                    "This controls cross-resolution attention quality.\n\n" +
                    "per_layer (recommended): Each transformer layer has its own\n" +
                    "position encoding. Best accuracy -- the model can learn\n" +
                    "different spatial relationships at different depths.\n" +
                    "Use this unless you have a specific reason not to.\n\n" +
                    "shared: All layers share one position encoding.\n" +
                    "Slightly fewer parameters. Use if training is unstable\n" +
                    "or you have very limited data (<50 labeled tiles).\n\n" +
                    "fixed: Non-learnable sinusoidal position encoding.\n" +
                    "Zero trainable parameters for position. Use only if\n" +
                    "you need maximum reproducibility or suspect position\n" +
                    "encoding is overfitting (very unlikely in practice).\n\n" +
                    "none: No position information at all. The model cannot\n" +
                    "distinguish where patches came from spatially. Only for\n" +
                    "ablation experiments -- not recommended for real use.\n\n" +
                    "Must match the setting used during MAE pretraining\n" +
                    "(locked automatically when an MAE encoder is loaded).");
            ropeModeCombo.setTooltip(ropeTooltip);
            grid.add(ropeLabel, 0, row);
            grid.add(ropeModeCombo, 1, row);

            root.getChildren().add(grid);
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

            return Optional.empty();
        }

        @Override
        public void applyParameters(Map<String, Object> params) {
            if (params.containsKey("model_config"))
                modelConfigCombo.setValue((String) params.get("model_config"));
            if (params.containsKey("patch_size"))
                patchSizeCombo.setValue(((Number) params.get("patch_size")).intValue());
            if (params.containsKey("level_scales"))
                levelScalesField.setText(String.valueOf(params.get("level_scales")));
            if (params.containsKey("rope_mode"))
                ropeModeCombo.setValue((String) params.get("rope_mode"));
        }

        @Override
        public void setLocked(boolean locked) {
            modelConfigCombo.setDisable(locked);
            patchSizeCombo.setDisable(locked);
            levelScalesField.setEditable(!locked);
            levelScalesField.setDisable(locked);
            ropeModeCombo.setDisable(locked);
        }
    }
}

