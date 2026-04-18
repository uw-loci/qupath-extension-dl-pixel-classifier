package qupath.ext.dlclassifier.classifier.handlers;

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
 * Handler for the Tiny UNet architecture -- a lightweight, fast-to-train
 * depthwise-separable U-Net aimed at simple 2-5 class microscopy segmentation.
 * <p>
 * Trains in seconds to minutes on small datasets (500-5000 tiles).
 * Recommended when the task is simple enough that the ~24M-parameter
 * ResNet34 UNet is overkill and training speed matters more than squeezing
 * out the last few points of mIoU.
 *
 * <h3>Size presets (selected via the backbone combo)</h3>
 * <ul>
 *   <li><b>tiny-16x4</b>: base=16, depth=4 (~138k params) - default, balanced</li>
 *   <li><b>nano-8x3</b>: base=8, depth=3 (~10k params) - fastest, 2-class tasks</li>
 *   <li><b>compact-16x3</b>: base=16, depth=3 (~36k params) - small model, shallow</li>
 *   <li><b>small-24x4</b>: base=24, depth=4 (~305k params) - extra capacity</li>
 * </ul>
 *
 * <h3>Normalization</h3>
 * <p>
 * Defaults to BatchRenorm, matching the rest of the extension. Advanced users
 * can override by setting the "tiny_unet_norm" handler parameter to "gn"
 * (GroupNorm, recommended when using torch.compile) or "bn" (plain BatchNorm,
 * not recommended -- kept for compatibility).
 *
 * @author UW-LOCI
 * @since 0.6.1
 */
public class TinyUNetHandler implements ClassifierHandler {

    /** Size presets (re-using the "backbone" combo for architecture sizing). */
    public static final List<String> BACKBONES = List.of(
            "tiny-16x4",
            "nano-8x3",
            "compact-16x3",
            "small-24x4"
    );

    private static final Map<String, String> BACKBONE_DISPLAY_NAMES = Map.of(
            "tiny-16x4",    "Tiny (default, ~138k params)",
            "nano-8x3",     "Nano (~10k params, 2-class tasks)",
            "compact-16x3", "Compact (~36k params, shallow)",
            "small-24x4",   "Small (~305k params, extra capacity)"
    );

    /** Decoded (base, depth) per preset. */
    private static final Map<String, int[]> BACKBONE_DIMS = Map.of(
            "tiny-16x4",    new int[]{16, 4},
            "nano-8x3",     new int[]{8,  3},
            "compact-16x3", new int[]{16, 3},
            "small-24x4",   new int[]{24, 4}
    );

    /**
     * Tile sizes must be divisible by 2**depth. Max depth across presets is 4,
     * so all listed sizes are divisible by 16.
     */
    public static final List<Integer> TILE_SIZES = List.of(
            128, 192, 256, 384, 512
    );

    @Override
    public String getType() {
        return "tiny-unet";
    }

    @Override
    public String getDisplayName() {
        return "Tiny UNet (fast)";
    }

    @Override
    public String getDescription() {
        return "Lightweight depthwise-separable U-Net. Trains in seconds to minutes "
                + "for simple 2-5 class microscopy tasks. No pretrained weights "
                + "required - good default for fluorescence or multi-channel data "
                + "where ImageNet priors do not transfer. See TINY_MODEL.md for "
                + "size presets and normalization options.";
    }

    @Override
    public String getBackboneDisplayName(String backbone) {
        return BACKBONE_DISPLAY_NAMES.getOrDefault(backbone, backbone);
    }

    @Override
    public TrainingConfig getDefaultTrainingConfig() {
        return TrainingConfig.builder()
                .modelType("tiny-unet")
                .backbone("tiny-16x4")
                .epochs(30)
                .batchSize(16)
                .learningRate(0.003)
                .weightDecay(1e-4)
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
                .maxTilesInMemory(100)
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
        return 7;
    }

    @Override
    public List<Integer> getSupportedTileSizes() {
        return TILE_SIZES;
    }

    @Override
    public Optional<String> validateChannelConfig(ChannelConfiguration channelConfig) {
        if (channelConfig == null) {
            return Optional.of("Channel configuration is required");
        }

        int numChannels = channelConfig.getNumChannels();
        if (numChannels < getMinChannels()) {
            return Optional.of(String.format(
                    "Tiny UNet requires at least %d channel(s), but %d selected",
                    getMinChannels(), numChannels));
        }
        if (numChannels > getMaxChannels()) {
            return Optional.of(String.format(
                    "Tiny UNet supports at most %d channels, but %d selected. "
                            + "For higher channel counts, use UNet.",
                    getMaxChannels(), numChannels));
        }

        return Optional.empty();
    }

    @Override
    public Map<String, Object> getArchitectureParams(TrainingConfig config) {
        Map<String, Object> params = new HashMap<>();
        params.put("architecture", "tiny-unet");
        params.put("available_backbones", BACKBONES);

        String backbone = config != null && config.getBackbone() != null
                ? config.getBackbone()
                : "tiny-16x4";
        int[] dims = BACKBONE_DIMS.getOrDefault(backbone, BACKBONE_DIMS.get("tiny-16x4"));
        params.put("backbone", backbone);
        params.put("base", dims[0]);
        params.put("depth", dims[1]);

        // Normalization: default BRN. Advanced override via handler parameters.
        String norm = "brn";
        if (config != null && config.getHandlerParameters() != null) {
            Object override = config.getHandlerParameters().get("tiny_unet_norm");
            if (override instanceof String s && !s.isEmpty()) {
                norm = s;
            }
        }
        params.put("norm", norm);

        // Tiny UNet does not use pretrained weights.
        params.put("use_pretrained", false);
        params.put("freeze_encoder_layers", 0);
        return params;
    }

    @Override
    public Set<WeightInitStrategy> getSupportedWeightInitStrategies() {
        return Set.of(WeightInitStrategy.SCRATCH, WeightInitStrategy.CONTINUE_TRAINING);
    }

    @Override
    public WeightInitStrategy getDefaultWeightInitStrategy() {
        return WeightInitStrategy.SCRATCH;
    }

    @Override
    public ClassifierMetadata buildMetadata(TrainingConfig config,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames) {
        String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String id = String.format("tinyunet_%s_%s", config.getBackbone(), timestamp);

        ClassifierMetadata.Builder builder = ClassifierMetadata.builder()
                .id(id)
                .name(String.format("Tiny UNet %s Classifier", config.getBackbone()))
                .description(String.format(
                        "Tiny UNet (%s), %d channels, %d classes",
                        config.getBackbone(),
                        channelConfig.getNumChannels(),
                        classNames.size()))
                .modelType("tiny-unet")
                .backbone(config.getBackbone())
                .inputSize(config.getTileSize(), config.getTileSize())
                .inputChannels(channelConfig.getNumChannels())
                .contextScale(config.getContextScale())
                .expectedChannelNames(channelConfig.getChannelNames())
                .normalizationStrategy(channelConfig.getNormalizationStrategy())
                .bitDepthTrained(channelConfig.getBitDepth())
                .trainingEpochs(config.getEpochs());

        for (int i = 0; i < classNames.size(); i++) {
            builder.addClass(i, classNames.get(i), getDefaultColor(i));
        }

        return builder.build();
    }

    private String getDefaultColor(int index) {
        String[] colors = {
                "#808080",
                "#FF0000",
                "#00FF00",
                "#0000FF",
                "#FFFF00",
                "#FF00FF",
                "#00FFFF",
                "#FFA500"
        };
        return colors[index % colors.length];
    }
}
