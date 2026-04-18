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
 * Handler for the Fast Pretrained architecture -- SMP U-Net with a small
 * ImageNet-pretrained mobile encoder and a scaled-down decoder.
 * <p>
 * Intended for RGB H&amp;E tasks with small annotation counts (under ~1000
 * tiles) where ImageNet priors give 10-15 points of Dice score vs training
 * from scratch. Trains in minutes on a modest GPU.
 *
 * <h3>Available encoders</h3>
 * <ul>
 *   <li><b>timm-tf_efficientnet_lite0</b> (default): ~4.2M params. No SE
 *       blocks or hard-swish -- clean compile/export story, optimized for
 *       mobile CPUs. Best balance of ImageNet accuracy and speed.</li>
 *   <li><b>timm-mobilenetv3_small_100</b>: ~2.0M params. Smaller and
 *       faster, lower ImageNet top-1 (67.7% vs 75.1%). Pick when VRAM
 *       or inference latency is tight.</li>
 * </ul>
 *
 * <h3>Decoder sizing</h3>
 * <p>
 * SMP's default decoder channels [256, 128, 64, 32, 16] dwarf these small
 * encoders. The handler uses [128, 64, 32, 16, 8] across all supported
 * encoders -- keeps decoder params under 1.5x encoder params per the
 * agent-A2 recommendation.
 *
 * <h3>Multi-channel inputs</h3>
 * <p>
 * SMP's first-conv adaptation works out of the box for 1-7 channels:
 * sums RGB weights for 1-channel, rescales for 2-channel, tile-initializes
 * extra channels for 4+. For fluorescence (usually 4+ channels), the
 * empirical tile-init is good enough for most cases; an explicit
 * mean-of-RGB init is left as a future optimization.
 *
 * @author UW-LOCI
 * @since 0.6.1
 */
public class FastPretrainedHandler implements ClassifierHandler {

    /** Small pretrained encoders. Order matters: index 0 is the default. */
    public static final List<String> BACKBONES = List.of(
            "timm-tf_efficientnet_lite0",
            "timm-mobilenetv3_small_100"
    );

    private static final Map<String, String> BACKBONE_DISPLAY_NAMES = Map.of(
            "timm-tf_efficientnet_lite0",
                    "EfficientNet-Lite0 (ImageNet, ~4.2M params, recommended)",
            "timm-mobilenetv3_small_100",
                    "MobileNetV3-Small (ImageNet, ~2.0M params, fastest)"
    );

    /** Most SMP encoders need tile sizes divisible by 32. */
    public static final List<Integer> TILE_SIZES = List.of(
            128, 192, 256, 384, 512
    );

    @Override
    public String getType() {
        return "fast-pretrained";
    }

    @Override
    public String getDisplayName() {
        return "Fast Pretrained (small RGB)";
    }

    @Override
    public String getDescription() {
        return "Small U-Net with ImageNet-pretrained mobile encoder "
                + "(EfficientNet-Lite0 or MobileNetV3-Small) and scaled-down decoder. "
                + "Best for RGB H&E segmentation with limited training data "
                + "(<1000 tiles) where ImageNet priors provide a 10-15 Dice bump "
                + "over training from scratch. See FAST_PRETRAINED.md.";
    }

    @Override
    public String getBackboneDisplayName(String backbone) {
        return BACKBONE_DISPLAY_NAMES.getOrDefault(backbone, backbone);
    }

    @Override
    public TrainingConfig getDefaultTrainingConfig() {
        return TrainingConfig.builder()
                .modelType("fast-pretrained")
                .backbone("timm-tf_efficientnet_lite0")
                .epochs(30)
                .batchSize(16)
                .learningRate(1e-3)
                .weightDecay(1e-4)
                .tileSize(256)
                .overlap(32)
                .validationSplit(0.2)
                .augmentation(true)
                .usePretrainedWeights(true)
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
                    "Fast Pretrained requires at least %d channel(s), but %d selected",
                    getMinChannels(), numChannels));
        }
        if (numChannels > getMaxChannels()) {
            return Optional.of(String.format(
                    "Fast Pretrained supports at most %d channels, but %d selected. "
                            + "For higher channel counts, use UNet or Tiny UNet.",
                    getMaxChannels(), numChannels));
        }

        return Optional.empty();
    }

    @Override
    public Map<String, Object> getArchitectureParams(TrainingConfig config) {
        Map<String, Object> params = new HashMap<>();
        params.put("architecture", "fast-pretrained");
        params.put("available_backbones", BACKBONES);

        String backbone = config != null && config.getBackbone() != null
                ? config.getBackbone()
                : "timm-tf_efficientnet_lite0";
        params.put("backbone", backbone);
        params.put("decoder_channels", List.of(128, 64, 32, 16, 8));

        // Discriminative LR: 1/5 instead of UNet's 1/10, per agent A2 -- mobile
        // encoders have less over-specialized features and benefit from more
        // aggressive decoder-side adaptation.
        params.put("discriminative_lr_ratio", 0.2);

        if (config != null) {
            params.put("use_pretrained", config.isUsePretrainedWeights());
            params.put("freeze_encoder_layers", config.getFreezeEncoderLayers());
        } else {
            params.put("use_pretrained", true);
            params.put("freeze_encoder_layers", 0);
        }
        return params;
    }

    @Override
    public Set<WeightInitStrategy> getSupportedWeightInitStrategies() {
        return Set.of(WeightInitStrategy.SCRATCH,
                WeightInitStrategy.BACKBONE_PRETRAINED,
                WeightInitStrategy.CONTINUE_TRAINING);
    }

    @Override
    public WeightInitStrategy getDefaultWeightInitStrategy() {
        return WeightInitStrategy.BACKBONE_PRETRAINED;
    }

    @Override
    public ClassifierMetadata buildMetadata(TrainingConfig config,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames) {
        String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String id = String.format("fastpre_%s_%s",
                config.getBackbone().replace("timm-", "").replace("-", "_"),
                timestamp);

        ClassifierMetadata.Builder builder = ClassifierMetadata.builder()
                .id(id)
                .name(String.format("Fast Pretrained %s Classifier", config.getBackbone()))
                .description(String.format(
                        "Fast Pretrained U-Net with %s backbone, %d channels, %d classes",
                        config.getBackbone(),
                        channelConfig.getNumChannels(),
                        classNames.size()))
                .modelType("fast-pretrained")
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
