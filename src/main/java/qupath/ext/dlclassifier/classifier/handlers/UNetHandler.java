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

/**
 * Handler for UNet architecture pixel classifiers.
 * <p>
 * UNet is a fully convolutional encoder-decoder network originally designed for
 * biomedical image segmentation. It uses skip connections between encoder and
 * decoder to preserve spatial information.
 *
 * <h3>Supported Backbones</h3>
 * <ul>
 *   <li>resnet18, resnet34, resnet50 (recommended)</li>
 *   <li>efficientnet-b0 through efficientnet-b4</li>
 *   <li>mobilenet_v2 (lightweight)</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class UNetHandler implements ClassifierHandler {

    /** Available backbone architectures for UNet encoder */
    public static final List<String> BACKBONES = List.of(
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "mobilenet_v2",
            // Histology-pretrained ResNet-50 encoders (weights from HuggingFace)
            "resnet50_lunit-swav",
            "resnet50_lunit-bt",
            "resnet50_kather100k",
            "resnet50_tcga-brca"
    );

    /** Human-readable display names for backbone encoders. */
    private static final Map<String, String> BACKBONE_DISPLAY_NAMES = Map.ofEntries(
            Map.entry("resnet18", "ResNet-18"),
            Map.entry("resnet34", "ResNet-34"),
            Map.entry("resnet50", "ResNet-50"),
            Map.entry("efficientnet-b0", "EfficientNet-B0"),
            Map.entry("efficientnet-b1", "EfficientNet-B1"),
            Map.entry("efficientnet-b2", "EfficientNet-B2"),
            Map.entry("mobilenet_v2", "MobileNet-V2"),
            Map.entry("resnet50_lunit-swav", "ResNet-50 Lunit SwAV (Histology)"),
            Map.entry("resnet50_lunit-bt", "ResNet-50 Lunit Barlow Twins (Histology)"),
            Map.entry("resnet50_kather100k", "ResNet-50 Kather100K (Histology)"),
            Map.entry("resnet50_tcga-brca", "ResNet-50 TCGA-BRCA (Histology)")
    );

    /**
     * Returns a human-readable display name for a backbone encoder.
     *
     * @param backbone the backbone identifier string
     * @return display name, or the backbone string itself if not found
     */
    public static String getStaticBackboneDisplayName(String backbone) {
        return BACKBONE_DISPLAY_NAMES.getOrDefault(backbone, backbone);
    }

    @Override
    public String getBackboneDisplayName(String backbone) {
        return BACKBONE_DISPLAY_NAMES.getOrDefault(backbone, backbone);
    }

    /** Supported tile sizes (must be divisible by 32 for UNet) */
    public static final List<Integer> TILE_SIZES = List.of(
            128, 256, 384, 512, 768, 1024
    );

    @Override
    public String getType() {
        return "unet";
    }

    @Override
    public String getDisplayName() {
        return "UNet";
    }

    @Override
    public String getDescription() {
        return "Encoder-decoder architecture with skip connections. " +
                "Excellent for semantic segmentation with strong boundary preservation. " +
                "Supports various pretrained backbones (ResNet, EfficientNet).";
    }

    @Override
    public TrainingConfig getDefaultTrainingConfig() {
        return TrainingConfig.builder()
                .modelType("unet")
                .backbone("resnet34")
                .epochs(50)
                .batchSize(8)
                .learningRate(0.001)
                .weightDecay(1e-4)
                .tileSize(512)
                .overlap(64)
                .validationSplit(0.2)
                .augmentation(true)
                .usePretrainedWeights(true)
                .freezeEncoderLayers(0)
                .build();
    }

    @Override
    public InferenceConfig getDefaultInferenceConfig() {
        return InferenceConfig.builder()
                .tileSize(512)
                .overlap(64)
                .blendMode(InferenceConfig.BlendMode.LINEAR)
                .outputType(InferenceConfig.OutputType.MEASUREMENTS)
                .minObjectSizeMicrons(10.0)
                .holeFillingMicrons(5.0)
                .boundarySmoothing(2.0)
                .maxTilesInMemory(50)
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
        return 64; // Practical upper limit
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
                    "UNet requires at least %d channel(s), but %d selected",
                    getMinChannels(), numChannels));
        }
        if (numChannels > getMaxChannels()) {
            return Optional.of(String.format(
                    "UNet supports at most %d channels, but %d selected",
                    getMaxChannels(), numChannels));
        }

        return Optional.empty();
    }

    @Override
    public Map<String, Object> getArchitectureParams(TrainingConfig config) {
        Map<String, Object> params = new HashMap<>();
        params.put("architecture", "unet");
        params.put("available_backbones", BACKBONES);
        params.put("encoder_depth", 5);
        params.put("decoder_channels", List.of(256, 128, 64, 32, 16));

        if (config != null) {
            params.put("backbone", config.getBackbone());
            params.put("use_pretrained", config.isUsePretrainedWeights());
            params.put("freeze_encoder_layers", config.getFreezeEncoderLayers());
        } else {
            params.put("backbone", "resnet34");
            params.put("use_pretrained", true);
            params.put("freeze_encoder_layers", 0);
        }
        return params;
    }

    // No handler-specific training UI needed for UNet.
    // Encoder selection and layer freezing are handled by TrainingDialog's
    // generic backbone combo and LayerFreezePanel respectively.

    @Override
    public ClassifierMetadata buildMetadata(TrainingConfig config,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames) {
        // Generate a unique ID
        String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String id = String.format("unet_%s_%s", config.getBackbone(), timestamp);

        ClassifierMetadata.Builder builder = ClassifierMetadata.builder()
                .id(id)
                .name(String.format("UNet %s Classifier", config.getBackbone()))
                .description(String.format("UNet with %s backbone, %d channels, %d classes",
                        config.getBackbone(),
                        channelConfig.getNumChannels(),
                        classNames.size()))
                .modelType("unet")
                .backbone(config.getBackbone())
                .inputSize(config.getTileSize(), config.getTileSize())
                .inputChannels(channelConfig.getNumChannels())
                .contextScale(config.getContextScale())
                .expectedChannelNames(channelConfig.getChannelNames())
                .normalizationStrategy(channelConfig.getNormalizationStrategy())
                .bitDepthTrained(channelConfig.getBitDepth())
                .trainingEpochs(config.getEpochs());

        // Add classes
        for (int i = 0; i < classNames.size(); i++) {
            String color = getDefaultColor(i);
            builder.addClass(i, classNames.get(i), color);
        }

        return builder.build();
    }

    /**
     * Returns a default color for a class index.
     */
    private String getDefaultColor(int index) {
        String[] colors = {
                "#808080", // Gray (background)
                "#FF0000", // Red
                "#00FF00", // Green
                "#0000FF", // Blue
                "#FFFF00", // Yellow
                "#FF00FF", // Magenta
                "#00FFFF", // Cyan
                "#FFA500"  // Orange
        };
        return colors[index % colors.length];
    }

}
