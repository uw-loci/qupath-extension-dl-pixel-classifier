package qupath.ext.dlclassifier.model;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Objects;

/**
 * Metadata describing a trained deep learning classifier.
 * <p>
 * This class contains all information needed to identify, load, and validate
 * a classifier for inference, including architecture details, class definitions,
 * channel configuration, and training provenance.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ClassifierMetadata {

    /** Metadata schema version for forward compatibility */
    public static final String VERSION = "1.0";

    // Identification
    private final String id;
    private final String name;
    private final String description;
    private final LocalDateTime createdAt;

    // Architecture
    private final String modelType;
    private final String backbone;
    private final int inputWidth;
    private final int inputHeight;
    private final int inputChannels;
    private final double downsample;
    private final int contextScale;

    // Channel configuration
    private final List<String> expectedChannelNames;
    private final ChannelConfiguration.NormalizationStrategy normalizationStrategy;
    private final int bitDepthTrained;

    // Classes
    private final List<ClassInfo> classes;

    // Training info
    private final String trainingImageName;
    private final int trainingEpochs;
    private final double finalLoss;
    private final double finalAccuracy;

    // Full training hyperparameters (may be null for older models)
    private final Map<String, Object> trainingSettings;

    // Normalization stats computed from training dataset (may be null for older models)
    private final List<Map<String, Double>> normalizationStats;

    // Resolution contract (may be NaN/0 for older models trained before
    // these fields were saved). Used at inference to detect cross-batch
    // pixel-size mismatch with the source image and warn the user.
    private final double trainingPixelSizeMicrons;
    private final int trainingTileSizePx;

    private ClassifierMetadata(Builder builder) {
        this.id = builder.id;
        this.name = builder.name;
        this.description = builder.description;
        this.createdAt = builder.createdAt;
        this.modelType = builder.modelType;
        this.backbone = builder.backbone;
        this.inputWidth = builder.inputWidth;
        this.inputHeight = builder.inputHeight;
        this.inputChannels = builder.inputChannels;
        this.downsample = builder.downsample;
        this.contextScale = builder.contextScale;
        this.expectedChannelNames = Collections.unmodifiableList(new ArrayList<>(builder.expectedChannelNames));
        this.normalizationStrategy = builder.normalizationStrategy;
        this.bitDepthTrained = builder.bitDepthTrained;
        this.classes = Collections.unmodifiableList(new ArrayList<>(builder.classes));
        this.trainingImageName = builder.trainingImageName;
        this.trainingEpochs = builder.trainingEpochs;
        this.finalLoss = builder.finalLoss;
        this.finalAccuracy = builder.finalAccuracy;
        this.trainingSettings = builder.trainingSettings != null
                ? Collections.unmodifiableMap(new LinkedHashMap<>(builder.trainingSettings))
                : null;
        this.normalizationStats = builder.normalizationStats != null
                ? Collections.unmodifiableList(new ArrayList<>(builder.normalizationStats))
                : null;
        this.trainingPixelSizeMicrons = builder.trainingPixelSizeMicrons;
        this.trainingTileSizePx = builder.trainingTileSizePx;
    }

    /**
     * Physical pixel size of the training data in microns per pixel,
     * or NaN when the model was trained on uncalibrated images / saved
     * before this contract was added.
     */
    public double getTrainingPixelSizeMicrons() {
        return trainingPixelSizeMicrons;
    }

    /**
     * Training tile size in pixels (== input_size), or 0 when not
     * recorded in metadata.
     */
    public int getTrainingTileSizePx() {
        return trainingTileSizePx;
    }

    // Getters

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public String getModelType() {
        return modelType;
    }

    public String getBackbone() {
        return backbone;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    /**
     * Gets the number of image channels this model requires (base count).
     * <p>
     * This is the number of channels selected during training, before any
     * context scale doubling. Use {@link #getEffectiveInputChannels()} to get
     * the actual model input channel count (which is doubled when
     * {@link #getContextScale()} > 1).
     *
     * @return base input channel count
     */
    public int getInputChannels() {
        return inputChannels;
    }

    /**
     * Gets the effective number of input channels the model expects.
     * <p>
     * When {@code contextScale > 1}, the model receives both a detail tile
     * and a context tile concatenated along the channel axis, so the effective
     * input channels are {@code inputChannels * 2}. When {@code contextScale == 1},
     * this is the same as {@link #getInputChannels()}.
     *
     * @return effective model input channels (doubled when context is enabled)
     */
    public int getEffectiveInputChannels() {
        return contextScale > 1 ? inputChannels * 2 : inputChannels;
    }

    /**
     * Gets the downsample factor used during training.
     * <p>
     * Inference must use the same downsample to match training resolution.
     *
     * @return downsample factor (1.0 = full resolution)
     */
    public double getDownsample() {
        return downsample;
    }

    /**
     * Gets the context scale factor for multi-scale inference.
     * <p>
     * When greater than 1, inference requires both a detail tile and a context tile.
     * The context tile covers contextScale times the area in each dimension,
     * downsampled to the same pixel size. Both tiles are concatenated along the
     * channel axis before being sent to the model.
     *
     * @return context scale factor (1 = single-scale, no context tile needed)
     */
    public int getContextScale() {
        return contextScale;
    }

    public List<String> getExpectedChannelNames() {
        return expectedChannelNames;
    }

    public ChannelConfiguration.NormalizationStrategy getNormalizationStrategy() {
        return normalizationStrategy;
    }

    public int getBitDepthTrained() {
        return bitDepthTrained;
    }

    public List<ClassInfo> getClasses() {
        return classes;
    }

    public int getNumClasses() {
        return classes.size();
    }

    public String getTrainingImageName() {
        return trainingImageName;
    }

    public int getTrainingEpochs() {
        return trainingEpochs;
    }

    public double getFinalLoss() {
        return finalLoss;
    }

    public double getFinalAccuracy() {
        return finalAccuracy;
    }

    /**
     * Returns per-channel normalization statistics computed from the training dataset,
     * or null if not available (older models without this data).
     * <p>
     * Each map contains keys: p1, p99, min, max, mean, std.
     */
    public List<Map<String, Double>> getNormalizationStats() {
        return normalizationStats;
    }

    /**
     * Returns true if this model has training dataset normalization statistics.
     */
    public boolean hasNormalizationStats() {
        return normalizationStats != null && !normalizationStats.isEmpty();
    }

    /**
     * Returns the full training hyperparameters map, or null if not available
     * (older models trained before this feature was added).
     * <p>
     * Keys include: learning_rate, batch_size, weight_decay, validation_split,
     * overlap, line_stroke_width, use_pretrained_weights, frozen_layers,
     * scheduler_type, loss_function, early_stopping_metric, early_stopping_patience,
     * mixed_precision, augmentation_config, class_weight_multipliers.
     */
    public Map<String, Object> getTrainingSettings() {
        return trainingSettings;
    }

    /**
     * Returns true if this model has saved training settings.
     */
    public boolean hasTrainingSettings() {
        return trainingSettings != null && !trainingSettings.isEmpty();
    }

    /**
     * Returns a list of class names.
     */
    public List<String> getClassNames() {
        return classes.stream().map(ClassInfo::name).toList();
    }

    /**
     * Converts this metadata to a Map for JSON serialization.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("version", VERSION);
        map.put("id", id);
        map.put("name", name);
        map.put("description", description);
        map.put("createdAt", createdAt != null ? createdAt.toString() : null);

        Map<String, Object> architecture = new HashMap<>();
        architecture.put("type", modelType);
        architecture.put("backbone", backbone);
        architecture.put("input_width", inputWidth);
        architecture.put("input_height", inputHeight);
        architecture.put("input_channels", inputChannels);
        architecture.put("effective_input_channels", getEffectiveInputChannels());
        architecture.put("downsample", downsample);
        architecture.put("context_scale", contextScale);
        map.put("architecture", architecture);

        Map<String, Object> channelConfig = new HashMap<>();
        channelConfig.put("expected_channels", expectedChannelNames);
        channelConfig.put("normalization_strategy", normalizationStrategy.name());
        channelConfig.put("bit_depth_trained", bitDepthTrained);
        map.put("channel_config", channelConfig);

        List<Map<String, Object>> classesInfo = new ArrayList<>();
        for (ClassInfo ci : classes) {
            Map<String, Object> classMap = new HashMap<>();
            classMap.put("index", ci.index());
            classMap.put("name", ci.name());
            classMap.put("color", ci.color());
            classesInfo.add(classMap);
        }
        map.put("classes", classesInfo);

        Map<String, Object> training = new HashMap<>();
        training.put("image_name", trainingImageName);
        training.put("epochs", trainingEpochs);
        training.put("final_loss", finalLoss);
        training.put("final_accuracy", finalAccuracy);
        map.put("training", training);

        if (trainingSettings != null && !trainingSettings.isEmpty()) {
            map.put("training_settings", trainingSettings);
        }

        if (normalizationStats != null && !normalizationStats.isEmpty()) {
            map.put("normalization_stats", normalizationStats);
        }

        return map;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ClassifierMetadata that = (ClassifierMetadata) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return String.format("ClassifierMetadata{id='%s', name='%s', type=%s, classes=%d}",
                id, name, modelType, classes.size());
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Information about a classification class.
     */
    public record ClassInfo(int index, String name, String color) {
        public ClassInfo {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("Class name cannot be null or empty");
            }
        }
    }

    /**
     * Builder for ClassifierMetadata.
     */
    public static class Builder {
        private String id;
        private String name;
        private String description = "";
        private LocalDateTime createdAt = LocalDateTime.now();
        private String modelType = "unet";
        private String backbone = "resnet34";
        private int inputWidth = 512;
        private int inputHeight = 512;
        private int inputChannels = 3;
        private double downsample = 1.0;
        private int contextScale = 1;
        private List<String> expectedChannelNames = new ArrayList<>();
        private ChannelConfiguration.NormalizationStrategy normalizationStrategy =
                ChannelConfiguration.NormalizationStrategy.PERCENTILE_99;
        private int bitDepthTrained = 8;
        private List<ClassInfo> classes = new ArrayList<>();
        private String trainingImageName = "";
        private int trainingEpochs = 0;
        private double finalLoss = 0.0;
        private double finalAccuracy = 0.0;
        private Map<String, Object> trainingSettings;
        private List<Map<String, Double>> normalizationStats;
        private double trainingPixelSizeMicrons = Double.NaN;
        private int trainingTileSizePx = 0;

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder createdAt(LocalDateTime createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder modelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        public Builder backbone(String backbone) {
            this.backbone = backbone;
            return this;
        }

        public Builder inputSize(int width, int height) {
            this.inputWidth = width;
            this.inputHeight = height;
            return this;
        }

        public Builder inputChannels(int channels) {
            this.inputChannels = channels;
            return this;
        }

        /**
         * Sets the downsample factor used during training.
         *
         * @param downsample downsample factor (1.0 = full resolution)
         */
        public Builder downsample(double downsample) {
            this.downsample = downsample;
            return this;
        }

        /**
         * Sets the context scale factor for multi-scale inference.
         *
         * @param contextScale context scale (1 = disabled, 2/4/8 = enabled)
         */
        public Builder contextScale(int contextScale) {
            this.contextScale = contextScale;
            return this;
        }

        public Builder expectedChannelNames(List<String> names) {
            this.expectedChannelNames = new ArrayList<>(names);
            return this;
        }

        public Builder normalizationStrategy(ChannelConfiguration.NormalizationStrategy strategy) {
            this.normalizationStrategy = strategy;
            return this;
        }

        public Builder bitDepthTrained(int bitDepth) {
            this.bitDepthTrained = bitDepth;
            return this;
        }

        public Builder classes(List<ClassInfo> classes) {
            this.classes = new ArrayList<>(classes);
            return this;
        }

        public Builder addClass(int index, String name, String color) {
            this.classes.add(new ClassInfo(index, name, color));
            return this;
        }

        public Builder trainingImageName(String imageName) {
            this.trainingImageName = imageName;
            return this;
        }

        public Builder trainingEpochs(int epochs) {
            this.trainingEpochs = epochs;
            return this;
        }

        public Builder finalLoss(double loss) {
            this.finalLoss = loss;
            return this;
        }

        public Builder finalAccuracy(double accuracy) {
            this.finalAccuracy = accuracy;
            return this;
        }

        /**
         * Sets the full training hyperparameters map for metadata persistence.
         * <p>
         * This allows reloading training settings when retraining a model.
         *
         * @param settings map of training parameter names to values
         */
        public Builder trainingSettings(Map<String, Object> settings) {
            this.trainingSettings = settings != null ? new LinkedHashMap<>(settings) : null;
            return this;
        }

        /**
         * Sets per-channel normalization statistics from the training dataset.
         *
         * @param stats list of per-channel stat maps (p1, p99, min, max, mean, std)
         */
        public Builder normalizationStats(List<Map<String, Double>> stats) {
            this.normalizationStats = stats != null ? new ArrayList<>(stats) : null;
            return this;
        }

        /**
         * Sets the training data's physical pixel size in microns per pixel.
         * Saved into the model's metadata.json as the resolution contract.
         */
        public Builder trainingPixelSizeMicrons(double value) {
            this.trainingPixelSizeMicrons = value;
            return this;
        }

        /**
         * Sets the training tile size in pixels (== input_size).
         */
        public Builder trainingTileSizePx(int value) {
            this.trainingTileSizePx = value;
            return this;
        }

        public ClassifierMetadata build() {
            if (id == null || id.isEmpty()) {
                throw new IllegalStateException("ID must be specified");
            }
            if (name == null || name.isEmpty()) {
                throw new IllegalStateException("Name must be specified");
            }
            if (classes.isEmpty()) {
                throw new IllegalStateException("At least one class must be defined");
            }
            return new ClassifierMetadata(this);
        }
    }
}
