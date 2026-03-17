package qupath.ext.dlclassifier.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration parameters for training a deep learning pixel classifier.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TrainingConfig {

    // Model architecture
    private final String modelType;
    private final String backbone;

    // Training hyperparameters
    private final int epochs;
    private final int batchSize;
    private final double learningRate;
    private final double weightDecay;

    // Tile parameters
    private final int tileSize;
    private final int overlap;
    private final double downsample;

    // Data configuration
    private final double validationSplit;
    private final Map<String, Boolean> augmentationConfig;

    // Transfer learning
    private final boolean usePretrainedWeights;
    private final int freezeEncoderLayers;
    private final List<String> frozenLayers;

    // Annotation rendering
    private final int lineStrokeWidth;

    // Class weight multipliers (user-supplied multipliers on auto-computed inverse-frequency weights)
    private final Map<String, Double> classWeightMultipliers;

    // Multi-scale context
    private final int contextScale;

    // Training strategy
    private final String schedulerType;
    private final String lossFunction;
    private final double focalGamma;
    private final double ohemHardRatio;
    private final String earlyStoppingMetric;
    private final int earlyStoppingPatience;
    private final boolean mixedPrecision;

    // Focus class for best model selection and early stopping
    private final String focusClass;       // null = disabled (use earlyStoppingMetric as-is)
    private final double focusClassMinIoU; // 0.0 = no minimum threshold

    // Intensity augmentation mode: "none", "brightfield", or "fluorescence"
    private final String intensityAugMode;

    // Gradient accumulation (effective batch = batchSize * gradientAccumulationSteps)
    private final int gradientAccumulationSteps;

    // Progressive resizing: train at half resolution first, then full
    private final boolean progressiveResize;

    // Continue training: path to a previously trained model's .pt file to load weights from
    private final String pretrainedModelPath;

    // Whole-image mode: use entire image as a single training tile (no tiling)
    private final boolean wholeImage;

    // Handler-specific UI parameters (e.g., MuViT model_config, patch_size, level_scales, rope_mode).
    // These override values from ClassifierHandler.getArchitectureParams() so that user
    // selections in the handler UI are actually sent to the Python backend.
    private final Map<String, Object> handlerParameters;

    // Transient runtime field: project-local directory for model output.
    // Not part of equals/hashCode/builder -- set at runtime by TrainingWorkflow
    // to redirect model saving directly into the project's classifiers directory.
    private String modelOutputDir;

    // Transient runtime overrides for whole-image mode.
    // When set (> 0), these override the builder-configured values so that
    // downstream code (backend, serialization) automatically uses the safe values.
    private int runtimeTileSize = -1;
    private int runtimeBatchSize = -1;
    private int runtimeGradAccumSteps = -1;

    private TrainingConfig(Builder builder) {
        this.modelType = builder.modelType;
        this.backbone = builder.backbone;
        this.epochs = builder.epochs;
        this.batchSize = builder.batchSize;
        this.learningRate = builder.learningRate;
        this.weightDecay = builder.weightDecay;
        this.tileSize = builder.tileSize;
        this.overlap = builder.overlap;
        this.downsample = builder.downsample;
        this.validationSplit = builder.validationSplit;
        this.augmentationConfig = Collections.unmodifiableMap(new LinkedHashMap<>(builder.augmentationConfig));
        this.usePretrainedWeights = builder.usePretrainedWeights;
        this.freezeEncoderLayers = builder.freezeEncoderLayers;
        this.frozenLayers = Collections.unmodifiableList(new ArrayList<>(builder.frozenLayers));
        this.lineStrokeWidth = builder.lineStrokeWidth;
        this.classWeightMultipliers = Collections.unmodifiableMap(new LinkedHashMap<>(builder.classWeightMultipliers));
        this.contextScale = builder.contextScale;
        this.schedulerType = builder.schedulerType;
        this.lossFunction = builder.lossFunction;
        this.focalGamma = builder.focalGamma;
        this.ohemHardRatio = builder.ohemHardRatio;
        this.earlyStoppingMetric = builder.earlyStoppingMetric;
        this.earlyStoppingPatience = builder.earlyStoppingPatience;
        this.mixedPrecision = builder.mixedPrecision;
        this.focusClass = builder.focusClass;
        this.focusClassMinIoU = builder.focusClassMinIoU;
        this.intensityAugMode = builder.intensityAugMode;
        this.gradientAccumulationSteps = builder.gradientAccumulationSteps;
        this.progressiveResize = builder.progressiveResize;
        this.pretrainedModelPath = builder.pretrainedModelPath;
        this.wholeImage = builder.wholeImage;
        this.handlerParameters = Collections.unmodifiableMap(new LinkedHashMap<>(builder.handlerParameters));
    }

    // Getters

    public String getModelType() {
        return modelType;
    }

    public String getBackbone() {
        return backbone;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getBatchSize() {
        return runtimeBatchSize > 0 ? runtimeBatchSize : batchSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getWeightDecay() {
        return weightDecay;
    }

    public int getTileSize() {
        return runtimeTileSize > 0 ? runtimeTileSize : tileSize;
    }

    public int getOverlap() {
        return overlap;
    }

    /**
     * Gets the downsample factor for tile extraction.
     * <p>
     * At downsample 1.0, tiles are extracted at full resolution.
     * At downsample 4.0, each tile covers 4x the spatial area,
     * providing more context for tissue-level classification.
     *
     * @return downsample factor (1.0 = full resolution)
     */
    public double getDownsample() {
        return downsample;
    }

    public double getValidationSplit() {
        return validationSplit;
    }

    /**
     * Gets the augmentation configuration map.
     *
     * @return map of augmentation type to enabled status
     */
    public Map<String, Boolean> getAugmentationConfig() {
        return augmentationConfig;
    }

    /**
     * Checks if any augmentation is enabled.
     *
     * @return true if at least one augmentation or intensity mode is enabled
     */
    public boolean isAugmentation() {
        boolean spatialAug = augmentationConfig.values().stream().anyMatch(v -> v);
        boolean intensityAug = intensityAugMode != null && !"none".equals(intensityAugMode);
        return spatialAug || intensityAug;
    }

    public boolean isUsePretrainedWeights() {
        return usePretrainedWeights;
    }

    public int getFreezeEncoderLayers() {
        return freezeEncoderLayers;
    }

    /**
     * Gets the list of layer names to freeze during training.
     *
     * @return list of layer names (e.g., "encoder.layer1", "encoder.layer2")
     */
    public List<String> getFrozenLayers() {
        return frozenLayers;
    }

    /**
     * Gets the stroke width for rendering line/polyline annotations as training masks.
     *
     * @return stroke width in pixels
     */
    public int getLineStrokeWidth() {
        return lineStrokeWidth;
    }

    /**
     * Gets the user-supplied class weight multipliers.
     * <p>
     * These multipliers are applied on top of auto-computed inverse-frequency weights.
     * A multiplier of 1.0 means no change; values &gt; 1.0 emphasize a class.
     *
     * @return map of class name to weight multiplier (empty map means no modification)
     */
    public Map<String, Double> getClassWeightMultipliers() {
        return classWeightMultipliers;
    }

    /**
     * Gets the multi-scale context scale factor.
     * <p>
     * When greater than 1, each training tile also has a context tile extracted
     * from a region {@code contextScale} times larger, downsampled to the same
     * pixel dimensions. The two tiles are concatenated along the channel axis,
     * doubling the model's input channels.
     *
     * @return context scale factor (1 = disabled, 2/4/8 = context enabled)
     */
    public int getContextScale() {
        return contextScale;
    }

    /**
     * Gets the learning rate scheduler type.
     *
     * @return scheduler type ("onecycle", "cosine", "step", or "none")
     */
    public String getSchedulerType() {
        return schedulerType;
    }

    /**
     * Gets the loss function type.
     *
     * @return loss function ("ce_dice", "cross_entropy", "focal_dice", or "focal")
     */
    public String getLossFunction() {
        return lossFunction;
    }

    /**
     * Gets the focal loss gamma parameter.
     * <p>
     * Controls focusing strength: higher gamma further down-weights
     * easy pixels. Standard value is 2.0. Only used when loss function
     * is "focal_dice" or "focal".
     *
     * @return gamma value (0.5-5.0, default 2.0)
     */
    public double getFocalGamma() {
        return focalGamma;
    }

    /**
     * Gets the OHEM hard pixel ratio.
     * <p>
     * Fraction of pixels kept per batch (hardest pixels only).
     * 1.0 means all pixels are kept (OHEM disabled).
     *
     * @return hard ratio (0.05-1.0, default 1.0 = disabled)
     */
    public double getOhemHardRatio() {
        return ohemHardRatio;
    }

    /**
     * Gets the metric used for early stopping.
     *
     * @return early stopping metric ("mean_iou" or "val_loss")
     */
    public String getEarlyStoppingMetric() {
        return earlyStoppingMetric;
    }

    /**
     * Gets the early stopping patience (epochs to wait without improvement).
     *
     * @return patience value
     */
    public int getEarlyStoppingPatience() {
        return earlyStoppingPatience;
    }

    /**
     * Checks whether mixed precision (AMP) training is enabled.
     *
     * @return true if mixed precision is enabled
     */
    public boolean isMixedPrecision() {
        return mixedPrecision;
    }

    /**
     * Gets the focus class name for best model selection and early stopping.
     * <p>
     * When set, the focus class's per-class IoU is used instead of mean IoU
     * or validation loss for determining the best model and early stopping.
     *
     * @return focus class name, or null if disabled (uses earlyStoppingMetric)
     */
    public String getFocusClass() {
        return focusClass;
    }

    /**
     * Gets the minimum IoU threshold for the focus class.
     * <p>
     * When greater than 0, early stopping is suppressed until the focus class
     * reaches this IoU, regardless of patience. Training will continue until
     * either max epochs or the threshold is reached and patience expires.
     *
     * @return minimum IoU threshold (0.0 = no threshold)
     */
    public double getFocusClassMinIoU() {
        return focusClassMinIoU;
    }

    /**
     * Gets the intensity augmentation mode.
     * <p>
     * Controls how color/intensity transforms are applied during training:
     * <ul>
     *   <li>{@code "none"} -- no intensity transforms</li>
     *   <li>{@code "brightfield"} -- RGB-correlated brightness/contrast/gamma (for H&amp;E)</li>
     *   <li>{@code "fluorescence"} -- per-channel independent intensity jitter</li>
     * </ul>
     *
     * @return intensity augmentation mode string
     */
    public String getIntensityAugMode() {
        return intensityAugMode;
    }

    /**
     * Gets the gradient accumulation steps.
     * <p>
     * When greater than 1, gradients are accumulated across this many batches
     * before a single optimizer step, effectively increasing the batch size
     * without using more GPU memory.
     *
     * @return accumulation steps (1 = no accumulation)
     */
    public int getGradientAccumulationSteps() {
        return runtimeGradAccumSteps > 0 ? runtimeGradAccumSteps : gradientAccumulationSteps;
    }

    /**
     * Checks whether progressive resizing is enabled.
     * <p>
     * When enabled, training starts at half tile resolution for the first 40% of epochs,
     * then switches to full resolution. This can speed up early training and
     * act as a form of regularization.
     *
     * @return true if progressive resizing is enabled
     */
    public boolean isProgressiveResize() {
        return progressiveResize;
    }

    /**
     * Gets the path to a previously trained model's .pt file for weight initialization.
     * <p>
     * When set, the model weights are loaded from this file before training begins.
     * Only network weights are loaded -- optimizer, scheduler, and early stopping
     * state all start fresh (correct for fine-tuning on new data).
     * <p>
     * If the new training has different classes, the segmentation head weights
     * won't match and are skipped (randomly initialized). Encoder/decoder weights
     * (the valuable part) transfer correctly.
     *
     * @return path to .pt file, or null if not using pretrained weights from a previous model
     */
    public String getPretrainedModelPath() {
        return pretrainedModelPath;
    }

    /**
     * Checks whether whole-image mode is enabled.
     * <p>
     * When enabled, the entire image is used as a single training tile
     * instead of extracting fixed-size patches. The effective tile size
     * is computed at export time from the actual image dimensions.
     *
     * @return true if whole-image mode is enabled
     */
    public boolean isWholeImage() {
        return wholeImage;
    }

    /**
     * Gets the handler-specific UI parameters.
     * <p>
     * These contain the actual user selections from the handler UI
     * (e.g., MuViT model_config, patch_size, level_scales, rope_mode).
     * They override the static defaults from
     * {@link qupath.ext.dlclassifier.classifier.ClassifierHandler#getArchitectureParams}.
     *
     * @return handler parameters map (never null, may be empty)
     */
    public Map<String, Object> getHandlerParameters() {
        return handlerParameters;
    }

    /**
     * Computes the effective tile size for whole-image mode.
     * <p>
     * Takes the maximum of image width and height, divides by the downsample
     * factor, and rounds up to the nearest multiple of 32 (required by
     * encoder downsampling). When whole-image mode is disabled, returns
     * the configured tile size unchanged.
     *
     * @param imageWidth  image width in pixels (at full resolution)
     * @param imageHeight image height in pixels (at full resolution)
     * @return effective tile size (multiple of 32)
     */
    public int computeEffectiveTileSize(int imageWidth, int imageHeight) {
        return computeEffectiveTileSize(imageWidth, imageHeight, Integer.MAX_VALUE);
    }

    /**
     * Computes the effective tile size for whole-image mode, capped at maxTileSize.
     * <p>
     * When the computed tile size exceeds maxTileSize (e.g., for ViT models
     * where global self-attention is O(n^2) in patch count), the result is
     * capped and the image will be tiled instead of processed whole.
     *
     * @param imageWidth   image width in pixels (at full resolution)
     * @param imageHeight  image height in pixels (at full resolution)
     * @param maxTileSize  maximum allowed tile size (from handler's supported tile sizes)
     * @return effective tile size (multiple of 32, capped at maxTileSize)
     */
    public int computeEffectiveTileSize(int imageWidth, int imageHeight, int maxTileSize) {
        if (!wholeImage) return tileSize;
        int maxDim = Math.max(imageWidth, imageHeight);
        int rawSize = (int) Math.ceil(maxDim / downsample);
        int rounded = ((rawSize + 31) / 32) * 32;
        return Math.min(rounded, maxTileSize);
    }

    /**
     * Gets the project-local model output directory.
     * <p>
     * When set, Python training scripts save model files directly to this
     * directory instead of the default {@code ~/.dlclassifier/models/}.
     * This is a transient runtime field set by {@code TrainingWorkflow}.
     *
     * @return model output directory path, or null if using default location
     */
    public String getModelOutputDir() {
        return modelOutputDir;
    }

    /**
     * Sets the project-local model output directory.
     *
     * @param modelOutputDir directory path, or null to use default location
     */
    public void setModelOutputDir(String modelOutputDir) {
        this.modelOutputDir = modelOutputDir;
    }

    /**
     * Adjusts batch size for large whole-image tiles to prevent GPU OOM.
     * <p>
     * When the effective tile size exceeds 1024px, batch size is reduced to 1
     * and gradient accumulation is increased to maintain the same effective
     * batch size. This trades training speed for memory safety.
     * <p>
     * Has no effect if batch size is already 1 or tile size is &lt;= 1024.
     *
     * @param effectiveTileSize the computed tile size in pixels
     */
    /**
     * Sets the runtime tile size override for whole-image mode.
     * <p>
     * When set, {@link #getTileSize()} returns this value instead of the
     * builder-configured tile size. This ensures the Python backend receives
     * the computed effective tile size.
     *
     * @param effectiveTileSize the computed effective tile size
     */
    public void setEffectiveTileSize(int effectiveTileSize) {
        this.runtimeTileSize = effectiveTileSize;
    }

    /**
     * Adjusts batch size for large tiles to prevent GPU OOM.
     * <p>
     * The threshold is model-dependent: ViT models (MuViT) use global
     * self-attention which is O(n^2) in patch count, so they need batch
     * reduction at much smaller tile sizes than CNN models (UNet).
     * <p>
     * When triggered, batch size is reduced to 1 and gradient accumulation
     * is increased to maintain the same effective batch size.
     *
     * @param effectiveTileSize the computed tile size in pixels
     */
    public void adjustBatchForTileSize(int effectiveTileSize) {
        if (batchSize <= 1) return;

        // ViT models need batch=1 at much smaller tile sizes than CNNs
        // because self-attention memory scales quadratically with patch count.
        int threshold = "muvit".equals(modelType) ? 256 : 1024;

        if (effectiveTileSize <= threshold) return;
        runtimeBatchSize = 1;
        runtimeGradAccumSteps = batchSize * gradientAccumulationSteps;
    }

    /**
     * Returns the effective tile step size (tileSize - overlap).
     */
    public int getStepSize() {
        return tileSize - overlap;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TrainingConfig that = (TrainingConfig) o;
        return epochs == that.epochs &&
                batchSize == that.batchSize &&
                Double.compare(that.learningRate, learningRate) == 0 &&
                Double.compare(that.weightDecay, weightDecay) == 0 &&
                tileSize == that.tileSize &&
                overlap == that.overlap &&
                Double.compare(that.downsample, downsample) == 0 &&
                Double.compare(that.validationSplit, validationSplit) == 0 &&
                usePretrainedWeights == that.usePretrainedWeights &&
                freezeEncoderLayers == that.freezeEncoderLayers &&
                lineStrokeWidth == that.lineStrokeWidth &&
                contextScale == that.contextScale &&
                earlyStoppingPatience == that.earlyStoppingPatience &&
                mixedPrecision == that.mixedPrecision &&
                Double.compare(that.focusClassMinIoU, focusClassMinIoU) == 0 &&
                Objects.equals(modelType, that.modelType) &&
                Objects.equals(backbone, that.backbone) &&
                Objects.equals(augmentationConfig, that.augmentationConfig) &&
                Objects.equals(frozenLayers, that.frozenLayers) &&
                Objects.equals(classWeightMultipliers, that.classWeightMultipliers) &&
                Objects.equals(schedulerType, that.schedulerType) &&
                Objects.equals(lossFunction, that.lossFunction) &&
                Double.compare(that.focalGamma, focalGamma) == 0 &&
                Double.compare(that.ohemHardRatio, ohemHardRatio) == 0 &&
                Objects.equals(earlyStoppingMetric, that.earlyStoppingMetric) &&
                Objects.equals(focusClass, that.focusClass) &&
                Objects.equals(intensityAugMode, that.intensityAugMode) &&
                gradientAccumulationSteps == that.gradientAccumulationSteps &&
                progressiveResize == that.progressiveResize &&
                wholeImage == that.wholeImage &&
                Objects.equals(pretrainedModelPath, that.pretrainedModelPath);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelType, backbone, epochs, batchSize, learningRate,
                weightDecay, tileSize, overlap, downsample, validationSplit, augmentationConfig,
                usePretrainedWeights, freezeEncoderLayers, frozenLayers, lineStrokeWidth,
                classWeightMultipliers, contextScale, schedulerType, lossFunction,
                focalGamma, ohemHardRatio,
                earlyStoppingMetric, earlyStoppingPatience, mixedPrecision,
                focusClass, focusClassMinIoU, intensityAugMode,
                gradientAccumulationSteps, progressiveResize, wholeImage, pretrainedModelPath);
    }

    @Override
    public String toString() {
        return String.format("TrainingConfig{model=%s, backbone=%s, epochs=%d, lr=%.6f, wd=%.4f, tile=%d, downsample=%.1f, contextScale=%d, lineStroke=%d, scheduler=%s, loss=%s, focalGamma=%.1f, ohemRatio=%.2f, esMetric=%s, esPat=%d, amp=%b, focusClass=%s, focusMinIoU=%.2f, intensityAug=%s, gradAccum=%d, progResize=%b, wholeImage=%b, pretrainedModel=%s}",
                modelType, backbone, epochs, learningRate, weightDecay, tileSize, downsample, contextScale, lineStrokeWidth,
                schedulerType, lossFunction, focalGamma, ohemHardRatio, earlyStoppingMetric, earlyStoppingPatience, mixedPrecision,
                focusClass, focusClassMinIoU, intensityAugMode,
                gradientAccumulationSteps, progressiveResize, wholeImage, pretrainedModelPath);
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for TrainingConfig.
     */
    public static class Builder {
        private String modelType = "unet";
        private String backbone = "resnet34";
        private int epochs = 50;
        private int batchSize = 8;
        private double learningRate = 0.001;
        private double weightDecay = 0.01;
        private int tileSize = 512;
        private int overlap = 64;
        private double downsample = 1.0;
        private double validationSplit = 0.2;
        private Map<String, Boolean> augmentationConfig = new LinkedHashMap<>();
        private boolean usePretrainedWeights = true;
        private int freezeEncoderLayers = 0;
        private List<String> frozenLayers = new ArrayList<>();
        private int lineStrokeWidth = 5;
        private Map<String, Double> classWeightMultipliers = new LinkedHashMap<>();
        private int contextScale = 1;
        private String schedulerType = "onecycle";
        private String lossFunction = "ce_dice";
        private double focalGamma = 2.0;
        private double ohemHardRatio = 1.0;
        private String earlyStoppingMetric = "mean_iou";
        private int earlyStoppingPatience = 15;
        private boolean mixedPrecision = true;
        private String focusClass = null;
        private double focusClassMinIoU = 0.0;
        private String intensityAugMode = "none";
        private int gradientAccumulationSteps = 1;
        private boolean progressiveResize = false;
        private String pretrainedModelPath = null;
        private boolean wholeImage = false;
        private Map<String, Object> handlerParameters = new LinkedHashMap<>();

        public Builder() {
            // Default augmentation configuration (spatial transforms only)
            augmentationConfig.put("flip_horizontal", true);
            augmentationConfig.put("flip_vertical", true);
            augmentationConfig.put("rotation_90", true);
            augmentationConfig.put("elastic_deformation", false);
        }

        /**
         * Copies all fields from an existing TrainingConfig into this builder.
         * Allows creating a modified copy via
         * {@code TrainingConfig.builder().from(existing).pretrainedModelPath(newPath).build()}.
         *
         * @param config the config to copy from
         * @return this builder
         */
        public Builder from(TrainingConfig config) {
            this.modelType = config.modelType;
            this.backbone = config.backbone;
            this.epochs = config.epochs;
            this.batchSize = config.batchSize;
            this.learningRate = config.learningRate;
            this.weightDecay = config.weightDecay;
            this.tileSize = config.tileSize;
            this.overlap = config.overlap;
            this.downsample = config.downsample;
            this.validationSplit = config.validationSplit;
            this.augmentationConfig = new LinkedHashMap<>(config.augmentationConfig);
            this.usePretrainedWeights = config.usePretrainedWeights;
            this.freezeEncoderLayers = config.freezeEncoderLayers;
            this.frozenLayers = new ArrayList<>(config.frozenLayers);
            this.lineStrokeWidth = config.lineStrokeWidth;
            this.classWeightMultipliers = new LinkedHashMap<>(config.classWeightMultipliers);
            this.contextScale = config.contextScale;
            this.schedulerType = config.schedulerType;
            this.lossFunction = config.lossFunction;
            this.focalGamma = config.focalGamma;
            this.ohemHardRatio = config.ohemHardRatio;
            this.earlyStoppingMetric = config.earlyStoppingMetric;
            this.earlyStoppingPatience = config.earlyStoppingPatience;
            this.mixedPrecision = config.mixedPrecision;
            this.focusClass = config.focusClass;
            this.focusClassMinIoU = config.focusClassMinIoU;
            this.intensityAugMode = config.intensityAugMode;
            this.gradientAccumulationSteps = config.gradientAccumulationSteps;
            this.progressiveResize = config.progressiveResize;
            this.pretrainedModelPath = config.pretrainedModelPath;
            this.wholeImage = config.wholeImage;
            this.handlerParameters = new LinkedHashMap<>(config.handlerParameters);
            return this;
        }

        public Builder modelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        /**
         * Alias for modelType() for more readable code.
         */
        public Builder classifierType(String classifierType) {
            return modelType(classifierType);
        }

        public Builder backbone(String backbone) {
            this.backbone = backbone;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder weightDecay(double weightDecay) {
            this.weightDecay = weightDecay;
            return this;
        }

        public Builder tileSize(int tileSize) {
            this.tileSize = tileSize;
            return this;
        }

        public Builder overlap(int overlap) {
            this.overlap = overlap;
            return this;
        }

        /**
         * Sets the downsample factor for tile extraction.
         * <p>
         * Higher downsample = more spatial context per tile but less detail.
         * Recommended: 2-4x for tissue-level classification, 1x for cell-level.
         *
         * @param downsample downsample factor (1.0-32.0)
         */
        public Builder downsample(double downsample) {
            this.downsample = downsample;
            return this;
        }

        public Builder validationSplit(double validationSplit) {
            this.validationSplit = validationSplit;
            return this;
        }

        /**
         * Sets detailed augmentation configuration.
         *
         * @param augmentationConfig map of augmentation type to enabled status
         */
        public Builder augmentation(Map<String, Boolean> augmentationConfig) {
            this.augmentationConfig = new LinkedHashMap<>(augmentationConfig);
            return this;
        }

        /**
         * Enables or disables all augmentation.
         *
         * @param enabled true to enable default augmentation, false to disable all
         */
        public Builder augmentation(boolean enabled) {
            if (enabled) {
                augmentationConfig.put("flip_horizontal", true);
                augmentationConfig.put("flip_vertical", true);
                augmentationConfig.put("rotation_90", true);
            } else {
                augmentationConfig.clear();
            }
            return this;
        }

        public Builder usePretrainedWeights(boolean usePretrainedWeights) {
            this.usePretrainedWeights = usePretrainedWeights;
            return this;
        }

        public Builder freezeEncoderLayers(int freezeEncoderLayers) {
            this.freezeEncoderLayers = freezeEncoderLayers;
            return this;
        }

        /**
         * Sets the list of layer names to freeze during training.
         * This provides fine-grained control over transfer learning.
         *
         * @param frozenLayers list of layer names to freeze
         */
        public Builder frozenLayers(List<String> frozenLayers) {
            this.frozenLayers = new ArrayList<>(frozenLayers);
            return this;
        }

        /**
         * Adds a single layer to freeze during training.
         *
         * @param layerName name of the layer to freeze
         */
        public Builder freezeLayer(String layerName) {
            this.frozenLayers.add(layerName);
            return this;
        }

        /**
         * Sets the stroke width for rendering line/polyline annotations as training masks.
         *
         * @param lineStrokeWidth stroke width in pixels (1-50)
         */
        public Builder lineStrokeWidth(int lineStrokeWidth) {
            this.lineStrokeWidth = lineStrokeWidth;
            return this;
        }

        /**
         * Sets class weight multipliers applied on top of auto-computed inverse-frequency weights.
         *
         * @param classWeightMultipliers map of class name to multiplier (default 1.0)
         */
        public Builder classWeightMultipliers(Map<String, Double> classWeightMultipliers) {
            this.classWeightMultipliers = new LinkedHashMap<>(classWeightMultipliers);
            return this;
        }

        /**
         * Sets the multi-scale context scale factor.
         * <p>
         * When greater than 1, training data export will also extract a context
         * tile from a region contextScale times larger, downsampled to the same
         * pixel size. The model receives detail + context channels (2*C input).
         *
         * @param contextScale context scale factor (1 = disabled, 2/4/8 = context)
         */
        public Builder contextScale(int contextScale) {
            this.contextScale = contextScale;
            return this;
        }

        /**
         * Sets the learning rate scheduler type.
         *
         * @param schedulerType "onecycle", "cosine", "step", or "none"
         */
        public Builder schedulerType(String schedulerType) {
            this.schedulerType = schedulerType;
            return this;
        }

        /**
         * Sets the loss function type.
         *
         * @param lossFunction "ce_dice", "cross_entropy", "focal_dice", or "focal"
         */
        public Builder lossFunction(String lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        /**
         * Sets the focal loss gamma parameter.
         *
         * @param focalGamma focusing strength (0.5-5.0, default 2.0)
         */
        public Builder focalGamma(double focalGamma) {
            this.focalGamma = focalGamma;
            return this;
        }

        /**
         * Sets the OHEM hard pixel ratio.
         * <p>
         * Only the hardest fraction of pixels contribute to the loss.
         * 1.0 means all pixels are kept (OHEM disabled).
         *
         * @param ohemHardRatio fraction of pixels to keep (0.05-1.0)
         */
        public Builder ohemHardRatio(double ohemHardRatio) {
            this.ohemHardRatio = ohemHardRatio;
            return this;
        }

        /**
         * Sets the metric used for early stopping.
         *
         * @param earlyStoppingMetric "mean_iou" or "val_loss"
         */
        public Builder earlyStoppingMetric(String earlyStoppingMetric) {
            this.earlyStoppingMetric = earlyStoppingMetric;
            return this;
        }

        /**
         * Sets the early stopping patience.
         *
         * @param earlyStoppingPatience epochs to wait without improvement (3-50)
         */
        public Builder earlyStoppingPatience(int earlyStoppingPatience) {
            this.earlyStoppingPatience = earlyStoppingPatience;
            return this;
        }

        /**
         * Sets whether to use mixed precision (AMP) training.
         *
         * @param mixedPrecision true to enable mixed precision on CUDA devices
         */
        public Builder mixedPrecision(boolean mixedPrecision) {
            this.mixedPrecision = mixedPrecision;
            return this;
        }

        /**
         * Sets the focus class for best model selection and early stopping.
         * <p>
         * When set, the focus class's per-class IoU is used instead of the
         * configured early stopping metric for determining the best model
         * and triggering early stopping.
         *
         * @param focusClass class name, or null to disable
         */
        public Builder focusClass(String focusClass) {
            this.focusClass = focusClass;
            return this;
        }

        /**
         * Sets the minimum IoU threshold for the focus class.
         * <p>
         * Early stopping is suppressed until the focus class reaches this IoU.
         *
         * @param focusClassMinIoU minimum IoU threshold (0.0-1.0, 0.0 = no threshold)
         */
        public Builder focusClassMinIoU(double focusClassMinIoU) {
            this.focusClassMinIoU = focusClassMinIoU;
            return this;
        }

        /**
         * Sets the intensity augmentation mode.
         *
         * @param intensityAugMode "none", "brightfield", or "fluorescence"
         */
        public Builder intensityAugMode(String intensityAugMode) {
            this.intensityAugMode = intensityAugMode;
            return this;
        }

        /**
         * Sets the gradient accumulation steps.
         * <p>
         * Effective batch size = batchSize * gradientAccumulationSteps.
         * Use when GPU memory is limited but larger effective batches are desired.
         *
         * @param steps accumulation steps (1-8, 1 = no accumulation)
         */
        public Builder gradientAccumulationSteps(int steps) {
            this.gradientAccumulationSteps = steps;
            return this;
        }

        /**
         * Sets whether to use progressive resizing.
         * <p>
         * Trains at half resolution for the first 40% of epochs, then full resolution.
         *
         * @param progressiveResize true to enable progressive resizing
         */
        public Builder progressiveResize(boolean progressiveResize) {
            this.progressiveResize = progressiveResize;
            return this;
        }

        /**
         * Sets the path to a previously trained model's .pt file for weight initialization.
         * <p>
         * When set, model weights are loaded from this file before training begins.
         * Optimizer and scheduler start fresh (fine-tuning behavior).
         *
         * @param pretrainedModelPath path to .pt file, or null to disable
         */
        public Builder pretrainedModelPath(String pretrainedModelPath) {
            this.pretrainedModelPath = pretrainedModelPath;
            return this;
        }

        /**
         * Sets whether to use whole-image mode (no tiling).
         * <p>
         * When enabled, the effective tile size is computed at export time
         * from actual image dimensions instead of using the configured tile size.
         *
         * @param wholeImage true to use whole-image mode
         */
        public Builder wholeImage(boolean wholeImage) {
            this.wholeImage = wholeImage;
            return this;
        }

        /**
         * Sets handler-specific UI parameters that override defaults from
         * {@code ClassifierHandler.getArchitectureParams()}.
         *
         * @param handlerParameters map of handler UI parameter names to values
         */
        public Builder handlerParameters(Map<String, Object> handlerParameters) {
            this.handlerParameters = new LinkedHashMap<>(handlerParameters);
            return this;
        }

        public TrainingConfig build() {
            if (modelType == null || modelType.isEmpty()) {
                throw new IllegalStateException("Model type must be specified");
            }
            if (!wholeImage && (tileSize < 64 || tileSize > 2048)) {
                throw new IllegalStateException("Tile size must be between 64 and 2048");
            }
            if (!wholeImage && (overlap < 0 || overlap >= tileSize / 2)) {
                throw new IllegalStateException("Overlap must be between 0 and half of tile size");
            }
            if (downsample < 1.0 || downsample > 32.0) {
                throw new IllegalStateException("Downsample must be between 1.0 and 32.0");
            }
            if (contextScale != 1 && contextScale != 2 && contextScale != 4 && contextScale != 8) {
                throw new IllegalStateException("Context scale must be 1, 2, 4, or 8");
            }
            if (epochs < 1) {
                throw new IllegalStateException("Epochs must be at least 1");
            }
            if (batchSize < 1) {
                throw new IllegalStateException("Batch size must be at least 1");
            }
            if (focusClassMinIoU < 0.0 || focusClassMinIoU > 1.0) {
                throw new IllegalStateException("Focus class min IoU must be between 0.0 and 1.0");
            }
            if (focalGamma < 0.0 || focalGamma > 10.0) {
                throw new IllegalStateException("Focal gamma must be between 0.0 and 10.0");
            }
            if (ohemHardRatio < 0.05 || ohemHardRatio > 1.0) {
                throw new IllegalStateException("OHEM hard ratio must be between 0.05 and 1.0");
            }
            if (gradientAccumulationSteps < 1 || gradientAccumulationSteps > 16) {
                throw new IllegalStateException("Gradient accumulation steps must be between 1 and 16");
            }
            return new TrainingConfig(this);
        }
    }
}
