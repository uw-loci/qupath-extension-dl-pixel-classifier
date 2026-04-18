package qupath.ext.dlclassifier.model;

import java.util.Objects;

/**
 * Configuration parameters for running inference with a trained classifier.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class InferenceConfig {

    /**
     * Application scope for inference.
     */
    public enum ApplicationScope {
        /** Apply only to selected annotations */
        SELECTED_ANNOTATIONS,
        /** Apply to all annotations in the image */
        ALL_ANNOTATIONS,
        /** Apply to the entire image (no annotation required) */
        WHOLE_IMAGE
    }

    /**
     * Output types for classification results.
     */
    public enum OutputType {
        /** Add classification measurements to parent annotations */
        MEASUREMENTS,
        /** Create detection objects from classification */
        OBJECTS,
        /** Create a classification overlay for visualization */
        OVERLAY,
        /** Batch-computed overlay with tile blending -- matches OBJECTS quality */
        RENDERED_OVERLAY
    }

    /**
     * Type of PathObject to create for OBJECTS output.
     */
    public enum OutputObjectType {
        /** Create PathDetectionObjects (default, non-editable) */
        DETECTION,
        /** Create PathAnnotationObjects (editable, selectable) */
        ANNOTATION
    }

    /**
     * Blend modes for tile boundary handling.
     */
    public enum BlendMode {
        /** Linear blending in overlap regions */
        LINEAR,
        /** Gaussian-weighted blending in overlap regions */
        GAUSSIAN,
        /** Keep only center predictions; zero boundary artifacts, ~4x slower */
        CENTER_CROP,
        /** No blending, use tile center values */
        NONE
    }

    // Tile parameters
    private final int tileSize;
    private final int overlap;
    private final double overlapPercent;
    private final BlendMode blendMode;

    // Output configuration
    private final OutputType outputType;
    private final OutputObjectType objectType;
    private final double minObjectSizeMicrons;
    private final double holeFillingMicrons;
    private final double boundarySmoothing;

    // Processing options
    private final int maxTilesInMemory;
    private final boolean useGPU;
    private final boolean useTTA;
    private final boolean multiPassAveraging;

    // Overlay probability smoothing
    private final double overlaySmoothingSigma;

    // Phase 3c: return uint8 class indices from Python instead of float32
    // probability maps. Disables smoothing, multi-pass, and tile blending
    // because those require floats.
    private final boolean useCompactArgmaxOutput;

    private InferenceConfig(Builder builder) {
        this.tileSize = builder.tileSize;
        this.overlap = builder.overlap;
        this.overlapPercent = builder.overlapPercent;
        this.blendMode = builder.blendMode;
        this.outputType = builder.outputType;
        this.objectType = builder.objectType;
        this.minObjectSizeMicrons = builder.minObjectSizeMicrons;
        this.holeFillingMicrons = builder.holeFillingMicrons;
        this.boundarySmoothing = builder.boundarySmoothing;
        this.maxTilesInMemory = builder.maxTilesInMemory;
        this.useGPU = builder.useGPU;
        this.useTTA = builder.useTTA;
        this.multiPassAveraging = builder.multiPassAveraging;
        this.overlaySmoothingSigma = builder.overlaySmoothingSigma;
        this.useCompactArgmaxOutput = builder.useCompactArgmaxOutput;
    }

    // Getters

    public int getTileSize() {
        return tileSize;
    }

    public int getOverlap() {
        return overlap;
    }

    public BlendMode getBlendMode() {
        return blendMode;
    }

    public OutputType getOutputType() {
        return outputType;
    }

    public OutputObjectType getObjectType() {
        return objectType;
    }

    public double getOverlapPercent() {
        return overlapPercent;
    }

    public double getMinObjectSizeMicrons() {
        return minObjectSizeMicrons;
    }

    public double getHoleFillingMicrons() {
        return holeFillingMicrons;
    }

    public double getBoundarySmoothing() {
        return boundarySmoothing;
    }

    public int getMaxTilesInMemory() {
        return maxTilesInMemory;
    }

    public boolean isUseGPU() {
        return useGPU;
    }

    /**
     * Checks whether Test-Time Augmentation (TTA) is enabled.
     * <p>
     * When enabled, inference runs the model on multiple augmented versions
     * of each tile (flips and 90-degree rotations) and averages the predictions.
     * This typically improves segmentation quality by 1-3% but is ~8x slower.
     *
     * @return true if TTA is enabled
     */
    public boolean isUseTTA() {
        return useTTA;
    }

    /**
     * Checks whether multi-pass tile averaging is enabled.
     * <p>
     * When enabled, each tile is inferred at 4 spatial offsets (2x2 grid)
     * and the probability maps are averaged before classification. This
     * eliminates tile boundary artifacts by ensuring each pixel's prediction
     * comes from multiple independent model evaluations with different context.
     * <p>
     * Approximately 4x slower but produces seamless results.
     *
     * @return true if multi-pass averaging is enabled
     */
    public boolean isMultiPassAveraging() {
        return multiPassAveraging;
    }

    /**
     * Returns the Gaussian sigma for overlay probability smoothing.
     * <p>
     * When > 0, a Gaussian blur is applied to probability maps before argmax
     * to smooth noisy per-pixel predictions. A sigma of 2.0 is a good default.
     *
     * @return sigma in pixels (0 = no smoothing)
     */
    public double getOverlaySmoothingSigma() {
        return overlaySmoothingSigma;
    }

    /**
     * Returns true if inference should request uint8 argmax output from
     * the Python side instead of float32 probability maps. When true,
     * overlay smoothing, multi-pass averaging, and tile boundary blending
     * are effectively disabled since they require floats.
     */
    public boolean isUseCompactArgmaxOutput() {
        return useCompactArgmaxOutput;
    }

    /**
     * Returns the effective tile step size (tileSize - overlap).
     */
    public int getStepSize() {
        return tileSize - overlap;
    }

    /**
     * Returns the overlap as a fraction of tile size.
     */
    public double getOverlapFraction() {
        return (double) overlap / tileSize;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        InferenceConfig that = (InferenceConfig) o;
        return tileSize == that.tileSize &&
                overlap == that.overlap &&
                Double.compare(that.overlapPercent, overlapPercent) == 0 &&
                Double.compare(that.minObjectSizeMicrons, minObjectSizeMicrons) == 0 &&
                Double.compare(that.holeFillingMicrons, holeFillingMicrons) == 0 &&
                Double.compare(that.boundarySmoothing, boundarySmoothing) == 0 &&
                maxTilesInMemory == that.maxTilesInMemory &&
                useGPU == that.useGPU &&
                useTTA == that.useTTA &&
                multiPassAveraging == that.multiPassAveraging &&
                Double.compare(that.overlaySmoothingSigma, overlaySmoothingSigma) == 0 &&
                useCompactArgmaxOutput == that.useCompactArgmaxOutput &&
                blendMode == that.blendMode &&
                outputType == that.outputType &&
                objectType == that.objectType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(tileSize, overlap, overlapPercent, blendMode, outputType, objectType,
                minObjectSizeMicrons, holeFillingMicrons, boundarySmoothing,
                maxTilesInMemory, useGPU, useTTA, multiPassAveraging, overlaySmoothingSigma,
                useCompactArgmaxOutput);
    }

    @Override
    public String toString() {
        return String.format("InferenceConfig{tile=%d, overlap=%d (%.1f%%), output=%s, objectType=%s, blend=%s, tta=%b%s%s}",
                tileSize, overlap, overlapPercent, outputType, objectType, blendMode, useTTA,
                multiPassAveraging ? ", multiPass" : "",
                useCompactArgmaxOutput ? ", argmax8" : "");
    }

    /**
     * Computes the effective per-side padding for tile overlap.
     * <p>
     * Both the overlay (expanded reads + center crop) and Apply Classifier
     * (TileProcessor batch inference) use this to decide how much context
     * to read around each tile. The result is a per-side padding value;
     * total overlap between adjacent tiles is {@code 2 * effectivePadding}.
     * <p>
     * Only one invariant is enforced here: {@code stride = tileSize -
     * 2*padding} must be &gt; 0 so the tiling loop advances. That means
     * {@code padding < tileSize / 2}. The user's configured overlap is
     * otherwise respected as-is: no silent minimum, no silent clamp
     * toward a "recommended" range. Prior implementations imposed
     * absolute / proportional floors and ceilings that overrode user
     * intent and, on small tiles, drove stride to zero -- the UI is the
     * right place to advise on quality tradeoffs, not this function.
     *
     * @param tileSize      tile size in pixels
     * @param configOverlap configured overlap in pixels (from user preferences)
     * @return effective padding per side; clamped only so stride &gt; 0
     */
    public static int computeEffectivePadding(int tileSize, int configOverlap) {
        if (tileSize <= 0) return 0;
        int maxPadding = (tileSize - 1) / 2;
        return Math.min(Math.max(0, configOverlap), maxPadding);
    }

    /**
     * Checks a tileSize / overlap pair against recommended quality and
     * speed ranges, returning a short advisory string when something is
     * likely to be problematic or {@code null} when the pair is in a
     * sensible range. Does not alter the values -- callers (typically
     * the training / inference dialog) use this purely to populate a
     * warning label so the user knows the tradeoff they are making.
     * <p>
     * Current ranges (all tileSize-relative so they hold for any size):
     * <ul>
     *   <li>Stride must stay &gt; 0. If the requested overlap is &ge;
     *       tileSize/2, the effective padding will silently clamp --
     *       surface that.</li>
     *   <li>Padding &lt; 12.5% of tileSize: tile boundaries will not
     *       have enough model context; visible seams are likely near
     *       high-contrast transitions.</li>
     *   <li>Padding &gt; 37.5% of tileSize: most of each tile is
     *       discarded context; inference is much slower with little
     *       quality benefit.</li>
     *   <li>tileSize &lt; 192: any fixed-px context budget becomes a
     *       large fraction of the tile, and the tile count grows
     *       quadratically with 1/tileSize, so inference is slower.</li>
     * </ul>
     *
     * @return a human-readable advisory, or {@code null} if the pair is
     *         within recommended ranges
     */
    public static String checkTileSettings(int tileSize, int configOverlap) {
        if (tileSize <= 0) return null;
        int padding = Math.max(0, configOverlap);
        if (padding >= tileSize / 2) {
            int clamped = (tileSize - 1) / 2;
            return String.format(
                    "Overlap %dpx is >= tileSize/2 (%dpx) -- stride would be zero "
                    + "and tiling cannot advance. Value will be clamped to %dpx "
                    + "at inference. Reduce overlap to fix this properly.",
                    configOverlap, tileSize / 2, clamped);
        }
        double pct = 100.0 * padding / tileSize;
        if (tileSize < 192) {
            return String.format(
                    "Small tile size (%dpx): inference will be slower (tile count "
                    + "grows as 1/tileSize^2) and per-tile context is limited, "
                    + "which tends to hurt edge predictions. 256-512px is the "
                    + "typical working range.",
                    tileSize);
        }
        if (pct < 12.5) {
            return String.format(
                    "Low overlap (%.1f%% of tileSize): tile boundaries may show "
                    + "visible seams in the overlay, especially at high-contrast "
                    + "class transitions. 12.5-25%% (%d-%dpx for this tile size) "
                    + "is the typical working range.",
                    pct, tileSize / 8, tileSize / 4);
        }
        if (pct > 37.5) {
            return String.format(
                    "High overlap (%.1f%% of tileSize): most of each tile is "
                    + "discarded context, so inference will be significantly "
                    + "slower with little quality benefit. 12.5-25%% is the "
                    + "typical working range.",
                    pct);
        }
        return null;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for InferenceConfig.
     */
    public static class Builder {
        private int tileSize = 512;
        private int overlap = 64;
        private double overlapPercent = 12.5; // Default: 64/512 = 12.5%
        private BlendMode blendMode = BlendMode.GAUSSIAN;
        private OutputType outputType = OutputType.MEASUREMENTS;
        private OutputObjectType objectType = OutputObjectType.DETECTION;
        private double minObjectSizeMicrons = 10.0;
        private double holeFillingMicrons = 5.0;
        private double boundarySmoothing = 2.0;
        private int maxTilesInMemory = 50;
        private boolean useGPU = true;
        private boolean useTTA = false;
        private boolean multiPassAveraging = false;
        private double overlaySmoothingSigma = 2.0;
        private boolean useCompactArgmaxOutput = false;

        public Builder tileSize(int tileSize) {
            this.tileSize = tileSize;
            return this;
        }

        /**
         * Sets the overlap in pixels.
         *
         * @param overlap overlap in pixels
         * @return this builder
         */
        public Builder overlap(int overlap) {
            this.overlap = overlap;
            return this;
        }

        /**
         * Sets the overlap as a percentage of tile size (0.0 to 50.0).
         * <p>
         * This will also calculate and set the pixel overlap value.
         * Values above 25% are rarely needed but may be useful for very
         * high resolution images with fine structures.
         *
         * @param percent overlap percentage (0.0 to 50.0)
         * @return this builder
         * @throws IllegalArgumentException if percent is out of range
         */
        public Builder overlapPercent(double percent) {
            if (percent < 0.0 || percent > 50.0) {
                throw new IllegalArgumentException("Overlap percent must be between 0.0 and 50.0");
            }
            this.overlapPercent = percent;
            // Calculate pixel overlap from percentage
            this.overlap = (int) Math.round(tileSize * percent / 100.0);
            return this;
        }

        public Builder blendMode(BlendMode blendMode) {
            this.blendMode = blendMode;
            return this;
        }

        public Builder outputType(OutputType outputType) {
            this.outputType = outputType;
            return this;
        }

        /**
         * Sets the type of PathObject to create for OBJECTS output.
         * <p>
         * Only relevant when outputType is OBJECTS.
         *
         * @param objectType DETECTION or ANNOTATION
         * @return this builder
         */
        public Builder objectType(OutputObjectType objectType) {
            this.objectType = objectType;
            return this;
        }

        public Builder minObjectSizeMicrons(double minSize) {
            this.minObjectSizeMicrons = minSize;
            return this;
        }

        /**
         * Alias for minObjectSizeMicrons() for more readable code.
         */
        public Builder minObjectSize(double minSize) {
            return minObjectSizeMicrons(minSize);
        }

        public Builder holeFillingMicrons(double holeFilling) {
            this.holeFillingMicrons = holeFilling;
            return this;
        }

        /**
         * Alias for holeFillingMicrons() for more readable code.
         */
        public Builder holeFilling(double holeFilling) {
            return holeFillingMicrons(holeFilling);
        }

        public Builder boundarySmoothing(double smoothing) {
            this.boundarySmoothing = smoothing;
            return this;
        }

        /**
         * Alias for boundarySmoothing() for more readable code.
         */
        public Builder smoothing(double smoothing) {
            return boundarySmoothing(smoothing);
        }

        public Builder maxTilesInMemory(int maxTiles) {
            this.maxTilesInMemory = maxTiles;
            return this;
        }

        public Builder useGPU(boolean useGPU) {
            this.useGPU = useGPU;
            return this;
        }

        /**
         * Sets whether to use Test-Time Augmentation (TTA).
         * <p>
         * TTA runs inference with D4 transforms (flips + 90-degree rotations)
         * and averages the predictions. Improves quality by 1-3% but ~8x slower.
         *
         * @param useTTA true to enable TTA
         * @return this builder
         */
        public Builder useTTA(boolean useTTA) {
            this.useTTA = useTTA;
            return this;
        }

        /**
         * Enables multi-pass tile averaging for seamless tile boundaries.
         * <p>
         * When enabled, each tile is inferred at 4 spatial offsets (2x2 grid)
         * and the probability maps are averaged. This eliminates boundary
         * artifacts at ~4x inference cost.
         *
         * @param enabled true to enable multi-pass averaging
         * @return this builder
         */
        public Builder multiPassAveraging(boolean enabled) {
            this.multiPassAveraging = enabled;
            return this;
        }

        /**
         * Sets the Gaussian sigma for overlay probability smoothing.
         * <p>
         * A Gaussian blur is applied to probability maps before argmax to smooth
         * noisy per-pixel predictions. Set to 0 to disable smoothing.
         *
         * @param sigma Gaussian sigma in pixels (0 = no smoothing, 2.0 = recommended)
         * @return this builder
         */
        public Builder overlaySmoothingSigma(double sigma) {
            this.overlaySmoothingSigma = Math.max(0.0, sigma);
            return this;
        }

        /**
         * Enables the compact uint8 argmax output path (Phase 3c). When on,
         * the Python side returns class indices directly; smoothing,
         * multi-pass averaging, and tile blending become no-ops.
         */
        public Builder useCompactArgmaxOutput(boolean enabled) {
            this.useCompactArgmaxOutput = enabled;
            return this;
        }

        public InferenceConfig build() {
            if (tileSize < 64 || tileSize > 8192) {
                throw new IllegalStateException("Tile size must be between 64 and 8192");
            }
            if (overlap < 0 || overlap > tileSize / 2) {
                throw new IllegalStateException("Overlap must be between 0 and half of tile size");
            }
            if (overlapPercent < 0.0 || overlapPercent > 50.0) {
                throw new IllegalStateException("Overlap percent must be between 0.0 and 50.0");
            }
            if (maxTilesInMemory < 1) {
                throw new IllegalStateException("Max tiles in memory must be at least 1");
            }
            return new InferenceConfig(this);
        }
    }
}
