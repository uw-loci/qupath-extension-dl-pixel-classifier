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

    // Overlay probability smoothing
    private final double overlaySmoothingSigma;

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
        this.overlaySmoothingSigma = builder.overlaySmoothingSigma;
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
                Double.compare(that.overlaySmoothingSigma, overlaySmoothingSigma) == 0 &&
                blendMode == that.blendMode &&
                outputType == that.outputType &&
                objectType == that.objectType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(tileSize, overlap, overlapPercent, blendMode, outputType, objectType,
                minObjectSizeMicrons, holeFillingMicrons, boundarySmoothing,
                maxTilesInMemory, useGPU, useTTA, overlaySmoothingSigma);
    }

    @Override
    public String toString() {
        return String.format("InferenceConfig{tile=%d, overlap=%d (%.1f%%), output=%s, objectType=%s, blend=%s, tta=%b}",
                tileSize, overlap, overlapPercent, outputType, objectType, blendMode, useTTA);
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
        private double overlaySmoothingSigma = 2.0;

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
