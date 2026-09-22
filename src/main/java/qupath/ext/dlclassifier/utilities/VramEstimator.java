package qupath.ext.dlclassifier.utilities;

import qupath.ext.dlclassifier.model.InferenceConfig;

/**
 * Predicts peak GPU memory for a training configuration.
 * <p>
 * This lives outside the training dialog so it can be tested. The estimate is
 * the number a user acts on before committing to a long run, and it has been
 * wrong in both directions:
 * <ul>
 *   <li>It measured the UNPADDED tile. The exporter writes
 *       {@code tileSize + 2 * contextPadding} pixels per side, so a 256 px tile
 *       at 20% overlap is really 358 px -- 1.96x the area, and none of it was
 *       counted.</li>
 *   <li>Its tiny-unet factor was calibrated at a different tile size and
 *       carried over unchanged.</li>
 * </ul>
 * Together those reported 18,018 MB (73%, green) for a configuration that
 * actually demanded ~32,000 MB on a 24,575 MB card, which then spilled silently
 * into shared memory on Windows instead of failing.
 *
 * <h2>Calibration</h2>
 * The tiny-unet factor is MEASURED, not derived. Two points from an RTX 3090
 * (tiny-unet small-24x4, base 24, 358 px padded tile, BF16, context scale 1):
 * <table border="1">
 *   <caption>Measured peaks</caption>
 *   <tr><th>batch</th><th>observed</th><th>predicted here</th></tr>
 *   <tr><td>40</td><td>21,437 MiB (nvidia-smi, steady state)</td><td>20,4xx MiB</td></tr>
 *   <tr><td>60</td><td>~32,057 MiB (24,274 dedicated + 7.6 GB spilled)</td><td>32,0xx MiB</td></tr>
 * </table>
 * The factor targets what nvidia-smi reports for the process -- reserved blocks
 * plus the CUDA context -- because that is what decides whether a run fits,
 * not {@code torch.cuda.max_memory_allocated()}, which is lower.
 *
 * <h2>Keeping it honest</h2>
 * A formula cannot be guaranteed against future changes: add a norm layer,
 * change the decoder, or turn on a new augmentation and the constant moves. Two
 * things keep it from drifting silently. {@code VramEstimatorTest} pins the
 * measured points, so a change that shifts memory fails a test rather than
 * mispredicting; and every training run logs predicted versus actual peak, so
 * drift shows up in the log of the very next run even when nobody runs tests.
 * When those disagree, re-measure and update the factor here -- do not adjust
 * it to taste.
 *
 * @author UW-LOCI
 * @since 0.9.0
 */
public final class VramEstimator {

    /**
     * Peak bytes per (batch element x base channel x pixel x element byte) for
     * tiny-unet. Large because BatchRenorm stores several full-resolution
     * tensors per layer and autograd keeps every one for the backward pass.
     * Measured; see the class javadoc for the two points behind it.
     */
    static final double TINY_UNET_ACTIVATION_FACTOR = 91.0;

    /** Multiplier applied when a second (context) input doubles the channels. */
    static final double CONTEXT_SCALE_MULTIPLIER = 1.1;

    private VramEstimator() {}

    /** How a configuration relates to the card it will run on. */
    public enum Fit {
        /** Comfortable. */
        OK,
        /** Fits, but with little headroom for augmentation spikes or validation. */
        TIGHT,
        /** Does not fit. On Windows this does not raise OOM; it silently spills
         *  to shared memory over PCIe and the run merely becomes very slow. */
        EXCEEDS
    }

    /** Fraction of the card above which a run is called tight rather than OK. */
    static final double TIGHT_FRACTION = 0.85;

    /**
     * Classifies an estimate against a card.
     * <p>
     * EXCEEDS is reserved for estimates above the card's real capacity. An
     * earlier version flagged anything over 85% as "EXCEEDS GPU!", which would
     * have condemned a measured-good configuration: batch 40 here peaks at
     * 21,437 MiB of 24,575, i.e. 87%, and runs fine.
     *
     * @param estimatedMb predicted peak, MiB
     * @param gpuTotalMb  card capacity, MiB
     * @return the classification
     */
    public static Fit classify(double estimatedMb, double gpuTotalMb) {
        if (gpuTotalMb <= 0) return Fit.OK;
        if (estimatedMb > gpuTotalMb) return Fit.EXCEEDS;
        if (estimatedMb > TIGHT_FRACTION * gpuTotalMb) return Fit.TIGHT;
        return Fit.OK;
    }

    /**
     * Result of an estimate.
     *
     * @param totalMb    predicted peak, MiB
     * @param calibrated whether the number rests on a real measurement for this
     *                   architecture. False means the shape of the formula is a
     *                   guess and the caller should say so.
     */
    public record Estimate(double totalMb, boolean calibrated) {}

    /**
     * The tile size the model actually receives.
     * <p>
     * The exporter surrounds every training tile with real image data so the
     * geometry matches inference, which is why a 256 px tile becomes 358 px at
     * 20% overlap. Whole-image mode has no surrounding data and so no padding.
     *
     * @param tileSize       configured tile size in pixels
     * @param overlapPercent tile overlap, 0-50, as the dialog's spinner reports it
     * @param wholeImage     whole-image training mode
     * @return the padded tile size in pixels
     */
    public static int paddedTileSize(int tileSize, int overlapPixels, boolean wholeImage) {
        if (wholeImage || tileSize <= 0) {
            return Math.max(0, tileSize);
        }
        // Mirrors TrainingWorkflow.computeTrainingContextPadding exactly.
        return tileSize + 2 * InferenceConfig.computeEffectivePadding(tileSize, Math.max(0, overlapPixels));
    }

    /**
     * Same, for callers holding the dialog's overlap PERCENTAGE rather than
     * pixels. Separated by name because silently mixing the two units is the
     * kind of mistake this class exists to stop.
     *
     * @param tileSize       configured tile size in pixels
     * @param overlapPercent tile overlap, 0-50
     * @param wholeImage     whole-image training mode
     * @return the padded tile size in pixels
     */
    public static int paddedTileSizeFromPercent(int tileSize, int overlapPercent, boolean wholeImage) {
        int overlapPx = (int) Math.round(tileSize * Math.max(0, overlapPercent) / 100.0);
        return paddedTileSize(tileSize, overlapPx, wholeImage);
    }

    /**
     * Predicts peak GPU memory for a training configuration.
     *
     * @param modelType      architecture id ("tiny-unet", "muvit", "unet", ...)
     * @param backbone       encoder / preset name
     * @param paddedTile     tile size the model receives, from
     *                       {@link #paddedTileSize}
     * @param batchSize      images per batch
     * @param mixedPrecision whether AMP is on
     * @param contextScale   context scale; above 1 doubles the input channels
     * @return the estimate
     */
    public static Estimate estimateMb(
            String modelType,
            String backbone,
            int paddedTile,
            int batchSize,
            boolean mixedPrecision,
            int contextScale) {

        double modelMb = modelSizeMb(modelType, backbone);
        double bytesPerElem = mixedPrecision ? 2.0 : 4.0;
        double contextFactor = contextScale > 1 ? CONTEXT_SCALE_MULTIPLIER : 1.0;

        if ("tiny-unet".equals(modelType)) {
            // Activations scale with feature-map volume, not parameter count:
            // batch x base channels x spatial area. A 305k-parameter model can
            // still need 20 GB at batch 40.
            int base = tinyUnetBase(backbone);
            double actBytes = (double) batchSize
                    * base
                    * paddedTile
                    * paddedTile
                    * bytesPerElem
                    * TINY_UNET_ACTIVATION_FACTOR
                    * contextFactor;
            // Weights + gradients + Adam's two moments, and small either way.
            double totalMb = actBytes / (1024.0 * 1024.0) + 5.0 * modelMb;
            return new Estimate(totalMb, true);
        }

        // Pretrained encoders: activations are still driven by feature maps,
        // but no measurement exists for these yet, so the older
        // parameter-proportional shape is kept and reported as uncalibrated.
        // The per-run "predicted vs actual" log line is what will supply the
        // first real data point; recalibrate here when it does.
        double actMultiplier = "muvit".equals(modelType) ? 10.0 : 4.0;
        if (mixedPrecision) actMultiplier *= 0.6;
        actMultiplier *= contextFactor;
        double areaScale = (double) paddedTile * paddedTile / (256.0 * 256.0);
        double totalMb = modelMb * (1 + 3 + actMultiplier * areaScale * batchSize);
        return new Estimate(totalMb, false);
    }

    /**
     * Largest batch size whose estimate stays within a budget.
     *
     * @param budgetMb       the ceiling, MiB
     * @param modelType      architecture id
     * @param backbone       encoder / preset name
     * @param paddedTile     padded tile size
     * @param mixedPrecision whether AMP is on
     * @param contextScale   context scale
     * @param startBatch     largest batch to consider
     * @return the largest batch that fits, or 0 if even a batch of 1 does not
     */
    public static int maxBatchWithin(
            double budgetMb,
            String modelType,
            String backbone,
            int paddedTile,
            boolean mixedPrecision,
            int contextScale,
            int startBatch) {
        for (int b = Math.max(1, startBatch); b >= 1; b--) {
            if (estimateMb(modelType, backbone, paddedTile, b, mixedPrecision, contextScale)
                            .totalMb()
                    <= budgetMb) {
                return b;
            }
        }
        return 0;
    }

    /** Base channel count for a tiny-unet preset. */
    public static int tinyUnetBase(String backbone) {
        if (backbone == null) return 16;
        return switch (backbone) {
            case "nano-8x3" -> 8;
            case "compact-16x3" -> 16;
            case "tiny-16x4" -> 16;
            case "small-24x4" -> 24;
            default -> 16;
        };
    }

    /** Approximate parameter footprint in MiB, encoder plus decoder. */
    public static double modelSizeMb(String modelType, String backbone) {
        if ("muvit".equals(modelType)) return 140.0;
        if ("tiny-unet".equals(modelType)) {
            if (backbone == null) return 1.5;
            return switch (backbone) {
                case "nano-8x3" -> 0.4;
                case "compact-16x3" -> 0.8;
                case "tiny-16x4" -> 1.5;
                case "small-24x4" -> 3.5;
                default -> 1.5;
            };
        }
        if (backbone == null) return 30.0;
        return switch (backbone.toLowerCase()) {
            case "resnet18" -> 47.0;
            case "resnet34" -> 87.0;
            case "resnet50" -> 100.0;
            case "resnet101" -> 170.0;
            case "resnet152" -> 230.0;
            case "efficientnet-b0" -> 21.0;
            case "efficientnet-b1" -> 31.0;
            case "efficientnet-b2" -> 36.0;
            case "efficientnet-b3" -> 48.0;
            case "efficientnet-b4" -> 76.0;
            case "efficientnet-b5" -> 120.0;
            case "densenet121" -> 32.0;
            case "densenet169" -> 56.0;
            case "densenet201" -> 80.0;
            case "mobilenet_v2" -> 14.0;
            case "timm-mobilenetv3_large_100" -> 22.0;
            case "resnet50_lunit-swav", "resnet50_lunit-bt", "resnet50_kather100k", "resnet50_tcga-brca" -> 100.0;
            case "h-optimus-0", "midnight" -> 4400.0;
            case "virchow" -> 2500.0;
            case "hibou-l", "dinov2-large" -> 1200.0;
            case "hibou-b" -> 350.0;
            case "uni", "conch", "phikon" -> 350.0;
            default -> {
                if (backbone.contains("50")) yield 100.0;
                if (backbone.contains("101")) yield 170.0;
                if (backbone.contains("optimus") || backbone.contains("midnight")) yield 4400.0;
                if (backbone.contains("virchow")) yield 2500.0;
                if (backbone.contains("hibou") || backbone.contains("dinov2")) yield 1200.0;
                yield 50.0;
            }
        };
    }
}
