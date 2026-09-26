package qupath.ext.dlclassifier.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Data types shared between the {@link ClassifierBackend} interface
 * and its Appose implementation.
 * <p>
 * Also contains utility methods for reading probability map files
 * produced by inference.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ClassifierClient {

    private static final Logger logger = LoggerFactory.getLogger(ClassifierClient.class);

    private ClassifierClient() {
        // Data holder class - no instantiation
    }

    /**
     * Reads a probability map from raw binary float32 data.
     *
     * @param output     inference output for one tile, in memory or on disk
     * @param numClasses number of classes (C dimension)
     * @param height     tile height (H dimension)
     * @param width      tile width (W dimension)
     * @return probability map with shape [height][width][numClasses] (HWC order for TileProcessor)
     * @throws IOException if reading fails
     */
    public static float[][][] readProbabilityMap(TileOutput output, int numClasses, int height, int width)
            throws IOException {
        byte[] bytes = output.bytes();

        // Validate payload size matches expected dimensions
        long expectedSize = (long) numClasses * height * width * Float.BYTES;
        if (bytes.length != expectedSize) {
            throw new IOException(String.format(
                    "Probability map size mismatch for %s: expected %d bytes (C=%d, H=%d, W=%d) but got %d bytes",
                    output.tileId(), expectedSize, numClasses, height, width, bytes.length));
        }

        java.nio.FloatBuffer buffer = java.nio.ByteBuffer.wrap(bytes)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();

        // Data is in CHW order from Python, convert to HWC for TileProcessor
        float[][][] result = new float[height][width][numClasses];
        float[] classMin = new float[numClasses];
        float[] classMax = new float[numClasses];
        double[] classSum = new double[numClasses];
        java.util.Arrays.fill(classMin, Float.MAX_VALUE);
        java.util.Arrays.fill(classMax, -Float.MAX_VALUE);

        for (int c = 0; c < numClasses; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float val = buffer.get(c * height * width + h * width + w);
                    result[h][w][c] = val;
                    if (val < classMin[c]) classMin[c] = val;
                    if (val > classMax[c]) classMax[c] = val;
                    classSum[c] += val;
                }
            }
        }

        // Log probability distribution diagnostics
        if (logger.isDebugEnabled()) {
            int totalPixels = height * width;
            StringBuilder sb = new StringBuilder("Probability map stats for ");
            sb.append(output.tileId()).append(": ");
            for (int c = 0; c < numClasses; c++) {
                double mean = classSum[c] / totalPixels;
                sb.append(String.format("C%d[min=%.3f, max=%.3f, mean=%.3f] ", c, classMin[c], classMax[c], mean));
            }
            logger.debug(sb.toString());
        }

        return result;
    }

    /**
     * Reads a compact uint8 argmax map from a raw binary file.
     * <p>
     * This is the Phase 3c fast path: Python returns class indices directly
     * instead of float32 probability maps. Used only when
     * {@code InferenceConfig.isUseCompactArgmaxOutput()} is true.
     *
     * @param output inference output for one tile (H*W bytes, uint8)
     * @param height tile height
     * @param width  tile width
     * @return class-index map with shape [height][width]
     * @throws IOException if reading fails or size mismatches
     */
    public static byte[][] readArgmaxMap(TileOutput output, int height, int width) throws IOException {
        byte[] bytes = output.bytes();
        long expectedSize = (long) height * width;
        if (bytes.length != expectedSize) {
            throw new IOException(String.format(
                    "Argmax map size mismatch for %s: expected %d bytes (H=%d, W=%d) but got %d bytes",
                    output.tileId(), expectedSize, height, width, bytes.length));
        }
        byte[][] result = new byte[height][width];
        for (int y = 0; y < height; y++) {
            System.arraycopy(bytes, y * width, result[y], 0, width);
        }
        return result;
    }

    // ==================== Data Classes ====================

    /**
     * Training progress information.
     */
    public record TrainingProgress(
            int epoch,
            int totalEpochs,
            double loss,
            double valLoss,
            double accuracy,
            double meanIoU,
            Map<String, Double> perClassIoU,
            Map<String, Double> perClassLoss,
            String device,
            String deviceInfo,
            String status,
            String setupPhase,
            Map<String, String> configSummary) {
        public double getProgress() {
            return (double) epoch / totalEpochs;
        }

        /**
         * Whether this is a setup phase update (not an epoch update).
         */
        public boolean isSetupPhase() {
            return "setup".equals(status) || "initializing".equals(status);
        }
    }

    /**
     * Training result information.
     * <p>
     * {@code finalLoss} and {@code finalAccuracy} reflect the <b>best</b> model
     * (the checkpoint that was actually saved), not the last training epoch.
     */
    public record TrainingResult(
            String jobId,
            String modelPath,
            double finalLoss,
            double finalAccuracy,
            int bestEpoch,
            double bestMeanIoU,
            boolean paused,
            int lastEpoch,
            int totalEpochs,
            String checkpointPath,
            boolean cancelled,
            String lastModelPath,
            String focusClassName,
            double focusClassIoU,
            boolean focusClassTargetMet,
            String quality,
            List<String> warnings) {
        /**
         * Canonical constructor - normalizes quality/warnings to safe defaults.
         */
        public TrainingResult {
            if (quality == null) {
                quality = "ok";
            }
            warnings = warnings == null ? Collections.emptyList() : List.copyOf(warnings);
        }

        /** Compact constructor for non-paused results. */
        public TrainingResult(
                String jobId,
                String modelPath,
                double finalLoss,
                double finalAccuracy,
                int bestEpoch,
                double bestMeanIoU) {
            this(
                    jobId,
                    modelPath,
                    finalLoss,
                    finalAccuracy,
                    bestEpoch,
                    bestMeanIoU,
                    false,
                    0,
                    0,
                    null,
                    false,
                    null,
                    null,
                    0.0,
                    true,
                    "ok",
                    Collections.emptyList());
        }

        /** Compact constructor for cancelled results (no save). */
        public TrainingResult(String jobId, String modelPath, double finalLoss, double finalAccuracy) {
            this(
                    jobId,
                    modelPath,
                    finalLoss,
                    finalAccuracy,
                    0,
                    0.0,
                    false,
                    0,
                    0,
                    null,
                    true,
                    null,
                    null,
                    0.0,
                    true,
                    "ok",
                    Collections.emptyList());
        }

        /**
         * Legacy 15-arg compact constructor (pre-quality/warnings).
         * Defaults {@code quality="ok"} and an empty warnings list so existing
         * callers in TrainingWorkflow / ApposeClassifierBackend continue to work
         * without churn. New callers (SSL pretraining) should use the full
         * 17-arg form.
         */
        public TrainingResult(
                String jobId,
                String modelPath,
                double finalLoss,
                double finalAccuracy,
                int bestEpoch,
                double bestMeanIoU,
                boolean paused,
                int lastEpoch,
                int totalEpochs,
                String checkpointPath,
                boolean cancelled,
                String lastModelPath,
                String focusClassName,
                double focusClassIoU,
                boolean focusClassTargetMet) {
            this(
                    jobId,
                    modelPath,
                    finalLoss,
                    finalAccuracy,
                    bestEpoch,
                    bestMeanIoU,
                    paused,
                    lastEpoch,
                    totalEpochs,
                    checkpointPath,
                    cancelled,
                    lastModelPath,
                    focusClassName,
                    focusClassIoU,
                    focusClassTargetMet,
                    "ok",
                    Collections.emptyList());
        }

        /** Returns true if training was cancelled. */
        public boolean isCancelled() {
            return cancelled || (modelPath == null && !paused);
        }

        /** Returns true if training was cancelled and a model was saved. */
        public boolean isCancelledWithSave() {
            return cancelled && modelPath != null;
        }

        /** Returns true if training was paused. */
        public boolean isPaused() {
            return paused;
        }

        /** Returns true if the run produced quality warnings (likely_collapse, warn, etc). */
        public boolean hasQualityWarnings() {
            return !"ok".equals(quality) || (warnings != null && !warnings.isEmpty());
        }
    }

    /**
     * Tile data for inference.
     */
    public record TileData(String id, String data, int x, int y) {}

    /**
     * Inference result containing predictions for each tile.
     */
    public record InferenceResult(Map<String, float[]> predictions) {}

    /**
     * Pixel-level inference result with file paths to probability maps.
     */
    public record PixelInferenceResult(Map<String, TileOutput> outputs, int numClasses) {}

    /**
     * Raw inference output for a single tile.
     *
     * <p>Exactly one of {@code data} and {@code path} is non-null. The
     * single-tile overlay path copies the result straight out of Appose
     * shared memory and keeps it in {@code data}; the multi-tile batch path
     * has Python write one file per tile and reports {@code path}. Callers
     * read either through {@link #bytes()} and release either through
     * {@link #discard()}, so neither has to know which kind it holds.
     *
     * <p>Keeping the overlay payload in memory is deliberate. Writing it to
     * disk only to read it back on the next line cost a full write plus read
     * of the probability map on every tile of an interactive overlay, which
     * is the dominant per-tile cost once the model is small.
     *
     * @param tileId tile identifier, used in size-mismatch messages
     * @param path   file holding the payload, or null when it is in memory
     * @param data   payload bytes, or null when the payload is on disk
     */
    public record TileOutput(String tileId, Path path, byte[] data) {

        public TileOutput {
            if ((path == null) == (data == null)) {
                throw new IllegalArgumentException(
                        "TileOutput requires exactly one of path or data (tileId=" + tileId + ")");
            }
        }

        /** Output already held in memory, with no file backing it. */
        public static TileOutput ofBytes(String tileId, byte[] data) {
            return new TileOutput(tileId, null, data);
        }

        /** Output written to a file by the Python side. */
        public static TileOutput ofPath(String tileId, Path path) {
            return new TileOutput(tileId, path, null);
        }

        /**
         * Returns the payload, reading the backing file when there is one.
         *
         * @return the raw bytes
         * @throws IOException if the backing file cannot be read
         */
        public byte[] bytes() throws IOException {
            return data != null ? data : Files.readAllBytes(path);
        }

        /**
         * Releases the output. Deletes the backing file when there is one;
         * does nothing for an in-memory payload. Safe to call more than once.
         */
        public void discard() {
            if (path == null) {
                return;
            }
            try {
                Files.deleteIfExists(path);
            } catch (IOException e) {
                logger.debug("Failed to delete tile output {}: {}", path, e.getMessage());
            }
        }
    }

    /**
     * Model information.
     */
    public record ModelInfo(String id, String name, String type, String path) {}

    /**
     * Pretrained encoder information.
     */
    public record EncoderInfo(String name, String displayName, String family, double paramsMillion, String license) {}

    /**
     * Segmentation architecture information.
     */
    public record ArchitectureInfo(String name, String displayName, String description) {}

    /**
     * Model layer information for freeze/unfreeze configuration.
     */
    public record LayerInfo(
            String name,
            String displayName,
            int paramCount,
            boolean isEncoder,
            int depth,
            boolean recommendedFreeze,
            String description) {}

    /**
     * One GT-class -> Predicted-class confusion bucket for a tile.
     * {@code pixels} is the number of labeled pixels where the model predicted
     * {@code pred} but the ground truth was {@code gt}; {@code gtTotal} is the
     * total count of {@code gt} pixels in that tile (so the dialog can show
     * "k% of gt-class was misread as pred-class").
     */
    public record ConfusionPair(String gt, String pred, long pixels, long gtTotal) {}

    /**
     * Per-tile evaluation result from post-training analysis.
     * Tiles with higher loss are more likely to represent annotation errors
     * or hard cases.
     */
    public record TileEvaluationResult(
            String filename,
            String split,
            double loss,
            double disagreementPct,
            Map<String, Double> perClassIoU,
            double meanIoU,
            int x,
            int y,
            String sourceImage,
            String sourceImageId,
            String disagreementImagePath,
            String lossHeatmapPath,
            String tileImagePath,
            String predictionMapPath,
            String confidenceMapPath,
            String groundTruthMaskPath,
            List<ConfusionPair> topConfusions,
            long disagreementPixels,
            List<Integer> disagreementConfHistogram,
            Map<String, Long> gtPixelTotals) {
        /**
         * Normalize {@code topConfusions}, {@code disagreementConfHistogram}
         * and {@code gtPixelTotals} to empty collections (never null) so
         * downstream code never has to null-check. The histogram has 20 fixed
         * bins of width 0.05 covering [0.0, 1.0]; bin i = count of disagree
         * pixels with confidence in [i*0.05, (i+1)*0.05) (bin 19 includes
         * confidence=1.0). When absent (legacy session, pre-feature), it's
         * empty -- callers should fall back to {@code disagreementPixels} (raw
         * total) or hide the column.
         * <p>
         * {@code gtPixelTotals} maps class name -> labeled ground-truth pixel
         * count in this tile, for every class present, whether or not it was
         * confused with anything. It is the denominator the session-wide
         * confusion matrix needs: {@link ConfusionPair#gtTotal()} only reaches
         * classes that were actually misread somewhere in the tile, so summing
         * those alone omits every perfectly-predicted tile and inflates each
         * percentage. Empty for legacy sessions saved before this field existed.
         */
        public TileEvaluationResult {
            topConfusions = topConfusions == null ? List.of() : List.copyOf(topConfusions);
            disagreementConfHistogram =
                    disagreementConfHistogram == null ? List.of() : List.copyOf(disagreementConfHistogram);
            gtPixelTotals = gtPixelTotals == null ? Map.of() : Map.copyOf(gtPixelTotals);
        }

        /**
         * Back-compat constructor for callers that haven't yet started
         * computing the disagreement-pixel count and confidence histogram.
         */
        public TileEvaluationResult(
                String filename,
                String split,
                double loss,
                double disagreementPct,
                Map<String, Double> perClassIoU,
                double meanIoU,
                int x,
                int y,
                String sourceImage,
                String sourceImageId,
                String disagreementImagePath,
                String lossHeatmapPath,
                String tileImagePath,
                String predictionMapPath,
                String confidenceMapPath,
                String groundTruthMaskPath,
                List<ConfusionPair> topConfusions) {
            this(
                    filename,
                    split,
                    loss,
                    disagreementPct,
                    perClassIoU,
                    meanIoU,
                    x,
                    y,
                    sourceImage,
                    sourceImageId,
                    disagreementImagePath,
                    lossHeatmapPath,
                    tileImagePath,
                    predictionMapPath,
                    confidenceMapPath,
                    groundTruthMaskPath,
                    topConfusions,
                    0L,
                    List.of(),
                    Map.of());
        }

        /**
         * Backward-compatible constructor for results without prediction/confidence/gt maps
         * (e.g. sessions saved before annotation adjustment was added).
         */
        public TileEvaluationResult(
                String filename,
                String split,
                double loss,
                double disagreementPct,
                Map<String, Double> perClassIoU,
                double meanIoU,
                int x,
                int y,
                String sourceImage,
                String sourceImageId,
                String disagreementImagePath,
                String lossHeatmapPath,
                String tileImagePath) {
            this(
                    filename,
                    split,
                    loss,
                    disagreementPct,
                    perClassIoU,
                    meanIoU,
                    x,
                    y,
                    sourceImage,
                    sourceImageId,
                    disagreementImagePath,
                    lossHeatmapPath,
                    tileImagePath,
                    null,
                    null,
                    null,
                    List.of());
        }

        /**
         * Backward-compatible constructor for sessions saved before
         * top-confusion pairs were added (defaults to empty list).
         */
        public TileEvaluationResult(
                String filename,
                String split,
                double loss,
                double disagreementPct,
                Map<String, Double> perClassIoU,
                double meanIoU,
                int x,
                int y,
                String sourceImage,
                String sourceImageId,
                String disagreementImagePath,
                String lossHeatmapPath,
                String tileImagePath,
                String predictionMapPath,
                String confidenceMapPath,
                String groundTruthMaskPath) {
            this(
                    filename,
                    split,
                    loss,
                    disagreementPct,
                    perClassIoU,
                    meanIoU,
                    x,
                    y,
                    sourceImage,
                    sourceImageId,
                    disagreementImagePath,
                    lossHeatmapPath,
                    tileImagePath,
                    predictionMapPath,
                    confidenceMapPath,
                    groundTruthMaskPath,
                    List.of());
        }
    }

    /**
     * Progress update during tile evaluation.
     */
    public record EvaluationProgress(int currentTile, int totalTiles, String message) {}
}
