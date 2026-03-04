package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

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
     * Reads a probability map from a raw binary float32 file.
     *
     * @param filePath   path to the binary file
     * @param numClasses number of classes (C dimension)
     * @param height     tile height (H dimension)
     * @param width      tile width (W dimension)
     * @return probability map with shape [height][width][numClasses] (HWC order for TileProcessor)
     * @throws IOException if reading fails
     */
    public static float[][][] readProbabilityMap(Path filePath, int numClasses,
                                                  int height, int width) throws IOException {
        byte[] bytes = java.nio.file.Files.readAllBytes(filePath);

        // Validate file size matches expected dimensions
        long expectedSize = (long) numClasses * height * width * Float.BYTES;
        if (bytes.length != expectedSize) {
            throw new IOException(String.format(
                    "Probability map size mismatch for %s: expected %d bytes (C=%d, H=%d, W=%d) but got %d bytes",
                    filePath.getFileName(), expectedSize, numClasses, height, width, bytes.length));
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
            sb.append(filePath.getFileName()).append(": ");
            for (int c = 0; c < numClasses; c++) {
                double mean = classSum[c] / totalPixels;
                sb.append(String.format("C%d[min=%.3f, max=%.3f, mean=%.3f] ", c, classMin[c], classMax[c], mean));
            }
            logger.debug(sb.toString());
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
            Map<String, String> configSummary
    ) {
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
            String checkpointPath
    ) {
        /** Compact constructor for non-paused results. */
        public TrainingResult(String jobId, String modelPath, double finalLoss, double finalAccuracy,
                              int bestEpoch, double bestMeanIoU) {
            this(jobId, modelPath, finalLoss, finalAccuracy, bestEpoch, bestMeanIoU, false, 0, 0, null);
        }

        /** Compact constructor for cancelled results. */
        public TrainingResult(String jobId, String modelPath, double finalLoss, double finalAccuracy) {
            this(jobId, modelPath, finalLoss, finalAccuracy, 0, 0.0, false, 0, 0, null);
        }

        /** Returns true if training was cancelled (no model produced and not paused). */
        public boolean isCancelled() {
            return modelPath == null && !paused;
        }

        /** Returns true if training was paused. */
        public boolean isPaused() {
            return paused;
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
    public record PixelInferenceResult(Map<String, String> outputPaths, int numClasses) {}

    /**
     * Model information.
     */
    public record ModelInfo(String id, String name, String type, String path) {}

    /**
     * Pretrained encoder information.
     */
    public record EncoderInfo(
            String name,
            String displayName,
            String family,
            double paramsMillion,
            String license
    ) {}

    /**
     * Segmentation architecture information.
     */
    public record ArchitectureInfo(
            String name,
            String displayName,
            String description
    ) {}

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
            String description
    ) {}

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
            String tileImagePath
    ) {}

    /**
     * Progress update during tile evaluation.
     */
    public record EvaluationProgress(
            int currentTile,
            int totalTiles,
            String message
    ) {}
}
