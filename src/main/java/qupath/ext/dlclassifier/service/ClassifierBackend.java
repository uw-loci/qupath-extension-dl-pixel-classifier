package qupath.ext.dlclassifier.service;

import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Backend interface for the DL classification service.
 * <p>
 * Implemented by {@link ApposeClassifierBackend} which uses Appose IPC
 * with shared memory for embedded Python execution.
 * <p>
 * Use {@link BackendFactory#getBackend()} to obtain the backend.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public interface ClassifierBackend {

    // ==================== Health & Status ====================

    /**
     * Checks if the backend is available and ready for requests.
     *
     * @return true if the backend can accept requests
     */
    boolean checkHealth();

    /**
     * Gets GPU availability information.
     *
     * @return GPU info string (name, memory), or "Unknown" if unavailable
     */
    String getGPUInfo();

    /**
     * Forces all GPU memory to be cleared.
     *
     * @return summary of what was freed, or null on failure
     */
    String clearGPUMemory();

    // ==================== Training ====================

    /**
     * Starts a training job.
     *
     * @param trainingConfig   training configuration
     * @param channelConfig    channel configuration
     * @param classNames       list of class names
     * @param trainingDataPath path to exported training data
     * @param progressCallback callback for progress updates
     * @param cancelledCheck   supplier that returns true when cancelled
     * @param jobIdCallback    optional callback to receive the job ID once assigned
     * @return training result containing model path and metrics
     * @throws IOException if communication fails
     */
    ClassifierClient.TrainingResult startTraining(
            TrainingConfig trainingConfig,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path trainingDataPath,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback) throws IOException;

    /**
     * Pauses a running training job at the end of the current epoch.
     *
     * @param jobId the job ID to pause
     * @throws IOException if communication fails
     */
    void pauseTraining(String jobId) throws IOException;

    /**
     * Resumes a paused training job.
     *
     * @param jobId            the job ID to resume
     * @param newDataPath      optional new data path
     * @param epochs           optional new total epochs
     * @param learningRate     optional new learning rate
     * @param batchSize        optional new batch size
     * @param progressCallback callback for progress updates
     * @param cancelledCheck   supplier that returns true when cancelled
     * @return training result
     * @throws IOException if communication fails
     */
    ClassifierClient.TrainingResult resumeTraining(
            String jobId,
            Path newDataPath,
            Integer epochs,
            Double learningRate,
            Integer batchSize,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException;

    /**
     * Finalizes training from a saved checkpoint by restoring the best model
     * weights and saving them as the final classifier.
     *
     * @param checkpointPath path to the training checkpoint file
     * @return training result with the saved model path and best metrics
     * @throws IOException if finalization fails
     */
    ClassifierClient.TrainingResult finalizeTraining(String checkpointPath) throws IOException;

    /**
     * Finalizes training from a saved checkpoint, saving model files directly
     * to the specified project directory.
     *
     * @param checkpointPath path to the training checkpoint file
     * @param modelOutputDir project-local directory for model output, or null for default
     * @return training result with the saved model path and best metrics
     * @throws IOException if finalization fails
     */
    default ClassifierClient.TrainingResult finalizeTraining(String checkpointPath,
            String modelOutputDir) throws IOException {
        return finalizeTraining(checkpointPath);
    }

    // ==================== Inference ====================

    /**
     * Runs pixel-level inference using binary tile transfer.
     * <p>
     * Returns {@code null} if the binary endpoint is not available (HTTP backend
     * returns 404). Callers should fall back to {@link #runPixelInference} in that case.
     *
     * @param modelPath         path to the trained model
     * @param rawTileBytes      concatenated tile pixels (HWC order)
     * @param tileIds           ordered list of tile IDs
     * @param tileHeight        tile height in pixels
     * @param tileWidth         tile width in pixels
     * @param numChannels       number of channels
     * @param dtype             data type ("uint8" or "float32")
     * @param channelConfig     channel configuration
     * @param inferenceConfig   inference configuration
     * @param outputDir         directory for probability map files
     * @param reflectionPadding reflection padding pixels
     * @return pixel inference result, or null if binary path unavailable
     * @throws IOException if communication fails
     */
    ClassifierClient.PixelInferenceResult runPixelInferenceBinary(
            String modelPath,
            byte[] rawTileBytes,
            List<String> tileIds,
            int tileHeight,
            int tileWidth,
            int numChannels,
            String dtype,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            Path outputDir,
            int reflectionPadding) throws IOException;

    /**
     * Runs pixel-level inference using base64/file-path tile transfer.
     *
     * @param modelPath       path to the trained model
     * @param tiles           list of tile data
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @param outputDir       directory for probability map files
     * @param reflectionPadding reflection padding pixels
     * @return pixel inference result
     * @throws IOException if communication fails
     */
    ClassifierClient.PixelInferenceResult runPixelInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            Path outputDir,
            int reflectionPadding) throws IOException;

    /**
     * Runs batch inference using binary tile transfer.
     *
     * @param modelPath       path to the trained model
     * @param rawTileBytes    concatenated tile pixels
     * @param tileIds         tile IDs
     * @param tileHeight      tile height
     * @param tileWidth       tile width
     * @param numChannels     number of channels
     * @param dtype           data type
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @return inference result, or null if binary path unavailable
     * @throws IOException if communication fails
     */
    ClassifierClient.InferenceResult runInferenceBinary(
            String modelPath,
            byte[] rawTileBytes,
            List<String> tileIds,
            int tileHeight,
            int tileWidth,
            int numChannels,
            String dtype,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig) throws IOException;

    /**
     * Runs batch inference using base64/file-path tile transfer.
     *
     * @param modelPath       path to the trained model
     * @param tiles           list of tile data
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @return inference result
     * @throws IOException if communication fails
     */
    ClassifierClient.InferenceResult runInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig) throws IOException;

    // ==================== Evaluation ====================

    /**
     * Evaluates all training tiles against the trained model to identify
     * annotation errors and hard cases.
     *
     * @param modelPath        path to the trained model directory
     * @param trainingDataPath path to training data directory (containing tile_manifest.json)
     * @param architecture     model architecture name (e.g. "unet")
     * @param backbone         encoder backbone name (e.g. "resnet34")
     * @param inputConfig      channel configuration as map
     * @param classNames       list of class names
     * @param classColors      map of class name to packed RGB color, or null
     * @param progressCallback callback for progress updates
     * @param cancelledCheck   supplier that returns true when cancelled
     * @return list of per-tile evaluation results sorted by loss descending
     * @throws IOException if evaluation fails
     */
    List<ClassifierClient.TileEvaluationResult> evaluateTiles(
            Path modelPath,
            Path trainingDataPath,
            String architecture,
            String backbone,
            Map<String, Object> inputConfig,
            List<String> classNames,
            Map<String, Integer> classColors,
            Consumer<ClassifierClient.EvaluationProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException;

    // ==================== Pretrained Model Info ====================

    /**
     * Gets the layer structure of a model for freeze/unfreeze configuration.
     *
     * @param architecture model architecture
     * @param encoder      encoder name
     * @param numChannels  number of input channels
     * @param numClasses   number of output classes
     * @return list of layer info objects
     * @throws IOException if communication fails
     */
    List<ClassifierClient.LayerInfo> getModelLayers(
            String architecture, String encoder,
            int numChannels, int numClasses) throws IOException;

    /**
     * Gets recommended freeze settings for a dataset size.
     *
     * @param datasetSize "small", "medium", or "large"
     * @param encoder     optional encoder name
     * @return map of depth to freeze recommendation
     * @throws IOException if communication fails
     */
    Map<Integer, Boolean> getFreezeRecommendations(
            String datasetSize, String encoder) throws IOException;
}
