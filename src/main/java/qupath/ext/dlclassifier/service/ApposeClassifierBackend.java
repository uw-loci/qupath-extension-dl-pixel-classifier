package qupath.ext.dlclassifier.service;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Service.Task;
import org.apposed.appose.TaskEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Backend implementation using Appose for embedded Python execution
 * with shared-memory tile transfer.
 * <p>
 * This backend eliminates the need for an external Python server by
 * managing a Python subprocess via Appose. Tile data is transferred
 * via shared memory (zero-copy), avoiding HTTP and file I/O overhead.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class ApposeClassifierBackend implements ClassifierBackend {

    private static final Logger logger = LoggerFactory.getLogger(ApposeClassifierBackend.class);
    private static final Gson gson = new Gson();

    // Limit concurrent Appose inference tasks. The Python side serializes GPU
    // access via inference_lock, but 16+ overlay threads submitting tasks
    // simultaneously creates 16 Python threads that mostly block and die.
    // Permits=1: serialize all overlay inference to avoid thread death.
    private static final Semaphore inferenceSemaphore = new Semaphore(1, true);

    // Retry settings -- concurrent overlay inference or stale worker state can
    // cause transient "thread death" errors when starting new tasks.
    private static final int MAX_TASK_RETRIES = 3;
    private static final long TASK_RETRY_DELAY_MS = 500;

    // Version compatibility warning from the Python package (set during health check)
    private static volatile String versionWarning;

    // Stores checkpoint info for paused training jobs so resume/finalize can
    // retrieve the checkpoint path and original training inputs.
    private static final ConcurrentHashMap<String, CheckpointInfo> checkpointStore = new ConcurrentHashMap<>();

    /**
     * Stores checkpoint state for a paused training job.
     *
     * @param path           path to the checkpoint file on disk
     * @param lastEpoch      the last completed epoch
     * @param originalInputs the original Appose task inputs (for resume)
     */
    public record CheckpointInfo(String path, int lastEpoch, Map<String, Object> originalInputs) {}

    // ==================== Health & Status ====================

    @Override
    public boolean checkHealth() {
        try {
            ApposeService appose = ApposeService.getInstance();
            if (!appose.isAvailable()) return false;

            // Retry on "thread death" -- concurrent overlay inference or stale
            // worker state can misroute errors to the health check task.
            for (int attempt = 0; attempt < MAX_TASK_RETRIES; attempt++) {
                try {
                    Task task = appose.runTask("health_check", Map.of());
                    Object healthy = task.outputs.get("healthy");

                    // Read version compatibility info from health check
                    Object warning = task.outputs.get("version_warning");
                    if (warning instanceof String w && !w.isEmpty()) {
                        versionWarning = w;
                        logger.warn("Python package version mismatch: {}", w);
                    } else {
                        versionWarning = null;
                        Object ver = task.outputs.get("server_version");
                        Object req = task.outputs.get("required_version");
                        if (ver != null) {
                            logger.info("dlclassifier-server v{} (required >= {})",
                                    ver, req != null ? req : "?");
                        }
                    }

                    return Boolean.TRUE.equals(healthy);
                } catch (Exception e) {
                    String msg = e.getMessage() != null ? e.getMessage() : "";
                    if (msg.toLowerCase().contains("thread death")
                            && attempt < MAX_TASK_RETRIES - 1) {
                        logger.debug("Health check got transient 'thread death' " +
                                "(attempt {}/{}), retrying after {}ms...",
                                attempt + 1, MAX_TASK_RETRIES,
                                TASK_RETRY_DELAY_MS);
                        Thread.sleep(TASK_RETRY_DELAY_MS);
                        continue;
                    }
                    throw e;
                }
            }
            return false;
        } catch (Exception e) {
            logger.debug("Appose health check failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public String getGPUInfo() {
        try {
            Task task = ApposeService.getInstance().runTask("health_check", Map.of());
            boolean available = Boolean.TRUE.equals(task.outputs.get("gpu_available"));
            if (available) {
                String name = String.valueOf(task.outputs.get("gpu_name"));
                int memoryMb = ((Number) task.outputs.get("gpu_memory_mb")).intValue();
                return String.format("%s (%d MB) [Appose]", name, memoryMb);
            }
            return "No GPU available (CPU mode) [Appose]";
        } catch (Exception e) {
            logger.debug("Appose GPU info failed: {}", e.getMessage());
            return "Unknown [Appose]";
        }
    }

    /**
     * Returns the version warning from the last health check, or null if versions match.
     */
    public static String getVersionWarning() {
        return versionWarning;
    }

    @Override
    public String clearGPUMemory() {
        try {
            Task task = ApposeService.getInstance().runTask("clear_gpu", Map.of());
            boolean success = Boolean.TRUE.equals(task.outputs.get("success"));
            String message = String.valueOf(task.outputs.get("message"));
            return success ? message : null;
        } catch (Exception e) {
            logger.error("Appose GPU clear failed: {}", e.getMessage());
            return null;
        }
    }

    // ==================== Training ====================

    @Override
    public ClassifierClient.TrainingResult startTraining(
            TrainingConfig trainingConfig,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path trainingDataPath,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        // Build input maps matching the train.py script expectations
        Map<String, Object> architecture = new HashMap<>();
        architecture.put("backbone", trainingConfig.getBackbone());
        architecture.put("input_size", List.of(trainingConfig.getTileSize(), trainingConfig.getTileSize()));
        architecture.put("downsample", trainingConfig.getDownsample());
        architecture.put("use_pretrained", trainingConfig.isUsePretrainedWeights());
        if (trainingConfig.getContextScale() > 1) {
            architecture.put("context_scale", trainingConfig.getContextScale());
        }
        List<String> frozenLayers = trainingConfig.getFrozenLayers();
        if (frozenLayers != null && !frozenLayers.isEmpty()) {
            architecture.put("frozen_layers", frozenLayers);
        }

        // Merge handler-specific architecture params (e.g., MuViT patch_size,
        // level_scales, rope_mode). Excludes keys already set above.
        ClassifierRegistry.getHandler(trainingConfig.getModelType()).ifPresent(handler -> {
            Map<String, Object> handlerParams = handler.getArchitectureParams(trainingConfig);
            for (Map.Entry<String, Object> entry : handlerParams.entrySet()) {
                String key = entry.getKey();
                // Skip keys that are already populated or UI-only metadata
                if (!architecture.containsKey(key)
                        && !"available_backbones".equals(key)
                        && !"architecture".equals(key)) {
                    architecture.put(key, entry.getValue());
                }
            }
        });

        // Override with actual handler UI selections (e.g., user-changed patch_size,
        // or MAE-locked model_config). These take priority over handler defaults.
        Map<String, Object> uiParams = trainingConfig.getHandlerParameters();
        if (uiParams != null && !uiParams.isEmpty()) {
            for (Map.Entry<String, Object> entry : uiParams.entrySet()) {
                String key = entry.getKey();
                if (!"available_backbones".equals(key)) {
                    architecture.put(key, entry.getValue());
                }
            }
        }

        Map<String, Object> inputConfig = buildInputConfig(channelConfig);

        Map<String, Object> trainingParams = new HashMap<>();
        trainingParams.put("epochs", trainingConfig.getEpochs());
        trainingParams.put("batch_size", trainingConfig.getBatchSize());
        trainingParams.put("learning_rate", trainingConfig.getLearningRate());
        trainingParams.put("weight_decay", trainingConfig.getWeightDecay());
        trainingParams.put("validation_split", trainingConfig.getValidationSplit());
        trainingParams.put("augmentation", trainingConfig.isAugmentation());
        // Send full augmentation config dict to Python (not just a boolean)
        Map<String, Object> augConfig = new HashMap<>();
        Map<String, Boolean> toggles = trainingConfig.getAugmentationConfig();
        augConfig.put("p_flip", toggles.getOrDefault("flip_horizontal", false)
                || toggles.getOrDefault("flip_vertical", false) ? 0.5 : 0.0);
        augConfig.put("p_rotate", toggles.getOrDefault("rotation_90", false) ? 0.5 : 0.0);
        augConfig.put("p_elastic", toggles.getOrDefault("elastic_deformation", false) ? 0.3 : 0.0);
        augConfig.put("intensity_mode", trainingConfig.getIntensityAugMode());
        trainingParams.put("augmentation_config", augConfig);
        trainingParams.put("scheduler", trainingConfig.getSchedulerType());
        trainingParams.put("loss_function", trainingConfig.getLossFunction());
        String lf = trainingConfig.getLossFunction();
        if ("focal_dice".equals(lf) || "focal".equals(lf)) {
            trainingParams.put("focal_gamma", trainingConfig.getFocalGamma());
        }
        if (trainingConfig.getOhemHardRatio() < 1.0) {
            trainingParams.put("ohem_hard_ratio", trainingConfig.getOhemHardRatio());
        }
        trainingParams.put("early_stopping", true);
        trainingParams.put("early_stopping_patience", trainingConfig.getEarlyStoppingPatience());
        trainingParams.put("early_stopping_metric", trainingConfig.getEarlyStoppingMetric());
        trainingParams.put("mixed_precision", trainingConfig.isMixedPrecision());
        if (trainingConfig.getGradientAccumulationSteps() > 1) {
            trainingParams.put("gradient_accumulation_steps", trainingConfig.getGradientAccumulationSteps());
        }
        if (trainingConfig.isProgressiveResize()) {
            trainingParams.put("progressive_resize", true);
        }
        if (trainingConfig.getFocusClass() != null) {
            trainingParams.put("focus_class", trainingConfig.getFocusClass());
            trainingParams.put("focus_class_min_iou", trainingConfig.getFocusClassMinIoU());
        }

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("model_type", trainingConfig.getModelType());
        inputs.put("architecture", architecture);
        inputs.put("input_config", inputConfig);
        inputs.put("training_params", trainingParams);
        inputs.put("classes", classNames);
        inputs.put("data_path", trainingDataPath.toString());

        // Continue training: load weights from a previously trained model
        if (trainingConfig.getPretrainedModelPath() != null) {
            inputs.put("pretrained_model_path", trainingConfig.getPretrainedModelPath());
        }

        // Project-local model output directory
        if (trainingConfig.getModelOutputDir() != null) {
            inputs.put("model_output_dir", trainingConfig.getModelOutputDir());
        }

        // Generate a synthetic job ID for Appose-based training
        String jobId = "appose-" + System.currentTimeMillis();
        inputs.put("pause_signal_path", getPauseSignalPath(jobId).toString());
        if (jobIdCallback != null) {
            jobIdCallback.accept(jobId);
        }

        // All Appose task operations need TCCL set for Groovy JSON serialization
        // (task.start/cancel/waitFor all go through Messages.encode -> Groovy).
        ClassLoader extensionCL = ApposeService.class.getClassLoader();
        ClassLoader originalCL = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(extensionCL);

        // No retry for training -- it is a long-running stateful operation.
        // If Appose reports "thread death", the Python-side may still be running.
        // Retrying would create a duplicate training process on the same GPU.
        try {
            return executeTrainingTask(appose, inputs, jobId, extensionCL,
                    progressCallback, cancelledCheck);
        } finally {
            Thread.currentThread().setContextClassLoader(originalCL);
        }
    }

    /**
     * Run MAE pretraining for MuViT encoder on unlabeled image tiles.
     * <p>
     * This trains the encoder in a self-supervised manner using masked image
     * reconstruction. The resulting weights can be loaded as pretrained weights
     * for downstream MuViT pixel classification.
     *
     * @param config           pretraining configuration (model size, epochs, mask ratio, etc.)
     * @param dataPath         directory of image tiles for pretraining
     * @param outputDir        directory to save pretrained encoder weights
     * @param progressCallback receives progress updates during pretraining
     * @param cancelledCheck   returns true if pretraining should be cancelled
     * @return result containing encoder_path and training metrics
     * @throws IOException if pretraining fails
     */
    public ClassifierClient.TrainingResult startPretraining(
            Map<String, Object> config,
            Path dataPath,
            Path outputDir,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("config", config);
        inputs.put("data_path", dataPath.toString());
        inputs.put("output_dir", outputDir.toString());

        logger.info("Starting MAE pretraining: config={}, data={}, output={}",
                config.get("model_config"), dataPath, outputDir);

        Task task = appose.createTask("pretrain_mae", inputs);

        // Wire up progress reporting
        task.listen(event -> {
            if (event.responseType == ResponseType.UPDATE && event.message != null) {
                try {
                    JsonObject json = JsonParser.parseString(event.message).getAsJsonObject();
                    ClassifierClient.TrainingProgress progress =
                            new ClassifierClient.TrainingProgress(
                                    json.has("epoch") ? json.get("epoch").getAsInt() : 0,
                                    json.has("total_epochs") ? json.get("total_epochs").getAsInt() : 0,
                                    json.has("train_loss") ? json.get("train_loss").getAsDouble() : 0.0,
                                    json.has("val_loss") ? json.get("val_loss").getAsDouble() : 0.0,
                                    json.has("accuracy") ? json.get("accuracy").getAsDouble() : 0.0,
                                    json.has("mean_iou") ? json.get("mean_iou").getAsDouble() : 0.0,
                                    Map.of(), Map.of(),
                                    json.has("device") ? json.get("device").getAsString() : "",
                                    json.has("device_info") ? json.get("device_info").getAsString() : "",
                                    json.has("status") ? json.get("status").getAsString() : "",
                                    json.has("setup_phase") ? json.get("setup_phase").getAsString() : "",
                                    Map.of()
                            );
                    if (progressCallback != null) {
                        progressCallback.accept(progress);
                    }
                } catch (Exception e) {
                    logger.warn("Failed to parse pretraining progress: {}", e.getMessage());
                }
            }
        });

        // Start task and poll for cancellation
        task.start();
        while (!task.status.isFinished()) {
            if (cancelledCheck != null && cancelledCheck.get()) {
                task.cancel();
                logger.info("MAE pretraining cancelled by user");
            }
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                task.cancel();
                throw new IOException("MAE pretraining interrupted", e);
            }
        }

        if (task.status == org.apposed.appose.Service.TaskStatus.FAILED) {
            throw new IOException("MAE pretraining failed: " + task.error);
        }

        String encoderPath = task.outputs.containsKey("encoder_path")
                ? task.outputs.get("encoder_path").toString() : "";
        int epochsCompleted = task.outputs.containsKey("epochs_completed")
                ? ((Number) task.outputs.get("epochs_completed")).intValue() : 0;
        double finalLoss = task.outputs.containsKey("final_loss")
                ? ((Number) task.outputs.get("final_loss")).doubleValue() : 0.0;

        logger.info("MAE pretraining complete: {} epochs, loss={}, path={}",
                epochsCompleted, finalLoss, encoderPath);

        return new ClassifierClient.TrainingResult(
                "mae-pretrain", encoderPath, finalLoss, 0.0, 0, 0.0);
    }

    @Override
    public void pauseTraining(String jobId) throws IOException {
        Path signalPath = getPauseSignalPath(jobId);
        Files.writeString(signalPath, "pause");
        logger.info("Pause signal written for job {}", jobId);
    }

    /**
     * Returns the file path used as a pause signal for the given job.
     * The Python train.py script polls for this file's existence.
     */
    private Path getPauseSignalPath(String jobId) {
        return Path.of(System.getProperty("java.io.tmpdir"), "dl-pause-" + jobId);
    }

    @Override
    public ClassifierClient.TrainingResult resumeTraining(
            String jobId,
            Path newDataPath,
            Integer epochs,
            Double learningRate,
            Integer batchSize,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException {

        CheckpointInfo checkpoint = checkpointStore.get(jobId);
        if (checkpoint == null) {
            throw new IOException("No checkpoint info stored for job: " + jobId);
        }

        ApposeService appose = ApposeService.getInstance();

        // Rebuild inputs using checkpoint's original config, overriding with resume params
        Map<String, Object> inputs = new HashMap<>(checkpoint.originalInputs());
        if (newDataPath != null) {
            inputs.put("data_path", newDataPath.toString());
        }
        inputs.put("checkpoint_path", checkpoint.path());
        inputs.put("start_epoch", checkpoint.lastEpoch());

        // Override training params with resume values
        @SuppressWarnings("unchecked")
        Map<String, Object> trainingParams = new HashMap<>(
                (Map<String, Object>) inputs.get("training_params"));
        if (epochs != null) trainingParams.put("epochs", epochs);
        if (learningRate != null) trainingParams.put("learning_rate", learningRate);
        if (batchSize != null) trainingParams.put("batch_size", batchSize);
        inputs.put("training_params", trainingParams);

        String newJobId = "appose-resume-" + System.currentTimeMillis();
        inputs.put("pause_signal_path", getPauseSignalPath(newJobId).toString());

        ClassLoader extensionCL = ApposeService.class.getClassLoader();
        ClassLoader originalCL = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(extensionCL);
        try {
            ClassifierClient.TrainingResult result = executeTrainingTask(
                    appose, inputs, newJobId, extensionCL,
                    progressCallback, cancelledCheck);
            // If paused again, store checkpoint for next resume
            if (result.isPaused() && result.checkpointPath() != null) {
                storeCheckpointInfo(newJobId, result.checkpointPath(),
                        result.lastEpoch(), inputs);
            }
            return result;
        } finally {
            Thread.currentThread().setContextClassLoader(originalCL);
        }
    }

    @Override
    public ClassifierClient.TrainingResult finalizeTraining(String checkpointPath) throws IOException {
        return finalizeTraining(checkpointPath, null);
    }

    @Override
    public ClassifierClient.TrainingResult finalizeTraining(String checkpointPath,
            String modelOutputDir) throws IOException {
        ApposeService appose = ApposeService.getInstance();
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", checkpointPath);
        if (modelOutputDir != null) {
            inputs.put("model_output_dir", modelOutputDir);
        }

        ClassLoader extensionCL = ApposeService.class.getClassLoader();
        ClassLoader originalCL = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(extensionCL);
        try {
            Task task = appose.runTask("finalize_training", inputs);

            String modelPath = String.valueOf(task.outputs.get("model_path"));
            double loss = ((Number) task.outputs.getOrDefault("final_loss", 0.0)).doubleValue();
            double acc = ((Number) task.outputs.getOrDefault("final_accuracy", 0.0)).doubleValue();
            int bestEpoch = ((Number) task.outputs.getOrDefault("best_epoch", 0)).intValue();
            double bestMIoU = ((Number) task.outputs.getOrDefault("best_mean_iou", 0.0)).doubleValue();
            int epochsTrained = ((Number) task.outputs.getOrDefault("epochs_trained", 0)).intValue();

            return new ClassifierClient.TrainingResult(
                    "finalized", modelPath, loss, acc, bestEpoch, bestMIoU);
        } catch (Exception e) {
            throw new IOException("Failed to finalize training: " + e.getMessage(), e);
        } finally {
            Thread.currentThread().setContextClassLoader(originalCL);
        }
    }

    /**
     * Stores checkpoint info for a paused training job.
     */
    void storeCheckpointInfo(String jobId, String path, int lastEpoch,
                             Map<String, Object> originalInputs) {
        checkpointStore.put(jobId, new CheckpointInfo(path, lastEpoch, originalInputs));
        logger.debug("Stored checkpoint info for job {}: epoch={}, path={}",
                jobId, lastEpoch, path);
    }

    /**
     * Retrieves checkpoint info for a paused training job.
     *
     * @param jobId the job ID
     * @return checkpoint info, or null if not found
     */
    public CheckpointInfo getCheckpointInfo(String jobId) {
        return checkpointStore.get(jobId);
    }

    /**
     * Executes a training task via Appose. No retry -- training is a long-running
     * stateful operation and retrying risks duplicate GPU processes.
     */
    private ClassifierClient.TrainingResult executeTrainingTask(
            ApposeService appose,
            Map<String, Object> inputs,
            String jobId,
            ClassLoader extensionCL,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException {

        Task task = appose.createTask("train", inputs);

        // Listen for progress events
        task.listen(event -> {
            if (event.responseType == ResponseType.UPDATE && event.message != null) {
                try {
                    ClassifierClient.TrainingProgress progress = parseProgressJson(event.message);
                    if (progressCallback != null) {
                        progressCallback.accept(progress);
                    }
                } catch (Exception e) {
                    logger.debug("Failed to parse training progress: {}", e.getMessage());
                }
            }
        });

        // Start the task
        task.start();

        // Poll for cancellation in a background thread.
        // The cancel thread also needs TCCL because task.cancel()
        // sends a JSON message via Groovy serialization.
        Thread cancelThread = new Thread(() -> {
            Thread.currentThread().setContextClassLoader(extensionCL);
            while (!task.status.isFinished()) {
                if (cancelledCheck != null && cancelledCheck.get()) {
                    logger.info("Training cancel requested, sending to Appose task");
                    task.cancel();
                    break;
                }
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "DLClassifier-ApposeTrainCancel");
        cancelThread.setDaemon(true);
        cancelThread.start();

        // Wait for completion
        try {
            task.waitFor();
        } catch (Exception e) {
            if (task.status == org.apposed.appose.Service.TaskStatus.CANCELED) {
                logger.info("Training cancelled via Appose");
                // Try to recover model_path in case cancel arrived after model was saved
                String cancelledModelPath = null;
                if (task.outputs != null && task.outputs.containsKey("model_path")) {
                    String mp = String.valueOf(task.outputs.get("model_path"));
                    if (!mp.isEmpty() && !"null".equals(mp)) {
                        cancelledModelPath = mp;
                    }
                }
                return new ClassifierClient.TrainingResult(jobId, cancelledModelPath, 0, 0);
            }
            // Do NOT retry training on "thread death". Training is a long-running
            // stateful operation -- the Python-side task may still be executing even
            // though Appose reported failure. Retrying would create a second concurrent
            // training run on the same GPU, causing deadlock or OOM.
            throw new IOException("Training failed: " + task.error, e);
        }

        // Check if training was paused (not cancelled, not failed)
        String status = String.valueOf(task.outputs.getOrDefault("status", "completed"));
        if ("paused".equals(status)) {
            String checkpointPath = String.valueOf(task.outputs.getOrDefault("checkpoint_path", ""));
            int lastEpoch = ((Number) task.outputs.getOrDefault("last_epoch", 0)).intValue();
            int totalEpochs = ((Number) task.outputs.getOrDefault("total_epochs", 0)).intValue();
            logger.info("Training paused at epoch {}/{}, checkpoint: {}", lastEpoch, totalEpochs, checkpointPath);

            // Store checkpoint info for resume/finalize
            storeCheckpointInfo(jobId, checkpointPath, lastEpoch, inputs);

            return new ClassifierClient.TrainingResult(
                    jobId, null, 0, 0, 0, 0, true, lastEpoch, totalEpochs, checkpointPath);
        }

        // Normal completion
        String modelPath = String.valueOf(task.outputs.get("model_path"));
        double finalLoss = ((Number) task.outputs.getOrDefault("final_loss", 0.0)).doubleValue();
        double finalAccuracy = ((Number) task.outputs.getOrDefault("final_accuracy", 0.0)).doubleValue();
        int bestEpoch = ((Number) task.outputs.getOrDefault("best_epoch", 0)).intValue();
        double bestMeanIoU = ((Number) task.outputs.getOrDefault("best_mean_iou", 0.0)).doubleValue();

        // Store checkpoint for potential "continue training"
        String completionCheckpoint = String.valueOf(task.outputs.getOrDefault("checkpoint_path", ""));
        int lastEpoch = ((Number) task.outputs.getOrDefault("last_epoch", 0)).intValue();
        int totalEpochs = ((Number) task.outputs.getOrDefault("total_epochs", 0)).intValue();
        if (!completionCheckpoint.isEmpty() && !"null".equals(completionCheckpoint)) {
            storeCheckpointInfo(jobId, completionCheckpoint, lastEpoch, inputs);
            logger.info("Stored completion checkpoint for continue-training: epoch {}, path {}",
                    lastEpoch, completionCheckpoint);
        }

        return new ClassifierClient.TrainingResult(jobId, modelPath, finalLoss, finalAccuracy,
                bestEpoch, bestMeanIoU, false, lastEpoch, totalEpochs, completionCheckpoint);
    }

    // ==================== Evaluation ====================

    @Override
    public List<ClassifierClient.TileEvaluationResult> evaluateTiles(
            Path modelPath,
            Path trainingDataPath,
            String architecture,
            String backbone,
            Map<String, Object> inputConfig,
            List<String> classNames,
            Map<String, Integer> classColors,
            Consumer<ClassifierClient.EvaluationProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        Map<String, Object> archMap = new HashMap<>();
        archMap.put("type", architecture);
        archMap.put("backbone", backbone);
        archMap.put("use_pretrained", false); // Loading trained weights, not ImageNet
        // Propagate context scale if present in input config
        Object contextScale = inputConfig.get("context_scale");
        if (contextScale != null) {
            archMap.put("context_scale", contextScale);
        }

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("model_path", modelPath.toString());
        inputs.put("data_path", trainingDataPath.toString());
        inputs.put("architecture", archMap);
        inputs.put("input_config", inputConfig);
        inputs.put("classes", classNames);
        if (classColors != null && !classColors.isEmpty()) {
            inputs.put("class_colors", classColors);
        }

        ClassLoader extensionCL = ApposeService.class.getClassLoader();
        ClassLoader originalCL = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(extensionCL);

        try {
            Task task = appose.createTask("evaluate_tiles", inputs);

            // Listen for progress events
            task.listen(event -> {
                if (event.responseType == ResponseType.UPDATE && event.message != null) {
                    try {
                        JsonObject msg = JsonParser.parseString(event.message).getAsJsonObject();
                        int currentTile = msg.has("current_tile") ? msg.get("current_tile").getAsInt() : 0;
                        int totalTiles = msg.has("total_tiles") ? msg.get("total_tiles").getAsInt() : 0;
                        if (progressCallback != null) {
                            progressCallback.accept(new ClassifierClient.EvaluationProgress(
                                    currentTile, totalTiles,
                                    String.format("Evaluating tile %d/%d", currentTile, totalTiles)));
                        }
                    } catch (Exception e) {
                        logger.debug("Failed to parse evaluation progress: {}", e.getMessage());
                    }
                }
            });

            task.start();

            // Poll for cancellation
            Thread cancelThread = new Thread(() -> {
                Thread.currentThread().setContextClassLoader(extensionCL);
                while (!task.status.isFinished()) {
                    if (cancelledCheck != null && cancelledCheck.get()) {
                        logger.info("Evaluation cancel requested");
                        task.cancel();
                        break;
                    }
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }, "DLClassifier-ApposeEvalCancel");
            cancelThread.setDaemon(true);
            cancelThread.start();

            task.waitFor();

            // Parse results
            String resultsJson = String.valueOf(task.outputs.get("results"));
            return parseEvaluationResults(resultsJson);

        } catch (Exception e) {
            if (e.getMessage() != null && e.getMessage().contains("CANCELED")) {
                logger.info("Evaluation cancelled");
                return List.of();
            }
            throw new IOException("Evaluation failed: " + e.getMessage(), e);
        } finally {
            Thread.currentThread().setContextClassLoader(originalCL);
        }
    }

    /**
     * Parses the JSON evaluation results from the Python script.
     */
    @SuppressWarnings("unchecked")
    private List<ClassifierClient.TileEvaluationResult> parseEvaluationResults(String json) {
        List<ClassifierClient.TileEvaluationResult> results = new ArrayList<>();
        if (json == null || json.isEmpty() || "null".equals(json)) {
            return results;
        }

        var jsonArray = JsonParser.parseString(json).getAsJsonArray();
        for (var element : jsonArray) {
            var obj = element.getAsJsonObject();

            Map<String, Double> perClassIoU = new LinkedHashMap<>();
            if (obj.has("per_class_iou") && !obj.get("per_class_iou").isJsonNull()) {
                var iouObj = obj.getAsJsonObject("per_class_iou");
                for (var entry : iouObj.entrySet()) {
                    if (!entry.getValue().isJsonNull()) {
                        perClassIoU.put(entry.getKey(), entry.getValue().getAsDouble());
                    }
                }
            }

            String disagreementImagePath = obj.has("disagreement_image") && !obj.get("disagreement_image").isJsonNull()
                    ? obj.get("disagreement_image").getAsString() : null;
            String lossHeatmapPath = obj.has("loss_heatmap") && !obj.get("loss_heatmap").isJsonNull()
                    ? obj.get("loss_heatmap").getAsString() : null;
            String tileImagePath = obj.has("tile_image") && !obj.get("tile_image").isJsonNull()
                    ? obj.get("tile_image").getAsString() : null;

            results.add(new ClassifierClient.TileEvaluationResult(
                    obj.has("filename") ? obj.get("filename").getAsString() : "",
                    obj.has("split") ? obj.get("split").getAsString() : "",
                    obj.has("loss") ? obj.get("loss").getAsDouble() : 0.0,
                    obj.has("disagreement_pct") ? obj.get("disagreement_pct").getAsDouble() : 0.0,
                    perClassIoU,
                    obj.has("mean_iou") ? obj.get("mean_iou").getAsDouble() : 0.0,
                    obj.has("x") ? obj.get("x").getAsInt() : 0,
                    obj.has("y") ? obj.get("y").getAsInt() : 0,
                    obj.has("source_image") ? obj.get("source_image").getAsString() : "",
                    obj.has("source_image_id") ? obj.get("source_image_id").getAsString() : "",
                    disagreementImagePath,
                    lossHeatmapPath,
                    tileImagePath
            ));
        }

        return results;
    }

    // ==================== Inference ====================

    @Override
    public ClassifierClient.PixelInferenceResult runPixelInferenceBinary(
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
            int reflectionPadding) throws IOException {

        ApposeService appose = ApposeService.getInstance();
        int numTiles = tileIds.size();

        // For single-tile overlay inference, use shared-memory path
        if (numTiles == 1) {
            return runSingleTilePixelInference(
                    appose, modelPath, rawTileBytes, tileIds.get(0),
                    tileHeight, tileWidth, numChannels, dtype,
                    channelConfig, inferenceConfig, outputDir, reflectionPadding);
        }

        // Multi-tile: use file-based output (same as HTTP backend)
        return runMultiTilePixelInference(
                appose, modelPath, rawTileBytes, tileIds,
                tileHeight, tileWidth, numChannels, dtype,
                channelConfig, inferenceConfig, outputDir, reflectionPadding);
    }

    /** Maximum retry attempts for transient "thread death" errors from Appose. */
    private static final int MAX_THREAD_DEATH_RETRIES = 2;
    /** Delay between retry attempts to let Appose Python threads clean up. */
    private static final long THREAD_DEATH_RETRY_DELAY_MS = 100;

    /**
     * Single-tile pixel inference with full shared-memory round-trip.
     * No file I/O -- probability map returned directly via shared memory.
     * <p>
     * Wrapped with {@link ApposeService#withExtensionClassLoader} because
     * NDArray allocation triggers ServiceLoader discovery of ShmFactory,
     * and QuPath's tile-rendering threads don't have the extension classloader
     * as their TCCL.
     * <p>
     * Includes retry logic for transient "thread death" errors from Appose.
     * These occur when a stale Python worker thread's death message is
     * misrouted to the currently-active task. The actual inference completes
     * successfully but the COMPLETION arrives after Appose already reported
     * FAILURE. Retrying with a new task resolves this.
     */
    private ClassifierClient.PixelInferenceResult runSingleTilePixelInference(
            ApposeService appose,
            String modelPath,
            byte[] rawTileBytes,
            String tileId,
            int tileHeight,
            int tileWidth,
            int numChannels,
            String dtype,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            Path outputDir,
            int reflectionPadding) throws IOException {

        // Throttle concurrent Appose task submissions. Without this, 16+
        // overlay threads each spawn a Python thread that blocks on
        // inference_lock, and many die with "thread death" before running.
        try {
            inferenceSemaphore.acquire();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Inference interrupted while waiting for semaphore", e);
        }

        try {
            return ApposeService.withExtensionClassLoader(() -> {
                IOException lastError = null;

                for (int attempt = 0; attempt < MAX_THREAD_DEATH_RETRIES; attempt++) {
                    // Create shared memory NDArray for input tile (fresh per attempt)
                    NDArray.Shape shape = new NDArray.Shape(
                            NDArray.Shape.Order.C_ORDER, tileHeight, tileWidth, numChannels);
                    NDArray inputNd = new NDArray(NDArray.DType.FLOAT32, shape);

                    try {
                        // Copy raw bytes into shared memory, converting dtype if needed.
                        // uint8 values are kept in [0, 255] range (not divided by 255)
                        // because normalization expects raw pixel values matching training stats.
                        FloatBuffer fbuf = inputNd.buffer().order(ByteOrder.nativeOrder()).asFloatBuffer();
                        if ("uint8".equals(dtype)) {
                            for (byte b : rawTileBytes) {
                                fbuf.put((float) (b & 0xFF));
                            }
                        } else {
                            // float32: copy raw bytes directly
                            FloatBuffer srcBuf = ByteBuffer.wrap(rawTileBytes)
                                    .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                            fbuf.put(srcBuf);
                        }

                        // Build input config dict
                        Map<String, Object> inputConfig = buildInputConfig(channelConfig);

                        Map<String, Object> inputs = new HashMap<>();
                        inputs.put("model_path", modelPath);
                        inputs.put("tile_data", inputNd);
                        inputs.put("tile_height", tileHeight);
                        inputs.put("tile_width", tileWidth);
                        inputs.put("num_channels", numChannels);
                        inputs.put("input_config", inputConfig);
                        inputs.put("reflection_padding", reflectionPadding);
                        if (inferenceConfig.isUseTTA()) {
                            inputs.put("use_tta", true);
                        }

                        Task task = appose.runTask("inference_pixel", inputs);

                        int numClasses = ((Number) task.outputs.get("num_classes")).intValue();
                        NDArray resultNd = (NDArray) task.outputs.get("probabilities");

                        try {
                            // Read probability map from shared memory and save as .bin file
                            // (matching the existing file-based contract for DLPixelClassifier)
                            Files.createDirectories(outputDir);
                            Path outputPath = outputDir.resolve(tileId + ".bin");

                            // Result is in CHW order (C classes, H height, W width) as float32
                            ByteBuffer resultBuf = resultNd.buffer().order(ByteOrder.nativeOrder());
                            byte[] resultBytes = new byte[resultBuf.remaining()];
                            resultBuf.get(resultBytes);
                            Files.write(outputPath, resultBytes);

                            Map<String, String> outputPaths = Map.of(tileId, outputPath.toString());
                            return new ClassifierClient.PixelInferenceResult(outputPaths, numClasses);
                        } finally {
                            resultNd.close();
                        }
                    } catch (IOException e) {
                        String msg = e.getMessage() != null ? e.getMessage() : "";
                        if (msg.toLowerCase().contains("thread death")
                                && attempt < MAX_THREAD_DEATH_RETRIES - 1) {
                            lastError = e;
                            logger.debug("Thread death on attempt {}, retrying after {}ms delay",
                                    attempt + 1, THREAD_DEATH_RETRY_DELAY_MS);
                            try {
                                Thread.sleep(THREAD_DEATH_RETRY_DELAY_MS);
                            } catch (InterruptedException ie) {
                                Thread.currentThread().interrupt();
                                throw new IOException("Interrupted during thread death retry", ie);
                            }
                            continue;
                        }
                        throw e;
                    } finally {
                        inputNd.close();
                    }
                }
                // Should not reach here, but satisfy compiler
                throw lastError != null ? lastError
                        : new IOException("Single-tile inference failed after retries");
            });
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Single-tile pixel inference failed", e);
        } finally {
            inferenceSemaphore.release();
        }
    }

    /**
     * Multi-tile pixel inference with file-based output.
     * Wrapped with extension classloader for ShmFactory ServiceLoader discovery.
     */
    private ClassifierClient.PixelInferenceResult runMultiTilePixelInference(
            ApposeService appose,
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
            int reflectionPadding) throws IOException {

        try {
            return ApposeService.withExtensionClassLoader(() -> {
                int numTiles = tileIds.size();
                int pixelsPerTile = tileHeight * tileWidth * numChannels;

                // Convert to float32 if needed.
                // uint8 values are kept in [0, 255] range (not divided by 255)
                // because normalization expects raw pixel values matching training stats.
                byte[] float32Bytes;
                if ("uint8".equals(dtype)) {
                    float32Bytes = new byte[numTiles * pixelsPerTile * Float.BYTES];
                    FloatBuffer fbuf = ByteBuffer.wrap(float32Bytes)
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                    for (byte b : rawTileBytes) {
                        fbuf.put((float) (b & 0xFF));
                    }
                } else {
                    float32Bytes = rawTileBytes;
                }

                // Create shared memory NDArray for all tiles
                NDArray.Shape shape = new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER,
                        numTiles, tileHeight, tileWidth, numChannels);
                NDArray inputNd = new NDArray(NDArray.DType.FLOAT32, shape);

                try {
                    ByteBuffer buf = inputNd.buffer().order(ByteOrder.nativeOrder());
                    buf.put(float32Bytes);

                    Map<String, Object> inputConfig = buildInputConfig(channelConfig);

                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("model_path", modelPath);
                    inputs.put("tile_data", inputNd);
                    inputs.put("tile_ids", tileIds);
                    inputs.put("tile_height", tileHeight);
                    inputs.put("tile_width", tileWidth);
                    inputs.put("num_channels", numChannels);
                    inputs.put("input_config", inputConfig);
                    inputs.put("output_dir", outputDir.toString());
                    inputs.put("reflection_padding", reflectionPadding);
                    if (inferenceConfig.isUseTTA()) {
                        inputs.put("use_tta", true);
                    }

                    Task task = appose.runTask("inference_pixel_batch", inputs);

                    int numClasses = ((Number) task.outputs.get("num_classes")).intValue();
                    @SuppressWarnings("unchecked")
                    Map<String, String> outputPaths = (Map<String, String>) task.outputs.get("output_paths");
                    return new ClassifierClient.PixelInferenceResult(
                            new HashMap<>(outputPaths), numClasses);
                } finally {
                    inputNd.close();
                }
            });
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Multi-tile pixel inference failed", e);
        }
    }

    @Override
    public ClassifierClient.PixelInferenceResult runPixelInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            Path outputDir,
            int reflectionPadding) throws IOException {
        // For the Appose backend, convert TileData to binary and delegate
        // to the binary path. This avoids needing a separate base64 script.
        // Note: This is a fallback path; the primary path uses runPixelInferenceBinary.
        logger.warn("Appose backend using base64 tile fallback -- this should not normally happen");
        throw new IOException("Appose backend does not support base64 tile transfer. " +
                "Use runPixelInferenceBinary instead.");
    }

    @Override
    public ClassifierClient.InferenceResult runInferenceBinary(
            String modelPath,
            byte[] rawTileBytes,
            List<String> tileIds,
            int tileHeight,
            int tileWidth,
            int numChannels,
            String dtype,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        try {
            return ApposeService.withExtensionClassLoader(() -> {
                int numTiles = tileIds.size();
                int pixelsPerTile = tileHeight * tileWidth * numChannels;

                // Convert to float32 if needed.
                // uint8 values are kept in [0, 255] range (not divided by 255)
                // because normalization expects raw pixel values matching training stats.
                byte[] float32Bytes;
                if ("uint8".equals(dtype)) {
                    float32Bytes = new byte[numTiles * pixelsPerTile * Float.BYTES];
                    FloatBuffer fbuf = ByteBuffer.wrap(float32Bytes)
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                    for (byte b : rawTileBytes) {
                        fbuf.put((float) (b & 0xFF));
                    }
                } else {
                    float32Bytes = rawTileBytes;
                }

                NDArray.Shape shape = new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER,
                        numTiles, tileHeight, tileWidth, numChannels);
                NDArray inputNd = new NDArray(NDArray.DType.FLOAT32, shape);

                try {
                    ByteBuffer buf = inputNd.buffer().order(ByteOrder.nativeOrder());
                    buf.put(float32Bytes);

                    Map<String, Object> inputConfig = buildInputConfig(channelConfig);

                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("model_path", modelPath);
                    inputs.put("tile_data", inputNd);
                    inputs.put("tile_ids", tileIds);
                    inputs.put("tile_height", tileHeight);
                    inputs.put("tile_width", tileWidth);
                    inputs.put("num_channels", numChannels);
                    inputs.put("input_config", inputConfig);

                    Task task = appose.runTask("inference_batch", inputs);

                    @SuppressWarnings("unchecked")
                    Map<String, Object> rawPredictions =
                            (Map<String, Object>) task.outputs.get("predictions");

                    Map<String, float[]> predictions = new HashMap<>();
                    for (Map.Entry<String, Object> entry : rawPredictions.entrySet()) {
                        @SuppressWarnings("unchecked")
                        List<Number> probs = (List<Number>) entry.getValue();
                        float[] probArray = new float[probs.size()];
                        for (int i = 0; i < probs.size(); i++) {
                            probArray[i] = probs.get(i).floatValue();
                        }
                        predictions.put(entry.getKey(), probArray);
                    }

                    return new ClassifierClient.InferenceResult(predictions);
                } finally {
                    inputNd.close();
                }
            });
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Batch inference failed", e);
        }
    }

    @Override
    public ClassifierClient.InferenceResult runInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig) throws IOException {
        // For the Appose backend, the base64 path is not supported.
        // Callers should use runInferenceBinary.
        logger.warn("Appose backend does not support base64 tile transfer for batch inference");
        throw new IOException("Appose backend does not support base64 tile transfer. " +
                "Use runInferenceBinary instead.");
    }

    // ==================== Pretrained Model Info ====================

    @Override
    public List<ClassifierClient.LayerInfo> getModelLayers(
            String architecture, String encoder,
            int numChannels, int numClasses) throws IOException {

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("architecture", architecture);
        inputs.put("encoder", encoder);
        inputs.put("num_channels", numChannels);
        inputs.put("num_classes", numClasses);

        Task task = ApposeService.getInstance().runTask("get_model_layers", inputs);

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> layers = (List<Map<String, Object>>) task.outputs.get("layers");

        List<ClassifierClient.LayerInfo> result = new ArrayList<>();
        if (layers != null) {
            for (Map<String, Object> layer : layers) {
                result.add(new ClassifierClient.LayerInfo(
                        String.valueOf(layer.get("name")),
                        String.valueOf(layer.get("display_name")),
                        ((Number) layer.getOrDefault("param_count", 0)).intValue(),
                        Boolean.TRUE.equals(layer.get("is_encoder")),
                        ((Number) layer.getOrDefault("depth", 0)).intValue(),
                        Boolean.TRUE.equals(layer.get("recommended_freeze")),
                        String.valueOf(layer.getOrDefault("description", ""))
                ));
            }
        }
        return result;
    }

    @Override
    public Map<Integer, Boolean> getFreezeRecommendations(
            String datasetSize, String encoder) throws IOException {

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("dataset_size", datasetSize);
        if (encoder != null) {
            inputs.put("encoder", encoder);
        }

        Task task = ApposeService.getInstance().runTask("get_freeze_recommendations", inputs);

        @SuppressWarnings("unchecked")
        Map<String, Object> recs = (Map<String, Object>) task.outputs.get("recommendations");

        Map<Integer, Boolean> result = new HashMap<>();
        if (recs != null) {
            for (Map.Entry<String, Object> entry : recs.entrySet()) {
                try {
                    result.put(Integer.parseInt(entry.getKey()),
                            Boolean.TRUE.equals(entry.getValue()));
                } catch (NumberFormatException e) {
                    logger.debug("Skipping non-integer freeze key: {}", entry.getKey());
                }
            }
        }
        return result;
    }

    // ==================== Helpers ====================

    /**
     * Builds the input configuration dict from a ChannelConfiguration.
     * <p>
     * Includes normalization strategy, per-channel flag, clip percentile,
     * fixed min/max range, and precomputed image-level stats when available.
     */
    /**
     * Builds the input config map for Python scripts from a channel configuration.
     *
     * @param channelConfig channel configuration
     * @return input config map
     */
    public static Map<String, Object> buildInputConfig(ChannelConfiguration channelConfig) {
        Map<String, Object> inputConfig = new HashMap<>();
        inputConfig.put("num_channels", channelConfig.getNumChannels());
        inputConfig.put("selected_channels", channelConfig.getSelectedChannels());

        Map<String, Object> normalization = new HashMap<>();
        normalization.put("strategy", channelConfig.getNormalizationStrategy().name().toLowerCase());
        normalization.put("per_channel", channelConfig.isPerChannelNormalization());
        normalization.put("clip_percentile", channelConfig.getClipPercentile());
        // Include fixed range values (previously missing for FIXED_RANGE strategy)
        normalization.put("min", channelConfig.getFixedMin());
        normalization.put("max", channelConfig.getFixedMax());

        // Include precomputed image-level normalization stats when available
        List<Map<String, Double>> precomputedStats = channelConfig.getPrecomputedChannelStats();
        if (precomputedStats != null && !precomputedStats.isEmpty()) {
            normalization.put("precomputed", true);
            normalization.put("channel_stats", precomputedStats);
        }

        inputConfig.put("normalization", normalization);

        return inputConfig;
    }

    /**
     * Parses a JSON training progress message from a task update event.
     */
    private static ClassifierClient.TrainingProgress parseProgressJson(String json) {
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        int epoch = obj.get("epoch").getAsInt();
        int totalEpochs = obj.get("total_epochs").getAsInt();
        double trainLoss = obj.has("train_loss") ? obj.get("train_loss").getAsDouble() : 0;
        double valLoss = obj.has("val_loss") ? obj.get("val_loss").getAsDouble() : 0;
        double accuracy = obj.has("accuracy") ? obj.get("accuracy").getAsDouble() : 0;
        double meanIoU = obj.has("mean_iou") ? obj.get("mean_iou").getAsDouble() : 0;

        Map<String, Double> perClassIoU = parseStringDoubleMap(obj, "per_class_iou");
        Map<String, Double> perClassLoss = parseStringDoubleMap(obj, "per_class_loss");

        // Device info (present in epoch-0 pre-training update, absent in later epochs)
        String device = obj.has("device") ? obj.get("device").getAsString() : null;
        String deviceInfo = obj.has("device_info") ? obj.get("device_info").getAsString() : null;

        // Setup phase info (present during model initialization)
        String status = obj.has("status") ? obj.get("status").getAsString() : null;
        String setupPhase = obj.has("setup_phase") ? obj.get("setup_phase").getAsString() : null;

        // Training config summary (present in "training_config" setup phase)
        Map<String, String> configSummary = new LinkedHashMap<>();
        if (obj.has("config") && !obj.get("config").isJsonNull()) {
            JsonObject cfg = obj.getAsJsonObject("config");
            for (String key : cfg.keySet()) {
                configSummary.put(key, cfg.get(key).getAsString());
            }
        }

        return new ClassifierClient.TrainingProgress(
                epoch, totalEpochs, trainLoss, valLoss, accuracy,
                meanIoU, perClassIoU, perClassLoss, device, deviceInfo,
                status, setupPhase, configSummary);
    }

    private static Map<String, Double> parseStringDoubleMap(JsonObject parent, String fieldName) {
        Map<String, Double> result = new LinkedHashMap<>();
        if (parent.has(fieldName) && !parent.get(fieldName).isJsonNull()) {
            JsonObject obj = parent.getAsJsonObject(fieldName);
            for (String key : obj.keySet()) {
                result.put(key, obj.get(key).getAsDouble());
            }
        }
        return result;
    }
}
