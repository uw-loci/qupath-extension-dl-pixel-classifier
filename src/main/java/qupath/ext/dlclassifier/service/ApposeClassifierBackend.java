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
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;

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

    // Maps old job IDs to new ones after resume, so pauseTraining() can
    // find the correct pause signal path even when called with the old ID.
    private static final ConcurrentHashMap<String, String> jobIdRedirects = new ConcurrentHashMap<>();

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
        trainingParams.put("discriminative_lr_ratio", trainingConfig.getDiscriminativeLrRatio());
        if (trainingConfig.getSeed() != null) {
            trainingParams.put("seed", trainingConfig.getSeed());
        }
        trainingParams.put("validation_split", trainingConfig.getValidationSplit());
        trainingParams.put("augmentation", trainingConfig.isAugmentation());
        // Send full augmentation config dict to Python (not just a boolean).
        // Advanced strength/probability values come from trainingConfig.getAugmentationParams()
        // (populated from preferences via AdvancedAugmentationDialog). Toggles zero out the
        // corresponding probability when a whole category is disabled.
        Map<String, Object> augConfig = new HashMap<>();
        Map<String, Boolean> toggles = trainingConfig.getAugmentationConfig();
        Map<String, Object> params = trainingConfig.getAugmentationParams();

        // Defaults match training_service.get_training_augmentation() hardcoded values
        double userPFlip = toDouble(params.get("p_flip"), 0.5);
        double userPRotate = toDouble(params.get("p_rotate"), 0.5);
        double userPElastic = toDouble(params.get("p_elastic"), 0.3);

        boolean flipOn = toggles.getOrDefault("flip_horizontal", false)
                || toggles.getOrDefault("flip_vertical", false);
        boolean rotateOn = toggles.getOrDefault("rotation_90", false);
        boolean elasticOn = toggles.getOrDefault("elastic_deformation", false);

        augConfig.put("p_flip", flipOn ? userPFlip : 0.0);
        augConfig.put("p_rotate", rotateOn ? userPRotate : 0.0);
        augConfig.put("p_elastic", elasticOn ? userPElastic : 0.0);

        // Forward all remaining strength/probability params verbatim; Python fills any missing
        if (params.containsKey("p_color")) augConfig.put("p_color", toDouble(params.get("p_color"), 0.3));
        if (params.containsKey("brightness_limit"))
            augConfig.put("brightness_limit", toDouble(params.get("brightness_limit"), 0.2));
        if (params.containsKey("contrast_limit"))
            augConfig.put("contrast_limit", toDouble(params.get("contrast_limit"), 0.2));
        if (params.containsKey("gamma_min"))
            augConfig.put("gamma_min", toInt(params.get("gamma_min"), 80));
        if (params.containsKey("gamma_max"))
            augConfig.put("gamma_max", toInt(params.get("gamma_max"), 120));
        if (params.containsKey("elastic_alpha"))
            augConfig.put("elastic_alpha", toDouble(params.get("elastic_alpha"), 120.0));
        if (params.containsKey("elastic_sigma_ratio"))
            augConfig.put("elastic_sigma_ratio", toDouble(params.get("elastic_sigma_ratio"), 0.05));
        if (params.containsKey("p_noise"))
            augConfig.put("p_noise", toDouble(params.get("p_noise"), 0.2));
        if (params.containsKey("noise_std_min"))
            augConfig.put("noise_std_min", toDouble(params.get("noise_std_min"), 0.04));
        if (params.containsKey("noise_std_max"))
            augConfig.put("noise_std_max", toDouble(params.get("noise_std_max"), 0.2));
        if (params.containsKey("scale_jitter_limit"))
            augConfig.put("scale_jitter_limit",
                    toDouble(params.get("scale_jitter_limit"), 0.0));
        if (params.containsKey("p_scale_jitter"))
            augConfig.put("p_scale_jitter",
                    toDouble(params.get("p_scale_jitter"), 0.5));

        augConfig.put("intensity_mode", trainingConfig.getIntensityAugMode());
        trainingParams.put("augmentation_config", augConfig);
        trainingParams.put("scheduler", trainingConfig.getSchedulerType());
        trainingParams.put("loss_function", trainingConfig.getLossFunction());
        String lf = trainingConfig.getLossFunction();
        if ("focal_dice".equals(lf) || "focal".equals(lf)) {
            trainingParams.put("focal_gamma", trainingConfig.getFocalGamma());
        }
        if ("boundary_ce".equals(lf) || "boundary_ce_dice".equals(lf)) {
            trainingParams.put("boundary_sigma", trainingConfig.getBoundarySigma());
            trainingParams.put("boundary_w_min", trainingConfig.getBoundaryWMin());
        }
        if (trainingConfig.getOhemHardRatio() < 1.0) {
            trainingParams.put("ohem_hard_ratio", trainingConfig.getOhemHardRatio());
            trainingParams.put("ohem_hard_ratio_start", trainingConfig.getOhemHardRatioStart());
            trainingParams.put("ohem_schedule", trainingConfig.getOhemSchedule());
            trainingParams.put("ohem_adaptive_floor", trainingConfig.isOhemAdaptiveFloor());
        }
        trainingParams.put("data_loader_workers", trainingConfig.getDataLoaderWorkers());
        trainingParams.put("in_memory_dataset", trainingConfig.getInMemoryDataset());
        // "disabled" as the metric means the user wants to train for the full
        // epoch count -- turn off the EarlyStopping instance on the Python side.
        boolean earlyStoppingEnabled =
                !"disabled".equalsIgnoreCase(trainingConfig.getEarlyStoppingMetric());
        trainingParams.put("early_stopping", earlyStoppingEnabled);
        trainingParams.put("early_stopping_patience", trainingConfig.getEarlyStoppingPatience());
        trainingParams.put("early_stopping_metric",
                earlyStoppingEnabled ? trainingConfig.getEarlyStoppingMetric() : "mean_iou");
        trainingParams.put("mixed_precision", trainingConfig.isMixedPrecision());
        trainingParams.put("fused_optimizer", trainingConfig.isFusedOptimizer());
        trainingParams.put("use_lr_finder", trainingConfig.isUseLrFinder());
        trainingParams.put("gpu_augmentation", trainingConfig.isGpuAugmentation());
        trainingParams.put("use_torch_compile", trainingConfig.isUseTorchCompile());
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
        // Pass classifier name so it survives in checkpoints for recovery
        if (trainingConfig.getClassifierName() != null && !trainingConfig.getClassifierName().isEmpty()) {
            inputs.put("classifier_name", trainingConfig.getClassifierName());
        }

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

        // Training retry is gated on whether Python actually started executing.
        // If Appose returns "thread death" *before any progress update fires*,
        // the worker died on task receipt (stale worker from a prior job or
        // idle overlay task). No Python code ran, so retrying is safe and will
        // land on the fresh worker that Appose auto-launched. If we ever saw
        // a progress update, Python has a running training loop -- do NOT
        // retry, as that would create a duplicate GPU training process.
        try {
            for (int attempt = 0; attempt < 2; attempt++) {
                try {
                    return executeTrainingTask(appose, inputs, jobId, extensionCL,
                            progressCallback, cancelledCheck);
                } catch (IOException e) {
                    String msg = e.getMessage() != null ? e.getMessage() : "";
                    boolean isThreadDeath = msg.toLowerCase().contains("thread death");
                    boolean pythonStarted = (e instanceof TrainingStartedException tse)
                            && tse.pythonStarted;
                    if (isThreadDeath && !pythonStarted && attempt == 0) {
                        logger.warn("Training hit 'thread death' before Python started " +
                                "(stale worker); retrying on the replacement worker. " +
                                "Error: {}", msg);
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new IOException("Interrupted during training retry", ie);
                        }
                        continue;
                    }
                    throw e;
                }
            }
            throw new IOException("Training retry loop exited unexpectedly");
        } finally {
            Thread.currentThread().setContextClassLoader(originalCL);
        }
    }

    /**
     * IOException subclass signaling whether Python started executing the
     * training task before failing. Used to gate the pre-start retry safely.
     */
    private static final class TrainingStartedException extends IOException {
        final boolean pythonStarted;
        TrainingStartedException(String msg, Throwable cause, boolean pythonStarted) {
            super(msg, cause);
            this.pythonStarted = pythonStarted;
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
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback) throws IOException {
        return startPretraining(config, dataPath, outputDir,
                progressCallback, cancelledCheck, jobIdCallback, null);
    }

    public ClassifierClient.TrainingResult startPretraining(
            Map<String, Object> config,
            Path dataPath,
            Path outputDir,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback,
            Supplier<String> cancelSaveModeSupplier) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        // Inject pause signal path into the config dict. Python reads
        // pause_signal_path from config.get(...), then on detection saves
        // pause_checkpoint.pt and returns status="paused".
        String jobId = "appose-mae-" + System.currentTimeMillis();
        Map<String, Object> configWithPause = new HashMap<>(config);
        configWithPause.put("pause_signal_path", getPauseSignalPath(jobId).toString());
        configWithPause.put("cancel_signal_path", getCancelSignalPath(jobId).toString());

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("config", configWithPause);
        inputs.put("data_path", dataPath.toString());
        inputs.put("output_dir", outputDir.toString());

        logger.info("Starting MAE pretraining: jobId={}, config={}, data={}, output={}",
                jobId, config.get("model_config"), dataPath, outputDir);
        if (jobIdCallback != null) {
            jobIdCallback.accept(jobId);
        }

        // See SSL pretraining for the rationale; MAE shares the same race
        // (worker dies on dispatch -> Java throws -> finally block deletes the
        // temp tile dir while Appose's silent auto-retry is still running).
        return runPretrainingWithThreadDeathRetry(appose, "pretrain_mae", inputs,
                jobId, "mae", outputDir, progressCallback, cancelledCheck,
                cancelSaveModeSupplier, getCancelSignalPath(jobId));
    }

    /**
     * Reads pretraining task outputs and produces a TrainingResult, handling
     * paused / cancelled / completed states uniformly across MAE and SSL.
     * Persists checkpoint info in checkpointStore on pause so resume/finalize
     * can find the saved state.
     */
    private ClassifierClient.TrainingResult buildPretrainingResult(
            Task task, String jobId, String label,
            Map<String, Object> inputs, Path outputDir) {
        String status = task.outputs.containsKey("status")
                ? String.valueOf(task.outputs.get("status")) : "completed";
        String encoderPath = task.outputs.containsKey("encoder_path")
                ? task.outputs.get("encoder_path").toString() : "";
        int epochsCompleted = task.outputs.containsKey("epochs_completed")
                ? ((Number) task.outputs.get("epochs_completed")).intValue() : 0;
        double finalLoss = task.outputs.containsKey("final_loss")
                ? ((Number) task.outputs.get("final_loss")).doubleValue() : 0.0;
        // Python may report a quality assessment ("ok", "warn", "likely_collapse",
        // "aborted_collapse") plus a list of human-readable warnings. Plumb both
        // through so the completion dialog can flag problems instead of silently
        // saving a degenerate encoder.
        String quality = task.outputs.containsKey("quality")
                ? String.valueOf(task.outputs.get("quality")) : "ok";
        java.util.List<String> warnings = extractStringList(task.outputs.get("warnings"));
        if ("aborted_collapse".equals(status) && "ok".equals(quality)) {
            // Defensive: if the abort happened but the quality field wasn't set,
            // tag it so downstream UI still flags the run as suspect.
            quality = "likely_collapse";
        }

        if ("paused".equals(status)) {
            // Python wrote pause_checkpoint.pt to outputDir; record it for resume/finalize.
            Path checkpointPath = outputDir.resolve("pause_checkpoint.pt");
            int totalEpochs = task.outputs.containsKey("total_epochs")
                    ? ((Number) task.outputs.get("total_epochs")).intValue()
                    : epochsCompleted;
            storeCheckpointInfo(jobId, checkpointPath.toString(), epochsCompleted, inputs);
            logger.info("{} pretraining paused: jobId={}, epoch={}/{}, checkpoint={}",
                    label.toUpperCase(), jobId, epochsCompleted, totalEpochs, checkpointPath);
            return new ClassifierClient.TrainingResult(
                    jobId, null, finalLoss, 0.0, 0, 0.0,
                    true, epochsCompleted, totalEpochs, checkpointPath.toString(),
                    false, null, null, 0.0, true,
                    quality, warnings);
        }

        logger.info("{} pretraining {}: {} epochs, loss={}, path={}, quality={}, warnings={}",
                label.toUpperCase(), status, epochsCompleted, finalLoss, encoderPath,
                quality, warnings.size());

        // Mark cancelled-with-save runs as cancelled in the TrainingResult so
        // SetupDLClassifier shows a "cancelled, partial encoder saved" dialog
        // instead of the celebratory completion dialog.
        boolean cancelled = "cancelled_saved".equals(status)
                || "cancelled".equals(status);
        return new ClassifierClient.TrainingResult(
                jobId, encoderPath, finalLoss, 0.0, 0, 0.0,
                false, epochsCompleted, epochsCompleted, null,
                cancelled, null, null, 0.0, true,
                quality, warnings);
    }

    /**
     * Coerces a task.outputs value (which may be null, a List, or a single
     * string from older Python builds) into a List&lt;String&gt;. Defensive against
     * malformed payloads -- never throws.
     */
    private static java.util.List<String> extractStringList(Object raw) {
        if (raw == null) {
            return java.util.Collections.emptyList();
        }
        if (raw instanceof java.util.List<?> list) {
            java.util.List<String> out = new java.util.ArrayList<>(list.size());
            for (Object item : list) {
                if (item != null) {
                    out.add(item.toString());
                }
            }
            return out;
        }
        return java.util.List.of(raw.toString());
    }

    /**
     * Run SSL (SimCLR/BYOL) self-supervised pretraining on SMP encoder backbone.
     *
     * @param config           SSL method and training hyperparameters
     * @param dataPath         directory of unlabeled image tiles
     * @param outputDir        where to save the pretrained encoder weights
     * @param progressCallback receives progress updates during pretraining
     * @param cancelledCheck   returns true if pretraining should be cancelled
     * @return result containing encoder_path and training metrics
     * @throws IOException if pretraining fails
     */
    public ClassifierClient.TrainingResult startSSLPretraining(
            Map<String, Object> config,
            Path dataPath,
            Path outputDir,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback) throws IOException {
        return startSSLPretraining(config, dataPath, outputDir,
                progressCallback, cancelledCheck, jobIdCallback, null);
    }

    public ClassifierClient.TrainingResult startSSLPretraining(
            Map<String, Object> config,
            Path dataPath,
            Path outputDir,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback,
            Supplier<String> cancelSaveModeSupplier) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        String jobId = "appose-ssl-" + System.currentTimeMillis();
        Map<String, Object> configWithPause = new HashMap<>(config);
        configWithPause.put("pause_signal_path", getPauseSignalPath(jobId).toString());
        configWithPause.put("cancel_signal_path", getCancelSignalPath(jobId).toString());

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("config", configWithPause);
        inputs.put("data_path", dataPath.toString());
        inputs.put("output_dir", outputDir.toString());

        logger.info("Starting SSL pretraining: jobId={}, method={}, encoder={}, data={}, output={}",
                jobId, config.get("method"), config.get("encoder_name"), dataPath, outputDir);
        if (jobIdCallback != null) {
            jobIdCallback.accept(jobId);
        }

        // Retry once if Appose's worker dies before Python starts. Without this,
        // the SSL/MAE poll loop bails on the FAILURE, the SetupDLClassifier
        // finally-block wipes the temp tiles directory, and Appose's silent
        // auto-retry then runs Python on an empty tile dir (1 image left ->
        // BatchNorm-on-batch-of-1 crash). Mirrors the supervised retry at
        // executeTrainingTask().
        return runPretrainingWithThreadDeathRetry(appose, "pretrain_ssl", inputs,
                jobId, "ssl", outputDir, progressCallback, cancelledCheck,
                cancelSaveModeSupplier, getCancelSignalPath(jobId));
    }

    /**
     * Polls a pretraining task to completion. If the task fails with
     * "thread death" before Python ever emits a progress event (i.e. the
     * Appose worker died on receipt), recreates the task once with the same
     * inputs and retries on a fresh worker. Returns the resulting
     * TrainingResult once the (possibly retried) task finishes successfully
     * or with a non-recoverable failure.
     */
    private ClassifierClient.TrainingResult runPretrainingWithThreadDeathRetry(
            ApposeService appose,
            String taskName,
            Map<String, Object> inputs,
            String jobId,
            String label,
            Path outputDir,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException {
        return runPretrainingWithThreadDeathRetry(appose, taskName, inputs,
                jobId, label, outputDir, progressCallback, cancelledCheck,
                null, null);
    }

    private ClassifierClient.TrainingResult runPretrainingWithThreadDeathRetry(
            ApposeService appose,
            String taskName,
            Map<String, Object> inputs,
            String jobId,
            String label,
            Path outputDir,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Supplier<String> cancelSaveModeSupplier,
            Path cancelSignalPath) throws IOException {

        for (int attempt = 0; attempt < 2; attempt++) {
            Task task = appose.createTask(taskName, inputs);
            boolean[] pythonStarted = {false};
            boolean[] cancelSignalWritten = {false};

            task.listen(event -> {
                if (event.responseType == ResponseType.UPDATE && event.message != null) {
                    pythonStarted[0] = true;
                    try {
                        JsonObject json = JsonParser.parseString(event.message).getAsJsonObject();
                        Map<String, String> extraData = new HashMap<>();
                        if (json.has("elapsed_sec"))
                            extraData.put("elapsed_sec", json.get("elapsed_sec").getAsString());
                        if (json.has("remaining_sec"))
                            extraData.put("remaining_sec", json.get("remaining_sec").getAsString());
                        if (json.has("epoch_sec"))
                            extraData.put("epoch_sec", json.get("epoch_sec").getAsString());
                        if (json.has("images_per_sec"))
                            extraData.put("images_per_sec", json.get("images_per_sec").getAsString());
                        if (json.has("config") && json.get("config").isJsonObject()) {
                            JsonObject cfg = json.getAsJsonObject("config");
                            if (cfg.has("message"))
                                extraData.put("message", cfg.get("message").getAsString());
                        }

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
                                        extraData
                                );
                        if (progressCallback != null) {
                            progressCallback.accept(progress);
                        }
                    } catch (Exception e) {
                        logger.warn("Failed to parse {} pretraining progress: {}",
                                label, e.getMessage());
                    }
                }
            });

            task.start();
            while (!task.status.isFinished()) {
                if (cancelledCheck != null && cancelledCheck.get()) {
                    // Write the chosen save mode ("best" / "last" / "none")
                    // to the cancel signal file BEFORE telling Appose to
                    // cancel. Python's watcher polls this path and reads
                    // the mode string when handling cancellation, so the
                    // user's pick from the JavaFX dialog is honored. We
                    // only write once per task to avoid spamming the
                    // filesystem on every poll cycle. If no supplier is
                    // wired (older callers), we still touch the file with
                    // "best" so Python's polling has a faster signal than
                    // Appose's task.cancel_requested round-trip.
                    if (!cancelSignalWritten[0] && cancelSignalPath != null) {
                        String mode = "best";
                        if (cancelSaveModeSupplier != null) {
                            try {
                                String supplied = cancelSaveModeSupplier.get();
                                if (supplied != null && !supplied.isBlank()) {
                                    mode = supplied;
                                }
                            } catch (Exception ex) {
                                logger.debug("cancelSaveModeSupplier threw {}; "
                                        + "defaulting mode=best", ex.toString());
                            }
                        }
                        try {
                            Files.writeString(cancelSignalPath, mode);
                            logger.info(
                                    "{} pretraining cancel signal written: "
                                    + "mode={} path={}",
                                    label.toUpperCase(), mode, cancelSignalPath);
                            cancelSignalWritten[0] = true;
                        } catch (IOException io) {
                            logger.warn("Failed to write cancel signal {}: {}",
                                    cancelSignalPath, io.toString());
                        }
                    }
                    task.cancel();
                    logger.info("{} pretraining cancelled by user", label.toUpperCase());
                }
                try {
                    // Tighter polling than the previous 200 ms reduces the
                    // worst-case lag between cancel click and the cancel
                    // signal being visible to Python. Python's watcher
                    // polls at 100 ms now too, so end-to-end click->break
                    // is bounded by ~200 ms plus the current batch.
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    task.cancel();
                    throw new IOException(
                            label.toUpperCase() + " pretraining interrupted", e);
                }
            }

            if (task.status == org.apposed.appose.Service.TaskStatus.FAILED) {
                String err = task.error == null ? "" : task.error;
                boolean isThreadDeath = err.toLowerCase().contains("thread death");
                if (isThreadDeath && !pythonStarted[0] && attempt == 0) {
                    logger.warn("{} pretraining hit 'thread death' before Python started "
                                    + "(stale Appose worker); retrying on a fresh task. "
                                    + "Error: {}", label.toUpperCase(), err);
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new IOException(
                                "Interrupted during " + label + " pretraining retry", ie);
                    }
                    continue;
                }
                throw new IOException(
                        label.toUpperCase() + " pretraining failed: " + err);
            }

            return buildPretrainingResult(task, jobId, label, inputs, outputDir);
        }
        throw new IOException(
                label.toUpperCase() + " pretraining retry loop exited unexpectedly");
    }

    /**
     * Re-launches a previously paused pretraining run from its saved
     * pause_checkpoint.pt. Reuses the original inputs but injects
     * checkpoint_path and a fresh pause_signal_path.
     *
     * @param taskName        Appose script name ("pretrain_mae" or "pretrain_ssl")
     * @param jobId           original (paused) job ID; used to look up checkpoint info
     * @param outputDir       where the previous run was writing
     * @param progressCallback receives progress updates
     * @param cancelledCheck   cancellation poll
     * @param jobIdCallback    invoked with the new resume job ID
     */
    public ClassifierClient.TrainingResult resumePretraining(
            String taskName,
            String jobId,
            Path outputDir,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback) throws IOException {

        CheckpointInfo checkpoint = checkpointStore.get(jobId);
        if (checkpoint == null) {
            throw new IOException("No pretraining checkpoint stored for job: " + jobId);
        }

        ApposeService appose = ApposeService.getInstance();

        // Rebuild inputs from the original run, swapping in the new pause path
        // and checkpoint path so the Python service knows to resume.
        Map<String, Object> inputs = new HashMap<>(checkpoint.originalInputs());
        @SuppressWarnings("unchecked")
        Map<String, Object> previousConfig = (Map<String, Object>) inputs.get("config");
        Map<String, Object> resumeConfig = new HashMap<>(
                previousConfig != null ? previousConfig : new HashMap<>());

        String newJobId = "appose-resume-" + System.currentTimeMillis();
        resumeConfig.put("pause_signal_path", getPauseSignalPath(newJobId).toString());
        resumeConfig.put("checkpoint_path", checkpoint.path());
        resumeConfig.put("start_epoch", checkpoint.lastEpoch());
        inputs.put("config", resumeConfig);

        jobIdRedirects.put(jobId, newJobId);
        logger.info("Resuming pretraining: task={}, newJobId={}, checkpoint={}, fromEpoch={}",
                taskName, newJobId, checkpoint.path(), checkpoint.lastEpoch());
        if (jobIdCallback != null) {
            jobIdCallback.accept(newJobId);
        }

        Task task = appose.createTask(taskName, inputs);
        task.listen(event -> {
            if (event.responseType == ResponseType.UPDATE && event.message != null) {
                try {
                    JsonObject json = JsonParser.parseString(event.message).getAsJsonObject();
                    Map<String, String> extraData = new HashMap<>();
                    if (json.has("config") && json.get("config").isJsonObject()) {
                        JsonObject cfg = json.getAsJsonObject("config");
                        if (cfg.has("message"))
                            extraData.put("message", cfg.get("message").getAsString());
                    }
                    ClassifierClient.TrainingProgress progress =
                            new ClassifierClient.TrainingProgress(
                                    json.has("epoch") ? json.get("epoch").getAsInt() : 0,
                                    json.has("total_epochs") ? json.get("total_epochs").getAsInt() : 0,
                                    json.has("train_loss") ? json.get("train_loss").getAsDouble() : 0.0,
                                    json.has("val_loss") ? json.get("val_loss").getAsDouble() : 0.0,
                                    0.0, 0.0, Map.of(), Map.of(),
                                    json.has("device") ? json.get("device").getAsString() : "",
                                    json.has("device_info") ? json.get("device_info").getAsString() : "",
                                    json.has("status") ? json.get("status").getAsString() : "",
                                    json.has("setup_phase") ? json.get("setup_phase").getAsString() : "",
                                    extraData);
                    if (progressCallback != null) {
                        progressCallback.accept(progress);
                    }
                } catch (Exception e) {
                    logger.debug("Failed to parse resume progress: {}", e.getMessage());
                }
            }
        });

        task.start();
        while (!task.status.isFinished()) {
            if (cancelledCheck != null && cancelledCheck.get()) {
                task.cancel();
                logger.info("Resumed pretraining cancelled by user");
            }
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                task.cancel();
                throw new IOException("Resumed pretraining interrupted", e);
            }
        }
        if (task.status == org.apposed.appose.Service.TaskStatus.FAILED) {
            throw new IOException("Resumed pretraining failed: " + task.error);
        }
        return buildPretrainingResult(task, newJobId,
                "pretrain_ssl".equals(taskName) ? "ssl" : "mae",
                inputs, outputDir);
    }

    /**
     * Finalizes a paused pretraining run by extracting the best encoder weights
     * from pause_checkpoint.pt and writing them as the final model.pt + metadata.json.
     *
     * @param checkpointPath path to pause_checkpoint.pt
     * @param outputDir      directory to write model.pt and metadata.json
     * @return a TrainingResult whose modelPath points at the saved encoder
     */
    public ClassifierClient.TrainingResult finalizePretraining(
            String checkpointPath, Path outputDir) throws IOException {
        ApposeService appose = ApposeService.getInstance();
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", checkpointPath);
        inputs.put("output_dir", outputDir.toString());

        ClassLoader extensionCL = ApposeService.class.getClassLoader();
        ClassLoader originalCL = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(extensionCL);
        try {
            Task task = appose.runTask("finalize_pretrain", inputs);
            String encoderPath = task.outputs.containsKey("encoder_path")
                    ? String.valueOf(task.outputs.get("encoder_path")) : "";
            double bestLoss = task.outputs.containsKey("best_loss")
                    ? ((Number) task.outputs.get("best_loss")).doubleValue() : 0.0;
            return new ClassifierClient.TrainingResult(
                    "pretrain-finalized", encoderPath, bestLoss, 0.0, 0, 0.0);
        } catch (Exception e) {
            throw new IOException("Failed to finalize pretraining: " + e.getMessage(), e);
        } finally {
            Thread.currentThread().setContextClassLoader(originalCL);
        }
    }

    @Override
    public void pauseTraining(String jobId) throws IOException {
        // Follow redirect chain: after resume, the old jobId maps to a new one
        // with a different pause signal path.
        String effectiveId = jobId;
        String redirect = jobIdRedirects.get(effectiveId);
        while (redirect != null) {
            effectiveId = redirect;
            redirect = jobIdRedirects.get(effectiveId);
        }
        Path signalPath = getPauseSignalPath(effectiveId);
        Files.writeString(signalPath, "pause");
        logger.info("Pause signal written for job {} (effective: {})", jobId, effectiveId);
    }

    /**
     * Returns the file path used as a pause signal for the given job.
     * The Python train.py script polls for this file's existence.
     */
    private Path getPauseSignalPath(String jobId) {
        return Path.of(System.getProperty("java.io.tmpdir"), "dl-pause-" + jobId);
    }

    /**
     * Returns the file path used as a cancel signal for the given pretraining
     * job. Java writes the chosen save mode ("best", "last", "none") to this
     * file when the user cancels; Python's watcher polls the path and reads
     * the content to decide whether to save the best epoch, the last epoch,
     * or skip saving entirely. Without this signal the Python side can only
     * see Appose's task.cancel_requested boolean and cannot tell which mode
     * the user picked in the JavaFX dialog.
     */
    private Path getCancelSignalPath(String jobId) {
        return Path.of(System.getProperty("java.io.tmpdir"), "dl-cancel-" + jobId);
    }

    @Override
    public ClassifierClient.TrainingResult resumeTraining(
            String jobId,
            Path newDataPath,
            Integer epochs,
            Double learningRate,
            Integer batchSize,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback) throws IOException {

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

        // Store old -> new job ID mapping so pauseTraining() can find the
        // correct pause signal path even if called with the old job ID.
        jobIdRedirects.put(jobId, newJobId);
        logger.info("Resume: new jobId={}, mapped from old={}", newJobId, jobId);
        if (jobIdCallback != null) {
            jobIdCallback.accept(newJobId);
        }

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

        // Track whether Python ever emitted a progress update. Used by the
        // caller to decide if a "thread death" failure is safely retryable
        // (no progress = worker died before any training code ran).
        final boolean[] pythonStarted = {false};

        // Listen for progress events
        task.listen(event -> {
            if (event.responseType == ResponseType.UPDATE && event.message != null) {
                pythonStarted[0] = true;
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
                // Python saves both best-epoch and last-epoch models on cancel.
                // Read all available outputs for the cancel result.
                String cancelBestPath = null;
                String cancelLastPath = null;
                int cancelBestEpoch = 0;
                double cancelBestMIoU = 0.0;
                double cancelLoss = 0.0;
                double cancelAcc = 0.0;
                int cancelLastEpoch = 0;
                int cancelTotalEpochs = 0;
                String cancelCheckpoint = null;
                if (task.outputs != null) {
                    String mp = String.valueOf(task.outputs.getOrDefault("model_path", ""));
                    if (!mp.isEmpty() && !"null".equals(mp)) cancelBestPath = mp;
                    String lp = String.valueOf(task.outputs.getOrDefault("last_model_path", ""));
                    if (!lp.isEmpty() && !"null".equals(lp)) cancelLastPath = lp;
                    cancelBestEpoch = task.outputs.containsKey("best_epoch")
                            ? ((Number) task.outputs.get("best_epoch")).intValue() : 0;
                    cancelBestMIoU = task.outputs.containsKey("best_mean_iou")
                            ? ((Number) task.outputs.get("best_mean_iou")).doubleValue() : 0.0;
                    cancelLoss = task.outputs.containsKey("final_loss")
                            ? ((Number) task.outputs.get("final_loss")).doubleValue() : 0.0;
                    cancelAcc = task.outputs.containsKey("final_accuracy")
                            ? ((Number) task.outputs.get("final_accuracy")).doubleValue() : 0.0;
                    cancelLastEpoch = task.outputs.containsKey("last_epoch")
                            ? ((Number) task.outputs.get("last_epoch")).intValue() : 0;
                    cancelTotalEpochs = task.outputs.containsKey("total_epochs")
                            ? ((Number) task.outputs.get("total_epochs")).intValue() : 0;
                    String cp = String.valueOf(task.outputs.getOrDefault("checkpoint_path", ""));
                    if (!cp.isEmpty() && !"null".equals(cp)) cancelCheckpoint = cp;
                }
                if (cancelBestPath != null) {
                    logger.info("Cancel with save: best model={}, last model={}, epoch {}/{}",
                            cancelBestPath, cancelLastPath, cancelLastEpoch, cancelTotalEpochs);
                }
                // Extract focus class info from cancel result
                String cancelFocusName = null;
                double cancelFocusIoU = 0.0;
                boolean cancelFocusMet = true;
                if (task.outputs != null && task.outputs.containsKey("focus_class_name")) {
                    cancelFocusName = String.valueOf(task.outputs.get("focus_class_name"));
                    cancelFocusIoU = task.outputs.containsKey("focus_class_iou")
                            ? ((Number) task.outputs.get("focus_class_iou")).doubleValue() : 0.0;
                    cancelFocusMet = task.outputs.containsKey("focus_class_target_met")
                            ? (Boolean) task.outputs.get("focus_class_target_met") : true;
                }
                return new ClassifierClient.TrainingResult(
                        jobId, cancelBestPath, cancelLoss, cancelAcc,
                        cancelBestEpoch, cancelBestMIoU, false,
                        cancelLastEpoch, cancelTotalEpochs, cancelCheckpoint,
                        true, cancelLastPath,
                        cancelFocusName, cancelFocusIoU, cancelFocusMet);
            }
            // Wrap so the caller can distinguish "worker died before Python ran"
            // (safe to retry) from "Python was running training" (NOT safe --
            // risks a duplicate GPU training process).
            throw new TrainingStartedException("Training failed: " + task.error,
                    e, pythonStarted[0]);
        }

        // Check if training was paused (not cancelled, not failed)
        String status = String.valueOf(task.outputs.getOrDefault("status", "completed"));
        if ("paused".equals(status)) {
            String checkpointPath = String.valueOf(task.outputs.getOrDefault("checkpoint_path", ""));
            int lastEpoch = ((Number) task.outputs.getOrDefault("last_epoch", 0)).intValue();
            int totalEpochs = ((Number) task.outputs.getOrDefault("total_epochs", 0)).intValue();
            int bestEpoch = ((Number) task.outputs.getOrDefault("best_epoch", 0)).intValue();
            double bestMeanIoU = ((Number) task.outputs.getOrDefault("best_mean_iou", 0.0)).doubleValue();
            double finalLoss = ((Number) task.outputs.getOrDefault("final_loss", 0.0)).doubleValue();
            double finalAccuracy = ((Number) task.outputs.getOrDefault("final_accuracy", 0.0)).doubleValue();
            logger.info("Training paused at epoch {}/{}, best epoch {} (mIoU={}), checkpoint: {}",
                    lastEpoch, totalEpochs, bestEpoch, bestMeanIoU, checkpointPath);

            // Store checkpoint info for resume/finalize
            storeCheckpointInfo(jobId, checkpointPath, lastEpoch, inputs);

            return new ClassifierClient.TrainingResult(
                    jobId, null, finalLoss, finalAccuracy, bestEpoch, bestMeanIoU,
                    true, lastEpoch, totalEpochs, checkpointPath,
                    false, null, null, 0.0, true);
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

        // Extract focus class IoU if present (null when no focus class was set)
        String focusClassName = task.outputs.containsKey("focus_class_name")
                ? String.valueOf(task.outputs.get("focus_class_name")) : null;
        double focusClassIoU = task.outputs.containsKey("focus_class_iou")
                ? ((Number) task.outputs.get("focus_class_iou")).doubleValue() : 0.0;
        boolean focusClassTargetMet = task.outputs.containsKey("focus_class_target_met")
                ? (Boolean) task.outputs.get("focus_class_target_met") : true;

        return new ClassifierClient.TrainingResult(jobId, modelPath, finalLoss, finalAccuracy,
                bestEpoch, bestMeanIoU, false, lastEpoch, totalEpochs, completionCheckpoint,
                false, null, focusClassName, focusClassIoU, focusClassTargetMet);
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
        int missingDisagree = 0;
        int missingLoss = 0;
        int missingTile = 0;
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
            String predictionMapPath = obj.has("prediction_map") && !obj.get("prediction_map").isJsonNull()
                    ? obj.get("prediction_map").getAsString() : null;
            String confidenceMapPath = obj.has("confidence_map") && !obj.get("confidence_map").isJsonNull()
                    ? obj.get("confidence_map").getAsString() : null;
            String groundTruthMaskPath = obj.has("ground_truth_mask") && !obj.get("ground_truth_mask").isJsonNull()
                    ? obj.get("ground_truth_mask").getAsString() : null;
            if (disagreementImagePath == null) missingDisagree++;
            if (lossHeatmapPath == null) missingLoss++;
            if (tileImagePath == null) missingTile++;

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
                    tileImagePath,
                    predictionMapPath,
                    confidenceMapPath,
                    groundTruthMaskPath
            ));
        }

        int total = results.size();
        if (total > 0 && (missingDisagree > 0 || missingLoss > 0 || missingTile > 0)) {
            logger.warn("Evaluation produced {} tiles; missing PNGs: "
                    + "disagreement={}, loss={}, tile={}. These tiles will "
                    + "show no overlay in the Training Area Issues dialog.",
                    total, missingDisagree, missingLoss, missingTile);
        } else if (total > 0) {
            logger.info("Evaluation produced {} tiles, all with loss/disagreement/tile PNGs",
                    total);
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
                        if (inferenceConfig.isUseCompactArgmaxOutput()) {
                            inputs.put("output_format", "argmax_uint8");
                        }
                        if (DLClassifierPreferences.isExperimentalTensorRT()) {
                            inputs.put("use_tensorrt", true);
                        }
                        if (DLClassifierPreferences.isExperimentalInt8()) {
                            inputs.put("use_int8", true);
                        }

                        Task task = appose.runTask("inference_pixel", inputs);

                        int numClasses = ((Number) task.outputs.get("num_classes")).intValue();
                        NDArray resultNd = (NDArray) task.outputs.get("probabilities");

                        try {
                            // Read result from shared memory and save as .bin file
                            // (matching the existing file-based contract for
                            // DLPixelClassifier). Phase 3c: content is either
                            // (C,H,W) float32 probabilities or (H,W) uint8
                            // argmax -- Java knows which via inferenceConfig.
                            Files.createDirectories(outputDir);
                            Path outputPath = outputDir.resolve(tileId + ".bin");

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
                    if (inferenceConfig.isUseCompactArgmaxOutput()) {
                        inputs.put("output_format", "argmax_uint8");
                    }
                    if (DLClassifierPreferences.isExperimentalTensorRT()) {
                        inputs.put("use_tensorrt", true);
                    }
                    if (DLClassifierPreferences.isExperimentalInt8()) {
                        inputs.put("use_int8", true);
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
                    // Pass experimental provider flags so the
                    // measurements path does not silently inherit
                    // stale provider state from the most recent
                    // pixel inference. Matches the pixel-path wiring.
                    inputs.put("use_tensorrt",
                            DLClassifierPreferences.isExperimentalTensorRT());
                    inputs.put("use_int8",
                            DLClassifierPreferences.isExperimentalInt8());

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

        // Training config summary (present in "training_config" and "training_batch" setup phases).
        // Values may be strings or numbers, so use getAsString() for primitives (Gson
        // converts numbers to their string representation).
        Map<String, String> configSummary = new LinkedHashMap<>();
        if (obj.has("config") && !obj.get("config").isJsonNull()) {
            JsonObject cfg = obj.getAsJsonObject("config");
            for (String key : cfg.keySet()) {
                var element = cfg.get(key);
                if (element.isJsonPrimitive()) {
                    configSummary.put(key, element.getAsString());
                } else {
                    configSummary.put(key, element.toString());
                }
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

    /** Coerces an Object (Number, String, or null) to a double with fallback. */
    private static double toDouble(Object value, double fallback) {
        if (value instanceof Number n) return n.doubleValue();
        if (value instanceof String s) {
            try { return Double.parseDouble(s); } catch (NumberFormatException ignored) {}
        }
        return fallback;
    }

    /** Coerces an Object (Number, String, or null) to an int with fallback. */
    private static int toInt(Object value, int fallback) {
        if (value instanceof Number n) return n.intValue();
        if (value instanceof String s) {
            try { return Integer.parseInt(s); } catch (NumberFormatException ignored) {}
        }
        return fallback;
    }
}
