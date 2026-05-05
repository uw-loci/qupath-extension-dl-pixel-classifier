package qupath.ext.dlclassifier.scripting;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.InferenceConfig.ApplicationScope;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.service.ApposeService;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * Generates runnable Groovy scripts from dialog configuration values.
 * <p>
 * The two public entry points produce a self-contained Groovy script that can
 * be pasted into QuPath's Script Editor and executed without the original
 * dialog GUI:
 * <ul>
 *   <li>{@link #generateTrainingScript} - emits a {@code TrainingConfig} builder
 *       chain wired into a {@code TrainingWorkflow.run()} call.</li>
 *   <li>{@link #generateInferenceScript} - emits an {@code InferenceConfig}
 *       builder chain wired into an {@code InferenceWorkflow.run()} call.</li>
 * </ul>
 *
 * <h2>Audit-driven invariant</h2>
 * Every {@code TrainingConfig.Builder} / {@code InferenceConfig.Builder} method
 * that has a corresponding getter on the config object is reachable here. The
 * top-level methods read as a sequence of {@code appendXxx} group helpers; if
 * a future field is added on the config, exactly one of those helpers must be
 * extended so the audit stays exhaustive. New helpers should also bump the
 * "emitted / skipped" telemetry below.
 *
 * <h2>Conditional emission policy</h2>
 * Each builder line falls into one of three buckets:
 * <ul>
 *   <li><b>Always emitted</b> - core, always-meaningful fields whose default
 *       carries user intent (for example {@code epochs}, {@code learningRate},
 *       {@code weightDecay} where 0 is a legitimate user pick).</li>
 *   <li><b>Non-default only</b> - fields whose default value is "off" or
 *       "absent"; emitting "false"/"0"/empty would just add noise (for
 *       example {@code progressiveResize}, {@code wholeImage},
 *       {@code frozenLayers}).</li>
 *   <li><b>Context-dependent</b> - fields that are only meaningful given
 *       another setting; for example {@code focalGamma} is only emitted when
 *       the loss is a focal-family loss, and the OBJECTS post-processing
 *       block is only emitted for {@code outputType == OBJECTS}.</li>
 * </ul>
 * The exact rule is documented inline at each call site.
 *
 * <h2>Byte-identical output</h2>
 * The generator must produce the same output for the same logical input
 * regardless of host JVM. To that end:
 * <ul>
 *   <li>Numbers are formatted via {@link #formatDouble} which uses
 *       {@link Locale#ROOT} so the decimal separator is always {@code '.'}
 *       and the exponent (when present) is locale-independent.</li>
 *   <li>Map literals iterate in insertion order when the source is a
 *       {@link LinkedHashMap} or {@link SortedMap}; otherwise entries are
 *       copied through a {@link TreeMap} so unordered inputs still produce
 *       a stable, alphabetical key order.</li>
 *   <li>Trailing zeros are stripped from {@code %g} output so that doubles
 *       like {@code 2.0} render as {@code "2"} consistently.</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ScriptGenerator {

    private static final Logger logger = LoggerFactory.getLogger(ScriptGenerator.class);

    // ---- Indentation & line shape ----------------------------------------
    /** Indent for builder-method lines inside a {@code .builder()} chain. */
    private static final String BUILDER_INDENT = "        ";
    /** Indent for entries inside a multi-line map literal. */
    private static final String MAP_ENTRY_INDENT = "            ";

    // ---- Loss-function keywords ------------------------------------------
    // The dialog stores the loss as a lowercase string; we test substrings
    // because the focal/boundary families have multiple variants
    // ("focal", "focal_dice", "boundary_ce", "boundary_ce_dice", ...).
    private static final String LOSS_KEYWORD_FOCAL = "focal";
    private static final String LOSS_KEYWORD_BOUNDARY = "boundary";

    // ---- OHEM schedule defaults ------------------------------------------
    /** OHEM schedule value that disables ratio progression. */
    private static final String OHEM_SCHEDULE_FIXED = "fixed";
    /** OHEM ratio at which OHEM is effectively disabled (no example mining). */
    private static final double OHEM_RATIO_DISABLED = 1.0;

    // ---- Other "leave it default" sentinel values ------------------------
    /** {@code intensityAugMode} value meaning "do not apply intensity aug". */
    private static final String INTENSITY_AUG_NONE = "none";

    private ScriptGenerator() {
        // Utility class - not instantiable.
    }

    /**
     * Returns the extension version string embedded in the generated header.
     */
    private static String getExtensionVersion() {
        return ApposeService.DL_SERVER_VERSION;
    }

    // ====================================================================
    // Inference Script
    // ====================================================================

    /**
     * Generates a Groovy inference script from the current dialog settings.
     * <p>
     * The script structure is:
     * <ol>
     *   <li>Header comment + imports</li>
     *   <li>{@code DLClassifierScripts.loadClassifier(...)}</li>
     *   <li>{@code InferenceConfig.builder()...build()} (4 group helpers)</li>
     *   <li>{@code ChannelConfiguration.builder()...build()}</li>
     *   <li>Annotation collection - depends on {@code scope}</li>
     *   <li>{@code InferenceWorkflow.builder()...build().run()}</li>
     *   <li>Result reporting</li>
     * </ol>
     *
     * @param classifierId  the classifier ID to load
     * @param config        the inference configuration
     * @param channelConfig the channel configuration
     * @param scope         the application scope
     * @return a runnable Groovy script string
     */
    public static String generateInferenceScript(String classifierId,
                                                  InferenceConfig config,
                                                  ChannelConfiguration channelConfig,
                                                  ApplicationScope scope) {
        EmissionStats stats = new EmissionStats();
        StringBuilder sb = new StringBuilder();

        appendInferenceHeader(sb);
        appendInferenceClassifierLoad(sb, classifierId);
        appendInferenceConfigBlock(sb, config, stats);
        appendChannelConfig(sb, channelConfig);
        appendInferenceTargetRegions(sb, scope);
        appendInferenceWorkflowRun(sb);

        logger.info("Generated inference script for classifier '{}' (scope={}): emitted {} fields, skipped {} fields",
                classifierId, scope, stats.emitted, stats.skipped);
        if (logger.isDebugEnabled() && !stats.skippedNames.isEmpty()) {
            logger.debug("Inference script skipped fields: {}", stats.skippedNames);
        }
        return sb.toString();
    }

    /** Header comment + imports for the inference script. */
    private static void appendInferenceHeader(StringBuilder sb) {
        appendLine(sb, "/**");
        appendLine(sb, " * DL Pixel Classifier - Inference Script");
        appendLine(sb, " * Generated by DL Pixel Classifier v" + getExtensionVersion());
        appendLine(sb, " */");
        appendLine(sb, "import qupath.ext.dlclassifier.controller.InferenceWorkflow");
        appendLine(sb, "import qupath.ext.dlclassifier.model.*");
        appendLine(sb, "import qupath.ext.dlclassifier.scripting.DLClassifierScripts");
        appendLine(sb, "");
    }

    /** Emits the classifier-load block. */
    private static void appendInferenceClassifierLoad(StringBuilder sb, String classifierId) {
        appendLine(sb, "// Load classifier");
        appendLine(sb, "def classifier = DLClassifierScripts.loadClassifier(" + quote(classifierId) + ")");
        appendLine(sb, "");
    }

    /** Emits the {@code InferenceConfig.builder()} chain. */
    private static void appendInferenceConfigBlock(StringBuilder sb, InferenceConfig config, EmissionStats stats) {
        appendLine(sb, "// Configure inference");
        appendLine(sb, "def inferenceConfig = InferenceConfig.builder()");
        appendInferenceTilingFields(sb, config, stats);
        appendInferenceOutputFields(sb, config, stats);
        appendInferenceSmoothingFields(sb, config, stats);
        appendInferenceStrategyFields(sb, config, stats);
        appendLine(sb, BUILDER_INDENT + ".build()");
        appendLine(sb, "");
    }

    /**
     * Emits the script's "where to run inference" block. The shape of this
     * block depends entirely on {@code scope}; the inference config itself is
     * scope-agnostic.
     */
    private static void appendInferenceTargetRegions(StringBuilder sb, ApplicationScope scope) {
        appendLine(sb, "// Get target regions");
        switch (scope) {
            case WHOLE_IMAGE:
                // Full-image scope synthesizes a single annotation covering the
                // whole server, so downstream code stays uniform.
                appendLine(sb, "import qupath.lib.objects.PathObjects");
                appendLine(sb, "import qupath.lib.roi.ROIs");
                appendLine(sb, "import qupath.lib.regions.ImagePlane");
                appendLine(sb, "def server = getCurrentServer()");
                appendLine(sb, "def fullROI = ROIs.createRectangleROI(0, 0, server.getWidth(), server.getHeight(), ImagePlane.getDefaultPlane())");
                appendLine(sb, "def annotations = [PathObjects.createAnnotationObject(fullROI)]");
                break;
            case SELECTED_ANNOTATIONS:
                // Fall back to all annotations when the user has nothing
                // selected at the time the generated script runs.
                appendLine(sb, "def annotations = getSelectedObjects().findAll { it.isAnnotation() }");
                appendLine(sb, "if (annotations.isEmpty()) {");
                appendLine(sb, "    annotations = getAnnotationObjects()");
                appendLine(sb, "}");
                break;
            case ALL_ANNOTATIONS:
            default:
                appendLine(sb, "def annotations = getAnnotationObjects()");
                break;
        }
        appendLine(sb, "println \"Classifying ${annotations.size()} region(s)...\"");
        appendLine(sb, "");
    }

    /** Emits the workflow runner + result reporting at the bottom of the script. */
    private static void appendInferenceWorkflowRun(StringBuilder sb) {
        appendLine(sb, "// Run inference");
        appendLine(sb, "def result = InferenceWorkflow.builder()");
        appendLine(sb, BUILDER_INDENT + ".classifier(classifier)");
        appendLine(sb, BUILDER_INDENT + ".config(inferenceConfig)");
        appendLine(sb, BUILDER_INDENT + ".channels(channelConfig)");
        appendLine(sb, BUILDER_INDENT + ".annotations(annotations)");
        appendLine(sb, BUILDER_INDENT + ".build()");
        appendLine(sb, BUILDER_INDENT + ".run()");
        appendLine(sb, "");
        appendLine(sb, "println \"Done! Processed ${result.processedAnnotations()} annotations, ${result.processedTiles()} tiles\"");
        appendLine(sb, "if (!result.success()) {");
        appendLine(sb, "    println \"WARNING: ${result.message()}\"");
        appendLine(sb, "}");
    }

    // ====================================================================
    // Training Script
    // ====================================================================

    /**
     * Generates a Groovy training script from the current dialog settings.
     * <p>
     * The script structure is:
     * <ol>
     *   <li>Header comment + imports</li>
     *   <li>{@code TrainingConfig.builder()...build()} (10 group helpers,
     *       grouped by concept: architecture, duration, batch+tile, learning
     *       rate, loss+OHEM, augmentation, performance, transfer, focus,
     *       reproducibility)</li>
     *   <li>{@code ChannelConfiguration.builder()...build()}</li>
     *   <li>Selected class list</li>
     *   <li>{@code TrainingWorkflow.builder()...build().run()}</li>
     *   <li>Result reporting</li>
     * </ol>
     *
     * @param classifierName  the classifier name
     * @param description     the classifier description
     * @param config          the training configuration
     * @param channelConfig   the channel configuration
     * @param selectedClasses the selected class names
     * @return a runnable Groovy script string
     */
    public static String generateTrainingScript(String classifierName,
                                                 String description,
                                                 TrainingConfig config,
                                                 ChannelConfiguration channelConfig,
                                                 List<String> selectedClasses) {
        EmissionStats stats = new EmissionStats();
        StringBuilder sb = new StringBuilder();

        appendTrainingHeader(sb);
        appendTrainingConfigBlock(sb, config, stats);
        appendChannelConfig(sb, channelConfig);
        appendSelectedClasses(sb, selectedClasses);
        appendTrainingWorkflowRun(sb, classifierName, description);

        logger.info("Generated training script for classifier '{}' (type={}): emitted {} fields, skipped {} fields",
                classifierName, config.getModelType(), stats.emitted, stats.skipped);
        if (logger.isDebugEnabled() && !stats.skippedNames.isEmpty()) {
            logger.debug("Training script skipped fields: {}", stats.skippedNames);
        }
        return sb.toString();
    }

    /** Header comment + imports for the training script. */
    private static void appendTrainingHeader(StringBuilder sb) {
        appendLine(sb, "/**");
        appendLine(sb, " * DL Pixel Classifier - Training Script");
        appendLine(sb, " * Generated by DL Pixel Classifier v" + getExtensionVersion());
        appendLine(sb, " */");
        appendLine(sb, "import qupath.ext.dlclassifier.controller.TrainingWorkflow");
        appendLine(sb, "import qupath.ext.dlclassifier.model.*");
        appendLine(sb, "");
    }

    /** Emits the {@code TrainingConfig.builder()} chain via group helpers. */
    private static void appendTrainingConfigBlock(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, "// Configure training");
        appendLine(sb, "def trainingConfig = TrainingConfig.builder()");
        appendArchitectureFields(sb, config, stats);
        appendDurationFields(sb, config, stats);
        appendBatchAndTileFields(sb, config, stats);
        appendLearningRateFields(sb, config, stats);
        appendLossAndOhemFields(sb, config, stats);
        appendAugmentationFields(sb, config, stats);
        appendPerformanceFields(sb, config, stats);
        appendTransferLearningFields(sb, config, stats);
        appendFocusClassFields(sb, config, stats);
        appendReproducibilityFields(sb, config, stats);
        appendLine(sb, BUILDER_INDENT + ".build()");
        appendLine(sb, "");
    }

    /** Emits the {@code def selectedClasses = [...]} line. */
    private static void appendSelectedClasses(StringBuilder sb, List<String> selectedClasses) {
        appendLine(sb, "// Classes for training");
        appendLine(sb, "def selectedClasses = " + formatStringList(selectedClasses));
        appendLine(sb, "");
    }

    /** Emits the workflow runner + result reporting at the bottom of the script. */
    private static void appendTrainingWorkflowRun(StringBuilder sb, String classifierName, String description) {
        appendLine(sb, "// Run training");
        appendLine(sb, "def result = TrainingWorkflow.builder()");
        appendLine(sb, BUILDER_INDENT + ".name(" + quote(classifierName) + ")");
        // Description is optional in the dialog; only wire the builder call
        // when the user actually typed something so the script does not carry
        // an empty .description("") line.
        if (description != null && !description.isEmpty()) {
            appendLine(sb, BUILDER_INDENT + ".description(" + quote(description) + ")");
        }
        appendLine(sb, BUILDER_INDENT + ".config(trainingConfig)");
        appendLine(sb, BUILDER_INDENT + ".channels(channelConfig)");
        appendLine(sb, BUILDER_INDENT + ".classes(selectedClasses)");
        appendLine(sb, BUILDER_INDENT + ".build()");
        appendLine(sb, BUILDER_INDENT + ".run()");
        appendLine(sb, "");
        appendLine(sb, "println \"Training complete! Success: ${result.success()}\"");
        appendLine(sb, "if (result.success()) {");
        appendLine(sb, "    println \"Final loss: ${result.finalLoss()}, Accuracy: ${result.finalAccuracy()}\"");
        appendLine(sb, "    println \"Classifier ID: ${result.classifierId()}\"");
        appendLine(sb, "} else {");
        appendLine(sb, "    println \"Training failed: ${result.message()}\"");
        appendLine(sb, "}");
    }

    // ====================================================================
    // Training Field Groups
    // ====================================================================
    //
    // Each group helper handles one conceptual cluster of builder calls.
    // The Javadoc lists which builder methods it emits and any cross-field
    // dependencies. Section banners use the form
    //     // --- TITLE-CASED GROUP NAME ---
    // and the banner text below MUST stay byte-identical -- the dialog test
    // suite asserts on the rendered banners.

    /**
     * Emits {@code classifierType}, {@code backbone}, and (when non-empty)
     * {@code handlerParameters}. {@code handlerParameters} is an
     * architecture-specific map populated by panels such as the MuViT
     * config (model_config / patch_size); empty means "use defaults".
     */
    private static void appendArchitectureFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Architecture & Backbone ---");
        emit(sb, stats, "classifierType", builderCall("classifierType", quote(config.getModelType())));
        emit(sb, stats, "backbone", builderCall("backbone", quote(config.getBackbone())));

        Map<String, Object> handlerParams = config.getHandlerParameters();
        if (handlerParams != null && !handlerParams.isEmpty()) {
            emit(sb, stats, "handlerParameters", buildMapLiteral(BUILDER_INDENT + ".handlerParameters", handlerParams));
        } else {
            skip(stats, "handlerParameters");
        }
    }

    /**
     * Emits {@code epochs}, {@code earlyStoppingMetric},
     * {@code earlyStoppingPatience}, {@code progressiveResize} (only when
     * true), and {@code pretrainedModelPath} (only when set).
     */
    private static void appendDurationFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Training Duration & Checkpointing ---");
        emit(sb, stats, "epochs", builderCall("epochs", String.valueOf(config.getEpochs())));
        emit(sb, stats, "earlyStoppingMetric", builderCall("earlyStoppingMetric", quote(config.getEarlyStoppingMetric())));
        emit(sb, stats, "earlyStoppingPatience", builderCall("earlyStoppingPatience", String.valueOf(config.getEarlyStoppingPatience())));

        // Default false; emit only when the user opted in.
        if (config.isProgressiveResize()) {
            emit(sb, stats, "progressiveResize", builderCall("progressiveResize", "true"));
        } else {
            skip(stats, "progressiveResize");
        }

        // Empty path means "no warm-start checkpoint"; do not emit.
        String pretrainedPath = config.getPretrainedModelPath();
        if (pretrainedPath != null && !pretrainedPath.isEmpty()) {
            emit(sb, stats, "pretrainedModelPath", builderCall("pretrainedModelPath", quote(pretrainedPath)));
        } else {
            skip(stats, "pretrainedModelPath");
        }
    }

    /**
     * Emits {@code batchSize}, {@code gradientAccumulationSteps} (only when
     * &gt; 1, i.e. accumulation is actually doing something), {@code tileSize},
     * {@code overlap}, {@code downsample}, {@code wholeImage} (only when
     * true), and {@code trainingPixelSizeMicrons} (only when finite -- NaN
     * means the project is uncalibrated and there is no useful value).
     */
    private static void appendBatchAndTileFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Batch, Memory & Tile Geometry ---");
        emit(sb, stats, "batchSize", builderCall("batchSize", String.valueOf(config.getBatchSize())));

        int accumulationSteps = config.getGradientAccumulationSteps();
        if (accumulationSteps > 1) {
            emit(sb, stats, "gradientAccumulationSteps",
                    builderCall("gradientAccumulationSteps", String.valueOf(accumulationSteps)));
        } else {
            skip(stats, "gradientAccumulationSteps");
        }

        emit(sb, stats, "tileSize", builderCall("tileSize", String.valueOf(config.getTileSize())));
        emit(sb, stats, "overlap", builderCall("overlap", String.valueOf(config.getOverlap())));
        emit(sb, stats, "downsample", builderCall("downsample", formatDouble(config.getDownsample())));

        if (config.isWholeImage()) {
            emit(sb, stats, "wholeImage", builderCall("wholeImage", "true"));
        } else {
            skip(stats, "wholeImage");
        }

        double trainingMpp = config.getTrainingPixelSizeMicrons();
        if (Double.isFinite(trainingMpp)) {
            emit(sb, stats, "trainingPixelSizeMicrons",
                    builderCall("trainingPixelSizeMicrons", formatDouble(trainingMpp)));
        } else {
            skip(stats, "trainingPixelSizeMicrons");
        }
    }

    /**
     * Emits {@code learningRate}, {@code discriminativeLrRatio},
     * {@code weightDecay}, {@code useLrFinder}, {@code fusedOptimizer},
     * {@code schedulerType}.
     * <p>
     * All are emitted unconditionally so the script reflects user intent
     * verbatim and round-trips through later edits. In particular,
     * {@code weightDecay = 0} is a legitimate choice (the builder default
     * is 0.01), so a "&gt; 0" gate would silently corrupt user intent.
     */
    private static void appendLearningRateFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Learning Rate & Optimizer ---");
        emit(sb, stats, "learningRate", builderCall("learningRate", formatDouble(config.getLearningRate())));
        emit(sb, stats, "discriminativeLrRatio",
                builderCall("discriminativeLrRatio", formatDouble(config.getDiscriminativeLrRatio())));
        emit(sb, stats, "weightDecay", builderCall("weightDecay", formatDouble(config.getWeightDecay())));
        emit(sb, stats, "useLrFinder", builderCall("useLrFinder", String.valueOf(config.isUseLrFinder())));
        emit(sb, stats, "fusedOptimizer", builderCall("fusedOptimizer", String.valueOf(config.isFusedOptimizer())));
        emit(sb, stats, "schedulerType", builderCall("schedulerType", quote(config.getSchedulerType())));
    }

    /**
     * Emits {@code lossFunction}, the loss-family-specific parameters
     * ({@code focalGamma} for focal losses; {@code boundarySigma} and
     * {@code boundaryWMin} for boundary losses), and the OHEM block.
     * <p>
     * Cross-field dependencies:
     * <ul>
     *   <li>{@code focalGamma} is emitted only when {@code lossFunction}
     *       contains {@value #LOSS_KEYWORD_FOCAL}.</li>
     *   <li>{@code boundarySigma} / {@code boundaryWMin} are emitted only
     *       when {@code lossFunction} contains {@value #LOSS_KEYWORD_BOUNDARY}.</li>
     *   <li>OHEM ratio fields are emitted only when OHEM is engaged
     *       (see {@link #isOhemEngaged}).</li>
     * </ul>
     */
    private static void appendLossAndOhemFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Loss & OHEM ---");
        String loss = config.getLossFunction();
        emit(sb, stats, "lossFunction", builderCall("lossFunction", quote(loss)));

        appendLossFamilyParams(sb, config, stats, loss);
        appendOhemFields(sb, config, stats);
    }

    /**
     * Emits the loss-family-specific parameters that ride along with the
     * chosen loss. Skipped entirely when the loss keyword does not match.
     */
    private static void appendLossFamilyParams(StringBuilder sb, TrainingConfig config,
                                               EmissionStats stats, String loss) {
        // focalGamma is only meaningful for focal-family losses ("focal",
        // "focal_dice"); for any other loss the value is ignored at runtime.
        if (loss != null && loss.contains(LOSS_KEYWORD_FOCAL)) {
            emit(sb, stats, "focalGamma", builderCall("focalGamma", formatDouble(config.getFocalGamma())));
        } else {
            skip(stats, "focalGamma");
        }

        // boundarySigma / boundaryWMin only apply to boundary-weighted losses
        // ("boundary_ce", "boundary_ce_dice"). Always emit them when relevant
        // so user-tuned values like sigma=2 are not silently reverted to the
        // builder default 3.0 when the script runs.
        if (loss != null && loss.contains(LOSS_KEYWORD_BOUNDARY)) {
            emit(sb, stats, "boundarySigma",
                    builderCall("boundarySigma", formatDouble(config.getBoundarySigma())));
            emit(sb, stats, "boundaryWMin",
                    builderCall("boundaryWMin", formatDouble(config.getBoundaryWMin())));
        } else {
            skip(stats, "boundarySigma");
            skip(stats, "boundaryWMin");
        }
    }

    /**
     * Emits the OHEM-related builder calls. The whole ratio/schedule trio is
     * emitted as a unit only when OHEM is actually doing something; the
     * adaptive-floor toggle is a separate opt-in flag.
     */
    private static void appendOhemFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        if (isOhemEngaged(config)) {
            emit(sb, stats, "ohemHardRatio",
                    builderCall("ohemHardRatio", formatDouble(config.getOhemHardRatio())));
            emit(sb, stats, "ohemHardRatioStart",
                    builderCall("ohemHardRatioStart", formatDouble(config.getOhemHardRatioStart())));
            emit(sb, stats, "ohemSchedule",
                    builderCall("ohemSchedule", quote(config.getOhemSchedule())));
        } else {
            skip(stats, "ohemHardRatio");
            skip(stats, "ohemHardRatioStart");
            skip(stats, "ohemSchedule");
        }

        if (config.isOhemAdaptiveFloor()) {
            emit(sb, stats, "ohemAdaptiveFloor", builderCall("ohemAdaptiveFloor", "true"));
        } else {
            skip(stats, "ohemAdaptiveFloor");
        }
    }

    /**
     * OHEM is "engaged" when either the hard-example ratio is below 1.0
     * (mining only the hardest fraction) or the schedule is something other
     * than {@value #OHEM_SCHEDULE_FIXED} (so the ratio progresses over
     * training). Both defaults together (ratio 1.0, schedule "fixed") mean
     * OHEM is off.
     */
    private static boolean isOhemEngaged(TrainingConfig config) {
        return config.getOhemHardRatio() < OHEM_RATIO_DISABLED
                || !OHEM_SCHEDULE_FIXED.equals(config.getOhemSchedule());
    }

    /**
     * Emits {@code augmentation} (flag map), {@code augmentationParams}
     * (per-flag tuning map), {@code intensityAugMode} (only when not
     * {@value #INTENSITY_AUG_NONE}), {@code contextScale} (only when &gt; 1),
     * {@code lineStrokeWidth} (only when &gt; 0), and
     * {@code classWeightMultipliers} (only when non-empty).
     */
    private static void appendAugmentationFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Augmentation & Preprocessing ---");

        Map<String, Boolean> augFlags = config.getAugmentationConfig();
        if (augFlags != null && !augFlags.isEmpty()) {
            emit(sb, stats, "augmentation", buildMapLiteral(BUILDER_INDENT + ".augmentation", augFlags));
        } else {
            skip(stats, "augmentation");
        }

        Map<String, Object> augParams = config.getAugmentationParams();
        if (augParams != null && !augParams.isEmpty()) {
            emit(sb, stats, "augmentationParams", buildMapLiteral(BUILDER_INDENT + ".augmentationParams", augParams));
        } else {
            skip(stats, "augmentationParams");
        }

        // Default "none" means no intensity-domain augmentation; only emit
        // when the user picked a real mode (e.g. "brightfield", "fluorescence").
        String intensityMode = config.getIntensityAugMode();
        if (intensityMode != null && !INTENSITY_AUG_NONE.equals(intensityMode)) {
            emit(sb, stats, "intensityAugMode", builderCall("intensityAugMode", quote(intensityMode)));
        } else {
            skip(stats, "intensityAugMode");
        }

        if (config.getContextScale() > 1) {
            emit(sb, stats, "contextScale", builderCall("contextScale", String.valueOf(config.getContextScale())));
        } else {
            skip(stats, "contextScale");
        }

        if (config.getLineStrokeWidth() > 0) {
            emit(sb, stats, "lineStrokeWidth", builderCall("lineStrokeWidth", String.valueOf(config.getLineStrokeWidth())));
        } else {
            skip(stats, "lineStrokeWidth");
        }

        Map<String, Double> classWeights = config.getClassWeightMultipliers();
        if (classWeights != null && !classWeights.isEmpty()) {
            emit(sb, stats, "classWeightMultipliers",
                    buildMapLiteral(BUILDER_INDENT + ".classWeightMultipliers", classWeights));
        } else {
            skip(stats, "classWeightMultipliers");
        }
    }

    /**
     * Emits {@code mixedPrecision} (always - the default is true and turning
     * it off is a meaningful decision), {@code gpuAugmentation} /
     * {@code useTorchCompile} (opt-in toggles, off by default and
     * platform-dependent), {@code dataLoaderWorkers} (only when non-zero),
     * and {@code inMemoryDataset} (always - "auto"/"on"/"off" are all
     * meaningful explicit choices).
     */
    private static void appendPerformanceFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Performance & Precision ---");
        emit(sb, stats, "mixedPrecision", builderCall("mixedPrecision", String.valueOf(config.isMixedPrecision())));

        if (config.isGpuAugmentation()) {
            emit(sb, stats, "gpuAugmentation", builderCall("gpuAugmentation", "true"));
        } else {
            skip(stats, "gpuAugmentation");
        }

        if (config.isUseTorchCompile()) {
            emit(sb, stats, "useTorchCompile", builderCall("useTorchCompile", "true"));
        } else {
            skip(stats, "useTorchCompile");
        }

        if (config.getDataLoaderWorkers() != 0) {
            emit(sb, stats, "dataLoaderWorkers",
                    builderCall("dataLoaderWorkers", String.valueOf(config.getDataLoaderWorkers())));
        } else {
            skip(stats, "dataLoaderWorkers");
        }

        emit(sb, stats, "inMemoryDataset", builderCall("inMemoryDataset", quote(config.getInMemoryDataset())));
    }

    /**
     * Emits {@code usePretrainedWeights} (always),
     * {@code freezeEncoderLayers} (only when &gt; 0), and {@code frozenLayers}
     * (only when non-empty).
     */
    private static void appendTransferLearningFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Transfer Learning & Layer Freezing ---");
        emit(sb, stats, "usePretrainedWeights",
                builderCall("usePretrainedWeights", String.valueOf(config.isUsePretrainedWeights())));

        if (config.getFreezeEncoderLayers() > 0) {
            emit(sb, stats, "freezeEncoderLayers",
                    builderCall("freezeEncoderLayers", String.valueOf(config.getFreezeEncoderLayers())));
        } else {
            skip(stats, "freezeEncoderLayers");
        }

        List<String> frozenLayers = config.getFrozenLayers();
        if (frozenLayers != null && !frozenLayers.isEmpty()) {
            emit(sb, stats, "frozenLayers", builderCall("frozenLayers", formatStringList(frozenLayers)));
        } else {
            skip(stats, "frozenLayers");
        }
    }

    /**
     * Emits {@code focusClass} (only when set) and, nested inside that gate,
     * {@code focusClassMinIoU} (only when &gt; 0). When no focus class is
     * picked, both are skipped together so the script does not carry an
     * orphan threshold.
     */
    private static void appendFocusClassFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Focus Class ---");
        String focusClass = config.getFocusClass();
        if (focusClass != null && !focusClass.isEmpty()) {
            emit(sb, stats, "focusClass", builderCall("focusClass", quote(focusClass)));
            if (config.getFocusClassMinIoU() > 0) {
                emit(sb, stats, "focusClassMinIoU",
                        builderCall("focusClassMinIoU", formatDouble(config.getFocusClassMinIoU())));
            } else {
                skip(stats, "focusClassMinIoU");
            }
        } else {
            skip(stats, "focusClass");
            skip(stats, "focusClassMinIoU");
        }
    }

    /**
     * Emits {@code seed} (only when set; null means non-deterministic),
     * {@code hasPerImageSplitRoles} (only when true), and
     * {@code validationSplit} (always).
     */
    private static void appendReproducibilityFields(StringBuilder sb, TrainingConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Reproducibility & Integrity ---");
        if (config.getSeed() != null) {
            emit(sb, stats, "seed", builderCall("seed", String.valueOf(config.getSeed())));
        } else {
            skip(stats, "seed");
        }

        if (config.isHasPerImageSplitRoles()) {
            emit(sb, stats, "hasPerImageSplitRoles", builderCall("hasPerImageSplitRoles", "true"));
        } else {
            skip(stats, "hasPerImageSplitRoles");
        }

        emit(sb, stats, "validationSplit",
                builderCall("validationSplit", formatDouble(config.getValidationSplit())));
    }

    // ====================================================================
    // Inference Field Groups
    // ====================================================================

    /**
     * Emits inference {@code tileSize}, {@code overlap}, and {@code blendMode}.
     * All three are core geometry and always meaningful; emitted always.
     */
    private static void appendInferenceTilingFields(StringBuilder sb, InferenceConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Tiling & Geometry ---");
        emit(sb, stats, "tileSize", builderCall("tileSize", String.valueOf(config.getTileSize())));
        emit(sb, stats, "overlap", builderCall("overlap", String.valueOf(config.getOverlap())));
        emit(sb, stats, "blendMode",
                builderCall("blendMode", "InferenceConfig.BlendMode." + config.getBlendMode().name()));
    }

    /**
     * Emits {@code outputType} (always) and the OBJECTS-only post-processing
     * block ({@code objectType}, {@code minObjectSize}, {@code holeFilling},
     * {@code smoothing}). The post-processing block is gated on
     * {@code outputType == OBJECTS} so it does not appear for MEASUREMENTS,
     * OVERLAY, or RENDERED_OVERLAY.
     */
    private static void appendInferenceOutputFields(StringBuilder sb, InferenceConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Output Configuration ---");
        emit(sb, stats, "outputType",
                builderCall("outputType", "InferenceConfig.OutputType." + config.getOutputType().name()));

        if (config.getOutputType() == InferenceConfig.OutputType.OBJECTS) {
            emit(sb, stats, "objectType",
                    builderCall("objectType", "InferenceConfig.OutputObjectType." + config.getObjectType().name()));
            emit(sb, stats, "minObjectSize",
                    builderCall("minObjectSize", formatDouble(config.getMinObjectSizeMicrons())));
            emit(sb, stats, "holeFilling",
                    builderCall("holeFilling", formatDouble(config.getHoleFillingMicrons())));
            emit(sb, stats, "boundarySmoothing",
                    builderCall("smoothing", formatDouble(config.getBoundarySmoothing())));
        } else {
            skip(stats, "objectType");
            skip(stats, "minObjectSize");
            skip(stats, "holeFilling");
            skip(stats, "boundarySmoothing");
        }
    }

    /**
     * Emits {@code overlaySmoothingSigma} and {@code useCompactArgmaxOutput}.
     * Both always emitted to round-trip user intent.
     */
    private static void appendInferenceSmoothingFields(StringBuilder sb, InferenceConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Smoothing & Output Format ---");
        emit(sb, stats, "overlaySmoothingSigma",
                builderCall("overlaySmoothingSigma", formatDouble(config.getOverlaySmoothingSigma())));
        emit(sb, stats, "useCompactArgmaxOutput",
                builderCall("useCompactArgmaxOutput", String.valueOf(config.isUseCompactArgmaxOutput())));
    }

    /**
     * Emits {@code useGPU}, {@code useTTA}, {@code multiPassAveraging}, and
     * {@code maxTilesInMemory}. All always emitted.
     */
    private static void appendInferenceStrategyFields(StringBuilder sb, InferenceConfig config, EmissionStats stats) {
        appendLine(sb, BUILDER_INDENT + "// --- Inference Strategy ---");
        emit(sb, stats, "useGPU", builderCall("useGPU", String.valueOf(config.isUseGPU())));
        emit(sb, stats, "useTTA", builderCall("useTTA", String.valueOf(config.isUseTTA())));
        emit(sb, stats, "multiPassAveraging",
                builderCall("multiPassAveraging", String.valueOf(config.isMultiPassAveraging())));
        emit(sb, stats, "maxTilesInMemory",
                builderCall("maxTilesInMemory", String.valueOf(config.getMaxTilesInMemory())));
    }

    // ====================================================================
    // Channel Config (shared by training & inference)
    // ====================================================================

    /**
     * Emits the {@code ChannelConfiguration.builder()} chain. Channel config
     * is fully populated from the dialog, so every field is always emitted
     * (no conditional gates here).
     */
    private static void appendChannelConfig(StringBuilder sb, ChannelConfiguration channelConfig) {
        appendLine(sb, "// Configure channels");
        appendLine(sb, "def channelConfig = ChannelConfiguration.builder()");
        appendLine(sb, BUILDER_INDENT + ".selectedChannels(" + formatIntList(channelConfig.getSelectedChannels()) + ")");
        appendLine(sb, BUILDER_INDENT + ".channelNames(" + formatStringList(channelConfig.getChannelNames()) + ")");
        appendLine(sb, BUILDER_INDENT + ".bitDepth(" + channelConfig.getBitDepth() + ")");
        appendLine(sb, BUILDER_INDENT + ".normalizationStrategy(ChannelConfiguration.NormalizationStrategy."
                + channelConfig.getNormalizationStrategy().name() + ")");
        appendLine(sb, BUILDER_INDENT + ".build()");
        appendLine(sb, "");
    }

    // ====================================================================
    // Emission helpers (record + write a builder line)
    // ====================================================================

    /**
     * Builds a {@code "        .methodName(argument)"} line at the standard
     * builder indent. Centralises the indent + method-name + parens shape so
     * call sites in the group helpers read at a glance.
     */
    private static String builderCall(String methodName, String argument) {
        return BUILDER_INDENT + "." + methodName + "(" + argument + ")";
    }

    /**
     * Appends a single Builder line and increments the per-script "emitted"
     * counter. The field name is recorded for telemetry / log analysis.
     */
    private static void emit(StringBuilder sb, EmissionStats stats, String fieldName, String line) {
        appendLine(sb, line);
        stats.emitted++;
        stats.emittedNames.add(fieldName);
    }

    /**
     * Records that a field was intentionally skipped (default-valued or
     * gated off). Bumps the counter and remembers the field name for the
     * DEBUG log.
     */
    private static void skip(EmissionStats stats, String fieldName) {
        stats.skipped++;
        stats.skippedNames.add(fieldName);
    }

    // ====================================================================
    // Map and list literal formatting
    // ====================================================================

    /**
     * Builds a multi-line Groovy map literal of the form:
     * <pre>
     *         .builderMethod([
     *             "key1": value1,
     *             "key2": value2
     *         ])
     * </pre>
     * The map is iterated in a deterministic order (see
     * {@link #ensureDeterministicOrder}) so two runs of the generator on
     * the same logical input produce byte-identical scripts.
     *
     * @param builderPrefix the leading {@code "        .builderMethod"} chunk
     *                      (without the open paren)
     * @param map           the entries to render
     */
    private static String buildMapLiteral(String builderPrefix, Map<String, ?> map) {
        StringBuilder lines = new StringBuilder();
        lines.append(builderPrefix).append("([\n");
        Map<String, ?> ordered = ensureDeterministicOrder(map);
        int index = 0;
        int lastIndex = ordered.size() - 1;
        for (Map.Entry<String, ?> entry : ordered.entrySet()) {
            String trailingComma = (index < lastIndex) ? "," : "";
            lines.append(MAP_ENTRY_INDENT)
                    .append(quote(entry.getKey()))
                    .append(": ")
                    .append(formatMapValue(entry.getValue()))
                    .append(trailingComma)
                    .append("\n");
            index++;
        }
        lines.append(BUILDER_INDENT).append("])");
        return lines.toString();
    }

    /**
     * Returns the input map as-is when it already has a stable iteration
     * order ({@link LinkedHashMap} or any {@link SortedMap}); otherwise
     * copies entries into a {@link TreeMap} so emission order is
     * reproducible across runs and JVMs.
     */
    private static Map<String, ?> ensureDeterministicOrder(Map<String, ?> map) {
        if (map instanceof LinkedHashMap || map instanceof SortedMap) {
            return map;
        }
        return new TreeMap<>(map);
    }

    /**
     * Renders a single map value for a Groovy literal. Numbers go in
     * locale-safe form, booleans as lowercase keywords, strings get
     * quoted/escaped, everything else falls back to a quoted toString().
     */
    private static String formatMapValue(Object value) {
        if (value == null) return "null";
        if (value instanceof Boolean) return value.toString();
        if (value instanceof Double || value instanceof Float) {
            return formatDouble(((Number) value).doubleValue());
        }
        if (value instanceof Number) return value.toString();
        return quote(String.valueOf(value));
    }

    /**
     * Formats a double for Groovy with a locale-safe decimal separator.
     * Uses {@code %.6g} for compact representation across small (1e-6) and
     * large values, then strips trailing zeros / trailing dots that
     * {@code %g} sometimes adds.
     */
    private static String formatDouble(double value) {
        if (Double.isNaN(value)) return "Double.NaN";
        if (Double.isInfinite(value)) {
            return value > 0 ? "Double.POSITIVE_INFINITY" : "Double.NEGATIVE_INFINITY";
        }
        // %.6g gives 6 significant digits without locale-specific separators.
        String raw = String.format(Locale.ROOT, "%.6g", value);
        return cleanupNumberString(raw);
    }

    /**
     * Cleans a {@code printf %g} result: trims trailing zeros after the
     * decimal point, drops a trailing dot, and leaves the exponent (when
     * present) intact.
     */
    private static String cleanupNumberString(String s) {
        // Split off any exponent ('e' or 'E'), clean the mantissa, reattach.
        int exponentPos = -1;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == 'e' || ch == 'E') {
                exponentPos = i;
                break;
            }
        }
        String mantissa = exponentPos < 0 ? s : s.substring(0, exponentPos);
        String exponent = exponentPos < 0 ? "" : s.substring(exponentPos);
        if (mantissa.indexOf('.') >= 0) {
            int end = mantissa.length();
            while (end > 1 && mantissa.charAt(end - 1) == '0') {
                end--;
            }
            if (end > 1 && mantissa.charAt(end - 1) == '.') {
                end--;
            }
            mantissa = mantissa.substring(0, end);
        }
        return mantissa + exponent;
    }

    /** Appends {@code line} followed by a newline. */
    private static void appendLine(StringBuilder sb, String line) {
        sb.append(line).append("\n");
    }

    /**
     * Quotes and escapes a string for use in Groovy source code. Escapes
     * backslash, double quote, newline, carriage return, and tab.
     */
    private static String quote(String value) {
        if (value == null) return "null";
        String escaped = value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
        return "\"" + escaped + "\"";
    }

    /**
     * Formats a list of strings as a Groovy literal, e.g.
     * {@code ["a", "b", "c"]}. Empty/null produces {@code "[]"}.
     */
    private static String formatStringList(List<String> items) {
        if (items == null || items.isEmpty()) return "[]";
        return items.stream()
                .map(ScriptGenerator::quote)
                .collect(Collectors.joining(", ", "[", "]"));
    }

    /**
     * Formats a list of integers as a Groovy literal, e.g.
     * {@code [0, 1, 2]}. Empty/null produces {@code "[]"}.
     */
    private static String formatIntList(List<Integer> items) {
        if (items == null || items.isEmpty()) return "[]";
        return items.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(", ", "[", "]"));
    }

    // ====================================================================
    // Telemetry
    // ====================================================================

    /**
     * Tracks per-script emission counts for log telemetry. A field is
     * "emitted" when its Builder method appears in the generated script
     * and "skipped" when intentionally omitted by emission policy.
     */
    private static final class EmissionStats {
        int emitted = 0;
        int skipped = 0;
        final List<String> emittedNames = new ArrayList<>();
        final List<String> skippedNames = new ArrayList<>();
    }
}
