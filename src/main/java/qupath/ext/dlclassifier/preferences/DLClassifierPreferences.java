package qupath.ext.dlclassifier.preferences;

import javafx.beans.property.*;
import javafx.collections.ObservableList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.service.warnings.InteractionWarningService;
import qupath.fx.prefs.controlsfx.PropertyItemBuilder;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;

/**
 * Persistent preferences for the DL Pixel Classifier extension.
 * <p>
 * All preferences are stored using QuPath's preference system and persist
 * across sessions.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class DLClassifierPreferences {

    private static final Logger logger = LoggerFactory.getLogger(DLClassifierPreferences.class);

    // Tile settings
    private static final IntegerProperty tileSize = PathPrefs.createPersistentPreference(
            "dlclassifier.tileSize", 512);

    private static final IntegerProperty tileOverlap = PathPrefs.createPersistentPreference(
            "dlclassifier.tileOverlap", 64);

    private static final DoubleProperty tileOverlapPercent = PathPrefs.createPersistentPreference(
            "dlclassifier.tileOverlapPercent", 12.5);

    // Object output settings
    private static final StringProperty defaultObjectType = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultObjectType", "DETECTION");

    // Training defaults
    private static final IntegerProperty defaultEpochs = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultEpochs", 50);

    private static final IntegerProperty defaultBatchSize = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultBatchSize", 8);

    private static final DoubleProperty defaultLearningRate = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultLearningRate", 0.0001);

    private static final BooleanProperty useAugmentation = PathPrefs.createPersistentPreference(
            "dlclassifier.useAugmentation", true);

    private static final BooleanProperty usePretrainedWeights = PathPrefs.createPersistentPreference(
            "dlclassifier.usePretrainedWeights", true);

    // Inference defaults
    private static final BooleanProperty useGPU = PathPrefs.createPersistentPreference(
            "dlclassifier.useGPU", true);

    private static final DoubleProperty minObjectSizeMicrons = PathPrefs.createPersistentPreference(
            "dlclassifier.minObjectSizeMicrons", 10.0);

    private static final DoubleProperty holeFillingMicrons = PathPrefs.createPersistentPreference(
            "dlclassifier.holeFillingMicrons", 5.0);

    // Normalization
    private static final StringProperty defaultNormalization = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultNormalization", "PERCENTILE_99");

    // ==================== Training Dialog Preferences ====================

    private static final StringProperty lastArchitecture = PathPrefs.createPersistentPreference(
            "dlclassifier.lastArchitecture", "unet");

    private static final StringProperty lastBackbone = PathPrefs.createPersistentPreference(
            "dlclassifier.lastBackbone", "resnet34");

    private static final IntegerProperty validationSplit = PathPrefs.createPersistentPreference(
            "dlclassifier.validationSplit", 20);

    private static final BooleanProperty rebalanceByDefault = PathPrefs.createPersistentPreference(
            "dlclassifier.rebalanceByDefault", true);

    private static final BooleanProperty augFlipHorizontal = PathPrefs.createPersistentPreference(
            "dlclassifier.augFlipHorizontal", true);

    private static final BooleanProperty augFlipVertical = PathPrefs.createPersistentPreference(
            "dlclassifier.augFlipVertical", true);

    private static final BooleanProperty augRotation = PathPrefs.createPersistentPreference(
            "dlclassifier.augRotation", true);

    private static final BooleanProperty augColorJitter = PathPrefs.createPersistentPreference(
            "dlclassifier.augColorJitter", false);

    private static final StringProperty augIntensityMode = PathPrefs.createPersistentPreference(
            "dlclassifier.augIntensityMode", "none");

    private static final BooleanProperty augElasticDeform = PathPrefs.createPersistentPreference(
            "dlclassifier.augElasticDeform", false);

    // ==================== Advanced augmentation strength/probability ====================
    // Defaults match the hardcoded values in training_service.get_training_augmentation()
    private static final DoubleProperty augPFlip = PathPrefs.createPersistentPreference(
            "dlclassifier.augPFlip", 0.5);
    private static final DoubleProperty augPRotate = PathPrefs.createPersistentPreference(
            "dlclassifier.augPRotate", 0.5);
    private static final DoubleProperty augPElastic = PathPrefs.createPersistentPreference(
            "dlclassifier.augPElastic", 0.3);
    private static final DoubleProperty augPColor = PathPrefs.createPersistentPreference(
            "dlclassifier.augPColor", 0.3);
    private static final DoubleProperty augBrightnessLimit = PathPrefs.createPersistentPreference(
            "dlclassifier.augBrightnessLimit", 0.2);
    private static final DoubleProperty augContrastLimit = PathPrefs.createPersistentPreference(
            "dlclassifier.augContrastLimit", 0.2);
    private static final IntegerProperty augGammaMin = PathPrefs.createPersistentPreference(
            "dlclassifier.augGammaMin", 80);
    private static final IntegerProperty augGammaMax = PathPrefs.createPersistentPreference(
            "dlclassifier.augGammaMax", 120);
    private static final DoubleProperty augElasticAlpha = PathPrefs.createPersistentPreference(
            "dlclassifier.augElasticAlpha", 120.0);
    private static final DoubleProperty augElasticSigmaRatio = PathPrefs.createPersistentPreference(
            "dlclassifier.augElasticSigmaRatio", 0.05);
    private static final DoubleProperty augPNoise = PathPrefs.createPersistentPreference(
            "dlclassifier.augPNoise", 0.2);
    private static final DoubleProperty augNoiseStdMin = PathPrefs.createPersistentPreference(
            "dlclassifier.augNoiseStdMin", 0.04);
    private static final DoubleProperty augNoiseStdMax = PathPrefs.createPersistentPreference(
            "dlclassifier.augNoiseStdMax", 0.2);

    // Tiling/resolution settings persisted from training dialog
    private static final DoubleProperty defaultDownsample = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultDownsample", 1.0);

    private static final IntegerProperty defaultContextScale = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultContextScale", 1);

    private static final IntegerProperty lastLineStrokeWidth = PathPrefs.createPersistentPreference(
            "dlclassifier.lastLineStrokeWidth", 0);

    // Training data export directory (empty = system temp)
    private static final StringProperty trainingExportDir = PathPrefs.createPersistentPreference(
            "dlclassifier.trainingExportDir", "");

    // ==================== Training Strategy Preferences ====================

    private static final StringProperty defaultScheduler = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultScheduler", "onecycle");

    private static final StringProperty defaultLossFunction = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultLossFunction", "ce_dice");

    private static final StringProperty defaultEarlyStoppingMetric = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultEarlyStoppingMetric", "mean_iou");

    private static final IntegerProperty defaultEarlyStoppingPatience = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultEarlyStoppingPatience", 15);

    private static final BooleanProperty defaultMixedPrecision = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultMixedPrecision", true);

    // UI mode
    private static final BooleanProperty advancedMode = PathPrefs.createPersistentPreference(
            "dlclassifier.advancedMode", false);

    private static final IntegerProperty defaultGradientAccumulation = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultGradientAccumulation", 1);

    private static final IntegerProperty defaultOhemHardPixelPct = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultOhemHardPixelPct", 100);

    private static final IntegerProperty defaultOhemHardPixelStartPct = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultOhemHardPixelStartPct", 100);

    private static final DoubleProperty defaultFocalGamma = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultFocalGamma", 2.0);

    // Boundary-softened CE parameters (loss_function = "boundary_ce" or
    // "boundary_ce_dice"). sigma is the EDT falloff length in pixels,
    // w_min is the floor weight at exact boundaries.
    private static final DoubleProperty defaultBoundarySigma = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultBoundarySigma", 3.0);
    private static final DoubleProperty defaultBoundaryWMin = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultBoundaryWMin", 0.1);

    private static final StringProperty defaultOhemSchedule = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultOhemSchedule", "fixed");

    // When true, OHEM uses global topk + per-class safety floor (lets hard
    // pixels concentrate on the weaker class). When false, each class keeps
    // a proportional share of the hard pixels -- the legacy behavior.
    private static final BooleanProperty defaultOhemAdaptiveFloor = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultOhemAdaptiveFloor", false);

    private static final BooleanProperty defaultProgressiveResize = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultProgressiveResize", false);

    private static final StringProperty defaultFocusClass = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultFocusClass", "");

    private static final DoubleProperty defaultFocusClassMinIoU = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultFocusClassMinIoU", 0.0);

    // ==================== Advanced Training Parameters ====================

    private static final DoubleProperty defaultWeightDecay = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultWeightDecay", 0.01);

    private static final DoubleProperty defaultDiscriminativeLrRatio = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultDiscriminativeLrRatio", 0.1);

    private static final IntegerProperty lastSeed = PathPrefs.createPersistentPreference(
            "dlclassifier.lastSeed", 0);

    // ==================== Inference Dialog Preferences ====================

    private static final StringProperty lastOutputType = PathPrefs.createPersistentPreference(
            "dlclassifier.lastOutputType", "OBJECTS");

    private static final StringProperty lastBlendMode = PathPrefs.createPersistentPreference(
            "dlclassifier.lastBlendMode", "GAUSSIAN");

    private static final DoubleProperty smoothing = PathPrefs.createPersistentPreference(
            "dlclassifier.smoothing", 1.0);

    private static final DoubleProperty overlaySmoothing = PathPrefs.createPersistentPreference(
            "dlclassifier.overlaySmoothing", 2.0);

    private static final StringProperty applicationScope = PathPrefs.createPersistentPreference(
            "dlclassifier.applicationScope", "ALL_ANNOTATIONS");

    private static final BooleanProperty createBackup = PathPrefs.createPersistentPreference(
            "dlclassifier.createBackup", false);

    // Multi-pass tile averaging for seamless inference
    private static final BooleanProperty multiPassAveraging = PathPrefs.createPersistentPreference(
            "dlclassifier.multiPassAveraging", false);

    // Phase 3c: compact uint8 argmax output instead of float32 probability maps.
    // When on, Python inference returns class indices directly (20x smaller
    // payload, no softmax). Disables overlay smoothing, multi-pass averaging,
    // and tile blending since those operate on floats.
    private static final BooleanProperty useCompactArgmaxOutput = PathPrefs.createPersistentPreference(
            "dlclassifier.useCompactArgmaxOutput", false);

    // Phase 4: experimental TensorRT inference path. Requires
    // onnxruntime-gpu built with TensorrtExecutionProvider. Builds a TRT
    // engine from model_static.onnx on first inference and caches it on
    // disk; silently falls back to CUDAExecutionProvider when TRT is
    // unavailable. Windows wheels are hit-or-miss, so default is off.
    private static final BooleanProperty experimentalTensorRT = PathPrefs.createPersistentPreference(
            "dlclassifier.experimentalTensorRT", false);

    // Phase 4: experimental INT8 PTQ. Requires experimentalTensorRT on.
    // Calibration uses a sample of training tiles; BatchRenorm is folded
    // to BatchNorm at export time (safe at eval() with rmax=dmax=inf).
    private static final BooleanProperty experimentalInt8 = PathPrefs.createPersistentPreference(
            "dlclassifier.experimentalInt8", false);

    // Preload training patches into RAM to skip per-batch disk I/O and
    // TIFF decode. "auto" = enable when the dataset fits in ~25% of free
    // RAM; "on" = always enable (may exhaust memory on huge datasets);
    // "off" = always stream from disk (the legacy behavior).
    private static final StringProperty defaultInMemoryDataset = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultInMemoryDataset", "auto");

    // One-time overlay notice dismissed
    private static final BooleanProperty overlayNoticeDismissed = PathPrefs.createPersistentPreference(
            "dlclassifier.overlayNoticeDismissed", false);

    // ==================== Environment ====================

    // Automatically rebuild the Python environment when extension version changes
    private static final BooleanProperty autoRebuildEnvironment = PathPrefs.createPersistentPreference(
            "dlclassifier.autoRebuildEnvironment", true);

    // Show the welcome/getting-started message when Train Classifier is first opened
    private static final BooleanProperty showWelcomeMessage = PathPrefs.createPersistentPreference(
            "dlclassifier.showWelcomeMessage", true);

    // ==================== Menu Appearance ====================

    // Show colored dot next to extension name in the Extensions menu
    private static final BooleanProperty showMenuDot = PathPrefs.createPersistentPreference(
            "dlclassifier.showMenuDot", true);

    // Color of the menu dot as packed ARGB integer (default: magenta = 0xFFFF00FF)
    private static final IntegerProperty menuDotColor = PathPrefs.createPersistentPreference(
            "dlclassifier.menuDotColor", 0xFFFF00FF);

    private DLClassifierPreferences() {
        // Utility class - no instantiation
    }

    private static final String CATEGORY = "DL Pixel Classifier";

    /**
     * Registers key preferences with QuPath's preference panel so they
     * appear in Edit > Preferences under "DL Pixel Classifier".
     *
     * @param qupath the QuPath GUI instance
     */
    public static void installPreferences(QuPathGUI qupath) {
        if (qupath == null)
            return;

        logger.info("Installing DL Pixel Classifier preferences");

        // Wire interaction-warning preference-toggle listeners. Fires
        // the relevant PreferenceWarning watchers when the user flips
        // the experimental TRT / INT8 toggles, so they get an
        // immediate popup instead of wondering why nothing changed.
        javafx.beans.value.ChangeListener<Boolean> providerToggleListener =
                (obs, oldV, newV) -> {
                    // Only fire on turn-ON transitions. Turning a
                    // provider OFF is not a situation that warrants a
                    // popup about cache invalidation (user knows they
                    // are giving up the provider).
                    if (!Boolean.TRUE.equals(newV)) return;
                    try {
                        var list = InteractionWarningService.evaluatePreferences();
                        var visible = InteractionWarningService.filterVisible(list);
                        if (!visible.isEmpty()) {
                            InteractionWarningService.showIfAny(visible, null);
                        }
                    } catch (RuntimeException ex) {
                        logger.warn("Preference-toggle interaction warning "
                                + "evaluation failed", ex);
                    }
                };
        experimentalTensorRT.addListener(providerToggleListener);
        experimentalInt8.addListener(providerToggleListener);

        ObservableList<org.controlsfx.control.PropertySheet.Item> items =
                qupath.getPreferencePane()
                        .getPropertySheet()
                        .getItems();

        items.add(new PropertyItemBuilder<>(useGPU, Boolean.class)
                .name("Use GPU for Inference")
                .category(CATEGORY)
                .description("Use GPU acceleration for inference when available. " +
                        "Falls back to CPU if no GPU is detected.")
                .build());

        items.add(new PropertyItemBuilder<>(trainingExportDir, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.DIRECTORY)
                .name("Training Data Export Directory")
                .category(CATEGORY)
                .description("Directory for exporting training data patches. " +
                        "If empty, the system temporary directory is used. " +
                        "Set to a directory on a larger drive if temp space is limited.")
                .build());

        items.add(new PropertyItemBuilder<>(autoRebuildEnvironment, Boolean.class)
                .name("Auto-Rebuild Environment on Update")
                .category(CATEGORY)
                .description("Automatically rebuild the Python environment in the background " +
                        "when the extension JAR is updated. While rebuilding, all workflow " +
                        "menu items (Train, Apply Classifier, Overlay) are temporarily " +
                        "disabled and re-enabled once the rebuild completes. " +
                        "Disable this if you prefer to rebuild manually via Utilities.")
                .build());

        items.add(new PropertyItemBuilder<>(overlaySmoothing, Double.class)
                .name("Overlay Prediction Smoothing")
                .category(CATEGORY)
                .description("Gaussian sigma for smoothing probability maps before " +
                        "classification in the prediction overlay. " +
                        "0 = no smoothing (raw model output, may appear noisy); " +
                        "1-2 = light; 3-5 = moderate (recommended for noisy models); " +
                        "5+ = heavy (may lose fine detail). " +
                        "Changes apply immediately if an overlay is active.")
                .build());

        items.add(new PropertyItemBuilder<>(multiPassAveraging, Boolean.class)
                .name("Overlay High-Quality Tile Averaging")
                .category(CATEGORY)
                .description("Run each tile at 4 spatial offsets and average the " +
                        "predictions to eliminate tile boundary artifacts. " +
                        "Recommended for context-scale models where seams are visible. " +
                        "Applies to both the overlay and Apply Classifier. " +
                        "~4x slower but produces seamless results.")
                .build());

        items.add(new PropertyItemBuilder<>(useCompactArgmaxOutput, Boolean.class)
                .name("Overlay Fast Argmax (uint8)")
                .category(CATEGORY)
                .description("Return class indices directly from inference instead of " +
                        "full probability maps. ~20x smaller payload and slightly " +
                        "faster per tile. " +
                        "Trade-off: disables overlay smoothing, multi-pass averaging, " +
                        "and tile boundary blending (these all require float " +
                        "probabilities). Leave off for highest-quality overlays; " +
                        "turn on for quick previews or lower memory use.")
                .build());

        items.add(new PropertyItemBuilder<>(experimentalTensorRT, Boolean.class)
                .name("Experimental: TensorRT Inference")
                .category(CATEGORY)
                .description("Use TensorRT for ONNX inference on CUDA. " +
                        "Builds a TRT engine from the static-shape ONNX on first " +
                        "inference and caches it next to the model. " +
                        "Typical speedup: 2-4x over plain CUDAExecutionProvider. " +
                        "Requires onnxruntime-gpu built with the " +
                        "TensorrtExecutionProvider. Silently falls back to " +
                        "CUDAExecutionProvider when unavailable. " +
                        "EXPERIMENTAL -- may not work on all Windows setups; " +
                        "report issues if encountered.")
                .build());

        items.add(new PropertyItemBuilder<>(experimentalInt8, Boolean.class)
                .name("Experimental: INT8 Quantization")
                .category(CATEGORY)
                .description("Quantize weights and activations to INT8 when " +
                        "TensorRT is enabled. Typical additional speedup: ~2x " +
                        "over FP16 TRT with ~1-2 point IoU drop on simple tasks. " +
                        "Requires a calibration pass (done once at engine " +
                        "build time). Only effective when " +
                        "'Experimental: TensorRT Inference' is also on. " +
                        "EXPERIMENTAL.")
                .build());

        items.add(new PropertyItemBuilder<>(defaultInMemoryDataset, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.CHOICE)
                .choices(javafx.collections.FXCollections.observableArrayList(
                        "auto", "on", "off"))
                .name("Training: Pre-Load Patches Into RAM")
                .category(CATEGORY)
                .description("Cache all training patches in RAM at startup to skip " +
                        "per-batch disk I/O and TIFF decode. " +
                        "auto = enable when the dataset fits in about 25% of free RAM " +
                        "(safe default); " +
                        "on = always enable (may run out of memory on large datasets); " +
                        "off = always stream from disk. " +
                        "Typically cuts per-epoch time by 30-70% on GPU-bound setups.")
                .build());

        items.add(new PropertyItemBuilder<>(showMenuDot, Boolean.class)
                .name("Show Menu Indicator Dot")
                .category(CATEGORY)
                .description("Show a colored dot next to the extension name in the " +
                        "Extensions menu for quick identification. " +
                        "Takes effect after restarting QuPath.")
                .build());

        items.add(new PropertyItemBuilder<>(menuDotColor, Integer.class)
                .propertyType(PropertyItemBuilder.PropertyType.COLOR)
                .name("Menu Indicator Dot Color")
                .category(CATEGORY)
                .description("Color of the indicator dot shown in the Extensions menu. " +
                        "Takes effect after restarting QuPath.")
                .build());
    }

    // ==================== Tile Settings ====================

    public static int getTileSize() {
        return tileSize.get();
    }

    public static void setTileSize(int size) {
        tileSize.set(size);
    }

    public static IntegerProperty tileSizeProperty() {
        return tileSize;
    }

    public static int getTileOverlap() {
        return tileOverlap.get();
    }

    public static void setTileOverlap(int overlap) {
        tileOverlap.set(overlap);
    }

    public static IntegerProperty tileOverlapProperty() {
        return tileOverlap;
    }

    public static double getTileOverlapPercent() {
        return tileOverlapPercent.get();
    }

    public static void setTileOverlapPercent(double percent) {
        tileOverlapPercent.set(percent);
    }

    public static DoubleProperty tileOverlapPercentProperty() {
        return tileOverlapPercent;
    }

    // ==================== Object Output Settings ====================

    public static String getDefaultObjectType() {
        return defaultObjectType.get();
    }

    public static void setDefaultObjectType(String type) {
        defaultObjectType.set(type);
    }

    public static StringProperty defaultObjectTypeProperty() {
        return defaultObjectType;
    }

    // ==================== Training Defaults ====================

    public static int getDefaultEpochs() {
        return defaultEpochs.get();
    }

    public static void setDefaultEpochs(int epochs) {
        defaultEpochs.set(epochs);
    }

    public static IntegerProperty defaultEpochsProperty() {
        return defaultEpochs;
    }

    public static int getDefaultBatchSize() {
        return defaultBatchSize.get();
    }

    public static void setDefaultBatchSize(int batchSize) {
        defaultBatchSize.set(batchSize);
    }

    public static IntegerProperty defaultBatchSizeProperty() {
        return defaultBatchSize;
    }

    public static double getDefaultLearningRate() {
        return defaultLearningRate.get();
    }

    public static void setDefaultLearningRate(double lr) {
        defaultLearningRate.set(lr);
    }

    public static DoubleProperty defaultLearningRateProperty() {
        return defaultLearningRate;
    }

    public static boolean isUseAugmentation() {
        return useAugmentation.get();
    }

    public static void setUseAugmentation(boolean use) {
        useAugmentation.set(use);
    }

    public static BooleanProperty useAugmentationProperty() {
        return useAugmentation;
    }

    public static boolean isUsePretrainedWeights() {
        return usePretrainedWeights.get();
    }

    public static void setUsePretrainedWeights(boolean use) {
        usePretrainedWeights.set(use);
    }

    public static BooleanProperty usePretrainedWeightsProperty() {
        return usePretrainedWeights;
    }

    // ==================== Inference Defaults ====================

    public static boolean isUseGPU() {
        return useGPU.get();
    }

    public static void setUseGPU(boolean use) {
        useGPU.set(use);
    }

    public static BooleanProperty useGPUProperty() {
        return useGPU;
    }

    // ==================== Object Output Settings ====================

    public static double getMinObjectSizeMicrons() {
        return minObjectSizeMicrons.get();
    }

    public static void setMinObjectSizeMicrons(double size) {
        minObjectSizeMicrons.set(size);
    }

    public static DoubleProperty minObjectSizeMicronsProperty() {
        return minObjectSizeMicrons;
    }

    public static double getHoleFillingMicrons() {
        return holeFillingMicrons.get();
    }

    public static void setHoleFillingMicrons(double size) {
        holeFillingMicrons.set(size);
    }

    public static DoubleProperty holeFillingMicronsProperty() {
        return holeFillingMicrons;
    }

    // ==================== Normalization ====================

    public static String getDefaultNormalization() {
        return defaultNormalization.get();
    }

    public static void setDefaultNormalization(String normalization) {
        defaultNormalization.set(normalization);
    }

    public static StringProperty defaultNormalizationProperty() {
        return defaultNormalization;
    }

    // ==================== Training Export Directory ====================

    public static String getTrainingExportDir() {
        return trainingExportDir.get();
    }

    public static void setTrainingExportDir(String dir) {
        trainingExportDir.set(dir);
    }

    public static StringProperty trainingExportDirProperty() {
        return trainingExportDir;
    }

    // ==================== Training Dialog Preferences ====================

    public static String getLastArchitecture() {
        return lastArchitecture.get();
    }

    public static void setLastArchitecture(String architecture) {
        lastArchitecture.set(architecture);
    }

    public static StringProperty lastArchitectureProperty() {
        return lastArchitecture;
    }

    public static String getLastBackbone() {
        return lastBackbone.get();
    }

    public static void setLastBackbone(String backbone) {
        lastBackbone.set(backbone);
    }

    public static StringProperty lastBackboneProperty() {
        return lastBackbone;
    }

    public static int getValidationSplit() {
        return validationSplit.get();
    }

    public static void setValidationSplit(int split) {
        validationSplit.set(split);
    }

    public static IntegerProperty validationSplitProperty() {
        return validationSplit;
    }

    public static boolean isRebalanceByDefault() {
        return rebalanceByDefault.get();
    }

    public static void setRebalanceByDefault(boolean rebalance) {
        rebalanceByDefault.set(rebalance);
    }

    public static BooleanProperty rebalanceByDefaultProperty() {
        return rebalanceByDefault;
    }

    public static boolean isAugFlipHorizontal() {
        return augFlipHorizontal.get();
    }

    public static void setAugFlipHorizontal(boolean flip) {
        augFlipHorizontal.set(flip);
    }

    public static BooleanProperty augFlipHorizontalProperty() {
        return augFlipHorizontal;
    }

    public static boolean isAugFlipVertical() {
        return augFlipVertical.get();
    }

    public static void setAugFlipVertical(boolean flip) {
        augFlipVertical.set(flip);
    }

    public static BooleanProperty augFlipVerticalProperty() {
        return augFlipVertical;
    }

    public static boolean isAugRotation() {
        return augRotation.get();
    }

    public static void setAugRotation(boolean rotation) {
        augRotation.set(rotation);
    }

    public static BooleanProperty augRotationProperty() {
        return augRotation;
    }

    public static boolean isAugColorJitter() {
        return augColorJitter.get();
    }

    public static void setAugColorJitter(boolean jitter) {
        augColorJitter.set(jitter);
    }

    public static BooleanProperty augColorJitterProperty() {
        return augColorJitter;
    }

    public static String getAugIntensityMode() {
        return augIntensityMode.get();
    }

    public static void setAugIntensityMode(String mode) {
        augIntensityMode.set(mode);
    }

    public static StringProperty augIntensityModeProperty() {
        return augIntensityMode;
    }

    public static boolean isAugElasticDeform() {
        return augElasticDeform.get();
    }

    public static void setAugElasticDeform(boolean deform) {
        augElasticDeform.set(deform);
    }

    public static BooleanProperty augElasticDeformProperty() {
        return augElasticDeform;
    }

    // ==================== Advanced augmentation strength/probability ====================

    public static double getAugPFlip() { return augPFlip.get(); }
    public static void setAugPFlip(double v) { augPFlip.set(v); }
    public static DoubleProperty augPFlipProperty() { return augPFlip; }

    public static double getAugPRotate() { return augPRotate.get(); }
    public static void setAugPRotate(double v) { augPRotate.set(v); }
    public static DoubleProperty augPRotateProperty() { return augPRotate; }

    public static double getAugPElastic() { return augPElastic.get(); }
    public static void setAugPElastic(double v) { augPElastic.set(v); }
    public static DoubleProperty augPElasticProperty() { return augPElastic; }

    public static double getAugPColor() { return augPColor.get(); }
    public static void setAugPColor(double v) { augPColor.set(v); }
    public static DoubleProperty augPColorProperty() { return augPColor; }

    public static double getAugBrightnessLimit() { return augBrightnessLimit.get(); }
    public static void setAugBrightnessLimit(double v) { augBrightnessLimit.set(v); }
    public static DoubleProperty augBrightnessLimitProperty() { return augBrightnessLimit; }

    public static double getAugContrastLimit() { return augContrastLimit.get(); }
    public static void setAugContrastLimit(double v) { augContrastLimit.set(v); }
    public static DoubleProperty augContrastLimitProperty() { return augContrastLimit; }

    public static int getAugGammaMin() { return augGammaMin.get(); }
    public static void setAugGammaMin(int v) { augGammaMin.set(v); }
    public static IntegerProperty augGammaMinProperty() { return augGammaMin; }

    public static int getAugGammaMax() { return augGammaMax.get(); }
    public static void setAugGammaMax(int v) { augGammaMax.set(v); }
    public static IntegerProperty augGammaMaxProperty() { return augGammaMax; }

    public static double getAugElasticAlpha() { return augElasticAlpha.get(); }
    public static void setAugElasticAlpha(double v) { augElasticAlpha.set(v); }
    public static DoubleProperty augElasticAlphaProperty() { return augElasticAlpha; }

    public static double getAugElasticSigmaRatio() { return augElasticSigmaRatio.get(); }
    public static void setAugElasticSigmaRatio(double v) { augElasticSigmaRatio.set(v); }
    public static DoubleProperty augElasticSigmaRatioProperty() { return augElasticSigmaRatio; }

    public static double getAugPNoise() { return augPNoise.get(); }
    public static void setAugPNoise(double v) { augPNoise.set(v); }
    public static DoubleProperty augPNoiseProperty() { return augPNoise; }

    public static double getAugNoiseStdMin() { return augNoiseStdMin.get(); }
    public static void setAugNoiseStdMin(double v) { augNoiseStdMin.set(v); }
    public static DoubleProperty augNoiseStdMinProperty() { return augNoiseStdMin; }

    public static double getAugNoiseStdMax() { return augNoiseStdMax.get(); }
    public static void setAugNoiseStdMax(double v) { augNoiseStdMax.set(v); }
    public static DoubleProperty augNoiseStdMaxProperty() { return augNoiseStdMax; }

    // ==================== Tiling/Resolution Settings ====================

    public static double getDefaultDownsample() {
        return defaultDownsample.get();
    }

    public static void setDefaultDownsample(double downsample) {
        defaultDownsample.set(downsample);
    }

    public static DoubleProperty defaultDownsampleProperty() {
        return defaultDownsample;
    }

    public static int getDefaultContextScale() {
        return defaultContextScale.get();
    }

    public static void setDefaultContextScale(int scale) {
        defaultContextScale.set(scale);
    }

    public static IntegerProperty defaultContextScaleProperty() {
        return defaultContextScale;
    }

    /** Line stroke width last used in the training dialog. 0 means "use QuPath's stroke thickness". */
    public static int getLastLineStrokeWidth() {
        return lastLineStrokeWidth.get();
    }

    public static void setLastLineStrokeWidth(int width) {
        lastLineStrokeWidth.set(width);
    }

    public static IntegerProperty lastLineStrokeWidthProperty() {
        return lastLineStrokeWidth;
    }

    // ==================== Training Strategy Preferences ====================

    public static String getDefaultScheduler() {
        return defaultScheduler.get();
    }

    public static void setDefaultScheduler(String scheduler) {
        defaultScheduler.set(scheduler);
    }

    public static StringProperty defaultSchedulerProperty() {
        return defaultScheduler;
    }

    public static String getDefaultLossFunction() {
        return defaultLossFunction.get();
    }

    public static void setDefaultLossFunction(String lossFunction) {
        defaultLossFunction.set(lossFunction);
    }

    public static StringProperty defaultLossFunctionProperty() {
        return defaultLossFunction;
    }

    public static String getDefaultEarlyStoppingMetric() {
        return defaultEarlyStoppingMetric.get();
    }

    public static void setDefaultEarlyStoppingMetric(String metric) {
        defaultEarlyStoppingMetric.set(metric);
    }

    public static StringProperty defaultEarlyStoppingMetricProperty() {
        return defaultEarlyStoppingMetric;
    }

    public static int getDefaultEarlyStoppingPatience() {
        return defaultEarlyStoppingPatience.get();
    }

    public static void setDefaultEarlyStoppingPatience(int patience) {
        defaultEarlyStoppingPatience.set(patience);
    }

    public static IntegerProperty defaultEarlyStoppingPatienceProperty() {
        return defaultEarlyStoppingPatience;
    }

    public static boolean isDefaultMixedPrecision() {
        return defaultMixedPrecision.get();
    }

    public static void setDefaultMixedPrecision(boolean enabled) {
        defaultMixedPrecision.set(enabled);
    }

    public static BooleanProperty defaultMixedPrecisionProperty() {
        return defaultMixedPrecision;
    }

    public static boolean isAdvancedMode() {
        return advancedMode.get();
    }

    public static void setAdvancedMode(boolean enabled) {
        advancedMode.set(enabled);
    }

    public static BooleanProperty advancedModeProperty() {
        return advancedMode;
    }

    public static int getDefaultGradientAccumulation() {
        return defaultGradientAccumulation.get();
    }

    public static void setDefaultGradientAccumulation(int steps) {
        defaultGradientAccumulation.set(steps);
    }

    public static int getDefaultOhemHardPixelPct() {
        return defaultOhemHardPixelPct.get();
    }

    public static void setDefaultOhemHardPixelPct(int pct) {
        defaultOhemHardPixelPct.set(pct);
    }

    public static int getDefaultOhemHardPixelStartPct() {
        return defaultOhemHardPixelStartPct.get();
    }

    public static void setDefaultOhemHardPixelStartPct(int pct) {
        defaultOhemHardPixelStartPct.set(pct);
    }

    public static String getDefaultOhemSchedule() {
        return defaultOhemSchedule.get();
    }

    public static void setDefaultOhemSchedule(String schedule) {
        defaultOhemSchedule.set(schedule != null ? schedule : "fixed");
    }

    public static boolean isDefaultOhemAdaptiveFloor() {
        return defaultOhemAdaptiveFloor.get();
    }

    public static void setDefaultOhemAdaptiveFloor(boolean enabled) {
        defaultOhemAdaptiveFloor.set(enabled);
    }

    public static BooleanProperty defaultOhemAdaptiveFloorProperty() {
        return defaultOhemAdaptiveFloor;
    }

    public static double getDefaultFocalGamma() {
        return defaultFocalGamma.get();
    }

    public static void setDefaultFocalGamma(double gamma) {
        defaultFocalGamma.set(gamma);
    }

    public static double getDefaultBoundarySigma() {
        return defaultBoundarySigma.get();
    }

    public static void setDefaultBoundarySigma(double sigma) {
        defaultBoundarySigma.set(sigma);
    }

    public static DoubleProperty defaultBoundarySigmaProperty() {
        return defaultBoundarySigma;
    }

    public static double getDefaultBoundaryWMin() {
        return defaultBoundaryWMin.get();
    }

    public static void setDefaultBoundaryWMin(double wMin) {
        defaultBoundaryWMin.set(wMin);
    }

    public static DoubleProperty defaultBoundaryWMinProperty() {
        return defaultBoundaryWMin;
    }

    public static boolean isDefaultProgressiveResize() {
        return defaultProgressiveResize.get();
    }

    public static void setDefaultProgressiveResize(boolean enabled) {
        defaultProgressiveResize.set(enabled);
    }

    public static String getDefaultFocusClass() {
        return defaultFocusClass.get();
    }

    public static void setDefaultFocusClass(String className) {
        defaultFocusClass.set(className != null ? className : "");
    }

    public static double getDefaultFocusClassMinIoU() {
        return defaultFocusClassMinIoU.get();
    }

    public static void setDefaultFocusClassMinIoU(double minIoU) {
        defaultFocusClassMinIoU.set(minIoU);
    }

    // ==================== Advanced Training Parameters ====================

    public static double getDefaultWeightDecay() { return defaultWeightDecay.get(); }
    public static void setDefaultWeightDecay(double v) { defaultWeightDecay.set(v); }

    public static double getDefaultDiscriminativeLrRatio() { return defaultDiscriminativeLrRatio.get(); }
    public static void setDefaultDiscriminativeLrRatio(double v) { defaultDiscriminativeLrRatio.set(v); }

    public static int getLastSeed() { return lastSeed.get(); }
    public static void setLastSeed(int v) { lastSeed.set(v); }

    // ==================== Inference Dialog Preferences ====================

    public static String getLastOutputType() {
        return lastOutputType.get();
    }

    public static void setLastOutputType(String type) {
        lastOutputType.set(type);
    }

    public static StringProperty lastOutputTypeProperty() {
        return lastOutputType;
    }

    public static String getLastBlendMode() {
        return lastBlendMode.get();
    }

    public static void setLastBlendMode(String mode) {
        lastBlendMode.set(mode);
    }

    public static StringProperty lastBlendModeProperty() {
        return lastBlendMode;
    }

    public static double getSmoothing() {
        return smoothing.get();
    }

    public static void setSmoothing(double value) {
        smoothing.set(value);
    }

    public static DoubleProperty smoothingProperty() {
        return smoothing;
    }

    public static double getOverlaySmoothing() {
        return overlaySmoothing.get();
    }

    public static void setOverlaySmoothing(double value) {
        overlaySmoothing.set(value);
    }

    public static DoubleProperty overlaySmoothingProperty() {
        return overlaySmoothing;
    }

    public static String getApplicationScope() {
        return applicationScope.get();
    }

    public static void setApplicationScope(String scope) {
        applicationScope.set(scope);
    }

    public static StringProperty applicationScopeProperty() {
        return applicationScope;
    }

    public static boolean isCreateBackup() {
        return createBackup.get();
    }

    public static void setCreateBackup(boolean backup) {
        createBackup.set(backup);
    }

    public static BooleanProperty createBackupProperty() {
        return createBackup;
    }

    // ==================== Environment ====================

    public static boolean isAutoRebuildEnvironment() {
        return autoRebuildEnvironment.get();
    }

    public static void setAutoRebuildEnvironment(boolean auto) {
        autoRebuildEnvironment.set(auto);
    }

    public static BooleanProperty autoRebuildEnvironmentProperty() {
        return autoRebuildEnvironment;
    }

    // ==================== Menu Appearance ====================

    public static boolean isShowMenuDot() {
        return showMenuDot.get();
    }

    public static void setShowMenuDot(boolean show) {
        showMenuDot.set(show);
    }

    public static BooleanProperty showMenuDotProperty() {
        return showMenuDot;
    }

    public static int getMenuDotColor() {
        return menuDotColor.get();
    }

    public static void setMenuDotColor(int argb) {
        menuDotColor.set(argb);
    }

    public static IntegerProperty menuDotColorProperty() {
        return menuDotColor;
    }

    // ==================== Multi-Pass Averaging ====================

    public static boolean isMultiPassAveraging() {
        return multiPassAveraging.get();
    }

    public static void setMultiPassAveraging(boolean enabled) {
        multiPassAveraging.set(enabled);
    }

    public static BooleanProperty multiPassAveragingProperty() {
        return multiPassAveraging;
    }

    // ==================== Compact Argmax Output (Phase 3c) ====================

    public static boolean isUseCompactArgmaxOutput() {
        return useCompactArgmaxOutput.get();
    }

    public static void setUseCompactArgmaxOutput(boolean enabled) {
        useCompactArgmaxOutput.set(enabled);
    }

    public static BooleanProperty useCompactArgmaxOutputProperty() {
        return useCompactArgmaxOutput;
    }

    // ==================== Phase 4 experimental inference ====================

    public static boolean isExperimentalTensorRT() {
        return experimentalTensorRT.get();
    }

    public static void setExperimentalTensorRT(boolean enabled) {
        experimentalTensorRT.set(enabled);
    }

    public static BooleanProperty experimentalTensorRTProperty() {
        return experimentalTensorRT;
    }

    public static boolean isExperimentalInt8() {
        return experimentalInt8.get();
    }

    public static void setExperimentalInt8(boolean enabled) {
        experimentalInt8.set(enabled);
    }

    public static BooleanProperty experimentalInt8Property() {
        return experimentalInt8;
    }

    /**
     * @return the in-memory dataset mode: "auto", "on", or "off".
     */
    public static String getDefaultInMemoryDataset() {
        String v = defaultInMemoryDataset.get();
        if ("auto".equals(v) || "on".equals(v) || "off".equals(v)) {
            return v;
        }
        return "auto";
    }

    public static void setDefaultInMemoryDataset(String mode) {
        if ("auto".equals(mode) || "on".equals(mode) || "off".equals(mode)) {
            defaultInMemoryDataset.set(mode);
        }
    }

    public static StringProperty defaultInMemoryDatasetProperty() {
        return defaultInMemoryDataset;
    }

    // ==================== Overlay Notice ====================

    public static boolean isOverlayNoticeDismissed() {
        return overlayNoticeDismissed.get();
    }

    public static void setOverlayNoticeDismissed(boolean dismissed) {
        overlayNoticeDismissed.set(dismissed);
    }

    public static BooleanProperty overlayNoticeDismissedProperty() {
        return overlayNoticeDismissed;
    }

    // ==================== Welcome Message ====================

    public static boolean isShowWelcomeMessage() {
        return showWelcomeMessage.get();
    }

    public static void setShowWelcomeMessage(boolean show) {
        showWelcomeMessage.set(show);
    }

    public static BooleanProperty showWelcomeMessageProperty() {
        return showWelcomeMessage;
    }
}
