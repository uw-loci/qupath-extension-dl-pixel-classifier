package qupath.ext.dlclassifier.preferences;

import javafx.beans.property.*;
import javafx.collections.ObservableList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
            "dlclassifier.defaultLearningRate", 0.001);

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

    // ==================== Inference Dialog Preferences ====================

    private static final StringProperty lastOutputType = PathPrefs.createPersistentPreference(
            "dlclassifier.lastOutputType", "RENDERED_OVERLAY");

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

    // One-time overlay notice dismissed
    private static final BooleanProperty overlayNoticeDismissed = PathPrefs.createPersistentPreference(
            "dlclassifier.overlayNoticeDismissed", false);

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
}
