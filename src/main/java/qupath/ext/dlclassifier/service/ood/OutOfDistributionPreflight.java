package qupath.ext.dlclassifier.service.ood;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.NormalizationStatsComputer;
import qupath.ext.dlclassifier.service.warnings.InteractionWarningService;
import qupath.ext.dlclassifier.service.warnings.OutOfDistributionWarning;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.images.servers.ImageServer;

/**
 * Orchestrates the pre-inference OOD check: samples image-level pixel
 * stats, compares them against the training stats in metadata, and
 * surfaces deviations via the requested {@link SurfaceMode}.
 * <p>
 * Cost: ~1-3 s of image sampling per call (4x4 tile grid, capped at
 * ~100k pixels per channel). Skipped when training stats are absent
 * from the model (old metadata) or when the user has dismissed the
 * warning via "Don't show again" in the popup.
 */
public final class OutOfDistributionPreflight {

    private static final Logger logger = LoggerFactory.getLogger(OutOfDistributionPreflight.class);

    /** How OOD deviations should reach the user. */
    public enum SurfaceMode {
        /** Modal dialog via {@link InteractionWarningService}. Blocks the caller until dismissed. */
        DIALOG,
        /** Non-blocking corner notification (warning toast). For paths that must not block, like overlay rendering. */
        NOTIFICATION,
        /** Log only. For headless / scripted callers where there is no UI to attach to. */
        LOG
    }

    private static final String NOTIFICATION_TITLE = "DL Pixel Classifier";

    private OutOfDistributionPreflight() {}

    /**
     * Run the OOD check and surface the result through {@code mode}.
     *
     * @param server      the inference image
     * @param metadata    classifier metadata (must have normalization stats; otherwise this is a no-op)
     * @param channelCfg  channel configuration used for inference
     * @param downsample  downsample factor for sampling tiles
     * @param mode        how to surface deviations to the user
     * @return true to proceed with inference, false if the user cancelled a blocking DIALOG warning;
     *     always true for NOTIFICATION and LOG modes
     */
    public static boolean run(
            ImageServer<BufferedImage> server,
            ClassifierMetadata metadata,
            ChannelConfiguration channelCfg,
            double downsample,
            SurfaceMode mode) {

        if (metadata == null || !metadata.hasNormalizationStats()) {
            return true;
        }
        if (InteractionWarningService.isSuppressed(OutOfDistributionWarning.ID)) {
            // User opted out -- skip even the sampling cost.
            return true;
        }

        List<Map<String, Double>> imageStats;
        try {
            imageStats = NormalizationStatsComputer.sampleImageStats(server, metadata, channelCfg, downsample);
        } catch (RuntimeException ex) {
            logger.warn("OOD preflight: image sampling failed, skipping check: {}", ex.getMessage());
            return true;
        }
        if (imageStats == null || imageStats.isEmpty()) {
            logger.debug("OOD preflight: no image stats available, skipping check");
            return true;
        }

        double sigma = DLClassifierPreferences.getOodSigmaThreshold();
        OodReport report = OutOfDistributionChecker.check(
                metadata.getNormalizationStats(), imageStats, channelCfg.getChannelNames(), sigma);

        if (!report.hasDeviation()) {
            logger.debug("OOD preflight: image stats within training distribution (sigma threshold={})", sigma);
            return true;
        }

        logger.warn(
                "OOD preflight: {} channel(s) outside training distribution (sigma threshold={}). {}",
                report.getDeviations().size(),
                sigma,
                summarizeForLog(report));

        switch (mode) {
            case DIALOG:
                return InteractionWarningService.showIfAny(List.of(new OutOfDistributionWarning(report)), null);
            case NOTIFICATION:
                Dialogs.showWarningNotification(NOTIFICATION_TITLE, buildNotificationText(report));
                return true;
            case LOG:
            default:
                return true;
        }
    }

    private static String buildNotificationText(OodReport report) {
        int n = report.getDeviations().size();
        String channels = (n == 1) ? "1 channel" : (n + " channels");
        return "Image distribution differs from training data ("
                + channels
                + "). Predictions may be unreliable. "
                + "See log for details, or use Apply Classifier for the full warning.";
    }

    private static String summarizeForLog(OodReport report) {
        StringBuilder sb = new StringBuilder();
        for (OodReport.ChannelDeviation d : report.getDeviations()) {
            sb.append(" [ch").append(d.channelIndex());
            if (d.channelName() != null) sb.append("/").append(d.channelName());
            sb.append(" ")
                    .append(d.worstMetric())
                    .append("=")
                    .append(String.format("%.2f", d.worstScore()))
                    .append("]");
        }
        return sb.toString().trim();
    }
}
