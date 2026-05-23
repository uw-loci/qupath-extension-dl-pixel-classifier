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
import qupath.lib.images.servers.ImageServer;

/**
 * Orchestrates the pre-inference OOD check: samples image-level pixel
 * stats, compares them against the training stats in metadata, and
 * either pops up a dialog (interactive path) or logs a warning
 * (headless path).
 * <p>
 * Cost: ~1-3 s of image sampling per call (4x4 tile grid, capped at
 * ~100k pixels per channel). Skipped when training stats are absent
 * from the model (old metadata) or when the user has dismissed the
 * warning via "Don't show again" in the popup.
 */
public final class OutOfDistributionPreflight {

    private static final Logger logger = LoggerFactory.getLogger(OutOfDistributionPreflight.class);

    private OutOfDistributionPreflight() {}

    /**
     * Run the OOD check and surface the result.
     *
     * @param server      the inference image
     * @param metadata    classifier metadata (must have normalization stats; otherwise this is a no-op)
     * @param channelCfg  channel configuration used for inference
     * @param downsample  downsample factor for sampling tiles
     * @param interactive when true, show a popup; when false (headless), only log
     * @return true to proceed with inference, false if the user cancelled a blocking warning
     */
    public static boolean run(
            ImageServer<BufferedImage> server,
            ClassifierMetadata metadata,
            ChannelConfiguration channelCfg,
            double downsample,
            boolean interactive) {

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

        OutOfDistributionWarning warning = new OutOfDistributionWarning(report);
        logger.warn(
                "OOD preflight: {} channel(s) outside training distribution (sigma threshold={}). {}",
                report.getDeviations().size(),
                sigma,
                summarizeForLog(report));

        if (!interactive) {
            // Headless: log only, do not block.
            return true;
        }

        return InteractionWarningService.showIfAny(List.of(warning), null);
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
