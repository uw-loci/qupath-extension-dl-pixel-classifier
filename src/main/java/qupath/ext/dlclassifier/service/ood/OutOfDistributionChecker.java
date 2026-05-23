package qupath.ext.dlclassifier.service.ood;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Per-channel out-of-distribution (OOD) comparison between image-level
 * pixel statistics and the training-time statistics stored in a
 * classifier's metadata.
 * <p>
 * Both sides are expected in the schema produced by
 * {@code NormalizationStatsComputer}: each channel is a map with keys
 * {@code p1, p99, min, max, mean, std} in the raw pixel range (uint8 in
 * [0, 255] or float as stored).
 * <p>
 * Limitations: this is a moment-based check. It catches gross intensity,
 * contrast, and dynamic-range shifts (different exposure, stain, sensor
 * bit-depth, missing/extra channels) but will not catch subtle texture
 * or morphology drift that still confuses the model. See option 2
 * (histogram / KL) and option 3 (model-based) for stronger signals.
 */
public final class OutOfDistributionChecker {

    private static final Logger logger = LoggerFactory.getLogger(OutOfDistributionChecker.class);

    /** Small constant to avoid division by zero when training std is ~0. */
    private static final double EPS = 1e-6;

    /** Default contrast-shift threshold: ln(2) means image std differs from training by 2x. */
    public static final double DEFAULT_STD_LOG_THRESHOLD = Math.log(2.0);

    private OutOfDistributionChecker() {}

    /**
     * Compare image stats against training stats and return a report of
     * channels that exceed the threshold on any measured metric.
     *
     * @param trainingStats    per-channel stats from model metadata; may be null
     * @param imageStats       per-channel stats sampled from the inference image; may be null
     * @param channelNames     channel names (for the report); may be null or shorter than stats
     * @param thresholdSigma   z-score threshold for mean / p1 / p99 deviations (default 3.0)
     * @param stdLogThreshold  |ln(img_std/train_std)| threshold (default ln(2) ~ 0.69)
     * @return report (may be empty); never null
     */
    public static OodReport check(
            List<Map<String, Double>> trainingStats,
            List<Map<String, Double>> imageStats,
            List<String> channelNames,
            double thresholdSigma,
            double stdLogThreshold) {

        List<OodReport.ChannelDeviation> deviations = new ArrayList<>();
        if (trainingStats == null || imageStats == null) {
            return new OodReport(deviations, thresholdSigma, stdLogThreshold);
        }

        int n = Math.min(trainingStats.size(), imageStats.size());
        if (n == 0) {
            return new OodReport(deviations, thresholdSigma, stdLogThreshold);
        }

        for (int c = 0; c < n; c++) {
            Map<String, Double> ts = trainingStats.get(c);
            Map<String, Double> is = imageStats.get(c);
            if (ts == null || is == null) continue;

            double trainMean = get(ts, "mean", 0.0);
            double trainStd = get(ts, "std", 0.0);
            double trainP1 = get(ts, "p1", 0.0);
            double trainP99 = get(ts, "p99", 0.0);

            double imgMean = get(is, "mean", trainMean);
            double imgStd = get(is, "std", trainStd);
            double imgP1 = get(is, "p1", trainP1);
            double imgP99 = get(is, "p99", trainP99);

            // Z-score-like measures use training std as the natural scale.
            // Falls back to a percent-of-mean scale when training std is
            // pathologically small (uniform channel), so we still flag
            // unambiguous drift instead of dividing by zero.
            double scale = Math.max(trainStd, Math.max(Math.abs(trainMean) * 0.01, EPS));
            double meanZ = Math.abs(imgMean - trainMean) / scale;
            double p1Z = Math.abs(imgP1 - trainP1) / scale;
            double p99Z = Math.abs(imgP99 - trainP99) / scale;

            // Contrast ratio: log so 0.5x and 2x are symmetric.
            double stdLogRatio = 0.0;
            if (trainStd > EPS && imgStd > EPS) {
                stdLogRatio = Math.abs(Math.log(imgStd / trainStd));
            } else if (trainStd > EPS || imgStd > EPS) {
                // One side is degenerate -- flag it as a large contrast shift
                stdLogRatio = stdLogThreshold * 2;
            }

            String worstMetric = "mean";
            double worstNormalized = meanZ / thresholdSigma;
            double worstRaw = meanZ;
            if (p1Z / thresholdSigma > worstNormalized) {
                worstNormalized = p1Z / thresholdSigma;
                worstRaw = p1Z;
                worstMetric = "p1";
            }
            if (p99Z / thresholdSigma > worstNormalized) {
                worstNormalized = p99Z / thresholdSigma;
                worstRaw = p99Z;
                worstMetric = "p99";
            }
            if (stdLogRatio / stdLogThreshold > worstNormalized) {
                worstNormalized = stdLogRatio / stdLogThreshold;
                worstRaw = stdLogRatio;
                worstMetric = "std";
            }

            if (worstNormalized >= 1.0) {
                String name = (channelNames != null && c < channelNames.size()) ? channelNames.get(c) : null;
                deviations.add(new OodReport.ChannelDeviation(
                        c,
                        name,
                        imgMean,
                        trainMean,
                        trainStd,
                        meanZ,
                        imgP1,
                        trainP1,
                        p1Z,
                        imgP99,
                        trainP99,
                        p99Z,
                        imgStd,
                        trainStd,
                        stdLogRatio,
                        worstRaw,
                        worstMetric));
                logger.info(
                        "OOD: channel {} '{}' worst metric '{}' score={} (threshold mean/p1/p99 sigma={}, std log={})",
                        c,
                        name,
                        worstMetric,
                        String.format("%.2f", worstRaw),
                        thresholdSigma,
                        String.format("%.2f", stdLogThreshold));
            }
        }

        return new OodReport(deviations, thresholdSigma, stdLogThreshold);
    }

    /**
     * Convenience overload with the default contrast threshold.
     */
    public static OodReport check(
            List<Map<String, Double>> trainingStats,
            List<Map<String, Double>> imageStats,
            List<String> channelNames,
            double thresholdSigma) {
        return check(trainingStats, imageStats, channelNames, thresholdSigma, DEFAULT_STD_LOG_THRESHOLD);
    }

    private static double get(Map<String, Double> m, String key, double fallback) {
        Double v = m.get(key);
        return (v == null || v.isNaN() || v.isInfinite()) ? fallback : v;
    }
}
