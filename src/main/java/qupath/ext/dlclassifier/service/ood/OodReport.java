package qupath.ext.dlclassifier.service.ood;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Result of an out-of-distribution check comparing image-level pixel
 * statistics against the training-time pixel statistics stored in a
 * classifier's metadata.
 * <p>
 * The report holds one {@link ChannelDeviation} per channel that exceeded
 * the configured threshold on at least one metric (mean, p1, p99, std).
 * An empty report means the image is within the model's training
 * distribution on every measured channel.
 */
public final class OodReport {

    /**
     * Per-metric deviation magnitude for one channel.
     * <p>
     * For mean / p1 / p99 the score is a z-score-like quantity:
     * {@code |image - training| / max(training_std, eps)}.
     * For std the score is {@code |ln(image_std / training_std)|}: 0 when
     * the contrast matches, {@code ln(2) ~ 0.69} when image contrast is
     * half or double the training contrast.
     */
    public record ChannelDeviation(
            int channelIndex,
            String channelName,
            double imageMean,
            double trainingMean,
            double trainingStd,
            double meanZ,
            double imageP1,
            double trainingP1,
            double p1Z,
            double imageP99,
            double trainingP99,
            double p99Z,
            double imageStd,
            double trainingStdValue,
            double stdLogRatio,
            double worstScore,
            String worstMetric) {}

    private final List<ChannelDeviation> deviations;
    private final double thresholdSigma;
    private final double stdLogThreshold;

    public OodReport(List<ChannelDeviation> deviations, double thresholdSigma, double stdLogThreshold) {
        this.deviations = Collections.unmodifiableList(new ArrayList<>(deviations));
        this.thresholdSigma = thresholdSigma;
        this.stdLogThreshold = stdLogThreshold;
    }

    public List<ChannelDeviation> getDeviations() {
        return deviations;
    }

    public boolean hasDeviation() {
        return !deviations.isEmpty();
    }

    public double getThresholdSigma() {
        return thresholdSigma;
    }

    public double getStdLogThreshold() {
        return stdLogThreshold;
    }

    /**
     * Human-readable summary suitable for the description field of an
     * {@link qupath.ext.dlclassifier.service.warnings.InteractionWarning}.
     * ASCII-only per project conventions.
     */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        sb.append("The current image's pixel distribution differs from the model's training data ");
        sb.append("on the following channel(s):\n\n");
        for (ChannelDeviation d : deviations) {
            sb.append("  Channel ").append(d.channelIndex());
            if (d.channelName() != null && !d.channelName().isEmpty()) {
                sb.append(" (").append(d.channelName()).append(")");
            }
            sb.append(": ");
            sb.append(formatMetric(d));
            sb.append("\n");
        }
        sb.append("\nPredictions may be unreliable when the image's pixel distribution ");
        sb.append("falls outside what the model saw during training. Consider:\n");
        sb.append("  - Applying stain / white-balance / exposure normalization first\n");
        sb.append("  - Retraining the model on data that includes this image's appearance\n");
        sb.append("  - Using a different model trained on similar images\n");
        return sb.toString();
    }

    private String formatMetric(ChannelDeviation d) {
        switch (d.worstMetric()) {
            case "mean":
                return String.format(
                        "mean=%.1f vs training %.1f +/- %.1f (z=%.1f)",
                        d.imageMean(), d.trainingMean(), d.trainingStd(), d.meanZ());
            case "p1":
                return String.format("p1=%.1f vs training %.1f (z=%.1f)", d.imageP1(), d.trainingP1(), d.p1Z());
            case "p99":
                return String.format("p99=%.1f vs training %.1f (z=%.1f)", d.imageP99(), d.trainingP99(), d.p99Z());
            case "std":
                return String.format(
                        "std=%.1f vs training %.1f (ratio=%.2fx)",
                        d.imageStd(),
                        d.trainingStdValue(),
                        d.trainingStdValue() > 0 ? d.imageStd() / d.trainingStdValue() : Double.NaN);
            default:
                return String.format("score=%.2f", d.worstScore());
        }
    }
}
