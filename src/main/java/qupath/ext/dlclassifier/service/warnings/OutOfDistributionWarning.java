package qupath.ext.dlclassifier.service.warnings;

import qupath.ext.dlclassifier.service.ood.OodReport;

/**
 * Runtime warning fired when an image's pixel distribution differs
 * substantially from the training-time distribution recorded in a
 * classifier's metadata.
 * <p>
 * Unlike the watcher-style {@link InferenceWarning}s in
 * {@code service.warnings.watchers}, this one is built ad hoc at
 * inference start because its description depends on the runtime
 * comparison result. The shared id keeps the per-watcher
 * "Don't show again" preference stable across instances.
 */
public final class OutOfDistributionWarning implements InteractionWarning {

    public static final String ID = "ood-distribution-shift";

    private final OodReport report;

    public OutOfDistributionWarning(OodReport report) {
        this.report = report;
    }

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        int n = report.getDeviations().size();
        if (n == 1) {
            return "Image distribution differs from training data (1 channel)";
        }
        return "Image distribution differs from training data (" + n + " channels)";
    }

    @Override
    public String getDescription() {
        return report.describe();
    }

    @Override
    public String getDocsAnchor() {
        return "ood-distribution-shift";
    }

    @Override
    public Severity getSeverity() {
        return Severity.WARN;
    }
}
