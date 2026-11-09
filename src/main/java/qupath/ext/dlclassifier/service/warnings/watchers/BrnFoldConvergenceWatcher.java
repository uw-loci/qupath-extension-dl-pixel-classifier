package qupath.ext.dlclassifier.service.warnings.watchers;

import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.warnings.PreferenceWarning;

/**
 * Fires when INT8 inference is enabled. The INT8 path depends on
 * BatchRenorm being folded to BatchNorm at export time, which is
 * only safe once BRN's running statistics have converged. Very
 * short training runs (&lt;20 epochs) can produce BRN running
 * stats that are still noisy, so the folded BN graph may drift
 * from the training-time behaviour.
 * <p>
 * The E.3 Python-side fix adds an epoch threshold on
 * {@code fold_brn_to_bn} that skips the BN-folded export when
 * training is too short. This watcher surfaces the caveat to the
 * user at preference-toggle time so they know why INT8 may fall
 * back to FP16 on short-trained models.
 * <p>
 * Severity is INFO -- composition is correct, this is an
 * expectation-setting popup.
 */
public final class BrnFoldConvergenceWatcher implements PreferenceWarning {

    public static final String ID = "int8-brn-fold-convergence";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "INT8 quantization requires converged BatchRenorm stats";
    }

    @Override
    public String getDescription() {
        return "INT8 inference via TensorRT folds BatchRenorm into "
                + "plain BatchNorm at export time. That fold is only "
                + "safe after BRN's running statistics have "
                + "converged -- models trained for fewer than ~20 "
                + "epochs may produce a BN-folded ONNX whose eval "
                + "behavior drifts from training. For short runs, "
                + "INT8 inference falls back to FP16 automatically "
                + "and logs a warning. Re-export a longer-trained "
                + "model to get the full INT8 speedup.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-7-brn-fold-convergence";
    }

    @Override
    public Severity getSeverity() {
        return Severity.INFO;
    }

    @Override
    public boolean check() {
        return DLClassifierPreferences.isExperimentalInt8();
    }
}
