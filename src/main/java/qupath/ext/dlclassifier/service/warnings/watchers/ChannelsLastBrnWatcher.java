package qupath.ext.dlclassifier.service.warnings.watchers;

import java.util.Map;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.service.warnings.InferenceWarning;

/**
 * Fires when the loaded model is known to contain BatchRenorm
 * layers, so the user understands why the channels_last inference
 * optimization is skipped (BRN's internal reshape ops can silently
 * undo the NHWC layout -- see inference_service.py
 * {@code _apply_channels_last}).
 * <p>
 * Detection: consults {@link ClassifierMetadata#getTrainingSettings()}
 * for a {@code norm} key equal to "brn". TinyUNet always records
 * this; SMP models that had {@code replace_bn_with_batchrenorm}
 * applied do NOT currently record it in metadata. Once metadata
 * gains an SMP-BRN flag, update the detection here.
 * <p>
 * Severity is INFO: the gate is correct, the user just wants to
 * know why they're not getting the ~1.1-1.3x inference speedup.
 */
public final class ChannelsLastBrnWatcher implements InferenceWarning {

    public static final String ID = "channels-last-brn";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getTitle() {
        return "channels_last inference layout skipped (BatchRenorm)";
    }

    @Override
    public String getDescription() {
        return "This model uses BatchRenorm. The channels_last "
                + "inference memory format is automatically skipped "
                + "for BRN models because the layer's internal "
                + "reshape operations can silently revert to the "
                + "default NCHW layout and waste the conversion. "
                + "This is expected behavior -- GroupNorm or plain "
                + "BatchNorm models still benefit from the "
                + "optimization.";
    }

    @Override
    public String getDocsAnchor() {
        return "section-7-channels-last-brn";
    }

    @Override
    public Severity getSeverity() {
        return Severity.INFO;
    }

    @Override
    public boolean check(InferenceConfig config, ClassifierMetadata metadata) {
        if (metadata == null) return false;
        Map<String, Object> settings = metadata.getTrainingSettings();
        if (settings == null) return false;
        Object norm = settings.get("norm");
        if (norm instanceof String s) {
            return "brn".equalsIgnoreCase(s);
        }
        // Fallback: some older metadata may carry the flag under
        // a different key. Extend here if/when we add an
        // SMP-specific BRN signal.
        return false;
    }
}
