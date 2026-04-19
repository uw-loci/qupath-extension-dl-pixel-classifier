package qupath.ext.dlclassifier.service.warnings;

import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;

/**
 * Watcher that runs at inference time. The metadata argument may
 * be null when the watcher only needs runtime inference
 * preferences (e.g. "TRT enabled but no TRT-capable model
 * cached"); guard accordingly.
 */
public interface InferenceWarning extends InteractionWarning {
    boolean check(InferenceConfig config, ClassifierMetadata metadata);
}
