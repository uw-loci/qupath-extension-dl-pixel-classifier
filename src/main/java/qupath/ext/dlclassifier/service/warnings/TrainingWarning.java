package qupath.ext.dlclassifier.service.warnings;

import qupath.ext.dlclassifier.model.TrainingConfig;

/**
 * Watcher that runs on a {@link TrainingConfig} before training
 * starts. Implementations inspect the config for a specific
 * pairwise interaction and return {@code true} when triggered.
 */
public interface TrainingWarning extends InteractionWarning {
    boolean check(TrainingConfig config);
}
