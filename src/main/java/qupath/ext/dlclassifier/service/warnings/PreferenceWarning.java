package qupath.ext.dlclassifier.service.warnings;

/**
 * Watcher that fires on a preference toggle rather than a
 * config object. The check is parameterless -- the watcher
 * consults {@link qupath.ext.dlclassifier.preferences.DLClassifierPreferences}
 * and any runtime state (e.g. cached model count) itself.
 */
public interface PreferenceWarning extends InteractionWarning {
    boolean check();
}
