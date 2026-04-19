package qupath.ext.dlclassifier.service.warnings;

import qupath.ext.dlclassifier.service.warnings.watchers.BrnFoldConvergenceWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.ChannelsLastBrnWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.ExperimentalProvidersToggleWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.InMemoryCacheWorkersWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.OhemFocalWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.PlateauValLossWatcher;
import qupath.ext.dlclassifier.service.warnings.watchers.TileOverlapSplitWatcher;

/**
 * Static startup hook that registers every
 * {@link InteractionWarning} shipped with the extension.
 * <p>
 * Called from {@code SetupDLClassifier.installExtension}. Add new
 * watchers here as they are authored; the list stays short because
 * each watcher is a one-line register call and the watchers
 * themselves own their titles, descriptions, and check logic.
 * <p>
 * When removing a deprecated watcher, keep its id reserved by
 * listing it in a source-level comment so future additions do not
 * accidentally reuse the id (which would also reuse the stored
 * "Don't show again" preference).
 */
public final class InteractionWarningRegistration {

    private InteractionWarningRegistration() {}

    public static void registerAll() {
        // Training-scope watchers.
        InteractionWarningService.register(new OhemFocalWatcher());
        InteractionWarningService.register(new InMemoryCacheWorkersWatcher());
        InteractionWarningService.register(new TileOverlapSplitWatcher());
        InteractionWarningService.register(new PlateauValLossWatcher());

        // Inference-scope watchers.
        InteractionWarningService.register(new ChannelsLastBrnWatcher());

        // Preference-toggle watchers.
        InteractionWarningService.register(new BrnFoldConvergenceWatcher());
        InteractionWarningService.register(new ExperimentalProvidersToggleWatcher());

        // Default-suppress regression tripwires whose underlying bug
        // has been fixed. They stay in the registry (so a future code
        // regression that reintroduces the bad composition can be
        // detected by flipping the preference back on) but do not
        // greet users with a popup on every training run. Users who
        // explicitly want the informational confirmation can uncheck
        // the "Don't show again" checkbox in preferences.
        suppressIfUnset(OhemFocalWatcher.ID);
        suppressIfUnset(PlateauValLossWatcher.ID);
    }

    /**
     * Mark a watcher's suppression preference as {@code true} only
     * when the user has not already interacted with it. We check the
     * current value (which defaults to false) and only write when it
     * is false -- this preserves explicit opt-ins from users who want
     * the tripwire visible.
     * <p>
     * Note: PathPrefs does not distinguish "never set" from "false"
     * out of the box. The heuristic here is that any explicit "show
     * this" action from the user would have set the value to false
     * (from the checkbox), which is indistinguishable from "never
     * seen". That is acceptable for regression tripwires: the cost
     * of occasional re-suppression on upgrade is low.
     */
    private static void suppressIfUnset(String watcherId) {
        if (!InteractionWarningService.isSuppressed(watcherId)) {
            InteractionWarningService.setSuppressed(watcherId, true);
        }
    }
}
