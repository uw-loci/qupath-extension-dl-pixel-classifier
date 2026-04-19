package qupath.ext.dlclassifier.service.warnings;

/**
 * A single pairwise option-interaction check that the extension
 * evaluates at training / inference / preference-change time.
 * <p>
 * Each watcher encodes one row from the 2026-04-19 interaction
 * table (see claude-reports) as a runtime check plus user-facing
 * text. The {@link InteractionWarningService} runs the relevant
 * watchers, collects those that returned {@code true} from their
 * {@code check(...)} method, and surfaces them in a single popup.
 * <p>
 * Two scopes are supported -- {@link TrainingWarning} and
 * {@link InferenceWarning} -- so a watcher does not need to
 * pretend to understand a config it has no business reading.
 * Watchers that apply at preference-toggle time implement
 * {@link PreferenceWarning}.
 * <p>
 * This is deliberately an interface (not an enum): individual
 * watcher classes can hold state (e.g. a cached computation),
 * and the registry is populated at startup in
 * {@link InteractionWarningService}.
 */
public interface InteractionWarning {

    /**
     * Stable identifier used as the key for the per-watcher
     * "Don't show again" preference. Pick something like
     * {@code "ohem-focal"} that will survive refactors.
     */
    String getId();

    /**
     * Short (under 60 chars) title shown as the row heading in
     * the popup.
     */
    String getTitle();

    /**
     * 1-3 sentence description of WHAT is happening and WHY it
     * matters. ASCII-only per project CLAUDE.md conventions.
     */
    String getDescription();

    /**
     * Optional docs anchor for a "Learn more" link, e.g. a path
     * into the interaction table, or a section heading in the
     * design-document report. Return {@code null} if none.
     */
    default String getDocsAnchor() {
        return null;
    }

    /**
     * Severity hint. Governs the icon used in the popup and
     * whether the popup can be silently suppressed. Defaults to
     * {@link Severity#WARN}.
     */
    default Severity getSeverity() {
        return Severity.WARN;
    }

    enum Severity {
        /** Informational -- "here's what we just did for you". */
        INFO,
        /** Default. Works but with a caveat. */
        WARN,
        /** Blocking. User must resolve before continuing. */
        BLOCKING
    }
}
