package qupath.ext.dlclassifier.ui.hud;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Turns a stream of control edits into one change at a time, compared against
 * either the edit before it or a named baseline.
 * <p>
 * Spinner arrows, a held key and typing fire many events; the tracker waits
 * until the inputs have been still for the settle interval and treats the
 * whole burst as one change. The result stays until the next change -- there
 * is no fade timer -- so a user can look away and come back to what their last
 * edit did.
 * <p>
 * Baselines are whole configurations to compare against: {@link #START} is
 * taken whenever data arrives, every loaded profile or model's settings becomes
 * one under its own name ({@link #settingsLoaded}), and the user can pin the
 * current settings ({@link #pinCurrent}). Comparing against a baseline answers
 * "how far is this from profile B?" however many settings that spans; loading
 * a profile is itself one change, named, however many controls it set.
 * <p>
 * No JavaFX here: the clock is a {@link Scheduler}, so tests run on a fake one.
 */
public final class ChangeTracker {

    /** The baseline taken whenever data arrives: compare to it to see everything since. */
    public static final String START = "Start";

    /** Runs a task after a delay; the returned handle cancels it. */
    public interface Scheduler {
        Runnable schedule(Runnable task, long delayMs);
    }

    /** What the HUD should show. */
    public enum Mode {
        /** No data yet, nothing changed since it arrived, or identical to the baseline. */
        AT_REST,
        /** A change, with per-gauge verdicts. */
        CHANGE,
        /** Controls changed, but no gauge moved visibly. */
        NO_GAUGE_MOVED
    }

    /**
     * One state of the HUD.
     *
     * @param now        the current reading
     * @param scale      the scale to draw it on; its reference is the baseline compared against,
     *                   or {@link #START} when comparing against the last edit
     * @param mode       what to show
     * @param changes    per-gauge changes; empty in AT_REST
     * @param controls   the controls that differ, "Label old -> new"; empty in AT_REST
     * @param changeName a name for the change when it was one action that set many controls
     *                   ("Profile B"), or null for an ordinary edit
     * @param comparedTo the baseline compared against, or null for "the edit before"
     * @param baselines  every baseline's name, oldest first
     */
    public record State(
            HudModel.Reading now,
            HudScale scale,
            Mode mode,
            Map<Gauge, HudDelta.Change> changes,
            List<String> controls,
            String changeName,
            String comparedTo,
            List<String> baselines) {

        /** @return true when changes are against a baseline rather than the edit before */
        public boolean pinned() {
            return comparedTo != null;
        }
    }

    private final Supplier<HudInputs> snapshot;
    private final Scheduler scheduler;
    private final long settleMs;
    private final double minPositionStep;
    private final Consumer<State> listener;

    private final Map<String, HudModel.Reading> baselines = new LinkedHashMap<>();
    private HudModel.Reading settled;
    private HudModel.Reading burstBefore;
    private String comparedTo;
    private String pendingSettingsName;
    private int pinCount;
    private HudScale scale;
    private Runnable cancelPending;
    private State last;

    /**
     * @param snapshot        reads the dialog's current inputs
     * @param scheduler       the clock
     * @param settleMs        how long the inputs must be still before a burst counts as one change
     * @param minPositionStep smallest visible needle travel, as a fraction of a scale
     * @param listener        receives every new state
     */
    public ChangeTracker(
            Supplier<HudInputs> snapshot,
            Scheduler scheduler,
            long settleMs,
            double minPositionStep,
            Consumer<State> listener) {
        this.snapshot = snapshot;
        this.scheduler = scheduler;
        this.settleMs = settleMs;
        this.minPositionStep = minPositionStep;
        this.listener = listener;
        reset();
    }

    /** Call from every input property's listener. */
    public void inputsChanged() {
        if (cancelPending == null) {
            burstBefore = settled;
        } else {
            cancelPending.run();
        }
        cancelPending = scheduler.schedule(this::settle, settleMs);
    }

    /**
     * Data was (re)loaded: every gauge may move at once, and nothing the user
     * did caused it. Shown at rest. Baselines are dropped -- their patch counts
     * belong to the old data -- and {@link #START} is taken afresh.
     */
    public void reset() {
        if (cancelPending != null) {
            cancelPending.run();
            cancelPending = null;
        }
        settled = HudModel.read(snapshot.get());
        baselines.clear();
        baselines.put(START, settled);
        comparedTo = null;
        pendingSettingsName = null;
        pinCount = 0;
        scale = HudScale.centredOn(settled);
        publishAtRest();
    }

    /**
     * A whole set of settings is about to be applied -- a profile, or the
     * settings of a saved model. The burst it causes is shown as one change
     * under this name, and its result becomes a baseline of the same name.
     * Call it just before the controls are set.
     *
     * @param name what to call it, e.g. the profile's name
     */
    public void settingsLoaded(String name) {
        pendingSettingsName = name == null || name.isBlank() ? "Loaded settings" : name.trim();
    }

    /**
     * Pins the current settings as a new baseline and compares against it.
     *
     * @return the name it was given
     */
    public String pinCurrent() {
        String name;
        do {
            name = "Pin " + (++pinCount);
        } while (baselines.containsKey(name));
        baselines.put(name, settled);
        compareTo(name);
        return name;
    }

    /**
     * Chooses what changes are measured against, and shows the result at once.
     *
     * @param name a baseline's name, or null for "the edit before"
     */
    public void compareTo(String name) {
        if (name != null && !baselines.containsKey(name)) {
            return;
        }
        comparedTo = name;
        scale = HudScale.centredOn(baselines.get(name == null ? START : name));
        if (name == null) {
            publishAtRest();
            return;
        }
        publishAgainst(baselines.get(name), null);
    }

    /**
     * Removes a baseline. {@link #START} cannot be removed.
     *
     * @param name the baseline
     */
    public void removeBaseline(String name) {
        if (START.equals(name) || baselines.remove(name) == null) {
            return;
        }
        if (name.equals(comparedTo)) {
            compareTo(null);
        } else {
            publish(withBaselines(last));
        }
    }

    /** @return the most recent state */
    public State state() {
        return last;
    }

    private void settle() {
        cancelPending = null;
        HudModel.Reading now = HudModel.read(snapshot.get());
        HudModel.Reading previous = settled;
        settled = now;
        String loadedName = pendingSettingsName;
        pendingSettingsName = null;
        if (loadedName != null) {
            baselines.put(loadedName, now);
        }
        if (now.inputs().sameSettings(previous.inputs())) {
            // Set and set back inside one burst, an event that changed no
            // setting this HUD reads, or a profile identical to what was there:
            // leave the last change on screen (with the new baseline listed).
            if (loadedName != null) {
                publish(withBaselines(last));
            }
            return;
        }
        HudModel.Reading before = comparedTo != null ? baselines.get(comparedTo) : burstBefore;
        publishAgainst(before, loadedName);
    }

    private void publishAgainst(HudModel.Reading before, String changeName) {
        List<String> controls = settled.inputs().changesFrom(before.inputs());
        if (controls.isEmpty()) {
            // Only reachable against a baseline: the current settings are that
            // baseline's, so there is nothing to compare.
            publishAtRest();
            return;
        }
        Map<Gauge, HudDelta.Change> changes = HudDelta.compare(before, settled, scale, minPositionStep);
        boolean anyMoved = changes.values().stream().anyMatch(HudDelta.Change::moved);
        publish(new State(
                settled,
                scale,
                anyMoved ? Mode.CHANGE : Mode.NO_GAUGE_MOVED,
                changes,
                controls,
                changeName,
                comparedTo,
                baselineNames()));
    }

    private void publishAtRest() {
        publish(new State(settled, scale, Mode.AT_REST, Map.of(), List.of(), null, comparedTo, baselineNames()));
    }

    private State withBaselines(State s) {
        return new State(
                s.now(),
                s.scale(),
                s.mode(),
                s.changes(),
                s.controls(),
                s.changeName(),
                s.comparedTo(),
                baselineNames());
    }

    private List<String> baselineNames() {
        return List.copyOf(new ArrayList<>(baselines.keySet()));
    }

    private void publish(State s) {
        last = Objects.requireNonNull(s);
        listener.accept(s);
    }
}
