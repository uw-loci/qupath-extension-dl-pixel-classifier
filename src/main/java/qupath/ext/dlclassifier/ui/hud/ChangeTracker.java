package qupath.ext.dlclassifier.ui.hud;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Turns a stream of control edits into one change at a time.
 * <p>
 * Spinner arrows, a held key and typing fire many events; the tracker waits
 * until the inputs have been still for the settle interval and treats the
 * whole burst as one change, compared against the reading from before the
 * burst began. The result stays until the next change -- there is no fade
 * timer -- so a user can look away and come back to what their last edit did.
 * <p>
 * No JavaFX here: the clock is a {@link Scheduler}, so tests run on a fake one.
 */
public final class ChangeTracker {

    /** Runs a task after a delay; the returned handle cancels it. */
    public interface Scheduler {
        Runnable schedule(Runnable task, long delayMs);
    }

    /** What the HUD should show. */
    public enum Mode {
        /** No data yet, or nothing has been changed since it arrived. */
        AT_REST,
        /** The last change, with per-gauge verdicts. */
        CHANGE,
        /** Controls changed, but no gauge moved visibly. */
        NO_GAUGE_MOVED
    }

    /**
     * One state of the HUD.
     *
     * @param now      the current reading
     * @param scale    the scale to draw it on
     * @param mode     what to show
     * @param changes  per-gauge changes; empty in AT_REST
     * @param controls the controls that changed, "Label old -> new"; empty in AT_REST
     * @param pinned   true when changes are against a pinned baseline, not the last edit
     */
    public record State(
            HudModel.Reading now,
            HudScale scale,
            Mode mode,
            Map<Gauge, HudDelta.Change> changes,
            List<String> controls,
            boolean pinned) {}

    private final Supplier<HudInputs> snapshot;
    private final Scheduler scheduler;
    private final long settleMs;
    private final double minPositionStep;
    private final Consumer<State> listener;

    private HudModel.Reading settled;
    private HudModel.Reading burstBefore;
    private HudModel.Reading pinnedReading;
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
     * did caused it. Shown at rest, with the scale re-centred on the new data.
     */
    public void reset() {
        if (cancelPending != null) {
            cancelPending.run();
            cancelPending = null;
        }
        settled = HudModel.read(snapshot.get());
        scale = HudScale.centredOn(settled);
        pinnedReading = null;
        publish(new State(settled, scale, Mode.AT_REST, Map.of(), List.of(), false));
    }

    /** Compare every later change against the current reading, not the last edit. */
    public void pin() {
        pinnedReading = settled;
        scale = HudScale.centredOn(settled);
        publish(new State(settled, scale, Mode.AT_REST, Map.of(), List.of(), true));
    }

    /** Go back to comparing against the last edit. */
    public void unpin() {
        pinnedReading = null;
        publish(new State(settled, scale, Mode.AT_REST, Map.of(), List.of(), false));
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
        if (now.inputs().sameSettings(previous.inputs())) {
            // Set and set back inside one burst, or an event that changed no
            // setting this HUD reads: leave the last change on screen.
            return;
        }
        HudModel.Reading before = pinnedReading != null ? pinnedReading : burstBefore;
        List<String> controls = now.inputs().changesFrom(before.inputs());
        if (controls.isEmpty()) {
            // Only reachable when pinned: the user walked back to the pinned
            // configuration, so there is nothing to compare.
            publish(new State(now, scale, Mode.AT_REST, Map.of(), List.of(), pinnedReading != null));
            return;
        }
        Map<Gauge, HudDelta.Change> changes = HudDelta.compare(before, now, scale, minPositionStep);
        boolean anyMoved = changes.values().stream().anyMatch(HudDelta.Change::moved);
        publish(new State(
                now, scale, anyMoved ? Mode.CHANGE : Mode.NO_GAUGE_MOVED, changes, controls, pinnedReading != null));
    }

    private void publish(State s) {
        last = Objects.requireNonNull(s);
        listener.accept(s);
    }
}
