package qupath.ext.dlclassifier.ui.hud;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.ui.hud.ChangeTracker.Mode;
import qupath.ext.dlclassifier.ui.hud.HudDelta.Verdict;

/**
 * "If a user makes a change, they shouldn't need to toggle it back and forth
 * to see what changed" -- the tracker's half of that, on a fake clock.
 */
class ChangeTrackerTest {

    /** A clock that only moves when told to. */
    private static final class FakeScheduler implements ChangeTracker.Scheduler {
        long now;
        final List<long[]> due = new ArrayList<>();
        final List<Runnable> tasks = new ArrayList<>();

        @Override
        public Runnable schedule(Runnable task, long delayMs) {
            long[] at = {now + delayMs};
            due.add(at);
            tasks.add(task);
            return () -> at[0] = Long.MAX_VALUE;
        }

        void advance(long ms) {
            now += ms;
            for (int i = 0; i < tasks.size(); i++) {
                if (due.get(i)[0] <= now) {
                    due.get(i)[0] = Long.MAX_VALUE;
                    tasks.get(i).run();
                }
            }
        }
    }

    private static final long SETTLE = 350;

    private FakeScheduler clock;
    private List<ChangeTracker.State> states;
    private int batch;
    private int epochs;
    private boolean earlyStopping;
    private ChangeTracker tracker;

    @BeforeEach
    void setUp() {
        clock = new FakeScheduler();
        states = new ArrayList<>();
        batch = 16;
        epochs = 40;
        earlyStopping = true;
        tracker = new ChangeTracker(this::snapshot, clock, SETTLE, 0.01, states::add);
    }

    private HudInputs snapshot() {
        return new HudInputs(
                "unet",
                "resnet18",
                256,
                20,
                false,
                4,
                1,
                batch,
                1,
                epochs,
                true,
                20,
                earlyStopping,
                "auto",
                3,
                8,
                "Tissue|img:Both",
                true,
                600,
                150,
                24_576,
                32L << 30);
    }

    @Test
    void opensAtRest() {
        assertThat(states).hasSize(1);
        assertThat(states.get(0).mode()).isEqualTo(Mode.AT_REST);
    }

    @Test
    void tenSpinnerTicksAreOneChange() {
        for (int i = 0; i < 10; i++) {
            batch++;
            tracker.inputsChanged();
            clock.advance(SETTLE / 3);
        }
        assertThat(states).hasSize(1); // still inside the burst
        clock.advance(SETTLE);
        assertThat(states).hasSize(2);
        ChangeTracker.State s = states.get(1);
        assertThat(s.mode()).isEqualTo(Mode.CHANGE);
        assertThat(s.controls()).containsExactly("Batch Size 16 -> 26");
    }

    @Test
    void twoControlsInOneBurstAreOneChangeListingBoth() {
        batch = 32;
        tracker.inputsChanged();
        epochs = 80;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        assertThat(states).hasSize(2);
        assertThat(states.get(1).controls()).containsExactly("Epochs 40 -> 80", "Batch Size 16 -> 32");
    }

    @Test
    void theHighlightStaysUntilTheNextChange() {
        epochs = 80;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        ChangeTracker.State shown = tracker.state();
        clock.advance(60_000); // the user looks away for a minute
        assertThat(tracker.state()).isSameAs(shown);
        assertThat(shown.changes().get(Gauge.LEARNING_STEPS).verdict()).isEqualTo(Verdict.BETTER);
    }

    @Test
    void setAndSetBackInsideOneBurstLeavesTheLastChangeOnScreen() {
        epochs = 80;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        ChangeTracker.State shown = tracker.state();
        batch = 64;
        tracker.inputsChanged();
        batch = 16;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        assertThat(tracker.state()).isSameAs(shown);
    }

    @Test
    void aChangeThatMovesNoGaugeIsStillReported() {
        earlyStopping = false;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        assertThat(tracker.state().mode()).isEqualTo(Mode.NO_GAUGE_MOVED);
        assertThat(tracker.state().controls()).containsExactly("Early stopping on -> off");
    }

    @Test
    void eachChangeIsAgainstTheOneBeforeIt() {
        epochs = 80;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        epochs = 160;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        assertThat(tracker.state().controls()).containsExactly("Epochs 80 -> 160");
        assertThat(tracker.state().changes().get(Gauge.LEARNING_STEPS).ratio()).isEqualTo(2.0);
    }

    @Test
    void aPinnedBaselineHoldsAcrossSeveralEdits() {
        assertThat(tracker.pinCurrent()).isEqualTo("Pin 1");
        epochs = 80;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        batch = 32;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        epochs = 160;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        ChangeTracker.State s = tracker.state();
        assertThat(s.pinned()).isTrue();
        assertThat(s.controls()).containsExactly("Epochs 40 -> 160", "Batch Size 16 -> 32");
    }

    @Test
    void walkingBackToThePinnedConfigurationIsAtRest() {
        tracker.pinCurrent();
        epochs = 80;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        epochs = 40;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        assertThat(tracker.state().mode()).isEqualTo(Mode.AT_REST);
        assertThat(tracker.state().pinned()).isTrue();
    }

    private void edit(Runnable change) {
        change.run();
        tracker.inputsChanged();
        clock.advance(SETTLE);
    }

    @Test
    void startIsAlwaysThereSoNobodyHasToRememberToPin() {
        edit(() -> epochs = 80);
        edit(() -> batch = 32);
        edit(() -> earlyStopping = false);
        assertThat(tracker.state().controls()).containsExactly("Early stopping on -> off");
        tracker.compareTo(ChangeTracker.START);
        ChangeTracker.State s = tracker.state();
        assertThat(s.comparedTo()).isEqualTo(ChangeTracker.START);
        assertThat(s.controls()).containsExactly("Epochs 40 -> 80", "Early stopping on -> off", "Batch Size 16 -> 32");
        assertThat(s.mode()).isEqualTo(Mode.CHANGE);
    }

    /**
     * Loading a profile sets many controls in one action. It is shown as ONE
     * change under the profile's name, and becomes a baseline of its own.
     */
    @Test
    void aLoadedProfileIsOneNamedChangeAndBecomesABaseline() {
        tracker.settingsLoaded("Profile B");
        batch = 4;
        tracker.inputsChanged();
        epochs = 120;
        tracker.inputsChanged();
        earlyStopping = false;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        ChangeTracker.State s = tracker.state();
        assertThat(s.changeName()).isEqualTo("Profile B");
        assertThat(s.controls()).hasSize(3);
        assertThat(s.baselines()).containsExactly(ChangeTracker.START, "Profile B");
        // An ordinary edit afterwards is not named
        edit(() -> batch = 8);
        assertThat(tracker.state().changeName()).isNull();
    }

    @Test
    void comparingToAProfileShowsOnlyWhatChangedSinceIt() {
        tracker.settingsLoaded("Profile B");
        edit(() -> {
            batch = 4;
            epochs = 120;
        });
        edit(() -> epochs = 60);
        tracker.compareTo("Profile B");
        assertThat(tracker.state().controls()).containsExactly("Epochs 120 -> 60");
        assertThat(tracker.state().changes().get(Gauge.LEARNING_STEPS).verdict())
                .isEqualTo(HudDelta.Verdict.WORSE);
        // and the reference tick moves to the profile
        assertThat(tracker.state().scale().reference(Gauge.LEARNING_STEPS))
                .isEqualTo(HudModel.read(withEpochsAndBatch(120, 4))
                        .get(Gauge.LEARNING_STEPS)
                        .value());
    }

    @Test
    void switchingBetweenTwoProfilesIsOneChange() {
        tracker.settingsLoaded("A");
        edit(() -> {
            batch = 4;
            epochs = 120;
        });
        tracker.settingsLoaded("B");
        edit(() -> {
            batch = 32;
            epochs = 20;
            earlyStopping = false;
        });
        ChangeTracker.State s = tracker.state();
        assertThat(s.changeName()).isEqualTo("B");
        assertThat(s.controls()).containsExactly("Epochs 120 -> 20", "Early stopping on -> off", "Batch Size 4 -> 32");
        tracker.compareTo("A");
        assertThat(tracker.state().controls()).isEqualTo(s.controls());
    }

    @Test
    void identicalSettingsToTheBaselineAreAtRest() {
        tracker.compareTo(ChangeTracker.START);
        assertThat(tracker.state().mode()).isEqualTo(Mode.AT_REST);
        assertThat(tracker.state().comparedTo()).isEqualTo(ChangeTracker.START);
    }

    @Test
    void baselinesCanBeRemovedButStartCannot() {
        String pin = tracker.pinCurrent();
        tracker.removeBaseline(ChangeTracker.START);
        tracker.removeBaseline(pin);
        assertThat(tracker.state().baselines()).containsExactly(ChangeTracker.START);
        assertThat(tracker.state().comparedTo()).isNull();
    }

    @Test
    void anUnknownBaselineIsIgnored() {
        tracker.compareTo("nope");
        assertThat(tracker.state().comparedTo()).isNull();
    }

    private HudInputs withEpochsAndBatch(int e, int b) {
        int oldE = epochs;
        int oldB = batch;
        epochs = e;
        batch = b;
        HudInputs in = snapshot();
        epochs = oldE;
        batch = oldB;
        return in;
    }

    @Test
    void reloadingDataIsNotAChange() {
        epochs = 80;
        tracker.inputsChanged();
        clock.advance(SETTLE);
        tracker.pinCurrent();
        tracker.reset();
        assertThat(tracker.state().mode()).isEqualTo(Mode.AT_REST);
        assertThat(tracker.state().changes()).isEmpty();
        assertThat(tracker.state().baselines()).containsExactly(ChangeTracker.START);
    }
}
