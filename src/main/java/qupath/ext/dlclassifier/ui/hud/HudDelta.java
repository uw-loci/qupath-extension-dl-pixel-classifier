package qupath.ext.dlclassifier.ui.hud;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * What one change did to each gauge. Pure: no JavaFX.
 * <p>
 * Each gauge is judged alone against its own polarity. There is deliberately
 * no overall verdict: the reason this HUD exists is a change that made one
 * gauge better and another worse (downsample 1 -> 8: faster per epoch, far
 * fewer learning steps), and an overall "good change" would have hidden it.
 */
public final class HudDelta {

    private HudDelta() {}

    /** How a gauge is coloured after a change. */
    public enum Verdict {
        /** Did not move visibly; drawn at rest. */
        SAME,
        /** Moved, better for the user: blue. */
        BETTER,
        /** Moved, worse for the user: orange. */
        WORSE,
        /** Moved, and the comparison cannot be made honestly (e.g. across architectures). */
        UNKNOWN,
        /** Has data now and had none before, or the reverse. Not coloured. */
        APPEARED_OR_GONE
    }

    /**
     * One gauge's change.
     *
     * @param gauge          the gauge
     * @param verdict        how to colour it
     * @param ratio          now / before for the raw value; NaN when not comparable
     * @param bandCrossed    the reading crossed one of the gauge's hard lines
     * @param positionBefore needle position before, 0-1 on the given scale; NaN when unavailable
     * @param positionNow    needle position now, 0-1; NaN when unavailable
     */
    public record Change(
            Gauge gauge,
            Verdict verdict,
            double ratio,
            boolean bandCrossed,
            double positionBefore,
            double positionNow) {

        /** @return true when the gauge should be highlighted at all */
        public boolean moved() {
            return verdict != Verdict.SAME;
        }
    }

    /**
     * Compares two readings.
     *
     * @param before         the reading before the change (or the pinned baseline)
     * @param now            the reading after it
     * @param scale          maps values to needle positions
     * @param minPositionStep smallest needle travel, as a fraction of the scale,
     *                       that the view can draw; smaller moves are SAME, so
     *                       hue is never spent on a change nobody can see
     * @return one change per gauge
     */
    public static Map<Gauge, Change> compare(
            HudModel.Reading before, HudModel.Reading now, HudScale scale, double minPositionStep) {
        Map<Gauge, Change> out = new EnumMap<>(Gauge.class);
        boolean sameArchitecture = Objects.equals(
                        before.inputs().architecture(), now.inputs().architecture())
                && Objects.equals(before.inputs().backbone(), now.inputs().backbone());
        for (Gauge g : Gauge.values()) {
            GaugeReading b = before.get(g);
            GaugeReading n = now.get(g);
            double pb = b.available() ? scale.position(g, b.value()) : Double.NaN;
            double pn = n.available() ? scale.position(g, n.value()) : Double.NaN;
            if (b.available() != n.available()) {
                out.put(g, new Change(g, Verdict.APPEARED_OR_GONE, Double.NaN, b.band() != n.band(), pb, pn));
                continue;
            }
            if (!n.available()) {
                out.put(g, new Change(g, Verdict.SAME, Double.NaN, false, pb, pn));
                continue;
            }
            double ratio = b.value() == 0 ? (n.value() == 0 ? 1.0 : Double.POSITIVE_INFINITY) : n.value() / b.value();
            boolean crossed = b.band() != n.band();
            // A band crossing is always shown, however small the needle move:
            // it is the change the user most needs to see.
            double ub = scale.unclampedPosition(g, b.value());
            double un = scale.unclampedPosition(g, n.value());
            boolean visible = crossed || (ub != un && !(Math.abs(un - ub) < minPositionStep));
            Verdict v;
            if (!sameArchitecture && (g == Gauge.EPOCH_SPEED || g == Gauge.TOTAL_TIME)) {
                // The work formula has no per-architecture cost term, and none
                // is measured. Painting a guess blue or orange would break the
                // one promise the colours make -- and so would SAME: an
                // architecture-only change leaves the formula's value exactly
                // where it was, while the real time per epoch can change many
                // times over. Checked before visibility for that reason.
                v = Verdict.UNKNOWN;
            } else if (!visible) {
                v = Verdict.SAME;
            } else {
                boolean up = n.value() > b.value();
                boolean better = (g.polarity() == Gauge.Polarity.HIGHER_BETTER) == up;
                v = better ? Verdict.BETTER : Verdict.WORSE;
            }
            out.put(g, new Change(g, v, ratio, crossed, pb, pn));
        }
        return Collections.unmodifiableMap(out);
    }
}
