package qupath.ext.dlclassifier.ui.hud;

import java.util.EnumMap;
import java.util.Map;

/**
 * Maps gauge values to needle positions in 0-1.
 * <p>
 * FRACTION gauges are linear from 0 to {@link #FRACTION_FULL_SCALE} of
 * capacity, so the region past 100% is visible as a hatched overrun. RELATIVE
 * gauges are logarithmic around a reference reading: the reference sits at the
 * middle and each end is {@link #RELATIVE_SPAN} times larger or smaller, so a
 * doubling moves the needle the same distance anywhere on the dial.
 * <p>
 * The reference is the first full reading (when classes are loaded) unless the
 * user pins another. It is a comparison point, not a recommendation.
 */
public final class HudScale {

    /** FRACTION gauges run to 120% of capacity so an overrun is drawn, not clipped. */
    public static final double FRACTION_FULL_SCALE = 1.2;

    /** RELATIVE gauges span this factor either side of the reference. */
    public static final double RELATIVE_SPAN = 16.0;

    private final Map<Gauge, Double> reference = new EnumMap<>(Gauge.class);

    /**
     * Uses a reading's values as the reference for every RELATIVE gauge that
     * has data in it.
     *
     * @param r the reference reading
     * @return a scale centred on it
     */
    public static HudScale centredOn(HudModel.Reading r) {
        HudScale s = new HudScale();
        for (Gauge g : Gauge.values()) {
            GaugeReading gr = r.get(g);
            if (g.scale() == Gauge.Scale.RELATIVE && gr.available() && gr.value() > 0) {
                s.reference.put(g, gr.value());
            }
        }
        return s;
    }

    /** @return the reference value for a RELATIVE gauge, or NaN when it has none yet */
    public double reference(Gauge g) {
        return reference.getOrDefault(g, Double.NaN);
    }

    /**
     * Needle position.
     *
     * @param g     the gauge
     * @param value its raw value
     * @return 0-1, clamped; 0.5 for a RELATIVE gauge with no reference yet
     */
    public double position(Gauge g, double value) {
        return clamp(unclampedPosition(g, value));
    }

    /**
     * Needle position before clamping to the dial. Used to decide whether a
     * change is visible: a needle already pegged at an end must still report a
     * further real change, even though it cannot travel any further.
     *
     * @param g     the gauge
     * @param value its raw value
     * @return the position, possibly outside 0-1; negative infinity for a
     *     RELATIVE value of zero
     */
    public double unclampedPosition(Gauge g, double value) {
        if (Double.isNaN(value)) return Double.NaN;
        if (g.scale() == Gauge.Scale.FRACTION) {
            return value / FRACTION_FULL_SCALE;
        }
        double ref = reference(g);
        if (Double.isNaN(ref)) return 0.5;
        if (value <= 0) return Double.NEGATIVE_INFINITY;
        double octaves = Math.log(value / ref) / Math.log(2);
        double halfSpan = Math.log(RELATIVE_SPAN) / Math.log(2);
        return 0.5 + 0.5 * octaves / halfSpan;
    }

    private static double clamp(double v) {
        return Math.max(0, Math.min(1, v));
    }
}
