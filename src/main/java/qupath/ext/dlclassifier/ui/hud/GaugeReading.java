package qupath.ext.dlclassifier.ui.hud;

/**
 * One gauge's reading.
 *
 * @param gauge      which gauge
 * @param available  false when the gauge has no data (classes not loaded, no
 *                   GPU detected, cache off); the view draws it dimmed
 * @param value      the raw value: a fraction for FRACTION gauges, an absolute
 *                   amount (pixels, steps, patches) for RELATIVE ones. NaN when
 *                   unavailable
 * @param calibrated false when the estimate behind it rests on no measurement;
 *                   the view draws the needle hollow
 * @param band       where the value sits against the gauge's hard lines
 * @param note       short reason for an unavailable or flagged reading, or ""
 */
public record GaugeReading(Gauge gauge, boolean available, double value, boolean calibrated, Band band, String note) {

    /**
     * Position against a gauge's hard lines. Each line is an identity the code
     * already enforces (card capacity, the untrainable step count), never a
     * threshold tuned to a dataset.
     */
    public enum Band {
        /** Nothing to flag. */
        NORMAL,
        /** Past a soft edge: TIGHT on GPU memory, past the auto-cache ceiling. */
        CAUTION,
        /** Past a hard line: exceeds the card, untrainable, a single validation patch. */
        LIMIT
    }

    static GaugeReading unavailable(Gauge gauge, String note) {
        return new GaugeReading(gauge, false, Double.NaN, true, Band.NORMAL, note);
    }
}
