package qupath.ext.dlclassifier.utilities;

/**
 * Shared blue -> yellow -> red colormap used for the per-pixel heatmaps and the
 * confidence slider track in the Training Area Issues dialog.
 * <p>
 * This MUST stay identical to the Python LUT in {@code evaluate_tiles.py}
 * ({@code save_loss_heatmap}) so the Java-side confidence overlay, the slider
 * track gradient, and the legend all render the exact same colors as the
 * Python-generated loss heatmap PNG:
 *
 * <pre>
 *   t = 0.0 -> blue   (0,   0,   255)
 *   t = 0.5 -> yellow (255, 255, 0)
 *   t = 1.0 -> red    (255, 0,   0)
 * </pre>
 */
public final class HeatmapColormap {

    private HeatmapColormap() {}

    /**
     * Returns the packed {@code 0xRRGGBB} color for {@code t} in [0, 1] on the
     * blue -> yellow -> red ramp. Values outside [0, 1] are clamped.
     */
    public static int rgb(double t) {
        double tt = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
        int r;
        int g;
        int b;
        if (tt < 0.5) {
            // blue -> yellow
            double s = tt * 2.0;
            r = (int) (255 * s);
            g = (int) (255 * s);
            b = (int) (255 * (1 - s));
        } else {
            // yellow -> red
            double s = (tt - 0.5) * 2.0;
            r = 255;
            g = (int) (255 * (1 - s));
            b = 0;
        }
        return (r << 16) | (g << 8) | b;
    }

    /** Returns the CSS/AWT hex string ({@code #RRGGBB}) for {@code t} in [0, 1]. */
    public static String hex(double t) {
        return String.format("#%06X", rgb(t));
    }

    /**
     * Builds a JavaFX {@code linear-gradient(...)} string that samples the ramp
     * across the sub-range {@code [t0, t1]} (so a slider spanning, say, 0.5-0.99
     * shows exactly the colors a pixel at those confidences would get).
     *
     * @param direction JavaFX gradient direction, e.g. {@code "to right"}
     * @param t0        ramp value at the start of the gradient
     * @param t1        ramp value at the end of the gradient
     * @param stops     number of color stops to emit (>= 2)
     */
    public static String cssLinearGradient(String direction, double t0, double t1, int stops) {
        int n = Math.max(2, stops);
        StringBuilder sb = new StringBuilder("linear-gradient(").append(direction);
        for (int i = 0; i < n; i++) {
            double f = (double) i / (n - 1);
            double t = t0 + (t1 - t0) * f;
            sb.append(", ").append(hex(t)).append(' ').append(String.format("%.1f%%", f * 100));
        }
        sb.append(')');
        return sb.toString();
    }
}
