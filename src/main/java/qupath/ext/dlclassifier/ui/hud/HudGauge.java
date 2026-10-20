package qupath.ext.dlclassifier.ui.hud;

import javafx.animation.Interpolator;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.Timeline;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.geometry.VPos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.util.Duration;

/**
 * Shared machinery for one instrument: its canvas, its needle animation, the
 * ratio chip, and the typography.
 * <p>
 * Everything the subclasses draw comes from {@link HudPalette}, so the
 * stylesheet remains the only place a colour is written down. Three rules from
 * the instrument-panel-ui skill are enforced here rather than in each gauge:
 * colour appears only on a gauge the last change moved; a coloured change
 * always carries a verdict mark (a check for better, an exclamation mark for
 * worse) and a ratio chip, so the panel reads in greyscale; and a gauge the
 * change did not move steps back rather than competing with the ones it did.
 */
abstract class HudGauge extends Region {

    /** Needle travel after an edit. Longer than this and the user is waiting. */
    static final double TRAVEL_MS = 220;

    /** First reading rises from the stop; the one piece of showmanship. */
    static final double RISE_MS = 500;

    /** A hard-line crossing pulses once, for this long. Never loops. */
    static final double PULSE_MS = 400;

    /**
     * Phase the crossing ring is frozen at when animation is off. A motionless
     * panel still has to show that a hard line was crossed, so the ring becomes
     * a static mark that stays until the next change.
     */
    static final double STATIC_PULSE_PHASE = 0.45;

    private static final String FAMILY = Font.getDefault().getFamily();

    static final Font TITLE_FONT = Font.font(FAMILY, FontWeight.SEMI_BOLD, 10);
    static final Font TITLE_SMALL_FONT = Font.font(FAMILY, FontWeight.SEMI_BOLD, 9);
    static final Font CHIP_FONT = Font.font(FAMILY, FontWeight.SEMI_BOLD, 11);
    static final Font CHIP_SMALL_FONT = Font.font(FAMILY, FontWeight.SEMI_BOLD, 10);
    static final Font SCALE_FONT = Font.font(FAMILY, FontWeight.NORMAL, 8);
    static final Font BADGE_FONT = Font.font(FAMILY, FontWeight.SEMI_BOLD, 8);

    /** Title letter-spacing, in pixels at a 10 px title: the skill's 0.08 em. */
    static final double TRACKING = 0.8;

    // The panel may display Unicode; this file may not contain it, so the
    // glyphs are built from their code points. MULTIPLICATION SIGN,
    // HORIZONTAL ELLIPSIS. The verdict marks are drawn as shapes, not glyphs,
    // so they do not depend on which fonts a machine has.
    private static final String TIMES = String.valueOf((char) 0x00D7);
    private static final String ELLIPSIS = String.valueOf((char) 0x2026);

    private static Text metrics;

    /** What one gauge needs in order to draw the current state. */
    record Frame(
            GaugeReading reading, HudDelta.Change change, double position, boolean steppedBack, boolean suppressNote) {

        static Frame blank() {
            return new Frame(null, null, Double.NaN, false, false);
        }

        boolean moved() {
            return change != null && change.moved();
        }

        HudDelta.Verdict verdict() {
            return change == null ? HudDelta.Verdict.SAME : change.verdict();
        }
    }

    /**
     * The mark at the front of a chip. It carries the VERDICT, not the
     * direction: the words already say which way the value went ("faster",
     * "fewer"). An up/down triangle there said the same thing twice and left
     * better-or-worse to colour alone -- in the downsample trap every chip read
     * "down", the good one and the bad ones alike.
     */
    enum Mark {
        NONE,
        BETTER,
        WORSE
    }

    /** A ratio chip: verdict mark plus words, tinted by the verdict. */
    record Chip(String text, HudPalette.Token tint, boolean plain, Mark mark) {

        Chip withText(String t) {
            return new Chip(t, tint, plain, mark);
        }
    }

    protected final HudPalette palette;
    protected final Gauge gauge;
    protected Frame frame = Frame.blank();

    /** Animated needle position, 0-1 on the gauge's scale. */
    protected final DoubleProperty needle = new SimpleDoubleProperty(0);

    /** Old needle position while a change is shown, else NaN. */
    protected double ghost = Double.NaN;

    /** Ghost and sweep appear at the end of the travel, so the eye lands still. */
    protected boolean ghostVisible;

    /** 0-1 while the crossing ring is running, -1 when there is none. */
    protected final DoubleProperty pulse = new SimpleDoubleProperty(-1);

    private final Canvas canvas = new Canvas();
    private final Tooltip tooltip = new Tooltip();
    private Timeline travel;
    private Timeline ring;

    HudGauge(Gauge gauge, HudPalette palette) {
        this.gauge = gauge;
        this.palette = palette;
        getChildren().add(canvas);
        needle.addListener((o, a, b) -> draw());
        pulse.addListener((o, a, b) -> draw());
        tooltip.setShowDelay(Duration.millis(250));
        tooltip.setWrapText(true);
        tooltip.setMaxWidth(300);
        Tooltip.install(this, tooltip);
    }

    /**
     * Renders a new state.
     *
     * @param f       the state
     * @param animate whether needles travel
     * @param rise    rise from the stop rather than travelling from the last
     *                position: the first reading, and the first after data loads
     */
    void show(Frame f, boolean animate, boolean rise) {
        this.frame = f;
        tooltip.setText(tooltipText());
        boolean reveal = f.moved() && !Double.isNaN(f.change().positionBefore());
        ghost = reveal ? f.change().positionBefore() : Double.NaN;
        ghostVisible = false;
        double target = Double.isNaN(f.position()) ? 0 : f.position();
        boolean crossed = f.change() != null && f.change().bandCrossed();

        if (travel != null) {
            travel.stop();
        }
        if (ring != null) {
            ring.stop();
        }
        if (!animate) {
            ghostVisible = reveal;
            pulse.set(crossed ? STATIC_PULSE_PHASE : -1);
            needle.set(target);
            draw();
            return;
        }
        pulse.set(-1);
        double from = rise ? 0 : needle.get();
        needle.set(from);
        travel = new Timeline(
                new KeyFrame(Duration.ZERO, new KeyValue(needle, from)),
                new KeyFrame(
                        Duration.millis(rise ? RISE_MS : TRAVEL_MS),
                        new KeyValue(needle, target, Interpolator.EASE_OUT)));
        travel.setOnFinished(e -> {
            ghostVisible = reveal;
            draw();
            if (crossed) {
                firePulse();
            }
        });
        travel.play();
    }

    private void firePulse() {
        ring = new Timeline(
                new KeyFrame(Duration.ZERO, new KeyValue(pulse, 0.0)),
                new KeyFrame(Duration.millis(PULSE_MS), new KeyValue(pulse, 1.0, Interpolator.EASE_OUT)));
        // Once. A panel that blinks teaches people to stop looking at it.
        ring.setOnFinished(e -> pulse.set(-1));
        ring.play();
    }

    @Override
    protected void layoutChildren() {
        canvas.setWidth(getWidth());
        canvas.setHeight(getHeight());
        draw();
    }

    private void draw() {
        double w = canvas.getWidth();
        double h = canvas.getHeight();
        if (w <= 0 || h <= 0) {
            return;
        }
        GraphicsContext gc = canvas.getGraphicsContext2D();
        // A pixel of margin: an antialiased edge drawn at the canvas boundary
        // by the previous state can otherwise survive an exact-size clear.
        gc.clearRect(-1, -1, w + 2, h + 2);
        if (frame.reading() == null) {
            return;
        }
        gc.setTextAlign(TextAlignment.LEFT);
        redraw(gc, w, h);
    }

    /** Draws the instrument into a cleared canvas of the given size. */
    protected abstract void redraw(GraphicsContext gc, double w, double h);

    // ---------------------------------------------------------------- colours

    /**
     * How far a gauge the last change did not move steps back.
     * <p>
     * The skill dims the whole gauge to 55%. Here the fade is applied to the
     * reading -- needle, fill, sweep -- and not to the scale, because the scale
     * carries the hard lines, and a TIGHT mark that disappears whenever the
     * user edits something else has stopped being a mark.
     */
    protected double dim() {
        return frame.steppedBack() ? 0.55 : 1.0;
    }

    /**
     * The colour the needle, fill and chip take.
     * <p>
     * Colour means change: a gauge sitting in its danger band but untouched by
     * the last edit reads neutral, and its band is shown by the dial's own
     * markings instead.
     */
    protected Color accent() {
        if (!frame.reading().available()) {
            return palette.get(HudPalette.Token.MUTED);
        }
        return switch (frame.verdict()) {
            case BETTER -> palette.get(HudPalette.Token.BETTER, dim());
            case WORSE -> palette.get(HudPalette.Token.WORSE, dim());
            case APPEARED_OR_GONE -> palette.get(HudPalette.Token.NEUTRAL_HOT, dim());
            case UNKNOWN -> palette.get(HudPalette.Token.TEXT_2, dim());
            case SAME -> palette.get(HudPalette.Token.NEUTRAL, dim());
        };
    }

    /** True when the needle is drawn as an outline: an estimate with nothing behind it. */
    protected boolean hollow() {
        return !frame.reading().calibrated() || frame.verdict() == HudDelta.Verdict.UNKNOWN;
    }

    protected Color titleColour() {
        if (!frame.reading().available()) {
            return palette.get(HudPalette.Token.MUTED);
        }
        // A gauge at a hard line lights its own title; that is position, not change.
        return frame.reading().band() == GaugeReading.Band.NORMAL
                ? palette.get(HudPalette.Token.TEXT_2, 0.3 + 0.7 * dim())
                : palette.get(HudPalette.Token.TEXT, 0.3 + 0.7 * dim());
    }

    // ------------------------------------------------------------------ chips

    /** @return the chip to draw under this gauge, or null for none */
    protected Chip chip() {
        GaugeReading r = frame.reading();
        if (!r.available()) {
            return r.note().isEmpty() || frame.suppressNote()
                    ? null
                    : new Chip(r.note(), HudPalette.Token.MUTED, true, Mark.NONE);
        }
        if (!frame.moved()) {
            return null;
        }
        HudDelta.Change c = frame.change();
        return switch (c.verdict()) {
            // No colour is honest here, so the words carry the whole message.
            case UNKNOWN -> new Chip("? not comparable", HudPalette.Token.TEXT_2, false, Mark.NONE);
            case APPEARED_OR_GONE -> new Chip("data arrived", HudPalette.Token.NEUTRAL_HOT, false, Mark.NONE);
            default -> new Chip(ratioText(gauge, c), tintFor(c.verdict()), false, markFor(c.verdict()));
        };
    }

    private static Mark markFor(HudDelta.Verdict v) {
        return switch (v) {
            case BETTER -> Mark.BETTER;
            case WORSE -> Mark.WORSE;
            default -> Mark.NONE;
        };
    }

    private static HudPalette.Token tintFor(HudDelta.Verdict v) {
        return switch (v) {
            case BETTER -> HudPalette.Token.BETTER;
            case WORSE -> HudPalette.Token.WORSE;
            default -> HudPalette.Token.NEUTRAL_HOT;
        };
    }

    /**
     * The change in the user's words: a ratio and a direction word, never an
     * absolute value. The verdict is the chip's mark, not part of this text. Rounded to two significant figures, and "slightly" below
     * a tenth, because a HUD that reads "x1.03 fewer" is reporting noise.
     */
    static String ratioText(Gauge g, HudDelta.Change c) {
        double ratio = c.ratio();
        String[] words = words(g);
        if (Double.isNaN(ratio) || Math.abs(ratio - 1) < 1e-9) {
            // Reachable when a hard line moves rather than the value: switching
            // the in-memory mode changes the ceiling the same cache is judged
            // against, so the band crosses while the reading stands still.
            return "band changed";
        }
        boolean up = ratio > 1;
        String word = up ? words[0] : words[1];
        double factor = up ? ratio : 1 / ratio;
        if (Double.isInfinite(factor)) {
            return up ? "from none" : "to none";
        }
        if (factor < 1.1) {
            return "slightly " + word;
        }
        return TIMES + format(factor) + " " + word;
    }

    /** Words for a rise and for a fall, in the terms the user thinks in. */
    private static String[] words(Gauge g) {
        return switch (g) {
            case TIME_PER_EPOCH, TOTAL_TIME -> new String[] {"slower", "faster"};
            case LEARNING_STEPS, TRAINING_DATA, VALIDATION -> new String[] {"more", "fewer"};
            case GPU_MEMORY, RAM_CACHE -> new String[] {"more", "less"};
        };
    }

    /**
     * The gauge's title as the panel writes it.
     *
     * @param g the gauge
     * @param r its reading
     * @return the title, marked as an upper bound where the reading says so
     */
    static String title(Gauge g, GaugeReading r) {
        String t = g.title().toUpperCase();
        // Total time is the one gauge whose only note is the early-stopping
        // upper bound, and a caveat that lives only in a tooltip is a caveat
        // nobody reads.
        if (g == Gauge.TOTAL_TIME && r.available() && !r.note().isEmpty()) {
            return t + " (MAX)";
        }
        return t;
    }

    /**
     * What the two ends of a dial mean, from the same table as the chips.
     *
     * @param g the gauge
     * @return the low end's word then the high end's
     */
    static String[] endWords(Gauge g) {
        String[] w = words(g);
        return new String[] {w[1].toUpperCase(), w[0].toUpperCase()};
    }

    private static String format(double f) {
        if (f >= 9.95) {
            return String.valueOf(Math.round(f));
        }
        // Integer tenths, so a whole ratio prints without ".0" and no float is compared for equality
        long tenths = Math.round(f * 10);
        return tenths % 10 == 0 ? String.valueOf(tenths / 10) : String.valueOf(tenths / 10.0);
    }

    /**
     * Draws a chip, or measures one.
     *
     * @param gc      the context; null to measure only
     * @param c       the chip
     * @param x       left edge
     * @param centreY vertical centre
     * @param font    the chip font
     * @return the chip's width
     */
    protected double drawChip(GraphicsContext gc, Chip c, double x, double centreY, Font font) {
        double padX = c.plain() ? 0 : 6;
        double textW = textWidth(c.text(), font);
        double markW = markWidth(c, font);
        double w = textW + markW + 2 * padX;
        if (gc == null) {
            return w;
        }
        double h = font.getSize() + 6;
        Color tint = palette.get(c.tint());
        if (!c.plain()) {
            gc.setFill(palette.get(c.tint(), 0.14));
            gc.fillRoundRect(x, centreY - h / 2, w, h, h, h);
            gc.setStroke(palette.get(c.tint(), 0.55));
            gc.setLineWidth(1);
            gc.strokeRoundRect(x + 0.5, centreY - h / 2 + 0.5, w - 1, h - 1, h, h);
        }
        if (c.mark() != Mark.NONE) {
            drawMark(gc, c.mark(), x + padX, centreY, font.getSize() * MARK_SCALE, tint);
        }
        gc.setFill(tint);
        gc.setFont(font);
        gc.setTextBaseline(VPos.CENTER);
        gc.fillText(c.text(), x + padX + markW, centreY + 0.5);
        return w;
    }

    /** Mark height as a share of the chip font size. */
    private static final double MARK_SCALE = 0.78;

    /** @return the room a chip's mark takes, including its gap to the text; 0 for none */
    protected static double markWidth(Chip c, Font font) {
        return c.mark() == Mark.NONE ? 0 : font.getSize() * MARK_SCALE + 4;
    }

    /**
     * A check for better, an exclamation mark for worse, drawn as strokes in
     * a square of side {@code s} whose left edge is {@code x}.
     */
    private static void drawMark(GraphicsContext gc, Mark m, double x, double cy, double s, Color tint) {
        gc.save();
        gc.setStroke(tint);
        gc.setFill(tint);
        gc.setLineWidth(Math.max(1.4, s * 0.17));
        gc.setLineCap(javafx.scene.shape.StrokeLineCap.ROUND);
        gc.setLineJoin(javafx.scene.shape.StrokeLineJoin.ROUND);
        if (m == Mark.BETTER) {
            gc.strokePolyline(
                    new double[] {x + s * 0.05, x + s * 0.38, x + s * 0.95},
                    new double[] {cy + s * 0.02, cy + s * 0.34, cy - s * 0.38},
                    3);
        } else {
            double mx = x + s * 0.5;
            gc.strokeLine(mx, cy - s * 0.45, mx, cy + s * 0.12);
            double r = Math.max(0.9, s * 0.11);
            gc.fillOval(mx - r, cy + s * 0.38 - r, 2 * r, 2 * r);
        }
        gc.restore();
    }

    // ------------------------------------------------------------------- type

    /**
     * Caps with letter-spacing. JavaFX has no {@code -fx-letter-spacing}, so
     * titles are drawn a character at a time.
     */
    protected static void drawTracked(GraphicsContext gc, String s, double x, double y, Font font, double tracking) {
        gc.setFont(font);
        gc.setTextBaseline(VPos.CENTER);
        double cursor = x;
        for (int i = 0; i < s.length(); i++) {
            String ch = s.substring(i, i + 1);
            gc.fillText(ch, cursor, y);
            cursor += textWidth(ch, font) + tracking;
        }
    }

    protected static double trackedWidth(String s, Font font, double tracking) {
        double w = 0;
        for (int i = 0; i < s.length(); i++) {
            w += textWidth(s.substring(i, i + 1), font);
        }
        return w + Math.max(0, s.length() - 1) * tracking;
    }

    /** @return the width this string renders at, in pixels */
    protected static double textWidth(String s, Font font) {
        if (metrics == null) {
            metrics = new Text();
        }
        metrics.setFont(font);
        metrics.setText(s);
        return metrics.getLayoutBounds().getWidth();
    }

    /**
     * Same as {@link #fit}, for text drawn with {@link #drawTracked}.
     *
     * @param s        the string
     * @param font     the font
     * @param tracking letter-spacing in pixels
     * @param max      the width to fit into
     * @return the string, shortened with an ellipsis until it fits
     */
    protected static String fitTracked(String s, Font font, double tracking, double max) {
        if (trackedWidth(s, font, tracking) <= max) {
            return s;
        }
        for (int n = s.length() - 1; n > 0; n--) {
            String candidate = s.substring(0, n) + ELLIPSIS;
            if (trackedWidth(candidate, font, tracking) <= max) {
                return candidate;
            }
        }
        return ELLIPSIS;
    }

    /** @return the string, shortened with an ellipsis until it fits */
    protected static String fit(String s, Font font, double max) {
        if (textWidth(s, font) <= max) {
            return s;
        }
        for (int n = s.length() - 1; n > 0; n--) {
            String candidate = s.substring(0, n) + ELLIPSIS;
            if (textWidth(candidate, font) <= max) {
                return candidate;
            }
        }
        return ELLIPSIS;
    }

    // --------------------------------------------------------------- tooltips

    /**
     * What hovering the gauge says: the honesty the colours cannot carry --
     * why an estimate is uncalibrated, why a comparison is unknowable, what a
     * hard line means.
     */
    private String tooltipText() {
        GaugeReading r = frame.reading();
        if (r == null) {
            return gauge.title();
        }
        StringBuilder sb = new StringBuilder(gauge.title()).append('\n').append(explain(gauge));
        if (!r.available()) {
            if (!r.note().isEmpty()) {
                sb.append('\n').append(r.note());
            }
            return sb.toString();
        }
        if (frame.verdict() == HudDelta.Verdict.UNKNOWN) {
            sb.append('\n')
                    .append("Not comparable: the work estimate has no per-architecture cost term, "
                            + "and none has been measured, so this change cannot honestly be called "
                            + "better or worse.");
        } else if (frame.moved()) {
            String verdict = frame.verdict() == HudDelta.Verdict.BETTER
                    ? " (better for you)"
                    : frame.verdict() == HudDelta.Verdict.WORSE ? " (worse for you)" : "";
            sb.append('\n')
                    .append("Last change: ")
                    .append(ratioText(gauge, frame.change()))
                    .append(verdict);
        }
        if (!r.calibrated()) {
            sb.append('\n').append("Rough: this estimate rests on no measurement for this architecture.");
        }
        if (r.band() != GaugeReading.Band.NORMAL) {
            sb.append('\n').append("At a hard line");
            if (!r.note().isEmpty()) {
                sb.append(": ").append(r.note());
            }
            sb.append('.');
        } else if (!r.note().isEmpty()) {
            sb.append('\n').append(r.note());
        }
        return sb.toString();
    }

    private static String explain(Gauge g) {
        return switch (g) {
            case GPU_MEMORY -> "Predicted peak GPU memory as a share of the card. Lower is better.";
            case RAM_CACHE -> "The in-memory patch cache as a share of free system RAM. Lower is better.";
            case TIME_PER_EPOCH ->
                "Relative work in one epoch: patches times the padded tile area. "
                        + "Lower is faster. Comparable within one architecture only.";
            case LEARNING_STEPS ->
                "Optimizer steps over the whole run: how much learning it buys. " + "Higher is better.";
            case TRAINING_DATA -> "Training patches the export would start from. Higher is better.";
            case VALIDATION -> "Validation patches. With very few, every score jumps. Higher is better.";
            case TOTAL_TIME -> "Time per epoch times epochs. Lower is faster.";
        };
    }
}
