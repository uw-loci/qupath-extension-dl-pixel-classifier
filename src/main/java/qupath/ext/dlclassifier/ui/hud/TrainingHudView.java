package qupath.ext.dlclassifier.ui.hud;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javafx.geometry.VPos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.Region;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.util.Duration;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator;
import qupath.ext.dlclassifier.utilities.VramEstimator;

/**
 * The training HUD: a small dark instrument cluster that answers "what did that
 * change just do?" without the user toggling the value back and forth.
 * <p>
 * Reading order, top to bottom: the change strip names the edit in the dialog's
 * own words; the two big dials pair the rate with the amount, because an epoch
 * that finishes sooner only because it does less has to read as fast AND empty;
 * the two bars carry the memory fractions, where the hard lines live; the three
 * minis back them up.
 * <p>
 * Three rules hold the whole thing together, from the instrument-panel-ui
 * skill. Colour means change -- a gauge sitting in its danger band but untouched
 * by the last edit is not orange, because then orange would mean two things.
 * Each gauge is judged alone, so one edit can light one dial blue and the other
 * orange, and there is no overall "good change" anywhere. And never colour
 * alone: every coloured gauge also carries a direction glyph and a ratio chip,
 * so the panel reads in greyscale and in a screenshot pasted into a bug report.
 * <p>
 * Every number comes from {@link ChangeTracker.State}. Nothing here recomputes
 * an estimate.
 */
public final class TrainingHudView extends Region {

    /**
     * The smallest needle travel this panel can show, as a fraction of a scale.
     * <p>
     * Two pixels of travel on the shortest scale the panel draws: a needle tip
     * 41 px out on a 240 degree arc, which is 172 px of arc, so 2 px is 1/86 of
     * it. {@link ChangeTracker} treats anything smaller as SAME, so hue is
     * never spent on a change nobody can see. On the dials' log scale that
     * works out at a factor of about 1.07, which leaves the skill's "slightly"
     * band (below x1.1) visible rather than suppressed.
     */
    public static final double MIN_POSITION_STEP = 0.0125;

    private static final double PREF_W = 440;
    private static final double PREF_H = 320;
    private static final double PAD = 10;
    private static final double GAP = 7;
    private static final double BAR_H = 25;
    private static final double BAR_GAP = 4;
    private static final double MINIS_GAP = 7;
    private static final double MINIS_H = 38;
    private static final double FOOT_H = 12;
    private static final double STRIP_LABEL_H = 11;
    private static final double STRIP_LINE_H = 15;

    /**
     * The change strip always reserves two lines for the control list, even
     * when one would do. A strip that grew when a preset touched three controls
     * would resize the dials underneath it mid-read, and a needle that changes
     * scale while the user is looking at it is worse than a line of air.
     */
    private static final double STRIP_LINES = 2;

    private static final double STRIP_H = STRIP_LABEL_H + STRIP_LINE_H * STRIP_LINES;
    private static final double BAR_TITLE_W = 84;
    private static final double BAR_CHIP_W = 108;

    private static final String FAMILY = Font.getDefault().getFamily();
    private static final Font STRIP_LABEL_FONT = Font.font(FAMILY, FontWeight.SEMI_BOLD, 9);
    private static final Font STRIP_FONT = Font.font(FAMILY, FontWeight.NORMAL, 12);
    private static final Font FOOT_FONT = Font.font(FAMILY, FontWeight.NORMAL, 8);
    private static final Font CHIP_FONT = Font.font(FAMILY, FontWeight.SEMI_BOLD, 11);

    /** Short, because every pixel it takes comes off the control list beside it. */
    private static final String PIN_TEXT = "Pin baseline";

    /** MIDDLE DOT, between the controls one burst touched. */
    private static final String SEP = "  " + (char) 0x00B7 + "  ";

    /**
     * Where GPU memory stops being comfortable, as a share of the card.
     * <p>
     * Found by asking {@link VramEstimator#classify} rather than copying its
     * {@code TIGHT_FRACTION}, which is package-private in another package: the
     * tick on the bar then cannot drift away from the band the model reports.
     */
    private static final double GPU_TIGHT_FRACTION = findTightFraction();

    private final HudPalette palette = new HudPalette();
    private final Canvas chrome = new Canvas();
    private final Region stripHover = new Region();
    private final Button pinButton = new Button();
    private final Tooltip stripTip = new Tooltip();

    private final Map<Gauge, HudGauge> gauges = new EnumMap<>(Gauge.class);
    private final GaugeBar gpuBar;
    private final GaugeBar ramBar;

    private ChangeTracker.State state;
    private boolean animate = true;
    private boolean everShown;
    private boolean dataSeen;
    private boolean wasPinned;
    private LocalTime pinnedAt;
    private String sharedNote = "";
    private Runnable onPin;
    private Runnable onUnpin;
    private double stripTextWidth = PREF_W - 2 * PAD - 86;
    private double dialsHeight = 134;

    public TrainingHudView() {
        getStyleClass().add("hud-root");
        getStylesheets()
                .add(TrainingHudView.class
                        .getResource("/qupath/ext/dlclassifier/ui/hud/hud.css")
                        .toExternalForm());
        setPrefSize(PREF_W, PREF_H);
        setMinSize(PREF_W, PREF_H);

        for (Gauge g : new Gauge[] {Gauge.TIME_PER_EPOCH, Gauge.LEARNING_STEPS}) {
            gauges.put(g, new GaugeDial(g, palette));
        }
        gpuBar = new GaugeBar(Gauge.GPU_MEMORY, palette, GaugeBar.Size.FULL);
        ramBar = new GaugeBar(Gauge.RAM_CACHE, palette, GaugeBar.Size.FULL);
        gpuBar.setColumns(BAR_TITLE_W, BAR_CHIP_W);
        ramBar.setColumns(BAR_TITLE_W, BAR_CHIP_W);
        gauges.put(Gauge.GPU_MEMORY, gpuBar);
        gauges.put(Gauge.RAM_CACHE, ramBar);
        for (Gauge g : new Gauge[] {Gauge.TRAINING_DATA, Gauge.VALIDATION, Gauge.TOTAL_TIME}) {
            gauges.put(g, new GaugeBar(g, palette, GaugeBar.Size.MINI));
        }

        pinButton.getStyleClass().add("hud-pin");
        pinButton.setText(PIN_TEXT);
        pinButton.setFocusTraversable(false);
        Tooltip pinTip = new Tooltip();
        pinTip.setShowDelay(Duration.millis(250));
        pinTip.setWrapText(true);
        pinTip.setMaxWidth(300);
        pinTip.setText("Compare every later change against the current reading, instead of "
                + "against the last edit. Use it to compare two configurations without "
                + "passing through the ones in between.");
        pinButton.setTooltip(pinTip);
        pinButton.setOnAction(e -> {
            boolean pinned = state != null && state.pinned();
            Runnable r = pinned ? onUnpin : onPin;
            if (r != null) {
                r.run();
            }
        });

        // Nothing is painted here; the region exists only to host the strip's
        // tooltip, and a Region with no background is not picked without this.
        stripHover.setPickOnBounds(true);
        stripTip.setShowDelay(Duration.millis(250));
        stripTip.setWrapText(true);
        stripTip.setMaxWidth(320);
        Tooltip.install(stripHover, stripTip);

        getChildren().add(chrome);
        getChildren().addAll(palette.probes());
        getChildren().add(stripHover);
        getChildren().add(pinButton);
        getChildren().addAll(new ArrayList<>(gauges.values()));
        updatePinButton();
    }

    /** @param onPin runs when the user asks to compare against the current reading */
    public void setOnPin(Runnable onPin) {
        this.onPin = onPin;
        updatePinButton();
    }

    /** @param onUnpin runs when the user asks to go back to comparing against the last edit */
    public void setOnUnpin(Runnable onUnpin) {
        this.onUnpin = onUnpin;
        updatePinButton();
    }

    /**
     * Turns needle travel on or off. Off, every reading lands immediately and a
     * hard-line crossing is marked by a ring that stays rather than pulsing.
     *
     * @param animate whether to animate
     */
    public void setAnimate(boolean animate) {
        this.animate = animate;
    }

    /**
     * Shows a state: the current reading, what the last change did to each
     * gauge, and the controls that changed.
     *
     * @param newState the state to draw
     */
    public void show(ChangeTracker.State newState) {
        this.state = newState;
        boolean hasData = newState.now().get(Gauge.TIME_PER_EPOCH).available();
        // Rising from the stop is for the first reading and for the moment the
        // patch counts arrive -- both of which the user did not cause.
        boolean rise = !everShown || (hasData && !dataSeen);
        everShown = true;
        dataSeen = dataSeen || hasData;
        if (newState.pinned() && !wasPinned) {
            pinnedAt = LocalTime.now();
        }
        wasPinned = newState.pinned();

        sharedNote = sharedNote(newState);
        // Before the gauges draw, not after: a bar that is handed its hard
        // lines afterwards paints its first frame without them, and the first
        // frame is the one the user opens the panel on.
        gpuBar.setLines(gpuLines());
        ramBar.setLines(ramLines(newState.now().inputs()));
        boolean highlighting = newState.mode() == ChangeTracker.Mode.CHANGE;
        for (Gauge g : Gauge.values()) {
            GaugeReading r = newState.now().get(g);
            HudDelta.Change c = newState.changes().get(g);
            double position = r.available() ? newState.scale().position(g, r.value()) : Double.NaN;
            boolean moved = c != null && c.moved();
            // Gauges the change did not move step back, so the eye goes to the
            // ones it did. They come back at the next change.
            boolean steppedBack = highlighting && r.available() && !moved;
            boolean suppress = !r.note().isEmpty() && r.note().equals(sharedNote);
            gauges.get(g).show(new HudGauge.Frame(r, c, position, steppedBack, suppress), animate, rise);
        }
        updatePinButton();
        stripTip.setText(stripTooltip(newState));
        requestLayout();
        drawChrome();
    }

    private void updatePinButton() {
        boolean pinned = state != null && state.pinned();
        pinButton.setText(pinned ? "Unpin" : PIN_TEXT);
        pinButton.setDisable(pinned ? onUnpin == null : onPin == null);
    }

    /**
     * The one chip the panel shows while the patch counts do not exist yet.
     * Five gauges go dark together at that point, and a row of five identical
     * chips would read as five faults rather than one missing input.
     */
    private static String sharedNote(ChangeTracker.State s) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Gauge g : Gauge.values()) {
            GaugeReading r = s.now().get(g);
            if (!r.available() && !r.note().isEmpty()) {
                counts.merge(r.note(), 1, Integer::sum);
            }
        }
        String best = "";
        int bestCount = 1;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() > bestCount) {
                best = e.getKey();
                bestCount = e.getValue();
            }
        }
        return best;
    }

    private static List<GaugeBar.Line> gpuLines() {
        return List.of(
                new GaugeBar.Line(GPU_TIGHT_FRACTION / HudScale.FRACTION_FULL_SCALE, "TIGHT", false),
                new GaugeBar.Line(1 / HudScale.FRACTION_FULL_SCALE, "100%", true));
    }

    private static List<GaugeBar.Line> ramLines(HudInputs in) {
        List<GaugeBar.Line> out = new ArrayList<>();
        // The ceiling depends on which cache mode the run will use, so the
        // tick moves with the mode; both fractions are the training script's.
        if ("auto".equals(in.inMemoryMode())) {
            out.add(new GaugeBar.Line(
                    PatchCountEstimator.AUTO_CACHE_FRACTION / HudScale.FRACTION_FULL_SCALE, "auto 50%", false));
        } else if ("bounded".equals(in.inMemoryMode())) {
            out.add(new GaugeBar.Line(
                    PatchCountEstimator.BOUNDED_CACHE_FRACTION / HudScale.FRACTION_FULL_SCALE, "cap 40%", false));
        }
        out.add(new GaugeBar.Line(1 / HudScale.FRACTION_FULL_SCALE, "100%", true));
        return out;
    }

    private static double findTightFraction() {
        double low = 0;
        double high = 1;
        for (int i = 0; i < 40; i++) {
            double mid = (low + high) / 2;
            if (VramEstimator.classify(mid, 1.0) == VramEstimator.Fit.OK) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return high;
    }

    // ----------------------------------------------------------------- layout

    @Override
    protected void layoutChildren() {
        palette.resolve();
        double w = getWidth();
        double h = getHeight();
        chrome.setWidth(w);
        chrome.setHeight(h);

        double pinW = Math.max(pinButton.prefWidth(-1), 78);
        double pinH = pinButton.prefHeight(-1);
        stripTextWidth = w - 2 * PAD - pinW - 10;
        stripHover.resizeRelocate(PAD, PAD, w - 2 * PAD, STRIP_H);
        pinButton.resizeRelocate(w - PAD - pinW, PAD + (STRIP_H - pinH) / 2, pinW, pinH);

        double dialsH = Math.max(90, h - fixedHeight());
        dialsHeight = dialsH;
        double y = PAD + STRIP_H + GAP;

        double cellW = (w - 2 * PAD - 16) / 2;
        gauges.get(Gauge.TIME_PER_EPOCH).resizeRelocate(PAD, y, cellW, dialsH);
        gauges.get(Gauge.LEARNING_STEPS).resizeRelocate(PAD + cellW + 16, y, cellW, dialsH);
        y += dialsH + GAP;

        gpuBar.resizeRelocate(PAD, y, w - 2 * PAD, BAR_H);
        y += BAR_H + BAR_GAP;
        ramBar.resizeRelocate(PAD, y, w - 2 * PAD, BAR_H);
        y += BAR_H + MINIS_GAP;

        Gauge[] minis = {Gauge.TRAINING_DATA, Gauge.VALIDATION, Gauge.TOTAL_TIME};
        double miniW = (w - 2 * PAD - 16) / 3;
        for (int i = 0; i < minis.length; i++) {
            gauges.get(minis[i]).resizeRelocate(PAD + i * (miniW + 8), y, miniW, MINIS_H);
        }
        drawChrome();
    }

    // ----------------------------------------------------------------- chrome

    /** The strip, the separators, the shared no-data chip and the estimate footnote. */
    private void drawChrome() {
        double w = chrome.getWidth();
        double h = chrome.getHeight();
        if (w <= 0 || h <= 0) {
            return;
        }
        GraphicsContext gc = chrome.getGraphicsContext2D();
        gc.clearRect(0, 0, w, h);
        gc.setTextBaseline(VPos.CENTER);

        // A hairline across the top reads as a bezel rather than a border.
        gc.setStroke(palette.get(HudPalette.Token.TICK, 0.35));
        gc.setLineWidth(1);
        gc.strokeLine(0, 0.5, w, 0.5);

        drawStrip(gc, w);

        double sepY = PAD + STRIP_H + GAP / 2;
        gc.setStroke(palette.get(HudPalette.Token.TICK, 0.5));
        gc.strokeLine(PAD, Math.round(sepY) + 0.5, w - PAD, Math.round(sepY) + 0.5);

        double minisTop = h - PAD - FOOT_H - MINIS_H;
        gc.setStroke(palette.get(HudPalette.Token.TICK, 0.3));
        gc.strokeLine(
                PAD, Math.round(minisTop - MINIS_GAP / 2) + 0.5, w - PAD, Math.round(minisTop - MINIS_GAP / 2) + 0.5);

        drawSharedChip(gc, w, h);
        drawFootnote(gc, w, h);
    }

    private void drawStrip(GraphicsContext gc, double w) {
        String label = stripLabel();
        gc.setFill(palette.get(HudPalette.Token.TEXT_2));
        HudGauge.drawTracked(gc, label, PAD, PAD + STRIP_LABEL_H / 2, STRIP_LABEL_FONT, HudGauge.TRACKING);

        boolean rest = state == null || state.mode() == ChangeTracker.Mode.AT_REST;
        gc.setFill(palette.get(rest ? HudPalette.Token.TEXT_2 : HudPalette.Token.TEXT));
        gc.setFont(STRIP_FONT);
        List<String> rows = wrap(stripDetail(), stripTextWidth, (int) STRIP_LINES);
        // Centre one line in the two-line space rather than leaving it adrift
        // at the top.
        double y = PAD + STRIP_LABEL_H + STRIP_LINE_H / 2 + (STRIP_LINES - rows.size()) * STRIP_LINE_H / 2;
        for (String row : rows) {
            gc.fillText(row, PAD, y);
            y += STRIP_LINE_H;
        }
    }

    private String stripLabel() {
        if (state == null || state.mode() == ChangeTracker.Mode.AT_REST) {
            return state != null && state.pinned() ? "COMPARED TO PIN" : "READY";
        }
        if (state.mode() == ChangeTracker.Mode.NO_GAUGE_MOVED) {
            return "NO GAUGE MOVED";
        }
        return state.pinned() ? "COMPARED TO PIN" : "LAST CHANGE";
    }

    private String stripDetail() {
        if (state == null) {
            return "Change any option to see what it does.";
        }
        String controls = String.join(SEP, state.controls());
        if (state.mode() == ChangeTracker.Mode.AT_REST) {
            return state.pinned() ? "Back at the pinned settings." : "Change any option to see what it does.";
        }
        return controls;
    }

    private String stripTooltip(ChangeTracker.State s) {
        StringBuilder sb = new StringBuilder();
        if (s.pinned()) {
            sb.append("Every gauge is compared against the reading pinned");
            if (pinnedAt != null) {
                sb.append(" at ").append(pinnedAt.format(DateTimeFormatter.ofPattern("HH:mm")));
            }
            sb.append(", not against the last edit.\n");
        }
        if (s.controls().isEmpty()) {
            sb.append("Nothing has changed yet.");
        } else {
            sb.append("Changed: ").append(String.join(", ", s.controls()));
        }
        if (s.mode() == ChangeTracker.Mode.NO_GAUGE_MOVED) {
            sb.append("\nNo gauge moved: this edit changes nothing the HUD measures.");
        }
        return sb.toString();
    }

    /** One chip for the whole panel while the patch counts do not exist. */
    private void drawSharedChip(GraphicsContext gc, double w, double h) {
        if (sharedNote.isEmpty()) {
            return;
        }
        double y = PAD + STRIP_H + GAP + dialsHeight - 10;
        double tw = HudGauge.textWidth(sharedNote, CHIP_FONT);
        double cw = tw + 18;
        double x = (w - cw) / 2;
        gc.setFill(palette.get(HudPalette.Token.RAISED));
        gc.fillRoundRect(x, y - 10, cw, 20, 10, 10);
        gc.setStroke(palette.get(HudPalette.Token.MUTED, 0.8));
        gc.setLineWidth(1);
        gc.strokeRoundRect(x + 0.5, y - 9.5, cw - 1, 19, 10, 10);
        gc.setFill(palette.get(HudPalette.Token.MUTED));
        gc.setFont(CHIP_FONT);
        gc.fillText(sharedNote, x + 9, y);
    }

    private double fixedHeight() {
        return 2 * PAD + STRIP_H + GAP + GAP + BAR_H + BAR_GAP + BAR_H + MINIS_GAP + MINIS_H + FOOT_H;
    }

    /**
     * Says once, quietly, that every patch-derived gauge rests on a count made
     * from the annotations rather than from an export that has happened.
     */
    private void drawFootnote(GraphicsContext gc, double w, double h) {
        if (state == null || !state.now().get(Gauge.TRAINING_DATA).available()) {
            return;
        }
        String s = "est. from the annotations; exact counts exist after export";
        gc.setFill(palette.get(HudPalette.Token.MUTED, 0.9));
        gc.setFont(FOOT_FONT);
        double tw = HudGauge.textWidth(s, FOOT_FONT);
        gc.fillText(s, w - PAD - tw, h - PAD - FOOT_H / 2);
    }

    /**
     * Breaks the control list over at most {@code maxLines}, never dropping a
     * control: once the last line is reached everything still unplaced goes on
     * it and is ellipsised, so the strip can run out of room but cannot quietly
     * fail to mention an edit. The whole list is always in the tooltip.
     */
    private static List<String> wrap(String s, double available, int maxLines) {
        List<String> out = new ArrayList<>();
        if (HudGauge.textWidth(s, STRIP_FONT) <= available || maxLines <= 1) {
            out.add(HudGauge.fit(s, STRIP_FONT, available));
            return out;
        }
        String[] parts = s.split(SEP, -1);
        StringBuilder row = new StringBuilder();
        for (int i = 0; i < parts.length; i++) {
            String candidate = row.length() == 0 ? parts[i] : row + SEP + parts[i];
            if (HudGauge.textWidth(candidate, STRIP_FONT) > available && row.length() > 0) {
                if (out.size() == maxLines - 1) {
                    // Last line: everything still unplaced goes on it, however
                    // long, and the ellipsis below says there was more.
                    row = new StringBuilder(candidate);
                    for (int j = i + 1; j < parts.length; j++) {
                        row.append(SEP).append(parts[j]);
                    }
                    break;
                }
                out.add(row.toString());
                row = new StringBuilder(parts[i]);
            } else {
                row = new StringBuilder(candidate);
            }
        }
        out.add(HudGauge.fit(row.toString(), STRIP_FONT, available));
        return out;
    }

    @Override
    protected double computePrefWidth(double height) {
        return PREF_W;
    }

    @Override
    protected double computePrefHeight(double width) {
        return PREF_H;
    }
}
