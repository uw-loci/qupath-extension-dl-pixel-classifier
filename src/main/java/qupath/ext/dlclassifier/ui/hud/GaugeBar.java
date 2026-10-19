package qupath.ext.dlclassifier.ui.hud;

import java.util.List;
import javafx.geometry.VPos;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;

/**
 * A horizontal gauge, in two sizes.
 * <p>
 * {@code FULL} is where "position means state" lives: GPU memory and the RAM
 * cache are fractions of something real, so their hard lines are ticks across
 * the track with the territory past them hatched, and the fill walking into the
 * hatching is the reading. {@code MINI} is the small secondary form -- title,
 * short bar, chip -- for the gauges that back up the big dials.
 */
final class GaugeBar extends HudGauge {

    /**
     * Titles start two pixels in. One is enough at 100% scaling, but at 150%
     * an odd logical offset lands the glyph's left edge on a half device pixel
     * and the first letter grows a stray mark.
     */
    private static final double TITLE_INSET = 2;

    /** Which size this bar is drawn at. */
    enum Size {
        FULL,
        MINI
    }

    /**
     * A hard line on the track.
     *
     * @param position where it sits, 0-1 on the gauge's scale
     * @param label    its name, small, beside the tick
     * @param dense    hatch the territory past it densely: past here is impossible,
     *                 not merely tight
     */
    record Line(double position, String label, boolean dense) {}

    private final Size size;
    private double titleWidth;
    private double chipWidth;
    private List<Line> lines = List.of();

    GaugeBar(Gauge gauge, HudPalette palette, Size size) {
        super(gauge, palette);
        this.size = size;
    }

    /** Column widths, so every bar in a column lines up. */
    void setColumns(double titleWidth, double chipWidth) {
        this.titleWidth = titleWidth;
        this.chipWidth = chipWidth;
    }

    /** The hard lines to mark, from the gauge's own capacities. */
    void setLines(List<Line> lines) {
        this.lines = List.copyOf(lines);
    }

    @Override
    protected void redraw(GraphicsContext gc, double w, double h) {
        if (size == Size.FULL) {
            drawFull(gc, w, h);
        } else {
            drawMini(gc, w, h);
        }
    }

    // ------------------------------------------------------------------- full

    private void drawFull(GraphicsContext gc, double w, double h) {
        double trackX = titleWidth;
        double trackW = Math.max(40, w - titleWidth - chipWidth);
        double trackH = 10;
        double trackY = h - trackH - 2;
        double centreY = trackY + trackH / 2;

        gc.setFill(titleColour());
        drawTracked(gc, title(gauge, frame.reading()), TITLE_INSET, centreY, TITLE_FONT, TRACKING);

        track(gc, trackX, trackY, trackW, trackH);
        boolean lit = frame.reading().available();
        if (lit) {
            // Sweep first, fill over it: the solid part is the reading now, the
            // translucent part the ground it covered.
            if (ghostVisible && !Double.isNaN(ghost)) {
                sweep(gc, trackX, trackY, trackW, trackH);
            }
            fill(gc, trackX, trackY, trackW, trackH, needle.get());
            hatching(gc, trackX, trackY, trackW, trackH);
        }
        hardLines(gc, trackX, trackY, trackW, trackH, lit);
        if (lit && ghostVisible && !Double.isNaN(ghost)) {
            ghostMark(gc, trackX + trackW * clamp(ghost), trackY, trackH);
        }

        Chip c = chip();
        if (c != null) {
            double available = chipWidth - 6;
            Chip fitted = new Chip(fit(c.text(), chipFont(), available - (c.plain() ? 0 : 12)), c.tint(), c.plain());
            drawChip(gc, fitted, w - chipWidth + 2, centreY, chipFont());
        }
        if (pulse.get() >= 0) {
            crossingOutline(gc, trackX, trackY, trackW, trackH);
        }
    }

    private void track(GraphicsContext gc, double x, double y, double w, double h) {
        gc.setFill(palette.get(HudPalette.Token.RAISED));
        gc.fillRoundRect(x, y, w, h, h, h);
        gc.setStroke(palette.get(HudPalette.Token.TICK, frame.reading().available() ? 0.8 : 0.45));
        gc.setLineWidth(1);
        gc.strokeRoundRect(x + 0.5, y + 0.5, w - 1, h - 1, h, h);
    }

    private void fill(GraphicsContext gc, double x, double y, double w, double h, double position) {
        double fw = Math.max(2, w * clamp(position));
        gc.save();
        clipTrack(gc, x, y, w, h);
        if (hollow()) {
            // Hollow is the bar's version of the dial's outline needle: the
            // estimate behind this reading rests on no measurement, and a solid
            // block would claim more than the arithmetic knows.
            Color c = accent();
            gc.setFill(Color.color(c.getRed(), c.getGreen(), c.getBlue(), 0.4));
            gc.fillRect(x, y, fw, h);
            gc.setStroke(c);
            gc.setLineWidth(1.4);
            gc.strokeRect(x + 0.7, y + 0.7, fw - 1.4, h - 1.4);
        } else {
            gc.setFill(accent());
            gc.fillRect(x, y, fw, h);
        }
        gc.restore();
    }

    /** The span the reading moved across, as an area rather than an edge. */
    private void sweep(GraphicsContext gc, double x, double y, double w, double h) {
        double a = clamp(ghost);
        double b = clamp(needle.get());
        double from = Math.min(a, b);
        double to = Math.max(a, b);
        if (to - from < 0.002) {
            return;
        }
        Color c = accent();
        gc.setFill(Color.color(c.getRed(), c.getGreen(), c.getBlue(), 0.45));
        gc.fillRect(x + w * from, y - 3, w * (to - from), h + 6);
    }

    private void ghostMark(GraphicsContext gc, double x, double y, double h) {
        gc.setStroke(palette.get(HudPalette.Token.TEXT_2, 0.75));
        gc.setLineWidth(2);
        gc.strokeLine(x, y - 3, x, y + h + 3);
    }

    /**
     * The territory past a hard line, hatched at 45 degrees. Sparse past a soft
     * edge (tight, but it runs), dense past a capacity that cannot be exceeded.
     */
    private void hatching(GraphicsContext gc, double x, double y, double w, double h) {
        for (Line l : lines) {
            double start = x + w * clamp(l.position());
            if (start >= x + w - 1) {
                continue;
            }
            gc.save();
            clipTrack(gc, x, y, w, h);
            gc.setStroke(palette.get(HudPalette.Token.TICK, l.dense() ? 1.0 : 0.5));
            gc.setLineWidth(1);
            // Dense past a capacity that cannot be exceeded, sparse past a soft
            // edge: the two must not read as the same prohibition.
            double step = l.dense() ? 4 : 9;
            for (double sx = start - h; sx < x + w; sx += step) {
                gc.strokeLine(sx, y + h, sx + h, y);
            }
            gc.restore();
        }
    }

    private void hardLines(GraphicsContext gc, double x, double y, double w, double h, boolean lit) {
        double labelY = y - 8;
        for (int i = 0; i < lines.size(); i++) {
            Line l = lines.get(i);
            double lx = x + w * clamp(l.position());
            boolean active = lit && crossed(l);
            gc.setStroke(palette.get(active ? HudPalette.Token.TEXT : HudPalette.Token.TICK, lit ? 1.0 : 0.5));
            gc.setLineWidth(active ? 1.8 : 1.2);
            // Dashed where the estimate behind the line is uncalibrated: the
            // line is where the arithmetic says, and the arithmetic is a guess.
            gc.setLineDashes(lit && !frame.reading().calibrated() ? new double[] {2, 2} : null);
            gc.strokeLine(lx, y - 2.5, lx, y + h + 2.5);
            gc.setLineDashes(null);

            double tw = textWidth(l.label(), SCALE_FONT);
            // Labels fall on opposite sides of their ticks so two lines close
            // together cannot write over each other.
            double tx = i == 0 ? lx - 3 - tw : lx + 3;
            if (active) {
                // The band the reading is in lights its own label. This is
                // position, not change, so it never takes a change colour.
                gc.setFill(palette.get(HudPalette.Token.TICK, 0.55));
                gc.fillRoundRect(tx - 3, labelY - 6, tw + 6, 12, 3, 3);
            }
            gc.setFill(palette.get(active ? HudPalette.Token.TEXT : HudPalette.Token.TEXT_2, lit ? 1.0 : 0.6));
            gc.setFont(SCALE_FONT);
            gc.setTextBaseline(VPos.CENTER);
            gc.fillText(l.label(), tx, labelY);
        }
    }

    private boolean crossed(Line l) {
        GaugeReading r = frame.reading();
        if (r.band() == GaugeReading.Band.LIMIT) {
            return true;
        }
        return r.band() == GaugeReading.Band.CAUTION && !l.dense();
    }

    private void crossingOutline(GraphicsContext gc, double x, double y, double w, double h) {
        double phase = pulse.get();
        double grow = 5 * phase;
        Color c = accent();
        gc.setStroke(Color.color(c.getRed(), c.getGreen(), c.getBlue(), 0.85 * (1 - phase)));
        gc.setLineWidth(1.6);
        gc.strokeRoundRect(x - grow, y - grow, w + 2 * grow, h + 2 * grow, h + grow, h + grow);
    }

    // ------------------------------------------------------------------- mini

    private void drawMini(GraphicsContext gc, double w, double h) {
        double barH = 6;
        double barW = w - 2;
        double barY = 15;
        boolean lit = frame.reading().available();

        gc.setFill(titleColour());
        drawTracked(gc, title(gauge, frame.reading()), TITLE_INSET, 6, TITLE_SMALL_FONT, 0.6);

        gc.setFill(palette.get(HudPalette.Token.RAISED));
        gc.fillRoundRect(0, barY, barW, barH, barH, barH);
        gc.setStroke(palette.get(HudPalette.Token.TICK, lit ? 0.55 : 0.3));
        gc.setLineWidth(1);
        gc.strokeRoundRect(0.5, barY + 0.5, barW - 1, barH - 1, barH, barH);

        if (lit) {
            if (gauge.scale() == Gauge.Scale.RELATIVE) {
                // The log scale's midpoint is the reference reading; without a
                // mark for it the bar's fill would mean nothing.
                gc.setStroke(palette.get(HudPalette.Token.TICK));
                gc.setLineWidth(1);
                gc.strokeLine(barW / 2, barY - 2, barW / 2, barY + barH + 2);
            }
            if (ghostVisible && !Double.isNaN(ghost)) {
                sweep(gc, 0, barY, barW, barH);
            }
            fill(gc, 0, barY, barW, barH, needle.get());
            if (ghostVisible && !Double.isNaN(ghost)) {
                ghostMark(gc, barW * clamp(ghost), barY, barH);
            }
            if (frame.reading().band() != GaugeReading.Band.NORMAL) {
                // A mini has no room for a tick and a label, so a bar hatched
                // end to end plus the lit title says "this is at a hard line".
                gc.save();
                clipTrack(gc, 0, barY, barW, barH);
                gc.setStroke(palette.get(HudPalette.Token.TICK, 0.95));
                gc.setLineWidth(1);
                for (double sx = -barH; sx < barW; sx += 4) {
                    gc.strokeLine(sx, barY + barH, sx + barH, barY);
                }
                gc.restore();
            }
        }

        Chip c = chip();
        if (c != null) {
            Chip fitted = new Chip(fit(c.text(), CHIP_SMALL_FONT, w - 14), c.tint(), c.plain());
            drawChip(gc, fitted, TITLE_INSET, h - 8, CHIP_SMALL_FONT);
        }
        if (pulse.get() >= 0) {
            crossingOutline(gc, 0, barY, barW, barH);
        }
    }

    // ----------------------------------------------------------------- shared

    private Font chipFont() {
        Chip c = chip();
        return c != null && c.plain() ? SCALE_FONT : CHIP_FONT;
    }

    private void clipTrack(GraphicsContext gc, double x, double y, double w, double h) {
        gc.beginPath();
        gc.moveTo(x + h / 2, y);
        gc.arc(x + h / 2, y + h / 2, h / 2, h / 2, 90, 180);
        gc.lineTo(x + w - h / 2, y + h);
        gc.arc(x + w - h / 2, y + h / 2, h / 2, h / 2, -90, 180);
        gc.closePath();
        gc.clip();
    }

    private static double clamp(double v) {
        return Double.isNaN(v) ? 0 : Math.max(0, Math.min(1, v));
    }
}
