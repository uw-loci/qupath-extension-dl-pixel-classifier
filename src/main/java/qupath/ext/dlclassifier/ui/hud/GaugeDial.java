package qupath.ext.dlclassifier.ui.hud;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.shape.ArcType;

/**
 * One of the two large dials: a 240-degree arc opening at the bottom, with a
 * tapered needle, a ghost needle at the previous reading and a filled sweep
 * between them.
 * <p>
 * These carry the pair the whole panel exists for -- time per epoch beside
 * learning steps, speed beside fuel -- so they get the size and the only
 * logarithmic scale on the panel, where a doubling moves the needle the same
 * distance wherever it sits and the reference reading is marked at the top.
 */
final class GaugeDial extends HudGauge {

    /** Leaves room above the face for the reference marking. */
    private static final double REF_BAND = 10;

    /**
     * The title sits below the face rather than inside it. The skill puts it
     * under the hub, but a 240 degree arc sweeps the needle through the whole
     * lower half of the face, so a title there is crossed by the needle at
     * either stop -- and "TIME PER EPOCH" is wider than the face at that
     * height anyway.
     */
    private static final double TITLE_BAND = 13;

    /** The chip, and the hard-line badge, below the title. */
    private static final double CHIP_BAND = 19;

    private static final double START_ANGLE = 210;
    private static final double SWEEP_ANGLE = -240;

    GaugeDial(Gauge gauge, HudPalette palette) {
        super(gauge, palette);
    }

    @Override
    protected void redraw(GraphicsContext gc, double w, double h) {
        double radius = Math.min((h - REF_BAND - TITLE_BAND - CHIP_BAND) / 2, w / 2 - 34);
        if (radius < 16) {
            return;
        }
        double cx = w / 2;
        double cy = REF_BAND + radius;
        boolean lit = frame.reading().available();

        face(gc, cx, cy, radius);
        ticks(gc, cx, cy, radius, lit);
        endLabels(gc, cx, cy, radius, lit);
        if (lit && gauge.scale() == Gauge.Scale.RELATIVE) {
            referenceMark(gc, cx, cy, radius);
        }
        if (lit && ghostVisible && !Double.isNaN(ghost)) {
            sweep(gc, cx, cy, radius);
            needle(gc, cx, cy, radius, ghost, palette.get(HudPalette.Token.TEXT_2, 0.4), false, false);
        }
        needle(gc, cx, cy, radius, needle.get(), accent(), hollow(), true);
        hub(gc, cx, cy);
        title(gc, cx, cy + radius + 6);
        chipRow(gc, cx, h);
        if (pulse.get() >= 0) {
            crossingRing(gc, cx, cy, radius);
        }
    }

    private void face(GraphicsContext gc, double cx, double cy, double r) {
        gc.setFill(palette.get(HudPalette.Token.RAISED));
        gc.fillOval(cx - r, cy - r, 2 * r, 2 * r);
        gc.setStroke(palette.get(HudPalette.Token.TICK, 0.45));
        gc.setLineWidth(1);
        gc.strokeOval(cx - r + 0.5, cy - r + 0.5, 2 * r - 1, 2 * r - 1);
        // A darker well around the hub: depth from the tokens already here,
        // rather than a new colour.
        gc.setFill(palette.get(HudPalette.Token.SURFACE, 0.45));
        double well = r * 0.46;
        gc.fillOval(cx - well, cy - well, 2 * well, 2 * well);
    }

    private void ticks(GraphicsContext gc, double cx, double cy, double r, boolean lit) {
        double outer = r - 5;
        gc.setStroke(palette.get(HudPalette.Token.TICK, lit ? 1.0 : 0.6));
        gc.setLineWidth(2);
        gc.strokeArc(
                cx - outer, cy - outer, 2 * outer, 2 * outer, START_ANGLE + SWEEP_ANGLE, -SWEEP_ANGLE, ArcType.OPEN);
        for (int i = 0; i <= 16; i++) {
            double p = i / 16.0;
            boolean major = i % 4 == 0;
            double len = major ? 8 : 4;
            gc.setLineWidth(major ? 1.6 : 1);
            line(gc, cx, cy, p, outer - 1, outer - 1 - len);
        }
    }

    /**
     * Which way is which, in the user's words, at the two stops. A reader who
     * has never seen the panel should not have to work out that the left end
     * of the time dial is the good end.
     */
    private void endLabels(GraphicsContext gc, double cx, double cy, double r, boolean lit) {
        String[] ends = endWords(gauge);
        double rr = r + 7;
        double dx = Math.cos(Math.toRadians(30)) * rr;
        double dy = Math.sin(Math.toRadians(30)) * rr;
        gc.setFill(palette.get(HudPalette.Token.TEXT_2, lit ? 0.8 : 0.5));
        double leftW = trackedWidth(ends[0], SCALE_FONT, 0.6);
        drawTracked(gc, ends[0], cx - dx - leftW, cy + dy, SCALE_FONT, 0.6);
        drawTracked(gc, ends[1], cx + dx, cy + dy, SCALE_FONT, 0.6);
    }

    /**
     * The comparison point for the log scale, at the middle of the arc. Without
     * it the needle's position means nothing at all; it is not a recommendation.
     */
    private void referenceMark(GraphicsContext gc, double cx, double cy, double r) {
        gc.setStroke(palette.get(HudPalette.Token.TEXT_2, 0.8));
        gc.setLineWidth(1.6);
        line(gc, cx, cy, 0.5, r - 2, r - 5);
        gc.setFill(palette.get(HudPalette.Token.TEXT_2, 0.8));
        String s = "REF";
        double tw = trackedWidth(s, SCALE_FONT, 0.6);
        drawTracked(gc, s, cx - tw / 2, REF_BAND / 2, SCALE_FONT, 0.6);
    }

    /** The size of the change as an area, not just as an angle. */
    private void sweep(GraphicsContext gc, double cx, double cy, double r) {
        double from = Math.min(ghost, needle.get());
        double to = Math.max(ghost, needle.get());
        if (to - from < 0.002) {
            return;
        }
        double outer = r - 8;
        double inner = Math.max(outer - 11, r * 0.5);
        double a0 = START_ANGLE + SWEEP_ANGLE * to;
        double extent = SWEEP_ANGLE * (from - to);
        gc.setFill(colourOrNeutral(0.45));
        gc.beginPath();
        gc.arc(cx, cy, outer, outer, a0, extent);
        gc.arc(cx, cy, inner, inner, a0 + extent, -extent);
        gc.closePath();
        gc.fill();
    }

    private Color colourOrNeutral(double opacity) {
        Color c = accent();
        return Color.color(c.getRed(), c.getGreen(), c.getBlue(), c.getOpacity() * opacity);
    }

    private void needle(
            GraphicsContext gc,
            double cx,
            double cy,
            double r,
            double pos,
            Color colour,
            boolean outline,
            boolean halo) {
        double a = Math.toRadians(START_ANGLE + SWEEP_ANGLE * clamp(pos));
        double ux = Math.cos(a);
        double uy = -Math.sin(a);
        double px = -uy;
        double py = ux;
        double tip = r - 5;
        double hub = 5.5;
        double half = 2.2;
        gc.beginPath();
        gc.moveTo(cx + ux * tip, cy + uy * tip);
        gc.lineTo(cx + ux * hub + px * half, cy + uy * hub + py * half);
        gc.lineTo(cx - ux * hub * 0.7 + px * half * 0.7, cy - uy * hub * 0.7 + py * half * 0.7);
        gc.lineTo(cx - ux * hub * 0.7 - px * half * 0.7, cy - uy * hub * 0.7 - py * half * 0.7);
        gc.lineTo(cx + ux * hub - px * half, cy + uy * hub - py * half);
        gc.closePath();
        if (outline) {
            // Hollow: an estimate with no measurement behind it, or a change
            // that cannot be called better or worse.
            gc.setFill(palette.get(HudPalette.Token.SURFACE, 0.85));
            gc.fill();
            gc.setStroke(colour);
            gc.setLineWidth(1.2);
            gc.stroke();
            return;
        }
        if (halo) {
            // A surface-coloured halo keeps the needle crisp over the sweep.
            gc.setStroke(palette.get(HudPalette.Token.SURFACE, 0.9));
            gc.setLineWidth(2.5);
            gc.stroke();
        }
        gc.setFill(colour);
        gc.fill();
    }

    private void hub(GraphicsContext gc, double cx, double cy) {
        gc.setFill(palette.get(HudPalette.Token.TICK));
        gc.fillOval(cx - 4.5, cy - 4.5, 9, 9);
        gc.setFill(palette.get(HudPalette.Token.SURFACE));
        gc.fillOval(cx - 2, cy - 2, 4, 4);
    }

    private void title(GraphicsContext gc, double cx, double y) {
        String s = title(gauge, frame.reading());
        double tw = trackedWidth(s, TITLE_FONT, TRACKING);
        gc.setFill(titleColour());
        drawTracked(gc, s, cx - tw / 2, y, TITLE_FONT, TRACKING);
    }

    private void chipRow(GraphicsContext gc, double cx, double h) {
        Chip c = chip();
        String badge = bandBadge();
        if (c == null && badge == null) {
            return;
        }
        double centreY = h - CHIP_BAND / 2 - 1;
        double badgeW = badge == null ? 0 : trackedWidth(badge, BADGE_FONT, 0.6) + 12;
        double chipW = c == null ? 0 : drawChip(null, c, 0, centreY, CHIP_FONT);
        double gap = badge != null && c != null ? 6 : 0;
        double x = cx - (badgeW + gap + chipW) / 2;
        if (badge != null) {
            drawBadge(gc, badge, x, centreY, badgeW);
            x += badgeW + gap;
        }
        if (c != null) {
            drawChip(gc, c, x, centreY, CHIP_FONT);
        }
    }

    /**
     * The gauge's hard line, named. A dial's limit is a property of the whole
     * configuration rather than a region of the scale -- "fewer than two
     * optimizer steps per epoch" is not a place on the dial -- so it is shown
     * as a lit label, not as a zone.
     */
    private String bandBadge() {
        GaugeReading r = frame.reading();
        if (!r.available() || r.band() == GaugeReading.Band.NORMAL) {
            return null;
        }
        if (gauge == Gauge.LEARNING_STEPS) {
            return "UNTRAINABLE";
        }
        return r.band() == GaugeReading.Band.LIMIT ? "LIMIT" : "CAUTION";
    }

    private void drawBadge(GraphicsContext gc, String text, double x, double centreY, double w) {
        double h = 14;
        gc.setFill(palette.get(HudPalette.Token.TICK, 0.35));
        gc.fillRoundRect(x, centreY - h / 2, w, h, 3, 3);
        gc.setStroke(palette.get(HudPalette.Token.TEXT, 0.8));
        gc.setLineWidth(1);
        gc.strokeRoundRect(x + 0.5, centreY - h / 2 + 0.5, w - 1, h - 1, 3, 3);
        gc.setFill(palette.get(HudPalette.Token.TEXT));
        drawTracked(gc, text, x + 6, centreY, BADGE_FONT, 0.6);
    }

    /** One expanding ring when a hard line is crossed. Never a loop. */
    private void crossingRing(GraphicsContext gc, double cx, double cy, double r) {
        double phase = pulse.get();
        double rr = r * (0.76 + 0.29 * phase);
        gc.setStroke(colourOrNeutral(0.85 * (1 - phase)));
        gc.setLineWidth(2);
        gc.strokeOval(cx - rr, cy - rr, 2 * rr, 2 * rr);
    }

    private void line(GraphicsContext gc, double cx, double cy, double p, double rOuter, double rInner) {
        double a = Math.toRadians(START_ANGLE + SWEEP_ANGLE * p);
        double ux = Math.cos(a);
        double uy = -Math.sin(a);
        gc.strokeLine(cx + ux * rOuter, cy + uy * rOuter, cx + ux * rInner, cy + uy * rInner);
    }

    private static double clamp(double v) {
        return Double.isNaN(v) ? 0 : Math.max(0, Math.min(1, v));
    }
}
