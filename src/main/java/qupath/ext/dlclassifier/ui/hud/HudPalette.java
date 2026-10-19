package qupath.ext.dlclassifier.ui.hud;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import javafx.scene.layout.Background;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The HUD's colour tokens, read out of hud.css instead of repeated in Java.
 * <p>
 * The gauges draw on a {@link javafx.scene.canvas.Canvas}, which CSS cannot
 * style, and JavaFX offers no way to ask a scene for the value of a looked-up
 * colour. So each token gets a zero-sized probe {@link Region} whose style
 * class sets its background from that token; once CSS has been applied the
 * palette reads the backgrounds back. The stylesheet stays the single place the
 * values exist, and there is no Java copy for it to drift from.
 */
final class HudPalette {

    private static final Logger logger = LoggerFactory.getLogger(HudPalette.class);

    /** The tokens, one per entry in the stylesheet's {@code .hud-root} block. */
    enum Token {
        SURFACE,
        RAISED,
        TICK,
        TEXT,
        TEXT_2,
        MUTED,
        NEUTRAL,
        NEUTRAL_HOT,
        BETTER,
        WORSE
    }

    private final Map<Token, Region> probes = new EnumMap<>(Token.class);
    private final Map<Token, Color> values = new EnumMap<>(Token.class);
    private boolean complete;
    private boolean warned;

    HudPalette() {
        for (Token t : Token.values()) {
            Region r = new Region();
            r.getStyleClass().add("hud-probe-" + t.name().toLowerCase().replace('_', '-'));
            r.setManaged(false);
            r.setMouseTransparent(true);
            r.resize(0, 0);
            probes.put(t, r);
        }
    }

    /** @return the probe nodes, to be added as children of the styled root */
    List<Region> probes() {
        return new ArrayList<>(probes.values());
    }

    /**
     * Reads the tokens off the probes. Call after CSS has been applied -- from
     * {@code layoutChildren}, which always runs later in the same pulse.
     *
     * @return true once every token has a value
     */
    boolean resolve() {
        if (complete) {
            return true;
        }
        boolean all = true;
        for (Token t : Token.values()) {
            Color c = colourOf(probes.get(t));
            if (c == null) {
                all = false;
            } else {
                values.put(t, c);
            }
        }
        complete = all;
        return all;
    }

    /** @return the token's colour, or grey when the stylesheet never loaded */
    Color get(Token t) {
        Color c = values.get(t);
        if (c != null) {
            return c;
        }
        if (!warned) {
            warned = true;
            // Grey rather than a hard-coded copy of the palette: a HUD that
            // draws in one flat colour says "the stylesheet is missing" at a
            // glance, and keeps the token values in one file.
            logger.warn("hud.css did not resolve its colour tokens; the training HUD will draw in grey");
        }
        return Color.GRAY;
    }

    /**
     * @param t       the token
     * @param opacity multiplier on its alpha
     * @return the token's colour, faded
     */
    Color get(Token t, double opacity) {
        Color c = get(t);
        return Color.color(c.getRed(), c.getGreen(), c.getBlue(), c.getOpacity() * opacity);
    }

    private static Color colourOf(Region r) {
        Background bg = r.getBackground();
        if (bg == null || bg.getFills().isEmpty()) {
            return null;
        }
        Paint p = bg.getFills().get(0).getFill();
        return p instanceof Color c ? c : null;
    }
}
