package qupath.ext.dlclassifier.ui;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Every {@link ThemeColors} text color is readable in QuPath's light and dark themes,
 * by the WCAG 2 contrast ratio (4.5:1 for normal text), against the theme backgrounds
 * text actually lands on, and every banner's text against its own banner.
 */
class ThemeColorsTest {

    /** Modena light: -fx-background is derive(#ececec, 26.4%). */
    private static final String LIGHT_BACKGROUND = "#f4f4f4";

    /** Modena light: -fx-control-inner-background (fields, list cells). */
    private static final String LIGHT_INNER = "#ffffff";

    /** QuPath dark.css: -fx-background is derive(rgb(45, 48, 50), -10%). */
    private static final String DARK_BACKGROUND = "#282b2d";

    /** QuPath dark.css: -fx-control-inner-background is derive(rgb(45, 48, 50), 10%). */
    private static final String DARK_INNER = "#323537";

    private static final double MIN_CONTRAST = 4.5;

    @Test
    void theLadderPicksTheRightSideForBothThemes() {
        // ladder() switches at 49-50% HSB brightness of -fx-background.
        assertTrue(hsbBrightness(LIGHT_BACKGROUND) >= 0.50, "light theme background reads as light");
        assertTrue(hsbBrightness(DARK_BACKGROUND) < 0.49, "dark theme background reads as dark");
    }

    @Test
    void textColorsAreReadableOnBothThemes() {
        List<String> bad = new ArrayList<>();
        int checked = 0;
        for (ThemeColors.Shade s : ThemeColors.TEXT_ON_THEME) {
            for (String bg : List.of(DARK_BACKGROUND, DARK_INNER)) {
                checked++;
                note(bad, s.onDark(), bg, "dark theme");
            }
            for (String bg : List.of(LIGHT_BACKGROUND, LIGHT_INNER)) {
                checked++;
                note(bad, s.onLight(), bg, "light theme");
            }
        }
        assertEquals(4 * ThemeColors.TEXT_ON_THEME.size(), checked);
        assertTrue(bad.isEmpty(), String.join("\n", bad));
    }

    @Test
    void bannerTextIsReadableOnItsBanner() {
        List<String> bad = new ArrayList<>();
        for (ThemeColors.Shade[] pair : ThemeColors.TEXT_ON_BANNER) {
            note(bad, pair[0].onDark(), pair[1].onDark(), "dark theme banner");
            note(bad, pair[0].onLight(), pair[1].onLight(), "light theme banner");
        }
        assertTrue(bad.isEmpty(), String.join("\n", bad));
    }

    @Test
    void cssUsesTheThemeBackground() {
        assertEquals("ladder(-fx-background, #b0b6bb 49%, #595959 50%)", ThemeColors.MUTED.toString());
    }

    private static void note(List<String> bad, String fg, String bg, String where) {
        double c = contrast(fg, bg);
        if (c < MIN_CONTRAST) {
            bad.add(String.format("%s: %s on %s is %.2f:1, below %.1f:1", where, fg, bg, c, MIN_CONTRAST));
        }
    }

    static double contrast(String a, String b) {
        double la = luminance(a);
        double lb = luminance(b);
        return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
    }

    private static double luminance(String hex) {
        int[] c = rgb(hex);
        double[] lin = new double[3];
        for (int i = 0; i < 3; i++) {
            double v = c[i] / 255.0;
            lin[i] = v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        }
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    }

    private static double hsbBrightness(String hex) {
        int[] c = rgb(hex);
        return Math.max(c[0], Math.max(c[1], c[2])) / 255.0;
    }

    private static int[] rgb(String hex) {
        int v = Integer.parseInt(hex.substring(1), 16);
        return new int[] {(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff};
    }
}
