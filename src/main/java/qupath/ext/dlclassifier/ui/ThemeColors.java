package qupath.ext.dlclassifier.ui;

import java.util.List;

/**
 * Colors for inline styles that read in QuPath's light AND dark themes.
 *
 * <p>A hex color in a style is chosen against one background. The extension's colors
 * were all chosen against the light theme, so in the dark theme its gray hints sat
 * gray-on-gray and its pale banners and fields glowed white. Each color here is a
 * pair, one for dark backgrounds and one for light, and the CSS engine picks between
 * them from the brightness of the theme's {@code -fx-background} with {@code ladder()},
 * the same way JavaFX's own Modena theme picks its text color. Nothing has to know
 * which theme is active, and a custom theme gets whichever side its background is on.
 *
 * <p>Use them inside a style string, where {@link Shade#toString()} gives the CSS:
 * {@code label.setStyle("-fx-text-fill: " + ThemeColors.MUTED + "; -fx-font-size: 11px;")}.
 * Text on a {@code *_BG} banner uses the {@code *_ON_BG} color of the same family, not
 * the plain one: the plain colors are measured against the theme's background, not
 * against the banner's. {@code ThemeColorsTest} checks every pairing for contrast in
 * both themes, and {@code NoHardCodedStyleColorsTest} fails on a new hex color in a style.
 */
public final class ThemeColors {

    private ThemeColors() {}

    /**
     * One color as a dark-background variant and a light-background variant.
     *
     * @param onDark  used when the theme's background is dark
     * @param onLight used when the theme's background is light
     */
    public record Shade(String onDark, String onLight) {
        /** The CSS expression that picks the variant from the theme's background. */
        public String css() {
            // Below 49% background brightness the dark variant, from 50% the light one,
            // as Modena's own text-color ladders do.
            return "ladder(-fx-background, " + onDark + " 49%, " + onLight + " 50%)";
        }

        @Override
        public String toString() {
            return css();
        }
    }

    /** Ordinary text: the theme's own text color, for text that was hard-coded near-black. */
    public static final String TEXT = "-fx-text-background-color";

    /** Secondary text: hints, captions, descriptions under an option. */
    public static final Shade MUTED = new Shade("#b0b6bb", "#595959");

    /** A caution in running text (not on a banner). */
    public static final Shade WARNING = new Shade("#f0b85a", "#8a5300");

    /** An error or a failed check in running text. */
    public static final Shade ERROR = new Shade("#ff8a80", "#b3261e");

    /** A passed check or a good state in running text. */
    public static final Shade SUCCESS = new Shade("#7fd18f", "#2e6b30");

    /** Neutral emphasis in running text: a status line, a note about what was assigned. */
    public static final Shade INFO = new Shade("#8ab8f0", "#2a4a7a");

    /** Background of a caution banner. */
    public static final Shade WARNING_BG = new Shade("#433817", "#fff3cd");

    /** Border or accent stripe of a caution banner. */
    public static final Shade WARNING_BORDER = new Shade("#8f7425", "#d39e00");

    /** Text on {@link #WARNING_BG}. */
    public static final Shade WARNING_ON_BG = new Shade("#f6dc96", "#6b4e00");

    /** Background of an informational banner. */
    public static final Shade INFO_BG = new Shade("#1f344c", "#e7f1ff");

    /** Border of an informational banner. */
    public static final Shade INFO_BORDER = new Shade("#3f6a96", "#b6d4fe");

    /** Text on {@link #INFO_BG}. */
    public static final Shade INFO_ON_BG = new Shade("#cfe2ff", "#0b3a75");

    /** Background of a confirmation banner. */
    public static final Shade SUCCESS_BG = new Shade("#1e3b26", "#d4edda");

    /** Border of a confirmation banner. */
    public static final Shade SUCCESS_BORDER = new Shade("#3c7a4c", "#c3e6cb");

    /** Text on {@link #SUCCESS_BG}. */
    public static final Shade SUCCESS_ON_BG = new Shade("#c8ecd0", "#1d4d2b");

    /** A neutral raised panel or a read-only field: slightly apart from the background. */
    public static final Shade PANEL_BG = new Shade("#3a3f43", "#ececec");

    /** A warm-tinted row (decoder layers in the freeze panel). */
    public static final Shade WARM_BG = new Shade("#3e352c", "#fff6ec");

    /** A cool-tinted row (encoder layers in the freeze panel). */
    public static final Shade COOL_BG = new Shade("#2b3846", "#eef6ff");

    /** A neutral border or divider. */
    public static final Shade BORDER = new Shade("#5c6267", "#c8c8c8");

    /** Every text color and the backgrounds it is drawn on, for the contrast test. */
    static final List<Shade> TEXT_ON_THEME = List.of(MUTED, WARNING, ERROR, SUCCESS, INFO);

    /** Every banner text color paired with its banner background, for the contrast test. */
    static final List<Shade[]> TEXT_ON_BANNER = List.of(
            new Shade[] {WARNING_ON_BG, WARNING_BG},
            new Shade[] {INFO_ON_BG, INFO_BG},
            new Shade[] {SUCCESS_ON_BG, SUCCESS_BG},
            new Shade[] {MUTED, PANEL_BG},
            new Shade[] {MUTED, WARM_BG},
            new Shade[] {MUTED, COOL_BG},
            new Shade[] {ERROR, PANEL_BG},
            // A banner whose background is set on its container and whose labels keep
            // their running-text colors.
            new Shade[] {WARNING, WARNING_BG},
            new Shade[] {ERROR, WARNING_BG},
            new Shade[] {MUTED, WARNING_BG},
            new Shade[] {INFO, INFO_BG},
            new Shade[] {MUTED, INFO_BG},
            new Shade[] {SUCCESS, SUCCESS_BG},
            new Shade[] {WARNING, PANEL_BG},
            new Shade[] {SUCCESS, PANEL_BG});
}
