package qupath.ext.dlclassifier.ui;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * No inline style in the extension hard-codes a text, background or border color.
 *
 * <p>A hex color is chosen against one theme's background. Every such color in the
 * extension had been chosen against the light theme, and in QuPath's dark theme its
 * hints were gray on gray and its banners and fields glowed white. Use
 * {@link ThemeColors} instead. A color that must not follow the theme (an image
 * backdrop, a data color scale) says so with a {@code theme-exempt:} comment on its line.
 * The Training HUD is exempt as a package: it draws on its own fixed dark surface.
 */
class NoHardCodedStyleColorsTest {

    private static final Pattern HARD_CODED = Pattern.compile(
            "-fx-(text-fill|background-color|border-color|control-inner-background|prompt-text-fill)\\s*:\\s*"
                    + "[^;\"]*(#[0-9A-Fa-f]{3}|\\b(red|green|blue|white|black|gray|grey|orange|yellow)\\b|rgba?\\()");

    /**
     * Two more ways to land below 4.5:1 without a hex color: the theme's accent as text
     * (3.1:1 light, 3.7:1 dark) and JavaFX's default link color (the same). Links come from
     * {@link ThemeColors#hyperlink(String)}; accent-colored notes use {@link ThemeColors#INFO}.
     */
    private static final Pattern UI_COLOR_STRING = Pattern.compile("\"#[0-9A-Fa-f]{3,8}\"");

    private static final Pattern LOW_CONTRAST_DEFAULTS =
            Pattern.compile("-fx-text-fill:\\s*-fx-accent|new\\s+(javafx\\.scene\\.control\\.)?Hyperlink\\(");

    @Test
    void stylesUseThemeColors() throws IOException {
        Path root = Path.of("src/main/java/qupath/ext/dlclassifier");
        List<String> found = new ArrayList<>();
        int files = 0;
        try (Stream<Path> s = Files.walk(root)) {
            for (Path p : (Iterable<Path>) s.filter(f -> f.toString().endsWith(".java"))::iterator) {
                if (p.toString().contains("/ui/hud/")
                        || p.getFileName().toString().equals("ThemeColors.java")) {
                    continue;
                }
                files++;
                List<String> lines = Files.readAllLines(p, StandardCharsets.UTF_8);
                for (int i = 0; i < lines.size(); i++) {
                    String line = lines.get(i);
                    // In the UI package a bare color string is almost always a text color picked
                    // in code (the training-data status line was one, and missed the rewrite).
                    boolean uiColorString = p.toString().contains("/ui/")
                            && UI_COLOR_STRING.matcher(line).find();
                    boolean hard = HARD_CODED.matcher(line).find()
                            || LOW_CONTRAST_DEFAULTS.matcher(line).find()
                            || uiColorString;
                    if (hard && !line.contains("theme-exempt:")) {
                        found.add(root.relativize(p) + ":" + (i + 1) + ": " + line.trim());
                    }
                }
            }
        }
        assertTrue(files > 50, "walked the extension's sources, found " + files + " files");
        assertTrue(
                found.isEmpty(),
                "Hard-coded colors in styles; use ThemeColors, or mark a data color 'theme-exempt:'\n"
                        + String.join("\n", found));
    }
}
