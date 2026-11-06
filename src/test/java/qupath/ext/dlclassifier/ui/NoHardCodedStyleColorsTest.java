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
                    if (HARD_CODED.matcher(line).find() && !line.contains("theme-exempt:")) {
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
