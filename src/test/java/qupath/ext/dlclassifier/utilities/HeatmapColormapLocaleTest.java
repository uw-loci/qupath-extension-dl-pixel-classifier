package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Locale;
import org.junit.jupiter.api.Test;

/** The gradient is CSS: a decimal comma in a stop makes JavaFX drop the style. */
class HeatmapColormapLocaleTest {

    @Test
    void stopsUseADecimalPointUnderGerman() {
        Locale previous = Locale.getDefault();
        String css;
        try {
            Locale.setDefault(Locale.GERMANY);
            css = HeatmapColormap.cssLinearGradient("to right", 0.0, 1.0, 3);
        } finally {
            Locale.setDefault(previous);
        }
        assertThat(css).startsWith("linear-gradient(to right, ");
        assertThat(css).contains(" 0.0%").contains(" 50.0%").contains(" 100.0%");
        assertThat(css).doesNotContain(",0%").doesNotContain("0,0%");
    }
}
