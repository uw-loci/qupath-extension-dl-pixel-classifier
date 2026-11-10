package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ChannelConfiguration.NormalizationStrategy;

/**
 * The training {@code config.json} is read by Python, so it must come out the
 * same in every locale.
 *
 * <p>Under a decimal-comma locale the class weights were written as
 * {@code 0,500000}, which {@code json.loads} reads as two numbers: the weight
 * array silently gained an entry per class. Under Turkish the normalization
 * name lost its dotted i.
 */
class AnnotationExtractorConfigLocaleTest {

    @TempDir
    Path dir;

    private String writeConfig(Locale locale) throws Exception {
        Method save = AnnotationExtractor.class.getDeclaredMethod(
                "saveProjectConfig",
                Path.class,
                List.class,
                long[].class,
                long.class,
                ChannelConfiguration.class,
                int.class,
                int.class,
                int.class,
                int.class,
                List.class,
                Map.class,
                double.class,
                int.class,
                int.class);
        save.setAccessible(true);
        ChannelConfiguration channels = ChannelConfiguration.builder()
                .selectedChannels(List.of(0, 1, 2))
                .channelNames(List.of("Red", "Green", "Blue"))
                .normalizationStrategy(NormalizationStrategy.PERCENTILE_99)
                .build();
        Locale previous = Locale.getDefault();
        try {
            Locale.setDefault(locale);
            save.invoke(
                    null,
                    dir,
                    List.of("Ignore*", "Tissue", "Immune"),
                    new long[] {600, 200, 200},
                    1000L,
                    channels,
                    256,
                    10,
                    2,
                    3,
                    List.of("IMAGE_1.tif"),
                    Map.of("Tissue", 1.5),
                    2.0,
                    1,
                    0);
        } finally {
            Locale.setDefault(previous);
        }
        return Files.readString(dir.resolve("config.json"));
    }

    private void assertMachineReadable(String text) {
        JsonObject root = JsonParser.parseString(text).getAsJsonObject();
        JsonArray weights = root.getAsJsonArray("class_weights");
        assertThat(weights.size()).isEqualTo(3);
        // 1000 / (3 * 600), 1000 / (3 * 200) * 1.5, 1000 / (3 * 200)
        assertThat(weights.get(0).getAsDouble()).isEqualTo(0.555556);
        assertThat(weights.get(1).getAsDouble()).isEqualTo(2.5);
        assertThat(weights.get(2).getAsDouble()).isEqualTo(1.666667);
        assertThat(text).contains("    0.555556,\n").contains("    2.500000,\n");
        String strategy = root.getAsJsonObject("channel_config")
                .getAsJsonObject("normalization")
                .get("strategy")
                .getAsString();
        assertThat(strategy).isEqualTo("percentile_99");
    }

    @Test
    void germanWritesDecimalPoints() throws Exception {
        assertMachineReadable(writeConfig(Locale.GERMANY));
    }

    @Test
    void turkishWritesDecimalPointsAndADottedI() throws Exception {
        assertMachineReadable(writeConfig(Locale.forLanguageTag("tr-TR")));
    }

    @Test
    void usIsUnchanged() throws Exception {
        assertMachineReadable(writeConfig(Locale.US));
    }
}
