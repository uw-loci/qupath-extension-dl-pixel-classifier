package qupath.ext.dlclassifier.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ChannelConfiguration.NormalizationStrategy;
import qupath.ext.dlclassifier.model.ClassifierMetadata;

/**
 * A FIXED_RANGE model is applied with the range it was trained on. Training saves the
 * bounds as normalization.min / max (buildInputConfig), but nothing read them back, so
 * the Apply dialog normalized with whatever its panel held. See
 * docs/NORMALIZATION_ROUNDTRIP.md.
 */
class ModelManagerFixedRangeTest {

    private static final String HEAD =
            "{\"id\":\"m\",\"name\":\"m\",\"architecture\":{},\"classes\":[{\"index\":0,\"name\":\"Tumor\"}]";

    private static ClassifierMetadata load(Path dir, String body) throws IOException {
        Files.writeString(dir.resolve("metadata.json"), HEAD + body + "}");
        return new ModelManager().loadMetadata(dir);
    }

    private static String block(String parent, String extra) {
        return ",\"" + parent + "\":{\"normalization_strategy\":\"FIXED_RANGE\",\"normalization\":{\"strategy\":"
                + "\"fixed_range\",\"per_channel\":true,\"clip_percentile\":99.0" + extra + "}}";
    }

    private static ChannelConfiguration panel() {
        return ChannelConfiguration.builder()
                .selectedChannels(List.of(0))
                .normalizationStrategy(NormalizationStrategy.PERCENTILE_99)
                .fixedRange(0, 255)
                .build();
    }

    @Test
    void theTrainedRangeIsReadAndApplied(@TempDir Path dir) throws IOException {
        ClassifierMetadata m = load(dir, block("input_config", ",\"min\":100.0,\"max\":4000.0"));
        assertTrue(m.hasFixedRange());
        assertEquals(100.0, m.getFixedMin(), 1e-9);
        assertEquals(4000.0, m.getFixedMax(), 1e-9);
        ChannelConfiguration applied = panel().withNormalizationOf(m);
        assertEquals(100.0, applied.getFixedMin(), 1e-9, "the model's range, not the panel's 0-255");
        assertEquals(4000.0, applied.getFixedMax(), 1e-9);
    }

    @Test
    void inputConfigWinsOverChannelConfig(@TempDir Path dir) throws IOException {
        ClassifierMetadata m = load(
                dir,
                block("channel_config", ",\"min\":1.0,\"max\":2.0")
                        + block("input_config", ",\"min\":100.0,\"max\":4000.0"));
        assertEquals(100.0, m.getFixedMin(), 1e-9);
        assertEquals(4000.0, m.getFixedMax(), 1e-9);
    }

    @Test
    void aModelWithoutTheRangeKeepsThePanels(@TempDir Path dir) throws IOException {
        ClassifierMetadata m = load(dir, block("input_config", ""));
        assertFalse(m.hasFixedRange());
        ChannelConfiguration applied = panel().withNormalizationOf(m);
        assertEquals(0.0, applied.getFixedMin(), 1e-9);
        assertEquals(255.0, applied.getFixedMax(), 1e-9);
    }

    @Test
    void reSavingKeepsTheRange(@TempDir Path dir) throws IOException {
        ClassifierMetadata m = load(dir, block("input_config", ",\"min\":100.0,\"max\":4000.0"));
        @SuppressWarnings("unchecked")
        Map<String, Object> norm =
                (Map<String, Object>) ((Map<String, Object>) m.toMap().get("channel_config")).get("normalization");
        assertEquals(100.0, ((Number) norm.get("min")).doubleValue(), 1e-9);
        assertEquals(4000.0, ((Number) norm.get("max")).doubleValue(), 1e-9);
    }
}
