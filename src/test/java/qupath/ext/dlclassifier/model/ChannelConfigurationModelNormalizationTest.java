package qupath.ext.dlclassifier.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The Apply Classifier dialog builds its channel configuration from its own panel,
 * which defaults to PERCENTILE_99 and takes per-channel from the image type. A model
 * trained any other way was applied with the wrong normalization from that dialog.
 * {@link ChannelConfiguration#withNormalizationOf} replaces the panel's normalization
 * with the model's and keeps everything else.
 */
class ChannelConfigurationModelNormalizationTest {

    private static ClassifierMetadata model(ChannelConfiguration.NormalizationStrategy s, boolean perChannel) {
        return ClassifierMetadata.builder()
                .id("m")
                .name("m")
                .normalizationStrategy(s)
                .perChannelNormalization(perChannel)
                .clipPercentile(99.5)
                .classes(List.of(
                        new ClassifierMetadata.ClassInfo(0, "a", "#ff0000"),
                        new ClassifierMetadata.ClassInfo(1, "b", "#00ff00")))
                .build();
    }

    private static ChannelConfiguration panelDefault() {
        return ChannelConfiguration.builder()
                .selectedChannels(List.of(2, 0, 1))
                .channelNames(List.of("Blue", "Red", "Green"))
                .bitDepth(16)
                .normalizationStrategy(ChannelConfiguration.NormalizationStrategy.PERCENTILE_99)
                .perChannelNormalization(true)
                .precomputedChannelStats(List.of(Map.of("p1", 1.0, "p99", 2.0)))
                .build();
    }

    @Test
    void everyStrategyAndFlagComesFromTheModel() {
        for (ChannelConfiguration.NormalizationStrategy s : ChannelConfiguration.NormalizationStrategy.values()) {
            for (boolean pc : new boolean[] {false, true}) {
                ChannelConfiguration c = panelDefault().withNormalizationOf(model(s, pc));
                assertEquals(s, c.getNormalizationStrategy(), s + "/" + pc);
                assertEquals(pc, c.isPerChannelNormalization(), s + "/" + pc);
                assertEquals(99.5, c.getClipPercentile(), 1e-9);
            }
        }
    }

    @Test
    void channelSelectionBitDepthAndStatsAreKept() {
        ChannelConfiguration c =
                panelDefault().withNormalizationOf(model(ChannelConfiguration.NormalizationStrategy.MIN_MAX, false));
        assertEquals(List.of(2, 0, 1), c.getSelectedChannels());
        assertEquals(List.of("Blue", "Red", "Green"), c.getChannelNames());
        assertEquals(16, c.getBitDepth());
        assertTrue(c.hasPrecomputedStats());
        assertFalse(c.isPerChannelNormalization());
    }
}
