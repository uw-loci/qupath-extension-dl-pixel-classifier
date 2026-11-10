package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ChannelConfiguration.NormalizationStrategy;

/**
 * The normalization name sent to Python must not depend on the default locale.
 *
 * <p>A no-argument {@code toLowerCase()} under Turkish turned {@code PERCENTILE_99}
 * into {@code percentile_99} spelled with a dotless i (U+0131), and the Python
 * side, which had no branch for an unknown name, returned every tile
 * unnormalized.
 */
class BuildInputConfigLocaleTest {

    private static final Locale TURKISH = Locale.forLanguageTag("tr-TR");

    @ParameterizedTest
    @EnumSource(NormalizationStrategy.class)
    void strategyNameIsTheSameUnderTurkish(NormalizationStrategy strategy) {
        Locale previous = Locale.getDefault();
        try {
            Locale.setDefault(TURKISH);
            ChannelConfiguration config = ChannelConfiguration.builder()
                    .selectedChannels(List.of(0, 1, 2))
                    .normalizationStrategy(strategy)
                    .build();

            @SuppressWarnings("unchecked")
            Map<String, Object> norm = (Map<String, Object>)
                    ApposeClassifierBackend.buildInputConfig(config).get("normalization");

            String expected =
                    switch (strategy) {
                        case MIN_MAX -> "min_max";
                        case PERCENTILE_99 -> "percentile_99";
                        case Z_SCORE -> "z_score";
                        case FIXED_RANGE -> "fixed_range";
                    };
            assertThat(norm).containsEntry("strategy", expected);
        } finally {
            Locale.setDefault(previous);
        }
    }
}
