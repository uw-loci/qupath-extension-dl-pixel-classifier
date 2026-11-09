package qupath.ext.dlclassifier.scripting;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ChannelConfiguration.NormalizationStrategy;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.InferenceConfig.ApplicationScope;

/**
 * Copy as Script carries every normalization setting. It emitted only the strategy,
 * so a replayed script fell back to pooled channels and the 99th percentile, and an
 * image with more than three channels was normalized differently from the dialog
 * (see docs/NORMALIZATION_ROUNDTRIP.md).
 */
class ScriptGeneratorNormalizationTest {

    private static String script(ChannelConfiguration ch) {
        return ScriptGenerator.generateInferenceScript(
                "test-classifier", InferenceConfig.builder().build(), ch, ApplicationScope.ALL_ANNOTATIONS);
    }

    @Test
    void everyStrategyEmitsPerChannelAndClipPercentile() {
        for (NormalizationStrategy strategy : NormalizationStrategy.values()) {
            for (boolean perChannel : new boolean[] {true, false}) {
                for (double clip : new double[] {95.0, 99.5}) {
                    String s = script(ChannelConfiguration.builder()
                            .selectedChannels(List.of(0, 1, 2, 3))
                            .normalizationStrategy(strategy)
                            .perChannelNormalization(perChannel)
                            .clipPercentile(clip)
                            .fixedRange(10, 4000)
                            .build());
                    String at = strategy + " perChannel=" + perChannel + " clip=" + clip;
                    assertThat(s)
                            .as(at)
                            .contains(".normalizationStrategy(ChannelConfiguration.NormalizationStrategy."
                                    + strategy.name() + ")");
                    assertThat(s).as(at).contains(".perChannelNormalization(" + perChannel + ")");
                    assertThat(s).as(at).contains(".clipPercentile(" + (clip == 95.0 ? "95" : "99.5"));
                    assertThat(s.contains(".fixedRange("))
                            .as(at)
                            .isEqualTo(strategy == NormalizationStrategy.FIXED_RANGE);
                }
            }
        }
    }
}
