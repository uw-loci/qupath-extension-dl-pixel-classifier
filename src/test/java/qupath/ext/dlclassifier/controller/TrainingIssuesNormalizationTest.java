package qupath.ext.dlclassifier.controller;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.service.ApposeClassifierBackend;

/**
 * Training Area Issues has to normalize the way training did.
 *
 * <p>The dataset statistics are computed inside the Python training service and
 * recorded in metadata.json. The Java {@code ChannelConfiguration} built from
 * live UI state never receives them, so the review pass used to send an
 * input_config with no {@code precomputed} flag and evaluate_tiles.py fell back
 * to normalizing every tile against its own percentiles.
 *
 * <p>That fallback is invisible on a varied tile and severe on a uniform one. On
 * one brightfield run it reported 38% of the annotated background as
 * background-called-tissue; running the exported model.onnx over the same
 * patches with the dataset statistics reproduced 0%, and re-running it with
 * per-tile statistics reproduced 38.2%. The model was fine and the review
 * dialog was wrong.
 */
class TrainingIssuesNormalizationTest {

    private static final List<Map<String, Double>> STATS = List.of(
            Map.of("p1", 82.0, "p99", 246.0, "min", 0.0, "max", 255.0, "mean", 199.5, "std", 45.3),
            Map.of("p1", 39.0, "p99", 245.0, "min", 0.0, "max", 255.0, "mean", 165.6, "std", 68.0),
            Map.of("p1", 79.0, "p99", 245.0, "min", 0.0, "max", 255.0, "mean", 188.5, "std", 50.5));

    private static ChannelConfiguration liveConfig() {
        return new ChannelConfiguration.Builder()
                .selectedChannels(List.of(0, 1, 2))
                .channelNames(List.of("Red", "Green", "Blue"))
                .bitDepth(8)
                .normalizationStrategy(ChannelConfiguration.NormalizationStrategy.PERCENTILE_99)
                .perChannelNormalization(false)
                .clipPercentile(99.0)
                .build();
    }

    private static ClassifierMetadata metadataWithStats(List<Map<String, Double>> stats) {
        return new ClassifierMetadata.Builder()
                .id("model-id")
                .name("model")
                .addClass(0, "Ignore*", "#B4B4B4")
                .addClass(1, "Other", "#FFC800")
                .normalizationStats(stats)
                .build();
    }

    @Test
    void theLiveConfigHasNoStatsOfItsOwn() {
        // The premise of the bug: nothing on the Java side knows them.
        assertThat(liveConfig().hasPrecomputedStats()).isFalse();
    }

    @Test
    void statsFromTheTrainedModelAreAttached() {
        ChannelConfiguration result =
                TrainingWorkflow.withTrainedNormalizationStats(liveConfig(), metadataWithStats(STATS), Path.of("m"));
        assertThat(result.hasPrecomputedStats()).isTrue();
        assertThat(result.getPrecomputedChannelStats()).hasSize(3);
        assertThat(result.getPrecomputedChannelStats().get(1)).containsEntry("p1", 39.0);
    }

    @Test
    void theStatsSurviveIntoTheConfigSentToPython() {
        // The end the bug was at: buildInputConfig only emits the block when
        // the configuration carries stats, so attaching them is what makes
        // evaluate_tiles.py take the precomputed path.
        ChannelConfiguration result =
                TrainingWorkflow.withTrainedNormalizationStats(liveConfig(), metadataWithStats(STATS), Path.of("m"));
        Map<String, Object> inputConfig = ApposeClassifierBackend.buildInputConfig(result);

        @SuppressWarnings("unchecked")
        Map<String, Object> norm = (Map<String, Object>) inputConfig.get("normalization");
        assertThat(norm).containsEntry("precomputed", Boolean.TRUE);
        assertThat((List<?>) norm.get("channel_stats")).hasSize(3);
    }

    @Test
    void theUnfixedPathSendsNoStatsAtAll() {
        // Guards the exact regression: without the attach step the config that
        // reaches Python has no precomputed block, which is the silent
        // fall-through to per-tile normalization.
        Map<String, Object> inputConfig = ApposeClassifierBackend.buildInputConfig(liveConfig());

        @SuppressWarnings("unchecked")
        Map<String, Object> norm = (Map<String, Object>) inputConfig.get("normalization");
        assertThat(norm).doesNotContainKey("precomputed");
        assertThat(norm).doesNotContainKey("channel_stats");
    }

    @Test
    void metadataWithoutStatsLeavesTheConfigAlone() {
        // An older model genuinely has none. Returning the original is correct;
        // inventing stats would be worse than the per-tile fallback.
        ChannelConfiguration original = liveConfig();
        assertThat(TrainingWorkflow.withTrainedNormalizationStats(original, metadataWithStats(null), Path.of("m")))
                .isSameAs(original);
        assertThat(TrainingWorkflow.withTrainedNormalizationStats(original, metadataWithStats(List.of()), Path.of("m")))
                .isSameAs(original);
    }

    @Test
    void missingMetadataIsNotAnError() {
        // The review button must still work when metadata.json cannot be read.
        ChannelConfiguration original = liveConfig();
        assertThat(TrainingWorkflow.withTrainedNormalizationStats(original, null, Path.of("m")))
                .isSameAs(original);
    }
}
