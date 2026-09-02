package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import com.google.gson.Gson;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Pins the number handling of the Gson used for {@code metadata.json}.
 * <p>
 * {@code ModelManager.saveClassifier} merges Python-authored fields back over the
 * Java metadata by deserializing them to {@code Object} and re-serializing. Under
 * Gson's default policy every JSON number becomes a {@code Double}, so
 * {@code "decoder_channels": [128, 64, ...]} is rewritten as
 * {@code [128.0, 64.0, ...]}. The Python inference loader passes that list to
 * {@code smp.Unet}, which reaches {@code nn.BatchNorm2d(128.0)} and fails --
 * issue #26, where the user's workaround was to hand-edit the floats back to
 * integers.
 */
class ModelManagerMetadataGsonTest {

    private final Gson gson = ModelManager.createMetadataGson();

    /** Round-trips through the Object-typed path the merge in saveClassifier uses. */
    private String roundTrip(String json) {
        return gson.toJson(gson.fromJson(json, Object.class));
    }

    @Test
    void integerArchitectureValuesSurviveTheRoundTrip() {
        String result = roundTrip("{\"decoder_channels\":[128,64,32,16,8]}");

        // The rendered text is what Python reads, so assert on it directly.
        assertThat(result)
                .as("decoder channels must stay integers -- PyTorch rejects BatchNorm2d(128.0)")
                .contains("128,")
                .doesNotContain("128.0");

        Object channels = gson.fromJson(result, Map.class).get("decoder_channels");
        assertThat(channels).isEqualTo(List.of(128L, 64L, 32L, 16L, 8L));
    }

    @Test
    void scalarIntegerFieldsSurviveTheRoundTrip() {
        // The same merge corrupted each of these in the issue #26 report.
        String result = roundTrip("{\"num_channels\":3,\"selected_channels\":[0,1,2],\"input_size\":[512,512]}");

        Map<?, ?> parsed = gson.fromJson(result, Map.class);
        assertThat(parsed.get("num_channels")).isEqualTo(3L);
        assertThat(parsed.get("selected_channels")).isEqualTo(List.of(0L, 1L, 2L));
        assertThat(parsed.get("input_size")).isEqualTo(List.of(512L, 512L));
    }

    @Test
    void genuineDecimalsAreNotTurnedIntoIntegers() {
        // The fix must not round real floats; normalization stats depend on them.
        // Gson may render these in exponent form (5.0E-4), which is valid JSON,
        // so compare the parsed values rather than the text.
        String result = roundTrip("{\"learning_rate\":0.0005,\"mean\":128.31163}");

        Map<?, ?> parsed = gson.fromJson(result, Map.class);
        assertThat(parsed.get("learning_rate")).isInstanceOf(Double.class).isEqualTo(0.0005);
        assertThat(parsed.get("mean")).isInstanceOf(Double.class).isEqualTo(128.31163);
    }
}
