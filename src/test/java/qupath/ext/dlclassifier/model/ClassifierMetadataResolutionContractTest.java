package qupath.ext.dlclassifier.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Guards the resolution contract surviving a metadata write.
 *
 * <p>{@code training_pixel_size_um} and {@code training_tile_size_px} are
 * written by the Python training service, but {@code ModelManager.saveClassifier}
 * then overwrites metadata.json with the Java side's {@code toMap()}. Because
 * {@code toMap()} emitted neither field and no builder set them, every saved
 * model lost both -- confirmed against a real model on disk, whose top-level
 * keys were only [architecture, channel_config, classes, createdAt,
 * description, id, name, training, version].
 *
 * <p>The cost is silent. {@code DLPixelClassifier} skips its pixel-size
 * mismatch warning when the field is absent, and
 * {@code inference_preprocess.resample_to_training_resolution} skips
 * resampling, so applying a model to a batch acquired at a different
 * resolution degrades predictions with nothing in the log -- which is the
 * cross-batch case the tool exists for.
 */
class ClassifierMetadataResolutionContractTest {

    private static ClassifierMetadata.Builder minimalBuilder() {
        return ClassifierMetadata.builder()
                .id("test-id")
                .name("test")
                .classes(List.of(new ClassifierMetadata.ClassInfo(0, "Tumor", "#ff0000")));
    }

    @Test
    @DisplayName("a set resolution contract survives toMap()")
    void resolutionContractIsEmitted() {
        ClassifierMetadata metadata = minimalBuilder()
                .trainingPixelSizeMicrons(0.2213)
                .trainingTileSizePx(256)
                .build();

        Map<String, Object> map = metadata.toMap();

        assertTrue(map.containsKey("training_pixel_size_um"), "pixel size must be persisted");
        assertTrue(map.containsKey("training_tile_size_px"), "tile size must be persisted");
        assertEquals(0.2213, (Double) map.get("training_pixel_size_um"), 1e-9);
        assertEquals(256, ((Number) map.get("training_tile_size_px")).intValue());
    }

    @Test
    @DisplayName("the effective pixel size is emitted as native x downsample")
    void effectivePixelSizeIsEmitted() {
        // training_pixel_size_um is the SOURCE image's pixel size; the model
        // trained on tiles read at native x downsample. An external consumer
        // resampling to the native value is too fine by exactly the downsample
        // factor, silently. The explicit field spares every reader that
        // arithmetic and the guess about whether a file predates it.
        ClassifierMetadata metadata =
                minimalBuilder().trainingPixelSizeMicrons(0.499).downsample(8.0).build();

        Map<String, Object> map = metadata.toMap();

        assertEquals(0.499, (Double) map.get("training_pixel_size_um"), 1e-9, "native size unchanged");
        assertEquals(3.992, (Double) map.get("training_effective_pixel_size_um"), 1e-9, "what the model actually saw");
        assertEquals(3.992, metadata.getTrainingEffectivePixelSizeMicrons(), 1e-9);
    }

    @Test
    @DisplayName("at downsample 1 the two pixel sizes agree")
    void effectiveEqualsNativeAtFullResolution() {
        Map<String, Object> map = minimalBuilder()
                .trainingPixelSizeMicrons(0.25)
                .downsample(1.0)
                .build()
                .toMap();

        assertEquals(
                (Double) map.get("training_pixel_size_um"),
                (Double) map.get("training_effective_pixel_size_um"),
                1e-12);
    }

    @Test
    @DisplayName("no native size means no effective size either")
    void effectiveIsOmittedWhenNativeIsUnknown() {
        // A training set whose images had differing pixel sizes records no
        // native size, and there genuinely is no single training resolution:
        // each image was read at its OWN native times the downsample, so the
        // model saw a mixture. Inventing one here would be a fabricated
        // measurement.
        Map<String, Object> map = minimalBuilder().downsample(4.0).build().toMap();

        assertFalse(map.containsKey("training_effective_pixel_size_um"));
        assertTrue(Double.isNaN(minimalBuilder().downsample(4.0).build().getTrainingEffectivePixelSizeMicrons()));
    }

    @Test
    @DisplayName("an unset contract is omitted rather than written as NaN/0")
    void unsetContractIsOmitted() {
        // Absent must stay absent. Writing NaN or 0 would look like a real
        // measurement to the reader and defeat the "older model" branch.
        Map<String, Object> map = minimalBuilder().build().toMap();

        assertFalse(map.containsKey("training_pixel_size_um"));
        assertFalse(map.containsKey("training_tile_size_px"));
    }

    @Test
    @DisplayName("an uncalibrated image does not write a bogus pixel size")
    void uncalibratedPixelSizeIsOmitted() {
        // Uncalibrated images give NaN, and some paths give 0. Neither is a
        // pixel size, and both must be dropped rather than recorded.
        assertFalse(minimalBuilder()
                .trainingPixelSizeMicrons(Double.NaN)
                .build()
                .toMap()
                .containsKey("training_pixel_size_um"));

        assertFalse(
                minimalBuilder().trainingPixelSizeMicrons(0.0).build().toMap().containsKey("training_pixel_size_um"));

        assertFalse(minimalBuilder().trainingTileSizePx(0).build().toMap().containsKey("training_tile_size_px"));
    }

    @Test
    @DisplayName("the contract round-trips through toMap() and back")
    void contractRoundTrips() {
        // The full loop that was broken: build -> toMap -> (write/read) ->
        // rebuild. The values must come back identical.
        ClassifierMetadata original = minimalBuilder()
                .trainingPixelSizeMicrons(0.65)
                .trainingTileSizePx(512)
                .build();

        Map<String, Object> map = original.toMap();

        ClassifierMetadata reloaded = minimalBuilder()
                .trainingPixelSizeMicrons((Double) map.get("training_pixel_size_um"))
                .trainingTileSizePx(((Number) map.get("training_tile_size_px")).intValue())
                .build();

        assertEquals(0.65, reloaded.getTrainingPixelSizeMicrons(), 1e-9);
        assertEquals(512, reloaded.getTrainingTileSizePx());
    }
}
