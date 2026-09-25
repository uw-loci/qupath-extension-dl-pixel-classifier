package qupath.ext.dlclassifier.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Patch counts in saved metadata.
 *
 * <p>How many patches a run trained on decides whether its batch size could
 * work at all, and how many it validated on decides how much a reported best
 * mean IoU is worth. Neither was recorded, so diagnosing a finished model
 * meant asking the user for a log they happened to still have.
 *
 * <p>The counts follow the same absent-rather-than-zero rule as the
 * resolution contract: a model saved before they existed must read as
 * "unknown", not as "trained on no patches".
 */
class ClassifierMetadataPatchCountsTest {

    private static ClassifierMetadata.Builder base() {
        return ClassifierMetadata.builder()
                .id("test")
                .name("test")
                .classes(List.of(new ClassifierMetadata.ClassInfo(0, "Tissue", "#ff0000")));
    }

    @Test
    @DisplayName("counts are written when recorded")
    void countsAreWritten() {
        Map<String, Object> map = base().trainingPatchCounts(80, 64, 16).build().toMap();

        assertEquals(80, map.get("training_patches_total"));
        assertEquals(64, map.get("training_patches_train"));
        assertEquals(16, map.get("training_patches_validation"));
    }

    @Test
    @DisplayName("unrecorded counts are absent, not zero")
    void unrecordedCountsAreAbsent() {
        // A key present with value 0 would read as a fact about the run.
        Map<String, Object> map = base().build().toMap();

        assertFalse(map.containsKey("training_patches_total"));
        assertFalse(map.containsKey("training_patches_train"));
        assertFalse(map.containsKey("training_patches_validation"));
    }

    @Test
    @DisplayName("a train-only split still records what it has")
    void partialCountsAreWrittenIndependently() {
        // validationSplit 0 is legitimate: no validation patches, but the
        // training count is still worth having.
        Map<String, Object> map = base().trainingPatchCounts(64, 64, 0).build().toMap();

        assertTrue(map.containsKey("training_patches_train"));
        assertFalse(
                map.containsKey("training_patches_validation"),
                "an empty validation split is absent, matching the unknown case");
    }

    @Test
    @DisplayName("negative counts are clamped rather than stored")
    void negativeCountsAreClamped() {
        ClassifierMetadata meta = base().trainingPatchCounts(-5, -1, -1).build();

        assertEquals(0, meta.getTrainingPatchesTotal());
        assertFalse(meta.toMap().containsKey("training_patches_total"));
    }

    @Test
    @DisplayName("getters expose what was recorded")
    void gettersRoundTrip() {
        ClassifierMetadata meta = base().trainingPatchCounts(80, 64, 16).build();

        assertEquals(80, meta.getTrainingPatchesTotal());
        assertEquals(64, meta.getTrainingPatchesTrain());
        assertEquals(16, meta.getTrainingPatchesValidation());
    }
}
