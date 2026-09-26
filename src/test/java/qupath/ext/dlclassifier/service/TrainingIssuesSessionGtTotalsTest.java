package qupath.ext.dlclassifier.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.dlclassifier.model.ClassifierMetadata;

/**
 * Round-trips the per-class ground-truth pixel totals through Save/Load Session.
 *
 * <p>Those totals are the Confusion Matrix tab's denominator. If the session
 * store drops them, reopening a saved session silently reverts the matrix to the
 * pair-derived denominator that overstated every percentage -- and does so
 * without any error, which is why this is worth a test rather than a glance.
 */
class TrainingIssuesSessionGtTotalsTest {

    private static ClassifierMetadata metadata() {
        return ClassifierMetadata.builder()
                .id("test")
                .name("test")
                .classes(List.of(new ClassifierMetadata.ClassInfo(0, "Tissue", "#ff0000")))
                .build();
    }

    private static ClassifierClient.TileEvaluationResult tile(String filename, Map<String, Long> gtTotals) {
        return new ClassifierClient.TileEvaluationResult(
                filename,
                "train",
                0.25,
                1.5,
                Map.of("Tissue", 0.9),
                0.9,
                10,
                20,
                "slide.tif",
                "id",
                null,
                null,
                null,
                null,
                null,
                null,
                List.of(new ClassifierClient.ConfusionPair("Ignore*", "Tissue", 12L, 100L)),
                12L,
                List.of(),
                gtTotals);
    }

    @Test
    @DisplayName("GT pixel totals survive save then load")
    void totalsRoundTrip(@TempDir Path modelDir) throws IOException {
        Map<String, Long> totals = new LinkedHashMap<>();
        totals.put("Ignore*", 100L);
        totals.put("Tissue", 900L);

        Path sessionDir =
                TrainingIssuesSessionStore.save(metadata(), modelDir, 2.0, 512, List.of(tile("a.tif", totals)), "note");

        var loaded = TrainingIssuesSessionStore.load(sessionDir);
        assertEquals(1, loaded.results().size());
        assertEquals(totals, loaded.results().get(0).gtPixelTotals());
    }

    @Test
    @DisplayName("a session saved without the field loads with empty totals")
    void legacyManifestLoadsWithoutTheField(@TempDir Path modelDir) throws IOException {
        Path sessionDir = TrainingIssuesSessionStore.save(
                metadata(), modelDir, 2.0, 512, List.of(tile("a.tif", Map.of("Tissue", 900L))), "note");

        // Strip the block the way a legacy writer would have left it out.
        Path manifestPath = sessionDir.resolve("session.json");
        JsonObject manifest =
                JsonParser.parseString(Files.readString(manifestPath)).getAsJsonObject();
        JsonObject firstTile = manifest.getAsJsonArray("tiles").get(0).getAsJsonObject();
        assertTrue(firstTile.has("gtPixelTotals"), "the writer must emit the block in the first place");
        firstTile.remove("gtPixelTotals");
        Files.writeString(manifestPath, manifest.toString());

        var loaded = TrainingIssuesSessionStore.load(sessionDir);
        assertEquals(1, loaded.results().size());
        assertTrue(loaded.results().get(0).gtPixelTotals().isEmpty(), "missing block must load as empty, not throw");
        // The rest of the tile is untouched, so the legacy fallback still has
        // its pair-derived denominator to work with.
        assertEquals(1, loaded.results().get(0).topConfusions().size());
    }
}
