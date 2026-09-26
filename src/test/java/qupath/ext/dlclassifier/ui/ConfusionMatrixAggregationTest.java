package qupath.ext.dlclassifier.ui;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.service.ClassifierClient;

/**
 * Pins the denominator of the Confusion Matrix tab.
 *
 * <p>The matrix used to divide each off-diagonal pixel count by a denominator
 * assembled from confusion pairs alone. A tile that predicted a class perfectly
 * emits no pair for that class, so its correct pixels never reached the
 * denominator while every other tile's errors stayed in the numerator. With 36
 * clean tiles and one tile that misread 98% of its {@code Ignore*} pixels, the
 * tab reported the bad tile's own 98% as the session-wide figure.
 *
 * <p>The fix carries a per-tile {@code gt_pixel_totals} map for every class
 * present, confused or not, and sums that instead. These tests fix the
 * arithmetic at the numbers the bug got wrong, and hold the legacy path (a
 * session saved before the field existed) at its old behaviour rather than
 * letting it throw or read as exact.
 */
class ConfusionMatrixAggregationTest {

    private static final String IGNORE = "Ignore*";
    private static final String TISSUE = "Tissue";

    /** GT pixels of {@code Ignore*} in each tile of the fixture. */
    private static final long IGNORE_PX_PER_TILE = 10_000L;

    /** Pixels of {@code Ignore*} the one bad tile predicted as {@code Tissue}. */
    private static final long LEAKED_PX = 9_800L;

    private static final int CLEAN_TILES = 36;

    /**
     * Builds a row. {@code gtTotals} empty models a legacy session.
     */
    private static TrainingAreaIssuesDialog.TileRow row(
            String filename,
            Map<String, Double> perClassIoU,
            List<ClassifierClient.ConfusionPair> pairs,
            Map<String, Long> gtTotals) {
        return new TrainingAreaIssuesDialog.TileRow(new ClassifierClient.TileEvaluationResult(
                filename,
                "train",
                0.5,
                0.0,
                perClassIoU,
                0.5,
                0,
                0,
                "slide.tif",
                "id",
                null,
                null,
                null,
                null,
                null,
                null,
                pairs,
                0L,
                List.of(),
                gtTotals));
    }

    private static Map<String, Double> bothClassesIoU() {
        Map<String, Double> iou = new LinkedHashMap<>();
        iou.put(IGNORE, 0.5);
        iou.put(TISSUE, 0.5);
        return iou;
    }

    private static Map<String, Long> fullTotals() {
        Map<String, Long> totals = new LinkedHashMap<>();
        totals.put(IGNORE, IGNORE_PX_PER_TILE);
        totals.put(TISSUE, IGNORE_PX_PER_TILE);
        return totals;
    }

    /**
     * 37 tiles: one leaks 98% of its {@code Ignore*} to {@code Tissue}, the
     * other 36 get {@code Ignore*} exactly right and so emit no pair at all.
     *
     * @param withTotals false to strip {@code gtPixelTotals}, i.e. a legacy session
     */
    private static List<TrainingAreaIssuesDialog.TileRow> thirtySevenTiles(boolean withTotals) {
        List<TrainingAreaIssuesDialog.TileRow> rows = new ArrayList<>();
        rows.add(row(
                "bad.tif",
                bothClassesIoU(),
                List.of(new ClassifierClient.ConfusionPair(IGNORE, TISSUE, LEAKED_PX, IGNORE_PX_PER_TILE)),
                withTotals ? fullTotals() : Map.of()));
        for (int i = 0; i < CLEAN_TILES; i++) {
            rows.add(row("clean" + i + ".tif", bothClassesIoU(), List.of(), withTotals ? fullTotals() : Map.of()));
        }
        return rows;
    }

    private static double pct(TrainingAreaIssuesDialog.ConfusionMatrixData data, String gt, String pred) {
        long total = data.gtTotals().getOrDefault(gt, 0L);
        long pixels = data.offDiag().getOrDefault(gt, Map.of()).getOrDefault(pred, 0L);
        return total > 0 ? (100.0 * pixels / total) : 0.0;
    }

    private static double diagonalPct(TrainingAreaIssuesDialog.ConfusionMatrixData data, String gt) {
        long total = data.gtTotals().getOrDefault(gt, 0L);
        long off = 0;
        for (Long v : data.offDiag().getOrDefault(gt, Map.of()).values()) {
            off += v;
        }
        return total > 0 ? (100.0 * Math.max(total - off, 0L) / total) : 0.0;
    }

    @Test
    @DisplayName("one bad tile among 37 reads as a few percent, not 98%")
    void oneBadTileDoesNotSpeakForTheSession() {
        var data = TrainingAreaIssuesDialog.aggregateConfusionMatrix(thirtySevenTiles(true));

        // Every tile's Ignore* pixels are in the denominator, clean ones included.
        assertEquals(37 * IGNORE_PX_PER_TILE, data.gtTotals().get(IGNORE), "denominator must span all 37 tiles");
        assertEquals(37, data.totalTiles());

        double leak = pct(data, IGNORE, TISSUE);
        assertEquals(100.0 * LEAKED_PX / (37 * IGNORE_PX_PER_TILE), leak, 1e-9);
        assertTrue(leak < 3.0, "expected a low single-digit figure, got " + leak);

        // The number the bug produced, for the record.
        assertTrue(leak < 98.0 / 10, "must be nowhere near the bad tile's own 98%, got " + leak);

        assertEquals(100.0 - leak, diagonalPct(data, IGNORE), 1e-9);
        assertTrue(data.exact(), "all tiles carry GT totals, so the matrix is exact");
    }

    @Test
    @DisplayName("a session with no GT totals still produces the old numbers")
    void legacySessionFallsBackToPairDerivedTotals() {
        var data = TrainingAreaIssuesDialog.aggregateConfusionMatrix(thirtySevenTiles(false));

        // Only the bad tile contributed a pair, so only its 10k pixels are known.
        assertEquals(IGNORE_PX_PER_TILE, data.gtTotals().get(IGNORE));
        assertEquals(98.0, pct(data, IGNORE, TISSUE), 1e-9);
        assertFalse(data.exact(), "no per-class totals available -- must not claim to be exact");

        // Both classes still reach the axes, via perClassIoU.
        assertEquals(List.of(IGNORE, TISSUE), data.classes());
    }

    @Test
    @DisplayName("mixing one legacy tile into a modern session downgrades to approximate")
    void oneLegacyTileMakesTheWholeMatrixApproximate() {
        List<TrainingAreaIssuesDialog.TileRow> rows = new ArrayList<>(thirtySevenTiles(true));
        rows.add(row(
                "old.tif",
                bothClassesIoU(),
                List.of(new ClassifierClient.ConfusionPair(IGNORE, TISSUE, 100L, IGNORE_PX_PER_TILE)),
                Map.of()));

        var data = TrainingAreaIssuesDialog.aggregateConfusionMatrix(rows);
        assertFalse(data.exact());
        // The legacy tile still contributes both its numerator and (once) its
        // own pair-derived denominator, so nothing is silently dropped.
        assertEquals(38 * IGNORE_PX_PER_TILE, data.gtTotals().get(IGNORE));
        assertEquals(LEAKED_PX + 100L, data.offDiag().get(IGNORE).get(TISSUE));
    }

    @Test
    @DisplayName("a session with no confusions at all is 100% on the diagonal")
    void allCorrectSessionIsFullyDiagonal() {
        List<TrainingAreaIssuesDialog.TileRow> rows = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            rows.add(row("clean" + i + ".tif", bothClassesIoU(), List.of(), fullTotals()));
        }

        var data = TrainingAreaIssuesDialog.aggregateConfusionMatrix(rows);
        assertTrue(data.offDiag().isEmpty(), "no pairs means no off-diagonal pixels");
        assertEquals(100.0, diagonalPct(data, IGNORE), 1e-9);
        assertEquals(100.0, diagonalPct(data, TISSUE), 1e-9);
        assertEquals(0.0, pct(data, IGNORE, TISSUE), 1e-9);
        assertTrue(data.exact(), "a perfect modern session is exact, not approximate");
    }

    @Test
    @DisplayName("a tile with no labeled pixels does not make the matrix approximate")
    void unlabeledTileIsNotTreatedAsLegacy() {
        List<TrainingAreaIssuesDialog.TileRow> rows = new ArrayList<>(thirtySevenTiles(true));
        // No GT totals and no pairs: nothing was labeled here. It carries no
        // information either way and must not be read as a legacy tile.
        rows.add(row("blank.tif", bothClassesIoU(), List.of(), Map.of()));

        var data = TrainingAreaIssuesDialog.aggregateConfusionMatrix(rows);
        assertTrue(data.exact(), "an unlabeled tile is not a legacy tile");
        assertEquals(37 * IGNORE_PX_PER_TILE, data.gtTotals().get(IGNORE));
    }

    @Test
    @DisplayName("aggregating nothing yields an empty matrix, not a crash")
    void emptySessionAggregatesCleanly() {
        var data = TrainingAreaIssuesDialog.aggregateConfusionMatrix(List.of());
        assertTrue(data.classes().isEmpty());
        assertEquals(0, data.totalTiles());
        assertTrue(data.exact());
    }
}
