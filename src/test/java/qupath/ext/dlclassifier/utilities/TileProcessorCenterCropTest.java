package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.InferenceConfig;

/**
 * Guards CENTER_CROP tile coverage.
 *
 * <p>{@code createCenterCropWeights} discarded a margin equal to the TOTAL
 * overlap on each side, while the tile grid advances by
 * {@code tileSize - overlap}. The kept band was therefore always narrower than
 * the stride, so a stripe of every step was contributed by no tile at all. At
 * the shipped defaults -- 256 px tiles with 51 px of context padding per side,
 * giving 102 px total overlap -- the kept band was 52 px against a 154 px
 * stride, leaving 102 px of every step uncovered. Once overlap reached
 * tileSize/2 the band collapsed entirely and the output went blank.
 *
 * <p>The margin to discard per side is half the total overlap.
 */
class TileProcessorCenterCropTest {

    /** Width of the contiguous run of 1.0 weights along the centre row. */
    private static int keptBandWidth(float[][] weights) {
        int tileSize = weights.length;
        int mid = tileSize / 2;
        int kept = 0;
        for (int x = 0; x < tileSize; x++) {
            if (weights[mid][x] > 0f) {
                kept++;
            }
        }
        return kept;
    }

    private static TileProcessor centerCrop(int tileSize, int overlap) {
        return new TileProcessor(tileSize, overlap, InferenceConfig.BlendMode.CENTER_CROP, 16);
    }

    @Test
    @DisplayName("the kept band covers the stride at shipped defaults")
    void keptBandCoversStrideAtDefaults() {
        int tileSize = 256;
        int overlap = 2 * 51; // context padding is per-side; overlap is the total
        int stride = tileSize - overlap;

        int kept = keptBandWidth(centerCrop(tileSize, overlap).createBlendWeights());

        assertEquals(154, stride, "guard the arithmetic this test is built on");
        assertTrue(
                kept >= stride,
                "kept band " + kept + " px must cover the " + stride + " px stride, or tiles leave gaps");
    }

    @Test
    @DisplayName("no gaps across a range of tile and overlap sizes")
    void noGapsAcrossConfigurations() {
        int[] tileSizes = {128, 256, 512};
        int[] paddings = {8, 32, 51, 64};
        for (int tileSize : tileSizes) {
            for (int padding : paddings) {
                int overlap = 2 * padding;
                if (overlap >= tileSize) {
                    continue; // degenerate: stride would be <= 0
                }
                int stride = tileSize - overlap;
                int kept = keptBandWidth(centerCrop(tileSize, overlap).createBlendWeights());
                assertTrue(
                        kept >= stride,
                        String.format(
                                "tileSize=%d overlap=%d: kept %d px < stride %d px", tileSize, overlap, kept, stride));
            }
        }
    }

    @Test
    @DisplayName("a large overlap does not blank the tile")
    void largeOverlapDoesNotBlankTheTile() {
        // overlap == tileSize/2 collapsed the old kept band to zero, producing
        // an all-zero weight map and therefore no output at all.
        int kept = keptBandWidth(centerCrop(256, 128).createBlendWeights());

        assertTrue(kept > 0, "weights must not be entirely zero");
    }

    @Test
    @DisplayName("the kept band is centred")
    void keptBandIsCentred() {
        float[][] weights = centerCrop(256, 102).createBlendWeights();
        int mid = 256 / 2;

        assertTrue(weights[mid][mid] > 0f, "the tile centre must always be kept");
        assertEquals(0f, weights[mid][0], "the extreme edge must always be discarded");
        assertEquals(0f, weights[0][mid], "the extreme edge must always be discarded");
    }
}
