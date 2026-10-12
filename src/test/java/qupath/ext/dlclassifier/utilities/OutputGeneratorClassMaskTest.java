package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Guards the per-class mask that contour tracing runs on (GitHub issue #29).
 *
 * <p>Issue #24 made class index 0 eligible for object generation, and
 * {@link OutputGeneratorClassSelectionTest} locked that in -- but it only
 * asserts WHICH indices are eligible, never what geometry comes out. The mask
 * still labelled foreground with the class index itself, so class 0 wrote 0
 * for its own pixels, the same value as the background. Tracing the range
 * [0,0] then matched every pixel and the first class came out as one object
 * covering the whole inference ROI, with coordinates identical to the ROI's.
 *
 * <p>These tests assert the mask's CONTENT, which is the level the defect
 * lived at: foreground must be distinguishable from background for every
 * class index, index 0 included.
 */
class OutputGeneratorClassMaskTest {

    /**
     * Three classes in a 4x4 map. Class 0 holds the top-left 2x2 (4 px),
     * class 1 a single pixel, class 2 the remaining 11.
     */
    private static int[][] threeClassMap() {
        return new int[][] {
            {0, 0, 2, 2},
            {0, 0, 2, 2},
            {2, 2, 1, 2},
            {2, 2, 2, 2},
        };
    }

    private static int countForeground(float[] mask) {
        int n = 0;
        for (float v : mask) {
            if (v == OutputGenerator.MASK_FOREGROUND) n++;
        }
        return n;
    }

    @Test
    @DisplayName("class 0's mask marks only class 0's pixels, not the whole map")
    void classZeroMaskIsNotTheWholeMap() {
        // The bug in one assertion: with foreground == targetClass == 0, every
        // one of the 16 values was 0 and the traced region was everything.
        float[] mask = OutputGenerator.classMaskValues(threeClassMap(), 0, 4, 4);

        assertEquals(4, countForeground(mask), "class 0 covers 4 of 16 pixels");
        assertNotEquals(16, countForeground(mask), "class 0 must not trace the entire region");
    }

    @Test
    @DisplayName("foreground is distinguishable from background for every class")
    void foregroundDiffersFromBackgroundForEveryClass() {
        // The identity that was violated, stated directly: whatever value
        // marks foreground, it cannot be the value that marks background,
        // for ANY class index. This is what makes the guard general rather
        // than a fix aimed at index 0.
        int[][] map = threeClassMap();
        for (int targetClass = 0; targetClass < 3; targetClass++) {
            float[] mask = OutputGenerator.classMaskValues(map, targetClass, 4, 4);
            int foreground = countForeground(mask);
            assertTrue(foreground > 0, "class " + targetClass + " must have foreground pixels");
            assertTrue(foreground < mask.length, "class " + targetClass + " must leave some background");
        }
        assertNotEquals(
                OutputGenerator.MASK_BACKGROUND,
                OutputGenerator.MASK_FOREGROUND,
                "foreground must not collide with background");
    }

    @Test
    @DisplayName("each class's foreground count matches its pixel count")
    void maskCountsMatchClassCounts() {
        int[][] map = threeClassMap();
        int[] expected = {4, 1, 11};

        for (int targetClass = 0; targetClass < 3; targetClass++) {
            float[] mask = OutputGenerator.classMaskValues(map, targetClass, 4, 4);
            assertEquals(
                    expected[targetClass], countForeground(mask), "class " + targetClass + " foreground pixel count");
        }
    }

    @Test
    @DisplayName("the masks partition the map -- no pixel is claimed twice or dropped")
    void masksPartitionTheMap() {
        // Catches a sentinel that accidentally matches a real class value, or
        // an off-by-one in the comparison, without naming any threshold.
        int[][] map = threeClassMap();
        int total = 0;
        for (int targetClass = 0; targetClass < 3; targetClass++) {
            total += countForeground(OutputGenerator.classMaskValues(map, targetClass, 4, 4));
        }

        assertEquals(16, total, "every pixel belongs to exactly one class mask");
    }

    @Test
    @DisplayName("the mask is binary -- it carries no class identity")
    void maskIsBinary() {
        // traceClassContours supplies the PathClass separately, so the mask
        // must not try to encode which class it represents.
        for (int targetClass = 0; targetClass < 3; targetClass++) {
            for (float v : OutputGenerator.classMaskValues(threeClassMap(), targetClass, 4, 4)) {
                assertTrue(
                        v == OutputGenerator.MASK_BACKGROUND || v == OutputGenerator.MASK_FOREGROUND,
                        "unexpected mask value " + v + " for class " + targetClass);
            }
        }
    }

    @Test
    @DisplayName("a class absent from the map yields an empty mask")
    void absentClassYieldsEmptyMask() {
        // Tracing an empty mask returns no geometry, which is the correct
        // outcome -- not an object covering everything.
        float[] mask = OutputGenerator.classMaskValues(threeClassMap(), 7, 4, 4);

        assertEquals(0, countForeground(mask));
    }

    @Test
    @DisplayName("a map that is entirely one class masks fully for it and empty for others")
    void uniformMapEdgeCase() {
        int[][] uniform = new int[][] {{0, 0}, {0, 0}};

        assertEquals(4, countForeground(OutputGenerator.classMaskValues(uniform, 0, 2, 2)));
        assertEquals(0, countForeground(OutputGenerator.classMaskValues(uniform, 1, 2, 2)));
    }
}
