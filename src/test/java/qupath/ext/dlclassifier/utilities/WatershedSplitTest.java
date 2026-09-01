package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayDeque;
import java.util.Deque;
import org.junit.jupiter.api.Test;

/**
 * Verifies the distance-transform watershed used to separate touching instances
 * of a class ({@link OutputGenerator#watershedForeground}).
 *
 * <p>The feature it backs relies on one non-obvious ImageJ contract: that
 * {@code MaximumFinder.findMaxima(edm, tolerance, ..., SEGMENTED, false, true)}
 * over a Euclidean distance map draws ridge lines between basins, so two abutting
 * blobs of the same class trace as two objects instead of one. These tests pin
 * that behavior against a synthetic "figure-8" (two overlapping disks) so a future
 * ImageJ bump or an argument-order slip is caught here rather than in the field.
 */
class WatershedSplitTest {

    private static final int TARGET = 1;

    @Test
    void splitsTwoTouchingDisksIntoTwoComponents() {
        int w = 120, h = 60;
        int[][] classMap = twoTouchingDisks(w, h);

        // Sanity: before watershed the two disks are a single connected component.
        assertThat(countComponents(rawForeground(classMap, TARGET, w, h), w, h))
                .as("touching disks should be one blob without watershed")
                .isEqualTo(1);

        boolean[] fg = OutputGenerator.watershedForeground(classMap, TARGET, w, h, 0.5);

        assertThat(countComponents(fg, w, h))
                .as("watershed should cut the two touching disks apart")
                .isEqualTo(2);
    }

    @Test
    void leavesAnIsolatedDiskAsOneComponent() {
        int w = 60, h = 60;
        int[][] classMap = new int[h][w];
        fillDisk(classMap, 30, 30, 18, TARGET);

        boolean[] fg = OutputGenerator.watershedForeground(classMap, TARGET, w, h, 0.5);

        assertThat(countComponents(fg, w, h))
                .as("a single convex disk must not be over-fragmented")
                .isEqualTo(1);
    }

    @Test
    void higherToleranceProducesNoMoreCutsThanLower() {
        int w = 120, h = 60;
        int[][] classMap = twoTouchingDisks(w, h);

        int lowTol = countComponents(OutputGenerator.watershedForeground(classMap, TARGET, w, h, 0.5), w, h);
        int highTol = countComponents(OutputGenerator.watershedForeground(classMap, TARGET, w, h, 8.0), w, h);

        // Higher tolerance merges nearby maxima, so it never splits MORE than a lower one.
        assertThat(highTol).as("higher tolerance = fewer or equal cuts").isLessThanOrEqualTo(lowTol);
    }

    // ---- helpers -----------------------------------------------------------

    /** Two disks whose edges overlap slightly, forming a figure-8 in one class. */
    private static int[][] twoTouchingDisks(int w, int h) {
        int[][] map = new int[h][w];
        int r = 22;
        int cy = h / 2;
        fillDisk(map, 40, cy, r, TARGET);
        fillDisk(map, 80, cy, r, TARGET); // centers 40px apart, r=22 -> ~4px overlap
        return map;
    }

    private static void fillDisk(int[][] map, int cx, int cy, int r, int value) {
        int h = map.length, w = map[0].length;
        int r2 = r * r;
        for (int y = Math.max(0, cy - r); y < Math.min(h, cy + r + 1); y++) {
            for (int x = Math.max(0, cx - r); x < Math.min(w, cx + r + 1); x++) {
                int dx = x - cx, dy = y - cy;
                if (dx * dx + dy * dy <= r2) map[y][x] = value;
            }
        }
    }

    private static boolean[] rawForeground(int[][] map, int target, int w, int h) {
        boolean[] fg = new boolean[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                fg[y * w + x] = map[y][x] == target;
            }
        }
        return fg;
    }

    /** 4-connected component count over a boolean foreground mask. */
    private static int countComponents(boolean[] fg, int w, int h) {
        boolean[] seen = new boolean[fg.length];
        int components = 0;
        Deque<int[]> stack = new ArrayDeque<>();
        for (int start = 0; start < fg.length; start++) {
            if (!fg[start] || seen[start]) continue;
            components++;
            seen[start] = true;
            stack.push(new int[] {start % w, start / w});
            while (!stack.isEmpty()) {
                int[] p = stack.pop();
                int px = p[0], py = p[1];
                int[][] nbrs = {{px - 1, py}, {px + 1, py}, {px, py - 1}, {px, py + 1}};
                for (int[] n : nbrs) {
                    int nx = n[0], ny = n[1];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    int idx = ny * w + nx;
                    if (fg[idx] && !seen[idx]) {
                        seen[idx] = true;
                        stack.push(new int[] {nx, ny});
                    }
                }
            }
        }
        return components;
    }
}
