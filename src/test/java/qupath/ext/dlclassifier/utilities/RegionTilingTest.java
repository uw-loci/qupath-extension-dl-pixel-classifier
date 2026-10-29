package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.geom.Path2D;
import org.junit.jupiter.api.Test;

class RegionTilingTest {

    /** An L: a 1000 x 100 bar along the top and a 100 x 1000 bar down the left. */
    private static Path2D lShape() {
        Path2D p = new Path2D.Double();
        p.moveTo(0, 0);
        p.lineTo(1000, 0);
        p.lineTo(1000, 100);
        p.lineTo(100, 100);
        p.lineTo(100, 1000);
        p.lineTo(0, 1000);
        p.closePath();
        return p;
    }

    @Test
    void tilesInsideTheBoundingBoxButOutsideAnLAreSkipped() {
        Path2D l = lShape();
        int touched = 0;
        int total = 0;
        for (int y = 0; y < 1000; y += 200) {
            for (int x = 0; x < 1000; x += 200) {
                total++;
                if (RegionTiling.tileTouches(l, x, y, 200, 200)) touched++;
            }
        }
        // top row (5) + left column below it (4)
        assertEquals(25, total);
        assertEquals(9, touched);
        assertFalse(RegionTiling.tileTouches(l, 400, 400, 200, 200));
        assertTrue(RegionTiling.tileTouches(l, 90, 900, 200, 200), "a tile that grazes the L is kept");
        assertTrue(RegionTiling.tileTouches(null, 400, 400, 200, 200), "no ROI: every tile counts");
    }

    @Test
    void classMapBytesCountsIntsAndRowHeaders() {
        assertEquals(10L * (16 + 4 * 100), RegionTiling.classMapBytes(100, 10));
    }

    @Test
    void aFullResolutionWholeSlideIsRefusedWithNumbersAndWaysOut() {
        // 100k x 100k at ds 1 is ~40 GB; a 32 GB heap with 30 GB free cannot hold it.
        String why = RegionTiling.objectsClassMapProblem(100_000, 100_000, 30_000_000_000L, 1.0);
        assertNotNull(why);
        assertTrue(why.contains("40.0 GB"), why);
        assertTrue(why.contains("Measurements output"), why);
        assertTrue(why.contains("Edit > Preferences > General > Set max memory %"), why);
    }

    @Test
    void aMapUnderHalfTheFreeHeapIsAllowed() {
        long bytes = RegionTiling.classMapBytes(10_000, 10_000); // ~400 MB
        assertNull(RegionTiling.objectsClassMapProblem(10_000, 10_000, 2 * bytes + 1, 1.0));
        assertNotNull(RegionTiling.objectsClassMapProblem(10_000, 10_000, 2 * bytes - 1_000_000, 1.0));
    }
}
