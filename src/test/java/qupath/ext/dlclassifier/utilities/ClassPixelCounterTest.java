package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Path2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Counts must equal a brute-force "is the pixel's centre inside the annotation" count over
 * the same class map, whatever the shape, downsample or tiling. Swept over three shapes,
 * four downsamples (including a non-integer one) and three tile sizes that do and do not
 * divide the map.
 */
class ClassPixelCounterTest {

    private static final int CLASSES = 4;

    /** A class map with structure, so a misplaced mask changes the counts. */
    private static int[][] classMap(int w, int h, long seed) {
        int[][] m = new int[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // Stripes plus deterministic speckle, so a shifted mask changes the counts.
                long hash = (x * 7919L + y * 104729L + seed * 31L) % 10;
                m[y][x] = (int) (((x / 7) + (y / 5) + (hash == 0 ? 1 : 0)) % CLASSES);
            }
        }
        return m;
    }

    /**
     * Reference: is the pixel centre inside the shape's OUTLINE. For a polygon that is the
     * shape itself. For Ellipse2D it is the four cubic Beziers Java draws it with, which
     * bulge outward by up to ~0.03% of the radius; Ellipse2D.contains() uses the exact
     * ellipse instead, so it disagrees with any outline-based count on a few boundary pixels.
     */
    private static long[] bruteForce(int[][] map, Shape original, double rx, double ry, double ds) {
        Shape shape = original == null ? null : new Path2D.Double(original);
        long[] c = new long[CLASSES];
        for (int y = 0; y < map.length; y++) {
            for (int x = 0; x < map[0].length; x++) {
                double cx = rx + (x + 0.5) * ds;
                double cy = ry + (y + 0.5) * ds;
                if (shape == null || shape.contains(cx, cy)) {
                    c[map[y][x]]++;
                }
            }
        }
        return c;
    }

    private static long[] tiled(int[][] map, Shape shape, double rx, double ry, double ds, int tile) {
        int h = map.length;
        int w = map[0].length;
        ClassPixelCounter counter = new ClassPixelCounter(shape, rx, ry, ds, w, h);
        for (int ty = 0; ty < h; ty += tile) {
            for (int tx = 0; tx < w; tx += tile) {
                // Tiles may run past the map edge, as the classifier's do; the counter clips.
                BufferedImage img = new BufferedImage(tile, tile, BufferedImage.TYPE_BYTE_GRAY);
                WritableRaster ras = img.getRaster();
                for (int y = 0; y < tile; y++) {
                    for (int x = 0; x < tile; x++) {
                        int mx = tx + x;
                        int my = ty + y;
                        ras.setSample(x, y, 0, (mx < w && my < h) ? map[my][mx] : 3);
                    }
                }
                counter.add(ras, tx, ty);
            }
        }
        long[] out = new long[CLASSES];
        long[] got = counter.counts();
        System.arraycopy(got, 0, out, 0, Math.min(got.length, CLASSES));
        return out;
    }

    @Test
    void matchesBruteForceAcrossShapesDownsamplesAndTilings() {
        double rx = 1000.3;
        double ry = 2000.7;
        int checked = 0;
        for (double ds : new double[] {1.0, 2.0, 3.5, 8.0}) {
            int w = (int) Math.ceil(613.0 / ds);
            int h = (int) Math.ceil(457.0 / ds);
            int[][] map = classMap(w, h, 42);
            Path2D poly = new Path2D.Double();
            poly.moveTo(rx + 13.1, ry + 4.2);
            poly.lineTo(rx + 600.4, ry + 90.9);
            poly.lineTo(rx + 300.6, ry + 450.3);
            poly.lineTo(rx + 40.8, ry + 300.1);
            poly.closePath();
            List<Shape> shapes = List.of(
                    new Ellipse2D.Double(rx + 20.25, ry + 30.75, 550.5, 400.5),
                    new Rectangle2D.Double(rx + 100.4, ry + 50.6, 300.2, 200.2),
                    poly);
            for (Shape s : shapes) {
                long[] expected = bruteForce(map, s, rx, ry, ds);
                for (int tile : new int[] {16, 37, 128}) {
                    assertArrayEquals(
                            expected,
                            tiled(map, s, rx, ry, ds, tile),
                            s.getClass().getSimpleName() + " ds=" + ds + " tile=" + tile);
                    checked++;
                }
            }
        }
        assertEquals(36, checked);
    }

    @Test
    void nullShapeCountsEveryPixelOfTheMapAndNothingBeyondIt() {
        int[][] map = classMap(50, 30, 1);
        long[] got = tiled(map, null, 0, 0, 1.0, 37);
        long total = 0;
        for (long c : got) {
            total += c;
        }
        assertEquals(50L * 30L, total);
        assertArrayEquals(bruteForce(map, null, 0, 0, 1.0), got);
    }

    @Test
    void areaOfARectangleComesOutRight() {
        // 300 x 200 full-res pixels at ds 2 is 150 x 100 map pixels.
        int[][] map = new int[200][200];
        ClassPixelCounter counter =
                new ClassPixelCounter(new Rectangle2D.Double(10, 20, 300, 200), 0, 0, 2.0, 200, 200);
        BufferedImage img = new BufferedImage(200, 200, BufferedImage.TYPE_BYTE_GRAY);
        counter.add(img.getRaster(), 0, 0);
        assertEquals(150L * 100L, counter.total());
        assertEquals(0, map[0][0]);
    }
}
