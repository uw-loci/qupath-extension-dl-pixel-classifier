package qupath.ext.dlclassifier.utilities;

import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.FlatteningPathIterator;
import java.awt.geom.PathIterator;
import java.awt.image.Raster;
import java.util.Arrays;

/**
 * Counts classified pixels by class inside an annotation, one tile at a time.
 *
 * <p>MEASUREMENTS output used to average each tile's probabilities to a single vector and
 * count each TILE as one pixel, so class percentages were per-tile votes, areas were a few
 * pixels' worth, and "Total pixels" was the tile count; pixels outside the annotation but
 * inside a tile counted too. This counts the actual classified pixels of the class map the
 * overlay and OBJECTS use, keeping only those whose center lies inside the annotation.
 *
 * <p>Works per tile so memory stays at one tile's mask: a whole-slide annotation at full
 * resolution would need tens of gigabytes as a single class map.
 *
 * <p>Coordinates: the region's top-left is ({@code regionX}, {@code regionY}) in full-resolution
 * pixels; the class map is at classifier resolution, one map pixel per {@code downsample}
 * full-resolution pixels. A map pixel (mx, my) covers full-resolution
 * [regionX + mx*ds, regionX + (mx+1)*ds), and counts when that square's center is inside the
 * shape (Java2D's non-antialiased fill rule).
 */
public final class ClassPixelCounter {

    private final Shape shape;
    private final double regionX;
    private final double regionY;
    private final double downsample;
    private final int mapWidth;
    private final int mapHeight;
    private long[] counts = new long[0];

    /**
     * @param shape      the annotation's shape in full-resolution image coordinates, or null to
     *                   count every pixel of the region
     * @param regionX    full-resolution x of the class map's left edge
     * @param regionY    full-resolution y of the class map's top edge
     * @param downsample full-resolution pixels per class-map pixel
     * @param mapWidth   class-map width; pixels at or beyond it are ignored
     * @param mapHeight  class-map height; pixels at or beyond it are ignored
     */
    public ClassPixelCounter(
            Shape shape, double regionX, double regionY, double downsample, int mapWidth, int mapHeight) {
        this.shape = shape;
        this.regionX = regionX;
        this.regionY = regionY;
        this.downsample = downsample;
        this.mapWidth = mapWidth;
        this.mapHeight = mapHeight;
    }

    /**
     * Adds one tile of class indices (band 0 of {@code tile}) placed at class-map position
     * ({@code outX}, {@code outY}).
     */
    public void add(Raster tile, int outX, int outY) {
        int w = Math.min(tile.getWidth(), mapWidth - outX);
        int h = Math.min(tile.getHeight(), mapHeight - outY);
        if (w <= 0 || h <= 0) {
            return;
        }
        boolean[] mask = shape == null ? null : mask(outX, outY, w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mask != null && !mask[y * w + x]) {
                    continue;
                }
                int c = tile.getSample(x, y, 0);
                if (c < 0) {
                    continue;
                }
                if (c >= counts.length) {
                    counts = Arrays.copyOf(counts, c + 1);
                }
                counts[c]++;
            }
        }
    }

    /** Pixel count per class index; index i is class i. Length is one past the highest class seen. */
    public long[] counts() {
        return counts.clone();
    }

    /** Total pixels counted, over every class. */
    public long total() {
        long t = 0;
        for (long c : counts) {
            t += c;
        }
        return t;
    }

    /**
     * The annotation over a w x h block of class-map pixels starting at (outX, outY), as a
     * row-major boolean mask: true where the pixel's center is inside the shape.
     *
     * <p>A scanline fill rather than Java2D's rasterizer: the rasterizer places edges with
     * limited sub-pixel precision, which put an ellipse's boundary pixels on the wrong side
     * of the center rule (10 of 43,065 even after fine flattening). Here each row's center
     * line is intersected with the shape's edges, flattened in mask-pixel space to 1e-4 of a
     * pixel, and spans are filled by the shape's own winding rule, so holes in a QuPath ROI
     * stay holes.
     */
    private boolean[] mask(int outX, int outY, int w, int h) {
        AffineTransform t = new AffineTransform();
        t.scale(1.0 / downsample, 1.0 / downsample);
        t.translate(-(regionX + outX * downsample), -(regionY + outY * downsample));
        PathIterator it = new FlatteningPathIterator(shape.getPathIterator(t), 1e-4, 16);
        boolean evenOdd = it.getWindingRule() == PathIterator.WIND_EVEN_ODD;

        // Edges as (x0, y0, x1, y1), closing each subpath.
        double[] edges = new double[64];
        int n = 0;
        double[] c = new double[6];
        double sx = 0, sy = 0, px = 0, py = 0;
        for (; !it.isDone(); it.next()) {
            int type = it.currentSegment(c);
            if (type == PathIterator.SEG_MOVETO) {
                if (px != sx || py != sy) {
                    edges = push(edges, n++, px, py, sx, sy);
                }
                sx = px = c[0];
                sy = py = c[1];
            } else if (type == PathIterator.SEG_LINETO) {
                edges = push(edges, n++, px, py, c[0], c[1]);
                px = c[0];
                py = c[1];
            } else if (type == PathIterator.SEG_CLOSE) {
                edges = push(edges, n++, px, py, sx, sy);
                px = sx;
                py = sy;
            }
        }
        if (px != sx || py != sy) {
            edges = push(edges, n++, px, py, sx, sy);
        }

        boolean[] inside = new boolean[w * h];
        double[] xs = new double[16];
        int[] dirs = new int[16];
        for (int y = 0; y < h; y++) {
            double yc = y + 0.5;
            int k = 0;
            for (int e = 0; e < n; e++) {
                double x0 = edges[4 * e], y0 = edges[4 * e + 1], x1 = edges[4 * e + 2], y1 = edges[4 * e + 3];
                int dir;
                if (y0 <= yc && yc < y1) {
                    dir = 1;
                } else if (y1 <= yc && yc < y0) {
                    dir = -1;
                } else {
                    continue;
                }
                if (k == xs.length) {
                    xs = Arrays.copyOf(xs, 2 * k);
                    dirs = Arrays.copyOf(dirs, 2 * k);
                }
                xs[k] = x0 + (yc - y0) * (x1 - x0) / (y1 - y0);
                dirs[k] = dir;
                k++;
            }
            sortByX(xs, dirs, k);
            int winding = 0;
            for (int a = 0; a < k - 1; a++) {
                winding += evenOdd ? 1 : dirs[a];
                boolean in = evenOdd ? (winding & 1) == 1 : winding != 0;
                if (!in) {
                    continue;
                }
                // pixel x is inside when xs[a] <= x + 0.5 < xs[a + 1]
                int from = Math.max(0, (int) Math.ceil(xs[a] - 0.5));
                int to = Math.min(w, (int) Math.ceil(xs[a + 1] - 0.5));
                for (int x = from; x < to; x++) {
                    inside[y * w + x] = true;
                }
            }
        }
        return inside;
    }

    private static double[] push(double[] edges, int i, double x0, double y0, double x1, double y1) {
        if (4 * i + 4 > edges.length) {
            edges = Arrays.copyOf(edges, edges.length * 2);
        }
        edges[4 * i] = x0;
        edges[4 * i + 1] = y0;
        edges[4 * i + 2] = x1;
        edges[4 * i + 3] = y1;
        return edges;
    }

    /** Insertion sort; a row crosses few edges. */
    private static void sortByX(double[] xs, int[] dirs, int k) {
        for (int i = 1; i < k; i++) {
            double x = xs[i];
            int d = dirs[i];
            int j = i - 1;
            while (j >= 0 && xs[j] > x) {
                xs[j + 1] = xs[j];
                dirs[j + 1] = dirs[j];
                j--;
            }
            xs[j + 1] = x;
            dirs[j + 1] = d;
        }
    }
}
