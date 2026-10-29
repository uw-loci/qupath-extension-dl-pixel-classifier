package qupath.ext.dlclassifier.utilities;

import java.awt.Shape;
import java.util.Locale;

/**
 * Decisions the unified OBJECTS / MEASUREMENTS pipeline makes about an annotation before and
 * while it walks the annotation's bounding box on the overlay's tile grid.
 */
public final class RegionTiling {

    /**
     * Share of the free heap the OBJECTS class map may take. Contour tracing and the objects
     * it builds need room on top of the map itself.
     */
    public static final double CLASS_MAP_HEAP_SHARE = 0.5;

    private RegionTiling() {}

    /**
     * Whether a tile's area (full-resolution coordinates) can contain any of the annotation.
     * A tile that cannot is skipped: it would only be classified to be clipped or masked away,
     * and a thin or L-shaped annotation's bounding box is mostly such tiles.
     * <p>
     * {@link Shape#intersects} may answer true when unsure but never false for a real overlap,
     * so skipping on false is safe. A null shape means "no annotation": every tile counts.
     */
    public static boolean tileTouches(Shape roiShape, double x, double y, double w, double h) {
        return roiShape == null || roiShape.intersects(x, y, w, h);
    }

    /** Bytes of an {@code int[height][width]} class map, including per-row array headers. */
    public static long classMapBytes(long width, long height) {
        return height * (16L + 4L * width);
    }

    /**
     * Why an OBJECTS run on this region cannot be done in memory, or null when it can.
     * <p>
     * OBJECTS holds a class map of the annotation's whole bounding box at the model's
     * resolution: about 40 GB for a 100k x 100k region at full resolution. Allocating that
     * fails with an OutOfMemoryError that can leave QuPath unstable, so the run is refused
     * up front with numbers and ways out instead.
     *
     * @param width          class-map width in classifier pixels
     * @param height         class-map height in classifier pixels
     * @param availableBytes heap QuPath can still allocate
     * @param downsample     the model's downsample, for the message
     */
    public static String objectsClassMapProblem(long width, long height, long availableBytes, double downsample) {
        long needed = classMapBytes(width, height);
        if (needed <= CLASS_MAP_HEAP_SHARE * availableBytes) {
            return null;
        }
        return String.format(
                Locale.ROOT,
                "This annotation is too large for Objects output: at the model's %.0fx resolution its "
                        + "classification map is %d x %d pixels and needs %.1f GB, but QuPath has %.1f GB of "
                        + "memory free. Split the annotation into smaller ones, use Measurements output "
                        + "(which does not need the map), or raise QuPath's memory limit "
                        + "(Edit > Preferences > General > Set max memory %%, then restart QuPath).",
                downsample,
                width,
                height,
                needed / 1e9,
                availableBytes / 1e9);
    }

    /** Heap the JVM can still allocate: the maximum minus what is in use now. */
    public static long availableHeapBytes() {
        Runtime rt = Runtime.getRuntime();
        return rt.maxMemory() - (rt.totalMemory() - rt.freeMemory());
    }
}
