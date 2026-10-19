package qupath.ext.dlclassifier.utilities;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import qupath.lib.roi.interfaces.ROI;

/**
 * Counts the patches an export would start from, before any pixel is read.
 * <p>
 * Runs the exporter's own {@link PatchSampler} on the annotation geometry, so
 * the count moves with tile size, resolution, overlap and the class selection
 * exactly as the export will. It is a candidate count: the export then drops
 * patches that fail the minimum-coverage and minimum-label filters, which need
 * the pixels. So it reads high, by a factor that depends on the annotations
 * rather than on the settings -- which is what a relative gauge needs.
 * <p>
 * This replaced an area-based estimate that stepped a fixed 75% of the tile
 * and ignored overlap. Line annotations have almost no area, so on the
 * workshop's sparse-stroke recipe it said ~45 patches against 140 exported.
 */
public final class PatchCountEstimator {

    private PatchCountEstimator() {}

    /**
     * The training script's "auto" rule: it preloads the dataset only when it
     * fits in this share of available RAM. Mirrors {@code _AUTO_CACHE_FRACTION}
     * in {@code scripts/train.py}; change both together.
     */
    public static final double AUTO_CACHE_FRACTION = 0.50;

    /**
     * The training script's "bounded" cap: it preloads at most this share of
     * available RAM and trains on a subset beyond it. Mirrors the default of
     * {@code cache_bounded_fraction} in {@code scripts/train.py}.
     */
    public static final double BOUNDED_CACHE_FRACTION = 0.40;

    /** How an image's patches are split. Mirrors the dialog's per-image role. */
    public enum Role {
        BOTH,
        TRAIN,
        VAL
    }

    /** One annotation, with the class it belongs to. */
    public record ClassedRoi(String className, ROI roi) {}

    /** The annotations of one image, captured when classes were loaded. */
    public record ImageAnnotations(String imageName, int width, int height, List<ClassedRoi> annotations) {}

    /** One image as it will be exported: its annotations and its current role. */
    public record RoledImage(ImageAnnotations image, Role role) {}

    /**
     * Patch counts.
     *
     * @param train patches the training set would start from
     * @param val   patches the validation set would start from
     */
    public record Counts(int train, int val) {
        public int total() {
            return train + val;
        }
    }

    /**
     * Counts candidate patches.
     *
     * @param images          the selected images and their roles
     * @param classes         the selected classes; annotations of other classes are ignored
     * @param tileSize        tile size in output pixels
     * @param overlapPixels   overlap in output pixels, as the exporter receives it
     * @param downsample      export downsample
     * @param validationSplit fraction (0-1) of a "Both" image's patches that go to validation
     * @param wholeImage      whole-image mode: one patch per image
     * @return the counts
     */
    public static Counts count(
            List<RoledImage> images,
            Set<String> classes,
            int tileSize,
            int overlapPixels,
            double downsample,
            double validationSplit,
            boolean wholeImage) {
        int train = 0;
        int val = 0;
        double split = Math.max(0, Math.min(1, validationSplit));
        for (RoledImage ri : images) {
            List<PatchSampler.AnnotationGeometry> geoms = new ArrayList<>();
            for (ClassedRoi a : ri.image().annotations()) {
                if (classes.contains(a.className())) {
                    geoms.add(new PatchSampler.AnnotationGeometry(a.roi(), PatchSampler.isSparseROI(a.roi())));
                }
            }
            if (geoms.isEmpty()) {
                continue;
            }
            int n = wholeImage
                    ? 1
                    : PatchSampler.generatePatchLocations(
                                    geoms,
                                    tileSize,
                                    overlapPixels,
                                    downsample,
                                    ri.image().width(),
                                    ri.image().height())
                            .size();
            switch (ri.role()) {
                case TRAIN -> train += n;
                case VAL -> val += n;
                default -> {
                    int v = (int) Math.round(n * split);
                    val += v;
                    train += n - v;
                }
            }
        }
        return new Counts(train, val);
    }

    /**
     * Bytes the in-memory dataset cache would need.
     * <p>
     * Mirrors the export format: {@code AnnotationExtractor.savePatch} writes
     * uint8 when there are at most four bands of 8-bit data and float32
     * otherwise; context doubles the image; masks are cached as one byte per
     * pixel.
     *
     * @param patches   number of patches
     * @param tileSize  tile size in pixels
     * @param channels  input channels
     * @param bitDepth  bit depth of the source
     * @param context   whether a context tile accompanies each patch
     * @return bytes
     */
    public static long cacheBytes(int patches, int tileSize, int channels, int bitDepth, boolean context) {
        int bytesPerPixel = (channels <= 4 && bitDepth == 8) ? 1 : 4;
        long perImage = (long) tileSize * tileSize * Math.max(1, channels) * bytesPerPixel;
        if (context) perImage *= 2L;
        long perMask = (long) tileSize * tileSize;
        return (long) Math.max(0, patches) * (perImage + perMask);
    }
}
