package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.ClassedRoi;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.Counts;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.ImageAnnotations;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.Role;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.RoledImage;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;

/**
 * The pre-export patch count follows the exporter's rules: the same sampler,
 * the same class filter, the same per-image roles.
 */
class PatchCountEstimatorTest {

    private static ImageAnnotations image(String name) {
        return new ImageAnnotations(
                name,
                10000,
                10000,
                List.of(
                        new ClassedRoi(
                                "A", ROIs.createRectangleROI(1000, 1000, 4000, 4000, ImagePlane.getDefaultPlane())),
                        new ClassedRoi(
                                "B", ROIs.createRectangleROI(6000, 6000, 2000, 2000, ImagePlane.getDefaultPlane()))));
    }

    private static int sampled(ImageAnnotations img, Set<String> classes, int tile, int overlap, double ds) {
        List<PatchSampler.AnnotationGeometry> g = img.annotations().stream()
                .filter(a -> classes.contains(a.className()))
                .map(a -> new PatchSampler.AnnotationGeometry(a.roi(), PatchSampler.isSparseROI(a.roi())))
                .toList();
        return PatchSampler.generatePatchLocations(g, tile, overlap, ds, img.width(), img.height())
                .size();
    }

    @Test
    void countIsExactlyTheExportersSamplerOnTheSelectedClasses() {
        ImageAnnotations img = image("i");
        for (int tile : new int[] {128, 256, 512}) {
            for (double ds : new double[] {1, 2, 4}) {
                Counts c = PatchCountEstimator.count(
                        List.of(new RoledImage(img, Role.TRAIN)), Set.of("A"), tile, tile / 5, ds, 0.2, false);
                assertThat(c.train())
                        .as("tile %d ds %s", tile, ds)
                        .isEqualTo(sampled(img, Set.of("A"), tile, tile / 5, ds));
                assertThat(c.val()).isZero();
            }
        }
    }

    @Test
    void rolesSendWholeImagesToOneSideAndBothFollowsTheSplit() {
        ImageAnnotations a = image("a");
        ImageAnnotations b = image("b");
        ImageAnnotations c = image("c");
        int n = sampled(a, Set.of("A", "B"), 256, 51, 1);
        Counts counts = PatchCountEstimator.count(
                List.of(new RoledImage(a, Role.TRAIN), new RoledImage(b, Role.VAL), new RoledImage(c, Role.BOTH)),
                Set.of("A", "B"),
                256,
                51,
                1,
                0.25,
                false);
        int bothVal = (int) Math.round(n * 0.25);
        assertThat(counts.train()).isEqualTo(n + (n - bothVal));
        assertThat(counts.val()).isEqualTo(n + bothVal);
    }

    @Test
    void anImageWithNoSelectedClassContributesNothing() {
        Counts c = PatchCountEstimator.count(
                List.of(new RoledImage(image("i"), Role.BOTH)), Set.of("Z"), 256, 0, 1, 0.2, false);
        assertThat(c.total()).isZero();
    }

    @Test
    void wholeImageModeIsOnePatchPerImage() {
        Counts c = PatchCountEstimator.count(
                List.of(new RoledImage(image("a"), Role.TRAIN), new RoledImage(image("b"), Role.VAL)),
                Set.of("A"),
                256,
                0,
                1,
                0.2,
                true);
        assertThat(c).isEqualTo(new Counts(1, 1));
    }

    @Test
    void cacheBytesFollowTheExportFormat() {
        // <= 4 bands of 8-bit: one byte per pixel per channel, plus a one-byte mask
        assertThat(PatchCountEstimator.cacheBytes(10, 100, 3, 8, false)).isEqualTo(10L * (100 * 100 * 3 + 100 * 100));
        // more bands or deeper data: float32
        assertThat(PatchCountEstimator.cacheBytes(10, 100, 6, 8, false))
                .isEqualTo(10L * (100 * 100 * 6 * 4 + 100 * 100));
        // context doubles the image, not the mask
        assertThat(PatchCountEstimator.cacheBytes(10, 100, 3, 8, true))
                .isEqualTo(10L * (2 * 100 * 100 * 3 + 100 * 100));
    }
}
