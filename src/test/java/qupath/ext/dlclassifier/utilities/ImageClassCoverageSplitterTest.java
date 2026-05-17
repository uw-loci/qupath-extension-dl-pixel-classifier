package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.utilities.ImageClassCoverageSplitter.ImageAssignment;
import qupath.ext.dlclassifier.utilities.ImageClassCoverageSplitter.ImageInput;
import qupath.ext.dlclassifier.utilities.ImageClassCoverageSplitter.Role;
import qupath.ext.dlclassifier.utilities.ImageClassCoverageSplitter.SplitResult;

/**
 * Coverage gate for {@link ImageClassCoverageSplitter}.
 * <p>
 * The motivating bug was a real training run where three classes ("Adipose",
 * "SmoothMuscle", "Muscularis mucosae") were never learned because the random
 * split sent every image containing them entirely to val (or entirely to
 * train). These tests exercise the canonical shape of that scenario.
 */
public class ImageClassCoverageSplitterTest {

    private static ImageInput<String> img(String name, Object... kv) {
        Map<String, Double> areas = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) {
            areas.put((String) kv[i], ((Number) kv[i + 1]).doubleValue());
        }
        return new ImageInput<>(name, areas);
    }

    private static long countRole(SplitResult<String> result, Role role) {
        return result.assignments().stream().filter(a -> a.role() == role).count();
    }

    @Test
    public void everyCoverableClassEndsUpInBothSplits_userScenario() {
        // 11 images, each carrying a different mix of classes (mimics the real
        // multi-tissue project where the naive split stranded entire classes).
        // valFraction=0.4 -> targetVal=4, large enough for the 7 coverable
        // classes to fit (with overlap) on the val side.
        List<ImageInput<String>> inputs = List.of(
                img("img1_colon", "Mucosa", 5e5, "SmoothMuscle", 3e5),
                img("img2_colon", "Mucosa", 4e5, "SmoothMuscle", 2e5),
                img("img3_adipose", "Adipose", 8e5, "Mucosa", 1e5),
                img("img4_adipose", "Adipose", 6e5, "Fat", 2e5),
                img("img5_skin", "Epidermis", 5e5, "Dermis", 4e5),
                img("img6_skin", "Epidermis", 4e5, "Dermis", 3e5),
                img("img7_mm", "Muscularis mucosae", 3e5, "Mucosa", 2e5),
                img("img8_mm", "Muscularis mucosae", 3e5, "SmoothMuscle", 1e5),
                img("img9_mixed", "Mucosa", 4e5, "Epidermis", 2e5),
                img("img10_mixed", "Dermis", 3e5, "Fat", 2e5),
                img("img11_mixed", "Adipose", 2e5, "SmoothMuscle", 2e5));

        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.4, 42L);

        assertTrue(countRole(result, Role.TRAIN) >= 1);
        assertTrue(countRole(result, Role.VAL) >= 1);

        // Every coverable class (>=2 images) must appear in BOTH splits.
        for (var cc : result.coverage().values()) {
            if (cc.imagesContaining() >= 2) {
                assertTrue(cc.inTrain() >= 1, "Coverable class '" + cc.className() + "' missing from train: " + cc);
                assertTrue(cc.inVal() >= 1, "Coverable class '" + cc.className() + "' missing from val: " + cc);
            }
        }
    }

    @Test
    public void doesNotSuggestImpossibleSplitPercentage() {
        // 4 images, 5 coverable classes. Earlier versions of the splitter
        // emitted "raise to 125%" because the formula was coverable/n. The
        // warning must never suggest a >100% (or 3-digit) split.
        List<ImageInput<String>> inputs = List.of(
                img("a", "A", 100.0, "B", 100.0),
                img("b", "C", 100.0, "D", 100.0),
                img("c", "E", 100.0, "A", 100.0),
                img("d", "B", 100.0, "C", 100.0, "D", 100.0, "E", 100.0));
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.25, 1L);
        for (String w : result.warnings()) {
            assertTrue(!w.matches(".*\\b1\\d\\d%.*"), "Warning suggested >=100% split: " + w);
        }
    }

    @Test
    public void doesNotWarnWhenSplitActuallyCoversEveryClass() {
        // Classes co-occur heavily, so even a small val set can cover all of
        // them. Earlier versions warned eagerly based on targetVal vs class
        // count regardless of actual coverage -- this test guards against
        // that regression.
        List<ImageInput<String>> inputs = List.of(
                img("img1", "tumor", 100.0, "mucosa", 100.0, "muscle", 100.0, "fat", 100.0),
                img("img2", "tumor", 100.0, "mucosa", 100.0, "muscle", 100.0, "fat", 100.0),
                img("img3", "tumor", 100.0, "mucosa", 100.0, "muscle", 100.0, "fat", 100.0),
                img("img4", "tumor", 100.0, "mucosa", 100.0, "muscle", 100.0, "fat", 100.0),
                img("img5", "tumor", 100.0, "mucosa", 100.0, "muscle", 100.0, "fat", 100.0));
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.2, 0L);
        // 4 classes, val=1 image, every image carries all classes -> full coverage.
        assertTrue(
                result.warnings().stream().noneMatch(w -> w.contains("could not fit")),
                "Should not warn about uncovered classes when split is fully covering: " + result.warnings());
    }

    @Test
    public void warnsWhenClassIsRareByProjectPercentage() {
        // 'adipose' is in 4 of 30 images (13%) -- under the 15% threshold,
        // but at/above the 4-slide limited-data floor. The "rare" warning
        // should fire ('noisy between epochs') without the limited-data
        // warning ('excluded from best-epoch') -- they're distinct branches.
        List<ImageInput<String>> inputs = new ArrayList<>();
        for (int i = 0; i < 26; i++) {
            inputs.add(img("img" + i, "mucosa", 100.0, "muscle", 100.0));
        }
        inputs.add(img("imgA", "adipose", 100.0, "mucosa", 100.0));
        inputs.add(img("imgB", "adipose", 100.0, "muscle", 100.0));
        inputs.add(img("imgC", "adipose", 100.0, "mucosa", 100.0));
        inputs.add(img("imgD", "adipose", 100.0, "muscle", 100.0));
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.25, 0L);
        assertTrue(
                result.warnings().stream().anyMatch(w -> w.contains("adipose") && w.contains("rare")),
                "Expected a rare-class warning for 'adipose', got: " + result.warnings());
        assertTrue(
                result.warnings().stream().noneMatch(w -> w.contains("adipose") && w.contains("limited-data")
                        || (w.contains("adipose") && w.contains("only") && w.contains("source slides"))),
                "Should NOT also flag adipose as limited-data at 4 slides, got: " + result.warnings());
        assertTrue(
                !result.limitedDataClasses().contains("adipose"),
                "adipose should not be in limitedDataClasses with 4 source slides");
    }

    @Test
    public void doesNotWarnRareWhenSmallProjectKeepsPercentageHigh() {
        // 'adipose' is in 2 of 8 images (25%) -- above the 15% rare-percentage
        // threshold so no rare-class warning fires, but BELOW the 4-slide
        // limited-data floor so the limited-data warning DOES fire. This
        // test asserts both: the no-rare guard, plus the limited-data
        // detection (since 2 slides is too thin for the val signal).
        List<ImageInput<String>> inputs = List.of(
                img("img1", "mucosa", 100.0, "muscle", 100.0),
                img("img2", "mucosa", 100.0, "muscle", 100.0),
                img("img3", "mucosa", 100.0, "muscle", 100.0),
                img("img4", "mucosa", 100.0, "muscle", 100.0),
                img("img5", "mucosa", 100.0, "muscle", 100.0),
                img("img6", "mucosa", 100.0, "muscle", 100.0),
                img("img7", "adipose", 100.0, "mucosa", 100.0),
                img("img8", "adipose", 100.0, "muscle", 100.0));
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.25, 0L);
        assertTrue(
                result.warnings().stream().noneMatch(w -> w.contains("adipose") && w.contains("rare")),
                "Should not warn about adipose at 25% project share, got: " + result.warnings());
        assertTrue(
                result.limitedDataClasses().contains("adipose"),
                "adipose (2/8 slides) should be flagged limited-data: " + result.limitedDataClasses());
    }

    @Test
    public void limitedDataClassesPopulatedAtBoundary() {
        // The 'mucosa' class is in all 8 images; 'rare3' is in 3 (< floor=4
        // limited); 'enough4' is in 4 (== floor, NOT limited).
        List<ImageInput<String>> inputs = List.of(
                img("img1", "mucosa", 100.0, "rare3", 100.0),
                img("img2", "mucosa", 100.0, "rare3", 100.0),
                img("img3", "mucosa", 100.0, "rare3", 100.0),
                img("img4", "mucosa", 100.0, "enough4", 100.0),
                img("img5", "mucosa", 100.0, "enough4", 100.0),
                img("img6", "mucosa", 100.0, "enough4", 100.0),
                img("img7", "mucosa", 100.0, "enough4", 100.0),
                img("img8", "mucosa", 100.0));
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.25, 0L);
        assertTrue(result.limitedDataClasses().contains("rare3"),
                "rare3 (3 slides) should be limited-data: " + result.limitedDataClasses());
        assertTrue(!result.limitedDataClasses().contains("enough4"),
                "enough4 (4 slides) should NOT be limited-data: " + result.limitedDataClasses());
        assertTrue(!result.limitedDataClasses().contains("mucosa"),
                "mucosa (8 slides) should NOT be limited-data: " + result.limitedDataClasses());
    }

    @Test
    public void detectLimitedDataClassesStaticHelperMatches() {
        // The static helper must produce the same set as split() does --
        // training launch uses the helper when the user assigned roles
        // manually and we never ran a full split.
        List<ImageInput<String>> inputs = List.of(
                img("img1", "common", 100.0, "rare", 100.0),
                img("img2", "common", 100.0, "rare", 100.0),
                img("img3", "common", 100.0),
                img("img4", "common", 100.0),
                img("img5", "common", 100.0));
        var fromHelper = ImageClassCoverageSplitter.detectLimitedDataClasses(inputs);
        var fromSplit = ImageClassCoverageSplitter.split(inputs, 0.4, 0L).limitedDataClasses();
        assertEquals(fromHelper, fromSplit, "Helper and split() must agree on limited-data set");
        assertTrue(fromHelper.contains("rare"), "rare (2 slides) should be limited-data");
        assertTrue(!fromHelper.contains("common"), "common (5 slides) should NOT be limited-data");
    }

    @Test
    public void warnsWhenSplitFailsToCoverEveryClass() {
        // 4 images, 2 val slots, but the val side cannot fit one image with
        // class C and one with class D simultaneously -- val=2 includes at
        // most 2 distinct class pairs. Classes B and D are in disjoint
        // images, so a 2-image val can cover only some of {A,B,C,D}.
        List<ImageInput<String>> inputs = List.of(
                img("img1", "A", 100.0, "B", 100.0),
                img("img2", "A", 100.0, "B", 100.0),
                img("img3", "C", 100.0, "D", 100.0),
                img("img4", "C", 100.0, "D", 100.0));
        // valFraction=0.25 -> targetVal=1, which can hold only one of {AB,CD},
        // so coverage will fail for two of the four classes.
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.25, 7L);
        assertTrue(
                result.warnings().stream().anyMatch(w -> w.contains("could not fit")),
                "Expected an 'uncovered classes' warning, got: " + result.warnings());
    }

    @Test
    public void singleImageClassesAreWarnedButNotErrored() {
        // 'Rare' only exists in one image -- impossible to put on both sides.
        List<ImageInput<String>> inputs = List.of(
                img("a", "Common", 100.0, "Rare", 50.0),
                img("b", "Common", 100.0),
                img("c", "Common", 100.0),
                img("d", "Common", 100.0));
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.25, 7L);
        assertNotNull(result);
        assertTrue(
                result.warnings().stream().anyMatch(w -> w.contains("Rare")),
                "Expected a warning about single-image class 'Rare', got: " + result.warnings());
    }

    @Test
    public void singleImageInputIsAssignedToTrain() {
        List<ImageInput<String>> inputs = List.of(img("solo", "X", 100.0));
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.2, 1L);
        assertEquals(1, result.assignments().size());
        assertEquals(Role.TRAIN, result.assignments().get(0).role());
    }

    @Test
    public void emptyInputProducesEmptyResult() {
        SplitResult<String> result = ImageClassCoverageSplitter.split(new ArrayList<>(), 0.2, 0L);
        assertTrue(result.assignments().isEmpty());
        assertTrue(result.coverage().isEmpty());
    }

    @Test
    public void seedIsReproducible() {
        List<ImageInput<String>> inputs = List.of(
                img("a", "X", 100.0, "Y", 100.0),
                img("b", "X", 100.0, "Y", 100.0),
                img("c", "X", 100.0, "Y", 100.0),
                img("d", "X", 100.0, "Y", 100.0));
        SplitResult<String> r1 = ImageClassCoverageSplitter.split(inputs, 0.5, 12345L);
        SplitResult<String> r2 = ImageClassCoverageSplitter.split(inputs, 0.5, 12345L);
        for (int i = 0; i < r1.assignments().size(); i++) {
            ImageAssignment<String> a1 = r1.assignments().get(i);
            ImageAssignment<String> a2 = r2.assignments().get(i);
            assertEquals(a1.handle(), a2.handle());
            assertEquals(a1.role(), a2.role(), "Same seed must produce same split for image " + a1.handle());
        }
    }

    @Test
    public void respectsValFractionWhenCoverageAllows() {
        // 10 images, all carrying the same 3 classes -- any split covers all
        // classes, so the splitter is free to honor the target val fraction.
        List<ImageInput<String>> inputs = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            inputs.add(img("img" + i, "A", 100.0, "B", 100.0, "C", 100.0));
        }
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.3, 99L);
        long val = countRole(result, Role.VAL);
        assertEquals(3, val, "Expected 30% of 10 = 3 val images");
    }
}
