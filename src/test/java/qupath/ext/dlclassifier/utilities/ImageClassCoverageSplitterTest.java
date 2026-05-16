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
    public void warnsWhenValFractionTooSmallToFitAllClasses() {
        // 7 coverable classes but only 2 val slots -- no permutation can fit
        // them all on the val side. Splitter must surface this as a warning
        // so the user knows to raise the val fraction.
        List<ImageInput<String>> inputs = List.of(
                img("img1", "A", 100.0, "B", 100.0),
                img("img2", "A", 100.0, "B", 100.0),
                img("img3", "C", 100.0, "D", 100.0),
                img("img4", "C", 100.0, "D", 100.0),
                img("img5", "E", 100.0, "F", 100.0),
                img("img6", "E", 100.0, "F", 100.0),
                img("img7", "G", 100.0),
                img("img8", "G", 100.0),
                img("img9", "A", 100.0, "C", 100.0),
                img("img10", "B", 100.0, "D", 100.0),
                img("img11", "E", 100.0, "G", 100.0));

        // 20% of 11 = 2 val slots, but there are 7 coverable classes.
        SplitResult<String> result = ImageClassCoverageSplitter.split(inputs, 0.2, 42L);

        // Splitter still produces a split (best-effort), but warns the user.
        assertTrue(
                result.warnings().stream().anyMatch(w -> w.contains("Raise the validation split")),
                "Expected a warning suggesting a higher val fraction, got: " + result.warnings());
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
