package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.model.ClassifierMetadata;

/**
 * Guards which classes become objects (GitHub issue #24).
 *
 * <p>Object generation used to start its loop at class index 1, treating index
 * 0 as background. Nothing in the pipeline makes that true -- unannotated
 * pixels train as {@code ignore_index=255}, never as class 0 -- so the
 * alphabetically-first class silently disappeared. A "Gland, Stroma"
 * classifier produced only Stroma, and users invented throwaway background
 * classes to work around it. The MEASUREMENTS output never had the rule and
 * reported every class from 0, so the two outputs disagreed.
 *
 * <p>The only exclusion is QuPath's own: a class whose name ends with
 * {@code '*'} is ignored.
 */
class OutputGeneratorClassSelectionTest {

    private static ClassifierMetadata.ClassInfo cls(int index, String name) {
        return new ClassifierMetadata.ClassInfo(index, name, "#ffffff");
    }

    @Test
    @DisplayName("the first class is not treated as background")
    void firstClassIsNotBackground() {
        // The exact configuration reported in issue #24: two real classes,
        // no background. Gland is index 0 and used to vanish.
        List<ClassifierMetadata.ClassInfo> classes = List.of(cls(0, "Gland"), cls(1, "Stroma"));

        List<Integer> eligible = OutputGenerator.classIndicesForObjects(classes, 2);

        assertEquals(List.of(0, 1), eligible, "both classes must generate objects");
    }

    @Test
    @DisplayName("classes ending in '*' are excluded, wherever they sit")
    void ignoredClassesAreExcluded() {
        // Trailing '*' is QuPath's convention and is how a user opts a class
        // out -- including one at index 0, which is how you ask for a
        // background without needing the concept of one.
        List<ClassifierMetadata.ClassInfo> classes =
                List.of(cls(0, "BG*"), cls(1, "Fibrin"), cls(2, "Ignore*"), cls(3, "Tissue"));

        List<Integer> eligible = OutputGenerator.classIndicesForObjects(classes, 4);

        assertEquals(List.of(1, 3), eligible);
    }

    @Test
    @DisplayName("a plain-named background class is NOT excluded by guesswork")
    void plainBackgroundNameIsStillGenerated() {
        // Deliberate: "BG" without the asterisk is an ordinary class name. We
        // do not pattern-match names to decide what a background is, because
        // someone's real class may legitimately be called BG or Background.
        List<ClassifierMetadata.ClassInfo> classes = List.of(cls(0, "BG"), cls(1, "Tumor"), cls(2, "Stroma"));

        List<Integer> eligible = OutputGenerator.classIndicesForObjects(classes, 3);

        assertEquals(List.of(0, 1, 2), eligible);
    }

    @Test
    @DisplayName("metadata shorter than the map falls back to synthetic names")
    void handlesMetadataShorterThanClassCount() {
        // Defensive: a model whose metadata lost a class entry must not throw
        // or drop channels it still predicts.
        List<Integer> eligible = OutputGenerator.classIndicesForObjects(List.of(cls(0, "Tumor")), 3);

        assertEquals(List.of(0, 1, 2), eligible);
    }

    @Test
    @DisplayName("null metadata does not throw")
    void handlesNullMetadata() {
        List<Integer> eligible = OutputGenerator.classIndicesForObjects(null, 2);

        assertEquals(List.of(0, 1), eligible);
    }

    @Test
    @DisplayName("every class can be excluded, yielding no objects")
    void allIgnoredYieldsNothing() {
        List<ClassifierMetadata.ClassInfo> classes = List.of(cls(0, "A*"), cls(1, "B*"));

        assertTrue(OutputGenerator.classIndicesForObjects(classes, 2).isEmpty());
    }
}
