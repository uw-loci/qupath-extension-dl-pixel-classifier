package qupath.ext.dlclassifier.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ClassifierNamesTest {

    @ParameterizedTest
    @ValueSource(
            strings = {
                "CMU-1 Tissue (ResNet-18)",
                "Collagen_Classifier_v1",
                "fly wing v2.3",
                "Nuclei, 20x, batch #4",
                "Tumour & stroma [draft]",
                "a"
            })
    void ordinaryNamesPass(String name) {
        assertNull(ClassifierNames.problemWith(name), name);
    }

    @ParameterizedTest
    @ValueSource(strings = {"a<b", "a>b", "a:b", "a\"b", "a/b", "a\\b", "a|b", "a?b", "a*b"})
    void fileSystemReservedCharactersAreRefusedWithTheListInTheMessage(String name) {
        String problem = ClassifierNames.problemWith(name);
        assertNotNull(problem, name);
        assertTrue(problem.contains("< > : \" / \\ | ? *"), problem);
    }

    @Test
    void emptyAndControlCharactersAreRefused() {
        assertEquals("Classifier name is required", ClassifierNames.problemWith(null));
        assertEquals("Classifier name is required", ClassifierNames.problemWith("   "));
        assertNotNull(ClassifierNames.problemWith("tab\there"));
        assertNotNull(ClassifierNames.problemWith("new\nline"));
    }

    @Test
    void folderStemMatchesTheTrainingWorkflowRule() {
        // TrainingWorkflow: name.toLowerCase().replaceAll("[^a-z0-9_-]", "_") + "_" + millis
        assertEquals("cmu-1_tissue__resnet-18_", ClassifierNames.toFolderStem("CMU-1 Tissue (ResNet-18)"));
        assertEquals("fly_wing_v2_3", ClassifierNames.toFolderStem("fly wing v2.3"));
    }
}
