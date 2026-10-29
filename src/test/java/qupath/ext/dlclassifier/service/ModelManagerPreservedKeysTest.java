package qupath.ext.dlclassifier.service;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * saveClassifier REPLACES the metadata.json the training script wrote, keeping only
 * the keys it knows. The resolution contract was lost on every run that way until it
 * was added to the list; this pins the calibration record (training_hardware) and
 * checks every key the Python side assigns after building the dict is either listed
 * or one Java writes itself.
 */
class ModelManagerPreservedKeysTest {

    @Test
    void trainingHardwareSurvivesTheRewrite() {
        assertTrue(ModelManager.PYTHON_PRESERVED_KEYS.contains("training_hardware"));
    }

    @Test
    void everyKeyPythonAddsLaterIsPreserved() throws IOException {
        Path py = Path.of("python_server/dlclassifier_server/services/training_service.py");
        String src = Files.readString(py, StandardCharsets.UTF_8);
        Matcher m = Pattern.compile("metadata\\[\"([a-z_]+)\"\\]\\s*=").matcher(src);
        int seen = 0;
        while (m.find()) {
            seen++;
            String key = m.group(1);
            assertTrue(
                    ModelManager.PYTHON_PRESERVED_KEYS.contains(key),
                    "training_service.py writes metadata[\"" + key + "\"], which saveClassifier would drop");
        }
        assertTrue(seen > 0, "pattern no longer matches training_service.py; update this test");
    }
}
