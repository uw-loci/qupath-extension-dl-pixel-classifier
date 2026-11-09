package qupath.ext.dlclassifier.model;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * Every in-memory cache mode the Train dialog offers reaches the training script.
 * "bounded" used to fall back to "auto", so the dialog's bounded option, its RAM-share
 * preference and its confirmation never took effect.
 */
class TrainingConfigInMemoryModeTest {

    private static String mode(String m) {
        return TrainingConfig.builder().inMemoryDataset(m).build().getInMemoryDataset();
    }

    @Test
    void theDialogsModesSurviveAndAnythingElseIsAuto() {
        for (String m : new String[] {"auto", "on", "bounded", "off"}) {
            assertEquals(m, mode(m), m);
        }
        assertEquals("auto", mode("Bounded"));
        assertEquals("auto", mode("nonsense"));
        assertEquals("auto", mode(null));
    }
}
