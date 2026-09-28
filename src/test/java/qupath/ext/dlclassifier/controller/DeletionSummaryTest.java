package qupath.ext.dlclassifier.controller;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Deleting several classifiers at once can partly succeed: a model directory
 * locked by another process on Windows, or one already removed outside
 * QuPath. Reporting only "deleted 3 of 5" would leave the user to work out
 * which two survived by reading the list, so the message names them.
 */
class DeletionSummaryTest {

    @Test
    void oneClassifierIsNamedRatherThanCounted() {
        assertThat(ModelManagementWorkflow.summariseDeletion(List.of("tissue-v2"), List.of()))
                .isEqualTo("Deleted 'tissue-v2'");
    }

    @Test
    void severalAreCounted() {
        assertThat(ModelManagementWorkflow.summariseDeletion(List.of("a", "b", "c"), List.of()))
                .isEqualTo("Deleted 3 classifiers");
    }

    @Test
    void aPartialFailureNamesWhatSurvived() {
        String msg = ModelManagementWorkflow.summariseDeletion(List.of("a", "b"), List.of("locked-model"));
        assertThat(msg).contains("Deleted 2").contains("locked-model");
    }

    @Test
    void aTotalFailureDoesNotClaimAnythingWasDeleted() {
        String msg = ModelManagementWorkflow.summariseDeletion(List.of(), List.of("x", "y"));
        assertThat(msg).doesNotContain("Deleted ");
        assertThat(msg).contains("x").contains("y");
    }

    @Test
    void everyFailureIsNamedNoMatterHowMany() {
        List<String> failed = List.of("m1", "m2", "m3", "m4");
        String msg = ModelManagementWorkflow.summariseDeletion(List.of("ok"), failed);
        assertThat(msg).contains(failed.toArray(new String[0]));
    }

    @Test
    void theMessageStaysAscii() {
        // This reaches a notification on Windows, where cp1252 turns a stray
        // Unicode character into a crash rather than a cosmetic problem.
        String msg = ModelManagementWorkflow.summariseDeletion(List.of("a", "b"), List.of("c"));
        assertThat(msg.chars().allMatch(c -> c < 128)).isTrue();
    }
}
