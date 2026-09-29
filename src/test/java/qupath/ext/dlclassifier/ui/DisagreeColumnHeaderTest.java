package qupath.ext.dlclassifier.ui;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * The disagreement-pixel column counts only disagreements at or above the
 * confidence threshold, while the Disagree% column beside it is unconditional.
 *
 * <p>With the threshold at 97% a row legitimately reads "19.7%" beside "0".
 * That looked like a defect in a workshop screenshot until the column's
 * meaning was chased through the code, so the threshold now travels in the
 * header.
 */
class DisagreeColumnHeaderTest {

    @Test
    void theHeaderNamesTheThresholdItCountsAt() {
        assertThat(TrainingAreaIssuesDialog.disagreeColumnHeader(0.97)).isEqualTo("Disagree px @97%");
    }

    @Test
    void itTracksTheSliderAcrossItsWholeRange() {
        // The slider runs 0.50 to 0.99.
        assertThat(TrainingAreaIssuesDialog.disagreeColumnHeader(0.50)).isEqualTo("Disagree px @50%");
        assertThat(TrainingAreaIssuesDialog.disagreeColumnHeader(0.99)).isEqualTo("Disagree px @99%");
    }

    @Test
    void itRoundsRatherThanTruncating() {
        // 0.975 must not render as 97%, which would name a threshold that is
        // not the one the counts were taken at.
        assertThat(TrainingAreaIssuesDialog.disagreeColumnHeader(0.975)).isEqualTo("Disagree px @98%");
    }

    @Test
    void theEdgesDoNotProduceNonsense() {
        assertThat(TrainingAreaIssuesDialog.disagreeColumnHeader(0.0)).isEqualTo("Disagree px @0%");
        assertThat(TrainingAreaIssuesDialog.disagreeColumnHeader(1.0)).isEqualTo("Disagree px @100%");
    }

    @Test
    void theHeaderStaysAscii() {
        // It renders on Windows under cp1252; a stray Unicode comparator here
        // is the class of thing that has crashed workflows in this project.
        String h = TrainingAreaIssuesDialog.disagreeColumnHeader(0.97);
        assertThat(h.chars().allMatch(c -> c < 128)).isTrue();
    }
}
