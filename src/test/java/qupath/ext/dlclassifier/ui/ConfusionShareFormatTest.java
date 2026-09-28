package qupath.ext.dlclassifier.ui;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Training Area Issues exists to show where the model went wrong, so a real
 * error must never render as no error.
 *
 * <p>The per-tile "Worst Confusion" column printed {@code %.0f%%}, so any
 * confusion under half a percent of its ground-truth class showed as "0% of
 * GT" -- the same text a class with no errors at all would get. The matrix
 * had the same problem one decimal further out, printing "0.0%". A user
 * checking whether their deliberate annotation errors were caught would read
 * either as "not caught".
 */
class ConfusionShareFormatTest {

    @Test
    void aGenuineZeroReadsAsZero() {
        assertThat(TrainingAreaIssuesDialog.formatConfusionShare(0, 10_000)).isEqualTo("0%");
    }

    @Test
    void aTinyButRealErrorIsNotPrintedAsZero() {
        // 3 pixels in a million is 0.0003%, which every earlier format
        // rounded to zero.
        String s = TrainingAreaIssuesDialog.formatConfusionShare(3, 1_000_000);
        assertThat(s).isEqualTo("<0.1%");
        assertThat(s).isNotEqualTo("0%").isNotEqualTo("0.0%");
    }

    @Test
    void oneMisclassifiedPixelStillShows() {
        assertThat(TrainingAreaIssuesDialog.formatConfusionShare(1, Long.MAX_VALUE / 2))
                .isEqualTo("<0.1%");
    }

    @Test
    void ordinaryValuesKeepOneDecimal() {
        assertThat(TrainingAreaIssuesDialog.formatConfusionShare(1_234, 10_000)).isEqualTo("12.3%");
        assertThat(TrainingAreaIssuesDialog.formatConfusionShare(50, 10_000)).isEqualTo("0.5%");
    }

    @Test
    void theBoundaryOfTheBoundIsNotItselfRoundedAway() {
        // Exactly 0.1% must print as a number, not as the "<0.1%" bound.
        assertThat(TrainingAreaIssuesDialog.formatConfusionShare(10, 10_000)).isEqualTo("0.1%");
        assertThat(TrainingAreaIssuesDialog.formatConfusionShare(9, 10_000)).isEqualTo("<0.1%");
    }

    @Test
    void anAbsentClassSaysSoRatherThanClaimingPerfection() {
        // No ground-truth pixels is not the same as no errors, and 0/0 must
        // not become "0%".
        assertThat(TrainingAreaIssuesDialog.formatConfusionShare(0, 0)).isEqualTo("n/a");
    }

    @Test
    void theOutputStaysAscii() {
        for (long[] c : new long[][] {{0, 10}, {3, 1_000_000}, {1234, 10_000}, {0, 0}}) {
            String s = TrainingAreaIssuesDialog.formatConfusionShare(c[0], c[1]);
            assertThat(s.chars().allMatch(ch -> ch < 128)).as("%s", s).isTrue();
        }
    }
}
