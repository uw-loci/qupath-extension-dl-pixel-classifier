package qupath.ext.dlclassifier.service.ood;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Behavioural tests for {@link OutOfDistributionChecker}. Verifies the
 * threshold logic on the four metrics (mean, p1, p99, std) and the
 * defensive paths (null inputs, missing keys, degenerate std).
 */
class OutOfDistributionCheckerTest {

    private static Map<String, Double> stats(double p1, double p99, double mean, double std) {
        Map<String, Double> m = new HashMap<>();
        m.put("p1", p1);
        m.put("p99", p99);
        m.put("min", p1);
        m.put("max", p99);
        m.put("mean", mean);
        m.put("std", std);
        return m;
    }

    @Test
    void nullInputsProduceEmptyReport() {
        OodReport r = OutOfDistributionChecker.check(null, null, null, 3.0);
        assertNotNull(r);
        assertFalse(r.hasDeviation());
    }

    @Test
    void matchingStatsTriggerNoWarning() {
        // Image stats identical to training -- no deviation expected.
        Map<String, Double> training = stats(10, 240, 125, 40);
        Map<String, Double> image = stats(10, 240, 125, 40);
        OodReport r = OutOfDistributionChecker.check(List.of(training), List.of(image), List.of("R"), 3.0);
        assertFalse(r.hasDeviation(), "Identical stats should not trigger a warning");
    }

    @Test
    void meanShiftBeyondThresholdTriggersWarning() {
        // image mean is 5 sigma above training mean -- should fire.
        Map<String, Double> training = stats(0, 255, 128, 20);
        Map<String, Double> image = stats(0, 255, 128 + 5 * 20, 20);
        OodReport r = OutOfDistributionChecker.check(List.of(training), List.of(image), List.of("R"), 3.0);
        assertTrue(r.hasDeviation());
        assertEquals(1, r.getDeviations().size());
        assertEquals("mean", r.getDeviations().get(0).worstMetric());
    }

    @Test
    void contrastCollapseTriggersStdWarning() {
        // Image std is 10x smaller than training -- contrast collapse.
        Map<String, Double> training = stats(50, 200, 125, 30);
        Map<String, Double> image = stats(120, 130, 125, 3);
        OodReport r = OutOfDistributionChecker.check(List.of(training), List.of(image), List.of("R"), 3.0);
        assertTrue(r.hasDeviation());
        // Could be flagged via std OR via p1/p99 shift; both are valid signals.
        String metric = r.getDeviations().get(0).worstMetric();
        assertTrue(
                metric.equals("std") || metric.equals("p1") || metric.equals("p99"),
                "Expected std/p1/p99 but got " + metric);
    }

    @Test
    void smallMeanShiftDoesNotTrigger() {
        // Image mean is 1 sigma above training -- well below the 3-sigma cutoff.
        Map<String, Double> training = stats(50, 200, 125, 20);
        Map<String, Double> image = stats(50, 200, 145, 20);
        OodReport r = OutOfDistributionChecker.check(List.of(training), List.of(image), List.of("R"), 3.0);
        assertFalse(r.hasDeviation());
    }

    @Test
    void multipleChannelsReportedIndependently() {
        // Channel 0 fine, channel 1 way off in mean.
        Map<String, Double> ch0Train = stats(0, 255, 128, 25);
        Map<String, Double> ch0Img = stats(0, 255, 130, 25);
        Map<String, Double> ch1Train = stats(0, 255, 128, 25);
        Map<String, Double> ch1Img = stats(0, 255, 220, 25); // ~3.7 sigma
        OodReport r = OutOfDistributionChecker.check(
                List.of(ch0Train, ch1Train), List.of(ch0Img, ch1Img), List.of("R", "G"), 3.0);
        assertEquals(1, r.getDeviations().size(), "Only the offending channel should be reported");
        assertEquals(1, r.getDeviations().get(0).channelIndex());
        assertEquals("G", r.getDeviations().get(0).channelName());
    }

    @Test
    void degenerateTrainingStdFallsBackToMeanFraction() {
        // training std == 0 (uniform channel). A modest shift should still be flagged
        // via the abs(mean) * 0.01 fallback scale, not divide-by-zero.
        Map<String, Double> training = stats(100, 100, 100, 0);
        Map<String, Double> image = stats(100, 100, 200, 0);
        OodReport r = OutOfDistributionChecker.check(List.of(training), List.of(image), List.of("R"), 3.0);
        assertTrue(r.hasDeviation(), "Degenerate training std should still allow detection");
    }

    @Test
    void missingKeysFallBackGracefully() {
        // Training entry is missing 'std' -- should not NPE.
        Map<String, Double> training = new HashMap<>();
        training.put("mean", 100.0);
        training.put("p1", 0.0);
        training.put("p99", 255.0);
        Map<String, Double> image = stats(0, 255, 100, 20);
        OodReport r = OutOfDistributionChecker.check(List.of(training), List.of(image), List.of("R"), 3.0);
        // The training std falls back to 0, then to the abs(mean) * 0.01 scale.
        // Image std=20 vs training std=0 still triggers the std-ratio path.
        assertNotNull(r);
    }

    @Test
    void describeIsAsciiOnly() {
        Map<String, Double> training = stats(0, 255, 128, 20);
        Map<String, Double> image = stats(0, 255, 128 + 5 * 20, 20);
        OodReport r = OutOfDistributionChecker.check(List.of(training), List.of(image), List.of("Red"), 3.0);
        String desc = r.describe();
        for (int i = 0; i < desc.length(); i++) {
            int cp = desc.codePointAt(i);
            assertTrue(cp >= 0 && cp <= 127, "Non-ASCII char at index " + i + " (codepoint " + cp + "): " + desc);
        }
    }
}
