package qupath.ext.dlclassifier.service;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Overlay suspension is counted so overlapping training and pretraining jobs
 * keep the overlay off until the last one finishes. Drives the FX-free
 * bookkeeping directly; the JavaFX toolkit is never started.
 */
class OverlayServiceSuspendTest {

    private final OverlayService svc = OverlayService.getInstance();

    @AfterEach
    void drain() {
        while (svc.trainingActiveProperty().get()) {
            svc.markResumed();
        }
    }

    @Test
    void singleJobSuspendsAndResumes() {
        svc.markSuspended();
        assertTrue(svc.trainingActiveProperty().get());
        svc.markResumed();
        assertFalse(svc.trainingActiveProperty().get());
    }

    @Test
    void overlappingJobsStaySuspendedUntilTheLastEnds() {
        svc.markSuspended(); // training
        svc.markSuspended(); // pretraining started meanwhile
        svc.markResumed(); // training finishes first
        assertTrue(svc.trainingActiveProperty().get(), "pretraining is still running");
        svc.markResumed();
        assertFalse(svc.trainingActiveProperty().get());
    }

    @Test
    void unmatchedResumeDoesNotLetTheCountGoNegative() {
        svc.markResumed();
        assertFalse(svc.trainingActiveProperty().get());
        svc.markSuspended();
        assertTrue(svc.trainingActiveProperty().get(), "a stray resume must not pre-cancel the next suspend");
        svc.markResumed();
        assertFalse(svc.trainingActiveProperty().get());
    }
}
