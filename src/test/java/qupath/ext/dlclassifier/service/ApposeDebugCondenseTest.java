package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Appose hands the debug listener the full request JSON for every task,
 * script body included. A 1000-tile overlay therefore wrote north of 10 MB
 * of near-identical text into the QuPath log and buried everything else.
 *
 * <p>The rule under test: the first of each shape is kept whole, at INFO, so
 * a run is fully documented from its opening lines rather than only after it
 * finishes. Everything that repeats after that is compacted and moved to
 * DEBUG. Failures are never touched, whether they repeat or not.
 */
class ApposeDebugCondenseTest {

    private static String request(String task, String script) {
        return "{\"task\":\"" + task + "\",\"requestType\":\"EXECUTE\",\"inputs\":{\"num_channels\":3},\"script\":\""
                + script + "\"}";
    }

    @Test
    void theFirstSightingOfAScriptIsKeptWhole() {
        String msg = request("t1", "import numpy as np; run()" + System.nanoTime());
        ApposeService.DebugLine line = ApposeService.condenseDebugLine(msg);
        assertThat(line.text()).isEqualTo(msg);
        assertThat(line.routine()).isFalse();
    }

    @Test
    void repeatsOfThatScriptAreCompactedAndRoutine() {
        // Roughly the size of the real per-tile inference script, which is
        // what made this worth doing: ~11 KB per tile, 1000 tiles per overlay.
        String script = "the same script body " + System.nanoTime() + " x".repeat(5000);
        String repeated = request("t2", script);
        ApposeService.condenseDebugLine(request("t1", script));

        ApposeService.DebugLine line = ApposeService.condenseDebugLine(repeated);
        assertThat(line.routine()).isTrue();
        // The task id survives: it is the only part of a repeat anyone can
        // correlate against a COMPLETION or a failure.
        assertThat(line.text()).contains("t2");
        assertThat(line.text()).doesNotContain("the same script body");
        assertThat(line.text().length()).isLessThan(repeated.length() / 50);
    }

    @Test
    void aFailureIsNeverElidedEvenWhenItRepeats() {
        String script = "failing script " + System.nanoTime();
        ApposeService.condenseDebugLine(request("t1", script));

        String failure = "{\"task\":\"t2\",\"responseType\":\"FAILURE\",\"script\":\"" + script + "\"}";
        ApposeService.DebugLine line = ApposeService.condenseDebugLine(failure);
        assertThat(line.text()).isEqualTo(failure);
        assertThat(line.routine()).isFalse();
    }

    @Test
    void theFirstResponseOfEachKindIsKeptAndLaterOnesAreNot() {
        String kind = "COMPLETION_" + System.nanoTime();
        String first = "{\"task\":\"t1\",\"responseType\":\"" + kind + "\",\"outputs\":{\"shm\":\"wnsm_aaa\"}}";
        String second = "{\"task\":\"t2\",\"responseType\":\"" + kind + "\",\"outputs\":{\"shm\":\"wnsm_bbb\"}}";

        assertThat(ApposeService.condenseDebugLine(first).routine()).isFalse();

        ApposeService.DebugLine line = ApposeService.condenseDebugLine(second);
        assertThat(line.routine()).isTrue();
        assertThat(line.text()).contains("t2").doesNotContain("wnsm_bbb");
    }

    @Test
    void workerLogLinesAreLeftAlone() {
        // The Python logger's own output carries epoch progress and warnings.
        // It has no responseType and no script, and must never be demoted.
        String workerLine = "2026-09-27 01:19:38,705 - dlclassifier_server - INFO - Loading static ONNX model";
        ApposeService.DebugLine line = ApposeService.condenseDebugLine(workerLine);
        assertThat(line.text()).isEqualTo(workerLine);
        assertThat(line.routine()).isFalse();
    }

    @Test
    void anEmptyOrNullMessageIsPassedThrough() {
        assertThat(ApposeService.condenseDebugLine(null).text()).isNull();
        assertThat(ApposeService.condenseDebugLine("").text()).isEmpty();
    }
}
