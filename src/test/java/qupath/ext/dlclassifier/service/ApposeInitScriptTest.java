package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import org.junit.jupiter.api.Test;

/**
 * Pins the shape of the Appose worker's init script.
 * <p>
 * Two invariants live here, and both fail silently rather than loudly.
 * <ul>
 *   <li>{@code import numpy} must stay the FIRST line. NumPy has to be imported
 *       before Appose's stdin reader thread starts or the worker deadlocks on
 *       Windows (numpy#24290 / Appose#23) -- with no error, just a hang.</li>
 *   <li>The protocol/stdio separation must appear ONLY when DataLoader workers
 *       are enabled. It repoints the worker's fd 0 / fd 1, so switching it on by
 *       default would change stdio for every user to fix a problem only worker
 *       processes have.</li>
 *   <li>{@code _DLC_HOST_PID} must carry THIS JVM's PID. The worker cannot
 *       derive it: its process tree is {@code java -> pixi -> python}, so
 *       {@code os.getppid()} returns pixi. A wrong value here is a watcher
 *       that kills a healthy worker, or one that never kills an orphan
 *       (GitHub issue #30).</li>
 * </ul>
 */
class ApposeInitScriptTest {

    @Test
    void workersOffLeavesTheWorkerStdioAlone() throws IOException {
        String script = ApposeService.getInstance().buildInitScript(false);

        // init_services.py both names the flag in a comment and tests for it,
        // so only a LINE that assigns it means the separation is switched on.
        assertThat(setsTheFlag(script)).isFalse();
        assertThat(script.lines().findFirst()).hasValue("import numpy");
    }

    @Test
    void workersOnEnablesTheProtocolSeparation() throws IOException {
        String script = ApposeService.getInstance().buildInitScript(true);

        assertThat(setsTheFlag(script)).isTrue();
        assertThat(script.lines().findFirst()).hasValue("import numpy");
        // The flag is only read by the guard inside init_services.py, so it has
        // to be set before that file's body runs.
        assertThat(script.indexOf("_DLC_PROTOCOL_FD_SEPARATION = True"))
                .isLessThan(script.indexOf("dlclassifier.appose"));
    }

    @Test
    void theHostPidIsThisJvmsOwnPid() throws IOException {
        String script = ApposeService.getInstance().buildInitScript(false);

        // Assert the IDENTITY, not a literal: the only correct value is the
        // PID of the process that wrote the script. Anything else -- pixi, the
        // worker, a stale constant -- points the watcher at the wrong process.
        assertThat(hostPidIn(script)).hasValue(ProcessHandle.current().pid());
    }

    @Test
    void theHostPidIsSetWhicheverWayTheWorkerStarts() throws IOException {
        // The separation flag is conditional; the host PID is not. A worker
        // started for a run with DataLoader workers needs the watcher just as
        // much as one started without.
        assertThat(hostPidIn(ApposeService.getInstance().buildInitScript(true)))
                .hasValue(ProcessHandle.current().pid());
    }

    @Test
    void theHostPidIsSetBeforeTheInitBodyRuns() throws IOException {
        String script = ApposeService.getInstance().buildInitScript(false);

        // Read via globals() at import time, so assigning it later would leave
        // the watcher silently falling back to getppid().
        assertThat(script.indexOf("_DLC_HOST_PID = ")).isLessThan(script.indexOf("dlclassifier.appose"));
    }

    @Test
    void thePythonSideReadsTheHostPidAndStillHasAFallback() throws IOException {
        // Guards against the PID being plumbed while the Python that consumes
        // it is edited away, and against the fallback being dropped (which
        // would leave a hand-run script with no watcher at all).
        String script = ApposeService.getInstance().buildInitScript(false);

        assertThat(script).contains("globals().get(\"_DLC_HOST_PID\")");
        assertThat(script).contains("host_pid = os.getppid()");
    }

    @Test
    void theExitPathWritesItsReasonToAFileDescriptor() throws IOException {
        // os._exit() flushes nothing, so a logger call alone can be lost
        // between emit and death -- which is how issue #30 arrived as
        // "exit code 1" with no reason attached. The os.write must come
        // FIRST, because after os._exit there is no "later".
        String script = ApposeService.getInstance().buildInitScript(false);

        assertThat(script).contains("os.write(2, ");
        assertThat(script.indexOf("os.write(2, ")).isLessThan(script.indexOf("os._exit(1)"));
    }

    /** The {@code _DLC_HOST_PID} the script assigns, if it assigns one. */
    private static java.util.OptionalLong hostPidIn(String script) {
        return script.lines()
                .map(String::strip)
                .filter(line -> line.startsWith("_DLC_HOST_PID = "))
                .mapToLong(line -> Long.parseLong(
                        line.substring("_DLC_HOST_PID = ".length()).strip()))
                .findFirst();
    }

    /** True when a line of the script actually assigns the flag. */
    private static boolean setsTheFlag(String script) {
        return script.lines().anyMatch(line -> line.strip().equals("_DLC_PROTOCOL_FD_SEPARATION = True"));
    }

    @Test
    void theGuardedBlockIsPresentInTheBundledInitScript() throws IOException {
        // Guards against the flag being plumbed while the Python side that acts
        // on it is edited away -- which would leave workers hanging again.
        String script = ApposeService.getInstance().buildInitScript(true);

        assertThat(script).contains("globals().get(\"_DLC_PROTOCOL_FD_SEPARATION\", False)");
        assertThat(script).contains("os.dup2");
        assertThat(script).contains("sys.stdin = _protocol_in");
    }
}
