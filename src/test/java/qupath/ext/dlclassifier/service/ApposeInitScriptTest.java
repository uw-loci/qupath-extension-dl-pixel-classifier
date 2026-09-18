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
