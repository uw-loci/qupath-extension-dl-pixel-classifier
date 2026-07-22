package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import qupath.ui.logviewer.api.manager.LoggerManager;

/**
 * Verifies that {@link SessionLogBuffer} captures the live log via QuPath's
 * logging framework -- the mechanism that lets the bug reporter attach the log
 * without a log file on disk.
 */
class SessionLogBufferTest {

    @Test
    void capturesLiveLogAfterInit() {
        // The buffer only works if a logging framework is discoverable; if this
        // assumption breaks, the whole feature is inert, so assert it explicitly.
        assertThat(LoggerManager.getCurrentLoggerManager())
                .as("a logging framework must be on the classpath for live capture")
                .isPresent();

        SessionLogBuffer.init();

        String marker = "SESSIONLOGBUFFER_MARKER_" + System.nanoTime();
        LoggerFactory.getLogger(SessionLogBufferTest.class).error(marker);

        assertThat(SessionLogBuffer.hasContent()).isTrue();
        assertThat(SessionLogBuffer.getText()).contains(marker);
    }
}
