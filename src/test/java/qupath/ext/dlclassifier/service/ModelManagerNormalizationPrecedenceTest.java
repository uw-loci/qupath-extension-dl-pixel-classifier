package qupath.ext.dlclassifier.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.dlclassifier.model.ClassifierMetadata;

/**
 * Pins which metadata block wins when both record {@code per_channel}.
 *
 * <p>{@code input_config.normalization} is written by the Python training
 * service from the same Appose config training actually normalized with, so it
 * is authoritative. {@code channel_config.normalization} comes from
 * {@code ClassifierMetadata.toMap()} and is only as good as what the builder
 * received.
 *
 * <p>The parse loop stops at the first parent carrying the flag. It checked
 * {@code channel_config} first, which shadowed the truthful value for every
 * model saved between 2026-06-17 -- when {@code toMap()} began emitting the
 * block from a builder default that was always {@code false} -- and 2026-09-22,
 * when the builders began setting it for real. Those models trained per-channel
 * and were applied joint, on the overlay, AdaBN, scripting and retrain paths.
 *
 * <p>Nothing pinned the ordering, so the regression was invisible. These tests
 * pin it, one per vintage of model that exists in the wild.
 */
class ModelManagerNormalizationPrecedenceTest {

    // loadMetadata dereferences "architecture" without a null check, so every
    // fixture needs one even when the test is about normalization.
    private static final String HEAD =
            "{\"id\":\"m\",\"name\":\"m\",\"architecture\":{},\"classes\":[{\"index\":0,\"name\":\"Tumor\"}]";

    private static ClassifierMetadata load(Path dir, String bodyJson) throws IOException {
        Files.writeString(dir.resolve("metadata.json"), HEAD + bodyJson + "}");
        return new ModelManager().loadMetadata(dir);
    }

    private static String norm(String parent, boolean perChannel, double clip) {
        return String.format(
                ",\"%s\":{\"normalization\":{\"per_channel\":%s,\"clip_percentile\":%s}}", parent, perChannel, clip);
    }

    @Test
    @DisplayName("input_config wins when the two blocks disagree")
    void inputConfigWinsOverChannelConfig(@TempDir Path dir) throws IOException {
        // The tier that was broken: training really used per-channel, toMap()
        // wrote the builder default false into channel_config.
        ClassifierMetadata m = load(dir, norm("channel_config", false, 99.0) + norm("input_config", true, 95.0));

        assertTrue(m.isPerChannelNormalization(), "the truthful input_config value must win");
        assertEquals(95.0, m.getClipPercentile(), 1e-9);
    }

    @Test
    @DisplayName("declaration order in the file does not decide it")
    void fileOrderDoesNotMatter(@TempDir Path dir) throws IOException {
        // Same content, blocks swapped in the JSON text. JSON objects are
        // unordered; the parser must not inherit an ordering from the file.
        ClassifierMetadata m = load(dir, norm("input_config", true, 95.0) + norm("channel_config", false, 99.0));

        assertTrue(m.isPerChannelNormalization());
        assertEquals(95.0, m.getClipPercentile(), 1e-9);
    }

    @Test
    @DisplayName("channel_config is still used when it is the only block")
    void channelConfigUsedWhenAlone(@TempDir Path dir) throws IOException {
        // Older models carry no input_config. The fallback must still work,
        // otherwise fixing the precedence would regress them.
        ClassifierMetadata m = load(dir, norm("channel_config", true, 98.0));

        assertTrue(m.isPerChannelNormalization());
        assertEquals(98.0, m.getClipPercentile(), 1e-9);
    }

    @Test
    @DisplayName("input_config alone is honoured")
    void inputConfigUsedWhenAlone(@TempDir Path dir) throws IOException {
        ClassifierMetadata m = load(dir, norm("input_config", true, 97.0));

        assertTrue(m.isPerChannelNormalization());
        assertEquals(97.0, m.getClipPercentile(), 1e-9);
    }

    @Test
    @DisplayName("neither block falls back to the documented defaults")
    void missingBlocksFallBackToDefaults(@TempDir Path dir) throws IOException {
        // Oldest tier: the true value is unknowable, so default and WARN.
        ClassifierMetadata m = load(dir, "");

        assertFalse(m.isPerChannelNormalization());
        assertEquals(99.0, m.getClipPercentile(), 1e-9);
    }

    @Test
    @DisplayName("agreeing blocks are unaffected")
    void agreeingBlocksAreStable(@TempDir Path dir) throws IOException {
        // Models trained after the builders were fixed: both blocks agree, so
        // the precedence change must be a no-op.
        ClassifierMetadata m = load(dir, norm("channel_config", true, 95.0) + norm("input_config", true, 95.0));

        assertTrue(m.isPerChannelNormalization());
        assertEquals(95.0, m.getClipPercentile(), 1e-9);
    }
}
