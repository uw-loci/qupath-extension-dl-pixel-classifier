package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * A classifier deleted while its overlay was still showing used to produce
 * over a thousand stack traces.
 *
 * <p>The model-directory lookup fell back to the bare classifier ID when
 * neither lookup found anything on disk. An ID is not a path, so every tile
 * asked the Python side to load a directory that cannot exist, and every
 * tile failed twice over: once as an Appose traceback, and once as QuPath's
 * own "Error requesting tile classification", which this extension cannot
 * suppress because QuPath logs it from the calling side.
 *
 * <p>The fix is to notice at construction, so the message is delivered once
 * before a single tile is requested. This exception is the signal, and it is
 * deliberately distinct from a backend failure: nothing is wrong with the
 * server and retrying will not help.
 */
class ModelFilesMissingExceptionTest {

    @Test
    void theMessageNamesTheClassifierAndItsId() {
        var e = new ModelFilesMissingException("CMU-1_Tissue_ResNet-18", "cmu-1_tissue_resnet-18_1790650688378");
        assertThat(e.getMessage()).contains("CMU-1_Tissue_ResNet-18").contains("cmu-1_tissue_resnet-18_1790650688378");
    }

    @Test
    void theMessageSaysWhatToSuspect() {
        // The user's next question is "why", and the two likely answers are
        // worth stating rather than making them guess.
        var e = new ModelFilesMissingException("model", "id");
        assertThat(e.getMessage()).containsIgnoringCase("deleted");
        assertThat(e.getMessage()).containsIgnoringCase("moved");
    }

    @Test
    void theClassifierNameIsAvailableForTheNotification() {
        var e = new ModelFilesMissingException("My Model", "id-1");
        assertThat(e.getClassifierName()).isEqualTo("My Model");
    }

    @Test
    void itIsAnIllegalStateNotAnIoProblem() {
        // Callers separate "the model is not there" from "inference failed".
        // Typing it as an IOException would put it back in the retry path
        // that produced the flood.
        var e = new ModelFilesMissingException("m", "i");
        assertThat(e).isInstanceOf(IllegalStateException.class);
        assertThat(e).isNotInstanceOf(java.io.IOException.class);
    }

    @Test
    void theMessageStaysAscii() {
        // It reaches a notification on Windows, where cp1252 turns a stray
        // Unicode character into a crash rather than a cosmetic problem.
        var e = new ModelFilesMissingException("model", "id");
        assertThat(e.getMessage().chars().allMatch(c -> c < 128)).isTrue();
    }
}
