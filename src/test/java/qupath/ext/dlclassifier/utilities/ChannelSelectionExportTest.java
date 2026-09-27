package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Training patches must carry exactly the selected channels, in the selected
 * order.
 *
 * <p>Two failures made this necessary. Selecting 6 of an 8-channel
 * fluorescence image exported 8-channel patches against a 6-channel model,
 * which died inside the first convolution with a message that never mentions
 * channels. Reordering channels without dropping any was worse: the counts
 * matched, nothing raised, and the model trained on file order while
 * inference fed the user's order.
 */
class ChannelSelectionExportTest {

    @Test
    void allChannelsInOrderNeedsNoExtraction() throws IOException {
        // The RGB path. Returning null here is what keeps 8-bit H&E patches
        // on the TIFF writer instead of silently moving them to .raw.
        assertThat(AnnotationExtractor.channelSelectionFor(List.of(0, 1, 2), 3)).isNull();
    }

    @Test
    void emptyOrNullSelectionMeansEveryBand() throws IOException {
        assertThat(AnnotationExtractor.channelSelectionFor(List.of(), 8)).isNull();
        assertThat(AnnotationExtractor.channelSelectionFor(null, 8)).isNull();
    }

    @Test
    void subsetIsExtracted() throws IOException {
        // The reported crash: 6 of 8 channels, DAPI and autofluorescence dropped.
        assertThat(AnnotationExtractor.channelSelectionFor(List.of(0, 1, 2, 3, 4, 5), 8))
                .containsExactly(0, 1, 2, 3, 4, 5);
    }

    @Test
    void reorderingIsExtractedEvenThoughTheCountMatches() throws IOException {
        // The silent case. A count check alone would pass this and train the
        // model on the wrong channel for every pixel.
        assertThat(AnnotationExtractor.channelSelectionFor(List.of(2, 0, 1), 3)).containsExactly(2, 0, 1);
    }

    @Test
    void anOutOfRangeChannelIsRejectedByName() {
        assertThatThrownBy(() -> AnnotationExtractor.channelSelectionFor(List.of(0, 9), 3))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("Channel 9")
                .hasMessageContaining("only 3");
    }

    @Test
    void extractionMatchesTheInferenceSideOrdering() {
        // BitDepthConverter.extractChannels is what the inference encoder
        // uses; the export path must agree with it or a model trained one way
        // is run the other.
        float[][][] data = {
            {{10, 20, 30}, {11, 21, 31}},
            {{12, 22, 32}, {13, 23, 33}},
        };
        float[][][] picked = BitDepthConverter.extractChannels(data, new int[] {2, 0});
        assertThat(picked[0][0]).containsExactly(30f, 10f);
        assertThat(picked[1][1]).containsExactly(33f, 13f);
    }
}
