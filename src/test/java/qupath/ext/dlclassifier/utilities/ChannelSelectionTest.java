package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The channel-selection contract, and the agreement between its producers.
 *
 * <p>Pixels handed to a model must carry exactly the selected channels in the
 * selected order. The Python side does not check this -- the inference scripts
 * say outright that Java has already done the selecting -- so every Java
 * producer has to get it right independently. Two of the four did not:
 *
 * <ul>
 *   <li>training export wrote every band against a subset model and died in
 *       the first convolution (0.9.7);</li>
 *   <li>the whole-slide batch and AdaBN calibration did the same thing,
 *       each having re-derived the decision locally (0.9.8).</li>
 * </ul>
 *
 * <p>The failure that motivated the shared helper is the one where nothing
 * raises: a REORDER keeps the channel count identical, so training used file
 * order while inference used the user's order and every pixel was classified
 * from the wrong channel.
 */
class ChannelSelectionTest {

    @Test
    void everyBandInOrderNeedsNoExtraction() throws IOException {
        // Returning null is what keeps 8-bit RGB on the untouched TIFF/uint8
        // paths. If this ever returns indices, every H&E export and every
        // RGB inference silently changes format.
        assertThat(ChannelSelection.indicesFor(List.of(0, 1, 2), 3)).isNull();
    }

    @Test
    void absentSelectionMeansEveryBand() throws IOException {
        assertThat(ChannelSelection.indicesFor(null, 8)).isNull();
        assertThat(ChannelSelection.indicesFor(List.of(), 8)).isNull();
        assertThat(ChannelSelection.channelCount(null, 8)).isEqualTo(8);
        assertThat(ChannelSelection.channelCount(List.of(), 8)).isEqualTo(8);
    }

    @Test
    void aSubsetIsExtracted() throws IOException {
        // The reported crash: 6 of 8 channels, DAPI and autofluorescence dropped.
        assertThat(ChannelSelection.indicesFor(List.of(0, 1, 2, 3, 4, 5), 8)).containsExactly(0, 1, 2, 3, 4, 5);
        assertThat(ChannelSelection.channelCount(List.of(0, 1, 2, 3, 4, 5), 8)).isEqualTo(6);
    }

    @Test
    void aReorderIsExtractedEvenThoughTheCountMatches() throws IOException {
        // The silent case. A count-only check passes this and trains the
        // model on the wrong channel for every pixel.
        assertThat(ChannelSelection.indicesFor(List.of(2, 0, 1), 3)).containsExactly(2, 0, 1);
        assertThat(ChannelSelection.channelCount(List.of(2, 0, 1), 3)).isEqualTo(3);
    }

    @Test
    void aNonContiguousSelectionIsExtracted() throws IOException {
        // "DAPI plus one marker" on an 8-channel panel.
        assertThat(ChannelSelection.indicesFor(List.of(5, 6), 8)).containsExactly(5, 6);
        assertThat(ChannelSelection.channelCount(List.of(5, 6), 8)).isEqualTo(2);
    }

    @Test
    void anOutOfRangeChannelIsRejectedByName() {
        assertThatThrownBy(() -> ChannelSelection.indicesFor(List.of(0, 9), 3))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("Channel 9")
                .hasMessageContaining("only 3");
    }

    @Test
    void theByteFastPathIsOnlyOfferedForAnIdentitySelection() throws IOException {
        // encodeTileRaw copies bands verbatim. Offering it for a subset or a
        // reorder is how a caller silently ignores the user's channel choice.
        assertThat(ChannelSelection.canUseByteFastPath(List.of(0, 1, 2), 3)).isTrue();
        assertThat(ChannelSelection.canUseByteFastPath(null, 3)).isTrue();
        assertThat(ChannelSelection.canUseByteFastPath(List.of(2, 1, 0), 3)).isFalse();
        assertThat(ChannelSelection.canUseByteFastPath(List.of(0, 2), 3)).isFalse();
    }

    /**
     * Every producer must reach the same answer for the same inputs.
     *
     * <p>This is the shape of the bug that got through twice: each site
     * derived the decision locally and one of them derived it differently.
     * Now they all call the same helper, so the property under test is that
     * the helper's two answers stay consistent with each other -- a count
     * that disagrees with the indices is exactly the desync that produced
     * "expected input[20, 8, 384, 384] to have 6 channels".
     */
    @Test
    void channelCountAlwaysAgreesWithTheExtractedIndices() throws IOException {
        int[][] cases = {
            {3, 0, 1, 2}, {3, 2, 0, 1}, {8, 0, 1, 2, 3, 4, 5}, {8, 5, 6}, {8, 7}, {1, 0},
        };
        for (int[] c : cases) {
            int numBands = c[0];
            List<Integer> selected = new java.util.ArrayList<>();
            for (int i = 1; i < c.length; i++) {
                selected.add(c[i]);
            }
            int[] indices = ChannelSelection.indicesFor(selected, numBands);
            int count = ChannelSelection.channelCount(selected, numBands);
            int actual = indices == null ? numBands : indices.length;
            assertThat(actual).as("bands=%d selected=%s", numBands, selected).isEqualTo(count);
        }
    }

    @Test
    void extractionAgreesWithTheInferenceSideEncoder() {
        // BitDepthConverter.extractChannels is what TileEncoder uses on the
        // inference side; the export side must produce the same ordering or a
        // model trained one way is run the other.
        float[][][] data = {
            {{10, 20, 30}, {11, 21, 31}},
            {{12, 22, 32}, {13, 23, 33}},
        };
        float[][][] picked = BitDepthConverter.extractChannels(data, new int[] {2, 0});
        assertThat(picked[0][0]).containsExactly(30f, 10f);
        assertThat(picked[1][1]).containsExactly(33f, 13f);
    }
}
