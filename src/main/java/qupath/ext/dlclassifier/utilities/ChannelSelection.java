package qupath.ext.dlclassifier.utilities;

import java.io.IOException;
import java.util.List;

/**
 * The one place that resolves a channel selection against an image's bands.
 *
 * <p>CONTRACT: pixels handed to a model carry exactly the channels named by
 * {@code ChannelConfiguration.getSelectedChannels()}, in that order. Four
 * places in this extension produce such pixels -- training export, the
 * single-tile overlay, the whole-slide batch, and AdaBN calibration -- and
 * the Python side does not select channels at all. It trusts Java to have
 * done it.
 *
 * <p>That trust was misplaced twice. Training export wrote every band while
 * the model was built for a subset, and the run died inside the first
 * convolution (fixed in 0.9.7). The whole-slide batch and AdaBN calibration
 * did the same thing, each having re-derived the decision locally (fixed in
 * 0.9.8). Reordering was worse than dropping: the channel count still
 * matched, so nothing raised, and the model simply trained on the wrong
 * channel for every pixel.
 *
 * <p>Every producer resolves the selection here so a fifth cannot quietly
 * invent a fifth answer.
 *
 * @author UW-LOCI
 * @since 0.9.8
 */
public final class ChannelSelection {

    private ChannelSelection() {
        // Utility class
    }

    /**
     * Resolves the channel indices to extract from an image.
     *
     * @param selected selected channel indices, or null/empty to mean all bands
     * @param numBands band count of the image being encoded
     * @return the indices to extract, or {@code null} when the selection is
     *         every band in file order and the image can be used untouched
     * @throws IOException if a selected index does not exist in the image
     */
    public static int[] indicesFor(List<Integer> selected, int numBands) throws IOException {
        if (selected == null || selected.isEmpty()) {
            return null;
        }
        int[] indices = new int[selected.size()];
        boolean identity = selected.size() == numBands;
        for (int i = 0; i < indices.length; i++) {
            int band = selected.get(i);
            if (band < 0 || band >= numBands) {
                throw new IOException(String.format(
                        "Channel %d was selected but the image has only %d channel(s) (0-%d). "
                                + "Reopen the channel list and reselect.",
                        band, numBands, numBands - 1));
            }
            indices[i] = band;
            if (band != i) {
                identity = false;
            }
        }
        return identity ? null : indices;
    }

    /**
     * Returns how many channels the model will receive.
     *
     * @param selected selected channel indices, or null/empty to mean all bands
     * @param numBands band count of the image being encoded
     * @return the channel count after selection
     */
    public static int channelCount(List<Integer> selected, int numBands) {
        return selected == null || selected.isEmpty() ? numBands : selected.size();
    }

    /**
     * Whether the uint8 simple-RGB fast path may be used.
     *
     * <p>{@code TileEncoder.encodeTileRaw} copies bands verbatim and cannot
     * express a subset or a reorder, so the fast path is only safe when the
     * selection asks for every band in file order. A caller that skips this
     * check silently ignores the user's channel choice on RGB images.
     *
     * @param selected selected channel indices, or null/empty to mean all bands
     * @param numBands band count of the image being encoded
     * @return true when the byte fast path preserves the selection
     * @throws IOException if a selected index does not exist in the image
     */
    public static boolean canUseByteFastPath(List<Integer> selected, int numBands) throws IOException {
        return indicesFor(selected, numBands) == null;
    }
}
