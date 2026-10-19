package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Pins where a multi-pass overlay pass may be trusted.
 * <p>
 * Multi-pass averaging shifts the tile by half a stride and averages the
 * shifted prediction into the original tile. It used to average in the WHOLE
 * original stride, half of which sits in the shifted tile's halo, where the
 * model has no context beyond the edge of what it was given. On a two-class
 * tissue model (QuPath 0.7, Windows, 0.9.26) two context-starved passes
 * outvoted two good ones on every tile, the 2-2 tie went to class 0, and the
 * overlay drew background stripes on the tile grid across solid tissue.
 * <p>
 * The rule now: a shifted pass counts only where the original stride overlaps
 * the shifted tile's OWN stride, i.e. its center crop.
 */
class MultiPassTrustedSpanTest {

    /** The geometry the overlay really uses: map = stride + 2*pad, original at pad + shift. */
    private static int[] span(int stride, int pad, int shift) {
        int origOffset = pad + shift;
        int shiftedOffset = pad;
        return DLPixelClassifier.trustedSpan(origOffset, shiftedOffset, stride, stride + 2 * pad);
    }

    @Test
    void theCaseThatStripedTheOverlay() {
        // tile 256 at the 25% center-crop floor: pad 64, stride 128, half-stride shift 64
        assertThat(span(128, 64, 64)).containsExactly(0, 64);
    }

    @Test
    void anUnshiftedPassIsTrustedEverywhere() {
        assertThat(span(128, 64, 0)).containsExactly(0, 128);
    }

    @Test
    void aShiftOfAWholeStrideOrMoreIsNeverTrusted() {
        assertThat(span(128, 64, 128)).isNull();
        assertThat(span(128, 64, 200)).isNull();
    }

    /**
     * The invariant, over a range rather than one shape: every trusted pixel
     * maps into the shifted tile's own stride, none falls in its halo, and the
     * trusted length is exactly stride - shift.
     */
    @Test
    void trustedPixelsNeverComeFromTheHalo() {
        int cases = 0;
        for (int stride = 16; stride <= 512; stride += 16) {
            for (int pad = 0; pad <= 128; pad += 8) {
                for (int shift = 0; shift < stride; shift += 4) {
                    int origOffset = pad + shift;
                    int[] s = span(stride, pad, shift);
                    assertThat(s)
                            .as("stride %d pad %d shift %d", stride, pad, shift)
                            .isNotNull();
                    assertThat(s[1] - s[0]).isEqualTo(stride - shift);
                    for (int x = s[0]; x < s[1]; x++) {
                        int inShifted = x + origOffset;
                        assertThat(inShifted).isBetween(pad, pad + stride - 1);
                    }
                    cases++;
                }
            }
        }
        assertThat(cases).isGreaterThan(10_000);
    }

    @Test
    void aClippedEdgeTileIsLimitedToThePixelsTheMapHas() {
        // Image edge: the shifted map is shorter than stride + 2*pad.
        int[] s = DLPixelClassifier.trustedSpan(128, 64, 128, 150);
        assertThat(s).containsExactly(0, 22);
    }

    @Test
    void aShiftedTileStartingBeforeTheImageIsClampedToTheMap() {
        // Left image edge: the shifted stride begins before pixel 0 of the map.
        int[] s = DLPixelClassifier.trustedSpan(0, -64, 128, 192);
        assertThat(s).containsExactly(0, 64);
    }
}
