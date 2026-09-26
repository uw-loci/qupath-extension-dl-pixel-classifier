package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;

import java.awt.Point;
import java.awt.color.ColorSpace;
import java.awt.image.BandedSampleModel;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link ImageCompat}, the TYPE_CUSTOM-safe image operations.
 * <p>
 * Multi-channel fluorescence regions come back from QuPath as
 * {@link BufferedImage#TYPE_CUSTOM}, which neither the {@code BufferedImage}
 * (width, height, type) constructor nor {@link java.awt.Graphics2D} can handle.
 * Every fixture here asserts {@code getType() == TYPE_CUSTOM} so a test that
 * stopped exercising the custom path would fail rather than quietly pass.
 */
class ImageCompatTest {

    private static final int BANDS = 7;

    /**
     * A 7-band float image of the shape a 7-plex fluorescence read produces.
     * Sample values encode (band, x, y) so a mirrored or offset copy is
     * identifiable from the value alone.
     */
    private static BufferedImage customImage(int width, int height, int bands, int dataType) {
        WritableRaster raster =
                Raster.createWritableRaster(new BandedSampleModel(dataType, width, height, bands), new Point(0, 0));
        int colorSpaceType = bands == 1 ? ColorSpace.TYPE_GRAY : ColorSpace.TYPE_7CLR;
        ColorSpace colorSpace = new ColorSpace(colorSpaceType, bands) {
            @Override
            public float[] toRGB(float[] colorvalue) {
                return new float[] {colorvalue[0], colorvalue[0], colorvalue[0]};
            }

            @Override
            public float[] fromRGB(float[] rgbvalue) {
                return new float[bands];
            }

            @Override
            public float[] toCIEXYZ(float[] colorvalue) {
                return new float[3];
            }

            @Override
            public float[] fromCIEXYZ(float[] colorvalue) {
                return new float[bands];
            }
        };
        ColorModel colorModel = new ComponentColorModel(colorSpace, false, false, ColorModel.OPAQUE, dataType);
        BufferedImage image = new BufferedImage(colorModel, raster, false, null);
        for (int b = 0; b < bands; b++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    raster.setSample(x, y, b, sampleValue(b, x, y));
                }
            }
        }
        return image;
    }

    private static BufferedImage customImage(int width, int height) {
        return customImage(width, height, BANDS, DataBuffer.TYPE_FLOAT);
    }

    /** Distinct per (band, x, y) and never zero, so a dropped sample is visible. */
    private static int sampleValue(int band, int x, int y) {
        return 1 + band * 10000 + y * 100 + x;
    }

    /** A standard-type image whose RGB encodes (x, y). */
    private static BufferedImage rgbImage(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, ((x + 1) << 8) | (y + 1));
            }
        }
        return image;
    }

    private static float sample(BufferedImage image, int x, int y, int band) {
        return image.getRaster().getSampleFloat(x, y, band);
    }

    // ------------------------------------------------------------------
    // The bug itself
    // ------------------------------------------------------------------

    @Test
    @DisplayName("the old allocation really does throw on a multi-channel fluorescence image")
    void oldAllocationThrowsOnCustomType() {
        BufferedImage custom = customImage(8, 8);
        assertThat(custom.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);

        // This is verbatim what the four call sites used to do, and it is the
        // "Unknown image type 0" that blocked all fluorescence training.
        assertThatExceptionOfType(IllegalArgumentException.class)
                .isThrownBy(() -> new BufferedImage(16, 16, custom.getType()))
                .withMessageContaining("Unknown image type 0");
    }

    @Test
    @DisplayName("Graphics2D silently zeroes the bands of a TYPE_CUSTOM image")
    void graphicsPathLosesCustomSamples() {
        BufferedImage source = customImage(4, 4);
        BufferedImage destination = ImageCompat.createCompatible(source, 8, 8);

        var g = destination.createGraphics();
        g.drawImage(source, 2, 2, null);
        g.dispose();

        // createGraphics() and drawImage() both succeed -- and carry nothing.
        // This is why the TYPE_CUSTOM paths work on the raster instead.
        assertThat(sample(source, 0, 0, 3)).isEqualTo(sampleValue(3, 0, 0));
        assertThat(sample(destination, 2, 2, 3)).isZero();
    }

    // ------------------------------------------------------------------
    // createCompatible
    // ------------------------------------------------------------------

    @Test
    @DisplayName("createCompatible preserves band count and data type of a 7-band float source")
    void createCompatibleKeepsBandsAndDataType() {
        BufferedImage source = customImage(8, 8);
        assertThat(source.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);

        BufferedImage created = ImageCompat.createCompatible(source, 33, 21);

        assertThat(created.getWidth()).isEqualTo(33);
        assertThat(created.getHeight()).isEqualTo(21);
        assertThat(created.getRaster().getNumBands()).isEqualTo(BANDS);
        assertThat(created.getRaster().getDataBuffer().getDataType()).isEqualTo(DataBuffer.TYPE_FLOAT);
        assertThat(created.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);
    }

    @Test
    @DisplayName("createCompatible keeps the standard type for a standard source")
    void createCompatibleKeepsStandardType() {
        BufferedImage created = ImageCompat.createCompatible(rgbImage(4, 4), 6, 7);

        assertThat(created.getType()).isEqualTo(BufferedImage.TYPE_INT_RGB);
        assertThat(created.getWidth()).isEqualTo(6);
        assertThat(created.getHeight()).isEqualTo(7);
    }

    @Test
    @DisplayName("createCompatible handles a 16-bit source too")
    void createCompatibleHandlesUnsignedShort() {
        BufferedImage source = customImage(8, 8, BANDS, DataBuffer.TYPE_USHORT);
        assertThat(source.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);

        BufferedImage created = ImageCompat.createCompatible(source, 10, 10);

        assertThat(created.getRaster().getDataBuffer().getDataType()).isEqualTo(DataBuffer.TYPE_USHORT);
        assertThat(created.getRaster().getNumBands()).isEqualTo(BANDS);
    }

    // ------------------------------------------------------------------
    // reflectionPad
    // ------------------------------------------------------------------

    @Test
    @DisplayName("reflectionPad places a TYPE_CUSTOM source at the offset, every band intact")
    void reflectionPadCopiesCustomSourceAtOffset() {
        BufferedImage source = customImage(4, 4);
        assertThat(source.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);

        BufferedImage padded = ImageCompat.reflectionPad(source, 8, 8, 2, 2);

        assertThat(padded.getWidth()).isEqualTo(8);
        assertThat(padded.getHeight()).isEqualTo(8);
        assertThat(padded.getRaster().getNumBands()).isEqualTo(BANDS);
        for (int b = 0; b < BANDS; b++) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    assertThat(sample(padded, x + 2, y + 2, b))
                            .as("band %d, source (%d,%d) at padded (%d,%d)", b, x, y, x + 2, y + 2)
                            .isEqualTo(sampleValue(b, x, y));
                }
            }
        }
    }

    @Test
    @DisplayName("reflectionPad mirrors the right edge strip of a TYPE_CUSTOM source, per band")
    void reflectionPadMirrorsRightEdgeOfCustomSource() {
        BufferedImage source = customImage(4, 4);
        BufferedImage padded = ImageCompat.reflectionPad(source, 8, 8, 2, 2);

        // Right gap is 8 - (2 + 4) = 2 columns, padded x = 6 and 7. Java2D's
        // mirroring maps the first padded column to source column srcW - 2 = 2
        // and the next to source column 1, so column 3 is not repeated.
        for (int b = 0; b < BANDS; b++) {
            for (int y = 0; y < 4; y++) {
                assertThat(sample(padded, 6, y + 2, b))
                        .as("band %d, padded (6,%d)", b, y + 2)
                        .isEqualTo(sampleValue(b, 2, y));
                assertThat(sample(padded, 7, y + 2, b))
                        .as("band %d, padded (7,%d)", b, y + 2)
                        .isEqualTo(sampleValue(b, 1, y));
            }
        }
    }

    @Test
    @DisplayName("reflectionPad mirrors the left edge strip of a TYPE_CUSTOM source, per band")
    void reflectionPadMirrorsLeftEdgeOfCustomSource() {
        BufferedImage source = customImage(4, 4);
        BufferedImage padded = ImageCompat.reflectionPad(source, 8, 8, 2, 2);

        // The left mirror writes destination columns [offsetX - w - 1, offsetX - 1),
        // i.e. only column 0 here, taking source column 0. Column 1 is deliberately
        // left blank -- an off-by-one in the original Graphics2D code that this
        // path reproduces rather than fixes.
        for (int b = 0; b < BANDS; b++) {
            for (int y = 0; y < 4; y++) {
                assertThat(sample(padded, 0, y + 2, b))
                        .as("band %d, padded (0,%d)", b, y + 2)
                        .isEqualTo(sampleValue(b, 0, y));
                assertThat(sample(padded, 1, y + 2, b))
                        .as("band %d, padded (1,%d) is the unwritten column", b, y + 2)
                        .isZero();
            }
        }
    }

    @Test
    @DisplayName("reflectionPad mirrors the top strip of a TYPE_CUSTOM source after the side mirrors")
    void reflectionPadMirrorsTopStripOfCustomSource() {
        BufferedImage source = customImage(4, 4);
        BufferedImage padded = ImageCompat.reflectionPad(source, 8, 8, 2, 2);

        // The top mirror reads back padded row offsetY (row 2), which by then
        // already carries the left/right reflections, and writes it to row 0.
        for (int b = 0; b < BANDS; b++) {
            for (int x = 0; x < 8; x++) {
                assertThat(sample(padded, x, 0, b))
                        .as("band %d, padded (%d,0) mirrors (%d,2)", b, x, x)
                        .isEqualTo(sample(padded, x, 2, b));
            }
        }
    }

    @Test
    @DisplayName("reflectionPad on a TYPE_CUSTOM source matches the Graphics2D geometry exactly")
    void reflectionPadCustomMatchesGraphicsGeometry() {
        // Same geometry, run through both implementations on a standard image:
        // whatever Java2D does -- flips, clipping, the unwritten column and row --
        // the raster path must do too.
        int[][] cases = {
            {4, 4, 8, 8, 2, 2},
            {4, 4, 8, 8, 0, 0},
            {5, 3, 9, 9, 3, 4},
            {6, 6, 8, 8, 1, 1},
            {3, 3, 10, 10, 4, 4},
            {4, 4, 8, 8, 4, 4},
            {4, 4, 8, 8, 0, 3},
        };
        for (int[] c : cases) {
            BufferedImage source = rgbImage(c[0], c[1]);
            BufferedImage viaGraphics = ImageCompat.reflectionPad(source, c[2], c[3], c[4], c[5]);
            BufferedImage viaRaster = ImageCompat.reflectionPadRaster(source, c[2], c[3], c[4], c[5]);
            for (int y = 0; y < c[3]; y++) {
                for (int x = 0; x < c[2]; x++) {
                    assertThat(viaRaster.getRGB(x, y))
                            .as(
                                    "src %dx%d target %dx%d offset (%d,%d) at (%d,%d)",
                                    c[0], c[1], c[2], c[3], c[4], c[5], x, y)
                            .isEqualTo(viaGraphics.getRGB(x, y));
                }
            }
        }
    }

    @Test
    @DisplayName("reflectionPad leaves a standard image on the Graphics2D path")
    void reflectionPadStandardImageUnchanged() {
        BufferedImage source = rgbImage(4, 4);
        BufferedImage padded = ImageCompat.reflectionPad(source, 8, 8, 2, 2);

        assertThat(padded.getType()).isEqualTo(BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                assertThat(padded.getRGB(x + 2, y + 2))
                        .as("source (%d,%d)", x, y)
                        .isEqualTo(source.getRGB(x, y));
            }
        }
        // Right-edge mirror, as above: padded column 6 is source column 2.
        assertThat(padded.getRGB(6, 3)).isEqualTo(source.getRGB(2, 1));
    }

    // ------------------------------------------------------------------
    // resizeBilinear
    // ------------------------------------------------------------------

    @Test
    @DisplayName("resizeBilinear on a TYPE_CUSTOM source keeps band count and data type")
    void resizeCustomKeepsBandsAndSize() {
        BufferedImage source = customImage(8, 8);
        assertThat(source.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);

        BufferedImage resized = ImageCompat.resizeBilinear(source, 16, 16);

        assertThat(resized.getWidth()).isEqualTo(16);
        assertThat(resized.getHeight()).isEqualTo(16);
        assertThat(resized.getRaster().getNumBands()).isEqualTo(BANDS);
        assertThat(resized.getRaster().getDataBuffer().getDataType()).isEqualTo(DataBuffer.TYPE_FLOAT);
    }

    @Test
    @DisplayName("resizeBilinear carries real sample values, not zeros, in every band")
    void resizeCustomCarriesSamples() {
        BufferedImage source = customImage(8, 8);
        BufferedImage resized = ImageCompat.resizeBilinear(source, 4, 4);

        for (int b = 0; b < BANDS; b++) {
            // Interior corner of a 2x downsample: source pixel centres 0 and 1
            // average, plus the band offset, so the value must sit in that band's
            // range and never collapse to zero.
            float value = sample(resized, 1, 1, b);
            assertThat(value)
                    .as("band %d at (1,1)", b)
                    .isGreaterThan(b * 10000f)
                    .isLessThan((b + 1) * 10000f + 1f);
        }
    }

    @Test
    @DisplayName("resizeBilinear interpolates a known TYPE_CUSTOM ramp")
    void resizeCustomInterpolatesRamp() {
        // Single-band float ramp v = x, so a 4 -> 8 upscale has predictable values:
        // destination x maps to source (x + 0.5) / 2 - 0.5.
        BufferedImage source = customImage(4, 1, 1, DataBuffer.TYPE_FLOAT);
        WritableRaster raster = source.getRaster();
        for (int x = 0; x < 4; x++) {
            raster.setSample(x, 0, 0, x);
        }
        assertThat(source.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);

        BufferedImage resized = ImageCompat.resizeBilinear(source, 8, 1);

        // x = 0 -> source -0.25, clamped to 0.0; x = 2 -> 0.75; x = 4 -> 1.75
        assertThat(sample(resized, 0, 0, 0)).isEqualTo(0f);
        assertThat(sample(resized, 2, 0, 0)).isEqualTo(0.75f);
        assertThat(sample(resized, 4, 0, 0)).isEqualTo(1.75f);
        assertThat(sample(resized, 7, 0, 0)).isEqualTo(3f);
    }

    @Test
    @DisplayName("resizeBilinear rounds rather than truncates on an integer raster")
    void resizeCustomRoundsIntegerSamples() {
        BufferedImage source = customImage(2, 1, 1, DataBuffer.TYPE_USHORT);
        source.getRaster().setSample(0, 0, 0, 0);
        source.getRaster().setSample(1, 0, 0, 10);
        assertThat(source.getType()).isEqualTo(BufferedImage.TYPE_CUSTOM);

        BufferedImage resized = ImageCompat.resizeBilinear(source, 4, 1);

        // Destination x = 2 maps to source 0.75, so the interpolated value is
        // exactly 7.5. Rounding gives 8; truncating -- which is what
        // setSamples(double[]) does on its own for an integer sample model --
        // would give 7.
        assertThat(sample(resized, 2, 0, 0)).isEqualTo(8f);
        assertThat(sample(resized, 3, 0, 0)).isEqualTo(10f);
    }

    @Test
    @DisplayName("resizeBilinear returns the same instance when no resize is needed")
    void resizeNoOpReturnsSource() {
        BufferedImage source = customImage(8, 8);
        assertThat(ImageCompat.resizeBilinear(source, 8, 8)).isSameAs(source);
    }

    @Test
    @DisplayName("resizeBilinear leaves a standard image on the Graphics2D path")
    void resizeStandardImageUsesGraphics() {
        BufferedImage source = rgbImage(8, 8);
        BufferedImage resized = ImageCompat.resizeBilinear(source, 4, 4);

        assertThat(resized.getType()).isEqualTo(BufferedImage.TYPE_INT_RGB);
        assertThat(resized.getWidth()).isEqualTo(4);
        assertThat(resized.getHeight()).isEqualTo(4);
        // Graphics2D bilinear on a real image is not all-black.
        boolean anyNonZero = false;
        for (int y = 0; y < 4 && !anyNonZero; y++) {
            for (int x = 0; x < 4; x++) {
                if ((resized.getRGB(x, y) & 0xFFFFFF) != 0) {
                    anyNonZero = true;
                    break;
                }
            }
        }
        assertThat(anyNonZero).isTrue();
    }

    @Test
    @DisplayName("isCustomType distinguishes the two families")
    void isCustomTypeDistinguishes() {
        assertThat(ImageCompat.isCustomType(customImage(4, 4))).isTrue();
        assertThat(ImageCompat.isCustomType(rgbImage(4, 4))).isFalse();
    }
}
