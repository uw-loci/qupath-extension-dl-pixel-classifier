package qupath.ext.dlclassifier.utilities;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;

/**
 * Image operations that survive {@link BufferedImage#TYPE_CUSTOM} sources.
 * <p>
 * QuPath hands back a TYPE_CUSTOM image for any region it cannot describe with a
 * standard {@code BufferedImage} type -- a multiplex fluorescence read with more
 * than four channels, or any 16-bit / float region. Two things then break, and
 * both of them bit us:
 * <ul>
 *   <li>{@code new BufferedImage(w, h, source.getType())} throws
 *       {@code IllegalArgumentException: Unknown image type 0}, because
 *       {@code TYPE_CUSTOM == 0} is not a type the constructor accepts. This is
 *       what blocked all training on multi-channel fluorescence images.</li>
 *   <li>{@link Graphics2D} does not carry the samples. {@code createGraphics()}
 *       succeeds and {@code drawImage} does not throw, but Java2D routes the
 *       pixels through the image's {@code ColorSpace}, so an N-band float
 *       destination comes back all zeros. A silent data loss, not a crash.</li>
 * </ul>
 * So the methods here branch: standard types keep the proven Java2D path
 * unchanged, and TYPE_CUSTOM sources are handled by copying samples through
 * {@link Raster} / {@link WritableRaster}, which is band- and precision-agnostic
 * and performs no colour conversion.
 * <p>
 * {@code AffineTransformOp}'s {@code filter(Raster, WritableRaster)} overload is
 * not an option for the resize: it throws
 * {@code ImagingOpException: Unable to transform src image} on both banded-float
 * and interleaved-ushort multi-band rasters, for bilinear and nearest-neighbour
 * alike. The bilinear resampling below is therefore done by hand.
 *
 * @author UW-LOCI
 */
public final class ImageCompat {

    private ImageCompat() {}

    /**
     * Whether an image has no standard {@code BufferedImage} type, and so must be
     * handled through its raster rather than through Java2D.
     */
    public static boolean isCustomType(BufferedImage image) {
        return image.getType() == BufferedImage.TYPE_CUSTOM;
    }

    /**
     * Allocates an image of the requested size that can hold the same kind of
     * pixels as {@code source}.
     * <p>
     * For a standard type this is {@code new BufferedImage(w, h, source.getType())}.
     * For TYPE_CUSTOM it is a raster built from the source's own sample model --
     * same band count, same data type -- wrapped in the source's colour model.
     *
     * @param source the image whose pixel layout should be matched
     * @param width  width of the new image
     * @param height height of the new image
     * @return a new, blank image of the requested size
     */
    public static BufferedImage createCompatible(BufferedImage source, int width, int height) {
        if (!isCustomType(source)) {
            return new BufferedImage(width, height, source.getType());
        }
        WritableRaster raster = source.getRaster().createCompatibleWritableRaster(width, height);
        return new BufferedImage(source.getColorModel(), raster, source.isAlphaPremultiplied(), null);
    }

    /**
     * Reflection-pads a source image to target dimensions.
     * The source is placed at (offsetX, offsetY) within the target; remaining
     * pixels are filled by reflecting the source content.
     * <p>
     * The reflection geometry -- including the one destination column and row
     * immediately adjacent to the source that the mirroring steps leave
     * untouched -- is identical for standard and TYPE_CUSTOM images;
     * {@code ImageCompatTest} pins the two paths against each other.
     *
     * @param source  the source image
     * @param targetW target width
     * @param targetH target height
     * @param offsetX X offset where source is placed in the target
     * @param offsetY Y offset where source is placed in the target
     * @return reflection-padded image of targetW x targetH
     */
    public static BufferedImage reflectionPad(
            BufferedImage source, int targetW, int targetH, int offsetX, int offsetY) {
        return isCustomType(source)
                ? reflectionPadRaster(source, targetW, targetH, offsetX, offsetY)
                : reflectionPadGraphics(source, targetW, targetH, offsetX, offsetY);
    }

    /**
     * Resizes an image to the requested dimensions with bilinear interpolation.
     *
     * @param source the image to resize
     * @param width  target width
     * @param height target height
     * @return a resized image, or {@code source} itself if it is already that size
     */
    public static BufferedImage resizeBilinear(BufferedImage source, int width, int height) {
        if (source.getWidth() == width && source.getHeight() == height) {
            return source;
        }
        return isCustomType(source)
                ? resizeBilinearRaster(source, width, height)
                : resizeBilinearGraphics(source, width, height);
    }

    // ------------------------------------------------------------------
    // Standard-type paths (Java2D). Unchanged behaviour -- do not "improve".
    // ------------------------------------------------------------------

    private static BufferedImage reflectionPadGraphics(
            BufferedImage source, int targetW, int targetH, int offsetX, int offsetY) {
        int srcW = source.getWidth();
        int srcH = source.getHeight();
        BufferedImage padded = new BufferedImage(targetW, targetH, source.getType());
        Graphics2D g = padded.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);

        // Draw the source at its offset position
        g.drawImage(source, offsetX, offsetY, null);

        // Reflect left border
        if (offsetX > 0) {
            int w = Math.min(offsetX, srcW);
            // Flip horizontally: draw the left strip of source mirrored
            g.drawImage(
                    source,
                    offsetX - 1,
                    offsetY,
                    offsetX - w - 1,
                    offsetY + srcH, // dest: reversed x
                    0,
                    0,
                    w,
                    srcH, // src
                    null);
        }

        // Reflect right border
        int rightGap = targetW - (offsetX + srcW);
        if (rightGap > 0) {
            int w = Math.min(rightGap, srcW);
            g.drawImage(
                    source,
                    offsetX + srcW,
                    offsetY,
                    offsetX + srcW + w,
                    offsetY + srcH,
                    srcW - 1,
                    0,
                    srcW - 1 - w,
                    srcH,
                    null);
        }

        // Reflect top border (over the full width of what we have so far)
        if (offsetY > 0) {
            int h = Math.min(offsetY, srcH);
            // Copy the top strip of the padded image and flip vertically
            BufferedImage topStrip = padded.getSubimage(0, offsetY, targetW, h);
            g.drawImage(topStrip, 0, offsetY - 1, targetW, offsetY - h - 1, 0, 0, targetW, h, null);
        }

        // Reflect bottom border
        int bottomGap = targetH - (offsetY + srcH);
        if (bottomGap > 0) {
            int h = Math.min(bottomGap, srcH);
            BufferedImage bottomStrip = padded.getSubimage(0, offsetY + srcH - h, targetW, h);
            g.drawImage(bottomStrip, 0, offsetY + srcH, targetW, offsetY + srcH + h, 0, h - 1, targetW, -1, null);
        }

        g.dispose();
        return padded;
    }

    private static BufferedImage resizeBilinearGraphics(BufferedImage source, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, source.getType());
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(source, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    // ------------------------------------------------------------------
    // TYPE_CUSTOM paths (raster sample copying). Package-private so the test
    // can run them against a standard image and compare with the Java2D result.
    // ------------------------------------------------------------------

    /**
     * Raster-level equivalent of {@link #reflectionPadGraphics}: same sequence of
     * blits, same source and destination rectangles, no colour conversion.
     */
    static BufferedImage reflectionPadRaster(BufferedImage source, int targetW, int targetH, int offsetX, int offsetY) {
        int srcW = source.getWidth();
        int srcH = source.getHeight();
        BufferedImage padded = createCompatible(source, targetW, targetH);
        WritableRaster dst = padded.getRaster();
        Raster src = source.getRaster();

        // Place the source at its offset position
        blit(src, dst, offsetX, offsetY, offsetX + srcW, offsetY + srcH, 0, 0, srcW, srcH);

        // Reflect left border
        if (offsetX > 0) {
            int w = Math.min(offsetX, srcW);
            blit(src, dst, offsetX - 1, offsetY, offsetX - w - 1, offsetY + srcH, 0, 0, w, srcH);
        }

        // Reflect right border
        int rightGap = targetW - (offsetX + srcW);
        if (rightGap > 0) {
            int w = Math.min(rightGap, srcW);
            blit(
                    src,
                    dst,
                    offsetX + srcW,
                    offsetY,
                    offsetX + srcW + w,
                    offsetY + srcH,
                    srcW - 1,
                    0,
                    srcW - 1 - w,
                    srcH);
        }

        // Reflect top border (over the full width of what we have so far, so this
        // reads back the left/right reflections just written)
        if (offsetY > 0) {
            int h = Math.min(offsetY, srcH);
            Raster topStrip = dst.createChild(0, offsetY, targetW, h, 0, 0, null);
            blit(topStrip, dst, 0, offsetY - 1, targetW, offsetY - h - 1, 0, 0, targetW, h);
        }

        // Reflect bottom border
        int bottomGap = targetH - (offsetY + srcH);
        if (bottomGap > 0) {
            int h = Math.min(bottomGap, srcH);
            Raster bottomStrip = dst.createChild(0, offsetY + srcH - h, targetW, h, 0, 0, null);
            blit(bottomStrip, dst, 0, offsetY + srcH, targetW, offsetY + srcH + h, 0, h - 1, targetW, -1);
        }

        return padded;
    }

    /**
     * Bilinear resize done on the raster, band by band, with samples read at
     * destination pixel centres and edge coordinates clamped. Integer sample
     * models get a rounded value; float models get the interpolated value as-is.
     */
    static BufferedImage resizeBilinearRaster(BufferedImage source, int width, int height) {
        Raster src = source.getRaster();
        BufferedImage resized = createCompatible(source, width, height);
        WritableRaster dst = resized.getRaster();

        int srcW = src.getWidth();
        int srcH = src.getHeight();
        int srcMinX = src.getMinX();
        int srcMinY = src.getMinY();
        int bands = Math.min(src.getNumBands(), dst.getNumBands());
        boolean integral = isIntegral(src);

        double scaleX = (double) srcW / width;
        double scaleY = (double) srcH / height;

        // Per-column interpolation setup, hoisted out of the row loop
        int[] x0 = new int[width];
        int[] x1 = new int[width];
        double[] fx = new double[width];
        for (int x = 0; x < width; x++) {
            double sx = (x + 0.5) * scaleX - 0.5;
            int i0 = (int) Math.floor(sx);
            fx[x] = sx - i0;
            x0[x] = clamp(i0, 0, srcW - 1);
            x1[x] = clamp(i0 + 1, 0, srcW - 1);
        }

        double[] rowTop = new double[srcW];
        double[] rowBottom = new double[srcW];
        double[] out = new double[width];
        for (int b = 0; b < bands; b++) {
            for (int y = 0; y < height; y++) {
                double sy = (y + 0.5) * scaleY - 0.5;
                int j0 = (int) Math.floor(sy);
                double fy = sy - j0;
                int yTop = clamp(j0, 0, srcH - 1);
                int yBottom = clamp(j0 + 1, 0, srcH - 1);
                src.getSamples(srcMinX, srcMinY + yTop, srcW, 1, b, rowTop);
                src.getSamples(srcMinX, srcMinY + yBottom, srcW, 1, b, rowBottom);
                for (int x = 0; x < width; x++) {
                    double top = rowTop[x0[x]] * (1 - fx[x]) + rowTop[x1[x]] * fx[x];
                    double bottom = rowBottom[x0[x]] * (1 - fx[x]) + rowBottom[x1[x]] * fx[x];
                    double value = top * (1 - fy) + bottom * fy;
                    out[x] = integral ? Math.rint(value) : value;
                }
                dst.setSamples(dst.getMinX(), dst.getMinY() + y, width, 1, b, out);
            }
        }
        return resized;
    }

    /**
     * Copies a source rectangle into a destination rectangle of a raster with
     * nearest-neighbour sampling, reproducing the flip and clipping behaviour of
     * {@code Graphics2D.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, null)}.
     * <p>
     * As in Java2D, a rectangle given with its second coordinate smaller than its
     * first means that axis is reversed, and a flip is applied when exactly one of
     * the source and destination rectangles is reversed along an axis. Samples
     * whose mapped source coordinate falls outside the source raster are skipped,
     * leaving the destination untouched there -- which is how a source rectangle
     * deliberately extended past the edge (the bottom-border reflection passes
     * {@code sy2 = -1}) ends up short by one row.
     */
    static void blit(
            Raster src, WritableRaster dst, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2) {
        int dxa = Math.min(dx1, dx2);
        int dya = Math.min(dy1, dy2);
        int sxa = Math.min(sx1, sx2);
        int sya = Math.min(sy1, sy2);
        int destW = Math.abs(dx2 - dx1);
        int destH = Math.abs(dy2 - dy1);
        int sampW = Math.abs(sx2 - sx1);
        int sampH = Math.abs(sy2 - sy1);
        if (destW <= 0 || destH <= 0 || sampW <= 0 || sampH <= 0) {
            return;
        }
        boolean flipX = (dx1 > dx2) != (sx1 > sx2);
        boolean flipY = (dy1 > dy2) != (sy1 > sy2);

        int srcMinX = src.getMinX();
        int srcMinY = src.getMinY();
        int srcMaxX = srcMinX + src.getWidth();
        int srcMaxY = srcMinY + src.getHeight();
        int dstMinX = dst.getMinX();
        int dstMinY = dst.getMinY();
        int dstMaxX = dstMinX + dst.getWidth();
        int dstMaxY = dstMinY + dst.getHeight();

        // Source coordinate for each destination column. Both bound checks are
        // monotonic in the column index, so the writable columns form one run;
        // the loop below stops at the first gap rather than assuming that.
        int[] srcX = new int[destW];
        int firstCol = -1;
        int lastCol = -1;
        for (int i = 0; i < destW; i++) {
            int xDst = dxa + i;
            int si = flipX ? sampW - 1 - i : i;
            int xSrc = sxa + (sampW == destW ? si : (int) ((si + 0.5) * sampW / destW));
            boolean ok = xDst >= dstMinX && xDst < dstMaxX && xSrc >= srcMinX && xSrc < srcMaxX;
            srcX[i] = ok ? xSrc : -1;
            if (ok) {
                if (firstCol < 0) {
                    firstCol = i;
                }
                lastCol = i;
            } else if (firstCol >= 0) {
                break;
            }
        }
        if (firstCol < 0) {
            return;
        }
        int runLength = lastCol - firstCol + 1;

        int bands = Math.min(src.getNumBands(), dst.getNumBands());
        double[] srcRow = new double[src.getWidth()];
        double[] outRow = new double[runLength];
        for (int b = 0; b < bands; b++) {
            for (int j = 0; j < destH; j++) {
                int yDst = dya + j;
                int sj = flipY ? sampH - 1 - j : j;
                int ySrc = sya + (sampH == destH ? sj : (int) ((sj + 0.5) * sampH / destH));
                if (yDst < dstMinY || yDst >= dstMaxY || ySrc < srcMinY || ySrc >= srcMaxY) {
                    continue;
                }
                src.getSamples(srcMinX, ySrc, src.getWidth(), 1, b, srcRow);
                for (int i = firstCol; i <= lastCol; i++) {
                    outRow[i - firstCol] = srcRow[srcX[i] - srcMinX];
                }
                dst.setSamples(dxa + firstCol, yDst, runLength, 1, b, outRow);
            }
        }
    }

    private static boolean isIntegral(Raster raster) {
        int type = raster.getDataBuffer().getDataType();
        return type != DataBuffer.TYPE_FLOAT && type != DataBuffer.TYPE_DOUBLE;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
