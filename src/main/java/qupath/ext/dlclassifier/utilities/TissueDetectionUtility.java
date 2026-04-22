package qupath.ext.dlclassifier.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.analysis.images.SimpleImage;
import qupath.lib.analysis.images.SimpleImages;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;

/**
 * Generates coarse "Tissue" annotations on whole-slide images so that the
 * supervised patch-extraction pipeline can be reused for MAE pretraining
 * without training on empty slide space.
 *
 * <p>Simple fixed-threshold or Otsu thresholding on a low-resolution
 * grayscale macro view, with optional Gaussian blur and a tiny
 * open-then-close morphological cleanup. The resulting binary mask is
 * vectorised via {@link ContourTracing#createTracedROI} and saved as one
 * {@link PathObject} per connected component under the class
 * {@code "Tissue"}.
 *
 * <p>Not a precise tissue detector -- the goal is only to reject
 * background. For H&amp;E this is good enough in the majority of cases;
 * IHC and fluorescence may need the fixed-threshold mode with tuning.
 *
 * <p>Also callable as a script -- {@link #detectTissue(Collection,
 * TissueDetectionParams, Consumer)} takes a plain parameters bean and
 * a progress callback.
 *
 * @author UW-LOCI
 * @since 0.6.3
 */
public final class TissueDetectionUtility {

    private static final Logger logger = LoggerFactory.getLogger(TissueDetectionUtility.class);

    /** Name of the {@link PathClass} created by this utility. */
    public static final String TISSUE_CLASS_NAME = "Tissue";

    private TissueDetectionUtility() {}

    /** Threshold methods supported for background vs. tissue segmentation. */
    public enum ThresholdMethod {
        /** Otsu threshold on the blurred grayscale histogram. */
        OTSU,
        /** Fixed grayscale threshold (0-255). Tissue is darker than the cutoff. */
        FIXED
    }

    /**
     * Plain parameters bean for a batch tissue-detection run. Lets the
     * method signature stay stable while we add knobs later.
     */
    public static final class TissueDetectionParams {
        /** Downsample factor used when reading the macro view. */
        public double downsample = 16.0;
        /** Gaussian blur sigma in microns. {@code 0} disables blur. */
        public double sigmaMicrons = 5.0;
        /** Threshold method. */
        public ThresholdMethod method = ThresholdMethod.OTSU;
        /** Fixed grayscale threshold (0-255), used only when {@link #method} is {@link ThresholdMethod#FIXED}. */
        public int fixedThreshold = 200;
        /**
         * Clamp any computed Otsu threshold to this {@code [min, max]}
         * grayscale range so an all-one-value image doesn't produce a
         * degenerate mask covering everything or nothing.
         */
        public int otsuMin = 20;
        /** See {@link #otsuMin}. */
        public int otsuMax = 220;
        /** Discard connected components smaller than this in microns^2. */
        public double minAreaMicronsSq = 100_000.0; // about 0.1 mm^2
        /** Remove any pre-existing "Tissue"-classed annotations on each image before adding new ones. */
        public boolean replaceExisting = true;
    }

    /**
     * Runs tissue detection on each entry in turn, saving the resulting
     * annotations back to the project. Blocks on the calling thread --
     * callers should run on a background worker.
     *
     * @param entries   project image entries to process
     * @param params    detection parameters (see {@link TissueDetectionParams})
     * @param progress  callback receiving a short per-image status line;
     *                  may be {@code null}
     * @return total number of tissue annotations created across all entries
     */
    public static int detectTissue(
            Collection<ProjectImageEntry<BufferedImage>> entries,
            TissueDetectionParams params,
            Consumer<String> progress) {

        int totalAnnotations = 0;
        int failed = 0;
        PathClass tissueClass = PathClass.getInstance(TISSUE_CLASS_NAME);

        int index = 0;
        int total = entries.size();
        for (ProjectImageEntry<BufferedImage> entry : entries) {
            index++;
            String name = entry.getImageName();
            if (progress != null) {
                progress.accept(String.format("Detecting tissue [%d/%d]: %s", index, total, name));
            }
            try {
                int count = detectTissueForEntry(entry, tissueClass, params);
                totalAnnotations += count;
                logger.info("Tissue detection on '{}': {} annotation(s) created", name, count);
            } catch (Exception ex) {
                failed++;
                logger.warn("Tissue detection failed for '{}': {}", name, ex.toString(), ex);
            }
        }
        if (progress != null) {
            progress.accept(String.format(
                    "Tissue detection complete: %d annotation(s) across %d image(s)%s",
                    totalAnnotations, total - failed,
                    failed > 0 ? ", " + failed + " image(s) failed (see log)" : ""));
        }
        return totalAnnotations;
    }

    private static int detectTissueForEntry(
            ProjectImageEntry<BufferedImage> entry,
            PathClass tissueClass,
            TissueDetectionParams params) throws Exception {

        ImageData<BufferedImage> imageData = entry.readImageData();
        try {
            ImageServer<BufferedImage> server = imageData.getServer();
            int fullW = server.getWidth();
            int fullH = server.getHeight();
            double downsample = Math.max(1.0, params.downsample);

            // Clamp requested downsample to something the server can actually
            // deliver without blowing memory on huge slides.
            RegionRequest region = RegionRequest.createInstance(
                    server.getPath(), downsample, 0, 0, fullW, fullH);
            BufferedImage thumbnail = server.readRegion(region);
            if (thumbnail == null) {
                logger.warn("Server returned null thumbnail for '{}' at downsample {}",
                        entry.getImageName(), downsample);
                return 0;
            }

            int w = thumbnail.getWidth();
            int h = thumbnail.getHeight();

            // Grayscale (mean of bands for RGB; max across bands for fluorescence-style
            // multichannel -- tissue is typically brighter in fluorescence, darker in RGB,
            // so we invert after to always produce "tissue > threshold").
            Raster raster = thumbnail.getRaster();
            int numBands = raster.getNumBands();
            float[] gray = new float[w * h];
            boolean isRgbLike = numBands >= 3;
            int[] pixel = new int[numBands];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    raster.getPixel(x, y, pixel);
                    if (isRgbLike) {
                        // RGB: tissue darker than background. Use mean and invert.
                        float mean = (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
                        gray[y * w + x] = 255.0f - mean; // tissue -> high
                    } else {
                        // Single/multi-channel non-RGB: assume tissue brighter than background.
                        int max = 0;
                        for (int b = 0; b < numBands; b++) max = Math.max(max, pixel[b]);
                        gray[y * w + x] = Math.min(255, max);
                    }
                }
            }

            // Optional Gaussian blur. Derived pixel sigma uses the thumbnail's effective
            // pixel size (full-res micron/pixel times downsample).
            PixelCalibration cal = server.getPixelCalibration();
            double mppFull = cal != null ? cal.getAveragedPixelSizeMicrons() : Double.NaN;
            double sigmaPx = 0.0;
            if (params.sigmaMicrons > 0 && Double.isFinite(mppFull) && mppFull > 0) {
                double mppThumb = mppFull * downsample;
                sigmaPx = params.sigmaMicrons / mppThumb;
            }
            if (sigmaPx >= 0.5) {
                gaussianBlurSeparable(gray, w, h, sigmaPx);
            }

            // Threshold.
            int threshold;
            if (params.method == ThresholdMethod.FIXED) {
                // User input is in original-image terms (darker = tissue for RGB),
                // but `gray` is already inverted for RGB. Keep the user's intuition
                // by inverting the fixed threshold in the RGB case.
                threshold = isRgbLike
                        ? Math.max(0, Math.min(255, 255 - params.fixedThreshold))
                        : Math.max(0, Math.min(255, params.fixedThreshold));
            } else {
                int otsu = otsuThreshold(gray);
                threshold = Math.max(params.otsuMin, Math.min(params.otsuMax, otsu));
                logger.debug("Otsu threshold for '{}': {} (clamped to [{}, {}] = {})",
                        entry.getImageName(), otsu, params.otsuMin, params.otsuMax, threshold);
            }

            // Binarise.
            float[] mask = new float[w * h];
            for (int i = 0; i < mask.length; i++) {
                mask[i] = gray[i] > threshold ? 1.0f : 0.0f;
            }

            // Tiny morphological cleanup: open (erode then dilate) removes single-pixel
            // noise; close (dilate then erode) fills single-pixel holes.
            morphologicalOpen(mask, w, h);
            morphologicalClose(mask, w, h);

            // Trace to an ROI in full-resolution coordinates.
            SimpleImage classImage = SimpleImages.createFloatImage(mask, w, h);
            ROI tissueROI = ContourTracing.createTracedROI(classImage, 1.0, 1.0, region);
            if (tissueROI == null || tissueROI.isEmpty()) {
                logger.info("No tissue detected on '{}'", entry.getImageName());
                return 0;
            }

            // Split the ROI into per-component annotations so the user can
            // delete individual slide corners / smears / floaters.
            List<ROI> components = qupath.lib.roi.RoiTools.splitROI(tissueROI);
            if (components == null || components.isEmpty()) components = List.of(tissueROI);

            double minPxSqAtThumb = 0;
            if (Double.isFinite(mppFull) && mppFull > 0) {
                double mppThumb = mppFull * downsample;
                minPxSqAtThumb = params.minAreaMicronsSq / (mppThumb * mppThumb);
            }
            // Component areas are in full-resolution pixel units (because the ROI
            // was scaled by ContourTracing).  Convert min area to full-res pixels.
            double minPxSqFull;
            if (Double.isFinite(mppFull) && mppFull > 0) {
                minPxSqFull = params.minAreaMicronsSq / (mppFull * mppFull);
            } else {
                // No calibration available -- fall back to a size in thumbnail
                // pixels scaled up by downsample^2. Treat minAreaMicronsSq as
                // a raw pixel count in that case (not ideal; documented).
                minPxSqFull = params.minAreaMicronsSq;
            }

            List<PathObject> newAnnotations = new ArrayList<>();
            for (ROI comp : components) {
                double areaPx = comp.getArea();
                if (areaPx < minPxSqFull) continue;
                newAnnotations.add(PathObjects.createAnnotationObject(comp, tissueClass));
            }
            if (newAnnotations.isEmpty()) {
                logger.info("Tissue detected on '{}' but all components below minimum area ({} um^2)",
                        entry.getImageName(), params.minAreaMicronsSq);
                return 0;
            }

            // Apply to hierarchy.
            PathObjectHierarchy hierarchy = imageData.getHierarchy();
            if (params.replaceExisting) {
                List<PathObject> toRemove = new ArrayList<>();
                for (PathObject po : hierarchy.getAnnotationObjects()) {
                    if (po.getPathClass() != null
                            && TISSUE_CLASS_NAME.equals(po.getPathClass().getName())) {
                        toRemove.add(po);
                    }
                }
                if (!toRemove.isEmpty()) {
                    hierarchy.removeObjects(toRemove, false);
                }
            }
            hierarchy.addObjects(newAnnotations);
            entry.saveImageData(imageData);
            return newAnnotations.size();
        } finally {
            if (imageData.getServer() != null) {
                try {
                    imageData.getServer().close();
                } catch (Exception ignored) {}
            }
        }
    }

    // --- Otsu threshold ---

    private static int otsuThreshold(float[] values) {
        int[] hist = new int[256];
        for (float v : values) {
            int idx = v < 0 ? 0 : v > 255 ? 255 : (int) v;
            hist[idx]++;
        }
        int total = values.length;
        double sum = 0;
        for (int t = 0; t < 256; t++) sum += t * hist[t];

        double sumB = 0;
        int wB = 0;
        double maxVar = -1;
        int bestT = 127;
        for (int t = 0; t < 256; t++) {
            wB += hist[t];
            if (wB == 0) continue;
            int wF = total - wB;
            if (wF == 0) break;
            sumB += t * hist[t];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double between = (double) wB * wF * (mB - mF) * (mB - mF);
            if (between > maxVar) {
                maxVar = between;
                bestT = t;
            }
        }
        return bestT;
    }

    // --- Separable gaussian blur ---

    private static void gaussianBlurSeparable(float[] data, int w, int h, double sigma) {
        int r = Math.max(1, (int) Math.ceil(3.0 * sigma));
        double[] k = new double[2 * r + 1];
        double sum = 0;
        double twoSigmaSq = 2.0 * sigma * sigma;
        for (int i = -r; i <= r; i++) {
            double v = Math.exp(-(i * i) / twoSigmaSq);
            k[i + r] = v;
            sum += v;
        }
        for (int i = 0; i < k.length; i++) k[i] /= sum;

        float[] tmp = new float[w * h];
        // Horizontal pass
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double acc = 0;
                for (int i = -r; i <= r; i++) {
                    int xi = Math.max(0, Math.min(w - 1, x + i));
                    acc += data[y * w + xi] * k[i + r];
                }
                tmp[y * w + x] = (float) acc;
            }
        }
        // Vertical pass
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double acc = 0;
                for (int i = -r; i <= r; i++) {
                    int yi = Math.max(0, Math.min(h - 1, y + i));
                    acc += tmp[yi * w + x] * k[i + r];
                }
                data[y * w + x] = (float) acc;
            }
        }
    }

    // --- Morphology (3x3 square SE) ---

    private static void morphologicalOpen(float[] mask, int w, int h) {
        float[] tmp = new float[mask.length];
        erode3x3(mask, tmp, w, h);
        dilate3x3(tmp, mask, w, h);
    }

    private static void morphologicalClose(float[] mask, int w, int h) {
        float[] tmp = new float[mask.length];
        dilate3x3(mask, tmp, w, h);
        erode3x3(tmp, mask, w, h);
    }

    private static void erode3x3(float[] src, float[] dst, int w, int h) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float v = 1.0f;
                for (int dy = -1; dy <= 1 && v > 0; dy++) {
                    int yy = Math.max(0, Math.min(h - 1, y + dy));
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = Math.max(0, Math.min(w - 1, x + dx));
                        if (src[yy * w + xx] < 0.5f) { v = 0.0f; break; }
                    }
                }
                dst[y * w + x] = v;
            }
        }
    }

    private static void dilate3x3(float[] src, float[] dst, int w, int h) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float v = 0.0f;
                for (int dy = -1; dy <= 1 && v < 1; dy++) {
                    int yy = Math.max(0, Math.min(h - 1, y + dy));
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = Math.max(0, Math.min(w - 1, x + dx));
                        if (src[yy * w + xx] > 0.5f) { v = 1.0f; break; }
                    }
                }
                dst[y * w + x] = v;
            }
        }
    }
}
