package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.analysis.images.SimpleImage;
import qupath.lib.analysis.images.SimpleImages;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Adjusts annotations within a single training tile based on model predictions.
 * <p>
 * Uses the prediction argmax and confidence maps saved during evaluation to
 * determine where the model disagrees with the ground truth. Pixels above
 * the confidence threshold are reassigned to the model's predicted class.
 * <p>
 * Only modifies annotation geometry within the tile boundary -- annotations
 * extending beyond the tile are surgically clipped and preserved outside.
 * <p>
 * Supports undo by tracking the removed and added annotation objects so the
 * caller can reverse the operation.
 *
 * @author UW-LOCI
 * @since 0.4.0
 */
public class AnnotationAdjuster {

    private static final Logger logger = LoggerFactory.getLogger(AnnotationAdjuster.class);

    /** Value used in masks for unlabeled pixels (must match AnnotationExtractor). */
    private static final int UNLABELED_INDEX = 255;

    private final double downsample;
    private final int patchSize;
    private final List<String> classNames;

    // Undo state from the most recent adjustment
    private UndoSnapshot lastUndo;

    /**
     * @param downsample training downsample factor
     * @param patchSize  training patch size in pixels at the downsampled level
     * @param classNames ordered list of class names (index 0, 1, 2, ...)
     */
    public AnnotationAdjuster(double downsample, int patchSize, List<String> classNames) {
        this.downsample = downsample;
        this.patchSize = patchSize;
        this.classNames = List.copyOf(classNames);
    }

    /**
     * Computes the adjustment preview without modifying any annotations.
     * Returns an RGBA BufferedImage showing which pixels would change,
     * colored by the class they would be reassigned to.
     *
     * @param predictionMapPath path to the argmax prediction PNG (uint8)
     * @param confidenceMapPath path to the confidence PNG (uint8 0-255)
     * @param groundTruthMaskPath path to the ground truth mask PNG
     * @param confidenceThreshold minimum confidence to accept prediction (0.0-1.0)
     * @param classColors map of class name to packed RGB color for rendering
     * @return preview image and statistics about the proposed changes
     * @throws IOException if any map file cannot be read
     */
    public PreviewResult computePreview(String predictionMapPath,
                                         String confidenceMapPath,
                                         String groundTruthMaskPath,
                                         double confidenceThreshold,
                                         Map<String, Integer> classColors)
            throws IOException {
        int[][] predMap = loadGrayscaleMap(predictionMapPath, "prediction");
        int[][] confMap = loadGrayscaleMap(confidenceMapPath, "confidence");
        int[][] gtMask = loadGrayscaleMap(groundTruthMaskPath, "ground truth");

        int h = predMap.length;
        int w = predMap[0].length;

        // Build the adjusted mask and track changes
        int[][] adjustedMask = new int[h][w];
        int totalChanged = 0;
        Map<String, Integer> changesPerClass = new LinkedHashMap<>();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int gt = gtMask[y][x];

                // Never create annotations in unlabeled areas
                if (gt == UNLABELED_INDEX || gt >= classNames.size()) {
                    adjustedMask[y][x] = gt;
                    continue;
                }

                int pred = predMap[y][x];
                double conf = confMap[y][x] / 255.0;

                if (conf >= confidenceThreshold && pred != gt && pred < classNames.size()) {
                    adjustedMask[y][x] = pred;
                    totalChanged++;
                    String className = classNames.get(pred);
                    changesPerClass.merge(className, 1, Integer::sum);
                } else {
                    adjustedMask[y][x] = gt;
                }
            }
        }

        // Build RGBA preview image: changed pixels colored by new class, others transparent
        BufferedImage preview = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (adjustedMask[y][x] != gtMask[y][x]) {
                    int pred = adjustedMask[y][x];
                    String className = pred < classNames.size() ? classNames.get(pred) : "";
                    int packed = classColors.getOrDefault(className, 0xFF00FF) & 0xFFFFFF;
                    preview.setRGB(x, y, 0xFF000000 | packed); // fully opaque
                } else {
                    preview.setRGB(x, y, 0x00000000); // transparent
                }
            }
        }

        return new PreviewResult(preview, adjustedMask, totalChanged, changesPerClass);
    }

    /**
     * Applies the annotation adjustment to the QuPath hierarchy.
     * <p>
     * Uses {@link RoiTools} for all ROI operations (intersection, difference)
     * and {@link ContourTracing#createTracedROI} for mask-to-ROI conversion,
     * avoiding direct JTS geometry manipulation.
     *
     * @param viewer   the active QuPath viewer
     * @param tileX    tile top-left X in full-resolution image coordinates
     * @param tileY    tile top-left Y in full-resolution image coordinates
     * @param adjustedMask the adjusted class mask (from {@link #computePreview})
     * @return result with counts of added/removed annotations
     */
    public AdjustmentResult applyAdjustment(QuPathViewer viewer,
                                             int tileX, int tileY,
                                             int[][] adjustedMask) {
        ImageData<?> imageData = viewer.getImageData();
        if (imageData == null) {
            return new AdjustmentResult(0, 0, 0, "No image data in viewer");
        }

        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        int regionSize = (int) (patchSize * downsample);

        // Tile boundary ROI for clipping operations
        ROI tileROI = ROIs.createRectangleROI(tileX, tileY, regionSize, regionSize,
                ImagePlane.getDefaultPlane());

        // RegionRequest for ContourTracing (maps pixel coords to image coords)
        RegionRequest region = RegionRequest.createInstance(
                imageData.getServer().getPath(), downsample,
                tileX, tileY, regionSize, regionSize);

        int h = adjustedMask.length;
        int w = adjustedMask[0].length;

        // --- Step 1: Trace new annotations from the adjusted mask ---
        List<PathObject> newAnnotations = new ArrayList<>();
        Set<Integer> classesPresent = new HashSet<>();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int v = adjustedMask[y][x];
                if (v != UNLABELED_INDEX && v < classNames.size()) {
                    classesPresent.add(v);
                }
            }
        }

        for (int classIdx : classesPresent) {
            String className = classNames.get(classIdx);
            PathClass pathClass = PathClass.fromString(className);

            // Create SimpleImage for this class
            float[] data = new float[w * h];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    data[y * w + x] = adjustedMask[y][x] == classIdx ? classIdx : 0;
                }
            }
            SimpleImage classImage = SimpleImages.createFloatImage(data, w, h);

            // Trace mask directly to ROI (no JTS geometry intermediate)
            ROI tracedROI = ContourTracing.createTracedROI(
                    classImage, classIdx, classIdx, region);
            if (tracedROI == null || tracedROI.isEmpty()) continue;

            // Clip to tile boundary (safety measure)
            ROI clippedROI = RoiTools.intersection(tracedROI, tileROI);
            if (clippedROI == null || clippedROI.isEmpty()) continue;

            PathObject annotation = PathObjects.createAnnotationObject(clippedROI, pathClass);
            newAnnotations.add(annotation);
        }

        // --- Step 2: Find existing annotations overlapping the tile ---
        List<PathObject> overlapping = new ArrayList<>();
        for (PathObject ann : hierarchy.getAnnotationObjects()) {
            if (ann.getPathClass() == null) continue;
            ROI annROI = ann.getROI();
            if (annROI == null) continue;

            // Quick bounding-box pre-filter before the full intersection test
            double ax = annROI.getBoundsX();
            double ay = annROI.getBoundsY();
            double ax2 = ax + annROI.getBoundsWidth();
            double ay2 = ay + annROI.getBoundsHeight();
            if (ax2 <= tileX || ax >= tileX + regionSize
                    || ay2 <= tileY || ay >= tileY + regionSize) {
                continue;
            }

            // Full intersection check via RoiTools
            try {
                if (RoiTools.intersectionArea(annROI, tileROI) > 0) {
                    overlapping.add(ann);
                }
            } catch (Exception e) {
                // Degenerate ROI; skip
            }
        }

        // --- Step 3: Surgically remove old annotation portions within the tile ---
        List<PathObject> removed = new ArrayList<>();
        Map<PathObject, PathObject> replaced = new LinkedHashMap<>();

        for (PathObject ann : overlapping) {
            try {
                ROI outerROI = RoiTools.difference(ann.getROI(), tileROI);
                if (outerROI == null || outerROI.isEmpty()) {
                    // Entire annotation is within the tile: remove it
                    removed.add(ann);
                } else {
                    // Annotation extends beyond tile: keep the outer part
                    PathObject replacement = PathObjects.createAnnotationObject(
                            outerROI, ann.getPathClass());
                    replaced.put(ann, replacement);
                }
            } catch (Exception e) {
                logger.warn("ROI difference failed for annotation {}: {}",
                        ann.getPathClass(), e.getMessage());
            }
        }

        // --- Step 4: Apply changes to hierarchy ---
        List<PathObject> toRemove = new ArrayList<>(removed);
        toRemove.addAll(replaced.keySet());
        hierarchy.removeObjects(toRemove, false);

        List<PathObject> outerReplacements = new ArrayList<>(replaced.values());
        hierarchy.addObjects(outerReplacements);
        hierarchy.addObjects(newAnnotations);

        hierarchy.fireHierarchyChangedEvent(this);

        // Save undo state
        lastUndo = new UndoSnapshot(hierarchy, toRemove, outerReplacements,
                newAnnotations, replaced);

        int totalRemoved = removed.size();
        int totalClipped = replaced.size();
        int totalAdded = newAnnotations.size();

        logger.info("Annotation adjustment applied: {} removed, {} clipped, {} added "
                + "within tile ({},{})", totalRemoved, totalClipped, totalAdded, tileX, tileY);

        return new AdjustmentResult(totalRemoved + totalClipped, totalAdded,
                countChangedPixels(adjustedMask),
                String.format("Removed %d, clipped %d, added %d annotations",
                        totalRemoved, totalClipped, totalAdded));
    }

    /**
     * Undoes the most recent annotation adjustment, restoring the original
     * annotations. Returns false if there is nothing to undo.
     */
    public boolean undoLastAdjustment() {
        if (lastUndo == null) {
            return false;
        }
        UndoSnapshot undo = lastUndo;
        lastUndo = null;

        PathObjectHierarchy hierarchy = undo.hierarchy;

        // Remove the new annotations and outer replacements we added
        List<PathObject> toRemove = new ArrayList<>();
        toRemove.addAll(undo.addedAnnotations);
        toRemove.addAll(undo.outerReplacements);
        hierarchy.removeObjects(toRemove, false);

        // Re-add the original annotations we removed
        hierarchy.addObjects(undo.removedAnnotations);

        hierarchy.fireHierarchyChangedEvent(this);

        logger.info("Annotation adjustment undone: restored {} annotations",
                undo.removedAnnotations.size());
        return true;
    }

    /** Whether an undo operation is available. */
    public boolean canUndo() {
        return lastUndo != null;
    }

    // ==================== Data Classes ====================

    /**
     * Result of computing a preview (before applying changes).
     */
    public record PreviewResult(
            BufferedImage previewImage,
            int[][] adjustedMask,
            int totalChangedPixels,
            Map<String, Integer> changesPerClass
    ) {}

    /**
     * Result after applying an annotation adjustment.
     */
    public record AdjustmentResult(
            int annotationsModified,
            int annotationsAdded,
            int pixelsChanged,
            String summary
    ) {}

    // ==================== Internal ====================

    private record UndoSnapshot(
            PathObjectHierarchy hierarchy,
            List<PathObject> removedAnnotations,
            List<PathObject> outerReplacements,
            List<PathObject> addedAnnotations,
            Map<PathObject, PathObject> replacementMap
    ) {}

    /**
     * Loads a grayscale (8-bit) PNG as an int[][height][width] array.
     */
    private static int[][] loadGrayscaleMap(String path, String kind) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IOException("No " + kind + " map path provided");
        }
        File file = new File(path);
        if (!file.exists()) {
            throw new IOException(kind + " map not found: " + path);
        }
        BufferedImage img = ImageIO.read(file);
        if (img == null) {
            throw new IOException("Failed to decode " + kind + " map: " + path);
        }
        int w = img.getWidth();
        int h = img.getHeight();
        int[][] map = new int[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // Extract the first channel value (grayscale)
                map[y][x] = img.getRaster().getSample(x, y, 0);
            }
        }
        return map;
    }

    /**
     * Counts the number of pixels that differ from UNLABELED in the adjusted mask.
     * Used for summary statistics.
     */
    private int countChangedPixels(int[][] adjustedMask) {
        int count = 0;
        for (int[] row : adjustedMask) {
            for (int v : row) {
                if (v != UNLABELED_INDEX) count++;
            }
        }
        return count;
    }
}
