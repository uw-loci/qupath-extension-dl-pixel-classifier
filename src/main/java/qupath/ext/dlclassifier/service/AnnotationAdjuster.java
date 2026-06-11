package qupath.ext.dlclassifier.service;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import javax.imageio.ImageIO;
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

    /**
     * Stroke width applied when rasterising line ROIs into the "current
     * annotations" mask. Matches the typical training default; differences
     * here only affect users with line annotations in a tile being re-adjusted.
     */
    private static final double LINE_STROKE_WIDTH = 8.0;

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
     * <p>
     * The "from" mask is built by rasterising the CURRENT annotations in the
     * viewer for the tile region -- not the saved training-time GT mask. This
     * way, a redo of the same tile sees the post-adjustment annotation state,
     * so the per-(from, to) transition breakdown stays accurate.
     *
     * @param viewer the active QuPath viewer (used to read live annotations)
     * @param tileX tile top-left X in full-resolution image coordinates
     * @param tileY tile top-left Y in full-resolution image coordinates
     * @param predictionMapPath path to the argmax prediction PNG (uint8)
     * @param confidenceMapPath path to the confidence PNG (uint8 0-255)
     * @param confidenceThreshold minimum confidence to accept prediction (0.0-1.0)
     * @param classColors map of class name to packed RGB color for rendering
     * @return preview image, full unfiltered mask, per-class + per-transition counts
     * @throws IOException if any map file cannot be read
     */
    public PreviewResult computePreview(
            QuPathViewer viewer,
            int tileX,
            int tileY,
            String predictionMapPath,
            String confidenceMapPath,
            double confidenceThreshold,
            Map<String, Integer> classColors)
            throws IOException {
        PreviewSession session = beginPreviewSession(viewer, tileX, tileY, predictionMapPath, confidenceMapPath);
        return previewAtThreshold(session, confidenceThreshold, classColors);
    }

    /**
     * Loads the prediction + confidence maps and rasterizes the current
     * annotations ONCE, returning a reusable {@link PreviewSession}.
     * <p>
     * The expensive step is rasterizing the live annotations; caching it lets
     * the confidence slider re-threshold the preview live (every drag event)
     * via {@link #previewAtThreshold} without touching the viewer or disk again.
     *
     * @param predictionMapPath path to the argmax prediction PNG (uint8)
     * @param confidenceMapPath path to the confidence PNG (uint8 0-255)
     * @throws IOException if any map file cannot be read
     */
    public PreviewSession beginPreviewSession(
            QuPathViewer viewer, int tileX, int tileY, String predictionMapPath, String confidenceMapPath)
            throws IOException {
        int[][] predMap = loadGrayscaleMap(predictionMapPath, "prediction");
        int[][] confMap = loadGrayscaleMap(confidenceMapPath, "confidence");

        // Crop padded maps to the center patchSize x patchSize region.
        // Training tiles exported with context padding are larger than patchSize
        // (e.g. 552x552 for 512 + 2*20 padding). If not cropped, ContourTracing
        // in applyAdjustment() maps pixel coordinates into the wrong image
        // region (shifted by the padding offset).
        int rawH = predMap.length;
        int rawW = predMap[0].length;
        predMap = cropToCenter(predMap, patchSize);
        confMap = cropToCenter(confMap, patchSize);
        if (predMap.length != rawH || predMap[0].length != rawW) {
            int padding = (rawW - patchSize) / 2;
            logger.info(
                    "Cropped padded maps from {}x{} to {}x{} (contextPadding={})",
                    rawW,
                    rawH,
                    patchSize,
                    patchSize,
                    padding);
        }

        int[][] currentMask = rasterizeCurrentAnnotations(viewer, tileX, tileY);
        return new PreviewSession(predMap, confMap, currentMask);
    }

    /**
     * Re-thresholds a cached {@link PreviewSession} into a {@link PreviewResult}.
     * Cheap (a single pass over already-loaded arrays), so it is safe to call on
     * every confidence-slider drag event for a live preview.
     */
    public PreviewResult previewAtThreshold(
            PreviewSession session, double confidenceThreshold, Map<String, Integer> classColors) {
        int[][] predMap = session.predMap;
        int[][] confMap = session.confMap;
        int[][] currentMask = session.currentMask;
        int h = session.h;
        int w = session.w;

        // Build the FULL unfiltered adjusted mask + per-transition counts.
        // The UI can later filter these via rebuildPreviewFromFilter().
        int[][] adjustedMask = new int[h][w];
        int totalChanged = 0;
        Map<String, Integer> changesPerClass = new LinkedHashMap<>();
        Map<TransitionKey, Integer> changesPerTransition = new LinkedHashMap<>();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int from = currentMask[y][x];

                // Never create annotations in unlabeled areas.
                if (from == UNLABELED_INDEX || from >= classNames.size()) {
                    adjustedMask[y][x] = from;
                    continue;
                }

                int pred = predMap[y][x];
                double conf = confMap[y][x] / 255.0;

                if (conf >= confidenceThreshold && pred != from && pred < classNames.size()) {
                    adjustedMask[y][x] = pred;
                    totalChanged++;
                    String fromName = classNames.get(from);
                    String toName = classNames.get(pred);
                    changesPerClass.merge(toName, 1, Integer::sum);
                    changesPerTransition.merge(new TransitionKey(fromName, toName), 1, Integer::sum);
                } else {
                    adjustedMask[y][x] = from;
                }
            }
        }

        // Build RGBA preview image: changed pixels colored by new class, others transparent
        BufferedImage preview = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (adjustedMask[y][x] != currentMask[y][x]) {
                    int pred = adjustedMask[y][x];
                    String className = pred < classNames.size() ? classNames.get(pred) : "";
                    int packed = classColors.getOrDefault(className, 0xFF00FF) & 0xFFFFFF;
                    preview.setRGB(x, y, 0xFF000000 | packed); // fully opaque
                } else {
                    preview.setRGB(x, y, 0x00000000); // transparent
                }
            }
        }

        return new PreviewResult(
                preview, adjustedMask, currentMask, totalChanged, changesPerClass, changesPerTransition);
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
    public AdjustmentResult applyAdjustment(QuPathViewer viewer, int tileX, int tileY, int[][] adjustedMask) {
        ImageData<?> imageData = viewer.getImageData();
        if (imageData == null) {
            return new AdjustmentResult(0, 0, 0, "No image data in viewer");
        }

        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        int regionSize = (int) (patchSize * downsample);

        // Tile boundary ROI for clipping operations
        ROI tileROI = ROIs.createRectangleROI(tileX, tileY, regionSize, regionSize, ImagePlane.getDefaultPlane());

        // RegionRequest for ContourTracing (maps pixel coords to image coords)
        RegionRequest region = RegionRequest.createInstance(
                imageData.getServer().getPath(), downsample, tileX, tileY, regionSize, regionSize);

        int h = adjustedMask.length;
        int w = adjustedMask[0].length;

        // --- Step 1: Trace new in-tile ROIs from the adjusted mask, per class.
        //
        // Use a binary 0/1 mask with thresholds (0.5, 1.5) so classIdx == 0
        // does NOT collapse to an all-zero image (which would cause
        // ContourTracing to trace the entire tile as a single giant square).
        Set<Integer> classesPresent = new HashSet<>();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int v = adjustedMask[y][x];
                if (v != UNLABELED_INDEX && v < classNames.size()) {
                    classesPresent.add(v);
                }
            }
        }

        Map<Integer, ROI> newClassROIs = new LinkedHashMap<>();
        for (int classIdx : classesPresent) {
            float[] data = new float[w * h];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    data[y * w + x] = adjustedMask[y][x] == classIdx ? 1.0f : 0.0f;
                }
            }
            SimpleImage classImage = SimpleImages.createFloatImage(data, w, h);
            ROI tracedROI = ContourTracing.createTracedROI(classImage, 0.5, 1.5, region);
            if (tracedROI == null || tracedROI.isEmpty()) continue;

            ROI clippedROI = RoiTools.intersection(tracedROI, tileROI);
            if (clippedROI == null || clippedROI.isEmpty()) continue;
            newClassROIs.put(classIdx, clippedROI);
        }

        // Map class name -> classIdx for quick lookup when matching existing
        // annotations to a new in-tile ROI.
        Map<String, Integer> classNameToIdx = new HashMap<>();
        for (Map.Entry<Integer, ROI> e : newClassROIs.entrySet()) {
            classNameToIdx.put(classNames.get(e.getKey()), e.getKey());
        }

        // --- Step 2: Find existing annotations overlapping the tile ---
        List<PathObject> overlapping = new ArrayList<>();
        for (PathObject ann : hierarchy.getAnnotationObjects()) {
            if (ann.getPathClass() == null) continue;
            ROI annROI = ann.getROI();
            if (annROI == null) continue;

            double ax = annROI.getBoundsX();
            double ay = annROI.getBoundsY();
            double ax2 = ax + annROI.getBoundsWidth();
            double ay2 = ay + annROI.getBoundsHeight();
            if (ax2 <= tileX || ax >= tileX + regionSize || ay2 <= tileY || ay >= tileY + regionSize) {
                continue;
            }

            try {
                if (RoiTools.intersectionArea(annROI, tileROI) > 0) {
                    overlapping.add(ann);
                }
            } catch (Exception e) {
                // Degenerate ROI; skip
            }
        }

        // --- Step 3: Process overlapping annotations.
        //
        // For each existing annotation that overlaps the tile:
        //   * If the class has a new in-tile ROI, accumulate the outer-of-tile
        //     portion into the per-class merge bucket and mark the original
        //     for removal. The new in-tile ROI will absorb the outer piece.
        //   * Otherwise (class not affected by this adjustment), keep the
        //     outer-of-tile portion as a clipped replacement, preserving
        //     existing behaviour for unaffected classes.
        List<PathObject> removed = new ArrayList<>();
        Map<PathObject, PathObject> replaced = new LinkedHashMap<>();
        Map<Integer, List<ROI>> sameClassOuterPieces = new LinkedHashMap<>();
        List<PathObject> mergedAbsorbed = new ArrayList<>();

        for (PathObject ann : overlapping) {
            String annClassName = ann.getPathClass().toString();
            Integer mergeIdx = classNameToIdx.get(annClassName);

            if (mergeIdx != null) {
                // Same class as a new in-tile ROI: absorb the outer portion
                // into the merge bucket regardless of whether it is empty.
                try {
                    ROI outerROI = RoiTools.difference(ann.getROI(), tileROI);
                    if (outerROI != null && !outerROI.isEmpty()) {
                        sameClassOuterPieces
                                .computeIfAbsent(mergeIdx, k -> new ArrayList<>())
                                .add(outerROI);
                    }
                } catch (Exception e) {
                    logger.warn(
                            "ROI difference failed for same-class merge {}: {}", ann.getPathClass(), e.getMessage());
                }
                mergedAbsorbed.add(ann);
                continue;
            }

            // Class is unaffected: clip and keep the outer part.
            try {
                ROI outerROI = RoiTools.difference(ann.getROI(), tileROI);
                if (outerROI == null || outerROI.isEmpty()) {
                    removed.add(ann);
                } else {
                    PathObject replacement = PathObjects.createAnnotationObject(outerROI, ann.getPathClass());
                    replaced.put(ann, replacement);
                }
            } catch (Exception e) {
                logger.warn("ROI difference failed for annotation {}: {}", ann.getPathClass(), e.getMessage());
            }
        }

        // --- Step 4: Build the merged per-class annotations and apply.
        List<PathObject> newAnnotations = new ArrayList<>();
        for (Map.Entry<Integer, ROI> e : newClassROIs.entrySet()) {
            int classIdx = e.getKey();
            ROI mergedROI = e.getValue();
            List<ROI> outerPieces = sameClassOuterPieces.get(classIdx);
            if (outerPieces != null) {
                for (ROI outer : outerPieces) {
                    try {
                        mergedROI = RoiTools.union(List.of(mergedROI, outer));
                    } catch (Exception ex) {
                        logger.warn("ROI union failed for class {}: {}", classNames.get(classIdx), ex.getMessage());
                    }
                }
            }
            if (mergedROI == null || mergedROI.isEmpty()) continue;
            PathClass pathClass = PathClass.fromString(classNames.get(classIdx));
            newAnnotations.add(PathObjects.createAnnotationObject(mergedROI, pathClass));
        }

        List<PathObject> toRemove = new ArrayList<>(removed);
        toRemove.addAll(replaced.keySet());
        toRemove.addAll(mergedAbsorbed);
        hierarchy.removeObjects(toRemove, false);

        List<PathObject> outerReplacements = new ArrayList<>(replaced.values());
        hierarchy.addObjects(outerReplacements);
        hierarchy.addObjects(newAnnotations);

        hierarchy.fireHierarchyChangedEvent(this);

        lastUndo = new UndoSnapshot(hierarchy, toRemove, outerReplacements, newAnnotations, replaced);

        int totalRemoved = removed.size();
        int totalClipped = replaced.size();
        int totalMerged = mergedAbsorbed.size();
        int totalAdded = newAnnotations.size();

        logger.info(
                "Annotation adjustment applied: {} removed, {} clipped, "
                        + "{} merged into new same-class annotations, {} added "
                        + "within tile ({},{})",
                totalRemoved,
                totalClipped,
                totalMerged,
                totalAdded,
                tileX,
                tileY);

        return new AdjustmentResult(
                totalRemoved + totalClipped + totalMerged,
                totalAdded,
                countChangedPixels(adjustedMask),
                String.format(
                        "Removed %d, clipped %d, merged %d, added %d annotations",
                        totalRemoved, totalClipped, totalMerged, totalAdded));
    }

    /**
     * Rebuilds a preview from an existing one, keeping only the (from -> to)
     * transitions in {@code allowed}. Unchecked transitions revert to the
     * current annotation class. Used to update the overlay live as the user
     * toggles per-transition checkboxes in the dialog.
     */
    public PreviewResult rebuildPreviewFromFilter(
            PreviewResult original, Set<TransitionKey> allowed, Map<String, Integer> classColors) {
        int[][] current = original.currentMask();
        int[][] full = original.adjustedMask();
        int h = current.length;
        int w = current[0].length;

        int[][] filtered = new int[h][w];
        int totalChanged = 0;
        Map<String, Integer> changesPerClass = new LinkedHashMap<>();
        Map<TransitionKey, Integer> changesPerTransition = new LinkedHashMap<>();
        BufferedImage preview = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int from = current[y][x];
                int to = full[y][x];
                if (from == to || from == UNLABELED_INDEX || from >= classNames.size()) {
                    filtered[y][x] = from;
                    preview.setRGB(x, y, 0x00000000);
                    continue;
                }
                String fromName = classNames.get(from);
                String toName = to < classNames.size() ? classNames.get(to) : "";
                TransitionKey key = new TransitionKey(fromName, toName);
                if (allowed.contains(key)) {
                    filtered[y][x] = to;
                    totalChanged++;
                    changesPerClass.merge(toName, 1, Integer::sum);
                    changesPerTransition.merge(key, 1, Integer::sum);
                    int packed = classColors.getOrDefault(toName, 0xFF00FF) & 0xFFFFFF;
                    preview.setRGB(x, y, 0xFF000000 | packed);
                } else {
                    filtered[y][x] = from;
                    preview.setRGB(x, y, 0x00000000);
                }
            }
        }

        return new PreviewResult(preview, filtered, current, totalChanged, changesPerClass, changesPerTransition);
    }

    /**
     * Rasterises the current annotations in {@code viewer} for the tile region
     * into a {@code patchSize x patchSize} class-index mask. Pixels with no
     * annotation get {@link #UNLABELED_INDEX}. Mirrors the rendering pipeline
     * in {@code AnnotationExtractor.createCombinedMask} so the resulting mask
     * is comparable to the saved training-time GT mask.
     * <p>
     * Larger-area annotations are painted first so smaller (more specific)
     * annotations end up on top, matching typical class-precedence intent.
     */
    public int[][] rasterizeCurrentAnnotations(QuPathViewer viewer, int tileX, int tileY) {
        BufferedImage mask = new BufferedImage(patchSize, patchSize, BufferedImage.TYPE_BYTE_GRAY);
        byte[] maskBytes = ((java.awt.image.DataBufferByte) mask.getRaster().getDataBuffer()).getData();
        Arrays.fill(maskBytes, (byte) UNLABELED_INDEX);

        ImageData<?> imageData = viewer != null ? viewer.getImageData() : null;
        if (imageData == null) {
            return bytesToIntArray(maskBytes, patchSize);
        }

        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        int regionSize = (int) (patchSize * downsample);
        Rectangle patchBounds = new Rectangle(tileX, tileY, regionSize, regionSize);

        List<PathObject> overlapping = new ArrayList<>();
        for (PathObject ann : hierarchy.getAnnotationObjects()) {
            if (ann.getPathClass() == null) continue;
            ROI roi = ann.getROI();
            if (roi == null) continue;
            int expandedStroke = (int) Math.ceil(LINE_STROKE_WIDTH * downsample);
            Rectangle annBounds = new Rectangle(
                    (int) roi.getBoundsX(),
                    (int) roi.getBoundsY(),
                    (int) roi.getBoundsWidth() + expandedStroke,
                    (int) roi.getBoundsHeight() + expandedStroke);
            if (!patchBounds.intersects(annBounds)) continue;
            overlapping.add(ann);
        }
        // Largest-area first; smaller (typically more specific) annotations paint on top.
        overlapping.sort(
                (a, b) -> Double.compare(b.getROI().getArea(), a.getROI().getArea()));

        Graphics2D g2d = mask.createGraphics();
        try {
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
            g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
            AffineTransform originalTransform = g2d.getTransform();
            for (PathObject ann : overlapping) {
                String cls = ann.getPathClass().toString();
                int idx = classNames.indexOf(cls);
                if (idx < 0 || idx >= UNLABELED_INDEX) continue;

                g2d.setColor(new Color(idx, idx, idx));
                g2d.setTransform(originalTransform);
                g2d.scale(1.0 / downsample, 1.0 / downsample);
                g2d.translate(-tileX, -tileY);

                ROI roi = ann.getROI();
                Shape shape = roi.getShape();
                if (roi.isLine()) {
                    g2d.setStroke(new BasicStroke(
                            (float) (LINE_STROKE_WIDTH * downsample), BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                    g2d.draw(shape);
                } else {
                    g2d.fill(shape);
                }
            }
        } finally {
            g2d.dispose();
        }

        return bytesToIntArray(maskBytes, patchSize);
    }

    private static int[][] bytesToIntArray(byte[] bytes, int size) {
        int[][] out = new int[size][size];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                out[y][x] = bytes[y * size + x] & 0xFF;
            }
        }
        return out;
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

        logger.info("Annotation adjustment undone: restored {} annotations", undo.removedAnnotations.size());
        return true;
    }

    /** Whether an undo operation is available. */
    public boolean canUndo() {
        return lastUndo != null;
    }

    // ==================== Data Classes ====================

    /**
     * Cached per-tile inputs (loaded prediction + confidence maps and the
     * rasterized current-annotation mask) so the confidence slider can
     * re-threshold the preview live without re-reading disk or re-rendering the
     * viewer. Created by {@link #beginPreviewSession}; consumed by
     * {@link #previewAtThreshold}.
     */
    public static final class PreviewSession {
        private final int[][] predMap;
        private final int[][] confMap;
        private final int[][] currentMask;
        private final int h;
        private final int w;

        private PreviewSession(int[][] predMap, int[][] confMap, int[][] currentMask) {
            this.predMap = predMap;
            this.confMap = confMap;
            this.currentMask = currentMask;
            this.h = predMap.length;
            this.w = predMap.length > 0 ? predMap[0].length : 0;
        }
    }

    /**
     * Identifies a single (from-class -> to-class) transition. Used to break
     * down the changes by class pair and to filter which transitions get
     * applied.
     */
    public record TransitionKey(String fromClass, String toClass) {}

    /**
     * Result of computing a preview (before applying changes).
     *
     * @param previewImage RGBA overlay of changed pixels coloured by new class
     * @param adjustedMask the proposed new mask (after applying the filter)
     * @param currentMask  the live-annotation mask the proposal was computed
     *                     against; needed by {@link #rebuildPreviewFromFilter}
     * @param totalChangedPixels number of pixels that differ between
     *                           {@code currentMask} and {@code adjustedMask}
     * @param changesPerClass    pixel count by target class (legacy)
     * @param changesPerTransition pixel count by (from, to) transition pair
     */
    public record PreviewResult(
            BufferedImage previewImage,
            int[][] adjustedMask,
            int[][] currentMask,
            int totalChangedPixels,
            Map<String, Integer> changesPerClass,
            Map<TransitionKey, Integer> changesPerTransition) {}

    /**
     * Result after applying an annotation adjustment.
     */
    public record AdjustmentResult(int annotationsModified, int annotationsAdded, int pixelsChanged, String summary) {}

    // ==================== Internal ====================

    private record UndoSnapshot(
            PathObjectHierarchy hierarchy,
            List<PathObject> removedAnnotations,
            List<PathObject> outerReplacements,
            List<PathObject> addedAnnotations,
            Map<PathObject, PathObject> replacementMap) {}

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

    /**
     * Crops a 2D array to a centered region of the given target size.
     * If the array is already at or smaller than the target size, returns
     * it unchanged.
     */
    private static int[][] cropToCenter(int[][] map, int targetSize) {
        int h = map.length;
        int w = map[0].length;
        if (h <= targetSize && w <= targetSize) {
            return map;
        }
        int cropH = Math.min(h, targetSize);
        int cropW = Math.min(w, targetSize);
        int offY = (h - cropH) / 2;
        int offX = (w - cropW) / 2;
        int[][] cropped = new int[cropH][cropW];
        for (int y = 0; y < cropH; y++) {
            System.arraycopy(map[y + offY], offX, cropped[y], 0, cropW);
        }
        return cropped;
    }
}
