package qupath.ext.dlclassifier.utilities;

import ij.plugin.filter.EDM;
import ij.plugin.filter.MaximumFinder;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.LinearRing;
import org.locationtech.jts.geom.MultiPolygon;
import org.locationtech.jts.geom.Polygon;
import org.locationtech.jts.operation.union.UnaryUnionOp;
import org.locationtech.jts.simplify.TopologyPreservingSimplifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.InferenceConfig.OutputObjectType;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.analysis.images.SimpleImage;
import qupath.lib.analysis.images.SimpleImages;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassTools;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.interfaces.ROI;

/**
 * Generates output from classification results.
 * <p>
 * This class converts raw classification probabilities into QuPath outputs:
 * <ul>
 *   <li>Measurements: Area and percentage per class</li>
 *   <li>Objects: Detection/annotation objects using QuPath's ContourTracing API</li>
 *   <li>Overlay: Classification overlay for visualization</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class OutputGenerator {

    private static final Logger logger = LoggerFactory.getLogger(OutputGenerator.class);

    private final ImageData<BufferedImage> imageData;
    private final ClassifierMetadata metadata;
    private final InferenceConfig config;
    private final double pixelSizeMicrons;

    /**
     * Creates a new output generator.
     *
     * @param imageData image data
     * @param metadata  classifier metadata
     * @param config    inference configuration
     */
    public OutputGenerator(ImageData<BufferedImage> imageData, ClassifierMetadata metadata, InferenceConfig config) {
        this.imageData = imageData;
        this.metadata = metadata;
        this.config = config;
        double ps = imageData.getServer().getPixelCalibration().getAveragedPixelSizeMicrons();
        // Guard against uncalibrated images where pixel size is NaN.
        // Use 1.0 as fallback so micron-based thresholds become pixel-based.
        this.pixelSizeMicrons = Double.isNaN(ps) || ps <= 0 ? 1.0 : ps;
    }

    /**
     * Generates measurements output for a parent annotation.
     *
     * @param parent      the parent annotation
     * @param predictions probability map [height][width][numClasses]
     */
    public void addMeasurements(PathObject parent, float[][][] predictions) {
        logger.info("Adding measurements to: {}", parent.getName());

        int height = predictions.length;
        int width = predictions[0].length;
        int numClasses = predictions[0][0].length;
        int totalPixels = height * width;

        // Calculate area per class
        double[] classAreas = new double[numClasses];
        int[] classCounts = new int[numClasses];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Find winning class
                int maxClass = 0;
                float maxProb = predictions[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (predictions[y][x][c] > maxProb) {
                        maxProb = predictions[y][x][c];
                        maxClass = c;
                    }
                }
                classCounts[maxClass]++;
            }
        }

        // Convert to area
        double pixelArea = pixelSizeMicrons * pixelSizeMicrons;
        for (int c = 0; c < numClasses; c++) {
            classAreas[c] = classCounts[c] * pixelArea;
        }

        // Add measurements
        var ml = parent.getMeasurementList();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();

        for (int c = 0; c < numClasses; c++) {
            String className = c < classes.size() ? classes.get(c).name() : "Class " + c;
            double percentage = 100.0 * classCounts[c] / totalPixels;

            ml.put("DL: " + className + " area (um^2)", classAreas[c]);
            ml.put("DL: " + className + " %", percentage);
        }

        logger.info("Added {} measurements", numClasses * 2);
    }

    /**
     * Creates detection objects from classification results using QuPath's ContourTracing API.
     *
     * @param predictions probability map [height][width][numClasses]
     * @param offsetX     X offset in image coordinates
     * @param offsetY     Y offset in image coordinates
     * @return list of created detection objects
     */
    public List<PathObject> createObjects(float[][][] predictions, int offsetX, int offsetY) {
        logger.info("Creating objects from predictions using ContourTracing");

        int height = predictions.length;
        int width = predictions[0].length;
        int numClasses = predictions[0][0].length;

        // Create classification map (argmax)
        int[][] classMap = createClassificationMap(predictions, height, width, numClasses);

        return createObjectsFromMergedMap(classMap, offsetX, offsetY, OutputObjectType.DETECTION);
    }

    /**
     * Creates a classification overlay image.
     *
     * @param predictions probability map [height][width][numClasses]
     * @return overlay image
     */
    public BufferedImage createOverlay(float[][][] predictions) {
        int height = predictions.length;
        int width = predictions[0].length;
        int numClasses = predictions[0][0].length;

        BufferedImage overlay = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        // Get class colors
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Find winning class
                int maxClass = 0;
                float maxProb = predictions[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (predictions[y][x][c] > maxProb) {
                        maxProb = predictions[y][x][c];
                        maxClass = c;
                    }
                }

                // Get color for class
                int rgb;
                if (maxClass < classes.size()) {
                    rgb = parseColor(classes.get(maxClass).color());
                } else {
                    rgb = 0xFF808080; // Gray default
                }

                // Apply alpha based on confidence
                int alpha = (int) (maxProb * 128); // Semi-transparent
                int argb = (alpha << 24) | (rgb & 0x00FFFFFF);
                overlay.setRGB(x, y, argb);
            }
        }

        return overlay;
    }

    /**
     * Parses a hex color string to RGB.
     */
    private int parseColor(String colorStr) {
        if (colorStr == null || colorStr.isEmpty()) {
            return 0xFF808080;
        }
        try {
            if (colorStr.startsWith("#")) {
                colorStr = colorStr.substring(1);
            }
            return Integer.parseInt(colorStr, 16) | 0xFF000000;
        } catch (NumberFormatException e) {
            return 0xFF808080;
        }
    }

    /**
     * Generates measurements output for a parent annotation from multiple tile results.
     *
     * @param parent      the parent annotation
     * @param tileResults list of probability maps for each tile
     * @param tileSpecs   list of tile specifications
     */
    public void addMeasurements(
            PathObject parent, List<float[][][]> tileResults, List<TileProcessor.TileSpec> tileSpecs) {
        if (tileResults.isEmpty()) {
            logger.warn("No tile results to process for measurements");
            return;
        }

        logger.info("Adding measurements from {} tiles to: {}", tileResults.size(), parent.getName());

        // Aggregate counts across all tiles
        int numClasses = tileResults.get(0)[0][0].length;
        long[] classCounts = new long[numClasses];
        long totalPixels = 0;

        for (float[][][] predictions : tileResults) {
            int height = predictions.length;
            int width = predictions[0].length;
            totalPixels += (long) height * width;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // Find winning class
                    int maxClass = 0;
                    float maxProb = predictions[y][x][0];
                    for (int c = 1; c < numClasses; c++) {
                        if (predictions[y][x][c] > maxProb) {
                            maxProb = predictions[y][x][c];
                            maxClass = c;
                        }
                    }
                    classCounts[maxClass]++;
                }
            }
        }

        // Convert to area
        double pixelArea = pixelSizeMicrons * pixelSizeMicrons;
        double[] classAreas = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            classAreas[c] = classCounts[c] * pixelArea;
        }

        // Add measurements
        var ml = parent.getMeasurementList();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();

        for (int c = 0; c < numClasses; c++) {
            String className = c < classes.size() ? classes.get(c).name() : "Class " + c;
            double percentage = totalPixels > 0 ? 100.0 * classCounts[c] / totalPixels : 0;

            ml.put("DL: " + className + " area (um^2)", classAreas[c]);
            ml.put("DL: " + className + " %", percentage);
        }

        ml.put("DL: Total pixels", totalPixels);
        // Classifier name stored in logger only; MeasurementList accepts only numeric values
        logger.info("Measurements from classifier: {}", metadata.getName());

        logger.info("Added {} measurements", numClasses * 2 + 2);
    }

    /**
     * Creates detection objects from multiple tile classification results.
     *
     * @param tileResults list of probability maps for each tile
     * @param tileSpecs   list of tile specifications
     * @param parentROI   the parent region for filtering
     * @return list of created detection objects
     */
    public List<PathObject> createDetectionObjects(
            List<float[][][]> tileResults, List<TileProcessor.TileSpec> tileSpecs, ROI parentROI) {
        List<PathObject> allDetections = new ArrayList<>();

        if (tileResults.size() != tileSpecs.size()) {
            logger.error("Mismatch between tile results ({}) and specs ({})", tileResults.size(), tileSpecs.size());
            return allDetections;
        }

        for (int i = 0; i < tileResults.size(); i++) {
            float[][][] predictions = tileResults.get(i);
            TileProcessor.TileSpec spec = tileSpecs.get(i);

            List<PathObject> detections = createObjects(predictions, spec.x(), spec.y());
            allDetections.addAll(detections);
        }

        // Filter and clip objects to the parent ROI using geometric intersection
        if (parentROI != null) {
            allDetections = clipObjectsToParent(allDetections, parentROI);
        }

        logger.info("Created {} detection objects from {} tiles", allDetections.size(), tileResults.size());
        return allDetections;
    }

    /**
     * Creates PathObjects from a merged classification map using QuPath's ContourTracing API.
     * <p>
     * This method processes the ENTIRE merged classification map at once, enabling
     * objects that span tile boundaries to be correctly identified as single objects.
     *
     * @param classMap   merged classification map [height][width] with class indices
     * @param offsetX    X offset in image coordinates
     * @param offsetY    Y offset in image coordinates
     * @param objectType type of PathObject to create (DETECTION or ANNOTATION)
     * @return list of created PathObjects
     */
    public List<PathObject> createObjectsFromMergedMap(
            int[][] classMap, int offsetX, int offsetY, OutputObjectType objectType) {
        return createObjectsFromMergedMap(classMap, offsetX, offsetY, 1.0, objectType);
    }

    /**
     * Creates detection/annotation objects from a merged classification map.
     * <p>
     * The downsample parameter maps classification pixels to image coordinates:
     * each pixel in the classMap covers a (downsample x downsample) area in the
     * full-resolution image. The offsets and region dimensions are in full-res
     * image coordinates.
     *
     * @param classMap    argmax classification map [height][width]
     * @param offsetX     region X offset in full-res image coordinates
     * @param offsetY     region Y offset in full-res image coordinates
     * @param downsample  downsample factor (classMap pixels per image pixel)
     * @param objectType  DETECTION or ANNOTATION
     * @return list of PathObjects
     */
    public List<PathObject> createObjectsFromMergedMap(
            int[][] classMap, int offsetX, int offsetY, double downsample, OutputObjectType objectType) {
        int height = classMap.length;
        int width = classMap[0].length;
        int fullResW = (int) (width * downsample);
        int fullResH = (int) (height * downsample);

        logger.info(
                "Creating objects from merged map ({}x{} @ ds={}) at ({},{}), type={}",
                width,
                height,
                downsample,
                offsetX,
                offsetY,
                objectType);

        List<PathObject> objects = new ArrayList<>();
        int numClasses = metadata.getClasses().size();

        // Create a RegionRequest mapping classMap pixels to full-res image space.
        // ContourTracing uses this to scale traced geometries to image coordinates.
        RegionRequest region = RegionRequest.createInstance(
                imageData.getServer().getPath(), downsample, offsetX, offsetY, fullResW, fullResH);

        // Process each class (skip background = class 0, skip ignored classes)
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
        for (int classIdx = 1; classIdx < numClasses; classIdx++) {
            String className = classIdx < classes.size() ? classes.get(classIdx).name() : "Class " + classIdx;
            if (PathClassTools.isIgnoredClass(PathClass.fromString(className))) {
                logger.debug("Skipping ignored class: {}", className);
                continue;
            }
            List<PathObject> classObjects = traceClassContours(classMap, classIdx, width, height, region, objectType);
            objects.addAll(classObjects);
        }

        logger.info("Created {} objects from merged classification map", objects.size());
        return objects;
    }

    /**
     * Creates objects from merged tiles using TileProcessor's complete merge.
     * <p>
     * This is the preferred method for OBJECTS output as it correctly handles
     * objects spanning tile boundaries.
     *
     * @param tileProcessor tile processor to use for merging
     * @param tileResults   probability maps for each tile
     * @param tileSpecs     tile specifications
     * @param parentROI     parent region for coordinate offset and filtering
     * @param numClasses    number of classification classes
     * @param objectType    type of PathObject to create
     * @return list of created PathObjects
     */
    public List<PathObject> createObjectsFromTiles(
            TileProcessor tileProcessor,
            List<float[][][]> tileResults,
            List<TileProcessor.TileSpec> tileSpecs,
            ROI parentROI,
            int numClasses,
            OutputObjectType objectType) {
        if (tileResults.isEmpty() || tileSpecs.isEmpty()) {
            logger.warn("No tile results to process for object creation");
            return Collections.emptyList();
        }

        int regionX = (int) parentROI.getBoundsX();
        int regionY = (int) parentROI.getBoundsY();
        int regionWidth = (int) parentROI.getBoundsWidth();
        int regionHeight = (int) parentROI.getBoundsHeight();

        // Use edge-aware merging to get complete merged result
        TileProcessor.MergedResult merged = tileProcessor.mergeTileResultsWithEdgeHandling(
                tileSpecs, tileResults, regionX, regionY, regionWidth, regionHeight, numClasses);

        // Apply probability smoothing to match overlay quality.
        // The overlay smooths each tile's probabilities before argmax;
        // here we smooth the merged map which is even better (crosses
        // tile boundaries). Uses the same sigma as the overlay.
        double sigma = config.getOverlaySmoothingSigma();
        int[][] classMap;
        if (sigma > 0) {
            float[][][] smoothed =
                    gaussianSmoothProbabilities(merged.probabilityMap(), regionWidth, regionHeight, sigma);
            classMap = computeArgmax(smoothed, regionWidth, regionHeight, numClasses);
        } else {
            classMap = merged.classificationMap();
        }

        // Create objects from the classification map using ContourTracing
        List<PathObject> objects = createObjectsFromMergedMap(classMap, regionX, regionY, objectType);

        // Clip objects to the parent ROI using geometric intersection
        if (parentROI != null) {
            objects = clipObjectsToParent(objects, parentROI);
        }

        return new ArrayList<>(objects);
    }

    // ==================== ContourTracing Integration ====================

    /**
     * Traces contours for a single class using QuPath's ContourTracing API
     * and creates PathObjects with post-processing applied.
     */
    private List<PathObject> traceClassContours(
            int[][] classMap,
            int targetClass,
            int width,
            int height,
            RegionRequest region,
            OutputObjectType objectType) {
        List<PathObject> objects = new ArrayList<>();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
        String className =
                targetClass < classes.size() ? classes.get(targetClass).name() : "Class " + targetClass;
        PathClass pathClass = PathClass.fromString(className);

        // Build the class mask. When touching-object separation is enabled, run a
        // distance-transform watershed on this class's binary mask first so that
        // abutting instances are cut apart before tracing; otherwise the mask is
        // the raw class map and touching instances trace as a single blob.
        SimpleImage classImage;
        if (config.isSeparateTouchingObjects()) {
            boolean[] splitMask =
                    watershedForeground(classMap, targetClass, width, height, config.getWatershedTolerance());
            classImage = createClassImageFromMask(splitMask, targetClass, width, height);
        } else {
            classImage = createClassImage(classMap, targetClass, width, height);
        }

        // Use ContourTracing to trace geometries for this class value
        Geometry geometry = ContourTracing.createTracedGeometry(classImage, targetClass, targetClass, region);

        if (geometry == null || geometry.isEmpty()) {
            return objects;
        }

        double maxObjectSizeMicrons = config.getMaxObjectSizeMicrons();

        // Split multi-geometries into individual objects
        for (int i = 0; i < geometry.getNumGeometries(); i++) {
            Geometry part = geometry.getGeometryN(i);
            if (part == null || part.isEmpty()) continue;

            // Apply post-processing (hole filling, smoothing)
            part = applyGeometryPostProcessing(part);
            if (part == null || part.isEmpty()) continue;

            // Check minimum size
            double areaMicrons = part.getArea() * pixelSizeMicrons * pixelSizeMicrons;
            if (areaMicrons < config.getMinObjectSizeMicrons()) continue;
            // Check maximum size (0 = no limit); drops mis-merged sheets
            if (maxObjectSizeMicrons > 0 && areaMicrons > maxObjectSizeMicrons) continue;

            // Convert to ROI
            ROI roi = GeometryTools.geometryToROI(part, ImagePlane.getDefaultPlane());

            // Create object based on type
            PathObject pathObject;
            if (objectType == OutputObjectType.ANNOTATION) {
                pathObject = PathObjects.createAnnotationObject(roi, pathClass);
            } else {
                pathObject = PathObjects.createDetectionObject(roi, pathClass);
            }

            // Add area measurements
            pathObject.getMeasurementList().put("Area (um^2)", areaMicrons);
            pathObject.getMeasurementList().put("Area (pixels)", part.getArea());
            if (config.isAddShapeMeasurements()) {
                addShapeMeasurements(pathObject, part);
            }

            objects.add(pathObject);
        }

        return objects;
    }

    /**
     * Creates a SimpleImage from the classification map for a specific class.
     * Pixels matching the target class are set to the class value; others are 0.
     */
    private SimpleImage createClassImage(int[][] classMap, int targetClass, int width, int height) {
        float[] data = new float[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                data[y * width + x] = classMap[y][x] == targetClass ? targetClass : 0;
            }
        }
        return SimpleImages.createFloatImage(data, width, height);
    }

    /**
     * Creates a SimpleImage from a boolean foreground mask (already watershed-cut).
     * Foreground pixels take the class value; everything else is 0, so the ridge
     * lines removed by the watershed become background and split the traced blob.
     */
    private SimpleImage createClassImageFromMask(boolean[] mask, int targetClass, int width, int height) {
        float[] data = new float[width * height];
        for (int i = 0; i < data.length; i++) {
            data[i] = mask[i] ? targetClass : 0;
        }
        return SimpleImages.createFloatImage(data, width, height);
    }

    /**
     * Separates touching instances of a single class using a distance-transform
     * watershed, and returns the resulting foreground mask.
     * <p>
     * The pipeline mirrors ImageJ's classic binary watershed: build a binary mask
     * for the target class, compute its Euclidean distance map (EDM), then run
     * {@link MaximumFinder} in SEGMENTED mode over the EDM. That places watershed
     * ridge lines between basins (one basin per distance maximum). The returned
     * mask is the original foreground with those ridge pixels removed, so a
     * subsequent contour trace yields one object per instance.
     *
     * @param classMap    merged classification map [height][width]
     * @param targetClass class index to isolate
     * @param width       map width
     * @param height      map height
     * @param tolerance   maxima merge tolerance (higher = fewer cuts)
     * @return foreground mask (row-major, length width*height); true = keep pixel
     */
    static boolean[] watershedForeground(int[][] classMap, int targetClass, int width, int height, double tolerance) {
        ByteProcessor mask = new ByteProcessor(width, height);
        byte[] maskPx = (byte[]) mask.getPixels();
        for (int y = 0; y < height; y++) {
            int rowOff = y * width;
            int[] row = classMap[y];
            for (int x = 0; x < width; x++) {
                maskPx[rowOff + x] = (row[x] == targetClass) ? (byte) 255 : 0;
            }
        }

        boolean[] fg = new boolean[width * height];
        try {
            // Euclidean distance map of the foreground (background value 0).
            FloatProcessor edm = new EDM().makeFloatEDM(mask, 0, false);
            // Watershed-segment the EDM: 255 = basin interior, 0 = ridge/background.
            ByteProcessor segmented = new MaximumFinder()
                    .findMaxima(edm, tolerance, ImageProcessor.NO_THRESHOLD, MaximumFinder.SEGMENTED, false, true);
            if (segmented == null) {
                for (int i = 0; i < fg.length; i++) fg[i] = (maskPx[i] & 0xFF) != 0;
                return fg;
            }
            byte[] segPx = (byte[]) segmented.getPixels();
            // Keep only original-foreground pixels that survived as basin interior.
            for (int i = 0; i < fg.length; i++) {
                fg[i] = ((maskPx[i] & 0xFF) != 0) && ((segPx[i] & 0xFF) != 0);
            }
        } catch (Exception e) {
            // Watershed is best-effort; on any failure fall back to the un-split mask
            // so object creation still succeeds (just without instance separation).
            logger.warn("Watershed split failed for class {}; using un-split mask: {}", targetClass, e.getMessage());
            for (int i = 0; i < fg.length; i++) fg[i] = (maskPx[i] & 0xFF) != 0;
        }
        return fg;
    }

    /**
     * Attaches scale-invariant shape descriptors to an object: circularity
     * (4*pi*area / perimeter^2, capped at 1.0) and solidity (area / convex-hull
     * area). Both are computed in the geometry's own units, so pixel scaling
     * cancels out.
     */
    private void addShapeMeasurements(PathObject pathObject, Geometry geometry) {
        try {
            double area = geometry.getArea();
            double perimeter = geometry.getLength();
            double circularity = perimeter > 0 ? Math.min(1.0, 4.0 * Math.PI * area / (perimeter * perimeter)) : 0.0;
            double convexArea = geometry.convexHull().getArea();
            double solidity = convexArea > 0 ? area / convexArea : 0.0;
            pathObject.getMeasurementList().put("Circularity", circularity);
            pathObject.getMeasurementList().put("Solidity", solidity);
        } catch (Exception e) {
            logger.debug("Failed to compute shape measurements: {}", e.getMessage());
        }
    }

    /**
     * Clips objects to the parent ROI using geometric intersection.
     * Objects that don't intersect the parent are removed.
     * Objects that partially intersect are clipped to the parent boundary.
     */
    public List<PathObject> clipObjectsToParent(List<PathObject> objects, ROI parentROI) {
        Geometry parentGeom = GeometryTools.roiToGeometry(parentROI);
        if (parentGeom == null) return objects;

        List<PathObject> clipped = new ArrayList<>();
        for (PathObject obj : objects) {
            Geometry objGeom = GeometryTools.roiToGeometry(obj.getROI());
            if (objGeom == null || objGeom.isEmpty()) continue;

            try {
                Geometry intersection = objGeom.intersection(parentGeom);
                if (intersection.isEmpty()) continue;

                // Check size after clipping (min, and max when set)
                double areaMicrons = intersection.getArea() * pixelSizeMicrons * pixelSizeMicrons;
                if (areaMicrons < config.getMinObjectSizeMicrons()) continue;
                double maxObjectSizeMicrons = config.getMaxObjectSizeMicrons();
                if (maxObjectSizeMicrons > 0 && areaMicrons > maxObjectSizeMicrons) continue;

                ROI clippedROI = GeometryTools.geometryToROI(intersection, ImagePlane.getDefaultPlane());

                // Recreate the object with the clipped ROI
                PathObject clippedObj;
                if (obj.isDetection()) {
                    clippedObj = PathObjects.createDetectionObject(clippedROI, obj.getPathClass());
                } else {
                    clippedObj = PathObjects.createAnnotationObject(clippedROI, obj.getPathClass());
                }

                // Copy measurements
                clippedObj.getMeasurementList().put("Area (um^2)", areaMicrons);
                clippedObj.getMeasurementList().put("Area (pixels)", intersection.getArea());
                // Recompute shape measurements on the clipped geometry so boundary
                // objects keep them (they were dropped by the recreate above).
                if (config.isAddShapeMeasurements()) {
                    addShapeMeasurements(clippedObj, intersection);
                }

                clipped.add(clippedObj);
            } catch (Exception e) {
                // Geometric operations can fail for edge cases; skip this object
                logger.debug("Failed to clip object to parent: {}", e.getMessage());
            }
        }
        return clipped;
    }

    // ==================== Geometry Post-Processing ====================

    /**
     * Applies hole filling and boundary smoothing to a geometry.
     */
    private Geometry applyGeometryPostProcessing(Geometry geometry) {
        if (geometry == null || geometry.isEmpty()) return geometry;

        // Hole filling: remove interior rings smaller than threshold
        double holeFillingMicrons = config.getHoleFillingMicrons();
        if (holeFillingMicrons > 0) {
            double holeAreaPixels = holeFillingMicrons / (pixelSizeMicrons * pixelSizeMicrons);
            geometry = removeSmallHoles(geometry, holeAreaPixels);
        }

        // Boundary smoothing via topology-preserving simplification
        double smoothingMicrons = config.getBoundarySmoothing();
        if (smoothingMicrons > 0) {
            double tolerancePixels = smoothingMicrons / pixelSizeMicrons;
            geometry = TopologyPreservingSimplifier.simplify(geometry, tolerancePixels);
        }

        // Fix any invalid geometry from post-processing
        if (!geometry.isValid()) {
            geometry = geometry.buffer(0);
        }

        return geometry;
    }

    /**
     * Removes interior rings (holes) smaller than the given area threshold.
     */
    private Geometry removeSmallHoles(Geometry geometry, double minHoleAreaPixels) {
        if (geometry instanceof Polygon polygon) {
            int numHoles = polygon.getNumInteriorRing();
            if (numHoles == 0) return geometry;

            var factory = geometry.getFactory();
            var shell = polygon.getExteriorRing();
            List<LinearRing> keptHoles = new ArrayList<>();

            for (int i = 0; i < numHoles; i++) {
                var hole = polygon.getInteriorRingN(i);
                double holeArea = Math.abs(org.locationtech.jts.algorithm.Area.ofRing(hole.getCoordinates()));
                if (holeArea >= minHoleAreaPixels) {
                    keptHoles.add((LinearRing) hole);
                }
            }

            return factory.createPolygon((LinearRing) shell, keptHoles.toArray(new LinearRing[0]));

        } else if (geometry instanceof MultiPolygon mp) {
            var factory = geometry.getFactory();
            Polygon[] processed = new Polygon[mp.getNumGeometries()];
            for (int i = 0; i < mp.getNumGeometries(); i++) {
                processed[i] = (Polygon) removeSmallHoles(mp.getGeometryN(i), minHoleAreaPixels);
            }
            return factory.createMultiPolygon(processed);
        }

        return geometry;
    }

    // ==================== Helper Methods ====================

    /**
     * Creates a classification map (argmax) from probability predictions.
     */
    private int[][] createClassificationMap(float[][][] predictions, int height, int width, int numClasses) {
        int[][] classMap = new int[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int maxClass = 0;
                float maxProb = predictions[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (predictions[y][x][c] > maxProb) {
                        maxProb = predictions[y][x][c];
                        maxClass = c;
                    }
                }
                classMap[y][x] = maxClass;
            }
        }
        return classMap;
    }

    // ==================== Geometry Union Utilities ====================

    /** Batch size for hierarchical geometry union. */
    private static final int UNION_BATCH_SIZE = 64;

    /** Threshold above which geometry union is parallelized via ForkJoinPool. */
    private static final int PARALLEL_THRESHOLD = 128;

    /**
     * Merges a list of ROIs into a single unified ROI using hierarchical
     * batched union. For large lists (>{@value PARALLEL_THRESHOLD} ROIs),
     * processing is parallelized using a ForkJoinPool.
     * <p>
     * This is useful for combining adjacent or overlapping detection ROIs
     * into a single annotation region.
     *
     * @param rois the ROIs to merge
     * @return the merged ROI, or {@code null} if the list is null or empty
     */
    public static ROI mergeGeometries(List<ROI> rois) {
        if (rois == null || rois.isEmpty()) return null;
        if (rois.size() == 1) return rois.get(0);

        ImagePlane plane = rois.get(0).getImagePlane();
        List<Geometry> geometries = rois.stream()
                .map(GeometryTools::roiToGeometry)
                .filter(Objects::nonNull)
                .toList();

        if (geometries.isEmpty()) return null;
        if (geometries.size() == 1) {
            return GeometryTools.geometryToROI(geometries.get(0), plane);
        }

        Geometry merged;
        if (geometries.size() > PARALLEL_THRESHOLD) {
            merged = ForkJoinPool.commonPool().invoke(new GeometryUnionTask(geometries, 0, geometries.size()));
        } else {
            merged = mergeGeometriesBatched(geometries, UNION_BATCH_SIZE, false);
        }

        return merged != null ? GeometryTools.geometryToROI(merged, plane) : null;
    }

    /**
     * Merges JTS Geometry objects using hierarchical batched union.
     * <p>
     * In each round, geometries are grouped into batches of {@code batchSize},
     * each batch is unified via {@link UnaryUnionOp}, and the results form the
     * input for the next round. This continues until a single geometry remains.
     *
     * @param geometries the geometries to merge
     * @param batchSize  number of geometries per batch
     * @param parallel   whether to use ForkJoinPool (ignored for small lists)
     * @return the merged geometry, or {@code null} if empty
     */
    public static Geometry mergeGeometriesBatched(List<Geometry> geometries, int batchSize, boolean parallel) {
        if (geometries == null || geometries.isEmpty()) return null;
        if (geometries.size() == 1) return geometries.get(0);

        if (parallel && geometries.size() > PARALLEL_THRESHOLD) {
            return ForkJoinPool.commonPool().invoke(new GeometryUnionTask(geometries, 0, geometries.size()));
        }

        // Iterative hierarchical batched union
        List<Geometry> current = new ArrayList<>(geometries);
        while (current.size() > 1) {
            List<Geometry> next = new ArrayList<>();
            for (int i = 0; i < current.size(); i += batchSize) {
                int end = Math.min(i + batchSize, current.size());
                List<Geometry> batch = current.subList(i, end);
                Geometry unified = UnaryUnionOp.union(batch);
                if (unified != null) {
                    next.add(unified);
                }
            }
            current = next;
        }
        return current.isEmpty() ? null : current.get(0);
    }

    /**
     * ForkJoinPool task for parallel hierarchical geometry union.
     * <p>
     * Recursively splits the geometry list at the midpoint. When a partition
     * is small enough ({@value UNION_BATCH_SIZE} or fewer), it is unified
     * directly with {@link UnaryUnionOp}. The two halves are then merged
     * with a single {@code union()} call.
     */
    private static class GeometryUnionTask extends RecursiveTask<Geometry> {
        private final List<Geometry> geometries;
        private final int start;
        private final int end;

        GeometryUnionTask(List<Geometry> geometries, int start, int end) {
            this.geometries = geometries;
            this.start = start;
            this.end = end;
        }

        @Override
        protected Geometry compute() {
            int size = end - start;
            if (size <= 0) return null;
            if (size <= UNION_BATCH_SIZE) {
                // Base case: union the batch directly
                return UnaryUnionOp.union(geometries.subList(start, end));
            }

            // Fork-join split at midpoint
            int mid = start + size / 2;
            GeometryUnionTask left = new GeometryUnionTask(geometries, start, mid);
            GeometryUnionTask right = new GeometryUnionTask(geometries, mid, end);
            left.fork();
            Geometry rightResult = right.compute();
            Geometry leftResult = left.join();

            if (leftResult != null && rightResult != null) {
                return leftResult.union(rightResult);
            }
            return leftResult != null ? leftResult : rightResult;
        }
    }

    // ==================== Probability Map Utilities ====================

    /**
     * Applies separable Gaussian smoothing to each class channel of a probability map.
     * Same algorithm as DLPixelClassifier.gaussianSmoothProbabilities() but operates
     * on the merged (full-region) probability map for cross-tile boundary smoothing.
     */
    private static float[][][] gaussianSmoothProbabilities(float[][][] probMap, int width, int height, double sigma) {
        int radius = (int) Math.ceil(sigma * 2.5);
        if (radius < 1) return probMap;

        float[] kernel = new float[2 * radius + 1];
        float kernelSum = 0;
        for (int i = -radius; i <= radius; i++) {
            kernel[i + radius] = (float) Math.exp(-0.5 * (i * i) / (sigma * sigma));
            kernelSum += kernel[i + radius];
        }
        for (int i = 0; i < kernel.length; i++) {
            kernel[i] /= kernelSum;
        }

        int numClasses = probMap[0][0].length;

        // Horizontal pass
        float[][][] temp = new float[height][width][numClasses];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < numClasses; c++) {
                    float sum = 0;
                    for (int k = -radius; k <= radius; k++) {
                        int xx = Math.max(0, Math.min(width - 1, x + k));
                        sum += kernel[k + radius] * probMap[y][xx][c];
                    }
                    temp[y][x][c] = sum;
                }
            }
        }

        // Vertical pass
        float[][][] result = new float[height][width][numClasses];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < numClasses; c++) {
                    float sum = 0;
                    for (int k = -radius; k <= radius; k++) {
                        int yy = Math.max(0, Math.min(height - 1, y + k));
                        sum += kernel[k + radius] * temp[yy][x][c];
                    }
                    result[y][x][c] = sum;
                }
            }
        }

        return result;
    }

    /**
     * Computes argmax classification from a probability map.
     */
    private static int[][] computeArgmax(float[][][] probMap, int width, int height, int numClasses) {
        int[][] classMap = new int[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int maxClass = 0;
                float maxProb = probMap[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (probMap[y][x][c] > maxProb) {
                        maxProb = probMap[y][x][c];
                        maxClass = c;
                    }
                }
                classMap[y][x] = maxClass;
            }
        }
        return classMap;
    }
}
