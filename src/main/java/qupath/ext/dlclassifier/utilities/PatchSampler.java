package qupath.ext.dlclassifier.utilities;

import qupath.lib.roi.interfaces.ROI;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Static utility methods for generating patch locations from annotations.
 * <p>
 * Handles both sparse annotations (lines, polylines) and dense annotations
 * (filled polygons). For sparse annotations, patches are centered on points
 * sampled along the ROI. For dense annotations, the bounding box is tiled
 * with 50% overlap.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class PatchSampler {

    private PatchSampler() {
        // Static utility class
    }

    /**
     * A candidate patch location in full-resolution image coordinates.
     */
    public record PatchLocation(int x, int y) {}

    /**
     * Geometry information for a single annotation, pairing its ROI with
     * whether it is sparse (line-like) or dense (area-like).
     */
    public record AnnotationGeometry(ROI roi, boolean isSparse) {}

    /**
     * Determines if an ROI represents sparse annotation (line, polyline).
     * <p>
     * Only true line / polyline ROIs are treated as sparse. Area ROIs
     * (polygon, rectangle, ellipse, freehand brush strokes) are always
     * filled when rasterised, regardless of how curved or elongated they
     * happen to be.
     * <p>
     * A previous implementation also classified area ROIs with area/bbox
     * ratio below 5% as sparse, but this mis-rendered curved tissue
     * annotations (whose axis-aligned bounding box is large relative to
     * their area) as stroked outlines instead of filled regions --
     * producing masks that only covered the contour and heatmaps that
     * dramatically under-represented the labelled area. If a user wants a
     * thin stroke label, they should use QuPath's line tool explicitly.
     */
    public static boolean isSparseROI(ROI roi) {
        return roi.isLine();
    }

    /**
     * Generates patch locations based on annotation positions.
     * <p>
     * For sparse annotations (lines), patches are centered on points sampled
     * along the line. For area annotations, the bounding box is tiled with
     * 50% overlap. Duplicate locations (by grid snapping) are eliminated.
     *
     * @param annotations list of annotation geometries
     * @param patchSize   patch size in pixels (at output resolution)
     * @param downsample  downsample factor (patch coverage = patchSize * downsample)
     * @param imgW        image width in full-resolution pixels
     * @param imgH        image height in full-resolution pixels
     * @return deduplicated list of patch locations in full-resolution coordinates
     */
    public static List<PatchLocation> generatePatchLocations(
            List<AnnotationGeometry> annotations, int patchSize, double downsample,
            int imgW, int imgH) {
        Set<String> locationKeys = new HashSet<>();
        List<PatchLocation> locations = new ArrayList<>();

        // Coverage per patch in full-res coordinates
        int coverage = (int) (patchSize * downsample);
        int step = coverage / 2; // 50% overlap between patches

        for (AnnotationGeometry ann : annotations) {
            ROI roi = ann.roi();
            int x0 = (int) roi.getBoundsX();
            int y0 = (int) roi.getBoundsY();
            int w = (int) roi.getBoundsWidth();
            int h = (int) roi.getBoundsHeight();

            if (ann.isSparse()) {
                // For sparse annotations, sample points along the ROI
                List<double[]> points = samplePointsAlongROI(roi, patchSize, downsample);
                for (double[] pt : points) {
                    // Center patch on the sampled point
                    int px = (int) pt[0] - coverage / 2;
                    int py = (int) pt[1] - coverage / 2;

                    // Clip to image bounds
                    px = Math.max(0, Math.min(px, imgW - coverage));
                    py = Math.max(0, Math.min(py, imgH - coverage));

                    // Snap to grid to avoid too many overlapping patches
                    int snapStep = Math.max(1, step / 2);
                    px = (px / snapStep) * snapStep;
                    py = (py / snapStep) * snapStep;

                    String key = px + "," + py;
                    if (!locationKeys.contains(key)) {
                        locationKeys.add(key);
                        locations.add(new PatchLocation(px, py));
                    }
                }
            } else {
                // For area annotations, tile the bounding box
                for (int py = y0 - coverage / 4; py < y0 + h; py += step) {
                    for (int px = x0 - coverage / 4; px < x0 + w; px += step) {
                        int clippedX = Math.max(0, Math.min(px, imgW - coverage));
                        int clippedY = Math.max(0, Math.min(py, imgH - coverage));

                        String key = clippedX + "," + clippedY;
                        if (!locationKeys.contains(key)) {
                            locationKeys.add(key);
                            locations.add(new PatchLocation(clippedX, clippedY));
                        }
                    }
                }
            }
        }

        return locations;
    }

    /**
     * Samples points along a ROI for patch generation.
     * Works for lines, polylines, and thin shapes.
     *
     * @param roi        the ROI to sample
     * @param patchSize  patch size in pixels (at output resolution)
     * @param downsample downsample factor
     * @return list of [x, y] points in full-resolution coordinates
     */
    public static List<double[]> samplePointsAlongROI(ROI roi, int patchSize, double downsample) {
        List<double[]> points = new ArrayList<>();

        // Get all polygon points from the ROI
        List<qupath.lib.geom.Point2> roiPoints = roi.getAllPoints();

        if (roiPoints.isEmpty()) {
            // Fallback: use center of bounding box
            points.add(new double[]{
                    roi.getBoundsX() + roi.getBoundsWidth() / 2,
                    roi.getBoundsY() + roi.getBoundsHeight() / 2
            });
            return points;
        }

        // Sample at intervals along the ROI path (in full-res coords)
        double sampleInterval = patchSize * downsample / 4.0; // Sample every quarter-patch

        for (int i = 0; i < roiPoints.size() - 1; i++) {
            double x1 = roiPoints.get(i).getX();
            double y1 = roiPoints.get(i).getY();
            double x2 = roiPoints.get(i + 1).getX();
            double y2 = roiPoints.get(i + 1).getY();

            double segLength = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            int numSamples = Math.max(1, (int) (segLength / sampleInterval));

            for (int s = 0; s <= numSamples; s++) {
                double t = (double) s / numSamples;
                double x = x1 + t * (x2 - x1);
                double y = y1 + t * (y2 - y1);
                points.add(new double[]{x, y});
            }
        }

        // Also add the last point
        if (!roiPoints.isEmpty()) {
            qupath.lib.geom.Point2 last = roiPoints.get(roiPoints.size() - 1);
            points.add(new double[]{last.getX(), last.getY()});
        }

        return points;
    }
}
