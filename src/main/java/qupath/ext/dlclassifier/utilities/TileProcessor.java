package qupath.ext.dlclassifier.utilities;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

/**
 * Processes images into tiles for deep learning inference.
 * <p>
 * This class handles tile generation, overlap management, and result merging
 * for pixel classification workflows.
 *
 * <h3>Tiling Strategy</h3>
 * <ul>
 *   <li>Generates overlapping tiles to avoid boundary artifacts</li>
 *   <li>Supports serpentine pattern for efficient memory access</li>
 *   <li>Handles edge cases at image boundaries</li>
 *   <li>Provides Gaussian or linear blending in overlap regions</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TileProcessor {

    private static final Logger logger = LoggerFactory.getLogger(TileProcessor.class);

    private final int tileSize;
    private final int overlap;
    private final double downsample;
    private final InferenceConfig.BlendMode blendMode;
    private final int maxTilesInMemory;

    /**
     * Creates a new tile processor.
     *
     * @param config inference configuration
     */
    public TileProcessor(InferenceConfig config) {
        this(config, 1.0);
    }

    /**
     * Creates a new tile processor with a specific downsample factor.
     *
     * @param config     inference configuration
     * @param downsample downsample factor (1.0 = full resolution)
     */
    public TileProcessor(InferenceConfig config, double downsample) {
        this.tileSize = config.getTileSize();
        this.overlap = config.getOverlap();
        this.downsample = downsample;
        this.blendMode = config.getBlendMode();
        this.maxTilesInMemory = config.getMaxTilesInMemory();

        logger.info(
                "TileProcessor initialized: size={}, overlap={}, downsample={}, blend={}",
                tileSize,
                overlap,
                downsample,
                blendMode);
    }

    /**
     * Creates a new tile processor with explicit parameters.
     *
     * @param tileSize         tile size in pixels
     * @param overlap          overlap in pixels
     * @param blendMode        blend mode for boundaries
     * @param maxTilesInMemory max tiles to keep in memory
     */
    public TileProcessor(int tileSize, int overlap, InferenceConfig.BlendMode blendMode, int maxTilesInMemory) {
        this.tileSize = tileSize;
        this.overlap = overlap;
        this.downsample = 1.0;
        this.blendMode = blendMode;
        this.maxTilesInMemory = maxTilesInMemory;
    }

    /**
     * Generates tile specifications for a region of interest.
     *
     * @param roi    the region to tile
     * @param server the image server
     * @return list of tile specifications
     */
    public List<TileSpec> generateTiles(ROI roi, ImageServer<BufferedImage> server) {
        return generateTiles(
                (int) roi.getBoundsX(),
                (int) roi.getBoundsY(),
                (int) roi.getBoundsWidth(),
                (int) roi.getBoundsHeight(),
                server);
    }

    /**
     * Generates tile specifications for a rectangular region.
     * <p>
     * Tiles are generated with edge information to support proper handling
     * at image boundaries. Edge tiles are positioned to maximize real pixel
     * context while ensuring full coverage of the requested region.
     *
     * @param x      left edge
     * @param y      top edge
     * @param width  region width
     * @param height region height
     * @param server the image server
     * @return list of tile specifications
     */
    public List<TileSpec> generateTiles(int x, int y, int width, int height, ImageServer<BufferedImage> server) {
        List<TileSpec> tiles = new ArrayList<>();

        int stepSize = tileSize - overlap;

        // Calculate grid dimensions
        int nCols = (int) Math.ceil((double) width / stepSize);
        int nRows = (int) Math.ceil((double) height / stepSize);

        // Ensure at least one tile
        if (nCols == 0) nCols = 1;
        if (nRows == 0) nRows = 1;

        // Effective image dimensions in the downsampled coordinate space
        int serverWidth = (int) (server.getWidth() / downsample);
        int serverHeight = (int) (server.getHeight() / downsample);

        logger.debug(
                "Generating tiles: grid {}x{}, step={}, image={}x{}",
                nCols,
                nRows,
                stepSize,
                serverWidth,
                serverHeight);

        int tileIndex = 0;
        for (int row = 0; row < nRows; row++) {
            // Serpentine pattern: reverse direction on odd rows
            boolean reverseDirection = (row % 2 == 1);

            for (int col = 0; col < nCols; col++) {
                int actualCol = reverseDirection ? (nCols - 1 - col) : col;

                int tileX = x + actualCol * stepSize;
                int tileY = y + row * stepSize;

                // Determine edge flags before any adjustment
                int edgeFlags = 0;

                // Check if we're at image boundaries
                boolean atLeftEdge = (tileX <= 0);
                boolean atTopEdge = (tileY <= 0);
                boolean atRightEdge = (tileX + tileSize >= serverWidth);
                boolean atBottomEdge = (tileY + tileSize >= serverHeight);

                // Adjust tile position if it would extend beyond image
                // This shifts the tile inward to stay within bounds
                int originalTileX = tileX;
                int originalTileY = tileY;

                if (tileX + tileSize > serverWidth) {
                    tileX = Math.max(0, serverWidth - tileSize);
                    atRightEdge = true;
                }
                if (tileY + tileSize > serverHeight) {
                    tileY = Math.max(0, serverHeight - tileSize);
                    atBottomEdge = true;
                }
                if (tileX < 0) {
                    tileX = 0;
                    atLeftEdge = true;
                }
                if (tileY < 0) {
                    tileY = 0;
                    atTopEdge = true;
                }

                // Build edge flags
                if (atTopEdge) edgeFlags |= EDGE_TOP;
                if (atRightEdge) edgeFlags |= EDGE_RIGHT;
                if (atBottomEdge) edgeFlags |= EDGE_BOTTOM;
                if (atLeftEdge) edgeFlags |= EDGE_LEFT;

                boolean isEdgeTile = (edgeFlags != 0);

                // Calculate valid region within tile
                // For edge tiles, some portion may be outside the originally requested region
                int validX = 0;
                int validY = 0;
                int validWidth = tileSize;
                int validHeight = tileSize;

                // Adjust valid region if tile extends beyond image bounds
                if (tileX + tileSize > serverWidth) {
                    validWidth = serverWidth - tileX;
                }
                if (tileY + tileSize > serverHeight) {
                    validHeight = serverHeight - tileY;
                }

                // For shifted tiles, adjust valid region start if needed
                // (when tile was shifted inward from its calculated position)
                if (tileX < originalTileX && originalTileX > 0) {
                    validX = originalTileX - tileX;
                    validWidth = Math.min(tileSize - validX, width - (originalTileX - x));
                }
                if (tileY < originalTileY && originalTileY > 0) {
                    validY = originalTileY - tileY;
                    validHeight = Math.min(tileSize - validY, height - (originalTileY - y));
                }

                // Create tile spec with full edge information
                TileSpec spec = new TileSpec(
                        tileIndex,
                        tileX,
                        tileY,
                        tileSize,
                        tileSize,
                        row,
                        actualCol,
                        validX,
                        validY,
                        validWidth,
                        validHeight,
                        isEdgeTile,
                        edgeFlags);
                tiles.add(spec);
                tileIndex++;
            }
        }

        logger.info(
                "Generated {} tiles for region {}x{} at ({},{}), {} edge tiles",
                tiles.size(),
                width,
                height,
                x,
                y,
                tiles.stream().filter(TileSpec::isEdgeTile).count());

        return tiles;
    }

    /**
     * Reads tile data from an image server.
     *
     * @param spec   tile specification
     * @param server image server
     * @return tile image
     * @throws IOException if reading fails
     */
    public BufferedImage readTile(TileSpec spec, ImageServer<BufferedImage> server) throws IOException {
        // Convert tile coordinates (in downsampled space) to full-res region
        int fullResX = (int) (spec.x() * downsample);
        int fullResY = (int) (spec.y() * downsample);
        int fullResW = (int) (spec.width() * downsample);
        int fullResH = (int) (spec.height() * downsample);

        RegionRequest request =
                RegionRequest.createInstance(server.getPath(), downsample, fullResX, fullResY, fullResW, fullResH);

        return server.readRegion(request);
    }

    /**
     * Creates blending weights for tile merging.
     *
     * @return 2D array of weights
     */
    public float[][] createBlendWeights() {
        float[][] weights = new float[tileSize][tileSize];

        switch (blendMode) {
            case GAUSSIAN -> createGaussianWeights(weights);
            case LINEAR -> createLinearWeights(weights);
            case CENTER_CROP -> createCenterCropWeights(weights);
            case NONE -> createUniformWeights(weights);
        }

        return weights;
    }

    /**
     * Creates Gaussian blending weights (strongest in center).
     */
    private void createGaussianWeights(float[][] weights) {
        double sigma = tileSize / 4.0;
        double centerX = tileSize / 2.0;
        double centerY = tileSize / 2.0;

        for (int y = 0; y < tileSize; y++) {
            for (int x = 0; x < tileSize; x++) {
                double dx = x - centerX;
                double dy = y - centerY;
                double distSq = dx * dx + dy * dy;
                weights[y][x] = (float) Math.exp(-distSq / (2 * sigma * sigma));
            }
        }
    }

    /**
     * Creates center-crop weights: 1.0 inside the center region, 0.0 in the overlap margin.
     * Only center predictions are kept, eliminating boundary artifacts at the cost of
     * ~4x more tiles needed for full coverage.
     */
    private void createCenterCropWeights(float[][] weights) {
        // `overlap` is the TOTAL overlap between neighbouring tiles, so the
        // margin to discard on each side is half of it. Using the total here
        // kept a band of (tileSize - 2*overlap) while the grid advances by
        // (tileSize - overlap), so the kept band was always narrower than the
        // stride and left uncovered stripes -- at tileSize 256 and 51 px of
        // context padding, a 52 px band against a 154 px stride, i.e. 102 px
        // of every step contributed by no tile at all. Once overlap reached
        // tileSize/2 the band collapsed to nothing and the output went blank.
        int margin = overlap / 2;
        for (int y = 0; y < tileSize; y++) {
            for (int x = 0; x < tileSize; x++) {
                boolean inCenter = x >= margin && x < tileSize - margin && y >= margin && y < tileSize - margin;
                weights[y][x] = inCenter ? 1.0f : 0.0f;
            }
        }
    }

    /**
     * Creates linear blending weights (fade to zero at edges in overlap region).
     */
    private void createLinearWeights(float[][] weights) {
        for (int y = 0; y < tileSize; y++) {
            for (int x = 0; x < tileSize; x++) {
                // Calculate weight based on distance from edge
                float weightX = 1.0f;
                float weightY = 1.0f;

                // Left edge
                if (x < overlap) {
                    weightX = (float) x / overlap;
                }
                // Right edge
                else if (x >= tileSize - overlap) {
                    weightX = (float) (tileSize - x) / overlap;
                }

                // Top edge
                if (y < overlap) {
                    weightY = (float) y / overlap;
                }
                // Bottom edge
                else if (y >= tileSize - overlap) {
                    weightY = (float) (tileSize - y) / overlap;
                }

                weights[y][x] = weightX * weightY;
            }
        }
    }

    /**
     * Creates uniform weights (1.0 everywhere).
     */
    private void createUniformWeights(float[][] weights) {
        for (int y = 0; y < tileSize; y++) {
            for (int x = 0; x < tileSize; x++) {
                weights[y][x] = 1.0f;
            }
        }
    }

    /**
     * Merges overlapping tile results into a single result map.
     *
     * @param tiles        tile specifications
     * @param tileResults  classification results per tile (numClasses per pixel)
     * @param outputWidth  output width
     * @param outputHeight output height
     * @param numClasses   number of classification classes
     * @return merged probability map [height][width][numClasses]
     */
    public float[][][] mergeTileResults(
            List<TileSpec> tiles, List<float[][][]> tileResults, int outputWidth, int outputHeight, int numClasses) {
        // Output arrays
        float[][][] output = new float[outputHeight][outputWidth][numClasses];
        float[][] weightSum = new float[outputHeight][outputWidth];

        // Get blend weights
        float[][] blendWeights = createBlendWeights();

        // Merge each tile
        for (int i = 0; i < tiles.size(); i++) {
            TileSpec spec = tiles.get(i);
            float[][][] tileResult = tileResults.get(i);

            for (int ty = 0; ty < tileSize && spec.y() + ty < outputHeight; ty++) {
                for (int tx = 0; tx < tileSize && spec.x() + tx < outputWidth; tx++) {
                    int outY = spec.y() + ty;
                    int outX = spec.x() + tx;

                    if (outY >= 0 && outX >= 0) {
                        float weight = blendWeights[ty][tx];

                        for (int c = 0; c < numClasses; c++) {
                            output[outY][outX][c] += tileResult[ty][tx][c] * weight;
                        }
                        weightSum[outY][outX] += weight;
                    }
                }
            }
        }

        // Normalize by weight sum
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                if (weightSum[y][x] > 0) {
                    for (int c = 0; c < numClasses; c++) {
                        output[y][x][c] /= weightSum[y][x];
                    }
                }
            }
        }

        return output;
    }

    /**
     * Result of complete tile merging, containing both probability map and classification map.
     *
     * @param probabilityMap merged probability map [height][width][numClasses]
     * @param classificationMap argmax classification map [height][width] with class indices
     */
    public record MergedResult(float[][][] probabilityMap, int[][] classificationMap) {}

    /**
     * Merges overlapping tile results into complete probability and classification maps.
     * <p>
     * This method performs both operations in a single pass for efficiency:
     * <ol>
     *   <li>Merges probabilities with weighted blending</li>
     *   <li>Computes argmax classification from merged probabilities</li>
     * </ol>
     * <p>
     * The classification map is computed AFTER merging, ensuring that objects
     * spanning tile boundaries are handled correctly (since probabilities are
     * blended first, then classified).
     *
     * @param tiles        tile specifications
     * @param tileResults  classification results per tile (numClasses per pixel)
     * @param outputWidth  output width
     * @param outputHeight output height
     * @param numClasses   number of classification classes
     * @return MergedResult containing both probability map and classification map
     */
    public MergedResult mergeTileResultsComplete(
            List<TileSpec> tiles, List<float[][][]> tileResults, int outputWidth, int outputHeight, int numClasses) {
        // Merge probabilities first
        float[][][] probMap = mergeTileResults(tiles, tileResults, outputWidth, outputHeight, numClasses);

        // Compute argmax classification from merged probabilities
        int[][] classMap = new int[outputHeight][outputWidth];

        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
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

        return new MergedResult(probMap, classMap);
    }

    /**
     * Merges tile results considering edge tile valid regions.
     * <p>
     * This variant uses the valid region information from TileSpec to properly
     * handle edge tiles, ensuring that only valid (non-padded) pixels contribute
     * to the merged output.
     *
     * @param tiles        tile specifications with edge information
     * @param tileResults  classification results per tile
     * @param regionX      X offset of the region in image coordinates
     * @param regionY      Y offset of the region in image coordinates
     * @param regionWidth  width of the region
     * @param regionHeight height of the region
     * @param numClasses   number of classification classes
     * @return MergedResult containing both probability map and classification map
     */
    public MergedResult mergeTileResultsWithEdgeHandling(
            List<TileSpec> tiles,
            List<float[][][]> tileResults,
            int regionX,
            int regionY,
            int regionWidth,
            int regionHeight,
            int numClasses) {
        // Output arrays (relative to region, not image)
        float[][][] output = new float[regionHeight][regionWidth][numClasses];
        float[][] weightSum = new float[regionHeight][regionWidth];

        // Get blend weights
        float[][] blendWeights = createBlendWeights();

        // Merge each tile
        for (int i = 0; i < tiles.size(); i++) {
            TileSpec spec = tiles.get(i);
            float[][][] tileResult = tileResults.get(i);

            // Process only the valid region of the tile
            for (int ty = spec.validY(); ty < spec.validY() + spec.validHeight(); ty++) {
                for (int tx = spec.validX(); tx < spec.validX() + spec.validWidth(); tx++) {
                    // Calculate output coordinates (relative to region)
                    int outX = (spec.x() + tx) - regionX;
                    int outY = (spec.y() + ty) - regionY;

                    // Skip if outside output bounds
                    if (outX < 0 || outX >= regionWidth || outY < 0 || outY >= regionHeight) {
                        continue;
                    }

                    // Get weight - use full blendWeights for non-edge pixels
                    float weight = blendWeights[ty][tx];

                    // For edge tiles, reduce weight near edges to avoid artifacts
                    if (spec.isEdgeTile()) {
                        // Reduce weight for pixels near the invalid (padded) regions
                        float edgeWeight = calculateEdgeWeight(spec, tx, ty, tileSize);
                        weight *= edgeWeight;
                    }

                    // Ensure we have valid tile data
                    if (ty < tileResult.length && tx < tileResult[0].length) {
                        for (int c = 0; c < numClasses; c++) {
                            output[outY][outX][c] += tileResult[ty][tx][c] * weight;
                        }
                        weightSum[outY][outX] += weight;
                    }
                }
            }
        }

        // Normalize by weight sum
        for (int y = 0; y < regionHeight; y++) {
            for (int x = 0; x < regionWidth; x++) {
                if (weightSum[y][x] > 0) {
                    for (int c = 0; c < numClasses; c++) {
                        output[y][x][c] /= weightSum[y][x];
                    }
                }
            }
        }

        // Compute argmax classification
        int[][] classMap = new int[regionHeight][regionWidth];
        for (int y = 0; y < regionHeight; y++) {
            for (int x = 0; x < regionWidth; x++) {
                int maxClass = 0;
                float maxProb = output[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (output[y][x][c] > maxProb) {
                        maxProb = output[y][x][c];
                        maxClass = c;
                    }
                }
                classMap[y][x] = maxClass;
            }
        }

        return new MergedResult(output, classMap);
    }

    /**
     * Calculates edge weight factor for pixels near tile boundaries.
     */
    private float calculateEdgeWeight(TileSpec spec, int tx, int ty, int size) {
        float weight = 1.0f;

        // Distance from valid region edges
        int distFromValidLeft = tx - spec.validX();
        int distFromValidRight = (spec.validX() + spec.validWidth()) - tx - 1;
        int distFromValidTop = ty - spec.validY();
        int distFromValidBottom = (spec.validY() + spec.validHeight()) - ty - 1;

        // Apply linear falloff near edges
        int fadeDistance = Math.max(1, overlap / 2);

        if (spec.isLeftEdge() && distFromValidLeft < fadeDistance) {
            weight *= (float) distFromValidLeft / fadeDistance;
        }
        if (spec.isRightEdge() && distFromValidRight < fadeDistance) {
            weight *= (float) distFromValidRight / fadeDistance;
        }
        if (spec.isTopEdge() && distFromValidTop < fadeDistance) {
            weight *= (float) distFromValidTop / fadeDistance;
        }
        if (spec.isBottomEdge() && distFromValidBottom < fadeDistance) {
            weight *= (float) distFromValidBottom / fadeDistance;
        }

        return Math.max(weight, 0.001f); // Ensure non-zero weight
    }

    // ==================== Getters ====================

    public int getTileSize() {
        return tileSize;
    }

    public int getOverlap() {
        return overlap;
    }

    public int getStepSize() {
        return tileSize - overlap;
    }

    public InferenceConfig.BlendMode getBlendMode() {
        return blendMode;
    }

    public int getMaxTilesInMemory() {
        return maxTilesInMemory;
    }

    public double getDownsample() {
        return downsample;
    }

    /**
     * Edge flags for tiles at image boundaries.
     */
    public static final int EDGE_TOP = 1;

    public static final int EDGE_RIGHT = 2;
    public static final int EDGE_BOTTOM = 4;
    public static final int EDGE_LEFT = 8;

    /**
     * Tile specification containing position, size, and edge information.
     * <p>
     * For tiles at image boundaries (edge tiles), the valid region indicates
     * which portion of the tile contains actual image data vs. padding.
     *
     * @param index       tile index in serpentine order
     * @param x           left edge in image coordinates
     * @param y           top edge in image coordinates
     * @param width       tile width in pixels
     * @param height      tile height in pixels
     * @param row         row index in tile grid
     * @param col         column index in tile grid
     * @param validX      X offset where valid data begins within tile
     * @param validY      Y offset where valid data begins within tile
     * @param validWidth  width of valid data region
     * @param validHeight height of valid data region
     * @param isEdgeTile  true if tile touches any image boundary
     * @param edgeFlags   bitmask of which edges touch boundaries (EDGE_TOP, EDGE_RIGHT, etc.)
     */
    public record TileSpec(
            int index,
            int x,
            int y,
            int width,
            int height,
            int row,
            int col,
            int validX,
            int validY,
            int validWidth,
            int validHeight,
            boolean isEdgeTile,
            int edgeFlags) {

        /**
         * Legacy constructor for backward compatibility.
         * Creates a non-edge tile where the entire tile is valid.
         */
        public TileSpec(int index, int x, int y, int width, int height, int row, int col) {
            this(index, x, y, width, height, row, col, 0, 0, width, height, false, 0);
        }

        /**
         * Returns the center X coordinate.
         */
        public int centerX() {
            return x + width / 2;
        }

        /**
         * Returns the center Y coordinate.
         */
        public int centerY() {
            return y + height / 2;
        }

        /**
         * Returns the center X of the valid region in image coordinates.
         */
        public int validCenterX() {
            return x + validX + validWidth / 2;
        }

        /**
         * Returns the center Y of the valid region in image coordinates.
         */
        public int validCenterY() {
            return y + validY + validHeight / 2;
        }

        /**
         * Returns true if this tile touches the top edge of the image.
         */
        public boolean isTopEdge() {
            return (edgeFlags & EDGE_TOP) != 0;
        }

        /**
         * Returns true if this tile touches the right edge of the image.
         */
        public boolean isRightEdge() {
            return (edgeFlags & EDGE_RIGHT) != 0;
        }

        /**
         * Returns true if this tile touches the bottom edge of the image.
         */
        public boolean isBottomEdge() {
            return (edgeFlags & EDGE_BOTTOM) != 0;
        }

        /**
         * Returns true if this tile touches the left edge of the image.
         */
        public boolean isLeftEdge() {
            return (edgeFlags & EDGE_LEFT) != 0;
        }

        /**
         * Returns true if this tile is at a corner (touches 2 edges).
         */
        public boolean isCornerTile() {
            return Integer.bitCount(edgeFlags) >= 2;
        }
    }
}
