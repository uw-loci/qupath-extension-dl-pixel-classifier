package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.utilities.TileEncoder;
import qupath.ext.dlclassifier.service.ClassifierClient.PixelInferenceResult;
import qupath.lib.classifiers.pixel.PixelClassifier;
import qupath.lib.classifiers.pixel.PixelClassifierMetadata;
import qupath.lib.common.ColorTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.images.servers.PixelType;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.RegionRequest;

import javafx.application.Platform;

import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Implements QuPath's {@link PixelClassifier} interface to integrate with
 * the native overlay system.
 * <p>
 * This classifier delegates tile inference to the Python DL server via
 * {@link ClassifierClient}. When used with QuPath's
 * {@code PixelClassificationOverlay}, tiles are classified on demand as
 * the user pans and zooms.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class DLPixelClassifier implements PixelClassifier {

    private static final Logger logger = LoggerFactory.getLogger(DLPixelClassifier.class);

    private final ClassifierMetadata metadata;
    private final ChannelConfiguration channelConfig;
    private final InferenceConfig inferenceConfig;
    private final double downsample;
    private final int contextScale;
    private final PixelClassifierMetadata pixelMetadata;
    private final IndexColorModel colorModel;
    private final ClassifierBackend backend;
    private final Path sharedTempDir;
    private final String modelDirPath;

    /** Colors resolved from PathClass cache, keyed by class index. Used by buildColorModel(). */
    private final Map<Integer, Integer> resolvedClassColors = new LinkedHashMap<>();

    /** Circuit breaker: stops retrying server requests after persistent failures. */
    private static final int MAX_CONSECUTIVE_ERRORS = 3;
    private final AtomicInteger consecutiveErrors = new AtomicInteger(0);
    private final AtomicInteger tilesCompleted = new AtomicInteger(0);
    private final AtomicBoolean errorNotified = new AtomicBoolean(false);
    private final AtomicBoolean contextResizeWarned = new AtomicBoolean(false);
    private final AtomicBoolean firstTileLogged = new AtomicBoolean(false);
    private final AtomicBoolean oversizedWarned = new AtomicBoolean(false);
    private volatile String lastErrorMessage;

    /** Set to true when the overlay is being removed, to suppress error counting on interrupted threads. */
    private volatile boolean shuttingDown = false;

    /** The inputPadding value used by QuPath for tile overlap, cached for blending calculations. */
    private final int inputPadding;

    /** Probability map cache and tile boundary blending. */
    private final TileBlendCache blendCache;

    /** Cached channel config with precomputed normalization stats (lazy init). */
    private volatile ChannelConfiguration channelConfigWithStats;
    private final Object statsLock = new Object();

    /** Tracks the current image to detect image switches and invalidate caches. */
    private volatile String currentServerPath;

    /**
     * Creates a new DL pixel classifier.
     *
     * @param metadata        classifier metadata from the server
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @param imageData       image data (used for pixel calibration)
     */
    public DLPixelClassifier(ClassifierMetadata metadata,
                             ChannelConfiguration channelConfig,
                             InferenceConfig inferenceConfig,
                             ImageData<BufferedImage> imageData) {
        this.metadata = metadata;
        this.channelConfig = channelConfig;
        this.inferenceConfig = inferenceConfig;
        this.downsample = metadata.getDownsample();
        this.contextScale = metadata.getContextScale();
        this.inputPadding = computeOverlayPadding(inferenceConfig.getTileSize());

        // Always use GAUSSIAN blending for overlays -- the cosine-bell S-curve
        // smoothly averages overlapping predictions from adjacent tiles,
        // eliminating visible grid artifacts at tile boundaries.
        InferenceConfig.BlendMode overlayBlendMode = InferenceConfig.BlendMode.GAUSSIAN;
        int overlayMaxBlendDist = -1;  // Use full inputPadding for blend zone

        this.blendCache = new TileBlendCache(100, inputPadding,
                overlayBlendMode, overlayMaxBlendDist,
                () -> OverlayService.getInstance().refreshOverlayForBlending());
        this.pixelMetadata = buildPixelMetadata(imageData);
        this.colorModel = buildColorModel();
        this.backend = BackendFactory.getBackend();

        // Resolve classifier ID to filesystem path for the Python server
        ModelManager modelManager = new ModelManager();
        this.modelDirPath = modelManager.getModelPath(metadata.getId())
                .map(p -> p.getParent().toString())
                .orElse(metadata.getId());

        try {
            this.sharedTempDir = Files.createTempDirectory("dl-overlay-");
        } catch (IOException e) {
            throw new RuntimeException("Failed to create temp directory for overlay", e);
        }

        // Log overlay configuration at creation time
        int imgW = imageData.getServer().getWidth();
        int imgH = imageData.getServer().getHeight();
        int tileSize = inferenceConfig.getTileSize();
        int stride = tileSize - 2 * inputPadding;
        long estTiles = (long) Math.ceil((imgW / downsample) / (double) stride)
                * (long) Math.ceil((imgH / downsample) / (double) stride);
        logger.info("DL overlay created: model={}, image={}x{}, downsample={}, "
                + "tileSize={}, padding={}, blendMode={}, est. tiles={}",
                metadata.getName(), imgW, imgH, downsample,
                tileSize, inputPadding, inferenceConfig.getBlendMode(), estTiles);
        if (estTiles > 500) {
            logger.warn("Large tile count ({}) -- overlay will take a long time to fill. "
                    + "Zoom into a smaller region for faster results.", estTiles);
        }
    }

    @Override
    public boolean supportsImage(ImageData<BufferedImage> imageData) {
        if (imageData == null || imageData.getServer() == null) return false;
        int imageChannels = imageData.getServer().nChannels();
        return imageChannels >= channelConfig.getNumChannels();
    }

    @Override
    public BufferedImage applyClassification(ImageData<BufferedImage> imageData,
                                              RegionRequest request) throws IOException {
        // Log first tile request at INFO level for diagnostic visibility
        if (firstTileLogged.compareAndSet(false, true)) {
            ImageServer<BufferedImage> diagServer = imageData.getServer();
            int imgW = diagServer.getWidth();
            int imgH = diagServer.getHeight();
            int tileSize = inferenceConfig.getTileSize();
            int stride = tileSize - 2 * inputPadding;  // visible stride (padding is per-side)
            int estTilesX = (int) Math.ceil((imgW / downsample) / (double) stride);
            int estTilesY = (int) Math.ceil((imgH / downsample) / (double) stride);
            logger.info("Overlay tile request started: image={}x{}, downsample={}, "
                    + "tileSize={}, padding={}, stride~={}, est. total tiles~={}",
                    imgW, imgH, downsample, tileSize, inputPadding, stride,
                    estTilesX * estTilesY);
            logger.info("First tile request: stride=({},{}) {}x{} @ downsample={}, "
                    + "expanded to tileSize={} with padding={} (real context for interior tiles)",
                    request.getX(), request.getY(),
                    request.getWidth(), request.getHeight(),
                    request.getDownsample(), tileSize, inputPadding);
        }

        // Guard: reject oversized tile requests. QuPath sometimes sends the entire
        // viewport as a single region (especially when zoomed out on large images).
        // Trying to read/process a region much larger than our tile size would OOM
        // or hang indefinitely. Return empty and warn once.
        int requestPixelsW = (int) (request.getWidth() / request.getDownsample());
        int requestPixelsH = (int) (request.getHeight() / request.getDownsample());
        int maxTilePixels = inferenceConfig.getTileSize() + 2 * inputPadding;
        if (requestPixelsW > maxTilePixels * 2 || requestPixelsH > maxTilePixels * 2) {
            if (oversizedWarned.compareAndSet(false, true)) {
                logger.warn("Rejecting oversized tile request: {}x{} pixels "
                        + "(max expected ~{}x{}). QuPath is requesting the full "
                        + "viewport as a single tile -- zoom in for the overlay to work.",
                        requestPixelsW, requestPixelsH, maxTilePixels, maxTilePixels);
            }
            return createEmptyClassificationImage(request);
        }

        // If shutting down (overlay being removed), return blank image instead of throwing.
        // Throwing IOException here causes QuPath's PixelClassificationOverlay to log ERROR,
        // which is noisy and misleading during normal overlay removal.
        if (shuttingDown || Thread.currentThread().isInterrupted()) {
            return createEmptyClassificationImage(request);
        }

        // Circuit breaker: stop retrying after persistent server errors
        if (consecutiveErrors.get() >= MAX_CONSECUTIVE_ERRORS) {
            throw new IOException("Classification disabled after " + MAX_CONSECUTIVE_ERRORS +
                    " consecutive server errors: " + lastErrorMessage);
        }

        // Detect image switch: when the viewer changes to a different image,
        // invalidate the blend cache and normalization stats so stale tiles
        // from the previous image are not reused.
        String serverPath = imageData.getServer().getPath();
        if (currentServerPath != null && !currentServerPath.equals(serverPath)) {
            logger.info("Image changed ({}), clearing blend cache and normalization stats",
                    imageData.getServer().getMetadata().getName());
            blendCache.clear();
            synchronized (statsLock) {
                channelConfigWithStats = null;
            }
            consecutiveErrors.set(0);
            errorNotified.set(false);
            contextResizeWarned.set(false);
        }
        currentServerPath = serverPath;

        // Cache-hit fast path: if this tile's prob map is already cached,
        // skip inference entirely and just argmax from cache.
        // ProbMaps are stride-sized (reflection padding is cropped on Python side),
        // so there is no overlap between adjacent tiles and no blending is possible.
        float[][][] cachedProbMap = blendCache.getIfCached(request.getX(), request.getY());
        if (cachedProbMap != null) {
            int cachedH = cachedProbMap.length;
            int cachedW = cachedProbMap[0].length;
            logger.debug("Cache hit at ({}, {}), dims={}x{}, cache size={}",
                    request.getX(), request.getY(), cachedW, cachedH, blendCache.size());
            consecutiveErrors.set(0);
            return createClassIndexImage(cachedProbMap, cachedW, cachedH);
        }

        ImageServer<BufferedImage> server = imageData.getServer();

        // Lazily compute image-level normalization stats on first tile request.
        // Capture into a local variable to avoid TOCTOU race: another thread
        // could null out the volatile field (via image-switch detection) between
        // our check and the use ~80 lines below.
        ChannelConfiguration channelCfg = channelConfigWithStats;
        if (channelCfg == null) {
            synchronized (statsLock) {
                channelCfg = channelConfigWithStats;
                if (channelCfg == null) {
                    channelCfg = NormalizationStatsComputer.compute(
                            server, metadata, channelConfig, contextScale, downsample);
                    channelConfigWithStats = channelCfg;
                }
            }
        }

        // Read expanded region from image for real context at tile boundaries.
        // QuPath sends stride-sized requests, but the model needs tileSize input.
        // Interior tiles: expanded = tileSize (real context on all sides).
        // Edge tiles: expanded < tileSize (clipped to image bounds).
        RegionRequest expanded = expandRequest(request, server);
        BufferedImage tileImage = server.readRegion(expanded);
        if (tileImage == null) {
            throw new IOException("Failed to read tile at " + expanded);
        }

        // Encode detail tile as raw binary (uint8 fast path for simple RGB, float32 for N-channel)
        String dtype;
        byte[] detailBytes;
        int detailChannels;
        if (TileEncoder.isSimpleRgb(tileImage)) {
            dtype = "uint8";
            detailBytes = TileEncoder.encodeTileRaw(tileImage);
            detailChannels = 3;
        } else {
            dtype = "float32";
            detailBytes = TileEncoder.encodeTileRawFloat(tileImage,
                    channelConfig.getSelectedChannels());
            detailChannels = channelConfig.getSelectedChannels().isEmpty()
                    ? tileImage.getRaster().getNumBands()
                    : channelConfig.getSelectedChannels().size();
        }

        // When multi-scale context is enabled, extract a context tile and interleave
        // channels per pixel: [detail_ch0..N, context_ch0..N] for each pixel.
        byte[] rawBytes;
        int numChannels;
        if (contextScale > 1) {
            BufferedImage contextImage = readContextTile(server, request,
                    tileImage.getWidth(), tileImage.getHeight());
            byte[] contextBytes;
            if ("uint8".equals(dtype)) {
                contextBytes = TileEncoder.encodeTileRaw(contextImage);
            } else {
                contextBytes = TileEncoder.encodeTileRawFloat(contextImage,
                        channelConfig.getSelectedChannels());
            }
            // Interleave detail + context channels per pixel (not sequential concat)
            int numPixels = tileImage.getWidth() * tileImage.getHeight();
            int bytesPerChannel = "uint8".equals(dtype) ? 1 : Float.BYTES;
            if (contextBytes.length != detailBytes.length) {
                // Defense-in-depth: if readContextTile resize didn't produce exact
                // dimensions (QuPath rounding at different downsamples), fall back
                // to detail-only rather than crashing with ArrayIndexOutOfBoundsException
                logger.warn("Context tile byte length ({}) != detail tile byte length ({}), " +
                        "skipping context for this tile", contextBytes.length, detailBytes.length);
                rawBytes = detailBytes;
                numChannels = detailChannels;
            } else {
                rawBytes = TileEncoder.interleaveContextChannels(
                        detailBytes, contextBytes, numPixels, detailChannels, bytesPerChannel);
                numChannels = detailChannels * 2;
            }
        } else {
            rawBytes = detailBytes;
            numChannels = detailChannels;
        }

        String tileId = String.format("%d_%d_%d_%d",
                request.getX(), request.getY(), request.getWidth(), request.getHeight());

        try {
            // Determine reflection padding.
            // Interior tiles: expanded image = tileSize, no reflection needed
            // (model sees real neighboring image data on all sides).
            // Edge tiles: expanded < tileSize, use reflection padding so the
            // model always gets >= tileSize input for consistent predictions.
            int expandedW = tileImage.getWidth();
            int expandedH = tileImage.getHeight();
            int tileSize = inferenceConfig.getTileSize();
            int reflectionPadding = (expandedW >= tileSize && expandedH >= tileSize)
                    ? 0 : inputPadding;
            PixelInferenceResult result = backend.runPixelInferenceBinary(
                    modelDirPath, rawBytes, List.of(tileId),
                    tileImage.getHeight(), tileImage.getWidth(), numChannels,
                    dtype, channelCfg, inferenceConfig, sharedTempDir,
                    reflectionPadding);

            // Fall back to JSON/PNG path if binary endpoint unavailable.
            // Context tiles are NOT supported in the JSON fallback path -- the model
            // receives only detail channels, producing incorrect results for context models.
            if (result == null) {
                if (contextScale > 1) {
                    logger.warn("Binary inference failed for context_scale={} model; " +
                            "JSON fallback does not support multi-scale context. " +
                            "Results will be incorrect.", contextScale);
                }
                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                javax.imageio.ImageIO.write(tileImage, "png", baos);
                String encoded = "data:image/png;base64," +
                        java.util.Base64.getEncoder().encodeToString(baos.toByteArray());
                List<ClassifierClient.TileData> tiles = List.of(
                        new ClassifierClient.TileData(tileId, encoded,
                                request.getX(), request.getY()));
                result = backend.runPixelInference(
                        modelDirPath, tiles, channelCfg, inferenceConfig,
                        sharedTempDir, reflectionPadding);
            }

            if (result == null || result.outputPaths() == null || result.outputPaths().isEmpty()) {
                throw new IOException("No inference result returned for tile");
            }

            String outputPath = result.outputPaths().get(tileId);
            if (outputPath == null) {
                throw new IOException("No output path for tile " + tileId);
            }

            // Read probability map
            int tileWidth = tileImage.getWidth();
            int tileHeight = tileImage.getHeight();
            float[][][] probMap = ClassifierClient.readProbabilityMap(
                    Path.of(outputPath), result.numClasses(), tileHeight, tileWidth);

            // Clean up this tile's prob map file (shared dir persists)
            try {
                Files.deleteIfExists(Path.of(outputPath));
            } catch (IOException e) {
                logger.debug("Failed to delete tile output: {}", outputPath);
            }

            // ProbMap matches expanded image dimensions (Python crops any
            // reflection padding back to the expanded size). Crop to the
            // stride region that QuPath expects.
            float[][][] strideProbMap = cropToStride(probMap, request);
            int strideW = strideProbMap[0].length;
            int strideH = strideProbMap.length;

            // Cache stride-sized probMap for fast path on repaint
            blendCache.cache(request.getX(), request.getY(), strideProbMap);

            // Success -- reset error counter and log progress
            consecutiveErrors.set(0);
            int completed = tilesCompleted.incrementAndGet();
            if (completed <= 10 || completed % 50 == 0) {
                logger.info("Overlay tile {} completed at ({}, {}), dims={}x{}, "
                        + "expanded={}x{}, reflPad={}",
                        completed, request.getX(), request.getY(),
                        strideW, strideH, expandedW, expandedH, reflectionPadding);
            }
            return createClassIndexImage(strideProbMap, strideW, strideH);

        } catch (IOException e) {
            // During shutdown, interrupted threads and missing temp files are expected.
            // Return blank image instead of re-throwing to avoid QuPath logging ERROR.
            if (shuttingDown || Thread.currentThread().isInterrupted()
                    || e instanceof java.io.InterruptedIOException) {
                logger.debug("Classification interrupted during shutdown");
                return createEmptyClassificationImage(request);
            }

            // "thread death" is transient (Python worker thread contention under high
            // concurrency). QuPath re-requests the tile on repaint, so don't count it
            // toward the circuit breaker -- it would trip on startup when many tiles
            // are requested simultaneously.
            String msg = e.getMessage() != null ? e.getMessage() : "";
            if (msg.toLowerCase().contains("thread death")) {
                logger.debug("Transient thread death for tile, will retry on repaint");
                throw e;
            }

            int errorCount = consecutiveErrors.incrementAndGet();
            lastErrorMessage = e.getMessage();

            if (errorCount >= MAX_CONSECUTIVE_ERRORS && errorNotified.compareAndSet(false, true)) {
                logger.error("Classification overlay disabled after {} consecutive errors: {}",
                        errorCount, e.getMessage());
                Platform.runLater(() -> {
                    var alert = new javafx.scene.control.Alert(
                            javafx.scene.control.Alert.AlertType.ERROR);
                    alert.setTitle("Classification Error");
                    alert.setHeaderText("Classification overlay has been disabled");
                    alert.setContentText("The server returned repeated errors:\n" +
                            lastErrorMessage + "\n\n" +
                            "Remove the overlay and check the server connection.");
                    alert.show();
                });
            }
            throw e;
        }
    }

    /**
     * Signals that this classifier is shutting down. In-flight tile requests
     * will detect this and exit without counting errors or showing dialogs.
     * Called by {@link OverlayService} before stopping the overlay.
     */
    public void shutdown() {
        shuttingDown = true;
    }

    /**
     * Cleans up resources used by this classifier (shared temp directory).
     * Called by {@link OverlayService} when the overlay is removed.
     */
    public void cleanup() {
        shuttingDown = true;

        // Shutdown the blend cache (clears data + stops refresh scheduler)
        blendCache.shutdown();

        try {
            if (sharedTempDir != null && Files.exists(sharedTempDir)) {
                Files.walk(sharedTempDir)
                        .sorted(Comparator.reverseOrder())
                        .forEach(path -> {
                            try { Files.deleteIfExists(path); }
                            catch (IOException ignored) {}
                        });
                logger.debug("Cleaned up shared temp dir: {}", sharedTempDir);
            }
        } catch (IOException e) {
            logger.warn("Failed to clean up shared temp dir: {}", sharedTempDir, e);
        }
    }

    @Override
    public PixelClassifierMetadata getMetadata() {
        return pixelMetadata;
    }

    /**
     * Reads a context tile centered on the same location as the given detail request,
     * but covering a larger area (contextScale times in each dimension) and downsampled
     * to the same pixel dimensions as the detail tile.
     * <p>
     * Three-tier strategy for handling image edges:
     * <ol>
     *   <li>Ideal: context region fits entirely within the image</li>
     *   <li>Clamped: context region is shifted to fit (image >= context size)</li>
     *   <li>Resized: image smaller than context region -- read entire image, resize to match</li>
     * </ol>
     *
     * @param server        image server
     * @param detailRequest the detail tile region request
     * @param expectedW     expected output width (detail tile pixel width)
     * @param expectedH     expected output height (detail tile pixel height)
     * @return context tile with dimensions matching expectedW x expectedH
     */
    private BufferedImage readContextTile(ImageServer<BufferedImage> server,
                                          RegionRequest detailRequest,
                                          int expectedW, int expectedH) throws IOException {
        int detailX = detailRequest.getX();
        int detailY = detailRequest.getY();
        int detailW = detailRequest.getWidth();
        int detailH = detailRequest.getHeight();

        // Context region covers contextScale times the area in each dimension
        int contextW = detailW * contextScale;
        int contextH = detailH * contextScale;

        int imgW = server.getWidth();
        int imgH = server.getHeight();

        int cx, cy, readW, readH;

        if (imgW >= contextW && imgH >= contextH) {
            // Tier 1/2: Image large enough -- clamp position to keep context within bounds
            int centerX = detailX + detailW / 2;
            int centerY = detailY + detailH / 2;
            cx = centerX - contextW / 2;
            cy = centerY - contextH / 2;
            cx = Math.max(0, Math.min(cx, imgW - contextW));
            cy = Math.max(0, Math.min(cy, imgH - contextH));
            readW = contextW;
            readH = contextH;
        } else {
            // Tier 3: Image smaller than context region -- read entire image
            cx = 0;
            cy = 0;
            readW = imgW;
            readH = imgH;
            if (contextResizeWarned.compareAndSet(false, true)) {
                logger.warn("Image ({}x{}) smaller than context region ({}x{}) -- " +
                        "reading entire image and resizing to match detail tile dimensions",
                        imgW, imgH, contextW, contextH);
            }
        }

        // Read at higher downsample so output has same pixel dimensions as detail tile
        double contextDownsample = detailRequest.getDownsample() * contextScale;
        RegionRequest contextRequest = RegionRequest.createInstance(
                server.getPath(), contextDownsample,
                cx, cy, readW, readH,
                detailRequest.getZ(), detailRequest.getT());

        BufferedImage contextImage = server.readRegion(contextRequest);
        if (contextImage == null) {
            throw new IOException("Failed to read context tile at " + contextRequest);
        }

        // Resize to expected dimensions if the read region was smaller than expected
        if (contextImage.getWidth() != expectedW || contextImage.getHeight() != expectedH) {
            BufferedImage resized = new BufferedImage(expectedW, expectedH, contextImage.getType());
            java.awt.Graphics2D g = resized.createGraphics();
            g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                    java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(contextImage, 0, 0, expectedW, expectedH, null);
            g.dispose();
            contextImage = resized;
        }

        return contextImage;
    }

    /**
     * Creates a TYPE_BYTE_INDEXED image where each pixel value is the argmax
     * class index from the probability map. Applies Gaussian smoothing to the
     * probability maps before argmax if configured (sigma > 0).
     */
    private BufferedImage createClassIndexImage(float[][][] probMap, int width, int height) {
        // Smooth probability maps before argmax to reduce noisy per-pixel predictions
        double sigma = inferenceConfig.getOverlaySmoothingSigma();
        if (sigma > 0) {
            probMap = gaussianSmoothProbabilities(probMap, width, height, sigma);
        }

        BufferedImage indexed = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_INDEXED, colorModel);
        var raster = indexed.getRaster();

        int numClasses = probMap[0][0].length;
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
                raster.setSample(x, y, 0, maxClass);
            }
        }

        return indexed;
    }

    /**
     * Applies separable Gaussian smoothing to each class channel of a probability map.
     * <p>
     * This smooths noisy per-pixel predictions before argmax, producing cleaner
     * classification boundaries without the tile-boundary artifacts that blending
     * approaches introduce. Uses a 1D Gaussian kernel applied horizontally then
     * vertically (separable convolution) for efficiency.
     *
     * @param probMap probability map [height][width][numClasses]
     * @param width   map width
     * @param height  map height
     * @param sigma   Gaussian sigma in pixels
     * @return smoothed probability map (new array)
     */
    private static float[][][] gaussianSmoothProbabilities(float[][][] probMap,
                                                            int width, int height, double sigma) {
        int radius = (int) Math.ceil(sigma * 2.5);
        if (radius < 1) return probMap;

        // Build 1D Gaussian kernel
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
     * Creates a blank classification image (all pixels = class 0) for the given request.
     * Used during shutdown to return a valid image instead of throwing, which prevents
     * QuPath's PixelClassificationOverlay from logging spurious ERROR messages.
     */
    private BufferedImage createEmptyClassificationImage(RegionRequest request) {
        int width = (int) (request.getWidth() / request.getDownsample());
        int height = (int) (request.getHeight() / request.getDownsample());
        // Ensure at least 1x1 to avoid invalid image
        width = Math.max(1, width);
        height = Math.max(1, height);
        return new BufferedImage(width, height, BufferedImage.TYPE_BYTE_INDEXED, colorModel);
    }

    /**
     * Builds the QuPath PixelClassifierMetadata from our classifier metadata.
     * <p>
     * Tile overlap (inputPadding) is computed from the preferred physical distance
     * in micrometers using the image's pixel calibration. This ensures consistent
     * CNN context at tile boundaries regardless of objective/resolution.
     */
    private PixelClassifierMetadata buildPixelMetadata(ImageData<BufferedImage> imageData) {
        PixelCalibration cal = imageData.getServer().getPixelCalibration();

        // Scale calibration by downsample factor so QuPath requests tiles at the correct resolution
        if (downsample > 1.0) {
            cal = cal.createScaledInstance(downsample, downsample);
        }

        // Build classification labels map
        Map<Integer, PathClass> labels = new LinkedHashMap<>();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
        List<ImageChannel> channels = new ArrayList<>();

        for (ClassifierMetadata.ClassInfo classInfo : classes) {
            int color = parseClassColor(classInfo.color(), classInfo.index());
            PathClass pathClass = PathClass.fromString(classInfo.name(), color);
            // Use the actual color from the resolved PathClass (may differ from metadata
            // if QuPath already has a cached PathClass with a different color)
            int resolvedColor = pathClass.getColor();
            resolvedClassColors.put(classInfo.index(), resolvedColor);
            labels.put(classInfo.index(), pathClass);
            channels.add(ImageChannel.getInstance(classInfo.name(), resolvedColor));
        }

        int tileSize = inferenceConfig.getTileSize();
        int padding = computeOverlayPadding(tileSize);

        return new PixelClassifierMetadata.Builder()
                .inputResolution(cal)
                .inputShape(tileSize, tileSize)
                .inputPadding(padding)
                .setChannelType(ImageServerMetadata.ChannelType.CLASSIFICATION)
                .outputPixelType(PixelType.UINT8)
                .classificationLabels(labels)
                .outputChannels(channels)
                .build();
    }

    /**
     * Computes overlay tile padding from InferenceConfig overlap.
     * <p>
     * Uses the larger of the configured overlap and tileSize/4 to ensure
     * sufficient CNN context for artifact-free tile boundaries.
     * The tileSize/4 floor guarantees at least 25% padding even when the
     * user sets a low overlap percentage.
     *
     * @param tileSize tile size in pixels
     * @return padding in pixels (at least 64, at most tileSize/2)
     */
    private int computeOverlayPadding(int tileSize) {
        // inputPadding is PER-SIDE: QuPath's visible stride = tileSize - 2*inputPadding.
        // Padding must be strictly less than tileSize/2 or the stride collapses to zero
        // and QuPath sends the entire viewport as a single (impossibly large) tile request.
        int configOverlap = inferenceConfig.getOverlap();
        int minContextPadding = tileSize / 4;  // 25% minimum padding per side
        int padding = Math.max(configOverlap, minContextPadding);
        // Clamp: at least 64, at most 3/8 of tileSize (ensures >= 25% visible stride)
        int maxPadding = Math.max(64, tileSize * 3 / 8);
        return Math.max(64, Math.min(padding, maxPadding));
    }

    /**
     * Expands a stride-sized RegionRequest by inputPadding on each side,
     * clipped to image bounds. QuPath sends stride-sized tiles without
     * surrounding context; this provides the model with the full receptive
     * field it was trained with.
     */
    private RegionRequest expandRequest(RegionRequest request, ImageServer<?> server) {
        double ds = request.getDownsample();
        int padFullRes = (int) (inputPadding * ds);
        int expX = Math.max(0, request.getX() - padFullRes);
        int expY = Math.max(0, request.getY() - padFullRes);
        int expRight = Math.min(server.getWidth(),
                request.getX() + request.getWidth() + padFullRes);
        int expBottom = Math.min(server.getHeight(),
                request.getY() + request.getHeight() + padFullRes);
        return RegionRequest.createInstance(
                server.getPath(), ds,
                expX, expY, expRight - expX, expBottom - expY,
                request.getZ(), request.getT());
    }

    /**
     * Crops an expanded probability map to the stride region that QuPath expects.
     * The stride region corresponds to the original (unexpanded) request within
     * the expanded tile. At image edges where expansion was clipped, the offset
     * is reduced accordingly.
     */
    private float[][][] cropToStride(float[][][] probMap, RegionRequest request) {
        int strideW = (int) (request.getWidth() / request.getDownsample());
        int strideH = (int) (request.getHeight() / request.getDownsample());
        int expandedW = probMap[0].length;
        int expandedH = probMap.length;

        // If already stride-sized (no expansion happened), return as-is
        if (expandedW <= strideW && expandedH <= strideH) {
            return probMap;
        }

        // Compute how much padding was actually added on the left/top
        // (less at image edges due to clipping)
        double ds = request.getDownsample();
        int padFullRes = (int) (inputPadding * ds);
        int expandedX = Math.max(0, request.getX() - padFullRes);
        int expandedY = Math.max(0, request.getY() - padFullRes);
        int offsetX = (int) ((request.getX() - expandedX) / ds);
        int offsetY = (int) ((request.getY() - expandedY) / ds);

        // Clamp crop to available dimensions
        int cropW = Math.min(strideW, expandedW - offsetX);
        int cropH = Math.min(strideH, expandedH - offsetY);

        int numClasses = probMap[0][0].length;
        float[][][] cropped = new float[cropH][cropW][numClasses];
        for (int y = 0; y < cropH; y++) {
            for (int x = 0; x < cropW; x++) {
                System.arraycopy(probMap[y + offsetY][x + offsetX], 0,
                        cropped[y][x], 0, numClasses);
            }
        }
        return cropped;
    }

    /**
     * Builds an IndexColorModel for the class indices, used for TYPE_BYTE_INDEXED images.
     * Uses colors resolved from PathClass cache (populated by buildPixelMetadata) so that
     * overlay colors match the annotation class colors the user sees in QuPath.
     */
    private IndexColorModel buildColorModel() {
        byte[] r = new byte[256];
        byte[] g = new byte[256];
        byte[] b = new byte[256];
        byte[] a = new byte[256];

        for (Map.Entry<Integer, Integer> entry : resolvedClassColors.entrySet()) {
            int idx = entry.getKey();
            if (idx < 0 || idx >= 256) continue;
            int color = entry.getValue();
            r[idx] = (byte) ColorTools.red(color);
            g[idx] = (byte) ColorTools.green(color);
            b[idx] = (byte) ColorTools.blue(color);
            a[idx] = (byte) 255;
        }

        return new IndexColorModel(8, 256, r, g, b, a);
    }

    /** Distinct color palette for fallback when class metadata lacks colors. */
    private static final int[][] FALLBACK_PALETTE = {
            {255, 0, 0}, {0, 170, 0}, {0, 0, 255}, {255, 255, 0},
            {255, 0, 255}, {0, 255, 255}, {255, 136, 0}, {136, 0, 255}
    };

    /**
     * Parses a hex color string to a packed RGB integer (QuPath format).
     * Falls back to a distinct palette color for the given class index.
     */
    private static int parseClassColor(String colorStr, int classIndex) {
        if (colorStr == null || colorStr.isEmpty() || "#808080".equals(colorStr)) {
            // Use distinct fallback color instead of gray
            int[] c = FALLBACK_PALETTE[classIndex % FALLBACK_PALETTE.length];
            return ColorTools.packRGB(c[0], c[1], c[2]);
        }
        try {
            String hex = colorStr.startsWith("#") ? colorStr.substring(1) : colorStr;
            int rgb = Integer.parseInt(hex, 16);
            return ColorTools.packRGB(
                    (rgb >> 16) & 0xFF,
                    (rgb >> 8) & 0xFF,
                    rgb & 0xFF);
        } catch (NumberFormatException e) {
            int[] c = FALLBACK_PALETTE[classIndex % FALLBACK_PALETTE.length];
            return ColorTools.packRGB(c[0], c[1], c[2]);
        }
    }

    /**
     * Parses a hex color string to a packed RGB integer (QuPath format).
     */
    private static int parseClassColor(String colorStr) {
        return parseClassColor(colorStr, 0);
    }
}
