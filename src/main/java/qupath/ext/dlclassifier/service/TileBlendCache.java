package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.InferenceConfig;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Manages a bounded cache of probability maps for tile boundary blending.
 * <p>
 * The cache stores raw probability maps from inference results keyed by
 * tile request coordinates. When blending is requested, it looks up
 * neighboring tiles (left, right, top, bottom) and applies a configurable
 * cross-fade at boundaries to eliminate visible seams in the overlay.
 * Supports LINEAR (ramp), GAUSSIAN (cosine bell), and CENTER_CROP modes.
 * <p>
 * Also tracks observed tile positions to compute the empirical step
 * between tiles, which is used to locate neighbor tiles in the cache.
 * <p>
 * After the initial batch of tiles is cached, a debounced one-shot
 * overlay refresh is scheduled so that all tiles get re-rendered with
 * proper bidirectional blending.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TileBlendCache {

    private static final Logger logger = LoggerFactory.getLogger(TileBlendCache.class);

    /** Cache of probability maps. Key = (requestX, requestY) packed into long. */
    private final ConcurrentHashMap<Long, float[][][]> probCache = new ConcurrentHashMap<>();

    /** Tracks insertion order for LRU eviction. */
    private final ConcurrentLinkedDeque<Long> probCacheOrder = new ConcurrentLinkedDeque<>();

    /** Maximum cached probability maps. */
    private final int maxSize;

    /** QuPath's inputPadding: extra pixels on each side of the visible tile. */
    private final int inputPadding;

    /** Observed tile request X positions for empirical step computation. */
    private final ConcurrentSkipListSet<Integer> seenTileX = new ConcurrentSkipListSet<>();
    /** Observed tile request Y positions for empirical step computation. */
    private final ConcurrentSkipListSet<Integer> seenTileY = new ConcurrentSkipListSet<>();
    /** Empirical step between tiles in full-res X coords. -1 = unknown. */
    private volatile int empiricalStepX = -1;
    /** Empirical step between tiles in full-res Y coords. -1 = unknown. */
    private volatile int empiricalStepY = -1;

    /** Debounced scheduler for viewer refresh after new tiles are cached. */
    private final ScheduledExecutorService refreshScheduler =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "dl-overlay-refresh");
                t.setDaemon(true);
                return t;
            });
    private volatile ScheduledFuture<?> pendingRefresh;

    /** Guards against repeated overlay refreshes -- only one refresh per session. */
    private volatile boolean hasRefreshed = false;

    /** Callback invoked when a deferred overlay refresh fires. */
    private final Runnable refreshCallback;

    /** Blend mode for weight computation (LINEAR, GAUSSIAN/cosine, CENTER_CROP). */
    private final InferenceConfig.BlendMode blendMode;

    /** Maximum blend distance in pixels. -1 = use full inputPadding. */
    private final int maxBlendDist;

    /**
     * Creates a new tile blend cache.
     *
     * @param maxSize         maximum number of probability maps to cache
     * @param inputPadding    QuPath's inputPadding (extra pixels on each side of visible tile)
     * @param blendMode       blend mode for weight computation
     * @param maxBlendDist    maximum blend distance (-1 = use full inputPadding)
     * @param refreshCallback called when a deferred overlay refresh fires
     */
    public TileBlendCache(int maxSize, int inputPadding,
                          InferenceConfig.BlendMode blendMode, int maxBlendDist,
                          Runnable refreshCallback) {
        this.maxSize = maxSize;
        this.inputPadding = inputPadding;
        this.blendMode = blendMode;
        this.maxBlendDist = maxBlendDist;
        this.refreshCallback = refreshCallback;
    }

    /**
     * Packs (x, y) request coordinates into a single long key.
     */
    public static long cacheKey(int requestX, int requestY) {
        return ((long) requestX << 32) | (requestY & 0xFFFFFFFFL);
    }

    /**
     * Returns the cached probability map for the given tile coordinates, or null.
     */
    public float[][][] getIfCached(int requestX, int requestY) {
        return probCache.get(cacheKey(requestX, requestY));
    }

    /**
     * Caches a probability map and tracks tile positions for step computation.
     * Evicts the oldest entry if over capacity.
     */
    public void cache(int requestX, int requestY, float[][][] probMap) {
        long key = cacheKey(requestX, requestY);
        probCache.put(key, probMap);
        probCacheOrder.addLast(key);

        // LRU eviction
        while (probCache.size() > maxSize) {
            Long oldest = probCacheOrder.pollFirst();
            if (oldest != null) {
                probCache.remove(oldest);
            }
        }

        // Track tile positions for empirical step computation
        seenTileX.add(requestX);
        seenTileY.add(requestY);
        if (empiricalStepX < 0 || empiricalStepY < 0) {
            computeEmpiricalStep();
        }
    }

    /**
     * Returns the current number of cached probability maps.
     */
    public int size() {
        return probCache.size();
    }

    /**
     * Returns the empirical tile step in X, or -1 if not yet computed.
     */
    public int getEmpiricalStepX() {
        return empiricalStepX;
    }

    /**
     * Returns the empirical tile step in Y, or -1 if not yet computed.
     */
    public int getEmpiricalStepY() {
        return empiricalStepY;
    }

    /**
     * Blends this tile's probability map with cached neighbor probability maps
     * at the visible tile boundaries using overlapping predictions.
     * <p>
     * QuPath tiles with {@code inputPadding} have overlapping coverage: tile A's
     * right padding and tile B's left padding predict the same image region.
     * This method blends at the <b>visible</b> boundary (where QuPath crops the
     * padding) using those overlapping predictions:
     * <ul>
     *   <li>At the visible boundary (d=0): ~50% self + ~50% neighbor</li>
     *   <li>Deeper inside visible region (d=blendDist): 100% self</li>
     * </ul>
     * <p>
     * The blend distance is capped at {@code inputPadding} (or a configured max)
     * since we cannot blend beyond the overlap region.
     * <p>
     * Horizontal blending (left/right) is applied first, then vertical (top/bottom).
     * This sequential approach handles corners naturally.
     *
     * @param probMap   this tile's raw probability map [height][width][numClasses]
     * @param requestX  tile request X coordinate (full-resolution image coords)
     * @param requestY  tile request Y coordinate (full-resolution image coords)
     * @param width     prob map width (= tileSize, the inputShape given to QuPath)
     * @param height    prob map height (= tileSize, the inputShape given to QuPath)
     * @return blended probability map (new array, original not modified)
     */
    public float[][][] blendWithNeighbors(float[][][] probMap, int requestX, int requestY,
                                           int width, int height) {
        // CENTER_CROP: every visible pixel is at the center of its tile, no blending needed
        if (blendMode == InferenceConfig.BlendMode.CENTER_CROP) {
            return probMap;
        }

        // Need empirical step to locate neighbors
        int stepX = empiricalStepX;
        int stepY = empiricalStepY;
        if (stepX <= 0 && stepY <= 0) {
            return probMap;  // Step not yet computed, skip blending
        }

        if (inputPadding <= 0) {
            return probMap;  // No padding = no overlap = can't blend
        }

        int numClasses = probMap[0][0].length;

        // Look up cached neighbors using empirical step
        float[][][] left   = (stepX > 0) ? probCache.get(cacheKey(requestX - stepX, requestY)) : null;
        float[][][] right  = (stepX > 0) ? probCache.get(cacheKey(requestX + stepX, requestY)) : null;
        float[][][] top    = (stepY > 0) ? probCache.get(cacheKey(requestX, requestY - stepY)) : null;
        float[][][] bottom = (stepY > 0) ? probCache.get(cacheKey(requestX, requestY + stepY)) : null;

        if (left == null && right == null && top == null && bottom == null) {
            return probMap;  // No neighbors available
        }

        // Blend distance: configurable max or full inputPadding
        int blendDist = (maxBlendDist > 0) ? Math.min(inputPadding, maxBlendDist) : inputPadding;

        // Create a copy for blending (don't modify cached original)
        float[][][] blended = deepCopyProbMap(probMap);

        // --- Horizontal blending (uses probMap as self source) ---

        // Right visible boundary: self's right edge of visible region overlaps
        // with right neighbor's left padding region (same image location)
        if (right != null) {
            int rh = Math.min(height, right.length);
            for (int y = 0; y < rh; y++) {
                for (int d = 0; d < blendDist; d++) {
                    int xSelf  = width - inputPadding - 1 - d;  // visible col near right edge
                    int xRight = inputPadding - 1 - d;          // same image loc in right neighbor
                    if (xSelf < 0 || xRight < 0 || xRight >= right[y].length) break;
                    float wSelf = blendWeight(d, blendDist);
                    float wNeighbor = 1.0f - wSelf;
                    int nc = Math.min(numClasses, right[y][xRight].length);
                    for (int c = 0; c < nc; c++) {
                        blended[y][xSelf][c] = wSelf * probMap[y][xSelf][c]
                                + wNeighbor * right[y][xRight][c];
                    }
                }
            }
        }

        // Left visible boundary: self's left edge of visible region overlaps
        // with left neighbor's right padding region (same image location)
        if (left != null) {
            int lh = Math.min(height, left.length);
            for (int y = 0; y < lh; y++) {
                for (int d = 0; d < blendDist; d++) {
                    int xSelf = inputPadding + d;           // visible col near left edge
                    int xLeft = width - inputPadding + d;   // same image loc in left neighbor
                    if (xSelf >= width || xLeft >= left[y].length) break;
                    float wSelf = blendWeight(d, blendDist);
                    float wNeighbor = 1.0f - wSelf;
                    int nc = Math.min(numClasses, left[y][xLeft].length);
                    for (int c = 0; c < nc; c++) {
                        blended[y][xSelf][c] = wSelf * probMap[y][xSelf][c]
                                + wNeighbor * left[y][xLeft][c];
                    }
                }
            }
        }

        // --- Vertical blending (uses blended as self source for corner handling) ---

        // Bottom visible boundary: self's bottom edge of visible region overlaps
        // with bottom neighbor's top padding region
        if (bottom != null) {
            for (int d = 0; d < blendDist; d++) {
                int ySelf   = height - inputPadding - 1 - d;  // visible row near bottom edge
                int yBottom = inputPadding - 1 - d;            // same image loc in bottom neighbor
                if (ySelf < 0 || yBottom < 0 || yBottom >= bottom.length) break;
                float wSelf = blendWeight(d, blendDist);
                float wNeighbor = 1.0f - wSelf;
                int bw = Math.min(width, bottom[yBottom].length);
                for (int x = 0; x < bw; x++) {
                    int nc = Math.min(numClasses, bottom[yBottom][x].length);
                    for (int c = 0; c < nc; c++) {
                        blended[ySelf][x][c] = wSelf * blended[ySelf][x][c]
                                + wNeighbor * bottom[yBottom][x][c];
                    }
                }
            }
        }

        // Top visible boundary: self's top edge of visible region overlaps
        // with top neighbor's bottom padding region
        if (top != null) {
            for (int d = 0; d < blendDist; d++) {
                int ySelf = inputPadding + d;            // visible row near top edge
                int yTop  = height - inputPadding + d;   // same image loc in top neighbor
                if (ySelf >= height || yTop >= top.length) break;
                float wSelf = blendWeight(d, blendDist);
                float wNeighbor = 1.0f - wSelf;
                int tw = Math.min(width, top[yTop].length);
                for (int x = 0; x < tw; x++) {
                    int nc = Math.min(numClasses, top[yTop][x].length);
                    for (int c = 0; c < nc; c++) {
                        blended[ySelf][x][c] = wSelf * blended[ySelf][x][c]
                                + wNeighbor * top[yTop][x][c];
                    }
                }
            }
        }

        return blended;
    }

    /**
     * Schedules a debounced, one-shot overlay refresh after the initial tile batch.
     * <p>
     * On the first render, tiles are computed without all neighbors cached, so blending
     * is incomplete. After the batch completes (debounced 1s after the last tile), the
     * overlay is recreated to force fresh tile requests. The cache-hit fast path serves
     * these re-requests instantly from the prob cache, now with all neighbors available
     * for proper bidirectional blending.
     * <p>
     * Only fires once per overlay session to avoid infinite refresh loops.
     */
    public void scheduleRefresh() {
        if (hasRefreshed) {
            logger.debug("BLEND scheduleRefresh skipped (already refreshed)");
            return;
        }

        ScheduledFuture<?> prev = pendingRefresh;
        if (prev != null) prev.cancel(false);
        logger.debug("BLEND scheduling refresh in 1s (cache size={})", probCache.size());
        pendingRefresh = refreshScheduler.schedule(() -> {
            hasRefreshed = true;
            try {
                logger.debug("Refreshing overlay for tile blending ({} cached prob maps)",
                        probCache.size());
                refreshCallback.run();
            } catch (Exception e) {
                logger.debug("Deferred overlay refresh failed: {}", e.getMessage());
            }
        }, 1000, TimeUnit.MILLISECONDS);
    }

    /**
     * Clears all cached data and resets state.
     */
    public void clear() {
        probCache.clear();
        probCacheOrder.clear();
        seenTileX.clear();
        seenTileY.clear();
        empiricalStepX = -1;
        empiricalStepY = -1;
        hasRefreshed = false;
    }

    /**
     * Clears all data and shuts down the refresh scheduler.
     */
    public void shutdown() {
        clear();
        ScheduledFuture<?> pending = pendingRefresh;
        if (pending != null) pending.cancel(false);
        refreshScheduler.shutdownNow();
    }

    // ==================== Internal Methods ====================

    /**
     * Computes the self-weight for a pixel at distance {@code d} from a tile boundary.
     * <p>
     * For GAUSSIAN mode, uses a cosine bell (smooth S-curve) that transitions
     * more gradually than linear -- better suited to ViT models where prediction
     * gradients are smooth due to global self-attention.
     *
     * @param d         distance from the boundary (0 = at boundary, blendDist-1 = interior)
     * @param blendDist total blend zone width in pixels
     * @return self-weight in [0.5, 1.0]
     */
    private float blendWeight(int d, int blendDist) {
        float t = (d + 0.5f) / blendDist;  // 0 at boundary, 1 at interior
        if (blendMode == InferenceConfig.BlendMode.GAUSSIAN) {
            // Cosine bell: smooth S-curve transition
            return (float) (0.5 * (1.0 + Math.cos(Math.PI * (1.0 - t))));
        }
        // LINEAR (default): current behavior
        return 0.5f + 0.5f * t;
    }

    /**
     * Computes the empirical step between tile requests by finding the minimum
     * non-zero gap between observed tile positions.
     */
    private void computeEmpiricalStep() {
        if (seenTileX.size() >= 2 && empiricalStepX < 0) {
            int minGap = Integer.MAX_VALUE;
            Integer prev = null;
            for (Integer pos : seenTileX) {
                if (prev != null) {
                    int gap = pos - prev;
                    if (gap > 0 && gap < minGap) minGap = gap;
                }
                prev = pos;
            }
            if (minGap != Integer.MAX_VALUE) {
                empiricalStepX = minGap;
                logger.info("BLEND empirical stepX = {} (from {} positions)",
                        minGap, seenTileX.size());
            }
        }
        if (seenTileY.size() >= 2 && empiricalStepY < 0) {
            int minGap = Integer.MAX_VALUE;
            Integer prev = null;
            for (Integer pos : seenTileY) {
                if (prev != null) {
                    int gap = pos - prev;
                    if (gap > 0 && gap < minGap) minGap = gap;
                }
                prev = pos;
            }
            if (minGap != Integer.MAX_VALUE) {
                empiricalStepY = minGap;
                logger.info("BLEND empirical stepY = {} (from {} positions)",
                        minGap, seenTileY.size());
            }
        }
    }

    /**
     * Creates a deep copy of a probability map so blending doesn't modify cached data.
     */
    private static float[][][] deepCopyProbMap(float[][][] src) {
        int h = src.length;
        float[][][] copy = new float[h][][];
        for (int y = 0; y < h; y++) {
            int w = src[y].length;
            copy[y] = new float[w][];
            for (int x = 0; x < w; x++) {
                copy[y][x] = src[y][x].clone();
            }
        }
        return copy;
    }
}
