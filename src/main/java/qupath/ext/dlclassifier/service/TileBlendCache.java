package qupath.ext.dlclassifier.service;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A bounded cache of the overlay's probability maps, keyed by tile request
 * coordinates, so a repaint does not re-run inference.
 * <p>
 * The name is historical: it once carried a cross-fade between neighbouring
 * tiles, which nothing ever called. See the note where that method used to
 * be. The overlay center-crops, and this class now only caches.
 * <p>
 * It also tracks observed tile positions to compute the empirical step
 * between tiles, and schedules a debounced one-shot overlay refresh after
 * the first batch so tiles that arrived before their neighbours get
 * repainted.
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

    /** Observed tile request X positions for empirical step computation. */
    private final ConcurrentSkipListSet<Integer> seenTileX = new ConcurrentSkipListSet<>();
    /** Observed tile request Y positions for empirical step computation. */
    private final ConcurrentSkipListSet<Integer> seenTileY = new ConcurrentSkipListSet<>();
    /** Empirical step between tiles in full-res X coords. -1 = unknown. */
    private volatile int empiricalStepX = -1;
    /** Empirical step between tiles in full-res Y coords. -1 = unknown. */
    private volatile int empiricalStepY = -1;

    /** Debounced scheduler for viewer refresh after new tiles are cached. */
    private final ScheduledExecutorService refreshScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "dl-overlay-refresh");
        t.setDaemon(true);
        return t;
    });

    private volatile ScheduledFuture<?> pendingRefresh;

    /** Cooldown period between overlay refreshes (ms). */
    private static final long REFRESH_COOLDOWN_MS = 5000;

    /** Timestamp of the last completed refresh. 0 = never refreshed. */
    private volatile long lastRefreshTime = 0;

    /** Callback invoked when a deferred overlay refresh fires. */
    private final Runnable refreshCallback;

    /**
     * Creates a new tile blend cache.
     *
     * @param maxSize         maximum number of probability maps to cache
     * @param refreshCallback called when a deferred overlay refresh fires
     */
    public TileBlendCache(int maxSize, Runnable refreshCallback) {
        this.maxSize = maxSize;
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

    /*
     * blendWithNeighbors() lived here and was never called -- verified by a
     * repo-wide search on 2026-09-27, with no caller in main, test or
     * resources. The overlay caches the center-cropped probability map and
     * returns it as-is, so the live overlay has only ever center-cropped,
     * whatever Blend Mode said. That is why center-crop was the only setting
     * that ever looked right in the overlay: blending was not on offer there.
     *
     * It is deleted rather than wired up, because wiring it as written would
     * not have worked: it needs neighbouring tiles to already be in the
     * cache, QuPath requests overlay tiles on demand in viewport order, and
     * the cache holds 100 entries against the ~1000 tiles of a whole-slide
     * overlay. Tiles would have blended against whatever happened to survive
     * eviction, so the result would have depended on pan and zoom history.
     *
     * Giving the overlay real blending is a design job, not a re-wiring job;
     * it is on TODO_LIST.md. Until then DLPixelClassifier enforces the
     * center-crop halo floor unconditionally, because center-crop is what it
     * does. Apply Classifier blends through TileProcessor and is unaffected.
     */

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
        long elapsed = System.currentTimeMillis() - lastRefreshTime;
        if (lastRefreshTime > 0 && elapsed < REFRESH_COOLDOWN_MS) {
            logger.debug(
                    "BLEND scheduleRefresh skipped ({}ms since last refresh, cooldown={}ms)",
                    elapsed,
                    REFRESH_COOLDOWN_MS);
            return;
        }

        ScheduledFuture<?> prev = pendingRefresh;
        if (prev != null) prev.cancel(false);
        logger.debug("BLEND scheduling refresh in 1s (cache size={})", probCache.size());
        pendingRefresh = refreshScheduler.schedule(
                () -> {
                    lastRefreshTime = System.currentTimeMillis();
                    try {
                        logger.debug("Refreshing overlay for tile blending ({} cached prob maps)", probCache.size());
                        refreshCallback.run();
                    } catch (Exception e) {
                        logger.debug("Deferred overlay refresh failed: {}", e.getMessage());
                    }
                },
                1000,
                TimeUnit.MILLISECONDS);
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
        lastRefreshTime = 0;
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
                logger.debug("Overlay empirical stepX = {} (from {} positions)", minGap, seenTileX.size());
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
                logger.debug("Overlay empirical stepY = {} (from {} positions)", minGap, seenTileY.size());
            }
        }
    }
}
