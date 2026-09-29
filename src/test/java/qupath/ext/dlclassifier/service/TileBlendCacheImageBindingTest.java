package qupath.ext.dlclassifier.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * The overlay's tile cache is keyed by tile coordinate alone, which identifies
 * a tile within one image and says nothing about which image it came from.
 *
 * <p>QuPath keeps a pixel-classification overlay on the viewer across an image
 * change, so one classifier -- and this cache with it -- can outlive the image
 * it was built for. Every tile whose coordinates collided was then served from
 * the wrong image, appearing as tile-shaped rectangles of the opposite class in
 * the middle of solid tissue, and differing on each visit because it depended on
 * what was still in the LRU.
 *
 * <p>Measured on a four-block case (2026-09-29): all four blocks matched the
 * previously-viewed image's prediction at those coordinates, and none matched
 * the displayed image's own.
 */
class TileBlendCacheImageBindingTest {

    private static final String A = "/slides/Tile 1.ome.tif";
    private static final String B = "/slides/Tile 2.ome.tif";

    private TileBlendCache cache;

    private TileBlendCache cache() {
        cache = new TileBlendCache(100, () -> {});
        return cache;
    }

    @AfterEach
    void tearDown() {
        if (cache != null) cache.shutdown();
    }

    private static float[][][] map(float v) {
        return new float[][][] {{{v}}};
    }

    @Test
    void aTileCachedForOneImageIsNeverServedForAnother() {
        TileBlendCache c = cache();
        c.cache(A, 4096, 2048, map(1f));
        assertThat(c.getIfCached(A, 4096, 2048)).isNotNull();
        assertThat(c.getIfCached(B, 4096, 2048)).isNull();
    }

    @Test
    void switchingImagesDropsWhatTheCacheHeld() {
        TileBlendCache c = cache();
        c.cache(A, 0, 0, map(1f));
        c.cache(A, 154, 0, map(1f));
        assertThat(c.size()).isEqualTo(2);
        c.getIfCached(B, 999, 999);
        assertThat(c.size()).isZero();
    }

    @Test
    void anInFlightTileFromThePreviousImageIsDroppedNotStored() {
        // The race the coordinate-only key allowed: the read path has already
        // bound the cache to the new image when a tile that was still being
        // computed for the old one arrives. Storing it would hand it straight
        // back on the next repaint of a colliding coordinate.
        TileBlendCache c = cache();
        c.getIfCached(A, 0, 0);
        c.getIfCached(B, 0, 0); // viewer moved to B; cache now bound to B
        c.cache(A, 512, 512, map(1f)); // straggler from A lands late
        assertThat(c.size()).isZero();
        assertThat(c.getIfCached(B, 512, 512)).isNull();
    }

    @Test
    void aStragglerDoesNotEvictTheNewImagesTiles() {
        // Rebinding on write instead of dropping would clear B's tiles here.
        TileBlendCache c = cache();
        c.getIfCached(B, 0, 0);
        c.cache(B, 0, 0, map(2f));
        c.cache(A, 0, 0, map(1f)); // straggler for the same coordinate
        assertThat(c.size()).isEqualTo(1);
        assertThat(c.getIfCached(B, 0, 0)).isNotNull();
        assertThat(c.getIfCached(B, 0, 0)[0][0][0]).isEqualTo(2f);
    }

    @Test
    void goingBackToTheFirstImageDoesNotResurrectItsOldTiles() {
        TileBlendCache c = cache();
        c.cache(A, 0, 0, map(1f));
        c.getIfCached(B, 0, 0);
        assertThat(c.getIfCached(A, 0, 0)).isNull();
    }

    @Test
    void theFirstImageSeenBindsTheCache() {
        // A write before any read still has to bind, or nothing would ever cache.
        TileBlendCache c = cache();
        c.cache(A, 10, 20, map(1f));
        assertThat(c.getIfCached(A, 10, 20)).isNotNull();
    }

    @Test
    void repeatedUseOfOneImageKeepsCaching() {
        // Guard against a fix that clears too eagerly and disables the cache.
        TileBlendCache c = cache();
        for (int i = 0; i < 10; i++) {
            c.getIfCached(A, i * 154, 0);
            c.cache(A, i * 154, 0, map(i));
        }
        assertThat(c.size()).isEqualTo(10);
        assertThat(c.getIfCached(A, 5 * 154, 0)[0][0][0]).isEqualTo(5f);
    }
}
