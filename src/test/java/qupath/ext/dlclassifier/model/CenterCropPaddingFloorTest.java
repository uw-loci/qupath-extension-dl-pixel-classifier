package qupath.ext.dlclassifier.model;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * The one place the extension overrides the user's tiling geometry.
 *
 * <p>CENTER_CROP gives each pixel to a single tile instead of averaging the
 * tiles that cover it. A U-Net with a ResNet encoder has a receptive field
 * wider than a 256px tile, so two tiles sharing a piece of tissue can predict
 * differently for it and center-crop turns that into a rectangular block on
 * the tile grid. Shifting the tile grid half a stride and counting the pixels
 * whose class changed, on a 6-channel model at tile 256: 10.65% at 20%
 * overlap, 7.08% at 25%, 3.42% at 37.5%, 3.39% at 43.8%. So the halo fixes
 * it, and 37.5% is where it stops paying for itself.
 *
 * <p>The floor therefore applies to CENTER_CROP and to nothing else, and the
 * raise is reported so a caller can explain why a run got slower.
 */
class CenterCropPaddingFloorTest {

    private static InferenceConfig config(int tileSize, int overlap, InferenceConfig.BlendMode mode) {
        return InferenceConfig.builder()
                .tileSize(tileSize)
                .overlap(overlap)
                .blendMode(mode)
                .build();
    }

    @Test
    void centerCropIsRaisedToTheFloorAndSaysSo() {
        // 20% of 256 is 51px, which is what the reported overlay ran at.
        InferenceConfig c = config(256, 51, InferenceConfig.BlendMode.CENTER_CROP);
        assertThat(c.effectivePadding()).isEqualTo(96); // 37.5% of 256
        assertThat(c.centerCropPaddingWasRaised()).isTrue();
    }

    @Test
    void anOverlapAlreadyAboveTheFloorIsLeftAlone() {
        InferenceConfig c = config(256, 112, InferenceConfig.BlendMode.CENTER_CROP);
        assertThat(c.effectivePadding()).isEqualTo(112);
        assertThat(c.centerCropPaddingWasRaised()).isFalse();
    }

    @Test
    void everyOtherBlendModeKeepsTheUsersOverlapExactly() {
        for (InferenceConfig.BlendMode mode : InferenceConfig.BlendMode.values()) {
            if (mode == InferenceConfig.BlendMode.CENTER_CROP) {
                continue;
            }
            InferenceConfig c = config(256, 51, mode);
            assertThat(c.effectivePadding()).as("%s must not be clamped", mode).isEqualTo(51);
            assertThat(c.centerCropPaddingWasRaised()).isFalse();
        }
    }

    @Test
    void theFloorScalesWithTileSizeRatherThanBeingAFixedPixelCount() {
        assertThat(config(512, 64, InferenceConfig.BlendMode.CENTER_CROP).effectivePadding())
                .isEqualTo(192);
        assertThat(config(64, 8, InferenceConfig.BlendMode.CENTER_CROP).effectivePadding())
                .isEqualTo(24);
    }

    @Test
    void strideStaysPositiveSoTheTilingLoopAdvances() {
        // The floor must never outrank the invariant that makes tiling
        // terminate. 37.5% leaves a quarter of the tile as stride, but the
        // clamp is what guarantees it rather than the arithmetic.
        // 64 is the smallest tile the builder accepts; 8192 the largest.
        for (int tileSize : new int[] {64, 128, 256, 511, 512, 1024, 8192}) {
            InferenceConfig c = config(tileSize, 0, InferenceConfig.BlendMode.CENTER_CROP);
            int stride = tileSize - 2 * c.effectivePadding();
            assertThat(stride).as("tileSize=%d", tileSize).isGreaterThan(0);
        }
    }

    @Test
    void aZeroTileSizeIsNotAnOpportunityToDivideByIt() {
        // The builder rejects a tile size outside 64..8192, so this can only
        // be reached through the static helper -- which several callers use
        // directly, including the VRAM estimate.
        assertThat(InferenceConfig.computeEffectivePadding(0, 51)).isZero();
        assertThat(InferenceConfig.computeEffectivePadding(-256, 51)).isZero();
    }
}
