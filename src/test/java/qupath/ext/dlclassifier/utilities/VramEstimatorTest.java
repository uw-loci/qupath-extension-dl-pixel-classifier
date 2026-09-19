package qupath.ext.dlclassifier.utilities;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

/**
 * Pins the VRAM estimate against real measurements.
 * <p>
 * This is the guard against silent drift. A formula cannot be guaranteed
 * correct across future changes: add a normalization layer, widen the decoder,
 * or turn on another augmentation and the constant moves. What CAN be
 * guaranteed is that such a change fails here loudly instead of quietly telling
 * a user that 130% of their VRAM is "73%, green" -- which is what the estimate
 * did before these numbers were measured.
 * <p>
 * <b>When this test fails after an intentional change:</b> re-measure, do not
 * retune to taste. Every training run logs predicted versus actual peak, so the
 * next run on a real GPU gives the new number; put it in the table below with
 * its provenance and update {@link VramEstimator#TINY_UNET_ACTIVATION_FACTOR}.
 *
 * <h2>Provenance of the measurements</h2>
 * RTX 3090 (24,575 MiB), tiny-unet small-24x4, base 24, tile 256 at 20% overlap
 * so the exporter writes 358 px tiles, BF16, context scale 1, 2026-09-19:
 * <ul>
 *   <li>batch 40: 21,437 MiB steady state (nvidia-smi, process total)</li>
 *   <li>batch 60: ~32,057 MiB demanded (24,274 MiB resident + ~7.6 GB spilled
 *       to shared memory, which is how Windows fails instead of raising OOM)</li>
 * </ul>
 */
class VramEstimatorTest {

    private static final int GPU_TOTAL_MB = 24575;
    private static final String MODEL = "tiny-unet";
    private static final String BACKBONE = "small-24x4";
    private static final int TILE = 256;
    private static final int OVERLAP_PCT = 20;

    @Test
    void theExporterPadsTheTileAndTheEstimatorMustSeeThat() {
        // 256 + 2*51. Estimating on 256 understated the area by 1.96x, which is
        // the single biggest reason the old estimate was low.
        assertThat(VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, false))
                .isEqualTo(358);
        // Whole-image mode has no surrounding data, so no padding.
        assertThat(VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, true))
                .isEqualTo(256);
        // Pixels and percent must agree; mixing the units is the bug class here.
        assertThat(VramEstimator.paddedTileSize(TILE, 51, false)).isEqualTo(358);
    }

    @Test
    void batch40MatchesTheMeasuredPeak() {
        int padded = VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, false);

        double predicted =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 40, true, 1).totalMb();

        // Measured 21,437 MiB. Allow 10%: the estimate is a planning aid, not a
        // simulator, and the allocator's high-water mark moves a little run to run.
        assertThat(predicted).isCloseTo(21437.0, within(0.10 * 21437.0));
    }

    @Test
    void batch60MatchesTheMeasuredDemandThatOverflowedTheCard() {
        int padded = VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, false);

        double predicted =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 60, true, 1).totalMb();

        assertThat(predicted).isCloseTo(32057.0, within(0.10 * 32057.0));
        // And it must read as not fitting: this configuration is what spilled.
        assertThat(predicted).isGreaterThan(GPU_TOTAL_MB);
    }

    @Test
    void theOldFormulaWouldStillFailThisTest() {
        // Regression anchor. The previous estimate was
        //   batch * base * 256^2 * 2 bytes * 100  (unpadded tile, factor 100)
        // which produced 18,018 MB for batch 60 -- 56% of the truth, and shown
        // in green. If a future edit reintroduces either mistake, the two tests
        // above fail; this documents the number they are protecting against.
        double oldEstimateBatch60 = (60.0 * 24 * 256 * 256 * 2 * 100) / (1024.0 * 1024.0) + 5 * 3.5;

        assertThat(oldEstimateBatch60).isCloseTo(18018.0, within(50.0));
        assertThat(oldEstimateBatch60).isLessThan(GPU_TOTAL_MB); // i.e. it claimed the run would fit
    }

    @Test
    void scalingIsLinearInBatchAndQuadraticInTileSize() {
        int padded = VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, false);
        double at20 =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 20, true, 1).totalMb();
        double at40 =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 40, true, 1).totalMb();
        double atDoubleTile = VramEstimator.estimateMb(MODEL, BACKBONE, 2 * padded, 20, true, 1)
                .totalMb();

        assertThat(at40 / at20).isCloseTo(2.0, within(0.01));
        assertThat(atDoubleTile / at20).isCloseTo(4.0, within(0.01));
    }

    @Test
    void mixedPrecisionHalvesTheActivationCost() {
        int padded = VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, false);
        double amp =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 40, true, 1).totalMb();
        double fp32 =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 40, false, 1).totalMb();

        assertThat(fp32 / amp).isCloseTo(2.0, within(0.01));
    }

    @Test
    void pretrainedEncodersReportThemselvesAsUncalibrated() {
        int padded = VramEstimator.paddedTileSizeFromPercent(512, 10, false);

        // No measurement exists for these yet, and the UI says so rather than
        // implying a precision we do not have.
        assertThat(VramEstimator.estimateMb("unet", "resnet50", padded, 8, true, 1)
                        .calibrated())
                .isFalse();
        assertThat(VramEstimator.estimateMb(MODEL, BACKBONE, padded, 8, true, 1).calibrated())
                .isTrue();
    }

    @Test
    void theMeasuredGoodConfigurationIsNotReportedAsTooBig() {
        int padded = VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, false);
        double at40 =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 40, true, 1).totalMb();
        double at60 =
                VramEstimator.estimateMb(MODEL, BACKBONE, padded, 60, true, 1).totalMb();

        // Batch 40 was measured at 87% of the card and trained fine, so it must
        // read as tight, never as "exceeds". Batch 60 is the one that spilled.
        assertThat(VramEstimator.classify(at40, GPU_TOTAL_MB)).isEqualTo(VramEstimator.Fit.TIGHT);
        assertThat(VramEstimator.classify(at60, GPU_TOTAL_MB)).isEqualTo(VramEstimator.Fit.EXCEEDS);
        assertThat(VramEstimator.classify(at40 / 4, GPU_TOTAL_MB)).isEqualTo(VramEstimator.Fit.OK);
    }

    @Test
    void maxBatchWithinFindsTheLargestFittingBatch() {
        int padded = VramEstimator.paddedTileSizeFromPercent(TILE, OVERLAP_PCT, false);
        double budget = GPU_TOTAL_MB * 0.85;

        int best = VramEstimator.maxBatchWithin(budget, MODEL, BACKBONE, padded, true, 1, 60);

        assertThat(best).isBetween(1, 60);
        assertThat(VramEstimator.estimateMb(MODEL, BACKBONE, padded, best, true, 1)
                        .totalMb())
                .isLessThanOrEqualTo(budget);
        assertThat(VramEstimator.estimateMb(MODEL, BACKBONE, padded, best + 1, true, 1)
                        .totalMb())
                .isGreaterThan(budget);
    }
}
