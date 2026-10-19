package qupath.ext.dlclassifier.ui.hud;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import qupath.ext.dlclassifier.controller.TrainingWorkflow;
import qupath.ext.dlclassifier.ui.hud.HudDelta.Verdict;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.ClassedRoi;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.ImageAnnotations;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.Role;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator.RoledImage;
import qupath.ext.dlclassifier.utilities.VramEstimator;
import qupath.lib.geom.Point2;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;

/**
 * The HUD's meaning, end to end: patch counts from the exporter's own sampler,
 * readings from the run's own arithmetic, verdicts per gauge.
 * <p>
 * Assertions are identities and directions, not snapshots of today's numbers,
 * so they keep holding when an estimator is recalibrated.
 */
class HudModelTest {

    private static final double STEP = 0.01;

    /** One 20k x 20k image: a dense tissue region and a long sparse stroke, two classes. */
    private static final List<RoledImage> IMAGES = List.of(new RoledImage(
            new ImageAnnotations(
                    "img",
                    20000,
                    20000,
                    List.of(
                            new ClassedRoi(
                                    "Tissue",
                                    ROIs.createRectangleROI(2000, 2000, 12000, 12000, ImagePlane.getDefaultPlane())),
                            new ClassedRoi(
                                    "Background",
                                    ROIs.createPolylineROI(
                                            List.of(
                                                    new Point2(500, 500),
                                                    new Point2(19000, 1500),
                                                    new Point2(19000, 19000)),
                                            ImagePlane.getDefaultPlane())))),
            Role.BOTH));

    private static HudInputs inputs(double downsample, int batch, int epochs, String arch, int tile) {
        int overlapPct = 20;
        int overlapPx = (int) Math.round(tile * overlapPct / 100.0);
        PatchCountEstimator.Counts c = PatchCountEstimator.count(
                IMAGES, Set.of("Tissue", "Background"), tile, overlapPx, downsample, 0.2, false);
        return new HudInputs(
                arch,
                "resnet18",
                tile,
                overlapPct,
                false,
                downsample,
                1,
                batch,
                1,
                epochs,
                true,
                20,
                true,
                "auto",
                3,
                8,
                "Tissue,Background|img:Both",
                true,
                c.train(),
                c.val(),
                24_576,
                32L << 30);
    }

    private static HudInputs inputs(double downsample, int batch, int epochs) {
        return inputs(downsample, batch, epochs, "unet", 256);
    }

    private static Map<Gauge, HudDelta.Change> change(HudInputs before, HudInputs after) {
        HudModel.Reading b = HudModel.read(before);
        return HudDelta.compare(b, HudModel.read(after), HudScale.centredOn(b), STEP);
    }

    /**
     * THE acceptance test (design doc section 3). Raising downsample makes an
     * epoch faster only by doing less. A HUD that shows only blue here is the
     * failure it was built to prevent.
     */
    @Test
    void downsampleEightIsFasterAndStarved() {
        Map<Gauge, HudDelta.Change> c = change(inputs(1, 16, 40), inputs(8, 16, 40));
        assertThat(c.get(Gauge.TIME_PER_EPOCH).verdict()).isEqualTo(Verdict.BETTER);
        assertThat(c.get(Gauge.LEARNING_STEPS).verdict()).isEqualTo(Verdict.WORSE);
        assertThat(c.get(Gauge.TRAINING_DATA).verdict()).isEqualTo(Verdict.WORSE);
        assertThat(c.get(Gauge.LEARNING_STEPS).ratio()).isLessThan(0.5);
    }

    @Test
    void patchCountsFallWithDownsampleAndRiseWithOverlap() {
        HudInputs ds1 = inputs(1, 16, 40);
        HudInputs ds4 = inputs(4, 16, 40);
        assertThat(ds4.trainPatches()).isLessThan(ds1.trainPatches());
        int tile = 256;
        PatchCountEstimator.Counts low = PatchCountEstimator.count(IMAGES, Set.of("Tissue"), tile, 0, 1, 0, false);
        PatchCountEstimator.Counts high = PatchCountEstimator.count(IMAGES, Set.of("Tissue"), tile, 128, 1, 0, false);
        assertThat(high.train()).isGreaterThan(low.train());
    }

    @Test
    void stepsMatchTheRunsOwnBudgetForEveryBatch() {
        for (int batch = 1; batch <= 64; batch++) {
            HudInputs in = inputs(4, batch, 40);
            var budget = TrainingWorkflow.computeStepBudget(in.trainPatches(), batch, 1, 40);
            GaugeReading steps = HudModel.read(in).get(Gauge.LEARNING_STEPS);
            assertThat(steps.value()).as("batch %d", batch).isEqualTo(budget.totalSteps());
            assertThat(steps.band() == GaugeReading.Band.LIMIT)
                    .as("batch %d", batch)
                    .isEqualTo(budget.isUntrainable());
        }
    }

    @Test
    void moreEpochsBuysStepsAndCostsTime() {
        Map<Gauge, HudDelta.Change> c = change(inputs(4, 16, 20), inputs(4, 16, 40));
        assertThat(c.get(Gauge.LEARNING_STEPS).verdict()).isEqualTo(Verdict.BETTER);
        assertThat(c.get(Gauge.LEARNING_STEPS).ratio()).isEqualTo(2.0);
        assertThat(c.get(Gauge.TOTAL_TIME).verdict()).isEqualTo(Verdict.WORSE);
        assertThat(c.get(Gauge.TOTAL_TIME).ratio()).isEqualTo(2.0);
        assertThat(c.get(Gauge.TIME_PER_EPOCH).verdict()).isEqualTo(Verdict.SAME);
    }

    @Test
    void largerBatchCostsMemoryAndSteps() {
        Map<Gauge, HudDelta.Change> c = change(inputs(4, 8, 40), inputs(4, 32, 40));
        assertThat(c.get(Gauge.GPU_MEMORY).verdict()).isEqualTo(Verdict.WORSE);
        assertThat(c.get(Gauge.LEARNING_STEPS).verdict()).isEqualTo(Verdict.WORSE);
    }

    @Test
    void everyPolarityIsExercisedInBothDirections() {
        Map<Gauge, HudDelta.Change> up = change(inputs(4, 16, 20), inputs(1, 16, 40));
        Map<Gauge, HudDelta.Change> down = change(inputs(1, 16, 40), inputs(4, 16, 20));
        for (Gauge g : List.of(Gauge.TIME_PER_EPOCH, Gauge.LEARNING_STEPS, Gauge.TRAINING_DATA, Gauge.TOTAL_TIME)) {
            assertThat(up.get(g).verdict()).as(g.name()).isIn(Verdict.BETTER, Verdict.WORSE);
            assertThat(down.get(g).verdict())
                    .as(g.name())
                    .isEqualTo(up.get(g).verdict() == Verdict.BETTER ? Verdict.WORSE : Verdict.BETTER);
        }
    }

    @Test
    void timeAcrossArchitecturesIsUnknownNotColoured() {
        Map<Gauge, HudDelta.Change> c = change(inputs(4, 16, 40, "unet", 256), inputs(4, 16, 40, "tiny-unet", 512));
        assertThat(c.get(Gauge.TIME_PER_EPOCH).verdict()).isEqualTo(Verdict.UNKNOWN);
        assertThat(c.get(Gauge.TOTAL_TIME).verdict()).isEqualTo(Verdict.UNKNOWN);
        // Data and steps do not depend on the architecture, so they stay judged.
        assertThat(c.get(Gauge.TRAINING_DATA).verdict()).isIn(Verdict.BETTER, Verdict.WORSE);
    }

    /**
     * The formula has no architecture term, so an architecture-only change
     * leaves its value untouched. Reading that as SAME would claim the time is
     * unchanged; the honest answer is "not comparable".
     */
    @Test
    void anArchitectureOnlyChangeIsUnknownNotSame() {
        Map<Gauge, HudDelta.Change> c = change(inputs(4, 16, 40, "unet", 256), inputs(4, 16, 40, "tiny-unet", 256));
        assertThat(c.get(Gauge.TIME_PER_EPOCH).verdict()).isEqualTo(Verdict.UNKNOWN);
        assertThat(c.get(Gauge.TOTAL_TIME).verdict()).isEqualTo(Verdict.UNKNOWN);
        assertThat(c.get(Gauge.TRAINING_DATA).verdict()).isEqualTo(Verdict.SAME);
    }

    @Test
    void gpuBandsAreExactlyTheEstimatorsOwn() {
        for (int batch = 1; batch <= 200; batch += 7) {
            HudInputs in = inputs(4, batch, 40);
            int padded = VramEstimator.paddedTileSizeFromPercent(256, 20, false);
            var est = VramEstimator.estimateMb("unet", "resnet18", padded, batch, true, 1);
            var fit = VramEstimator.classify(est.totalMb(), 24_576);
            GaugeReading g = HudModel.read(in).get(Gauge.GPU_MEMORY);
            GaugeReading.Band expected =
                    switch (fit) {
                        case OK -> GaugeReading.Band.NORMAL;
                        case TIGHT -> GaugeReading.Band.CAUTION;
                        case EXCEEDS -> GaugeReading.Band.LIMIT;
                    };
            assertThat(g.band()).as("batch %d", batch).isEqualTo(expected);
            assertThat(g.calibrated()).isEqualTo(est.calibrated());
        }
    }

    @Test
    void aBandCrossingIsShownHoweverSmallTheMove() {
        // Find the batch where the card goes from OK to TIGHT, then step across it by one.
        int crossing = -1;
        for (int b = 1; b < 400 && crossing < 0; b++) {
            if (HudModel.read(inputs(4, b, 40)).get(Gauge.GPU_MEMORY).band()
                    != HudModel.read(inputs(4, b + 1, 40)).get(Gauge.GPU_MEMORY).band()) {
                crossing = b;
            }
        }
        assertThat(crossing).isPositive();
        HudModel.Reading b = HudModel.read(inputs(4, crossing, 40));
        // A step far larger than any one-batch move: only the crossing can make it visible.
        Map<Gauge, HudDelta.Change> c =
                HudDelta.compare(b, HudModel.read(inputs(4, crossing + 1, 40)), HudScale.centredOn(b), 0.5);
        assertThat(c.get(Gauge.GPU_MEMORY).bandCrossed()).isTrue();
        assertThat(c.get(Gauge.GPU_MEMORY).verdict()).isEqualTo(Verdict.WORSE);
    }

    @Test
    void aMoveBelowDisplayResolutionIsNotColoured() {
        HudModel.Reading b = HudModel.read(inputs(4, 16, 40));
        HudModel.Reading n = HudModel.read(inputs(4, 16, 41));
        Map<Gauge, HudDelta.Change> c = HudDelta.compare(b, n, HudScale.centredOn(b), 0.5);
        assertThat(c.get(Gauge.LEARNING_STEPS).verdict()).isEqualTo(Verdict.SAME);
    }

    @Test
    void beforeClassesLoadOnlyGpuMemoryHasData() {
        HudInputs in = new HudInputs(
                "unet",
                "resnet18",
                256,
                20,
                false,
                4,
                1,
                16,
                1,
                40,
                true,
                20,
                true,
                "auto",
                3,
                8,
                "",
                false,
                0,
                0,
                24_576,
                32L << 30);
        HudModel.Reading r = HudModel.read(in);
        for (Gauge g : Gauge.values()) {
            assertThat(r.get(g).available()).as(g.name()).isEqualTo(g == Gauge.GPU_MEMORY);
        }
    }

    @Test
    void autoCacheCautionStartsWhereTheTrainingScriptDeclines() {
        HudInputs base = inputs(1, 16, 40);
        long bytes = PatchCountEstimator.cacheBytes(base.trainPatches() + base.valPatches(), 256, 3, 8, false);
        long atCeiling = (long) Math.floor(bytes / PatchCountEstimator.AUTO_CACHE_FRACTION);
        HudInputs declines = withFreeRam(base, atCeiling);
        HudInputs accepts = withFreeRam(base, atCeiling * 2);
        assertThat(HudModel.read(declines).get(Gauge.RAM_CACHE).band()).isEqualTo(GaugeReading.Band.CAUTION);
        assertThat(HudModel.read(accepts).get(Gauge.RAM_CACHE).band()).isEqualTo(GaugeReading.Band.NORMAL);
    }

    private static HudInputs withFreeRam(HudInputs i, long free) {
        return new HudInputs(
                i.architecture(),
                i.backbone(),
                i.tileSize(),
                i.overlapPercent(),
                i.wholeImage(),
                i.downsample(),
                i.contextScale(),
                i.batchSize(),
                i.accumulation(),
                i.epochs(),
                i.mixedPrecision(),
                i.validationSplitPercent(),
                i.earlyStopping(),
                i.inMemoryMode(),
                i.channels(),
                i.bitDepth(),
                i.dataSelection(),
                i.patchesKnown(),
                i.trainPatches(),
                i.valPatches(),
                i.gpuTotalMb(),
                free);
    }
}
