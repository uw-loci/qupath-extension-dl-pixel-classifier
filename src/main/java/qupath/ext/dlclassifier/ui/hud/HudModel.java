package qupath.ext.dlclassifier.ui.hud;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import qupath.ext.dlclassifier.controller.TrainingWorkflow;
import qupath.ext.dlclassifier.utilities.PatchCountEstimator;
import qupath.ext.dlclassifier.utilities.VramEstimator;

/**
 * Turns a snapshot of the Train dialog into gauge readings. Pure: no JavaFX.
 * <p>
 * Every number comes from the arithmetic the run itself uses --
 * {@link VramEstimator}, {@link TrainingWorkflow#computeStepBudget},
 * {@link PatchCountEstimator} -- so the HUD and the dialog's own labels cannot
 * disagree.
 */
public final class HudModel {

    /**
     * Validation is a forward pass only; training is forward plus backward,
     * which costs about two forwards. So a validation patch costs about a third
     * of a training patch. A standard rule of thumb, not a measurement: it only
     * decides how much a change of validation split moves the time gauges.
     */
    static final double VALIDATION_COST = 1.0 / 3.0;

    private HudModel() {}

    /** All readings for one snapshot. */
    public record Reading(HudInputs inputs, Map<Gauge, GaugeReading> gauges) {
        public GaugeReading get(Gauge g) {
            return gauges.get(g);
        }
    }

    /**
     * Reads every gauge.
     *
     * @param in the dialog snapshot
     * @return the readings
     */
    public static Reading read(HudInputs in) {
        Map<Gauge, GaugeReading> m = new EnumMap<>(Gauge.class);
        int paddedTile = VramEstimator.paddedTileSizeFromPercent(in.tileSize(), in.overlapPercent(), in.wholeImage());

        m.put(Gauge.GPU_MEMORY, gpu(in, paddedTile));
        m.put(Gauge.RAM_CACHE, ram(in));

        if (!in.patchesKnown()) {
            String why = "Load classes to see data and time";
            for (Gauge g : new Gauge[] {
                Gauge.EPOCH_SPEED, Gauge.LEARNING_STEPS, Gauge.TRAINING_DATA, Gauge.VALIDATION, Gauge.TOTAL_TIME
            }) {
                m.put(g, GaugeReading.unavailable(g, why));
            }
            return new Reading(in, Collections.unmodifiableMap(m));
        }

        double contextFactor = in.contextScale() > 1 ? 2.0 : 1.0;
        double perPatch = (double) paddedTile * paddedTile * contextFactor;
        double epochWork = (in.trainPatches() + VALIDATION_COST * in.valPatches()) * perPatch;
        if (epochWork > 0) {
            m.put(Gauge.EPOCH_SPEED, relative(Gauge.EPOCH_SPEED, 1.0 / epochWork, GaugeReading.Band.NORMAL, ""));
        } else {
            m.put(Gauge.EPOCH_SPEED, GaugeReading.unavailable(Gauge.EPOCH_SPEED, "no patches"));
        }

        TrainingWorkflow.StepBudget budget =
                TrainingWorkflow.computeStepBudget(in.trainPatches(), in.batchSize(), in.accumulation(), in.epochs());
        m.put(
                Gauge.LEARNING_STEPS,
                relative(
                        Gauge.LEARNING_STEPS,
                        budget.totalSteps(),
                        budget.isUntrainable() ? GaugeReading.Band.LIMIT : GaugeReading.Band.NORMAL,
                        budget.isUntrainable() ? budget.stepsPerEpoch() + " step per epoch" : ""));

        m.put(
                Gauge.TRAINING_DATA,
                relative(
                        Gauge.TRAINING_DATA,
                        in.trainPatches(),
                        in.trainPatches() == 0 ? GaugeReading.Band.LIMIT : GaugeReading.Band.NORMAL,
                        in.trainPatches() == 0 ? "no training patches" : ""));

        GaugeReading.Band valBand = in.valPatches() <= 1 ? GaugeReading.Band.LIMIT : GaugeReading.Band.NORMAL;
        String valNote = in.valPatches() == 0
                ? "no validation patches"
                : in.valPatches() == 1 ? "one patch: every score step is 0 or 100%" : "";
        m.put(Gauge.VALIDATION, relative(Gauge.VALIDATION, in.valPatches(), valBand, valNote));

        m.put(
                Gauge.TOTAL_TIME,
                relative(
                        Gauge.TOTAL_TIME,
                        epochWork * Math.max(0, in.epochs()),
                        GaugeReading.Band.NORMAL,
                        in.earlyStopping() ? "upper bound: early stopping may end sooner" : ""));

        return new Reading(in, Collections.unmodifiableMap(m));
    }

    private static GaugeReading gpu(HudInputs in, int paddedTile) {
        if (in.gpuTotalMb() <= 0) {
            return GaugeReading.unavailable(Gauge.GPU_MEMORY, "No GPU detected");
        }
        VramEstimator.Estimate e = VramEstimator.estimateMb(
                in.architecture(), in.backbone(), paddedTile, in.batchSize(), in.mixedPrecision(), in.contextScale());
        VramEstimator.Fit fit = VramEstimator.classify(e.totalMb(), in.gpuTotalMb());
        GaugeReading.Band band =
                switch (fit) {
                    case OK -> GaugeReading.Band.NORMAL;
                    case TIGHT -> GaugeReading.Band.CAUTION;
                    case EXCEEDS -> GaugeReading.Band.LIMIT;
                };
        String note = e.calibrated() ? "" : "rough: not yet calibrated for this architecture";
        return new GaugeReading(Gauge.GPU_MEMORY, true, e.totalMb() / in.gpuTotalMb(), e.calibrated(), band, note);
    }

    private static GaugeReading ram(HudInputs in) {
        if ("off".equals(in.inMemoryMode())) {
            return GaugeReading.unavailable(Gauge.RAM_CACHE, "In-memory cache is off");
        }
        if (!in.patchesKnown()) {
            return GaugeReading.unavailable(Gauge.RAM_CACHE, "Load classes to see data and time");
        }
        if (in.freeRamBytes() <= 0) {
            return GaugeReading.unavailable(Gauge.RAM_CACHE, "Free RAM unknown on this system");
        }
        long bytes = PatchCountEstimator.cacheBytes(
                in.trainPatches() + in.valPatches(),
                in.tileSize(),
                in.channels(),
                in.bitDepth(),
                in.contextScale() > 1);
        double frac = (double) bytes / in.freeRamBytes();
        GaugeReading.Band band;
        String note = "";
        if (frac > 1.0) {
            band = GaugeReading.Band.LIMIT;
            note = "more than free RAM";
        } else if ("auto".equals(in.inMemoryMode()) && frac >= PatchCountEstimator.AUTO_CACHE_FRACTION) {
            band = GaugeReading.Band.CAUTION;
            note = "auto will decline the cache";
        } else if ("bounded".equals(in.inMemoryMode()) && frac > PatchCountEstimator.BOUNDED_CACHE_FRACTION) {
            band = GaugeReading.Band.CAUTION;
            note = "bounded will cache a subset";
        } else {
            band = GaugeReading.Band.NORMAL;
        }
        return new GaugeReading(Gauge.RAM_CACHE, true, frac, true, band, note);
    }

    private static GaugeReading relative(Gauge g, double value, GaugeReading.Band band, String note) {
        return new GaugeReading(g, true, value, true, band, note);
    }
}
