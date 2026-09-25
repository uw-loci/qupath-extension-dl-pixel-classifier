package qupath.ext.dlclassifier.controller;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Optimizer-step arithmetic behind the pre-flight guard.
 *
 * <p>A batch size at or above the training-patch count makes the DataLoader
 * yield one batch, so an epoch buys one optimizer step and the epoch count
 * stops being a training budget. Runs in that state look healthy for their
 * whole length: training loss falls, the progress bar advances, and accuracy
 * settles at the majority class's pixel share.
 *
 * <p>Six of fourteen nano runs analyzed on 2026-09-25 failed to learn, and
 * three of those were this: batch 80 against 64 training patches, 150 epochs,
 * 150 gradient updates. See
 * {@code claude-reports/2026-09-25_nano-tissue-training-failures.md}.
 *
 * <p>These tests pin the arithmetic only. The check deliberately does not
 * encode anything about learning rate, downsample or weight decay, which
 * separated those runs just as cleanly but are properties of one dataset
 * rather than of the training loop.
 */
class StepBudgetTest {

    @Test
    @DisplayName("batch larger than the training set yields one step per epoch")
    void oversizedBatchGivesOneStep() {
        // The observed failure: 64 train patches, batch 80, 150 epochs.
        var budget = TrainingWorkflow.computeStepBudget(64, 80, 1, 150);

        assertEquals(1, budget.batchesPerEpoch(), "the loader keeps its short final batch");
        assertEquals(1, budget.stepsPerEpoch());
        assertEquals(150, budget.totalSteps(), "150 epochs bought 150 gradient updates");
        assertTrue(budget.isUntrainable());
    }

    @Test
    @DisplayName("batch exactly equal to the training set is still one step")
    void batchEqualToDatasetGivesOneStep() {
        // The boundary: a strict > comparison would let this through.
        var budget = TrainingWorkflow.computeStepBudget(64, 64, 1, 100);

        assertEquals(1, budget.stepsPerEpoch());
        assertTrue(budget.isUntrainable());
    }

    @Test
    @DisplayName("the configuration that worked is not flagged")
    void workingConfigurationPasses() {
        // tiny4: downsample 2, batch 20. Best run on disk, mIoU 0.981.
        var budget = TrainingWorkflow.computeStepBudget(64, 20, 1, 150);

        assertEquals(4, budget.batchesPerEpoch(), "ceil(64/20), short final batch kept");
        assertEquals(4, budget.stepsPerEpoch());
        assertEquals(600, budget.totalSteps());
        assertFalse(budget.isUntrainable());
    }

    @Test
    @DisplayName("gradient accumulation divides the step count")
    void accumulationReducesSteps() {
        // 8 batches accumulated 4 at a time: 2 optimizer steps per epoch.
        var budget = TrainingWorkflow.computeStepBudget(64, 8, 4, 50);

        assertEquals(8, budget.batchesPerEpoch());
        assertEquals(2, budget.stepsPerEpoch());
        assertFalse(budget.isUntrainable(), "two steps is thin, but not the broken case");
    }

    @Test
    @DisplayName("accumulation can push an otherwise fine batch size under the bar")
    void accumulationCanMakeItUntrainable() {
        // Batch 20 alone gives 4 steps; accumulating 4 collapses it to 1.
        // Accumulation is the quieter half of this: the batch size looks
        // reasonable on its own.
        var budget = TrainingWorkflow.computeStepBudget(64, 20, 4, 150);

        assertEquals(4, budget.batchesPerEpoch());
        assertEquals(1, budget.stepsPerEpoch());
        assertTrue(budget.isUntrainable());
    }

    @Test
    @DisplayName("the accumulation tail flushes, so both divisions round up")
    void accumulationTailIsNotDropped() {
        // 5 batches accumulated 2 at a time. training_service steps on the
        // last batch of the epoch as well as every accumulation-th batch,
        // so the odd batch still produces a step: ceil(5/2) = 3, not 2.
        var budget = TrainingWorkflow.computeStepBudget(50, 10, 2, 10);

        assertEquals(5, budget.batchesPerEpoch());
        assertEquals(3, budget.stepsPerEpoch());
    }

    @Test
    @DisplayName("degenerate inputs do not throw or divide by zero")
    void degenerateInputsAreSafe() {
        var noPatches = TrainingWorkflow.computeStepBudget(0, 16, 1, 50);
        assertEquals(0, noPatches.batchesPerEpoch());
        assertFalse(noPatches.isUntrainable(), "nothing exported is a different problem, reported elsewhere");

        // A zero or negative batch size should not reach here, but the guard
        // runs on a live config and must not be the thing that crashes.
        var zeroBatch = TrainingWorkflow.computeStepBudget(64, 0, 0, 10);
        assertEquals(64, zeroBatch.batchesPerEpoch(), "batch and accumulation both floor at 1");
        assertEquals(64, zeroBatch.stepsPerEpoch());
        assertFalse(zeroBatch.isUntrainable());
    }

    @Test
    @DisplayName("a large dataset is unaffected by the guard")
    void largeDatasetIsFine() {
        var budget = TrainingWorkflow.computeStepBudget(4000, 16, 1, 50);

        assertEquals(250, budget.batchesPerEpoch());
        assertEquals(12500, budget.totalSteps());
        assertFalse(budget.isUntrainable());
    }
}
