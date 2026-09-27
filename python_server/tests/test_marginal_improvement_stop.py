"""The marginal-improvement exit: stop a run that has arrived.

EarlyStopping asks "has the model stopped setting records?". A converged run
keeps setting them, by fourth-decimal amounts, and so never stops. The real
100-epoch run in ``REAL_CONVERGED_MIOU`` below reached 0.9876 mean IoU at
epoch 43 and 0.9913 at epoch 93: fifty more epochs for 0.0037.

This exit asks whether the BEST score improved by at least ``min_improvement``
over the last ``window`` epochs. It is opt-in, because whether 0.0037 is worth
fifty epochs depends entirely on what an epoch costs.
"""

import pytest

from dlclassifier_server.services.training_service import MarginalImprovementStop


def stop_epoch(values, window=20, min_improvement=0.01, mode="max"):
    """Return the epoch the exit fires on, or None if it never does."""
    ms = MarginalImprovementStop(
        window=window, min_improvement=min_improvement, mode=mode
    )
    for i, v in enumerate(values, start=1):
        if ms(i, v):
            return i
    return None


# The real run, epochs 1-100. Best 0.9913 at epoch 93.
REAL_CONVERGED_MIOU = [
    0.1859,
    0.1976,
    0.6161,
    0.4610,
    0.4434,
    0.5871,
    0.7505,
    0.7507,
    0.7433,
    0.6005,
    0.7621,
    0.8380,
    0.9522,
    0.9567,
    0.9583,
    0.9425,
    0.9736,
    0.9690,
    0.9527,
    0.9706,
    0.9762,
    0.9478,
    0.9661,
    0.9346,
    0.9707,
    0.9818,
    0.9761,
    0.9785,
    0.9664,
    0.7700,
    0.9192,
    0.9642,
    0.7563,
    0.8513,
    0.9461,
    0.9746,
    0.9171,
    0.8842,
    0.9361,
    0.9427,
    0.8735,
    0.9394,
    0.9876,
    0.9844,
    0.9844,
    0.9811,
    0.9825,
    0.9825,
    0.9842,
    0.9729,
    0.9666,
    0.9813,
    0.9847,
    0.9870,
    0.9469,
    0.9610,
    0.9665,
    0.9721,
    0.9812,
    0.9887,
    0.9703,
    0.9804,
    0.9780,
    0.9861,
    0.9879,
    0.9892,
    0.9893,
    0.9862,
    0.9879,
    0.9871,
    0.9889,
    0.9867,
    0.9874,
    0.9868,
    0.9881,
    0.9888,
    0.9888,
    0.9902,
    0.9904,
    0.9906,
    0.9903,
    0.9909,
    0.9862,
    0.9896,
    0.9907,
    0.9908,
    0.9906,
    0.9897,
    0.9873,
    0.9631,
    0.9822,
    0.9860,
    0.9913,
    0.9836,
    0.9801,
    0.9905,
    0.9903,
    0.9725,
    0.9671,
    0.9470,
]


class TestStopsAConvergedRun:
    def test_the_defaults_stop_the_real_run_well_before_100(self):
        # 63 fewer epochs; the cost is stated in the next test.
        assert stop_epoch(REAL_CONVERGED_MIOU) == 37

    def test_and_the_cost_of_stopping_there_is_small(self):
        # Guards the premise. If this ever changes, the default threshold is
        # buying something different from what it was chosen to buy.
        assert max(REAL_CONVERGED_MIOU[:37]) == pytest.approx(0.9818)
        assert max(REAL_CONVERGED_MIOU) == pytest.approx(0.9913)

    def test_a_tighter_threshold_runs_longer_and_keeps_more(self):
        stopped = stop_epoch(REAL_CONVERGED_MIOU, min_improvement=0.005)
        assert stopped == 63
        assert max(REAL_CONVERGED_MIOU[:stopped]) == pytest.approx(0.9887)

    def test_a_longer_window_runs_longer(self):
        assert stop_epoch(REAL_CONVERGED_MIOU, window=30) > stop_epoch(
            REAL_CONVERGED_MIOU, window=20
        )

    def test_a_perfectly_flat_metric_stops_one_window_in(self):
        assert stop_epoch([0.5] * 60) == 21


class TestLeavesImprovingRunsAlone:
    def test_a_run_still_climbing_fast_is_not_stopped(self):
        # +0.01 every epoch: 0.2 over any 20-epoch window.
        assert stop_epoch([0.1 + 0.01 * i for i in range(60)]) is None

    def test_no_decision_before_a_full_window_has_elapsed(self):
        # Twenty flat epochs are not yet evidence of anything.
        assert stop_epoch([0.5] * 20) is None

    def test_improvement_is_measured_on_the_best_not_the_last_epoch(self):
        # A run that spikes to 0.9 at epoch 25 and then sits at 0.1 has
        # improved; the noisy tail must not read as a collapse.
        values = [0.1] * 24 + [0.9] + [0.1] * 20
        assert stop_epoch(values, window=20, min_improvement=0.01) == 21


class TestLossMode:
    def test_a_falling_loss_counts_as_improvement(self):
        losses = [1.0 - 0.01 * i for i in range(60)]
        assert stop_epoch(losses, mode="min") is None

    def test_a_flat_loss_stops(self):
        assert stop_epoch([0.3] * 60, mode="min") == 21

    def test_a_rising_loss_stops(self):
        # Getting worse is not improvement, and must not be read as such by
        # a sign error.
        assert stop_epoch([0.1 + 0.01 * i for i in range(60)], mode="min") == 21


class TestGuards:
    def test_a_missing_metric_is_ignored_rather_than_crashing(self):
        ms = MarginalImprovementStop(window=2, min_improvement=0.01)
        assert ms(1, None) is False

    def test_window_and_threshold_are_clamped_to_sane_values(self):
        ms = MarginalImprovementStop(window=0, min_improvement=-5.0)
        assert ms.window == 1
        assert ms.min_improvement == 0.0
