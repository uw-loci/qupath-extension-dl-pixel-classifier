"""Early stopping must not kill a run that is still converging.

Two real runs drove this, both recorded in
``claude-reports/data/2026-09-26_early-stopping-fixtures.json``:

* A validation metric measured on 6 patches swung 0.53 mean IoU between
  consecutive epochs. The highest spike became a bar the model had to clear
  within the patience window; it could not, and the run was stopped at epoch
  25 holding 0.886. Resumed from the same checkpoint it reached 0.976 by
  epoch 35.
* Mean IoU is computed from argmax, so it cannot move until predictions cross
  a decision boundary. A second run sat at exactly 0.1861 for eleven epochs
  while validation loss fell 37%, and was stopped one epoch before the
  predictions flipped.

The rule under test: an epoch counts against the model only when it is BOTH
meaningfully worse than the best AND validation loss has stopped improving.
The stalled cases below are as important as the converging ones -- a change
that keeps every run alive forever would pass half this file and be useless.
"""

import random

import pytest

from dlclassifier_server.services.training_service import EarlyStopping


class _Model:
    """Stand-in for nn.Module; EarlyStopping only calls state_dict()."""

    def state_dict(self):
        return {}


def stop_epoch(mious, val_losses, patience=10, restore=False):
    """Return the epoch early stopping fires on, or None if it never does."""
    es = EarlyStopping(
        patience=patience, min_delta=0.0, restore_best_weights=restore, mode="max"
    )
    model = _Model()
    for i, (m, vl) in enumerate(zip(mious, val_losses), start=1):
        if es(i, m, model, vl):
            return i
    return None


# --- the two real runs -----------------------------------------------------

REAL_NOISY_MIOU = [
    0.3301, 0.3382, 0.7121, 0.4465, 0.6457, 0.4793, 0.7307, 0.7353, 0.6415,
    0.7933, 0.5332, 0.6835, 0.8276, 0.8692, 0.8859, 0.6078, 0.6005, 0.7956,
    0.6288, 0.8837, 0.3517, 0.5695, 0.7305, 0.8289, 0.8647,
]
REAL_NOISY_VAL = [
    0.5774, 0.5218, 0.3413, 0.4658, 0.3597, 0.5786, 0.2807, 0.2392, 0.3916,
    0.2321, 0.5714, 0.3138, 0.1808, 0.1414, 0.1375, 0.4051, 0.3610, 0.2132,
    0.4632, 0.1858, 0.8795, 0.3292, 0.2255, 0.2007, 0.1543,
]

# Same run, resumed. It kept climbing well past where it was stopped.
RESUMED_MIOU = REAL_NOISY_MIOU + [
    0.8870, 0.7679, 0.8507, 0.8186, 0.9059, 0.8256, 0.8744, 0.9266, 0.9162,
    0.9756, 0.9626, 0.9603, 0.9441, 0.9331, 0.9745, 0.9154, 0.9697, 0.9014,
    0.9435, 0.6801,
]
RESUMED_VAL = REAL_NOISY_VAL + [
    0.1534, 0.2075, 0.1432, 0.1920, 0.0956, 0.1875, 0.1730, 0.1108, 0.0779,
    0.0324, 0.0397, 0.0510, 0.0514, 0.0578, 0.0378, 0.0758, 0.0379, 0.0983,
    0.0486, 0.3248,
]

COLLAPSE_MIOU = [0.1861] * 11 + [
    0.6480, 0.9105, 0.9173, 0.9315, 0.9935, 0.9998, 0.9632, 0.9259, 0.9519,
    0.9879, 0.9893, 0.9721, 0.9414, 0.9450, 0.9352, 0.9365,
]
COLLAPSE_VAL = [
    0.6394, 0.6473, 0.7623, 0.7100, 0.7199, 0.7153, 0.5903, 0.5593, 0.5758,
    0.5266, 0.4827, 0.4715, 0.2549, 0.2274, 0.0837, 0.1682, 0.0652, 0.0480,
    0.0793, 0.0966, 0.1211, 0.0658, 0.0538, 0.0629, 0.0596, 0.0715, 0.0684,
]


class TestKeepsConvergingRunsAlive:
    def test_noisy_metric_does_not_stop_at_the_observed_epoch(self):
        # The run really was stopped at 25, keeping 0.886, when 0.976 was
        # still ahead of it.
        assert stop_epoch(REAL_NOISY_MIOU, REAL_NOISY_VAL) is None

    def test_the_epoch_it_would_have_reached_is_much_better(self):
        # Guards the premise: stopping early cost real quality, it was not a
        # wash. If this ever fails the fixture has been mangled.
        assert max(RESUMED_MIOU[:25]) == pytest.approx(0.8859)
        assert max(RESUMED_MIOU) == pytest.approx(0.9756)

    def test_resumed_run_survives_past_the_old_stop(self):
        stopped = stop_epoch(RESUMED_MIOU, RESUMED_VAL)
        assert stopped is None or stopped > 35, (
            "must not stop before the epoch that produced the best model, got %s"
            % stopped
        )

    def test_flat_quantised_metric_with_falling_loss_does_not_stop(self):
        # mean IoU frozen at 0.1861 for eleven epochs while val_loss fell 37%.
        assert stop_epoch(COLLAPSE_MIOU, COLLAPSE_VAL) is None


class TestStillStopsDeadRuns:
    def test_stalled_at_a_good_score_still_stops(self):
        rng = random.Random(11)
        miou = [0.90 + rng.gauss(0, 0.01) for _ in range(40)]
        val = [0.20 + rng.gauss(0, 0.005) for _ in range(40)]
        assert stop_epoch(miou, val) is not None

    def test_flat_metric_with_rising_loss_still_stops(self):
        # Overfitting: the metric is going nowhere and the loss is climbing.
        rng = random.Random(11)
        miou = [0.20 + rng.gauss(0, 0.01) for _ in range(40)]
        val = [0.90 + rng.gauss(0, 0.01) + 0.004 * i for i in range(40)]
        assert stop_epoch(miou, val) is not None

    def test_perfectly_flat_metric_and_flat_loss_still_stops(self):
        # The case a naive tolerance rule gets wrong: with zero spread, any
        # "within tolerance of best" test accepts equality forever.
        rng = random.Random(11)
        miou = [0.1861] * 40
        val = [0.60 + rng.gauss(0, 0.004) for _ in range(40)]
        stopped = stop_epoch(miou, val)
        assert stopped is not None, "a run with no movement at all must stop"

    def test_a_genuinely_declining_run_stops(self):
        miou = [0.9 - 0.02 * i for i in range(40)]
        val = [0.1 + 0.02 * i for i in range(40)]
        assert stop_epoch(miou, val) is not None


class TestBackwardCompatibility:
    def test_secondary_value_is_optional(self):
        # Callers that monitor val_loss itself pass nothing, and the old
        # three-argument form must keep working.
        es = EarlyStopping(patience=3, min_delta=0.0, restore_best_weights=False, mode="max")
        model = _Model()
        fired = [es(i, 0.5, model) for i in range(1, 8)]
        assert any(fired), "a flat metric with no secondary signal must stop"
