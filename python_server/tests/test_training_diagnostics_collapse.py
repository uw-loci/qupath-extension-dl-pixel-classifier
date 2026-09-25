"""Collapse detection in TrainingDiagnostics.

A model that predicts one class over the whole image satisfies an exact
identity rather than a threshold: for the predicted class there are no false
negatives, so its IoU equals accuracy (both are that class's share of the
labeled pixels), and every other class has zero true positives, so IoU is 0.

These tests pin that identity, because the earlier collapse check keyed on
"is the dominant class above 0.5 IoU" -- which is a statement about how
balanced the data happens to be, not about whether the model collapsed. It
misses a collapse onto a class that holds 45% of the pixels and it can fire on
a badly imbalanced run that has not collapsed at all.
"""

from dlclassifier_server.services.training_diagnostics import TrainingDiagnostics

CLASSES = ["Tissue", "Ignore*"]


def _epoch(accuracy, ious, train_loss=0.5):
    return {
        "epoch": 1,
        "train_loss": train_loss,
        "val_loss": train_loss,
        "accuracy": accuracy,
        "per_class_iou": dict(ious),
        "per_class_loss": {c: 0.5 for c in ious},
        "mean_iou": sum(ious.values()) / max(len(ious), 1),
    }


def _collapsed_epoch(prior=0.586, onto="Ignore*"):
    """The observed failure: everything predicted as one class."""
    ious = {c: 0.0 for c in CLASSES}
    ious[onto] = prior
    return _epoch(prior, ious)


def _healthy_epoch():
    return _epoch(0.95, {"Tissue": 0.90, "Ignore*": 0.93})


def test_detects_collapse_onto_majority_class():
    diag = TrainingDiagnostics(classes=CLASSES)
    history = [_collapsed_epoch() for _ in range(10)]

    warnings = diag.run_checks(history, min_epochs=10)

    assert any("predicting 'Ignore*' everywhere" in w for w in warnings)


def test_detects_collapse_when_classes_are_nearly_balanced():
    """The case a fixed IoU threshold misses.

    Collapsing onto a class that holds 45% of the pixels pins accuracy at
    0.45. A check asking "is the dominant class above 0.5" reads that as
    healthy; the identity does not care about the split.
    """
    diag = TrainingDiagnostics(classes=CLASSES)
    history = [_collapsed_epoch(prior=0.45) for _ in range(10)]

    warnings = diag.run_checks(history, min_epochs=10)

    assert any("predicting 'Ignore*' everywhere" in w for w in warnings)


def test_quiet_on_a_model_that_learned_both_classes():
    diag = TrainingDiagnostics(classes=CLASSES)
    history = [_healthy_epoch() for _ in range(10)]

    warnings = diag.run_checks(history, min_epochs=10)

    assert not any("everywhere" in w for w in warnings)


def test_quiet_on_an_imbalanced_run_that_did_not_collapse():
    """Weak on one class is not the same as predicting one class.

    Tissue IoU 0.12 is poor, but it is non-zero and accuracy sits well above
    the Ignore* IoU, so the identity does not hold and this check stays out
    of it. Other checks own "class is doing badly".
    """
    diag = TrainingDiagnostics(classes=CLASSES)
    history = [_epoch(0.72, {"Tissue": 0.12, "Ignore*": 0.68}) for _ in range(10)]

    warnings = diag.run_checks(history, min_epochs=10)

    assert not any("everywhere" in w for w in warnings)


def test_quiet_once_the_run_has_escaped():
    """A run that collapsed early and recovered needs no warning.

    This is the 150-epoch run that escaped at epoch 110: reporting the
    collapse after it ended would send the user chasing a fixed problem.
    """
    diag = TrainingDiagnostics(classes=CLASSES)
    history = [_collapsed_epoch() for _ in range(8)] + [
        _healthy_epoch() for _ in range(4)
    ]

    warnings = diag.run_checks(history, min_epochs=10)

    assert not any("everywhere" in w for w in warnings)


def test_reports_a_late_collapse():
    """The mirror of the escape case: healthy, then collapsed and staying so."""
    diag = TrainingDiagnostics(classes=CLASSES)
    history = [_healthy_epoch() for _ in range(7)] + [
        _collapsed_epoch() for _ in range(3)
    ]

    warnings = diag.run_checks(history, min_epochs=10)

    assert any("everywhere" in w for w in warnings)


def test_warns_once_not_every_ten_epochs():
    diag = TrainingDiagnostics(classes=CLASSES)
    history = [_collapsed_epoch() for _ in range(10)]

    first = diag.run_checks(history, min_epochs=10)
    history.extend(_collapsed_epoch() for _ in range(10))
    second = diag.run_checks(history, min_epochs=10)

    assert any("everywhere" in w for w in first)
    assert not any("everywhere" in w for w in second)


def test_single_class_model_is_not_a_collapse():
    """With one class there is nothing to collapse onto."""
    diag = TrainingDiagnostics(classes=["Tissue"])
    history = [_epoch(0.99, {"Tissue": 0.99}) for _ in range(10)]

    warnings = diag.run_checks(history, min_epochs=10)

    assert not any("everywhere" in w for w in warnings)


def test_brief_collapse_is_not_reported():
    """Two collapsed epochs inside a run that is otherwise fine.

    Early epochs often look collapsed before the model finds the minority
    class; the check needs a sustained state, and the latest epoch still
    collapsed, before it says anything.
    """
    diag = TrainingDiagnostics(classes=CLASSES)
    history = (
        [_collapsed_epoch() for _ in range(2)]
        + [_healthy_epoch() for _ in range(7)]
        + [_collapsed_epoch()]
    )

    warnings = diag.run_checks(history, min_epochs=10)

    assert not any("everywhere" in w for w in warnings)
