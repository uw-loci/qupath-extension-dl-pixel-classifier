"""AdaBN calibration finds the layers training actually produces.

Training swaps every BatchNorm2d for BatchRenorm2d (utils/batchrenorm.py), which is an
nn.Module, not a torch _BatchNorm. The calibration looked only for _BatchNorm, so on any
model this extension trains it found nothing, saved an unchanged copy and said it was done.
"""

import torch
import torch.nn as nn

from dlclassifier_server.services.inference_service import InferenceService
from dlclassifier_server.utils.batchrenorm import BatchRenorm2d


def _model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 1),
        BatchRenorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 4, 1),
        nn.BatchNorm2d(4),
        nn.Dropout(0.5),
    )


def test_batchrenorm_and_batchnorm_are_both_calibrated():
    model = _model()
    br, bn, drop = model[1], model[4], model[5]
    weights_before = [p.detach().clone() for p in model.parameters()]
    saved = InferenceService._prepare_adabn(model)
    assert {id(m) for m, _ in saved} == {id(br), id(bn)}
    assert br.training and bn.training and not drop.training and not model[0].training

    # A new acquisition: inputs centered far from what the running stats describe.
    torch.manual_seed(1)
    batches = [torch.randn(16, 3, 8, 8) * 0.5 + 4.0 for _ in range(30)]
    with torch.no_grad():
        target_br = model[0](torch.cat(batches)).mean((0, 2, 3))
        start = (br.running_mean - target_br).abs().max().item()
        for b in batches:
            model(b)
    for m, momentum in saved:
        m.momentum = momentum

    moved = 1 - (br.running_mean - target_br).abs().max().item() / start
    # 30 batches at 0.1 close 1 - 0.9**30 = 96% of the gap; BatchRenorm2d's own 0.01 closes 26%.
    assert moved > 0.9, moved
    assert bn.running_mean.abs().max().item() > 0, "the BatchNorm layer moved too"
    assert br.momentum == 0.01 and bn.momentum == 0.1, "each layer's own momentum is restored"
    for before, after in zip(weights_before, model.parameters()):
        assert torch.equal(before, after), "calibration must not change any weight"


def test_a_model_without_batch_norm_layers_has_nothing_to_calibrate():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.GroupNorm(2, 4), nn.ReLU())
    assert InferenceService._prepare_adabn(model) == []
