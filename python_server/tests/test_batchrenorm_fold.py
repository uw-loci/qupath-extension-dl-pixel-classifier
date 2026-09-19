"""Regression guard for the BatchRenorm -> BatchNorm export fold.

The fold builds a fresh ``nn.BatchNorm2d`` per layer. A default-constructed
module lands on the CPU in float32, so on a CUDA run the replacement used to
stay behind while the rest of the model was on the GPU. The subsequent ONNX
export died with "weight is on cpu, different from other tensors on cuda:0",
and the run lost ``model_static_bn.onnx`` -- the INT8/TensorRT variant --
while ``model.onnx`` and ``model_static.onnx`` wrote fine. Because the export
is wrapped in a broad ``except``, the only symptom was one WARNING line in a
90-minute log. Observed on the 100-epoch SpeedTest4 run, 2026-09-19.

CI has no GPU, so the device half of the guard is skipped there. The dtype
half exercises the same defect -- a fresh module not inheriting the source's
placement -- and does run everywhere.
"""

import pytest
import torch
import torch.nn as nn

from dlclassifier_server.utils.batchrenorm import BatchRenorm2d, fold_brn_to_bn


def _brn_model(num_features: int = 8) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, num_features, 3, padding=1),
        BatchRenorm2d(num_features),
        nn.ReLU(),
    )


def test_fold_replaces_every_brn_layer():
    model = _brn_model().eval()
    fold_brn_to_bn(model)
    assert not any(isinstance(m, BatchRenorm2d) for m in model.modules())
    assert sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()) == 1


def test_fold_preserves_running_stats():
    model = _brn_model().eval()
    brn = model[1]
    with torch.no_grad():
        brn.running_mean.fill_(0.25)
        brn.running_var.fill_(1.75)
        brn.weight.fill_(2.0)
        brn.bias.fill_(-0.5)

    fold_brn_to_bn(model)
    bn = model[1]
    assert torch.allclose(bn.running_mean, torch.full_like(bn.running_mean, 0.25))
    assert torch.allclose(bn.running_var, torch.full_like(bn.running_var, 1.75))
    assert torch.allclose(bn.weight, torch.full_like(bn.weight, 2.0))
    assert torch.allclose(bn.bias, torch.full_like(bn.bias, -0.5))


def test_fold_inherits_source_dtype():
    # A fresh BatchNorm2d is float32. If the fold does not carry the source
    # module's dtype across, a float64 model comes back mixed-precision and
    # the export trips the same way it does on a mixed-device model.
    model = _brn_model().eval().to(dtype=torch.float64)
    fold_brn_to_bn(model)
    bn = model[1]
    assert bn.weight.dtype == torch.float64
    assert bn.bias.dtype == torch.float64
    assert bn.running_mean.dtype == torch.float64
    assert bn.running_var.dtype == torch.float64
    # Integral bookkeeping must survive the dtype move untouched.
    assert bn.num_batches_tracked.dtype == torch.int64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fold_keeps_model_on_one_device():
    # The defect itself: every tensor in the folded model must sit on the
    # same device, or torch.onnx.export refuses to trace it.
    model = _brn_model().eval().cuda()
    fold_brn_to_bn(model)

    devices = {t.device for t in model.parameters()}
    devices |= {t.device for t in model.buffers()}
    assert devices == {torch.device("cuda", torch.cuda.current_device())}

    # And the folded model still runs a forward pass on GPU input.
    out = model(torch.randn(1, 3, 16, 16, device="cuda"))
    assert out.device.type == "cuda"
