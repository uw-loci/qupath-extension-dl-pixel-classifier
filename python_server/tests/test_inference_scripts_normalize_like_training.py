"""The three JAR inference scripts must normalize a tile exactly as training did.

They used to carry their own copy of the normalization code. When 931ec34
(0.9.15) made per_channel=false pool the per-channel statistics in training,
the copies kept applying channel 0's (red's) statistics to every channel, so
every brightfield model trained since was served different preprocessing than
it learned on. Nothing caught it, because the round-trip tests only ever ran
the package's function on both sides; the scripts were never executed.

This runs each script's own _normalize_tile, with a real InferenceService,
against the function the training dataset calls.
"""

import pathlib

import numpy as np
import pytest

from dlclassifier_server.services.inference_service import InferenceService
from dlclassifier_server.utils.normalization import normalize as training_normalize

SCRIPTS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/main/resources/qupath/ext/dlclassifier/scripts"
)
NAMES = ["inference_pixel.py", "inference_pixel_batch.py", "inference_batch.py"]


def _stats(p1, p99, lo, hi, mean, std):
    return {"p1": p1, "p99": p99, "min": lo, "max": hi, "mean": mean, "std": std}


# The shipped brightfield model's shape of statistics: red's floor far above
# green's and blue's. Under red-only normalization everything in green and blue
# below red's p1 is clipped flat.
HE_STATS = [
    _stats(91.0, 245.0, 20.0, 255.0, 190.0, 30.0),
    _stats(52.0, 240.0, 5.0, 255.0, 150.0, 45.0),
    _stats(70.0, 242.0, 10.0, 255.0, 170.0, 38.0),
]


def _script_normalizer(name, service):
    src = (SCRIPTS / name).read_text().replace("\r\n", "\n")
    start = src.index("def _normalize_tile(")
    end = src.index("\n\n", src.index("return", start))
    ns = {"inference_service": service}
    exec(compile(src[start:end], f"{name}:_normalize_tile", "exec"), ns)
    return ns["_normalize_tile"]


@pytest.fixture(scope="module")
def service():
    return InferenceService(device="cpu")


@pytest.fixture
def tile():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(64, 64, 3)).astype(np.float32)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "strategy", ["percentile_99", "min_max", "z_score", "fixed_range"]
)
@pytest.mark.parametrize("per_channel", [False, True])
def test_script_matches_training(name, strategy, per_channel, service, tile):
    cfg = {
        "normalization": {
            "strategy": strategy,
            "per_channel": per_channel,
            "precomputed": True,
            "channel_stats": HE_STATS,
        }
    }
    expected = training_normalize(tile.copy(), cfg)
    got = _script_normalizer(name, service)(tile.copy(), cfg)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_joint_normalization_does_not_use_reds_floor(name, service):
    # A green value between green's p1 (52) and red's p1 (91) must not be
    # clipped to 0: pooled p1 is 52, so 70 maps above zero.
    img = np.full((4, 4, 3), 70.0, dtype=np.float32)
    cfg = {
        "normalization": {
            "strategy": "percentile_99",
            "per_channel": False,
            "precomputed": True,
            "channel_stats": HE_STATS,
        }
    }
    out = _script_normalizer(name, service)(img, cfg)
    assert out[..., 1].min() > 0.05


@pytest.mark.parametrize("name", NAMES)
def test_script_has_no_private_normalizer(name):
    src = (SCRIPTS / name).read_text()
    assert "channel_stats[0]" not in src
    assert "def _apply_precomputed_stats" not in src
