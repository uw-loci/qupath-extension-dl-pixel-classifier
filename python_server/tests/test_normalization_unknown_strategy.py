"""An unknown normalization name must fail, not return the tile unscaled.

Before this check, a name that matched no branch fell through every ``if`` in
normalization.py and the tile came back with raw 0-255 / 0-65535 values. The
name that triggered it in practice was a Java ``toLowerCase()`` run under a
Turkish default locale: ``PERCENTILE_99`` -> ``percent\\u0131le_99``,
``MIN_MAX`` -> ``m\\u0131n_max``, ``FIXED_RANGE`` -> ``f\\u0131xed_range``.
"""

import numpy as np
import pytest

from dlclassifier_server.utils.normalization import (
    KNOWN_STRATEGIES,
    _normalize_single,
    _normalize_with_stats,
    check_strategy,
    normalize,
)

TURKISH_LOWERED = ["percent\u0131le_99", "m\u0131n_max", "f\u0131xed_range"]


def _tile():
    return (np.arange(48, dtype=np.float32).reshape(4, 4, 3) * 5).copy()


@pytest.mark.parametrize("name", TURKISH_LOWERED + ["PERCENTILE_99", "", None])
def test_per_tile_path_refuses_an_unknown_name(name):
    with pytest.raises(ValueError, match="Unknown normalization strategy"):
        normalize(_tile(), {"normalization": {"strategy": name}})


@pytest.mark.parametrize("name", TURKISH_LOWERED)
def test_precomputed_path_refuses_an_unknown_name(name):
    stats = [{"p1": 0, "p99": 200, "min": 0, "max": 235, "mean": 100, "std": 50}] * 3
    config = {
        "normalization": {
            "strategy": name,
            "precomputed": True,
            "channel_stats": stats,
            "per_channel": True,
        }
    }
    with pytest.raises(ValueError, match="Unknown normalization strategy"):
        normalize(_tile(), config)


def test_the_message_is_ascii_and_shows_the_dotless_i():
    with pytest.raises(ValueError) as err:
        check_strategy("percent\u0131le_99")
    message = str(err.value)
    message.encode("ascii")  # cp1252-safe for Windows logs
    assert "percent\\u0131le_99" in message
    assert "percentile_99" in message


def test_the_private_helpers_refuse_it_too():
    with pytest.raises(ValueError):
        _normalize_single(_tile()[..., 0], {}, "m\u0131n_max")
    with pytest.raises(ValueError):
        _normalize_with_stats(_tile()[..., 0], {"min": 0, "max": 1}, "m\u0131n_max")


@pytest.mark.parametrize("name", KNOWN_STRATEGIES)
def test_every_known_name_still_normalizes_into_the_unit_range(name):
    out = normalize(
        _tile(), {"normalization": {"strategy": name, "min": 0, "max": 255}}
    )
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.max() > 0.0


def test_a_missing_name_keeps_the_percentile_default():
    out = normalize(_tile(), {"normalization": {}})
    assert out.max() == pytest.approx(1.0)
