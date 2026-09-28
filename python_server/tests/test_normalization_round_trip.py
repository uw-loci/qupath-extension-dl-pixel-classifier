"""The preprocessing a model is trained on must be the preprocessing it is run on.

Two defects lived here together, and both were invisible because each side
looked correct on its own.

The statistics computed from the training patches were written into
metadata.json and used by inference, but never used BY TRAINING -- the dataset
normalized each patch against that patch's own percentiles. So the model
learned on one preprocessing and was served another. Measured on a real
brightfield model, the two disagreed on 1.2% of pixels on average and up to
8.5% on a single tile.

And when ``per_channel`` is false -- which means "one shared transform, so the
ratios between channels survive" -- the precomputed path took CHANNEL ZERO's
statistics and applied them to every channel. On an H&E model red's p1 was 97
while green's was 43, so everything below 97 in green and blue was clipped
flat. That is not a joint statistic, it is the red channel's.
"""

import numpy as np
import pytest

from dlclassifier_server.utils.normalization import _pool_stats, normalize


def _stats(p1, p99, mean, std, lo=0.0, hi=255.0):
    return {"p1": p1, "p99": p99, "min": lo, "max": hi, "mean": mean, "std": std}


class TestPooling:
    def test_percentiles_widen_to_cover_every_channel(self):
        # The real H&E numbers that motivated this.
        pooled = _pool_stats(
            [
                _stats(97, 245, 214, 40),
                _stats(43, 245, 189, 67),
                _stats(83, 245, 205, 48),
            ]
        )
        assert pooled["p1"] == 43  # not 97, which would clip green flat
        assert pooled["p99"] == 245

    def test_a_single_channel_is_returned_unchanged(self):
        one = _stats(10, 200, 100, 30)
        assert _pool_stats([one]) == one

    def test_pooled_spread_counts_the_gap_between_channel_means(self):
        # Channels with identical spread but different means are jointly MORE
        # spread than either alone. Averaging the standard deviations would
        # miss that and understate the range a z-score maps.
        same_mean = _pool_stats([_stats(0, 255, 100, 10), _stats(0, 255, 100, 10)])
        far_apart = _pool_stats([_stats(0, 255, 50, 10), _stats(0, 255, 150, 10)])
        assert same_mean["std"] == pytest.approx(10.0)
        assert far_apart["std"] > same_mean["std"]
        assert far_apart["mean"] == pytest.approx(100.0)

    def test_no_stats_is_not_a_crash(self):
        assert _pool_stats([]) == {}


class TestSharedTransformPreservesChannelRatios:
    def test_per_channel_false_applies_one_transform_to_all_channels(self):
        # The point of per_channel=False: a pixel brighter in red than green
        # stays brighter in red than green afterwards.
        cfg = {
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "precomputed": True,
                "channel_stats": [
                    _stats(97, 245, 214, 40),
                    _stats(43, 245, 189, 67),
                    _stats(83, 245, 205, 48),
                ],
            }
        }
        img = np.zeros((4, 4, 3), np.float32)
        img[..., 0], img[..., 1], img[..., 2] = 200.0, 120.0, 60.0
        out = normalize(img.copy(), cfg)
        assert out[0, 0, 0] > out[0, 0, 1] > out[0, 0, 2]

    def test_green_is_no_longer_clipped_flat_by_reds_floor(self):
        # Every value below red's p1 of 97 used to collapse to 0.
        cfg = {
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "precomputed": True,
                "channel_stats": [
                    _stats(97, 245, 214, 40),
                    _stats(43, 245, 189, 67),
                    _stats(83, 245, 205, 48),
                ],
            }
        }
        dark = np.full((2, 2, 3), 60.0, np.float32)
        darker = np.full((2, 2, 3), 50.0, np.float32)
        assert (
            normalize(dark.copy(), cfg)[0, 0, 1]
            > normalize(darker.copy(), cfg)[0, 0, 1]
        )

    def test_per_channel_true_still_normalizes_each_channel_separately(self):
        cfg = {
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": True,
                "precomputed": True,
                "channel_stats": [_stats(0, 100, 50, 10), _stats(0, 200, 100, 20)],
            }
        }
        img = np.full((2, 2, 2), 100.0, np.float32)
        out = normalize(img.copy(), cfg)
        # 100 is the top of channel 0's range and the middle of channel 1's.
        assert out[0, 0, 0] == pytest.approx(1.0)
        assert out[0, 0, 1] == pytest.approx(0.5)


class TestTrainingAndInferenceAgree:
    def test_the_same_config_gives_the_same_answer_on_both_sides(self):
        # Training and inference now call this same function with the same
        # dict. The property that matters is that supplying the stats does
        # not depend on WHO is asking.
        cfg = {
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "precomputed": True,
                "channel_stats": [
                    _stats(97, 245, 214, 40),
                    _stats(43, 245, 189, 67),
                    _stats(83, 245, 205, 48),
                ],
            }
        }
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 255, (8, 8, 3)).astype(np.float32)
        np.testing.assert_array_equal(
            normalize(img.copy(), cfg), normalize(img.copy(), cfg)
        )

    def test_precomputed_and_per_tile_differ_which_is_why_this_matters(self):
        # Guards the premise: if these two agreed, none of the above would be
        # worth doing, and a future change that made them agree trivially
        # (by ignoring the stats) should fail here.
        stats_cfg = {
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "precomputed": True,
                "channel_stats": [_stats(97, 245, 214, 40)],
            }
        }
        tile_cfg = {
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "clip_percentile": 99.0,
            }
        }
        rng = np.random.default_rng(1)
        img = rng.uniform(0, 255, (16, 16, 3)).astype(np.float32)
        assert not np.allclose(
            normalize(img.copy(), stats_cfg), normalize(img.copy(), tile_cfg)
        )


class TestDegenerateStatsDoNotReturnRawPixels:
    """Statistics with no range used to leave the image completely unscaled.

    ``_normalize_with_stats`` guards its division with ``if p_max > p_min``,
    and when that guard fails it returns the input untouched -- raw 0-255
    where the model expects 0-1. Nothing downstream checks the range, so
    inference produced silent nonsense and training produced a NaN loss
    within two epochs. A constant channel or a blank annotated region is
    enough to trigger it.
    """

    FLAT = {
        "p1": 128.0,
        "p99": 128.0,
        "min": 128.0,
        "max": 128.0,
        "mean": 128.0,
        "std": 0.0,
    }

    def _cfg(self, strategy, stats):
        return {
            "normalization": {
                "strategy": strategy,
                "per_channel": False,
                "precomputed": True,
                "clip_percentile": 99.0,
                "channel_stats": stats,
            }
        }

    def test_a_flat_channel_falls_back_instead_of_passing_pixels_through(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 255, (8, 8, 3)).astype(np.float32)
        out = normalize(img.copy(), self._cfg("percentile_99", [self.FLAT]))
        assert out.max() <= 1.0 + 1e-6, "returned unnormalized pixels"
        assert out.min() >= -1e-6
        assert out.max() > out.min(), "fallback still has to normalize something"

    def test_one_dead_channel_among_good_ones_still_triggers_the_fallback(self):
        # The dangerous case: two channels scaled, one raw. Harder to notice
        # than everything being wrong.
        good = _stats(10, 200, 100, 30)
        img = np.random.default_rng(1).uniform(0, 255, (8, 8, 3)).astype(np.float32)
        out = normalize(img.copy(), self._cfg("percentile_99", [good, self.FLAT, good]))
        assert out.max() <= 1.0 + 1e-6

    @pytest.mark.parametrize(
        "strategy,stats",
        [
            ("percentile_99", {"p1": 5.0, "p99": 5.0}),
            ("min_max", {"min": 7.0, "max": 7.0}),
            ("z_score", {"mean": 3.0, "std": 0.0}),
        ],
    )
    def test_every_strategy_checks_the_field_it_actually_divides_by(
        self, strategy, stats
    ):
        img = np.random.default_rng(2).uniform(0, 255, (8, 8, 3)).astype(np.float32)
        out = normalize(img.copy(), self._cfg(strategy, [stats]))
        assert out.max() <= 1.0 + 1e-6, "%s returned unnormalized pixels" % strategy

    def test_usable_stats_are_still_used(self):
        # The fallback must not swallow the normal case: these stats have a
        # range, so they should be applied rather than ignored.
        img = np.full((4, 4, 3), 150.0, np.float32)
        out = normalize(
            img.copy(), self._cfg("percentile_99", [_stats(100, 200, 150, 25)])
        )
        assert out[0, 0, 0] == pytest.approx(0.5)
