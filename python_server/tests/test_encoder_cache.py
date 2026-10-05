"""The pretrained-encoder cache: what counts as stale, and what does not.

Pre-warming the Hub cache is what lets a first training run work without the
network. Its risk is the mirror image: cached weights that quietly fall behind.
These tests pin the decision, because the obvious version of it is wrong in a
way that only shows up as noise -- see the module docstring.
"""

from dlclassifier_server.services.encoder_cache import (
    CURRENT,
    OFFERED_ENCODERS,
    STALE,
    UNKNOWN,
    classify_weights,
)


class TestClassifyWeights:
    def test_the_weights_we_hold_are_still_published(self):
        assert classify_weights({"aaa"}, {"aaa"}) == CURRENT

    def test_newer_weights_published(self):
        assert classify_weights({"aaa"}, {"bbb"}) == STALE

    def test_one_matching_file_is_enough(self):
        # A repository can publish the same weights in several formats, and
        # the cache holds whichever one was asked for. Requiring every hash to
        # match would call that stale.
        assert classify_weights({"aaa"}, {"aaa", "bbb"}) == CURRENT

    def test_an_extra_cached_format_does_not_make_it_stale(self):
        assert classify_weights({"aaa", "ccc"}, {"aaa"}) == CURRENT

    def test_silence_from_the_hub_is_not_staleness(self):
        # An unreachable Hub, or a repository listing no weight file, says
        # nothing about what is on disk. Reporting that as stale would send
        # people to re-download weights that are fine.
        assert classify_weights({"aaa"}, set()) == UNKNOWN

    def test_an_empty_cache_is_not_staleness_either(self):
        assert classify_weights(set(), {"aaa"}) == UNKNOWN
        assert classify_weights(set(), set()) == UNKNOWN


class TestOfferedEncoders:
    def test_the_offered_set_is_not_empty(self):
        assert OFFERED_ENCODERS

    def test_entries_are_unique(self):
        assert len(OFFERED_ENCODERS) == len(set(OFFERED_ENCODERS))

    def test_mobilenetv3_small_is_pre_warmed(self):
        # The encoder the workshop asked for by name; it is the one most
        # likely to be reached for on a laptop with no GPU.
        assert "timm-mobilenetv3_small_100" in OFFERED_ENCODERS
