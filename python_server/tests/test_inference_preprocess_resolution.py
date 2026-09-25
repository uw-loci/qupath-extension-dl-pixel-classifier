"""The training resolution a standalone caller must resample to.

``training_pixel_size_um`` in metadata.json is the NATIVE pixel size of the
source images. Training tiles are read with a downsample relative to each
server's full resolution, so the model consumed ``native * downsample``.
Resampling to the native value leaves the input too fine by exactly the
downsample factor, silently: the model runs and the structures in view are
simply the wrong size for it.

Every model on the reference project except three had downsample > 1, so this
affected almost all of them. It survived because nothing in the repository
calls this module -- it ships only for standalone consumers.
"""

import numpy as np
import pytest

from dlclassifier_server.inference_preprocess import (
    preprocess_for_inference,
    resample_to_training_resolution,
    training_pixel_size,
)


def _meta(native=0.499, downsample=1.0, effective=None, **arch):
    md = {"architecture": {"downsample": downsample, **arch}}
    if native is not None:
        md["training_pixel_size_um"] = native
    if effective is not None:
        md["training_effective_pixel_size_um"] = effective
    return md


class TestTrainingPixelSize:
    def test_prefers_the_explicit_effective_field(self):
        md = _meta(native=0.499, downsample=8.0, effective=3.992)
        assert training_pixel_size(md) == pytest.approx(3.992)

    @pytest.mark.parametrize(
        "downsample,expected", [(1.0, 0.499), (2.0, 0.998), (4.0, 1.996), (8.0, 3.992)]
    )
    def test_derives_it_for_models_written_before_that_field(
        self, downsample, expected
    ):
        """The fallback is what repairs existing models.

        architecture.downsample has always been recorded correctly, including
        on the oldest model on disk, so no model needs retraining.
        """
        md = _meta(native=0.499, downsample=downsample)
        assert training_pixel_size(md) == pytest.approx(expected)

    def test_missing_downsample_is_treated_as_full_resolution(self):
        md = {"training_pixel_size_um": 0.25, "architecture": {}}
        assert training_pixel_size(md) == pytest.approx(0.25)

    def test_no_native_size_means_unknown(self):
        """A training set of mixed pixel sizes records no native size.

        TrainingDialog sets it to NaN when the selected images disagree, and
        toMap() then omits the field, so there is nothing to derive from --
        correctly, because each image was read at its OWN native times the
        downsample and the model saw a mixture of resolutions.
        """
        assert training_pixel_size({"architecture": {"downsample": 4.0}}) is None

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), "not a number", None])
    def test_unusable_native_values_are_unknown(self, bad):
        md = {"training_pixel_size_um": bad, "architecture": {"downsample": 2.0}}
        assert training_pixel_size(md) is None

    def test_garbage_effective_field_falls_back_rather_than_crashing(self):
        md = _meta(native=0.5, downsample=2.0, effective="oops")
        assert training_pixel_size(md) == pytest.approx(1.0)

    def test_zero_downsample_does_not_collapse_the_size(self):
        md = _meta(native=0.5, downsample=0.0)
        assert training_pixel_size(md) == pytest.approx(0.5)


class TestResampleToTrainingResolution:
    @staticmethod
    def _img(h=64, w=64):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_same_resolution_image_is_still_downsampled(self):
        """The failure that was silent.

        Source and training images share a native pixel size, so the old code
        computed ratio 1.0 and did nothing -- while the model expected data 8x
        coarser. Nothing told the caller a step was still owed.
        """
        out, scale = resample_to_training_resolution(
            self._img(64, 64), 0.499, _meta(0.499, 8.0)
        )

        assert scale == pytest.approx(0.499 / 3.992)
        assert out.shape[:2] == (8, 8)

    def test_different_resolution_image_targets_the_effective_size(self):
        # Source 1.0 um/px against a model that trained at 3.992: the image
        # must shrink to about a quarter, not grow to reach 0.499.
        out, scale = resample_to_training_resolution(
            self._img(64, 64), 1.0, _meta(0.499, 8.0)
        )

        assert scale == pytest.approx(1.0 / 3.992)
        assert out.shape[0] < 64, "must downsample, not upsample toward the native size"

    def test_downsample_one_model_is_unchanged(self):
        out, scale = resample_to_training_resolution(
            self._img(), 0.499, _meta(0.499, 1.0)
        )

        assert scale == 1.0
        assert out.shape[:2] == (64, 64)

    def test_unknown_training_resolution_is_a_no_op(self):
        out, scale = resample_to_training_resolution(
            self._img(), 0.499, {"architecture": {"downsample": 4.0}}
        )

        assert scale == 1.0
        assert out.shape[:2] == (64, 64)

    def test_unknown_source_resolution_is_a_no_op(self):
        out, scale = resample_to_training_resolution(
            self._img(), None, _meta(0.499, 8.0)
        )

        assert scale == 1.0

    def test_opt_out_is_respected(self):
        out, scale = resample_to_training_resolution(
            self._img(), 0.499, _meta(0.499, 8.0), resample=False
        )

        assert scale == 1.0
        assert out.shape[:2] == (64, 64)


class TestContextScaleModels:
    def test_context_model_is_refused_with_an_explanation(self):
        """Better than a channel-count mismatch deep inside the model.

        A context model takes a detail tile and a wider downsampled context
        tile concatenated on the channel axis. Nothing here builds the second
        one, so the old behaviour handed 3 channels to a 6-channel model.
        """
        md = _meta(
            0.499, 2.0, context_scale=4, input_channels=3, effective_input_channels=6
        )

        with pytest.raises(NotImplementedError, match="context_scale"):
            preprocess_for_inference(np.zeros((64, 64, 3), dtype=np.uint8), 0.499, md)

    def test_single_scale_model_is_not_refused(self):
        md = _meta(0.499, 1.0, context_scale=1, input_channels=3)
        md["input_config"] = {"num_channels": 3}

        out = preprocess_for_inference(np.zeros((64, 64, 3), dtype=np.uint8), 0.499, md)

        assert out.shape[:2] == (64, 64)
