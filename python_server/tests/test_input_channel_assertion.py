"""The consumer-side channel check.

The Appose inference scripts do not select channels; they trust Java to have
done it during encoding. Four separate Java call sites produce model input and
two of them got it wrong at different times, each surfacing as a conv2d shape
error a dozen frames deep that never mentioned channels:

    RuntimeError: Given groups=1, weight of size [64, 6, 7, 7], expected
    input[20, 8, 384, 384] to have 6 channels, but got 8 channels instead

This check lives at the consumer because that is the only place that sees
every producer, including one that does not exist yet.
"""

import pytest
import torch
import torch.nn as nn

from dlclassifier_server.services.inference_service import InferenceService


def _torch_model(in_channels):
    return ("pytorch", nn.Sequential(nn.Conv2d(in_channels, 8, 3), nn.ReLU()))


class TestDetectsModelInputWidth:
    def test_reads_channels_from_the_stem_convolution(self):
        for ch in (1, 3, 6, 8):
            assert InferenceService.expected_input_channels(_torch_model(ch)) == ch

    def test_reads_the_first_conv_not_a_later_one(self):
        # A later conv has a different in_channels; taking it would report a
        # width the caller never supplies.
        model = ("pytorch", nn.Sequential(nn.Conv2d(6, 16, 3), nn.Conv2d(16, 32, 3)))
        assert InferenceService.expected_input_channels(model) == 6

    def test_returns_none_when_it_cannot_tell(self):
        # No convolution to read; the check must decline rather than guess.
        assert (
            InferenceService.expected_input_channels(("pytorch", nn.Identity())) is None
        )


class TestTheAssertion:
    def test_matching_channels_pass_silently(self):
        InferenceService.assert_input_channels(_torch_model(6), 6)

    def test_the_real_failure_is_named(self):
        with pytest.raises(ValueError) as e:
            InferenceService.assert_input_channels(_torch_model(6), 8, "batch tiles")
        msg = str(e.value)
        assert "8 channel(s)" in msg
        assert "built for 6" in msg
        # The message has to name the cause, or it is no better than the
        # conv2d error it replaces.
        assert "channel selection" in msg
        assert "batch tiles" in msg

    def test_it_declines_rather_than_guessing(self):
        # Unknown on either side means no opinion: never block a run over a
        # model whose shape could not be read.
        InferenceService.assert_input_channels(("pytorch", nn.Identity()), 8)
        InferenceService.assert_input_channels(_torch_model(6), None)

    def test_a_context_scale_model_is_judged_on_its_real_width(self):
        # context_scale doubles the model's input; the check compares against
        # what the model actually has, so a correct doubled payload passes.
        InferenceService.assert_input_channels(_torch_model(12), 12)
        with pytest.raises(ValueError):
            InferenceService.assert_input_channels(_torch_model(12), 6)


class TestOnnxShape:
    def test_reads_nchw_input_spec(self):
        class _Sess:
            def get_inputs(self):
                class _I:
                    shape = [1, 6, 256, 256]

                return [_I()]

        assert InferenceService.expected_input_channels(("onnx", _Sess())) == 6

    def test_a_dynamic_channel_axis_is_not_a_number(self):
        class _Sess:
            def get_inputs(self):
                class _I:
                    shape = ["batch", "channels", 256, 256]

                return [_I()]

        assert InferenceService.expected_input_channels(("onnx", _Sess())) is None
