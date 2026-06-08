"""Regression guard for the static-ONNX baked input shape.

The static ONNX must be baked at the size the model is actually fed at
inference. For context-scale models that is tileSize + 2*context_padding
rounded up to the encoder spatial divisor (mirroring the inference path's
DLPixelClassifier.contextInferencePad == inputPadding == context_padding and
pad_to_multiple). Plain models keep the bare tile size. A regression here makes
context-model inference silently take the static-shape fallback (or, worse,
center-crop the context away), which is exactly the eval-vs-inference mismatch
this work fixed. See docs/CONTEXT_AND_DOWNSAMPLE_DESIGN.md.
"""

from dlclassifier_server.utils.spatial import compute_static_onnx_hw


def test_plain_model_uses_bare_tile_size():
    # No context: bake at the configured tile size, unchanged.
    assert compute_static_onnx_hw([512, 512], 1, 0, 32) == (512, 512)
    # context_scale > 1 but no padding -> still unchanged.
    assert compute_static_onnx_hw([512, 512], 4, 0, 32) == (512, 512)


def test_context_model_bakes_padded_size():
    # 512 + 2*128 = 768, already a multiple of 32.
    assert compute_static_onnx_hw([512, 512], 4, 128, 32) == (768, 768)


def test_context_model_rounds_up_to_divisor():
    # 512 + 2*20 = 552 -> next multiple of 32 is 576.
    assert compute_static_onnx_hw([512, 512], 2, 20, 32) == (576, 576)
    # 512 + 2*64 = 640, already a multiple of 32.
    assert compute_static_onnx_hw([512, 512], 4, 64, 32) == (640, 640)


def test_divisor_one_no_rounding():
    # TinyUNet handles its own alignment (divisor 1): pad but do not round.
    assert compute_static_onnx_hw([512, 512], 4, 20, 1) == (552, 552)


def test_non_square_tile():
    assert compute_static_onnx_hw([256, 512], 4, 128, 32) == (512, 768)
