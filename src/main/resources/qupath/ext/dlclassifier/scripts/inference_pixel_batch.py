"""
Batch pixel-level inference via Appose.

Saves probability maps to disk as .bin files (same format as HTTP backend).
Used when multiple tiles need spatial probability maps and the caller
expects file-based output (e.g. InferenceWorkflow for OBJECTS output).

Inputs:
    model_path: str
    tile_data: NDArray - concatenated tiles (N*H*W*C) float32
    tile_ids: list of str
    tile_height: int
    tile_width: int
    num_channels: int
    input_config: dict
    output_dir: str
    reflection_padding: int (default 0)

Outputs:
    output_paths: dict mapping tile_id -> output file path
    num_classes: int
"""

import os
import numpy as np
import logging

logger = logging.getLogger("dlclassifier.appose.inference_pixel_batch")

if inference_service is None:
    raise RuntimeError(
        "Inference service not initialized: " + globals().get("init_error", "unknown")
    )


# --- Normalization: the training function, not a copy of it ---
# Tiles are normalized by the SAME function the training dataset calls,
# dlclassifier_server.utils.normalization.normalize (reached through
# inference_service._normalize), so preprocessing round-trips by
# construction. See docs/NORMALIZATION_ROUNDTRIP.md.
#
# This used to be an inline copy, kept because the installed package might
# be older than the JAR (2026-02). It drifted: when 931ec34 (0.9.15) made
# per_channel=false POOL the per-channel statistics in training, the copies
# kept using channel 0's (red's) for every channel, so every brightfield
# model trained since was served different preprocessing than it learned on
# (output matched red-only normalization on 100% of pixels, the training
# preprocessing on 85.9%). The copies also lacked training's fallback for
# statistics with no range. The reason for copying is gone: init_services
# refuses to start an inference service whose package version differs from
# the JAR's, so this function is always the one training used.
def _normalize_tile(img, input_config):
    """Normalize one tile exactly as training normalized its patches."""
    return inference_service._normalize(img, input_config)


# --- ONNX spatial-divisor padding (tiny-unet / muvit edge-tile fix) ---
# Same issue and fix as inference_pixel.py: tiny-unet's exported ONNX graph
# does not carry forward()'s internal pad-to-2^depth, so odd-sized boundary
# tiles crash the U-Net skip Concat. All tiles in this batch share one
# (tile_height, tile_width) -- see the reshape below -- so a single pad makes
# the whole batch divisible. SMP/PyTorch resolve to divisor 1 (no-op).
# (Duplicated from inference_pixel.py because each JAR-bundled script is a
# self-contained payload; the installed package may predate this fix.)


def _onnx_pad_divisor(model_path):
    """Spatial factor the model's ONNX graph needs inputs padded to.

    Reads the model metadata once (cached per worker). Returns 1 for SMP /
    unknown architectures so the caller becomes a no-op for them.
    """
    cache = globals().setdefault("_onnx_pad_divisor_cache", {})
    if model_path in cache:
        return cache[model_path]
    div = 1
    try:
        import json as _json

        with open(os.path.join(model_path, "metadata.json")) as _f:
            _arch = _json.load(_f).get("architecture", {}) or {}
        _mtype = str(_arch.get("type", "")).lower()
        if _mtype == "tiny-unet":
            div = 1 << int(_arch.get("depth", 4))
        elif _mtype == "muvit":
            _ps = int(_arch.get("patch_size", 16))
            _scales = str(_arch.get("level_scales", "1"))
            _mx = max((int(s) for s in _scales.split(",") if s.strip()), default=1)
            div = _ps * _mx
    except Exception as _e:
        logger.warning("Could not read ONNX pad divisor for %s: %s", model_path, _e)
        div = 1
    cache[model_path] = div
    return div


def _infer_batch_with_divisor_padding(
    model_tuple, tiles, reflection_padding, use_tta, pad_div
):
    """Run a uniform-size tile batch padded to a divisible size, then crop.

    All tiles share one shape (the buffer reshape guarantees it), so one
    pad spec applies to the whole batch. Returns a list of (C, H0, W0) maps.
    """
    if not tiles:
        return []
    h0 = tiles[0].shape[0]
    w0 = tiles[0].shape[1]
    rh = min(int(reflection_padding), max(0, h0 - 1))
    rw = min(int(reflection_padding), max(0, w0 - 1))
    extra_h = (pad_div - (h0 + 2 * rh) % pad_div) % pad_div
    extra_w = (pad_div - (w0 + 2 * rw) % pad_div) % pad_div
    pad_spec = ((rh, rh + extra_h), (rw, rw + extra_w), (0, 0))
    padded = []
    for _t in tiles:
        try:
            padded.append(np.pad(_t, pad_spec, mode="reflect"))
        except Exception:
            padded.append(np.pad(_t, pad_spec, mode="edge"))
    out = inference_service._infer_batch_spatial(
        model_tuple, padded, reflection_padding=0, use_tta=use_tta
    )
    return [np.ascontiguousarray(p[:, rh : rh + h0, rw : rw + w0]) for p in out]


# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: model_path, tile_data, tile_ids, tile_height, tile_width, num_channels, input_config, output_dir
# Optional inputs: reflection_padding
tile_nd = tile_data
try:
    reflection_padding
except NameError:
    reflection_padding = 0
try:
    use_tta
except NameError:
    use_tta = False
try:
    output_format
except NameError:
    output_format = "prob_fp32"
try:
    use_tensorrt
except NameError:
    use_tensorrt = False
try:
    use_int8
except NameError:
    use_int8 = False

num_tiles = len(tile_ids)
os.makedirs(output_dir, exist_ok=True)

# Read from shared memory
raw = tile_nd.ndarray().reshape(num_tiles, tile_height, tile_width, num_channels).copy()

# Normalize tiles (channel selection already handled by Java during encoding)
preprocessed = []
for i in range(num_tiles):
    img = _normalize_tile(raw[i], input_config)
    preprocessed.append(img)

# Serialize GPU access AND provider-toggle + model-load + inference
# under the same lock. Without this, a concurrent thread can evict
# the cached session between our provider-toggle and model-load
# calls and race with our forward pass.
with inference_lock:
    # Phase 4: experimental TensorRT / INT8 provider selection.
    # Must be inside the lock so eviction cannot race with another
    # thread's _load_model on the same session.
    if hasattr(inference_service, "set_experimental_providers"):
        inference_service.set_experimental_providers(
            use_tensorrt=use_tensorrt, use_int8=use_int8
        )
    model_tuple = inference_service._load_model(model_path)
    # Contract check: the scripts do not select channels, Java does.
    # Catch a producer that forgot, here, where the model's real
    # input size is known.
    inference_service.assert_input_channels(model_tuple, num_channels, "batch tiles")
    # tiny-unet/muvit ONNX graphs need inputs divisible by the network's
    # downsampling factor (skip-connection Concat). SMP/PyTorch -> divisor 1
    # (no-op). See _onnx_pad_divisor above.
    pad_div = _onnx_pad_divisor(model_path) if model_tuple[0] == "onnx" else 1
    if pad_div > 1:
        all_prob_maps = _infer_batch_with_divisor_padding(
            model_tuple, preprocessed, reflection_padding, use_tta, pad_div
        )
    else:
        all_prob_maps = inference_service._infer_batch_spatial(
            model_tuple,
            preprocessed,
            reflection_padding=reflection_padding,
            use_tta=use_tta,
        )
    inference_service._cleanup_after_inference()

# Save probability maps to disk (outside lock -- file I/O, no GPU).
# Phase 3c: when output_format == "argmax_uint8", write (H, W) uint8 class
# indices instead of (C, H, W) float32 probabilities. Same .bin extension;
# the Java side knows which reader to use based on the flag it passed in.
output_paths = {}
num_classes = 0
for tile_id, prob_map in zip(tile_ids, all_prob_maps):
    num_classes = prob_map.shape[0]
    output_path = os.path.join(output_dir, "%s.bin" % tile_id)
    if output_format == "argmax_uint8":
        argmax_hw = np.argmax(prob_map, axis=0).astype(np.uint8)
        argmax_hw.tofile(output_path)
    else:
        prob_map.astype(np.float32).tofile(output_path)
    output_paths[tile_id] = output_path

task.outputs["output_paths"] = output_paths
task.outputs["num_classes"] = num_classes
task.outputs["output_format"] = output_format
