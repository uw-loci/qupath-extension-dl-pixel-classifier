"""
Batch inference via Appose shared memory.

Returns per-class average probabilities (not spatial maps).
Used for MEASUREMENTS output type.

Inputs:
    model_path: str
    tile_data: NDArray - concatenated tiles (N*H*W*C) float32
    tile_ids: list of str
    tile_height: int
    tile_width: int
    num_channels: int
    input_config: dict
    dtype: str ("uint8" or "float32")

Outputs:
    predictions: dict mapping tile_id -> list of per-class probabilities
"""

import numpy as np
import logging

logger = logging.getLogger("dlclassifier.appose.inference_batch")

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


# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: model_path, tile_data, tile_ids, tile_height, tile_width, num_channels, input_config
tile_nd = tile_data

# Phase 4: experimental TensorRT / INT8 provider selection. Default to
# whatever state the service holds (no change) if Java did not pass
# the flags through for the MEASUREMENTS path. Without this the
# measurements path would silently inherit stale provider state from
# the last pixel inference.
try:
    use_tensorrt
except NameError:
    use_tensorrt = False
try:
    use_int8
except NameError:
    use_int8 = False

num_tiles = len(tile_ids)

# Read from shared memory
raw = tile_nd.ndarray().reshape(num_tiles, tile_height, tile_width, num_channels).copy()

# Normalize tiles (channel selection already handled by Java during encoding)
preprocessed = []
for i in range(num_tiles):
    img = _normalize_tile(raw[i], input_config)
    preprocessed.append(img)

# Serialize GPU access AND the provider-toggle / model-load / inference
# trio under the same lock. Without this, a pixel-inference thread can
# evict a cached session while we are about to use it, and concurrent
# reloads race on the GPU. Matches the E.4 fix in inference_pixel*.py.
with inference_lock:
    if hasattr(inference_service, "set_experimental_providers"):
        inference_service.set_experimental_providers(
            use_tensorrt=use_tensorrt, use_int8=use_int8
        )
    model_tuple = inference_service._load_model(model_path)
    # Contract check: the scripts do not select channels, Java does.
    # Catch a producer that forgot, here, where the model's real
    # input size is known.
    inference_service.assert_input_channels(model_tuple, num_channels, "batch tiles")
    all_prob_maps = inference_service._infer_batch_spatial(model_tuple, preprocessed)
    inference_service._cleanup_after_inference()

# Average spatial dims for per-class probabilities (outside lock -- CPU-only numpy)
predictions = {}
for tile_id, prob_map in zip(tile_ids, all_prob_maps):
    class_probs = prob_map.mean(axis=(1, 2))
    predictions[tile_id] = class_probs.tolist()

task.outputs["predictions"] = predictions
