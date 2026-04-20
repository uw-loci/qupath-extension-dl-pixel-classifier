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
    raise RuntimeError("Inference service not initialized: " + globals().get("init_error", "unknown"))


# --- Inline normalization for precomputed image-level stats ---
# (See inference_pixel.py for rationale on why this is inlined)

def _apply_precomputed_stats(img, stats, strategy):
    """Normalize a single channel/image using pre-computed statistics."""
    if strategy == "percentile_99":
        p_min = stats.get("p1", float(img.min()))
        p_max = stats.get("p99", float(img.max()))
        img = np.clip(img, p_min, p_max)
        if p_max > p_min:
            img = (img - p_min) / (p_max - p_min)
    elif strategy == "min_max":
        i_min = stats.get("min", float(img.min()))
        i_max = stats.get("max", float(img.max()))
        if i_max > i_min:
            img = (img - i_min) / (i_max - i_min)
    elif strategy == "z_score":
        mean = stats.get("mean", float(img.mean()))
        std = stats.get("std", float(img.std()))
        if std > 0:
            img = (img - mean) / std
            img = np.clip(img, -5, 5)
            img = (img + 5) / 10
    elif strategy == "fixed_range":
        fixed_min = stats.get("min", 0)
        fixed_max = stats.get("max", 255)
        img = np.clip(img, fixed_min, fixed_max)
        if fixed_max > fixed_min:
            img = (img - fixed_min) / (fixed_max - fixed_min)
    return img


def _normalize_tile(img, input_config):
    """Normalize a tile, using precomputed image-level stats when available."""
    norm_config = input_config.get("normalization", {})
    precomputed = norm_config.get("precomputed", False)
    channel_stats = norm_config.get("channel_stats", None)

    if precomputed and channel_stats:
        strategy = norm_config.get("strategy", "percentile_99")
        per_channel = norm_config.get("per_channel", False)
        if per_channel and img.ndim == 3 and img.shape[2] > 1:
            for c in range(min(img.shape[2], len(channel_stats))):
                img[..., c] = _apply_precomputed_stats(
                    img[..., c], channel_stats[c], strategy)
        else:
            stats = channel_stats[0] if channel_stats else {}
            img = _apply_precomputed_stats(img, stats, strategy)
        return img

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
            use_tensorrt=use_tensorrt, use_int8=use_int8)
    model_tuple = inference_service._load_model(model_path)
    all_prob_maps = inference_service._infer_batch_spatial(model_tuple, preprocessed)
    inference_service._cleanup_after_inference()

# Average spatial dims for per-class probabilities (outside lock -- CPU-only numpy)
predictions = {}
for tile_id, prob_map in zip(tile_ids, all_prob_maps):
    class_probs = prob_map.mean(axis=(1, 2))
    predictions[tile_id] = class_probs.tolist()

task.outputs["predictions"] = predictions
