"""
Per-tile pixel inference via Appose shared memory.

Inputs (from Java):
    model_path: str - path to model directory
    tile_data: NDArray - shared memory tile (H, W, C) float32
    tile_height: int
    tile_width: int
    num_channels: int
    input_config: dict - normalization config
    reflection_padding: int (default 0)

Outputs:
    probabilities: NDArray - shared memory probability map (C, H, W) float32
    num_classes: int
"""
import numpy as np
import logging

logger = logging.getLogger("dlclassifier.appose.inference")

# Access persistent globals from init script
if inference_service is None:
    raise RuntimeError("Inference service not initialized: " + globals().get("init_error", "unknown"))


# --- Inline normalization for precomputed image-level stats ---
# This logic is defined here (in the JAR-bundled script) rather than
# delegating entirely to inference_service._normalize(), because the
# dlclassifier_server package installed in the Appose pixi environment
# may be an older version that does not support precomputed stats.
# Scripts loaded from JAR resources are always current.

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

    # Fall back to per-tile normalization via installed InferenceService
    return inference_service._normalize(img, input_config)


# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: model_path, tile_data, tile_height, tile_width, num_channels, input_config
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

# Zero-copy read from shared memory NDArray.
# Copy immediately so the shared memory segment can be reused by Java.
tile_array = tile_nd.ndarray().reshape(tile_height, tile_width, num_channels).copy()

# Normalize (thread-safe, no GPU access)
tile_array = _normalize_tile(tile_array, input_config)

# Channel selection is handled by Java during tile encoding.
# Do NOT re-select here -- it would drop context channels for multi-scale models.

# Serialize GPU access. Appose runs each task in its own thread, so
# without this lock, 10+ overlay threads would race on model loading,
# CUDA memory, and forward passes simultaneously. The GPU can only run
# one batch at a time anyway, so serializing here prevents OOM and
# thread-safety issues (torch.compile is NOT thread-safe).
with inference_lock:
    model_tuple = inference_service._load_model(model_path)
    prob_maps = inference_service._infer_batch_spatial(
        model_tuple, [tile_array],
        reflection_padding=reflection_padding,
        gpu_batch_size=1,
        use_tta=use_tta
    )
    prob_map = prob_maps[0]  # (C, H, W) float32
    inference_service._cleanup_after_inference()

num_classes = prob_map.shape[0]

# Write result to shared memory NDArray (outside lock -- no GPU needed).
# NDArray auto-creates SharedMemory of the correct size when shm=None.
# Phase 3c: optionally return uint8 (H, W) argmax instead of float32 (C,H,W).
from appose import NDArray as PyNDArray

if output_format == "argmax_uint8":
    argmax_hw = np.argmax(prob_map, axis=0).astype(np.uint8)
    out_nd = PyNDArray(dtype="uint8", shape=[tile_height, tile_width])
    np.copyto(out_nd.ndarray(), argmax_hw)
    task.outputs["probabilities"] = out_nd
    task.outputs["num_classes"] = num_classes
    task.outputs["output_format"] = "argmax_uint8"
else:
    out_nd = PyNDArray(dtype="float32", shape=[num_classes, tile_height, tile_width])
    np.copyto(out_nd.ndarray(), prob_map)
    task.outputs["probabilities"] = out_nd
    task.outputs["num_classes"] = num_classes
    task.outputs["output_format"] = "prob_fp32"
