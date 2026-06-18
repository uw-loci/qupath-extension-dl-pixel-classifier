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


# --- ONNX spatial-divisor padding (tiny-unet / muvit edge-tile fix) ---
# TinyUNet's forward() reflection-pads its input up to a multiple of 2**depth
# and crops back, so PyTorch inference handles any tile size. The EXPORTED
# ONNX graph does NOT carry that dynamic padding, and get_spatial_divisor()
# returns 1 for tiny-unet (correct only for the PyTorch path). The result:
# odd-sized boundary tiles reach the ONNX graph and crash the U-Net skip
# Concat ("Non concat axis dimensions must match: ... 91 and 90"). We fix it
# at the call site -- pad the tile so the array fed to ONNX is divisible by
# the network's downsampling factor, then crop our padding off the output.
# SMP architectures return divisor 1 here (no-op): their ONNX alignment is
# already handled by the static-shape logic inside _infer_batch_spatial, so
# their working path is left completely untouched.

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
        import os as _os
        with open(_os.path.join(model_path, "metadata.json")) as _f:
            _arch = (_json.load(_f).get("architecture", {}) or {})
        _mtype = str(_arch.get("type", "")).lower()
        if _mtype == "tiny-unet":
            div = 1 << int(_arch.get("depth", 4))
        elif _mtype == "muvit":
            _ps = int(_arch.get("patch_size", 16))
            _scales = str(_arch.get("level_scales", "1"))
            _mx = max((int(s) for s in _scales.split(",") if s.strip()),
                      default=1)
            div = _ps * _mx
    except Exception as _e:
        logger.warning("Could not read ONNX pad divisor for %s: %s",
                       model_path, _e)
        div = 1
    cache[model_path] = div
    return div


def _infer_with_divisor_padding(model_tuple, tile_array, reflection_padding,
                                use_tta, pad_div):
    """Run ONNX inference padding the tile to a divisible size, then crop back.

    Adds the reflection context ourselves and passes reflection_padding=0 so
    we control the exact final (divisible) size fed to the graph. Returns a
    (C, H0, W0) probability map matching the input tile size.
    """
    h0 = tile_array.shape[0]
    w0 = tile_array.shape[1]
    # Clamp context so reflect-pad never exceeds the (small edge) tile dim.
    rh = min(int(reflection_padding), max(0, h0 - 1))
    rw = min(int(reflection_padding), max(0, w0 - 1))
    extra_h = (pad_div - (h0 + 2 * rh) % pad_div) % pad_div
    extra_w = (pad_div - (w0 + 2 * rw) % pad_div) % pad_div
    pad_spec = ((rh, rh + extra_h), (rw, rw + extra_w), (0, 0))
    try:
        padded = np.pad(tile_array, pad_spec, mode="reflect")
    except Exception:
        # reflect requires pad <= dim-1; tiny edge tiles fall back to edge.
        padded = np.pad(tile_array, pad_spec, mode="edge")
    prob_maps = inference_service._infer_batch_spatial(
        model_tuple, [padded], reflection_padding=0,
        gpu_batch_size=1, use_tta=use_tta)
    full = prob_maps[0]  # (C, padded_H, padded_W)
    cropped = full[:, rh:rh + h0, rw:rw + w0]
    return np.ascontiguousarray(cropped)


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
try:
    use_tensorrt
except NameError:
    use_tensorrt = False
try:
    use_int8
except NameError:
    use_int8 = False

# Zero-copy read from shared memory NDArray.
# Copy immediately so the shared memory segment can be reused by Java.
tile_array = tile_nd.ndarray().reshape(tile_height, tile_width, num_channels).copy()

# Normalize (thread-safe, no GPU access)
tile_array = _normalize_tile(tile_array, input_config)

# Channel selection is handled by Java during tile encoding.
# Do NOT re-select here -- it would drop context channels for multi-scale models.

# Serialize GPU access AND provider-toggle + model-load + inference
# under the same lock. Appose runs each task in its own thread; without
# this lock, a concurrent thread can call set_experimental_providers,
# evict the _model_cache entry from under us, and then race with the
# reload. The GPU can only run one batch at a time anyway, so
# serializing here is correctness-preserving.
with inference_lock:
    # Phase 4: apply experimental ORT provider flags. This must run
    # INSIDE the lock so an eviction cannot race with another thread's
    # _load_model call on the same session.
    if hasattr(inference_service, "set_experimental_providers"):
        inference_service.set_experimental_providers(
            use_tensorrt=use_tensorrt, use_int8=use_int8)
    model_tuple = inference_service._load_model(model_path)
    # tiny-unet/muvit ONNX graphs need inputs divisible by the network's
    # downsampling factor or the skip-connection Concat fails on odd-sized
    # boundary tiles. SMP/PyTorch resolve to divisor 1 (no-op). See the
    # _onnx_pad_divisor comment above.
    pad_div = _onnx_pad_divisor(model_path) if model_tuple[0] == "onnx" else 1
    if pad_div > 1:
        prob_map = _infer_with_divisor_padding(
            model_tuple, tile_array, reflection_padding, use_tta, pad_div)
    else:
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
