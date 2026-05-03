"""Portable inference preprocessing helpers.

This module is the **public inference-preprocessing API** for models
trained by the DL Pixel Classifier extension. Both QuPath's inference
path and any standalone Python inference code (e.g. an external
fly-wing pipeline that loads ``model.pt`` directly) call into the same
functions here, so the preprocessing applied to model inputs is
identical in both environments.

Whatever the model was trained against is the contract; the contract is
encoded in ``metadata.json`` next to ``model.pt``. This module reads that
metadata and applies the corresponding preprocessing steps, in order:

    1. Channel selection / ordering (if the model was trained on a
       specific subset / order of channels).
    2. Cast and value-range rescale (uint16 -> float32, 0..255 -> 0..1).
    3. Pixel-size resample (target resolution -> training resolution),
       skippable via ``resample=False``.
    4. Per-channel normalization (mean/std or percentile, as recorded
       in ``input_config.normalization``).

Standalone Python usage::

    import imageio.v3 as iio
    from dlclassifier_server.inference_preprocess import (
        load_metadata, preprocess_for_inference,
    )

    meta = load_metadata("/path/to/model_dir")
    img = iio.imread("/path/to/target_image.tif")  # HWC, source resolution
    source_um_per_px = 0.25  # microns per pixel of the target image
    ready = preprocess_for_inference(img, source_um_per_px, meta)
    # `ready` is now identical in shape / dtype / range to what the model
    # saw during training, ready to be tiled and run through model.pt.

When loading a model produced before this contract was added, the
metadata may lack ``training_pixel_size_um`` or normalization fields. In
that case, ``preprocess_for_inference`` skips the missing steps and logs
a warning -- predictions are still computed but cross-batch correctness
cannot be guaranteed.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils.normalization import normalize as _normalize_with_config

logger = logging.getLogger(__name__)


# Tolerance for "pixel sizes are effectively equal" check; below this
# fractional difference, resampling is a no-op. 1% is well within the
# precision of typical microscope calibration.
_PIXEL_SIZE_EQ_TOLERANCE = 0.01


def load_metadata(model_dir) -> Dict[str, Any]:
    """Read and return ``metadata.json`` from a saved model directory.

    Args:
        model_dir: Path to the directory containing ``metadata.json``.

    Returns:
        Parsed metadata dict.

    Raises:
        FileNotFoundError: when ``metadata.json`` is missing.
        ValueError: when the file is not valid JSON.
    """
    path = Path(model_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            "metadata.json not found in %s -- this directory does not "
            "contain a model produced by the DL Pixel Classifier "
            "extension." % model_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        raise ValueError("metadata.json in %s is not valid JSON: %s"
                         % (model_dir, e)) from e


def select_and_order_channels(
    image: np.ndarray, metadata: Dict[str, Any]
) -> np.ndarray:
    """Apply the model's saved channel selection / ordering.

    Reads ``input_config.selected_channels`` from metadata if present.
    No-op when the metadata records no channel selection (model was
    trained on all channels in their native order) or the input is
    grayscale.

    Args:
        image: HWC or HW numpy array.
        metadata: Output of :func:`load_metadata`.

    Returns:
        Image with channels selected/reordered to match training.
    """
    if image.ndim != 3:
        return image
    input_config = metadata.get("input_config", {})
    selected = input_config.get("selected_channels")
    if not selected:
        return image
    selected = list(selected)
    n_ch = image.shape[2]
    if any(c < 0 or c >= n_ch for c in selected):
        raise ValueError(
            "metadata selected_channels=%s but image has only %d "
            "channels" % (selected, n_ch))
    return image[..., selected]


def cast_and_rescale(
    image: np.ndarray, metadata: Dict[str, Any]
) -> np.ndarray:
    """Cast to float32 and rescale to the training value range.

    Uses ``input_config.bit_depth`` from metadata if present to compute
    the source range. Common cases:
      - uint8 input  -> divide by 255.0
      - uint16 input -> divide by 65535.0
      - float input  -> passthrough (already 0..1 or normalized)

    Args:
        image: HWC or HW numpy array, any numeric dtype.
        metadata: Output of :func:`load_metadata`.

    Returns:
        float32 array with values in 0..1 (when applicable).
    """
    arr = image
    if arr.dtype == np.float32:
        return arr
    if arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        denom = float(info.max)
        return (arr.astype(np.float32) / denom).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def resample_to_training_resolution(
    image: np.ndarray,
    source_um_per_px: Optional[float],
    metadata: Dict[str, Any],
    *,
    resample: bool = True,
) -> Tuple[np.ndarray, float]:
    """Resample ``image`` to the model's training pixel size.

    The decision is driven entirely by metadata: training pixel size is
    read from ``metadata['training_pixel_size_um']``. If the metadata
    lacks that field (older models), or the source pixel size is
    unknown, this is a no-op.

    Anti-aliasing on downsample, bilinear on upsample. Identity within
    a 1% tolerance on the ratio.

    Args:
        image: HWC or HW numpy array.
        source_um_per_px: Pixel size of the input image, in microns.
            None or NaN means "unknown" -- function is a no-op.
        metadata: Output of :func:`load_metadata`.
        resample: When False, skip the resize step entirely (still
            returns the input image and ``scale_factor=1.0``).
            Documented opt-out for advanced users who want to handle
            scale themselves.

    Returns:
        ``(resampled_image, scale_factor)`` where ``scale_factor`` is
        ``source_um_per_px / training_um_per_px`` (i.e. how many
        training pixels each source pixel covers). Returns
        ``(image, 1.0)`` when no resample is performed.
    """
    if not resample:
        return image, 1.0

    train_px = metadata.get("training_pixel_size_um")
    if train_px is None or train_px <= 0:
        logger.debug(
            "metadata lacks training_pixel_size_um -- skipping resample")
        return image, 1.0
    if source_um_per_px is None:
        logger.debug(
            "source_um_per_px not provided -- skipping resample")
        return image, 1.0
    try:
        source_px = float(source_um_per_px)
    except (TypeError, ValueError):
        return image, 1.0
    if source_px <= 0 or source_px != source_px:  # NaN check
        return image, 1.0

    # scale factor: source per training, i.e. how to multiply source
    # dimensions to reach the model's expected resolution. Larger source
    # pixel (coarser image) -> need to upsample -> scale > 1. Smaller
    # source pixel (finer image) -> need to downsample -> scale < 1.
    # Resize target dims = source_dims * (source_px / train_px).
    ratio = source_px / float(train_px)
    if abs(ratio - 1.0) < _PIXEL_SIZE_EQ_TOLERANCE:
        return image, 1.0

    # Anti-aliased resize. We import lazily so the module is importable
    # without skimage installed (only the resample step needs it).
    try:
        from skimage.transform import resize as _sk_resize
    except ImportError as e:
        raise ImportError(
            "skimage is required for pixel-size resampling. "
            "Either install scikit-image, or pass resample=False to "
            "skip this step.") from e

    if image.ndim == 2:
        h, w = image.shape
        new_h = max(1, int(round(h * ratio)))
        new_w = max(1, int(round(w * ratio)))
        out = _sk_resize(
            image, (new_h, new_w),
            order=1,  # bilinear
            mode="reflect",
            anti_aliasing=(ratio < 1.0),
            preserve_range=True,
        ).astype(image.dtype, copy=False)
    elif image.ndim == 3:
        h, w, c = image.shape
        new_h = max(1, int(round(h * ratio)))
        new_w = max(1, int(round(w * ratio)))
        out = _sk_resize(
            image, (new_h, new_w, c),
            order=1,
            mode="reflect",
            anti_aliasing=(ratio < 1.0),
            preserve_range=True,
        ).astype(image.dtype, copy=False)
    else:
        raise ValueError(
            "image must be HW or HWC; got shape %s" % (image.shape,))

    logger.info(
        "Resampled %s -> %s (source=%.4f um/px, training=%.4f um/px, "
        "ratio=%.4f)",
        image.shape, out.shape, source_px, train_px, ratio)
    return out, ratio


def normalize_for_inference(
    image: np.ndarray, metadata: Dict[str, Any]
) -> np.ndarray:
    """Apply the model's saved per-channel normalization.

    Delegates to the shared
    :func:`dlclassifier_server.utils.normalization.normalize`, which
    handles both precomputed (image-level) and per-tile fallback
    strategies and is the same code path the inference and training
    services use.

    Args:
        image: HWC or HW numpy array.
        metadata: Output of :func:`load_metadata`.

    Returns:
        Normalized image array (same shape).
    """
    input_config = metadata.get("input_config", {})
    return _normalize_with_config(image, input_config)


def preprocess_for_inference(
    image: np.ndarray,
    source_um_per_px: Optional[float],
    metadata: Dict[str, Any],
    *,
    resample: bool = True,
) -> np.ndarray:
    """Run the full preprocessing pipeline in canonical order.

    This is what 99% of standalone-Python callers should use. The
    per-step functions exist for advanced users who need to interleave
    custom logic (e.g., a per-tile preprocessing pipeline that reads
    tiles from disk one at a time).

    Order: select_and_order_channels -> cast_and_rescale ->
    resample_to_training_resolution -> normalize_for_inference.

    Args:
        image: HWC or HW numpy array, any numeric dtype.
        source_um_per_px: Pixel size of the input image, in microns.
            None when unknown -- resample step becomes a no-op.
        metadata: Output of :func:`load_metadata`.
        resample: Forwards to
            :func:`resample_to_training_resolution`.

    Returns:
        float32 array, channel-selected, value-rescaled,
        pixel-size-corrected, and normalized.
    """
    img = select_and_order_channels(image, metadata)
    img = cast_and_rescale(img, metadata)
    img, _ = resample_to_training_resolution(
        img, source_um_per_px, metadata, resample=resample)
    img = normalize_for_inference(img, metadata)
    return img


__all__ = [
    "load_metadata",
    "select_and_order_channels",
    "cast_and_rescale",
    "resample_to_training_resolution",
    "normalize_for_inference",
    "preprocess_for_inference",
]
