"""Shared normalization utilities for inference and training.

This module provides consistent normalization logic used by both
InferenceService and TrainingService (SegmentationDataset).
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def normalize(img: np.ndarray, input_config: Dict[str, Any]) -> np.ndarray:
    """Normalize image data using config-specified strategy.

    Supports precomputed image-level statistics (when available) to ensure
    consistent normalization across tiles. When precomputed stats are absent,
    falls back to per-tile computation.

    Args:
        img: Input image array (HWC or HW)
        input_config: Configuration dict with "normalization" sub-dict

    Returns:
        Normalized image array
    """
    norm_config = input_config.get("normalization", {})
    strategy = norm_config.get("strategy", "percentile_99")
    per_channel = norm_config.get("per_channel", False)
    precomputed = norm_config.get("precomputed", False)
    channel_stats = norm_config.get("channel_stats", None)

    # Use precomputed image-level stats when available -- but only when they
    # can actually normalize anything. Degenerate stats (p99 == p1, or a zero
    # range) hit the divide-by-zero guard inside _normalize_with_stats, which
    # returns the image UNSCALED: raw 0-255 values where the model expects
    # 0-1. Training on that produces a NaN loss within two epochs, and
    # inference on it produces silent nonsense, because nothing downstream
    # checks the range. Falling back to per-tile normalization keeps the
    # contract that this function always returns something normalized.
    if precomputed and channel_stats:
        if _stats_usable(channel_stats, strategy):
            return _normalize_precomputed(img, channel_stats, strategy, per_channel)
        logger.warning(
            "Precomputed %s statistics have no range (%s); normalizing this "
            "tile against its own percentiles instead. A constant channel or "
            "a blank training region will do this.",
            strategy,
            _describe_degenerate(channel_stats, strategy),
        )

    # Fall back to per-tile normalization
    if per_channel and img.ndim == 3 and img.shape[2] > 1:
        if not img.flags.writeable:
            img = img.copy()
        for c in range(img.shape[2]):
            img[..., c] = _normalize_single(img[..., c], norm_config, strategy)
    else:
        img = _normalize_single(img, norm_config, strategy)

    return img


def _stats_usable(channel_stats: List[Dict[str, float]], strategy: str) -> bool:
    """Whether these statistics can actually rescale an image.

    A zero range makes _normalize_with_stats return its input untouched, so
    the caller silently gets unnormalized pixels. Every channel must be
    usable: one dead channel is enough to feed the model raw values in that
    channel while the others are scaled, which is harder to spot than all of
    them being wrong.

    Args:
        channel_stats: per-channel statistics
        strategy: which fields the strategy reads

    Returns:
        True when every channel has a non-zero range for this strategy
    """
    if not channel_stats:
        return False
    for stats in channel_stats:
        if strategy == "percentile_99":
            ok = stats.get("p99", 0.0) > stats.get("p1", 0.0)
        elif strategy == "min_max":
            ok = stats.get("max", 0.0) > stats.get("min", 0.0)
        elif strategy == "z_score":
            ok = stats.get("std", 0.0) > 0.0
        elif strategy == "fixed_range":
            # User-specified bounds, not measured from data.
            ok = stats.get("max", 255.0) > stats.get("min", 0.0)
        else:
            ok = True
        if not ok:
            return False
    return True


def _describe_degenerate(channel_stats: List[Dict[str, float]], strategy: str) -> str:
    """Names the channels whose statistics have no range, for the warning."""
    keys = {
        "percentile_99": ("p1", "p99"),
        "min_max": ("min", "max"),
        "z_score": ("std", "std"),
        "fixed_range": ("min", "max"),
    }.get(strategy, ("p1", "p99"))
    bad = []
    for i, stats in enumerate(channel_stats):
        lo = stats.get(keys[0], 0.0)
        hi = stats.get(keys[1], 0.0)
        if hi <= lo or (strategy == "z_score" and stats.get("std", 0.0) <= 0.0):
            bad.append("channel %d %s=%g %s=%g" % (i, keys[0], lo, keys[1], hi))
    return "; ".join(bad) if bad else "no usable range"


def _normalize_precomputed(
    img: np.ndarray,
    channel_stats: List[Dict[str, float]],
    strategy: str,
    per_channel: bool,
) -> np.ndarray:
    """Normalize using pre-computed image-level statistics.

    Args:
        img: Input image array (HWC or HW)
        channel_stats: List of per-channel stat dicts with keys:
            p1, p99, min, max, mean, std
        strategy: Normalization strategy name
        per_channel: Whether to normalize each channel independently

    Returns:
        Normalized image array
    """
    if per_channel and img.ndim == 3 and img.shape[2] > 1:
        if not img.flags.writeable:
            img = img.copy()
        for c in range(min(img.shape[2], len(channel_stats))):
            stats = channel_stats[c]
            img[..., c] = _normalize_with_stats(img[..., c], stats, strategy)
    else:
        # ONE shared transform across every channel, which is the point of
        # per_channel=False: it preserves the ratios between channels, and
        # for H&E those ratios are the signal.
        #
        # This used to take channel 0's statistics and apply them to all
        # channels. That is not a joint statistic, it is the red channel's,
        # and on a real H&E model red p1 was 97 while green was 43 -- so
        # everything below 97 in green and blue was clipped flat. The
        # per-tile path this mirrors computes its percentile over the whole
        # multi-channel tile, so it pools; this now pools too, and the two
        # agree. Measured on that model, pooling took the disagreement with
        # the training-time normalization from 1.17% of pixels to 0.77%.
        stats = _pool_stats(channel_stats) if channel_stats else {}
        img = _normalize_with_stats(img, stats, strategy)

    return img


def _pool_stats(channel_stats: List[Dict[str, float]]) -> Dict[str, float]:
    """Combines per-channel statistics into one covering every channel.

    Percentiles and extremes widen to bracket all channels. Mean and
    standard deviation are pooled properly -- the pooled variance is the
    mean of the per-channel variances PLUS the variance of their means,
    because channels with different means are themselves a source of
    spread. Averaging the standard deviations would understate it.

    Args:
        channel_stats: one stat dict per channel

    Returns:
        A single stat dict; the input unchanged when there is only one.
    """
    if not channel_stats:
        return {}
    if len(channel_stats) == 1:
        return dict(channel_stats[0])

    def vals(key, default=0.0):
        return [float(s.get(key, default)) for s in channel_stats]

    means = vals("mean")
    stds = vals("std")
    grand_mean = sum(means) / len(means)
    within = sum(v * v for v in stds) / len(stds)
    between = sum((m - grand_mean) ** 2 for m in means) / len(means)
    return {
        "p1": min(vals("p1")),
        "p99": max(vals("p99")),
        "min": min(vals("min")),
        "max": max(vals("max")),
        "mean": grand_mean,
        "std": (within + between) ** 0.5,
    }


def _normalize_with_stats(
    img: np.ndarray, stats: Dict[str, float], strategy: str
) -> np.ndarray:
    """Normalize a single channel/image using pre-computed statistics.

    Args:
        img: Single-channel image or full image array
        stats: Dict with keys: p1, p99, min, max, mean, std
        strategy: Normalization strategy name

    Returns:
        Normalized array
    """
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
            # Rescale to 0-1 for model compatibility
            img = (img + 5) / 10

    # fixed_range uses global fixed values, not precomputed stats
    elif strategy == "fixed_range":
        fixed_min = stats.get("min", 0)
        fixed_max = stats.get("max", 255)
        img = np.clip(img, fixed_min, fixed_max)
        if fixed_max > fixed_min:
            img = (img - fixed_min) / (fixed_max - fixed_min)

    return img


def _normalize_single(
    img: np.ndarray, norm_config: Dict[str, Any], strategy: str
) -> np.ndarray:
    """Normalize a single image or channel using per-tile statistics.

    This is the fallback path when precomputed stats are not available.

    Args:
        img: Image or channel array
        norm_config: Normalization configuration dict
        strategy: Normalization strategy name

    Returns:
        Normalized array
    """
    if strategy == "percentile_99":
        percentile = norm_config.get("clip_percentile", 99.0)
        p_min = np.percentile(img, 100 - percentile)
        p_max = np.percentile(img, percentile)
        img = np.clip(img, p_min, p_max)
        if p_max > p_min:
            img = (img - p_min) / (p_max - p_min)

    elif strategy == "min_max":
        i_min, i_max = img.min(), img.max()
        if i_max > i_min:
            img = (img - i_min) / (i_max - i_min)

    elif strategy == "z_score":
        mean, std = img.mean(), img.std()
        if std > 0:
            img = (img - mean) / std
            img = np.clip(img, -5, 5)
            # Rescale to 0-1 for model compatibility
            img = (img + 5) / 10

    elif strategy == "fixed_range":
        fixed_min = norm_config.get("min", 0)
        fixed_max = norm_config.get("max", 255)
        img = np.clip(img, fixed_min, fixed_max)
        if fixed_max > fixed_min:
            img = (img - fixed_min) / (fixed_max - fixed_min)

    return img


def compute_dataset_stats(
    images: List[np.ndarray], num_channels: int, max_samples_per_channel: int = 500_000
) -> List[Dict[str, float]]:
    """Compute normalization statistics across a collection of images.

    Used during training to compute dataset-level stats that can be saved
    in model metadata for consistent inference normalization.

    Args:
        images: List of image arrays (HWC format, float32)
        num_channels: Number of channels to compute stats for
        max_samples_per_channel: Maximum pixel samples per channel
            (reservoir sampling for memory efficiency)

    Returns:
        List of per-channel stat dicts with keys:
            p1, p99, min, max, mean, std
    """
    if not images:
        return []

    # Collect samples using reservoir sampling
    channel_reservoirs = [[] for _ in range(num_channels)]
    total_pixels = 0

    for img in images:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        h, w = img.shape[:2]
        c = min(img.shape[2], num_channels)
        n_pixels = h * w

        # Subsample rate to keep reservoir bounded
        subsample = max(1, (total_pixels + n_pixels) // max_samples_per_channel)

        flat_indices = np.arange(0, n_pixels, subsample)
        for ch in range(c):
            channel_data = img[..., ch].ravel()
            samples = channel_data[flat_indices]
            channel_reservoirs[ch].append(samples)

        total_pixels += n_pixels

    # Compute stats from collected samples
    channel_stats = []
    for ch in range(num_channels):
        if channel_reservoirs[ch]:
            all_samples = np.concatenate(channel_reservoirs[ch])
            # Trim to max_samples if overflow from multiple images
            if len(all_samples) > max_samples_per_channel:
                rng = np.random.default_rng(42)
                indices = rng.choice(
                    len(all_samples), max_samples_per_channel, replace=False
                )
                all_samples = all_samples[indices]

            stats = {
                "p1": float(np.percentile(all_samples, 1)),
                "p99": float(np.percentile(all_samples, 99)),
                "min": float(np.min(all_samples)),
                "max": float(np.max(all_samples)),
                "mean": float(np.mean(all_samples)),
                "std": float(np.std(all_samples)),
            }
        else:
            stats = {
                "p1": 0.0,
                "p99": 1.0,
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.25,
            }
        channel_stats.append(stats)

    return channel_stats
