"""Training service for deep learning models.

Includes:
- Data augmentation via albumentations
- Learning rate scheduling (cosine annealing, one-cycle, step decay)
- Early stopping with configurable metric (loss or mean IoU)
- Combined CE + Dice loss for improved segmentation quality
- Mixed precision training (AMP) for CUDA devices
- GPU memory monitoring and management
- Support for CUDA, Apple MPS, and CPU training
"""

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, StepLR, OneCycleLR, ReduceLROnPlateau
)
from torch.nn.utils import clip_grad_norm_
from PIL import Image

from .gpu_manager import GPUManager, get_gpu_manager
from ..utils.normalization import normalize as normalize_image
from ..utils.normalization import compute_dataset_stats
from ..utils.batchrenorm import (
    replace_bn_with_batchrenorm, set_batchrenorm_limits
)

logger = logging.getLogger(__name__)

# Try to import albumentations for augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("albumentations not available - augmentation will be disabled")

# Try to import kornia for GPU-side augmentation. Optional; when unavailable
# the training path silently falls back to CPU albumentations.
try:
    import kornia.augmentation as K  # noqa: N814
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


class PerChannelIntensityJitter(A.ImageOnlyTransform):
    """Apply random brightness/contrast independently to each channel.

    Works with arbitrary number of channels (fluorescence, multi-spectral).
    Each channel receives independent random scaling, unlike albumentations'
    built-in transforms which assume RGB correlation.
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2,
                 always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def apply(self, img, **params):
        result = img.copy().astype(np.float32)
        num_ch = img.shape[2] if img.ndim == 3 else 1
        for c in range(num_ch):
            brightness = 1.0 + np.random.uniform(
                -self.brightness_limit, self.brightness_limit)
            contrast = 1.0 + np.random.uniform(
                -self.contrast_limit, self.contrast_limit)
            if img.ndim == 3:
                ch = result[:, :, c]
            else:
                ch = result
            mean_val = ch.mean()
            ch[:] = np.clip(
                (ch - mean_val) * contrast + mean_val * brightness,
                0, 255 if img.dtype == np.uint8 else ch.max() * 2)
        return result.astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit")


def get_training_augmentation(
    image_size: int = 512,
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    p_elastic: float = 0.3,
    p_color: float = 0.3,
    p_noise: float = 0.2,
    intensity_mode: str = "none",
    # Advanced strength parameters (defaults preserve legacy behavior)
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    gamma_min: int = 80,
    gamma_max: int = 120,
    elastic_alpha: float = 120.0,
    elastic_sigma_ratio: float = 0.05,
    noise_std_min: float = 0.04,
    noise_std_max: float = 0.2,
) -> Optional[A.Compose]:
    """Create training augmentation pipeline.

    Args:
        image_size: Expected image size (for padding/cropping)
        p_flip: Probability of flip transforms
        p_rotate: Probability of rotation
        p_elastic: Probability of elastic deformation
        p_color: Probability of color jitter (used when intensity_mode != "none")
        p_noise: Probability of noise addition
        intensity_mode: "none", "brightfield", or "fluorescence"
        brightness_limit: Max brightness adjustment as fraction (0.0 - 0.5)
        contrast_limit: Max contrast adjustment as fraction (0.0 - 0.5)
        gamma_min: Minimum gamma percent for brightfield mode
        gamma_max: Maximum gamma percent for brightfield mode
        elastic_alpha: Elastic deformation magnitude
        elastic_sigma_ratio: Smoothness of elastic deformation as fraction of alpha
        noise_std_min: Minimum Gaussian noise std (fraction of image max)
        noise_std_max: Maximum Gaussian noise std (fraction of image max)

    Returns:
        Albumentations Compose object or None if not available
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    # Ensure noise std range is valid (min <= max, both in [0, 1])
    noise_lo = max(0.0, min(noise_std_min, noise_std_max))
    noise_hi = max(noise_lo, noise_std_max)

    # Ensure gamma range is valid (min <= max)
    gamma_lo = min(gamma_min, gamma_max)
    gamma_hi = max(gamma_min, gamma_max)

    transforms = [
        # Spatial transforms (applied to both image and mask)
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip),
        A.RandomRotate90(p=p_rotate),

        # Rotation with interpolation (fills with reflection)
        A.Rotate(
            limit=45,
            interpolation=1,  # INTER_LINEAR
            border_mode=2,    # BORDER_REFLECT
            p=p_rotate * 0.5
        ),

        # Elastic deformation - good for biological tissue
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_alpha * elastic_sigma_ratio,
            p=p_elastic
        ),

        # Grid distortion - another spatial augmentation
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=p_elastic * 0.5
        ),
    ]

    # Intensity transforms -- mode-dependent
    if intensity_mode == "brightfield":
        # RGB-correlated brightness/contrast/gamma (standard for H&E)
        transforms.append(A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(gamma_lo, gamma_hi), p=1.0),
        ], p=p_color))
    elif intensity_mode == "fluorescence":
        # Per-channel independent intensity jitter (for fluorescence/multi-spectral)
        transforms.append(PerChannelIntensityJitter(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p_color))

    transforms.extend([
        # Blur - simulates slight defocus
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.1),

        # Noise - std_range is fraction of image max value
        A.GaussNoise(std_range=(noise_lo, noise_hi), p=p_noise),
    ])

    return A.Compose(transforms, additional_targets={})


def get_validation_transform() -> Optional[A.Compose]:
    """Create validation transform (no augmentation, just normalization)."""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    # No transforms for validation - just return as-is
    return None


def build_gpu_augmentation(aug_config: Optional[Dict[str, Any]]):
    """Build a kornia.AugmentationSequential pipeline for GPU-side augmentation.

    Mirrors a subset of the CPU albumentations pipeline but runs on tensors
    already on the GPU, which is dramatically faster once the per-batch
    CPU-decode cost is eliminated by the in-memory dataset preload.

    Supported (mask-consistent when geometric):
    - HorizontalFlip / VerticalFlip
    - 90-degree rotation (RandomRotation with multiples of 90)
    - Color jitter (brightfield mode)
    - Gaussian noise and blur at low probability

    Deferred to CPU albumentations (too costly or non-trivial on GPU for the
    marginal benefit):
    - ElasticTransform, GridDistortion, arbitrary-angle Rotate

    Args:
        aug_config: augmentation config dict (same one passed to
            get_training_augmentation) -- keys like ``p_flip``, ``p_rotate``,
            ``p_color``, ``intensity_mode``, etc.

    Returns:
        A kornia AugmentationSequential instance with ``data_keys=["input",
        "mask"]`` so geometric ops apply consistently to image and mask.
        Returns ``None`` if kornia is unavailable or aug_config is empty.
    """
    if not KORNIA_AVAILABLE:
        return None
    if not aug_config:
        return None

    cfg = dict(aug_config)  # shallow copy
    p_flip = float(cfg.get("p_flip", 0.5))
    p_rotate = float(cfg.get("p_rotate", 0.5))
    p_color = float(cfg.get("p_color", 0.3))
    p_noise = float(cfg.get("p_noise", 0.2))
    brightness_limit = float(cfg.get("brightness_limit", 0.2))
    contrast_limit = float(cfg.get("contrast_limit", 0.2))
    noise_std_min = float(cfg.get("noise_std_min", 0.04))
    noise_std_max = float(cfg.get("noise_std_max", 0.2))
    intensity_mode = cfg.get("intensity_mode", "none")

    # same_on_batch=False applies a different sample from each transform to
    # each element in the batch -- the usual correct behaviour for training.
    augs = [
        K.RandomHorizontalFlip(p=p_flip),
        K.RandomVerticalFlip(p=p_flip),
        # 90-degree rotation -- pick uniformly from {0, 90, 180, 270} per sample.
        K.RandomRotation(degrees=[0.0, 270.0], p=p_rotate,
                         resample="nearest", align_corners=False),
    ]

    # Intensity augmentation (image only -- kornia handles the mask skip).
    # Brightfield: RGB-correlated jitter (ColorJitter). Fluorescence and
    # multi-channel are not cleanly expressible via ColorJitter (it is
    # defined for 3-channel RGB), so fall back to per-channel random
    # brightness+contrast via RandomBrightness + RandomContrast which both
    # operate on arbitrary channel counts.
    if intensity_mode == "brightfield":
        augs.append(K.ColorJitter(
            brightness=brightness_limit,
            contrast=contrast_limit,
            p=p_color,
        ))
    else:
        # Works for 1-N channel inputs; brightness/contrast independently.
        augs.append(K.RandomBrightness(
            brightness=(max(0.0, 1.0 - brightness_limit),
                        1.0 + brightness_limit),
            p=p_color,
        ))
        augs.append(K.RandomContrast(
            contrast=(max(0.0, 1.0 - contrast_limit),
                      1.0 + contrast_limit),
            p=p_color,
        ))

    # Low-probability blur and noise -- cheap regularization.
    augs.append(K.RandomGaussianBlur(
        kernel_size=(3, 3), sigma=(0.1, 1.5), p=0.1,
    ))
    augs.append(K.RandomGaussianNoise(
        mean=0.0, std=noise_std_max, p=p_noise,
    ))
    # noise_std_min is informational for the CPU path (range); kornia's
    # RandomGaussianNoise uses a single std, so we use the upper bound.

    return K.AugmentationSequential(
        *augs,
        data_keys=["input", "mask"],
        same_on_batch=False,
    )


class EarlyStopping:
    """Early stopping to stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore best model weights when stopping
        mode: "min" if lower metric is better (e.g. loss),
              "max" if higher metric is better (e.g. IoU)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.should_stop = False

        if mode == "min":
            self._is_better = lambda current, best: current < best - self.min_delta
        else:
            self._is_better = lambda current, best: current > best + self.min_delta

    def __call__(self, epoch: int, metric_value: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            epoch: Current epoch number
            metric_value: Current value of the monitored metric
            model: The model being trained

        Returns:
            True if training should stop, False otherwise
        """
        if self._is_better(metric_value, self.best_score):
            # Improvement found
            self.best_score = metric_value
            self.best_epoch = epoch
            self.counter = 0

            if self.restore_best_weights:
                # Save best model state
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            logger.debug(f"Early stopping: new best {metric_value:.4f} at epoch {epoch}")
            return False
        else:
            # No improvement
            self.counter += 1
            logger.debug(f"Early stopping: no improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}. "
                           f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True

        return False

    def restore_best(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_state is not None and self.restore_best_weights:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored best model weights from epoch {self.best_epoch}")


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation with ignore_index support.

    Computes per-class Dice loss and averages across classes.
    """

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            inputs: Model logits of shape (N, C, H, W)
            targets: Ground truth labels of shape (N, H, W)

        Returns:
            Scalar Dice loss (1 - mean Dice coefficient)
        """
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)

        # Create valid pixel mask
        valid_mask = (targets != self.ignore_index)

        # One-hot encode targets (only valid pixels)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        targets_one_hot = F.one_hot(targets_safe, num_classes).permute(0, 3, 1, 2).float()

        # Zero out invalid pixels
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded

        # Per-class Dice
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


class CombinedCEDiceLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss for segmentation.

    The CE component handles per-pixel classification while the Dice component
    optimizes region overlap directly, making this combination the modern
    standard for segmentation tasks.

    Args:
        class_weights: Optional per-class weights for the CE component
        ignore_index: Label index to ignore (default 255)
        ce_weight: Weight for Cross-Entropy component (default 0.5)
        dice_weight: Weight for Dice component (default 0.5)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clamp(-50.0, 50.0)
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """Focal loss for segmentation -- down-weights easy pixels.

    Modulates per-pixel CE by (1 - p_t)^gamma so that well-classified
    pixels contribute less gradient and hard pixels contribute more.
    When gamma=0 this reduces to standard cross-entropy.

    Args:
        gamma: Focusing parameter (0 = CE, 2.0 = standard focal)
        class_weights: Optional per-class weights (alpha)
        ignore_index: Label index to ignore (default 255)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model logits of shape (N, C, H, W)
            targets: Ground truth labels of shape (N, H, W)

        Returns:
            Scalar focal loss averaged over valid pixels
        """
        # Per-pixel CE (unreduced)
        ce = F.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Compute p_t (probability of the true class)
        probs = F.softmax(inputs, dim=1)  # (N, C, H, W)
        # Clamp targets so gather doesn't fail on ignore_index
        targets_safe = targets.clone()
        valid_mask = targets != self.ignore_index
        targets_safe[~valid_mask] = 0
        p_t = probs.gather(1, targets_safe.unsqueeze(1)).squeeze(1)  # (N, H, W)

        # Focal modulation
        focal_weight = (1.0 - p_t) ** self.gamma

        # Per-class alpha weighting
        if self.class_weights is not None:
            alpha = self.class_weights[targets_safe]
            focal_weight = focal_weight * alpha

        loss = focal_weight * ce

        # Average over valid pixels only
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return loss.sum() * 0.0  # no valid pixels
        return loss[valid_mask].mean()


class OHEMCrossEntropyLoss(nn.Module):
    """Online Hard Example Mining -- keeps only the hardest K% of pixels.

    More aggressive than focal loss: completely ignores easy pixels
    instead of down-weighting them. When hard_ratio=1.0 this is
    equivalent to standard cross-entropy.

    Two selection strategies are supported:
      * adaptive_floor=False (default, legacy): for each class, keep the
        hardest (hard_ratio * class_pixel_count) pixels.  The resulting
        mixture matches each class's natural frequency in the batch, so
        majority classes dominate when class imbalance is extreme.
      * adaptive_floor=True: global topk over all valid pixels (hard pixels
        naturally flow to the hardest class), then top up any present class
        that fell below a per-class floor.  The floor is
        min(class_count, max(min_pixels_per_class, k // (num_classes *
        floor_divisor))), so a class can never contribute fewer than that
        but also never more than it actually has.

    Args:
        hard_ratio: Fraction of pixels to keep (0.05-1.0)
        class_weights: Optional per-class weights
        ignore_index: Label index to ignore (default 255)
        adaptive_floor: If True, use the global-topk + per-class-floor
            strategy described above.  Default False preserves the original
            per-class proportional behavior for backwards compatibility.
        min_pixels_per_class: Absolute lower bound for the per-class floor
            in adaptive_floor mode.
        floor_divisor: Controls how large the per-class floor is relative
            to equal allocation.  floor = k // (num_classes * floor_divisor),
            so divisor=1 equalises classes, divisor=4 barely floors anything.
            Default 2 = a quarter of equal-slot allocation, leaving most of
            the budget for the hardest pixels.
    """

    def __init__(
        self,
        hard_ratio: float = 0.25,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        adaptive_floor: bool = False,
        min_pixels_per_class: int = 32,
        floor_divisor: int = 2,
    ):
        super().__init__()
        self.hard_ratio = hard_ratio
        self.ignore_index = ignore_index
        self.adaptive_floor = adaptive_floor
        self.min_pixels_per_class = int(min_pixels_per_class)
        self.floor_divisor = max(1, int(floor_divisor))
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index, reduction="none"
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute OHEM loss with per-class minimum representation.

        Args:
            inputs: Model logits of shape (N, C, H, W)
            targets: Ground truth labels of shape (N, H, W)

        Returns:
            Scalar loss averaged over the selected hard pixels
        """
        per_pixel = self.ce(inputs, targets)  # (N, H, W)

        # Flatten and filter to valid pixels
        valid_mask = targets != self.ignore_index
        valid_losses = per_pixel[valid_mask]
        valid_targets = targets[valid_mask]

        if valid_losses.numel() == 0:
            return per_pixel.sum() * 0.0

        # When ratio is 1.0, keep everything (standard CE)
        if self.hard_ratio >= 1.0:
            return valid_losses.mean()

        k = max(1, int(valid_losses.numel() * self.hard_ratio))
        num_classes = inputs.shape[1]
        selected_mask = torch.zeros_like(valid_losses, dtype=torch.bool)

        if self.adaptive_floor:
            # Step 1: global topk -- hard pixels naturally concentrate on
            # whichever class currently has the worst predictions.
            actual_k = min(k, valid_losses.numel())
            _, global_top = torch.topk(valid_losses, actual_k)
            selected_mask[global_top] = True

            # Step 2: per-class safety floor so no present class drops to
            # zero (or near-zero) pixels in the gradient.
            floor_base = max(
                self.min_pixels_per_class,
                k // max(1, num_classes * self.floor_divisor),
            )
            for c in range(num_classes):
                class_mask = valid_targets == c
                class_count = class_mask.sum().item()
                if class_count == 0:
                    continue
                floor = min(class_count, floor_base)
                current = (class_mask & selected_mask).sum().item()
                if current >= floor:
                    continue
                needed = floor - current
                class_indices = torch.where(class_mask)[0]
                already = selected_mask[class_indices]
                candidate_losses = valid_losses[class_indices].clone()
                candidate_losses[already] = -1.0  # exclude already-selected
                _, top_indices = torch.topk(
                    candidate_losses, min(needed, class_count - current)
                )
                selected_mask[class_indices[top_indices]] = True
        else:
            # Legacy per-class proportional: each class keeps its own
            # hardest (hard_ratio * class_count) pixels, then fill up to
            # the global k with the hardest remaining pixels.
            for c in range(num_classes):
                class_mask = valid_targets == c
                class_count = class_mask.sum().item()
                if class_count == 0:
                    continue
                class_losses = valid_losses[class_mask]
                class_k = max(1, int(class_count * self.hard_ratio))
                if class_k >= class_count:
                    selected_mask[class_mask] = True
                else:
                    _, top_indices = torch.topk(class_losses, class_k)
                    class_indices = torch.where(class_mask)[0]
                    selected_mask[class_indices[top_indices]] = True

            selected_count = selected_mask.sum().item()
            if selected_count < k:
                remaining = valid_losses.clone()
                remaining[selected_mask] = -1.0  # exclude already-selected
                fill_k = k - int(selected_count)
                _, fill_indices = torch.topk(
                    remaining, min(fill_k, remaining.numel())
                )
                selected_mask[fill_indices] = True

        return valid_losses[selected_mask].mean()


class _CombinedPixelDiceLoss(nn.Module):
    """Generic combiner for any pixel-level loss + Dice loss.

    OHEM/focal apply to the pixel-loss component only; Dice stays as-is
    since it operates on region overlap and is not pixel-selectable.

    Logits are clamped to [-50, 50] to prevent extreme per-pixel losses
    when a pretrained model produces very confident wrong predictions
    (common during continue-training). Normal training logits (~5-15)
    are unaffected by this clamp.

    Args:
        pixel_loss: Any pixel-level loss module (CE, Focal, OHEM)
        dice_loss: DiceLoss instance
        pixel_weight: Weight for pixel-loss component (default 0.5)
        dice_weight: Weight for Dice component (default 0.5)
    """

    # Max absolute logit value. With 3 classes, this bounds per-pixel CE at
    # ~50 before class weighting. Empirically: scale=10 logits produce CE~7,
    # scale=200 produce CE~180 unclamped vs ~39 clamped. Normal training
    # logits (scale 5-15) are unaffected.
    LOGIT_CLAMP = 50.0

    def __init__(
        self,
        pixel_loss: nn.Module,
        dice_loss: DiceLoss,
        pixel_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.dice_loss = dice_loss
        self.pixel_weight = pixel_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clamp(-self.LOGIT_CLAMP, self.LOGIT_CLAMP)
        px = self.pixel_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.pixel_weight * px + self.dice_weight * dice


class SegmentationDataset(Dataset):
    """Dataset for segmentation training with augmentation support."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        input_config: Dict[str, Any],
        augment: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        context_dir: Optional[str] = None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_config = input_config
        self.augment = augment and ALBUMENTATIONS_AVAILABLE
        self.context_dir = Path(context_dir) if context_dir else None

        # Find all images (including raw float32 files from N-channel export)
        self.image_files = sorted(list(self.images_dir.glob("*.tiff")) +
                                  list(self.images_dir.glob("*.tif")) +
                                  list(self.images_dir.glob("*.png")) +
                                  list(self.images_dir.glob("*.raw")))
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")
        if self.context_dir:
            logger.info(f"Multi-scale context enabled from {context_dir}")

        # Setup augmentation
        if self.augment:
            aug_config = augmentation_config or {}
            intensity_mode = aug_config.get("intensity_mode", "none")
            # Set color probability based on intensity mode
            p_color = 0.3 if intensity_mode != "none" else 0.0
            self.transform = get_training_augmentation(
                image_size=aug_config.get("image_size", 512),
                p_flip=aug_config.get("p_flip", 0.5),
                p_rotate=aug_config.get("p_rotate", 0.5),
                p_elastic=aug_config.get("p_elastic", 0.3),
                p_color=aug_config.get("p_color", p_color),
                p_noise=aug_config.get("p_noise", 0.2),
                intensity_mode=intensity_mode,
                brightness_limit=aug_config.get("brightness_limit", 0.2),
                contrast_limit=aug_config.get("contrast_limit", 0.2),
                gamma_min=aug_config.get("gamma_min", 80),
                gamma_max=aug_config.get("gamma_max", 120),
                elastic_alpha=aug_config.get("elastic_alpha", 120.0),
                elastic_sigma_ratio=aug_config.get("elastic_sigma_ratio", 0.05),
                noise_std_min=aug_config.get("noise_std_min", 0.04),
                noise_std_max=aug_config.get("noise_std_max", 0.2),
            )
            logger.info(
                f"Augmentation enabled (intensity_mode={intensity_mode}, "
                f"p_flip={aug_config.get('p_flip', 0.5)}, "
                f"p_rotate={aug_config.get('p_rotate', 0.5)}, "
                f"p_elastic={aug_config.get('p_elastic', 0.3)}, "
                f"p_color={aug_config.get('p_color', p_color)}, "
                f"p_noise={aug_config.get('p_noise', 0.2)})"
            )
        else:
            self.transform = None
            if augment and not ALBUMENTATIONS_AVAILABLE:
                logger.warning("Augmentation requested but albumentations not available")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load detail image (supports TIFF, PNG, and raw float32 files)
        img_path = self.image_files[idx]
        img_array = self._load_patch(img_path)

        # Handle channels
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]

        # Load and concatenate context tile when multi-scale is enabled
        if self.context_dir is not None:
            ctx_path = self.context_dir / img_path.name
            if ctx_path.exists():
                ctx_array = self._load_patch(ctx_path)
                if ctx_array.ndim == 2:
                    ctx_array = ctx_array[..., np.newaxis]
                # Resize context tile if spatial dims don't match (edge tiles near image boundary)
                if ctx_array.shape[0] != img_array.shape[0] or ctx_array.shape[1] != img_array.shape[1]:
                    from PIL import Image as _PILResize
                    h, w = img_array.shape[:2]
                    resized = []
                    for c in range(ctx_array.shape[2]):
                        ch = _PILResize.fromarray(ctx_array[:, :, c])
                        ch = ch.resize((w, h), _PILResize.BILINEAR)
                        resized.append(np.array(ch))
                    ctx_array = np.stack(resized, axis=2)
                # Concatenate detail + context along channel axis: (H,W,C) + (H,W,C) -> (H,W,2C)
                img_array = np.concatenate([img_array, ctx_array], axis=2)
            else:
                logger.warning(f"Context tile not found: {ctx_path}, duplicating detail tile")
                img_array = np.concatenate([img_array, img_array], axis=2)

        # Normalize BEFORE augmentation
        img_array = self._normalize(img_array)

        # Load mask
        mask_name = img_path.stem + ".png"
        mask_path = self.masks_dir / mask_name
        if mask_path.exists():
            mask = Image.open(mask_path)
            mask_array = np.array(mask, dtype=np.int64)
        else:
            # No mask - create empty
            mask_array = np.zeros(img_array.shape[:2], dtype=np.int64)

        # Apply augmentation
        if self.transform is not None:
            # Albumentations expects uint8 or float32 in [0, 1] for image
            # and int for mask
            transformed = self.transform(image=img_array, mask=mask_array)
            img_array = transformed["image"]
            mask_array = transformed["mask"]

        # Convert to tensors (HWC -> CHW for image)
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1).astype(np.float32))
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64))

        return img_tensor, mask_tensor

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image data.

        Delegates to shared normalization module which supports both
        per-tile and precomputed image-level statistics.
        """
        return normalize_image(img, self.input_config)

    @staticmethod
    def _load_patch(img_path: Path) -> np.ndarray:
        """Load a training patch from file.

        Supports:
        - .raw: N-channel float32 (12-byte header: H,W,C as int32 + float32 data)
        - .tif/.tiff: Multi-channel TIFF via tifffile (falls back to PIL)
        - .png and others: Standard formats via PIL
        """
        suffix = img_path.suffix.lower()
        if suffix == '.raw':
            with open(img_path, 'rb') as f:
                header = np.frombuffer(f.read(12), dtype=np.int32)
                h, w, c = int(header[0]), int(header[1]), int(header[2])
                data = np.frombuffer(f.read(), dtype=np.float32)
            return data.reshape(h, w, c).copy()
        if suffix in ('.tif', '.tiff'):
            try:
                import tifffile
                arr = tifffile.imread(str(img_path)).astype(np.float32)
                # tifffile may return (C,H,W) for multi-channel; convert to HWC
                if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
                    arr = arr.transpose(1, 2, 0)
                return arr
            except ImportError:
                pass  # fall through to PIL
        img = Image.open(img_path)
        return np.array(img, dtype=np.float32)


def _validate_mae_architecture(pretrained_path, architecture, log):
    """Check MAE metadata architecture against segmentation config.

    Warns on mismatches but does not raise -- mismatched configs may still
    partially transfer encoder weights.
    """
    meta_path = Path(pretrained_path).parent / "metadata.json"
    if not meta_path.exists():
        log.info("No metadata.json next to pretrained model -- "
                 "skipping architecture validation")
        return
    try:
        with open(meta_path) as f:
            mae_meta = json.load(f)
    except Exception as e:
        log.warning("Could not read pretrained metadata.json: %s", e)
        return
    mae_arch = mae_meta.get("architecture", {})
    # Compare model_config
    mae_cfg = mae_arch.get("model_config", "")
    seg_cfg = architecture.get("model_config",
                               architecture.get("backbone", ""))
    if mae_cfg and seg_cfg and mae_cfg != seg_cfg:
        log.warning("MAE pretrained with model_config='%s' but "
                    "segmentation uses '%s'. Layer counts differ -- "
                    "encoder weights may not transfer.",
                    mae_cfg, seg_cfg)
    # Compare patch_size
    mae_ps = mae_arch.get("patch_size")
    seg_ps = architecture.get("patch_size")
    if mae_ps and seg_ps and int(mae_ps) != int(seg_ps):
        log.warning("MAE pretrained with patch_size=%d but "
                    "segmentation uses %d.", int(mae_ps), int(seg_ps))
    # Compare level_scales
    mae_ls = str(mae_arch.get("level_scales", ""))
    seg_ls = str(architecture.get("level_scales", ""))
    if mae_ls and seg_ls and mae_ls != seg_ls:
        log.warning("MAE pretrained with level_scales='%s' but "
                    "segmentation uses '%s'.", mae_ls, seg_ls)
    # Compare input_channels
    mae_ch = mae_arch.get("input_channels")
    seg_ch = architecture.get("input_channels",
                              architecture.get("num_channels"))
    if mae_ch and seg_ch and int(mae_ch) != int(seg_ch):
        log.warning("MAE pretrained with %d channels but "
                    "segmentation uses %d.", int(mae_ch), int(seg_ch))
    # Compare rope_mode
    mae_rope = mae_arch.get("rope_mode")
    seg_rope = architecture.get("rope_mode")
    if mae_rope and seg_rope and mae_rope != seg_rope:
        log.warning("MAE pretrained with rope_mode='%s' but "
                    "segmentation uses '%s'.", mae_rope, seg_rope)


class TrainingService:
    """Service for training deep learning models.

    Features:
    - Multiple model architectures via segmentation-models-pytorch
    - Data augmentation via albumentations
    - Learning rate scheduling (cosine annealing, step decay, one-cycle)
    - Early stopping with patience
    - Transfer learning with layer freezing
    - Class weighting for imbalanced datasets
    - GPU memory monitoring and management
    - Support for CUDA, Apple MPS, and CPU
    """

    def __init__(
        self,
        device: str = "auto",
        gpu_manager: Optional[GPUManager] = None
    ):
        """Initialize training service.

        Args:
            device: Device to use ("cuda", "mps", "cpu", or "auto")
            gpu_manager: Optional GPUManager instance (uses singleton if not provided)
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()

        if device == "auto":
            self.device = self.gpu_manager.device_type
        else:
            self.device = device

        logger.info(f"TrainingService initialized on device: {self.device}")

    def train(
        self,
        model_type: str,
        architecture: Dict[str, Any],
        input_config: Dict[str, Any],
        training_params: Dict[str, Any],
        classes: List[str],
        data_path: str,
        progress_callback: Optional[Callable] = None,
        cancel_flag: Optional[threading.Event] = None,
        frozen_layers: Optional[List[str]] = None,
        pause_flag: Optional[threading.Event] = None,
        checkpoint_path: Optional[str] = None,
        start_epoch: int = 0,
        setup_callback: Optional[Callable] = None,
        pretrained_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train a model.

        Args:
            model_type: Type of model architecture (e.g., "unet", "deeplabv3plus")
            architecture: Architecture configuration dict
            input_config: Input configuration (channels, normalization)
            training_params: Training hyperparameters
            classes: List of class names
            data_path: Path to training data
            progress_callback: Optional callback for progress updates.
                Signature: (epoch, train_loss, val_loss, accuracy,
                            per_class_iou, per_class_loss, mean_iou)
            cancel_flag: Optional threading event for cancellation
            frozen_layers: Optional list of layer names to freeze for transfer learning
            pause_flag: Optional threading event for pause requests
            checkpoint_path: Optional path to checkpoint for resuming training
            start_epoch: Epoch to start from when resuming (0-based)
            setup_callback: Optional callback for setup phase status.
                Signature: (phase_name: str) where phase_name is one of:
                "creating_model", "loading_data", "computing_stats",
                "configuring_optimizer", "loading_checkpoint", "starting_training"
            pretrained_model_path: Optional path to a previously trained model's
                .pt file for weight initialization (fine-tuning). Only network
                weights are loaded; optimizer/scheduler/early stopping start fresh.
                Ignored when checkpoint_path is set (checkpoint takes precedence).

        Training params can include:
            - epochs: Number of training epochs (default: 50)
            - batch_size: Batch size (default: 8)
            - learning_rate: Initial learning rate (default: 0.0001)
            - weight_decay: L2 regularization (default: 1e-4)
            - augmentation: Enable data augmentation (default: True)
            - scheduler: Learning rate scheduler type ("cosine", "step", "onecycle", "none")
            - scheduler_config: Scheduler-specific parameters
            - early_stopping: Enable early stopping (default: True)
            - early_stopping_patience: Epochs to wait (default: 10)
            - early_stopping_min_delta: Minimum improvement (default: 0.001)
        """
        logger.info(f"Starting training: {model_type}")

        try:
            return self._run_training(
                model_type=model_type,
                architecture=architecture,
                input_config=input_config,
                training_params=training_params,
                classes=classes,
                data_path=data_path,
                progress_callback=progress_callback,
                cancel_flag=cancel_flag,
                frozen_layers=frozen_layers,
                pause_flag=pause_flag,
                checkpoint_path=checkpoint_path,
                start_epoch=start_epoch,
                setup_callback=setup_callback,
                pretrained_model_path=pretrained_model_path
            )
        except BaseException as exc:
            # On CUDA OOM, the exception's traceback keeps _run_training()'s
            # frame alive -- and that frame holds model, optimizer, data
            # loaders, etc. on the GPU (45+ GB).  Strip the traceback to
            # release those references so gc.collect() can free them.
            # For non-OOM errors, just do normal cleanup and re-raise.
            if "OutOfMemory" in type(exc).__name__:
                import traceback as _tb
                error_text = _tb.format_exc()
                exc.__traceback__ = None
                self._cleanup_training_memory()
                raise RuntimeError(error_text) from None
            raise
        finally:
            self._cleanup_training_memory()

    def _cleanup_training_memory(self) -> None:
        """Free GPU memory after training completes or fails.

        On the normal path, _run_training()'s locals are already released.
        On the error path, the traceback keeps the frame alive, but we still
        call gc.collect() + empty_cache() here to free any unreferenced
        cache. The heavy lifting (deleting model, optimizer, etc.) is done
        inside _run_training()'s own finally block because only that frame
        can delete its own local variables.
        """
        import gc
        gc.collect()
        self.gpu_manager.clear_cache()
        self.gpu_manager.log_memory_status(prefix="Training cleanup: ")

    def _run_training(
        self,
        model_type: str,
        architecture: Dict[str, Any],
        input_config: Dict[str, Any],
        training_params: Dict[str, Any],
        classes: List[str],
        data_path: str,
        progress_callback: Optional[Callable] = None,
        cancel_flag: Optional[threading.Event] = None,
        frozen_layers: Optional[List[str]] = None,
        pause_flag: Optional[threading.Event] = None,
        checkpoint_path: Optional[str] = None,
        start_epoch: int = 0,
        setup_callback: Optional[Callable] = None,
        pretrained_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal training implementation. Called by train() with cleanup guarantee."""

        # Reproducibility seed
        seed = training_params.get("seed", None)
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"Reproducibility seed set: {seed}")

        def _report_setup(phase, data=None):
            if setup_callback:
                try:
                    setup_callback(phase, data)
                except Exception:
                    pass  # Never let status reporting break training

        # Compute effective input channels: doubled when context_scale > 1
        # (detail + context tiles are concatenated along the channel axis)
        context_scale = architecture.get("context_scale", 1)
        base_channels = input_config["num_channels"]
        effective_channels = base_channels * 2 if context_scale > 1 else base_channels
        if effective_channels != base_channels:
            logger.info(f"Context scale {context_scale}: model input channels "
                        f"{base_channels} -> {effective_channels} (detail + context)")

        # Log context padding if present (real-data border around training tiles)
        context_padding = input_config.get("context_padding", 0)
        if context_padding > 0:
            patch_size = architecture.get("input_size", [512, 512])[0]
            logger.info("Context padding: %dpx per side (effective tile: %dx%d)",
                        context_padding,
                        patch_size + 2 * context_padding,
                        patch_size + 2 * context_padding)

        # Create model with optional frozen layers
        _report_setup("creating_model")
        if frozen_layers:
            from .pretrained_models import get_pretrained_service
            pretrained_service = get_pretrained_service()

            model = pretrained_service.create_model_with_frozen_layers(
                architecture=model_type,
                encoder=architecture.get("backbone", "resnet34"),
                num_channels=effective_channels,
                num_classes=len(classes),
                frozen_layers=frozen_layers
            )
            logger.info(f"Created model with {len(frozen_layers)} frozen layer groups")
        else:
            model = self._create_model(
                model_type=model_type,
                architecture=architecture,
                num_channels=effective_channels,
                num_classes=len(classes)
            )

        # Replace BatchNorm with BatchRenorm to eliminate tiling artifacts
        # during sliding-window inference (see arXiv:2503.19545).
        # Skip for MuViT which uses LayerNorm (no BatchNorm to replace).
        if model_type != "muvit":
            replace_bn_with_batchrenorm(model)
            set_batchrenorm_limits(model, rmax=1.0, dmax=0.0)

        model = model.to(self.device)

        # torch.compile (opt-in, experimental).  Gated on Linux + CUDA +
        # Triton availability; silently falls back to eager on any failure.
        # BRN's in-place buffer updates are the known weak spot for compile
        # graph-breaks (agent B2); tiny-unet with norm="gn" is the safest
        # target. Users enable this explicitly via the "Experimental:
        # torch.compile" checkbox in the Training dialog.
        if training_params.get("use_torch_compile", False):
            model = self._try_training_compile(model, model_type, architecture)

        # Load pretrained weights from a previously trained model (fine-tuning).
        # Only loads network weights -- optimizer/scheduler start fresh.
        # Skipped when checkpoint_path is set (checkpoint loading restores everything).
        if pretrained_model_path and not checkpoint_path:
            _report_setup("loading_pretrained_weights")
            try:
                pt_path = Path(pretrained_model_path)
                if not pt_path.exists():
                    logger.warning("Pretrained model not found: %s -- "
                                   "training from scratch.",
                                   pretrained_model_path)
                else:
                    logger.info("Loading pretrained weights from: %s",
                                pretrained_model_path)
                    saved = torch.load(pretrained_model_path,
                                       map_location=self.device,
                                       weights_only=True)

                    # Handle both bare state_dict and
                    # {"model_state_dict": ...} checkpoint format
                    if isinstance(saved, dict) and "model_state_dict" in saved:
                        state_dict = saved["model_state_dict"]
                    else:
                        state_dict = saved

                    # Detect MAE checkpoint and strip "mae." prefix so that
                    # encoder keys (mae.encoder.* -> encoder.*) match the
                    # MuViTSegmentation model's state_dict.
                    mae_prefix = "mae."
                    has_mae_keys = any(
                        k.startswith(mae_prefix) for k in state_dict)
                    if has_mae_keys:
                        logger.info(
                            "Detected MAE checkpoint -- stripping 'mae.' "
                            "prefix for encoder weight transfer.")
                        state_dict = {
                            (k[len(mae_prefix):]
                             if k.startswith(mae_prefix) else k): v
                            for k, v in state_dict.items()
                        }
                        _validate_mae_architecture(
                            pretrained_model_path, architecture, logger)

                    # Shape-aware key matching
                    model_state = model.state_dict()
                    matched_keys = []
                    mismatched_keys = []
                    for key in state_dict:
                        if key in model_state:
                            if state_dict[key].shape == model_state[key].shape:
                                matched_keys.append(key)
                            else:
                                mismatched_keys.append(key)
                                logger.warning(
                                    "  Shape mismatch '%s': "
                                    "pretrained=%s vs model=%s -- skipping",
                                    key, list(state_dict[key].shape),
                                    list(model_state[key].shape))

                    # Build filtered state dict with only shape-compatible keys
                    filtered_state = {k: state_dict[k] for k in matched_keys}
                    model.load_state_dict(filtered_state, strict=False)

                    # Diagnostics
                    enc_loaded = sum(
                        1 for k in matched_keys if k.startswith("encoder"))
                    logger.info(
                        "Loaded %d/%d weight tensors (%d encoder keys)",
                        len(matched_keys), len(model_state), enc_loaded)
                    if mismatched_keys:
                        logger.info("  Skipped %d shape-mismatched keys",
                                    len(mismatched_keys))

                    # Critical warning if encoder transfer failed
                    if enc_loaded == 0:
                        logger.warning(
                            "*** 0 encoder keys loaded from pretrained "
                            "model! ***  The pretrained weights were NOT "
                            "applied to the encoder. Check that architecture "
                            "configs match. Training will proceed from "
                            "random initialization.")
            except Exception as e:
                logger.warning("Failed to load pretrained weights: %s -- "
                               "training from scratch", e)

        # Create datasets
        _report_setup("loading_data")
        data_path = Path(data_path)
        augmentation_config = training_params.get("augmentation_config", {})

        # GPU-side augmentation via kornia.  Opt-in through
        # training_params["gpu_augmentation"], auto-disabled when kornia is
        # unavailable or when the device is not CUDA.  When active, CPU-side
        # albumentations is skipped in the dataset to avoid double-augment.
        # self.device is a STRING in TrainingService (set in __init__), unlike
        # InferenceService which uses a torch.device instance.
        use_gpu_aug = bool(training_params.get("gpu_augmentation", False))
        device_str = self.device if isinstance(self.device, str) else self.device.type
        gpu_augment_on_cuda = (
            use_gpu_aug
            and KORNIA_AVAILABLE
            and device_str == "cuda"
        )
        if use_gpu_aug and not KORNIA_AVAILABLE:
            logger.warning(
                "gpu_augmentation requested but kornia is not installed; "
                "falling back to CPU albumentations."
            )
        elif use_gpu_aug and not gpu_augment_on_cuda:
            logger.info(
                "gpu_augmentation requested but device is %s; "
                "falling back to CPU albumentations (kornia needs CUDA to "
                "be a meaningful speedup).",
                self.device,
            )
        gpu_augment = (
            build_gpu_augmentation(augmentation_config)
            if gpu_augment_on_cuda else None
        )
        if gpu_augment is not None:
            logger.info(
                "GPU augmentation enabled via kornia; CPU albumentations "
                "will be skipped on training dataset."
            )

        # Multi-scale context: when context_scale > 1, load context tiles from context/ dirs
        train_context_dir = None
        val_context_dir = None
        if context_scale > 1:
            train_ctx = data_path / "train" / "context"
            val_ctx = data_path / "validation" / "context"
            if train_ctx.exists():
                train_context_dir = str(train_ctx)
                logger.info(f"Multi-scale context enabled (scale={context_scale})")
            else:
                logger.warning(f"context_scale={context_scale} but no context/ directory found")
            if val_ctx.exists():
                val_context_dir = str(val_ctx)

        cpu_augment_enabled = (
            training_params.get("augmentation", True)
            and gpu_augment is None
        )
        train_dataset = SegmentationDataset(
            images_dir=str(data_path / "train" / "images"),
            masks_dir=str(data_path / "train" / "masks"),
            input_config=input_config,
            augment=cpu_augment_enabled,
            augmentation_config=augmentation_config,
            context_dir=train_context_dir
        )

        val_dataset = SegmentationDataset(
            images_dir=str(data_path / "validation" / "images"),
            masks_dir=str(data_path / "validation" / "masks"),
            input_config=input_config,
            augment=False,  # Never augment validation
            context_dir=val_context_dir
        )

        # Validate datasets before proceeding
        train_count = len(train_dataset)
        val_count = len(val_dataset)

        if train_count == 0:
            raise ValueError(
                f"Training dataset is empty (0 patches in "
                f"{data_path / 'train' / 'images'}). "
                f"This usually means the annotations are too small to produce "
                f"any tiles at the current downsample level. "
                f"Try: (1) using a lower downsample value, "
                f"(2) making annotations larger, or "
                f"(3) adding annotations to more images."
            )

        if val_count == 0:
            logger.warning(
                "Validation dataset is empty -- training will proceed "
                "but early stopping and best-model tracking will be unreliable. "
                "Consider adding more annotations or reducing the validation split."
            )

        batch_size_requested = training_params.get("batch_size", 8)
        if batch_size_requested > train_count:
            logger.warning(
                f"Batch size ({batch_size_requested}) is larger than the "
                f"training set ({train_count} patches). "
                f"Effective batch size will be {train_count}."
            )

        if train_count < 5:
            logger.warning(
                f"Very small training set ({train_count} patches). "
                f"Training may be unreliable. Consider adding more annotations "
                f"or reducing the downsample to generate more patches."
            )

        # Warn about small batch sizes and their interaction with BatchRenorm.
        # BN/BatchRenorm computes mean/var over (N, H, W) per forward pass;
        # with batch=1-2, these statistics are extremely noisy.
        _accum = training_params.get("gradient_accumulation_steps", 1)
        _use_pt = architecture.get("use_pretrained", False)
        if batch_size_requested < 4 and not _use_pt:
            logger.warning(
                "Batch size %d is small for training from scratch. "
                "BatchNorm statistics will be very noisy. "
                "Consider batch_size >= 4, or use pretrained weights "
                "(which freeze encoder BN to pretrained running stats).",
                batch_size_requested
            )
        if batch_size_requested <= 2 and _accum > 1:
            logger.warning(
                "Gradient accumulation (%d steps) does not help "
                "BatchNorm stability: BN computes statistics per "
                "forward pass (batch_size=%d), not per optimizer step "
                "(effective_batch=%d). Consider increasing batch_size "
                "instead of accumulation_steps for better BN stats.",
                _accum, batch_size_requested,
                batch_size_requested * _accum
            )

        # Compute dataset-level normalization statistics for consistent inference
        _report_setup("computing_stats")
        try:
            train_images = []
            for i in range(min(len(train_dataset), 200)):  # Sample up to 200 patches
                img_path = train_dataset.image_files[i]
                img_arr = SegmentationDataset._load_patch(img_path)
                if img_arr.ndim == 2:
                    img_arr = img_arr[..., np.newaxis]
                # Concatenate context tile if multi-scale is enabled
                if train_context_dir:
                    ctx_path = Path(train_context_dir) / img_path.name
                    if ctx_path.exists():
                        ctx_arr = SegmentationDataset._load_patch(ctx_path)
                        if ctx_arr.ndim == 2:
                            ctx_arr = ctx_arr[..., np.newaxis]
                        # Resize context if spatial dims differ (edge tiles)
                        if ctx_arr.shape[0] != img_arr.shape[0] or ctx_arr.shape[1] != img_arr.shape[1]:
                            from PIL import Image as _PILResize
                            h, w = img_arr.shape[:2]
                            resized = []
                            for c in range(ctx_arr.shape[2]):
                                ch = _PILResize.fromarray(ctx_arr[:, :, c])
                                ch = ch.resize((w, h), _PILResize.BILINEAR)
                                resized.append(np.array(ch))
                            ctx_arr = np.stack(resized, axis=2)
                        img_arr = np.concatenate([img_arr, ctx_arr], axis=2)
                train_images.append(img_arr)

            # Use actual channel count (includes context channels if present)
            stats_channels = train_images[0].shape[2] if train_images else input_config["num_channels"]
            dataset_norm_stats = compute_dataset_stats(
                train_images, num_channels=stats_channels
            )
            logger.info(f"Computed dataset normalization stats from "
                        f"{len(train_images)} training patches ({stats_channels} channels)")
        except Exception as e:
            logger.warning(f"Failed to compute dataset normalization stats: {e}. "
                           f"Using default [0, 255] range stats.")
            # Provide default stats so metadata always has normalization_stats.
            # Consumers need this to build the model correctly; a missing field
            # causes silent failures in external programs.
            num_ch = effective_channels
            dataset_norm_stats = [
                {"p1": 0.0, "p99": 255.0, "min": 0.0, "max": 255.0,
                 "mean": 127.5, "std": 73.9}
                for _ in range(num_ch)
            ]

        # Create data loaders
        batch_size = training_params.get("batch_size", 8)
        # DataLoader workers: 0 = main thread (safe). >0 = spawn worker
        # processes to overlap I/O/augmentation with GPU compute. Exposed
        # via the "Training: DataLoader Workers" preference; older Appose
        # releases hang when >0, which is why the default is 0.
        dl_workers = int(training_params.get("data_loader_workers", 0))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dl_workers,
            persistent_workers=dl_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dl_workers,
            persistent_workers=dl_workers > 0,
        )

        # Compute class distribution from training masks for diagnostic logging.
        # This helps diagnose class imbalance issues when reviewing training logs.
        try:
            _class_pixel_counts = np.zeros(len(classes), dtype=np.int64)
            _class_patch_counts = np.zeros(len(classes), dtype=np.int64)
            _sample_count = min(len(train_dataset), 500)
            for _i in range(_sample_count):
                _mask_path = (train_dataset.masks_dir /
                              train_dataset.image_files[_i].name)
                if _mask_path.exists():
                    _mask = np.array(
                        SegmentationDataset._load_patch(_mask_path))
                    for _c in range(len(classes)):
                        _px = int((_mask == _c).sum())
                        _class_pixel_counts[_c] += _px
                        if _px > 0:
                            _class_patch_counts[_c] += 1
            _total_px = _class_pixel_counts.sum()
            if _total_px > 0:
                _class_dist = {
                    classes[c]: {
                        "pixel_pct": round(
                            100.0 * _class_pixel_counts[c] / _total_px, 1),
                        "patch_pct": round(
                            100.0 * _class_patch_counts[c] / _sample_count, 1),
                    }
                    for c in range(len(classes))
                }
            else:
                _class_dist = None
        except Exception:
            _class_dist = None

        # Setup optimizer - only optimize trainable parameters
        _report_setup("configuring_optimizer")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)

        logger.info(f"Model parameters: {total_params:,} total, {trainable_count:,} trainable "
                   f"({100*trainable_count/total_params:.1f}%)")

        learning_rate = training_params.get("learning_rate", 0.0001)
        weight_decay = training_params.get("weight_decay", 0.01)

        # Discriminative learning rates for pretrained encoders (fast.ai style)
        use_pretrained = architecture.get("use_pretrained", False)
        has_frozen = frozen_layers is not None and len(frozen_layers) > 0
        # Default 0.1 matches UNet with ResNet encoders. Handlers can override
        # via the architecture dict (e.g., Fast Pretrained uses 0.2 for mobile
        # encoders -- see agent report A2).
        discriminative_lr_ratio = training_params.get(
            "discriminative_lr_ratio",
            architecture.get("discriminative_lr_ratio", 0.1),
        )
        param_groups = self._create_param_groups(
            model, learning_rate, discriminative_lr_ratio
        ) if (use_pretrained or has_frozen) else None

        # Fused AdamW: single CUDA kernel for the param update, saves 2-5 ms/step
        # on tiny models. Safe since PyTorch 2.0 but requires all params on the
        # same CUDA device. Opt-out via training_params["fused_optimizer"]=False.
        # self.device is a string here (set in __init__), not a torch.device.
        _fused_device_str = (
            self.device if isinstance(self.device, str) else self.device.type
        )
        use_fused = (
            training_params.get("fused_optimizer", True)
            and torch.cuda.is_available()
            and _fused_device_str == "cuda"
        )
        fused_kwargs = {"fused": True} if use_fused else {}

        if param_groups and len(param_groups) > 1:
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.99), eps=1e-5,
                weight_decay=weight_decay,
                **fused_kwargs,
            )
            lr_parts = " ".join(
                f"{g.get('group_name', '?')}={g['lr']:.6f}" for g in param_groups
            )
            logger.info(f"Using AdamW with discriminative LRs (ratio={discriminative_lr_ratio}, "
                       f"fused={use_fused}): {lr_parts}")
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                betas=(0.9, 0.99), eps=1e-5,
                weight_decay=weight_decay,
                **fused_kwargs,
            )
            logger.info(f"Using AdamW optimizer (lr={learning_rate}, wd={weight_decay}, "
                       f"betas=(0.9, 0.99), eps=1e-5, fused={use_fused})")

        # Scheduler setup is deferred until after criterion is created (LR finder needs it)
        epochs = training_params.get("epochs", 50)

        # Setup early stopping
        early_stopping = None
        early_stopping_metric = training_params.get("early_stopping_metric", "mean_iou")

        # Focus class: override metric behavior for best model selection and early stopping
        focus_class = training_params.get("focus_class", None)
        focus_class_min_iou = training_params.get("focus_class_min_iou", 0.0)
        if focus_class:
            logger.info(f"Focus class '{focus_class}' IoU will be used for "
                       f"best model selection and early stopping")
            if focus_class_min_iou > 0:
                logger.info(f"  Min IoU threshold: {focus_class_min_iou:.2f} "
                           f"-- early stopping suppressed until reached")

        if training_params.get("early_stopping", True):
            # When focus class is set, always use "max" mode (higher IoU is better)
            es_mode = "max" if (focus_class or early_stopping_metric == "mean_iou") else "min"
            early_stopping = EarlyStopping(
                patience=training_params.get("early_stopping_patience", 15),
                min_delta=training_params.get("early_stopping_min_delta", 0.001),
                restore_best_weights=True,
                mode=es_mode
            )
            logger.info(f"Early stopping enabled: metric={early_stopping_metric}, "
                       f"mode={es_mode}, patience={early_stopping.patience}")

        # Load class weights and unlabeled index from exported config
        unlabeled_index = 255
        class_weights = None
        config_path = data_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                export_config = json.load(f)
            unlabeled_index = export_config.get("unlabeled_index", 255)
            weights_list = export_config.get("class_weights", None)
            if weights_list:
                class_weights = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
                logger.info(f"Using class weights: {weights_list}")

        loss_function = training_params.get("loss_function", "ce_dice")
        focal_gamma = training_params.get("focal_gamma", 2.0)
        ohem_hard_ratio = training_params.get("ohem_hard_ratio", 1.0)
        # Start value for OHEM anneal.  Defaults to the end value, i.e.
        # "fixed" (no anneal).  When start > end, the ratio linearly anneals
        # from start to end over the first 75% of epochs.  This lets the user
        # skip the warm-up phase when continuing from an existing model that
        # has already learned easy pixels.
        ohem_hard_ratio_start = training_params.get(
            "ohem_hard_ratio_start", ohem_hard_ratio)
        # Back-compat: old "fixed"/"anneal" schedule param still honored if
        # ohem_hard_ratio_start was not explicitly provided.  The default
        # anneal behavior used to start at 100% regardless of start value.
        ohem_schedule = training_params.get("ohem_schedule", "fixed")
        if "ohem_hard_ratio_start" not in training_params and ohem_schedule == "anneal":
            ohem_hard_ratio_start = 1.0
        # Derived schedule flag for downstream checks.
        _ohem_anneals = ohem_hard_ratio_start > ohem_hard_ratio

        # When True, OHEM selects the GLOBAL hardest K pixels then tops up
        # any present class that fell below a per-class safety floor.  This
        # lets hard pixels concentrate on the worst-performing class (usually
        # the minority) while guaranteeing every class keeps some signal.
        # Default False preserves the legacy per-class proportional behavior.
        ohem_adaptive_floor = bool(training_params.get(
            "ohem_adaptive_floor", False))

        # Build the base pixel-level loss
        dice = DiceLoss(ignore_index=unlabeled_index)

        if loss_function == "focal_dice":
            pixel_loss = FocalLoss(
                gamma=focal_gamma,
                class_weights=class_weights,
                ignore_index=unlabeled_index,
            )
            criterion = _CombinedPixelDiceLoss(pixel_loss, dice)
            logger.info(f"Using Focal (gamma={focal_gamma}) + Dice loss")
        elif loss_function == "focal":
            criterion = FocalLoss(
                gamma=focal_gamma,
                class_weights=class_weights,
                ignore_index=unlabeled_index,
            )
            logger.info(f"Using Focal loss (gamma={focal_gamma})")
        elif loss_function == "ce_dice":
            criterion = CombinedCEDiceLoss(
                class_weights=class_weights,
                ignore_index=unlabeled_index
            )
            logger.info("Using Combined CE + Dice loss")
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=unlabeled_index
            )
            logger.info("Using CrossEntropy loss")

        # Wrap with OHEM if active (applies to pixel-loss component only)
        if ohem_hard_ratio < 1.0:
            if isinstance(criterion, (_CombinedPixelDiceLoss, CombinedCEDiceLoss)):
                # Replace the pixel-loss component with OHEM-wrapped version
                if isinstance(criterion, _CombinedPixelDiceLoss):
                    inner_pixel = criterion.pixel_loss
                else:
                    inner_pixel = criterion.ce_loss
                ohem_pixel = OHEMCrossEntropyLoss(
                    hard_ratio=ohem_hard_ratio,
                    class_weights=class_weights,
                    ignore_index=unlabeled_index,
                    adaptive_floor=ohem_adaptive_floor,
                )
                criterion = _CombinedPixelDiceLoss(ohem_pixel, dice)
                sched_note = ""
                if _ohem_anneals:
                    sched_note = (f" [anneal: {ohem_hard_ratio_start * 100:.0f}%"
                                  f" -> {ohem_hard_ratio * 100:.0f}%"
                                  f" over first 75% of epochs]")
                    # Seed the OHEM module at the start value so the first
                    # epoch actually begins at ohem_hard_ratio_start rather
                    # than the configured end value.
                    ohem_pixel.hard_ratio = ohem_hard_ratio_start
                mode_note = " (adaptive per-class floor)" if ohem_adaptive_floor else ""
                logger.info(
                    f"OHEM active: keeping hardest {ohem_hard_ratio * 100:.0f}%%"
                    f" of pixels{mode_note} (pixel-loss component only, Dice unchanged)"
                    f"{sched_note}"
                )
            else:
                # Pure pixel loss (CE or Focal) -- wrap entirely
                criterion = OHEMCrossEntropyLoss(
                    hard_ratio=ohem_hard_ratio,
                    class_weights=class_weights,
                    ignore_index=unlabeled_index,
                    adaptive_floor=ohem_adaptive_floor,
                )
                if _ohem_anneals:
                    criterion.hard_ratio = ohem_hard_ratio_start
                sched_note = ""
                if _ohem_anneals:
                    sched_note = (f" [anneal: {ohem_hard_ratio_start * 100:.0f}%"
                                  f" -> {ohem_hard_ratio * 100:.0f}%"
                                  f" over first 75% of epochs]")
                mode_note = " (adaptive per-class floor)" if ohem_adaptive_floor else ""
                logger.info(
                    f"OHEM active: keeping hardest {ohem_hard_ratio * 100:.0f}%%"
                    f" of pixels{mode_note}{sched_note}"
                )

        # Setup learning rate scheduler (deferred until after criterion for LR finder)
        accumulation_steps = training_params.get("gradient_accumulation_steps", 1)
        scheduler_config = training_params.get("scheduler_config", {})
        scheduler_config["accumulation_steps"] = accumulation_steps
        scheduler_type = training_params.get("scheduler", "onecycle")

        # LR Finder: auto-run before OneCycleLR on new training (not checkpoint resume).
        # Can be skipped via training_params["use_lr_finder"]=False, in which case
        # we use a heuristic max_lr = base_lr * sqrt(batch_size / 8) so OneCycleLR
        # has a reasonable peak.  Saves ~10s of presweep time for tiny models.
        use_lr_finder = training_params.get("use_lr_finder", True)
        if (scheduler_type == "onecycle" and checkpoint_path is None
                and not use_lr_finder):
            import math as _math
            heuristic_lr = learning_rate * _math.sqrt(
                max(1, training_params.get("batch_size", 8)) / 8.0
            )
            scheduler_config["max_lr"] = heuristic_lr
            logger.info(
                "LR Finder disabled, using heuristic max_lr=%.6f (base_lr=%.6f, "
                "batch_size=%d)",
                heuristic_lr, learning_rate,
                training_params.get("batch_size", 8),
            )

        if (scheduler_type == "onecycle" and checkpoint_path is None
                and use_lr_finder):
            _report_setup("finding_learning_rate")
            try:
                suggested_lr, finder_lrs, finder_losses = self.find_learning_rate(
                    model, train_loader, criterion)
                # Guard rails: reject suggestions that are too small or too large.
                # Cap at both an absolute ceiling (0.01) and a relative ceiling
                # (10x the user's specified lr).  The relative cap prevents the
                # finder from proposing wildly different values than what the
                # user chose -- a 100x override causes training instability.
                min_reasonable_lr = 1e-5
                max_reasonable_lr = min(0.01, learning_rate * 10)
                if suggested_lr is not None and suggested_lr >= min_reasonable_lr:
                    if suggested_lr > max_reasonable_lr:
                        logger.warning(
                            "LR Finder suggested lr=%.4f which exceeds "
                            "safety ceiling (%.4f). Capping to %.4f to "
                            "prevent divergence.",
                            suggested_lr, max_reasonable_lr, max_reasonable_lr)
                        suggested_lr = max_reasonable_lr
                    logger.info(f"LR Finder suggested lr={suggested_lr:.6f}")
                    scheduler_config["max_lr"] = suggested_lr
                elif suggested_lr is not None:
                    logger.warning(
                        f"LR Finder suggested lr={suggested_lr:.2e} is below"
                        f" floor ({min_reasonable_lr:.0e}), using base lr"
                        f" {learning_rate:.6f} instead")
                else:
                    logger.info("LR Finder could not suggest LR, using default max_lr")
            except Exception as e:
                logger.warning(f"LR Finder failed: {e} -- using default max_lr")

        scheduler = self._create_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            scheduler_config=scheduler_config,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

        # Save scheduler_config back into training_params so it survives in
        # checkpoints. On resume, the config (including LR Finder max_lr)
        # is read from training_params["scheduler_config"].
        training_params["scheduler_config"] = scheduler_config

        # Setup mixed precision training with BF16 auto-detection
        use_mixed_precision = (
            training_params.get("mixed_precision", True)
            and self.device == "cuda"
        )
        use_bf16 = use_mixed_precision and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        # BF16 doesn't need a GradScaler (wider dynamic range avoids underflow)
        scaler = None if use_bf16 else (
            torch.amp.GradScaler("cuda") if use_mixed_precision else None
        )
        if use_mixed_precision:
            dtype_name = "BF16" if use_bf16 else "FP16"
            logger.info(f"Mixed precision training enabled ({dtype_name})")

        # Determine best-model tracking mode (same metric as early stopping)
        # When focus class is set, always use "max" mode (higher IoU is better)
        best_score_mode = "max" if (focus_class or early_stopping_metric == "mean_iou") else "min"
        best_score = float("-inf") if best_score_mode == "max" else float("inf")
        best_model_state = None
        training_history = []

        # Training diagnostics: automated checks for common issues
        from .training_diagnostics import TrainingDiagnostics
        _diagnostics = TrainingDiagnostics(classes=classes)

        def _is_best(current, best):
            if best_score_mode == "max":
                return current > best
            return current < best

        # Restore from checkpoint if resuming
        if checkpoint_path:
            _report_setup("loading_checkpoint")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore scheduler (recreate OneCycleLR with remaining steps)
            if scheduler is not None:
                if isinstance(scheduler, OneCycleLR):
                    remaining_epochs = epochs - start_epoch
                    steps_per_epoch = -(-len(train_loader) // accumulation_steps)
                    remaining_steps = remaining_epochs * max(steps_per_epoch, 1)
                    if remaining_steps > 0:
                        resume_config = training_params.get("scheduler_config", {})
                        max_lr = resume_config.get(
                            "max_lr", optimizer.param_groups[0]["lr"] * 10)
                        scheduler = OneCycleLR(
                            optimizer,
                            max_lr=max_lr,
                            total_steps=remaining_steps,
                            pct_start=resume_config.get("pct_start", 0.3),
                            anneal_strategy=resume_config.get("anneal_strategy", "cos"),
                            div_factor=resume_config.get("div_factor", 25.0),
                            final_div_factor=resume_config.get("final_div_factor", 1e4)
                        )
                elif "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logger.info("Restored scheduler state from checkpoint")

            # Restore early stopping state (handle both old and new format)
            if early_stopping is not None and "early_stopping" in checkpoint:
                es_state = checkpoint["early_stopping"]
                if "best_score" in es_state:
                    early_stopping.best_score = es_state["best_score"]
                elif "best_loss" in es_state:
                    early_stopping.best_score = es_state["best_loss"]
                early_stopping.best_epoch = es_state["best_epoch"]
                early_stopping.counter = es_state["counter"]
                if "best_state" in es_state and es_state["best_state"] is not None:
                    early_stopping.best_state = es_state["best_state"]

            # Restore training history and best model (handle both formats)
            training_history = checkpoint.get("training_history", [])
            if "best_score" in checkpoint:
                best_score = checkpoint["best_score"]
            elif "best_loss" in checkpoint:
                best_score = checkpoint["best_loss"]
            if "best_model_state" in checkpoint:
                best_model_state = checkpoint["best_model_state"]

            # Re-apply frozen layers (load_state_dict restores weights but
            # doesn't freeze parameters -- requires_grad is not in state_dict)
            if frozen_layers:
                frozen_count = 0
                for param_name, param in model.named_parameters():
                    if any(layer in param_name for layer in frozen_layers):
                        param.requires_grad = False
                        frozen_count += 1
                if frozen_count > 0:
                    logger.info(f"Re-froze {frozen_count} parameters after "
                                f"checkpoint resume ({len(frozen_layers)} layer groups)")

            logger.info(f"Resumed from checkpoint at epoch {start_epoch}, "
                       f"best_score={best_score:.4f}")

        # Progressive resizing: train at half-resolution first, then full
        progressive_resize = training_params.get("progressive_resize", False)
        phase1_end_epoch = 0  # No phase transition by default
        if progressive_resize and checkpoint_path is None:
            phase1_end_epoch = max(1, int(epochs * 0.4))
            input_size = architecture.get("input_size", [512, 512])
            small_size = max(input_size[0] // 2, 64)
            logger.info(f"Progressive resizing: phase 1 (epochs 1-{phase1_end_epoch}) "
                       f"at {small_size}x{small_size}, phase 2 at "
                       f"{input_size[0]}x{input_size[1]}")

            # Create half-resolution datasets using torchvision transforms
            from torchvision.transforms import InterpolationMode
            import torchvision.transforms.functional as TF

            class ResizedDataset(Dataset):
                """Wrapper that resizes images and masks to a target size."""
                def __init__(self, base_dataset, target_size):
                    self.base = base_dataset
                    self.target_size = target_size

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, idx):
                    image, mask = self.base[idx]
                    # image: (C, H, W), mask: (H, W)
                    image = TF.resize(image, [self.target_size, self.target_size],
                                      interpolation=InterpolationMode.BILINEAR,
                                      antialias=True)
                    mask = TF.resize(mask.unsqueeze(0),
                                     [self.target_size, self.target_size],
                                     interpolation=InterpolationMode.NEAREST
                                     ).squeeze(0)
                    return image, mask

            small_train_dataset = ResizedDataset(train_dataset, small_size)
            small_val_dataset = ResizedDataset(val_dataset, small_size)
            small_train_loader = DataLoader(
                small_train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=dl_workers,
                persistent_workers=dl_workers > 0)
            small_val_loader = DataLoader(
                small_val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=dl_workers,
                persistent_workers=dl_workers > 0)

            # Start with small loaders
            active_train_loader = small_train_loader
            active_val_loader = small_val_loader
        else:
            active_train_loader = train_loader
            active_val_loader = val_loader

        # Emit training configuration summary for diagnostic logging
        try:
            backbone = architecture.get("backbone", "resnet34")
            input_size = architecture.get("input_size", [512, 512])
            tile_size_str = (f"{input_size[0]}x{input_size[1]}"
                            if isinstance(input_size, (list, tuple))
                            else str(input_size))

            # Optimizer description
            if param_groups and len(param_groups) > 1:
                optimizer_desc = (f"AdamW (wd={weight_decay}, betas=0.9/0.99)"
                                  " [discriminative LRs]")
                # Read live LRs from optimizer (scheduler may have changed them)
                disc_lr_parts = ", ".join(
                    f"{g.get('group_name', '?')}={g['lr']:.6f}"
                    for g in optimizer.param_groups
                )
            else:
                optimizer_desc = (f"AdamW (lr={learning_rate}, wd={weight_decay},"
                                  f" betas=0.9/0.99)")
                disc_lr_parts = None

            # Scheduler description
            if scheduler is None:
                sched_desc = "None"
            elif scheduler_type == "onecycle":
                sched_desc = "OneCycleLR"
            elif scheduler_type == "cosine":
                sc = scheduler_config if isinstance(scheduler_config, dict) else {}
                sched_desc = (f"CosineAnnealingWarmRestarts"
                              f" (T_0={sc.get('T_0', '?')})")
            elif scheduler_type == "step":
                sc = scheduler_config if isinstance(scheduler_config, dict) else {}
                sched_desc = (f"StepLR (step_size={sc.get('step_size', '?')},"
                              f" gamma={sc.get('gamma', 0.1)})")
            elif scheduler_type == "plateau":
                sc = scheduler_config if isinstance(scheduler_config, dict) else {}
                sched_desc = (f"ReduceOnPlateau (factor={sc.get('factor', 0.5)},"
                              f" patience={sc.get('patience', 10)})")
            else:
                sched_desc = scheduler_type

            # Mixed precision
            if use_mixed_precision:
                mp_desc = "BF16" if use_bf16 else "FP16"
            else:
                mp_desc = "Off"

            # Normalization -- input_config["normalization"] may be a
            # dict (current format) or a bare strategy string.
            norm_config_val = input_config.get("normalization", {})
            if isinstance(norm_config_val, dict):
                norm_method = norm_config_val.get("strategy", "percentile_99")
                per_channel = norm_config_val.get("per_channel", False)
            else:
                norm_method = str(norm_config_val) if norm_config_val else "percentile_99"
                per_channel = input_config.get("per_channel", False)
            norm_desc = f"{norm_method} (per_channel={per_channel})"

            # Augmentation list
            aug_enabled = training_params.get("augmentation", True)
            if aug_enabled and augmentation_config:
                aug_items = [k for k, v in augmentation_config.items()
                             if v and k != "enabled"]
                aug_desc = ", ".join(aug_items) if aug_items else "default"
            elif aug_enabled:
                aug_desc = "default"
            else:
                aug_desc = "Off"

            # Early stopping
            if early_stopping is not None:
                es_desc = (f"{early_stopping_metric}"
                           f" (patience={early_stopping.patience})")
                if focus_class:
                    es_desc += f", focus={focus_class}"
                    if focus_class_min_iou > 0:
                        es_desc += f" (min IoU={focus_class_min_iou:.2f})"
            else:
                es_desc = "Off"

            config_summary = {
                "Architecture": f"{model_type} ({backbone})",
                "Optimizer": optimizer_desc,
                "Scheduler": sched_desc,
                "Loss": self._format_loss_desc(
                    loss_function, focal_gamma, ohem_hard_ratio, ohem_schedule,
                    ohem_hard_ratio_start, ohem_adaptive_floor),
                "Batch Size": (f"{batch_size} (accumulation={accumulation_steps},"
                               f" effective={batch_size * accumulation_steps})"),
                "Tile Size": tile_size_str,
                "Mixed Precision": mp_desc,
                "Classes": f"{len(classes)} ({', '.join(classes)})",
                "Training Patches": (f"{len(train_dataset)} train /"
                                     f" {len(val_dataset)} val"),
                "Normalization": norm_desc,
                "Channels": str(effective_channels),
                "Early Stopping": es_desc,
                "Augmentation": aug_desc,
                "Progressive Resize": "On" if progressive_resize else "Off",
            }

            # Conditional entries
            if disc_lr_parts:
                config_summary["Discriminative LRs"] = disc_lr_parts
            if (scheduler_type == "onecycle" and checkpoint_path is None
                    and scheduler_config.get("max_lr") is not None):
                lr_val = scheduler_config["max_lr"]
                if isinstance(lr_val, (int, float)):
                    finder_note = f"max_lr={lr_val:.6f}"
                    if lr_val != learning_rate:
                        finder_note += (
                            f" (user lr={learning_rate:.6f}, "
                            f"finder override: {lr_val/learning_rate:.0f}x)")
                    config_summary["LR Finder"] = finder_note
                    # Show effective peak LRs per param group so the user
                    # sees what OneCycleLR will actually ramp to
                    if param_groups and len(param_groups) > 1:
                        base_lr = max(g["lr"] for g in optimizer.param_groups)
                        if base_lr > 0:
                            peak_parts = ", ".join(
                                f"{g.get('group_name', '?')}="
                                f"{lr_val * (g['lr'] / base_lr):.6f}"
                                for g in optimizer.param_groups
                            )
                        else:
                            peak_parts = f"all={lr_val:.6f}"
                        config_summary["Peak LRs (OneCycleLR)"] = peak_parts

            # Class distribution: pixel % and patch presence %
            if _class_dist:
                dist_parts = []
                for cname, cinfo in _class_dist.items():
                    dist_parts.append(
                        f"{cname}={cinfo['pixel_pct']}% pixels, "
                        f"in {cinfo['patch_pct']}% of patches")
                config_summary["Class Distribution"] = "; ".join(dist_parts)

                # Class weight info if set
                weights_list = export_config.get("class_weights", None)
                if weights_list:
                    w_parts = [f"{classes[i]}={w:.2f}"
                               for i, w in enumerate(weights_list)
                               if i < len(classes)]
                    config_summary["Class Weights"] = ", ".join(w_parts)

            _report_setup("training_config", config_summary)
        except Exception:
            pass  # Never let config summary break training

        # Pre-flight VRAM estimate: warn if batch * tile likely exceeds GPU memory.
        # Model weights + optimizer state (AdamW ~3x model) + forward activations +
        # backward gradients. ViT models need much more activation memory than CNNs.
        try:
            if self.device == "cuda":
                mem_info = self.gpu_manager.get_memory_info()
                total_mb = mem_info.get("total_mb", 0)
                allocated_mb = mem_info.get("allocated_mb", 0)
                free_mb = total_mb - allocated_mb
                model_mb = self.gpu_manager.estimate_model_memory(model)
                # Rough estimate: model weights + optimizer (3x) + activations
                # ViTs need ~8-12x model size for activations; CNNs need ~3-5x
                act_multiplier = 10.0 if model_type == "muvit" else 4.0
                # Mixed precision roughly halves activation/gradient memory
                if use_mixed_precision:
                    act_multiplier *= 0.6
                # Context scale doubles input channels, increasing activations
                context_scale = architecture.get("context_scale", 1)
                if context_scale > 1:
                    act_multiplier *= 1.5
                tiles_per_batch = batch_size
                tile_pixels = architecture.get("input_size", [512, 512])
                tile_area = tile_pixels[0] * tile_pixels[1] if isinstance(tile_pixels, (list, tuple)) else 512 * 512
                # If progressive resizing is active, estimate for the LARGER
                # phase 2 tile size (that's where OOM would actually occur)
                if progressive_resize:
                    effective_area = tile_area  # full res for phase 2
                    phase_note = " (phase 2 full-res)"
                else:
                    effective_area = tile_area
                    phase_note = ""
                # Scale activation estimate by tile area relative to 256x256 baseline
                area_scale = effective_area / (256 * 256)
                estimated_mb = model_mb * (1 + 3 + act_multiplier * area_scale * tiles_per_batch)
                if estimated_mb > free_mb * 0.9:
                    logger.warning(
                        "VRAM estimate: %.0f MB needed vs %.0f MB free%s -- "
                        "OOM likely. Reduce batch size, tile size, or model size.",
                        estimated_mb, free_mb, phase_note
                    )
                else:
                    logger.info(
                        "VRAM estimate: %.0f MB needed, %.0f MB free%s -- OK",
                        estimated_mb, free_mb, phase_note
                    )
        except Exception:
            pass  # Never let estimation break training

        # Training loop
        _report_setup("starting_training")
        num_classes = len(classes)

        # Initialize best-tracking variables. When resuming from a checkpoint,
        # restore from early_stopping state so pause/completion reports are accurate.
        if checkpoint_path and early_stopping is not None:
            best_epoch = early_stopping.best_epoch
            best_mean_iou = best_score if best_score_mode == "max" else 0.0
        else:
            best_epoch = 0
            best_mean_iou = 0.0
        best_loss = 0.0
        best_accuracy = 0.0

        # Auto-detect encoder prefix for BN freezing.  SMP models use
        # "encoder.", but future architectures might use "backbone." or
        # "features.".  Detect the prefix once here so the per-epoch
        # freezing loop does not depend on a hardcoded string.
        _encoder_prefix = None
        for candidate in ("encoder", "backbone", "features"):
            if hasattr(model, candidate):
                _encoder_prefix = candidate + "."
                break
        if _encoder_prefix is None and model_type != "muvit":
            logger.debug("Could not detect encoder prefix on model; "
                         "encoder BN freezing will be skipped")

        # BatchRenorm warmup: linearly increase rmax/dmax over first 20% of
        # epochs from BatchNorm-equivalent (1, 0) to full BatchRenorm (3, 5).
        # Skip warmup when continuing from a pretrained model that already has
        # calibrated BatchRenorm statistics -- the warmup would revert to
        # standard BatchNorm behavior and destabilize validation.
        brenorm_rmax_target = 3.0
        brenorm_dmax_target = 5.0
        has_pretrained_weights = (pretrained_model_path or checkpoint_path
                                  or (frozen_layers and len(frozen_layers) > 0)
                                  or architecture.get("use_pretrained", False))
        if has_pretrained_weights:
            brenorm_warmup_epochs = 0
            set_batchrenorm_limits(model, rmax=brenorm_rmax_target,
                                  dmax=brenorm_dmax_target)
            logger.info("BatchRenorm warmup skipped (pretrained encoder weights)")
        else:
            brenorm_warmup_epochs = max(1, int(epochs * 0.2))

        for epoch in range(start_epoch, epochs):
            # Progressive resizing: switch to full resolution at phase boundary
            if progressive_resize and epoch == phase1_end_epoch and checkpoint_path is None:
                logger.info(f"Progressive resize: switching to full resolution at epoch {epoch+1}")
                active_train_loader = train_loader
                active_val_loader = val_loader
                # Temporarily boost BatchRenorm running stat momentum so stats
                # adapt faster to the new resolution's feature distribution.
                # Default momentum=0.01 means ~100 batches for significant
                # update; 0.1 means ~10 batches.  Boost for 1 epoch then
                # restore (checked at epoch == phase1_end_epoch + 1 below).
                if model_type != "muvit":
                    _bn_original_momentum = 0.01
                    from ..utils.batchrenorm import BatchRenorm2d as _BRN
                    for m in model.modules():
                        if isinstance(m, _BRN):
                            _bn_original_momentum = m.momentum
                            m.momentum = min(0.1, _bn_original_momentum * 10)
                    logger.info("Boosted BatchRenorm momentum %.3f -> %.3f "
                                "for resolution adaptation (1 epoch)",
                                _bn_original_momentum,
                                min(0.1, _bn_original_momentum * 10))
                # Recreate OneCycleLR for remaining epochs
                if isinstance(scheduler, OneCycleLR):
                    remaining_epochs = epochs - phase1_end_epoch
                    remaining_steps = remaining_epochs * (
                        -(-len(train_loader) // accumulation_steps))
                    remaining_steps = max(remaining_steps, 1)
                    max_lr = scheduler_config.get(
                        "max_lr", optimizer.param_groups[0]["lr"] * 10)
                    scheduler = OneCycleLR(
                        optimizer,
                        max_lr=max_lr,
                        total_steps=remaining_steps,
                        pct_start=scheduler_config.get("pct_start", 0.3),
                        anneal_strategy=scheduler_config.get("anneal_strategy", "cos"),
                        div_factor=scheduler_config.get("div_factor", 25.0),
                        final_div_factor=scheduler_config.get("final_div_factor", 1e4)
                    )
            # Restore BatchRenorm momentum after progressive resize adaptation
            if (progressive_resize and model_type != "muvit"
                    and epoch == phase1_end_epoch + 1
                    and checkpoint_path is None):
                from ..utils.batchrenorm import BatchRenorm2d as _BRN
                for m in model.modules():
                    if isinstance(m, _BRN):
                        m.momentum = _bn_original_momentum
                logger.info("Restored BatchRenorm momentum to %.3f",
                            _bn_original_momentum)

            # Check for cancellation
            if cancel_flag and cancel_flag.is_set():
                logger.info("Training cancelled")
                break

            # Clear GPU cache at epoch start to prevent memory accumulation
            self.gpu_manager.clear_cache()

            # Log memory status at start of epoch
            self.gpu_manager.log_memory_status(prefix=f"Epoch {epoch+1}/{epochs} start: ")

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # BatchRenorm warmup: relax clipping bounds gradually
            if epoch < brenorm_warmup_epochs:
                progress = (epoch + 1) / brenorm_warmup_epochs
                cur_rmax = 1.0 + (brenorm_rmax_target - 1.0) * progress
                cur_dmax = brenorm_dmax_target * progress
                set_batchrenorm_limits(model, rmax=cur_rmax, dmax=cur_dmax)
                if epoch == 0:
                    logger.info(
                        f"BatchRenorm warmup: {brenorm_warmup_epochs} epochs "
                        f"(rmax 1->{brenorm_rmax_target}, "
                        f"dmax 0->{brenorm_dmax_target})")
            elif epoch == brenorm_warmup_epochs:
                # Ensure final values are set exactly
                set_batchrenorm_limits(
                    model, rmax=brenorm_rmax_target, dmax=brenorm_dmax_target)

            # OHEM anneal: linearly decrease hard_ratio from
            # ohem_hard_ratio_start to ohem_hard_ratio over the first 75% of
            # epochs.  Lets the user choose whether to warm up from all pixels
            # (start=100%) or pick up directly at the target (start=end).
            # Only applies when start > end and OHEM is active.
            if _ohem_anneals and ohem_hard_ratio < 1.0:
                _ohem_module = None
                if isinstance(criterion, _CombinedPixelDiceLoss):
                    if isinstance(criterion.pixel_loss, OHEMCrossEntropyLoss):
                        _ohem_module = criterion.pixel_loss
                elif isinstance(criterion, OHEMCrossEntropyLoss):
                    _ohem_module = criterion
                if _ohem_module is not None:
                    anneal_end = max(1, int(epochs * 0.75))
                    if epoch < anneal_end:
                        progress = epoch / anneal_end
                        cur_ratio = (ohem_hard_ratio_start
                                     - (ohem_hard_ratio_start - ohem_hard_ratio) * progress)
                    else:
                        cur_ratio = ohem_hard_ratio
                    _ohem_module.hard_ratio = cur_ratio
                    if epoch == 0:
                        logger.info(
                            "OHEM anneal: %.0f%% -> %.0f%% over %d epochs",
                            ohem_hard_ratio_start * 100,
                            ohem_hard_ratio * 100, anneal_end)
                    elif epoch == anneal_end:
                        logger.info(
                            "OHEM anneal complete: hard_ratio=%.0f%%",
                            ohem_hard_ratio * 100)

            # Train epoch
            model.train()

            # Freeze ALL encoder BN/BatchRenorm layers to eval mode when
            # using pretrained encoder weights.  With small batches (4-8),
            # batch statistics are extremely noisy and corrupt the running
            # stats, causing wild validation oscillation (mIoU swings from
            # 0.76 to 0.05 between epochs).  The pretrained encoder already
            # has high-quality running stats from millions of patches --
            # updating them with 6-sample batches can only hurt.
            # Decoder BN stays in train mode (no pretrained stats to preserve).
            if has_pretrained_weights and _encoder_prefix:
                frozen_bn_count = 0
                for name, module in model.named_modules():
                    if name.startswith(_encoder_prefix):
                        module.eval()
                        frozen_bn_count += 1
                if epoch == start_epoch and frozen_bn_count > 0:
                    logger.info("Set %d %s* modules to eval mode "
                                "(BN uses pretrained running stats, "
                                "weights still trainable)",
                                frozen_bn_count, _encoder_prefix)

            train_loss = 0.0

            # Gradient accumulation support
            effective_batches = 0
            optimizer.zero_grad()

            total_batches = len(active_train_loader)
            # Log every ~10% of batches (at least every 50 batches) so
            # long epochs on slow devices (MPS) show visible progress.
            log_interval = max(1, min(50, total_batches // 10))
            epoch_start_time = time.time()

            for batch_idx, (images, masks) in enumerate(active_train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # GPU-side augmentation: mask-consistent geometric ops plus
                # intensity jitter and low-p noise/blur.  Kornia expects the
                # mask as a 4D float tensor (N, 1, H, W); we unsqueeze,
                # transform, then squeeze back to int64 (N, H, W).
                if gpu_augment is not None:
                    masks_aug = masks.unsqueeze(1).float()
                    images, masks_aug = gpu_augment(images, masks_aug)
                    # Round is a no-op for images but makes mask values
                    # (nearest-interpolated) safely castable to long.
                    masks = masks_aug.squeeze(1).round().long()

                if use_mixed_precision:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    # Capture loss for logging BEFORE backward.
                    # When loss is already float32 (CE promotes under
                    # autocast), loss.float() returns the SAME tensor
                    # object -- backward() can then invalidate its
                    # storage, causing .item() to return 0.
                    train_loss += loss.detach().float().item()
                    scaled_loss = loss.float() / accumulation_steps
                    if scaler is not None:
                        # FP16 path: use GradScaler
                        scaler.scale(scaled_loss).backward()
                    else:
                        # BF16 path: no scaler needed
                        scaled_loss.backward()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    train_loss += loss.detach().item()
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()

                # Step optimizer every accumulation_steps batches (or on last batch)
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(active_train_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                    effective_batches += 1

                    # Step scheduler if using OneCycleLR (per-optimizer-step)
                    if isinstance(scheduler, OneCycleLR):
                        scheduler.step()

                # Log peak GPU memory after first batch of first epoch
                # (and first batch after progressive resize switch) to show
                # actual VRAM usage vs the pre-flight estimate.
                if batch_idx == 0 and self.device == "cuda":
                    is_first_epoch = (epoch == start_epoch)
                    is_phase2_start = (progressive_resize
                                       and epoch == phase1_end_epoch)
                    if is_first_epoch or is_phase2_start:
                        phase_label = ("Phase 2 (full-res) " if is_phase2_start
                                       else "")
                        self.gpu_manager.log_memory_status(
                            prefix=f"  {phase_label}After first batch: ",
                            include_peak=True)
                        peak_mb = self.gpu_manager.get_peak_allocated_mb()
                        total_gpu = self.gpu_manager.get_memory_mb()
                        headroom = total_gpu - peak_mb
                        if headroom > peak_mb * 0.5:
                            logger.info(
                                "  %.0f MB headroom -- a larger batch size "
                                "may train faster", headroom)

                # Periodic batch progress logging
                if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                    elapsed = time.time() - epoch_start_time
                    avg_loss = train_loss / (batch_idx + 1)
                    batches_done = batch_idx + 1
                    if elapsed > 0 and batches_done > 1:
                        sec_per_batch = elapsed / batches_done
                        remaining = sec_per_batch * (total_batches - batches_done)
                        eta_str = f", ETA {remaining:.0f}s"
                    else:
                        eta_str = ""
                    logger.info(
                        f"  Epoch {epoch+1} batch {batches_done}/{total_batches} "
                        f"loss={avg_loss:.4f} ({elapsed:.0f}s elapsed{eta_str})")
                    # Report batch-level progress to Java UI via setup_callback
                    if setup_callback:
                        try:
                            setup_callback("training_batch", {
                                "epoch": epoch + 1,
                                "total_epochs": epochs,
                                "batch": batches_done,
                                "total_batches": total_batches,
                                "batch_loss": avg_loss,
                                "elapsed_seconds": round(elapsed, 1),
                            })
                        except Exception:
                            pass

                # Check cancellation between batches
                if cancel_flag and cancel_flag.is_set():
                    logger.info("Training cancelled mid-epoch at batch %d/%d",
                                batch_idx + 1, total_batches)
                    break

            train_loss /= max(len(active_train_loader), 1)

            # Abort early if loss has diverged to NaN. This typically means
            # the learning rate is too high.  Continuing wastes GPU time on
            # a model that will never recover.
            if math.isnan(train_loss):
                _nan_epoch_count = getattr(self, '_nan_epoch_count', 0) + 1
                self._nan_epoch_count = _nan_epoch_count
                if _nan_epoch_count >= 2:
                    logger.error(
                        "Training loss is NaN for %d consecutive epochs. "
                        "Aborting -- the learning rate is likely too high. "
                        "Try a lower learning rate or use ReduceOnPlateau "
                        "scheduler instead of OneCycleLR.",
                        _nan_epoch_count)
                    raise RuntimeError(
                        f"Training diverged: loss is NaN for "
                        f"{_nan_epoch_count} consecutive epochs. "
                        f"The learning rate (max_lr={scheduler_config.get('max_lr', '?')}) "
                        f"is likely too high.")
            else:
                self._nan_epoch_count = 0

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            # Per-class accumulators
            class_tp = torch.zeros(num_classes, device=self.device)
            class_fp = torch.zeros(num_classes, device=self.device)
            class_fn = torch.zeros(num_classes, device=self.device)
            class_loss_sum = torch.zeros(num_classes, device=self.device)
            class_pixel_count = torch.zeros(num_classes, device=self.device)

            val_total_batches = len(active_val_loader)
            val_log_interval = max(1, min(50, val_total_batches // 10))
            val_start_time = time.time()

            with torch.no_grad():
                for val_batch_idx, (images, masks) in enumerate(active_val_loader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    if use_mixed_precision:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            outputs = model(images)
                        # Compute val loss in FP32 to avoid BF16 overflow.
                        # BF16 logits from confident pretrained models can
                        # overflow exp() in cross-entropy, producing inf loss.
                        loss = criterion(outputs.float(), masks)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    # Only count labeled pixels for accuracy
                    labeled_mask = masks != unlabeled_index
                    total += labeled_mask.sum().item()
                    correct += ((predicted == masks) & labeled_mask).sum().item()

                    # Per-class TP/FP/FN
                    for c in range(num_classes):
                        pred_c = (predicted == c) & labeled_mask
                        true_c = (masks == c) & labeled_mask
                        class_tp[c] += (pred_c & true_c).sum()
                        class_fp[c] += (pred_c & ~true_c).sum()
                        class_fn[c] += (~pred_c & true_c).sum()

                    # Per-class loss (unreduced, must include ignore_index
                    # to avoid CUDA assertion on unlabeled pixels)
                    per_pixel_loss = F.cross_entropy(
                        outputs, masks, reduction='none',
                        ignore_index=unlabeled_index)
                    # Clamp to prevent Infinity from -log(0) when model is
                    # very confident but wrong, which would poison the mean
                    per_pixel_loss = torch.clamp(per_pixel_loss, max=100.0)
                    for c in range(num_classes):
                        c_mask = (masks == c) & labeled_mask
                        c_count = c_mask.sum()
                        if c_count > 0:
                            class_loss_sum[c] += per_pixel_loss[c_mask].sum()
                            class_pixel_count[c] += c_count

                    # Periodic validation progress logging
                    if (val_batch_idx + 1) % val_log_interval == 0:
                        val_elapsed = time.time() - val_start_time
                        logger.info(
                            f"  Epoch {epoch+1} val batch "
                            f"{val_batch_idx+1}/{val_total_batches} "
                            f"({val_elapsed:.0f}s elapsed)")

            val_loss /= max(len(active_val_loader), 1)
            accuracy = correct / max(total, 1)

            # Compute per-class IoU and loss
            per_class_iou = {}
            per_class_loss = {}
            for c in range(num_classes):
                denom = (class_tp[c] + class_fp[c] + class_fn[c]).item()
                iou = class_tp[c].item() / denom if denom > 0 else 0.0
                per_class_iou[classes[c]] = round(iou, 4)

                px_count = class_pixel_count[c].item()
                c_loss = class_loss_sum[c].item() / px_count if px_count > 0 else 0.0
                per_class_loss[classes[c]] = round(c_loss, 4)

            iou_values = list(per_class_iou.values())
            mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0
            mean_iou = round(mean_iou, 4)

            # Record history
            training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "learning_rate": current_lr,
                "per_class_iou": per_class_iou,
                "per_class_loss": per_class_loss,
                "mean_iou": mean_iou
            })

            # Run training diagnostics periodically to detect common issues
            _hist_len = len(training_history)
            if _hist_len >= 15 and _hist_len % 10 == 0:
                _diagnostics.run_checks(training_history)

            # Log with per-class breakdown
            # Use scientific notation for very small train_loss to avoid
            # misleading 0.0000 display when loss < 0.00005 (common when
            # overfitting a small training set).
            tl_fmt = f"{train_loss:.2e}" if 0 < train_loss < 0.0001 else f"{train_loss:.4f}"
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={tl_fmt}, "
                       f"val_loss={val_loss:.4f}, acc={accuracy:.4f}, "
                       f"mIoU={mean_iou:.4f}, lr={current_lr:.6f}")
            iou_parts = " ".join(f"{k}={v:.3f}" for k, v in per_class_iou.items())
            loss_parts = " ".join(f"{k}={v:.4f}" for k, v in per_class_loss.items())
            logger.info(f"  IoU: {iou_parts}")
            logger.info(f"  Loss: {loss_parts}")

            if progress_callback:
                progress_callback(epoch + 1, train_loss, val_loss, accuracy,
                                  per_class_iou, per_class_loss, mean_iou)

            # Step scheduler (for epoch-based schedulers)
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                if isinstance(scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau needs the metric value
                    if focus_class and focus_class in per_class_iou:
                        plateau_metric = per_class_iou[focus_class]
                    else:
                        plateau_metric = mean_iou if early_stopping_metric == "mean_iou" else val_loss
                    scheduler.step(plateau_metric)
                elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step(epoch + 1)
                else:
                    scheduler.step()

            # Save best model checkpoint (independent of early stopping)
            if focus_class and focus_class in per_class_iou:
                current_metric = per_class_iou[focus_class]
            else:
                current_metric = mean_iou if early_stopping_metric == "mean_iou" else val_loss
            if _is_best(current_metric, best_score):
                best_score = current_metric
                best_epoch = epoch + 1
                best_loss = val_loss
                best_accuracy = accuracy
                best_mean_iou = mean_iou
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if focus_class:
                    metric_name = f"{focus_class} IoU"
                else:
                    metric_name = "mIoU" if early_stopping_metric == "mean_iou" else "loss"
                logger.info(f"  New best model at epoch {epoch+1} ({metric_name}={current_metric:.4f})")
                # Persist full checkpoint to disk for crash recovery.
                # If training is interrupted, this file supports both model
                # recovery (finalize_training.py) and training resume.
                self._save_best_in_progress(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopping=early_stopping,
                    best_model_state=best_model_state,
                    model_type=model_type,
                    best_epoch=best_epoch,
                    best_score=best_score,
                    best_score_mode=best_score_mode,
                    training_config={
                        "model_type": model_type,
                        "architecture": architecture,
                        "input_config": input_config,
                        "training_params": training_params,
                        "classes": classes,
                    },
                    training_history=training_history,
                    normalization_stats=dataset_norm_stats,
                )

            # Log focus class status
            if focus_class:
                fc_iou = per_class_iou.get(focus_class, 0.0)
                logger.info(f"  Focus class '{focus_class}' IoU={fc_iou:.4f}"
                           f" (min threshold={focus_class_min_iou:.2f})")

            # Check early stopping
            if early_stopping is not None:
                if focus_class and focus_class in per_class_iou:
                    es_value = per_class_iou[focus_class]
                else:
                    es_value = mean_iou if early_stopping_metric == "mean_iou" else val_loss

                # Suppress early stopping if focus class hasn't reached minimum IoU
                if (focus_class and focus_class_min_iou > 0
                        and focus_class in per_class_iou
                        and per_class_iou[focus_class] < focus_class_min_iou):
                    # Reset early stopping counter -- don't stop yet
                    early_stopping.counter = 0

                if early_stopping(epoch + 1, es_value, model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Check for pause request
            if pause_flag and pause_flag.is_set():
                logger.info(f"Training paused at epoch {epoch+1}")
                # Run diagnostics at pause
                pause_warnings = _diagnostics.run_all_checks(training_history)
                if pause_warnings:
                    logger.info("=== Training Diagnostics (%d warnings) ===",
                                len(pause_warnings))
                checkpoint_save_path = self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopping=early_stopping,
                    training_history=training_history,
                    best_score=best_score,
                    best_score_mode=best_score_mode,
                    best_model_state=best_model_state,
                    model_type=model_type,
                    training_config={
                        "model_type": model_type,
                        "architecture": architecture,
                        "input_config": input_config,
                        "training_params": training_params,
                        "classes": classes,
                    },
                    normalization_stats=dataset_norm_stats,
                )
                # Free GPU memory during pause
                model = model.cpu()
                self.gpu_manager.clear_cache()
                self.gpu_manager.log_memory_status(prefix="Paused (GPU freed): ")

                pause_result = {
                    "status": "paused",
                    "checkpoint_path": checkpoint_save_path,
                    "epoch": epoch + 1,
                    "last_epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                    "per_class_iou": per_class_iou,
                    "per_class_loss": per_class_loss,
                    "mean_iou": mean_iou,
                    "best_epoch": best_epoch,
                    "best_mean_iou": best_score if best_score_mode == "max" else 0.0,
                    "epochs_trained": len(training_history),
                    "final_loss": val_loss,
                    "final_accuracy": accuracy,
                }
                if focus_class:
                    pause_result["focus_class_name"] = focus_class
                    pause_result["focus_class_iou"] = best_score
                    pause_result["focus_class_target_met"] = (
                        best_score >= focus_class_min_iou
                        if focus_class_min_iou > 0 else True
                    )
                    pause_result["focus_class_min_iou"] = focus_class_min_iou
                return pause_result

        # Handle cancellation: save both best-epoch and last-epoch models
        # so Java can let the user choose which to keep (or discard both).
        # The pause check above already returned for paused training.
        was_cancelled = (cancel_flag and cancel_flag.is_set()
                         and not (pause_flag and pause_flag.is_set()))
        if was_cancelled:
            logger.info("Saving progress after cancellation...")
            cancel_checkpoint = self._save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                early_stopping=early_stopping,
                training_history=training_history,
                best_score=best_score, best_score_mode=best_score_mode,
                best_model_state=best_model_state, model_type=model_type,
                training_config={
                    "model_type": model_type, "architecture": architecture,
                    "input_config": input_config,
                    "training_params": training_params, "classes": classes,
                },
                normalization_stats=dataset_norm_stats,
            )
            # Save last-epoch model (current model state)
            _cls_name = training_params.get("classifier_name")
            cancel_last_model_path = self._save_model(
                model=model, model_type=model_type,
                architecture=architecture, input_config=input_config,
                classes=classes, data_path=str(data_path),
                training_history=training_history,
                normalization_stats=dataset_norm_stats,
                classifier_name=_cls_name
            )
            logger.info("Saved last-epoch model after cancel: %s",
                        cancel_last_model_path)
            # Save best-epoch model if we have a separate best state
            cancel_best_model_path = ""
            if best_model_state is not None and best_epoch != len(training_history):
                model.load_state_dict(best_model_state)
                model = model.to(self.device)
                cancel_best_model_path = self._save_model(
                    model=model, model_type=model_type,
                    architecture=architecture, input_config=input_config,
                    classes=classes, data_path=str(data_path),
                    training_history=training_history,
                    normalization_stats=dataset_norm_stats,
                    classifier_name=_cls_name
                )
                logger.info("Saved best model (epoch %d) after cancel: %s",
                            best_epoch, cancel_best_model_path)
            else:
                # Best epoch IS the last epoch -- same model
                cancel_best_model_path = cancel_last_model_path
            # Clean up in-progress best model (proper models saved above)
            self._cleanup_best_in_progress(
                model_type, training_params.get("classifier_name"))
            # Free GPU memory
            model = model.cpu()
            self.gpu_manager.clear_cache()
            self.gpu_manager.log_memory_status(prefix="Cancelled (GPU freed): ")
            cancel_result = {
                "status": "cancelled",
                "model_path": cancel_best_model_path,
                "last_model_path": cancel_last_model_path,
                "checkpoint_path": cancel_checkpoint,
                "best_epoch": best_epoch,
                "best_mean_iou": best_mean_iou,
                "final_loss": best_loss,
                "final_accuracy": best_accuracy,
                "epoch": len(training_history),
                "last_epoch": len(training_history),
                "total_epochs": epochs,
                "epochs_trained": len(training_history),
            }
            if focus_class:
                cancel_result["focus_class_name"] = focus_class
                cancel_result["focus_class_iou"] = best_score
                cancel_result["focus_class_target_met"] = (
                    best_score >= focus_class_min_iou if focus_class_min_iou > 0 else True
                )
                cancel_result["focus_class_min_iou"] = focus_class_min_iou
            return cancel_result

        # Log final memory status
        self.gpu_manager.log_memory_status(prefix="Training complete: ")

        # Run all diagnostic checks at completion
        completion_warnings = _diagnostics.run_all_checks(training_history)
        if completion_warnings:
            logger.info("=== Training Diagnostics (%d warnings) ===",
                        len(completion_warnings))

        # Save checkpoint for potential "continue training" (before restoring best weights).
        # This preserves the last-epoch model/optimizer/scheduler state for seamless resume.
        completion_checkpoint_path = self._save_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            early_stopping=early_stopping, training_history=training_history,
            best_score=best_score, best_score_mode=best_score_mode,
            best_model_state=best_model_state, model_type=model_type,
            training_config={
                "model_type": model_type, "architecture": architecture,
                "input_config": input_config, "training_params": training_params,
                "classes": classes,
            },
            normalization_stats=dataset_norm_stats,
        )

        # Clear cache before restoring weights
        self.gpu_manager.clear_cache()

        # Restore best model weights before saving
        if early_stopping is not None and early_stopping.best_state is not None:
            early_stopping.restore_best(model)
            model = model.to(self.device)
        elif best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(self.device)
            logger.info(f"Restored best model weights (score={best_score:.4f})")

        # Save final model (include dataset normalization stats when available)
        model_path = self._save_model(
            model=model,
            model_type=model_type,
            architecture=architecture,
            input_config=input_config,
            classes=classes,
            data_path=str(data_path),
            training_history=training_history,
            normalization_stats=dataset_norm_stats,
            classifier_name=training_params.get("classifier_name")
        )

        # Clean up in-progress best model (final model saved above)
        self._cleanup_best_in_progress(
            model_type, training_params.get("classifier_name"))

        # Report focus class IoU so Java can warn when target not met
        result = {
            "model_path": model_path,
            "final_loss": best_loss,
            "final_accuracy": best_accuracy,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_mean_iou": best_mean_iou,
            "epochs_trained": len(training_history),
            "early_stopped": early_stopping.should_stop if early_stopping else False,
            "checkpoint_path": completion_checkpoint_path,
            "epoch": len(training_history),
            "last_epoch": len(training_history),
            "total_epochs": epochs,
        }
        if focus_class:
            result["focus_class_name"] = focus_class
            result["focus_class_iou"] = best_score  # best_score IS focus class IoU when focus_class is set
            result["focus_class_target_met"] = (
                best_score >= focus_class_min_iou if focus_class_min_iou > 0 else True
            )
            result["focus_class_min_iou"] = focus_class_min_iou
            if best_score < focus_class_min_iou and focus_class_min_iou > 0:
                logger.warning(
                    "Focus class '%s' IoU %.4f did not meet target %.4f",
                    focus_class, best_score, focus_class_min_iou)
                if best_score == 0.0:
                    logger.warning(
                        "Focus class '%s' had 0.0 IoU -- check that this class "
                        "has sufficient samples in the validation split",
                        focus_class)
        return result

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        scheduler_config: Dict[str, Any],
        epochs: int,
        steps_per_epoch: int
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler.

        Args:
            optimizer: The optimizer
            scheduler_type: Type of scheduler ("cosine", "step", "onecycle", "none")
            scheduler_config: Scheduler-specific configuration
            epochs: Total training epochs
            steps_per_epoch: Number of batches per epoch

        Returns:
            Learning rate scheduler or None
        """
        if scheduler_type == "none" or scheduler_type is None:
            logger.info("No learning rate scheduler")
            return None

        if scheduler_type == "cosine":
            # Cosine annealing with warm restarts
            T_0 = scheduler_config.get("T_0", max(epochs // 3, 1))  # Restart every T_0 epochs
            T_mult = scheduler_config.get("T_mult", 2)  # Double period after each restart
            eta_min = scheduler_config.get("eta_min", 1e-6)  # Minimum learning rate

            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            logger.info(f"Using CosineAnnealingWarmRestarts scheduler (T_0={T_0}, T_mult={T_mult})")
            return scheduler

        elif scheduler_type == "step":
            # Step decay
            step_size = scheduler_config.get("step_size", epochs // 3)
            gamma = scheduler_config.get("gamma", 0.1)

            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"Using StepLR scheduler (step_size={step_size}, gamma={gamma})")
            return scheduler

        elif scheduler_type == "onecycle":
            # One-cycle policy (good for finding optimal LR)
            # Support discriminative LRs: pass list of max_lr per param group
            config_max_lr = scheduler_config.get("max_lr", None)
            if len(optimizer.param_groups) > 1:
                if isinstance(config_max_lr, (int, float)):
                    # LR finder returned a single scalar -- scale per group
                    # by preserving the ratio each group has to the base LR.
                    # The base LR is the max among the initial group LRs
                    # (decoder/head), so encoder keeps its 1/10 ratio.
                    base_lr = max(g["lr"] for g in optimizer.param_groups)
                    if base_lr > 0:
                        max_lr = [
                            config_max_lr * (g["lr"] / base_lr)
                            for g in optimizer.param_groups
                        ]
                    else:
                        max_lr = [config_max_lr] * len(optimizer.param_groups)
                else:
                    # No finder result -- default to 10x each group's initial LR
                    max_lr = [g["lr"] * 10 for g in optimizer.param_groups]
            else:
                max_lr = config_max_lr if isinstance(config_max_lr, (int, float)) \
                    else optimizer.param_groups[0]["lr"] * 10

            accumulation_steps = scheduler_config.get("accumulation_steps", 1)
            total_steps = epochs * -(-steps_per_epoch // accumulation_steps)
            # Ensure at least 1 step
            total_steps = max(total_steps, 1)

            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=scheduler_config.get("pct_start", 0.3),
                anneal_strategy=scheduler_config.get("anneal_strategy", "cos"),
                div_factor=scheduler_config.get("div_factor", 25.0),
                final_div_factor=scheduler_config.get("final_div_factor", 1e4)
            )
            logger.info(f"Using OneCycleLR scheduler (max_lr={max_lr}, total_steps={total_steps})")
            return scheduler

        elif scheduler_type == "plateau":
            # ReduceLROnPlateau: reduce LR when metric stops improving
            mode = scheduler_config.get("mode", "max")  # "max" for mIoU
            factor = scheduler_config.get("factor", 0.5)
            patience = scheduler_config.get("patience", 10)
            min_lr = scheduler_config.get("min_lr", 1e-7)

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr
            )
            logger.info(f"Using ReduceLROnPlateau scheduler (mode={mode}, "
                       f"factor={factor}, patience={patience}, min_lr={min_lr})")
            return scheduler

        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using none")
            return None

    def _create_param_groups(self, model, learning_rate, discriminative_lr_ratio=0.1):
        """Create parameter groups with discriminative LRs for transfer learning.

        Fast.ai-style: encoder gets a fraction of the LR, decoder and head get full LR.
        This prevents catastrophic forgetting of pretrained features while allowing
        the decoder to adapt quickly.

        Args:
            model: The segmentation model
            learning_rate: Base learning rate for decoder/head
            discriminative_lr_ratio: Ratio applied to encoder LR (default 0.1 = 1/10th)

        Returns:
            List of param group dicts if multiple groups, else flat param list
        """
        encoder_params = []
        decoder_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            elif "decoder" in name:
                decoder_params.append(param)
            else:  # segmentation_head or other
                head_params.append(param)

        groups = []
        if encoder_params:
            groups.append({
                "params": encoder_params,
                "lr": learning_rate * discriminative_lr_ratio,
                "group_name": "encoder"
            })
        if decoder_params:
            groups.append({
                "params": decoder_params,
                "lr": learning_rate,
                "group_name": "decoder"
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": learning_rate,
                "group_name": "head"
            })

        if len(groups) > 1:
            return groups
        # Fall back to flat list if only one group
        return encoder_params + decoder_params + head_params

    @staticmethod
    def _format_loss_desc(
        loss_function: str, focal_gamma: float, ohem_hard_ratio: float,
        ohem_schedule: str = "fixed",
        ohem_hard_ratio_start: float = None,
        ohem_adaptive_floor: bool = False,
    ) -> str:
        """Format a human-readable loss description for the config summary."""
        desc = loss_function
        if loss_function in ("focal_dice", "focal"):
            desc += f" (gamma={focal_gamma})"
        if ohem_hard_ratio < 1.0:
            start = ohem_hard_ratio_start if ohem_hard_ratio_start is not None else ohem_hard_ratio
            anneals = start > ohem_hard_ratio
            mode = "adaptive floor" if ohem_adaptive_floor else "per-class proportional"
            if anneals:
                desc += (f" + OHEM (anneal {start * 100:.0f}%"
                         f" -> {ohem_hard_ratio * 100:.0f}%, {mode})")
            else:
                desc += f" + OHEM (keep {ohem_hard_ratio * 100:.0f}%, {mode})"
        return desc

    def find_learning_rate(self, model, train_loader, criterion,
                           start_lr=1e-7, end_lr=10, num_iter=100):
        """Fast.ai-style LR range test.

        Exponentially increases LR from start_lr to end_lr over num_iter steps,
        recording loss at each step. Suggests the LR at the steepest descent point.

        Args:
            model: The model to test
            train_loader: Training data loader
            criterion: Loss function
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to test

        Returns:
            Tuple of (suggested_lr, lrs, losses) or (None, [], []) on failure
        """
        # Save original model state to restore after test
        original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=start_lr, weight_decay=0.01)

        gamma = (end_lr / start_lr) ** (1 / num_iter)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        lrs, losses = [], []
        best_loss = float('inf')
        smoothed_loss = 0
        # Lighter smoothing for short sweeps -- 0.98 needs hundreds of steps
        # to settle; 0.9 responds in ~10 steps, better for 100-iter sweeps
        beta = 0.9

        model.train()
        data_iter = iter(train_loader)

        for i in range(num_iter):
            try:
                images, masks = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, masks = next(data_iter)

            images = images.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Exponential smoothing with bias correction
            smoothed_loss = beta * smoothed_loss + (1 - beta) * loss.item()
            corrected = smoothed_loss / (1 - beta ** (i + 1))

            if corrected < best_loss:
                best_loss = corrected
            if corrected > 4 * best_loss and i > 5:
                break  # Diverging

            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(corrected)

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Restore original model state
        model.load_state_dict(original_state)
        model = model.to(self.device)

        if len(lrs) < 10:
            logger.warning("LR finder: too few iterations to suggest LR")
            return None, lrs, losses

        suggested_lr = self._find_valley(lrs, losses)
        return suggested_lr, lrs, losses

    @staticmethod
    def _find_valley(lrs, losses):
        """Find the LR at the steepest descent in the loss curve (valley method).

        Returns None if the loss curve is too flat (no meaningful descent found),
        which lets the caller fall back to the user's configured learning rate.

        Args:
            lrs: List of learning rates
            losses: List of smoothed losses

        Returns:
            Suggested learning rate at steepest descent, or None if not found
        """
        if len(lrs) < 10:
            return None
        # Compute gradient (finite differences on log-lr scale)
        log_lrs = [np.log10(lr) for lr in lrs]
        gradients = []
        for i in range(1, len(losses) - 1):
            grad = (losses[i + 1] - losses[i - 1]) / (log_lrs[i + 1] - log_lrs[i - 1])
            gradients.append(grad)
        if not gradients:
            return None

        # Steepest descent = most negative gradient
        min_grad = float(np.min(gradients))
        min_idx = int(np.argmin(gradients))

        # Reject if curve is essentially flat -- the steepest gradient must be
        # meaningfully negative relative to the loss scale. If the total loss
        # drop is < 5% of the starting loss, there's no real signal.
        loss_range = max(losses) - min(losses)
        if loss_range < 0.05 * losses[0]:
            logger.info(
                f"LR finder: loss curve is flat (range {loss_range:.4f},"
                f" start {losses[0]:.4f}) -- no reliable suggestion")
            return None

        # Also reject if the steepest descent is positive (loss only goes up)
        if min_grad >= 0:
            logger.info("LR finder: no descent found in loss curve")
            return None

        # The LR is at idx+1 (offset from gradient computation)
        suggested = lrs[min_idx + 1]
        return suggested

    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        early_stopping,
        training_history: List[Dict[str, Any]],
        best_score: float,
        best_score_mode: str,
        best_model_state: Optional[Dict],
        model_type: str,
        training_config: Dict[str, Any],
        normalization_stats: Optional[List[Dict[str, float]]] = None
    ) -> str:
        """Save a training checkpoint for pause/resume.

        Returns:
            Path to the saved checkpoint file.
        """
        import time

        checkpoint_dir = Path(os.path.expanduser("~/.dlclassifier/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        import re
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cls_name = training_config.get("training_params", {}).get(
            "classifier_name", "")
        if cls_name:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", cls_name).lower()
            checkpoint_path = checkpoint_dir / f"checkpoint_{safe_name}_{timestamp}.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_{model_type}_{timestamp}.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": training_history,
            "best_score": best_score,
            "best_score_mode": best_score_mode,
            "training_config": training_config,
        }

        if best_model_state is not None:
            checkpoint["best_model_state"] = best_model_state

        if normalization_stats is not None:
            checkpoint["normalization_stats"] = normalization_stats

        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            # Save scheduler state for all types except OneCycleLR
            # (which must be recreated with remaining steps on resume).
            # ReduceOnPlateau has internal state (best, num_bad_epochs,
            # last_epoch, cooldown_counter) that should be preserved.
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if early_stopping is not None:
            checkpoint["early_stopping"] = {
                "best_score": early_stopping.best_score,
                "mode": early_stopping.mode,
                "best_epoch": early_stopping.best_epoch,
                "counter": early_stopping.counter,
                "best_state": early_stopping.best_state,
            }

        torch.save(checkpoint, str(checkpoint_path))
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Write companion metadata.json so the checkpoint is self-documenting
        # and recoverable without parsing the .pt file
        try:
            meta = {
                "type": "checkpoint",
                "model_type": model_type,
                "architecture": training_config.get("architecture", {}),
                "classes": training_config.get("classes", []),
                "input_config": training_config.get("input_config", {}),
                "training_params": training_config.get("training_params", {}),
                "epochs_completed": len(training_history),
                "best_score": best_score,
                "classifier_name": training_config.get("training_params", {}).get(
                    "classifier_name", ""),
            }
            meta_path = checkpoint_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as meta_err:
            logger.warning(f"Failed to write checkpoint metadata: {meta_err}")

        return str(checkpoint_path)

    def _save_best_in_progress(
        self,
        model,
        optimizer,
        scheduler,
        early_stopping,
        best_model_state: Dict[str, Any],
        model_type: str,
        best_epoch: int,
        best_score: float,
        best_score_mode: str,
        training_config: Dict[str, Any],
        training_history: List[Dict[str, Any]],
        normalization_stats: Optional[List[Dict[str, float]]] = None
    ) -> str:
        """Save a full training checkpoint to disk for crash recovery.

        Called every time a new best epoch is found during training.
        The file is overwritten each time, keeping only the latest best.
        If training is interrupted (crash, power loss), this file can be
        used to either recover the best model (finalize_training.py) or
        resume training (same format as _save_checkpoint).

        The filename includes the classifier name (if available) so that
        multiple concurrent or sequential training runs don't overwrite
        each other's checkpoints.

        Returns:
            Path to the saved file.
        """
        checkpoint_dir = Path(os.path.expanduser("~/.dlclassifier/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use classifier_name for unique per-model checkpoint filenames.
        # Falls back to model_type if classifier_name is not available.
        import re
        cls_name = training_config.get("training_params", {}).get(
            "classifier_name", "")
        if cls_name:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", cls_name).lower()
            save_path = checkpoint_dir / f"best_in_progress_{safe_name}.pt"
        else:
            save_path = checkpoint_dir / f"best_in_progress_{model_type}.pt"

        data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_model_state": best_model_state,
            "best_score": best_score,
            "best_score_mode": best_score_mode,
            "training_config": training_config,
            "training_history": training_history,
        }

        if normalization_stats is not None:
            data["normalization_stats"] = normalization_stats

        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            data["scheduler_state_dict"] = scheduler.state_dict()

        if early_stopping is not None:
            data["early_stopping"] = {
                "best_score": early_stopping.best_score,
                "mode": early_stopping.mode,
                "best_epoch": early_stopping.best_epoch,
                "counter": early_stopping.counter,
                "best_state": early_stopping.best_state,
            }

        torch.save(data, str(save_path))
        logger.info(f"  Best checkpoint saved to disk (epoch {best_epoch}): {save_path}")
        return str(save_path)

    def _cleanup_best_in_progress(self, model_type: str,
                                   classifier_name: Optional[str] = None) -> None:
        """Remove the in-progress best model file after training completes normally.

        Called after successful completion or cancellation, when the final model
        has been saved properly via _save_model().
        """
        import re
        checkpoint_dir = Path(os.path.expanduser("~/.dlclassifier/checkpoints"))
        if classifier_name:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", classifier_name).lower()
            save_path = checkpoint_dir / f"best_in_progress_{safe_name}.pt"
        else:
            save_path = checkpoint_dir / f"best_in_progress_{model_type}.pt"
        if save_path.exists():
            save_path.unlink()
            logger.debug(f"Cleaned up in-progress best model: {save_path}")

    def _try_training_compile(self, model, model_type: str,
                              architecture: Dict[str, Any]):
        """Attempt to wrap a training model with torch.compile().

        Experimental and opt-in.  Requires Linux + CUDA + sm_70+ + Triton.
        Silently returns the eager model on any failure.  Warns when the
        architecture has a known graph-break risk so the user gets a clear
        signal in the training log.

        Args:
            model: The model on its training device.
            model_type: "unet", "muvit", "tiny-unet", "fast-pretrained", ...
            architecture: architecture dict from the request.

        Returns:
            Either the original model (on failure / unsupported) or the
            torch.compile()-wrapped model.
        """
        import platform

        if not hasattr(torch, "compile"):
            logger.info("torch.compile unavailable in this PyTorch build; "
                        "using eager mode.")
            return model
        if platform.system() != "Linux":
            logger.info("torch.compile is Linux-gated in this build "
                        "(Windows/macOS: Triton support is incomplete); "
                        "using eager mode.")
            return model
        # self.device is a string in TrainingService.
        _compile_device_str = (
            self.device if isinstance(self.device, str) else self.device.type
        )
        if _compile_device_str != "cuda":
            logger.info("torch.compile requires CUDA; device=%s", self.device)
            return model
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] < 7:
                logger.info(
                    "torch.compile skipped: GPU capability %d.%d < 7.0 "
                    "(Triton Inductor requires Volta or newer).",
                    cap[0], cap[1],
                )
                return model
        except Exception as e:
            logger.debug("Could not query device capability: %s", e)
        try:
            import triton  # noqa: F401
        except ImportError:
            logger.info("torch.compile requires triton -- not installed. "
                        "Falling back to eager mode.")
            return model

        # Warn about known problematic combinations.
        brn_risk = (
            model_type != "muvit"
            and not (model_type == "tiny-unet"
                     and str(architecture.get("norm", "brn")) == "gn")
        )
        if brn_risk:
            logger.warning(
                "torch.compile enabled with BatchRenorm in the model "
                "(model_type=%s, norm=%s). Expect graph-breaks around the "
                "BRN buffer updates; the speedup may be limited. For the "
                "best result, use tiny-unet with norm=gn.",
                model_type, architecture.get("norm", "brn"),
            )

        try:
            compiled = torch.compile(model, mode="reduce-overhead",
                                     dynamic=False)
            logger.info("Training model wrapped with torch.compile "
                        "(mode=reduce-overhead, dynamic=False)")
            return compiled
        except Exception as e:
            logger.warning(
                "torch.compile failed (%s); training continues in eager mode.",
                e,
            )
            return model

    def _create_model(
        self,
        model_type: str,
        architecture: Dict[str, Any],
        num_channels: int,
        num_classes: int
    ):
        """Create a segmentation model."""
        try:
            import segmentation_models_pytorch as smp
            from .pretrained_models import PretrainedModelsService, get_pretrained_service

            encoder_name = architecture.get("backbone", "resnet34")
            # Default to False (no pretrained weights) for safety -- callers
            # must explicitly opt in.  Java always sets this via the
            # architecture dict; the default only matters if a new code path
            # omits it.  Matches the default in the training code (~line 1200).
            encoder_weights = "imagenet" if architecture.get("use_pretrained", False) else None

            # Map model types to smp classes
            model_map = {
                "unet": smp.Unet,
                "unetplusplus": smp.UnetPlusPlus,
                "deeplabv3": smp.DeepLabV3,
                "deeplabv3plus": smp.DeepLabV3Plus,
                "fpn": smp.FPN,
                "pspnet": smp.PSPNet,
                "manet": smp.MAnet,
                "linknet": smp.Linknet,
                "pan": smp.PAN,
            }

            # Tiny UNet: hand-rolled depthwise-separable U-Net for fast
            # training on simple 2-5 class tasks. Defaults to BatchRenorm.
            if model_type == "tiny-unet":
                from ..models.tiny_unet import TinyUNet
                model = TinyUNet(
                    in_channels=num_channels,
                    n_classes=num_classes,
                    base=int(architecture.get("base", 16)),
                    depth=int(architecture.get("depth", 4)),
                    norm=str(architecture.get("norm", "brn")),
                )
                logger.info(
                    "Created Tiny UNet model (preset=%s, base=%d, depth=%d, norm=%s)",
                    architecture.get("backbone", "tiny-16x4"),
                    int(architecture.get("base", 16)),
                    int(architecture.get("depth", 4)),
                    architecture.get("norm", "brn"),
                )
                return model

            # Fast Pretrained: SMP U-Net with a small ImageNet-pretrained
            # mobile encoder (EfficientNet-Lite0 or MobileNetV3-Small) and
            # a scaled-down decoder.  Intended for small RGB H&E datasets
            # where ImageNet priors are worth keeping.
            if model_type == "fast-pretrained":
                fp_encoder = architecture.get(
                    "backbone", "timm-tf_efficientnet_lite0"
                )
                fp_decoder = architecture.get(
                    "decoder_channels", [128, 64, 32, 16, 8]
                )
                model = smp.Unet(
                    encoder_name=fp_encoder,
                    encoder_weights=encoder_weights,
                    in_channels=num_channels,
                    classes=num_classes,
                    decoder_channels=fp_decoder,
                )
                logger.info(
                    "Created Fast Pretrained UNet (encoder=%s, decoder=%s, "
                    "pretrained=%s, in_channels=%d, classes=%d)",
                    fp_encoder, fp_decoder,
                    encoder_weights is not None,
                    num_channels, num_classes,
                )
                return model

            # MuViT transformer: delegate to dedicated factory
            if model_type == "muvit":
                from .muvit_model import create_muvit_model
                model = create_muvit_model(
                    architecture=architecture,
                    num_channels=num_channels,
                    num_classes=num_classes,
                )
                cfg = architecture.get("model_config",
                                      architecture.get("backbone", "muvit-base"))
                logger.info("Created MuViT model (config=%s)", cfg)
                return model

            if model_type not in model_map:
                raise ValueError(f"Unknown model type: {model_type}. "
                               f"Available: {list(model_map.keys()) + ['muvit']}")

            # Check if this is a histology-pretrained encoder
            if encoder_name in PretrainedModelsService.HISTOLOGY_ENCODERS:
                smp_encoder, hub_id = PretrainedModelsService.HISTOLOGY_ENCODERS[encoder_name]

                # Create model with imagenet weights first (correct architecture)
                model = model_map[model_type](
                    encoder_name=smp_encoder,
                    encoder_weights="imagenet",
                    in_channels=num_channels,
                    classes=num_classes
                )

                # Replace encoder weights with histology-pretrained weights
                pretrained_service = get_pretrained_service()
                pretrained_service._load_histology_weights(
                    model, hub_id, smp_encoder, num_channels)

                logger.info(f"Created {model_type} model with histology encoder "
                           f"{encoder_name} (weights: {hub_id})")
                return model

            # Check if this is a foundation model encoder (on-demand download from HuggingFace)
            # Integration approach inspired by LazySlide (MIT License).
            # Zheng, Y. et al. Nature Methods (2026). https://doi.org/10.1038/s41592-026-03044-7
            if encoder_name in PretrainedModelsService.FOUNDATION_ENCODERS:
                _, hub_id = PretrainedModelsService.FOUNDATION_ENCODERS[encoder_name]

                logger.info("Loading foundation model encoder: %s "
                            "(downloading on first use, may require HF_TOKEN for gated models)",
                            encoder_name)

                # Use SMP's timm universal encoder (tu-) prefix for foundation model integration.
                # This uses timm's feature extraction mode to produce multi-scale outputs
                # compatible with UNet/FPN decoders.
                smp_encoder_name = "tu-" + hub_id.replace("hf_hub:", "")
                try:
                    model = model_map[model_type](
                        encoder_name=smp_encoder_name,
                        encoder_weights=None,
                        in_channels=num_channels,
                        classes=num_classes
                    )
                    logger.info("Created %s model with foundation encoder %s via SMP timm adapter",
                                model_type, encoder_name)
                    return model
                except Exception as e:
                    raise ValueError(
                        f"Foundation model encoder '{encoder_name}' failed to "
                        f"load via SMP timm adapter (tu-{hub_id}): {e}. "
                        f"This may be caused by: (1) the model requires a "
                        f"HuggingFace token (set HF_TOKEN environment variable), "
                        f"(2) network issues downloading the model, or "
                        f"(3) incompatible timm/SMP versions. "
                        f"Try a different encoder or check the Python Console "
                        f"for details."
                    ) from e

            model = model_map[model_type](
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=num_channels,
                classes=num_classes
            )

            logger.info(f"Created {model_type} model with {encoder_name} encoder "
                       f"(pretrained={encoder_weights is not None})")

            return model

        except ImportError:
            logger.error("segmentation_models_pytorch not installed")
            raise

    def _save_model(
        self,
        model,
        model_type: str,
        architecture: Dict[str, Any],
        input_config: Dict[str, Any],
        classes: List[str],
        data_path: str,
        training_history: Optional[List[Dict[str, Any]]] = None,
        normalization_stats: Optional[List[Dict[str, float]]] = None,
        classifier_name: Optional[str] = None
    ) -> str:
        """Save the trained model."""
        import time

        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_{timestamp}"
        output_dir = Path(os.path.expanduser("~/.dlclassifier/models")) / model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Export to ONNX (skip for MuViT -- custom ops not ONNX-compatible)
        if model_type == "muvit":
            logger.info("Skipping ONNX export for MuViT model (not supported)")
        else:
            try:
                model.eval()
                input_size = architecture.get("input_size", [512, 512])
                # Detect actual in_channels from model weights (handles context_scale > 1
                # where model has 2*C channels but input_config.num_channels is C)
                try:
                    actual_channels = model.encoder.conv1.weight.shape[1]
                except AttributeError:
                    actual_channels = input_config["num_channels"]
                dummy_input = torch.randn(1, actual_channels, input_size[0], input_size[1])
                dummy_input = dummy_input.to(self.device)

                onnx_path = output_dir / "model.onnx"
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    opset_version=14,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 2: "height", 3: "width"}
                    }
                )
                logger.info(f"Exported ONNX model to {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

        # Read class colors from training config.json if available
        class_colors = {}
        try:
            config_path = Path(data_path) / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    train_config = json.load(f)
                for cls_info in train_config.get("classes", []):
                    if "color" in cls_info and "name" in cls_info:
                        class_colors[cls_info["name"]] = cls_info["color"]
        except Exception as e:
            logger.warning(f"Could not read class colors from config.json: {e}")

        # Build class list with colors
        class_list = []
        for i, c in enumerate(classes):
            entry = {"index": i, "name": c}
            if c in class_colors:
                entry["color"] = class_colors[c]
            class_list.append(entry)

        # Save metadata
        # Compute effective input channels: doubled when context_scale > 1
        # because detail + context tiles are concatenated along channel axis.
        # Store this explicitly so any consumer can read the actual model
        # input size without needing to know the context_scale doubling rule.
        base_ch = int(architecture.get("input_channels",
                                       input_config.get("num_channels", 3)))
        ctx_scale = int(architecture.get("context_scale", 1))
        effective_ch = base_ch * 2 if ctx_scale > 1 else base_ch

        # Embed training dataset stats into input_config.normalization so
        # the metadata is self-contained for normalization. At runtime Java
        # rebuilds input_config from scratch (via buildInputConfig) with
        # image-level stats, so these persisted values don't interfere.
        saved_input_config = dict(input_config)
        if normalization_stats:
            saved_norm = dict(saved_input_config.get("normalization", {}))
            saved_norm["precomputed"] = True
            saved_norm["channel_stats"] = normalization_stats
            saved_input_config["normalization"] = saved_norm

        display_name = classifier_name if classifier_name else f"{model_type.upper()} Classifier"
        metadata = {
            "id": model_id,
            "name": display_name,
            "architecture": {
                "type": model_type,
                "use_batchrenorm": True,
                **architecture,
                "effective_input_channels": effective_ch
            },
            "input_config": saved_input_config,
            "classes": class_list
        }

        # Also store normalization_stats at top level for backward compat
        if normalization_stats:
            metadata["normalization_stats"] = normalization_stats
            logger.info(f"Saved normalization stats for {len(normalization_stats)} channels")

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save training history if provided
        if training_history:
            history_path = output_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(training_history, f, indent=2)
            logger.info(f"Saved training history ({len(training_history)} epochs)")

        logger.info(f"Model saved to {output_dir}")
        return str(output_dir)
