"""Self-supervised pretraining service for SMP encoder backbones.

Supports SimCLR (contrastive) and BYOL (self-distillation) methods
for CNN encoders available through Segmentation Models PyTorch (SMP):
ResNet, EfficientNet, MobileNet, etc.

After pretraining, encoder weights are saved with SMP-native key
prefixes (encoder.*) so they load directly into an SMP segmentation
model via the existing shape-matched partial loading mechanism in
training_service.py. No prefix stripping is needed -- unlike MAE
which saves with a mae.* prefix.

Output format:
  model.pt      -- {"model_state_dict": {encoder.* keys only}}
  metadata.json -- architecture, method, normalization stats
"""
import copy
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .gpu_manager import GPUManager, get_gpu_manager

logger = logging.getLogger(__name__)

# Try to import albumentations for SSL augmentations
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("albumentations not available -- SSL augmentations disabled")


# ==================== Augmentation ====================

class PerChannelIntensityJitter(A.ImageOnlyTransform):
    """Per-channel random brightness/contrast for multi-channel images.

    Each channel receives independent random scaling, appropriate for
    fluorescence and multi-spectral microscopy where channels are
    physically independent signals.
    """

    def __init__(self, brightness_limit=0.4, contrast_limit=0.4,
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
                0, 255 if img.dtype == np.uint8 else max(ch.max() * 2, 1.0))
        return result.astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit")


# Ruifrok & Johnston (2001) RGB <-> HED stain decomposition matrix for H&E.
# Used by HEDStainJitter below. ASCII-only; no Unicode in code.
_HED_RGB_FROM_HED = np.array([
    [0.65, 0.70, 0.29],   # H
    [0.07, 0.99, 0.11],   # E
    [0.27, 0.57, 0.78],   # residual
], dtype=np.float32)
_HED_HED_FROM_RGB = np.linalg.inv(_HED_RGB_FROM_HED).astype(np.float32)


class HEDStainJitter(A.ImageOnlyTransform):
    """Stain-aware jitter for H&E in HED color space.

    Decomposes RGB into hematoxylin/eosin/residual channels, perturbs each
    channel by a random multiplicative + additive shift, recomposes RGB.
    Models realistic stain variation across slides without breaking tissue
    morphology (unlike generic hue jitter).

    Args:
        sigma: Per-channel multiplicative jitter std (typical 0.02-0.05).
        bias: Per-channel additive jitter std (typical 0.005-0.02).
    """

    def __init__(self, sigma=0.05, bias=0.01,
                 always_apply=False, p=0.8):
        super().__init__(always_apply, p)
        self.sigma = float(sigma)
        self.bias = float(bias)

    def apply(self, img, **params):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        is_uint8 = img.dtype == np.uint8
        rgb = img.astype(np.float32) / (255.0 if is_uint8 else 1.0)
        # Avoid log(0)
        rgb = np.clip(rgb, 1e-6, 1.0)
        od = -np.log(rgb)
        hed = od @ _HED_HED_FROM_RGB.T
        alpha = 1.0 + np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        beta = np.random.uniform(-self.bias, self.bias, size=3).astype(np.float32)
        hed = hed * alpha + beta
        od_out = hed @ _HED_RGB_FROM_HED.T
        rgb_out = np.exp(-od_out)
        rgb_out = np.clip(rgb_out, 0.0, 1.0)
        if is_uint8:
            return (rgb_out * 255.0).astype(np.uint8)
        return rgb_out.astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("sigma", "bias")


def get_ssl_augmentation(
    tile_size: int = 256,
    intensity_mode: str = "brightfield",
    num_channels: int = 3,
    stain_aug: bool = True,
) -> A.Compose:
    """Create strong augmentation pipeline for SSL pretraining.

    SSL methods rely on strong augmentations to learn invariant
    representations. This pipeline is stronger than standard
    supervised training augmentations.

    Args:
        tile_size: Output spatial size after augmentation.
        intensity_mode: "brightfield", "fluorescence", or "none".
        num_channels: Number of image channels (affects color transforms).

    Returns:
        albumentations.Compose pipeline.
    """
    spatial = [
        A.RandomResizedCrop(
            size=(tile_size, tile_size),
            scale=(0.2, 1.0),
            ratio=(0.75, 1.333),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    # Intensity transforms depend on imaging modality and channel count
    intensity = []
    if intensity_mode == "brightfield" and num_channels == 3:
        # H&E-tuned: lower hue (full hue rotation breaks stain semantics),
        # very rare ToGray (collapses both stains -> trivial pretext task),
        # plus optional stain-space jitter which is the right axis of
        # variation for histology.
        intensity = [
            A.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.2, hue=0.04, p=0.8),
            A.ToGray(p=0.05),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        ]
        if stain_aug:
            intensity.append(HEDStainJitter(sigma=0.05, bias=0.01, p=0.8))
    elif intensity_mode == "fluorescence" or num_channels != 3:
        intensity = [
            PerChannelIntensityJitter(
                brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        ]
    else:
        # "none" -- still apply mild jitter for diversity
        intensity = [
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ]

    blur_noise = [
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
        A.GaussNoise(std_range=(0.02, 0.15), p=0.3),
    ]

    return A.Compose(spatial + intensity + blur_noise)


# ==================== Dataset ====================

class SSLImageDataset(Dataset):
    """Dataset for SSL pretraining that returns pairs of augmented views.

    For each image, independently applies a stochastic augmentation
    pipeline twice to produce (view1, view2). Supports in-memory
    preloading for performance with num_workers=0.
    """

    SUPPORTED_EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.raw'}

    def __init__(
        self,
        image_dir: str,
        tile_size: int = 256,
        normalize_stats: Optional[Dict] = None,
        augmentation: Optional[A.Compose] = None,
        preload: bool = True,
    ):
        self.tile_size = tile_size
        self.normalize_stats = normalize_stats
        self.augmentation = augmentation
        self.preload = preload

        image_dir = Path(image_dir)
        candidates = sorted([
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
               and not p.name.startswith('.')
        ])

        if not candidates:
            raise ValueError("No image files found in %s" % image_dir)

        # Validate file headers
        self.image_paths = []
        skipped = []
        for p in candidates:
            if self._validate_file(p):
                self.image_paths.append(p)
            else:
                skipped.append(p.name)

        if skipped:
            logger.warning(
                "Skipped %d of %d files (invalid headers): %s",
                len(skipped), len(candidates),
                ", ".join(skipped[:10]) + ("..." if len(skipped) > 10 else ""))

        if not self.image_paths:
            raise ValueError(
                "No valid image files found in %s "
                "(%d files had invalid headers)" % (image_dir, len(skipped)))

        logger.info("SSLImageDataset: %d valid images in %s",
                     len(self.image_paths), image_dir)

        # In-memory preload for performance (critical with num_workers=0)
        self._cache = None
        if preload:
            self._preload_images()

    def _preload_images(self):
        """Load all images into memory.

        Refuses to preload when memory headroom is unknown or insufficient.
        The previous behavior (assume infinite memory when psutil is missing)
        produced silent disk-paging stalls on Windows hosts without psutil
        installed -- 20k tiles at 512x512 is ~15 GB decoded, which routinely
        tipped systems with 32-48 GB total RAM into the page file once the
        JVM, GPU driver, and dataloader workers added their share.
        """
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            logger.warning(
                "psutil not available -- cannot verify host memory "
                "headroom. Falling back to disk streaming to avoid "
                "silently exhausting system RAM. Install psutil to "
                "enable in-memory preload.")
            self.preload = False
            return

        # Estimate decoded-bytes-per-tile from the actual tile_size instead
        # of a flat 1 MB/tile. SSL targets 512x512x3 uint8 = 0.75 MB raw,
        # but PyTorch tensors hold a float copy after transform (3 MB), and
        # dataloader workers may briefly hold an extra augmented copy.
        # Assume 3 channels (typical for SSL) and ~4x the raw size to cover
        # transient peaks. Channels=1 makes the estimate 3x conservative,
        # which biases toward the safer disk-streaming path.
        bytes_per_tile = self.tile_size * self.tile_size * 3 * 4
        estimated_mb = len(self.image_paths) * bytes_per_tile / (1024 * 1024)
        # Reserve 25% of available RAM for the OS, JVM, GPU driver, and
        # transient buffers; only preload if dataset fits in the rest.
        budget_mb = available_mb * 0.50
        if estimated_mb > budget_mb:
            logger.warning(
                "Dataset too large for in-memory preload "
                "(~%.0f MB, %.0f MB available, %.0f MB budget). "
                "Using disk streaming instead.",
                estimated_mb, available_mb, budget_mb)
            self.preload = False
            return

        self._cache = []
        loaded = 0
        for p in self.image_paths:
            try:
                img = self._load_image(p)
                if img.ndim == 2:
                    img = img[..., np.newaxis]
                self._cache.append(img)
                loaded += 1
            except Exception as e:
                logger.warning("Failed to preload %s: %s", p.name, e)
                self._cache.append(None)

        logger.info("Preloaded %d/%d images into memory",
                     loaded, len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _validate_file(path: Path) -> bool:
        """Quick header check -- rejects files that are not valid images."""
        try:
            with open(path, 'rb') as f:
                header = f.read(12)
            if len(header) < 4:
                return False
            suffix = path.suffix.lower()
            if suffix in ('.tif', '.tiff'):
                return header[:4] in (
                    b'II*\x00', b'MM\x00*',
                    b'II+\x00', b'MM\x00+',
                )
            if suffix == '.png':
                return header[:4] == b'\x89PNG'
            if suffix in ('.jpg', '.jpeg'):
                return header[:2] == b'\xff\xd8'
            if suffix == '.bmp':
                return header[:2] == b'BM'
            if suffix == '.raw':
                return len(header) >= 12
            return True
        except OSError:
            return False

    def _get_image(self, idx: int) -> np.ndarray:
        """Get image by index, from cache or disk."""
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx].copy()

        try:
            img = self._load_image(self.image_paths[idx])
        except Exception as e:
            logger.warning("Failed to load %s: %s -- using random substitute",
                           self.image_paths[idx].name, e)
            for _ in range(5):
                alt = np.random.randint(0, len(self.image_paths))
                if alt != idx:
                    try:
                        img = self._load_image(self.image_paths[alt])
                        break
                    except Exception:
                        continue
            else:
                img = np.zeros(
                    (self.tile_size, self.tile_size, 1), dtype=np.float32)

        if img.ndim == 2:
            img = img[..., np.newaxis]
        return img

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._get_image(idx)

        # Ensure image is at least tile_size in both dimensions
        h, w = img.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            pad_h = max(0, self.tile_size - h)
            pad_w = max(0, self.tile_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        # Apply augmentation twice independently to get two views
        if self.augmentation is not None:
            # albumentations expects uint8 or float32 HWC
            view1 = self.augmentation(image=img)["image"]
            view2 = self.augmentation(image=img)["image"]
        else:
            # Fallback: random crop + flip
            view1 = self._basic_augment(img)
            view2 = self._basic_augment(img)

        # Normalize both views
        view1 = self._normalize(view1)
        view2 = self._normalize(view2)

        # HWC -> CHW float32
        v1 = torch.from_numpy(view1.transpose(2, 0, 1).astype(np.float32))
        v2 = torch.from_numpy(view2.transpose(2, 0, 1).astype(np.float32))
        return v1, v2

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Apply channel-wise normalization."""
        if self.normalize_stats:
            mean = np.array(self.normalize_stats.get("mean", [0.5]),
                            dtype=np.float32)
            std = np.array(self.normalize_stats.get("std", [0.5]),
                           dtype=np.float32)
            std = np.maximum(std, 1e-6)
            img = (img.astype(np.float32) - mean) / std
        return img

    def _basic_augment(self, img: np.ndarray) -> np.ndarray:
        """Simple fallback augmentation when albumentations unavailable."""
        h, w = img.shape[:2]
        # Random crop to tile_size
        if h > self.tile_size or w > self.tile_size:
            y = np.random.randint(0, max(1, h - self.tile_size))
            x = np.random.randint(0, max(1, w - self.tile_size))
            img = img[y:y + self.tile_size, x:x + self.tile_size]
        # Random flip and rotate
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=0).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k=k, axes=(0, 1)).copy()
        return img

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image file as float32 HWC array."""
        path_str = str(path)

        if path_str.endswith('.raw'):
            with open(path_str, 'rb') as f:
                header = np.frombuffer(f.read(12), dtype=np.int32)
                h, w, c = int(header[0]), int(header[1]), int(header[2])
                data = np.frombuffer(f.read(), dtype=np.float32)
            return data.reshape(h, w, c).copy()

        if path_str.endswith(('.tif', '.tiff')):
            try:
                import tifffile
                arr = tifffile.imread(path_str).astype(np.float32)
                if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
                    arr = arr.transpose(1, 2, 0)
                elif arr.ndim == 2:
                    arr = arr[..., np.newaxis]
                return arr
            except ImportError:
                pass

        from PIL import Image
        pil_img = Image.open(path_str)
        return np.array(pil_img, dtype=np.float32) / 255.0

    @property
    def n_channels(self) -> int:
        """Number of channels from first image."""
        img = self._get_image(0)
        return img.shape[2] if img.ndim == 3 else 1


# ==================== Projection Head ====================

class ProjectionHead(nn.Module):
    """MLP projection head for SSL methods.

    SimCLR uses: encoder -> projector (2-layer MLP with BN + ReLU).
    BYOL adds:   projector -> predictor (same architecture).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048,
                 out_dim: int = 256, use_bn: bool = True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== Loss Functions ====================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) for SimCLR.

    Args:
        z1: Projections from view 1, shape (B, D).
        z2: Projections from view 2, shape (B, D).
        temperature: Temperature scaling (lower = sharper).

    Returns:
        Scalar loss.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]

    # Concatenate: [z1_0, ..., z1_B-1, z2_0, ..., z2_B-1]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """BYOL regression loss (negative cosine similarity).

    Args:
        p: Predictions from online network, shape (B, D).
        z: Projections from target network (detached), shape (B, D).

    Returns:
        Scalar loss.
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()


# ==================== Model Wrappers ====================

class SimCLRModel(nn.Module):
    """SimCLR wrapper: encoder + global avg pool + projection head."""

    def __init__(self, encoder: nn.Module, encoder_out_dim: int,
                 hidden_dim: int = 2048, projection_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = ProjectionHead(
            encoder_out_dim, hidden_dim, projection_dim, use_bn=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning projection vector.

        Args:
            x: (B, C, H, W) input images.

        Returns:
            z: (B, projection_dim) projection vectors.
        """
        # SMP encoders return list of feature maps at different scales
        features = self.encoder(x)
        h = self.pool(features[-1]).flatten(1)  # Deepest features
        return self.projector(h)


class BYOLModel(nn.Module):
    """BYOL wrapper: online (encoder+projector+predictor) + target (EMA)."""

    def __init__(self, encoder: nn.Module, encoder_out_dim: int,
                 hidden_dim: int = 2048, projection_dim: int = 256):
        super().__init__()
        # Online network
        self.online_encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.online_projector = ProjectionHead(
            encoder_out_dim, hidden_dim, projection_dim, use_bn=True)
        self.predictor = ProjectionHead(
            projection_dim, hidden_dim, projection_dim, use_bn=True)

        # Target network (deep copy, no gradients)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, ema_decay: float):
        """Exponential moving average update of target from online."""
        for op, tp in zip(self.online_encoder.parameters(),
                          self.target_encoder.parameters()):
            tp.data = ema_decay * tp.data + (1 - ema_decay) * op.data
        for op, tp in zip(self.online_projector.parameters(),
                          self.target_projector.parameters()):
            tp.data = ema_decay * tp.data + (1 - ema_decay) * op.data

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Forward pass returning (prediction, target_projection) pairs.

        Args:
            x1, x2: Two augmented views, each (B, C, H, W).

        Returns:
            (p1, z2): online prediction for view1, target projection for view2.
        """
        # Online path (view 1)
        h1 = self.pool(self.online_encoder(x1)[-1]).flatten(1)
        z1 = self.online_projector(h1)
        p1 = self.predictor(z1)

        # Target path (view 2, no gradients)
        with torch.no_grad():
            h2 = self.pool(self.target_encoder(x2)[-1]).flatten(1)
            z2 = self.target_projector(h2)

        return p1, z2


# ==================== Pretraining Service ====================

class SSLPretrainingService:
    """Service for SSL pretraining of SMP encoder backbones.

    Supports SimCLR (contrastive) and BYOL (self-distillation) methods.
    Trains only the encoder backbone; the resulting weights transfer to
    any SMP segmentation model with the matching backbone via the
    existing pretrained weight loading mechanism.
    """

    def __init__(
        self,
        device: str = "auto",
        gpu_manager: Optional[GPUManager] = None,
    ):
        self.gpu_manager = gpu_manager or get_gpu_manager()
        if device == "auto":
            self.device = self.gpu_manager.device
            self._device_str = self.gpu_manager.device_type
        else:
            self._device_str = device
            self.device = torch.device(device)

        logger.info("SSLPretrainingService on device: %s", self._device_str)

    def pretrain(
        self,
        config: Dict[str, Any],
        data_path: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
        setup_callback: Optional[Callable] = None,
        cancel_flag: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Run SSL pretraining.

        Args:
            config: Dict with keys:
                method ("simclr" or "byol"), encoder_name, epochs,
                batch_size, learning_rate, weight_decay, warmup_epochs,
                temperature (SimCLR), ema_decay (BYOL), projection_dim,
                hidden_dim, tile_size, intensity_mode, checkpoint_interval
            data_path: Directory of image tiles
            output_dir: Where to save pretrained encoder weights
            progress_callback: fn(epoch, total, loss, lr)
            setup_callback: fn(phase, data=None)
            cancel_flag: threading.Event to signal cancellation

        Returns:
            Dict with status, encoder_path, epochs_completed,
            final_loss, best_loss
        """

        def _report(phase, data=None):
            if setup_callback:
                try:
                    setup_callback(phase, data)
                except Exception:
                    pass

        # --- Parse config ---
        method = config.get("method", "simclr").lower()
        encoder_name = config.get("encoder_name", "resnet34")
        pretrained_model_path = config.get("pretrained_model_path", None)
        epochs = int(config.get("epochs", 100))
        batch_size = int(config.get("batch_size", 64 if method == "simclr" else 32))
        grad_accum_steps = int(config.get("grad_accum_steps", 1))
        learning_rate = float(config.get("learning_rate", 3e-4))
        # Optional BYOL/SimCLR-paper LR scaling: lr = base_lr * batch_size / 256.
        # Both papers train at batch sizes much larger than 32, and the scaled
        # LR is what they actually use; the unscaled value is closer to a
        # batch=256 setting and can be too aggressive at small batch.
        scale_lr_by_batch = bool(config.get("scale_lr_by_batch", False))
        if scale_lr_by_batch:
            base_lr = learning_rate
            learning_rate = base_lr * batch_size / 256.0
            logger.info(
                "LR scaled by batch ratio: base=%.2e * (%d/256) = %.2e",
                base_lr, batch_size, learning_rate)
        # Default WD: 1e-6 for BYOL (per the BYOL paper), 1e-4 for SimCLR.
        # Overweight WD on BN/bias is one of the main BYOL collapse triggers
        # on small datasets; param-group split below excludes BN/bias from WD.
        wd_default = 1e-6 if method == "byol" else 1e-4
        weight_decay = float(config.get("weight_decay", wd_default))
        warmup_epochs = int(config.get("warmup_epochs", 10))
        temperature = float(config.get("temperature", 0.5))
        ema_decay = float(config.get("ema_decay", 0.996))
        # Schedule tau from a softer initial value to ema_decay_final over
        # the run. Constant tau on small datasets converges to a degenerate
        # target network; cosine ramp toward 1.0 is the BYOL-paper recipe.
        ema_decay_final = float(config.get("ema_decay_final", 1.0))
        projection_dim = int(config.get("projection_dim", 256))
        hidden_dim = int(config.get("hidden_dim", 2048))
        tile_size = int(config.get("tile_size", 256))
        intensity_mode = config.get("intensity_mode", "brightfield")
        stain_aug = bool(config.get("stain_aug", True))
        checkpoint_interval = int(config.get("checkpoint_interval", 50))
        # Collapse guard: track projection-output std-dev each epoch and
        # abort if it stays below this threshold for N consecutive epochs.
        collapse_threshold = float(config.get("collapse_threshold", 0.01))
        collapse_patience = int(config.get("collapse_patience", 3))
        # Encoder-side collapse guard. The projection-output probe misses
        # cases where the encoder pools to near-constant features but the
        # projector compensates by inflating its output magnitude -- the
        # L2-normalized proj_std then stays just above the threshold even
        # though the encoder is dead. Healthy pooled-encoder std is ~1.0;
        # values below ~0.5 mean the encoder is no longer carrying signal.
        encoder_collapse_threshold = float(
            config.get("encoder_collapse_threshold", 0.5))
        encoder_collapse_patience = int(
            config.get("encoder_collapse_patience", 2))

        if method not in ("simclr", "byol"):
            raise ValueError("Unsupported SSL method: %s" % method)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Locate images ---
        _report("loading_data")
        data_path = Path(data_path)
        if (data_path / "train" / "images").is_dir():
            image_dir = data_path / "train" / "images"
            logger.info("Detected training tile structure: %s", image_dir)
        elif (data_path / "images").is_dir():
            image_dir = data_path / "images"
        else:
            image_dir = data_path

        # --- Compute normalization stats ---
        _report("computing_stats")
        norm_stats = self._compute_norm_stats(image_dir)

        # --- Create augmentation pipeline ---
        # Detect channel count from first image
        tmp_dataset = SSLImageDataset.__new__(SSLImageDataset)
        tmp_dataset.image_paths = sorted([
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in SSLImageDataset.SUPPORTED_EXTENSIONS
               and not p.name.startswith('.')
               and SSLImageDataset._validate_file(p)
        ])[:1]
        if tmp_dataset.image_paths:
            tmp_dataset._cache = None
            first_img = tmp_dataset._load_image(tmp_dataset.image_paths[0])
            num_channels = 1 if first_img.ndim == 2 else first_img.shape[2]
        else:
            num_channels = 3

        augmentation = None
        if ALBUMENTATIONS_AVAILABLE:
            augmentation = get_ssl_augmentation(
                tile_size=tile_size,
                intensity_mode=intensity_mode,
                num_channels=num_channels,
                stain_aug=stain_aug,
            )
            logger.info("SSL augmentation: mode=%s, channels=%d, stain_aug=%s",
                         intensity_mode, num_channels, stain_aug)
        else:
            logger.warning(
                "albumentations unavailable -- using basic augmentation only")

        # --- Create dataset ---
        dataset = SSLImageDataset(
            image_dir=str(image_dir),
            tile_size=tile_size,
            normalize_stats=norm_stats,
            augmentation=augmentation,
            preload=True,
        )
        num_channels = dataset.n_channels
        num_images = len(dataset)
        logger.info("Dataset: %d images, %d channels, tile_size=%d",
                     num_images, num_channels, tile_size)

        # Hard floor: SSL needs more than 1 sample per batch or BatchNorm in
        # the projection head crashes ("Expected more than 1 value per channel
        # when training, got input size [1, 2048]"). Refuse early with a
        # readable error instead of letting the user wait minutes for the
        # opaque BN failure deep in the forward pass. We also require enough
        # images to sustain a 2-sample batch with drop_last=False.
        SSL_MIN_DATASET = 4
        if num_images < SSL_MIN_DATASET:
            raise ValueError(
                "SSL pretraining requires at least {n} unlabeled images, but "
                "found only {actual} in {path}. With fewer than {n} images "
                "the dataloader cannot form a batch large enough for the "
                "projection head's BatchNorm to train. Check that tile "
                "extraction completed (look for 'Extracted N tiles' in the "
                "log) and that the data directory wasn't deleted before "
                "training started.".format(
                    n=SSL_MIN_DATASET, actual=num_images, path=image_dir))

        # Dataset size warnings
        if num_images < 10:
            logger.warning(
                "Very few images (%d). SSL pretraining may not converge well. "
                "Consider collecting more unlabeled images.", num_images)
        elif num_images < 50:
            logger.info(
                "Small dataset (%d images). BYOL is recommended over SimCLR "
                "for small datasets (less sensitive to batch size).",
                num_images)

        # Auto-adjust batch size
        if batch_size > num_images:
            old_bs = batch_size
            batch_size = max(2, num_images // 2)  # Need >= 2 for SimCLR
            if grad_accum_steps == 1:
                grad_accum_steps = max(1, old_bs // batch_size)
            logger.info(
                "Batch size reduced %d -> %d (dataset has only %d images), "
                "grad accum = %d",
                old_bs, batch_size, num_images, grad_accum_steps)

        # SimCLR benefits from large effective batch sizes
        effective_batch = batch_size * grad_accum_steps
        if method == "simclr" and effective_batch < 64:
            # Auto-increase gradient accumulation to target effective batch 256
            target = min(256, num_images)
            grad_accum_steps = max(1, target // batch_size)
            effective_batch = batch_size * grad_accum_steps
            logger.info(
                "SimCLR: auto-adjusted grad_accum=%d for effective batch=%d",
                grad_accum_steps, effective_batch)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Must be 0: multiprocessing hangs in Appose
            pin_memory=self._device_str == "cuda",
            drop_last=len(dataset) > batch_size,
        )

        # --- Create SMP encoder ---
        _report("creating_model")
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch not installed. "
                "Install with: pip install segmentation-models-pytorch")

        # Create a minimal SMP model to extract the encoder
        temp_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # Random init; pretrained weights loaded below
            in_channels=num_channels,
            classes=1,  # Minimal decoder (we discard it)
        )
        encoder = temp_model.encoder
        encoder_out_dim = encoder.out_channels[-1]
        del temp_model  # Free decoder memory

        logger.info(
            "SMP encoder: %s, out_dim=%d, in_channels=%d",
            encoder_name, encoder_out_dim, num_channels)

        # --- Load pretrained weights (domain-adaptive pretraining) ---
        if pretrained_model_path:
            _report("loading_pretrained_weights")
            loaded = self._load_pretrained_encoder(
                encoder, pretrained_model_path, encoder_name)
            if loaded > 0:
                logger.info(
                    "Domain-adaptive SSL: loaded %d encoder weight tensors "
                    "from %s", loaded, pretrained_model_path)
            else:
                logger.warning(
                    "No encoder weights loaded from %s -- "
                    "training from scratch", pretrained_model_path)

        # --- Build SSL model ---
        if method == "simclr":
            model = SimCLRModel(
                encoder=encoder,
                encoder_out_dim=encoder_out_dim,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
            )
        else:  # byol
            model = BYOLModel(
                encoder=encoder,
                encoder_out_dim=encoder_out_dim,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
            )

        model = model.to(self.device)

        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "%s model: %.1fM parameters (%.1fM trainable), device=%s",
            method.upper(), param_count / 1e6,
            trainable_count / 1e6, self.device)

        # --- Optimizer ---
        # Exclude BatchNorm and bias parameters from weight decay. These
        # parameters do not benefit from L2 regularization, and applying WD
        # to BN scales destabilizes the EMA target in BYOL on small data.
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Match BN scale/shift (running stats are buffers, not params)
            # and any bias term across linear/conv layers.
            lname = name.lower()
            if param.ndim <= 1 or lname.endswith(".bias") or "bn" in lname \
                    or "batchnorm" in lname or "norm." in lname:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        logger.info(
            "Optimizer param groups: %d decay (wd=%.2e), %d no-decay",
            len(decay_params), weight_decay, len(no_decay_params))
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        # Cosine schedule with linear warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # BYOL EMA decay schedule: cosine ramp from `ema_decay` (initial)
        # to `ema_decay_final` (typically 1.0) over the run. Index is the
        # 1-based epoch number used in the training loop.
        def get_ema_decay(epoch_1based: int) -> float:
            if method != "byol":
                return ema_decay
            if epochs <= 1:
                return ema_decay_final
            progress = max(0.0, min(1.0, (epoch_1based - 1) / (epochs - 1)))
            return ema_decay_final - (ema_decay_final - ema_decay) * 0.5 \
                * (1.0 + float(np.cos(np.pi * progress)))

        # --- Mixed precision ---
        use_amp = self._device_str == "cuda"
        amp_dtype = None
        if use_amp:
            amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                         else torch.float16)
        use_grad_scaler = use_amp and amp_dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

        # --- Pre-flight VRAM estimate ---
        # Component-wise estimate calibrated against the empirical peak we
        # observe after the first batch (resnet34 BYOL batch=32 256x256 AMP
        # peaks at ~1.8 GB). The previous formula multiplied a per-batch
        # activation multiplier through the whole model+optimizer budget,
        # producing ~16x overestimates.
        vram_msg = None
        # Threshold for refusing the run pre-flight. The previous "warn at
        # 90% of free" was too lenient: the estimator under-counts SimCLR
        # at large tiles (a tile=512 batch=128 run estimated 14 GB but
        # measured 22.6 GB peak, peg-thrashed VRAM at 92%, dropped to
        # 1.9 img/s, and would have taken 12 days). Hard-stop at 85% of
        # total instead so the user reconfigures before burning hours.
        vram_estimate_pct_limit = float(
            config.get("vram_estimate_pct_limit", 85.0))
        abort_on_vram_risk = bool(config.get("abort_on_vram_risk", True))
        try:
            if self._device_str == "cuda":
                mem_info = self.gpu_manager.get_memory_info()
                total_mb = mem_info.get("total_mb", 0)
                allocated_mb = mem_info.get("allocated_mb", 0)
                free_mb = total_mb - allocated_mb
                model_mb = self.gpu_manager.estimate_model_memory(model)
                # BYOL has online + target weights (target = no grads/optim)
                model_factor = 2.0 if method == "byol" else 1.0
                # Trainable subset gets gradients + optimizer state. For BYOL
                # the target half is frozen, so only ~half of the weights have
                # backward state.
                trainable_mb = model_mb / model_factor
                # AdamW: 2 fp32 momentum buffers per trainable param.
                optimizer_mb = 2.0 * trainable_mb
                grad_mb = trainable_mb
                # Activations scale with encoder size, batch, and tile area.
                # Coefficient 0.5 is calibrated from a tile=512 batch=128
                # SimCLR run that peaked at 22.6 GB (the previous 0.3
                # value undershot by ~60%). Slight overshoot at small tiles
                # is fine -- the goal is to stop catastrophic configs early.
                area_scale = (tile_size * tile_size) / (256 * 256)
                act_per_tile_mb = 0.5 * trainable_mb * area_scale
                if use_amp:
                    act_per_tile_mb *= 0.5
                # SSL forwards two augmented views per sample.
                act_mb = act_per_tile_mb * batch_size * 2.0
                # cuDNN autotune workspace + allocator fragmentation. The
                # 250 MB figure was too low; cuDNN can hold 1+ GB during
                # benchmark search at large tile sizes.
                workspace_mb = 800.0
                estimated_mb = (model_mb + optimizer_mb + grad_mb
                                + act_mb + workspace_mb)
                # Express as percentage of TOTAL VRAM (not free) so the
                # threshold is interpretable independent of what other
                # processes happen to be holding right now.
                pct_of_total = (estimated_mb / total_mb * 100.0
                                if total_mb > 0 else 0.0)
                vram_msg = (
                    "VRAM: ~%.0f MB needed, %.0f MB free (%.0f MB total, "
                    "%.0f%% of total)"
                    % (estimated_mb, free_mb, total_mb, pct_of_total))
                if pct_of_total > vram_estimate_pct_limit:
                    vram_msg = (
                        "Estimated VRAM ~%.0f MB is %.0f%% of total %.0f MB "
                        "(limit %.0f%%). At this fill the GPU driver will page "
                        "activations through PCIe shared memory, dropping "
                        "throughput to ~1-2 img/s and stretching a 100-epoch "
                        "run to days. Reduce batch size or tile size, "
                        "or switch to a smaller backbone."
                        % (estimated_mb, pct_of_total, total_mb,
                           vram_estimate_pct_limit))
                    logger.error(vram_msg)
                    _report("vram_estimate", {"message": vram_msg,
                                              "estimated_mb": estimated_mb,
                                              "free_mb": free_mb,
                                              "total_mb": total_mb,
                                              "pct_of_total": pct_of_total,
                                              "limit_pct": vram_estimate_pct_limit,
                                              "blocked": True})
                    if abort_on_vram_risk:
                        raise RuntimeError(vram_msg)
                else:
                    logger.info(vram_msg)
                    _report("vram_estimate", {"message": vram_msg,
                                              "estimated_mb": estimated_mb,
                                              "free_mb": free_mb,
                                              "total_mb": total_mb,
                                              "pct_of_total": pct_of_total,
                                              "limit_pct": vram_estimate_pct_limit,
                                              "blocked": False})
        except RuntimeError:
            raise  # propagate the abort
        except Exception:
            pass  # Never let estimation break training

        # --- Optional torch.compile ---
        use_compile = config.get("torch_compile", False)
        if use_compile and self._device_str == "cuda":
            try:
                import platform
                if platform.system() == "Linux":
                    model = torch.compile(model)
                    logger.info("torch.compile enabled")
                else:
                    logger.info("torch.compile skipped (requires Linux)")
            except Exception as e:
                logger.warning("torch.compile failed, continuing without: %s", e)

        # --- Pause signal path ---
        pause_signal_path = config.get("pause_signal_path", None)

        # --- Resume from a paused checkpoint, if requested ---
        # Java's resumePretraining injects checkpoint_path + start_epoch into
        # the config dict so we know to skip ahead and restore model/optim state.
        checkpoint_path = config.get("checkpoint_path", None)
        start_epoch = int(config.get("start_epoch", 0))
        best_loss = float('inf')
        best_state = None
        best_epoch = 0
        history = []
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                ckpt = torch.load(str(checkpoint_path), map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                best_loss = ckpt.get("best_loss", float('inf'))
                best_state = ckpt.get("best_state", None)
                history = ckpt.get("history", [])
                start_epoch = max(start_epoch, int(ckpt.get("epoch", 0)))
                logger.info(
                    "Resumed SSL pretraining from %s at epoch %d (best_loss=%.6f)",
                    checkpoint_path, start_epoch, best_loss)
            except Exception as e:
                logger.error(
                    "Failed to load resume checkpoint %s: %s -- starting fresh",
                    checkpoint_path, e)
                start_epoch = 0
                best_loss = float('inf')
                best_state = None
                history = []

        # --- Training loop ---
        _report("starting_training")
        import time as _time
        logger.info(
            "%s pretraining: %d epochs, batch=%d (effective %d, accum=%d), "
            "lr=%.2e, encoder=%s",
            method.upper(), epochs, batch_size, effective_batch,
            grad_accum_steps, learning_rate, encoder_name)
        if method == "simclr":
            logger.info("SimCLR temperature=%.3f, projection_dim=%d",
                         temperature, projection_dim)
        else:
            logger.info(
                "BYOL ema_decay schedule: %.4f -> %.4f, projection_dim=%d",
                ema_decay, ema_decay_final, projection_dim)
        training_start_time = _time.monotonic()

        # Collapse-guard state
        collapse_streak = 0
        encoder_collapse_streak = 0
        # Set to "proj_std", "proj_std_immediate", or "encoder_std" when a
        # probe trips, so the post-loop warning can accurately describe
        # which signal failed instead of using a generic message.
        aborted_reason = None
        # Set when the training loop exits because the collapse probe
        # tripped, so the post-loop save path can return a status that
        # the Java side surfaces as a warning instead of pretending the
        # run completed normally.
        aborted_collapse = False

        first_batch_done = False
        try:
            for epoch in range(start_epoch + 1, epochs + 1):
                if cancel_flag and cancel_flag.is_set():
                    logger.info("Pretraining cancelled at epoch %d", epoch)
                    break

                # Check pause signal
                if pause_signal_path and Path(pause_signal_path).exists():
                    logger.info("Pause signal detected at epoch %d", epoch)
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "best_state": best_state,
                        "history": history,
                        "config": config,
                    }
                    torch.save(ckpt, str(output_dir / "pause_checkpoint.pt"))
                    logger.info("Saved pause checkpoint to %s",
                                output_dir / "pause_checkpoint.pt")
                    # Consume the pause signal so a resume doesn't trip on it
                    try:
                        Path(pause_signal_path).unlink()
                    except Exception:
                        pass
                    self._cleanup_training_memory()
                    return {
                        "status": "paused",
                        "encoder_path": "",
                        "epochs_completed": epoch - 1,
                        "final_loss": history[-1]["loss"] if history else 0.0,
                        "best_loss": best_loss,
                    }

                epoch_start = _time.monotonic()

                # Clear GPU cache at epoch start
                if self._device_str == "cuda":
                    self.gpu_manager.clear_cache()

                model.train()
                epoch_loss = 0.0
                n_batches = 0
                # Ring buffer of recent view1 batches for the collapse probe.
                # 8 batches at batch=32 gives 256 samples in 256-d, which is
                # enough to make per-dim std a stable estimator.
                proj_probe_size = 8
                last_view1_batches = []
                # Diagnostic accumulators (epoch averages). These let us
                # distinguish "data pipeline produces near-identical inputs"
                # from "encoder produces near-constant features" from
                # "projector collapsed" from "gradients aren't flowing".
                epoch_input_std_sum = 0.0
                epoch_grad_norm_sum = 0.0
                epoch_grad_norm_count = 0

                optimizer.zero_grad(set_to_none=True)

                for batch_idx, (view1, view2) in enumerate(dataloader):
                    if cancel_flag and cancel_flag.is_set():
                        break

                    view1 = view1.to(self.device, non_blocking=True)
                    view2 = view2.to(self.device, non_blocking=True)

                    # Per-batch input std (across all elements). Tracks
                    # whether augmentation+normalization is producing
                    # diverse inputs at all. Healthy ~0.2-0.5 for
                    # standardized image tensors; <<0.05 means the data
                    # pipeline is squashing the signal.
                    with torch.no_grad():
                        epoch_input_std_sum += float(view1.float().std().item())

                    if not first_batch_done:
                        logger.info(
                            "First batch: shape=%s, dtype=%s, device=%s",
                            list(view1.shape), view1.dtype, view1.device)
                        logger.info(
                            "First batch input stats: min=%.4f max=%.4f "
                            "mean=%.4f std=%.4f",
                            float(view1.float().min().item()),
                            float(view1.float().max().item()),
                            float(view1.float().mean().item()),
                            float(view1.float().std().item()))
                        # One-time: save 8 augmented (view1, view2) pairs
                        # so the user can eyeball whether the augmentation
                        # pipeline is producing reasonable positive pairs.
                        try:
                            self._save_aug_pair_grid(
                                view1, view2,
                                output_dir / "aug_pairs_epoch1.png",
                                norm_stats=norm_stats)
                            logger.info(
                                "Saved augmentation-pair preview to %s",
                                output_dir / "aug_pairs_epoch1.png")
                        except Exception as e:
                            logger.debug(
                                "Failed to save aug-pair preview: %s", e)

                    # Forward + loss
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            loss = self._compute_loss(
                                model, view1, view2, method,
                                temperature, ema_decay)
                        loss = loss / grad_accum_steps
                        if use_grad_scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    else:
                        loss = self._compute_loss(
                            model, view1, view2, method,
                            temperature, ema_decay)
                        loss = loss / grad_accum_steps
                        loss.backward()

                    if not first_batch_done:
                        logger.info(
                            "First forward+backward pass complete, loss=%.6f",
                            loss.item() * grad_accum_steps)
                        if self._device_str == "cuda":
                            peak_mb = self.gpu_manager.get_peak_allocated_mb()
                            mem_info = self.gpu_manager.get_memory_info()
                            total_mb = mem_info.get("total_mb", 0)
                            peak_pct = ((peak_mb / total_mb * 100)
                                        if total_mb > 0 else 0)
                            logger.info(
                                "Peak GPU memory after first batch: "
                                "%.0f MB / %.0f MB (%.0f%% used)",
                                peak_mb, total_mb, peak_pct)
                            # Steady-state will exceed first-batch peak
                            # (cuDNN autotune cache, BYOL EMA buffers,
                            # accumulated allocator fragmentation), so
                            # 70% measured here usually means 80%+ during
                            # training -- the regime where WDDM paging
                            # collapses throughput. Bail before that.
                            peak_pct_limit = float(
                                config.get("vram_peak_pct_limit", 70.0))
                            blocked = peak_pct > peak_pct_limit
                            msg = ("Peak VRAM: %.0f MB / %.0f MB (%.0f%%)"
                                   % (peak_mb, total_mb, peak_pct))
                            if blocked:
                                msg = (
                                    "Measured peak VRAM %.0f MB is %.0f%% of "
                                    "total %.0f MB after the first batch "
                                    "(limit %.0f%%). Steady-state will be "
                                    "higher (cuDNN cache, allocator drift); "
                                    "the GPU driver will start paging through "
                                    "PCIe and drop throughput to ~1-2 img/s. "
                                    "Reduce batch size or tile size and "
                                    "rerun."
                                    % (peak_mb, peak_pct, total_mb,
                                       peak_pct_limit))
                                logger.error(msg)
                            _report("peak_memory", {
                                "message": msg,
                                "peak_mb": peak_mb,
                                "total_mb": total_mb,
                                "peak_pct": peak_pct,
                                "limit_pct": peak_pct_limit,
                                "blocked": blocked,
                            })
                            if blocked and abort_on_vram_risk:
                                raise RuntimeError(msg)
                        first_batch_done = True

                    # Optimizer step
                    if (batch_idx + 1) % grad_accum_steps == 0 or \
                            (batch_idx + 1) == len(dataloader):
                        # Capture grad norm BEFORE clipping so we can tell
                        # whether gradients are vanishing (~0) or being
                        # clipped every step (>>1).
                        if use_amp and use_grad_scaler:
                            scaler.unscale_(optimizer)
                            grad_norm_val = float(
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 1.0).item())
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm_val = float(
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 1.0).item())
                            optimizer.step()
                        epoch_grad_norm_sum += grad_norm_val
                        epoch_grad_norm_count += 1
                        optimizer.zero_grad(set_to_none=True)

                        # BYOL: EMA update after each optimizer step
                        if method == "byol":
                            model.update_target(get_ema_decay(epoch))

                    epoch_loss += loss.item() * grad_accum_steps
                    n_batches += 1
                    last_view1_batches.append(view1.detach())
                    if len(last_view1_batches) > proj_probe_size:
                        last_view1_batches.pop(0)

                # If cancel arrived mid-epoch, skip the rest of the
                # epoch bookkeeping (scheduler, collapse probe, metrics,
                # history append, progress log) and break the outer loop
                # immediately. Without this, a cancel mid-batch still
                # causes a full epoch's worth of "Epoch X/Y: loss=..."
                # output to fire after the user requested cancel, which
                # makes the UI feel unresponsive.
                if cancel_flag and cancel_flag.is_set():
                    logger.info(
                        "Cancel detected mid-epoch %d; exiting without "
                        "completing epoch bookkeeping", epoch)
                    break

                scheduler.step()

                # --- Collapse guard ---
                # Measure per-feature std-dev of L2-normalized projections.
                # A healthy SSL model spreads samples across the unit sphere
                # -> std around 1/sqrt(D). A collapsed model maps everything
                # to one point -> std ~ 0.
                #
                # We sample multiple batches in train-mode BN so the metric
                # matches the regime we just trained in. Eval-mode BN with a
                # single batch was producing huge epoch-to-epoch swings that
                # looked like collapse but were just BN running-stats noise
                # (BYOL with batch=32 has very noisy running estimates).
                proj_std = None
                encoder_std = None
                proj_pre_norm_std = None
                if last_view1_batches:
                    try:
                        # Stay in train() so BN uses batch stats, matching
                        # the regime the projector was just optimized in.
                        # The probe is no_grad so no parameter updates occur.
                        # BN running stats receive a few extra updates here,
                        # which is benign relative to ~1500 training steps/epoch.
                        feats = []
                        zs_pre = []
                        zs = []
                        with torch.no_grad():
                            inner = model
                            if hasattr(inner, "_orig_mod"):
                                inner = inner._orig_mod
                            for v in last_view1_batches:
                                if method == "byol":
                                    feat = inner.pool(
                                        inner.online_encoder(v)[-1]
                                    ).flatten(1)
                                    z_pre = inner.online_projector(feat)
                                else:
                                    feat = inner.pool(
                                        inner.encoder(v)[-1]
                                    ).flatten(1)
                                    z_pre = inner.projector(feat)
                                feats.append(feat)
                                zs_pre.append(z_pre)
                                zs.append(F.normalize(z_pre, dim=1))
                            feat_all = torch.cat(feats, dim=0)
                            z_pre_all = torch.cat(zs_pre, dim=0)
                            z_all = torch.cat(zs, dim=0)
                            encoder_std = float(
                                feat_all.std(dim=0).mean().item())
                            proj_pre_norm_std = float(
                                z_pre_all.std(dim=0).mean().item())
                            proj_std = float(z_all.std(dim=0).mean().item())
                    except Exception as e:
                        logger.debug("Projection-std probe failed: %s", e)
                        proj_std = None

                # --- Epoch metrics ---
                epoch_end = _time.monotonic()
                epoch_duration = epoch_end - epoch_start
                elapsed = epoch_end - training_start_time
                avg_epoch_sec = elapsed / epoch
                remaining_sec = avg_epoch_sec * (epochs - epoch)
                images_per_sec = (n_batches * batch_size) / max(
                    epoch_duration, 0.001)

                avg_loss = epoch_loss / max(1, n_batches)
                current_lr = optimizer.param_groups[0]['lr']
                avg_input_std = (epoch_input_std_sum / max(1, n_batches))
                avg_grad_norm = (
                    epoch_grad_norm_sum / max(1, epoch_grad_norm_count))

                history.append({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "lr": current_lr,
                    "epoch_sec": round(epoch_duration, 2),
                    "proj_std": proj_std if proj_std is not None else -1.0,
                    "encoder_std": (
                        encoder_std if encoder_std is not None else -1.0),
                    "proj_pre_norm_std": (
                        proj_pre_norm_std
                        if proj_pre_norm_std is not None else -1.0),
                    "input_std": avg_input_std,
                    "grad_norm": avg_grad_norm,
                })

                # Update collapse streak and decide whether to abort.
                # Two paths:
                #   (a) standard streak: proj_std < threshold for N epochs.
                #   (b) immediate-collapse fast-path: proj_std < threshold/2
                #       on epoch 1 means the projector started in a collapsed
                #       configuration and won't recover. Bail right away
                #       rather than burning two more epochs (epoch 1 alone
                #       took >3 hours in the 22.6 GB-VRAM-thrashing run).
                immediate_collapse_threshold = collapse_threshold / 2.0
                if proj_std is not None and proj_std < collapse_threshold:
                    collapse_streak += 1
                    logger.warning(
                        "Collapse warning epoch %d: proj_std=%.6f < %.6f "
                        "(streak %d/%d)",
                        epoch, proj_std, collapse_threshold,
                        collapse_streak, collapse_patience)
                else:
                    collapse_streak = 0
                immediate_collapse = (
                    proj_std is not None
                    and epoch == 1
                    and proj_std < immediate_collapse_threshold)
                if immediate_collapse:
                    logger.error(
                        "Aborting: severe collapse on epoch 1 "
                        "(proj_std=%.6f < %.6f). Embeddings are already "
                        "near-constant; further epochs will not recover. "
                        "Lower temperature (0.1-0.2 for SimCLR), reduce "
                        "batch size, or switch to BYOL with ema_decay=0.99.",
                        proj_std, immediate_collapse_threshold)
                    aborted_collapse = True
                    aborted_reason = "proj_std_immediate"
                    break
                if collapse_streak >= collapse_patience:
                    logger.error(
                        "Aborting: representation collapse detected "
                        "(proj_std < %.6f for %d consecutive epochs). "
                        "Use checkpoint_epoch_<earlier>.pt for downstream.",
                        collapse_threshold, collapse_patience)
                    aborted_collapse = True
                    aborted_reason = "proj_std"
                    break

                # Encoder-side collapse: catches the failure mode where the
                # projector inflates magnitude to compensate for a dead
                # encoder. proj_std (post-L2-norm) can sit just above
                # threshold while encoder_std collapses to ~0.25 (run on
                # 2026-05-02 collapsed at epoch 24 this way and trained
                # uselessly for 17 more epochs before user cancel).
                if (encoder_std is not None
                        and encoder_std < encoder_collapse_threshold):
                    encoder_collapse_streak += 1
                    logger.warning(
                        "Encoder-collapse warning epoch %d: "
                        "encoder_std=%.4f < %.4f (streak %d/%d)",
                        epoch, encoder_std, encoder_collapse_threshold,
                        encoder_collapse_streak,
                        encoder_collapse_patience)
                else:
                    encoder_collapse_streak = 0
                if encoder_collapse_streak >= encoder_collapse_patience:
                    logger.error(
                        "Aborting: encoder collapse detected "
                        "(encoder_std < %.4f for %d consecutive epochs). "
                        "The projector may still appear healthy via "
                        "proj_std, but pooled encoder features have "
                        "lost their spread. Lower learning rate, tighten "
                        "gradient clip, or lower ema_decay for BYOL.",
                        encoder_collapse_threshold,
                        encoder_collapse_patience)
                    aborted_collapse = True
                    aborted_reason = "encoder_std"
                    break

                # Best model tracking with disk persistence
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = epoch
                    best_state = self._extract_encoder_state(model, method)
                    # Persist to disk for crash recovery
                    best_ckpt_path = output_dir / "best_in_progress.pt"
                    torch.save({
                        "model_state_dict": best_state,
                        "epoch": epoch,
                        "loss": best_loss,
                    }, str(best_ckpt_path))

                # Progress callback with timing data
                if progress_callback:
                    try:
                        progress_callback(
                            epoch, epochs, avg_loss, current_lr,
                            elapsed_sec=elapsed,
                            remaining_sec=remaining_sec,
                            epoch_sec=epoch_duration,
                            images_per_sec=images_per_sec)
                    except Exception:
                        pass

                # Dense epoch logging with timing
                proj_std_str = ("%.4f" % proj_std) if proj_std is not None else "n/a"
                enc_std_str = (
                    "%.4f" % encoder_std if encoder_std is not None else "n/a")
                pre_std_str = (
                    "%.4f" % proj_pre_norm_std
                    if proj_pre_norm_std is not None else "n/a")
                logger.info(
                    "Epoch %d/%d: loss=%.6f, lr=%.2e, "
                    "input_std=%.4f, encoder_std=%s, "
                    "proj_pre_std=%s, proj_std=%s, grad_norm=%.4f "
                    "(%.1fs, %.1f img/s, ~%.0fs remaining)",
                    epoch, epochs, avg_loss, current_lr,
                    avg_input_std, enc_std_str, pre_std_str, proj_std_str,
                    avg_grad_norm,
                    epoch_duration, images_per_sec, remaining_sec)

                # Periodic checkpoint
                if checkpoint_interval > 0 and \
                        epoch % checkpoint_interval == 0:
                    ckpt_path = output_dir / (
                        "checkpoint_epoch_%d.pt" % epoch)
                    ckpt_state = self._extract_encoder_state(model, method)
                    torch.save(
                        {"model_state_dict": ckpt_state}, str(ckpt_path))
                    logger.info("Saved checkpoint: %s", ckpt_path)

        except Exception as exc:
            # OOM-specific handling: strip traceback to release GPU memory
            if "OutOfMemory" in type(exc).__name__:
                import traceback as _tb
                error_text = _tb.format_exc()
                exc.__traceback__ = None
                self._cleanup_training_memory()
                raise RuntimeError(
                    "GPU out of memory during SSL pretraining. "
                    "Try reducing batch size or tile size.\n\n"
                    + error_text) from None
            raise
        finally:
            self._cleanup_training_memory()

        # --- Handle cancellation with save ---
        # Save-mode is set by the JavaFX cancel dialog and forwarded via
        # the cancel signal file (best/last/none). The pretrain script
        # stores the resolved mode in config["_cancel_save_mode_state"];
        # default "best" matches pre-existing behavior when the user
        # cancels without going through the dialog (e.g. window-close).
        was_cancelled = cancel_flag and cancel_flag.is_set()
        if was_cancelled:
            cancel_mode = "best"
            try:
                cancel_mode = (
                    config.get("_cancel_save_mode_state", {}).get(
                        "mode", "best") or "best"
                ).lower()
            except Exception:
                cancel_mode = "best"
            if cancel_mode not in ("best", "last", "none"):
                cancel_mode = "best"
            logger.info(
                "Cancel handler: save mode = %s (%d epochs completed, "
                "best_state %s)",
                cancel_mode, len(history),
                "available" if best_state is not None else "missing")

            # No-save path: user explicitly chose Discard.
            if cancel_mode == "none":
                logger.info("Cancel mode 'none': skipping encoder save")
                return {
                    "status": "cancelled",
                    "encoder_path": "",
                    "epochs_completed": len(history),
                    "final_loss": history[-1]["loss"] if history else 0.0,
                    "best_loss": best_loss if best_state is not None else 0.0,
                    "quality": "cancelled",
                    "warnings": [
                        f"Run was cancelled by the user at epoch "
                        f"{len(history)} of {epochs}; no encoder saved "
                        f"per the user's selection."
                    ],
                }

            # 'last' is only meaningful if we actually trained at least one
            # batch. With zero history, fall back to the no-save path.
            if cancel_mode == "last" and not history:
                logger.info(
                    "Cancel mode 'last' but no epochs completed; "
                    "nothing to save")
                return {
                    "status": "cancelled",
                    "encoder_path": "",
                    "epochs_completed": 0,
                    "final_loss": 0.0,
                    "best_loss": 0.0,
                    "quality": "cancelled",
                    "warnings": [
                        f"Run was cancelled before any epoch completed; "
                        f"nothing to save."
                    ],
                }

            # 'best' with no best_state is also a no-save path.
            if cancel_mode == "best" and best_state is None:
                logger.info("Cancelled with no completed epochs")
                return {
                    "status": "cancelled",
                    "encoder_path": "",
                    "epochs_completed": 0,
                    "final_loss": 0.0,
                    "best_loss": 0.0,
                    "quality": "cancelled",
                    "warnings": [
                        f"Run was cancelled by the user before any epoch "
                        f"completed; no encoder saved."
                    ],
                }

            # Save path: pick which weights to write.
            if cancel_mode == "last":
                save_state = self._extract_encoder_state(model, method)
                save_epoch = len(history)
                save_loss = history[-1]["loss"] if history else 0.0
                save_label = "last epoch's weights"
            else:  # "best"
                save_state = best_state
                save_epoch = best_epoch
                save_loss = best_loss
                save_label = "best epoch's weights"

            encoder_path = str(output_dir / "model.pt")
            torch.save({"model_state_dict": save_state}, encoder_path)
            self._save_metadata(
                output_dir, method, encoder_name, num_channels,
                tile_size, history, best_loss, best_epoch,
                batch_size, effective_batch, learning_rate,
                len(dataset), temperature, ema_decay,
                projection_dim, pretrained_model_path, norm_stats,
                run_name=config.get("run_name", ""))
            logger.info(
                "Cancelled at epoch %d, saved %s (epoch %d, loss=%.6f) "
                "to %s",
                len(history), save_label, save_epoch, save_loss,
                encoder_path)
            # Run the same quality heuristics on the cancelled run.
            # A user who cancels because the loss looked wrong should
            # see the diagnostic ("loss near random" / "BYOL collapse-
            # shaped") just like a clean-completion run -- otherwise
            # they save a bad encoder thinking they "captured" useful
            # weights from an early stop.
            cq, cwarn = self._assess_training_quality(
                method=method,
                history=history,
                dataset_size=len(dataset),
                batch_size=batch_size,
            )
            cwarn.insert(0,
                f"Run was cancelled by the user at epoch "
                f"{len(history)} of {epochs}. The encoder reflects the "
                f"{save_label} (epoch {save_epoch}), not a fully trained "
                f"model.")
            return {
                "status": "cancelled_saved",
                "encoder_path": encoder_path,
                "epochs_completed": len(history),
                "final_loss": history[-1]["loss"] if history else 0.0,
                "best_loss": best_loss,
                "quality": cq if cq != "ok" else "cancelled",
                "warnings": cwarn,
            }

        # --- Run trajectory health check BEFORE saving ---
        # Surface suspicious loss curves (collapse-shaped trajectories,
        # implausibly low final loss, near-zero proj_std) so a "trained"
        # encoder isn't silently presented as a success when it's likely
        # a collapsed model. Loud log lines + a "warnings" field on the
        # return dict so the Java side can show the user.
        quality, warnings = self._assess_training_quality(
            method=method,
            history=history,
            dataset_size=len(dataset),
            batch_size=batch_size,
        )
        # If the collapse probe explicitly aborted, escalate quality and
        # prepend an unambiguous warning so the user sees it first. The
        # heuristic check may not catch a 3-epoch run on its own.
        if aborted_collapse:
            quality = "likely_collapse"
            if aborted_reason == "encoder_std":
                msg = (
                    "Training aborted by the encoder-collapse probe -- "
                    "the pooled encoder features lost variance "
                    "(encoder_std < %.2f for %d consecutive epochs) "
                    "while the projector continued inflating its output "
                    "magnitude to compensate. The saved encoder is "
                    "almost certainly unusable for downstream "
                    "supervised training. Likely causes for this run: "
                    "(1) one-batch gradient spike during cosine LR "
                    "ramp-up that the 1.0 grad-clip didn't catch -- try "
                    "lowering the clip threshold; "
                    "(2) for BYOL, ema_decay too high for dataset size "
                    "-- try 0.98 with ema_decay_final=0.999 on small "
                    "datasets (<100k tiles); "
                    "(3) scale_lr_by_batch may be cutting LR too low "
                    "for warmup, then the post-warmup LR transition "
                    "destabilizes the encoder -- try unscaled LR.") % (
                    encoder_collapse_threshold, encoder_collapse_patience)
            else:
                msg = (
                    "Training aborted by the collapse probe -- the "
                    "encoder's embeddings degenerated to a near-constant "
                    "output (proj_std below threshold for %d consecutive "
                    "epochs). The saved encoder is almost certainly "
                    "unusable for downstream supervised training. "
                    "Causes: weak augmentations producing trivially "
                    "easy positive pairs, temperature too high "
                    "(try 0.1-0.2 for SimCLR on histology), or for "
                    "BYOL try ema_decay=0.99.") % collapse_patience
            warnings.insert(0, msg)
        if warnings:
            logger.warning("=" * 60)
            logger.warning("SSL pretraining QUALITY WARNINGS (%s):", quality.upper())
            for w in warnings:
                logger.warning("  - %s", w)
            logger.warning("Recommended actions:")
            logger.warning("  - Inspect the encoder before using it for "
                           "supervised training.")
            logger.warning("  - For BYOL: lower batch size (64-128 for "
                           "<50k tiles), drop ema_decay (start) to 0.99, "
                           "shorten epochs, or switch to SimCLR.")
            logger.warning("  - See per-epoch loss/proj_std in metadata.json.")
            logger.warning("=" * 60)

        # --- Save best model ---
        _report("saving_model")

        if best_state is None:
            best_state = self._extract_encoder_state(model, method)
            best_epoch = len(history)

        encoder_path = str(output_dir / "model.pt")
        torch.save({"model_state_dict": best_state}, encoder_path)

        n_encoder_keys = sum(
            1 for k in best_state if k.startswith("encoder."))
        logger.info("Saved %d encoder.* keys to %s",
                     n_encoder_keys, encoder_path)

        # Save metadata
        self._save_metadata(
            output_dir, method, encoder_name, num_channels,
            tile_size, history, best_loss, best_epoch,
            batch_size, effective_batch, learning_rate,
            len(dataset), temperature, ema_decay,
            projection_dim, pretrained_model_path, norm_stats,
            run_name=config.get("run_name", ""))

        # Clean up crash-recovery checkpoint
        best_ckpt = output_dir / "best_in_progress.pt"
        if best_ckpt.exists():
            best_ckpt.unlink()
        # Clean up pause checkpoint after successful completion
        pause_ckpt = output_dir / "pause_checkpoint.pt"
        if pause_ckpt.exists():
            try:
                pause_ckpt.unlink()
            except Exception:
                pass

        elapsed_total = _time.monotonic() - training_start_time
        status_tag = "" if not warnings else " [REVIEW WARNINGS]"
        logger.info(
            "%s pretraining complete%s: %d epochs (best at %d), "
            "best_loss=%.6f, %.1f min total -> %s",
            method.upper(), status_tag, len(history), best_epoch,
            best_loss, elapsed_total / 60, output_dir)

        self.gpu_manager.clear_cache()

        # Distinct status when the run was halted by the collapse probe.
        # The Java side reads this and shows the warning to the user
        # instead of the standard "Encoder saved! Final loss: ..." message.
        result_status = "aborted_collapse" if aborted_collapse else "completed"
        return {
            "status": result_status,
            "encoder_path": encoder_path,
            "epochs_completed": len(history),
            "final_loss": history[-1]["loss"] if history else 0.0,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "elapsed_sec": round(elapsed_total, 1),
            "quality": quality,
            "warnings": warnings,
        }

    def _assess_training_quality(self, method, history, dataset_size, batch_size):
        """Inspect the loss/proj_std trajectory for collapse-shaped runs.

        Returns (quality_label, warnings_list). quality is one of
        "ok", "warn", "likely_collapse". warnings is a list of plain-English
        strings safe to log and surface to the UI.

        Heuristics are tuned for BYOL on small histology datasets, where
        the most common failure mode is the loss freefalling to ~0 after
        ~10-20 epochs (the encoder has learned a constant). The proj_std
        probe in the training loop already aborts hard collapses; this
        check catches softer collapses that were near-but-not-below the
        abort threshold, plus other "this looks wrong" patterns that
        would otherwise be silently saved as a finished encoder.
        """
        warnings = []
        if not history:
            return "ok", warnings

        losses = [h["loss"] for h in history if h.get("loss") is not None]
        if len(losses) < 5:
            return "ok", warnings

        final_loss = losses[-1]
        first_loss = losses[0]
        min_loss = min(losses)
        # proj_std is -1.0 when not computed (e.g. SimCLR or older runs).
        proj_stds = [h.get("proj_std", -1.0) for h in history]
        proj_stds = [s for s in proj_stds if s is not None and s >= 0.0]

        is_byol = method.lower() == "byol"
        is_simclr = method.lower() == "simclr"

        # 0. SimCLR loss-near-random: InfoNCE loss for batch N has uniform-
        #    random baseline ln(2N) (each anchor sees 1 positive vs 2N-1
        #    negatives, so ~ln(2N) when predictions are uniform). If the
        #    loss never moves meaningfully below that baseline, the encoder
        #    isn't learning -- positive pairs aren't being aligned. This is
        #    distinct from hard projection collapse (which the proj_std
        #    probe catches): an alignment-collapsed encoder can still
        #    spread its embeddings on the unit sphere (high proj_std) while
        #    failing to bring positive pairs closer than negatives.
        #
        #    Reported case that motivated this heuristic: resnet34 SimCLR,
        #    batch=64, 20 epochs, loss settled at 4.82 with random baseline
        #    ln(128)~=4.85 -- only 0.03 below random, but the existing
        #    heuristics were all gated on is_byol so the run was reported
        #    as "ok" with no warnings.
        if is_simclr and len(losses) >= 5:
            try:
                ln_random = math.log(2 * max(2, batch_size))
            except (TypeError, ValueError):
                ln_random = None
            if ln_random and ln_random > 0:
                # Final loss within 5% of the random baseline.
                if final_loss > 0.95 * ln_random:
                    warnings.append(
                        f"SimCLR final loss {final_loss:.4f} is within 5% "
                        f"of the uniform-random baseline ln(2N)~={ln_random:.4f} "
                        f"(batch={batch_size}). The encoder is not learning "
                        f"to align positive pairs. Common causes: "
                        f"temperature too high (try 0.1-0.2 for histology), "
                        f"learning rate too low for this batch size, or the "
                        f"augmentation pipeline isn't producing meaningful "
                        f"positive pairs. Encoder is unlikely to help "
                        f"downstream."
                    )
                # Total drop from start to end is a small fraction of the
                # baseline -- the loss curve is essentially flat. Tunable:
                # 5% of ln_random over the whole run is a generous floor.
                drop = first_loss - final_loss
                if drop < 0.05 * ln_random:
                    warnings.append(
                        f"SimCLR loss barely changed: {first_loss:.4f} -> "
                        f"{final_loss:.4f} ({drop:+.4f}) over {len(losses)} "
                        f"epochs vs. baseline ln(2N)~={ln_random:.4f}. "
                        f"With normal learning the loss should fall well "
                        f"below the baseline within the first 10-20 epochs."
                    )

        # 1. Final loss implausibly low (BYOL only -- SimCLR loss is on
        #    a different scale and naturally rides much lower).
        if is_byol and final_loss < 0.05:
            warnings.append(
                f"Final BYOL loss is {final_loss:.4f}. Healthy BYOL on "
                f"histology typically settles in 0.05-0.20; values below "
                f"~0.02 are characteristic of representation collapse "
                f"(encoder outputs a constant)."
            )

        # 2. Loss freefall in the second half: monotonic-ish decrease and
        #    a final value much smaller than the mid-training loss is the
        #    classic collapse signature.
        if is_byol and len(losses) >= 10:
            mid = losses[len(losses) // 2]
            if mid > 0 and final_loss / mid < 0.15:
                warnings.append(
                    f"BYOL loss dropped from ~{mid:.3f} (mid-training) "
                    f"to {final_loss:.4f} (final), a {(1 - final_loss/mid)*100:.0f}% "
                    f"drop in the second half. This shape is typical of "
                    f"BYOL representation collapse, not normal convergence."
                )

        # 3. Late-stage proj_std near the abort threshold even if abort
        #    didn't fire. Use the last 5 epochs' median to ignore noise.
        if proj_stds and len(proj_stds) >= 5:
            tail = sorted(proj_stds[-5:])
            tail_median = tail[len(tail) // 2]
            if tail_median < 0.05:
                warnings.append(
                    f"Final embedding spread (proj_std median = "
                    f"{tail_median:.4f}) is very low. Healthy encoders "
                    f"have proj_std around 1/sqrt(D) ~ 0.06-0.1+; values "
                    f"under 0.05 indicate the encoder is mapping inputs "
                    f"to nearly the same vector."
                )

        # 4. Dataset/batch ratio sanity at end of training. If we hit
        #    very low loss on a tiny per-epoch dataset, flag it.
        steps_per_epoch = max(1, dataset_size // max(1, batch_size))
        if is_byol and final_loss < 0.10 and steps_per_epoch < 50:
            warnings.append(
                f"Only ~{steps_per_epoch} batches per epoch "
                f"({dataset_size} tiles / batch {batch_size}). BYOL "
                f"target updates infrequently with so few steps, and "
                f"the encoder can find the trivial solution before the "
                f"target diverges. A loss this low on this little data "
                f"is suspicious."
            )

        # 5. Best epoch is suspiciously early -- often a sign that BYOL
        #    drove loss into the collapse regime and never recovered.
        best_idx = losses.index(min_loss)
        if is_byol and len(losses) >= 20 and best_idx < len(losses) // 4:
            warnings.append(
                f"Best loss occurred at epoch {best_idx + 1} of "
                f"{len(losses)} -- very early. This often indicates the "
                f"encoder collapsed soon after and never produced a "
                f"better representation."
            )

        # Overall quality label
        if not warnings:
            return "ok", warnings
        # If multiple BYOL collapse-shaped warnings fire, escalate.
        collapse_signals = sum(
            1 for w in warnings
            if "collapse" in w.lower() or "constant" in w.lower()
            or "trivial" in w.lower())
        if collapse_signals >= 2:
            return "likely_collapse", warnings
        return "warn", warnings

    def _cleanup_training_memory(self):
        """Free GPU memory after training completes or fails."""
        import gc
        gc.collect()
        self.gpu_manager.clear_cache()
        self.gpu_manager.log_memory_status(prefix="SSL cleanup: ")

    @staticmethod
    def _save_aug_pair_grid(view1, view2, out_path, norm_stats=None,
                             num_pairs=8):
        """Save a grid of (view1, view2) augmented pairs as a PNG.

        Lets the user eyeball whether augmentations are producing
        meaningful positive pairs. If view1 and view2 look identical,
        the augmentation pipeline is too weak; if they look like
        unrelated images, it is too aggressive.

        Inputs are expected to be normalized (mean/std), so we
        de-normalize for display when norm_stats are provided.
        """
        import numpy as np
        try:
            import imageio.v3 as iio
        except Exception:
            import imageio as iio

        n = min(num_pairs, view1.shape[0])
        v1 = view1[:n].detach().float().cpu().numpy()
        v2 = view2[:n].detach().float().cpu().numpy()

        if norm_stats is not None:
            mean = np.asarray(
                norm_stats.get("mean", [0.0]), dtype=np.float32)
            std = np.asarray(
                norm_stats.get("std", [1.0]), dtype=np.float32)
            mean = mean.reshape(1, -1, 1, 1)
            std = std.reshape(1, -1, 1, 1)
            v1 = v1 * std + mean
            v2 = v2 * std + mean

        v1 = np.clip(v1, 0.0, 1.0)
        v2 = np.clip(v2, 0.0, 1.0)

        # (B, C, H, W) -> (B, H, W, C); fold 1-channel to RGB; take RGB
        # of multi-channel.
        def _to_rgb(arr):
            if arr.shape[1] == 1:
                arr = np.repeat(arr, 3, axis=1)
            elif arr.shape[1] >= 3:
                arr = arr[:, :3]
            return np.transpose(arr, (0, 2, 3, 1))

        v1 = _to_rgb(v1)
        v2 = _to_rgb(v2)

        # Build a 2-row grid: row 0 = view1, row 1 = view2
        h, w = v1.shape[1], v1.shape[2]
        gap = 4
        grid = np.ones(
            (2 * h + gap, n * w + (n - 1) * gap, 3), dtype=np.float32)
        for i in range(n):
            x = i * (w + gap)
            grid[0:h, x:x + w] = v1[i]
            grid[h + gap:2 * h + gap, x:x + w] = v2[i]

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(out_path), (grid * 255).astype(np.uint8))

    @staticmethod
    def _save_metadata(
        output_dir, method, encoder_name, num_channels, tile_size,
        history, best_loss, best_epoch, batch_size, effective_batch,
        learning_rate, num_images, temperature, ema_decay,
        projection_dim, pretrained_model_path, norm_stats,
        run_name=None,
    ):
        """Save metadata.json alongside model.pt."""
        metadata = {
            "model_type": "ssl_pretrained",
            "ssl_method": method,
            "run_name": run_name or "",
            "architecture": {
                "type": "smp",
                "encoder_name": encoder_name,
                "input_channels": num_channels,
                "tile_size": tile_size,
            },
            "pretraining": {
                "method": method,
                "epochs": len(history),
                "best_epoch": best_epoch,
                "final_loss": history[-1]["loss"] if history else 0.0,
                "best_loss": best_loss,
                "batch_size": batch_size,
                "effective_batch_size": effective_batch,
                "learning_rate": learning_rate,
                "num_images": num_images,
                "temperature": temperature if method == "simclr" else None,
                "ema_decay": ema_decay if method == "byol" else None,
                "projection_dim": projection_dim,
                "domain_adaptive": pretrained_model_path is not None,
                "source_model": (pretrained_model_path
                                 if pretrained_model_path else None),
            },
            "normalization_stats": norm_stats,
        }
        with open(str(Path(output_dir) / "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def _compute_loss(model, view1, view2, method, temperature, ema_decay):
        """Compute SSL loss for a batch."""
        if method == "simclr":
            z1 = model(view1)
            z2 = model(view2)
            return nt_xent_loss(z1, z2, temperature)
        else:  # byol
            # Symmetric loss: both directions
            p1, z2 = model(view1, view2)
            p2, z1 = model(view2, view1)
            return byol_loss(p1, z2) + byol_loss(p2, z1)

    @staticmethod
    def _extract_encoder_state(model, method: str) -> Dict[str, torch.Tensor]:
        """Extract encoder weights with SMP-compatible encoder.* prefix.

        For SimCLR: keys are already encoder.* (from SimCLRModel.encoder).
        For BYOL: keys are online_encoder.* -> remap to encoder.*
        """
        state = {}
        for key, value in model.state_dict().items():
            if method == "simclr" and key.startswith("encoder."):
                state[key] = value.cpu().clone()
            elif method == "byol" and key.startswith("online_encoder."):
                save_key = key.replace("online_encoder.", "encoder.", 1)
                state[save_key] = value.cpu().clone()
        return state

    @staticmethod
    def _load_pretrained_encoder(
        encoder: nn.Module,
        model_path: str,
        expected_backbone: str,
    ) -> int:
        """Load encoder weights from a previously trained model.

        Handles three source formats:
          1. Full SMP segmentation model (keys like encoder.*, decoder.*, etc.)
          2. SSL pretrained encoder (keys like encoder.* only)
          3. Checkpoint with model_state_dict wrapper

        Args:
            encoder: The SMP encoder module to load weights into.
            model_path: Path to .pt file.
            expected_backbone: Expected backbone name (for logging).

        Returns:
            Number of weight tensors loaded into the encoder.
        """
        model_path = str(model_path)
        logger.info("Loading pretrained weights from: %s", model_path)

        try:
            saved = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning("Failed to load %s: %s", model_path, e)
            return 0

        # Unwrap checkpoint format
        if isinstance(saved, dict) and "model_state_dict" in saved:
            state_dict = saved["model_state_dict"]
        elif isinstance(saved, dict) and "state_dict" in saved:
            state_dict = saved["state_dict"]
        else:
            state_dict = saved

        if not isinstance(state_dict, dict):
            logger.warning("Unexpected format in %s -- not a state dict", model_path)
            return 0

        # Strip "mae." prefix if present (MAE checkpoint)
        has_mae = any(k.startswith("mae.") for k in state_dict)
        if has_mae:
            state_dict = {
                (k[4:] if k.startswith("mae.") else k): v
                for k, v in state_dict.items()
            }
            logger.info("Detected MAE checkpoint -- stripped 'mae.' prefix")

        # Extract only encoder.* keys, strip "encoder." prefix to match
        # the bare encoder module's state_dict keys
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                bare_key = key[len("encoder."):]
                encoder_state[bare_key] = value

        if not encoder_state:
            logger.warning(
                "No encoder.* keys found in %s (keys: %s...)",
                model_path,
                list(state_dict.keys())[:5])
            return 0

        # Shape-aware matching
        model_state = encoder.state_dict()
        matched = {}
        mismatched = []
        for key in encoder_state:
            if key in model_state:
                if encoder_state[key].shape == model_state[key].shape:
                    matched[key] = encoder_state[key]
                else:
                    mismatched.append(key)

        if mismatched:
            logger.warning(
                "Shape mismatches for %d keys (skipped): %s",
                len(mismatched), mismatched[:5])

        if not matched:
            logger.warning(
                "No matching encoder weights found. "
                "Source has %d encoder keys, target has %d keys.",
                len(encoder_state), len(model_state))
            return 0

        encoder.load_state_dict(matched, strict=False)
        logger.info(
            "Loaded %d/%d encoder weight tensors from pretrained model "
            "(backbone: %s)",
            len(matched), len(model_state), expected_backbone)

        return len(matched)

    def _compute_norm_stats(self, image_dir: Path) -> Dict[str, Any]:
        """Compute per-channel mean/std from a sample of images."""
        all_images = sorted([
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in SSLImageDataset.SUPPORTED_EXTENSIONS
               and not p.name.startswith('.')
               and SSLImageDataset._validate_file(p)
        ])
        if not all_images:
            return {"mean": [0.5], "std": [0.5]}

        sample_paths = all_images[:min(200, len(all_images))]
        tmp = SSLImageDataset.__new__(SSLImageDataset)
        tmp._cache = None
        pixel_sums = None
        pixel_sq_sums = None
        total_pixels = 0

        for p in sample_paths:
            try:
                arr = tmp._load_image(p)
                if arr.ndim == 2:
                    arr = arr[..., np.newaxis]
                n_pixels = arr.shape[0] * arr.shape[1]
                if pixel_sums is None:
                    pixel_sums = arr.sum(axis=(0, 1))
                    pixel_sq_sums = (arr ** 2).sum(axis=(0, 1))
                else:
                    pixel_sums += arr.sum(axis=(0, 1))
                    pixel_sq_sums += (arr ** 2).sum(axis=(0, 1))
                total_pixels += n_pixels
            except Exception:
                continue

        if total_pixels == 0 or pixel_sums is None:
            return {"mean": [0.5], "std": [0.5]}

        mean = (pixel_sums / total_pixels).tolist()
        std = np.sqrt(
            pixel_sq_sums / total_pixels - np.array(mean) ** 2
        ).tolist()

        logger.info("Normalization stats (first 3 ch): mean=%s, std=%s",
                     mean[:3], std[:3])
        return {"mean": mean, "std": std}
