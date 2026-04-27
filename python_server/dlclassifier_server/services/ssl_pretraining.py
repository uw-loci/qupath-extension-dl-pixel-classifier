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


def get_ssl_augmentation(
    tile_size: int = 256,
    intensity_mode: str = "brightfield",
    num_channels: int = 3,
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
        intensity = [
            A.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.2, hue=0.1, p=0.8),
            A.ToGray(p=0.2),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        ]
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
        """Load all images into memory."""
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            logger.info("psutil not available -- skipping memory check, "
                        "preloading all images")
            available_mb = float('inf')

        # Estimate dataset size: assume ~1MB per tile (conservative)
        estimated_mb = len(self.image_paths) * 1.0
        if estimated_mb > available_mb * 0.25:
            logger.warning(
                "Dataset too large for in-memory preload "
                "(~%.0f MB, available %.0f MB). Using disk streaming.",
                estimated_mb, available_mb)
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
        weight_decay = float(config.get("weight_decay", 1e-4))
        warmup_epochs = int(config.get("warmup_epochs", 10))
        temperature = float(config.get("temperature", 0.5))
        ema_decay = float(config.get("ema_decay", 0.996))
        projection_dim = int(config.get("projection_dim", 256))
        hidden_dim = int(config.get("hidden_dim", 2048))
        tile_size = int(config.get("tile_size", 256))
        intensity_mode = config.get("intensity_mode", "brightfield")
        checkpoint_interval = int(config.get("checkpoint_interval", 50))

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
            )
            logger.info("SSL augmentation: mode=%s, channels=%d",
                         intensity_mode, num_channels)
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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        # Cosine schedule with linear warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # --- Mixed precision ---
        use_amp = self._device_str == "cuda"
        amp_dtype = None
        if use_amp:
            amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                         else torch.float16)
        use_grad_scaler = use_amp and amp_dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

        # --- Pre-flight VRAM estimate ---
        try:
            if self._device_str == "cuda":
                mem_info = self.gpu_manager.get_memory_info()
                total_mb = mem_info.get("total_mb", 0)
                allocated_mb = mem_info.get("allocated_mb", 0)
                free_mb = total_mb - allocated_mb
                model_mb = self.gpu_manager.estimate_model_memory(model)
                # BYOL has ~2x encoder memory (online + target)
                model_factor = 2.0 if method == "byol" else 1.0
                # SSL: model + optimizer (3x) + activations (~4x CNN)
                act_multiplier = 4.0
                if use_amp:
                    act_multiplier *= 0.6
                area_scale = (tile_size * tile_size) / (256 * 256)
                estimated_mb = (model_mb * model_factor
                                * (1 + 3 + act_multiplier * area_scale * batch_size))
                if estimated_mb > free_mb * 0.9:
                    logger.warning(
                        "VRAM estimate: %.0f MB needed vs %.0f MB free -- "
                        "OOM likely. Reduce batch size or tile size.",
                        estimated_mb, free_mb)
                else:
                    logger.info(
                        "VRAM estimate: %.0f MB needed, %.0f MB free -- OK",
                        estimated_mb, free_mb)
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
            logger.info("BYOL ema_decay=%.4f, projection_dim=%d",
                         ema_decay, projection_dim)

        best_loss = float('inf')
        best_state = None
        best_epoch = 0
        history = []
        training_start_time = _time.monotonic()

        first_batch_done = False
        try:
            for epoch in range(1, epochs + 1):
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

                optimizer.zero_grad(set_to_none=True)

                for batch_idx, (view1, view2) in enumerate(dataloader):
                    if cancel_flag and cancel_flag.is_set():
                        break

                    view1 = view1.to(self.device, non_blocking=True)
                    view2 = view2.to(self.device, non_blocking=True)

                    if not first_batch_done:
                        logger.info(
                            "First batch: shape=%s, dtype=%s, device=%s",
                            list(view1.shape), view1.dtype, view1.device)

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
                            logger.info(
                                "Peak GPU memory after first batch: %.0f MB",
                                peak_mb)
                        first_batch_done = True

                    # Optimizer step
                    if (batch_idx + 1) % grad_accum_steps == 0 or \
                            (batch_idx + 1) == len(dataloader):
                        if use_amp and use_grad_scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        # BYOL: EMA update after each optimizer step
                        if method == "byol":
                            model.update_target(ema_decay)

                    epoch_loss += loss.item() * grad_accum_steps
                    n_batches += 1

                scheduler.step()

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

                history.append({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "lr": current_lr,
                    "epoch_sec": round(epoch_duration, 2),
                })

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
                logger.info(
                    "Epoch %d/%d: loss=%.6f, lr=%.2e "
                    "(%.1fs, %.1f img/s, ~%.0fs remaining)",
                    epoch, epochs, avg_loss, current_lr,
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
        was_cancelled = cancel_flag and cancel_flag.is_set()
        if was_cancelled:
            if best_state is not None:
                # Save best model even on cancellation
                encoder_path = str(output_dir / "model.pt")
                torch.save({"model_state_dict": best_state}, encoder_path)
                self._save_metadata(
                    output_dir, method, encoder_name, num_channels,
                    tile_size, history, best_loss, best_epoch,
                    batch_size, effective_batch, learning_rate,
                    len(dataset), temperature, ema_decay,
                    projection_dim, pretrained_model_path, norm_stats)
                logger.info(
                    "Cancelled at epoch %d but saved best model "
                    "(epoch %d, loss=%.6f) to %s",
                    len(history), best_epoch, best_loss, encoder_path)
                return {
                    "status": "cancelled_saved",
                    "encoder_path": encoder_path,
                    "epochs_completed": len(history),
                    "final_loss": history[-1]["loss"] if history else 0.0,
                    "best_loss": best_loss,
                }
            else:
                logger.info("Cancelled with no completed epochs")
                return {
                    "status": "cancelled",
                    "encoder_path": "",
                    "epochs_completed": 0,
                    "final_loss": 0.0,
                    "best_loss": 0.0,
                }

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
            projection_dim, pretrained_model_path, norm_stats)

        # Clean up crash-recovery checkpoint
        best_ckpt = output_dir / "best_in_progress.pt"
        if best_ckpt.exists():
            best_ckpt.unlink()

        elapsed_total = _time.monotonic() - training_start_time
        logger.info(
            "%s pretraining complete: %d epochs (best at %d), "
            "best_loss=%.6f, %.1f min total -> %s",
            method.upper(), len(history), best_epoch,
            best_loss, elapsed_total / 60, output_dir)

        self.gpu_manager.clear_cache()

        return {
            "status": "completed",
            "encoder_path": encoder_path,
            "epochs_completed": len(history),
            "final_loss": history[-1]["loss"] if history else 0.0,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "elapsed_sec": round(elapsed_total, 1),
        }

    def _cleanup_training_memory(self):
        """Free GPU memory after training completes or fails."""
        import gc
        gc.collect()
        self.gpu_manager.clear_cache()
        self.gpu_manager.log_memory_status(prefix="SSL cleanup: ")

    @staticmethod
    def _save_metadata(
        output_dir, method, encoder_name, num_channels, tile_size,
        history, best_loss, best_epoch, batch_size, effective_batch,
        learning_rate, num_images, temperature, ema_decay,
        projection_dim, pretrained_model_path, norm_stats,
    ):
        """Save metadata.json alongside model.pt."""
        metadata = {
            "model_type": "ssl_pretrained",
            "ssl_method": method,
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
