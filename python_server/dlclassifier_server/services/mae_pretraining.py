"""MAE (Masked Autoencoder) pretraining service for MuViT encoder.

Trains the MuViT encoder in a self-supervised manner using masked image
modeling (SimMIM-style). No labels required -- the model learns to
reconstruct randomly masked patches from visible context across multiple
resolution levels.

After pretraining, encoder weights are saved as a standard model.pt that
can be loaded via the "Continue from model" feature in the training
dialog. The encoder.* keys transfer directly to MuViTSegmentation while
MAE-specific keys (decoder, mask_token) are automatically skipped.
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .gpu_manager import GPUManager, get_gpu_manager
from .muvit_model import MODEL_CONFIGS, extract_multi_resolution

logger = logging.getLogger(__name__)


# ==================== Dataset ====================

class MAEImageDataset(Dataset):
    """Dataset for MAE pretraining on unlabeled image tiles.

    Loads image tiles from a directory and applies random geometric
    augmentations (flip, rotate90). No masks or labels needed.
    Images larger than tile_size are randomly cropped; smaller images
    are reflection-padded.
    """

    SUPPORTED_EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.raw'}

    def __init__(
        self,
        image_dir: str,
        tile_size: int = 256,
        normalize_stats: Optional[Dict] = None,
    ):
        self.tile_size = tile_size
        self.normalize_stats = normalize_stats

        image_dir = Path(image_dir)
        self.image_paths = sorted([
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ])

        if not self.image_paths:
            raise ValueError("No image files found in %s" % image_dir)

        logger.info("MAEImageDataset: found %d images in %s",
                     len(self.image_paths), image_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])

        # Ensure HWC
        if img.ndim == 2:
            img = img[..., np.newaxis]

        h, w = img.shape[:2]

        # Random crop if larger than tile_size
        if h > self.tile_size or w > self.tile_size:
            y = np.random.randint(0, max(1, h - self.tile_size))
            x = np.random.randint(0, max(1, w - self.tile_size))
            img = img[y:y + self.tile_size, x:x + self.tile_size]

        # Pad if smaller
        h, w = img.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            pad_h = self.tile_size - h
            pad_w = self.tile_size - w
            img = np.pad(img,
                         ((0, pad_h), (0, pad_w), (0, 0)),
                         mode='reflect')

        # Normalize
        if self.normalize_stats:
            mean = np.array(self.normalize_stats.get("mean", [0.5]),
                            dtype=np.float32)
            std = np.array(self.normalize_stats.get("std", [0.5]),
                           dtype=np.float32)
            std = np.maximum(std, 1e-6)
            img = (img - mean) / std

        # Random geometric augmentation (no color aug for reconstruction)
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=0).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k=k, axes=(0, 1)).copy()

        # HWC -> CHW float32
        img_chw = img.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(img_chw)

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
        img = Image.open(path_str)
        return np.array(img, dtype=np.float32) / 255.0

    @property
    def n_channels(self):
        """Number of channels from first image."""
        img = self._load_image(self.image_paths[0])
        return 1 if img.ndim == 2 else img.shape[2]


# ==================== MAE Model ====================

class MuViTMAEWrapper(nn.Module):
    """SimMIM-style masked autoencoder around MuViT2d encoder.

    Masking strategy: patches in the level-0 input image are zeroed out.
    The encoder processes the partially-masked multi-resolution input
    (context levels remain intact, providing global cues). A lightweight
    MLP decoder predicts original pixel values from encoder features.
    MSE loss is computed only on masked patch locations.

    The encoder learns useful representations because:
    - 75% of detail-level patches are masked (strong bottleneck)
    - Cross-resolution attention lets context tokens guide reconstruction
    - Per-patch target normalization prevents trivial solutions
    """

    def __init__(
        self,
        n_channels: int,
        patch_size: int = 16,
        levels: Tuple[float, ...] = (1.0, 4.0),
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        rope: str = "per_layer",
        mask_ratio: float = 0.75,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.levels = levels
        self.dim = dim
        self.mask_ratio = mask_ratio

        from muvit import MuViT2d

        self.encoder = MuViT2d(
            n_channels=n_channels,
            patch_size=patch_size,
            levels=levels,
            dim=dim,
            depth=depth,
            heads=heads,
            rope=rope,
            use_level_embed=True,
        )

        # Lightweight decoder for pixel reconstruction
        decoder_layers = []
        current_dim = dim
        for _ in range(decoder_depth):
            decoder_layers.extend([
                nn.Linear(current_dim, decoder_dim),
                nn.GELU(),
                nn.LayerNorm(decoder_dim),
            ])
            current_dim = decoder_dim
        decoder_layers.append(
            nn.Linear(decoder_dim, patch_size * patch_size * n_channels)
        )
        self.decoder = nn.Sequential(*decoder_layers)

        logger.info(
            "Created MuViTMAEWrapper: dim=%d, depth=%d, heads=%d, "
            "mask_ratio=%.2f, levels=%s",
            dim, depth, heads, mask_ratio, levels
        )

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, H, W) -> (B, N, patch_size^2 * C)."""
        B, C, H, W = imgs.shape
        p = self.patch_size
        # (B, C, nH, nW, p, p) -> (B, nH, nW, C, p, p) -> (B, N, C*p*p)
        patches = imgs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        return patches.reshape(B, -1, C * p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: mask, encode, decode, compute reconstruction loss.

        Args:
            x: (B, C, H, W) input images

        Returns:
            Scalar MSE loss on masked patches
        """
        B, C, H, W = x.shape
        p = self.patch_size
        patches_h = H // p
        patches_w = W // p
        n_patches = patches_h * patches_w
        n_masked = int(n_patches * self.mask_ratio)

        # Random mask for level 0
        noise = torch.rand(B, n_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, n_patches, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :n_masked], True)

        # Create multi-resolution views
        imgs, bboxes = extract_multi_resolution(x, self.levels, self.patch_size)

        # Zero out masked patches in level-0 image only
        mask_spatial = mask.float().view(B, patches_h, patches_w)
        mask_pixels = mask_spatial.repeat_interleave(
            p, dim=1).repeat_interleave(p, dim=2)
        masked_imgs = imgs.clone()
        masked_imgs[:, 0] = imgs[:, 0] * (1.0 - mask_pixels.unsqueeze(1))

        # Encode (processes masked image + intact context levels)
        features = self.encoder(masked_imgs, bboxes)  # (B, N_total, dim)

        # Extract level 0 features only
        level0_features = features[:, :n_patches, :]

        # Decode to pixel predictions
        pred_pixels = self.decoder(level0_features)  # (B, N, p*p*C)

        # Original patches as target
        orig_patches = self._patchify(x)  # (B, N, p*p*C)

        # Per-patch normalization of targets (standard in MAE literature)
        target_mean = orig_patches.mean(dim=-1, keepdim=True)
        target_var = orig_patches.var(dim=-1, keepdim=True)
        target = (orig_patches - target_mean) / (target_var + 1e-6).sqrt()

        # MSE loss on masked patches only
        loss = F.mse_loss(pred_pixels[mask], target[mask])
        return loss


# ==================== Pretraining Service ====================

class MAEPretrainingService:
    """Service for MAE pretraining of MuViT encoder.

    Trains the MuViT encoder using masked image modeling on unlabeled
    tile images. The resulting weights (model.pt) contain encoder.* keys
    that transfer directly to MuViTSegmentation via the existing
    pretrained weight loading mechanism (shape-matched partial load).
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

        logger.info("MAEPretrainingService on device: %s", self._device_str)

    def pretrain(
        self,
        config: Dict[str, Any],
        data_path: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
        setup_callback: Optional[Callable] = None,
        cancel_flag: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Run MAE pretraining.

        Args:
            config: Dict with keys:
                model_config, patch_size, level_scales, rope_mode,
                mask_ratio, epochs, batch_size, learning_rate,
                weight_decay, warmup_epochs, tile_size
            data_path: Directory of image tiles (or training data root
                       with train/images/ subdirectory)
            output_dir: Where to save pretrained encoder weights
            progress_callback: fn(epoch, total, loss, lr)
            setup_callback: fn(phase, data=None)
            cancel_flag: threading.Event to signal cancellation

        Returns:
            Dict with status, encoder_path, epochs_completed, final_loss
        """

        def _report(phase, data=None):
            if setup_callback:
                try:
                    setup_callback(phase, data)
                except Exception:
                    pass

        # Parse config
        model_config = config.get("model_config", "muvit-base")
        patch_size = int(config.get("patch_size", 16))
        level_scales = config.get("level_scales", "1,4")
        levels = tuple(float(s.strip()) for s in level_scales.split(","))
        rope_mode = config.get("rope_mode", "per_layer")
        mask_ratio = float(config.get("mask_ratio", 0.75))
        epochs = int(config.get("epochs", 100))
        batch_size = int(config.get("batch_size", 8))
        grad_accum_steps = int(config.get("grad_accum_steps", 1))
        learning_rate = float(config.get("learning_rate", 1.5e-4))
        weight_decay = float(config.get("weight_decay", 0.05))
        warmup_epochs = int(config.get("warmup_epochs", 5))
        tile_size = int(config.get("tile_size", 256))
        checkpoint_interval = int(config.get("checkpoint_interval", 50))

        cfg = MODEL_CONFIGS.get(model_config, MODEL_CONFIGS["muvit-base"])

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

        # --- Compute normalization stats from sample ---
        _report("computing_stats")
        norm_stats = self._compute_norm_stats(image_dir)

        # --- Create dataset ---
        dataset = MAEImageDataset(
            image_dir=str(image_dir),
            tile_size=tile_size,
            normalize_stats=norm_stats,
        )
        num_channels = dataset.n_channels
        num_images = len(dataset)
        logger.info("Dataset: %d images, %d channels, tile_size=%d",
                     num_images, num_channels, tile_size)

        # Dataset size recommendations
        if num_images < 10:
            logger.warning(
                "Very few images (%d). MAE pretraining may not be effective. "
                "Consider collecting more unlabeled images for better results.",
                num_images)
        elif num_images < 50:
            logger.info(
                "Small dataset (%d images). Random masking will create "
                "diverse reconstruction tasks. Consider 300-500 epochs.",
                num_images)
        elif num_images < 200:
            logger.info(
                "Moderate dataset (%d images). 100-200 epochs recommended.",
                num_images)
        else:
            logger.info(
                "Good dataset size (%d images). 50-100 epochs should suffice.",
                num_images)

        # Auto-adjust batch size if larger than dataset
        if batch_size > num_images:
            old_bs = batch_size
            batch_size = max(1, num_images // 2)
            if grad_accum_steps == 1:
                grad_accum_steps = max(1, old_bs // batch_size)
            logger.info(
                "Batch size reduced %d -> %d (dataset has only %d images), "
                "grad accum = %d",
                old_bs, batch_size, num_images, grad_accum_steps)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=self._device_str == "cuda",
            drop_last=len(dataset) > batch_size,
        )

        # --- Create model ---
        _report("creating_model")
        num_heads = config.get("num_heads")
        if num_heads is not None:
            num_heads = int(num_heads)
        else:
            num_heads = cfg["heads"]

        model = MuViTMAEWrapper(
            n_channels=num_channels,
            patch_size=patch_size,
            levels=levels,
            dim=cfg["dim"],
            depth=cfg["enc_layers"],
            heads=num_heads,
            rope=rope_mode,
            mask_ratio=mask_ratio,
        )
        model = model.to(self.device)

        param_count = sum(p.numel() for p in model.parameters())
        logger.info("MAE model: %.1fM parameters", param_count / 1e6)

        # --- Optimizer: AdamW with (0.9, 0.95) as per MAE convention ---
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

        # --- Training loop ---
        _report("starting_training")
        effective_batch = batch_size * grad_accum_steps
        logger.info(
            "MAE pretraining: %d epochs, batch=%d (effective %d, accum=%d), "
            "lr=%.2e, mask=%.0f%%",
            epochs, batch_size, effective_batch, grad_accum_steps,
            learning_rate, mask_ratio * 100)

        best_loss = float('inf')
        best_state = None
        history = []

        for epoch in range(1, epochs + 1):
            if cancel_flag and cancel_flag.is_set():
                logger.info("Pretraining cancelled at epoch %d", epoch)
                break

            model.train()
            epoch_loss = 0.0
            n_batches = 0

            optimizer.zero_grad(set_to_none=True)

            for batch_idx, images in enumerate(dataloader):
                if cancel_flag and cancel_flag.is_set():
                    break

                images = images.to(self.device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        loss = model(images)
                    loss = loss / grad_accum_steps
                    if use_grad_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    loss = model(images)
                    loss = loss / grad_accum_steps
                    loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or \
                        (batch_idx + 1) == len(dataloader):
                    if use_amp and use_grad_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    elif use_amp:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0)
                        optimizer.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * grad_accum_steps
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(1, n_batches)
            current_lr = optimizer.param_groups[0]['lr']

            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "lr": current_lr,
            })

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

            if progress_callback:
                try:
                    progress_callback(epoch, epochs, avg_loss, current_lr)
                except Exception:
                    pass

            if epoch % 10 == 0 or epoch == 1:
                logger.info("Epoch %d/%d: loss=%.6f, lr=%.2e",
                            epoch, epochs, avg_loss, current_lr)

            # Periodic checkpoint
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                ckpt_path = output_dir / ("checkpoint_epoch_%d.pt" % epoch)
                torch.save(model.state_dict(), str(ckpt_path))
                logger.info("Saved checkpoint: %s", ckpt_path)

        # --- Save ---
        _report("saving_model")

        if best_state is not None:
            model.load_state_dict(best_state)

        encoder_path = str(output_dir / "model.pt")
        torch.save(model.state_dict(), encoder_path)

        metadata = {
            "model_type": "muvit_mae",
            "architecture": {
                "type": "muvit",
                "model_config": model_config,
                "patch_size": patch_size,
                "level_scales": level_scales,
                "rope_mode": rope_mode,
                "dim": cfg["dim"],
                "depth": cfg["enc_layers"],
                "heads": num_heads,
                "input_channels": num_channels,
            },
            "pretraining": {
                "epochs": len(history),
                "mask_ratio": mask_ratio,
                "final_loss": history[-1]["loss"] if history else 0.0,
                "best_loss": best_loss,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_images": len(dataset),
            },
            "normalization_stats": norm_stats,
        }
        with open(str(output_dir / "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "MAE pretraining complete: %d epochs, best_loss=%.6f -> %s",
            len(history), best_loss, output_dir)

        self.gpu_manager.clear_cache()

        return {
            "status": "completed",
            "encoder_path": encoder_path,
            "epochs_completed": len(history),
            "final_loss": history[-1]["loss"] if history else 0.0,
            "best_loss": best_loss,
        }

    def _compute_norm_stats(self, image_dir: Path) -> Dict[str, Any]:
        """Compute per-channel mean/std from a sample of images."""
        all_images = sorted([
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in MAEImageDataset.SUPPORTED_EXTENSIONS
        ])
        if not all_images:
            return {"mean": [0.5], "std": [0.5]}

        sample_paths = all_images[:min(200, len(all_images))]
        # Use a temporary dataset instance for its _load_image method
        tmp = MAEImageDataset.__new__(MAEImageDataset)
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
