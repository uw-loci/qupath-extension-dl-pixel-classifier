"""
MuViT model wrapper for the DL pixel classifier.

Wraps the muvit library's MuViTMAE2d encoder with a lightweight pixel
classification head to produce per-pixel segmentation predictions
compatible with the existing training and inference pipeline.

The muvit package exposes:
  - muvit.mae.MuViTMAE2d : full MAE model (encoder + decoder)
  - muvit.mae.MuViTEncoder: encoder-only (extracted via .encoder)
  - muvit.data.MuViTDataset: base dataset class

To create a standalone encoder for segmentation, we instantiate MuViTMAE2d
with a minimal decoder and extract its .encoder attribute.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Model size configurations
# num_layers maps to MuViTMAE2d's num_layers parameter
MODEL_CONFIGS = {
    "muvit-small": {"num_layers": 6, "dec_dim": 128},
    "muvit-base": {"num_layers": 12, "dec_dim": 256},
    "muvit-large": {"num_layers": 16, "dec_dim": 384},
}


def extract_multi_resolution(
    x: torch.Tensor,
    levels: Tuple[float, ...],
    patch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract multi-resolution views and bounding boxes from input tensor.

    Given a (B, C, H, W) tensor, produce:
    - imgs: (B, L, C, H, W) - same pixel size at each level
    - bboxes: (B, L, 2, 2) - [min_yx, max_yx] in normalized [0,1] coords

    Level 0 (scale=1): full-resolution tile as-is.
    Level k (scale>1): center crop of 1/scale of the tile, upsampled back
    to full pixel size.  Simulates a larger physical region at lower
    effective resolution.

    Args:
        x: (B, C, H, W) input images
        levels: tuple of scale factors, e.g. (1.0, 4.0)
        patch_size: minimum crop size

    Returns:
        imgs (B, L, C, H, W), bboxes (B, L, 2, 2)
    """
    B, C, H, W = x.shape
    imgs = []
    bboxes = []

    for scale in levels:
        if scale == 1.0:
            imgs.append(x)
            bbox = torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]],
                device=x.device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1, -1)
            bboxes.append(bbox)
        else:
            crop_frac = 1.0 / scale
            crop_h = max(patch_size, int(H * crop_frac))
            crop_w = max(patch_size, int(W * crop_frac))
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2

            crop = x[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
            if crop.shape[2] != H or crop.shape[3] != W:
                crop = F.interpolate(crop, size=(H, W), mode="bilinear",
                                     align_corners=False)
            imgs.append(crop)

            frac_start_h = start_h / H
            frac_start_w = start_w / W
            frac_end_h = (start_h + crop_h) / H
            frac_end_w = (start_w + crop_w) / W
            bbox = torch.tensor(
                [[frac_start_h, frac_start_w], [frac_end_h, frac_end_w]],
                device=x.device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1, -1)
            bboxes.append(bbox)

    imgs = torch.stack(imgs, dim=1)
    bboxes = torch.stack(bboxes, dim=1)
    return imgs, bboxes


class MuViTSegmentation(nn.Module):
    """MuViT encoder with a pixel classification segmentation head.

    Uses the muvit library's MuViTMAE2d to create an encoder, then adds a
    convolutional decoder head that upsamples encoder features to the input
    resolution and produces per-pixel class logits.

    The model accepts standard (B, C, H, W) input tensors. Multi-resolution
    level extraction is handled internally by cropping the input at different
    scales (centered crops at each level's spatial extent).
    """

    def __init__(
        self,
        in_channels: int,
        classes: int,
        model_config: str = "muvit-base",
        patch_size: int = 16,
        level_scales: str = "1,4",
        rope_mode: str = "per_layer",
        num_heads: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = classes
        self.model_config = model_config
        self.patch_size = patch_size
        self.level_scales_str = level_scales
        self.levels = tuple(float(s.strip()) for s in level_scales.split(","))
        self.rope_mode = rope_mode

        cfg = MODEL_CONFIGS.get(model_config, MODEL_CONFIGS["muvit-base"])
        self.dec_dim = cfg["dec_dim"]

        try:
            from muvit.mae import MuViTMAE2d
        except ImportError:
            raise ImportError(
                "muvit package not installed. Install with: pip install muvit"
            )

        # Create MAE model and extract its encoder.
        # We pass a minimal decoder (1 layer) since we discard it.
        mae = MuViTMAE2d(
            in_channels=in_channels,
            levels=self.levels,
            patch_size=patch_size,
            num_layers=cfg["num_layers"],
            num_layers_decoder=1,
        )
        self.encoder = mae.encoder

        # Probe encoder output dimension with a dummy forward pass
        self.encoder_dim = self._probe_encoder_dim()

        # Segmentation head: spatial Conv2d on encoder features
        # Uses compute_features() output shape (B, L, D, H', W')
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.encoder_dim, self.dec_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.dec_dim, self.num_classes, kernel_size=1),
        )

        logger.info(
            "Created MuViTSegmentation: config=%s, patch=%d, levels=%s, "
            "encoder_dim=%d, num_layers=%d, classes=%d",
            model_config, patch_size, self.levels,
            self.encoder_dim, cfg["num_layers"], classes,
        )

    def _probe_encoder_dim(self) -> int:
        """Determine encoder output dimension via a dummy forward pass.

        Uses compute_features() which returns (B, L, D, H', W') spatial
        features -- the same method used in the real forward pass.
        """
        size = self.patch_size * 2  # minimal valid spatial size
        dummy_img = torch.zeros(
            1, len(self.levels), self.in_channels, size, size)
        dummy_bbox = torch.zeros(1, len(self.levels), 2, 2)
        dummy_bbox[:, :, 1, :] = 1.0
        with torch.no_grad():
            out = self.encoder.compute_features(dummy_img, dummy_bbox)
        # out shape: (B, L, D, H', W') -- D is at index 2
        dim = out.shape[2]
        logger.debug("Probed encoder dim: %d", dim)
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing (B, num_classes, H, W) logits.

        Accepts standard (B, C, H, W) input and handles multi-resolution
        extraction internally.
        """
        B, C, H, W = x.shape

        # Extract multi-resolution views
        imgs, bboxes = extract_multi_resolution(x, self.levels, self.patch_size)

        # Encoder: (B, L, C, H, W) + (B, L, 2, 2) -> spatial (B, L, D, H', W')
        features = self.encoder.compute_features(imgs, bboxes)

        # Use level-0 (highest resolution) features: (B, D, H', W')
        level0 = features[:, 0]

        # Apply segmentation head: (B, num_classes, H', W')
        logits = self.seg_head(level0)

        # Upsample to original input size
        if logits.shape[2] != H or logits.shape[3] != W:
            logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                                   align_corners=False)

        return logits


def create_muvit_model(
    architecture: Dict[str, Any],
    num_channels: int,
    num_classes: int,
) -> nn.Module:
    """Factory function to create a MuViT segmentation model.

    Args:
        architecture: Dict with keys: model_config, patch_size, level_scales,
                      rope_mode, num_heads (all optional with defaults)
        num_channels: Number of input image channels
        num_classes: Number of output segmentation classes

    Returns:
        MuViTSegmentation model instance
    """
    # Check both "model_config" (Python convention) and "backbone" (Java convention).
    # Must not use a valid default for model_config -- that shadows the backbone value.
    model_config = architecture.get("model_config")
    if model_config not in MODEL_CONFIGS:
        model_config = architecture.get("backbone", "muvit-base")
    if model_config not in MODEL_CONFIGS:
        model_config = "muvit-base"

    patch_size = int(architecture.get("patch_size", 16))
    level_scales = str(architecture.get("level_scales", "1,4"))
    rope_mode = str(architecture.get("rope_mode", "per_layer"))
    num_heads = architecture.get("num_heads")
    if num_heads is not None:
        num_heads = int(num_heads)

    # Ensure int types: Gson round-trips may produce floats (3.0 instead of 3)
    # which cause TypeError in nn.Linear when multiplied with np.prod().
    return MuViTSegmentation(
        in_channels=int(num_channels),
        classes=int(num_classes),
        model_config=model_config,
        patch_size=patch_size,
        level_scales=level_scales,
        rope_mode=rope_mode,
        num_heads=num_heads,
    )
