"""
MuViT model wrapper for the DL pixel classifier.

Wraps the MuViT encoder with a lightweight pixel classification head to produce
per-pixel segmentation predictions compatible with the existing training and
inference pipeline.

Phase 1: Trains from scratch (no MAE pretraining). Single forward pass produces
(B, num_classes, H, W) output matching the smp model interface.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Model size configurations: (encoder_layers, hidden_dim, decoder_layers, decoder_dim)
MODEL_CONFIGS = {
    "muvit-small": {"enc_layers": 6, "dim": 256, "dec_layers": 1, "dec_dim": 128, "heads": 4},
    "muvit-base": {"enc_layers": 12, "dim": 512, "dec_layers": 2, "dec_dim": 256, "heads": 8},
    "muvit-large": {"enc_layers": 16, "dim": 768, "dec_layers": 2, "dec_dim": 384, "heads": 12},
}


class MuViTSegmentation(nn.Module):
    """MuViT encoder with a pixel classification segmentation head.

    Wraps the muvit library's MuViT2d encoder and adds a simple convolutional
    decoder head that upsamples encoder features to the input resolution and
    produces per-pixel class logits.

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
        self.dim = cfg["dim"]
        self.enc_layers = cfg["enc_layers"]
        self.dec_dim = cfg["dec_dim"]
        heads = num_heads if num_heads is not None else cfg["heads"]

        try:
            from muvit import MuViT2d
        except ImportError:
            raise ImportError(
                "muvit package not installed. Install with: pip install muvit"
            )

        self.encoder = MuViT2d(
            n_channels=in_channels,
            patch_size=patch_size,
            levels=self.levels,
            dim=self.dim,
            depth=self.enc_layers,
            heads=heads,
            rope=rope_mode,
            use_level_embed=True,
        )

        # Segmentation head: project encoder features to class logits
        # MuViT encoder output: (B, N_total, dim) where N_total is across all levels
        # We only decode the highest-resolution level (level 0) for pixel prediction
        self.seg_head = nn.Sequential(
            nn.Linear(self.dim, self.dec_dim),
            nn.GELU(),
            nn.Linear(self.dec_dim, self.num_classes),
        )

        # Track input size for reshape
        self._input_h = None
        self._input_w = None

        logger.info(
            "Created MuViTSegmentation: config=%s, patch=%d, levels=%s, "
            "dim=%d, depth=%d, heads=%d, rope=%s, classes=%d",
            model_config, patch_size, self.levels,
            self.dim, self.enc_layers, heads, rope_mode, classes,
        )

    def _extract_multi_resolution(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract multi-resolution crops and bounding boxes from input.

        Given a (B, C, H, W) tensor, produce:
        - imgs: (B, L, C, H, W) - same pixel size at each level
        - bboxes: (B, L, 2, 2) - [min_yx, max_yx] in normalized [0,1] coords

        Level 0: full-resolution center crop
        Level k (k>0): larger physical region downsampled to same pixel size
        """
        B, C, H, W = x.shape
        L = len(self.levels)

        imgs = []
        bboxes = []

        for level_idx, scale in enumerate(self.levels):
            if scale == 1.0:
                # Level 0: use the full tile as-is
                imgs.append(x)
                # Bbox covers the full [0,1] range
                bbox = torch.tensor(
                    [[0.0, 0.0], [1.0, 1.0]],
                    device=x.device, dtype=torch.float32
                ).unsqueeze(0).expand(B, -1, -1)
                bboxes.append(bbox)
            else:
                # Higher scale levels: simulate a larger physical region
                # by extracting a center crop that's 1/scale of the tile,
                # then upsampling back to full resolution.
                # This means the context level "sees" the center at lower effective
                # resolution, equivalent to a context_scale approach.
                crop_frac = 1.0 / scale
                crop_h = max(self.patch_size, int(H * crop_frac))
                crop_w = max(self.patch_size, int(W * crop_frac))
                start_h = (H - crop_h) // 2
                start_w = (W - crop_w) // 2

                crop = x[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
                # Upsample to match level 0 dimensions
                if crop.shape[2] != H or crop.shape[3] != W:
                    crop = F.interpolate(crop, size=(H, W), mode="bilinear",
                                         align_corners=False)
                imgs.append(crop)

                # Bbox in world coordinates: this crop covers a smaller physical
                # region but is viewed at lower resolution
                frac_start_h = start_h / H
                frac_start_w = start_w / W
                frac_end_h = (start_h + crop_h) / H
                frac_end_w = (start_w + crop_w) / W
                bbox = torch.tensor(
                    [[frac_start_h, frac_start_w], [frac_end_h, frac_end_w]],
                    device=x.device, dtype=torch.float32
                ).unsqueeze(0).expand(B, -1, -1)
                bboxes.append(bbox)

        # Stack to (B, L, C, H, W) and (B, L, 2, 2)
        imgs = torch.stack(imgs, dim=1)
        bboxes = torch.stack(bboxes, dim=1)
        return imgs, bboxes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing (B, num_classes, H, W) logits.

        Accepts standard (B, C, H, W) input and handles multi-resolution
        extraction internally.
        """
        B, C, H, W = x.shape
        self._input_h = H
        self._input_w = W

        # Extract multi-resolution views
        imgs, bboxes = self._extract_multi_resolution(x)

        # MuViT encoder: (B, L, C, H, W) + (B, L, 2, 2) -> (B, N_total, dim)
        features = self.encoder(imgs, bboxes)

        # Compute number of patches for level 0 (highest resolution)
        patches_h = H // self.patch_size
        patches_w = W // self.patch_size
        n_patches_level0 = patches_h * patches_w

        # Extract only level 0 tokens (first n_patches_level0 tokens)
        level0_features = features[:, :n_patches_level0, :]  # (B, n_patches, dim)

        # Apply segmentation head
        logits = self.seg_head(level0_features)  # (B, n_patches, num_classes)

        # Reshape to spatial grid and upsample to input resolution
        logits = logits.permute(0, 2, 1)  # (B, num_classes, n_patches)
        logits = logits.view(B, self.num_classes, patches_h, patches_w)

        # Upsample to original input size
        if patches_h != H or patches_w != W:
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
    model_config = architecture.get("model_config", "muvit-base")
    # Also accept from "backbone" key (how Java sends it)
    if model_config not in MODEL_CONFIGS:
        backbone = architecture.get("backbone", "muvit-base")
        if backbone in MODEL_CONFIGS:
            model_config = backbone

    patch_size = int(architecture.get("patch_size", 16))
    level_scales = str(architecture.get("level_scales", "1,4"))
    rope_mode = str(architecture.get("rope_mode", "per_layer"))
    num_heads = architecture.get("num_heads")
    if num_heads is not None:
        num_heads = int(num_heads)

    return MuViTSegmentation(
        in_channels=num_channels,
        classes=num_classes,
        model_config=model_config,
        patch_size=patch_size,
        level_scales=level_scales,
        rope_mode=rope_mode,
        num_heads=num_heads,
    )
