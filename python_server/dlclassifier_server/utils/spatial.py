"""Spatial padding helpers for training / inference forward passes.

Centralizes the "round H,W up to the model's required divisibility and
crop the output back" pattern so per-model spatial constraints do not
leak into the training loop or the overlay inference path.

Background
----------
U-Net-style architectures pool the spatial dimensions ``depth`` times,
which requires ``H, W`` to be divisible by ``2**depth`` (16 for the
default TinyUNet, 32 for SMP's ResNet/EfficientNet/MobileNet encoders).
MuViT additionally requires divisibility by ``patch_size`` times the
largest ``level_scale``. The training pipeline adds context padding
around every tile (``tile_size + 2 * context_pad``), and there is no
UI-level guarantee that the post-padding size is still divisible --
for example tile=512 + context=20 produces 552, which is not divisible
by 16 or 32.

The fix is the standard U-Net trick: reflection-pad the input to the
next multiple, run the model, crop the output back. This module is the
single home for that logic so every call site stays honest.

See also: claude-reports/2026-04-17_input-size-divisibility.md for
history and per-architecture requirements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_spatial_divisor(model_type: str, architecture: Dict[str, Any]) -> int:
    """Return the spatial factor that inputs must be divisible by.

    Returns 1 when no external padding is needed -- e.g., TinyUNet
    handles its own alignment inside :meth:`TinyUNet.forward`.

    Args:
        model_type: classifier type identifier (matches handler getType()).
        architecture: architecture dict from the training/inference request.

    Returns:
        Positive integer spatial factor. Callers pass this to
        :func:`pad_to_multiple` directly.
    """
    if not model_type:
        return 32
    model_type = model_type.lower()

    # TinyUNet reflects-pads inside its own forward() so external padding
    # would be redundant (and would double the work).
    if model_type == "tiny-unet":
        return 1

    if model_type == "muvit":
        patch_size = int(architecture.get("patch_size", 16))
        # level_scales is a comma-separated string like "1,4"; inputs must
        # be divisible by patch_size * max_scale so that the coarsest
        # scale still tiles evenly.
        level_scales = architecture.get("level_scales", "1")
        try:
            parts = [int(s) for s in str(level_scales).split(",") if s.strip()]
            max_scale = max(parts) if parts else 1
        except ValueError:
            max_scale = 1
        return patch_size * max_scale

    # Everything else goes through segmentation_models_pytorch U-Net-style
    # decoders (unet, unetplusplus, deeplabv3, fpn, pan, manet, linknet,
    # pspnet, fast-pretrained). SMP encoders used here all pool to 1/32.
    return 32


def pad_to_multiple(
    x: torch.Tensor, factor: int
) -> Tuple[torch.Tensor, int, int]:
    """Reflection-pad an NCHW tensor so H and W are multiples of ``factor``.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        factor: Divisibility requirement. ``<= 1`` means pass-through.

    Returns:
        ``(padded, pad_h, pad_w)`` where ``padded`` is ``x`` extended on
        the bottom/right edges by at most ``factor - 1`` pixels in each
        dimension. Hand ``pad_h`` / ``pad_w`` back to
        :func:`crop_to_original` to get the original-size output.
    """
    if factor is None or factor <= 1:
        return x, 0, 0
    h, w = x.shape[-2:]
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    # F.pad order for 4D: (left, right, top, bottom).
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), pad_h, pad_w


def crop_to_original(
    y: torch.Tensor, pad_h: int, pad_w: int
) -> torch.Tensor:
    """Undo :func:`pad_to_multiple` on the model's output tensor.

    Args:
        y: Model output with shape ``(N, C, H', W')``.
        pad_h: Pixels that were added to the bottom edge (``>= 0``).
        pad_w: Pixels that were added to the right edge (``>= 0``).

    Returns:
        ``y`` cropped to ``(N, C, H' - pad_h, W' - pad_w)``.
    """
    if not pad_h and not pad_w:
        return y
    h_out = y.shape[-2] - pad_h
    w_out = y.shape[-1] - pad_w
    return y[..., :h_out, :w_out]
