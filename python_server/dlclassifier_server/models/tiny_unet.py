"""TinyUNet: lightweight depthwise-separable U-Net for fast microscopy segmentation.

Designed for 2-5 class segmentation with 1-7 channel inputs, trainable in
seconds-to-minutes on small datasets (500-5000 tiles). Defaults to
BatchRenorm (matches the rest of the extension, best empirical accuracy),
with GroupNorm and BatchNorm as alternatives for torch.compile friendliness
or maximum simplicity.

Design notes
------------
* Requires H and W divisible by 2**depth (16 at depth=4).
* Expected size at base=16, depth=4, in=3, out=3: ~150k params, ~1 GMACs
  at 256x256 input. Exact count logged at instantiation.
* Kaiming fan_out initialization is critical for depthwise convs -- PyTorch
  default under-scales them (fan_in=9), causing several epochs of flat
  training loss before progress.
* When norm="brn", the existing training_service.py warmup loop
  (set_batchrenorm_limits) applies unchanged because BRN modules register
  the same "rmax"/"dmax" buffers as the rest of the codebase's BRN uses.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.batchrenorm import BatchRenorm2d

logger = logging.getLogger(__name__)

NormKind = Literal["brn", "gn", "bn"]


def _make_norm(kind: str, channels: int) -> nn.Module:
    """Factory for the per-layer normalization module."""
    if kind == "brn":
        return BatchRenorm2d(channels)
    if kind == "gn":
        # GroupNorm with 8 groups; clamp if channels < 8.
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    raise ValueError(
        "Unknown norm kind '%s'. Use 'brn' (default), 'gn', or 'bn'." % kind
    )


class DSConv(nn.Module):
    """Depthwise-separable conv block.

    Sequence: DW 3x3 -> norm -> SiLU -> PW 1x1 -> norm -> SiLU.
    """

    def __init__(self, in_c: int, out_c: int, norm: str = "brn"):
        super().__init__()
        self.dw = nn.Conv2d(
            in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False
        )
        self.n1 = _make_norm(norm, in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.n2 = _make_norm(norm, out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.dw(x)))
        x = self.act(self.n2(self.pw(x)))
        return x


class TinyUNet(nn.Module):
    """Lightweight U-Net with depthwise-separable conv blocks.

    Parameters
    ----------
    in_channels : number of input channels (1-7 typical).
    n_classes : number of output classes (2-5 typical).
    base : number of channels in the first stage. Default 16.
    depth : number of downsampling stages. Default 4.
        With depth=4, encoder produces channel stages [base, base*2, base*4,
        base*8, base*16] and input must be divisible by 16.
    norm : one of "brn" (default), "gn", "bn". See module docstring.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 3,
        base: int = 16,
        depth: int = 4,
        norm: str = "brn",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1, got %d" % depth)
        if base < 4:
            raise ValueError("base must be >= 4, got %d" % base)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base = base
        self.depth = depth
        self.norm_kind = norm

        channels = [base * (2 ** i) for i in range(depth + 1)]

        # Stem: full 3x3 conv for the first layer.  Depthwise on 1-7 channels
        # is too thin to provide useful features -- this one dense conv is
        # worth its weight.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, base),
            nn.SiLU(inplace=True),
        )

        # Encoder: depth stages of (downsample, DSConv)
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList(
            DSConv(channels[i], channels[i + 1], norm=norm) for i in range(depth)
        )

        # Decoder: depth stages of (upsample, 1x1 merge, DSConv)
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            merged_in = channels[i] + channels[i - 1]
            self.dec.append(
                nn.Sequential(
                    nn.Conv2d(merged_in, channels[i - 1], kernel_size=1, bias=False),
                    _make_norm(norm, channels[i - 1]),
                    nn.SiLU(inplace=True),
                    DSConv(channels[i - 1], channels[i - 1], norm=norm),
                )
            )

        # 1x1 classifier head
        self.head = nn.Conv2d(base, n_classes, kernel_size=1)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Created TinyUNet(in=%d, classes=%d, base=%d, depth=%d, norm=%s) "
            "-> %d trainable params",
            in_channels, n_classes, base, depth, norm, n_params,
        )

    def _init_weights(self) -> None:
        # Kaiming fan_out is critical for depthwise convs (fan_in=9 too small).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # BatchRenorm2d inherits BN-like affine params -- default ones/zeros ok.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        mod = 1 << self.depth
        if h % mod != 0 or w % mod != 0:
            raise ValueError(
                "TinyUNet(depth=%d) requires H and W divisible by %d, got %dx%d"
                % (self.depth, mod, h, w)
            )

        x = self.stem(x)
        skips = [x]
        for enc in self.enc:
            x = self.pool(x)
            x = enc(x)
            skips.append(x)

        x = skips[-1]
        for i, dec in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skips[-2 - i]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)
