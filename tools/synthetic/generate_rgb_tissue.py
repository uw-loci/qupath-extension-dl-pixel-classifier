#!/usr/bin/env python3
"""Generate a synthetic brightfield RGB (H&E-like) slide with perfect ground truth.

The DL pixel classifier has no dataset whose answer we know exactly, so a regression
run can only ever say "the overlay looks the same as last time". This generator makes
an image whose label map is known by construction: a release test can train on it,
run inference, and score the resulting objects against ground_truth.png.

Three regions are drawn:

  Background (2) -- slide glass: near-white with mild illumination shading.
  Stroma     (0) -- eosinophilic wash, collagen striations, sparse ELONGATED nuclei.
  Tumor      (1) -- nests 200-2000 px across: dense ROUND haematoxylin nuclei and a
                    slightly more basophilic cytoplasm.

Stroma and Tumor are separable from local texture alone -- nuclear shape, nuclear
density and cytoplasm tint all differ inside a few hundred pixels. Nothing about the
class is encoded at whole-image scale. That is deliberate: this suite exists to catch
tiling artifacts, so a tile must be classifiable without the rest of the slide. A slow
multiplicative stain gradient IS painted across the slide, because cross-tile
normalization behaviour is one of the things under test -- but it never decides a class.

--difficulty sets how AMBIGUOUS that boundary is, and nothing else. It is not a
cosmetic knob: a fixture the model is always certain about cannot reproduce a
blend-mode regression, because neighbouring tiles never disagree about a pixel
neither of them doubts. easy keeps the hard-edged nests with a dark rim; medium
drops the rim, spreads the transition over tens of pixels, lets it wander off the
true boundary, and scatters tumor-like clusters through the stroma and stroma-like
clearings through the nests; hard widens all of that and flattens the interiors of
the largest nests. The label mask NEVER moves: every difficulty rasterizes the same
polygons, so for a given seed all three write a byte-identical ground_truth.png and
annotations.geojson and differ only in image.tif.

Three thin stroma tracts run between flanking rows of nests at every difficulty.
They are narrower than a 256 px tile stride on purpose -- a structure the tile grid
cannot resolve is where a blended overlay and a centre-cropped one visibly part.

Ground truth is exact rather than traced: the label mask is rasterized from the same
polygons that are written to annotations.geojson, so --annotation-mode perfect agrees
with the mask to the pixel.

Outputs (under --out, default tools/synthetic/data/rgb/):

    image.tif           RGB uint8, tiled TIFF, resolution tags set from --pixel-size-um
    ground_truth.png    uint8 label mask, mode L, 0=Stroma 1=Tumor 2=Background
    annotations.geojson QuPath-importable FeatureCollection (File > Import objects)
    manifest.json       seed, difficulty, class names/colors, pixel size, dims, areas

Usage:
    python tools/synthetic/generate_rgb_tissue.py
    python tools/synthetic/generate_rgb_tissue.py --out /tmp/rgb --seed 7 \
        --width 4000 --height 3000 --annotation-mode sparse --difficulty hard

Run it in the DL classifier venv:
    python_server/venv/bin/python tools/synthetic/generate_rgb_tissue.py

ASCII-only output (the project runs on Windows cp1252).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

GENERATOR = "generate_rgb_tissue.py"
GENERATOR_VERSION = "1.1.0"

# Label values in ground_truth.png. Background is glass, not a tissue class.
LBL_STROMA = 0
LBL_TUMOR = 1
LBL_BACKGROUND = 2

# Base colors, sampled off real H&E. (R, G, B) before the stain gradient.
GLASS_RGB = (246, 244, 248)
STROMA_RGB = (228, 168, 190)
STROMA_NUC_RGB = (108, 72, 148)
TUMOR_RGB = (212, 156, 198)
TUMOR_NUC_RGB = (60, 36, 106)

# Nuclear packing. Stroma is sparse and stays sparse; tumor density is scaled by the
# local tumorness field, which is how nests fade out at their margins.
STROMA_DENSITY = 0.42
TUMOR_DENSITY = 0.86

# Row band height for the texture pass. Trades peak memory against loop overhead.
BAND = 384

# How ambiguous the Stroma/Tumor boundary is. None of this touches the ground truth:
# every knob here perturbs the PICTURE only, and the label mask is still rasterized
# from the nest polygons. Lengths are in pixels at 8000x6000 and scale with the image.
#
# The point of the harder settings is to produce pixels a model is genuinely unsure
# about, because two overlapping tiles that see different context will then resolve
# them differently -- which is the only way this fixture can catch a blend-mode
# regression (CENTER_CROP filling the overlay with tile-grid blocks). A dataset the
# model is always certain about passes the suite whether or not the bug is present.
DIFFICULTY_PRESETS = {
    "easy": {
        # Hard-edged nests with a dark rim: the mask boundary IS the picture boundary.
        "soft_px": 0.0,
        "edge_noise_px": 0.0,
        "edge_noise_cell": 110.0,
        "draw_rim": True,
        "n_islands": 0,
        "n_gaps": 0,
        "n_ambiguous": 0,
        "flat_core": 0.0,
    },
    "medium": {
        # No rim, a transition tens of pixels wide that wanders off the true boundary,
        # tumor-like cell clusters loose in the stroma and stroma-like holes in nests.
        "soft_px": 48.0,
        "edge_noise_px": 30.0,
        "edge_noise_cell": 110.0,
        "draw_rim": False,
        "n_islands": 16,
        "n_gaps": 12,
        "n_ambiguous": 0,
        "flat_core": 0.0,
    },
    "hard": {
        # Medium, plus nests several 256 px tiles across whose interior is nearly
        # featureless, plus regions whose density sits between the two classes.
        "soft_px": 62.0,
        "edge_noise_px": 42.0,
        "edge_noise_cell": 170.0,
        "draw_rim": False,
        "n_islands": 24,
        "n_gaps": 18,
        "n_ambiguous": 7,
        "flat_core": 0.62,
    },
}

# Thin stroma corridors between two rows of nests. Narrower than a typical tile
# stride on purpose: a structure the tile grid cannot resolve is exactly where a
# blended overlay and a centre-cropped one part company. Wide enough that even the
# hardest softening leaves a stroma core rather than swallowing the tract whole.
N_TRACTS = 3
TRACT_WIDTH_PX = (90.0, 210.0)

# Nests several 256 px tiles across. They exist at every difficulty -- only hard
# flattens their interiors -- so that the ground truth depends on the seed alone.
N_GIANT_NESTS = 2

# Polygon simplification tolerance for sparse mode, in full-resolution pixels. Well
# under the width of a nucleus. Perfect mode does not simplify at all -- it emits the
# exact vertices the label mask was rasterized from, and the files stay small anyway.
SIMPLIFY_TOL_PX = 0.6

Image.MAX_IMAGE_PIXELS = None


def _log(msg):
    print("[synth-rgb] " + msg, flush=True)


def _argb(r, g, b):
    """Pack an opaque RGB into QuPath's signed 32-bit colorRGB."""
    v = (255 << 24) | (int(r) << 16) | (int(g) << 8) | int(b)
    return v - (1 << 32) if v >= (1 << 31) else v


CLASS_INFO = {
    LBL_STROMA: {"name": "Stroma", "rgb": (0, 200, 0)},
    LBL_TUMOR: {"name": "Tumor", "rgb": (200, 0, 0)},
    LBL_BACKGROUND: {"name": "Background", "rgb": (180, 180, 180)},
}


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


class Blob:
    """A star-shaped closed polygon: an ellipse whose radius wobbles with angle.

    Star-shaped matters. It makes "is this whole shape inside the tissue?" a cheap
    per-vertex radius comparison instead of a polygon clip, which is why this file
    needs no geometry library.
    """

    def __init__(self, cx, cy, rx, ry, phases, amps):
        self.cx = float(cx)
        self.cy = float(cy)
        self.rx = float(rx)
        self.ry = float(ry)
        # Plain Python floats, so an f32 input array stays f32 through modulation()
        # instead of being promoted to f64 by a numpy scalar (NEP 50).
        self.amps = [float(a) for a in amps]
        self.phases = [float(p) for p in phases]
        self.harmonics = [float(k) for k in range(2, 2 + len(self.amps))]

    def modulation(self, theta):
        """Radius multiplier at angle theta (array or scalar), in normalized space."""
        th = np.asarray(theta)
        if th.dtype.kind != "f":
            th = th.astype(np.float64)
        m = np.ones_like(th)
        for k, a, p in zip(self.harmonics, self.amps, self.phases):
            m = m + a * np.sin(k * th + p)
        return m

    def max_modulation(self):
        return 1.0 + float(np.sum(np.abs(self.amps)))

    def max_radius_px(self):
        return self.max_modulation() * max(self.rx, self.ry)

    def bbox(self, pad=0.0):
        """Axis-aligned bounds, padded. Used to keep field work off the whole band."""
        mx = self.max_modulation()
        return (
            self.cx - self.rx * mx - pad,
            self.cy - self.ry * mx - pad,
            self.cx + self.rx * mx + pad,
            self.cy + self.ry * mx + pad,
        )

    def signed_distance(self, x, y):
        """Approximate distance to the boundary in pixels, positive inside.

        Exact along the axes and close enough elsewhere for a soft transition; it is
        never used to decide a label, only how fast the texture changes.
        """
        u = (x - np.float32(self.cx)) / np.float32(self.rx)
        v = (y - np.float32(self.cy)) / np.float32(self.ry)
        th = np.arctan2(v, u)
        r = np.hypot(u, v)
        return (self.modulation(th) - r) * np.float32(min(self.rx, self.ry))

    def radius_toward(self, x, y):
        """Distance from the centre to the boundary, along the ray to (x, y)."""
        u = (x - self.cx) / self.rx
        v = (y - self.cy) / self.ry
        th = math.atan2(v, u)
        m = float(self.modulation(np.float64(th)))
        return math.hypot(self.rx * m * math.cos(th), self.ry * m * math.sin(th))

    def vertices(self, n):
        th = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False)
        m = self.modulation(th)
        x = self.cx + self.rx * m * np.cos(th)
        y = self.cy + self.ry * m * np.sin(th)
        return np.stack([x, y], axis=1)

    def contains(self, pts, margin_px=0.0):
        """Vectorized inside test with an inward margin, for a star-shaped blob."""
        u = (pts[:, 0] - self.cx) / self.rx
        v = (pts[:, 1] - self.cy) / self.ry
        th = np.arctan2(v, u)
        r = np.hypot(u, v)
        # A normalized margin of m costs at least m * min(rx, ry) real pixels, so
        # dividing by the smaller semi-axis is the conservative direction.
        marg = float(margin_px) / min(self.rx, self.ry)
        return r < (self.modulation(th) - marg)


def _make_blob(rng, cx, cy, rx, ry, wobble, n_harm=4):
    amps = wobble * rng.uniform(0.6, 1.4, size=n_harm) / np.arange(2, 2 + n_harm)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=n_harm)
    return Blob(cx, cy, rx, ry, phases, amps)


def build_tissue(rng, width, height):
    """One large irregular tissue footprint, guaranteed to stay inside the image."""
    return _make_blob(
        rng,
        cx=width * 0.5 + rng.uniform(-0.01, 0.01) * width,
        cy=height * 0.5 + rng.uniform(-0.01, 0.01) * height,
        rx=width * 0.472,
        ry=height * 0.472,
        wobble=0.11,
        n_harm=5,
    )


def _outline_gap(a, b, n=96):
    """Smallest distance between two blob outlines, in pixels.

    Vertex-to-vertex, so it slightly overestimates the true polygon distance and the
    clearances it enforces come out conservative. Returns 0.0 when one blob swallows
    the other, which vertex distance alone would not notice.
    """
    if (
        a.contains(np.array([[b.cx, b.cy]]))[0]
        or b.contains(np.array([[a.cx, a.cy]]))[0]
    ):
        return 0.0
    pa = a.vertices(n)
    pb = b.vertices(n)
    d = np.hypot(pa[:, 0][:, None] - pb[None, :, 0], pa[:, 1][:, None] - pb[None, :, 1])
    return float(d.min())


def _is_clear(cand, placed, min_gap):
    """True when cand keeps at least min_gap from every blob already placed."""
    reach = cand.max_radius_px()
    for other in placed:
        if (
            math.hypot(cand.cx - other.cx, cand.cy - other.cy)
            > reach + other.max_radius_px() + min_gap
        ):
            continue
        if _outline_gap(cand, other) < min_gap:
            return False
    return True


def build_giants(rng, tissue, width, height, scale, n_giant):
    """Nests several 256 px tiles across. Placed first, because they need the room.

    Hard mode only. Their interiors are later flattened almost featureless, so a tile
    landing well inside one has very little to go on.
    """
    giants = []
    for _ in range(n_giant):
        for _ in range(3000):
            diam = rng.uniform(1100.0, 1700.0) * scale
            aniso = rng.uniform(0.82, 1.22)
            blob = _make_blob(
                rng,
                rng.uniform(0.0, width),
                rng.uniform(0.0, height),
                0.5 * diam * aniso,
                0.5 * diam / aniso,
                wobble=0.22,
                n_harm=4,
            )
            if not tissue.contains(blob.vertices(128), margin_px=40.0 * scale).all():
                continue
            if not _is_clear(blob, giants, 150.0 * scale):
                continue
            giants.append(blob)
            break
    return giants


def build_tracts(rng, tissue, width, height, scale, existing, n_tracts):
    """Two rows of round nests flanking a thin stroma corridor.

    The corridor is genuine ground-truth Stroma, a couple of hundred pixels wide at
    most and a few thousand long. It is deliberately narrower than a 256 px tile
    stride: a tile that lands on one cannot see both sides of it, which is where a
    blended overlay and a centre-cropped one visibly disagree.
    """
    w_lo = TRACT_WIDTH_PX[0] * scale
    w_hi = TRACT_WIDTH_PX[1] * scale

    accepted = []
    widths = []
    for _ in range(n_tracts):
        for _ in range(600):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            dx, dy = math.cos(ang), math.sin(ang)
            nx, ny = -dy, dx
            length = rng.uniform(1500.0, 2600.0) * scale
            tract_w = rng.uniform(w_lo, w_hi)
            sx = rng.uniform(0.0, width)
            sy = rng.uniform(0.0, height)

            keep = []
            min_gap = min(45.0 * scale, tract_w * 0.45)
            for side in (1.0, -1.0):
                pos = 0.0
                prev_r = None
                while True:
                    rad = rng.uniform(100.0, 220.0) * scale
                    pos = rad if prev_r is None else pos + prev_r + rad * 1.15
                    if pos + rad > length:
                        break
                    prev_r = rad
                    proto = _make_blob(rng, 0.0, 0.0, rad, rad, wobble=0.22, n_harm=4)
                    # Circular blob, so the modulation at the angle pointing back at
                    # the centreline gives the exact radius on that side.
                    theta = math.atan2(-side * ny, -side * nx)
                    m = float(proto.modulation(np.float64(theta)))
                    off = tract_w * 0.5 + rad * m
                    blob = Blob(
                        sx + dx * pos + side * nx * off,
                        sy + dy * pos + side * ny * off,
                        rad,
                        rad,
                        proto.phases,
                        proto.amps,
                    )
                    if not tissue.contains(
                        blob.vertices(96), margin_px=30.0 * scale
                    ).all():
                        continue
                    if not _is_clear(blob, existing + accepted + keep, min_gap):
                        continue
                    keep.append(blob)

            if len(keep) >= 6:
                accepted.extend(keep)
                widths.append(tract_w)
                break

    return accepted, widths


def build_nests(rng, tissue, width, height, target_frac, scale, preplaced):
    """Clustered tumor nests, non-overlapping and wholly inside the tissue.

    target_frac is the share of the TISSUE area we aim to fill with tumor, counting
    whatever giants and tract nests were placed first.
    """
    target_area = target_frac * math.pi * tissue.rx * tissue.ry

    n_clusters = 7
    centers = []
    tries = 0
    while len(centers) < n_clusters and tries < 4000:
        tries += 1
        cx = rng.uniform(0.0, width)
        cy = rng.uniform(0.0, height)
        if tissue.contains(np.array([[cx, cy]]), margin_px=0.10 * min(width, height))[
            0
        ]:
            centers.append((cx, cy))
    if not centers:
        centers = [(width * 0.5, height * 0.5)]

    spread = 0.10 * min(width, height)

    nests = []
    area = sum(math.pi * b.rx * b.ry for b in preplaced)
    attempts = 0
    max_attempts = 20000
    while area < target_area and attempts < max_attempts:
        attempts += 1
        ccx, ccy = centers[rng.integers(0, len(centers))]
        cx = ccx + rng.normal(0.0, spread)
        cy = ccy + rng.normal(0.0, spread)

        # Across-size 220..1800 px at 8000x6000, then squashed one way or the other.
        # Squared draw: area fills up fast with big nests, so bias the draw small or
        # the slide ends up with a handful of giants and no small ones to test on.
        diam = (220.0 + 1580.0 * rng.uniform(0.0, 1.0) ** 2.0) * scale
        aniso = rng.uniform(0.72, 1.38)
        rx = 0.5 * diam * aniso
        ry = 0.5 * diam / aniso

        blob = _make_blob(rng, cx, cy, rx, ry, wobble=0.30, n_harm=4)

        if not tissue.contains(blob.vertices(96), margin_px=30.0 * scale).all():
            continue
        if not _is_clear(blob, preplaced + nests, 45.0 * scale):
            continue

        nests.append(blob)
        area += math.pi * rx * ry

    return nests


def build_perturbations(rng, lab, tissue, width, height, scale, cfg):
    """Picture-only distortions of the class boundary. None of these touch the mask.

    islands   tumor-like cell clusters sitting in ground-truth Stroma
    gaps      stroma-like clearings inside ground-truth Tumor
    ambiguous regions pushed to an intermediate density, whichever class they are
    """

    def place(n, want_label, r_lo, r_hi, wobble):
        out = []
        for _ in range(n):
            for _ in range(1500):
                rad = rng.uniform(r_lo, r_hi) * scale
                aniso = rng.uniform(0.7, 1.4)
                blob = _make_blob(
                    rng,
                    rng.uniform(rad, width - rad),
                    rng.uniform(rad, height - rad),
                    rad * aniso,
                    rad / aniso,
                    wobble=wobble,
                    n_harm=4,
                )
                if not tissue.contains(blob.vertices(64), margin_px=60.0 * scale).all():
                    continue
                if want_label is not None:
                    v = blob.vertices(48)
                    px = np.clip(v[:, 0].astype(np.int64), 0, width - 1)
                    py = np.clip(v[:, 1].astype(np.int64), 0, height - 1)
                    if not np.all(lab[py, px] == want_label):
                        continue
                out.append(blob)
                break
        return out

    islands = place(cfg["n_islands"], LBL_STROMA, 45.0, 130.0, 0.34)
    gaps = place(cfg["n_gaps"], LBL_TUMOR, 40.0, 110.0, 0.34)
    ambiguous = place(cfg["n_ambiguous"], None, 220.0, 460.0, 0.26)
    return islands, gaps, ambiguous


# --------------------------------------------------------------------------
# Rasterization
# --------------------------------------------------------------------------


def _ring_int(pts):
    return [(float(x), float(y)) for x, y in pts]


def rasterize_labels(tissue, nests, width, height, verts_tissue, verts_nest, draw_rim):
    """Label mask, plus the nest-boundary band when the difficulty still draws a rim.

    The mask is rasterized from the same polygons that become the annotations, at
    every difficulty. Nothing in the difficulty knob reaches this function except
    whether a rim is drawn, and the rim is picture only.
    """
    lab = Image.new("L", (width, height), LBL_BACKGROUND)
    d = ImageDraw.Draw(lab)
    d.polygon(_ring_int(tissue.vertices(verts_tissue)), fill=LBL_STROMA)

    rim = Image.new("L", (width, height), 0) if draw_rim else None
    dr = ImageDraw.Draw(rim) if draw_rim else None
    rim_w = max(3, int(round(7.0 * min(width, height) / 6000.0)))
    for nest in nests:
        ring = _ring_int(nest.vertices(verts_nest))
        d.polygon(ring, fill=LBL_TUMOR)
        if dr is not None:
            dr.line(ring + [ring[0]], fill=1, width=rim_w, joint="curve")

    rim_arr = np.asarray(rim, dtype=np.uint8) if draw_rim else None
    return np.asarray(lab, dtype=np.uint8), rim_arr


# --------------------------------------------------------------------------
# Texture synthesis
# --------------------------------------------------------------------------


def _hash32(ix, iy, salt):
    """Stable per-cell 32-bit hash. Deterministic, order-independent, no tables."""
    h = (
        (ix.astype(np.uint32) * np.uint32(73856093))
        ^ (iy.astype(np.uint32) * np.uint32(19349663))
        ^ np.uint32(salt & 0xFFFFFFFF)
    )
    h ^= h >> np.uint32(16)
    h *= np.uint32(2246822519)
    h ^= h >> np.uint32(13)
    h *= np.uint32(3266489917)
    h ^= h >> np.uint32(16)
    return h


def _byte01(h, shift):
    """One byte of a hash as a float32 in [0, 1)."""
    return ((h >> np.uint32(shift)) & np.uint32(0xFF)).astype(np.float32) / np.float32(
        256.0
    )


def _nuclei_field(x, y, cell, r_lo, r_hi, aniso, density, salt, edge=1.6, fade=None):
    """Jittered-grid nuclei, evaluated pointwise. Returns coverage in [0, 1].

    One nucleus per grid cell, present with probability `density`, centre jittered
    inside the cell. Each pixel checks its own cell and the eight neighbours, so a
    nucleus may be up to one cell wide before it starts getting clipped.

    `aniso` > 1 elongates the nucleus and gives it a random orientation, which is
    what separates spindly stromal fibroblasts from round tumor nuclei.

    `density` may be a per-pixel array, which is how a nest thins out towards its
    margin; pass `fade` as well in that case (see below).
    """
    inv = np.float32(1.0 / cell)
    gx = np.floor(x * inv).astype(np.int32)
    gy = np.floor(y * inv).astype(np.int32)
    acc = np.zeros(x.shape, dtype=np.float32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ix = gx + np.int32(dx)
            iy = gy + np.int32(dy)
            ha = _hash32(ix, iy, salt)
            hb = _hash32(ix, iy, salt + 0x9E3779B9)

            jx = _byte01(ha, 0)
            jy = _byte01(ha, 8)
            rr = _byte01(ha, 16)
            pr = _byte01(ha, 24)

            cx = (ix.astype(np.float32) + jx) * np.float32(cell)
            cy = (iy.astype(np.float32) + jy) * np.float32(cell)
            ddx = x - cx
            ddy = y - cy

            rad = np.float32(r_lo) + np.float32(r_hi - r_lo) * rr
            if aniso > 1.0:
                # A random unit vector from two hash bytes -- cheaper than a sin/cos
                # pair, and there are nine of these per pixel.
                ux = _byte01(hb, 0) * np.float32(2.0) - np.float32(1.0)
                uy = _byte01(hb, 8) * np.float32(2.0) - np.float32(1.0)
                nrm = np.sqrt(ux * ux + uy * uy) + np.float32(1e-6)
                ux /= nrm
                uy /= nrm
                pa = ddx * ux + ddy * uy
                pb = ddy * ux - ddx * uy
                sa = rad * np.float32(aniso)
                sb = rad / np.float32(aniso)
                d2 = (pa / sa) ** 2 + (pb / sb) ** 2
            else:
                d2 = (ddx * ddx + ddy * ddy) / (rad * rad)

            val = np.float32(edge) * (np.float32(1.0) - d2)
            np.clip(val, 0.0, 1.0, out=val)
            if fade is None:
                val *= (pr < density).astype(np.float32)
            else:
                # With a spatially varying density a hard presence test slices nuclei
                # in half wherever the threshold contour crosses one. Fading them in
                # over a narrow band of the threshold instead reads as weaker staining
                # at the nest margin, which is what a real nest does anyway.
                val *= np.clip((density - pr) * np.float32(fade), 0.0, 1.0)
            np.maximum(acc, val, out=acc)

    return acc


def _vnoise(x, y, cell, salt):
    """Two-octave value noise in [-1, 1] on a hash lattice. Smooth and seamless.

    Used to make the apparent class boundary wander off the true one in a spatially
    correlated way -- the kind of error a model makes, rather than per-pixel grain.
    """
    out = np.zeros(x.shape, dtype=np.float32)
    amp = np.float32(1.0)
    norm = np.float32(0.0)
    for octave in range(2):
        c = np.float32(cell / (2.7**octave))
        fx = x / c
        fy = y / c
        ix = np.floor(fx).astype(np.int32)
        iy = np.floor(fy).astype(np.int32)
        tx = fx - ix.astype(np.float32)
        ty = fy - iy.astype(np.float32)
        tx = tx * tx * (np.float32(3.0) - np.float32(2.0) * tx)
        ty = ty * ty * (np.float32(3.0) - np.float32(2.0) * ty)
        s = salt + octave * 0x7F4A7C15
        v00 = _byte01(_hash32(ix, iy, s), 0)
        v10 = _byte01(_hash32(ix + np.int32(1), iy, s), 0)
        v01 = _byte01(_hash32(ix, iy + np.int32(1), s), 0)
        v11 = _byte01(_hash32(ix + np.int32(1), iy + np.int32(1), s), 0)
        a = v00 + (v10 - v00) * tx
        b = v01 + (v11 - v01) * tx
        out += amp * (a + (b - a) * ty)
        norm += amp
        amp *= np.float32(0.5)
    return out / norm * np.float32(2.0) - np.float32(1.0)


def _smoothstep01(u):
    """Clamped smoothstep of u, which is already centred on 0 and scaled to +/-1."""
    t = np.clip(np.float32(0.5) + np.float32(0.5) * u, 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def _soft_indicator(blob, xg, yg, soft_px, noise_amp, noise_cell, salt):
    """Blob membership in [0, 1] with a transition soft_px wide, optionally wandering."""
    s = blob.signed_distance(xg, yg)
    if noise_amp > 0.0:
        s = s + np.float32(noise_amp) * _vnoise(xg, yg, noise_cell, salt)
    return _smoothstep01(s / np.float32(max(soft_px, 1e-3)) * np.float32(2.0))


def _blob_slice(blob, y0, y1, width, pad):
    """Row/column window of a band that a padded blob can reach, or None."""
    bx0, by0, bx1, by1 = blob.bbox(pad)
    r0 = max(y0, int(math.floor(by0)))
    r1 = min(y1, int(math.ceil(by1)) + 1)
    c0 = max(0, int(math.floor(bx0)))
    c1 = min(width, int(math.ceil(bx1)) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def class_fields(lab_band, y0, y1, width, geom, cfg, scale):
    """Local tumorness and local flatness for one row band.

    tumorness drives the texture blend; flatness compresses the contrast between the
    two textures. Both are PICTURE fields -- the label mask has already been decided
    and neither of these is allowed anywhere near it.
    """
    rows = y1 - y0
    tf = np.zeros((rows, width), dtype=np.float32)
    flat = np.zeros((rows, width), dtype=np.float32)
    soft = float(cfg["soft_px"]) * scale

    if soft <= 0.0:
        # Easy: the picture boundary is the mask boundary, no transition at all.
        tf[:] = (lab_band == LBL_TUMOR).astype(np.float32)
        return tf, flat

    amp = float(cfg["edge_noise_px"]) * scale
    ncell = float(cfg["edge_noise_cell"]) * scale
    pad = soft * 1.5 + amp * 1.5 + 4.0

    def window(blob):
        sl = _blob_slice(blob, y0, y1, width, pad)
        if sl is None:
            return None
        r0, r1, c0, c1 = sl
        xg = np.arange(c0, c1, dtype=np.float32)[None, :]
        yg = np.arange(r0, r1, dtype=np.float32)[:, None]
        return (
            sl,
            np.broadcast_to(xg, (r1 - r0, c1 - c0)).copy(),
            np.broadcast_to(yg, (r1 - r0, c1 - c0)).copy(),
        )

    for blob in geom["nests"]:
        w = window(blob)
        if w is None:
            continue
        (r0, r1, c0, c1), xg, yg = w
        ind = _soft_indicator(blob, xg, yg, soft, amp, ncell, 0x1B873593)
        view = tf[r0 - y0 : r1 - y0, c0:c1]
        np.maximum(view, ind, out=view)

    # Giants: normal contrast at the margin, near-featureless deep inside.
    core = float(cfg["flat_core"])
    if core > 0.0:
        for blob in geom["giants"]:
            w = window(blob)
            if w is None:
                continue
            (r0, r1, c0, c1), xg, yg = w
            s = blob.signed_distance(xg, yg)
            depth = _smoothstep01(
                (s - np.float32(260.0 * scale)) / np.float32(200.0 * scale)
            )
            view = flat[r0 - y0 : r1 - y0, c0:c1]
            np.maximum(view, depth * np.float32(core), out=view)

    # Tumor-like cell clusters loose in the stroma.
    for blob in geom["islands"]:
        w = window(blob)
        if w is None:
            continue
        (r0, r1, c0, c1), xg, yg = w
        ind = _soft_indicator(blob, xg, yg, soft * 0.55, amp * 0.4, ncell, 0x2545F491)
        view = tf[r0 - y0 : r1 - y0, c0:c1]
        np.maximum(view, ind * np.float32(0.92), out=view)

    # Stroma-like clearings inside nests.
    for blob in geom["gaps"]:
        w = window(blob)
        if w is None:
            continue
        (r0, r1, c0, c1), xg, yg = w
        ind = _soft_indicator(blob, xg, yg, soft * 0.55, amp * 0.4, ncell, 0x9E3779B1)
        view = tf[r0 - y0 : r1 - y0, c0:c1]
        np.minimum(view, np.float32(1.0) - ind * np.float32(0.9), out=view)

    # Regions dragged to an intermediate density, whichever class they really are.
    for blob in geom["ambiguous"]:
        w = window(blob)
        if w is None:
            continue
        (r0, r1, c0, c1), xg, yg = w
        wgt = _soft_indicator(blob, xg, yg, soft * 2.2, amp, ncell, 0x85EBCA77)
        tview = tf[r0 - y0 : r1 - y0, c0:c1]
        tview *= np.float32(1.0) - wgt
        tview += np.float32(0.5) * wgt
        fview = flat[r0 - y0 : r1 - y0, c0:c1]
        np.maximum(fview, wgt * np.float32(0.7), out=fview)

    return tf, flat


def _stain_gradient(x, y, width, height, phases):
    """Slow multiplicative stain density across the slide. Mild, and class-blind."""
    fx = x / np.float32(width)
    fy = y / np.float32(height)
    g = np.float32(1.0)
    g = g + np.float32(0.11) * np.sin(
        np.float32(2.0 * math.pi) * fx + np.float32(phases[0])
    )
    g = g + np.float32(0.08) * np.sin(
        np.float32(2.0 * math.pi * 0.7) * fy + np.float32(phases[1])
    )
    g = g + np.float32(0.05) * np.sin(
        np.float32(2.0 * math.pi * 0.55) * (fx + fy) + np.float32(phases[2])
    )
    return g


def _render_glass(band, idx, x, y, width, height, phases, rng):
    """Slide glass: near-white, gently vignetted, carrying the same slide shading."""
    vx = (x / np.float32(width)) - np.float32(0.5)
    vy = (y / np.float32(height)) - np.float32(0.5)
    shade = np.float32(1.0) - np.float32(0.05) * (vx * vx + vy * vy) * np.float32(4.0)
    g = np.float32(1.0) + np.float32(0.25) * (
        _stain_gradient(x, y, width, height, phases) - np.float32(1.0)
    )
    for c in range(3):
        col = np.float32(GLASS_RGB[c]) * shade
        col = np.float32(255.0) - (np.float32(255.0) - col) * g
        col += rng.normal(0.0, 2.2, size=col.shape).astype(np.float32)
        np.clip(col, 0.0, 255.0, out=col)
        band[idx, c] = col.astype(np.uint8)


def _collagen(x, y, scale):
    """Two crossing wave trains whose phase is itself slowly warped, so the fibre
    bundles undulate instead of reading as a plaid."""
    warp_a = np.float32(1.3) * np.sin(
        (x * np.float32(0.15) - y * np.float32(0.21)) / (np.float32(95.0) * scale)
    )
    warp_b = np.float32(1.5) * np.sin(
        (x * np.float32(0.19) + y * np.float32(0.11)) / (np.float32(72.0) * scale)
        + np.float32(1.7)
    )
    return np.float32(0.62) * np.sin(
        (x * np.float32(0.9) + y * np.float32(0.42)) / (np.float32(7.5) * scale)
        + warp_a
    ) + np.float32(0.38) * np.sin(
        (x * np.float32(-0.35) + y * np.float32(0.94)) / (np.float32(12.0) * scale)
        + warp_b
    )


def synthesize_image(lab, rim, width, height, seed, phases, geom, cfg):
    """Paint the RGB slide one row band at a time.

    Tissue is rendered as a continuous blend between the two textures, driven by the
    local tumorness field rather than by the label. At easy that field is the label,
    so the blend degenerates to the old hard switch; at medium and hard it crosses the
    boundary gradually and wanders off it, which is the whole point of the knob.
    """
    out = np.empty((height, width, 3), dtype=np.uint8)
    scale = np.float32(min(width, height) / 6000.0)
    fade = None if cfg["soft_px"] <= 0.0 else 6.0
    t0 = time.time()

    for y0 in range(0, height, BAND):
        y1 = min(height, y0 + BAND)
        rng = np.random.default_rng(seed * 1000003 + y0)

        lab_band = lab[y0:y1]
        m = lab_band.ravel()
        xs = np.tile(np.arange(width, dtype=np.float32), y1 - y0)
        ys = np.repeat(np.arange(y0, y1, dtype=np.float32), width)
        band = out[y0:y1].reshape(-1, 3)

        tf, flat = class_fields(lab_band, y0, y1, width, geom, cfg, float(scale))
        tf = tf.ravel()
        flat = flat.ravel()

        idx = np.flatnonzero(m == LBL_BACKGROUND)
        if idx.size:
            _render_glass(band, idx, xs[idx], ys[idx], width, height, phases, rng)

        idx = np.flatnonzero(m != LBL_BACKGROUND)
        if not idx.size:
            continue
        x = xs[idx]
        y = ys[idx]
        t = tf[idx]
        fl = flat[idx]
        n = idx.size

        # Stromal nuclei fade out as tumorness rises, and vice versa. Skipping the
        # pixels where one texture contributes nothing keeps the blended render
        # roughly as cheap as the two disjoint passes it replaced.
        cov_s = np.zeros(n, dtype=np.float32)
        sel = np.flatnonzero(t < 0.985)
        if sel.size:
            cov_s[sel] = (
                _nuclei_field(
                    x[sel],
                    y[sel],
                    cell=float(38.0 * scale),
                    r_lo=float(3.2 * scale),
                    r_hi=float(4.4 * scale),
                    aniso=2.6,
                    density=STROMA_DENSITY,
                    salt=0x51ED270B,
                    edge=1.5,
                )
                * np.float32(0.88)
                * (np.float32(1.0) - t[sel])
            )

        cov_t = np.zeros(n, dtype=np.float32)
        sel = np.flatnonzero(t > 0.015)
        if sel.size:
            dens = (
                np.float32(TUMOR_DENSITY)
                * t[sel]
                * (np.float32(1.0) - np.float32(0.35) * fl[sel])
            )
            cov_t[sel] = _nuclei_field(
                x[sel],
                y[sel],
                cell=float(9.5 * scale),
                r_lo=float(3.1 * scale),
                r_hi=float(4.5 * scale),
                aniso=1.0,
                density=dens if fade is not None else np.float32(TUMOR_DENSITY),
                salt=0x2F6E12C5,
                edge=1.9,
                fade=fade,
            ) * np.float32(0.95)

        cov = np.clip(cov_s + cov_t, 0.0, 1.0)
        # Which kind of nucleus actually covers this pixel, not which class it is.
        nuc_w = cov_t / np.maximum(cov_s + cov_t, np.float32(1e-6))
        fib = _collagen(x, y, scale) * (np.float32(1.0) - t)
        g = _stain_gradient(x, y, width, height, phases)
        sd = np.float32(3.4) + (np.float32(3.0) - np.float32(3.4)) * t

        for c in range(3):
            base = (
                np.float32(STROMA_RGB[c]) + np.float32(TUMOR_RGB[c] - STROMA_RGB[c]) * t
            )
            nucc = (
                np.float32(STROMA_NUC_RGB[c])
                + np.float32(TUMOR_NUC_RGB[c] - STROMA_NUC_RGB[c]) * nuc_w
            )
            # Low internal contrast: pull the nuclei towards the cytoplasm they sit in.
            nucc += (base - nucc) * (np.float32(0.55) * fl)
            col = base * (np.float32(1.0) - cov) + nucc * cov
            col += fib * np.float32(6.0)
            # Stain gradient acts on optical density, not brightness, so white
            # stays white.
            col = np.float32(255.0) - (np.float32(255.0) - col) * g
            col += rng.normal(0.0, sd).astype(np.float32)
            np.clip(col, 0.0, 255.0, out=col)
            band[idx, c] = col.astype(np.uint8)

        # A crisp dark rim is the single strongest boundary cue, so only easy keeps it.
        if rim is not None:
            r = rim[y0:y1].ravel()
            edge = np.flatnonzero((m == LBL_STROMA) & (r > 0))
            if edge.size:
                band[edge] = (band[edge].astype(np.float32) * np.float32(0.90)).astype(
                    np.uint8
                )
            edge = np.flatnonzero((m == LBL_TUMOR) & (r > 0))
            if edge.size:
                band[edge] = (band[edge].astype(np.float32) * np.float32(0.82)).astype(
                    np.uint8
                )

    _log("texture pass took %.1f s" % (time.time() - t0))
    return out


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------


def _rdp(pts, tol):
    """Douglas-Peucker, iterative. pts is an open ring; the ends are always kept."""
    n = len(pts)
    if n < 4 or tol <= 0.0:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = pts[i]
        bx, by = pts[j]
        seg = pts[i + 1 : j]
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9:
            dist = np.hypot(seg[:, 0] - ax, seg[:, 1] - ay)
        else:
            dist = np.abs(dy * (seg[:, 0] - ax) - dx * (seg[:, 1] - ay)) / length
        k = int(np.argmax(dist))
        if dist[k] > tol:
            mid = i + 1 + k
            keep[mid] = True
            stack.append((i, mid))
            stack.append((mid, j))
    return pts[keep]


def _ring(pts, width, height, tol=SIMPLIFY_TOL_PX):
    """Closed, simplified, in-bounds, rounded ring as a list of [x, y] pairs."""
    pts = np.asarray(pts, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    simp = _rdp(closed, tol)
    if len(simp) < 4:
        simp = closed
    simp[:, 0] = np.clip(simp[:, 0], 0.0, width - 1.0)
    simp[:, 1] = np.clip(simp[:, 1], 0.0, height - 1.0)
    out = [[round(float(px), 2), round(float(py), 2)] for px, py in simp]
    out[-1] = list(out[0])
    return out


def _feature(rings, label):
    info = CLASS_INFO[label]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": rings},
        "properties": {
            "objectType": "annotation",
            "classification": {"name": info["name"], "colorRGB": _argb(*info["rgb"])},
        },
    }


def annotations_perfect(tissue, nests, width, height, verts_tissue, verts_nest):
    """Exact annotations: the very polygons the label mask was rasterized from.

    Interior rings carry the containment, so the three classes tile the image with
    no overlap: Background is the image rectangle with the tissue punched out, and
    Stroma is the tissue with the nests punched out.
    """
    tissue_ring = _ring(tissue.vertices(verts_tissue), width, height, tol=0.0)
    nest_rings = [_ring(n.vertices(verts_nest), width, height, tol=0.0) for n in nests]

    frame = [
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0],
        [0.0, 0.0],
    ]
    feats = [
        _feature([frame, list(reversed(tissue_ring))], LBL_BACKGROUND),
        _feature([tissue_ring] + [list(reversed(r)) for r in nest_rings], LBL_STROMA),
    ]
    feats.extend(_feature([r], LBL_TUMOR) for n, r in zip(nests, nest_rings))
    return feats


def annotations_sparse(lab, width, height, rng):
    """What a user actually brushes: ~15 patches over a few percent of the tissue.

    Each patch must sit wholly inside one class, checked against the rasterized
    mask rather than against the geometry, so a patch can never straddle a border.
    """
    scale = min(width, height) / 6000.0
    plan = (
        [(LBL_TUMOR, 115.0 * scale, 230.0 * scale)] * 6
        + [(LBL_STROMA, 230.0 * scale, 370.0 * scale)] * 7
        + [(LBL_BACKGROUND, 200.0 * scale, 320.0 * scale)] * 2
    )

    feats = []
    for label, r_lo, r_hi in plan:
        for _ in range(900):
            rad = rng.uniform(r_lo, r_hi)
            aniso = rng.uniform(0.75, 1.33)
            cx = rng.uniform(rad, width - rad)
            cy = rng.uniform(rad, height - rad)
            blob = _make_blob(
                rng, cx, cy, rad * aniso, rad / aniso, wobble=0.26, n_harm=4
            )

            verts = blob.vertices(72)
            # Vertices plus a shrunken copy: cheap proof the interior is pure too.
            centre = np.array([blob.cx, blob.cy])
            probe = np.vstack([verts, centre + 0.55 * (verts - centre)])
            px = np.clip(probe[:, 0].astype(np.int64), 0, width - 1)
            py = np.clip(probe[:, 1].astype(np.int64), 0, height - 1)
            if not np.all(lab[py, px] == label):
                continue
            feats.append((label, _ring(verts, width, height), blob))
            break

    return [_feature([ring], label) for label, ring, _ in feats]


def annotation_coverage(feats, lab, width, height):
    """Rasterize the emitted GeoJSON and score it against the label mask.

    Returns (covered_px_by_label, impure_px). This is the file's own audit: it reads
    the features back the way QuPath will, honouring interior rings, so "perfect"
    mode proves it reproduces the mask and "sparse" mode proves no patch straddles
    a class border.
    """
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    name2label = {info["name"]: label for label, info in CLASS_INFO.items()}
    for ft in feats:
        label = name2label[ft["properties"]["classification"]["name"]]
        rings = ft["geometry"]["coordinates"]
        draw.polygon([tuple(p) for p in rings[0]], fill=label)
        for hole in rings[1:]:
            draw.polygon([tuple(p) for p in hole], fill=255)

    drawn = np.asarray(canvas, dtype=np.uint8)
    painted = drawn != 255
    covered = {label: int(np.count_nonzero(drawn == label)) for label in CLASS_INFO}
    impure = int(np.count_nonzero(painted & (drawn != lab)))
    return covered, impure


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args(argv):
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Generate a synthetic H&E-like RGB slide with perfect ground truth."
    )
    p.add_argument("--out", default=str(here / "data" / "rgb"), help="output directory")
    p.add_argument(
        "--seed", type=int, default=1234, help="RNG seed (fully determines output)"
    )
    p.add_argument("--width", type=int, default=8000, help="image width in pixels")
    p.add_argument("--height", type=int, default=6000, help="image height in pixels")
    p.add_argument(
        "--annotation-mode",
        choices=("perfect", "sparse"),
        default="perfect",
        help="perfect = exact class boundaries; sparse = ~15 brushed training patches",
    )
    p.add_argument(
        "--pixel-size-um", type=float, default=0.5, help="pixel size in microns"
    )
    p.add_argument(
        "--difficulty",
        choices=tuple(DIFFICULTY_PRESETS),
        default="medium",
        help=(
            "how ambiguous the Stroma/Tumor boundary looks; affects the picture only, "
            "never the ground truth"
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    width = int(args.width)
    height = int(args.height)
    if width < 512 or height < 512:
        _log("ERROR: --width and --height must be at least 512")
        return 2
    if args.pixel_size_um <= 0.0:
        _log("ERROR: --pixel-size-um must be positive")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DIFFICULTY_PRESETS[args.difficulty]
    scale = min(width, height) / 6000.0

    # Three independent streams. Geometry is drawn first and separately so that the
    # ground truth does not move when only the difficulty changes: easy and medium
    # produce a byte-identical ground_truth.png for a given seed. Hard is the one
    # exception, because its giant nests are real tumor and do change the mask.
    geo_rng = np.random.default_rng(args.seed)
    tex_rng = np.random.default_rng(args.seed ^ 0x5EED1234)
    ann_rng = np.random.default_rng(args.seed ^ 0x5A5A5A5A)
    phases = tex_rng.uniform(0.0, 2.0 * math.pi, size=3)

    # Vertex counts scale with the image so the boundary stays smooth at any size.
    verts_tissue = 720
    verts_nest = 128

    _log(
        "seed=%d dims=%dx%d mode=%s difficulty=%s"
        % (args.seed, width, height, args.annotation_mode, args.difficulty)
    )

    tissue = build_tissue(geo_rng, width, height)
    giants = build_giants(geo_rng, tissue, width, height, scale, N_GIANT_NESTS)
    tract_nests, tract_widths = build_tracts(
        geo_rng, tissue, width, height, scale, giants, N_TRACTS
    )
    preplaced = giants + tract_nests
    nests = preplaced + build_nests(
        geo_rng, tissue, width, height, 0.19, scale, preplaced
    )
    _log(
        "placed %d tumor nests (%d giant, %d flanking %d stroma tracts %s px wide)"
        % (
            len(nests),
            len(giants),
            len(tract_nests),
            len(tract_widths),
            "/".join("%d" % round(w) for w in tract_widths) if tract_widths else "-",
        )
    )

    lab, rim = rasterize_labels(
        tissue, nests, width, height, verts_tissue, verts_nest, cfg["draw_rim"]
    )

    # Picture-only perturbations, chosen AFTER the mask exists so they can be told
    # which class they are sitting in without ever being able to change it.
    islands, gaps, ambiguous = build_perturbations(
        tex_rng, lab, tissue, width, height, scale, cfg
    )
    geom = {
        "nests": nests,
        "giants": giants,
        "islands": islands,
        "gaps": gaps,
        "ambiguous": ambiguous,
    }
    if islands or gaps or ambiguous:
        _log(
            "picture-only perturbations: %d tumor-like islands in stroma, "
            "%d stroma-like gaps in nests, %d intermediate regions"
            % (len(islands), len(gaps), len(ambiguous))
        )

    counts = np.bincount(lab.ravel(), minlength=3)
    total = float(width) * float(height)
    for label in (LBL_STROMA, LBL_TUMOR, LBL_BACKGROUND):
        _log(
            "class %d %-10s %12d px  %6.2f %%"
            % (
                label,
                CLASS_INFO[label]["name"],
                counts[label],
                100.0 * counts[label] / total,
            )
        )

    img = synthesize_image(lab, rim, width, height, args.seed, phases, geom, cfg)

    tif_path = out_dir / "image.tif"
    ppcm = 1.0e4 / float(args.pixel_size_um)
    tifffile.imwrite(
        tif_path,
        img,
        photometric="rgb",
        planarconfig="contig",
        tile=(512, 512),
        compression="deflate",
        resolution=(ppcm, ppcm),
        resolutionunit="CENTIMETER",
        metadata=None,
    )
    _log("wrote %s (%.1f MB)" % (tif_path.name, tif_path.stat().st_size / 1e6))

    gt_path = out_dir / "ground_truth.png"
    Image.fromarray(lab, mode="L").save(gt_path, optimize=True)
    _log("wrote %s (%.1f MB)" % (gt_path.name, gt_path.stat().st_size / 1e6))

    if args.annotation_mode == "perfect":
        feats = annotations_perfect(
            tissue, nests, width, height, verts_tissue, verts_nest
        )
    else:
        feats = annotations_sparse(lab, width, height, ann_rng)
    covered, impure = annotation_coverage(feats, lab, width, height)

    geo_path = out_dir / "annotations.geojson"
    with open(geo_path, "w", encoding="ascii") as f:
        json.dump(
            {"type": "FeatureCollection", "features": feats},
            f,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    n_verts = sum(len(r) for ft in feats for r in ft["geometry"]["coordinates"])
    _log(
        "wrote %s (%d annotations, %d vertices, %.2f MB)"
        % (geo_path.name, len(feats), n_verts, geo_path.stat().st_size / 1e6)
    )

    tissue_px = float(counts[LBL_STROMA] + counts[LBL_TUMOR])
    annotated_tissue = float(covered[LBL_STROMA] + covered[LBL_TUMOR])
    annotated_all = float(sum(covered.values()))
    px_area_um2 = float(args.pixel_size_um) ** 2
    _log(
        "annotations cover %d tissue px (%.2f %% of tissue), %d px in all; "
        "%d px disagree with the mask (%.4f %%)"
        % (
            int(annotated_tissue),
            100.0 * annotated_tissue / tissue_px if tissue_px else 0.0,
            int(annotated_all),
            impure,
            100.0 * impure / total,
        )
    )

    manifest = {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "modality": "brightfield_rgb",
        "seed": int(args.seed),
        "annotation_mode": args.annotation_mode,
        "difficulty": args.difficulty,
        "difficulty_settings": dict(cfg),
        "pixel_size_um": float(args.pixel_size_um),
        "image": {
            "file": "image.tif",
            "width": width,
            "height": height,
            "channels": 3,
            "dtype": "uint8",
            "photometric": "rgb",
        },
        "ground_truth": {
            "file": "ground_truth.png",
            "dtype": "uint8",
            "encoding": "label_index",
        },
        "annotations": {
            "file": "annotations.geojson",
            "count": len(feats),
            "annotated_classes": sorted(
                {CLASS_INFO[label]["name"] for label in covered if covered[label] > 0}
            ),
        },
        "classes": [
            {
                "label": int(label),
                "name": CLASS_INFO[label]["name"],
                "rgb": list(CLASS_INFO[label]["rgb"]),
                "color_rgb_qupath": _argb(*CLASS_INFO[label]["rgb"]),
                "is_tissue": label != LBL_BACKGROUND,
                "area_px": int(counts[label]),
                "area_fraction": round(float(counts[label]) / total, 6),
                "area_um2": round(float(counts[label]) * px_area_um2, 2),
                "annotated_area_px": int(round(covered.get(label, 0.0))),
            }
            for label in (LBL_STROMA, LBL_TUMOR, LBL_BACKGROUND)
        ],
        "tumor_nests": len(nests),
        "giant_nests": len(giants),
        "giant_nests_flattened": bool(cfg["flat_core"] > 0.0),
        "stroma_tracts": [int(round(w)) for w in tract_widths],
        "picture_only_perturbations": {
            "tumor_like_islands_in_stroma": len(islands),
            "stroma_like_gaps_in_nests": len(gaps),
            "intermediate_regions": len(ambiguous),
        },
        "tissue_area_px": int(tissue_px),
        "annotated_fraction_of_tissue": (
            round(annotated_tissue / tissue_px, 6) if tissue_px else 0.0
        ),
        "annotation_mask_disagreement_px": impure,
    }
    man_path = out_dir / "manifest.json"
    with open(man_path, "w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    # --- self-check: the three files must agree on geometry ---
    with tifffile.TiffFile(tif_path) as tf:
        page = tf.pages[0]
        tif_shape = (int(page.imagelength), int(page.imagewidth))
        tiled = page.tilewidth is not None and page.tilewidth > 0
    with Image.open(gt_path) as im:
        png_shape = (im.height, im.width)
        png_mode = im.mode
    # Sub-pixel slack: GeoJSON coordinates are rounded to two decimals, so a handful
    # of boundary pixels may land on the other side when the polygons are re-filled.
    slack = max(64, int(0.0005 * total))
    ok = (
        tif_shape == (height, width)
        and png_shape == (height, width)
        and tiled
        and impure <= slack
    )
    _log(
        "VERIFY tif=%dx%d png=%dx%d(mode %s) tiled=%s impure=%d/%d -> %s"
        % (
            tif_shape[1],
            tif_shape[0],
            png_shape[1],
            png_shape[0],
            png_mode,
            tiled,
            impure,
            slack,
            "OK" if ok else "MISMATCH",
        )
    )

    print("")
    print("Synthetic RGB tissue written to %s" % out_dir)
    print(
        "  image.tif            %dx%d RGB uint8, tiled, %.1f um/px"
        % (width, height, args.pixel_size_um)
    )
    print("  ground_truth.png     0=Stroma 1=Tumor 2=Background")
    print("  annotations.geojson  %s, %d features" % (args.annotation_mode, len(feats)))
    print(
        "  manifest.json        seed %d, difficulty %s, %d tumor nests"
        % (args.seed, args.difficulty, len(nests))
    )
    print(
        "  class fractions      Stroma %.2f %%  Tumor %.2f %%  Background %.2f %%"
        % (
            100.0 * counts[LBL_STROMA] / total,
            100.0 * counts[LBL_TUMOR] / total,
            100.0 * counts[LBL_BACKGROUND] / total,
        )
    )
    print(
        "  annotated tissue     %.2f %% of Stroma+Tumor area"
        % (100.0 * annotated_tissue / tissue_px)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
