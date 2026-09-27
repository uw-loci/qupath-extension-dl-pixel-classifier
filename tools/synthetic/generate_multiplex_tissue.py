#!/usr/bin/env python3
"""Generate a synthetic multiplex immunofluorescence image with perfect ground truth.

The output is modelled on the LuCa-7color component data (PerkinElmer Vectra, float32,
one plane per unmixed marker, pixel size about 0.498 um). It is meant to feed a release
regression test: train the DL pixel classifier on it, then score the objects it produces
against a label mask that is known to be exactly right.

What the image contains
-----------------------
Two tissue classes plus background, five marker channels:

    CD8   (Opal 540)  punctate immune cells, dense in stroma, rare in tumor
    FoxP3 (Opal 570)  punctate nuclear signal, dense in stroma
    CD68  (Opal 620)  larger diffuse macrophages, present in both classes
    PD1   (Opal 650)  punctate immune cells, dense in stroma
    CK    (Opal 690)  bright membranous signal, confined to tumor nests

On top of the cells each channel carries a low tissue autofluorescence haze whose
TEXTURE differs by class: streaky and fibrillar in stroma, fine and granular inside
tumor nests. That matters, because it means a patch of stroma that happens to contain
no immune cell is still distinguishable from a tumor nest and from background using
only a few hundred pixels of context. Nothing about the class layout requires
whole-image context -- the suite exists to catch tiling artifacts, so the ground truth
must not itself depend on global information.

Channels bleed into each other by a few percent (most between spectral neighbours), so
no single channel separates the classes on its own. A slow multiplicative illumination
gradient runs across the slide, because cross-tile normalization behaviour is one of
the things under test.

Difficulty
----------
--difficulty {easy,medium,hard}, default medium, sets how AMBIGUOUS the picture is and
nothing else. easy stops CK exactly where Tumor stops; medium fades it across a soft
margin, strands CK-positive cells in ground-truth Stroma and leaves CK-negative holes
inside ground-truth Tumor; hard widens all of that past a tile stride, holds whole nests
at an intermediate CK level, and drains the internal contrast from the giant nests.

For one seed, ground_truth.png and annotations.geojson are BYTE-IDENTICAL across all
three difficulties -- only image.tif moves. That is what makes a difference in results
attributable to ambiguity rather than to a different answer key, and it is why the
generator runs three separate RNG streams (geometry, texture, annotations) instead of
one: a difficulty knob draws a different COUNT of random numbers, so a single stream
would shift every later draw and quietly move the label mask.

Two structures exist at every difficulty, because they are geometry: three thin stroma
corridors 90-210 px wide (narrower than a 256 px tile stride, which is where a blended
overlay and a centre-cropped one part company), and two tumor regions several 256 px
tiles across, so a tile can sit inside one and see no boundary at all.

Outputs (under --out)
---------------------
    image.tif            float32 OME-TIFF, N channels, CYX, tiled, channel names set
    ground_truth.png     uint8 label mask, 0=Stroma 1=Tumor 2=Background
    annotations.geojson  QuPath-importable annotations (File > Import objects from file)
    manifest.json        seed, difficulty and its settings, dims, channel names/stats,
                         class areas, pixel size

The generator checks its own output before it exits: it rasterizes the GeoJSON back with
pixel-centre containment and reports the agreement with ground_truth.png, the worst
per-feature purity, and any vertex outside the image. In perfect mode at the default size
that agreement is 100.0000 pct with zero mismatched pixels -- if a future change breaks
that, the summary says so rather than shipping annotations that quietly disagree with the
mask the suite scores against.

Usage
-----
    python_server/venv/bin/python tools/synthetic/generate_multiplex_tissue.py
    python_server/venv/bin/python tools/synthetic/generate_multiplex_tissue.py \
        --out /tmp/synth --seed 7 --width 2048 --height 1536 \
        --annotation-mode sparse --difficulty hard

Deterministic for a given --seed, --width, --height, --channels and --difficulty.
numpy / scipy / tifffile / Pillow only. ASCII-only output (Windows cp1252).
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

GENERATOR = "generate_multiplex_tissue"
VERSION = "1.0.0"

# --- classes -------------------------------------------------------------------
# Label values written into ground_truth.png. Do not renumber: the release test and
# the sibling RGB generator share this contract.
STROMA, TUMOR, BACKGROUND = 0, 1, 2
CLASS_NAMES = {STROMA: "Stroma", TUMOR: "Tumor", BACKGROUND: "Background"}
CLASS_RGB = {
    STROMA: (0, 170, 60),
    TUMOR: (200, 0, 0),
    BACKGROUND: (110, 110, 110),
}

# --- markers -------------------------------------------------------------------
# One entry per channel, in emission-wavelength order (bleed-through uses that order).
# "kind" picks the cell generator and "density" is one cell per N pixels of that class.
#
# Amplitudes live in a relative frame where the median cell peak is 1.0, so "af" (the
# autofluorescence haze weight) and "amp_sigma" (the lognormal spread of cell
# brightness) between them fix the SHAPE of the intensity histogram. Only the absolute
# scale is fitted afterwards, by pushing each channel's p99 onto target_p99 -- which is
# why those two knobs, not target_p99, are what to touch if max/p99 drifts.
#
# target_p99 mirrors the real LuCa component data: sparse markers p99 ~2.5-3.5 with a
# tail to roughly 10x that, cytokeratin p99 ~18-25 with a shorter tail.
ROLES = [
    dict(
        name="CD8 (Opal 540)",
        kind="punctate",
        target_p99=3.4,
        per_stroma=1400,
        per_tumor=16000,
        sigma=1.75,
        af=0.075,
        amp_sigma=0.45,
    ),
    dict(
        name="FoxP3 (Opal 570)",
        kind="punctate",
        target_p99=2.6,
        per_stroma=1500,
        per_tumor=9000,
        sigma=1.45,
        af=0.060,
        amp_sigma=0.42,
    ),
    dict(
        name="CD68 (Opal 620)",
        kind="blob",
        target_p99=2.9,
        per_stroma=2600,
        per_tumor=5200,
        sigma=4.6,
        af=0.085,
        amp_sigma=0.50,
    ),
    dict(
        name="PD1 (Opal 650)",
        kind="punctate",
        target_p99=2.4,
        per_stroma=1800,
        per_tumor=7000,
        sigma=1.60,
        af=0.070,
        amp_sigma=0.45,
    ),
    dict(
        name="CK (Opal 690)",
        kind="membrane",
        target_p99=18.0,
        per_stroma=0,
        per_tumor=0,
        sigma=0.0,
        af=0.045,
        amp_sigma=0.35,
    ),
]
CK_ROLE_INDEX = 4

# Spectral bleed-through, as a fraction of the donor channel. Neighbouring Opal dyes
# overlap the most; distant ones leak a little through the unmixing residual.
BLEED_BY_DISTANCE = {1: 0.045, 2: 0.012}
BLEED_FAR = 0.004

# Detector noise, as a fraction of the channel's target p99, and a shot-noise term
# that grows with the square root of the signal.
READ_NOISE_FRAC = 0.014
SHOT_NOISE_K = 0.22

# --- difficulty ----------------------------------------------------------------
# How AMBIGUOUS the picture is, and nothing else. Every knob below perturbs the
# PIXELS; none of them reaches the label mask or the annotations, so for one seed
# ground_truth.png and annotations.geojson are byte-identical across all three
# difficulties and only image.tif moves. That is the whole point: a difference in
# results is then attributable to ambiguity and not to a different answer key.
#
# Why harder settings are worth having at all: a tiling artifact only shows up where
# the model is genuinely unsure, because that is where two tiles seeing different
# context resolve the same pixel differently. On data the model finds easy, a blend
# regression passes the suite whether or not the bug is there.
#
#   transition_px   width of the CK falloff across a nest boundary, in pixels
#   stranded_cells  CK-positive cells in ground-truth Stroma, as a fraction of the
#                   tumor cell count
#   gap_frac        fraction of tumor area left CK-negative
#   dim_nest_frac   fraction of tumor area held at intermediate CK intensity
#   flat_core       how far the giant nests lose their internal contrast (0 to 1)
#   texture_contrast   1.0 keeps the stroma/tumor autofluorescence textures distinct,
#                   lower blends them toward each other
#   immune_contrast 1.0 keeps the immune density step at the nest edge, lower moves
#                   the in-nest density toward the stromal density
DIFFICULTY_PRESETS = {
    "easy": {
        # The mask boundary IS the picture boundary: CK stops where Tumor stops.
        "transition_px": 0.0,
        "stranded_cells": 0.0,
        "gap_frac": 0.0,
        "dim_nest_frac": 0.0,
        "flat_core": 0.0,
        "texture_contrast": 1.0,
        "immune_contrast": 1.0,
    },
    "medium": {
        # A soft margin tens of pixels wide, CK-positive cells loose in the stroma,
        # and CK-negative holes inside nests. These are wider than they first look
        # like they need to be: at transition_px 40 a 32x32 window almost never lands
        # wholly inside the ambiguous band, and medium scored the same as easy.
        "transition_px": 70.0,
        "stranded_cells": 0.13,
        "gap_frac": 0.14,
        "dim_nest_frac": 0.0,
        "flat_core": 0.40,
        "texture_contrast": 0.72,
        "immune_contrast": 0.78,
    },
    "hard": {
        # Medium, widened past a tile stride, plus whole nests at an intermediate CK
        # level and giant nests whose interiors are nearly featureless.
        "transition_px": 140.0,
        "stranded_cells": 0.24,
        "gap_frac": 0.22,
        "dim_nest_frac": 0.28,
        "flat_core": 0.80,
        "texture_contrast": 0.48,
        "immune_contrast": 0.52,
    },
}

# Thin stroma corridors running between tumor regions, present at EVERY difficulty
# because they are geometry. Deliberately narrower than a 256 px tile stride: a
# structure the tile grid cannot resolve is where a blended overlay and a
# centre-cropped one part company. These lengths are ABSOLUTE pixels and do not
# scale with the image, because the tile stride they are measured against does not
# scale either.
N_TRACTS = 3
TRACT_WIDTH_PX = (90.0, 210.0)

# A couple of tumor regions several 256 px tiles across, so a tile in the middle of
# one sees no boundary at all. They exist at every difficulty -- only `flat_core`
# changes what is inside them -- so the ground truth still depends on the seed alone.
N_GIANT_NESTS = 2
GIANT_NEST_RADIUS_PX = (430.0, 560.0)

# tiling_metrics.py measures one square of the image rather than all of it (its
# --max-size, default 2048, taken from the CENTRE by default), so the FIRST giant nest
# is required to fit inside that square. Without the constraint both landed in the right
# half on the default seed, the measured square's largest inscribed tumor region was
# 481 px -- under two tiles -- and the one structure those nests exist for was not being
# measured at all. Keep this in step with that tool's --max-size and --crop defaults.
METRICS_CROP_PX = 2048


def metrics_crop_box(height, width, size=METRICS_CROP_PX):
    """The square tiling_metrics.py measures by default: centred, clipped to fit."""
    side = min(size, height, width)
    return (height - side) // 2, (width - side) // 2, side


# How far the CK falloff reaches OUTSIDE the true nest boundary, as a fraction of
# transition_px -- nonzero so the boundary itself is still strongly CK-positive and the
# ambiguity lands in ground-truth Stroma, where it can actually mislead a model.
#
# Capped at a fraction of the NARROWEST stroma corridor, and that cap is the important
# part: at transition_px 140 an uncapped outward reach of 70 px would flood a 90 px
# corridor from both sides and leave no stroma core at all. Those corridors would then
# be unlabelable at hard, and the suite would report failures that have nothing to do
# with tiling. The cap is arithmetic against TRACT_WIDTH_PX rather than a tuned number,
# so widening the transition or narrowing the corridors cannot silently break it.
EDGE_OUT_FRACTION = 0.5
EDGE_OUT_CORRIDOR_SHARE = 0.30

MIN_REGION_PX = 400  # regions smaller than this are absorbed into their neighbour
TILE = (512, 512)
GEOJSON_MAX_BYTES = 5 * 1024 * 1024
# Simplification ladder for perfect mode, tried in order until the file fits the budget.
# 0.0 only drops collinear vertices from the pixel staircase, so it is lossless; the
# larger values trade a sub-pixel boundary shift for a smaller file.
SIMPLIFY_EPSILONS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def log(msg):
    print("[synth-mplex] " + msg, flush=True)


# ------------------------------------------------------------------------------
# random fields
# ------------------------------------------------------------------------------


def smooth_noise(rng, shape, sigma):
    """Unit-variance Gaussian noise blurred by sigma (scalar or per-axis tuple)."""
    field = rng.standard_normal(shape, dtype=np.float32)
    ndimage.gaussian_filter(field, sigma, output=field, mode="reflect")
    std = float(field.std())
    if std > 0:
        field /= std
    return field


def fbm(rng, shape, sigmas, weights):
    """Sum of blurred noise octaves, normalized to unit variance."""
    out = np.zeros(shape, dtype=np.float32)
    for sigma, weight in zip(sigmas, weights):
        out += np.float32(weight) * smooth_noise(rng, shape, sigma)
    std = float(out.std())
    if std > 0:
        out /= std
    return out


# ------------------------------------------------------------------------------
# geometry: tissue outline and tumor nests
# ------------------------------------------------------------------------------


def build_tissue_mask(rng, height, width, coverage):
    """A single irregular tissue section occupying roughly `coverage` of the slide."""
    scale = math.sqrt(height * width) / 3464.0  # 4000x3000 is the reference scale
    field = fbm(
        rng,
        (height, width),
        [140.0 * scale, 55.0 * scale, 22.0 * scale],
        [1.0, 0.45, 0.20],
    )

    # Elliptical falloff so the section sits inside the slide with a background rim.
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt((yy / 0.94) ** 2 + (xx / 0.94) ** 2)
    score = field - 3.0 * np.clip(radius - 0.58, 0.0, None)

    thresh = float(np.quantile(score, 1.0 - coverage))
    mask = score > thresh

    # Smooth away single-pixel speckle, then drop everything but the main section.
    soft = ndimage.gaussian_filter(mask.astype(np.float32), 4.0 * scale, mode="reflect")
    mask = soft > 0.5
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(sizes.argmax())
    return mask


def _blob_profile(rng, n_harmonics=4, amplitude=0.16):
    """Random radial perturbation r(theta)/r0 for an irregular but convex-ish blob."""
    orders = np.arange(2, 2 + n_harmonics)
    amps = amplitude * rng.uniform(0.4, 1.0, size=n_harmonics) / orders**0.5
    phases = rng.uniform(0.0, 2.0 * math.pi, size=n_harmonics)
    return orders, amps.astype(np.float32), phases.astype(np.float32)


def _profile_radius(theta, orders, amps, phases, r0):
    out = np.ones_like(theta)
    for order, amp, phase in zip(orders, amps, phases):
        out = out + amp * np.cos(order * theta + phase)
    return r0 * out


def _stamp_blob(rng, mask, cy, cx, r0, out=None):
    """OR one irregular blob of nominal radius r0 into mask (and into `out` if given)."""
    height, width = mask.shape
    orders, amps, phases = _blob_profile(rng)
    reach = int(math.ceil(r0 * 1.45)) + 2
    y0, y1 = max(0, cy - reach), min(height, cy + reach + 1)
    x0, x1 = max(0, cx - reach), min(width, cx + reach + 1)
    dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
    dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
    rr = np.hypot(dy, dx)
    th = np.arctan2(dy, dx)
    blob = rr < _profile_radius(th, orders, amps, phases, r0)
    np.logical_or(mask[y0:y1, x0:x1], blob, out=mask[y0:y1, x0:x1])
    if out is not None:
        np.logical_or(out[y0:y1, x0:x1], blob, out=out[y0:y1, x0:x1])


def build_tumor_mask(rng, tissue, target_frac):
    """Irregular tumor nests inside the tissue, covering ~target_frac of the tissue.

    Returns (tumor, giant) where `giant` marks the handful of nests big enough that a
    256 px tile can sit wholly inside one. Both come out of the geometry RNG stream
    only, so neither depends on the difficulty.
    """
    height, width = tissue.shape
    scale = math.sqrt(height * width) / 3464.0
    r_min, r_max = 55.0 * scale, 230.0 * scale

    # Keep nest centres away from the tissue edge so nests are not mostly clipped.
    edge_dist = ndimage.distance_transform_edt(tissue).astype(np.float32)
    eligible = edge_dist > (r_min * 0.8)

    tumor = np.zeros_like(tissue)
    giant = np.zeros_like(tissue)
    tissue_area = int(tissue.sum())
    want = target_frac * tissue_area
    placed = []

    # Giant nests first, so they get the room. Their radius is absolute pixels, not
    # scaled: "several 256 px tiles across" is a statement about the tile grid. On a
    # small test image they are clamped to whatever actually fits.
    giant_lo = min(GIANT_NEST_RADIUS_PX[0], min(height, width) / 7.0)
    giant_hi = min(GIANT_NEST_RADIUS_PX[1], min(height, width) / 5.0)
    n_giant = 0
    crop_y, crop_x, crop_side = metrics_crop_box(height, width)
    for index in range(N_GIANT_NESTS):
        # The first one has to fit inside the measurement crop; the rest go anywhere.
        # Two passes so that an unsatisfiable constraint costs a nest's placement
        # rather than the nest itself.
        for require_in_crop in (index == 0, False):
            for _attempt in range(4000):
                cy = int(rng.integers(0, height))
                cx = int(rng.integers(0, width))
                r0 = float(
                    rng.uniform(min(giant_lo, giant_hi), max(giant_lo, giant_hi))
                )
                if require_in_crop and not (
                    cy - r0 >= crop_y
                    and cy + r0 <= crop_y + crop_side
                    and cx - r0 >= crop_x
                    and cx + r0 <= crop_x + crop_side
                ):
                    continue
                if float(edge_dist[cy, cx]) < r0 * 1.15:
                    continue
                if any(
                    math.hypot(cy - py, cx - px) < 1.10 * (r0 + pr)
                    for py, px, pr in placed
                ):
                    continue
                _stamp_blob(rng, tumor, cy, cx, r0, out=giant)
                placed.append((cy, cx, r0))
                n_giant += 1
                break
            if n_giant > index:
                break
            if require_in_crop:
                log(
                    "WARNING: no giant nest fits inside the %d px measurement crop at "
                    "(%d,%d); placing it anywhere instead" % (crop_side, crop_x, crop_y)
                )

    attempts = 0
    max_attempts = 20000

    while tumor.sum() < want and attempts < max_attempts:
        attempts += 1
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        if not eligible[cy, cx]:
            continue
        r0 = float(rng.uniform(r_min, r_max))
        r0 = min(r0, float(edge_dist[cy, cx]) * 1.35)
        if r0 < r_min * 0.6:
            continue
        # Nests may cluster but should not fuse into one region-sized blob.
        too_close = False
        for py, px, pr in placed:
            if math.hypot(cy - py, cx - px) < 0.85 * (r0 + pr):
                too_close = True
                break
        if too_close:
            continue

        _stamp_blob(rng, tumor, cy, cx, r0)
        placed.append((cy, cx, r0))

    np.logical_and(tumor, tissue, out=tumor)
    np.logical_and(giant, tumor, out=giant)
    if n_giant < N_GIANT_NESTS:
        log(
            "WARNING: only %d of %d giant nests fit; a tile may not be able to sit "
            "inside one with no boundary in view" % (n_giant, N_GIANT_NESTS)
        )
    log(
        "placed %d tumor nests (%d giant) in %d attempts"
        % (len(placed), n_giant, attempts)
    )
    return tumor, giant


def carve_stroma_tracts(rng, tumor, tissue, giant):
    """Cut thin stroma corridors across the tumor, between tumor regions.

    Geometry, so it happens at every difficulty. The corridors are 90-210 px wide --
    narrower than a 256 px tile stride on purpose, since a structure the tile grid
    cannot resolve is exactly where a blended overlay and a centre-cropped one differ.
    """
    height, width = tumor.shape
    lo = min(TRACT_WIDTH_PX[0], min(height, width) / 12.0)
    hi = min(TRACT_WIDTH_PX[1], min(height, width) / 6.0)
    widths = []
    for _ in range(N_TRACTS):
        width_px = float(rng.uniform(min(lo, hi), max(lo, hi)))
        # A gently wandering path clear across the slide, so it is bound to cross
        # several nests rather than skirting them.
        span = np.linspace(0.0, 1.0, 6)
        if rng.random() < 0.5:
            ys = span * height
            xs = span * width + rng.uniform(-0.18, 0.18, size=6) * width
        else:
            ys = span * height + rng.uniform(-0.18, 0.18, size=6) * height
            xs = span * width
        ys = np.clip(ys, 0.0, height - 1.0)
        xs = np.clip(xs, 0.0, width - 1.0)

        # Rasterize the centreline, then take everything within half the width of it.
        line = np.zeros((height, width), dtype=bool)
        for i in range(len(span) - 1):
            steps = int(max(abs(ys[i + 1] - ys[i]), abs(xs[i + 1] - xs[i]))) + 1
            yy = np.linspace(ys[i], ys[i + 1], steps).astype(np.int64)
            xx = np.linspace(xs[i], xs[i + 1], steps).astype(np.int64)
            line[yy, xx] = True
        dist = ndimage.distance_transform_edt(~line)
        tract = dist <= (width_px * 0.5)
        # Never cut the giant nests. A corridor through one would split it into halves
        # too small to still hold a tile with no boundary in sight, which is the one
        # property those nests exist for.
        np.logical_and(tract, ~giant, out=tract)
        np.logical_and(tumor, ~tract, out=tumor)
        widths.append(width_px)

    np.logical_and(tumor, tissue, out=tumor)
    log(
        "carved %d stroma tracts, widths %s px"
        % (len(widths), ", ".join("%.0f" % w for w in widths))
    )
    return tumor


def enforce_min_area(labels, min_area, n_classes=3, max_passes=6):
    """Absorb regions below min_area into whatever class surrounds them.

    Keeps the label mask and the polygons in agreement: after this there is no region
    too small to annotate, so the GeoJSON needs no area filter of its own.
    """
    struct = np.ones((3, 3), bool)
    absorbed = 0
    for _ in range(max_passes):
        changed = False
        for cls in range(n_classes):
            # 4-connectivity, to match how class_polygons splits regions.
            comps, count = ndimage.label(labels == cls)
            if count == 0:
                continue
            sizes = np.bincount(comps.ravel())
            small = [i for i in range(1, count + 1) if sizes[i] < min_area]
            if not small:
                continue
            boxes = ndimage.find_objects(comps)
            for idx in small:
                sl = boxes[idx - 1]
                grown = tuple(
                    slice(max(0, s.start - 2), min(dim, s.stop + 2))
                    for s, dim in zip(sl, labels.shape)
                )
                sub_comp = comps[grown] == idx
                if not sub_comp.any():
                    continue
                ring = ndimage.binary_dilation(sub_comp, struct) & ~sub_comp
                neighbours = labels[grown][ring]
                if neighbours.size == 0:
                    continue
                winner = int(np.bincount(neighbours, minlength=n_classes).argmax())
                if winner == cls:
                    continue
                labels[grown][sub_comp] = winner
                absorbed += 1
                changed = True
        if not changed:
            break
    if absorbed:
        log("absorbed %d region(s) smaller than %d px" % (absorbed, min_area))
    return labels


# ------------------------------------------------------------------------------
# cells
# ------------------------------------------------------------------------------

# The brightest cell in the image is what sets max/p99, and an uncapped lognormal makes
# that ratio grow with the number of cells drawn -- so a 4000x3000 slide would have a
# visibly longer tail than a 1200x900 one from the same settings. Capping at a fixed
# multiple of sigma pins the ratio to the distribution instead of to the image size.
AMP_CAP_SIGMAS = 1.9


def cell_amplitudes(rng, sigma, count):
    """Lognormal cell brightness with a size-independent ceiling."""
    amps = rng.lognormal(0.0, sigma, size=count)
    return np.minimum(amps, math.exp(AMP_CAP_SIGMAS * sigma))


def _gauss_kernel(sigma_y, sigma_x):
    ry = max(2, int(math.ceil(3.0 * sigma_y)))
    rx = max(2, int(math.ceil(3.0 * sigma_x)))
    dy = np.arange(-ry, ry + 1, dtype=np.float32)[:, None]
    dx = np.arange(-rx, rx + 1, dtype=np.float32)[None, :]
    return np.exp(-(dy * dy) / (2.0 * sigma_y**2) - (dx * dx) / (2.0 * sigma_x**2))


# Membrane ring width. Named because _membrane_kernel and _flat_kernel must be called
# with the SAME value or their footprints differ and the per-cell blend breaks.
MEMBRANE_THICKNESS = 1.35


def _kernel_reach(radius, thickness):
    """Half-size of a cell kernel. Shared so blendable kernels cannot drift apart."""
    return int(math.ceil(radius + 3.0 * thickness))


def _membrane_kernel(radius, thickness):
    reach = _kernel_reach(radius, thickness)
    dy = np.arange(-reach, reach + 1, dtype=np.float32)[:, None]
    dx = np.arange(-reach, reach + 1, dtype=np.float32)[None, :]
    dist = np.hypot(dy, dx)
    ring = np.exp(-((dist - radius) ** 2) / (2.0 * thickness**2))
    cyto = 0.22 * np.exp(-(dist**2) / (2.0 * (radius * 0.75) ** 2))
    return ring + cyto


def _flat_kernel(radius, thickness):
    """A featureless disc the size of a cell -- the membrane kernel with the ring gone.

    Shares _membrane_kernel's reach so the two can be blended elementwise; both call
    _kernel_reach rather than repeating the arithmetic.
    """
    reach = _kernel_reach(radius, thickness)
    dy = np.arange(-reach, reach + 1, dtype=np.float32)[:, None]
    dx = np.arange(-reach, reach + 1, dtype=np.float32)[None, :]
    dist = np.hypot(dy, dx)
    return np.exp(-(dist**4) / (2.0 * (radius * 1.15) ** 4))


def stamp(image, kernel, cy, cx, amplitude):
    """Add amplitude * kernel centred on (cy, cx), clipped at the image border."""
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    y0, y1 = cy - ry, cy + ry + 1
    x0, x1 = cx - rx, cx + rx + 1
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(image.shape[0], y1), min(image.shape[1], x1)
    if sy0 >= sy1 or sx0 >= sx1:
        return
    image[sy0:sy1, sx0:sx1] += (
        np.float32(amplitude) * kernel[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
    )


def sample_points(rng, mask, count):
    """Rejection-sample `count` pixel positions uniformly inside mask."""
    if count <= 0:
        return np.empty((0, 2), dtype=np.int64)
    height, width = mask.shape
    out = []
    budget = max(40 * count, 4000)
    while len(out) < count and budget > 0:
        draw = max(count - len(out), 64) * 3
        budget -= draw
        ys = rng.integers(0, height, size=draw)
        xs = rng.integers(0, width, size=draw)
        keep = mask[ys, xs]
        for y, x in zip(ys[keep], xs[keep]):
            out.append((int(y), int(x)))
            if len(out) >= count:
                break
    return np.array(out, dtype=np.int64) if out else np.empty((0, 2), dtype=np.int64)


def jittered_grid(rng, mask, spacing, jitter):
    """Jittered lattice points that fall inside mask -- packed cells, no rejection loop."""
    height, width = mask.shape
    gy = np.arange(spacing // 2, height, spacing)
    gx = np.arange(spacing // 2, width, spacing)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    yy = yy.ravel() + rng.integers(-jitter, jitter + 1, size=yy.size)
    xx = xx.ravel() + rng.integers(-jitter, jitter + 1, size=xx.size)
    np.clip(yy, 0, height - 1, out=yy)
    np.clip(xx, 0, width - 1, out=xx)
    keep = mask[yy, xx]
    return np.stack([yy[keep], xx[keep]], axis=1)


# ------------------------------------------------------------------------------
# image synthesis
# ------------------------------------------------------------------------------


def select_roles(n_channels):
    """Pick n_channels marker roles, always keeping cytokeratin last.

    CK is what marks tumor, so it must survive any channel count; extra channels above
    five are autofluorescence-only planes (they still carry bleed-through and texture).
    """
    if n_channels < 2:
        raise ValueError("--channels must be at least 2")
    if n_channels <= len(ROLES):
        roles = [dict(r) for r in ROLES[: n_channels - 1]] + [
            dict(ROLES[CK_ROLE_INDEX])
        ]
    else:
        roles = [dict(r) for r in ROLES]
        for extra in range(n_channels - len(ROLES)):
            roles.append(
                dict(
                    name="Autofluorescence %d" % (extra + 1),
                    kind="haze",
                    target_p99=2.0,
                    per_stroma=0,
                    per_tumor=0,
                    sigma=0.0,
                    af=0.10,
                    amp_sigma=0.4,
                )
            )
    return roles


def ck_modulation(rng, labels, giant, cfg):
    """Per-pixel CK behaviour for one difficulty. Never touches `labels`.

    Returns (gain, region, flat):
      gain    multiplies each CK cell's brightness -- carries the nest-edge falloff,
              the CK-negative gaps and the dim nests
      region  where CK cells may be placed, which extends OUTSIDE ground-truth Tumor
              once there is a transition band
      flat    0 to 1, how far a cell there should lose its membrane definition
    """
    height, width = labels.shape
    scale = math.sqrt(height * width) / 3464.0
    tumor = labels == TUMOR
    transition = cfg["transition_px"]

    if transition <= 0.0:
        # Crisp: CK stops exactly where Tumor stops.
        gain = tumor.astype(np.float32)
        region = tumor.copy()
    else:
        # Signed distance to the nest boundary, positive inside, then a smoothstep from
        # `outward` px outside the boundary (CK gone) to `inward` px inside (CK full).
        # The split is asymmetric on purpose: most of a wide transition is spent INSIDE
        # the nest, weakening CK over a broad strip of ground-truth Tumor, while the
        # part that spills into Stroma stays narrow enough to leave the thin corridors
        # a stroma core. See EDGE_OUT_CORRIDOR_SHARE.
        outward = min(
            EDGE_OUT_FRACTION * transition,
            EDGE_OUT_CORRIDOR_SHARE * TRACT_WIDTH_PX[0],
        )
        d_in = ndimage.distance_transform_edt(tumor)
        d_out = ndimage.distance_transform_edt(~tumor)
        signed = (d_in - d_out).astype(np.float32)
        t = np.clip((signed + outward) / transition, 0.0, 1.0)
        gain = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        region = signed > -outward
        del d_in, d_out, signed

    # CK-negative gaps inside ground-truth Tumor: a low-frequency field thresholded
    # at the requested area fraction, so the holes are blobs rather than speckle.
    if cfg["gap_frac"] > 0.0 and tumor.any():
        field = fbm(rng, (height, width), [30.0 * scale, 11.0 * scale], [1.0, 0.55])
        cut = float(np.quantile(field[tumor], cfg["gap_frac"]))
        gain[(field < cut) & tumor] *= 0.10
        del field

    # Whole nests held at an intermediate CK level. Picked per connected region so the
    # result reads as "this nest is dim", not as noise.
    if cfg["dim_nest_frac"] > 0.0 and tumor.any():
        comps, count = ndimage.label(tumor)
        if count:
            sizes = np.bincount(comps.ravel())
            sizes[0] = 0
            order = rng.permutation(np.arange(1, count + 1))
            budget = cfg["dim_nest_frac"] * float(tumor.sum())
            chosen, used = [], 0.0
            for comp_id in order:
                if used >= budget:
                    break
                chosen.append(int(comp_id))
                used += float(sizes[comp_id])
            if chosen:
                # One factor PER nest, not one shared by all of them: a set of dim
                # nests that are all dim to exactly the same degree is a giveaway.
                lut = np.ones(count + 1, dtype=np.float32)
                lut[chosen] = rng.uniform(0.30, 0.50, size=len(chosen))
                gain *= lut[comps]
        del comps

    # Giant-nest interiors lose their internal contrast, ramping up away from the rim
    # so the edge of a giant nest still looks like an edge.
    flat = np.zeros((height, width), dtype=np.float32)
    if cfg["flat_core"] > 0.0 and giant.any():
        d_giant = ndimage.distance_transform_edt(giant).astype(np.float32)
        ramp = np.clip((d_giant - 70.0) / 150.0, 0.0, 1.0)
        flat = (np.float32(cfg["flat_core"]) * ramp).astype(np.float32)
        del d_giant, ramp
    return gain, region, flat


def build_bleed_matrix(n_channels):
    matrix = np.eye(n_channels, dtype=np.float32)
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            dist = abs(i - j)
            matrix[i, j] = BLEED_BY_DISTANCE.get(dist, BLEED_FAR)
    return matrix


def synthesize(rng, labels, roles, cfg, giant, verbose=True):
    """Build the float32 channel stack from the label mask.

    `rng` must be the TEXTURE stream: this is the only function the difficulty reaches,
    so the geometry and annotation streams stay untouched by it.
    """
    height, width = labels.shape
    n_channels = len(roles)
    tumor = labels == TUMOR
    stroma = labels == STROMA
    stroma_area = int(stroma.sum())
    tumor_area = int(tumor.sum())

    ck_gain, ck_region, flat = ck_modulation(rng, labels, giant, cfg)

    # Class-dependent autofluorescence texture. Streaky in stroma (collagen-like),
    # fine and granular inside nests -- a local cue that survives with zero cells
    # in the field of view.
    if verbose:
        log("building autofluorescence texture fields")
    streak_a = smooth_noise(rng, (height, width), (1.4, 13.0))
    streak_b = smooth_noise(rng, (height, width), (13.0, 1.4))
    coarse = fbm(rng, (height, width), [5.0, 14.0], [1.0, 0.6])
    fibrous = 0.55 + 0.42 * streak_a + 0.42 * streak_b + 0.30 * coarse
    del streak_a, streak_b
    granular = (
        0.55 + 0.50 * fbm(rng, (height, width), [1.6, 4.0], [1.0, 0.5]) + 0.18 * coarse
    )
    del coarse
    np.clip(fibrous, 0.02, None, out=fibrous)
    np.clip(granular, 0.02, None, out=granular)

    # texture_contrast pulls the two textures toward a common mixture, which removes
    # the cue that lets a cell-free window still be placed. At 1.0 they stay distinct.
    blend = np.float32(1.0 - cfg["texture_contrast"])
    mixture = 0.5 * (fibrous + 0.55 * granular)
    haze = np.zeros((height, width), dtype=np.float32)
    haze[stroma] = ((1.0 - blend) * fibrous + blend * mixture)[stroma]
    haze[tumor] = ((1.0 - blend) * 0.55 * granular + blend * mixture)[tumor]
    del fibrous, granular, mixture
    # A flattened giant-nest core loses most of its texture as well, so a tile in the
    # middle of one has very little local evidence either way.
    haze *= 1.0 - 0.6 * flat

    # Tumor cells: a packed jittered lattice inside the nests, drawn as CK-positive
    # membranes so the tumor signal is membranous rather than a filled blob. The
    # lattice covers ck_region, which reaches outside ground-truth Tumor whenever the
    # difficulty asks for a soft margin.
    tumor_cells = jittered_grid(rng, ck_region, spacing=13, jitter=3)
    # Extra CK-positive cells stranded well inside ground-truth Stroma, clustered
    # around the nest margins the way an invasive front behaves.
    stranded = np.empty((0, 2), dtype=np.int64)
    if cfg["stranded_cells"] > 0.0:
        n_stranded = int(cfg["stranded_cells"] * len(tumor_cells))
        band = stroma & ~ck_region
        if cfg["transition_px"] > 0.0:
            near = ndimage.distance_transform_edt(~tumor) < (3.0 * cfg["transition_px"])
            band = band & near
            del near
        stranded = sample_points(rng, band, n_stranded)
        del band
    if verbose:
        log(
            "CK cells: %d in the nest lattice, %d stranded in stroma"
            % (len(tumor_cells), len(stranded))
        )

    channels = np.zeros((n_channels, height, width), dtype=np.float32)
    counts = []
    for idx, role in enumerate(roles):
        plane = channels[idx]
        plane += np.float32(role["af"]) * haze

        n_cells = 0
        if role["kind"] == "membrane":
            radii = [5.0 + 0.5 * k for k in range(4)]
            membranes = [_membrane_kernel(r, MEMBRANE_THICKNESS) for r in radii]
            # The flat counterpart of each membrane: same radius and the same
            # thickness, so the footprints match and the two can be blended per cell.
            # That blend is what drains the internal contrast from a giant-nest core.
            flats = [_flat_kernel(r, MEMBRANE_THICKNESS) for r in radii]
            cells = (
                np.concatenate([tumor_cells, stranded], axis=0)
                if len(stranded)
                else tumor_cells
            )
            amps = cell_amplitudes(rng, role["amp_sigma"], len(cells))
            picks = rng.integers(0, len(membranes), size=len(cells))
            # Stranded cells are dimmer than the nest proper, and their gain field
            # reads ~0 out there, so give them their own modest amplitude instead.
            stranded_amp = rng.uniform(0.35, 0.85, size=len(cells))
            n_lattice = len(tumor_cells)
            for i, ((cy, cx), amp, pick) in enumerate(zip(cells, amps, picks)):
                cy, cx = int(cy), int(cx)
                if i < n_lattice:
                    amp *= float(ck_gain[cy, cx])
                else:
                    amp *= float(stranded_amp[i])
                if amp <= 1e-4:
                    continue
                f = float(flat[cy, cx])
                if f <= 0.0:
                    kernel = membranes[pick]
                else:
                    kernel = (1.0 - f) * membranes[pick] + f * flats[pick]
                    amp *= 1.0 - 0.65 * f
                stamp(plane, kernel, cy, cx, amp)
                n_cells += 1
        elif role["kind"] in ("punctate", "blob"):
            n_stroma = (
                int(stroma_area / role["per_stroma"]) if role["per_stroma"] else 0
            )
            # immune_contrast moves the in-nest density toward the stromal density, so
            # cell density stops being a clean second opinion on the class.
            per_tumor = role["per_tumor"]
            if per_tumor and role["per_stroma"]:
                per_tumor = role["per_stroma"] + cfg["immune_contrast"] * (
                    per_tumor - role["per_stroma"]
                )
            n_tumor = int(tumor_area / per_tumor) if per_tumor else 0
            pts_s = sample_points(rng, stroma, n_stroma)
            pts_t = sample_points(rng, tumor, n_tumor)
            pts = np.concatenate([pts_s, pts_t], axis=0) if len(pts_t) else pts_s
            n_cells = len(pts)
            if role["kind"] == "punctate":
                kernels = [
                    _gauss_kernel(role["sigma"] * a, role["sigma"] * b)
                    for a, b in ((1.0, 1.0), (1.25, 0.85), (0.85, 1.25), (1.15, 1.15))
                ]
            else:
                kernels = [
                    _gauss_kernel(role["sigma"] * a, role["sigma"] * b)
                    for a, b in (
                        (1.0, 1.0),
                        (1.4, 0.7),
                        (0.7, 1.4),
                        (1.2, 1.0),
                        (1.0, 1.2),
                    )
                ]
            amps = cell_amplitudes(rng, role["amp_sigma"], n_cells)
            picks = rng.integers(0, len(kernels), size=n_cells)
            for (cy, cx), amp, pick in zip(pts, amps, picks):
                stamp(plane, kernels[pick], int(cy), int(cx), amp)
        counts.append(n_cells)

        # Scale to the reference dynamic range: mostly dark, long bright tail.
        p99 = float(np.percentile(plane, 99.0))
        gain = (role["target_p99"] / p99) if p99 > 1e-9 else 1.0
        plane *= np.float32(gain)
        role["_gain"] = gain
        role["_cells"] = n_cells
        if verbose:
            log(
                "channel %d %-18s cells=%-6d gain=%.3f"
                % (idx, role["name"], n_cells, gain)
            )
    del haze

    # Spectral bleed-through between channels (a few percent, mostly to neighbours).
    bleed = build_bleed_matrix(n_channels)
    if verbose:
        log("mixing spectral bleed-through")
    mixed = np.tensordot(bleed, channels, axes=([1], [0])).astype(np.float32)
    channels = mixed

    # Slow illumination gradient across the slide, plus a low-frequency blotch.
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    illum = (
        1.0
        + 0.19 * xx
        + 0.11 * yy
        + 0.09 * fbm(rng, (height, width), [max(height, width) / 7.0], [1.0])
    )
    np.clip(illum, 0.45, 1.8, out=illum)
    channels *= illum[None, :, :]
    illum_range = (float(illum.min()), float(illum.max()))
    del illum

    # Detector noise: a flat read term plus a signal-dependent shot term.
    for idx, role in enumerate(roles):
        plane = channels[idx]
        read = READ_NOISE_FRAC * role["target_p99"]
        plane += rng.normal(0.0, read, size=plane.shape).astype(np.float32)
        plane += (
            SHOT_NOISE_K
            * np.sqrt(np.clip(plane, 0.0, None))
            * rng.standard_normal(plane.shape, dtype=np.float32)
        )
    np.clip(channels, 0.0, None, out=channels)
    return channels, illum_range, counts


def channel_stats(plane):
    p1, p50, p99 = np.percentile(plane, [1.0, 50.0, 99.0])
    return dict(
        p1=round(float(p1), 5),
        p50=round(float(p50), 5),
        p99=round(float(p99), 5),
        min=round(float(plane.min()), 5),
        max=round(float(plane.max()), 5),
        mean=round(float(plane.mean()), 5),
        frac_below_0p05=round(float((plane < 0.05).mean()), 5),
    )


# ------------------------------------------------------------------------------
# mask -> polygons
# ------------------------------------------------------------------------------
# Boundary tracing walks the pixel-EDGE lattice, not pixel centres. A contour through
# pixel centres would shrink every region by half a pixel on each side, which shows up
# as a systematic bias when the annotations are scored against the label mask. On the
# edge lattice, corner (r, c) is the top-left corner of pixel (r, c), so the polygon
# for a single pixel is exactly its unit square and the round trip is lossless.
_CRACK_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # E, S, W, N as (dy, dx)


def _ahead(cy, cx, d):
    """The two pixels straddling the step ahead: (ahead-left, ahead-right)."""
    if d == 0:  # heading east
        return (cy - 1, cx), (cy, cx)
    if d == 1:  # heading south
        return (cy, cx), (cy, cx - 1)
    if d == 2:  # heading west
        return (cy, cx - 1), (cy - 1, cx - 1)
    return (cy - 1, cx - 1), (cy - 1, cx)  # heading north


def trace_crack(padded, start_pixel):
    """Trace the pixel-edge outline of one region, keeping foreground on the right.

    `padded` has a one-pixel false border, so the 2x2 lookups never leave the array.
    `start_pixel` is the region's first pixel in raster order; its top-left corner is
    on the outline and heading east keeps the region on the right.
    """
    py, px = start_pixel
    cy, cx, d = py, px, 0
    corners = [(cy, cx)]
    limit = 4 * padded.size + 16
    while len(corners) <= limit:
        (ly, lx), (ry, rx) = _ahead(cy, cx, d)
        if not padded[ry, rx]:
            d = (d + 1) % 4  # wall gone: turn right
        elif padded[ly, lx]:
            d = (d + 3) % 4  # blocked ahead: turn left
        cy += _CRACK_DIRS[d][0]
        cx += _CRACK_DIRS[d][1]
        if (cy, cx) == (py, px):
            break
        corners.append((cy, cx))
    return corners


def _ring_from_component(mask, comp_labels, comp_id, bbox):
    """Trace one labelled component, returning (x, y) image-edge coordinates."""
    grown = tuple(
        slice(max(0, s.start - 1), min(dim, s.stop + 1))
        for s, dim in zip(bbox, mask.shape)
    )
    sub = comp_labels[grown] == comp_id
    padded = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = sub
    ys, xs = np.nonzero(padded)
    order = np.lexsort((xs, ys))
    start = (int(ys[order[0]]), int(xs[order[0]]))
    corners = trace_crack(padded, start)
    oy = grown[0].start - 1
    ox = grown[1].start - 1
    return [(c[1] + ox, c[0] + oy) for c in corners]


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker on an open polyline; iterative, so long rings are safe."""
    n = len(points)
    if n < 3:
        return list(points)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    pts = np.asarray(points, dtype=np.float64)
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = pts[i]
        bx, by = pts[j]
        seg = pts[i + 1 : j]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            dist = np.hypot(seg[:, 0] - ax, seg[:, 1] - ay)
        else:
            dist = np.abs(dy * (seg[:, 0] - ax) - dx * (seg[:, 1] - ay)) / norm
        k = int(dist.argmax())
        if dist[k] > epsilon:
            split = i + 1 + k
            keep[split] = True
            stack.append((i, split))
            stack.append((split, j))
    return [tuple(points[i]) for i in np.flatnonzero(keep)]


def simplify_ring(ring, epsilon):
    """Simplify a closed ring: split at the point farthest from the start, then RDP."""
    if len(ring) < 8 or epsilon < 0:
        return _close_ring(ring)
    pts = np.asarray(ring, dtype=np.float64)
    far = int(np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1]).argmax())
    if far < 2 or far > len(ring) - 2:
        out = rdp(list(ring) + [ring[0]], epsilon)
    else:
        first = rdp(list(ring[: far + 1]), epsilon)
        second = rdp(list(ring[far:]) + [ring[0]], epsilon)
        out = first[:-1] + second
    if len(out) < 4:
        return _close_ring(ring)
    return _close_ring(out)


def _close_ring(ring):
    ring = [(int(x), int(y)) for x, y in ring]
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def _signed_area(ring):
    total = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def orient(ring, counter_clockwise):
    """RFC 7946 wants exterior rings counter-clockwise and holes clockwise."""
    area = _signed_area(ring)
    if (area > 0) != counter_clockwise:
        return list(reversed(ring))
    return ring


def class_polygons(labels, cls, epsilon):
    """Polygons (outer ring plus hole rings) covering every region of one class."""
    mask = labels == cls
    if not mask.any():
        return []
    # 4-connectivity for both the regions and their holes, because that is how
    # trace_crack resolves a diagonal touch: as two regions meeting at a corner. A
    # diagonally pinched region therefore becomes two polygons sharing a corner, whose
    # union is still exactly the class mask.
    comps, count = ndimage.label(mask)
    boxes = ndimage.find_objects(comps)

    # Holes are complement components that do not reach the image border.
    inv_comps, inv_count = ndimage.label(~mask)
    holes_by_owner = {}
    if inv_count:
        border = set(
            np.unique(
                np.concatenate(
                    [
                        inv_comps[0, :],
                        inv_comps[-1, :],
                        inv_comps[:, 0],
                        inv_comps[:, -1],
                    ]
                )
            )
        )
        border.discard(0)
        inv_boxes = ndimage.find_objects(inv_comps)
        for hole_id in range(1, inv_count + 1):
            if hole_id in border:
                continue
            sl = inv_boxes[hole_id - 1]
            # Walk west from a hole pixel: the first non-hole pixel must be foreground,
            # and it belongs to the component that encloses the hole.
            sub = inv_comps[sl] == hole_id
            ys, xs = np.nonzero(sub)
            y = int(ys[0]) + sl[0].start
            x = int(xs[0]) + sl[1].start
            while x > 0 and inv_comps[y, x] == hole_id:
                x -= 1
            owner = int(comps[y, x])
            if owner == 0:
                continue
            holes_by_owner.setdefault(owner, []).append((hole_id, sl))

    polygons = []
    for comp_id in range(1, count + 1):
        outer = _ring_from_component(mask, comps, comp_id, boxes[comp_id - 1])
        rings = [orient(simplify_ring(outer, epsilon), True)]
        for hole_id, hole_slice in holes_by_owner.get(comp_id, ()):
            ring = _ring_from_component(mask, inv_comps, hole_id, hole_slice)
            ring = simplify_ring(ring, epsilon)
            if len(ring) >= 4:
                rings.append(orient(ring, False))
        polygons.append(rings)
    return polygons


# ------------------------------------------------------------------------------
# GeoJSON
# ------------------------------------------------------------------------------


def qupath_color(rgb):
    """QuPath's colorRGB is a signed 32-bit ARGB integer."""
    r, g, b = rgb
    value = (0xFF << 24) | (r << 16) | (g << 8) | b
    return value - (1 << 32) if value >= (1 << 31) else value


def make_feature(rings, cls, name=None):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[int(x), int(y)] for x, y in ring] for ring in rings],
        },
        "properties": {
            "objectType": "annotation",
            "name": name or CLASS_NAMES[cls],
            "classification": {
                "name": CLASS_NAMES[cls],
                "colorRGB": qupath_color(CLASS_RGB[cls]),
            },
        },
    }


def perfect_annotations(labels):
    """One annotation per ground-truth region, simplified to fit the size budget."""
    for epsilon in SIMPLIFY_EPSILONS:
        features = []
        for cls in (STROMA, TUMOR, BACKGROUND):
            for rings in class_polygons(labels, cls, epsilon):
                features.append(make_feature(rings, cls))
        payload = {"type": "FeatureCollection", "features": features}
        text = json.dumps(payload, separators=(",", ":"))
        size = len(text.encode("ascii"))
        log(
            "perfect mode: epsilon=%.1f px -> %d features, %.2f MB"
            % (epsilon, len(features), size / 1048576.0)
        )
        if size <= GEOJSON_MAX_BYTES:
            return payload, text, epsilon
    return payload, text, SIMPLIFY_EPSILONS[-1]


# (class, maximum patches, share of the area budget). The counts are ceilings, not
# quotas: each class stops as soon as it has covered its share. Tumor gets the most
# slots because it has the least room -- a nest is only a couple of hundred pixels
# across, so reaching a given area there takes several patches where stroma needs one
# or two. That is also how a user brushes small nests.
SPARSE_PLAN = ((TUMOR, 12, 0.45), (STROMA, 7, 0.45), (BACKGROUND, 4, 0.10))
SPARSE_MAX_GROWTH = 1.6  # how far one patch may exceed its even share of the budget
SPARSE_EDGE_MARGIN = 0.92  # fraction of the free room a patch is allowed to fill


def sparse_annotations(rng, labels, target_frac=0.07):
    """A dozen or so brushed-looking patches, each wholly inside one class.

    Mimics a user who paints a handful of representative patches instead of the whole
    section, which is how the classifier actually gets trained. Every patch sits inside
    its class with room to spare, so the sparse labels are a subset of the perfect ones
    rather than an approximation of them -- a sparse-mode run that scores badly is the
    classifier generalizing poorly, not the annotations being wrong.
    """
    tissue_area = int((labels != BACKGROUND).sum())
    features = []
    for cls, n_blobs, share in SPARSE_PLAN:
        mask = labels == cls
        if not mask.any():
            continue
        # The distance transform gives the largest patch that fits at each pixel, so a
        # patch can be kept strictly inside its class without ever clipping it. The mask
        # is PADDED first: distance_transform_edt measures distance to the nearest zero
        # WITHIN the array, so without the pad a class that reaches the image edge (the
        # background rim does) reports unlimited room there and its patches run off the
        # image.
        inscribed = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1].astype(
            np.float32
        )
        budget = target_frac * tissue_area * share
        even_r = math.sqrt(budget / max(n_blobs, 1) / math.pi)
        candidates = inscribed > max(12.0, even_r * 0.25)
        if not candidates.any():
            continue
        centres = sample_points(rng, candidates, n_blobs * 8)
        if len(centres) == 0:
            continue
        # Roomiest spots first, so the blobs come out as close to full size as the
        # class geometry allows.
        room = inscribed[centres[:, 0], centres[:, 1]]
        centres = centres[np.argsort(-room)]

        placed = []
        remaining = budget
        for cy, cx in centres:
            if len(placed) >= n_blobs or remaining <= 0.0:
                break
            cy, cx = int(cy), int(cx)
            # Size against what is LEFT of the budget, not an even split: a cramped
            # class (small tumor nests) then lets its later patches take up the slack
            # instead of quietly undershooting the requested coverage.
            slots_left = n_blobs - len(placed)
            want_r = math.sqrt(remaining / slots_left / math.pi)
            r_cap = min(want_r, even_r) * SPARSE_MAX_GROWTH

            orders, amps, phases = _blob_profile(rng, n_harmonics=3, amplitude=0.20)
            theta = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
            unit = _profile_radius(theta, orders, amps, phases, 1.0)
            # Scale against the profile's OWN maximum rather than an assumed worst-case
            # perturbation, so "stays inside the class" is arithmetic, not a fudge
            # factor that a change to _blob_profile could quietly invalidate.
            reach = float(unit.max())
            room = float(inscribed[cy, cx]) * SPARSE_EDGE_MARGIN
            r = min(r_cap, room / reach)
            if r < 10.0:
                continue
            if any(
                math.hypot(cy - py, cx - px) < 1.15 * (r * reach + pr)
                for py, px, pr in placed
            ):
                continue
            ring = [
                (
                    int(round(cx + r * u * math.cos(t))),
                    int(round(cy + r * u * math.sin(t))),
                )
                for t, u in zip(theta, unit)
            ]
            ring = orient(_close_ring(ring), True)
            placed.append((cy, cx, r * reach))
            remaining -= abs(_signed_area(ring))
            features.append(
                make_feature(
                    [ring], cls, name="%s %d" % (CLASS_NAMES[cls], len(placed))
                )
            )
    payload = {"type": "FeatureCollection", "features": features}
    text = json.dumps(payload, separators=(",", ":"))
    return payload, text, 0.0


# ------------------------------------------------------------------------------
# verification
# ------------------------------------------------------------------------------


def rasterize_feature(rings, shape):
    """Render one polygon (with holes) back to a mask, in its own bounding box.

    Scanline fill with the even-odd rule, testing pixel CENTRES -- the same
    containment rule QuPath uses when it turns an ROI into labels. Pillow's polygon
    fill is not usable here: it paints the boundary as well, which would over-report
    agreement by a one-pixel ring around every region.
    """
    pts = np.concatenate([np.asarray(r, dtype=np.float64) for r in rings])
    x_lo = max(0, int(math.floor(pts[:, 0].min())))
    x_hi = min(shape[1] - 1, int(math.ceil(pts[:, 0].max())))
    y_lo = max(0, int(math.floor(pts[:, 1].min())))
    y_hi = min(shape[0] - 1, int(math.ceil(pts[:, 1].max())))
    if x_hi < x_lo or y_hi < y_lo:
        return None

    segs = [np.asarray(r, dtype=np.float64) for r in rings]
    ax = np.concatenate([s[:-1, 0] for s in segs])
    ay = np.concatenate([s[:-1, 1] for s in segs])
    bx = np.concatenate([s[1:, 0] for s in segs])
    by = np.concatenate([s[1:, 1] for s in segs])
    keep = ay != by  # horizontal edges never cross
    ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
    if ax.size == 0:
        return None
    slope = (bx - ax) / (by - ay)
    y_min = np.minimum(ay, by)
    y_max = np.maximum(ay, by)

    height = y_hi - y_lo + 1
    width = x_hi - x_lo + 1
    # Bucket each edge into the scanlines it crosses, so no row scans every edge.
    first = np.ceil(y_min - 0.5).astype(np.int64) - y_lo
    last = np.ceil(y_max - 0.5).astype(np.int64) - 1 - y_lo
    np.clip(first, 0, height, out=first)
    np.clip(last, -1, height - 1, out=last)
    counts = np.clip(last - first + 1, 0, None)
    total = int(counts.sum())
    if total == 0:
        return None
    edge_of = np.repeat(np.arange(ax.size), counts)
    starts = np.repeat(first, counts)
    bases = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    row_of = starts + (np.arange(total) - bases)
    order = np.argsort(row_of, kind="stable")
    row_of = row_of[order]
    edge_of = edge_of[order]
    bounds = np.searchsorted(row_of, np.arange(height + 1))

    mask = np.zeros((height, width), dtype=bool)
    for i in range(height):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        idx = edge_of[lo:hi]
        yc = y_lo + i + 0.5
        xs = np.sort(ax[idx] + (yc - ay[idx]) * slope[idx])
        for left, right in zip(xs[0::2], xs[1::2]):
            j0 = int(math.floor(left - 0.5)) + 1
            j1 = int(math.ceil(right - 0.5)) - 1
            j0 = max(j0, x_lo)
            j1 = min(j1, x_hi)
            if j1 >= j0:
                mask[i, j0 - x_lo : j1 - x_lo + 1] = True
    return (y_lo, x_lo), mask


def verify_annotations(payload, labels):
    """Rasterize the GeoJSON and compare it against the ground truth.

    This is the check that the dataset is worth scoring against: if the polygons and
    the label mask disagree, a classifier trained from the polygons can never match the
    mask. Returns a dict of measured agreement, coverage and worst-case purity.
    """
    recon = np.full(labels.shape, 255, dtype=np.uint8)
    worst = 1.0
    worst_name = "none"
    out_of_bounds = 0
    height, width = labels.shape
    name_to_cls = {v: k for k, v in CLASS_NAMES.items()}
    for feature in payload["features"]:
        cls = name_to_cls[feature["properties"]["classification"]["name"]]
        rings = [
            [(int(x), int(y)) for x, y in ring]
            for ring in feature["geometry"]["coordinates"]
        ]
        # A vertex outside the image is not something QuPath should have to cope with,
        # and the rasterizer below clips silently, so count them here instead.
        for ring in rings:
            out_of_bounds += sum(
                1 for x, y in ring if not (0 <= x <= width and 0 <= y <= height)
            )
        out = rasterize_feature(rings, labels.shape)
        if out is None:
            continue
        (y0, x0), mask = out
        window = labels[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
        mask = mask[: window.shape[0], : window.shape[1]]
        if not mask.any():
            continue
        purity = float((window[mask] == cls).mean())
        if purity < worst:
            worst = purity
            worst_name = feature["properties"].get("name", CLASS_NAMES[cls])
        recon[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]][mask] = cls

    annotated = recon != 255
    tissue = labels != BACKGROUND
    tissue_px = int(tissue.sum())
    n_annotated = int(annotated.sum())
    return {
        "image_fraction": round(n_annotated / float(labels.size), 6),
        "tissue_fraction": (
            round(float((annotated & tissue).sum()) / tissue_px, 6)
            if tissue_px
            else 0.0
        ),
        "agreement": (
            round(float((recon[annotated] == labels[annotated]).mean()), 6)
            if n_annotated
            else 0.0
        ),
        "mismatched_pixels": int((recon[annotated] != labels[annotated]).sum()),
        "worst_feature_purity": round(worst, 6),
        "worst_feature": worst_name,
        "out_of_bounds_vertices": out_of_bounds,
    }


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------


def deterministic_uuid(args):
    """A UUID-shaped string derived from the generation parameters, not from chance."""
    key = "%s|%s|%d|%d|%d|%d|%.6f|%s" % (
        GENERATOR,
        VERSION,
        args.seed,
        args.width,
        args.height,
        args.channels,
        args.pixel_size_um,
        args.difficulty,
    )
    d = hashlib.sha1(key.encode("ascii")).hexdigest()
    return "urn:uuid:%s-%s-%s-%s-%s" % (d[0:8], d[8:12], d[12:16], d[16:20], d[20:32])


def parse_args(argv):
    repo = Path(__file__).resolve().parents[2]
    default_out = repo / "tools" / "synthetic" / "data" / "multiplex"
    parser = argparse.ArgumentParser(
        description="Generate a synthetic multiplex IF image with perfect ground truth."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="output directory (default: %s)" % default_out,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260927,
        help="RNG seed; the output is deterministic for a given seed",
    )
    parser.add_argument("--width", type=int, default=4000, help="image width in pixels")
    parser.add_argument(
        "--height", type=int, default=3000, help="image height in pixels"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=5,
        help="number of marker channels (2-8; CK is always last)",
    )
    parser.add_argument(
        "--annotation-mode",
        choices=("perfect", "sparse"),
        default="perfect",
        help="perfect: polygons follow the ground truth exactly; "
        "sparse: 15-20 brushed patches covering 5-10 pct of the tissue",
    )
    parser.add_argument(
        "--difficulty",
        choices=tuple(DIFFICULTY_PRESETS),
        default="medium",
        help="how ambiguous the PICTURE is; ground_truth.png and annotations.geojson "
        "are byte-identical across all three for a given seed",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.498,
        help="pixel size in microns, written to the TIFF and manifest",
    )
    parser.add_argument(
        "--tissue-fraction",
        type=float,
        default=0.84,
        help="fraction of the slide covered by tissue",
    )
    parser.add_argument(
        "--tumor-fraction",
        type=float,
        default=0.30,
        help="fraction of the tissue covered by tumor nests",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.width < 256 or args.height < 256:
        print("ERROR: --width and --height must be at least 256", file=sys.stderr)
        return 2
    if not 2 <= args.channels <= 8:
        print("ERROR: --channels must be between 2 and 8", file=sys.stderr)
        return 2
    if args.pixel_size_um <= 0:
        print("ERROR: --pixel-size-um must be positive", file=sys.stderr)
        return 2

    started = time.time()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DIFFICULTY_PRESETS[args.difficulty]
    height, width = args.height, args.width

    # THREE independent RNG streams, and the reason matters: the difficulty knob draws
    # a different number of random numbers, so a single shared stream would shift every
    # draw after it and the label mask and annotations would change with the difficulty.
    # Split this way, geometry and annotations see the same numbers at every difficulty,
    # and only image.tif moves. Same derivation as the sibling RGB generator.
    geo_rng = np.random.default_rng(args.seed)
    tex_rng = np.random.default_rng(args.seed ^ 0x5EED1234)
    ann_rng = np.random.default_rng(args.seed ^ 0x5A5A5A5A)

    log(
        "seed=%d size=%dx%d channels=%d mode=%s difficulty=%s"
        % (
            args.seed,
            width,
            height,
            args.channels,
            args.annotation_mode,
            args.difficulty,
        )
    )

    # --- geometry (geo stream only: never sees the difficulty) ---
    tissue = build_tissue_mask(geo_rng, height, width, args.tissue_fraction)
    tumor, giant = build_tumor_mask(geo_rng, tissue, args.tumor_fraction)
    tumor = carve_stroma_tracts(geo_rng, tumor, tissue, giant)
    np.logical_and(giant, tumor, out=giant)
    labels = np.full((height, width), BACKGROUND, dtype=np.uint8)
    labels[tissue] = STROMA
    labels[tumor] = TUMOR
    labels = enforce_min_area(labels, MIN_REGION_PX)
    # enforce_min_area may have reassigned slivers, so re-clip the giant-nest mask to
    # what is still Tumor before anything reads it.
    np.logical_and(giant, labels == TUMOR, out=giant)

    # --- pixels (tex stream only: this is where the difficulty lives) ---
    roles = select_roles(args.channels)
    channels, illum_range, _ = synthesize(tex_rng, labels, roles, cfg, giant)

    # --- write the image ---
    names = [r["name"] for r in roles]
    image_path = out_dir / "image.tif"
    log("writing %s" % image_path.name)
    tifffile.imwrite(
        image_path,
        channels,
        photometric="minisblack",
        tile=TILE,
        ome=True,
        compression="zlib",
        # No PhysicalSize*Unit: micrometer is the OME default, and the schema spells it
        # with the micro sign. Writing "um" instead makes Bio-Formats throw
        # EnumerationException on the WHOLE OME block, fall back to a plain TIFF reader,
        # and present the five planes as five timepoints of a one-channel image -- which
        # is what it did before this comment existed. Leaving the attribute out gets the
        # right unit without putting a non-ASCII byte in the file.
        metadata={
            "axes": "CYX",
            "Channel": {"Name": names},
            "PhysicalSizeX": args.pixel_size_um,
            "PhysicalSizeY": args.pixel_size_um,
            # Pin the OME UUID to the generation parameters. tifffile otherwise mints a
            # fresh random one per write, which makes two runs of the same seed differ
            # in bytes while the pixels are identical -- awkward for a suite that wants
            # to checksum its inputs.
            "UUID": deterministic_uuid(args),
        },
        resolution=(10000.0 / args.pixel_size_um, 10000.0 / args.pixel_size_um),
        resolutionunit="CENTIMETER",
    )

    # --- write the ground truth ---
    mask_path = out_dir / "ground_truth.png"
    Image.fromarray(labels, mode="L").save(mask_path, optimize=True)

    # --- write the annotations ---
    if args.annotation_mode == "perfect":
        payload, text, epsilon = perfect_annotations(labels)
    else:
        payload, text, epsilon = sparse_annotations(ann_rng, labels)
    geojson_path = out_dir / "annotations.geojson"
    geojson_path.write_text(text, encoding="ascii")
    check = verify_annotations(payload, labels)

    # --- manifest ---
    stats = [channel_stats(channels[i]) for i in range(len(roles))]
    class_counts = {cls: int((labels == cls).sum()) for cls in CLASS_NAMES}
    total_px = float(height * width)
    px_area = args.pixel_size_um**2
    manifest = {
        "generator": GENERATOR,
        "generator_version": VERSION,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "difficulty_settings": dict(cfg),
        "pixel_size_um": args.pixel_size_um,
        "image": {
            "file": image_path.name,
            "width": width,
            "height": height,
            "channels": len(roles),
            "dtype": "float32",
            "axes": "CYX",
            "format": "OME-TIFF",
            "tiled": True,
            "tile_size": list(TILE),
            "compression": "zlib",
        },
        "channels": [
            {
                "index": i,
                "name": roles[i]["name"],
                "role": roles[i]["kind"],
                "target_p99": roles[i]["target_p99"],
                "gain": round(float(roles[i].get("_gain", 1.0)), 6),
                "cells": int(roles[i].get("_cells", 0)),
                "stats": stats[i],
            }
            for i in range(len(roles))
        ],
        "classes": [
            {
                "label": cls,
                "name": CLASS_NAMES[cls],
                "color_rgb": list(CLASS_RGB[cls]),
                "color_qupath": qupath_color(CLASS_RGB[cls]),
                "pixels": class_counts[cls],
                "area_fraction": round(class_counts[cls] / total_px, 6),
                "area_um2": round(class_counts[cls] * px_area, 2),
            }
            for cls in sorted(CLASS_NAMES)
        ],
        "ground_truth": {
            "file": mask_path.name,
            "encoding": "uint8 label mask",
            "labels": {str(cls): CLASS_NAMES[cls] for cls in sorted(CLASS_NAMES)},
        },
        "annotations": {
            "file": geojson_path.name,
            "mode": args.annotation_mode,
            "features": len(payload["features"]),
            "bytes": len(text.encode("ascii")),
            "simplify_epsilon_px": epsilon,
            "annotated_image_fraction": check["image_fraction"],
            "annotated_tissue_fraction": check["tissue_fraction"],
            "rasterized_agreement": check["agreement"],
            "rasterized_mismatched_pixels": check["mismatched_pixels"],
            "worst_feature_purity": check["worst_feature_purity"],
            "out_of_bounds_vertices": check["out_of_bounds_vertices"],
            "worst_feature": check["worst_feature"],
        },
        "bleed_through": [
            [round(float(v), 4) for v in row] for row in build_bleed_matrix(len(roles))
        ],
        "illumination_gain_range": [round(illum_range[0], 4), round(illum_range[1], 4)],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )

    # --- summary ---
    elapsed = time.time() - started
    print("")
    print("Wrote synthetic multiplex dataset to %s" % out_dir)
    print(
        "  image.tif            %dx%d x %d ch, float32, tiled %dx%d, %.1f MB"
        % (
            width,
            height,
            len(roles),
            TILE[0],
            TILE[1],
            image_path.stat().st_size / 1048576.0,
        )
    )
    print(
        "  ground_truth.png     uint8 labels 0=Stroma 1=Tumor 2=Background, %.1f KB"
        % (mask_path.stat().st_size / 1024.0)
    )
    print(
        "  annotations.geojson  %s mode, %d features, %.2f MB, epsilon=%.1f px"
        % (
            args.annotation_mode,
            len(payload["features"]),
            len(text.encode("ascii")) / 1048576.0,
            epsilon,
        )
    )
    print(
        "  manifest.json        seed=%d difficulty=%s pixel size=%.3f um"
        % (args.seed, args.difficulty, args.pixel_size_um)
    )
    print("")
    print("Class areas:")
    for cls in sorted(CLASS_NAMES):
        print(
            "  %-11s %10d px  %6.2f pct  %12.0f um2"
            % (
                CLASS_NAMES[cls],
                class_counts[cls],
                100.0 * class_counts[cls] / total_px,
                class_counts[cls] * px_area,
            )
        )
    print("")
    print("Channel stats (units match the LuCa component data):")
    print(
        "  %-2s %-18s %8s %8s %8s %8s %8s %8s"
        % ("ch", "name", "p1", "p50", "p99", "max", "mean", "<0.05")
    )
    for i, role in enumerate(roles):
        s = stats[i]
        print(
            "  %-2d %-18s %8.4f %8.4f %8.4f %8.3f %8.4f %8.3f"
            % (
                i,
                role["name"],
                s["p1"],
                s["p50"],
                s["p99"],
                s["max"],
                s["mean"],
                s["frac_below_0p05"],
            )
        )
    print("")
    print("Annotation check (rasterized back and compared to ground_truth.png):")
    print(
        "  annotated      %.2f pct of the image, %.2f pct of the tissue"
        % (100.0 * check["image_fraction"], 100.0 * check["tissue_fraction"])
    )
    print(
        "  agreement      %.4f pct (%d mismatched pixels)"
        % (100.0 * check["agreement"], check["mismatched_pixels"])
    )
    print(
        "  worst feature  purity %.4f (%s)"
        % (check["worst_feature_purity"], check["worst_feature"])
    )
    print("  vertices outside the image extent: %d" % check["out_of_bounds_vertices"])
    print(
        "Illumination gain across the slide: %.3f to %.3f"
        % (illum_range[0], illum_range[1])
    )
    print("Done in %.1f s" % elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
