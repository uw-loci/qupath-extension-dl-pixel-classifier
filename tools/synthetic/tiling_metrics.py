#!/usr/bin/env python3
"""Measure how much a model's answer depends on where the tile grid falls.

This is the pre-release check for tiling artifacts. It exists because a real
one shipped: with the overlay taking each pixel from a single tile and a 20%
halo, whole regions came back as the wrong class in rectangles aligned to the
tile grid, while the per-tile training preview looked near-perfect. Nothing in
the test suite could see it, because every existing test scores one tile at a
time and the defect only exists BETWEEN tiles.

Three numbers, in decreasing order of how much you should trust them:

  shift invariance   Run the same tiling twice with the grid offset half a
                     stride, and count pixels whose class changed. Needs no
                     reference at all, and the pixels it counts are exactly
                     the ones a user sees as blocks. This is the headline.

  vs ground truth    Only available on synthetic data, where the mask is
                     known. Says whether the model is right, which shift
                     invariance does not: a model that answers "Stroma"
                     everywhere is perfectly shift invariant.

  vs consensus       The same model at its native tile size, averaged over
                     many grid offsets. A stand-in for ground truth on real
                     slides. Do NOT use a single whole-image forward pass for
                     this -- a model trained at 256x256 is out of
                     distribution at 2000x1500, and an early version of this
                     measurement drew the wrong conclusion from exactly that
                     mistake.

Usage:
    python_server/venv/bin/python tools/synthetic/tiling_metrics.py \\
        --model /path/to/classifiers/dl/<id> \\
        --image tools/synthetic/data/rgb/image.tif \\
        [--ground-truth tools/synthetic/data/rgb/ground_truth.png] \\
        [--sweep] [--json out.json]

Run it in the DL classifier venv. ASCII-only output (Windows cp1252).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "python_server"))

# The geometries worth checking before a release. The first is what the
# extension shipped as its default overlap; the third is the floor it now
# enforces for center-crop.
DEFAULT_SWEEP = [
    (256, 0.200, "crop"),
    (256, 0.200, "linear"),
    (256, 0.250, "crop"),
    (256, 0.250, "linear"),
    (256, 0.375, "crop"),
    (256, 0.375, "linear"),
    (512, 0.250, "crop"),
    (512, 0.250, "linear"),
]


def _log(msg):
    print("[tiling] " + msg, flush=True)


def load_image(path, selected_channels):
    """Reads a TIFF as float32 H,W,C, keeping only the model's channels."""
    import tifffile

    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CYX -> YXC
    arr = arr.astype(np.float32)
    if selected_channels:
        arr = arr[:, :, list(selected_channels)]
    return arr


def build_model(model_dir, device):
    """Loads the model the way production inference does."""
    import torch

    from dlclassifier_server.services.inference_service import InferenceService

    meta = json.loads((model_dir / "metadata.json").read_text())
    svc = InferenceService(device="cpu")
    kind, obj = svc._load_model(str(model_dir))
    if kind == "onnx":
        # The static variant bakes one tile size into the graph, which is the
        # one thing this tool must vary. Fall back to the dynamic export.
        import onnxruntime as ort

        dyn = model_dir / "model.onnx"
        if not dyn.exists():
            raise SystemExit(
                "need model.onnx (dynamic shape) to vary tile size; only a static export is present"
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(str(dyn), providers=providers)
        name = sess.get_inputs()[0].name

        def run(batch):
            return sess.run(None, {name: batch})[0]

    else:
        obj.eval().to(device)

        def run(batch):
            with torch.no_grad():
                t = torch.from_numpy(batch).to(device)
                return torch.softmax(obj(t), 1).cpu().numpy()

    return meta, svc, run


def tiled(run, nchw, tile, pad, mode, offset=0):
    """One tile grid over the whole image.

    mode 'crop' keeps the middle band of each tile and nothing else, which is
    what the overlay does. Any other mode accumulates a weighted average of
    every tile covering the pixel, which is what Apply Classifier does.
    """
    _, c, H, W = nchw.shape
    stride = tile - 2 * pad
    if stride <= 0:
        raise ValueError("padding %d leaves no stride at tile %d" % (pad, tile))

    if mode == "crop":
        out = np.zeros((H, W), np.uint8)
        covered = np.zeros((H, W), bool)
    else:
        r = np.minimum(np.arange(tile), np.arange(tile)[::-1]).astype(np.float32)
        w1 = np.clip((r + 0.5) / max(pad, 1), 0, 1)
        if mode == "gauss":
            w1 = 0.5 - 0.5 * np.cos(np.pi * w1)
        wmap = np.maximum(np.outer(w1, w1), 1e-6)
        acc = None
        den = np.zeros((H, W), np.float32)

    for cy in range(-offset, H, stride):
        for cx in range(-offset, W, stride):
            y0 = min(max(cy - pad, 0), max(H - tile, 0))
            x0 = min(max(cx - pad, 0), max(W - tile, 0))
            patch = nchw[:, :, y0 : y0 + tile, x0 : x0 + tile]
            ph, pw = patch.shape[2], patch.shape[3]
            if ph < tile or pw < tile:
                patch = np.pad(
                    patch,
                    ((0, 0), (0, 0), (0, tile - ph), (0, tile - pw)),
                    mode="reflect",
                )
            prob = run(np.ascontiguousarray(patch))[0]
            if mode == "crop":
                ky, kx = max(cy, 0), max(cx, 0)
                y1, x1 = min(ky + stride, H), min(kx + stride, W)
                if y1 <= ky or x1 <= kx:
                    continue
                out[ky:y1, kx:x1] = prob[
                    :, ky - y0 : y1 - y0, kx - x0 : x1 - x0
                ].argmax(0)
                covered[ky:y1, kx:x1] = True
            else:
                if acc is None:
                    acc = np.zeros((prob.shape[0], H, W), np.float32)
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                acc[:, y0:y1, x0:x1] += (
                    prob[:, : y1 - y0, : x1 - x0] * wmap[: y1 - y0, : x1 - x0]
                )
                den[y0:y1, x0:x1] += wmap[: y1 - y0, : x1 - x0]

    if mode == "crop":
        return out, float((~covered).mean())
    return (acc / np.maximum(den, 1e-6)).argmax(0).astype(np.uint8), float(
        (den == 0).mean()
    )


def consensus(run, nchw, tile, shift=32):
    """The model's own answer, averaged over every shift-offset placement."""
    _, _, H, W = nchw.shape
    acc, den = None, np.zeros((H, W), np.float32)
    ys = sorted(set(list(range(0, max(H - tile, 0) + 1, shift)) + [max(H - tile, 0)]))
    xs = sorted(set(list(range(0, max(W - tile, 0) + 1, shift)) + [max(W - tile, 0)]))
    for y0 in ys:
        for x0 in xs:
            prob = run(
                np.ascontiguousarray(nchw[:, :, y0 : y0 + tile, x0 : x0 + tile])
            )[0]
            if acc is None:
                acc = np.zeros((prob.shape[0], H, W), np.float32)
            acc[:, y0 : y0 + tile, x0 : x0 + tile] += prob
            den[y0 : y0 + tile, x0 : x0 + tile] += 1
    return (acc / np.maximum(den, 1)[None]).argmax(0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        required=True,
        help="classifier dir with metadata.json and model.onnx / model.pt",
    )
    ap.add_argument("--image", required=True, help="image to measure on")
    ap.add_argument("--ground-truth", help="label PNG; enables the accuracy column")
    ap.add_argument(
        "--gt-class-map",
        help="JSON list mapping model class index -> ground-truth label value",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="run the standard pre-release geometry sweep",
    )
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument(
        "--overlap", type=float, default=0.20, help="fraction of the tile, per side"
    )
    ap.add_argument("--mode", default="crop", choices=("crop", "linear", "gauss"))
    ap.add_argument(
        "--consensus-shift",
        type=int,
        default=32,
        help="0 to skip the consensus reference",
    )
    ap.add_argument(
        "--max-size",
        type=int,
        default=2048,
        help="measure a square of this size rather than the whole image",
    )
    ap.add_argument(
        "--crop",
        default="center",
        choices=("center", "topleft"),
        help="which square to measure when the image is larger than --max-size",
    )
    ap.add_argument("--json", help="write results here as well as printing them")
    args = ap.parse_args()

    model_dir = Path(args.model)
    device = "cpu"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    meta, svc, run = build_model(model_dir, device)
    ic = meta.get("input_config", {})
    full = load_image(Path(args.image), ic.get("selected_channels"))
    fh, fw = full.shape[:2]
    # WHICH square gets measured is not a detail. A whole-slide image is
    # mostly background, and a corner of one can be entirely background --
    # every tiling is then perfectly shift invariant and the measurement says
    # nothing. This bit a synthetic dataset built for this tool: both of its
    # deliberately tile-spanning regions sat outside the default top-left
    # square, so the one structure they existed to exercise was never
    # measured, and nothing said so.
    # A SQUARE, clipped to the shorter side, not each axis independently.
    # The generators place their tile-spanning structures inside the region
    # this computes, so the two must agree exactly; clipping per axis made
    # them diverge on any image smaller than max_size in one dimension --
    # which is every quick test run.
    ch = cw = min(args.max_size, fh, fw)
    if args.crop == "center":
        y0, x0 = (fh - ch) // 2, (fw - cw) // 2
    else:
        y0, x0 = 0, 0
    img = full[y0 : y0 + ch, x0 : x0 + cw]
    H, W = img.shape[:2]
    _log(
        "image %dx%d, measuring %dx%d at (%d,%d) [%s], %d channels, device %s"
        % (fw, fh, W, H, x0, y0, args.crop, img.shape[2], device)
    )

    norm = svc._normalize(img, ic)
    nchw = np.transpose(norm, (2, 0, 1))[None].astype(np.float32)

    gt = None
    if args.ground_truth:
        from PIL import Image

        gt = np.array(Image.open(args.ground_truth))[y0 : y0 + H, x0 : x0 + W]
        if args.gt_class_map:
            mapping = json.loads(args.gt_class_map)
            remap = np.full(256, 255, np.uint8)
            for model_idx, gt_value in enumerate(mapping):
                remap[gt_value] = model_idx
            gt = remap[gt]
        _log("ground truth loaded, %d labelled values" % len(np.unique(gt)))

    ref = None
    if args.consensus_shift > 0:
        _log("building consensus reference (shift %d)..." % args.consensus_shift)
        ref = consensus(run, nchw, args.tile, args.consensus_shift)

    cases = DEFAULT_SWEEP if args.sweep else [(args.tile, args.overlap, args.mode)]
    rows = []
    print()
    print(
        "tile  overlap  mode    shift-invariance   vs truth   vs consensus   uncovered"
    )
    for tile, frac, mode in cases:
        pad = int(round(tile * frac))
        stride = tile - 2 * pad
        a, gap = tiled(run, nchw, tile, pad, mode, offset=0)
        b, _ = tiled(run, nchw, tile, pad, mode, offset=stride // 2)
        row = {
            "tile": tile,
            "overlap_fraction": frac,
            "padding_px": pad,
            "stride_px": stride,
            "mode": mode,
            "shift_disagreement_pct": round(100.0 * float((a != b).mean()), 3),
            "uncovered_pct": round(100.0 * gap, 3),
        }
        if gt is not None:
            valid = gt != 255
            row["error_vs_truth_pct"] = round(
                100.0 * float((a[valid] != gt[valid]).mean()), 3
            )
        if ref is not None:
            row["disagreement_vs_consensus_pct"] = round(
                100.0 * float((a != ref).mean()), 3
            )
        rows.append(row)
        print(
            "%4d  %6.1f%%  %-6s  %13.2f%%   %8s   %12s   %7.2f%%"
            % (
                tile,
                100 * frac,
                mode,
                row["shift_disagreement_pct"],
                ("%.2f%%" % row["error_vs_truth_pct"]) if gt is not None else "-",
                (
                    ("%.2f%%" % row["disagreement_vs_consensus_pct"])
                    if ref is not None
                    else "-"
                ),
                row["uncovered_pct"],
            )
        )

    print()
    print("shift-invariance is the one to watch: it needs no reference, and the")
    print("pixels it counts are the ones that appear as blocks on the tile grid.")

    # A region the model answers uniformly is perfectly shift invariant and
    # proves nothing. Say so rather than letting a flat 0.00% read as success.
    first, _ = tiled(
        run, nchw, cases[0][0], int(round(cases[0][0] * cases[0][1])), cases[0][2]
    )
    counts = np.bincount(first.ravel())
    dominant = counts.max() / float(first.size)
    if dominant > 0.95:
        print()
        print(
            "WARNING: the model predicts one class over %.1f%% of the measured"
            % (100 * dominant)
        )
        print("region, so every tiling agrees trivially and these numbers mean")
        print("little. Measure somewhere with both classes in it -- try --crop")
        print("topleft, a larger --max-size, or a different image.")
    if any(r["uncovered_pct"] > 0 for r in rows):
        print("WARNING: a row left pixels uncovered. Those pixels hold whatever the")
        print("output array was initialised to, and every other number in that row")
        print("is measured partly against fill rather than against a prediction.")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "model": str(model_dir),
                    "image": args.image,
                    "device": device,
                    "rows": rows,
                },
                indent=2,
            )
        )
        _log("wrote " + args.json)


if __name__ == "__main__":
    main()
