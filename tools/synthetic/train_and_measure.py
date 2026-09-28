#!/usr/bin/env python3
"""Train on a synthetic dataset and measure the tiling artifact end to end.

This closes the loop the rest of the suite leaves open. The generators write
data and assert things about its statistics; tiling_metrics.py measures a
model. Neither proves that a model trained on this data can be measured, or
that the measurement moves when the data gets harder. This does both, without
QuPath: it exports patches in the layout the training service expects, trains,
and then measures the result.

What it is NOT: a substitute for running the real thing in QuPath. It skips
the Java exporter, the Appose boundary, and the overlay. It answers one
question -- does the fixture detect what it was built to detect -- and a green
result here does not mean the shipped path works.

Usage:
    python_server/venv/bin/python tools/synthetic/train_and_measure.py \\
        --data tools/synthetic/data/rgb \\
        --work /tmp/synth-run --epochs 20

    # the comparison the suite exists for
    for d in easy medium hard; do
        python_server/venv/bin/python tools/synthetic/generate_rgb_tissue.py \\
            --out data/$d --difficulty $d --width 3072 --height 2048
        python_server/venv/bin/python tools/synthetic/train_and_measure.py \\
            --data data/$d --work work/$d --epochs 20
    done

Run it in the DL classifier venv. ASCII-only output (Windows cp1252).
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "python_server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Background is a real class in the generated masks. Dropping it would leave
# the model nowhere to put glass, and it would classify slide as tissue.
UNLABELED = 255


def _log(msg):
    print("[train] " + msg, flush=True)


def write_raw_float(path, hwc):
    """The .raw format the training dataset reads: 12-byte H,W,C header."""
    h, w, c = hwc.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", h, w, c))
        f.write(np.ascontiguousarray(hwc, dtype=np.float32).tobytes())


def export_patches(image, mask, out_dir, patch, stride, val_fraction, min_label, rng):
    """Cuts the slide into patches and writes the training-service layout.

    Splits by a checkerboard of patch BLOCKS rather than at random, so a
    validation patch never overlaps a training patch. Random splitting with
    any overlap leaks pixels across the boundary and the validation score
    stops meaning anything -- the same trap TileOverlapSplitWatcher exists
    for on the Java side.
    """
    H, W = mask.shape
    for split in ("train", "validation"):
        for kind in ("images", "masks"):
            (out_dir / split / kind).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "validation": 0}
    block = max(1, int(round(1.0 / max(val_fraction, 1e-6))))
    idx = 0
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            m = mask[y : y + patch, x : x + patch]
            labelled = float((m != UNLABELED).mean())
            if labelled < min_label:
                continue
            # Every block-th patch in a checkerboard goes to validation.
            split = (
                "validation" if ((y // patch) + (x // patch)) % block == 0 else "train"
            )
            name = "patch_%05d" % idx
            idx += 1
            write_raw_float(
                out_dir / split / "images" / (name + ".raw"),
                image[y : y + patch, x : x + patch],
            )
            Image.fromarray(m.astype(np.uint8)).save(
                out_dir / split / "masks" / (name + ".png")
            )
            counts[split] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data",
        required=True,
        help="a generator output dir (image.tif, ground_truth.png, manifest.json)",
    )
    ap.add_argument(
        "--work", required=True, help="scratch dir for patches and the trained model"
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patch", type=int, default=256, help="model input size")
    ap.add_argument(
        "--context-pad",
        type=int,
        default=0,
        help="export patches this much larger per side; training random-crops "
        "back to --patch, so this is how much tile-framing variation the "
        "model sees (the 'overlap' field of a training profile)",
    )
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument(
        "--min-label",
        type=float,
        default=0.5,
        help="minimum labelled fraction to keep a patch",
    )
    ap.add_argument("--max-patches", type=int, default=400)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--keep", action="store_true", help="keep the exported patches")
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--json", help="write the measurement rows here")
    args = ap.parse_args()

    import tifffile

    from dlclassifier_server.services.training_service import TrainingService

    data = Path(args.data)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text())

    raw = tifffile.imread(str(data / "image.tif"))
    if raw.ndim == 2:
        raw = raw[:, :, None]
    elif raw.ndim == 3 and raw.shape[0] <= 16 and raw.shape[0] < raw.shape[-1]:
        raw = np.transpose(raw, (1, 2, 0))
    image = raw.astype(np.float32)
    mask = np.array(Image.open(data / "ground_truth.png"))
    if image.shape[:2] != mask.shape:
        raise SystemExit(
            "image %s and mask %s disagree" % (image.shape[:2], mask.shape)
        )

    classes = (
        [c["name"] for c in manifest["classes"]]
        if "classes" in manifest
        else ["Stroma", "Tumor", "Background"]
    )
    n_channels = image.shape[2]
    difficulty = manifest.get(
        "difficulty", manifest.get("difficulty_settings", {}).get("name", "unspecified")
    )
    _log(
        "%s: %dx%d, %d channels, classes %s, difficulty %s"
        % (data.name, image.shape[1], image.shape[0], n_channels, classes, difficulty)
    )

    export = work / "export"
    if export.exists():
        shutil.rmtree(export)
    rng = np.random.default_rng(0)
    export_size = args.patch + 2 * args.context_pad
    counts = export_patches(
        image,
        mask,
        export,
        export_size,
        args.stride,
        args.val_fraction,
        args.min_label,
        rng,
    )
    _log(
        "exported %d train / %d validation patches at %dpx (model input %d, context pad %d)"
        % (
            counts["train"],
            counts["validation"],
            export_size,
            args.patch,
            args.context_pad,
        )
    )
    if counts["train"] == 0 or counts["validation"] == 0:
        raise SystemExit("one split is empty; lower --min-label or --stride")
    if args.max_patches and counts["train"] > args.max_patches:
        # Workshop scale: a participant brushes a few regions, not a slide.
        # Keeping a deterministic subset makes the small-data regime testable.
        for split, keep in (
            ("train", args.max_patches),
            ("validation", max(2, args.max_patches // 4)),
        ):
            imgs = sorted((export / split / "images").glob("*.raw"))
            for extra in imgs[keep:]:
                extra.unlink()
                (export / split / "masks" / (extra.stem + ".png")).unlink(missing_ok=True)
            counts[split] = min(counts[split], keep)
        _log("trimmed to %d train / %d validation patches" % (counts["train"], counts["validation"]))

    (export / "config.json").write_text(
        json.dumps(
            {
                "patch_size": args.patch,
                "context_padding": args.context_pad,
                "downsample": 1.0,
                "unlabeled_index": UNLABELED,
                "classes": [{"index": i, "name": n} for i, n in enumerate(classes)],
                "channel_config": {"num_channels": n_channels, "context_scale": 1},
            },
            indent=2,
        )
    )

    svc = TrainingService()
    result = svc.train(
        model_type="unet",
        architecture={
            "type": "unet",
            "backbone": args.backbone,
            "encoder_depth": 5,
            "decoder_channels": [256, 128, 64, 32, 16],
            "input_channels": n_channels,
            "input_size": [args.patch, args.patch],
            "use_pretrained": True,
            "use_batchrenorm": True,
            "downsample": 1.0,
            "context_scale": 1,
        },
        input_config={
            "num_channels": n_channels,
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": True,
                "clip_percentile": 99.0,
            },
        },
        training_params={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 0.001,
            "validation_split": 0.0,
            "early_stopping": False,
            "mixed_precision": True,
            "data_loader_workers": 0,
            "seed": 1,
        },
        classes=classes,
        data_path=str(export),
    )

    model_path = result.get("model_path") or result.get("output_dir")
    _log(
        "trained: %s"
        % json.dumps(
            {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
        )[:300]
    )
    if not model_path:
        raise SystemExit("training returned no model path: %s" % list(result))

    if not args.keep:
        shutil.rmtree(export, ignore_errors=True)

    if args.skip_measure:
        return

    # Measure the trained model on the slide it was trained on. That is
    # deliberate: this asks whether the TILING is stable, not whether the
    # model generalises. A model that has memorised the slide and still
    # changes its answer when the grid moves is showing the artifact in its
    # purest form.
    import tiling_metrics as tm

    meta, isvc, run = tm.build_model(Path(model_path), "cuda" if _cuda() else "cpu")
    ic = meta.get("input_config", {})
    crop = min(2048, image.shape[0], image.shape[1])
    y0 = (image.shape[0] - crop) // 2
    x0 = (image.shape[1] - crop) // 2
    sub = image[y0 : y0 + crop, x0 : x0 + crop]
    gt = mask[y0 : y0 + crop, x0 : x0 + crop]
    nchw = np.transpose(isvc._normalize(sub, ic), (2, 0, 1))[None].astype(np.float32)

    rows = []
    print()
    print("difficulty  tile  overlap  mode    shift-invariance   error vs truth")
    for tile, frac, mode in [
        (256, 0.200, "crop"),
        (256, 0.375, "crop"),
        (256, 0.200, "linear"),
    ]:
        pad = int(round(tile * frac))
        stride = tile - 2 * pad
        a, _ = tm.tiled(run, nchw, tile, pad, mode, offset=0)
        b, _ = tm.tiled(run, nchw, tile, pad, mode, offset=stride // 2)
        valid = gt != UNLABELED
        row = {
            "difficulty": difficulty,
            "tile": tile,
            "overlap_fraction": frac,
            "mode": mode,
            "shift_disagreement_pct": round(100.0 * float((a != b).mean()), 3),
            "error_vs_truth_pct": round(
                100.0 * float((a[valid] != gt[valid]).mean()), 3
            ),
        }
        rows.append(row)
        print(
            "%-10s %5d  %6.1f%%  %-6s  %13.2f%%   %13.2f%%"
            % (
                difficulty,
                tile,
                100 * frac,
                mode,
                row["shift_disagreement_pct"],
                row["error_vs_truth_pct"],
            )
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"model": str(model_path), "rows": rows}, indent=2)
        )
        _log("wrote " + args.json)


def _cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
