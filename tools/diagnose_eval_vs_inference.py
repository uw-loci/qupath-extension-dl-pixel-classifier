#!/usr/bin/env python3
"""Phase 0 diagnostic: does the per-tile eval runtime match the inference runtime?

The Training Area Issues per-tile preview runs the trained model with PyTorch eager
(model.pt) over the exported on-disk tiles. Production inference runs the SAME weights
through ONNX (model_static_bn.onnx / model_static.onnx / model.onnx). This script feeds
one exported tile through BOTH runtimes on a byte-identical input tensor and reports how
much the outputs differ.

It isolates the "runtime mismatch" candidate (PyTorch vs ONNX) from the "geometry
mismatch" candidate (different tile reconstruction), which is analysed separately in
docs/CONTEXT_AND_DOWNSAMPLE_DESIGN.md. A large disagreement here means the runtime itself
diverges even on identical input; a small disagreement points the finger at geometry
instead.

Usage:
    python tools/diagnose_eval_vs_inference.py \
        --model /path/to/classifiers/dl/<id> \
        --data  /path/to/training_data_dir \
        [--split train|validation] [--tiles 5]

    --model  classifier dir containing model.pt + model.onnx + metadata.json
    --data   the exported training-data dir (train/images, train/masks, train/context, ...)

Run it in the DL classifier venv:
    python_server/venv/bin/python tools/diagnose_eval_vs_inference.py ...

ASCII-only output (the project runs on Windows cp1252).
"""

import argparse
import json
import sys
from pathlib import Path

# Make the dlclassifier_server package importable without installing.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "python_server"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dlclassifier_server.services.training_service import (  # noqa: E402
    SegmentationDataset,
    TrainingService,
)

from dlclassifier_server.utils.batchrenorm import replace_bn_with_batchrenorm  # noqa: E402


def _log(msg):
    print("[diagnose] " + msg, flush=True)


def build_pytorch_model(model_dir, input_config, classes):
    """Mirror evaluate_tiles.py model construction + model.pt load."""
    meta_path = model_dir / "metadata.json"
    arch = {}
    if meta_path.exists():
        with open(meta_path) as f:
            arch = json.load(f).get("architecture", {})

    context_scale = int(arch.get("context_scale", 1))
    num_channels = int(input_config.get("num_channels", 3))
    if context_scale > 1:
        num_channels *= 2

    svc = TrainingService(gpu_manager=None)
    model = svc._create_model(
        model_type=arch.get("type", "unet"),
        architecture=arch,
        num_channels=num_channels,
        num_classes=len(classes),
    )

    pt_path = model_dir / "model.pt"
    if not pt_path.exists():
        raise FileNotFoundError("No model.pt at %s" % pt_path)
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    has_brn = any(k.endswith(".rmax") or k.endswith(".dmax") for k in state)
    if has_brn:
        replace_bn_with_batchrenorm(model)
        _log("auto-detected BatchRenorm from state dict")
    model.load_state_dict(state)
    model.eval()
    return model, context_scale, num_channels


def load_onnx_session(model_dir):
    """Mirror inference_service preference order: static_bn -> static -> dynamic."""
    import onnxruntime as ort

    for name, label in (
        ("model_static_bn.onnx", "static-BN"),
        ("model_static.onnx", "static"),
        ("model.onnx", "dynamic"),
    ):
        path = model_dir / name
        if path.exists():
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            return sess, label
    raise FileNotFoundError("No ONNX model (model_static_bn/static/.onnx) at %s" % model_dir)


def center_fit(arr_nchw, target_hw):
    """Center-pad (edge) or center-crop arr (N,C,H,W) to target (H,W).

    Returns (fitted_array, (off_y, off_x), did_pad). Mirrors the static-shape
    handling in inference_service._infer_batch_spatial closely enough for a
    runtime-only comparison.
    """
    _, _, h, w = arr_nchw.shape
    th, tw = target_hw
    if (h, w) == (th, tw):
        return arr_nchw, (0, 0), False
    if h <= th and w <= tw:
        py0 = (th - h) // 2
        px0 = (tw - w) // 2
        out = np.pad(
            arr_nchw,
            ((0, 0), (0, 0), (py0, th - h - py0), (px0, tw - w - px0)),
            mode="edge",
        )
        return out, (py0, px0), True
    # crop (handles oversized in either dim by cropping that dim)
    cy0 = max(0, (h - th) // 2)
    cx0 = max(0, (w - tw) // 2)
    out = arr_nchw[:, :, cy0 : cy0 + min(th, h), cx0 : cx0 + min(tw, w)]
    return out, (cy0, cx0), False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="classifier dir (model.pt + .onnx + metadata.json)")
    ap.add_argument("--data", required=True, help="exported training-data dir")
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--tiles", type=int, default=5, help="number of tiles to test")
    args = ap.parse_args()

    model_dir = Path(args.model)
    data_root = Path(args.data)

    cfg_path = data_root / "config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    classes = config.get("classes") or config.get("class_names")
    if not classes:
        meta = json.loads((model_dir / "metadata.json").read_text())
        classes = meta.get("classes") or meta.get("class_names") or []
    input_config = config.get("input_config", config)

    _log("model dir : %s" % model_dir)
    _log("data dir  : %s (%s split)" % (data_root, args.split))
    _log("classes   : %s" % classes)

    model, context_scale, num_channels = build_pytorch_model(model_dir, input_config, classes)
    _log("context_scale=%d, model input channels=%d" % (context_scale, num_channels))

    sess, onnx_label = load_onnx_session(model_dir)
    onnx_in = sess.get_inputs()[0]
    _log("ONNX variant=%s, baked input shape=%s" % (onnx_label, onnx_in.shape))

    images_dir = data_root / args.split / "images"
    masks_dir = data_root / args.split / "masks"
    context_dir = None
    if context_scale > 1:
        ctx = data_root / args.split / "context"
        if ctx.exists():
            context_dir = str(ctx)
        else:
            _log("WARNING: context_scale>1 but no %s -- eval will duplicate detail tile" % ctx)

    ds = SegmentationDataset(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        input_config=input_config,
        augment=False,
        context_dir=context_dir,
    )
    n = min(args.tiles, len(ds))
    _log("evaluating %d / %d tiles" % (n, len(ds)))
    _log("")

    agree_pcts = []
    max_abs_diffs = []
    for i in range(n):
        img_t, _mask = ds[i]  # (C,H,W) float32, normalized, channel order [detail.., context..]
        x = img_t.unsqueeze(0).numpy().astype(np.float32)
        _, c, h, w = x.shape

        with torch.no_grad():
            pt_logits = model(img_t.unsqueeze(0)).cpu().numpy()  # (1, K, H, W)

        # ONNX: adapt to baked H/W if static, else feed as-is.
        baked = onnx_in.shape
        target_hw = None
        if len(baked) == 4 and isinstance(baked[2], int) and isinstance(baked[3], int):
            target_hw = (baked[2], baked[3])
        if target_hw and target_hw != (h, w):
            x_fit, (oy, ox), did_pad = center_fit(x, target_hw)
            note = "%s to %s" % ("padded" if did_pad else "cropped", target_hw)
        else:
            x_fit, (oy, ox), did_pad, note = x, (0, 0), False, "as-is"

        onnx_out = sess.run(None, {onnx_in.name: x_fit})[0]  # (1, K, Hf, Wf)

        # Bring both to a common HxW for comparison (center region).
        ph, pw = pt_logits.shape[2], pt_logits.shape[3]
        oh, ow = onnx_out.shape[2], onnx_out.shape[3]
        ch, cw = min(ph, oh), min(pw, ow)

        def _center(a, ch, cw):
            ah, aw = a.shape[2], a.shape[3]
            y0 = (ah - ch) // 2
            x0 = (aw - cw) // 2
            return a[:, :, y0 : y0 + ch, x0 : x0 + cw]

        pt_c = _center(pt_logits, ch, cw)
        on_c = _center(onnx_out, ch, cw)

        pt_arg = pt_c.argmax(axis=1)
        on_arg = on_c.argmax(axis=1)
        agree = float((pt_arg == on_arg).mean()) * 100.0
        max_abs = float(np.abs(pt_c - on_c).max())
        agree_pcts.append(agree)
        max_abs_diffs.append(max_abs)
        _log(
            "tile %2d: input(C=%d,H=%d,W=%d) onnx=%s | argmax agreement=%.2f%% "
            "max|logit diff|=%.4g" % (i, c, h, w, note, agree, max_abs)
        )

    _log("")
    if agree_pcts:
        _log(
            "SUMMARY over %d tiles: mean argmax agreement=%.2f%% (min %.2f%%), "
            "mean max|logit diff|=%.4g"
            % (
                len(agree_pcts),
                float(np.mean(agree_pcts)),
                float(np.min(agree_pcts)),
                float(np.mean(max_abs_diffs)),
            )
        )
        _log("")
        _log("Interpretation:")
        _log("  agreement ~100%% and tiny logit diff -> runtimes match; the eval-vs-")
        _log("    production mismatch is GEOMETRY (see CONTEXT_AND_DOWNSAMPLE_DESIGN.md).")
        _log("  large disagreement -> the ONNX runtime itself diverges from model.pt on")
        _log("    identical input (ONNX export / static-shape / BatchRenorm). Fix that first.")
        _log("  note any 'padded/cropped to' lines: if the ONNX baked H/W differs from the")
        _log("    trained tile size, that is itself a geometry/shape divergence to record.")


if __name__ == "__main__":
    main()
