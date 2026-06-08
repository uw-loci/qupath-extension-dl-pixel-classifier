"""
Post-training tile evaluation via Appose.

Runs the trained model over all training tiles and computes per-tile
quality metrics (loss, disagreement, per-class IoU) to help identify
annotation errors and hard cases.

Inputs:
    model_path: str        - path to trained model directory
    data_path: str         - path to training data directory (with tile_manifest.json)
    architecture: dict     - model architecture config
    input_config: dict     - channel/normalization config
    classes: list of str   - class names

Outputs:
    results: str (JSON)    - per-tile evaluation results sorted by loss descending
    total_tiles: int
    tiles_evaluated: int
"""

import json
import logging
import threading

logger = logging.getLogger("dlclassifier.appose.evaluate_tiles")

if inference_service is None:
    raise RuntimeError(
        "Services not initialized: " + globals().get("init_error", "unknown")
    )

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

from dlclassifier_server.services.training_service import (
    TrainingService,
    SegmentationDataset,
)
from dlclassifier_server.utils.batchrenorm import replace_bn_with_batchrenorm

# Default class colors (used when class_colors input is not provided)
DEFAULT_COLORS = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 255, 0),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 128, 0),  # orange
    (128, 0, 255),  # purple
    (0, 128, 255),  # light blue
    (255, 0, 128),  # pink
]


def get_class_color(class_idx, class_name, class_colors_map):
    """Get (R, G, B) for a class index, using provided colors or defaults."""
    if class_colors_map and class_name in class_colors_map:
        packed = class_colors_map[class_name]
        # Packed RGB integer: 0xRRGGBB (may be negative in Java signed int)
        packed = packed & 0xFFFFFF
        r = (packed >> 16) & 0xFF
        g = (packed >> 8) & 0xFF
        b = packed & 0xFF
        return (r, g, b)
    return DEFAULT_COLORS[class_idx % len(DEFAULT_COLORS)]


def save_disagreement_map(
    pred, mask, ignore_index, classes, class_colors_map, output_path, patch_size
):
    """Save a color-coded RGBA disagreement map.

    Pixels where pred != mask are colored by the PREDICTED class.
    Pixels where pred == mask (agreement) are fully transparent.
    Unlabeled pixels (ignore_index) are fully transparent.
    """
    h, w = pred.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    labeled = mask != ignore_index
    disagree = (pred != mask) & labeled

    for c_idx, c_name in enumerate(classes):
        # Pixels where model predicted this class but ground truth differs
        class_disagree = disagree & (pred == c_idx)
        if not class_disagree.any():
            continue
        r, g, b = get_class_color(c_idx, c_name, class_colors_map)
        rgba[class_disagree] = [r, g, b, 255]

    img = Image.fromarray(rgba, "RGBA")
    img.save(str(output_path))


def save_loss_heatmap(loss, mask, ignore_index, output_path):
    """Save a per-pixel loss heatmap as an RGBA PNG.

    Uses a blue -> yellow -> red colormap:
      0.0 (low loss)  -> blue  (0, 0, 255)
      0.5 (medium)    -> yellow (255, 255, 0)
      1.0 (high loss) -> red   (255, 0, 0)
    Unlabeled pixels are fully transparent.
    """
    h, w = loss.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    labeled = mask != ignore_index
    if not labeled.any():
        img = Image.fromarray(rgba, "RGBA")
        img.save(str(output_path))
        return

    # Normalize loss to 0-1 (clamp max at 5.0 to avoid outlier compression)
    loss_vals = loss[labeled].astype(np.float32)
    max_loss = min(float(loss_vals.max()), 5.0) if loss_vals.size > 0 else 1.0
    if max_loss <= 0:
        max_loss = 1.0
    norm = np.clip(loss.astype(np.float32) / max_loss, 0.0, 1.0)

    # Build blue->yellow->red LUT (256 entries)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            # blue -> yellow
            s = t * 2.0
            lut[i] = [int(255 * s), int(255 * s), int(255 * (1 - s))]
        else:
            # yellow -> red
            s = (t - 0.5) * 2.0
            lut[i] = [255, int(255 * (1 - s)), 0]

    indices = (norm * 255).astype(np.uint8)
    rgba[..., 0] = lut[indices, 0]
    rgba[..., 1] = lut[indices, 1]
    rgba[..., 2] = lut[indices, 2]
    rgba[labeled, 3] = 255  # opaque for labeled pixels only

    img = Image.fromarray(rgba, "RGBA")
    img.save(str(output_path))


def save_tile_image(img_path, output_path):
    """Save a displayable RGB PNG of the training tile."""
    suffix = img_path.suffix.lower()
    if suffix == ".raw":
        # Raw float32 N-channel - load and convert to RGB
        with open(img_path, "rb") as f:
            header = np.frombuffer(f.read(12), dtype=np.int32)
            h, w, c = int(header[0]), int(header[1]), int(header[2])
            data = np.frombuffer(f.read(), dtype=np.float32)
        arr = data.reshape(h, w, c)
        # Take first 3 channels or single channel
        if c >= 3:
            arr = arr[:, :, :3]
        elif c == 1:
            arr = np.repeat(arr, 3, axis=2)
        else:
            arr = np.concatenate(
                [arr, np.zeros((h, w, 3 - c), dtype=np.float32)], axis=2
            )
        # Normalize to 0-255
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            arr = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
    elif suffix in (".tif", ".tiff"):
        try:
            import tifffile

            arr = tifffile.imread(str(img_path))
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
                    arr = arr.transpose(1, 2, 0)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    arr = arr[:, :, :3]
                elif arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=2)
                vmin, vmax = arr.min(), arr.max()
                if vmax > vmin:
                    arr = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
                img = Image.fromarray(arr, "RGB")
            else:
                img = Image.fromarray(arr)
                img = img.convert("RGB")
        except ImportError:
            img = Image.open(img_path).convert("RGB")
    else:
        img = Image.open(img_path).convert("RGB")
    img.save(str(output_path))


# Set up cancellation bridge
cancel_flag = threading.Event()


def watch_cancel():
    while not cancel_flag.is_set():
        if task.cancel_requested:
            cancel_flag.set()
            logger.info("Evaluation cancellation requested")
            break
        import time

        time.sleep(0.5)


cancel_watcher = threading.Thread(target=watch_cancel, daemon=True)
cancel_watcher.start()

try:
    # Load tile manifest
    manifest_path = Path(data_path) / "tile_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"tile_manifest.json not found in {data_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    patches_meta = manifest["patches"]
    patch_size = manifest.get("patch_size", 512)
    manifest_downsample = manifest.get("downsample", 1.0)

    logger.info(
        "Loaded manifest: %d tiles, patch_size=%d", len(patches_meta), patch_size
    )

    # Build filename -> manifest entry lookup (by both full name and stem,
    # because the global split may update val entries to .raw while the
    # evaluate script finds .tiff files, or vice versa)
    meta_by_filename = {}
    meta_by_stem = {}
    for entry in patches_meta:
        meta_by_filename[entry["filename"]] = entry
        stem = entry["filename"].rsplit(".", 1)[0]
        meta_by_stem[stem] = entry

    # Load model metadata to get correct architecture config
    model_dir = Path(model_path)
    model_metadata_path = model_dir / "metadata.json"
    model_arch = dict(architecture)  # start with what was passed
    if model_metadata_path.exists():
        with open(model_metadata_path) as f:
            model_metadata = json.load(f)
        # Use the saved architecture config (includes use_batchrenorm, etc.)
        saved_arch = model_metadata.get("architecture", {})
        model_arch.update(saved_arch)
        logger.info("Loaded model metadata from %s", model_metadata_path)

    training_service = TrainingService(gpu_manager=gpu_manager)
    num_channels = input_config.get("num_channels", 3)
    context_scale = model_arch.get("context_scale", 1)
    if context_scale > 1:
        num_channels = num_channels * 2
    num_classes = len(classes)
    model_type = model_arch.get("type", "unet")

    # Use the same model creation logic as training
    model = training_service._create_model(
        model_type=model_type,
        architecture=model_arch,
        num_channels=num_channels,
        num_classes=num_classes,
    )

    # Load trained weights
    pt_path = model_dir / "model.pt"
    if pt_path.exists():
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

        # Auto-detect BatchRenorm from state dict keys (rmax/dmax are
        # unique to BatchRenorm2d). More robust than metadata flag which
        # may be lost when Java overwrites metadata.json.
        has_batchrenorm = any(
            k.endswith(".rmax") or k.endswith(".dmax") for k in state_dict
        )
        if has_batchrenorm:
            replace_bn_with_batchrenorm(model)
            logger.info("Auto-detected BatchRenorm from state dict keys")

        model.load_state_dict(state_dict)
        logger.info("Loaded model weights from %s", pt_path)
    else:
        raise FileNotFoundError(f"No model.pt found at {model_dir}")

    device = torch.device(training_service.device)
    model = model.to(device)
    model.eval()

    # Faithful preview: run the SAME runtime as production (ONNX) when available
    # so the per-tile maps match what applying the classifier to the image
    # produces. Production uses ONNX (model_static*/model.onnx); this per-tile
    # eval historically used model.pt (PyTorch eager), which can diverge from
    # ONNX even on byte-identical input. We use the dynamic model.onnx here: it
    # accepts the exported (padded, multiple-of-32) tile size directly and is
    # numerically equivalent to the static variants production prefers, without
    # the static-shape match constraint. Falls back to the PyTorch model when
    # ONNX is unavailable (e.g. MuViT, or a failed export).
    onnx_session = None
    onnx_input_name = None
    onnx_path = model_dir / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if str(training_service.device) == "cuda"
                else ["CPUExecutionProvider"]
            )
            onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            onnx_input_name = onnx_session.get_inputs()[0].name
            logger.info(
                "Per-tile eval using ONNX runtime (%s) for production parity",
                onnx_path,
            )
        except Exception as e:
            logger.warning(
                "ONNX runtime unavailable for eval (%s); "
                "falling back to model.pt (PyTorch): %s",
                onnx_path,
                e,
            )
            onnx_session = None
    else:
        logger.info("No model.onnx found; per-tile eval uses model.pt (PyTorch eager)")

    # Determine ignore index from config
    config_path = Path(data_path) / "config.json"
    ignore_index = 255
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        ignore_index = config.get("unlabeled_index", 255)

    # class_colors: optional dict of class_name -> packed RGB int from Java
    class_colors_map = globals().get("class_colors", None)
    if class_colors_map:
        logger.info("Received class colors for %d classes", len(class_colors_map))

    # Create disagreement output directory in the model directory (persistent).
    # Use split-scoped subdirectories so train/val tiles with matching stems
    # cannot overwrite each other's PNGs.
    disagree_dir = model_dir / "disagreement"
    disagree_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Disagreement maps will be saved under %s", disagree_dir)

    # Create datasets for train and val splits (no augmentation)
    data_root = Path(data_path)
    results = []
    total_tiles = 0

    for split_name, split_dir in [("train", "train"), ("val", "validation")]:
        images_dir = data_root / split_dir / "images"
        masks_dir = data_root / split_dir / "masks"
        context_dir = None
        if context_scale > 1:
            ctx = data_root / split_dir / "context"
            if ctx.exists():
                context_dir = str(ctx)

        if not images_dir.exists():
            continue

        dataset = SegmentationDataset(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            input_config=input_config,
            augment=False,
            context_dir=context_dir,
        )

        if len(dataset) == 0:
            continue

        # Per-split subdirectory to prevent stem collisions between train/val.
        split_disagree_dir = disagree_dir / split_name
        split_disagree_dir.mkdir(parents=True, exist_ok=True)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0
        )

        total_tiles += len(dataset)
        tile_idx = 0

        for batch_images, batch_masks in loader:
            if cancel_flag.is_set():
                break

            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)
            batch_size = batch_images.shape[0]

            with torch.no_grad():
                if onnx_session is not None:
                    # Production parity: same ONNX runtime as applying the model.
                    onnx_out = onnx_session.run(
                        None,
                        {onnx_input_name: batch_images.detach().cpu().numpy()},
                    )[0]
                    logits = torch.from_numpy(onnx_out).to(device)
                else:
                    logits = model(batch_images)
                # Per-tile cross-entropy loss
                losses = F.cross_entropy(
                    logits, batch_masks, ignore_index=ignore_index, reduction="none"
                )

            preds = logits.argmax(dim=1)

            # Softmax probabilities for confidence maps (used by annotation adjustment)
            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values  # [B, H, W], range 0.0-1.0

            for i in range(batch_size):
                if cancel_flag.is_set():
                    break

                file_idx = tile_idx + i
                if file_idx >= len(dataset.image_files):
                    break

                filename = dataset.image_files[file_idx].name
                stem = dataset.image_files[file_idx].stem

                # Try exact filename first, then stem-based fallback.
                # The global split may change val entry filenames from .tiff
                # to .raw (via resolveActualFilename), while train entries
                # keep the original .tiff name.
                meta = meta_by_filename.get(filename)
                if meta is None:
                    meta = meta_by_stem.get(stem, {})

                mask_i = batch_masks[i]
                pred_i = preds[i]
                loss_i = losses[i]

                # Compute loss only on labeled pixels
                labeled_mask = mask_i != ignore_index
                labeled_count = labeled_mask.sum().item()

                if labeled_count == 0:
                    continue

                tile_loss = loss_i[labeled_mask].mean().item()

                # Disagreement: fraction of labeled pixels where pred != mask
                disagree_mask_flat = pred_i[labeled_mask] != mask_i[labeled_mask]
                disagreements = disagree_mask_flat.sum().item()
                disagreement_pct = disagreements / labeled_count

                # Confidence histogram of disagree pixels, 20 bins covering
                # [0.0, 1.0) at width 0.05 (last bin includes 1.0). The Java
                # dialog uses this to render a "Disagree px @ >= confidence"
                # column that updates live as the user moves the confidence
                # slider, without re-running evaluation. Storing the raw
                # histogram (instead of a single count at a fixed threshold)
                # costs 20 ints/tile -- trivial RAM/disk vs. the prediction
                # and confidence maps already saved per tile.
                conf_flat = confidence[i][labeled_mask]
                disagree_conf = conf_flat[disagree_mask_flat]
                if disagree_conf.numel() > 0:
                    # bin = min(19, floor(conf * 20)); clamp guards conf == 1.0.
                    bin_idx = (disagree_conf * 20).long().clamp_(max=19)
                    disagree_hist = torch.bincount(bin_idx, minlength=20).cpu().tolist()
                else:
                    disagree_hist = [0] * 20

                # Per-class IoU
                per_class_iou = {}
                for c_idx, c_name in enumerate(classes):
                    pred_c = pred_i == c_idx
                    mask_c = mask_i == c_idx
                    intersection = (pred_c & mask_c).sum().item()
                    union = (pred_c | mask_c).sum().item()
                    if union > 0:
                        per_class_iou[c_name] = intersection / union
                    else:
                        per_class_iou[c_name] = float("nan")

                # Mean IoU (only over classes present in this tile)
                valid_ious = [
                    v for v in per_class_iou.values() if not (v != v)
                ]  # filter NaN
                mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

                # Top GT->Pred confusion pairs, computed over labeled pixels only.
                # Built from a NxN counts matrix (rows=GT, cols=Pred) via a single
                # bincount on (gt*num_classes + pred); off-diagonal entries are
                # the misclassifications. Reported with the source GT class's
                # total pixel count so the dialog can show "k% of GT_class".
                top_confusions = []
                try:
                    num_classes = len(classes)
                    gt_lab = mask_i[labeled_mask].long()
                    pr_lab = pred_i[labeled_mask].long()
                    combined = gt_lab * num_classes + pr_lab
                    conf_counts = torch.bincount(
                        combined, minlength=num_classes * num_classes
                    ).view(num_classes, num_classes)
                    gt_totals = conf_counts.sum(dim=1)
                    pairs = []
                    for gt_idx in range(num_classes):
                        for pred_idx in range(num_classes):
                            if gt_idx == pred_idx:
                                continue
                            pixels = int(conf_counts[gt_idx, pred_idx].item())
                            if pixels == 0:
                                continue
                            pairs.append(
                                {
                                    "gt": classes[gt_idx],
                                    "pred": classes[pred_idx],
                                    "pixels": pixels,
                                    "gt_total": int(gt_totals[gt_idx].item()),
                                }
                            )
                    pairs.sort(key=lambda p: p["pixels"], reverse=True)
                    # Keep the full set of non-zero off-diagonal pairs so the
                    # Confusion Matrix tab in TrainingAreaIssuesDialog has
                    # pixel-exact aggregation. JSON payload grows by ~50 KB
                    # for a 1000-tile, 7-class session -- negligible.
                    top_confusions = pairs
                except Exception as conf_err:
                    logger.warning(
                        "confusion-pair computation failed for [%s] %s: %s",
                        split_name,
                        stem,
                        conf_err,
                    )

                # Save disagreement map, loss heatmap, and tile image.
                # Each save is wrapped independently so a failure in one step
                # does not null out the others (e.g. a bad .raw header should
                # not suppress the heatmap PNG).
                disagree_path = None
                loss_heatmap_path = None
                tile_img_path = None
                prediction_map_path = None
                confidence_map_path = None
                gt_mask_path = None

                try:
                    pred_np = pred_i.cpu().numpy()
                    mask_np = mask_i.cpu().numpy()
                    loss_np = loss_i.cpu().numpy()
                    conf_np = confidence[i].cpu().numpy()
                except Exception as convert_err:
                    logger.warning(
                        "Failed to convert tensors for [%s] %s: %s",
                        split_name,
                        stem,
                        convert_err,
                    )
                    pred_np = mask_np = loss_np = conf_np = None

                if pred_np is not None:
                    disagree_file = split_disagree_dir / f"{stem}_disagree.png"
                    try:
                        save_disagreement_map(
                            pred_np,
                            mask_np,
                            ignore_index,
                            classes,
                            class_colors_map,
                            disagree_file,
                            patch_size,
                        )
                        disagree_path = str(disagree_file)
                    except Exception as save_err:
                        logger.warning(
                            "save_disagreement_map failed for [%s] %s: %s",
                            split_name,
                            stem,
                            save_err,
                        )

                    loss_file = split_disagree_dir / f"{stem}_loss.png"
                    try:
                        save_loss_heatmap(loss_np, mask_np, ignore_index, loss_file)
                        loss_heatmap_path = str(loss_file)
                    except Exception as save_err:
                        logger.warning(
                            "save_loss_heatmap failed for [%s] %s: %s",
                            split_name,
                            stem,
                            save_err,
                        )

                    # Prediction map: argmax class indices as uint8 grayscale PNG.
                    # Used by annotation adjustment to know what the model predicted.
                    pred_file = split_disagree_dir / f"{stem}_pred.png"
                    try:
                        Image.fromarray(pred_np.astype(np.uint8)).save(str(pred_file))
                        prediction_map_path = str(pred_file)
                    except Exception as save_err:
                        logger.warning(
                            "save prediction map failed for [%s] %s: %s",
                            split_name,
                            stem,
                            save_err,
                        )

                    # Confidence map: max softmax probability scaled to 0-255.
                    # Used by annotation adjustment threshold.
                    conf_file = split_disagree_dir / f"{stem}_conf.png"
                    try:
                        conf_uint8 = (conf_np * 255).astype(np.uint8)
                        Image.fromarray(conf_uint8).save(str(conf_file))
                        confidence_map_path = str(conf_file)
                    except Exception as save_err:
                        logger.warning(
                            "save confidence map failed for [%s] %s: %s",
                            split_name,
                            stem,
                            save_err,
                        )

                    # Ground truth mask: saved alongside prediction/confidence so
                    # annotation adjustment can compare without needing the
                    # original training data directory.
                    gt_file = split_disagree_dir / f"{stem}_gt.png"
                    try:
                        Image.fromarray(mask_np.astype(np.uint8)).save(str(gt_file))
                        gt_mask_path = str(gt_file)
                    except Exception as save_err:
                        logger.warning(
                            "save ground truth mask failed for [%s] %s: %s",
                            split_name,
                            stem,
                            save_err,
                        )

                tile_file = split_disagree_dir / f"{stem}_tile.png"
                try:
                    img_path = dataset.image_files[file_idx]
                    save_tile_image(img_path, tile_file)
                    tile_img_path = str(tile_file)
                except Exception as save_err:
                    logger.warning(
                        "save_tile_image failed for [%s] %s: %s",
                        split_name,
                        stem,
                        save_err,
                    )

                results.append(
                    {
                        "filename": filename,
                        "split": split_name,
                        "loss": round(tile_loss, 4),
                        "disagreement_pct": round(disagreement_pct, 4),
                        "disagreement_pixels": int(disagreements),
                        "disagreement_conf_histogram": disagree_hist,
                        "per_class_iou": {
                            k: round(v, 4) if v == v else None
                            for k, v in per_class_iou.items()
                        },
                        "mean_iou": round(mean_iou, 4),
                        "x": meta.get("x", 0),
                        "y": meta.get("y", 0),
                        "source_image": meta.get("source_image", ""),
                        "source_image_id": meta.get("source_image_id", ""),
                        "disagreement_image": disagree_path,
                        "loss_heatmap": loss_heatmap_path,
                        "tile_image": tile_img_path,
                        "prediction_map": prediction_map_path,
                        "confidence_map": confidence_map_path,
                        "ground_truth_mask": gt_mask_path,
                        "top_confusions": top_confusions,
                    }
                )

            tile_idx += batch_size

            # Progress update
            evaluated_so_far = sum(1 for r in results)
            task.update(
                message=json.dumps(
                    {
                        "status": "evaluating",
                        "current_tile": evaluated_so_far,
                        "total_tiles": total_tiles,
                    }
                ),
                current=evaluated_so_far,
                maximum=total_tiles,
            )

    # Sort by loss descending
    results.sort(key=lambda r: r["loss"], reverse=True)

    logger.info("Evaluation complete: %d/%d tiles evaluated", len(results), total_tiles)

    task.outputs["results"] = json.dumps(results)
    task.outputs["total_tiles"] = total_tiles
    task.outputs["tiles_evaluated"] = len(results)

except Exception as e:
    logger.error("Evaluation failed: %s", e)
    raise
finally:
    cancel_flag.set()
