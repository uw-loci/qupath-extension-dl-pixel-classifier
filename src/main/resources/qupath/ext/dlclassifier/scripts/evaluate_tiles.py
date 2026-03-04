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
    raise RuntimeError("Services not initialized: " + globals().get("init_error", "unknown"))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

from dlclassifier_server.services.training_service import (
    TrainingService, SegmentationDataset
)
from dlclassifier_server.utils.batchrenorm import replace_bn_with_batchrenorm


# Default class colors (used when class_colors input is not provided)
DEFAULT_COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 128, 255),    # light blue
    (255, 0, 128),    # pink
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


def save_disagreement_map(pred, mask, ignore_index, classes, class_colors_map,
                          output_path, patch_size):
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

    img = Image.fromarray(rgba, 'RGBA')
    img.save(str(output_path))


def save_tile_image(img_path, output_path):
    """Save a displayable RGB PNG of the training tile."""
    suffix = img_path.suffix.lower()
    if suffix == '.raw':
        # Raw float32 N-channel - load and convert to RGB
        with open(img_path, 'rb') as f:
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
            arr = np.concatenate([arr, np.zeros((h, w, 3 - c), dtype=np.float32)], axis=2)
        # Normalize to 0-255
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            arr = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        img = Image.fromarray(arr, 'RGB')
    elif suffix in ('.tif', '.tiff'):
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
                img = Image.fromarray(arr, 'RGB')
            else:
                img = Image.fromarray(arr)
                img = img.convert('RGB')
        except ImportError:
            img = Image.open(img_path).convert('RGB')
    else:
        img = Image.open(img_path).convert('RGB')
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

    logger.info("Loaded manifest: %d tiles, patch_size=%d", len(patches_meta), patch_size)

    # Build filename -> manifest entry lookup
    meta_by_filename = {}
    for entry in patches_meta:
        meta_by_filename[entry["filename"]] = entry

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
        num_classes=num_classes
    )

    # Load trained weights
    pt_path = model_dir / "model.pt"
    if pt_path.exists():
        state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)

        # Auto-detect BatchRenorm from state dict keys (rmax/dmax are
        # unique to BatchRenorm2d). More robust than metadata flag which
        # may be lost when Java overwrites metadata.json.
        has_batchrenorm = any(
            k.endswith('.rmax') or k.endswith('.dmax')
            for k in state_dict)
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

    # Create disagreement output directory in the model directory (persistent)
    disagree_dir = model_dir / "disagreement"
    disagree_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Disagreement maps will be saved to %s", disagree_dir)

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
            context_dir=context_dir
        )

        if len(dataset) == 0:
            continue

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
                logits = model(batch_images)
                # Per-tile cross-entropy loss
                losses = F.cross_entropy(
                    logits, batch_masks,
                    ignore_index=ignore_index,
                    reduction='none'
                )

            preds = logits.argmax(dim=1)

            for i in range(batch_size):
                if cancel_flag.is_set():
                    break

                file_idx = tile_idx + i
                if file_idx >= len(dataset.image_files):
                    break

                filename = dataset.image_files[file_idx].name
                # Strip extension to match manifest (manifest uses .tiff)
                stem = dataset.image_files[file_idx].stem
                manifest_filename = stem + ".tiff"

                meta = meta_by_filename.get(manifest_filename, {})

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
                disagreements = (pred_i[labeled_mask] != mask_i[labeled_mask]).sum().item()
                disagreement_pct = disagreements / labeled_count

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
                        per_class_iou[c_name] = float('nan')

                # Mean IoU (only over classes present in this tile)
                valid_ious = [v for v in per_class_iou.values()
                              if not (v != v)]  # filter NaN
                mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

                # Save disagreement map and tile image
                disagree_path = None
                tile_img_path = None
                try:
                    pred_np = pred_i.cpu().numpy()
                    mask_np = mask_i.cpu().numpy()
                    disagree_file = disagree_dir / f"{stem}_disagree.png"
                    save_disagreement_map(
                        pred_np, mask_np, ignore_index, classes,
                        class_colors_map, disagree_file, patch_size
                    )
                    disagree_path = str(disagree_file)

                    # Save tile image (from original training data)
                    tile_file = disagree_dir / f"{stem}_tile.png"
                    img_path = dataset.image_files[file_idx]
                    save_tile_image(img_path, tile_file)
                    tile_img_path = str(tile_file)
                except Exception as save_err:
                    logger.warning("Failed to save disagreement map for %s: %s",
                                   stem, save_err)

                results.append({
                    "filename": manifest_filename,
                    "split": split_name,
                    "loss": round(tile_loss, 4),
                    "disagreement_pct": round(disagreement_pct, 4),
                    "per_class_iou": {k: round(v, 4) if v == v else None
                                      for k, v in per_class_iou.items()},
                    "mean_iou": round(mean_iou, 4),
                    "x": meta.get("x", 0),
                    "y": meta.get("y", 0),
                    "source_image": meta.get("source_image", ""),
                    "source_image_id": meta.get("source_image_id", ""),
                    "disagreement_image": disagree_path,
                    "tile_image": tile_img_path,
                })

            tile_idx += batch_size

            # Progress update
            evaluated_so_far = sum(1 for r in results)
            task.update(
                message=json.dumps({
                    "status": "evaluating",
                    "current_tile": evaluated_so_far,
                    "total_tiles": total_tiles,
                }),
                current=evaluated_so_far,
                maximum=total_tiles
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
