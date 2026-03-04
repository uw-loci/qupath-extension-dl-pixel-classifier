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

from dlclassifier_server.services.training_service import (
    TrainingService, SegmentationDataset
)
from dlclassifier_server.utils.batchrenorm import replace_bn_with_batchrenorm

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

    # Replace BatchNorm with BatchRenorm if the model was trained with it
    if model_arch.get("use_batchrenorm", False):
        replace_bn_with_batchrenorm(model)
        logger.info("Applied BatchRenorm replacement for evaluation")

    # Load trained weights
    pt_path = model_dir / "model.pt"
    if pt_path.exists():
        state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
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
