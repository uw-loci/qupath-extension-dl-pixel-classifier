"""
Training task with progress reporting via Appose events.

Inputs:
    model_type: str
    architecture: dict
    input_config: dict
    training_params: dict
    classes: list of str
    data_path: str
    pause_signal_path: str (optional) - file path used as pause signal
    checkpoint_path: str (optional) - checkpoint to resume from
    start_epoch: int (optional) - epoch to resume from

Outputs:
    status: str ("completed" or "paused")
    model_path: str
    final_loss: float
    final_accuracy: float
    best_epoch: int
    best_mean_iou: float
    epochs_trained: int
    checkpoint_path: str (when paused)
    last_epoch: int (when paused)
    total_epochs: int
"""
import json
import os
import threading
import time
import logging

logger = logging.getLogger("dlclassifier.appose.train")

if inference_service is None:
    raise RuntimeError("Services not initialized: " + globals().get("init_error", "unknown"))

# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: model_type, architecture, input_config, training_params, classes, data_path
# Optional inputs (use try/except NameError pattern):

try:
    pause_signal_path
except NameError:
    pause_signal_path = None

try:
    checkpoint_path
except NameError:
    checkpoint_path = None

try:
    start_epoch
except NameError:
    start_epoch = 0

try:
    pretrained_model_path
except NameError:
    pretrained_model_path = None

try:
    model_output_dir
except NameError:
    model_output_dir = None

# Import training service (heavier import, done here rather than init)
from dlclassifier_server.services.training_service import TrainingService
import dlclassifier_server.services.training_service as _tsm
import numpy as _np

# Fix: np.frombuffer returns read-only arrays; ensure _load_patch always
# returns writable arrays so normalization can modify in-place.
# This patches the installed package which may be stale (Appose pip cache).
_orig_lp = _tsm.SegmentationDataset._load_patch
@staticmethod
def _writable_load_patch(img_path):
    arr = _orig_lp(img_path)
    return arr.copy() if not arr.flags.writeable else arr
_tsm.SegmentationDataset._load_patch = _writable_load_patch


# Safety net: resize context tiles if they don't match detail tile dimensions.
# Edge tiles should be skipped at export, but stale pip packages may still
# have mismatched tiles from older exports. This patches the installed package.
from PIL import Image as _PILImage

_orig_getitem = _tsm.SegmentationDataset.__getitem__
_ctx_resize_warned = [False]

def _safe_getitem(self, idx):
    """__getitem__ with context tile resize for edge-case size mismatch."""
    img_path = self.image_files[idx]
    img_array = self._load_patch(img_path)
    if img_array.ndim == 2:
        img_array = img_array[..., _np.newaxis]

    if self.context_dir is not None:
        ctx_path = self.context_dir / img_path.name
        if ctx_path.exists():
            ctx_array = self._load_patch(ctx_path)
            if ctx_array.ndim == 2:
                ctx_array = ctx_array[..., _np.newaxis]
            # Resize context tile if spatial dimensions don't match detail tile
            if ctx_array.shape[0] != img_array.shape[0] or ctx_array.shape[1] != img_array.shape[1]:
                if not _ctx_resize_warned[0]:
                    logger.warning("Context tile %s has shape %s but detail is %s -- "
                                   "resizing (edge tile from old export?)",
                                   ctx_path.name, ctx_array.shape, img_array.shape)
                    _ctx_resize_warned[0] = True
                h, w = img_array.shape[:2]
                resized_channels = []
                for c in range(ctx_array.shape[2]):
                    ch = _PILImage.fromarray(ctx_array[:, :, c])
                    ch = ch.resize((w, h), _PILImage.BILINEAR)
                    resized_channels.append(_np.array(ch))
                ctx_array = _np.stack(resized_channels, axis=2)
            img_array = _np.concatenate([img_array, ctx_array], axis=2)
        else:
            img_array = _np.concatenate([img_array, img_array], axis=2)

    img_array = self._normalize(img_array)

    mask_name = img_path.stem + ".png"
    mask_path = self.masks_dir / mask_name
    if mask_path.exists():
        mask = _PILImage.open(mask_path)
        mask_array = _np.array(mask, dtype=_np.int64)
    else:
        mask_array = _np.zeros(img_array.shape[:2], dtype=_np.int64)

    if self.transform is not None:
        transformed = self.transform(image=img_array, mask=mask_array)
        img_array = transformed["image"]
        mask_array = transformed["mask"]

    if img_array.ndim == 2:
        img_array = img_array[..., _np.newaxis]
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1).astype(_np.float32))
    mask_tensor = torch.from_numpy(mask_array.astype(_np.int64))
    return img_tensor, mask_tensor

_tsm.SegmentationDataset.__getitem__ = _safe_getitem

training_service = TrainingService(gpu_manager=gpu_manager)

# Redirect model and checkpoint saving to project directory when specified.
# This calls the original pip package methods (ONNX export, metadata writing, etc.)
# then moves all output files to the project directory.
if model_output_dir:
    import shutil as _shutil
    from pathlib import Path as _Path

    _orig_save_model = training_service._save_model

    def _redirected_save_model(*args, **kwargs):
        orig_path = _orig_save_model(*args, **kwargs)
        dst = _Path(model_output_dir)
        dst.mkdir(parents=True, exist_ok=True)
        src = _Path(orig_path)
        for item in src.iterdir():
            _shutil.move(str(item), str(dst / item.name))
        _shutil.rmtree(str(src), ignore_errors=True)
        logger.info("Moved model files to project: %s", dst)
        return str(dst)

    training_service._save_model = _redirected_save_model

    _orig_save_ckpt = training_service._save_checkpoint

    def _redirected_save_checkpoint(*args, **kwargs):
        orig_path = _orig_save_ckpt(*args, **kwargs)
        dst_dir = _Path(model_output_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = _Path(orig_path)
        dst = dst_dir / src.name
        _shutil.move(str(src), str(dst))
        logger.info("Moved checkpoint to project: %s", dst)
        return str(dst)

    training_service._save_checkpoint = _redirected_save_checkpoint

    _orig_save_best = training_service._save_best_in_progress

    def _redirected_save_best(*args, **kwargs):
        orig_path = _orig_save_best(*args, **kwargs)
        dst_dir = _Path(model_output_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = _Path(orig_path)
        dst = dst_dir / src.name
        _shutil.copy2(str(src), str(dst))
        logger.info("Copied best-in-progress to project: %s", dst)
        return str(dst)

    training_service._save_best_in_progress = _redirected_save_best

# Log device and training configuration for diagnostics
import torch
device_name = training_service.device
cuda_available = torch.cuda.is_available()
device_info = "CPU"
logger.info("Training device: %s (CUDA available: %s)", device_name, cuda_available)
if device_name == "cuda":
    device_info = torch.cuda.get_device_name(0)
    logger.info("GPU: %s", device_info)
elif device_name == "cpu":
    logger.warning("Training on CPU -- this will be very slow. Check pixi.toml CUDA configuration.")
logger.info("Model: %s, backbone: %s", model_type, architecture.get("backbone", "unknown"))
logger.info("Classes: %s", classes)
logger.info("Epochs: %s, batch_size: %s, lr: %s",
    training_params.get("epochs"), training_params.get("batch_size"), training_params.get("learning_rate"))
logger.info("Data path: %s", data_path)
if checkpoint_path:
    logger.info("Resuming from checkpoint: %s (start_epoch=%d)", checkpoint_path, start_epoch)
if pretrained_model_path:
    logger.info("Loading pretrained weights from: %s", pretrained_model_path)

# Send pre-training status update so the Java UI can show device info
total_epochs = training_params.get("epochs", 50)
task.update(
    message=json.dumps({
        "status": "initializing",
        "device": device_name,
        "device_info": device_info,
        "cuda_available": cuda_available,
        "epoch": 0,
        "total_epochs": total_epochs,
    }),
    current=0,
    maximum=total_epochs
)


def setup_callback(phase, data=None):
    """Forward setup phase updates to Appose task events."""
    import math
    total_epochs = training_params.get("epochs", 50)
    msg = {
        "status": "setup",
        "setup_phase": phase,
        "epoch": 0,
        "total_epochs": total_epochs,
    }
    if data:
        # Sanitize floats to prevent NaN/Inf breaking JSON protocol
        safe_data = {}
        for k, v in data.items():
            if isinstance(v, float) and not math.isfinite(v):
                safe_data[k] = 0.0
            else:
                safe_data[k] = v
        msg["config"] = safe_data
        # For batch-level progress, use the epoch from data for progress bar
        if "epoch" in data:
            msg["epoch"] = data["epoch"]
    task.update(
        message=json.dumps(msg),
        current=msg["epoch"],
        maximum=total_epochs
    )


def progress_callback(epoch, train_loss, val_loss, accuracy,
                       per_class_iou, per_class_loss, mean_iou):
    """Forward training progress to Appose task events."""
    import math
    # Guard against NaN/Inf: Python json.dumps serializes float('nan') as bare
    # NaN token which is NOT valid JSON. Gson's JsonParser rejects it, silently
    # dropping ALL progress updates. See docs/APPOSE_DEV_GUIDE.md.
    def _safe(v):
        return v if isinstance(v, (int, str)) or (isinstance(v, float) and math.isfinite(v)) else 0.0

    def _safe_dict(d):
        return {k: _safe(v) for k, v in d.items()} if d else {}

    total_epochs = training_params.get("epochs", 50)
    task.update(
        message=json.dumps({
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": _safe(train_loss),
            "val_loss": _safe(val_loss),
            "accuracy": _safe(accuracy),
            "mean_iou": _safe(mean_iou),
            "per_class_iou": _safe_dict(per_class_iou),
            "per_class_loss": _safe_dict(per_class_loss),
        }),
        current=epoch,
        maximum=total_epochs
    )


# Set up cancellation bridge: Appose cancel -> threading.Event
cancel_flag = threading.Event()


def watch_cancel():
    """Poll for Appose cancellation request and set the cancel flag."""
    while not cancel_flag.is_set():
        if task.cancel_requested:
            cancel_flag.set()
            logger.info("Training cancellation requested via Appose")
            break
        time.sleep(0.5)


cancel_watcher = threading.Thread(target=watch_cancel, daemon=True)
cancel_watcher.start()

# Set up pause bridge: file signal -> threading.Event
pause_flag = threading.Event()


def watch_pause():
    """Poll for pause signal file and set the pause flag."""
    if not pause_signal_path:
        return
    while not pause_flag.is_set() and not cancel_flag.is_set():
        if os.path.exists(pause_signal_path):
            pause_flag.set()
            logger.info("Pause requested via signal file")
            try:
                os.remove(pause_signal_path)
            except Exception:
                pass
            break
        time.sleep(0.5)


pause_watcher = threading.Thread(target=watch_pause, daemon=True)
pause_watcher.start()

# Extract frozen layers from architecture dict (Java puts them there)
frozen_layers = architecture.get("frozen_layers", None)

# Pretrained weight loading is inlined here rather than passed to
# training_service.train(), because the installed pip package may be stale
# (Appose caches pip installs and doesn't reinstall on git push).
# This script is loaded from JAR resources every run, so it's always current.
_pretrained_patched_class = None
_pretrained_orig_frozen = None

if pretrained_model_path and not checkpoint_path:
    _pretrained_applied = [False]

    def _load_pretrained_weights(model):
        """Load weights from a previously trained model onto the new model."""
        if _pretrained_applied[0]:
            return model
        _pretrained_applied[0] = True
        try:
            logger.info("Loading pretrained weights from: %s", pretrained_model_path)
            saved = torch.load(pretrained_model_path, map_location='cpu',
                               weights_only=True)

            # Handle both bare state_dict and checkpoint format
            if isinstance(saved, dict) and "model_state_dict" in saved:
                state_dict = saved["model_state_dict"]
            else:
                state_dict = saved

            # Detect MAE checkpoint and strip "mae." prefix so that
            # encoder keys (mae.encoder.* -> encoder.*) match the
            # MuViTSegmentation model's state_dict.
            mae_prefix = "mae."
            has_mae_keys = any(k.startswith(mae_prefix) for k in state_dict)
            if has_mae_keys:
                logger.info("Detected MAE checkpoint -- stripping 'mae.' "
                            "prefix for encoder weight transfer.")
                state_dict = {
                    (k[len(mae_prefix):] if k.startswith(mae_prefix) else k): v
                    for k, v in state_dict.items()
                }

            # Detect shape mismatches (e.g. different class count) and skip those keys
            model_state = model.state_dict()
            matched = {}
            mismatched = []
            for key in state_dict:
                if key in model_state:
                    if state_dict[key].shape == model_state[key].shape:
                        matched[key] = state_dict[key]
                    else:
                        mismatched.append(key)
                        logger.warning("  Shape mismatch for '%s': "
                                       "pretrained=%s vs model=%s -- skipping",
                                       key, list(state_dict[key].shape),
                                       list(model_state[key].shape))

            model.load_state_dict(matched, strict=False)
            logger.info("Loaded %d/%d weight tensors from pretrained model",
                        len(matched), len(model_state))
            if mismatched:
                logger.info("  Skipped %d mismatched keys "
                            "(likely segmentation head due to class count change)",
                            len(mismatched))
            # Warn if very few weights matched (architecture mismatch)
            if len(matched) == 0 and len(model_state) > 0:
                logger.warning("NO pretrained weights matched! "
                               "Architecture mismatch between encoder and model. "
                               "Training will start from random initialization.")
            elif len(matched) < len(model_state) * 0.5:
                logger.warning("Only %d%% of model weights loaded from pretrained -- "
                               "check architecture settings match the encoder.",
                               int(100 * len(matched) / len(model_state)))
        except Exception as e:
            logger.warning("Failed to load pretrained weights: %s -- "
                           "training will start from scratch", e)
        return model

    # Monkey-patch _create_model on the instance so weights are loaded
    # right after model creation (before to(device) in _run_training)
    _orig_create = training_service._create_model
    def _create_with_pretrained(*args, **kwargs):
        return _load_pretrained_weights(_orig_create(*args, **kwargs))
    training_service._create_model = _create_with_pretrained

    # Also patch the frozen-layers model creation path (different code path
    # in _run_training that bypasses _create_model)
    if frozen_layers:
        try:
            from dlclassifier_server.services.pretrained_models import \
                PretrainedModelsService as _PMS
            _pretrained_orig_frozen = _PMS.create_model_with_frozen_layers
            _pretrained_patched_class = _PMS
            def _frozen_with_pretrained(self, *args, **kwargs):
                return _load_pretrained_weights(
                    _pretrained_orig_frozen(self, *args, **kwargs))
            _PMS.create_model_with_frozen_layers = _frozen_with_pretrained
        except Exception as e:
            logger.warning("Could not patch frozen-layers path for "
                           "pretrained weights: %s", e)

try:
    result = training_service.train(
        model_type=model_type,
        architecture=architecture,
        input_config=input_config,
        training_params=training_params,
        classes=classes,
        data_path=data_path,
        progress_callback=progress_callback,
        cancel_flag=cancel_flag,
        frozen_layers=frozen_layers,
        pause_flag=pause_flag,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch,
        setup_callback=setup_callback
    )
except Exception as e:
    logger.error("Training failed: %s", e)
    raise
finally:
    # Signal watchers to stop so daemon threads terminate cleanly
    cancel_flag.set()
    pause_flag.set()
    # Restore class-level patches (Python process persists across Appose tasks)
    if _pretrained_patched_class is not None and _pretrained_orig_frozen is not None:
        _pretrained_patched_class.create_model_with_frozen_layers = _pretrained_orig_frozen

status = result.get("status", "completed")
task.outputs["status"] = status
task.outputs["model_path"] = result.get("model_path", "")
task.outputs["last_model_path"] = result.get("last_model_path", "")
task.outputs["final_loss"] = result.get("final_loss", 0.0)
task.outputs["final_accuracy"] = result.get("final_accuracy", 0.0)
task.outputs["best_epoch"] = result.get("best_epoch", 0)
task.outputs["best_mean_iou"] = result.get("best_mean_iou", 0.0)
task.outputs["epochs_trained"] = result.get("epochs_trained", 0)
task.outputs["checkpoint_path"] = result.get("checkpoint_path", "")
task.outputs["last_epoch"] = result.get("epoch", 0)
task.outputs["total_epochs"] = result.get("total_epochs", total_epochs)
