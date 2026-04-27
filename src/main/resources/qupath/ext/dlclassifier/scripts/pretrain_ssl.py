"""
SSL pretraining task for SMP encoder backbones via Appose.

Supports SimCLR (contrastive) and BYOL (self-distillation) methods
for CNN encoder backbones (ResNet, EfficientNet, MobileNet, etc.).

Inputs:
    config: dict - SSL method and training configuration
    data_path: str - directory of image tiles for pretraining
    output_dir: str - directory to save pretrained encoder weights

Outputs:
    status: str ("completed" or "cancelled")
    encoder_path: str - path to saved encoder weights
    epochs_completed: int
    final_loss: float
    best_loss: float
"""
import json
import logging
import threading
import time

logger = logging.getLogger("dlclassifier.appose.pretrain_ssl")

if inference_service is None:
    raise RuntimeError(
        "Services not initialized: "
        + globals().get("init_error", "unknown"))

# Appose 0.10.0+: inputs are injected directly into script scope.
# Required: config, data_path, output_dir

import torch
from dlclassifier_server.services.ssl_pretraining import SSLPretrainingService

ssl_service = SSLPretrainingService(gpu_manager=gpu_manager)

# Log device info
device_name = str(ssl_service.device)
cuda_available = torch.cuda.is_available()
device_info = "CPU"
ssl_method = config.get("method", "simclr")
logger.info("SSL pretraining (%s) device: %s (CUDA: %s)",
            ssl_method, device_name, cuda_available)
if ssl_service._device_str == "cuda":
    device_info = torch.cuda.get_device_name(0)
    logger.info("GPU: %s", device_info)
elif ssl_service._device_str == "cpu":
    logger.warning("Pretraining on CPU -- this will be extremely slow.")

total_epochs = config.get("epochs", 100)

# Initial status update
task.update(
    message=json.dumps({
        "status": "initializing",
        "device": ssl_service._device_str,
        "device_info": device_info,
        "cuda_available": cuda_available,
        "ssl_method": ssl_method,
        "epoch": 0,
        "total_epochs": total_epochs,
    }),
    current=0,
    maximum=total_epochs
)

logger.info("Config: %s", json.dumps(config, indent=2))
logger.info("Data path: %s", data_path)
logger.info("Output dir: %s", output_dir)


def setup_callback(phase, data=None):
    """Forward setup phase updates to Appose."""
    msg = {
        "status": "setup",
        "setup_phase": phase,
        "epoch": 0,
        "total_epochs": total_epochs,
    }
    if data:
        msg["config"] = data
    task.update(message=json.dumps(msg), current=0, maximum=total_epochs)


def progress_callback(epoch, total, loss, lr,
                      elapsed_sec=0, remaining_sec=0,
                      epoch_sec=0, images_per_sec=0):
    """Forward training progress to Appose."""
    import math
    # Guard against NaN/Inf which produce invalid JSON that Gson cannot parse
    safe_loss = loss if math.isfinite(loss) else 0.0
    safe_lr = lr if math.isfinite(lr) else 0.0
    try:
        task.update(
            message=json.dumps({
                "epoch": epoch,
                "total_epochs": total,
                "train_loss": safe_loss,
                "val_loss": safe_loss,
                "accuracy": 0.0,
                "mean_iou": 0.0,
                "ssl_lr": safe_lr,
                "ssl_method": ssl_method,
                "device": ssl_service._device_str,
                "device_info": device_info,
                "elapsed_sec": round(elapsed_sec, 1),
                "remaining_sec": round(remaining_sec, 1),
                "epoch_sec": round(epoch_sec, 1),
                "images_per_sec": round(images_per_sec, 1),
            }),
            current=epoch,
            maximum=total
        )
    except Exception as e:
        logger.debug("Failed to send progress update: %s", e)


# Cancellation bridge
cancel_flag = threading.Event()


def watch_cancel():
    while not cancel_flag.is_set():
        if task.cancel_requested:
            cancel_flag.set()
            logger.info("SSL pretraining cancellation requested")
            break
        time.sleep(0.5)


cancel_watcher = threading.Thread(target=watch_cancel, daemon=True)
cancel_watcher.start()

try:
    result = ssl_service.pretrain(
        config=config,
        data_path=data_path,
        output_dir=output_dir,
        progress_callback=progress_callback,
        setup_callback=setup_callback,
        cancel_flag=cancel_flag,
    )
except Exception as e:
    logger.error("SSL pretraining failed: %s", e)
    raise
finally:
    cancel_flag.set()

task.outputs["status"] = result.get("status", "completed")
task.outputs["encoder_path"] = result.get("encoder_path", "")
task.outputs["epochs_completed"] = result.get("epochs_completed", 0)
task.outputs["final_loss"] = result.get("final_loss", 0.0)
task.outputs["best_loss"] = result.get("best_loss", 0.0)
