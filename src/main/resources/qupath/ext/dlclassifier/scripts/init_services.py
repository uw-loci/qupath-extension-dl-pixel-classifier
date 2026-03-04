"""
Appose worker initialization script.

Called once via pythonService.init() when the worker process starts.
Sets up persistent globals that remain available across all task() calls.

CRITICAL: All output must go to sys.stderr, NOT sys.stdout.
Appose uses stdout for its JSON-based IPC protocol.
Any print() call corrupts the protocol and crashes communication.
"""
import sys
import os
import logging
import threading
import time

# Configure logging to stderr (stdout is reserved for Appose JSON protocol)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("dlclassifier.appose")

# The dlclassifier_server package is installed in the pixi environment
# via the git URL in pixi.toml. No sys.path manipulation needed.

# Import and initialize persistent services.
# ImportError is NOT caught here -- if critical packages (torch, smp, etc.)
# are missing, the init must fail loudly so the Java side knows the
# environment is broken. Runtime errors (e.g. no GPU) are caught and
# handled gracefully so the service can still run in CPU mode.
from dlclassifier_server.services.gpu_manager import GPUManager
from dlclassifier_server.services.inference_service import InferenceService
from dlclassifier_server.services.model_registry import ModelRegistry

try:
    gpu_manager = GPUManager()
    inference_service = InferenceService(device="auto", gpu_manager=gpu_manager)
    model_registry = ModelRegistry()

    logger.info("DL classifier services initialized successfully")
    logger.info("Device: %s", inference_service._device_str)

    # Threading lock for GPU operations.
    # Appose runs each task in its own thread. Without serialization,
    # concurrent tile inference tasks race on model loading, CUDA memory
    # allocation, and forward passes. PyTorch CUDA ops are thread-safe
    # but concurrent batches can OOM and torch.compile is NOT thread-safe.
    inference_lock = threading.Lock()

    # --- Monkey-patch _load_model for stale pip package ---
    # The installed dlclassifier_server package may be an older version
    # that defaults to in_channels=3 instead of reading num_channels from
    # the model metadata. This patch ensures multi-channel models (e.g.
    # multiplex IF with 8 channels) load correctly.
    import json as _json
    from pathlib import Path as _Path

    _orig_load_model = InferenceService._load_model

    def _patched_load_model(self, model_path, compile_model=True):
        """Load model with correct in_channels detected from checkpoint."""
        import torch
        import segmentation_models_pytorch as smp

        # Use cache if available
        if model_path in self._model_cache:
            model_tuple = self._model_cache[model_path]
            if (compile_model and model_tuple[0] == "pytorch"
                    and model_path not in self._compiled_models):
                if hasattr(self, '_try_compile_model'):
                    self._try_compile_model(model_path, model_tuple)
            return self._model_cache[model_path]

        model_dir = _Path(model_path)

        # ONNX path -- defer to original (ONNX doesn't need in_channels)
        onnx_path = model_dir / "model.onnx"
        if onnx_path.exists():
            return _orig_load_model(self, model_path, compile_model)

        # PyTorch path -- detect in_channels from checkpoint weights
        pt_path = model_dir / "model.pt"
        if pt_path.exists():
            logger.info("Loading PyTorch model from %s (patched)", pt_path)

            metadata_path = model_dir / "metadata.json"
            with open(metadata_path) as f:
                metadata = _json.load(f)

            arch = metadata.get("architecture", {})
            input_config = metadata.get("input_config", {})
            model_type = arch.get("type", "unet")
            encoder_name = arch.get("backbone", "resnet34")
            # Check both Python-style (input_config.num_channels) and
            # Java-style (architecture.input_channels) metadata formats.
            # Java's ModelManager.saveClassifier() overwrites Python's
            # metadata.json, replacing input_config with architecture format.
            metadata_channels = input_config.get(
                "num_channels", arch.get("input_channels", 3))
            num_classes = len(metadata.get("classes",
                                           [{"index": 0}, {"index": 1}]))

            # Load checkpoint first so we can detect the actual in_channels
            # from the encoder's first conv layer weight shape.
            # This is the ground truth -- metadata may be wrong (known bug
            # where training saves num_channels=3 for multi-channel models).
            state_dict = torch.load(pt_path, map_location=self.device,
                                    weights_only=True)

            num_channels = metadata_channels
            conv1_key = "encoder.conv1.weight"
            if conv1_key in state_dict:
                checkpoint_channels = state_dict[conv1_key].shape[1]
                if checkpoint_channels != metadata_channels:
                    logger.warning(
                        "Metadata num_channels=%d but checkpoint "
                        "encoder.conv1.weight has %d input channels. "
                        "Using checkpoint value.",
                        metadata_channels, checkpoint_channels)
                num_channels = checkpoint_channels
            else:
                # Some encoder architectures use different key names.
                # Search for the first conv weight to detect in_channels.
                for key in state_dict:
                    if ("conv" in key and "weight" in key
                            and state_dict[key].dim() == 4):
                        checkpoint_channels = state_dict[key].shape[1]
                        if checkpoint_channels != metadata_channels:
                            logger.warning(
                                "Metadata num_channels=%d but checkpoint "
                                "%s has %d input channels. "
                                "Using checkpoint value.",
                                metadata_channels, key, checkpoint_channels)
                        num_channels = checkpoint_channels
                        break

            # Resolve custom encoder names (e.g. histology-pretrained) to
            # base SMP encoder names. The checkpoint state_dict already
            # contains the trained weights, so we only need the right
            # architecture -- no need to download pretrained weights.
            smp_encoder_name = encoder_name
            try:
                from dlclassifier_server.services.pretrained_models import (
                    PretrainedModelsService)
                if encoder_name in PretrainedModelsService.HISTOLOGY_ENCODERS:
                    smp_encoder_name = (
                        PretrainedModelsService.HISTOLOGY_ENCODERS
                        [encoder_name][0])
                    logger.info(
                        "Resolved custom encoder '%s' -> '%s' for SMP",
                        encoder_name, smp_encoder_name)
            except ImportError:
                pass  # Server package not available; fall through

            logger.info("Model: %s/%s, in_channels=%d (metadata=%d), "
                        "classes=%d",
                        model_type, encoder_name, num_channels,
                        metadata_channels, num_classes)

            model_map = {
                "unet": smp.Unet,
                "unetplusplus": smp.UnetPlusPlus,
                "deeplabv3": smp.DeepLabV3,
                "deeplabv3plus": smp.DeepLabV3Plus,
                "fpn": smp.FPN,
                "pspnet": smp.PSPNet,
                "manet": smp.MAnet,
                "linknet": smp.Linknet,
                "pan": smp.PAN,
            }

            model_cls = model_map.get(model_type, smp.Unet)
            model = model_cls(
                encoder_name=smp_encoder_name,
                encoder_weights=None,
                in_channels=num_channels,
                classes=num_classes
            )

            # Auto-detect BatchRenorm from state dict keys (rmax/dmax are
            # unique to BatchRenorm2d). Metadata flag may be lost when Java
            # overwrites metadata.json, so detection from weights is robust.
            has_batchrenorm = any(
                k.endswith('.rmax') or k.endswith('.dmax')
                for k in state_dict)
            if has_batchrenorm:
                from dlclassifier_server.utils.batchrenorm import (
                    replace_bn_with_batchrenorm)
                replace_bn_with_batchrenorm(model)
                logger.info("Auto-detected BatchRenorm from state dict keys")

            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            self._model_cache[model_path] = ("pytorch", model)

            if compile_model and hasattr(self, '_try_compile_model'):
                self._try_compile_model(model_path, ("pytorch", model))

            return self._model_cache[model_path]

        raise FileNotFoundError("No model found at %s" % model_path)

    InferenceService._load_model = _patched_load_model
    logger.info("Patched InferenceService._load_model for multi-channel support")

except Exception as e:
    logger.error("Failed to initialize DL classifier services: %s", e)
    # Store error so tasks can report it -- imports succeeded but
    # runtime initialization failed (e.g. GPU not available)
    init_error = str(e)
    gpu_manager = None
    inference_service = None
    model_registry = None


# --- Parent process watcher ---
# Safety net: if QuPath is force-killed (Task Manager, kill -9), the JVM
# shutdown hook never runs, so stdin never closes and this process lives
# forever. This daemon thread polls the parent PID and exits if it dies.

def _parent_alive(pid):
    """Check if a process with the given PID is still running."""
    if sys.platform == 'win32':
        # os.kill(pid, 0) does NOT work on Windows -- signal 0 maps to
        # CTRL_C_EVENT which crashes the process. Use the Win32 API instead.
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)  # Signal 0 = existence check (Unix only)
            return True
        except PermissionError:
            return True  # Process exists but no permission
        except OSError:
            return False  # Process does not exist


def _watch_parent():
    """Exit if parent process (Java/QuPath) dies."""
    ppid = os.getppid()
    if ppid <= 1:
        return  # No meaningful parent to watch (already orphaned or init)
    logger.info("Parent process watcher started (parent PID: %d)", ppid)
    while True:
        time.sleep(3)
        try:
            # Check 1: parent PID changed (Linux reparents to init/systemd)
            current_ppid = os.getppid()
            if current_ppid != ppid:
                logger.warning("Parent process changed (%d -> %d), exiting",
                               ppid, current_ppid)
                os._exit(1)
            # Check 2: parent PID no longer exists
            if not _parent_alive(ppid):
                logger.warning("Parent process %d no longer exists, exiting",
                               ppid)
                os._exit(1)
        except Exception as e:
            # Never crash the thread -- log and keep watching
            logger.debug("Parent watcher check error: %s", e)


_parent_watcher = threading.Thread(target=_watch_parent, daemon=True)
_parent_watcher.start()
