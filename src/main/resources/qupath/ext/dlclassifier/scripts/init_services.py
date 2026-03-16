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

# --- Version enforcement ---
# The JAR bundles these scripts; the Python package is pip-installed in pixi.
# Both sides MUST be in sync. When EITHER side is updated, bump the version
# below to match the new pyproject.toml version and the health check will
# block until the user rebuilds the pixi environment.
_REQUIRED_PYTHON_VERSION = "0.3.5"


def _parse_version(v):
    """Parse a version string into a comparable tuple of ints."""
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


import dlclassifier_server as _dls
_installed_version = getattr(_dls, "__version__", "unknown")
_installed_tuple = _parse_version(_installed_version)
_required_tuple = _parse_version(_REQUIRED_PYTHON_VERSION)

if _installed_tuple < _required_tuple:
    _msg = (
        "PYTHON PACKAGE OUT OF DATE: installed dlclassifier-server v%s "
        "but the extension requires v%s or newer. "
        "Go to DL Pixel Classifier > Rebuild Python Environment to update."
        % (_installed_version, _REQUIRED_PYTHON_VERSION))
    logger.error(_msg)
    # Block initialization -- health check will return False and the Java
    # side will show the version_warning to the user.
    version_warning = _msg
    gpu_manager = None
    inference_service = None
    model_registry = None
    inference_lock = threading.Lock()
else:
    version_warning = None
    logger.info("dlclassifier-server v%s (required >= %s) -- OK",
                _installed_version, _REQUIRED_PYTHON_VERSION)

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

        # --- Register additional model factories ---
        # These are optional; failure to load them should not block init.
        try:
            from dlclassifier_server.services.pretrained_models import (
                PretrainedModelsService)
            pretrained_service = PretrainedModelsService()
            logger.info("PretrainedModelsService loaded (%d encoders)",
                        len(pretrained_service.list_encoders()))
        except Exception as e:
            pretrained_service = None
            logger.warning("PretrainedModelsService not available: %s", e)

        try:
            from dlclassifier_server.services.muvit_model import (
                create_muvit_model)
            logger.info("MuViT model factory loaded")
        except Exception as e:
            logger.warning("MuViT model factory not available: %s", e)

        try:
            from dlclassifier_server.utils.batchrenorm import (
                replace_bn_with_batchrenorm)
            logger.info("BatchRenorm utility loaded")
        except Exception as e:
            logger.warning("BatchRenorm utility not available: %s", e)

    except Exception as e:
        logger.error("Failed to initialize DL classifier services: %s", e)
        # Store error so tasks can report it -- imports succeeded but
        # runtime initialization failed (e.g. GPU not available)
        init_error = str(e)
        gpu_manager = None
        inference_service = None
        model_registry = None
        inference_lock = threading.Lock()


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
