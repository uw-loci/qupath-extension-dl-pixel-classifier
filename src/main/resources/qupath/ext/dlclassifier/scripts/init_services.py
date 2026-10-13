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
import numpy  # Must be imported during init, not in a task -- see Appose #23

# Configure logging to stderr (stdout is reserved for Appose JSON protocol)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("dlclassifier.appose")

# --- Appose protocol / stdio separation (apposed/appose#31) ---------------
# Active only when Java prepends `_DLC_PROTOCOL_FD_SEPARATION = True`, which
# it does when the DataLoader-workers preference is > 0. With workers at 0
# (the default) nothing here runs and the streams are untouched.
#
# PyTorch DataLoader worker processes inherit this process's fd 0 / fd 1,
# which are Appose's JSON protocol pipes, and deadlock during their own stdio
# setup before the first batch is ever delivered: on Linux the forked child
# hangs closing the inherited pipe in multiprocessing.util._close_stdin(); on
# Windows the spawned child hangs inside interpreter startup flushing the
# inherited handle. Upstream issue apposed/appose#31 is open and unmerged, so
# we apply the same fd-level separation from here instead.
#
# Keep private duplicates of fd 0 / fd 1 for the protocol, then point the real
# fd 0 / fd 1 at os.devnull so children inherit harmless empty streams. This
# runs in the init script, which the worker executes BEFORE starting its I/O
# loop, so no reader is mid-read when the swap happens.
#
# Do NOT "harden" this by giving sys.stdin a wrapper whose fileno() returns a
# real descriptor: that variant was measured and reintroduced the hang.
if globals().get("_DLC_PROTOCOL_FD_SEPARATION", False):
    import io as _io

    _protocol_in = _io.TextIOWrapper(_io.FileIO(os.dup(0), "r"), encoding="utf-8")
    _protocol_out = _io.TextIOWrapper(
        _io.FileIO(os.dup(1), "w"), encoding="utf-8", line_buffering=True
    )
    _devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(_devnull_fd, 0)
    os.dup2(_devnull_fd, 1)
    os.close(_devnull_fd)
    # The worker reads protocol requests with input(), i.e. from sys.stdin, so
    # the protocol MUST stay on sys.stdin -- unlike the upstream patch, which
    # points sys.stdin at devnull because it reads its own private handle.
    # sys.stdin / sys.stdout are what keep these objects alive.
    sys.stdin = _protocol_in
    sys.stdout = _protocol_out
    logger.info(
        "Appose protocol separated from fd 0/1 for DataLoader workers "
        "(apposed/appose#31)"
    )

# The dlclassifier_server package is installed in the pixi environment
# via the git URL in pixi.toml. No sys.path manipulation needed.

# --- Version enforcement ---
# The JAR bundles these scripts; the Python package is pip-installed in pixi.
# Both sides MUST be in sync. This version MUST match pyproject.toml and
# __init__.py exactly. When EITHER side is updated, ALL THREE version
# locations must be bumped together (see memory/feedback_version_sync.md).
_REQUIRED_PYTHON_VERSION = "0.9.26"


def _parse_version(v):
    """Parse a version string into a comparable tuple of ints.
    Strips dev/pre-release suffixes (e.g., '0.5.1.dev0' -> (0, 5, 1))."""
    try:
        # Strip common suffixes: .dev0, -dev, .rc1, etc.
        clean = v.split("-")[0].split(".dev")[0].split("rc")[0]
        return tuple(int(x) for x in clean.split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _is_dev_version(v):
    """Check if a version string indicates a development build."""
    return "dev" in v.lower() if v else False


import dlclassifier_server as _dls

_installed_version = getattr(_dls, "__version__", "unknown")
_installed_tuple = _parse_version(_installed_version)
_required_tuple = _parse_version(_REQUIRED_PYTHON_VERSION)

# Dev builds skip strict version matching -- both sides are from main
# branch and may be at slightly different commits.  Only check that the
# major.minor match so completely wrong versions are still caught.
if _is_dev_version(_REQUIRED_PYTHON_VERSION):
    # Dev mode: accept any version with matching major.minor
    _version_ok = (
        _installed_tuple[:2] == _required_tuple[:2]
        if len(_installed_tuple) >= 2 and len(_required_tuple) >= 2
        else True
    )
else:
    # Release mode: exact major.minor.patch match
    _version_ok = _installed_tuple == _required_tuple

if not _version_ok:
    if _installed_tuple < _required_tuple:
        _msg = (
            "PYTHON PACKAGE OUT OF DATE: installed dlclassifier-server v%s "
            "but the extension requires v%s. "
            "Go to Extensions > DL Pixel Classifier > Utilities > "
            "Rebuild DL Environment to update."
            % (_installed_version, _REQUIRED_PYTHON_VERSION)
        )
    else:
        _msg = (
            "PYTHON PACKAGE VERSION MISMATCH: installed dlclassifier-server v%s "
            "but the extension expects v%s. "
            "Go to Extensions > DL Pixel Classifier > Utilities > "
            "Rebuild DL Environment to resync."
            % (_installed_version, _REQUIRED_PYTHON_VERSION)
        )
    logger.error(_msg)
    # Block initialization -- health check will return unhealthy and the
    # Java side will show the version_warning and disable training.
    version_warning = _msg
    gpu_manager = None
    inference_service = None
    model_registry = None
    inference_lock = threading.Lock()
else:
    version_warning = None
    logger.info(
        "dlclassifier-server v%s (required: %s) -- OK",
        _installed_version,
        _REQUIRED_PYTHON_VERSION,
    )

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
                PretrainedModelsService,
            )

            pretrained_service = PretrainedModelsService()
            logger.info(
                "PretrainedModelsService loaded (%d encoders)",
                len(pretrained_service.list_encoders()),
            )
        except Exception as e:
            pretrained_service = None
            logger.warning("PretrainedModelsService not available: %s", e)

        try:
            from dlclassifier_server.services.muvit_model import create_muvit_model

            logger.info("MuViT model factory loaded")
        except Exception as e:
            logger.warning("MuViT model factory not available: %s", e)

        try:
            from dlclassifier_server.utils.batchrenorm import (
                replace_bn_with_batchrenorm,
            )

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
# forever. This daemon thread polls the host process and exits if it dies.
#
# WATCH THE JVM, NOT os.getppid(). The worker's real process tree is
# `java -> pixi -> python`, so os.getppid() returns PIXI, a launcher whose
# lifetime is not QuPath's. Measured 2026-10-12 on Windows: the worker logged
# parent PID 11284, which Win32_Process reported as pixi.exe, while QuPath was
# PID 2820. Watching pixi is wrong in both directions -- the worker outlives a
# force-killed QuPath whose pixi happens to persist, and a worker whose pixi
# exits on its own is killed although QuPath is alive and still talking to it
# over the Appose pipes. Java therefore prepends `_DLC_HOST_PID` (see
# ApposeService.buildInitScript); os.getppid() is only the fallback for an
# older JAR or a script run by hand.


def _get_process_create_time_win32(pid):
    """Get process creation time on Windows. Returns int or None."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        creation = wintypes.FILETIME()
        exit_t = wintypes.FILETIME()
        kern = wintypes.FILETIME()
        user = wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_t),
            ctypes.byref(kern),
            ctypes.byref(user),
        )
        if ok:
            return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return None
    finally:
        kernel32.CloseHandle(handle)


def _get_process_create_time_linux(pid):
    """Get process start time (clock ticks since boot) on Linux, or None.

    Field 22 of /proc/<pid>/stat. The comm field (2) may itself contain
    spaces and parentheses, so split after the last ')'.
    """
    try:
        with open("/proc/%d/stat" % pid, "rb") as fh:
            data = fh.read().decode("utf-8", "replace")
        fields = data[data.rindex(")") + 1 :].split()
        return int(fields[19])  # field 22 overall, 20th after the comm field
    except Exception:
        return None


def _get_process_create_time(pid):
    """Process creation time on this platform, or None if unavailable."""
    if sys.platform == "win32":
        return _get_process_create_time_win32(pid)
    if sys.platform.startswith("linux"):
        return _get_process_create_time_linux(pid)
    return None


def _parent_alive(pid, expected_create_time=None):
    """Check whether the *original* host process is still running.

    PIDs are recycled -- quickly on Windows, eventually anywhere -- so a
    process at the same PID is not necessarily the same process. When a
    creation time was captured at startup, it has to match; a PID that is
    alive with a different creation time counts as dead.
    """
    if sys.platform == "win32":
        create_time = _get_process_create_time_win32(pid)
        if create_time is None:
            return False  # Process does not exist
        if expected_create_time is not None and create_time != expected_create_time:
            return False  # PID was recycled -- different process
        return True
    try:
        os.kill(pid, 0)  # Signal 0 = existence check (Unix only)
    except PermissionError:
        pass  # Process exists but no permission to signal it
    except OSError:
        return False  # Process does not exist
    if expected_create_time is not None:
        create_time = _get_process_create_time_linux(pid)
        # Only a *known, different* start time proves recycling; an
        # unreadable /proc entry must not be read as "the host died".
        if create_time is not None and create_time != expected_create_time:
            return False
    return True


def _die(reason):
    """Write a reason to fd 2 and leave immediately.

    os._exit() runs no atexit handlers and flushes no buffers, so a
    logger call alone can be lost between emit and process death. An
    os.write() to fd 2 has already reached the pipe when it returns, so
    the reason survives -- which matters because "exit code 1, no output"
    is otherwise indistinguishable from a dozen other failures (GitHub
    issue #30).
    """
    try:
        os.write(2, ("DLCLASSIFIER WORKER EXIT: " + reason + "\n").encode("utf-8", "replace"))
    except Exception:
        pass  # Nothing left to try; still exit.
    logger.warning("Worker exiting: %s", reason)
    os._exit(1)


def _watch_parent():
    """Exit if the host process (the QuPath JVM) dies."""
    host_pid = globals().get("_DLC_HOST_PID")
    watching_jvm = isinstance(host_pid, int) and host_pid > 1
    if not watching_jvm:
        # Fallback for an older JAR or a hand-run script: the immediate
        # parent, which is pixi rather than the JVM. See the banner above.
        host_pid = os.getppid()
    if host_pid <= 1:
        return  # No meaningful host to watch (already orphaned or init)

    # Capture the host's creation time at startup so a recycled PID is not
    # mistaken for a live host.
    host_create_time = _get_process_create_time(host_pid)

    logger.info(
        "Parent process watcher started (host PID: %d, source: %s, start time: %s)",
        host_pid,
        "jvm" if watching_jvm else "getppid",
        host_create_time if host_create_time is not None else "unavailable",
    )
    while True:
        time.sleep(1)
        try:
            # Reparenting only means anything when host_pid IS the parent.
            # Watching the JVM, a changed getppid() just means pixi exited,
            # which says nothing about whether QuPath is still running.
            if not watching_jvm and os.getppid() != host_pid:
                _die("parent process changed (%d -> %d)" % (host_pid, os.getppid()))
            alive = _parent_alive(host_pid, host_create_time)
        except Exception as e:
            # Never crash the thread -- a failed check is not a dead host.
            logger.debug("Parent watcher check error: %s", e)
            continue
        if not alive:
            _die("host process %d is gone (watching %s)"
                 % (host_pid, "jvm" if watching_jvm else "getppid"))


_parent_watcher = threading.Thread(target=_watch_parent, daemon=True)
_parent_watcher.start()
