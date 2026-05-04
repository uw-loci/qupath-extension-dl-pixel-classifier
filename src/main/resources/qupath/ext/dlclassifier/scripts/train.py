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

try:
    classifier_name
except NameError:
    classifier_name = None

# Import training service (heavier import, done here rather than init)
from dlclassifier_server.services.training_service import TrainingService
import dlclassifier_server.services.training_service as _tsm
import numpy as _np

# Fix: np.frombuffer returns read-only arrays; ensure _load_patch always
# returns writable arrays so normalization can modify in-place.
# This patches the installed package which may be stale (Appose pip cache).
# Guard: only patch once per worker process (Appose reuses the worker).
if not getattr(_tsm.SegmentationDataset, '_patched_writable', False):
    _orig_lp = _tsm.SegmentationDataset._load_patch
    @staticmethod
    def _writable_load_patch(img_path):
        arr = _orig_lp(img_path)
        return arr.copy() if not arr.flags.writeable else arr
    _tsm.SegmentationDataset._load_patch = _writable_load_patch
    _tsm.SegmentationDataset._patched_writable = True


# Safety net: resize context tiles if they don't match detail tile dimensions.
# Edge tiles should be skipped at export, but stale pip packages may still
# have mismatched tiles from older exports. This patches the installed package.
from PIL import Image as _PILImage

# Guard: only patch __getitem__ once per worker process.
if not getattr(_tsm.SegmentationDataset, '_patched_getitem', False):
    _orig_getitem = _tsm.SegmentationDataset.__getitem__
    _ctx_resize_warned = [False]

    def _safe_getitem(self, idx):
        """__getitem__ with context tile resize for edge-case size mismatch."""
        # Use the in-memory cache if the preload patch populated it.
        # Cache holds the pre-context-concatenated image and the raw mask so
        # normalize + augmentation still run per-batch.
        _cached_imgs = getattr(self, '_cached_images', None)
        _cached_masks = getattr(self, '_cached_masks', None)
        if _cached_imgs is not None and _cached_imgs[idx] is not None:
            # .copy() so the augmentation pipeline can mutate freely without
            # corrupting the cached source.
            img_array = _cached_imgs[idx].copy()
            mask_array = (_cached_masks[idx].astype(_np.int64)
                          if _cached_masks is not None and _cached_masks[idx] is not None
                          else _np.zeros(img_array.shape[:2], dtype=_np.int64))
        else:
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

            mask_name = img_path.stem + ".png"
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                mask = _PILImage.open(mask_path)
                mask_array = _np.array(mask, dtype=_np.int64)
            else:
                mask_array = _np.zeros(img_array.shape[:2], dtype=_np.int64)

        img_array = self._normalize(img_array)

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
    _tsm.SegmentationDataset._patched_getitem = True

# In-memory dataset preload. Sidesteps the num_workers hang (Appose
# limitation) by eliminating per-batch disk I/O + TIFF decode entirely:
# every patch is loaded once at dataset construction and served from a
# numpy cache. Normalization and augmentation still run per-batch.
#
# Activation:
#   "auto" -> enable when the dataset fits in ~50% of available RAM;
#   "on"   -> force (may OOM on huge datasets);
#   "off"  -> leave the disk-streaming path alone.
#
# The patch wraps SegmentationDataset.__init__ so every dataset (train,
# val, and progressive-resize small loaders) picks it up automatically.
# Patched once per worker process (Appose reuses workers across tasks).
_in_memory_mode = training_params.get("in_memory_dataset", "auto")
_bounded_fraction = float(
    training_params.get("cache_bounded_fraction", 0.40))


def _query_available_ram_bytes():
    """Best-effort 'how much RAM can the cache safely use' query.

    psutil.virtual_memory().available reads from a Windows perf counter
    that can lag the real value by tens of GB when antivirus, the
    standby cache, or another process is mid-flux. On at least one
    deployment we observed psutil reporting 4 GB while Task Manager
    reported 27 GB available -- the cache aborted on a stale read.
    Mitigate by also calling MemoryStatusEx.ullAvailPhys (Windows) or
    parsing /proc/meminfo MemAvailable (Linux) and taking the larger
    of the two; both are cheap. Returns (bytes, source_str) so the
    log can show which value won.
    """
    psutil_bytes = None
    os_bytes = None
    try:
        import psutil as _ps
        psutil_bytes = _ps.virtual_memory().available
    except Exception:
        pass

    import sys as _sys
    if _sys.platform == "win32":
        try:
            import ctypes as _ct
            from ctypes import wintypes as _wt

            class _MEMORYSTATUSEX(_ct.Structure):
                _fields_ = [
                    ("dwLength", _ct.c_ulong),
                    ("dwMemoryLoad", _ct.c_ulong),
                    ("ullTotalPhys", _ct.c_ulonglong),
                    ("ullAvailPhys", _ct.c_ulonglong),
                    ("ullTotalPageFile", _ct.c_ulonglong),
                    ("ullAvailPageFile", _ct.c_ulonglong),
                    ("ullTotalVirtual", _ct.c_ulonglong),
                    ("ullAvailVirtual", _ct.c_ulonglong),
                    ("sullAvailExtendedVirtual", _ct.c_ulonglong),
                ]
            stat = _MEMORYSTATUSEX()
            stat.dwLength = _ct.sizeof(_MEMORYSTATUSEX)
            if _ct.windll.kernel32.GlobalMemoryStatusEx(_ct.byref(stat)):
                os_bytes = int(stat.ullAvailPhys)
        except Exception:
            pass
    elif _sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as _f:
                for _line in _f:
                    if _line.startswith("MemAvailable:"):
                        os_bytes = int(_line.split()[1]) * 1024
                        break
        except Exception:
            pass

    if psutil_bytes is None and os_bytes is None:
        return None, "unavailable"
    if psutil_bytes is None:
        return os_bytes, "os"
    if os_bytes is None:
        return psutil_bytes, "psutil"
    # Both readings: take the larger. Significant disagreement is logged
    # so we know when the perf-counter path is producing stale numbers.
    if max(psutil_bytes, os_bytes) > 1.5 * min(psutil_bytes, os_bytes):
        logger.info(
            "RAM query: psutil=%.2f GB, OS=%.2f GB -- significant "
            "disagreement; using OS value (psutil perf counter may be "
            "stale)",
            psutil_bytes / 1e9, os_bytes / 1e9)
    return max(psutil_bytes, os_bytes), (
        "os" if os_bytes >= psutil_bytes else "psutil")


# Threshold: dataset must fit in this fraction of available RAM for
# 'auto' to enable. 0.50 (was 0.25) is more aggressive but reasonable:
# the cache only stores tile data + uint8 masks, not gradients or
# activations, and the dataloader still has its own working set.
_AUTO_CACHE_FRACTION = 0.50

if _in_memory_mode in ("auto", "on", "bounded") and not getattr(
        _tsm.SegmentationDataset, "_patched_preload", False):
    _orig_sd_init = _tsm.SegmentationDataset.__init__
    try:
        import psutil as _psutil
        _psutil_ok = True
    except ImportError:
        _psutil_ok = False
        if _in_memory_mode == "auto":
            logger.warning("In-memory cache: psutil not available -- "
                           "'auto' mode will rely on OS-native RAM query. "
                           "Install psutil for cross-validation.")

    def _init_with_cache(self, *args, **kwargs):
        _orig_sd_init(self, *args, **kwargs)
        self._cached_images = None
        self._cached_masks = None

        n = len(self.image_files)
        if n == 0:
            return

        # Estimate bytes by loading one tile. Cache stores native dtype +
        # uint8 masks; normalize happens per-batch.
        try:
            first_img = _tsm.SegmentationDataset._load_patch(self.image_files[0])
            if first_img.ndim == 2:
                first_img = first_img[..., _np.newaxis]
            per_img_bytes = first_img.nbytes
            # context doubles the stored channels
            if self.context_dir is not None:
                per_img_bytes *= 2
            # masks stored as uint8 (classes < 256), promoted to int64 per batch
            per_mask_bytes = first_img.shape[0] * first_img.shape[1]
            total_bytes = n * (per_img_bytes + per_mask_bytes)
        except Exception as _e:
            logger.warning("In-memory cache: byte estimate failed (%s); skipping preload", _e)
            return

        # Bounded mode: pick a stratified random subset that fits in
        # `_bounded_fraction` of available RAM, slice self.image_files
        # in place so the rest of the dataloader (and the cache below)
        # only sees the subset. Each class is guaranteed >= 1 patch via
        # a rare-class floor. This is the explicit "train on a subset"
        # path -- the user has confirmed this in a pre-flight dialog.
        if _in_memory_mode == "bounded":
            available, source = _query_available_ram_bytes()
            if available is None:
                logger.warning(
                    "Bounded cache: cannot query available RAM; "
                    "skipping preload, falling back to disk streaming")
                return
            cap_bytes = _bounded_fraction * available
            per_patch_bytes = per_img_bytes + per_mask_bytes
            if cap_bytes >= total_bytes:
                logger.info(
                    "Bounded cache: full dataset fits in %.0f%% of "
                    "%.2f GB available (source=%s); preloading entire "
                    "dataset (no subsetting needed)",
                    _bounded_fraction * 100, available / 1e9, source)
                # Fall through to the regular preload path -- n stays the same.
            else:
                n_subset = max(1, int(cap_bytes // per_patch_bytes))
                if n_subset >= n:
                    n_subset = n  # paranoia
                # Stratified subset: try to read class info from masks
                # (sparse-sample). The same per-patch class presence the
                # AnnotationExtractor uses for stratified train/val split.
                logger.info(
                    "Bounded cache: building stratified subset of %d "
                    "patches (out of %d) to fit %.2f GB cap "
                    "(%.0f%% of %.2f GB %s)",
                    n_subset, n,
                    cap_bytes / 1e9,
                    _bounded_fraction * 100, available / 1e9, source)
                try:
                    import random as _random
                    _rng = _random.Random(42)
                    # Class presence per patch (sparse-sample masks).
                    presence = []
                    for i in range(n):
                        mp = self.masks_dir / (
                            self.image_files[i].stem + ".png")
                        seen = set()
                        if mp.exists():
                            try:
                                _m = _PILImage.open(mp)
                                _w, _h = _m.size
                                # 5x5 grid sample is enough to identify
                                # the dominant class without loading the
                                # full mask for all 56k patches.
                                _ar = _np.asarray(_m)
                                step_y = max(1, _h // 5)
                                step_x = max(1, _w // 5)
                                for _y in range(0, _h, step_y):
                                    for _x in range(0, _w, step_x):
                                        seen.add(int(_ar[_y, _x]))
                            except Exception:
                                pass
                        if not seen:
                            seen.add(0)
                        presence.append(seen)

                    # Group by dominant class (smallest class index in
                    # the patch -- biases toward rare classes which is
                    # what we want for the floor).
                    by_class = {}
                    for i, seen in enumerate(presence):
                        cls = min(seen)
                        by_class.setdefault(cls, []).append(i)

                    selected = []
                    # Floor: 1 patch per class present, sampled randomly.
                    for cls, idxs in sorted(by_class.items()):
                        _rng.shuffle(idxs)
                        selected.append(idxs[0])
                    floor_count = len(selected)

                    # Fill the rest proportionally to class frequency.
                    remaining = n_subset - floor_count
                    if remaining > 0:
                        # Available-after-floor pool per class.
                        pool = {cls: idxs[1:]
                                for cls, idxs in by_class.items()}
                        total_pool = sum(len(v) for v in pool.values())
                        if total_pool > 0:
                            for cls, idxs in pool.items():
                                # Proportional take, clamped to pool size.
                                take = int(round(
                                    remaining * len(idxs) / total_pool))
                                take = min(take, len(idxs))
                                selected.extend(idxs[:take])
                            # Any rounding shortfall: fill randomly.
                            if len(selected) < n_subset:
                                leftover = []
                                taken = set(selected)
                                for idxs in pool.values():
                                    for j in idxs:
                                        if j not in taken:
                                            leftover.append(j)
                                _rng.shuffle(leftover)
                                need = n_subset - len(selected)
                                selected.extend(leftover[:need])

                    # Trim to exactly n_subset if proportional rounding
                    # overshot (rare but possible).
                    selected = selected[:n_subset]
                    selected.sort()  # stable order for reproducibility

                    # Slice the dataset in place. From this point on,
                    # len(self) reflects the subset; the cache below
                    # only allocates n_subset entries.
                    self.image_files = [self.image_files[i] for i in selected]
                    n = len(self.image_files)
                    total_bytes = n * per_patch_bytes

                    # Per-class count for the user-visible report.
                    cls_counts = {}
                    for i in selected:
                        cls = min(presence[i])
                        cls_counts[cls] = cls_counts.get(cls, 0) + 1
                    classes_str = ", ".join(
                        "class %d: %d" % (c, cls_counts[c])
                        for c in sorted(cls_counts))
                    coverage_pct = 100.0 * n / len(presence)
                    logger.warning(
                        "Bounded cache active: training on %d / %d "
                        "patches (%.1f%% coverage). Per-class subset "
                        "counts: %s. The model will NOT see patches "
                        "outside this subset in this run.",
                        n, len(presence), coverage_pct, classes_str)
                    # Surface the same info to the Java progress monitor
                    # via setup_callback. The Java side renders this as
                    # a banner so the user can see it without scrolling
                    # the log.
                    try:
                        if 'setup_callback' in globals():
                            setup_callback("bounded_cache_subset", {
                                "subset_size": n,
                                "full_size": len(presence),
                                "coverage_pct": round(coverage_pct, 2),
                                "per_class_counts": {
                                    str(c): cls_counts[c]
                                    for c in sorted(cls_counts)},
                                "subset_bytes_gb":
                                    round(total_bytes / 1e9, 2),
                                "available_bytes_gb":
                                    round(available / 1e9, 2),
                            })
                    except Exception:
                        pass
                except Exception as _e:
                    logger.error(
                        "Bounded cache: subset selection failed (%s); "
                        "skipping cache, falling back to disk streaming",
                        _e)
                    return
            # Continue to the preload loop below with the (possibly
            # subset-sliced) self.image_files.

        elif _in_memory_mode == "auto":
            available, source = _query_available_ram_bytes()
            if available is None:
                logger.warning(
                    "In-memory cache: cannot query available RAM "
                    "(psutil + OS-native both failed); skipping preload")
                return
            if total_bytes >= _AUTO_CACHE_FRACTION * available:
                logger.info(
                    "In-memory cache: 'auto' declined for %s -- "
                    "estimate %.2f GB >= %.0f%% of %.2f GB available "
                    "(source=%s)",
                    self.images_dir.name if hasattr(self.images_dir, 'name')
                    else str(self.images_dir),
                    total_bytes / 1e9,
                    _AUTO_CACHE_FRACTION * 100, available / 1e9, source)
                return
            logger.info(
                "In-memory cache: 'auto' will preload %.2f GB "
                "(%.0f%% of %.2f GB available, source=%s)",
                total_bytes / 1e9,
                100 * total_bytes / available,
                available / 1e9, source)
        elif _in_memory_mode == "on":
            available, source = _query_available_ram_bytes()
            if available is not None and total_bytes > available:
                logger.warning(
                    "In-memory cache: 'on' forced but estimate %.2f GB "
                    "exceeds %.2f GB available (source=%s) -- preload "
                    "may fail with MemoryError",
                    total_bytes / 1e9, available / 1e9, source)

        logger.info("In-memory cache: preloading %d patches (~%.2f GB) from %s...",
                    n, total_bytes / 1e9,
                    self.images_dir.name if hasattr(self.images_dir, 'name')
                    else str(self.images_dir))

        cached_imgs = [None] * n
        cached_masks = [None] * n
        _log_every = max(1, n // 10)
        try:
            for i in range(n):
                img_path = self.image_files[i]
                img_arr = _tsm.SegmentationDataset._load_patch(img_path)
                if img_arr.ndim == 2:
                    img_arr = img_arr[..., _np.newaxis]
                if self.context_dir is not None:
                    ctx_path = self.context_dir / img_path.name
                    if ctx_path.exists():
                        ctx_arr = _tsm.SegmentationDataset._load_patch(ctx_path)
                        if ctx_arr.ndim == 2:
                            ctx_arr = ctx_arr[..., _np.newaxis]
                        if (ctx_arr.shape[0] != img_arr.shape[0]
                                or ctx_arr.shape[1] != img_arr.shape[1]):
                            h, w = img_arr.shape[:2]
                            resized_ch = []
                            for c in range(ctx_arr.shape[2]):
                                ch = _PILImage.fromarray(ctx_arr[:, :, c])
                                ch = ch.resize((w, h), _PILImage.BILINEAR)
                                resized_ch.append(_np.array(ch))
                            ctx_arr = _np.stack(resized_ch, axis=2)
                        img_arr = _np.concatenate([img_arr, ctx_arr], axis=2)
                    else:
                        img_arr = _np.concatenate([img_arr, img_arr], axis=2)
                cached_imgs[i] = img_arr

                mask_path = self.masks_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    mask = _PILImage.open(mask_path)
                    cached_masks[i] = _np.array(mask, dtype=_np.uint8)
                else:
                    cached_masks[i] = _np.zeros(img_arr.shape[:2], dtype=_np.uint8)

                if (i + 1) % _log_every == 0 or (i + 1) == n:
                    logger.info("  cache %d/%d", i + 1, n)
        except MemoryError:
            logger.error("In-memory cache: MemoryError during preload at %d/%d "
                         "-- falling back to disk streaming", i, n)
            self._cached_images = None
            self._cached_masks = None
            return
        except Exception as _e:
            logger.error("In-memory cache: preload failed (%s) -- "
                         "falling back to disk streaming", _e)
            self._cached_images = None
            self._cached_masks = None
            return

        self._cached_images = cached_imgs
        self._cached_masks = cached_masks
        logger.info("In-memory cache: preload complete (%d patches)", n)

    _tsm.SegmentationDataset.__init__ = _init_with_cache
    _tsm.SegmentationDataset._patched_preload = True

# DataLoader num_workers bootstrap: the installed pip package may still
# hardcode num_workers=0 in training_service._run_training (prior to
# 0.5.5-dev). When the user has set the "Training: DataLoader Workers"
# preference > 0, monkey-patch torch.utils.data.DataLoader.__init__ so
# every DataLoader instantiated after this point uses the requested count,
# regardless of the installed server version. Bootstraps cleanly off once
# dlclassifier-server >= 0.5.5-dev honors training_params natively.
_dl_workers_pref = int(training_params.get("data_loader_workers", 0))
# Suppress the monkey-patch whenever the in-memory cache is active.
# The cache is attached to the dataset instance before any DataLoader
# is built, so on Windows/Appose spawn each worker pickles a copy of
# the cache (RAM usage scales linearly with worker count). The
# training_service now also forces num_workers=0 in that case; we
# must not re-upgrade here. Matches the D.1 fix.
_cache_mode_for_workers = str(
    training_params.get("in_memory_dataset", "auto")).lower()
if (_cache_mode_for_workers in ("auto", "on", "bounded")
        and _dl_workers_pref > 0):
    logger.info(
        "DataLoader bootstrap: monkey-patch suppressed because "
        "in-memory cache is active (would duplicate cache per "
        "worker on Windows/Appose spawn).")
    _dl_workers_pref = 0

if _dl_workers_pref > 0:
    from torch.utils.data import DataLoader as _DataLoader
    if not getattr(_DataLoader, "_patched_num_workers", False):
        _orig_dl_init = _DataLoader.__init__

        def _patched_dl_init(self, *args, **kwargs):
            # Only upgrade when the caller explicitly passed num_workers=0
            # (or omitted it so it defaults to 0). This lets future
            # call-sites that opt out of worker processes stay at 0.
            if kwargs.get("num_workers", 0) == 0:
                kwargs["num_workers"] = _dl_workers_pref
                # persistent_workers reduces per-epoch worker startup cost
                # for small datasets; only meaningful when num_workers > 0.
                kwargs.setdefault("persistent_workers", True)
            return _orig_dl_init(self, *args, **kwargs)

        _DataLoader.__init__ = _patched_dl_init
        _DataLoader._patched_num_workers = True
        logger.info(
            "DataLoader bootstrap: forcing num_workers=%d, persistent_workers=True "
            "(installed pip package may still hardcode num_workers=0)",
            _dl_workers_pref)

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
        # Keep the source in ~/.dlclassifier/checkpoints/ as well. That is the
        # central registry the Java side scans for crash-recovery discovery
        # (CheckpointScanner, surfaced via project-open toast and Train dialog
        # banner). Both copies are cleaned up on normal completion/cancel via
        # _cleanup_best_in_progress.
        logger.info("Best-in-progress saved to project and central registry: %s", dst)
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


# Track best epoch metrics as fallback for stale pip packages that may
# not populate these fields in the result dict.
_epoch_tracker = {"best_epoch": 0, "best_mean_iou": 0.0, "last_epoch": 0,
                  "last_loss": 0.0, "last_acc": 0.0}


def progress_callback(epoch, train_loss, val_loss, accuracy,
                       per_class_iou, per_class_loss, mean_iou):
    """Forward training progress to Appose task events."""
    import math
    # Update fallback tracker
    _epoch_tracker["last_epoch"] = epoch
    _epoch_tracker["last_loss"] = val_loss if isinstance(val_loss, (int, float)) else 0.0
    _epoch_tracker["last_acc"] = accuracy if isinstance(accuracy, (int, float)) else 0.0
    if isinstance(mean_iou, (int, float)) and mean_iou > _epoch_tracker["best_mean_iou"]:
        _epoch_tracker["best_epoch"] = epoch
        _epoch_tracker["best_mean_iou"] = mean_iou
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

# Include classifier name in training_params so it survives in checkpoints
if classifier_name:
    training_params["classifier_name"] = classifier_name

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
task.outputs["checkpoint_path"] = result.get("checkpoint_path", "")
task.outputs["total_epochs"] = result.get("total_epochs", total_epochs)

# Use result values, falling back to progress_callback tracker for stale pip packages
task.outputs["best_epoch"] = result.get("best_epoch", 0) or _epoch_tracker["best_epoch"]
task.outputs["best_mean_iou"] = result.get("best_mean_iou", 0.0) or _epoch_tracker["best_mean_iou"]
task.outputs["last_epoch"] = result.get("epoch", 0) or _epoch_tracker["last_epoch"]
task.outputs["final_loss"] = result.get("final_loss", 0.0) or _epoch_tracker["last_loss"]
task.outputs["final_accuracy"] = result.get("final_accuracy", 0.0) or _epoch_tracker["last_acc"]
task.outputs["epochs_trained"] = result.get("epochs_trained", 0) or _epoch_tracker["last_epoch"]

# Forward focus class info if present (from pause, cancel, or completion)
if "focus_class_name" in result:
    task.outputs["focus_class_name"] = result["focus_class_name"]
    task.outputs["focus_class_iou"] = result.get("focus_class_iou", 0.0)
    task.outputs["focus_class_target_met"] = result.get("focus_class_target_met", True)
    task.outputs["focus_class_min_iou"] = result.get("focus_class_min_iou", 0.0)
