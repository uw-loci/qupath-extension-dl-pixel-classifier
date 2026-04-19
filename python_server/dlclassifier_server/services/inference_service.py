"""Inference service for deep learning models.

Supports:
- CUDA, Apple MPS, and CPU inference
- ONNX and PyTorch model loading with caching
- GPU-batched inference for maximum throughput
- FP16 mixed precision on CUDA
- torch.compile() for kernel fusion speedup
- Batch and pixel-level inference modes
- Binary buffer input for zero-copy tile transfer
- Multiple normalization strategies
"""

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .gpu_manager import GPUManager, get_gpu_manager
from ..utils.normalization import normalize as normalize_image
from ..utils.batchrenorm import replace_bn_with_batchrenorm
from ..utils.spatial import (
    get_spatial_divisor,
    pad_to_multiple,
    crop_to_original,
)

logger = logging.getLogger(__name__)

# Default GPU batch size for batched inference
DEFAULT_GPU_BATCH_SIZE = 16


def _write_pixel_outputs(
    output_dir: str,
    tile_ids: List[str],
    prob_maps: List[np.ndarray],
    output_format: str,
) -> Dict[str, str]:
    """Serialize per-tile inference outputs to disk.

    Two wire formats are supported; Java's ClassifierClient picks the
    matching reader based on the preference that was passed in.

    - ``prob_fp32`` (default): float32, shape ``(C, H, W)``, CHW order.
      Required for tile blending, multi-pass averaging, and overlay
      smoothing.
    - ``argmax_uint8``: uint8, shape ``(H, W)``, class indices only.
      ~20x smaller; skips softmax. Disables any downstream logic that
      requires per-class probabilities.

    See Phase 3c notes in the TrainingDialog preferences and
    docs/TINY_MODEL.md. Same ``.bin`` extension in both cases -- the
    caller knows which one it asked for.
    """
    output_paths = {}
    for tile_id, prob_map in zip(tile_ids, prob_maps):
        output_path = os.path.join(output_dir, "%s.bin" % tile_id)
        if output_format == "argmax_uint8":
            # prob_map shape: (C, H, W). Argmax over channel dim first.
            argmax_hw = np.argmax(prob_map, axis=0).astype(np.uint8)
            argmax_hw.tofile(output_path)
        else:
            prob_map.astype(np.float32).tofile(output_path)
        output_paths[tile_id] = output_path
    return output_paths


class InferenceService:
    """Service for running model inference.

    Features:
    - Automatic device selection (CUDA > MPS > CPU)
    - Model caching for efficient batch processing
    - GPU-batched inference (configurable batch size)
    - FP16 mixed precision on CUDA via torch.amp.autocast
    - torch.compile() for PyTorch 2.x kernel fusion
    - ONNX inference with appropriate execution providers
    - PyTorch inference as fallback
    - Multiple normalization strategies
    """

    def __init__(
        self,
        device: str = "auto",
        gpu_manager: Optional[GPUManager] = None
    ):
        """Initialize inference service.

        Args:
            device: Device to use ("cuda", "mps", "cpu", or "auto")
            gpu_manager: Optional GPUManager instance (uses singleton if not provided)
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()

        if device == "auto":
            self.device = self.gpu_manager.device
            self._device_str = self.gpu_manager.device_type
        else:
            self._device_str = device
            self.device = torch.device(device)

        self._model_cache: Dict[str, Tuple[str, Any]] = {}
        # Track which models have been compiled to avoid recompilation
        self._compiled_models: set = set()
        # Track which cached models have channels_last memory format so that
        # _infer_batch_spatial knows to convert input tensors to NHWC.
        self._channels_last_models: set = set()
        # Model paths where the static-shape ONNX variant does not match
        # runtime tile geometry; _load_model skips it and uses model.onnx.
        self._onnx_skip_static: set = set()
        # Phase 4: experimental provider flags. When flipped on mid-session,
        # already-cached sessions are unaffected (they stay on their prior
        # provider); newly-loaded models pick up the new provider list.
        self._use_tensorrt: bool = False
        self._use_int8: bool = False
        self._onnx_providers = self._get_onnx_providers()

        logger.info("InferenceService initialized on device: %s", self._device_str)

    def set_experimental_providers(
        self, use_tensorrt: bool = False, use_int8: bool = False
    ) -> None:
        """Toggle experimental ORT execution providers.

        Called from the Appose task wrappers with values sourced from
        the Java preference pane. On a provider change, evicts cached
        ONNX sessions so the next ``_load_model`` builds a fresh
        session with the new providers. Without this eviction, a user
        toggling TRT mid-session would see no change until they
        restarted the classifier (E.4 audit row).

        PyTorch-loaded models are kept -- provider flags only affect
        ORT, and rebuilding a PyTorch session is unnecessary (and
        would trigger a deepcopy of weights to GPU).
        """
        changed = (use_tensorrt != self._use_tensorrt
                   or use_int8 != self._use_int8)
        self._use_tensorrt = bool(use_tensorrt)
        self._use_int8 = bool(use_int8)
        if changed:
            self._onnx_providers = self._get_onnx_providers()
            evicted = 0
            for _k, _v in list(self._model_cache.items()):
                model_type, _ = _v
                if model_type == "onnx":
                    del self._model_cache[_k]
                    self._onnx_skip_static.discard(_k)
                    evicted += 1
            logger.info(
                "Experimental ORT providers updated: trt=%s int8=%s, "
                "evicted %d cached ONNX session(s). Next inference "
                "will reload with the new provider chain.",
                self._use_tensorrt, self._use_int8, evicted)

    def _get_onnx_providers(self, model_path: Optional[str] = None) -> List[Any]:
        """Get available ONNX execution providers based on device and flags.

        Phase 4: when ``self._use_tensorrt`` is on and
        ``TensorrtExecutionProvider`` is available, prepend it with a
        configured options dict (engine cache + optional INT8). Silent
        fallback to CUDAExecutionProvider when TRT is not installed.
        Users enable this via the "Experimental: TensorRT Inference"
        preference.

        Returns:
            List of execution providers. Items may be strings or
            ``(name, options_dict)`` tuples as accepted by ORT's
            ``InferenceSession(providers=...)``.
        """
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except ImportError:
            logger.warning("ONNX Runtime not available")
            return ["CPUExecutionProvider"]

        if self._device_str == "cuda" and "CUDAExecutionProvider" in available:
            # Phase 4 experimental TRT EP. Engine build is slow (seconds
            # to a minute); caching the engine next to the model avoids
            # paying that cost on every session start.
            if self._use_tensorrt and "TensorrtExecutionProvider" in available:
                cache_dir = os.path.expanduser(
                    "~/.dlclassifier/tensorrt_cache")
                os.makedirs(cache_dir, exist_ok=True)
                trt_opts = {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_fp16_enable": True,
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                }
                # Per-model prefix so retrained classifiers do not
                # collide in the shared cache dir. The prefix is a
                # short SHA1 of the model path AND metadata.json
                # mtime so a retrain into the same directory
                # produces a fresh prefix and the stale engine
                # (especially INT8 calibration cache) is not
                # silently reused. ORT 1.17+ honours
                # trt_engine_cache_prefix; on 1.16 this option is
                # ignored and engines pile up unprefixed -- the
                # pyproject/pixi pin must be bumped to >= 1.17 to
                # make this effective everywhere. Probe via
                # `ort.__version__` to log a warning on old builds.
                if model_path:
                    import hashlib
                    _h = hashlib.sha1()
                    _h.update(model_path.encode("utf-8"))
                    try:
                        _meta = os.path.join(model_path, "metadata.json")
                        if os.path.exists(_meta):
                            _h.update(str(int(
                                os.path.getmtime(_meta))).encode("utf-8"))
                    except OSError:
                        pass
                    _digest = _h.hexdigest()[:12]
                    trt_opts["trt_engine_cache_prefix"] = _digest
                    try:
                        _ort_ver = tuple(int(x) for x in
                                         ort.__version__.split(".")[:2])
                        if _ort_ver < (1, 17):
                            logger.warning(
                                "onnxruntime %s < 1.17 silently ignores "
                                "trt_engine_cache_prefix; engines from "
                                "different models may collide under one "
                                "prefix in %s. Bump the onnxruntime pin "
                                "to >= 1.17 for per-model isolation.",
                                ort.__version__, cache_dir,
                            )
                    except (AttributeError, ValueError):
                        pass
                if self._use_int8:
                    trt_opts["trt_int8_enable"] = True
                return [
                    ("TensorrtExecutionProvider", trt_opts),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            if self._use_tensorrt:
                logger.warning(
                    "Experimental TensorRT requested but "
                    "TensorrtExecutionProvider not in %s -- using "
                    "CUDAExecutionProvider instead.", available)
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self._device_str == "mps" and "CoreMLExecutionProvider" in available:
            # MPS devices can use CoreML for ONNX
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    # ==================== Public API ====================

    def run_batch(
        self,
        model_path: str,
        tiles: List[Dict[str, Any]],
        input_config: Dict[str, Any],
        gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
        use_amp: bool = True,
        compile_model: bool = True,
    ) -> Dict[str, List[float]]:
        """Run inference on a batch of tiles with GPU batching.

        Tiles are loaded, normalized, then processed in GPU batches for
        maximum throughput. Each GPU batch is a single forward pass.

        Args:
            model_path: Path to the model directory
            tiles: List of tile dictionaries with 'id' and 'data' keys
            input_config: Input configuration (channels, normalization)
            gpu_batch_size: Max tiles per GPU forward pass (default 16)
            use_amp: Use FP16 mixed precision on CUDA (default True)
            compile_model: Use torch.compile on PyTorch models (default True)

        Returns:
            Dict mapping tile_id to list of per-class probabilities
        """
        model_tuple = self._load_model(model_path, compile_model=compile_model)

        # Pre-process all tiles: load, normalize, select channels
        tile_ids = []
        preprocessed = []
        selected = input_config.get("selected_channels")

        for tile in tiles:
            tile_ids.append(tile["id"])
            img_array = self._load_tile_data(tile["data"])
            img_array = self._normalize(img_array, input_config)
            if selected:
                img_array = img_array[:, :, selected]
            preprocessed.append(img_array)

        # Batched inference -> list of per-tile probability maps (C, H, W)
        all_prob_maps = self._infer_batch_spatial(
            model_tuple, preprocessed,
            gpu_batch_size=gpu_batch_size, use_amp=use_amp
        )

        # Average spatial dimensions to get per-class probabilities
        predictions = {}
        for tile_id, prob_map in zip(tile_ids, all_prob_maps):
            class_probs = prob_map.mean(axis=(1, 2))
            predictions[tile_id] = class_probs.tolist()

        return predictions

    def run_pixel_inference(
        self,
        model_path: str,
        tiles: List[Dict[str, Any]],
        input_config: Dict[str, Any],
        output_dir: str,
        reflection_padding: int = 0,
        gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
        use_amp: bool = True,
        compile_model: bool = True,
        output_format: str = "prob_fp32",
    ) -> Dict[str, str]:
        """Run inference returning per-pixel probability maps saved as files.

        This mode is used for pixel classification (OBJECTS/OVERLAY output types)
        where full spatial probability maps are needed for tile blending.

        Args:
            model_path: Path to the model directory
            tiles: List of tile data dicts with 'id' and 'data' (file path or base64)
            input_config: Input configuration (channels, normalization)
            output_dir: Directory to save probability map files
            reflection_padding: Pixels of reflection padding to add around tiles
            gpu_batch_size: Max tiles per GPU forward pass (default 16)
            use_amp: Use FP16 mixed precision on CUDA (default True)
            compile_model: Use torch.compile on PyTorch models (default True)
            output_format: "prob_fp32" (default) writes (C,H,W) float32
                per tile; "argmax_uint8" writes (H,W) uint8 class indices
                (20x smaller, no softmax, but no blending/smoothing).

        Returns:
            Dict mapping tile_id to output file path
        """
        model_tuple = self._load_model(model_path, compile_model=compile_model)
        os.makedirs(output_dir, exist_ok=True)

        # Pre-process all tiles
        tile_ids = []
        preprocessed = []
        selected = input_config.get("selected_channels")

        for tile in tiles:
            tile_ids.append(tile["id"])
            img_array = self._load_tile_data(tile["data"])
            img_array = self._normalize(img_array, input_config)
            if selected:
                img_array = img_array[:, :, selected]
            preprocessed.append(img_array)

        # Batched inference with reflection padding
        all_prob_maps = self._infer_batch_spatial(
            model_tuple, preprocessed,
            reflection_padding=reflection_padding,
            gpu_batch_size=gpu_batch_size, use_amp=use_amp
        )

        # Save each probability map to disk
        output_paths = _write_pixel_outputs(
            output_dir, tile_ids, all_prob_maps, output_format
        )

        # Clear GPU cache after batch
        self._cleanup_after_inference()

        logger.info("Pixel inference complete: %d tiles -> %s (format=%s)",
                     len(output_paths), output_dir, output_format)
        return output_paths

    def run_batch_from_buffer(
        self,
        model_path: str,
        raw_bytes: bytes,
        tile_ids: List[str],
        tile_height: int,
        tile_width: int,
        num_channels: int,
        input_config: Dict[str, Any],
        gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
        use_amp: bool = True,
        compile_model: bool = True,
    ) -> Dict[str, List[float]]:
        """Run inference on tiles from a raw binary buffer.

        Tiles are packed as raw uint8 RGB in a single contiguous buffer,
        avoiding PNG/Base64 encode/decode overhead.

        Args:
            model_path: Path to the model directory
            raw_bytes: Concatenated raw tile pixels (uint8, HWC per tile)
            tile_ids: List of tile IDs matching the order in raw_bytes
            tile_height: Height of each tile in pixels
            tile_width: Width of each tile in pixels
            num_channels: Number of channels per tile
            input_config: Input configuration (channels, normalization)
            gpu_batch_size: Max tiles per GPU forward pass
            use_amp: Use FP16 mixed precision on CUDA
            compile_model: Use torch.compile on PyTorch models

        Returns:
            Dict mapping tile_id to list of per-class probabilities
        """
        model_tuple = self._load_model(model_path, compile_model=compile_model)

        # Determine dtype from metadata (default uint8 for backward compat)
        dtype_str = input_config.get("dtype", "uint8")
        np_dtype = np.float32 if dtype_str == "float32" else np.uint8
        bytes_per_element = 4 if np_dtype == np.float32 else 1

        # Reshape entire buffer into tiles at once
        num_tiles = len(tile_ids)
        expected_size = (num_tiles * tile_height * tile_width
                         * num_channels * bytes_per_element)
        if len(raw_bytes) != expected_size:
            raise ValueError(
                "Buffer size mismatch: expected %d bytes "
                "(%d tiles x %d x %d x %d x %d bytes/elem) but got %d bytes"
                % (expected_size, num_tiles, tile_height, tile_width,
                   num_channels, bytes_per_element, len(raw_bytes)))

        all_tiles = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(
            num_tiles, tile_height, tile_width, num_channels
        )
        if np_dtype != np.float32:
            all_tiles = all_tiles.astype(np.float32)

        # Normalize and select channels
        preprocessed = []
        selected = input_config.get("selected_channels")
        for i in range(num_tiles):
            img_array = self._normalize(all_tiles[i], input_config)
            if selected:
                img_array = img_array[:, :, selected]
            preprocessed.append(img_array)

        # Batched inference
        all_prob_maps = self._infer_batch_spatial(
            model_tuple, preprocessed,
            gpu_batch_size=gpu_batch_size, use_amp=use_amp
        )

        predictions = {}
        for tile_id, prob_map in zip(tile_ids, all_prob_maps):
            class_probs = prob_map.mean(axis=(1, 2))
            predictions[tile_id] = class_probs.tolist()

        return predictions

    def run_pixel_inference_from_buffer(
        self,
        model_path: str,
        raw_bytes: bytes,
        tile_ids: List[str],
        tile_height: int,
        tile_width: int,
        num_channels: int,
        input_config: Dict[str, Any],
        output_dir: str,
        reflection_padding: int = 0,
        gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
        use_amp: bool = True,
        compile_model: bool = True,
        output_format: str = "prob_fp32",
    ) -> Dict[str, str]:
        """Run pixel-level inference on tiles from a raw binary buffer.

        Args:
            model_path: Path to the model directory
            raw_bytes: Concatenated raw tile pixels (uint8 or float32, HWC per tile)
            tile_ids: List of tile IDs
            tile_height: Height of each tile in pixels
            tile_width: Width of each tile in pixels
            num_channels: Number of channels per tile
            input_config: Input configuration (channels, normalization, dtype)
            output_dir: Directory to save probability map files
            reflection_padding: Pixels of reflection padding
            gpu_batch_size: Max tiles per GPU forward pass
            use_amp: Use FP16 mixed precision on CUDA
            compile_model: Use torch.compile on PyTorch models

        Returns:
            Dict mapping tile_id to output file path
        """
        model_tuple = self._load_model(model_path, compile_model=compile_model)
        os.makedirs(output_dir, exist_ok=True)

        # Determine dtype from metadata (default uint8 for backward compat)
        dtype_str = input_config.get("dtype", "uint8")
        np_dtype = np.float32 if dtype_str == "float32" else np.uint8
        bytes_per_element = 4 if np_dtype == np.float32 else 1

        num_tiles = len(tile_ids)
        expected_size = (num_tiles * tile_height * tile_width
                         * num_channels * bytes_per_element)
        if len(raw_bytes) != expected_size:
            raise ValueError(
                "Buffer size mismatch: expected %d bytes but got %d bytes" % (
                    expected_size, len(raw_bytes)))

        all_tiles = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(
            num_tiles, tile_height, tile_width, num_channels
        )
        if np_dtype != np.float32:
            all_tiles = all_tiles.astype(np.float32)

        preprocessed = []
        selected = input_config.get("selected_channels")
        for i in range(num_tiles):
            img_array = self._normalize(all_tiles[i], input_config)
            if selected:
                img_array = img_array[:, :, selected]
            preprocessed.append(img_array)

        all_prob_maps = self._infer_batch_spatial(
            model_tuple, preprocessed,
            reflection_padding=reflection_padding,
            gpu_batch_size=gpu_batch_size, use_amp=use_amp
        )

        output_paths = _write_pixel_outputs(
            output_dir, tile_ids, all_prob_maps, output_format
        )

        self._cleanup_after_inference()

        logger.info("Pixel inference (binary) complete: %d tiles -> %s (format=%s)",
                     len(output_paths), output_dir, output_format)
        return output_paths

    def run_batch_files(
        self,
        model_path: str,
        tile_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Run inference on tile files.

        Args:
            model_path: Path to the model directory
            tile_paths: List of paths to tile image files

        Returns:
            List of result dictionaries with path and probabilities
        """
        model_tuple = self._load_model(model_path)

        # Pre-process all tiles
        preprocessed = []
        for path in tile_paths:
            img_array = self._load_image(path)
            img_array = img_array.astype(np.float32) / 255.0
            preprocessed.append(img_array)

        # Batched inference
        all_prob_maps = self._infer_batch_spatial(model_tuple, preprocessed)

        results = []
        for path, prob_map in zip(tile_paths, all_prob_maps):
            class_probs = prob_map.mean(axis=(1, 2))
            results.append({
                "path": path,
                "probabilities": class_probs.tolist()
            })

        return results

    # ==================== Batched Inference Core ====================

    def _infer_batch_spatial(
        self,
        model_tuple: Tuple[str, Any],
        images: List[np.ndarray],
        reflection_padding: int = 0,
        gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
        use_amp: bool = True,
        use_tta: bool = False,
    ) -> List[np.ndarray]:
        """Run batched inference on a list of preprocessed images.

        Groups images into GPU batches, runs a single forward pass per batch,
        and returns per-tile spatial probability maps.

        Args:
            model_tuple: (model_type, model) from _load_model
            images: List of normalized image arrays (H, W, C)
            reflection_padding: Pixels of reflection padding to add
            gpu_batch_size: Max images per forward pass
            use_amp: Use FP16 mixed precision on CUDA
            use_tta: Use Test-Time Augmentation (D4 transforms)

        Returns:
            List of probability maps, each with shape (C, H, W)
        """
        if not images:
            return []

        model_type, model = model_tuple

        # Wrap model with TTA if requested (PyTorch models only)
        if use_tta and model_type != "onnx":
            model = self._wrap_with_tta(model)

        # Apply reflection padding and convert HWC -> CHW for all tiles
        pad = reflection_padding
        tensors_nchw = []
        for img_array in images:
            if pad > 0:
                max_pad = min(img_array.shape[0], img_array.shape[1]) // 2
                effective_pad = min(pad, max_pad)
                if img_array.ndim == 2:
                    img_array = np.pad(img_array,
                                       ((effective_pad, effective_pad),
                                        (effective_pad, effective_pad)),
                                       mode='reflect')
                else:
                    img_array = np.pad(img_array,
                                       ((effective_pad, effective_pad),
                                        (effective_pad, effective_pad),
                                        (0, 0)),
                                       mode='reflect')

            if img_array.ndim == 2:
                img_array = img_array[..., np.newaxis]

            # HWC -> CHW, float32
            chw = img_array.transpose(2, 0, 1).astype(np.float32)
            tensors_nchw.append(chw)

        # Process in GPU batches
        all_probs = []

        for batch_start in range(0, len(tensors_nchw), gpu_batch_size):
            batch_end = min(batch_start + gpu_batch_size, len(tensors_nchw))
            batch_arrays = tensors_nchw[batch_start:batch_end]

            # Stack into (N, C, H, W) batch
            batch_np = np.stack(batch_arrays, axis=0)

            if model_type == "onnx":
                # Runtime shape check for static-shape ONNX. If the batch
                # shape doesn't match the baked-in shape, fall back to the
                # dynamic model.onnx variant by evicting this cached session
                # and marking this model to skip static on future loads.
                # See Phase 3b notes in
                # claude-reports/2026-04-17_input-size-divisibility.md.
                static_shape = getattr(
                    model, "_dlclassifier_onnx_static_shape", None)
                if static_shape is not None:
                    _, c0, h0, w0 = static_shape
                    _, cb, hb, wb = batch_np.shape
                    if (cb, hb, wb) != (c0, h0, w0):
                        logger.info(
                            "Batch shape %s differs from static ONNX shape "
                            "%s -- falling back to dynamic model.onnx",
                            (cb, hb, wb), (c0, h0, w0))
                        for key, val in list(self._model_cache.items()):
                            if val is model_tuple:
                                del self._model_cache[key]
                                self._onnx_skip_static.add(key)
                                model_tuple = self._load_model(key)
                                model_type, model = model_tuple
                                break
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: batch_np})
                batch_logits = outputs[0]  # (N, C, H, W)
            else:
                # PyTorch inference with optional AMP
                batch_tensor = torch.from_numpy(batch_np).to(self.device)
                # Match memory format when the model was converted to NHWC.
                if getattr(model, "_dlclassifier_channels_last", False):
                    batch_tensor = batch_tensor.contiguous(
                        memory_format=torch.channels_last
                    )
                # Per-architecture spatial auto-pad. See
                # claude-reports/2026-04-17_input-size-divisibility.md.
                divisor = int(
                    getattr(model, "_dlclassifier_spatial_divisor", 1)
                )
                batch_tensor, pad_h, pad_w = pad_to_multiple(
                    batch_tensor, divisor
                )
                with torch.no_grad():
                    if use_amp and self._device_str == "cuda":
                        # Prefer BF16 on Ampere+ GPUs, fall back to FP16
                        amp_dtype = (torch.bfloat16
                                     if torch.cuda.is_bf16_supported()
                                     else torch.float16)
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            outputs = model(batch_tensor)
                    else:
                        outputs = model(batch_tensor)
                    outputs = crop_to_original(outputs, pad_h, pad_w)
                    batch_logits = outputs.cpu().float().numpy()

            # Softmax per-tile and collect
            for i in range(batch_logits.shape[0]):
                probs = self._softmax(batch_logits[i])
                # Crop reflection padding from output
                if pad > 0:
                    effective_pad = min(pad,
                                        min(images[batch_start + i].shape[0],
                                            images[batch_start + i].shape[1]) // 2)
                    if effective_pad > 0:
                        probs = probs[:, effective_pad:-effective_pad,
                                      effective_pad:-effective_pad]
                all_probs.append(probs)

        return all_probs

    # ==================== Single-tile Appose API ====================

    def infer_single_tile(
        self,
        model_path: str,
        tile_array: np.ndarray,
        input_config: Dict[str, Any],
        reflection_padding: int = 0,
        use_tta: bool = False,
    ) -> np.ndarray:
        """Run inference on a single pre-loaded tile array.

        Optimized for Appose shared-memory calls where the tile is already
        loaded as a numpy array (no base64/file I/O needed).

        Args:
            model_path: Path to model directory
            tile_array: numpy array (H, W, C), float32, already normalized
            input_config: dict with normalization config
            reflection_padding: pixels of padding
            use_tta: Use Test-Time Augmentation (D4 transforms)

        Returns:
            numpy array (C, H, W), float32 probability map
        """
        model_tuple = self._load_model(model_path)
        prob_maps = self._infer_batch_spatial(
            model_tuple,
            [tile_array],
            reflection_padding=reflection_padding,
            gpu_batch_size=1,
            use_tta=use_tta
        )
        return prob_maps[0]  # (C, H, W)

    # ==================== Single-tile inference (legacy compat) ====================

    def _infer_tile(self, model_tuple: Tuple[str, Any], img_array: np.ndarray) -> np.ndarray:
        """Run inference on a single tile, returning per-class average probabilities.

        Args:
            model_tuple: (model_type, model) from _load_model
            img_array: Image array (H, W, C) normalized

        Returns:
            Per-class average probabilities
        """
        results = self._infer_batch_spatial(model_tuple, [img_array])
        return results[0].mean(axis=(1, 2))

    def _infer_tile_spatial(
        self,
        model_tuple: Tuple[str, Any],
        img_array: np.ndarray,
        reflection_padding: int = 0
    ) -> np.ndarray:
        """Run inference on a single tile, returning full spatial probability map.

        Args:
            model_tuple: (model_type, model) from _load_model
            img_array: Image array (H, W, C) normalized
            reflection_padding: Pixels of reflection padding

        Returns:
            Probability map with shape (C, H, W) where C is num_classes
        """
        results = self._infer_batch_spatial(
            model_tuple, [img_array],
            reflection_padding=reflection_padding
        )
        return results[0]

    # ==================== TTA Support ====================

    @staticmethod
    def _wrap_with_tta(model):
        """Wrap a PyTorch model with Test-Time Augmentation (D4 transforms).

        Uses ttach library for horizontal/vertical flips and 90-degree rotations.
        The wrapper averages predictions across all augmented views.

        Args:
            model: PyTorch segmentation model

        Returns:
            TTA-wrapped model, or original model if ttach is not available
        """
        try:
            import ttach as tta
            transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ])
            wrapped = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
            logger.debug("TTA: wrapped model with D4 transforms")
            return wrapped
        except ImportError:
            logger.warning("ttach not installed -- TTA disabled. "
                          "Install with: pip install ttach")
            return model

    # ==================== Model Loading ====================

    def _cleanup_after_inference(self) -> None:
        """Clear GPU cache after inference batch."""
        self.gpu_manager.clear_cache()

    def _load_model(
        self,
        model_path: str,
        compile_model: bool = True,
    ) -> Tuple[str, Any]:
        """Load a model from disk.

        Prefers ONNX models for inference efficiency, falls back to PyTorch.
        Applies torch.compile() to PyTorch models on CUDA for kernel fusion.

        Args:
            model_path: Path to model directory
            compile_model: Whether to apply torch.compile (PyTorch + CUDA only)

        Returns:
            Tuple of (model_type, model) where model_type is "onnx" or "pytorch"
        """
        if model_path in self._model_cache:
            model_tuple = self._model_cache[model_path]
            # Apply torch.compile if requested and not yet compiled
            if (compile_model and model_tuple[0] == "pytorch"
                    and model_path not in self._compiled_models):
                self._try_compile_model(model_path, model_tuple)
            return self._model_cache[model_path]

        model_dir = Path(model_path)

        # Prefer the static-shape ONNX when present. It bakes fixed H/W into
        # the graph (faster on ORT CPU/CUDA; required by Phase 4 TensorRT).
        # If the runtime tile shape does not match the baked-in shape, the
        # ORT session.run() call will raise and _infer_batch_spatial falls
        # back to the dynamic model on a re-load. See
        # claude-reports/2026-04-17_input-size-divisibility.md for shape
        # coordination notes.
        # Phase 4: when INT8 TRT is enabled, prefer the BN-folded static
        # ONNX so TRT's calibrator can fuse conv+BN cleanly. Falls through
        # to the ordinary static variant if the BN-folded file is absent
        # (older models exported before the Phase 4 upgrade).
        onnx_static_bn_path = model_dir / "model_static_bn.onnx"
        onnx_static_path = model_dir / "model_static.onnx"
        onnx_path = model_dir / "model.onnx"
        def _try_load_onnx(path: Path, label: str, static_shape=None):
            try:
                logger.info("Loading %s ONNX model from %s", label, path)
                import onnxruntime as ort

                # Build providers with a per-model TRT cache prefix so
                # engines from different classifiers do not collide in
                # the shared cache dir. Fallback to self._onnx_providers
                # (no prefix) if model_path is unavailable.
                per_model_providers = self._get_onnx_providers(
                    model_path=model_path)
                session = ort.InferenceSession(
                    str(path),
                    providers=per_model_providers
                )
                # Tag the session so _infer_batch_spatial can tell whether
                # it is safe to feed an arbitrary-shaped batch.
                session._dlclassifier_onnx_variant = label
                session._dlclassifier_onnx_static_shape = static_shape
                self._model_cache[model_path] = ("onnx", session)
                return ("onnx", session)
            except Exception as e:
                logger.warning(
                    "%s ONNX loading failed (%s): %s", label, path, e)
                return None

        if (self._use_tensorrt and self._use_int8
                and onnx_static_bn_path.exists()
                and model_path not in self._onnx_skip_static):
            try:
                with open(model_dir / "metadata.json") as f:
                    meta = json.load(f)
                bn_shape = meta.get(
                    "onnx_variants", {}).get(
                    "static_bn", {}).get("shape")
            except Exception:
                bn_shape = None
            loaded = _try_load_onnx(
                onnx_static_bn_path, "static-BN INT8",
                static_shape=bn_shape)
            if loaded is not None:
                return loaded

        if onnx_static_path.exists() and model_path not in self._onnx_skip_static:
            # Read baked-in shape from metadata so runtime can check matches.
            static_shape = None
            try:
                with open(model_dir / "metadata.json") as f:
                    meta = json.load(f)
                variants = meta.get("onnx_variants", {}) or {}
                static_shape = variants.get("static", {}).get("shape")
            except Exception:
                static_shape = None
            loaded = _try_load_onnx(
                onnx_static_path, "static", static_shape=static_shape)
            if loaded is not None:
                return loaded

        if onnx_path.exists():
            loaded = _try_load_onnx(onnx_path, "dynamic")
            if loaded is not None:
                return loaded

        # Try PyTorch
        pt_path = model_dir / "model.pt"
        if pt_path.exists():
            logger.info("Loading PyTorch model from %s", pt_path)

            # Load and validate metadata
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    "Classifier metadata not found: %s. "
                    "The classifier directory may be incomplete."
                    % metadata_path)
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    "Classifier metadata is corrupt (%s): %s"
                    % (metadata_path, e))

            # Create model architecture
            import segmentation_models_pytorch as smp

            arch = metadata.get("architecture", {})
            input_config = metadata.get("input_config", {})
            model_type = arch.get("type", "unet")
            encoder_name = arch.get("backbone", "resnet34")
            # Use effective_input_channels when available (written by v0.3.8+
            # models). This is the actual model input size including context
            # channel doubling. Fall back to manual computation for older models.
            context_scale = arch.get("context_scale", 1)
            if "effective_input_channels" in arch:
                metadata_channels = arch["effective_input_channels"]
            else:
                metadata_channels = arch.get("input_channels",
                                             input_config.get("num_channels", 3))
                if context_scale > 1:
                    metadata_channels = metadata_channels * 2
            num_classes = len(metadata.get("classes", [{"index": 0}, {"index": 1}]))

            # Load checkpoint first so we can detect actual in_channels from
            # the encoder's first conv layer -- this is ground truth when
            # metadata is wrong (e.g. Java overwrites with incorrect value).
            try:
                state_dict = torch.load(
                    pt_path, map_location=self.device, weights_only=True)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load model weights from %s: %s. "
                    "File may be corrupt." % (pt_path, e))

            # Detect MAE checkpoint accidentally used as classifier
            mae_keys = [k for k in state_dict if k.startswith("mae.")]
            if mae_keys:
                raise RuntimeError(
                    "This model file contains MAE pretraining weights "
                    "(not a trained classifier). Use 'Continue from model' "
                    "during training to load MAE weights as initialization, "
                    "then train a segmentation classifier.")

            # Detect actual in_channels from checkpoint weights
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
            # base SMP encoder names. The checkpoint already contains the
            # trained weights so we only need the right architecture.
            smp_encoder_name = encoder_name
            try:
                from .pretrained_models import PretrainedModelsService
                if encoder_name in PretrainedModelsService.HISTOLOGY_ENCODERS:
                    smp_encoder_name = (
                        PretrainedModelsService.HISTOLOGY_ENCODERS
                        [encoder_name][0])
                    logger.info(
                        "Resolved custom encoder '%s' -> '%s' for SMP",
                        encoder_name, smp_encoder_name)
            except ImportError:
                pass

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

            # MuViT transformer: use dedicated factory
            if model_type == "muvit":
                from .muvit_model import create_muvit_model
                model = create_muvit_model(
                    architecture=arch,
                    num_channels=num_channels,
                    num_classes=num_classes,
                )
            elif model_type == "tiny-unet":
                from ..models.tiny_unet import TinyUNet
                model = TinyUNet(
                    in_channels=num_channels,
                    n_classes=num_classes,
                    base=int(arch.get("base", 16)),
                    depth=int(arch.get("depth", 4)),
                    norm=str(arch.get("norm", "brn")),
                )
                logger.info(
                    "Reconstructed Tiny UNet (base=%d, depth=%d, norm=%s)",
                    int(arch.get("base", 16)),
                    int(arch.get("depth", 4)),
                    arch.get("norm", "brn"),
                )
            elif model_type == "fast-pretrained":
                # SMP U-Net with a small mobile encoder and scaled decoder.
                # encoder_weights=None because the saved state_dict already
                # holds the (possibly fine-tuned) weights.
                fp_decoder = arch.get("decoder_channels", [128, 64, 32, 16, 8])
                model = smp.Unet(
                    encoder_name=smp_encoder_name,
                    encoder_weights=None,
                    in_channels=num_channels,
                    classes=num_classes,
                    decoder_channels=fp_decoder,
                )
                logger.info(
                    "Reconstructed Fast Pretrained UNet (encoder=%s, decoder=%s)",
                    smp_encoder_name, fp_decoder,
                )
            else:
                model_cls = model_map.get(model_type, smp.Unet)
                model = model_cls(
                    encoder_name=smp_encoder_name,
                    encoder_weights=None,
                    in_channels=num_channels,
                    classes=num_classes
                )

            # Auto-detect BatchRenorm from state dict keys (rmax/dmax are
            # unique to BatchRenorm2d). More robust than metadata flag which
            # may be lost when Java overwrites metadata.json.
            has_batchrenorm = any(
                k.endswith('.rmax') or k.endswith('.dmax')
                for k in state_dict)
            if has_batchrenorm:
                replace_bn_with_batchrenorm(model)
                logger.info("Auto-detected BatchRenorm from state dict keys")

            # Pre-validate shapes to produce clear error messages
            model_state = model.state_dict()
            shape_mismatches = []
            for key in state_dict:
                if (key in model_state
                        and state_dict[key].shape != model_state[key].shape):
                    shape_mismatches.append(
                        "%s: checkpoint=%s model=%s" % (
                            key, list(state_dict[key].shape),
                            list(model_state[key].shape)))

            if shape_mismatches:
                detail = "\n  ".join(shape_mismatches[:5])
                extra = ""
                if len(shape_mismatches) > 5:
                    extra = ("\n  ... and %d more"
                             % (len(shape_mismatches) - 5))
                raise RuntimeError(
                    "Model architecture mismatch: %d weights have "
                    "incompatible shapes.\n  %s%s\n"
                    "The model.pt may be from a different training run "
                    "or architecture configuration."
                    % (len(shape_mismatches), detail, extra))

            missing = set(model_state.keys()) - set(state_dict.keys())
            if missing and len(missing) / max(1, len(model_state)) > 0.5:
                raise RuntimeError(
                    "Model architecture mismatch: %d/%d expected weights "
                    "missing from checkpoint. The model.pt does not match "
                    "the architecture in metadata.json."
                    % (len(missing), len(model_state)))

            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                err = str(e)
                if "size mismatch" in err or "Missing key" in err:
                    raise RuntimeError(
                        "Model weights do not match architecture. "
                        "The classifier may need to be retrained. "
                        "Detail: %s" % err) from None
                raise

            model = model.to(self.device)
            model.eval()

            # channels_last memory format: improves throughput on Tensor Cores
            # for conv-heavy models on Ampere+ GPUs.  Skip for MuViT (transformer,
            # no benefit) and for TinyUNet with BatchRenorm (BRN internal reshapes
            # silently undo the NHWC propagation).
            if self._apply_channels_last(model, model_type, arch):
                self._channels_last_models.add(model_path)

            # Stash the architecture's spatial divisor on the model so
            # _infer_batch_spatial can auto-pad inputs. See
            # claude-reports/2026-04-17_input-size-divisibility.md.
            try:
                setattr(
                    model,
                    "_dlclassifier_spatial_divisor",
                    int(get_spatial_divisor(model_type, arch)),
                )
            except Exception as e:
                logger.debug("spatial divisor tagging failed: %s", e)
                setattr(model, "_dlclassifier_spatial_divisor", 32)

            self._model_cache[model_path] = ("pytorch", model)

            # Apply torch.compile if requested
            if compile_model:
                self._try_compile_model(model_path, ("pytorch", model))

            return self._model_cache[model_path]

        raise FileNotFoundError("No model found at %s" % model_path)

    def _apply_channels_last(self, model, model_type: str,
                             arch: Dict[str, Any]) -> bool:
        """Convert a loaded PyTorch model to channels_last memory format.

        Gains 10-30% throughput on convnets on Ampere+ GPUs with Tensor
        Cores.  Gate carefully: transformers (MuViT) do not benefit and
        BatchRenorm silently undoes NHWC propagation via internal reshape
        operations (agent report B2).

        Args:
            model: The loaded nn.Module
            model_type: "unet", "muvit", "tiny-unet", "fast-pretrained", ...
            arch: architecture dict from metadata (used to detect BRN use)

        Returns:
            True if channels_last was successfully applied, False otherwise.
            The caller records the result so input tensors can match format.
        """
        if self._device_str != "cuda":
            return False
        # Transformers -- NHWC offers no benefit and can hurt.
        if model_type == "muvit":
            return False
        # ANY BatchRenorm module (including SMP UNets that were
        # BRN-converted at training time by replace_bn_with_batchrenorm)
        # silently undoes NHWC propagation via internal reshape ops.
        # Before this check the gate looked only at
        # model_type == "tiny-unet", so SMP + BRN models slipped
        # through and paid for the format conversion without the
        # speedup. See E.5 audit row.
        try:
            from ..utils.batchrenorm import BatchRenorm2d
            if any(isinstance(m, BatchRenorm2d) for m in model.modules()):
                logger.debug(
                    "channels_last skipped: model contains BatchRenorm "
                    "(model_type=%s)", model_type,
                )
                return False
        except ImportError:
            # If the utility is missing we err on the side of safety
            # and keep the legacy tiny-unet check below.
            pass
        # Fallback to the legacy metadata-based check in case the
        # state_dict path did not contain rmax/dmax but the norm param
        # was still set to "brn" (older TinyUNet checkpoints).
        if model_type == "tiny-unet" and str(arch.get("norm", "brn")) == "brn":
            logger.debug(
                "channels_last skipped: tiny-unet with BRN norm param (%s)",
                arch.get("norm", "brn"),
            )
            return False
        try:
            model.to(memory_format=torch.channels_last)
            # Tag for _infer_batch_spatial so input tensors match the layout.
            setattr(model, "_dlclassifier_channels_last", True)
            logger.info(
                "channels_last memory format enabled for %s inference",
                model_type,
            )
            return True
        except Exception as e:
            logger.debug(
                "channels_last not applied (%s): %s", model_type, e
            )
            return False

    def _try_compile_model(self, model_path: str,
                           model_tuple: Tuple[str, Any]) -> None:
        """Attempt to apply torch.compile() to a PyTorch model.

        Only applies on CUDA with PyTorch 2.x+ when Triton is available.
        The Inductor backend used by torch.compile requires Triton for
        GPU kernel generation. Triton is Linux-only; on Windows,
        torch.compile() wraps the model lazily but crashes on the first
        forward pass with TritonMissing. We check upfront to avoid that.

        Args:
            model_path: Cache key for the model
            model_tuple: (model_type, model) tuple
        """
        if self._device_str != "cuda":
            return
        if not hasattr(torch, "compile"):
            return

        # Triton is required by the Inductor backend for GPU kernel generation.
        # It is Linux-only -- not available on Windows. Without this check,
        # torch.compile() wraps the model successfully (lazy), but the first
        # model(tensor) call raises TritonMissing at code-generation time.
        try:
            import triton  # noqa: F401
        except ImportError:
            logger.info("Triton not available -- skipping torch.compile() "
                        "(eager mode will be used)")
            self._compiled_models.add(model_path)  # Don't retry
            return

        model_type, model = model_tuple
        if model_type != "pytorch":
            return

        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            self._model_cache[model_path] = ("pytorch", compiled)
            self._compiled_models.add(model_path)
            logger.info("Model compiled with torch.compile() for %s", model_path)
        except Exception as e:
            logger.warning("torch.compile failed, using eager mode: %s", e)
            self._compiled_models.add(model_path)  # Don't retry

    # ==================== Utilities ====================

    def _softmax(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Compute softmax.

        Args:
            x: Input logits
            axis: Axis along which to compute softmax

        Returns:
            Softmax probabilities
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _load_tile_data(self, tile_data: str) -> np.ndarray:
        """Load tile image from a data URL or file path.

        Args:
            tile_data: Either a data URL (data:image/png;base64,...) or a
                       filesystem path to an image file.

        Returns:
            Image as numpy array

        Raises:
            ValueError: If tile_data is neither a data URL nor an existing file
        """
        if tile_data.startswith("data:"):
            return self._decode_base64(tile_data)

        if os.path.isfile(tile_data):
            return self._load_image(tile_data)

        raise ValueError(
            "Tile data is neither a data URL (data:...) nor an existing file: "
            "%s..." % tile_data[:80]
        )

    def _decode_base64(self, data: str) -> np.ndarray:
        """Decode base64 image data.

        Args:
            data: Base64 encoded image (with or without data URL prefix)

        Returns:
            Image as numpy array
        """
        # Remove data URL prefix if present
        if data.startswith("data:"):
            data = data.split(",")[1]

        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img, dtype=np.float32)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file.

        Supports standard image formats via PIL as well as multi-channel
        TIFF files (>4 bands, 16-bit) via tifffile, and raw float32 files
        exported by the Java training pipeline.

        Args:
            path: Path to image file

        Returns:
            Image as numpy array (HWC float32)
        """
        if path.endswith('.raw'):
            return self._load_raw_tile(path)
        if path.endswith(('.tif', '.tiff')):
            try:
                import tifffile
                arr = tifffile.imread(path).astype(np.float32)
                # tifffile may return (C,H,W) for multi-channel; convert to HWC
                if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
                    arr = arr.transpose(1, 2, 0)
                elif arr.ndim == 2:
                    arr = arr[..., np.newaxis]
                return arr
            except ImportError:
                pass  # fall through to PIL
        img = Image.open(path)
        return np.array(img, dtype=np.float32)

    @staticmethod
    def _load_raw_tile(path: str) -> np.ndarray:
        """Load a raw float32 tile exported by the Java training pipeline.

        File format: 12-byte header (3x int32: height, width, channels)
        followed by H*W*C float32 values in HWC order, all little-endian.
        """
        with open(path, 'rb') as f:
            header = np.frombuffer(f.read(12), dtype=np.int32)
            h, w, c = int(header[0]), int(header[1]), int(header[2])
            data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(h, w, c)

    def _normalize(self, img: np.ndarray, input_config: Dict[str, Any]) -> np.ndarray:
        """Normalize image data.

        Delegates to shared normalization module which supports both
        per-tile and precomputed image-level statistics.

        Args:
            img: Input image array
            input_config: Configuration with normalization settings

        Returns:
            Normalized image array
        """
        return normalize_image(img, input_config)

    def clear_model_cache(self) -> None:
        """Clear the model cache to free memory."""
        self._model_cache.clear()
        self._compiled_models.clear()
        self._cleanup_after_inference()
        logger.info("Model cache cleared")
