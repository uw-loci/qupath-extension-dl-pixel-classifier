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

logger = logging.getLogger(__name__)

# Default GPU batch size for batched inference
DEFAULT_GPU_BATCH_SIZE = 16


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
        self._onnx_providers = self._get_onnx_providers()

        logger.info("InferenceService initialized on device: %s", self._device_str)

    def _get_onnx_providers(self) -> List[str]:
        """Get available ONNX execution providers based on device.

        Returns:
            List of ONNX execution provider names
        """
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except ImportError:
            logger.warning("ONNX Runtime not available")
            return ["CPUExecutionProvider"]

        if self._device_str == "cuda" and "CUDAExecutionProvider" in available:
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
        output_paths = {}
        for tile_id, prob_map in zip(tile_ids, all_prob_maps):
            output_path = os.path.join(output_dir, "%s.bin" % tile_id)
            prob_map.astype(np.float32).tofile(output_path)
            output_paths[tile_id] = output_path

        # Clear GPU cache after batch
        self._cleanup_after_inference()

        logger.info("Pixel inference complete: %d tiles -> %s",
                     len(output_paths), output_dir)
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

        output_paths = {}
        for tile_id, prob_map in zip(tile_ids, all_prob_maps):
            output_path = os.path.join(output_dir, "%s.bin" % tile_id)
            prob_map.astype(np.float32).tofile(output_path)
            output_paths[tile_id] = output_path

        self._cleanup_after_inference()

        logger.info("Pixel inference (binary) complete: %d tiles -> %s",
                     len(output_paths), output_dir)
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
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: batch_np})
                batch_logits = outputs[0]  # (N, C, H, W)
            else:
                # PyTorch inference with optional AMP
                batch_tensor = torch.from_numpy(batch_np).to(self.device)
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

        # Try ONNX first
        onnx_path = model_dir / "model.onnx"
        if onnx_path.exists():
            try:
                logger.info("Loading ONNX model from %s", onnx_path)
                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(onnx_path),
                    providers=self._onnx_providers
                )
                self._model_cache[model_path] = ("onnx", session)
                return ("onnx", session)
            except Exception as e:
                logger.warning("ONNX loading failed, trying PyTorch: %s", e)

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

            self._model_cache[model_path] = ("pytorch", model)

            # Apply torch.compile if requested
            if compile_model:
                self._try_compile_model(model_path, ("pytorch", model))

            return self._model_cache[model_path]

        raise FileNotFoundError("No model found at %s" % model_path)

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
