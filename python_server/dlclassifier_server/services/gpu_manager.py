"""GPU management service with MPS support and memory monitoring.

Provides:
- Device detection (CUDA > MPS > CPU priority)
- Memory usage monitoring (CUDA)
- Cache clearing for GPU memory management
- Model memory estimation
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and provides device information.

    Features:
    - Automatic device detection (CUDA, Apple MPS, or CPU fallback)
    - GPU memory monitoring and cache clearing (CUDA)
    - Model memory footprint estimation
    - Detailed device info for API responses

    Priority: CUDA > MPS > CPU
    """

    def __init__(self):
        """Initialize GPU manager and detect available devices."""
        self._device_type: str = "cpu"
        self._device = None
        self._device_name: str = "CPU"
        self._memory_mb: int = 0
        self._cuda_version: Optional[str] = None
        self._compute_capability: Optional[tuple] = None

        self._detect_device()

    def _detect_device(self) -> None:
        """Detect available GPU device with priority CUDA > MPS > CPU."""
        try:
            import torch

            # Check CUDA first (highest priority)
            if torch.cuda.is_available():
                self._device_type = "cuda"
                self._device = torch.device("cuda")
                self._device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                self._memory_mb = props.total_memory // (1024 * 1024)
                self._cuda_version = torch.version.cuda
                self._compute_capability = torch.cuda.get_device_capability(0)
                logger.info(
                    f"CUDA GPU detected: {self._device_name} "
                    f"({self._memory_mb} MB, CUDA {self._cuda_version})"
                )
                return

            # Check Apple MPS (Metal Performance Shaders)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device_type = "mps"
                self._device = torch.device("mps")
                self._device_name = "Apple Silicon (MPS)"
                logger.info("Apple MPS device detected")
                return

            # CPU fallback
            self._device_type = "cpu"
            self._device = torch.device("cpu")
            self._device_name = "CPU"
            logger.info("No GPU available, using CPU")

        except ImportError:
            logger.warning("PyTorch not installed, GPU detection unavailable")
            self._device_type = "cpu"
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self._device_type = "cpu"

    @property
    def device_type(self) -> str:
        """Get device type string ('cuda', 'mps', or 'cpu')."""
        return self._device_type

    @property
    def device(self):
        """Get torch.device object."""
        if self._device is None:
            import torch
            self._device = torch.device(self._device_type)
        return self._device

    def is_available(self) -> bool:
        """Check if GPU is available (CUDA or MPS)."""
        return self._device_type != "cpu"

    def get_device_name(self) -> str:
        """Get GPU device name."""
        return self._device_name

    def get_memory_mb(self) -> int:
        """Get total GPU memory in MB (CUDA only, 0 for others)."""
        return self._memory_mb

    def get_cuda_version(self) -> Optional[str]:
        """Get CUDA version (None if not CUDA)."""
        return self._cuda_version

    def get_device(self) -> str:
        """Get the device string for PyTorch."""
        return self._device_type

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory usage.

        Returns:
            Dict with memory information:
            - For CUDA: allocated_mb, reserved_mb, max_allocated_mb, total_mb
            - For MPS: device info (limited introspection)
            - For CPU: system RAM indicator
        """
        if self._device_type == "cuda":
            try:
                import torch
                return {
                    "device": "cuda",
                    "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                    "total_mb": self._memory_mb,
                    "utilization_percent": (
                        100 * torch.cuda.memory_allocated() /
                        (self._memory_mb * 1024**2)
                    ) if self._memory_mb > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory info: {e}")
                return {"device": "cuda", "error": str(e)}

        elif self._device_type == "mps":
            # MPS has limited memory introspection
            return {
                "device": "mps",
                "info": "Apple MPS - limited memory introspection available"
            }

        return {
            "device": "cpu",
            "info": "Using system RAM"
        }

    def clear_cache(self) -> None:
        """Clear GPU memory cache.

        Frees cached memory to help prevent out-of-memory errors.
        Safe to call on any device type.
        """
        try:
            import torch

            if self._device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logger.debug("Cleared CUDA memory cache")

            elif self._device_type == "mps":
                # MPS cache clearing (PyTorch 2.0+)
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    logger.debug("Cleared MPS memory cache")

        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")

    def estimate_model_memory(self, model) -> float:
        """Estimate model memory footprint in MB.

        Args:
            model: PyTorch nn.Module

        Returns:
            Estimated memory in megabytes
        """
        try:
            param_size = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            )
            buffer_size = sum(
                b.numel() * b.element_size()
                for b in model.buffers()
            )
            total_bytes = param_size + buffer_size
            return total_bytes / (1024**2)
        except Exception as e:
            logger.warning(f"Failed to estimate model memory: {e}")
            return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU info for API response.

        Returns:
            Dict with device information suitable for /health/gpu endpoint
        """
        info = {
            "available": self._device_type != "cpu",
            "device_type": self._device_type,
            "device_string": str(self.device),
            "name": self._device_name
        }

        if self._device_type == "cuda":
            info.update({
                "cuda_version": self._cuda_version,
                "compute_capability": self._compute_capability,
                "total_memory_mb": self._memory_mb,
                **self.get_memory_info()
            })

        elif self._device_type == "mps":
            info.update({
                "mps_available": True,
                "backend": "Metal Performance Shaders"
            })

        return info

    def log_memory_status(self, prefix: str = "",
                          include_peak: bool = False) -> None:
        """Log current memory status (useful during training).

        Args:
            prefix: Optional prefix for log message (e.g., epoch number)
            include_peak: If True, also log peak and reserved memory
        """
        if self._device_type == "cuda":
            mem_info = self.get_memory_info()
            allocated = mem_info.get('allocated_mb', 0)
            total = mem_info.get('total_mb', 0)
            pct = mem_info.get('utilization_percent', 0)
            if include_peak:
                peak = mem_info.get('max_allocated_mb', 0)
                reserved = mem_info.get('reserved_mb', 0)
                logger.info(
                    f"{prefix}GPU Memory: "
                    f"{allocated:.0f}MB allocated, "
                    f"{peak:.0f}MB peak, "
                    f"{reserved:.0f}MB reserved / "
                    f"{total:.0f}MB total "
                    f"({100 * peak / total:.0f}% peak utilization)"
                    if total > 0 else
                    f"{prefix}GPU Memory: {allocated:.0f}MB allocated"
                )
            else:
                logger.info(
                    f"{prefix}GPU Memory: "
                    f"{allocated:.1f}MB allocated / "
                    f"{total:.0f}MB total "
                    f"({pct:.1f}%)"
                )
        elif self._device_type == "mps":
            logger.info(f"{prefix}Using Apple MPS (memory introspection limited)")
        else:
            logger.info(f"{prefix}Using CPU")

    def get_peak_allocated_mb(self) -> float:
        """Get peak GPU memory allocated since last reset (CUDA only).

        Returns:
            Peak allocated memory in MB, or 0 for non-CUDA devices.
        """
        if self._device_type == "cuda":
            try:
                import torch
                return torch.cuda.max_memory_allocated() / (1024**2)
            except Exception:
                return 0.0
        return 0.0


# Singleton instance for shared access
_gpu_manager_instance: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get or create the singleton GPUManager instance.

    Returns:
        Shared GPUManager instance
    """
    global _gpu_manager_instance
    if _gpu_manager_instance is None:
        _gpu_manager_instance = GPUManager()
    return _gpu_manager_instance
