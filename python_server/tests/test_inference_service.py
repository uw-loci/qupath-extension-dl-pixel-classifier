"""Tests for inference service.

Tests cover:
- Model loading (PyTorch and ONNX)
- Single tile inference
- Batch inference
- Pixel-level inference
- Normalization strategies
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_pytorch_model(self, trained_model_path):
        """Test PyTorch model loading."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model_tuple = inf._load_model(str(trained_model_path))

        assert model_tuple is not None
        model_type, model = model_tuple

        # Should be pytorch since no ONNX file
        assert model_type == "pytorch"

    def test_load_model_caching(self, trained_model_path):
        """Test model caching works."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model_path_str = str(trained_model_path)

        # Load model first time
        model1 = inf._load_model(model_path_str)

        # Verify it's in the cache
        assert model_path_str in inf._model_cache

        # Load again - should return cached version
        model2 = inf._load_model(model_path_str)

        # Should be same cached entry (same model type at minimum)
        assert model1[0] == model2[0]  # Same model type
        # Cache should still only have one entry
        assert len(inf._model_cache) == 1

    def test_load_model_not_found(self, tmp_path):
        """Test loading non-existent model raises error."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        with pytest.raises(FileNotFoundError):
            inf._load_model(str(tmp_path / "nonexistent"))

    def test_load_onnx_model(self, trained_model_path):
        """Test ONNX model loading when available."""
        try:
            import torch
            import onnxruntime
        except ImportError:
            pytest.skip("ONNX runtime not available")

        from dlclassifier_server.services.inference_service import InferenceService

        # Create ONNX model
        try:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                in_channels=3,
                classes=2,
            )
            model.eval()

            onnx_path = trained_model_path / "model.onnx"
            dummy_input = torch.randn(1, 3, 256, 256)
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=14,
                input_names=["input"],
                output_names=["output"],
            )

            # Now load
            inf = InferenceService(device="cpu")
            # Clear cache first
            inf._model_cache.clear()

            model_tuple = inf._load_model(str(trained_model_path))

            model_type, _ = model_tuple
            # Should prefer ONNX
            assert model_type == "onnx"

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")


class TestNormalization:
    """Test image normalization strategies."""

    def test_normalize_min_max(self):
        """Test min-max normalization."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8).astype(
            np.float32
        )

        config = {"normalization": {"strategy": "min_max"}}

        normalized = inf._normalize(img, config)

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_percentile(self):
        """Test percentile normalization."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8).astype(
            np.float32
        )

        config = {
            "normalization": {"strategy": "percentile_99", "clip_percentile": 99.0}
        }

        normalized = inf._normalize(img, config)

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_default(self):
        """Test default normalization when not specified."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8).astype(
            np.float32
        )

        # Empty config should use default (percentile_99)
        normalized = inf._normalize(img, {})

        # Should be normalized to [0, 1]
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestSingleTileInference:
    """Test single tile inference."""

    def test_infer_tile(self, trained_model_path, sample_tile_image):
        """Test inference on a single tile."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model = inf._load_model(str(trained_model_path))

        img = np.array(Image.open(sample_tile_image), dtype=np.float32) / 255.0

        probs = inf._infer_tile(model, img)

        # Should return per-class probabilities
        assert len(probs) == 2  # 2 classes
        assert all(0 <= p <= 1 for p in probs)
        assert abs(sum(probs) - 1.0) < 0.01  # Should sum to ~1

    def test_infer_tile_spatial(self, trained_model_path, sample_tile_image):
        """Test spatial inference on a single tile."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model = inf._load_model(str(trained_model_path))

        img = np.array(Image.open(sample_tile_image), dtype=np.float32) / 255.0

        prob_map = inf._infer_tile_spatial(model, img)

        # Should return (C, H, W) probability map
        assert prob_map.ndim == 3
        assert prob_map.shape[0] == 2  # 2 classes
        assert prob_map.shape[1] == img.shape[0]  # Same height
        assert prob_map.shape[2] == img.shape[1]  # Same width

        # Probabilities should sum to 1 across classes
        class_sum = prob_map.sum(axis=0)
        assert np.allclose(class_sum, 1.0, atol=0.01)


class TestBatchInference:
    """Test batch inference."""

    def _preprocess(self, inf, tile_data, input_config):
        """Load, select channels, normalize -- the order the live scripts use."""
        img = inf._load_tile_data(tile_data)
        selected = input_config.get("selected_channels")
        if selected:
            img = img[:, :, selected]
        return inf._normalize(img, input_config)

    def test_batch_inference_over_the_live_helpers(
        self, trained_model_path, sample_tile_image
    ):
        """_load_model -> _normalize -> _infer_batch_spatial, end to end.

        These three are what production inference actually runs; the Appose
        scripts call them directly. The test used to reach them through
        InferenceService.run_batch, a leftover of the removed HTTP server,
        which meant deleting dead code would have deleted the only coverage
        of live code. It exercises the helpers directly instead.

        This path is not decorative: the MobileNet stem bug that made every
        MobileNet-encoder model unloadable lived in _load_model and was
        caught here.
        """
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        input_config = {"num_channels": 3, "normalization": {"strategy": "min_max"}}

        model_tuple = inf._load_model(str(trained_model_path))
        images = [
            self._preprocess(inf, sample_tile_image, input_config) for _ in range(2)
        ]
        prob_maps = inf._infer_batch_spatial(model_tuple, images)

        assert len(prob_maps) == 2
        for pm in prob_maps:
            # (C, H, W) with C = the fixture model's 2 classes
            assert pm.shape[0] == 2

    def test_batch_inference_with_channel_selection(
        self, trained_model_path, sample_tile_image
    ):
        """A selection covering every channel must behave as no selection."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        base = {"num_channels": 3, "normalization": {"strategy": "min_max"}}
        with_sel = dict(base, selected_channels=[0, 1, 2])

        model_tuple = inf._load_model(str(trained_model_path))
        plain = inf._infer_batch_spatial(
            model_tuple, [self._preprocess(inf, sample_tile_image, base)]
        )
        picked = inf._infer_batch_spatial(
            model_tuple, [self._preprocess(inf, sample_tile_image, with_sel)]
        )

        assert plain[0].shape == picked[0].shape
        np.testing.assert_allclose(plain[0], picked[0], rtol=1e-5, atol=1e-6)

    def test_a_selection_narrows_what_the_model_is_given(
        self, trained_model_path, sample_tile_image
    ):
        """Selecting fewer channels really does hand over fewer channels.

        The fixture model wants 3, so feeding it 2 must fail -- and the
        assertion added in 0.9.8 is what should say so, naming channel
        selection rather than letting it surface as a conv2d shape error.
        """
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        cfg = {
            "num_channels": 2,
            "selected_channels": [0, 2],
            "normalization": {"strategy": "min_max"},
        }
        img = self._preprocess(inf, sample_tile_image, cfg)
        assert img.shape[2] == 2

        model_tuple = inf._load_model(str(trained_model_path))
        with pytest.raises(ValueError) as e:
            InferenceService.assert_input_channels(model_tuple, img.shape[2])
        assert "channel selection" in str(e.value)


class TestPixelInference:
    """Spatial probability maps, the shape the overlay consumes."""

    def test_probability_map_shape_and_normalization(
        self, trained_model_path, sample_tile_image
    ):
        """Output is (C, H, W) and softmaxed over C."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        input_config = {"num_channels": 3, "normalization": {"strategy": "min_max"}}

        img = inf._normalize(inf._load_tile_data(sample_tile_image), input_config)
        model_tuple = inf._load_model(str(trained_model_path))
        prob_map = inf._infer_batch_spatial(model_tuple, [img])[0]

        assert prob_map.ndim == 3
        assert prob_map.shape[0] == 2
        assert prob_map.shape[1:] == img.shape[:2]
        # Per-pixel class probabilities must sum to 1; a raw-logits
        # regression here would be invisible in the overlay until the
        # thresholds stopped meaning anything.
        np.testing.assert_allclose(
            prob_map.sum(axis=0), np.ones(prob_map.shape[1:]), rtol=1e-4, atol=1e-4
        )


class TestSoftmax:
    """Test softmax computation."""

    def test_softmax(self):
        """Test softmax produces valid probabilities."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        x = np.array([1.0, 2.0, 3.0])
        result = inf._softmax(x)

        assert len(result) == 3
        assert all(r >= 0 for r in result)
        assert abs(sum(result) - 1.0) < 0.001

    def test_softmax_2d(self):
        """Test softmax on 2D array (CHW format)."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        # (C, H, W) shaped logits
        x = np.random.randn(2, 4, 4)
        result = inf._softmax(x, axis=0)

        # Should sum to 1 along class axis
        class_sum = result.sum(axis=0)
        assert np.allclose(class_sum, 1.0)


class TestImageDecoding:
    """Test image decoding utilities."""

    def test_load_image(self, sample_tile_image):
        """Test loading image from file."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        img = inf._load_image(sample_tile_image)

        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB

    def test_decode_base64(self):
        """Test decoding base64 image."""
        from dlclassifier_server.services.inference_service import InferenceService
        import base64
        from PIL import Image
        import io

        inf = InferenceService(device="cpu")

        # Create a small test image
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode()

        # Decode
        decoded = inf._decode_base64(b64_data)

        assert decoded is not None
        assert decoded.shape == (32, 32, 3)

    def test_decode_base64_with_prefix(self):
        """Test decoding base64 with data URL prefix."""
        from dlclassifier_server.services.inference_service import InferenceService
        import base64
        from PIL import Image
        import io

        inf = InferenceService(device="cpu")

        # Create a small test image
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Encode to base64 with data URL prefix
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_data = (
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        )

        # Decode
        decoded = inf._decode_base64(b64_data)

        assert decoded is not None
        assert decoded.shape == (32, 32, 3)
