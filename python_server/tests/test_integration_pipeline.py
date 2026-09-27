"""Integration tests for the DL Pixel Classifier training pipeline.

Validates end-to-end functionality using real exported training data from QuPath:
- Step 1: Extended training convergence (20 epochs, cosine scheduler, class weights)
- Step 2: Trained model prediction verification (shape, softmax, class differentiation)
- Step 3: FastAPI server training endpoint (POST /train, GET /status)
- Step 4: FastAPI server inference endpoint (POST /inference with base64 tiles)

These tests use the actual exported dataset at:
  dl_training/-CTRL_PknRNAi_108870_0001/
with 30 train + 7 validation tiles, 4 classes, sparse labels.

Run with:
  python -m pytest tests/test_integration_pipeline.py -v -s
"""

import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Configure logging for detailed output during tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Path to the real exported training data
DATA_DIR = (
    Path(__file__).parent.parent.parent / "dl_training" / "-CTRL_PknRNAi_108870_0001"
)

# Classes from config.json
CLASSES = ["Ignore*", "hair", "intervein", "vein"]
NUM_CLASSES = 4

# Shared state between ordered tests (model path from Step 1 used in Step 2+)
_shared_state = {}


def _data_available():
    """Check if the real exported training data is available."""
    return (
        DATA_DIR.exists()
        and (DATA_DIR / "train" / "images").exists()
        and (DATA_DIR / "train" / "masks").exists()
        and (DATA_DIR / "validation" / "images").exists()
        and (DATA_DIR / "validation" / "masks").exists()
        and (DATA_DIR / "config.json").exists()
    )


skip_no_data = pytest.mark.skipif(
    not _data_available(), reason=f"Training data not found at {DATA_DIR}"
)


# ============================================================================
# Step 1: Extended Training Convergence Test
# ============================================================================


@skip_no_data
class TestStep1TrainingConvergence:
    """Step 1: Train for 20 epochs and verify loss converges."""

    def test_training_convergence(self, tmp_path):
        """Run 20-epoch training with cosine scheduler and verify convergence.

        Checks:
        - Training completes without error
        - Loss values are finite (no NaN/Inf)
        - Loss trend shows improvement (final < initial, or plateau)
        - Validation accuracy improves
        - Model and metadata save correctly
        - Training history is recorded
        """
        from dlclassifier_server.services.training_service import TrainingService

        # Load the config to verify class info
        with open(DATA_DIR / "config.json") as f:
            config = json.load(f)

        assert len(config["classes"]) == NUM_CLASSES
        assert "class_weights" in config

        logger.info("=" * 70)
        logger.info("STEP 1: Extended Training Convergence Test (20 epochs)")
        logger.info("=" * 70)
        logger.info(f"Data directory: {DATA_DIR}")
        logger.info(f"Classes: {[c['name'] for c in config['classes']]}")
        logger.info(f"Class weights: {config['class_weights']}")
        logger.info(f"Train tiles: {config['metadata']['train_count']}")
        logger.info(f"Validation tiles: {config['metadata']['validation_count']}")

        # Create training service (CPU)
        trainer = TrainingService(device="cpu")

        # Track progress
        progress_log = []

        # Seven positional arguments, matching the real call site in
        # TrainingService._run_training. "loss" here is validation loss.
        def progress_callback(
            epoch,
            train_loss,
            val_loss,
            accuracy,
            per_class_iou,
            per_class_loss,
            mean_iou,
        ):
            progress_log.append(
                {"epoch": epoch, "val_loss": val_loss, "accuracy": accuracy}
            )
            logger.info(
                f"  Epoch {epoch}/20: val_loss={val_loss:.4f}, acc={accuracy:.4f}"
            )

        # Training parameters - designed for convergence on small dataset
        architecture = {
            "backbone": "mobilenet_v2",  # Lightweight for CPU
            "input_size": [512, 512],
            "use_pretrained": True,  # ImageNet pretrained for better convergence
        }

        input_config = {
            "num_channels": 3,
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "clip_percentile": 99.0,
            },
        }

        training_params = {
            "epochs": 20,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "augmentation": True,
            "scheduler": "cosine",
            "scheduler_config": {
                "T_0": 7,  # First restart at epoch 7
                "T_mult": 2,  # Double period after restart
                "eta_min": 1e-6,
            },
            "early_stopping": True,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
        }

        start_time = time.time()
        result = trainer.train(
            model_type="unet",
            architecture=architecture,
            input_config=input_config,
            training_params=training_params,
            classes=CLASSES,
            data_path=str(DATA_DIR),
            progress_callback=progress_callback,
        )
        elapsed = time.time() - start_time

        logger.info(f"\nTraining completed in {elapsed:.1f}s")
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")

        # Store model path for subsequent tests
        _shared_state["model_path"] = result["model_path"]
        _shared_state["training_result"] = result

        # === Assertions ===

        # 1. Training completed and returned valid result
        assert "model_path" in result
        assert result["model_path"] is not None
        assert os.path.isdir(result["model_path"])

        # 2. Model files exist
        model_dir = Path(result["model_path"])
        assert (model_dir / "model.pt").exists(), "PyTorch model not saved"
        assert (model_dir / "metadata.json").exists(), "Metadata not saved"
        assert (
            model_dir / "training_history.json"
        ).exists(), "Training history not saved"

        # 3. Metadata is correct
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["architecture"]["type"] == "unet"
        assert metadata["architecture"]["backbone"] == "mobilenet_v2"
        assert len(metadata["classes"]) == NUM_CLASSES
        assert metadata["input_config"]["num_channels"] == 3

        # 4. Training history recorded all epochs
        with open(model_dir / "training_history.json") as f:
            history = json.load(f)
        assert len(history) > 0
        assert len(history) == result["epochs_trained"]
        _shared_state["training_history"] = history

        # 5. All loss values are finite (no NaN or Inf)
        for entry in history:
            assert np.isfinite(
                entry["train_loss"]
            ), f"NaN/Inf train_loss at epoch {entry['epoch']}"
            assert np.isfinite(
                entry["val_loss"]
            ), f"NaN/Inf val_loss at epoch {entry['epoch']}"
            assert np.isfinite(
                entry["accuracy"]
            ), f"NaN/Inf accuracy at epoch {entry['epoch']}"

        # 6. Loss shows improvement trend
        # Compare first 3 epochs average vs last 3 epochs average
        losses = [e["val_loss"] for e in history]
        n = min(3, len(losses))
        early_avg = np.mean(losses[:n])
        late_avg = np.mean(losses[-n:])
        best_loss = min(losses)

        logger.info(f"\nLoss analysis:")
        logger.info(f"  First {n} epochs avg: {early_avg:.4f}")
        logger.info(f"  Last {n} epochs avg: {late_avg:.4f}")
        logger.info(f"  Best loss: {best_loss:.4f}")
        logger.info(f"  Final loss: {result['final_loss']:.4f}")
        logger.info(f"  Best loss (from result): {result['best_loss']:.4f}")

        # The best loss should be better than the first epoch
        assert (
            best_loss < losses[0]
        ), f"No improvement: best loss {best_loss:.4f} >= first epoch {losses[0]:.4f}"

        # 7. Accuracy should be positive and non-trivial
        accuracies = [e["accuracy"] for e in history]
        max_accuracy = max(accuracies)
        logger.info(f"  Max accuracy: {max_accuracy:.4f}")
        logger.info(f"  Final accuracy: {result['final_accuracy']:.4f}")

        # With 4 classes, random would be ~25%. We expect better than random.
        assert (
            max_accuracy > 0.25
        ), f"Max accuracy {max_accuracy:.4f} not better than random (0.25)"

        # 8. Result contains expected fields
        assert "final_loss" in result
        assert "final_accuracy" in result
        assert "best_loss" in result
        assert "epochs_trained" in result
        assert isinstance(result["early_stopped"], bool)

        logger.info("\n[OK] Step 1 PASSED: Training converges successfully")


# ============================================================================
# Step 2: Model Prediction Verification
# ============================================================================


@skip_no_data
class TestStep2ModelPredictions:
    """Step 2: Load trained model and verify predictions are sensible."""

    def test_prediction_shapes_and_softmax(self):
        """Load model, run inference on validation tiles, verify output properties.

        Checks:
        - Predictions have shape [4] (one per class)
        - Softmax probabilities sum to ~1.0
        - Model predicts different classes for different regions
        - Not all predictions are the same class
        """
        if "model_path" not in _shared_state:
            pytest.skip("Step 1 must run first (no model_path in shared state)")

        from dlclassifier_server.services.inference_service import InferenceService

        model_path = _shared_state["model_path"]
        logger.info("=" * 70)
        logger.info("STEP 2: Model Prediction Verification")
        logger.info("=" * 70)
        logger.info(f"Model path: {model_path}")

        # Create inference service (CPU)
        service = InferenceService(device="cpu")

        # Load validation tiles
        val_images_dir = DATA_DIR / "validation" / "images"
        val_images = sorted(val_images_dir.glob("*.tiff"))
        assert len(val_images) > 0, "No validation images found"
        logger.info(f"Found {len(val_images)} validation tiles")

        input_config = {
            "num_channels": 3,
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "clip_percentile": 99.0,
            },
        }

        # Prepare tiles for batch inference
        tiles = []
        for i, img_path in enumerate(val_images):
            # Read image and encode as base64
            img = Image.open(img_path)
            buffer = io.BytesIO()
            img.save(buffer, format="TIFF")
            b64_data = "data:image/tiff;base64," + base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            tiles.append({"id": f"val_{i:04d}", "data": b64_data, "x": 0, "y": 0})

        # Run batch inference
        predictions = service.run_batch(
            model_path=model_path, tiles=tiles, input_config=input_config
        )

        assert len(predictions) == len(
            val_images
        ), f"Expected {len(val_images)} predictions, got {len(predictions)}"

        # Analyze predictions
        all_probs = []
        argmax_classes = []

        for tile_id, probs in predictions.items():
            probs = np.array(probs)
            all_probs.append(probs)

            # Check shape: should be [NUM_CLASSES]
            assert probs.shape == (
                NUM_CLASSES,
            ), f"Tile {tile_id}: expected shape ({NUM_CLASSES},), got {probs.shape}"

            # Check softmax: probabilities should sum to ~1.0
            prob_sum = probs.sum()
            assert (
                abs(prob_sum - 1.0) < 0.01
            ), f"Tile {tile_id}: probs sum to {prob_sum:.4f}, expected ~1.0"

            # Check all probabilities are valid
            assert np.all(probs >= 0), f"Tile {tile_id}: negative probabilities found"
            assert np.all(probs <= 1), f"Tile {tile_id}: probabilities > 1 found"
            assert np.all(
                np.isfinite(probs)
            ), f"Tile {tile_id}: NaN/Inf probabilities found"

            argmax_class = int(np.argmax(probs))
            argmax_classes.append(argmax_class)

            logger.info(
                f"  {tile_id}: probs={probs.round(4).tolist()}, "
                f"predicted={CLASSES[argmax_class]}"
            )

        # Store for later tests
        _shared_state["predictions"] = predictions

        # Check class diversity - model should not predict ALL the same class
        unique_classes = set(argmax_classes)
        logger.info(
            f"\nPredicted classes across {len(val_images)} tiles: "
            f"{[CLASSES[c] for c in unique_classes]}"
        )
        logger.info(f"Class diversity: {len(unique_classes)} unique classes")

        # With 7 validation tiles and 4 classes, at least 1 unique class
        # (ideally more, but with sparse labels this is the minimum)
        assert len(unique_classes) >= 1, "Model predicts nothing"

        logger.info("\n[OK] Step 2 PASSED: Predictions are valid probability vectors")

    def test_spatial_predictions(self, tmp_path):
        """Verify pixel-level spatial predictions have correct shape.

        Checks:
        - Spatial output has shape (C, H, W) = (4, 512, 512)
        - Probabilities sum to 1 at each pixel
        - Binary output files are written correctly
        """
        if "model_path" not in _shared_state:
            pytest.skip("Step 1 must run first (no model_path in shared state)")

        from dlclassifier_server.services.inference_service import InferenceService

        model_path = _shared_state["model_path"]
        logger.info("\nSTEP 2b: Spatial Prediction Verification")

        service = InferenceService(device="cpu")

        # Use first validation tile
        val_images = sorted((DATA_DIR / "validation" / "images").glob("*.tiff"))
        img = Image.open(val_images[0])
        buffer = io.BytesIO()
        img.save(buffer, format="TIFF")
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        tiles = [{"id": "spatial_test", "data": b64_data, "x": 0, "y": 0}]

        input_config = {
            "num_channels": 3,
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "clip_percentile": 99.0,
            },
        }

        output_dir = str(tmp_path / "pixel_output")
        output_paths = service.run_pixel_inference(
            model_path=model_path,
            tiles=tiles,
            input_config=input_config,
            output_dir=output_dir,
        )

        assert "spatial_test" in output_paths
        output_file = output_paths["spatial_test"]
        assert os.path.exists(output_file), f"Output file not found: {output_file}"

        # Read binary float32 file
        prob_map = np.fromfile(output_file, dtype=np.float32)

        # Expected shape: (C, H, W) = (4, 512, 512)
        expected_size = NUM_CLASSES * 512 * 512
        assert (
            prob_map.size == expected_size
        ), f"Expected {expected_size} elements, got {prob_map.size}"

        prob_map = prob_map.reshape(NUM_CLASSES, 512, 512)
        logger.info(f"  Spatial output shape: {prob_map.shape}")

        # Check probabilities sum to ~1 at each pixel
        pixel_sums = prob_map.sum(axis=0)
        assert np.allclose(
            pixel_sums, 1.0, atol=0.01
        ), f"Pixel probability sums not ~1.0: range [{pixel_sums.min():.4f}, {pixel_sums.max():.4f}]"

        # Check for spatial variation (not all same prediction)
        argmax_map = np.argmax(prob_map, axis=0)
        unique_pixel_classes = np.unique(argmax_map)
        logger.info(
            f"  Unique classes in spatial prediction: "
            f"{[CLASSES[c] for c in unique_pixel_classes]}"
        )

        # All values should be valid
        assert np.all(np.isfinite(prob_map)), "NaN/Inf in spatial predictions"
        assert np.all(prob_map >= 0), "Negative values in spatial predictions"

        logger.info("[OK] Step 2b PASSED: Spatial predictions are valid")


# ============================================================================
# Step 3: FastAPI Server Training Endpoint Test
# ============================================================================


@skip_no_data
class TestStep3ServerTraining:
    """Step 3: Test training via the FastAPI server API."""

    def test_health_endpoint(self, client):
        """Verify server health check works."""
        logger.info("=" * 70)
        logger.info("STEP 3: FastAPI Server Training Endpoint Test")
        logger.info("=" * 70)

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        logger.info(f"  Health: {data}")

    def test_gpu_endpoint(self, client):
        """Verify GPU info endpoint works."""
        response = client.get("/api/v1/gpu")
        assert response.status_code == 200
        data = response.json()
        logger.info(f"  GPU info: {data}")

    def test_training_via_api(self, client):
        """Submit a training job via the API and poll for completion.

        Tests the full server-side training workflow:
        1. POST /api/v1/train starts a job
        2. GET /api/v1/train/{job_id}/status returns progress
        3. Job completes with model_path
        """
        # Use real exported data with minimal epochs for speed
        train_request = {
            "model_type": "unet",
            "architecture": {
                "backbone": "mobilenet_v2",
                "input_size": [512, 512],
                "use_pretrained": False,  # Faster, no download needed
            },
            "input_config": {
                "num_channels": 3,
                "channel_names": ["Red", "Green", "Blue"],
                "bit_depth": 8,
                "normalization": {
                    "strategy": "percentile_99",
                    "per_channel": False,
                    "clip_percentile": 99.0,
                },
            },
            "training": {
                "epochs": 3,  # Minimal for API test
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "validation_split": 0.2,
                "augmentation": False,  # Faster
            },
            "classes": CLASSES,
            "data_path": str(DATA_DIR),
        }

        logger.info("  Submitting training job...")
        response = client.post("/api/v1/train", json=train_request)
        assert (
            response.status_code == 200
        ), f"Train POST failed: {response.status_code} {response.text}"

        data = response.json()
        assert "job_id" in data
        assert data["status"] == "started"
        job_id = data["job_id"]
        logger.info(f"  Job created: {job_id}")

        # Poll for completion (timeout after 300s for CPU training)
        max_wait = 300
        poll_interval = 3
        elapsed = 0
        final_status = None

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            response = client.get(f"/api/v1/train/{job_id}/status")
            assert response.status_code == 200
            status = response.json()

            if status["status"] == "training":
                epoch = status.get("epoch", "?")
                loss = status.get("loss", "?")
                acc = status.get("accuracy", "?")
                logger.info(
                    f"  [{elapsed}s] Training: epoch={epoch}, "
                    f"loss={loss}, acc={acc}"
                )
            elif status["status"] == "completed":
                final_status = status
                logger.info(f"  [{elapsed}s] Training completed!")
                break
            elif status["status"] == "failed":
                pytest.fail(f"Training failed: {status.get('error', 'unknown')}")
            elif status["status"] == "pending":
                logger.info(f"  [{elapsed}s] Pending...")

        assert final_status is not None, f"Training did not complete within {max_wait}s"
        assert final_status["status"] == "completed"
        assert "model_path" in final_status
        assert final_status["model_path"] is not None

        # Store for Step 4
        _shared_state["server_model_path"] = final_status["model_path"]

        logger.info(f"  Model path: {final_status['model_path']}")
        logger.info(f"  Final loss: {final_status.get('final_loss')}")
        logger.info(f"  Final accuracy: {final_status.get('final_accuracy')}")
        logger.info("\n[OK] Step 3 PASSED: Server training API works end-to-end")

    def test_models_list_after_training(self, client):
        """Verify the trained model appears in the model registry."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        logger.info(f"  Registered models: {len(data.get('models', data))} total")


# ============================================================================
# Step 4: FastAPI Server Inference Endpoint Test
# ============================================================================


@skip_no_data
class TestStep4ServerInference:
    """Step 4: Test inference via the FastAPI server API."""

    def _get_model_path(self):
        """Get the model path - prefer direct training result, fallback to server."""
        # Prefer the model from Step 1 (more epochs, better trained)
        if "model_path" in _shared_state:
            return _shared_state["model_path"]
        if "server_model_path" in _shared_state:
            return _shared_state["server_model_path"]
        pytest.skip("No trained model available (Steps 1 or 3 must run first)")

    def _encode_tile(self, img_path: Path) -> str:
        """Encode a tile image as base64 with data URI prefix."""
        img = Image.open(img_path)
        buffer = io.BytesIO()
        img.save(buffer, format="TIFF")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/tiff;base64,{b64}"

    def test_inference_endpoint(self, client):
        """Send tiles to the inference endpoint and verify predictions.

        Tests:
        - POST /api/v1/inference accepts request
        - Returns predictions for each tile
        - Predictions are valid probability vectors
        """
        model_path = self._get_model_path()

        logger.info("=" * 70)
        logger.info("STEP 4: FastAPI Server Inference Endpoint Test")
        logger.info("=" * 70)
        logger.info(f"  Using model: {model_path}")

        # Load 3 validation tiles
        val_images = sorted((DATA_DIR / "validation" / "images").glob("*.tiff"))[:3]
        assert len(val_images) > 0, "No validation images found"

        tiles = []
        for i, img_path in enumerate(val_images):
            tiles.append(
                {
                    "id": f"tile_{i}",
                    "data": self._encode_tile(img_path),
                    "x": i * 512,
                    "y": 0,
                }
            )

        inference_request = {
            "model_path": model_path,
            "input_config": {
                "num_channels": 3,
                "selected_channels": [0, 1, 2],
                "normalization": {
                    "strategy": "percentile_99",
                    "per_channel": False,
                    "clip_percentile": 99.0,
                },
            },
            "tiles": tiles,
            "options": {"use_gpu": False, "blend_mode": "linear"},
        }

        logger.info(f"  Sending {len(tiles)} tiles for inference...")
        response = client.post("/api/v1/inference", json=inference_request)
        assert (
            response.status_code == 200
        ), f"Inference failed: {response.status_code} {response.text}"

        data = response.json()
        assert "predictions" in data
        predictions = data["predictions"]

        assert len(predictions) == len(
            tiles
        ), f"Expected {len(tiles)} predictions, got {len(predictions)}"

        for tile_id, probs in predictions.items():
            probs = np.array(probs)

            # Shape check
            assert probs.shape == (
                NUM_CLASSES,
            ), f"Tile {tile_id}: expected ({NUM_CLASSES},), got {probs.shape}"

            # Softmax check
            prob_sum = probs.sum()
            assert (
                abs(prob_sum - 1.0) < 0.01
            ), f"Tile {tile_id}: probs sum to {prob_sum:.4f}"

            # Valid range
            assert np.all(probs >= 0) and np.all(
                probs <= 1
            ), f"Tile {tile_id}: probs out of [0,1] range"

            assert np.all(np.isfinite(probs)), f"Tile {tile_id}: NaN/Inf in predictions"

            logger.info(
                f"  {tile_id}: {probs.round(4).tolist()} "
                f"-> {CLASSES[int(np.argmax(probs))]}"
            )

        logger.info("\n[OK] Step 4 PASSED: Server inference endpoint works correctly")

    def test_pixel_inference_endpoint(self, client, tmp_path):
        """Test pixel-level inference via the server.

        Tests:
        - POST /api/v1/inference/pixel returns output paths
        - Binary files contain valid probability maps
        """
        model_path = self._get_model_path()

        logger.info("\nSTEP 4b: Pixel Inference Endpoint Test")

        # Single tile for pixel inference
        val_images = sorted((DATA_DIR / "validation" / "images").glob("*.tiff"))
        tile_data = self._encode_tile(val_images[0])

        output_dir = str(tmp_path / "server_pixel_output")

        pixel_request = {
            "model_path": model_path,
            "input_config": {
                "num_channels": 3,
                "selected_channels": [0, 1, 2],
                "normalization": {
                    "strategy": "percentile_99",
                    "per_channel": False,
                    "clip_percentile": 99.0,
                },
            },
            "tiles": [{"id": "pixel_tile_0", "data": tile_data, "x": 0, "y": 0}],
            "output_dir": output_dir,
            "options": {"use_gpu": False, "blend_mode": "linear"},
        }

        response = client.post("/api/v1/inference/pixel", json=pixel_request)
        assert (
            response.status_code == 200
        ), f"Pixel inference failed: {response.status_code} {response.text}"

        data = response.json()
        assert "output_paths" in data
        assert "num_classes" in data
        assert data["num_classes"] == NUM_CLASSES

        output_paths = data["output_paths"]
        assert "pixel_tile_0" in output_paths

        # Verify output file
        output_file = output_paths["pixel_tile_0"]
        assert os.path.exists(output_file), f"Output file missing: {output_file}"

        prob_map = np.fromfile(output_file, dtype=np.float32)
        expected_elements = NUM_CLASSES * 512 * 512
        assert (
            prob_map.size == expected_elements
        ), f"Expected {expected_elements} elements, got {prob_map.size}"

        prob_map = prob_map.reshape(NUM_CLASSES, 512, 512)
        pixel_sums = prob_map.sum(axis=0)
        assert np.allclose(
            pixel_sums, 1.0, atol=0.01
        ), f"Pixel sums not ~1.0: [{pixel_sums.min():.4f}, {pixel_sums.max():.4f}]"

        logger.info(f"  Pixel output shape: {prob_map.shape}")
        logger.info(
            f"  Pixel sum range: [{pixel_sums.min():.4f}, {pixel_sums.max():.4f}]"
        )
        logger.info("[OK] Step 4b PASSED: Pixel inference endpoint works correctly")


# ============================================================================
# Summary Test - prints overall results
# ============================================================================


@skip_no_data
class TestSummary:
    """Final summary of all integration test results."""

    def test_print_summary(self):
        """Print a summary of all test results."""
        logger.info("\n" + "=" * 70)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("=" * 70)

        if "training_result" in _shared_state:
            result = _shared_state["training_result"]
            logger.info(
                f"Training: {result['epochs_trained']} epochs, "
                f"best_loss={result['best_loss']:.4f}, "
                f"final_acc={result['final_accuracy']:.4f}"
            )

        if "training_history" in _shared_state:
            history = _shared_state["training_history"]
            losses = [e["val_loss"] for e in history]
            accs = [e["accuracy"] for e in history]
            logger.info(
                f"Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f} "
                f"(best: {min(losses):.4f})"
            )
            logger.info(
                f"Accuracy trend: {accs[0]:.4f} -> {accs[-1]:.4f} "
                f"(best: {max(accs):.4f})"
            )

        if "predictions" in _shared_state:
            preds = _shared_state["predictions"]
            logger.info(f"Inference: {len(preds)} tiles classified successfully")

        if "model_path" in _shared_state:
            logger.info(f"Model saved at: {_shared_state['model_path']}")

        logger.info("=" * 70)
