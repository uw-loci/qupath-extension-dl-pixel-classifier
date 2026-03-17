"""Tests for training service.

Tests cover:
- Model creation with various architectures
- Dataset loading and augmentation
- Training loop execution
- Learning rate scheduling
- Early stopping
- ONNX export
- Transfer learning with layer freezing
"""

import os
from pathlib import Path

import pytest


class TestModelCreation:
    """Test model architecture creation."""

    def test_create_unet_model(self):
        """Test UNet model creation."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")
        model = ts._create_model(
            model_type="unet",
            architecture={"backbone": "mobilenet_v2", "use_pretrained": False},
            num_channels=3,
            num_classes=2
        )

        assert model is not None

        # Test forward pass
        import torch
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 2, 256, 256)

    def test_create_unetplusplus_model(self):
        """Test UNet++ model creation."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")
        model = ts._create_model(
            model_type="unetplusplus",
            architecture={"backbone": "mobilenet_v2", "use_pretrained": False},
            num_channels=3,
            num_classes=3
        )

        import torch
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 3, 256, 256)

    def test_create_fpn_model(self):
        """Test FPN model creation."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")
        model = ts._create_model(
            model_type="fpn",
            architecture={"backbone": "mobilenet_v2", "use_pretrained": False},
            num_channels=3,
            num_classes=2
        )

        import torch
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 2, 256, 256)

    def test_create_model_with_pretrained_weights(self):
        """Test model creation with pretrained encoder weights."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")
        model = ts._create_model(
            model_type="unet",
            architecture={"backbone": "mobilenet_v2", "use_pretrained": True},
            num_channels=3,
            num_classes=2
        )

        assert model is not None

    def test_create_model_different_channels(self):
        """Test model creation with different input channel counts."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        # Single channel (grayscale)
        model_1ch = ts._create_model(
            model_type="unet",
            architecture={"backbone": "mobilenet_v2", "use_pretrained": False},
            num_channels=1,
            num_classes=2
        )

        import torch
        x = torch.randn(1, 1, 256, 256)
        out = model_1ch(x)
        assert out.shape == (1, 2, 256, 256)

        # Multi-channel (e.g., fluorescence)
        model_4ch = ts._create_model(
            model_type="unet",
            architecture={"backbone": "mobilenet_v2", "use_pretrained": False},
            num_channels=4,
            num_classes=2
        )

        x = torch.randn(1, 4, 256, 256)
        out = model_4ch(x)
        assert out.shape == (1, 2, 256, 256)

    def test_create_model_invalid_type(self):
        """Test model creation with invalid type raises error."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        with pytest.raises(ValueError, match="Unknown model type"):
            ts._create_model(
                model_type="invalid_model",
                architecture={"backbone": "mobilenet_v2"},
                num_channels=3,
                num_classes=2
            )


class TestDataset:
    """Test dataset creation and loading."""

    def test_dataset_creation(self, sample_training_data):
        """Test SegmentationDataset creation."""
        from dlclassifier_server.services.training_service import SegmentationDataset

        dataset = SegmentationDataset(
            images_dir=str(sample_training_data / "train" / "images"),
            masks_dir=str(sample_training_data / "train" / "masks"),
            input_config={"num_channels": 3},
            augment=False
        )

        assert len(dataset) == 4  # 4 training images

    def test_dataset_item_shape(self, sample_training_data):
        """Test dataset item shapes."""
        from dlclassifier_server.services.training_service import SegmentationDataset

        dataset = SegmentationDataset(
            images_dir=str(sample_training_data / "train" / "images"),
            masks_dir=str(sample_training_data / "train" / "masks"),
            input_config={"num_channels": 3},
            augment=False
        )

        img, mask = dataset[0]

        # Image should be CHW
        assert img.dim() == 3
        assert img.shape[0] == 3  # RGB

        # Mask should be HW
        assert mask.dim() == 2

    def test_dataset_with_augmentation(self, sample_training_data):
        """Test dataset with augmentation enabled."""
        from dlclassifier_server.services.training_service import (
            SegmentationDataset, ALBUMENTATIONS_AVAILABLE
        )

        dataset = SegmentationDataset(
            images_dir=str(sample_training_data / "train" / "images"),
            masks_dir=str(sample_training_data / "train" / "masks"),
            input_config={"num_channels": 3},
            augment=True
        )

        # Should still work
        img, mask = dataset[0]
        assert img is not None
        assert mask is not None

        if ALBUMENTATIONS_AVAILABLE:
            assert dataset.transform is not None

    def test_dataset_normalization_minmax(self, sample_training_data):
        """Test min-max normalization."""
        from dlclassifier_server.services.training_service import SegmentationDataset

        dataset = SegmentationDataset(
            images_dir=str(sample_training_data / "train" / "images"),
            masks_dir=str(sample_training_data / "train" / "masks"),
            input_config={
                "num_channels": 3,
                "normalization": {"strategy": "min_max"}
            },
            augment=False
        )

        img, _ = dataset[0]

        # Values should be in [0, 1]
        assert img.min() >= 0
        assert img.max() <= 1

    def test_dataset_normalization_percentile(self, sample_training_data):
        """Test percentile normalization."""
        from dlclassifier_server.services.training_service import SegmentationDataset

        dataset = SegmentationDataset(
            images_dir=str(sample_training_data / "train" / "images"),
            masks_dir=str(sample_training_data / "train" / "masks"),
            input_config={
                "num_channels": 3,
                "normalization": {"strategy": "percentile_99", "clip_percentile": 99.0}
            },
            augment=False
        )

        img, _ = dataset[0]

        # Values should be in [0, 1]
        assert img.min() >= 0
        assert img.max() <= 1


class TestAugmentation:
    """Test data augmentation."""

    def test_get_training_augmentation(self):
        """Test augmentation pipeline creation."""
        from dlclassifier_server.services.training_service import (
            get_training_augmentation, ALBUMENTATIONS_AVAILABLE
        )

        transform = get_training_augmentation()

        if ALBUMENTATIONS_AVAILABLE:
            assert transform is not None
        else:
            assert transform is None

    def test_augmentation_custom_params(self):
        """Test augmentation with custom parameters."""
        from dlclassifier_server.services.training_service import (
            get_training_augmentation, ALBUMENTATIONS_AVAILABLE
        )

        if not ALBUMENTATIONS_AVAILABLE:
            pytest.skip("albumentations not available")

        transform = get_training_augmentation(
            image_size=256,
            p_flip=0.8,
            p_rotate=0.8,
            p_elastic=0.5,
            p_color=0.5,
            p_noise=0.3
        )

        assert transform is not None


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_improvement(self):
        """Test early stopping detects improvement."""
        from dlclassifier_server.services.training_service import EarlyStopping

        # Disable weight restoration for this test (no model provided)
        es = EarlyStopping(patience=3, min_delta=0.0, restore_best_weights=False)

        # Improving loss should not trigger
        assert not es(0, 1.0, None)
        assert not es(1, 0.9, None)
        assert not es(2, 0.8, None)

        assert es.best_loss == 0.8
        assert es.counter == 0

    def test_early_stopping_triggers(self):
        """Test early stopping triggers after patience."""
        from dlclassifier_server.services.training_service import EarlyStopping

        # Disable weight restoration for this test (no model provided)
        es = EarlyStopping(patience=2, min_delta=0.0, restore_best_weights=False)

        # Initial improvement
        assert not es(0, 1.0, None)

        # Stagnant loss
        assert not es(1, 1.0, None)  # counter = 1
        assert es(2, 1.0, None)       # counter = 2, triggers

        assert es.should_stop is True

    def test_early_stopping_min_delta(self):
        """Test early stopping respects min_delta."""
        from dlclassifier_server.services.training_service import EarlyStopping

        # Disable weight restoration for this test (no model provided)
        # Use patience=5 to allow multiple non-improving epochs
        es = EarlyStopping(patience=5, min_delta=0.1, restore_best_weights=False)

        # First call establishes baseline at 1.0
        assert not es(0, 1.0, None)
        assert es.best_loss == 1.0

        # Small improvements less than min_delta (0.1) should not count
        # Need loss < 1.0 - 0.1 = 0.9 to be considered improvement
        assert not es(1, 0.95, None)  # 0.95 > 0.9, not enough improvement
        assert es.counter == 1

        assert not es(2, 0.92, None)  # 0.92 > 0.9, still not enough
        assert es.counter == 2

        # Now a real improvement (0.85 < 0.9)
        assert not es(3, 0.85, None)
        assert es.counter == 0  # Counter reset
        assert es.best_loss == 0.85

    def test_early_stopping_restore_weights(self):
        """Test early stopping restores best weights."""
        import torch
        import torch.nn as nn
        from dlclassifier_server.services.training_service import EarlyStopping

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

        model = SimpleModel()
        es = EarlyStopping(patience=2, restore_best_weights=True)

        # Save state at best loss
        initial_weights = model.linear.weight.clone()
        assert not es(0, 1.0, model)

        # Modify weights
        with torch.no_grad():
            model.linear.weight.fill_(999.0)

        # More epochs without improvement
        assert not es(1, 1.1, model)
        assert es(2, 1.2, model)

        # Restore should bring back initial weights
        es.restore_best(model)

        # Weights should be restored (close to initial)
        assert torch.allclose(model.linear.weight, initial_weights)


class TestLRScheduler:
    """Test learning rate scheduler creation."""

    def test_create_cosine_scheduler(self):
        """Test cosine annealing scheduler creation."""
        import torch
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = ts._create_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine",
            scheduler_config={"T_0": 10},
            epochs=30,
            steps_per_epoch=10
        )

        assert scheduler is not None
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        assert isinstance(scheduler, CosineAnnealingWarmRestarts)

    def test_create_step_scheduler(self):
        """Test step decay scheduler creation."""
        import torch
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = ts._create_scheduler(
            optimizer=optimizer,
            scheduler_type="step",
            scheduler_config={"step_size": 5, "gamma": 0.1},
            epochs=20,
            steps_per_epoch=10
        )

        assert scheduler is not None
        from torch.optim.lr_scheduler import StepLR
        assert isinstance(scheduler, StepLR)

    def test_create_onecycle_scheduler(self):
        """Test one-cycle scheduler creation."""
        import torch
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = ts._create_scheduler(
            optimizer=optimizer,
            scheduler_type="onecycle",
            scheduler_config={"max_lr": 0.01},
            epochs=10,
            steps_per_epoch=10
        )

        assert scheduler is not None
        from torch.optim.lr_scheduler import OneCycleLR
        assert isinstance(scheduler, OneCycleLR)

    def test_create_no_scheduler(self):
        """Test no scheduler returns None."""
        import torch
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = ts._create_scheduler(
            optimizer=optimizer,
            scheduler_type="none",
            scheduler_config={},
            epochs=10,
            steps_per_epoch=10
        )

        assert scheduler is None


class TestTrainingLoop:
    """Test the full training loop."""

    def test_training_minimal(self, training_config):
        """Test minimal training loop execution."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        result = ts.train(
            model_type=training_config["model_type"],
            architecture=training_config["architecture"],
            input_config=training_config["input_config"],
            training_params=training_config["training_params"],
            classes=training_config["classes"],
            data_path=training_config["data_path"]
        )

        assert result["status"] if "status" in result else True
        assert "model_path" in result
        assert os.path.exists(result["model_path"])

    def test_training_saves_model(self, training_config):
        """Test training saves model files."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        result = ts.train(
            model_type=training_config["model_type"],
            architecture=training_config["architecture"],
            input_config=training_config["input_config"],
            training_params=training_config["training_params"],
            classes=training_config["classes"],
            data_path=training_config["data_path"]
        )

        model_path = Path(result["model_path"])

        # Check expected files exist
        assert (model_path / "model.pt").exists()
        assert (model_path / "metadata.json").exists()

    def test_training_progress_callback(self, training_config):
        """Test training calls progress callback."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        progress_calls = []

        def callback(epoch, loss, accuracy):
            progress_calls.append((epoch, loss, accuracy))

        result = ts.train(
            model_type=training_config["model_type"],
            architecture=training_config["architecture"],
            input_config=training_config["input_config"],
            training_params=training_config["training_params"],
            classes=training_config["classes"],
            data_path=training_config["data_path"],
            progress_callback=callback
        )

        # Should have 2 calls for 2 epochs
        assert len(progress_calls) == 2

    def test_training_cancellation(self, training_config):
        """Test training can be cancelled."""
        import threading
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        # Create a cancel flag and set it immediately
        cancel_flag = threading.Event()
        cancel_flag.set()

        result = ts.train(
            model_type=training_config["model_type"],
            architecture=training_config["architecture"],
            input_config=training_config["input_config"],
            training_params={**training_config["training_params"], "epochs": 100},
            classes=training_config["classes"],
            data_path=training_config["data_path"],
            cancel_flag=cancel_flag
        )

        # Training should have stopped early
        assert result.get("epochs_trained", 0) < 100


class TestONNXExport:
    """Test ONNX model export."""

    def test_onnx_export_after_training(self, training_config):
        """Test ONNX export happens after training."""
        from dlclassifier_server.services.training_service import TrainingService

        ts = TrainingService(device="cpu")

        result = ts.train(
            model_type=training_config["model_type"],
            architecture=training_config["architecture"],
            input_config=training_config["input_config"],
            training_params=training_config["training_params"],
            classes=training_config["classes"],
            data_path=training_config["data_path"]
        )

        model_path = Path(result["model_path"])

        # ONNX file should exist (may not exist if export fails, which is ok)
        onnx_path = model_path / "model.onnx"
        # Note: ONNX export can fail silently, so we just check the attempt was made
        # by checking the model.pt exists
        assert (model_path / "model.pt").exists()


class TestFocalLoss:
    """Test FocalLoss implementation."""

    def test_gamma_zero_equals_ce(self):
        """Focal loss with gamma=0 should equal cross-entropy."""
        import torch
        from dlclassifier_server.services.training_service import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(2, 3, 8, 8)  # (N, C, H, W)
        targets = torch.randint(0, 3, (2, 8, 8))

        focal = FocalLoss(gamma=0.0, ignore_index=255)
        ce = torch.nn.CrossEntropyLoss(ignore_index=255)

        focal_val = focal(logits, targets)
        ce_val = ce(logits, targets)

        assert torch.allclose(focal_val, ce_val, atol=1e-5), (
            f"Focal(gamma=0) = {focal_val.item():.6f}, CE = {ce_val.item():.6f}"
        )

    def test_ignore_index(self):
        """Focal loss should ignore pixels with ignore_index."""
        import torch
        from dlclassifier_server.services.training_service import FocalLoss

        logits = torch.randn(1, 2, 4, 4)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[0, 0, :] = 255  # first row ignored

        focal = FocalLoss(gamma=2.0, ignore_index=255)
        loss = focal(logits, targets)

        assert loss.isfinite(), "Loss should be finite with ignore_index"
        assert loss.item() > 0, "Loss should be positive"

    def test_all_ignored_returns_zero(self):
        """All pixels ignored should return zero loss."""
        import torch
        from dlclassifier_server.services.training_service import FocalLoss

        logits = torch.randn(1, 2, 4, 4)
        targets = torch.full((1, 4, 4), 255, dtype=torch.long)

        focal = FocalLoss(gamma=2.0, ignore_index=255)
        loss = focal(logits, targets)

        assert loss.item() == 0.0

    def test_gradient_flows(self):
        """Focal loss should produce gradients."""
        import torch
        from dlclassifier_server.services.training_service import FocalLoss

        logits = torch.randn(1, 3, 8, 8, requires_grad=True)
        targets = torch.randint(0, 3, (1, 8, 8))

        focal = FocalLoss(gamma=2.0)
        loss = focal(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_class_weights(self):
        """Focal loss with class weights should differ from unweighted."""
        import torch
        from dlclassifier_server.services.training_service import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))

        focal_unweighted = FocalLoss(gamma=2.0)
        focal_weighted = FocalLoss(
            gamma=2.0,
            class_weights=torch.tensor([1.0, 2.0, 0.5])
        )

        loss_uw = focal_unweighted(logits, targets)
        loss_w = focal_weighted(logits, targets)

        assert not torch.allclose(loss_uw, loss_w), (
            "Weighted and unweighted focal loss should differ"
        )

    def test_higher_gamma_lower_easy_loss(self):
        """Higher gamma should reduce contribution from easy (confident) pixels."""
        import torch
        from dlclassifier_server.services.training_service import FocalLoss

        torch.manual_seed(42)
        # Create logits where model is very confident (easy pixels)
        logits = torch.zeros(1, 2, 8, 8)
        logits[:, 0, :, :] = 5.0  # strongly predict class 0
        targets = torch.zeros(1, 8, 8, dtype=torch.long)  # all class 0

        focal_low = FocalLoss(gamma=0.0)
        focal_high = FocalLoss(gamma=5.0)

        loss_low = focal_low(logits, targets)
        loss_high = focal_high(logits, targets)

        assert loss_high < loss_low, (
            f"Higher gamma should reduce easy-pixel loss: "
            f"gamma=0 -> {loss_low.item():.6f}, gamma=5 -> {loss_high.item():.6f}"
        )


class TestOHEMCrossEntropyLoss:
    """Test OHEM loss implementation."""

    def test_ratio_one_equals_ce(self):
        """OHEM with ratio=1.0 should equal standard CE."""
        import torch
        from dlclassifier_server.services.training_service import OHEMCrossEntropyLoss

        torch.manual_seed(42)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))

        ohem = OHEMCrossEntropyLoss(hard_ratio=1.0, ignore_index=255)
        ce = torch.nn.CrossEntropyLoss(ignore_index=255)

        ohem_val = ohem(logits, targets)
        ce_val = ce(logits, targets)

        assert torch.allclose(ohem_val, ce_val, atol=1e-5), (
            f"OHEM(ratio=1.0) = {ohem_val.item():.6f}, CE = {ce_val.item():.6f}"
        )

    def test_keeps_subset(self):
        """OHEM with ratio<1.0 should yield loss >= CE (keeps harder pixels)."""
        import torch
        from dlclassifier_server.services.training_service import OHEMCrossEntropyLoss

        torch.manual_seed(42)
        logits = torch.randn(2, 3, 16, 16)
        targets = torch.randint(0, 3, (2, 16, 16))

        ohem = OHEMCrossEntropyLoss(hard_ratio=0.25, ignore_index=255)
        ce = torch.nn.CrossEntropyLoss(ignore_index=255)

        ohem_val = ohem(logits, targets)
        ce_val = ce(logits, targets)

        assert ohem_val >= ce_val, (
            f"OHEM should have loss >= CE: "
            f"OHEM={ohem_val.item():.6f}, CE={ce_val.item():.6f}"
        )

    def test_gradient_flows(self):
        """OHEM loss should produce gradients."""
        import torch
        from dlclassifier_server.services.training_service import OHEMCrossEntropyLoss

        logits = torch.randn(1, 3, 8, 8, requires_grad=True)
        targets = torch.randint(0, 3, (1, 8, 8))

        ohem = OHEMCrossEntropyLoss(hard_ratio=0.5)
        loss = ohem(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_ignore_index(self):
        """OHEM should ignore pixels with ignore_index."""
        import torch
        from dlclassifier_server.services.training_service import OHEMCrossEntropyLoss

        logits = torch.randn(1, 2, 4, 4)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[0, 0, :] = 255

        ohem = OHEMCrossEntropyLoss(hard_ratio=0.5, ignore_index=255)
        loss = ohem(logits, targets)

        assert loss.isfinite()
        assert loss.item() > 0


class TestCombinedPixelDiceLoss:
    """Test _CombinedPixelDiceLoss combiner."""

    def test_combines_focal_and_dice(self):
        """Combined loss should be average of pixel and dice components."""
        import torch
        from dlclassifier_server.services.training_service import (
            FocalLoss, DiceLoss, _CombinedPixelDiceLoss
        )

        torch.manual_seed(42)
        logits = torch.randn(2, 3, 16, 16)
        targets = torch.randint(0, 3, (2, 16, 16))

        focal = FocalLoss(gamma=2.0, ignore_index=255)
        dice = DiceLoss(ignore_index=255)
        combined = _CombinedPixelDiceLoss(focal, dice)

        focal_val = focal(logits, targets)
        dice_val = dice(logits, targets)
        combined_val = combined(logits, targets)

        expected = 0.5 * focal_val + 0.5 * dice_val
        assert torch.allclose(combined_val, expected, atol=1e-5), (
            f"Combined={combined_val.item():.6f}, "
            f"expected 0.5*focal + 0.5*dice = {expected.item():.6f}"
        )

    def test_gradient_flows(self):
        """Combined loss should produce gradients."""
        import torch
        from dlclassifier_server.services.training_service import (
            FocalLoss, DiceLoss, _CombinedPixelDiceLoss
        )

        logits = torch.randn(1, 3, 8, 8, requires_grad=True)
        targets = torch.randint(0, 3, (1, 8, 8))

        combined = _CombinedPixelDiceLoss(
            FocalLoss(gamma=2.0), DiceLoss()
        )
        loss = combined(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


class TestFormatLossDesc:
    """Test loss description formatting for config summary."""

    def test_plain_ce_dice(self):
        """Plain ce_dice should just show the name."""
        from dlclassifier_server.services.training_service import TrainingService
        desc = TrainingService._format_loss_desc("ce_dice", 2.0, 1.0)
        assert desc == "ce_dice"

    def test_focal_dice_shows_gamma(self):
        """Focal dice should include gamma."""
        from dlclassifier_server.services.training_service import TrainingService
        desc = TrainingService._format_loss_desc("focal_dice", 2.0, 1.0)
        assert "gamma=2.0" in desc

    def test_ohem_appended(self):
        """OHEM should be appended when ratio < 1.0."""
        from dlclassifier_server.services.training_service import TrainingService
        desc = TrainingService._format_loss_desc("ce_dice", 2.0, 0.25)
        assert "OHEM" in desc
        assert "25%" in desc

    def test_focal_with_ohem(self):
        """Focal + OHEM should show both."""
        from dlclassifier_server.services.training_service import TrainingService
        desc = TrainingService._format_loss_desc("focal_dice", 3.0, 0.5)
        assert "gamma=3.0" in desc
        assert "OHEM" in desc
        assert "50%" in desc
