"""Per-epoch cost fields in training_history.json and the hardware block in
metadata.json -- the data the Training HUD will calibrate epoch time and GPU
memory from."""

import json
import math
from pathlib import Path

from dlclassifier_server.services.training_service import (
    TrainingService,
    _epoch_cost_record,
    _json_safe,
    _training_hardware,
)


class _FakeGpu:
    def get_device_name(self):
        return "NVIDIA GeForce RTX 3090"

    def get_memory_mb(self):
        return 24576

    def get_cuda_version(self):
        return "12.4"


def test_cost_record_sums_and_rounds():
    r = _epoch_cost_record(
        train_seconds=41.23456,
        val_seconds=4.7654,
        train_batches=120,
        val_batches=30,
        peak_vram_mb=6806.4321,
    )
    assert r == {
        "epoch_seconds": 46.0,
        "train_seconds": 41.235,
        "val_seconds": 4.765,
        "train_batches": 120,
        "val_batches": 30,
        "peak_vram_mb": 6806.4,
    }


def test_cost_record_keeps_unknowns_as_none():
    r = _epoch_cost_record(1.0, 0.5, 3, 1, None)
    assert r["peak_vram_mb"] is None
    r = _epoch_cost_record(1.0, float("nan"), 3, 1, float("inf"))
    assert r["val_seconds"] is None
    assert r["epoch_seconds"] is None
    assert r["peak_vram_mb"] is None


def test_cost_record_never_negative():
    # A clock step backwards must not record negative time.
    r = _epoch_cost_record(-0.002, 0.1, 1, 1, None)
    assert r["train_seconds"] == 0.0


def test_json_safe_replaces_non_finite_floats_at_any_depth():
    data = [
        {"train_loss": float("nan"), "per_class_iou": {"a": float("inf"), "b": 0.5}},
        {"x": (1.0, float("-inf"))},
    ]
    out = _json_safe(data)
    assert out[0]["train_loss"] is None
    assert out[0]["per_class_iou"] == {"a": None, "b": 0.5}
    assert out[1]["x"] == [1.0, None]
    json.dumps(out, allow_nan=False)  # must not raise


def test_hardware_block_names_the_device():
    hw = _training_hardware(_FakeGpu(), "cuda")
    assert hw["device"] == "cuda"
    assert hw["device_name"] == "NVIDIA GeForce RTX 3090"
    assert hw["total_memory_mb"] == 24576
    assert hw["cuda_version"] == "12.4"
    assert "torch_version" in hw


def test_training_writes_cost_fields_and_hardware(training_config):
    ts = TrainingService(device="cpu")
    result = ts.train(
        model_type=training_config["model_type"],
        architecture=training_config["architecture"],
        input_config=training_config["input_config"],
        training_params=training_config["training_params"],
        classes=training_config["classes"],
        data_path=training_config["data_path"],
    )
    model_dir = Path(result["model_path"])

    # Strict parse: no NaN/Infinity tokens allowed.
    history = json.loads(
        (model_dir / "training_history.json").read_text(),
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
    )
    assert len(history) == training_config["training_params"]["epochs"]
    for entry in history:
        assert entry["train_seconds"] > 0
        assert entry["val_seconds"] >= 0
        assert math.isclose(
            entry["epoch_seconds"],
            entry["train_seconds"] + entry["val_seconds"],
            abs_tol=0.002,
        )
        assert entry["train_batches"] >= 1
        assert entry["val_batches"] >= 1
        assert entry["peak_vram_mb"] is None  # CPU run

    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["training_hardware"]["device"] == "cpu"
    assert meta["training_hardware"]["device_name"]
