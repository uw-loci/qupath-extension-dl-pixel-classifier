"""The bundled train.py patches SegmentationDataset.__init__ (in-memory cache)
and DataLoader.__init__ (worker count, validation pinning) inside the Appose
worker. Appose reuses that worker process and runs the script again, in fresh
globals, for every training task.

Until 2026-10-29 both patches were installed once per process behind a flag,
so the FIRST run's cache mode and worker count stuck for every later run in
that worker. A benchmark found it: its 'auto' arms, run after an 'off' arm,
took the validation-only cache branch. This runs the real block from the
script twice in one process, the way the worker does, and checks the second
run's settings win.
"""

import pathlib
import types

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/main/resources/qupath/ext/dlclassifier/scripts/train.py"
)
START = "_in_memory_mode = training_params.get("
END = 'logger.info("train.py: constructing TrainingService")'


class _SegDS(Dataset):
    """Stand-in for training_service.SegmentationDataset."""

    def __init__(self, n=4, is_val=False):
        self.n = n
        self._is_validation_half = is_val

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros(1)


@pytest.fixture
def block():
    src = SCRIPT.read_text().replace("\r\n", "\n")
    return compile(src[src.index(START) : src.index(END)], "train.py:patches", "exec")


@pytest.fixture
def tsm():
    orig_dl = DataLoader.__init__
    mod = types.SimpleNamespace(SegmentationDataset=_SegDS)
    orig_sd = _SegDS.__init__
    yield mod
    DataLoader.__init__ = orig_dl
    if hasattr(DataLoader, "_dl_orig_init"):
        del DataLoader._dl_orig_init
    _SegDS.__init__ = orig_sd
    if hasattr(_SegDS, "_dl_orig_init"):
        del _SegDS._dl_orig_init


def _run(block, tsm, **params):
    """One execution of the script's block in fresh globals, as Appose does."""
    ns = {
        "training_params": params,
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
        "_tsm": tsm,
        "_np": __import__("numpy"),
    }
    exec(block, ns)
    return ns


def _ds(is_val=False):
    """An instance built without the (patched) __init__, which would preload."""
    d = _SegDS.__new__(_SegDS)
    d.n = 4
    d._is_validation_half = is_val
    return d


def _workers(ds, **kw):
    return DataLoader(ds, batch_size=1, **kw).num_workers


def test_cache_mode_of_the_second_run_wins(block, tsm):
    _run(block, tsm, in_memory_dataset="off", data_loader_workers=0)
    off_init = _SegDS.__init__
    assert off_init.__globals__["_in_memory_mode"] == "off"

    _run(block, tsm, in_memory_dataset="auto", data_loader_workers=0)
    assert _SegDS.__init__ is not off_init
    assert _SegDS.__init__.__globals__["_in_memory_mode"] == "auto"
    assert _SegDS.__init__.__globals__["_cache_validation_only"] is False


def test_patches_wrap_the_true_original_not_each_other(block, tsm):
    original = _SegDS.__init__
    for mode in ("off", "auto", "on", "off"):
        _run(block, tsm, in_memory_dataset=mode, data_loader_workers=0)
        assert _SegDS._dl_orig_init is original
        assert _SegDS.__init__.__globals__["_orig_sd_init"] is original


def test_worker_count_does_not_leak_into_the_next_run(block, tsm):
    _run(block, tsm, in_memory_dataset="off", data_loader_workers=2)
    assert _workers(_ds(), num_workers=0) == 2

    _run(block, tsm, in_memory_dataset="off", data_loader_workers=0)
    assert _workers(_ds(), num_workers=0) == 0


def test_cache_on_suppresses_the_worker_upgrade(block, tsm):
    _run(block, tsm, in_memory_dataset="auto", data_loader_workers=2)
    assert _workers(_ds(), num_workers=0) == 0


def test_validation_stays_single_process_even_when_asked_for_workers(block, tsm):
    # The installed training_service passes its worker count to the
    # validation loader itself; the one-copy validation cache budget relies
    # on this override.
    _run(block, tsm, in_memory_dataset="off", data_loader_workers=2)
    assert _workers(_ds(is_val=True), num_workers=2) == 0
    assert _workers(_ds(), num_workers=2) == 2


def test_other_datasets_are_left_alone(block, tsm):
    class _Other(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return torch.zeros(1)

    _run(block, tsm, in_memory_dataset="off", data_loader_workers=2)
    assert _workers(_Other(), num_workers=0) == 0
