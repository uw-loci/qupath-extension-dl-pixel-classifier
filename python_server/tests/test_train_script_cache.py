"""Regression guard for the in-memory cache block in the bundled train.py.

`train.py` is a JAR resource executed with `exec()` inside the Appose worker, so
nothing imports it and nothing type-checks it. A name used before assignment is
invisible until a user runs the exact configuration that reaches that line.

That is what shipped in dece0ed: the validation-only budget block read
`total_bytes` while the estimate that defines it sat *below*. Every run with
`in_memory_dataset="off"` died on the validation dataset with

    UnboundLocalError: cannot access local variable 'total_bytes'

and nothing caught it, because the only prior cache=off run predated the
feature and every cache=auto run takes a different branch.

These tests lift `_init_with_cache` out of the script and drive it with stubs,
so the four reachable paths are exercised for real rather than eyeballed.
"""

import pathlib
import textwrap
import types

import numpy as np
import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/main/resources/qupath/ext/dlclassifier/scripts/train.py"
)


def _extract_init_with_cache() -> str:
    src = SCRIPT.read_text()
    start = src.index("    def _init_with_cache")
    end = src.index("    _tsm.SegmentationDataset.__init__ = _init_with_cache")
    return textwrap.dedent(src[start:end])


@pytest.fixture(scope="module")
def fn_src():
    if not SCRIPT.exists():  # pragma: no cover
        pytest.skip(f"bundled script not found at {SCRIPT}")
    src = _extract_init_with_cache()
    compile(src, "train.py:_init_with_cache", "exec")
    return src


def _run(fn_src, *, is_validation, n_patches, avail_gb, workers, mode="off"):
    """Exec the real function against stubs; return the log lines it emitted."""
    logs = []

    class _DS:
        @staticmethod
        def _load_patch(_path):
            return np.zeros((358, 358, 3), dtype=np.float32)

    ns = {
        "_orig_sd_init": lambda self, *a, **k: None,
        "logger": types.SimpleNamespace(
            info=lambda m, *a: logs.append(m % a if a else m),
            warning=lambda m, *a: logs.append(m % a if a else m),
            error=lambda m, *a: logs.append(m % a if a else m),
        ),
        "_np": np,
        "_tsm": types.SimpleNamespace(SegmentationDataset=_DS),
        "_PILImage": types.SimpleNamespace(open=lambda _p: None),
        "_query_available_ram_bytes": lambda: (int(avail_gb * 1e9), "test"),
        "_cache_validation_only": mode == "off",
        "_in_memory_mode": mode,
        "_AUTO_CACHE_FRACTION": 0.50,
        "_bounded_fraction": 0.40,
        "training_params": {"data_loader_workers": workers},
    }
    exec(fn_src, ns)  # noqa: S102 -- exercising the shipped script is the point

    half = "validation" if is_validation else "train"
    ds = types.SimpleNamespace(
        image_files=[
            pathlib.Path(f"/d/{half}/images/p{i}.tif") for i in range(n_patches)
        ],
        images_dir=pathlib.Path(f"/d/{half}/images"),
        masks_dir=pathlib.Path(f"/d/{half}/masks"),
        context_dir=None,
    )
    ns["_init_with_cache"](ds)
    return logs


def test_total_bytes_defined_before_the_block_that_reads_it(fn_src):
    # Cheap static guard, so a reorder is caught even if the stubs drift.
    assert fn_src.index("total_bytes = n * (") < fn_src.index("_needed = total_bytes")


def test_cache_off_validation_half_does_not_raise(fn_src):
    # The exact configuration that crashed: in_memory_dataset="off", the
    # validation dataset, DataLoader workers > 0.
    logs = _run(fn_src, is_validation=True, n_patches=958, avail_gb=14.0, workers=2)
    assert logs, "expected the validation-cache path to log a decision"
    assert any("Validation cache" in m for m in logs)


def test_cache_off_training_half_stays_out_of_the_way(fn_src):
    # With the cache off, the training set must keep streaming from disk and
    # must not be preloaded behind the user's back.
    logs = _run(fn_src, is_validation=False, n_patches=3830, avail_gb=14.0, workers=2)
    assert logs == []


def test_validation_budget_counts_exactly_one_copy(fn_src):
    # The validation loader is pinned to num_workers=0, so the cache is never
    # pickled into a child process and the budget must count one copy however
    # many workers the run asks for. Counting 1 + workers is what declined a
    # 1.60 GB validation set at 4.79 GB against a 3.25 GB cap on 2026-09-20 --
    # in the exact configuration the cache exists to serve.
    logs = _run(fn_src, is_validation=True, n_patches=958, avail_gb=1.2, workers=2)
    assert any("declined" in m for m in logs)
    assert any("x 1 copy/copies" in m for m in logs)


def test_validation_cache_engages_despite_workers(fn_src):
    # The regression that motivated the pin: 1.60 GB of validation patches
    # against ~13 GB available used to need 4.79 GB (1 + 2 workers) and lose
    # to the 25% cap. One copy fits, so the cache must engage.
    logs = _run(fn_src, is_validation=True, n_patches=958, avail_gb=13.0, workers=2)
    assert any("preloading" in m for m in logs), logs
    assert not any("declined" in m for m in logs), logs


def test_generous_ram_lets_the_validation_cache_engage(fn_src):
    logs = _run(fn_src, is_validation=True, n_patches=958, avail_gb=64.0, workers=2)
    assert any("preloading" in m for m in logs)
    assert not any("declined" in m for m in logs)


def test_auto_mode_path_is_unaffected(fn_src):
    logs = _run(
        fn_src,
        is_validation=False,
        n_patches=3830,
        avail_gb=30.0,
        workers=0,
        mode="auto",
    )
    assert any("'auto' will preload" in m for m in logs)
