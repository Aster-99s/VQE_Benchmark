"""Import-level compatibility of quantumlib.vqe with current NumPy and
qiskit-ibm-runtime releases.

NumPy 2.0 removed ``np.float_``; qiskit-ibm-runtime 0.20+ removed several fake
backends (``FakeBelem``, ``FakeFez``, ...).  The engine must import without
either, and must not reference the removed NumPy alias anywhere.
"""

import pathlib
import re

import pytest

pytest.importorskip("qiskit_nature")

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE_DIRS = [ROOT / "quantumlib" / "vqe", ROOT / "aqngd"]
ENGINE_FILES = [ROOT / "AQNGDOptimizer.py"]


def _engine_sources():
    for d in ENGINE_DIRS:
        yield from sorted(d.glob("*.py"))
    for f in ENGINE_FILES:
        if f.exists():
            yield f


def test_engine_imports():
    from quantumlib.vqe import core

    assert hasattr(core, "VQEOptimizer")


def test_no_removed_numpy_aliases():
    pattern = re.compile(r"\b(np|numpy)\.(float_|complex_)(?!\w)")
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in _engine_sources()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, offenders


def test_fake_backends_are_optional():
    """Removed fake backends resolve to None instead of failing the import."""
    from quantumlib.vqe import core

    import qiskit_ibm_runtime.fake_provider as fp

    for name in ("FakeBelem", "FakeManilaV2", "FakeCairoV2"):
        resolved = core._try_import_fake(name)
        assert resolved is getattr(fp, name, None)
    assert core._try_import_fake("FakeBackendThatDoesNotExist") is None
