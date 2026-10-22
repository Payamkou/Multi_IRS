"""os.replace retry helper shared by every atomic writer (M3)."""
import os

import pytest

import atomic


def test_replace_retries_transient_permission_error(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "a"
    src.write_text("x")
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("locked")
        return real(a, b)
    monkeypatch.setattr(atomic.os, "replace", flaky)
    atomic.replace(src, dst, delay=0.0)
    assert len(calls) == 3 and dst.read_text() == "x" and not src.exists()


def test_replace_gives_up_after_five_tries(tmp_path, monkeypatch):
    calls = []

    def always(a, b):
        calls.append(1)
        raise PermissionError("locked")
    monkeypatch.setattr(atomic.os, "replace", always)
    with pytest.raises(PermissionError):
        atomic.replace(tmp_path / "a.tmp", tmp_path / "a", delay=0.0)
    assert len(calls) == 5


def test_all_writers_use_the_helper():
    """No project module calls os.replace directly (atomic.py itself excepted); scans every code module."""
    import glob
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mods = [p for p in glob.glob(os.path.join(here, "**", "*.py"), recursive=True)
            if os.sep + "tests" + os.sep not in p and os.path.basename(p) != "atomic.py"]
    rels = {os.path.relpath(p, here).replace(os.sep, "/") for p in mods}
    assert {"train.py", "teacher.py", "models/fonr.py", "experiments/sweeps.py"} <= rels
    for p in mods:
        with open(p, encoding="utf-8") as f:
            assert "os.replace(" not in f.read(), os.path.relpath(p, here)
