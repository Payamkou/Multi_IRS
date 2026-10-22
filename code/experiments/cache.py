"""Result-cache validation shared by all experiments (spec §12).

A results npz is reused only if its stored ``cfg_hash`` and its stored artifact fingerprint
both match the current ones; otherwise the caller recomputes (a one-line notice is printed).

``artifact_fingerprint(cfg)`` -> {relpath: "size:sha256"} for
``checkpoints/{fonr,maso,tias}_best.pt`` and ``datasets/teacher_labels.pt`` (missing files map
to "missing"). Extra files, e.g. a sweep dataset, go in via ``extra`` (paths relative to cfg.root).
Store it with ``fingerprint_array(fp)`` under the npz key ``artifact_fingerprint``.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from config import config_hash

NETS = ("fonr", "maso", "tias")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _entry(path) -> str:
    if not os.path.exists(path):
        return "missing"
    return f"{os.path.getsize(path)}:{file_sha256(path)}"


def artifact_fingerprint(cfg, nets=NETS, extra=()) -> dict:
    rels = [f"checkpoints/{n}_best.pt" for n in nets] + ["datasets/teacher_labels.pt"] + list(extra)
    return {r: _entry(os.path.join(cfg.root, r)) for r in rels}


def fingerprint_array(fp: dict) -> np.ndarray:
    return np.array(json.dumps(fp, sort_keys=True))


def check_cache(npz_path, cfg, keys: tuple, fingerprint: dict, extra_keys: dict | None = None) -> bool:
    """True if ``npz_path`` exists and its ``cfg_hash`` (= config_hash(cfg, keys)), its
    ``artifact_fingerprint`` and any ``extra_keys`` {npz_key: expected_str} match.
    Prints a one-line notice when an existing file is stale (the caller then recomputes)."""
    cfg_hash = config_hash(cfg, keys)
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path) as z:
            stored = {k: str(z[k]) for k in ("cfg_hash", "artifact_fingerprint", *(extra_keys or {}))}
    except (KeyError, ValueError, OSError) as e:
        print(f"[cache] {os.path.basename(str(npz_path))}: unreadable/old format ({e}); recomputing", flush=True)
        return False
    want = {"cfg_hash": str(cfg_hash), "artifact_fingerprint": str(fingerprint_array(fingerprint)),
            **{k: str(v) for k, v in (extra_keys or {}).items()}}
    bad = [k for k in want if stored[k] != want[k]]
    if bad:
        print(f"[cache] {os.path.basename(str(npz_path))}: {', '.join(bad)} changed; recomputing", flush=True)
        return False
    return True
