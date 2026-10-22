import dataclasses
import os
import time

import torch

import ao
import data
import fp_core as fp
import teacher
from config import Config


def _cfg(tmp_path, **kw):
    base = dict(M=12, K=3, n_train=16, n_val=8, n_test=8, root=str(tmp_path), ao_max_iter=20)
    base.update(kw)
    return dataclasses.replace(Config(), **base)


def test_teacher_cached_and_consistent(tmp_path):
    cfg = _cfg(tmp_path)
    path = teacher.build_teacher_labels(cfg, chunk=5)
    assert os.path.exists(path)
    for split in ("train", "val", "test"):
        t = teacher.load_teacher(cfg, split)
        assert torch.allclose(t["phi_star"].abs(), torch.ones_like(t["phi_star"].abs()))
        ch = data.build_or_load(cfg, split)["ch"].to("cpu", torch.complex128)
        w = fp.wsr(ch, t["p_star"], t["phi_star"])
        assert torch.allclose(w, t["wsr_ao"], rtol=1e-9)
        assert t["wsr_traj"].shape == (len(w), cfg.ao_max_iter + 1)
    t = teacher.load_teacher(cfg, "test")
    for k in ("phi_ref1000", "p_ref1000", "wsr_ref1000", "n_iter_ref1000"):
        assert k in t
    assert (t["wsr_ref1000"] >= t["wsr_ao"] - 1e-9).all()
    m1 = os.path.getmtime(path)
    time.sleep(0.05)
    teacher.build_teacher_labels(cfg, chunk=5)
    assert os.path.getmtime(path) == m1


def test_teacher_hash_mismatch_rebuilds(tmp_path):
    cfg = _cfg(tmp_path)
    path = teacher.build_teacher_labels(cfg, chunk=8)
    m1 = os.path.getmtime(path)
    time.sleep(0.05)
    cfg2 = dataclasses.replace(cfg, ao_max_iter=10)
    teacher.build_teacher_labels(cfg2, chunk=8)
    assert os.path.getmtime(path) > m1
    assert teacher.load_teacher(cfg2, "val")["wsr_traj"].shape[1] == 11


def test_build_all_datasets_points(tmp_path):
    cfg = _cfg(tmp_path, sweep_M=(8,), zeroshot_M=(8, 10), sweep_K=(2,), sweep_crn_dbm=(20, 25),
               n_test=2, n_train=2, n_val=2)
    out = teacher.build_all_datasets(cfg)
    for k in ("train", "val", "test", "sweep_M_8", "sweep_M_10", "sweep_K_2",
              "sweep_power_20", "sweep_power_25"):
        assert os.path.exists(out[k])
