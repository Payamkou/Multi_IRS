import dataclasses
import math
import os

import pytest
import torch

import channels as ch
import data
from config import Config, seed_for


def test_path_loss_values():
    assert abs(float(ch.path_loss_db(torch.tensor(100.))) - 90.5) < 0.01
    assert abs(float(ch.path_loss_db(torch.tensor(1000.))) - 128.1) < 1e-4


def test_ula_unit_modulus():
    a = ch.ula_steering(torch.linspace(-1, 1, 11, dtype=torch.float64), 16)
    assert a.shape == (11, 16)
    assert torch.allclose(a.abs(), torch.ones(11, 16, dtype=torch.float64))
    assert torch.allclose(a[:, 0], torch.ones(11, dtype=a.dtype))


def test_shapes_and_association():
    cfg = Config()
    c, th = ch.generate_channels(cfg, 4, seed=5)
    assert c.hd.shape == (4, 6) and c.b.shape == (4, 2, 6, 80)
    assert c.x.shape == (4, 2, 6) and c.w.shape == (4, 6) and c.P.shape == (4, 6)
    assert th.shape == (4, 2, 80)
    assert (c.B, c.N, c.K, c.M) == (4, 2, 6, 80)
    assert c.hd.dtype == torch.complex128 and c.b.dtype == torch.complex128
    assert torch.all(c.x.sum(1) == 1)
    am = c.b.norm(dim=-1).argmax(1)  # (B,K)
    assert torch.equal(c.x.argmax(1), am)
    assert torch.equal(ch.associate(c.b), c.x)
    assert float(th.min()) >= 0 and float(th.max()) < 2 * math.pi


def test_node_types_k6():
    c, _ = ch.generate_channels(Config(), 2, seed=1)
    assert torch.allclose(c.w[0], torch.tensor([0.90, 0.30, 0.80, 0.20, 0.85, 0.25], dtype=torch.float64))
    assert torch.allclose(c.P[0], torch.tensor([0.1, 0.01] * 3, dtype=torch.float64))


def test_overrides_and_slice_to():
    cfg = Config()
    c, th = ch.generate_channels(cfg, 3, seed=2, M=20, K=10, crn_dbm=30)
    assert c.b.shape == (3, 2, 10, 20) and th.shape == (3, 2, 20)
    assert torch.allclose(c.P[0, :2], torch.tensor([1.0, 0.1], dtype=torch.float64))
    s = c.slice(torch.tensor([0, 2]))
    assert s.B == 2
    t = c.to("cpu", torch.complex64)
    assert t.b.dtype == torch.complex64 and t.P.dtype == torch.float32 and t.x.dtype == torch.float32


def test_deterministic_and_seed_dependent():
    cfg = Config()
    a, _ = ch.generate_channels(cfg, 2, seed=9)
    b, _ = ch.generate_channels(cfg, 2, seed=9)
    d, _ = ch.generate_channels(cfg, 2, seed=10)
    assert torch.equal(a.b, b.b) and not torch.equal(a.b, d.b)


def test_min_user_irs_distance():
    cfg = Config()
    g = torch.Generator().manual_seed(0)
    u = ch.sample_users(cfg, 10000, 1, g).reshape(-1, 2)
    for p in cfg.irs_pos:
        d = (u - torch.tensor(p, dtype=u.dtype)).norm(dim=-1)
        assert float(d.min()) > 60.0


def test_cascade_scale_sane():
    # mean cascaded gain should be far above the noise floor in code units
    c, _ = ch.generate_channels(Config(), 50, seed=3)
    assert float(c.hd.abs().pow(2).mean()) > 1.0


def test_cache_roundtrip(tmp_path):
    cfg = dataclasses.replace(Config(), root=str(tmp_path), n_test=5, n_val=3)
    d1 = data.build_or_load(cfg, "val")
    path = os.path.join(str(tmp_path), "datasets", "val.pt")
    assert os.path.exists(path)
    m1 = os.path.getmtime(path)
    d2 = data.build_or_load(cfg, "val")
    assert os.path.getmtime(path) == m1
    assert d1["ch"].B == 3
    assert torch.equal(d1["ch"].b, d2["ch"].b) and torch.equal(d1["theta0"], d2["theta0"])
    assert d2["ch"].b.dtype == torch.complex128
    assert d2["meta"]["split"] == "val"
    s = data.build_or_load(cfg, "sweep_M", 20)
    assert s["ch"].B == 5 and s["ch"].M == 20
    assert os.path.exists(os.path.join(str(tmp_path), "datasets", "sweep_M_20.pt"))
    sk = data.build_or_load(cfg, "sweep_K", 10)
    assert sk["ch"].K == 10
    sp = data.build_or_load(cfg, "sweep_power", 30)
    assert float(sp["ch"].P[0, 0]) == pytest.approx(1.0)
    assert float(sp["ch"].P[0, 1]) == pytest.approx(0.1)


def test_seed_registry_unique():
    cfg = Config()
    seeds = [seed_for(s) for s in ("train", "val", "test")]
    seeds += [seed_for("sweep_M", m) for m in cfg.sweep_M]
    seeds += [seed_for("sweep_K", k) for k in cfg.sweep_K]
    seeds += [seed_for("sweep_power", p) for p in cfg.sweep_crn_dbm]
    seeds += [seed_for("model_init")]
    allseeds = seeds + [s + 7 for s in seeds[:-1]]
    assert len(set(allseeds)) == len(allseeds)
