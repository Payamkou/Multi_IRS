"""Spec §6.2 / §6.5: unfolded base class, FONR, shared feasibility and size tests."""
import dataclasses
import json
import os

import pytest
import torch

import ao
import fp_core as fp
from channels import generate_channels
from config import Config
from models import build_model
from models.common import Trajectory, UnfoldedNet
from models.fonr import FONR, fonr_mu_init, theta_step

C128, F64 = torch.complex128, torch.float64
MU0 = 1e-3  # any positive step; feasibility/size tests do not depend on its value


def _make(B, M, K, seed=11, cdtype=C128):
    cfg = dataclasses.replace(Config(), M=M, K=K)
    ch, theta0 = generate_channels(cfg, B, seed=seed, M=M, K=K)
    ch = ch.to("cpu", cdtype)
    p0, phi0 = ao.init_state(ch, theta0)
    return ch, p0, phi0


def _model(name):
    cfg = Config()
    if name == "fonr":
        return build_model("fonr", cfg, mu_init=MU0)
    return build_model(name, cfg)


MODELS = ["fonr"]  # MASO and TIAS are appended in Tasks 6-7


def test_fonr_param_count():
    net = FONR(L=10, T=3, mu_init=MU0)
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert n == 30
    assert not net.mu.is_complex()
    assert torch.all(net.mu == MU0)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("cdtype", [C128, torch.complex64])
def test_feasibility_all_layers(name, cdtype):
    net = _model(name)
    ch, p0, phi0 = _make(4, 20, 6, cdtype=cdtype)
    with torch.no_grad():
        tr = net(ch, p0, phi0)
    assert isinstance(tr, Trajectory)
    L = tr.alpha_in.shape[0]
    assert tr.p.shape == (L + 1, 4, 6) and tr.phi.shape == (L + 1, 4, 2, 20)
    assert tr.phi.dtype == cdtype
    assert torch.allclose(tr.phi.abs(), torch.ones_like(tr.phi.real), atol=1e-6)
    P = ch.P.unsqueeze(0)
    assert torch.all(tr.p >= 0) and torch.all(tr.p <= P * (1 + 1e-12))


@pytest.mark.parametrize("name", MODELS)
def test_size_handling(name):
    net = _model(name)
    for M, K in [(20, 6), (180, 6), (80, 60)]:
        ch, p0, phi0 = _make(2, M, K)
        with torch.no_grad():
            tr = net(ch, p0, phi0)
        assert tr.phi.shape[-1] == M and tr.p.shape[-1] == K
        assert torch.isfinite(tr.p).all() and torch.isfinite(tr.phi.real).all()


@pytest.mark.parametrize("name", MODELS)
def test_same_init_as_ao(name):
    net = _model(name)
    cfg = dataclasses.replace(Config(), M=20)
    ch, theta0 = generate_channels(cfg, 3, seed=5, M=20)
    ch = ch.to("cpu", C128)
    p0, phi0 = ao.init_state(ch, theta0)
    with torch.no_grad():
        tr = net(ch, p0, phi0)
    assert torch.equal(tr.phi[0], phi0)
    assert torch.equal(tr.p[0], p0)


@pytest.mark.parametrize("name", MODELS)
def test_stepper_matches_forward(name):
    net = _model(name)
    ch, p0, phi0 = _make(3, 20, 6)
    with torch.no_grad():
        tr = net(ch, p0, phi0)
        st = net.init(ch, p0, phi0)
        for i in range(tr.alpha_in.shape[0]):
            st = net.step(i, ch, st)
    assert torch.allclose(st[0], tr.p[-1]) and torch.allclose(st[1], tr.phi[-1])


def test_alpha_in_recorded():
    net = FONR(L=2, T=1, mu_init=MU0)
    ch, p0, phi0 = _make(3, 20, 6)
    with torch.no_grad():
        tr = net(ch, p0, phi0)
    _, aux0 = fp.deterministic_block(ch, p0, phi0)
    assert torch.allclose(tr.alpha_in[0], aux0.alpha)
    _, aux1 = fp.deterministic_block(ch, tr.p[1], tr.phi[1])
    assert torch.allclose(tr.alpha_in[1], aux1.alpha)


def test_fonr_untrained_close_to_mm():
    ch, p0, phi0 = _make(64, 80, 6, seed=31)
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    lam = ao.lambda_max(ch, p_new, aux, "efficient")                  # (B,N)
    mu = torch.where(lam > 0, 1.0 / (2.0 * lam.clamp_min(1e-300)), torch.zeros_like(lam))
    g = fp.grad_matrix_free(ch, p_new, phi0, aux)
    phi_f = theta_step(phi0, g, mu[..., None])
    phi_mm = ao.mm_phase_step(ch, p_new, phi0, aux, "efficient")
    r0 = fp.wsr(ch, p_new, phi0)
    d_f = (fp.wsr(ch, p_new, phi_f) - r0).mean()
    d_mm = (fp.wsr(ch, p_new, phi_mm) - r0).mean()
    assert torch.sign(d_f) == torch.sign(d_mm)


def test_theta_step_formula():
    gen = torch.Generator().manual_seed(0)
    th = torch.rand(2, 2, 5, generator=gen, dtype=F64) * 6.28
    phi = torch.exp(1j * th)
    g = torch.randn(2, 2, 5, generator=gen, dtype=C128)
    out = theta_step(phi, g, torch.tensor(0.3, dtype=F64))
    expect = torch.exp(1j * (th + 0.3 * 2 * (-1j * phi.conj() * g).real))
    assert torch.allclose(out, expect)


def test_fonr_mu_init_cached(tmp_path, monkeypatch):
    cfg = dataclasses.replace(Config(), M=16, n_train=12, n_val=2, n_test=2, root=str(tmp_path))
    mu = fonr_mu_init(cfg)
    # reference computation
    from data import build_or_load
    d = build_or_load(cfg, "train")
    ch = d["ch"].to("cpu", C128)
    p0, phi0 = ao.init_state(ch, d["theta0"])
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    lam = ao.lambda_max(ch, p_new, aux, "efficient")
    med = torch.quantile(lam[lam > 0], 0.5)  # true median (mean of middle pair)
    assert mu == pytest.approx(1.0 / (2.0 * float(med)), rel=1e-12)
    path = os.path.join(str(tmp_path), "logs", "fonr_mu_init.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert len(json.load(f)) == 1
    import models.fonr as mf
    monkeypatch.setattr(mf, "build_or_load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no cache")))
    assert fonr_mu_init(cfg) == mu


def test_build_model_fonr():
    net = build_model("fonr", Config(), mu_init=MU0)
    assert isinstance(net, FONR) and isinstance(net, UnfoldedNet)
    assert net.L == 10 and net.T == 3
    with pytest.raises(ValueError):
        build_model("nope", Config())
