"""Spec §6.2 / §6.5: unfolded base class, FONR, shared feasibility and size tests."""
import dataclasses
import json
import math
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
    if name == "maso":
        return build_model("maso", cfg, step_scale=1.0)
    return build_model(name, cfg)


MODELS = ["fonr", "maso"]  # TIAS is appended in Task 7


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
    for M, K in [(20, 6), (160, 6), (180, 6), (80, 60)]:
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


# ---------------------------------------------------------------- MASO (spec 6.3)
from models.maso import MASO, ComplexLinear, complex_relu  # noqa: E402


def _capture_mlp_inputs(net):
    seen = []
    hooks = [m.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
             for m in net.psi]
    return seen, hooks


def test_maso_structure_and_param_count():
    net = build_model("maso", Config(), step_scale=1.0)
    assert isinstance(net, MASO) and net.L == 10 and net.N == 2 and net.M_train == 80
    assert len(net.psi) == 20
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    per = 2 * (2 * 160 * 160 + 2 * 160) + (2 * 160 * 80 + 2 * 80)
    assert n == 20 * per
    assert all(not p.is_complex() and p.dtype == torch.float32 for p in net.parameters())


def test_complex_relu_and_linear():
    z = torch.tensor([1 - 2j, -3 + 4j, -1 - 1j], dtype=C128)
    assert torch.equal(complex_relu(z), torch.tensor([1 + 0j, 0 + 4j, 0j], dtype=C128))
    torch.manual_seed(0)
    lin = ComplexLinear(5, 3)
    x = torch.randn(7, 5, dtype=C128)
    W = torch.complex(lin.W_re.double(), lin.W_im.double())
    b = torch.complex(lin.b_re.double(), lin.b_im.double())
    ref = torch.stack([W @ x[i] + b for i in range(7)])  # per-row, no batched complex matmul
    out = lin(x)
    assert out.dtype == C128 and torch.allclose(out, ref, atol=1e-12)


def test_maso_padding_small_M():
    torch.manual_seed(0)
    net = MASO(L=1, N=2, M_train=80, step_scale=1.0)
    ch, p0, phi0 = _make(3, 20, 6)
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    g = fp.grad_matrix_free(ch, p_new, phi0, aux)
    seen, hooks = _capture_mlp_inputs(net)
    with torch.no_grad():
        lam = net.step_sizes(0, phi0, g)
    for h in hooks:
        h.remove()
    assert lam.shape == (3, 2, 20) and torch.all(lam >= 0)
    assert len(seen) == 2
    ghat = g / (torch.linalg.vector_norm(g, dim=-1, keepdim=True) + 1e-12)
    for n, z in enumerate(seen):
        assert z.shape == (3, 160)
        assert torch.equal(z[:, :20], phi0[:, n]) and torch.equal(z[:, 80:100], ghat[:, n])
        assert torch.all(z[:, 20:80] == 0) and torch.all(z[:, 100:] == 0)


def test_maso_truncation_large_M():
    torch.manual_seed(0)
    net = MASO(L=3, N=2, M_train=80, step_scale=1.0)
    ch, p0, phi0 = _make(2, 160, 6)
    seen, hooks = _capture_mlp_inputs(net)
    with torch.no_grad():
        tr = net(ch, p0, phi0)
    for h in hooks:
        h.remove()
    assert all(z.shape == (2, 160) for z in seen)
    for l in range(1, 4):
        assert torch.equal(tr.phi[l][..., 80:], phi0[..., 80:])
    assert not torch.allclose(tr.phi[-1][..., :80], phi0[..., :80])
    # gradient direction normalised over the FULL M before truncation
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    g = fp.grad_matrix_free(ch, p_new, phi0, aux)
    ghat = g / (torch.linalg.vector_norm(g, dim=-1, keepdim=True) + 1e-12)
    assert torch.equal(seen[0][:, :80], phi0[:, 0, :80])
    assert torch.equal(seen[0][:, 80:], ghat[:, 0, :80])


def test_maso_K_agnostic_input():
    net = MASO(L=1, N=2, M_train=80, step_scale=1.0)
    for K in (6, 60):
        ch, p0, phi0 = _make(2, 80, K)
        seen, hooks = _capture_mlp_inputs(net)
        with torch.no_grad():
            net(ch, p0, phi0)
        for h in hooks:
            h.remove()
        assert all(z.shape == (2, 2 * 80) for z in seen)
    assert net.psi[0][0].W_re.shape == (160, 160)


def test_maso_update_formula():
    torch.manual_seed(1)
    net = MASO(L=1, N=2, M_train=80, step_scale=1.0)
    ch, p0, phi0 = _make(2, 80, 6)
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    with torch.no_grad():
        out = net.phase_update(0, ch, p_new, phi0, aux)
        g = fp.grad_matrix_free(ch, p_new, phi0, aux)
        lam = net.step_sizes(0, phi0, g)
    r = g - (g * phi0.conj()).real * phi0
    u = phi0 + lam * r
    assert torch.allclose(out, u / u.abs(), atol=1e-12)


def test_maso_grad_flows_complex64():
    net = MASO(L=2, N=2, M_train=80, step_scale=1.0)
    ch, p0, phi0 = _make(2, 80, 6, cdtype=torch.complex64)
    tr = net(ch, p0, phi0)
    fp.wsr(ch, tr.p[-1], tr.phi[-1]).mean().backward()
    assert net.psi[0][0].W_re.grad is not None and torch.isfinite(net.psi[0][0].W_re.grad).all()


def test_maso_step_scale_buffer_and_scaling():
    torch.manual_seed(2)
    net = MASO(L=1, N=2, M_train=80, step_scale=3.5)
    assert "step_scale" in dict(net.named_buffers()) and "step_scale" in net.state_dict()
    assert all(p is not net.step_scale for p in net.parameters())
    assert not net.step_scale.requires_grad
    net2 = MASO(L=1, N=2, M_train=80, step_scale=1.0)
    net2.load_state_dict(net.state_dict())
    assert float(net2.step_scale) == 3.5
    ch, p0, phi0 = _make(2, 20, 6)
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    g = fp.grad_matrix_free(ch, p_new, phi0, aux)
    out = []
    hooks = [m.register_forward_hook(lambda mod, a, o: out.append(o.detach())) for m in net.psi]
    with torch.no_grad():
        lam = net.step_sizes(0, phi0, g)
    for h in hooks:
        h.remove()
    raw = torch.stack(out, dim=1)[..., :20]  # Softplus(Re Psi(z))
    assert lam.dtype == F64 and torch.allclose(lam, 3.5 * raw.to(F64), rtol=1e-6)


def test_build_model_maso_step_scale_from_fonr_mu(monkeypatch):
    import models.fonr as mf
    monkeypatch.setattr(mf, "fonr_mu_init", lambda cfg: 0.25)
    net = build_model("maso", Config())
    assert float(net.step_scale) == pytest.approx(2 * 0.25 / math.log(2))
    assert float(build_model("maso", Config(), step_scale=7.0).step_scale) == 7.0


def test_maso_degenerate_normalisation_keeps_phi():
    # r is tangent (Re{r phi*} = 0) so |phi + lam r| >= 1 in exact arithmetic; the guard is defensive.
    import models.maso as mm
    _, _, phi0 = _make(2, 20, 6)
    u = phi0.clone()
    u[:, :, 3] = 0
    out = mm._normalise_or_keep(u, phi0)
    assert torch.equal(out[:, :, 3], phi0[:, :, 3])
    assert torch.allclose(out.abs(), torch.ones_like(out.real))
    phi32 = phi0.to(torch.complex64)
    u32 = phi32.clone()
    u32[:, :, 5] = 1e-39  # below float32 tiny
    out32 = mm._normalise_or_keep(u32, phi32)
    assert torch.equal(out32[:, :, 5], phi32[:, :, 5])
    assert torch.isfinite(out32.real).all()
