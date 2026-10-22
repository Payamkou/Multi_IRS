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


MODELS = ["fonr", "maso", "tias"]


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


def _mu_cfg(tmp_path, **kw):
    base = dict(M=16, n_train=12, n_val=2, n_test=2, root=str(tmp_path), train_device="cpu")
    base.update(kw)
    return dataclasses.replace(Config(), **base)


def _ref_lams(cfg):
    from data import build_or_load
    d = build_or_load(cfg, "train")
    ch = d["ch"].to("cpu", C128)
    p0, phi0 = ao.init_state(ch, d["theta0"])
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    lam = ao.lambda_max(ch, p_new, aux, "efficient")
    return lam[lam > 0]


def test_fonr_mu_init_cached(tmp_path, monkeypatch):
    from models.fonr import fonr_mu_median
    cfg = _mu_cfg(tmp_path)
    mu = fonr_mu_init(cfg)
    pos = _ref_lams(cfg)
    med = torch.quantile(pos, 0.5)  # true median (mean of middle pair)
    assert fonr_mu_median(cfg) == pytest.approx(1.0 / (2.0 * float(med)), rel=1e-12)
    path = os.path.join(str(tmp_path), "logs", "fonr_mu_init.json")
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)
    assert len(cache) == 1
    e = next(iter(cache.values()))
    for k in ("median_mu", "chosen_q", "mu_init", "per_q", "config_hash"):
        assert k in e, k
    q = e["chosen_q"]
    lam_q = torch.quantile(pos, q) if q < 1 else pos.max()
    assert mu == pytest.approx(1.0 / (2.0 * float(lam_q)), rel=1e-12)
    assert e["per_q"][str(q)]["finite"] is True
    for k in ("mu", "finite", "max_grad_abs"):
        assert k in e["per_q"][str(q)]
    import models.fonr as mf
    monkeypatch.setattr(mf, "build_or_load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no cache")))
    assert fonr_mu_init(cfg) == mu
    assert fonr_mu_median(cfg) == pytest.approx(1.0 / (2.0 * float(med)), rel=1e-12)


def test_fonr_mu_init_first_finite_q(tmp_path, monkeypatch):
    import models.fonr as mf
    calls = []

    def fake(cfg, mu0, batches):
        calls.append(mu0)
        return (len(calls) >= 2, 1.0)  # q=0.5 "non-finite", q=0.9 finite

    monkeypatch.setattr(mf, "_probe_q", fake)
    cfg = _mu_cfg(tmp_path)
    mu = fonr_mu_init(cfg)
    pos = _ref_lams(cfg)
    assert mu == pytest.approx(1.0 / (2.0 * float(torch.quantile(pos, 0.9))), rel=1e-12)
    assert len(calls) == 2
    with open(os.path.join(str(tmp_path), "logs", "fonr_mu_init.json"), encoding="utf-8") as f:
        e = next(iter(json.load(f).values()))
    assert e["chosen_q"] == 0.9 and e["probe_failed"] is False
    assert e["per_q"]["0.5"]["finite"] is False and e["per_q"]["0.9"]["finite"] is True


def test_fonr_mu_init_none_finite_falls_back(tmp_path, monkeypatch):
    import models.fonr as mf
    monkeypatch.setattr(mf, "_probe_q", lambda cfg, mu0, b: (False, float("inf")))
    cfg = _mu_cfg(tmp_path)
    mu = fonr_mu_init(cfg)
    pos = _ref_lams(cfg)
    assert mu == pytest.approx(1.0 / (2.0 * float(pos.max())), rel=1e-12)
    with open(os.path.join(str(tmp_path), "logs", "fonr_mu_init.json"), encoding="utf-8") as f:
        e = next(iter(json.load(f).values()))
    assert e["chosen_q"] == 1.0 and e["probe_failed"] is True and len(e["per_q"]) == 4


def test_maso_step_scale_uses_median_mu(tmp_path, monkeypatch):
    import models.fonr as mf
    from models.fonr import fonr_mu_median
    monkeypatch.setattr(mf, "_probe_q", lambda cfg, mu0, b: (False, float("inf")))  # mu_init != median
    cfg = _mu_cfg(tmp_path)
    assert fonr_mu_init(cfg) != pytest.approx(fonr_mu_median(cfg))
    net = build_model("maso", cfg)
    assert float(net.step_scale) == pytest.approx(2 * fonr_mu_median(cfg) / math.log(2), rel=1e-6)


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
    monkeypatch.setattr(mf, "fonr_mu_median", lambda cfg: 0.25)
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


# ---------------------------------------------------------------- TIAS (spec 6.4, §9 test 5)
import copy  # noqa: E402

from channels import ChannelBatch  # noqa: E402
from models.tias import TIAS, tias_features, tias_messages  # noqa: E402


def _tias_graph_case():
    """Random channels with IRS 1 (index 0) empty and user 2 linked to no IRS."""
    ch, p0, phi0 = _make(3, 20, 6, seed=21)
    x = torch.zeros_like(ch.x)
    x[:, 1, :] = 1.0
    x[:, 1, 2] = 0.0  # user 2: no IRS link at all
    ch = ChannelBatch(ch.hd, ch.b, x, ch.w, ch.P, ch.sigma2)
    return ch, p0, phi0


def test_tias_messages_equal_gradient_and_diag():
    for case in ("default", "graph"):
        ch, p0, phi0 = _make(4, 20, 6, seed=17) if case == "default" else _tias_graph_case()
        p_new, aux = fp.deterministic_block(ch, p0, phi0)
        for p in (p0, p_new):  # p0: generic c_k != 0; p_new: the state phase_update sees
            u1, u2 = tias_messages(ch, p, phi0, aux)
            assert u1.shape == phi0.shape and u1.is_complex()
            assert u2.shape == phi0.shape and not u2.is_complex()
            g = fp.grad_matrix_free(ch, p, phi0, aux)
            assert torch.allclose(u1, g, rtol=1e-10, atol=0)
            assert torch.allclose(u2, fp.diag_A(ch, p, aux), rtol=1e-10, atol=0)
            # explicit matrices: g_n = v_n - A_n phi_n, u2 = diag(A_n)
            A = fp.curvature_A(ch, p, aux)
            Aphi = (A * phi0.unsqueeze(-2)).sum(-1)  # multiply-sum, no batched complex matmul
            g_exp = fp.excitation_v(ch, p, aux) - Aphi
            scale = float(g_exp.abs().max())
            assert torch.allclose(u1, g_exp, rtol=1e-10, atol=1e-10 * scale)
            assert torch.allclose(u2, torch.diagonal(A, dim1=-2, dim2=-1).real, rtol=1e-10, atol=0)
        if case == "graph":
            u1, u2 = tias_messages(ch, p_new, phi0, aux)
            assert torch.all(u1[:, 0] == 0) and torch.all(u2[:, 0] == 0)
            torch.manual_seed(0)
            net = TIAS(L=1)
            with torch.no_grad():
                phi1 = net.phase_update(0, ch, p_new, phi0, aux)
            assert torch.equal(phi1[:, 0], phi0[:, 0])
            assert torch.allclose(phi1.abs(), torch.ones_like(phi1.real), atol=1e-12)


def test_tias_param_count_independent_of_size():
    for L in (1, 10, 25):
        net = TIAS(L=L, hidden=32)
        assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 1249
    net = build_model("tias", Config())
    assert isinstance(net, TIAS) and net.L == 10
    assert sum(p.numel() for p in net.parameters()) == 1249
    assert all(not p.is_complex() and p.dtype == torch.float32 for p in net.parameters())
    # the same weights run at any M and K (no size-dependent parameters)
    for M, K in [(20, 6), (180, 6), (80, 60)]:
        ch, p0, phi0 = _make(2, M, K)
        with torch.no_grad():
            net(ch, p0, phi0)
    assert sum(p.numel() for p in net.parameters()) == 1249


def test_tias_features_scale_invariant():
    # (a) units change: hd, b x 1e3 with sigma^2 x 1e6 is the same physical system
    ch, p0, phi0 = _make(4, 20, 6, seed=23)
    s = 1e3
    chs = ChannelBatch(ch.hd * s, ch.b * s, ch.x, ch.w, ch.P, ch.sigma2 * s * s)
    feats = []
    for c in (ch, chs):
        aux = fp.aux_vars(c, p0, phi0)
        u1, u2 = tias_messages(c, p0, phi0, aux)
        feats.append(tias_features(c, u1, u2, phi0))
    assert feats[0].shape == (4, 2, 20, 4)
    assert torch.allclose(feats[0], feats[1], rtol=0, atol=1e-9)
    # (b) the feature map itself: u1 -> a u1, u2 -> a2 u2 (b -> 1e3 b with c and Xi p fixed
    #     gives a = 1e3, a2 = 1e6). Exact only while the fixed eps = 1e-12 terms are negligible,
    #     i.e. |u1|, u2 and |u1|/u2 all >> 1e-12 before and after scaling.
    gen = torch.Generator().manual_seed(4)
    u1 = 1e3 * torch.randn(3, 2, 30, dtype=C128, generator=gen)
    u2 = torch.rand(3, 2, 30, dtype=F64, generator=gen) + 0.1
    phi = torch.exp(1j * 6.28 * torch.rand(3, 2, 30, dtype=F64, generator=gen))
    ch3, _, _ = _make(3, 30, 6, seed=5)
    f0 = tias_features(ch3, u1, u2, phi)
    for a, a2 in [(1e3, 1e6), (1e6, 1e6), (1e3, 1.0), (1.0, 1e3)]:
        assert torch.allclose(tias_features(ch3, a * u1, a2 * u2, phi), f0, rtol=0, atol=1e-9)


def test_tias_features_definition():
    ch, p0, phi0 = _tias_graph_case()
    aux = fp.aux_vars(ch, p0, phi0)
    u1, u2 = tias_messages(ch, p0, phi0, aux)
    f = tias_features(ch, u1, u2, phi0)
    eps = 1e-12
    z = u1 * phi0.conj()
    assert torch.allclose(f[..., 0], z.real / (u1.abs() + eps))
    assert torch.allclose(f[..., 1], z.imag / (u1.abs() + eps))
    assert torch.allclose(f[..., 3], (ch.x.sum(-1, keepdim=True) / 6).expand(-1, -1, 20))
    # feature 3: standardised over the sample's nodes that have users; 0 at user-less nodes
    raw = torch.log(u1.abs() / (u2 + eps) + eps)[:, 1]
    ref = (raw - raw.mean(-1, keepdim=True)) / raw.std(-1, unbiased=False, keepdim=True)
    assert torch.allclose(f[:, 1, :, 2], ref)
    assert torch.all(f[:, 0, :, 2] == 0)


def test_tias_update_formula():
    torch.manual_seed(3)
    net = TIAS(L=1)
    ch, p0, phi0 = _make(2, 20, 6)
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    with torch.no_grad():
        out = net.phase_update(0, ch, p_new, phi0, aux)
        u1, u2 = tias_messages(ch, p_new, phi0, aux)
        feats = tias_features(ch, u1, u2, phi0)
        s = copy.deepcopy(net.mlp).double()(feats).squeeze(-1)
    delta = 1e-12 * u2.amax(dim=(-2, -1), keepdim=True)
    u = phi0 + s * u1 / (u2 + delta)
    assert torch.allclose(out, u / u.abs(), atol=1e-12)


def test_tias_grad_flows_complex64():
    net = TIAS(L=2)
    ch, p0, phi0 = _make(2, 80, 6, cdtype=torch.complex64)
    tr = net(ch, p0, phi0)
    fp.wsr(ch, tr.p[-1], tr.phi[-1]).mean().backward()
    w = net.mlp[0].weight
    assert w.dtype == torch.float32 and w.grad is not None and torch.isfinite(w.grad).all()


@pytest.mark.parametrize("cdtype", [C128, torch.complex64])
def test_tias_grads_finite_when_u1_zero_on_active_irs(cdtype, monkeypatch):
    # all active nodes at the f3 floor (u1 == 0) -> per-sample var == 0
    ch, p0, phi0 = _tias_graph_case()
    ch = ch.to("cpu", cdtype)
    phi = phi0.to(cdtype).clone().requires_grad_(True)
    u1 = torch.zeros(phi.shape, dtype=cdtype, requires_grad=True)
    u2 = (torch.rand(phi.shape, dtype=ch.P.dtype) + 0.1) * ch.x.sum(-1, keepdim=True).gt(0)
    u2.requires_grad_(True)
    f = tias_features(ch, u1, u2, phi)
    assert torch.isfinite(f).all()
    f.pow(2).sum().backward()
    for t in (u1, u2, phi):
        g = torch.view_as_real(t.grad.resolve_conj()) if t.grad.is_complex() else t.grad
        assert torch.isfinite(g).all()
    # one TIAS layer with c_k == 0 for every user: u1 == 0 exactly, u2 > 0 on the active IRS
    import models.tias as mt
    real_obs = fp.obs_scalar
    monkeypatch.setattr(mt.fp, "obs_scalar", lambda *a: 0.0 * real_obs(*a))
    net = TIAS(L=1)
    phi_in = phi0.to(cdtype).clone().requires_grad_(True)
    p_new, aux = fp.deterministic_block(ch, ch.P, phi_in)
    u1c, u2c = tias_messages(ch, p_new, phi_in, aux)
    assert torch.all(u1c == 0) and torch.all(u2c[:, 1] > 0)
    out = net.phase_update(0, ch, p_new, phi_in, aux)
    fp.wsr(ch, p_new, out).sum().backward()
    assert torch.isfinite(phi_in.grad.real).all() and torch.isfinite(phi_in.grad.imag).all()
    for prm in net.parameters():
        assert prm.grad is not None and torch.isfinite(prm.grad).all()


@pytest.mark.parametrize("cdtype", [C128, torch.complex64])
def test_tias_grads_finite_all_empty_sample(cdtype):
    ch, p0, phi0 = _make(2, 20, 6, cdtype=cdtype)
    x = ch.x.clone()
    x[0] = 0.0  # sample 0: no user has any IRS link -> u2 == 0 everywhere, delta at its floor
    ch = ChannelBatch(ch.hd, ch.b, x, ch.w, ch.P, ch.sigma2)
    net = TIAS(L=2)
    tr = net(ch, p0, phi0)
    assert torch.equal(tr.phi[-1][0], phi0[0])
    fp.wsr(ch, tr.p[-1], tr.phi[-1]).sum().backward()
    for prm in net.parameters():
        assert prm.grad is not None and torch.isfinite(prm.grad).all()


def test_fonr_complex64_grads_finite_with_zero_power_user():
    """A user with p = 0 stays at 0 through every layer; sqrt(wt p) must not give NaN grads."""
    torch.manual_seed(0)
    net = FONR(L=10, T=3, mu_init=MU0)
    ch, p0, phi0 = _make(4, 20, 6, cdtype=torch.complex64)
    p0 = p0.clone()
    p0[:, 1] = 0.0
    tr = net(ch, p0, phi0)
    assert torch.all(tr.p[:, :, 1] == 0)
    H = fp.effective_channel(ch, tr.phi[-1])
    eps = fp.aux_vars(ch, tr.p[-1], tr.phi[-1]).eps
    loss = -fp.wsr(ch, tr.p[-1], tr.phi[-1], base="e").mean() \
        - fp.surrogate_f3(ch, tr.p[-1], tr.phi[-1], ch.w, eps).mean()
    loss.backward()
    assert net.mu.grad is not None and torch.isfinite(net.mu.grad).all()
    assert net.mu.grad.abs().sum() > 0
