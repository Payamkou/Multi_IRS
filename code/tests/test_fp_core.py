"""Spec §9 tests 1, 2 and 4: gradient, power law, transform identities."""
import dataclasses

import pytest
import torch

import fp_core as fp
from channels import generate_channels
from config import Config

C128, F64 = torch.complex128, torch.float64


@pytest.fixture
def setup():
    cfg = dataclasses.replace(Config(), M=16, K=6)
    chb, theta0 = generate_channels(cfg, 3, seed=123)
    chb = chb.to("cpu", C128)
    phi = torch.exp(1j * theta0).to(C128)
    gen = torch.Generator().manual_seed(5)
    p = chb.P * (0.2 + 0.8 * torch.rand(chb.B, chb.K, generator=gen, dtype=F64))
    return chb, p, phi


def _rel_close(a, b, tol):
    return float((a - b).abs().max()) <= tol * max(float(b.abs().max()), 1e-300)


def test_shapes_and_effective_channel(setup):
    chb, p, phi = setup
    H = fp.effective_channel(chb, phi)
    assert H.shape == (chb.B, chb.K)
    # explicit loop reference (test only)
    ref = chb.hd.clone()
    for n in range(chb.N):
        for k in range(chb.K):
            ref[:, k] += chb.x[:, n, k] * (chb.b[:, n, k, :].conj() * phi[:, n, :]).sum(-1)
    assert torch.allclose(H, ref, atol=1e-12, rtol=1e-12)
    aux = fp.aux_vars(chb, p, phi)
    for name in ("alpha", "wt", "eps", "H"):
        assert getattr(aux, name).shape == (chb.B, chb.K)
    assert aux.Xi.shape == (chb.B,) and aux.D.shape == (chb.B,)
    assert fp.wsr(chb, p, phi).shape == (chb.B,)
    assert torch.allclose(fp.wsr(chb, p, phi, base="2") * torch.log(torch.tensor(2.0, dtype=F64)),
                          fp.wsr(chb, p, phi, base="e"))


def test_gradient_matches_explicit(setup):
    chb, p, phi = setup
    p_new, aux = fp.deterministic_block(chb, p, phi)
    g = fp.grad_matrix_free(chb, p_new, phi, aux)
    v = fp.excitation_v(chb, p_new, aux)
    A = fp.curvature_A(chb, p_new, aux)
    ref = v - torch.einsum("bnml,bnl->bnm", A, phi)
    assert g.shape == (chb.B, chb.N, chb.M)
    assert float((g - ref).abs().max()) <= 1e-9 * float(ref.abs().max())
    d = fp.diag_A(chb, p_new, aux)
    assert torch.allclose(d.to(C128), torch.diagonal(A, dim1=-2, dim2=-1), rtol=1e-12, atol=0)


def test_gradient_matches_autograd(setup):
    chb, p, phi0 = setup
    aux = fp.aux_vars(chb, p, phi0)
    pd, wt, eps = p.detach(), aux.wt.detach(), aux.eps.detach()
    # evaluate at a different phi than the one eps came from
    phi = (phi0 * torch.exp(0.3j * torch.ones_like(phi0.real))).detach().requires_grad_(True)
    f = fp.surrogate_f3(chb, pd, phi, wt, eps).sum()
    f.backward()
    g = fp.grad_matrix_free(chb, pd, phi.detach(), aux)
    assert torch.allclose(phi.grad, 2 * g, rtol=1e-8, atol=1e-8 * float(g.abs().max()))


def test_power_update_maximises_f2(setup):
    chb, p, phi = setup
    aux = fp.aux_vars(chb, p, phi)
    p_new = fp.power_update(chb, aux, p)
    a = torch.sqrt(aux.wt) * (aux.eps.conj() * aux.H).real          # (B,K)
    c = aux.H.abs() ** 2 * aux.Xi.unsqueeze(-1)                       # (B,K)

    def f2(pk):
        return 2 * torch.sqrt(pk) * a.unsqueeze(-1) - pk * c.unsqueeze(-1)

    t = torch.linspace(0, 1, 20001, dtype=F64)
    grid = chb.P.unsqueeze(-1) * t                                     # (B,K,G)
    best_grid = f2(grid).max(-1).values
    closed = f2(p_new.unsqueeze(-1)).squeeze(-1)
    scale = best_grid.abs().clamp_min(1e-300)
    assert bool(((best_grid - closed) <= 1e-9 * scale).all())


def _tiny(chb):
    hd, b = chb.hd.clone(), chb.b.clone()
    hd[0, 0] = 0
    b[:, :, 0] = 0
    return dataclasses.replace(chb, hd=hd, b=b)


def test_power_update_bounds(setup):
    chb, p, phi = setup
    for c in (chb, _tiny(chb)):
        p_new, _ = fp.deterministic_block(c, p, phi)
        assert bool((p_new >= 0).all()) and bool((p_new <= c.P).all())


def test_power_update_tiny_channel_is_finite(setup):
    chb, p, phi = setup
    c = _tiny(chb)
    p_new, aux = fp.deterministic_block(c, p, phi)
    assert bool(torch.isfinite(p_new).all())
    for t in (aux.alpha, aux.wt, aux.eps, aux.Xi, aux.H, aux.D):
        assert bool(torch.isfinite(t).all())
    g = fp.grad_matrix_free(c, p_new, phi, aux)
    assert bool(torch.isfinite(g).all())
    assert bool(torch.isfinite(fp.wsr(c, p_new, phi)).all())


def _f1(chb, p, H, alpha):
    D = fp.received_power(chb, p, H).unsqueeze(-1)
    S = p * H.abs() ** 2
    w = chb.w
    return (w * torch.log1p(alpha) - w * alpha + w * (1 + alpha) * S / D).sum(-1)


def test_lagrangian_dual_stationary(setup):
    chb, p, phi = setup
    H = fp.effective_channel(chb, phi)
    D = fp.received_power(chb, p, H)
    gamma = fp.sinr(p, H, D)
    alpha = gamma.clone().requires_grad_(True)
    _f1(chb, p, H, alpha).sum().backward()
    assert float(alpha.grad.abs().max()) <= 1e-10 * float(chb.w.max())
    assert torch.allclose(_f1(chb, p, H, gamma), fp.wsr(chb, p, phi, base="e"), rtol=1e-12)
    assert torch.allclose(fp.aux_vars(chb, p, phi).alpha, gamma)


def test_quadratic_transform_optimal_eps(setup):
    chb, p, phi = setup
    aux = fp.aux_vars(chb, p, phi)
    f_opt = fp.surrogate_f3(chb, p, phi, aux.wt, aux.eps)
    gen = torch.Generator().manual_seed(9)
    for scale in (1e-3, 1e-1, 1.0):
        d = torch.complex(torch.randn(aux.eps.shape, generator=gen, dtype=F64),
                          torch.randn(aux.eps.shape, generator=gen, dtype=F64))
        d = d * scale * aux.eps.abs().mean()
        f_p = fp.surrogate_f3(chb, p, phi, aux.wt, aux.eps + d)
        assert bool((f_p < f_opt).all())
    # at the optimum, f3 equals the alpha-fixed objective sum_k wt_k p_k|H_k|^2/D
    target = (aux.wt * p * aux.H.abs() ** 2).sum(-1) / aux.D
    assert torch.allclose(f_opt, target, rtol=1e-10)


def test_contractions_deterministic_and_exact_multithreaded():
    """Regression: torch 2.8 CPU batched complex128 matmul with a 1-row left operand
    returns wrong values nondeterministically when multithreaded. The fp_core
    contractions must not route through it."""
    cfg = dataclasses.replace(Config(), M=180, K=60)
    chb, theta0 = generate_channels(cfg, 2, seed=11, M=180, K=60)
    chb = chb.to("cpu", C128)
    phi = torch.exp(1j * theta0).to(C128)
    p = chb.P.clone()
    old = torch.get_num_threads()
    torch.set_num_threads(max(old, 4))
    try:
        xc = chb.x.to(C128)
        H_ref = chb.hd + (xc.unsqueeze(-1) * chb.b.conj() * phi.unsqueeze(2)).sum(-1).sum(1)
        aux = fp.aux_vars(chb, p, phi)
        c = torch.sqrt(aux.wt * p) * aux.eps - aux.Xi.unsqueeze(-1) * p * H_ref
        g_ref = (xc * c.unsqueeze(1)).unsqueeze(-1).mul(chb.b).sum(2)
        coef = torch.sqrt(aux.wt * p) * aux.eps - aux.Xi.unsqueeze(-1) * p * chb.hd
        v_ref = (xc * coef.unsqueeze(1)).unsqueeze(-1).mul(chb.b).sum(2)
        for _ in range(15):
            assert _rel_close(fp.effective_channel(chb, phi), H_ref, 1e-12)
            assert _rel_close(fp.grad_matrix_free(chb, p, phi, aux), g_ref, 1e-12)
            assert _rel_close(fp.excitation_v(chb, p, aux), v_ref, 1e-12)
    finally:
        torch.set_num_threads(old)
