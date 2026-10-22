"""Spec §9 tests 3 (AO monotonicity) and 8 (explicit == efficient AO), plus MM-step guards."""
import dataclasses

import pytest
import torch

import ao
import fp_core as fp
from channels import generate_channels
from config import Config

C128, F64 = torch.complex128, torch.float64


def _make(B, M, K, seed=11):
    cfg = dataclasses.replace(Config(), M=M, K=K)
    ch, theta0 = generate_channels(cfg, B, seed=seed, M=M, K=K)
    ch = ch.to("cpu", C128)
    p0, phi0 = ao.init_state(ch, theta0)
    return ch, p0, phi0


def _rel_err(a, b):
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-300)


@pytest.mark.parametrize("B,M,K", [(2, 180, 60), (2, 16, 6)])
def test_explicit_equals_efficient(B, M, K):
    ch, p0, phi0 = _make(B, M, K)
    se, sf = ao.AOStepper("explicit"), ao.AOStepper("efficient")
    st_e, st_f = se.init(ch, p0, phi0), sf.init(ch, p0, phi0)
    # per-step pieces agree too
    p_new, aux = fp.deterministic_block(ch, p0, phi0)
    assert _rel_err(ao.lambda_max(ch, p_new, aux, "efficient"),
                    ao.lambda_max(ch, p_new, aux, "explicit")) < 1e-10
    assert _rel_err(ao.A_phi(ch, p_new, phi0, aux, "efficient"),
                    ao.A_phi(ch, p_new, phi0, aux, "explicit")) < 1e-10
    for i in range(20):
        st_e = se.step(i, ch, st_e)
        st_f = sf.step(i, ch, st_f)
    assert _rel_err(st_f[0], st_e[0]) < 1e-9
    assert _rel_err(st_f[1], st_e[1]) < 1e-9
    assert torch.allclose(st_f[1].abs(), torch.ones_like(st_f[1].real), atol=1e-12)


@pytest.mark.parametrize("mode", ["explicit", "efficient"])
def test_f3_monotone_in_phase_step(mode):
    ch, p, phi = _make(6, 16, 6, seed=21)
    for _ in range(5):
        p_new, aux = fp.deterministic_block(ch, p, phi)
        f_before = fp.surrogate_f3(ch, p_new, phi, aux.wt, aux.eps)
        phi_new = ao.mm_phase_step(ch, p_new, phi, aux, mode)
        f_after = fp.surrogate_f3(ch, p_new, phi_new, aux.wt, aux.eps)
        assert bool(((f_after - f_before) >= -1e-10 * f_before.abs()).all())
        p, phi = p_new, phi_new


def test_wsr_monotone_over_iterations():
    ch, p0, phi0 = _make(8, 32, 6, seed=31)
    res = ao.run_ao(ch, p0, phi0, mode="efficient", tol=1e-5, max_iter=100)
    tr = res.wsr_traj
    assert tr.shape == (8, 101)
    d = tr[:, 1:] - tr[:, :-1]
    assert bool((d >= -1e-9 * tr[:, :-1].abs()).all())
    assert torch.allclose(tr[:, 0], fp.wsr(ch, p0, phi0))
    assert torch.allclose(tr[:, -1], fp.wsr(ch, res.p, res.phi))
    assert bool((res.p <= ch.P * (1 + 1e-12)).all()) and bool((res.p >= 0).all())


@pytest.mark.parametrize("mode", ["explicit", "efficient"])
def test_mm_step_keeps_phases_when_irs_has_no_users(mode):
    ch, p, phi = _make(3, 16, 6, seed=41)
    x = torch.zeros_like(ch.x)
    x[:, 0, :] = 1.0
    ch = dataclasses.replace(ch, x=x)
    p_new, aux = fp.deterministic_block(ch, p, phi)
    phi_new = ao.mm_phase_step(ch, p_new, phi, aux, mode)
    assert torch.equal(phi_new[:, 1], phi[:, 1])
    assert not torch.equal(phi_new[:, 0], phi[:, 0])
    res = ao.run_ao(ch, p, phi, mode=mode, tol=1e-5, max_iter=10)
    assert torch.equal(res.phi[:, 1], phi[:, 1])


def test_mm_step_zero_argument_keeps_phase(monkeypatch):
    ch, p, phi = _make(2, 16, 6, seed=51)
    m = 5
    p_new, aux = fp.deterministic_block(ch, p, phi)
    orig = ao.A_phi

    def patched(ch_, p_, phi_, aux_, mode_):
        out = orig(ch_, p_, phi_, aux_, mode_).clone()
        v = fp.excitation_v(ch_, p_, aux_)
        lam = ao.lambda_max(ch_, p_, aux_, mode_)
        full = v + lam[..., None] * phi_
        out[:, 0, m] = full[:, 0, m]
        return out

    monkeypatch.setattr(ao, "A_phi", patched)
    phi_new = ao.mm_phase_step(ch, p_new, phi, aux, "efficient")
    assert torch.equal(phi_new[:, 0, m], phi[:, 0, m])
    assert bool(torch.isfinite(phi_new.real).all())
    others = torch.ones(16, dtype=torch.bool)
    others[m] = False
    assert not torch.equal(phi_new[:, 0, others], phi[:, 0, others])


def test_stopping_rule():
    ch, p0, phi0 = _make(8, 32, 6, seed=61)
    res = ao.run_ao(ch, p0, phi0, tol=1e-5, max_iter=100)
    assert res.n_iter.dtype in (torch.int64, torch.int32)
    assert bool((res.n_iter >= 1).all()) and bool((res.n_iter <= 100).all())
    idx = torch.arange(8)
    n = res.n_iter
    r_t = res.wsr_traj[idx, n]
    r_prev = res.wsr_traj[idx, n - 1]
    rel = (r_t - r_prev).abs() / r_prev.abs()
    assert bool(((rel < 1e-5) | (n == 100)).all())
    # padding after convergence
    for b in range(8):
        assert bool((res.wsr_traj[b, int(n[b]):] == r_t[b]).all())
    # converged before stop: earlier steps did not satisfy the rule
    for b in range(8):
        nb = int(n[b])
        if nb > 1:
            tr = res.wsr_traj[b]
            rel_prev = (tr[1:nb] - tr[:nb - 1]).abs() / tr[:nb - 1].abs()
            assert bool((rel_prev >= 1e-5).all())


def test_power_only_keeps_phases():
    ch, p0, phi0 = _make(4, 16, 6, seed=71)
    res = ao.run_power_only(ch, p0, phi0, tol=1e-5, max_iter=100)
    assert torch.equal(res.phi, phi0)
    d = res.wsr_traj[:, 1:] - res.wsr_traj[:, :-1]
    assert bool((d >= -1e-9 * res.wsr_traj[:, :-1].abs()).all())
    full = ao.run_ao(ch, p0, phi0, tol=1e-5, max_iter=100)
    assert bool((full.wsr_traj[:, -1] >= res.wsr_traj[:, 0]).all())
