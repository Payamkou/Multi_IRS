"""Spec §8: training losses (supervised phase alignment, hybrid unsupervised, deep supervision)."""
import dataclasses

import pytest
import torch

import ao
import fp_core as fp
from channels import generate_channels
from config import Config
from models.fonr import FONR
from train import loss_deep_supervision, loss_supervised, loss_unsup_layer, rho_weights

C128, F64 = torch.complex128, torch.float64


def _make(B=4, M=16, K=6, seed=3, cdtype=C128):
    cfg = dataclasses.replace(Config(), M=M, K=K)
    ch, theta0 = generate_channels(cfg, B, seed=seed, M=M, K=K)
    ch = ch.to("cpu", cdtype)
    p0, phi0 = ao.init_state(ch, theta0)
    return ch, p0, phi0


def test_supervised_loss_zero_at_label():
    _, _, phi = _make()
    assert float(loss_supervised(phi, phi)) == pytest.approx(0.0, abs=1e-12)


def test_supervised_loss_two_at_antipode():
    _, _, phi = _make()
    assert float(loss_supervised(-phi, phi)) == pytest.approx(2.0, abs=1e-12)


def test_supervised_loss_is_mean_over_bnm():
    _, _, phi = _make()
    gen = torch.Generator().manual_seed(1)
    d = torch.rand(phi.shape, generator=gen, dtype=F64)
    got = loss_supervised(phi * torch.exp(1j * d), phi)
    assert float(got) == pytest.approx(float((1 - torch.cos(d)).mean()), rel=1e-12)


def test_unsup_no_grad_through_alpha():
    ch, p0, phi0 = _make()
    p = p0.clone().requires_grad_(True)
    phi = phi0.clone().requires_grad_(True)
    alpha_in = fp.aux_vars(ch, p0, phi0).alpha.clone().requires_grad_(True)
    loss_unsup_layer(ch, p, phi, alpha_in).backward()
    assert alpha_in.grad is None
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert phi.grad is not None and torch.isfinite(phi.grad).all() and phi.grad.abs().sum() > 0


def test_unsup_nats():
    ch, p, phi = _make()
    alpha = fp.aux_vars(ch, p, phi).alpha
    got = loss_unsup_layer(ch, p, phi, alpha, a=1.0)
    want = -fp.wsr(ch, p, phi, base="e").mean()
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_unsup_f3_equals_ln_wsr_at_alpha_gamma():
    """Composite = mean(-a R_nats - (1-a) f3), eps closed-form at the given state."""
    ch, p, phi = _make()
    aux = fp.aux_vars(ch, p, phi)
    f3 = fp.surrogate_f3(ch, p, phi, aux.wt, aux.eps)
    got = loss_unsup_layer(ch, p, phi, aux.alpha, a=0.25)
    want = (-0.25 * fp.wsr(ch, p, phi, base="e") - 0.75 * f3).mean()
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_rho_weights_sum_to_one():
    rho = rho_weights(10)
    assert rho.shape == (10,)
    assert float(rho.sum()) == pytest.approx(1.0, abs=1e-15)
    assert torch.allclose(rho, torch.arange(1, 11, dtype=F64) / 55.0)
    assert torch.all(rho[1:] > rho[:-1])


def test_deep_supervision_is_rho_weighted_sum():
    ch, p0, phi0 = _make()
    net = FONR(L=10, T=3, mu_init=1e-3)
    with torch.no_grad():
        tr = net(ch, p0, phi0)
        got = loss_deep_supervision(ch, tr, a=0.5)
        want = sum((l / 55.0) * loss_unsup_layer(ch, tr.p[l], tr.phi[l], tr.alpha_in[l - 1], 0.5)
                   for l in range(1, 11))
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_deep_supervision_grads_finite_complex64_zero_power_user():
    ch, p0, phi0 = _make(cdtype=torch.complex64)
    p0 = p0.clone()
    p0[:, 2] = 0.0
    net = FONR(L=10, T=3, mu_init=1e-3)
    tr = net(ch, p0, phi0)
    loss = loss_deep_supervision(ch, tr)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(net.mu.grad).all() and net.mu.grad.abs().sum() > 0
