"""FP-AO baseline/teacher (spec §6.1) with the MM phase step, power-only FP and the §5.1 sanity check.

Two implementations of the identical MM step (decision D2):
* "explicit": builds A_n (M x M), lambda_max by an M x M eigen-decomposition, A_n phi by matmul;
* "efficient": never builds A_n; lambda_max from the K x K Gram matrix
  G_n = Xi diag(sqrt(x p)) B_n^H B_n diag(sqrt(x p)) (same nonzero spectrum as A_n),
  A_n phi = v_n - g_n with the matrix-free gradient.
No Python loops over K, M or N; loops are over AO iterations only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

import fp_core as fp
from channels import ChannelBatch

C128 = torch.complex128
_MODES = ("explicit", "efficient")


def _check_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")


# ---------------------------------------------------------------- MM pieces
def lambda_max(ch: ChannelBatch, p: torch.Tensor, aux: fp.Aux, mode: str) -> torch.Tensor:
    """Largest eigenvalue of A_n  -> (B,N) real."""
    _check_mode(mode)
    if mode == "explicit":
        return torch.linalg.eigvalsh(fp.curvature_A(ch, p, aux))[..., -1]
    s = fp.safe_sqrt(ch.x.to(p.dtype) * p.unsqueeze(1))                       # (B,N,K)
    gram = torch.einsum("bnkm,bnlm->bnkl", ch.b.conj(), ch.b)                # b_k^H b_l
    scale = aux.Xi.view(-1, 1, 1, 1) * s.unsqueeze(-1) * s.unsqueeze(-2)    # (B,N,K,K)
    G = scale.to(gram.dtype) * gram
    return torch.linalg.eigvalsh(G)[..., -1]


def A_phi(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor, aux: fp.Aux, mode: str) -> torch.Tensor:
    """A_n phi_n  -> (B,N,M)."""
    _check_mode(mode)
    if mode == "explicit":
        return (fp.curvature_A(ch, p, aux) * phi.unsqueeze(-2)).sum(-1)  # no batched matvec (see fp_core)
    return fp.excitation_v(ch, p, aux) - fp.grad_matrix_free(ch, p, phi, aux)


def mm_phase_step(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor, aux: fp.Aux,
                  mode: str) -> torch.Tensor:
    """phi_n <- exp(j angle(v_n + (lambda_max I - A_n) phi_n)), guarded (spec §6.1, D1)."""
    v = fp.excitation_v(ch, p, aux)
    lam = lambda_max(ch, p, aux, mode)                         # (B,N)
    arg = v + lam[..., None] * phi - A_phi(ch, p, phi, aux, mode)
    has_users = ch.x.sum(-1) > 0                                # (B,N)
    ok = (arg.abs() > 1e-300) & has_users[..., None]
    phi_new = torch.where(ok, arg / arg.abs().clamp_min(1e-300), phi)
    return phi_new


# ---------------------------------------------------------------- AO
@dataclass
class AOResult:
    p: torch.Tensor         # (B,K)
    phi: torch.Tensor       # (B,N,M)
    wsr_traj: torch.Tensor  # (B, max_iter+1) bits; [:,0] at init, padded after convergence
    n_iter: torch.Tensor    # (B,) int64 iterations performed


def init_state(ch: ChannelBatch, theta0: torch.Tensor):
    """p0 = peak powers, phi0 = exp(j theta0)."""
    phi0 = torch.exp(1j * theta0.to(device=ch.b.device, dtype=ch.P.dtype)).to(ch.b.dtype)
    return ch.P.clone(), phi0


class AOStepper:
    """One FP-AO iteration per ``step``; state = (p, phi)."""

    def __init__(self, mode: str = "efficient"):
        _check_mode(mode)
        self.mode = mode

    def init(self, ch: ChannelBatch, p0: torch.Tensor, phi0: torch.Tensor):
        return (p0, phi0)

    def step(self, i: int, ch: ChannelBatch, state):
        p, phi = state
        p_new, aux = fp.deterministic_block(ch, p, phi)
        phi = mm_phase_step(ch, p_new, phi, aux, self.mode)
        return (p_new, phi)


class _PowerOnlyStepper:
    def init(self, ch, p0, phi0):
        return (p0, phi0)

    def step(self, i, ch, state):
        p, phi = state
        p_new, _ = fp.deterministic_block(ch, p, phi)
        return (p_new, phi)


def _run(stepper, ch: ChannelBatch, p0: torch.Tensor, phi0: torch.Tensor,
         tol: float, max_iter: int) -> AOResult:
    B = ch.B
    p, phi = stepper.init(ch, p0, phi0)
    r_prev = fp.wsr(ch, p, phi)
    traj = r_prev.new_empty(B, max_iter + 1)
    traj[:, 0] = r_prev
    n_iter = torch.zeros(B, dtype=torch.int64, device=p.device)
    active = torch.ones(B, dtype=torch.bool, device=p.device)
    t = 0
    for t in range(1, max_iter + 1):
        p_c, phi_c = stepper.step(t - 1, ch, (p, phi))
        p = torch.where(active[:, None], p_c, p)
        phi = torch.where(active[:, None, None], phi_c, phi)
        r = fp.wsr(ch, p, phi)
        r = torch.where(active, r, r_prev)
        traj[:, t] = r
        n_iter = torch.where(active, torch.full_like(n_iter, t), n_iter)
        rel = (r - r_prev).abs() / r_prev.abs().clamp_min(1e-300)
        active = active & ~(rel < tol)
        r_prev = r
        if not bool(active.any()):
            break
    if t < max_iter:
        traj[:, t + 1:] = r_prev[:, None]
    return AOResult(p=p, phi=phi, wsr_traj=traj, n_iter=n_iter)


def run_ao(ch: ChannelBatch, p0: torch.Tensor, phi0: torch.Tensor, mode: str = "efficient",
           tol: float = 1e-5, max_iter: int = 100) -> AOResult:
    """Full FP-AO (power + MM phases); each sample is frozen once ΔR/R < tol."""
    return _run(AOStepper(mode), ch, p0, phi0, tol, max_iter)


def run_power_only(ch: ChannelBatch, p0: torch.Tensor, phi0: torch.Tensor,
                   tol: float = 1e-5, max_iter: int = 100) -> AOResult:
    """Power law only, phases frozen at phi0 (spec §5.1 reference)."""
    return _run(_PowerOnlyStepper(), ch, p0, phi0, tol, max_iter)


# ---------------------------------------------------------------- sanity check
def sanity_check(cfg) -> dict:
    """Spec §5.1: full AO vs power-only FP on the val split; pass iff gain >= 10 %."""
    from data import build_or_load

    d = build_or_load(cfg, "val")
    ch = d["ch"].to("cpu", C128)
    p0, phi0 = init_state(ch, d["theta0"])
    full = run_ao(ch, p0, phi0, mode="efficient", tol=cfg.ao_tol, max_iter=cfg.ao_max_iter)
    po = run_power_only(ch, p0, phi0, tol=cfg.ao_tol, max_iter=cfg.ao_max_iter)
    ao_mean = float(full.wsr_traj[:, -1].mean())
    po_mean = float(po.wsr_traj[:, -1].mean())
    gain_pct = 100.0 * (ao_mean - po_mean) / po_mean
    out = {
        "ao_mean": ao_mean,
        "power_only_mean": po_mean,
        "gain_pct": gain_pct,
        "passed": bool(gain_pct >= 10.0),
        "init_mean": float(full.wsr_traj[:, 0].mean()),
        "ao_mean_n_iter": float(full.n_iter.double().mean()),
        "ao_max_n_iter": int(full.n_iter.max()),
        "power_only_mean_n_iter": float(po.n_iter.double().mean()),
        "n_samples": int(ch.B),
        "split": "val",
        "mode": "efficient",
        "tol": cfg.ao_tol,
        "max_iter": cfg.ao_max_iter,
        "config_hash": d["meta"].get("config_hash"),
    }
    logdir = os.path.join(cfg.root, "logs")
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "sanity_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
