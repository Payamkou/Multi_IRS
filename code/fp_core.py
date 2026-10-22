"""Fractional-programming core shared by every method (spec §6, corrected).

Conventions
-----------
* phi: (B,N,M) complex unit-modulus; p: (B,K) real powers in W; sigma^2 = ch.sigma2.
* All transforms use the natural log (Shen-Yu). ``wsr(base="2")`` reports bits.
* Gradients are Wirtinger gradients w.r.t. phi* (torch's ``phi.grad`` of a real
  loss equals twice this).
* No Python loops over K, M or N: everything is einsum / broadcasting.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from channels import ChannelBatch

_DEN = 1e-300  # spec: every power-law denominator gets "+ 1e-300"


def _den_eps(t: torch.Tensor) -> float:
    """1e-300, raised to the dtype's smallest normal if 1e-300 underflows (float32)."""
    return max(_DEN, torch.finfo(t.dtype).tiny)


def _x(ch: ChannelBatch, like: torch.Tensor) -> torch.Tensor:
    return ch.x.to(dtype=like.dtype)


@dataclass
class Aux:
    alpha: torch.Tensor  # (B,K) Lagrangian dual auxiliary, = SINR
    wt: torch.Tensor     # (B,K) adjusted weight w(1+alpha)
    eps: torch.Tensor    # (B,K) complex quadratic-transform variable (beta == eps)
    Xi: torch.Tensor     # (B,)  sum_k |eps_k|^2
    H: torch.Tensor      # (B,K) complex effective channel at the state aux was built from
    D: torch.Tensor      # (B,)  total received power + sigma^2


# ---------------------------------------------------------------- system model
def effective_channel(ch: ChannelBatch, phi: torch.Tensor) -> torch.Tensor:
    """H_k = hd_k + sum_n x_{n,k} b_{n,k}^H phi_n  -> (B,K)."""
    return ch.hd + torch.einsum("bnk,bnkm,bnm->bk", _x(ch, ch.b), ch.b.conj(), phi)


def received_power(ch: ChannelBatch, p: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """D = sum_i p_i |H_i|^2 + sigma^2  -> (B,)."""
    return (p * H.abs() ** 2).sum(-1) + ch.sigma2


def sinr(p: torch.Tensor, H: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """gamma_k = p_k|H_k|^2 / (D - p_k|H_k|^2)  -> (B,K)."""
    S = p * H.abs() ** 2
    return S / (D.unsqueeze(-1) - S)


def wsr(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor, base: str = "2") -> torch.Tensor:
    """Weighted sum rate (B,); base "2" gives bits, base "e" gives nats."""
    H = effective_channel(ch, phi)
    r = (ch.w * torch.log1p(sinr(p, H, received_power(ch, p, H)))).sum(-1)
    if base == "e":
        return r
    if base == "2":
        return r / torch.log(torch.tensor(2.0, dtype=r.dtype, device=r.device))
    raise ValueError(f"base must be '2' or 'e', got {base!r}")


# ---------------------------------------------------------------- FP transforms
def aux_vars(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor) -> Aux:
    """alpha = gamma, wt = w(1+alpha), eps = sqrt(wt p) H / D, Xi = sum |eps|^2."""
    H = effective_channel(ch, phi)
    D = received_power(ch, p, H)
    alpha = sinr(p, H, D)
    wt = ch.w * (1 + alpha)
    eps = torch.sqrt(wt * p) * H / D.unsqueeze(-1)
    Xi = (eps.abs() ** 2).sum(-1)
    return Aux(alpha=alpha, wt=wt, eps=eps, Xi=Xi, H=H, D=D)


def power_update(ch: ChannelBatch, aux: Aux, p_cur: torch.Tensor) -> torch.Tensor:
    """Spec §6 power law: p = min{P, wt (max(0, Re{eps* H}))^2 / (|H|^2 Xi)^2}.

    Computed as wt * s^2 with s = max(0, Re{eps*H}) / (|H|^2 Xi + 1e-300)
    (algebraically identical, avoids squaring an underflowing denominator).
    ``p_cur`` is accepted for interface uniformity; the law does not depend on it.
    """
    num = (aux.eps.conj() * aux.H).real.clamp_min(0.0)
    den = aux.H.abs() ** 2 * aux.Xi.unsqueeze(-1)
    s = num / (den + _den_eps(den))
    p_new = aux.wt * s ** 2
    return torch.minimum(p_new.clamp_min(0.0), ch.P)


def surrogate_f3(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor,
                 wt: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """f3 = sum_k 2Re{eps_k* sqrt(wt_k p_k) H_k} - Xi D, Xi from ``eps``  -> (B,)."""
    H = effective_channel(ch, phi)
    D = received_power(ch, p, H)
    Xi = (eps.abs() ** 2).sum(-1)
    return 2 * (eps.conj() * torch.sqrt(wt * p) * H).real.sum(-1) - Xi * D


def deterministic_block(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor):
    """Aux from the INPUT (p, phi), then the power law. Returns (p_new, aux)."""
    aux = aux_vars(ch, p, phi)
    return power_update(ch, aux, p), aux


# ---------------------------------------------------------------- phase sub-problem
def obs_scalar(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor, aux: Aux) -> torch.Tensor:
    """c_k = sqrt(wt_k p_k) eps_k - Xi p_k H_k(phi), H recomputed from phi  -> (B,K)."""
    H = effective_channel(ch, phi)
    return torch.sqrt(aux.wt * p) * aux.eps - aux.Xi.unsqueeze(-1) * p * H


def grad_matrix_free(ch: ChannelBatch, p: torch.Tensor, phi: torch.Tensor, aux: Aux) -> torch.Tensor:
    """g_n = sum_k x_{n,k} c_k b_{n,k} = grad_{phi_n*} f3  -> (B,N,M)."""
    c = obs_scalar(ch, p, phi, aux)
    return torch.einsum("bnk,bk,bnkm->bnm", _x(ch, ch.b), c, ch.b)


def excitation_v(ch: ChannelBatch, p: torch.Tensor, aux: Aux) -> torch.Tensor:
    """v_n = sum_k x_{n,k} (sqrt(wt p) eps - Xi p hd) b_{n,k}  (hd NOT conjugated)."""
    coef = torch.sqrt(aux.wt * p) * aux.eps - aux.Xi.unsqueeze(-1) * p * ch.hd
    return torch.einsum("bnk,bk,bnkm->bnm", _x(ch, ch.b), coef, ch.b)


def curvature_A(ch: ChannelBatch, p: torch.Tensor, aux: Aux) -> torch.Tensor:
    """A_n = Xi sum_k x_{n,k} p_k b_{n,k} b_{n,k}^H  -> (B,N,M,M) (explicit)."""
    wgt = _x(ch, ch.b) * (aux.Xi.unsqueeze(-1) * p).unsqueeze(1).to(ch.b.dtype)  # (B,N,K)
    return torch.einsum("bnk,bnkm,bnkl->bnml", wgt, ch.b, ch.b.conj())


def diag_A(ch: ChannelBatch, p: torch.Tensor, aux: Aux) -> torch.Tensor:
    """[A_n]_mm = Xi sum_k x_{n,k} p_k |b_{n,k,m}|^2  -> (B,N,M) real."""
    wgt = ch.x.to(p.dtype) * (aux.Xi.unsqueeze(-1) * p).unsqueeze(1)  # (B,N,K)
    return torch.einsum("bnk,bnkm->bnm", wgt, ch.b.abs() ** 2)
