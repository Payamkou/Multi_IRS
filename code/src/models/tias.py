"""TIAS: bipartite user <-> IRS-element message passing with one shared node MLP (spec §6.4).

Per layer l (after the deterministic block, which gives p_new and aux from the layer's input):
  1. variable -> observation:  c_k = sqrt(wt_k p_k) eps_k - Xi p_k H_k(phi)   (fp.obs_scalar)
  2. observation -> variable:  m_{k->(n,m)} = [b_{n,k,m} c_k, Xi p_k |b_{n,k,m}|^2] on edges with
     x_{n,k} = 1, sum-pooled to u1 = g_n[m] (Wirtinger gradient) and u2 = [A_n]_mm.
  3. node update, ONE MLP shared by all N*M nodes and all L layers:
        s     = Softplus(F_GNN(features))                       (B,N,M) >= 0
        Delta = s * u1 / (u2 + delta),   delta = 1e-12 * max_{n,m} u2  per sample (Delta := 0 where u2 = 0)
        phi  <- (phi + Delta) / |phi + Delta|
     Nodes on an IRS with no users have u1 = u2 = 0 -> Delta = 0; phi is copied there exactly.

Scale-free node features (eps = 1e-12):
  f1 = Re{u1 phi*}/(|u1|+eps), f2 = Im{u1 phi*}/(|u1|+eps),
  f3 = log(|u1|/(u2+eps) + eps) standardised per sample (population std, clamped >= 1e-12;
       mean/std are detached constants for backprop),
  f4 = K_n / K.
Builder's choice (recorded): the f3 mean/std run over the sample's nodes that HAVE users
(K_n > 0) and f3 = 0 at user-less nodes. When every IRS has users this is exactly the spec's
"over the sample's N*M nodes"; otherwise the log(eps) = -27.6 values of empty IRSs (16.5% of
default samples) would dominate the statistics and collapse the spread of the active nodes.

Contractions over k are broadcast multiply-and-sum (not einsum/matmul): torch's CPU
multithreaded batched complex matmul with a 1-row left operand is wrong (see fp_core).
Parameters are float32 and cast to the features' real dtype at use (complex128 evaluation works).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import fp_core as fp

from .common import UnfoldedNet
from .maso import _normalise_or_keep

EPS = 1e-12
STD_MIN = 1e-12
DELTA_REL = 1e-12


def tias_messages(ch, p: torch.Tensor, phi: torch.Tensor, aux: fp.Aux):
    """Sum-pooled observation->variable messages: u1 (B,N,M) complex, u2 (B,N,M) real."""
    c = fp.obs_scalar(ch, p, phi, aux)                                   # (B,K) observation nodes
    xr = ch.x.to(p.dtype)                                                # (B,N,K) edges
    u1 = ((xr.to(c.dtype) * c.unsqueeze(1)).unsqueeze(-1) * ch.b).sum(2)
    xp = xr * (aux.Xi.unsqueeze(-1) * p).unsqueeze(1)                    # (B,N,K) x Xi p_k
    u2 = (xp.unsqueeze(-1) * ch.b.abs() ** 2).sum(2)
    return u1, u2


def tias_features(ch, u1: torch.Tensor, u2: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Scale-free node features (B,N,M,4) in the real dtype of u2."""
    rd = u2.dtype
    a1 = u1.abs()
    z = u1 * phi.conj()
    f1 = z.real / (a1 + EPS)
    f2 = z.imag / (a1 + EPS)
    raw = torch.log(a1 / (u2 + EPS) + EPS)                               # (B,N,M)
    deg = ch.x.to(rd).sum(-1)                                            # (B,N) K_n
    act = (deg > 0).unsqueeze(-1).expand_as(raw).to(rd)                  # nodes with users
    cnt = act.sum((-2, -1), keepdim=True).clamp_min(1.0)
    # statistics are detached (no gradient through mean/std) and the variance is floored
    # BEFORE sqrt: sqrt'(0) = inf gave NaN when all active nodes sit at the f3 floor.
    rd_ = raw.detach()
    mean = (rd_ * act).sum((-2, -1), keepdim=True) / cnt
    var = (((rd_ - mean) * act) ** 2).sum((-2, -1), keepdim=True) / cnt
    std = var.clamp_min(STD_MIN ** 2).sqrt()
    f3 = torch.where(act > 0, (raw - mean) / std, torch.zeros_like(raw))
    f4 = (deg / ch.K).unsqueeze(-1).expand_as(raw)
    return torch.stack([f1, f2, f3, f4], dim=-1)


class TIAS(UnfoldedNet):
    def __init__(self, L: int = 10, hidden: int = 32):
        super().__init__(L)
        self.hidden = int(hidden)
        h = self.hidden
        self.mlp = nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, 1), nn.Softplus())

    def node_scale(self, feats: torch.Tensor) -> torch.Tensor:
        """s = Softplus(F_GNN(feats)) (B,N,M); float32 params cast to the features' dtype."""
        y = feats
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                y = F.linear(y, m.weight.to(y.dtype), m.bias.to(y.dtype))
            else:
                y = m(y)
        return y.squeeze(-1)

    def phase_update(self, l, ch, p_new, phi, aux):
        u1, u2 = tias_messages(ch, p_new, phi, aux)
        s = self.node_scale(tias_features(ch, u1, u2, phi))
        delta = DELTA_REL * u2.amax(dim=(-2, -1), keepdim=True)
        # double-where: Delta only where u2 > 0, with a safe denominator (1) elsewhere, so no
        # s/delta-with-delta-at-its-floor appears in backward for user-less nodes or samples
        pos = u2 > 0
        den = torch.where(pos, u2 + delta, torch.ones_like(u2))
        step = torch.where(pos, s * u1 / den, torch.zeros_like(u1))
        out = _normalise_or_keep(phi + step, phi)
        has_users = (ch.x.sum(-1) > 0).unsqueeze(-1)                     # (B,N,1)
        return torch.where(has_users, out, phi)
