"""FONR: DUFP-style nested projected-gradient unfolding (spec §6.2).

Per layer: deterministic block, then T inner steps per IRS on the phase angles
    theta <- theta + mu[l,t] * 2 Re{-j conj(phi) * g},  phi = exp(j theta),
with g = grad_matrix_free(ch, p_new, phi, aux) recomputed at every inner step.
Trainable: one real scalar mu[l,t] per (layer, inner step) -> L*T parameters.
"""
from __future__ import annotations

import json
import os

import torch
from torch import nn

import fp_core as fp
from ao import init_state, lambda_max
from config import config_hash
from data import CHANNEL_KEYS, build_or_load

from .common import UnfoldedNet


def theta_step(phi: torch.Tensor, g: torch.Tensor, mu) -> torch.Tensor:
    """exp(j(angle(phi) + mu * 2 Re{-j conj(phi) g})); mu real, broadcastable to phi."""
    theta = torch.angle(phi)
    mu = torch.as_tensor(mu, device=theta.device).to(theta.dtype)
    grad = 2.0 * (-1j * phi.conj() * g).real
    return torch.polar(torch.ones_like(theta), theta + mu * grad).to(phi.dtype)


class FONR(UnfoldedNet):
    def __init__(self, L: int = 10, T: int = 3, mu_init: float = 1.0):
        super().__init__(L)
        self.T = int(T)
        self.mu = nn.Parameter(torch.full((self.L, self.T), float(mu_init)))

    def phase_update(self, l, ch, p_new, phi, aux):
        for t in range(self.T):
            g = fp.grad_matrix_free(ch, p_new, phi, aux)
            phi = theta_step(phi, g, self.mu[l, t])
        return phi


# ---------------------------------------------------------------- mu initialisation
_MU_KEYS = CHANNEL_KEYS  # mu0 depends only on the training channels


def fonr_mu_init(cfg, chunk: int = 1000) -> float:
    """mu0 = 1/(2 median lambda_max(A_n)) over the training split (efficient mode),
    at the initial state after one deterministic block; entries with lambda = 0
    (IRS with no users) are excluded. Cached in {cfg.root}/logs/fonr_mu_init.json."""
    key = config_hash(cfg, _MU_KEYS)
    path = os.path.join(cfg.root, "logs", "fonr_mu_init.json")
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if key in cache:
            return float(cache[key]["mu_init"])
    d = build_or_load(cfg, "train")
    ch_all = d["ch"].to("cpu", torch.complex128)
    lams = []
    for s in range(0, ch_all.B, chunk):
        ch = ch_all.slice(slice(s, s + chunk))
        p0, phi0 = init_state(ch, d["theta0"][s:s + chunk])
        p_new, aux = fp.deterministic_block(ch, p0, phi0)
        lams.append(lambda_max(ch, p_new, aux, "efficient").reshape(-1))
    lam = torch.cat(lams)
    pos = lam[lam > 0]
    med = float(torch.quantile(pos, 0.5))
    mu0 = 1.0 / (2.0 * med)
    cache[key] = {"mu_init": mu0, "median_lambda_max": med, "n_entries": int(pos.numel()),
                  "n_zero": int((lam <= 0).sum()), "split": "train", "mode": "efficient"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)
    return mu0
