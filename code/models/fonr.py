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
_MU_KEYS = CHANNEL_KEYS + ("L", "T", "loss_a")  # lambda stats: train channels; q-probe: L, T, loss
_MU_VERSION = "mu-v2-qprobe"
Q_GRID = (0.5, 0.9, 0.99, 1.0)
PROBE_SAMPLES, PROBE_BATCH = 1000, 200


def _probe_device(cfg) -> str:
    dev = cfg.train_device
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


def _probe_q(cfg, mu0: float, batches) -> tuple:
    """complex64 forward+backward of loss_deep_supervision for FONR(mu0) built at seed
    model_init on each (ch, p0, phi0) batch -> (all finite, max |grad|)."""
    from config import seed_for
    from train import loss_deep_supervision  # lazy: train imports models
    dev = batches[0][0].b.device
    torch.manual_seed(seed_for("model_init"))
    net = FONR(L=cfg.L, T=cfg.T, mu_init=mu0).to(dev)
    finite, gmax = True, 0.0
    for ch, p0, phi0 in batches:
        net.zero_grad(set_to_none=True)
        loss = loss_deep_supervision(ch, net(ch, p0, phi0), cfg.loss_a)
        loss.backward()
        g = net.mu.grad
        ok = bool(torch.isfinite(loss.detach())) and g is not None and bool(torch.isfinite(g).all())
        finite = finite and ok
        if g is not None:  # NaN-propagating max (Python max() would drop a NaN)
            gmax = float(torch.maximum(torch.tensor(gmax), g.detach().abs().max().double().cpu()))
    return finite, gmax


def _mu_entry(cfg, chunk: int = 1000) -> dict:
    """Compute once and cache in {cfg.root}/logs/fonr_mu_init.json (keyed by config hash).

    median_mu = 1/(2 median lambda_max) over the training split (efficient mode) at the
    initial state after one deterministic block; lambda = 0 entries (IRS with no users) excluded.
    mu_init   = 1/(2 Q_q(lambda_max)) with q the first of Q_GRID whose complex64 deep-supervision
    forward+backward on the first 1000 TRAIN samples (batches of 200, cfg.train_device) gives
    finite loss and grads in all batches (pre-registered, train-only rule); none -> q = 1.0,
    recorded as failure."""
    key = config_hash(cfg, _MU_KEYS) + ":" + _MU_VERSION
    path = os.path.join(cfg.root, "logs", "fonr_mu_init.json")
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if key in cache:
            return cache[key]
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

    dev = _probe_device(cfg)
    n = min(PROBE_SAMPLES, d["ch"].B)
    batches = []
    for s in range(0, n, PROBE_BATCH):
        sl = slice(s, min(s + PROBE_BATCH, n))
        c = d["ch"].slice(sl).to(dev, torch.complex64)
        p0, phi0 = init_state(c, d["theta0"][sl])
        batches.append((c, p0, phi0))
    per_q, chosen = {}, None
    for q in Q_GRID:
        lam_q = float(torch.quantile(pos, q)) if q < 1.0 else float(pos.max())
        mu_q = 1.0 / (2.0 * lam_q)
        finite, gmax = _probe_q(cfg, mu_q, batches)
        per_q[str(q)] = {"mu": mu_q, "lambda_q": lam_q, "finite": bool(finite), "max_grad_abs": gmax}
        if finite and chosen is None:
            chosen = q
            break  # first finite q wins; later q are not needed
    failed = chosen is None
    if failed:
        chosen = 1.0
    entry = {"median_mu": 1.0 / (2.0 * med), "median_lambda_max": med, "chosen_q": chosen,
             "mu_init": per_q[str(chosen)]["mu"], "probe_failed": failed, "per_q": per_q,
             "n_entries": int(pos.numel()), "n_zero": int((lam <= 0).sum()), "split": "train",
             "mode": "efficient", "probe_device": dev, "probe_samples": n, "config_hash": key}
    cache[key] = entry
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)
    return entry


def fonr_mu_init(cfg) -> float:
    """FONR initial step mu0 = 1/(2 Q_q(lambda_max)) with the chosen q (see _mu_entry)."""
    return float(_mu_entry(cfg)["mu_init"])


def fonr_mu_median(cfg) -> float:
    """Median-based mu = 1/(2 median lambda_max) (MASO step_scale reference; unchanged)."""
    return float(_mu_entry(cfg)["median_mu"])
