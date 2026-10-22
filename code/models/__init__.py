"""Neural unfolded models (spec §6.2-6.5)."""
from __future__ import annotations

from .common import Trajectory, UnfoldedNet


def build_model(name: str, cfg, **kw) -> UnfoldedNet:
    name = name.lower()
    if name == "fonr":
        from .fonr import FONR, fonr_mu_init
        mu_init = kw.pop("mu_init", None)
        if mu_init is None:
            mu_init = fonr_mu_init(cfg)
        return FONR(L=kw.pop("L", cfg.L), T=kw.pop("T", cfg.T), mu_init=mu_init)
    if name == "maso":
        import math
        from . import fonr as _fonr
        from .maso import MASO
        step_scale = kw.pop("step_scale", None)
        if step_scale is None:  # c = 2 mu0 / ln 2 -> untrained step c*Softplus(0) = 2 mu0
            step_scale = 2.0 * _fonr.fonr_mu_init(cfg) / math.log(2.0)
        return MASO(L=kw.pop("L", cfg.L), N=kw.pop("N", cfg.N),
                    M_train=kw.pop("M_train", cfg.maso_M_train), step_scale=step_scale)
    if name == "tias":
        raise NotImplementedError(f"{name} is implemented in a later task")
    raise ValueError(f"unknown model {name!r}")
