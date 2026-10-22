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
        from .maso import MASO
        return MASO(L=kw.pop("L", cfg.L), N=kw.pop("N", cfg.N), M_train=kw.pop("M_train", cfg.maso_M_train))
    if name == "tias":
        raise NotImplementedError(f"{name} is implemented in a later task")
    raise ValueError(f"unknown model {name!r}")
