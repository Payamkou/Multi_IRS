"""Cached dataset construction."""
from __future__ import annotations

import os

import torch

from channels import ChannelBatch, generate_channels
from config import Config, seed_for

_SWEEPS = ("sweep_M", "sweep_K", "sweep_power")


def _path(cfg: Config, split: str, point) -> str:
    name = split if point is None else f"{split}_{point}"
    return os.path.join(cfg.root, "datasets", name + ".pt")


def build_or_load(cfg: Config, split: str, point=None) -> dict:
    path = _path(cfg, split, point)
    if os.path.exists(path):
        blob = torch.load(path, weights_only=False)
        return {"ch": ChannelBatch(**blob["ch"]), "theta0": blob["theta0"], "meta": blob["meta"]}
    kw = {}
    if split in _SWEEPS:
        if point is None:
            raise ValueError(f"{split} requires a point")
        kw = {"sweep_M": {"M": int(point)}, "sweep_K": {"K": int(point)},
              "sweep_power": {"crn_dbm": float(point)}}[split]
        B = cfg.n_test
    else:
        B = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}[split]
    seed = seed_for(split, point)
    chb, theta0 = generate_channels(cfg, B, seed, **kw)
    meta = {"split": split, "point": point, "seed": seed, "init_seed": seed + 7, "B": B,
            "M": chb.M, "K": chb.K, "N": chb.N,
            "crn_dbm": kw.get("crn_dbm", cfg.crn_dbm)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {"ch": {"hd": chb.hd, "b": chb.b, "x": chb.x, "w": chb.w, "P": chb.P, "sigma2": chb.sigma2},
            "theta0": theta0, "meta": meta}
    torch.save(blob, path)
    return {"ch": chb, "theta0": theta0, "meta": meta}
