"""All datasets and AO teacher labels (efficient FP-AO, complex128)."""
from __future__ import annotations

import os
import time

import torch

import ao
import atomic
import data
import fp_core as fp
from config import Config, config_hash

REF_MAX_ITER = 1000
_VERSION = "teacher-v1-ref1000"


def _teacher_path(cfg: Config) -> str:
    return os.path.join(cfg.root, "datasets", "teacher_labels.pt")


def _hash(cfg: Config) -> str:
    return config_hash(cfg, data.CHANNEL_KEYS + ("ao_tol", "ao_max_iter")) + ":" + _VERSION


def build_all_datasets(cfg: Config) -> dict:
    out = {}
    for s in ("train", "val", "test"):
        data.build_or_load(cfg, s)
        out[s] = data._path(cfg, s, None)
    pts = {"sweep_M": sorted(set(cfg.sweep_M) | set(cfg.zeroshot_M)),
           "sweep_K": list(cfg.sweep_K), "sweep_power": list(cfg.sweep_crn_dbm)}
    for s, ps in pts.items():
        for p in ps:
            data.build_or_load(cfg, s, p)
            out[f"{s}_{int(p)}"] = data._path(cfg, s, p)
            print(f"dataset {s} {p} ready", flush=True)
    return out


def _run_split(cfg, split, device, chunk, with_ref):
    d = data.build_or_load(cfg, split)
    ch, theta0 = d["ch"], d["theta0"]
    B = ch.hd.shape[0]
    acc = {}

    def put(k, v):
        acc.setdefault(k, []).append(v.cpu())

    for lo in range(0, B, chunk):
        sl = slice(lo, min(lo + chunk, B))
        c = ch.slice(sl).to(device, torch.complex128)
        p0, phi0 = ao.init_state(c, theta0[sl].to(device))
        r = ao.run_ao(c, p0, phi0, mode="efficient", tol=cfg.ao_tol, max_iter=cfg.ao_max_iter)
        put("phi_star", r.phi)
        put("p_star", r.p)
        put("wsr_ao", fp.wsr(c, r.p, r.phi))
        put("n_iter", r.n_iter)
        put("wsr_traj", r.wsr_traj)
        if with_ref:
            rr = ao.run_ao(c, p0, phi0, mode="efficient", tol=cfg.ao_tol, max_iter=REF_MAX_ITER)
            put("phi_ref1000", rr.phi)
            put("p_ref1000", rr.p)
            put("wsr_ref1000", fp.wsr(c, rr.p, rr.phi))
            put("n_iter_ref1000", rr.n_iter)
        print(f"teacher {split} {sl.stop}/{B} meanWSR={float(acc['wsr_ao'][-1].mean()):.4f} "
              f"t={time.strftime('%H:%M:%S')}", flush=True)
    return {k: torch.cat(v) for k, v in acc.items()}


def build_teacher_labels(cfg: Config, splits=("train", "val", "test"), device=None, chunk: int = 500) -> str:
    device = device or cfg.eval_device
    path = _teacher_path(cfg)
    h = _hash(cfg)
    blob = {}
    if os.path.exists(path):
        old = torch.load(path, weights_only=False)
        if old.get("config_hash") == h:
            blob = old
    if blob and all(s in blob["splits"] for s in splits):
        return path
    blob = blob or {"config_hash": h, "splits": {}}
    for s in splits:
        if s not in blob["splits"]:
            blob["splits"][s] = _run_split(cfg, s, device, chunk, with_ref=(s == "test"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    atomic.replace(tmp, path)
    return path


def load_teacher(cfg: Config, split: str) -> dict:
    blob = torch.load(_teacher_path(cfg), weights_only=False)
    if blob["config_hash"] != _hash(cfg):
        raise RuntimeError("teacher labels stale: rebuild with build_teacher_labels")
    return blob["splits"][split]
