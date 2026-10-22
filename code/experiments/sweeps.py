"""Experiments 4-8 (spec §7 rows 4-8, §7.1, §7.2) and the supplementary GPU timing.

All neural models are the trained default-setting checkpoints (M = cfg.M, K = cfg.K, CRN 20 dBm),
evaluated WITHOUT retraining. FONR/TIAS run natively at any M/K; trained MASO zero-pads (M < M_train)
or truncates (M > M_train, lambda = 0 on the tail) per spec §6.3. The one exception is the
runtime-vs-M experiment, where MASO is instantiated untrained at each M (``MASO(M_train=M)``,
step_scale copied from the trained checkpoint; timing does not depend on weight values).

WSR experiments: AO is run fresh (efficient mode, cfg.ao_tol, cfg.ao_max_iter, complex128 on
cfg.eval_device) on every sweep set; its per-sample results are cached in the npz. Neural WSR is the
layer-L output, complex128, all samples. ``ao_baseline_max_iter`` (= cfg.ao_max_iter) is stored.

Timing experiments: batch 1, ``timing.time_steps`` with cfg.timing_warmup discarded runs; per-sample
time = mean of the first cfg.L (=10) AO iterations / neural layers; samples = first
cfg.timing_subset sweep samples (None -> all), the same subset for every method. AO is timed with
``ao.AOStepper`` (explicit and efficient), never ``run_ao``. Paper runs: CPU, complex128,
torch threads = cfg.timing_threads.

Result files (``{cfg.root}/results``; each with a per-(method, point) CSV of the same stem)
-------------------------------------------------------------------------------------------
* ``exp_04_runtime_vs_M.npz``  ``time_{m}_M{M}`` (S,) s, m in ao_explicit/ao_efficient/fonr/maso/tias;
  ``points``, ``maso_M_train``, ``timing_idx``, ``timing_n_samples``, ``timing_threads``, ``device``,
  ``cdtype``. Fig. ``avg_time_vs_Number_of_RIS_M`` (log10 y-axis).
* ``exp_05_power_sweep.npz``   ``wsr_{m}_P{dBm}`` (B,), m in ao/fonr/maso/tias; ``n_iter_ao_P{dBm}``,
  ``wsr_init_P{dBm}``, ``eta_ratio_of_means_{n}_P{dBm}``, ``eta_mean_of_ratios_{n}_P{dBm}`` (%).
  Fig. ``etha_vs_CRN_transmit_power`` plots the ratio of means.
* ``exp_06_wsr_vs_K.npz``      ``wsr_{m}_K{K}`` (B,), ``n_iter_ao_K{K}``, ``wsr_init_K{K}``.
  Fig. ``WSR_vs_Number_of_active_users_K``.
* ``exp_07_runtime_vs_K.npz``  ``time_{m}_K{K}`` (S,) as exp_04 (trained MASO).
  Fig. ``avg_time_vs_Number_of_active_users_K``.
* ``exp_08_zeroshot_M.npz``    ``wsr_{m}_M{M}`` (B,) for M in ``M_values`` = cfg.zeroshot_M ∪ {cfg.M};
  ``M_reference`` = cfg.M (training M, drawn as the reference point). Fig. ``WSR_vs_M_ZeroShot``.
* ``exp_supp_gpu_timing.npz``  (no figure) exp_04/exp_07 timings (keys as above) plus exp_01-style
  step timings on the default test split: ``step_time_{n}`` (S, L) s for the nets,
  ``ao_cumtime_{mode}`` (S,) s to convergence and ``ao_niter_{mode}`` (S,), ``ao_steptime_{mode}``
  (S,) mean s/iteration; all methods on cuda, complex64.
Every npz also carries ``cfg_json``, ``cfg_hash``, ``artifact_fingerprint`` (checkpoints, teacher
labels and the sweep datasets used); a cache is reused only if all match (experiments.cache).
Non-default ``device``/``cdtype`` runs of exp_04/exp_07 go to ``..._{device}_{cdtype}.npz``, no figure.

Summary keys (each run_* returns ``(npz_path, summary)``; all floats unless noted)
-----------------------------------------------------------------------------------
Methods m: ao_explicit, ao_efficient, fonr, maso, tias for timing; ao, fonr, maso, tias for WSR.
* run_runtime_vs_M:  ``runtime_ms_{m}_M{M}`` (mean), ``runtime_median_ms_{m}_M{M}``,
  ``timing_threads``, ``timing_n_samples``
* run_power_sweep:   ``eta_ratio_of_means_{n}_P{dBm}``, ``eta_mean_of_ratios_{n}_P{dBm}`` (%),
  ``eta_range_{n}`` = (min, max) over powers of the ratio of means (tuple of 2 floats),
  ``eta_min_{n}``, ``eta_max_{n}``, ``wsr_{m}_P{dBm}`` (mean WSR), ``ao_n_iter_mean_P{dBm}``,
  ``ao_baseline_max_iter``
* run_wsr_vs_K:      ``wsr_{m}_K{K}`` (mean), ``eta_{n}_K{K}`` (ratio of means, %),
  ``ao_n_iter_mean_K{K}``, ``ao_baseline_max_iter``
* run_runtime_vs_K:  ``runtime_ms_{m}_K{K}``, ``runtime_median_ms_{m}_K{K}``, ``timing_threads``,
  ``timing_n_samples``
* run_zeroshot_M:    ``wsr_{m}_M{M}_zeroshot`` (mean), ``eta_{n}_M{M}_zeroshot`` (%),
  ``M_reference``, ``ao_baseline_max_iter``
* run_gpu_timing:    ``gpu_available`` (0/1); if 1 also ``runtime_ms_{m}_M{M}``, ``runtime_ms_{m}_K{K}``,
  ``step_time_ms_{m}`` (mean per layer / per AO iteration), ``ao_conv_time_ms_{mode}``,
  ``ao_conv_n_failed_{mode}``, ``timing_threads``, ``timing_n_samples``. Returns
  ``(None, {"gpu_available": 0.0})`` without CUDA. GPU complex64 only: a sample whose cuSOLVER
  eigvalsh raises LinAlgError (explicit AO, ill-conditioned A_n) is stored as NaN; means/medians
  are NaN-aware and ``runtime_n_failed_{m}_{tag}{x}`` (present only when > 0) counts them.
  CPU runs never swallow the error.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

import ao
import data
import plotting
import timing
import train
from config import Config, config_hash, seed_for
from experiments.cache import artifact_fingerprint, check_cache, fingerprint_array
from experiments.convergence import (_atomic_npz, _ci95, _minor_label, _results_dir, _timing_idx, _wsr_per_layer,
                                     _write_csv, load_trained_models)
from models.maso import MASO

NETS = ("fonr", "maso", "tias")
AO_MODES = ("explicit", "efficient")
TIME_METHODS = ("ao_explicit", "ao_efficient") + NETS
WSR_METHODS = ("ao",) + NETS
C64, C128 = torch.complex64, torch.complex128
WSR_KEYS = train.TRAIN_KEYS + ("ao_tol", "ao_max_iter", "n_test", "eval_device")
TIME_KEYS = train.TRAIN_KEYS + ("n_test", "timing_threads", "timing_warmup", "timing_subset")
LABEL = {"ao": "FP-AO", "ao_explicit": "FP-AO (explicit)", "ao_efficient": "FP-AO (efficient)",
         **plotting.NET_LABEL}


# ---------------------------------------------------------------- helpers
def _cname(cdtype) -> str:
    return str(cdtype).replace("torch.", "")


def _ds(split: str, points) -> list:
    return [f"datasets/{split}_{int(x)}.pt" for x in points]


def _meta(cfg, keys, fpr) -> dict:
    return {"cfg_json": np.array(json.dumps(cfg.to_dict())), "cfg_hash": np.array(config_hash(cfg, keys)),
            "artifact_fingerprint": fingerprint_array(fpr)}


def _on(model, device, cdtype):
    return copy.deepcopy(model).to_precision(cdtype).to(device).eval()


def _untrained_maso(cfg: Config, M: int, step_scale: float) -> MASO:
    """MASO built at M_train = M with fixed-seed untrained weights (timing only)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for("model_init"))
        return MASO(L=cfg.L, N=cfg.N, M_train=int(M), step_scale=step_scale)


def _sample_times(stepper, d, idx, cfg, device, cdtype) -> np.ndarray:
    """(S,) per-sample mean of the first cfg.L step times, batch 1.

    On CUDA only, a sample whose step raises ``torch.linalg.LinAlgError`` (cuSOLVER eigvalsh in
    complex64 can fail on an ill-conditioned A_n in explicit AO) is recorded as NaN; on CPU it raises."""
    out = np.empty(len(idx))
    for j, i in enumerate(idx):
        try:
            out[j] = timing.time_steps(stepper, d["ch"].slice([int(i)]), d["theta0"][[int(i)]], cfg.L,
                                       device, cdtype, warmup=cfg.timing_warmup).mean()
        except torch.linalg.LinAlgError:
            if torch.device(device).type != "cuda":
                raise
            out[j] = np.nan
    return out


def _fig(npz_fn, z, name, cfg):
    import matplotlib.pyplot as plt
    fig = npz_fn(z)
    out = plotting.save_fig(fig, name, cfg.paper_dir)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- generic sweep bodies
def _runtime_arrays(cfg, split, points, tag, device, cdtype, models) -> dict:
    """time_{m}_{tag}{x} for every method and point (threads must already be set)."""
    idx = _timing_idx(cfg)
    out = {}
    nets = {n: _on(m, device, cdtype) for n, m in models.items()}
    scale = float(models["maso"].step_scale)
    for x in points:
        d = data.build_or_load(cfg, split, x)
        for mode in AO_MODES:
            out[f"time_ao_{mode}_{tag}{x}"] = _sample_times(ao.AOStepper(mode), d, idx, cfg, device, cdtype)
        for n in NETS:
            net = nets[n]
            if tag == "M" and n == "maso":
                net = _on(_untrained_maso(cfg, x, scale), device, cdtype)
            out[f"time_{n}_{tag}{x}"] = _sample_times(net, d, idx, cfg, device, cdtype)
        msg = ", ".join(f"{m} {1e3 * np.nanmean(out[f'time_{m}_{tag}{x}']):.4f}" for m in TIME_METHODS)
        nf = {m: int(np.isnan(out[f"time_{m}_{tag}{x}"]).sum()) for m in TIME_METHODS}
        if any(nf.values()):
            print(f"[runtime {tag}={x}] WARNING samples failed (LinAlgError, recorded NaN): {nf}", flush=True)
        print(f"[runtime {tag}={x} {device}/{_cname(cdtype)}] ms/iter: {msg}", flush=True)
    return out


def _wsr_arrays(cfg, split, points, tag, models, chunk: int = 250) -> dict:
    """Fresh efficient AO (cfg tol/max_iter, complex128) and layer-L net WSR per point."""
    out = {}
    dev = cfg.eval_device
    for x in points:
        d = data.build_or_load(cfg, split, x)
        ch, theta0 = d["ch"], d["theta0"]
        w, it, w0 = [], [], []
        for lo in range(0, ch.B, chunk):
            sl = slice(lo, min(lo + chunk, ch.B))
            c = ch.slice(sl).to(dev, C128)
            p0, phi0 = ao.init_state(c, theta0[sl].to(dev))
            r = ao.run_ao(c, p0, phi0, mode="efficient", tol=cfg.ao_tol, max_iter=cfg.ao_max_iter)
            w.append(r.wsr_traj[:, -1].cpu().numpy())
            w0.append(r.wsr_traj[:, 0].cpu().numpy())
            it.append(r.n_iter.cpu().numpy())
        out[f"wsr_ao_{tag}{x}"] = np.concatenate(w)
        out[f"wsr_init_{tag}{x}"] = np.concatenate(w0)
        out[f"n_iter_ao_{tag}{x}"] = np.concatenate(it)
        for n, m in models.items():
            out[f"wsr_{n}_{tag}{x}"] = _wsr_per_layer(m, ch, theta0, dev)[:, -1]
        msg = ", ".join(f"{m} {out[f'wsr_{m}_{tag}{x}'].mean():.4f}" for m in WSR_METHODS)
        print(f"[wsr {tag}={x}] mean WSR: {msg}", flush=True)
    return out


def _with_threads(cfg, fn):
    prev = torch.get_num_threads()
    threads = timing.set_timing_threads(cfg.timing_threads)
    try:
        return fn(), threads
    finally:
        torch.set_num_threads(prev)


def _runtime_csv(path, z, tag, points):
    rows = []
    for x in points:
        for m in TIME_METHODS:
            t = z[f"time_{m}_{tag}{x}"]
            rows.append({"method": m, tag: int(x), "mean_ms": 1e3 * np.nanmean(t),
                         "median_ms": 1e3 * np.nanmedian(t), "n_samples": int(np.isfinite(t).sum()),
                         "n_failed": int(np.isnan(t).sum())})
    _write_csv(path, ["method", tag, "mean_ms", "median_ms", "n_samples", "n_failed"], rows)


def _wsr_csv(path, z, tag, points, extra=None):
    rows = []
    for x in points:
        ao_m = z[f"wsr_ao_{tag}{x}"].mean()
        for m in WSR_METHODS:
            w = z[f"wsr_{m}_{tag}{x}"]
            row = {"method": m, tag: int(x), "wsr_mean": w.mean(), "wsr_ci95": float(_ci95(w[:, None])[0]),
                   "wsr_median": np.median(w), "eta_ratio_of_means_pct": 100 * w.mean() / ao_m,
                   "eta_mean_of_ratios_pct": 100 * np.mean(w / z[f"wsr_ao_{tag}{x}"]),
                   "ao_n_iter_mean": z[f"n_iter_ao_{tag}{x}"].mean()}
            rows.append(row)
    _write_csv(path, list(rows[0]), rows)


def _summ_runtime(z, tag, points) -> dict:
    s = {"timing_threads": float(z["timing_threads"]), "timing_n_samples": float(z["timing_n_samples"])}
    for x in points:
        for m in TIME_METHODS:
            t = z[f"time_{m}_{tag}{x}"]
            s[f"runtime_ms_{m}_{tag}{x}"] = float(1e3 * np.nanmean(t))
            s[f"runtime_median_ms_{m}_{tag}{x}"] = float(1e3 * np.nanmedian(t))
            if np.isnan(t).any():  # GPU complex64 only (see _sample_times)
                s[f"runtime_n_failed_{m}_{tag}{x}"] = float(np.isnan(t).sum())
    return s


def _summ_wsr(z, tag, points, suffix="", eta_prefix="eta_") -> dict:
    s = {"ao_baseline_max_iter": float(z["ao_baseline_max_iter"])}
    for x in points:
        ao_m = float(z[f"wsr_ao_{tag}{x}"].mean())
        for m in WSR_METHODS:
            s[f"wsr_{m}_{tag}{x}{suffix}"] = float(z[f"wsr_{m}_{tag}{x}"].mean())
        for n in NETS:
            s[f"{eta_prefix}{n}_{tag}{x}{suffix}"] = float(100 * z[f"wsr_{n}_{tag}{x}"].mean() / ao_m)
        s[f"ao_n_iter_mean_{tag}{x}"] = float(z[f"n_iter_ao_{tag}{x}"].mean())
    return s


def _load(p):
    return dict(np.load(p)) if not isinstance(p, dict) else p


# ---------------------------------------------------------------- runtime experiments (4, 7)
# ---------------------------------------------------------------- shared cache specs / summaries
# exp -> (file stem, dataset split, points(cfg), tag, sweep config key); used by run_* AND report.py
EXPERIMENTS = {
    "runtime_M": ("exp_04_runtime_vs_M", "sweep_M", lambda c: tuple(c.sweep_M), "M", "sweep_M"),
    "runtime_K": ("exp_07_runtime_vs_K", "sweep_K", lambda c: tuple(c.sweep_K), "K", "sweep_K"),
    "power": ("exp_05_power_sweep", "sweep_power", lambda c: tuple(c.sweep_crn_dbm), "P", "sweep_crn_dbm"),
    "wsr_K": ("exp_06_wsr_vs_K", "sweep_K", lambda c: tuple(c.sweep_K), "K", "sweep_K"),
    "zeroshot_M": ("exp_08_zeroshot_M", "sweep_M", lambda c: zeroshot_points(c), "M", "zeroshot_M"),
}
GPU_EXP = "gpu"
GPU_KEYS = TIME_KEYS + ("sweep_M", "sweep_K", "ao_tol", "ao_max_iter")


def cache_spec(cfg: Config, exp: str, device: str = "cpu", cdtype=C128) -> tuple:
    """(npz path, cfg-hash keys, extra fingerprint files, extra npz keys or None) of experiment ``exp``
    (a key of EXPERIMENTS or "gpu"). The single source of the cache rule for run_* and report.py."""
    res = os.path.join(cfg.root, "results")
    if exp == GPU_EXP:
        files = _ds("sweep_M", cfg.sweep_M) + _ds("sweep_K", cfg.sweep_K) + ["datasets/test.pt"]
        return os.path.join(res, "exp_supp_gpu_timing.npz"), GPU_KEYS, files, None
    stem, split, pts, _, sweep_key = EXPERIMENTS[exp]
    files = _ds(split, pts(cfg))
    if exp.startswith("runtime_"):
        if not (torch.device(device).type == "cpu" and cdtype == C128):
            stem = f"{stem}_{torch.device(device).type}_{_cname(cdtype)}"
        return (os.path.join(res, stem + ".npz"), TIME_KEYS + (sweep_key,), files,
                {"device": str(device), "cdtype": _cname(cdtype)})
    return os.path.join(res, stem + ".npz"), WSR_KEYS + (sweep_key,), files, None


def is_cached(cfg: Config, exp: str, device: str = "cpu", cdtype=C128) -> bool:
    """True if experiment ``exp``'s npz exists and matches the current config and artifacts."""
    path, keys, files, extra = cache_spec(cfg, exp, device, cdtype)
    return check_cache(path, cfg, keys, artifact_fingerprint(cfg, extra=files), extra)


def summarize_power(z, pts) -> dict:
    """Summary of exp_05 (keys in the module docstring)."""
    s = _summ_wsr(z, "P", pts, eta_prefix="eta_ratio_of_means_")
    for n in NETS:
        rom = [float(z[f"eta_ratio_of_means_{n}_P{P}"]) for P in pts]
        for P in pts:
            s[f"eta_mean_of_ratios_{n}_P{P}"] = float(z[f"eta_mean_of_ratios_{n}_P{P}"])
        s[f"eta_range_{n}"] = (min(rom), max(rom))
        s[f"eta_min_{n}"], s[f"eta_max_{n}"] = min(rom), max(rom)
    return s


def summarize_gpu(z, Ms, Ks) -> dict:
    """Summary of the supplementary GPU timing (keys in the module docstring)."""
    s = {"gpu_available": 1.0}
    s.update(_summ_runtime(z, "M", Ms))
    s.update(_summ_runtime(z, "K", Ks))
    for m in TIME_METHODS:
        s[f"step_time_ms_{m}"] = float(1e3 * np.nanmean(_step_times(z, m)))
    for mode in AO_MODES:
        s[f"ao_conv_time_ms_{mode}"] = float(1e3 * np.nanmean(z[f"ao_cumtime_{mode}"]))
        s[f"ao_conv_n_failed_{mode}"] = float(np.isnan(z[f"ao_cumtime_{mode}"]).sum())
    return s


def summarize(cfg: Config, exp: str, z) -> dict:
    """Summary dict of experiment ``exp`` from its npz (exactly what the matching run_* returns)."""
    z = _load(z)
    if exp == GPU_EXP:
        return summarize_gpu(z, tuple(cfg.sweep_M), tuple(cfg.sweep_K))
    _, _, pts, tag, _ = EXPERIMENTS[exp]
    pts = pts(cfg)
    if exp.startswith("runtime_"):
        return _summ_runtime(z, tag, pts)
    if exp == "power":
        return summarize_power(z, pts)
    if exp == "zeroshot_M":
        s = _summ_wsr(z, "M", pts, suffix="_zeroshot")
        s["M_reference"] = float(z["M_reference"])
        return s
    return _summ_wsr(z, tag, pts)


def _run_runtime(cfg, exp, device, cdtype, force_rerun, plot_fn, fig_name):
    default = torch.device(device).type == "cpu" and cdtype == C128
    _, split, pts_fn, tag, _ = EXPERIMENTS[exp]
    points = pts_fn(cfg)
    _results_dir(cfg)
    path, keys, files, extra = cache_spec(cfg, exp, device, cdtype)
    if force_rerun or not check_cache(path, cfg, keys, artifact_fingerprint(cfg, extra=files), extra):
        models = load_trained_models(cfg)
        out, threads = _with_threads(cfg, lambda: _runtime_arrays(cfg, split, points, tag, device, cdtype, models))
        idx = _timing_idx(cfg)
        out.update(points=np.array(points), timing_idx=idx, timing_n_samples=np.int64(len(idx)),
                   timing_threads=np.int64(threads), device=np.array(str(device)), cdtype=np.array(_cname(cdtype)),
                   **_meta(cfg, keys, artifact_fingerprint(cfg, extra=files)))
        if tag == "M":
            out["maso_M_train"] = np.array(points)
        _atomic_npz(path, **out)
        _runtime_csv(path[:-4] + ".csv", out, tag, points)
    z = _load(path)
    if default:
        _fig(plot_fn, z, fig_name, cfg)
    return Path(path), summarize(cfg, exp, z)


def run_runtime_vs_M(cfg: Config, device: str = "cpu", cdtype=C128, force_rerun: bool = False):
    """exp_04: per-iteration/per-layer time vs M (K = cfg.K); untrained MASO(M_train=M)."""
    return _run_runtime(cfg, "runtime_M", device, cdtype, force_rerun, plot_fig4, "avg_time_vs_Number_of_RIS_M")


def run_runtime_vs_K(cfg: Config, device: str = "cpu", cdtype=C128, force_rerun: bool = False):
    """exp_07: per-iteration/per-layer time vs K (M = cfg.M); trained models."""
    return _run_runtime(cfg, "runtime_K", device, cdtype, force_rerun, plot_fig7,
                        "avg_time_vs_Number_of_active_users_K")


# ---------------------------------------------------------------- WSR experiments (5, 6, 8)
def _run_wsr(cfg, exp, force_rerun, post=None):
    _, split, pts_fn, tag, _ = EXPERIMENTS[exp]
    points = pts_fn(cfg)
    _results_dir(cfg)
    path, keys, files, extra = cache_spec(cfg, exp)
    if force_rerun or not check_cache(path, cfg, keys, artifact_fingerprint(cfg, extra=files), extra):
        out = _wsr_arrays(cfg, split, points, tag, load_trained_models(cfg))
        out.update(points=np.array(points), ao_baseline_max_iter=np.int64(cfg.ao_max_iter),
                   **_meta(cfg, keys, artifact_fingerprint(cfg, extra=files)))
        if post:
            post(out)
        _atomic_npz(path, **out)
        _wsr_csv(path[:-4] + ".csv", out, tag, points)
    return path


def _eta_post(points):
    def post(out):
        for P in points:
            a = out[f"wsr_ao_P{P}"]
            for n in NETS:
                w = out[f"wsr_{n}_P{P}"]
                out[f"eta_ratio_of_means_{n}_P{P}"] = np.float64(100 * w.mean() / a.mean())
                out[f"eta_mean_of_ratios_{n}_P{P}"] = np.float64(100 * np.mean(w / a))
    return post


def run_power_sweep(cfg: Config, force_rerun: bool = False):
    """exp_05: efficiency eta vs CRN peak power (PMN = CRN - 10 dB), models trained at 20 dBm."""
    pts = tuple(cfg.sweep_crn_dbm)
    path = _run_wsr(cfg, "power", force_rerun, _eta_post(pts))
    z = _load(path)
    s = summarize(cfg, "power", z)
    _fig(plot_fig5, z, "etha_vs_CRN_transmit_power", cfg)
    return Path(path), s


def run_wsr_vs_K(cfg: Config, force_rerun: bool = False):
    """exp_06: mean WSR vs K (M = cfg.M) for AO and the three nets."""
    path = _run_wsr(cfg, "wsr_K", force_rerun)
    z = _load(path)
    _fig(plot_fig6, z, "WSR_vs_Number_of_active_users_K", cfg)
    return Path(path), summarize(cfg, "wsr_K", z)


def zeroshot_points(cfg: Config) -> tuple:
    """cfg.zeroshot_M plus the training M (reference point), sorted."""
    return tuple(sorted({int(m) for m in cfg.zeroshot_M} | {int(cfg.M)}))


def run_zeroshot_M(cfg: Config, force_rerun: bool = False):
    """exp_08: zero-shot mean WSR vs M with the trained models (MASO pads/truncates, spec §6.3)."""
    pts = zeroshot_points(cfg)

    def post(out):
        out["M_values"] = np.array(pts)
        out["M_reference"] = np.int64(cfg.M)

    path = _run_wsr(cfg, "zeroshot_M", force_rerun, post)
    z = _load(path)
    s = summarize(cfg, "zeroshot_M", z)
    _fig(plot_fig8, z, "WSR_vs_M_ZeroShot", cfg)
    return Path(path), s


# ---------------------------------------------------------------- supplementary GPU timing
def run_gpu_timing(cfg: Config, force_rerun: bool = False):
    """exp_04/exp_07 timings and exp_01 step timings on cuda, complex64, for all methods. No figure."""
    if not torch.cuda.is_available():
        print("[gpu timing] CUDA not available; skipping the supplementary GPU timing", flush=True)
        return None, {"gpu_available": 0.0}
    device, cdtype = "cuda", C64
    _results_dir(cfg)
    path, keys, files, _ = cache_spec(cfg, GPU_EXP)
    Ms, Ks = tuple(cfg.sweep_M), tuple(cfg.sweep_K)
    if force_rerun or not check_cache(path, cfg, keys, artifact_fingerprint(cfg, extra=files)):
        models = load_trained_models(cfg)
        idx = _timing_idx(cfg)

        def body():
            out = _runtime_arrays(cfg, "sweep_M", Ms, "M", device, cdtype, models)
            out.update(_runtime_arrays(cfg, "sweep_K", Ks, "K", device, cdtype, models))
            d = data.build_or_load(cfg, "test")
            for n, m in models.items():
                net = _on(m, device, cdtype)
                out[f"step_time_{n}"] = np.stack([
                    timing.time_steps(net, d["ch"].slice([int(i)]), d["theta0"][[int(i)]], net.L, device, cdtype,
                                      warmup=cfg.timing_warmup) for i in idx])
            for mode in AO_MODES:
                st = ao.AOStepper(mode)
                runs = []
                for i in idx:
                    try:
                        runs.append(timing.time_until_converged(st, d["ch"].slice([int(i)]), d["theta0"][[int(i)]],
                                                                cfg.ao_tol, cfg.ao_max_iter, device, cdtype)[0])
                    except torch.linalg.LinAlgError:  # cuSOLVER complex64 eigvalsh failure -> NaN
                        runs.append(np.array([np.nan]))
                out[f"ao_cumtime_{mode}"] = np.array([r.sum() for r in runs])
                out[f"ao_niter_{mode}"] = np.array([len(r) if np.isfinite(r).all() else -1 for r in runs])
                out[f"ao_steptime_{mode}"] = np.array([r.mean() for r in runs])
                nf = int(np.isnan(out[f"ao_cumtime_{mode}"]).sum())
                if nf:
                    print(f"[gpu timing] WARNING AO {mode}: {nf} samples failed (LinAlgError, NaN)", flush=True)
            print("[gpu timing] exp_01 step timings done", flush=True)
            return out

        out, threads = _with_threads(cfg, body)
        out.update(timing_idx=idx, timing_n_samples=np.int64(len(idx)), timing_threads=np.int64(threads),
                   device=np.array(device), cdtype=np.array(_cname(cdtype)), sweep_M=np.array(Ms),
                   sweep_K=np.array(Ks), gpu_name=np.array(torch.cuda.get_device_name(0)),
                   **_meta(cfg, keys, artifact_fingerprint(cfg, extra=files)))
        _atomic_npz(path, **out)
        rows = []
        for tag, pts in (("M", Ms), ("K", Ks)):
            for x in pts:
                for m in TIME_METHODS:
                    t = out[f"time_{m}_{tag}{x}"]
                    rows.append({"experiment": f"runtime_vs_{tag}", "method": m, "point": int(x),
                                 "mean_ms": 1e3 * np.nanmean(t), "median_ms": 1e3 * np.nanmedian(t),
                                 "n_failed": int(np.isnan(t).sum())})
        for m in TIME_METHODS:
            t = _step_times(out, m)
            rows.append({"experiment": "step_time", "method": m, "point": "",
                         "mean_ms": 1e3 * np.nanmean(t), "median_ms": 1e3 * np.nanmedian(t),
                         "n_failed": int(np.isnan(t).sum())})
        _write_csv(path[:-4] + ".csv", ["experiment", "method", "point", "mean_ms", "median_ms", "n_failed"], rows)
    return Path(path), summarize(cfg, GPU_EXP, _load(path))


def _step_times(z, m) -> np.ndarray:
    """(S,) per-sample mean per-step seconds (nets: per layer; AO: per iteration until convergence)."""
    if m.startswith("ao_"):
        return z[f"ao_steptime_{m[3:]}"]
    return z[f"step_time_{m}"].mean(1)


# ---------------------------------------------------------------- figures
def _plot_lines(ax, z, methods, key, points, value):
    for m in methods:
        st = plotting.STYLE[LABEL[m]]
        ax.plot(points, [value(z[key.format(m=m, x=x)]) for x in points], color=st["color"],
                marker=st["marker"], label=LABEL[m])
    ax.set_xticks(points)


def _plot_runtime(z, tag, xlabel):
    fig, ax = plotting.new_fig()
    pts = [int(x) for x in z["points"]]
    _plot_lines(ax, z, TIME_METHODS, "time_{m}_" + tag + "{x}", pts, lambda t: 1e3 * np.nanmean(t))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(_minor_label))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Avg. time per iteration / layer (ms)")
    ax.legend(loc="best", ncol=2, fontsize=plotting.FONT_SIZE - 2)
    return fig


def plot_fig4(npz):
    """Mean per-iteration/per-layer CPU time vs M, log10 y-axis."""
    return _plot_runtime(_load(npz), "M", "Number of IRS elements $M$")


def plot_fig7(npz):
    """Mean per-iteration/per-layer CPU time vs K."""
    return _plot_runtime(_load(npz), "K", "Number of active users $K$")


def plot_fig5(npz):
    """eta (ratio of means, %) vs CRN peak power."""
    z = _load(npz)
    fig, ax = plotting.new_fig()
    pts = [int(x) for x in z["points"]]
    _plot_lines(ax, z, NETS, "eta_ratio_of_means_{m}_P{x}", pts, float)
    ax.axhline(100.0, color="black", linestyle=(0, (5, 2)), linewidth=0.9, label="FP-AO (100%)")
    ax.set_xlabel("CRN peak transmit power (dBm)")
    ax.set_ylabel(r"Efficiency $\eta$ (%)")
    ax.legend(loc="best", fontsize=plotting.FONT_SIZE - 2)
    return fig


def _plot_wsr(z, tag, xlabel):
    fig, ax = plotting.new_fig()
    pts = [int(x) for x in z["points"]]
    _plot_lines(ax, z, WSR_METHODS, "wsr_{m}_" + tag + "{x}", pts, lambda w: w.mean())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("WSR (bits/s/Hz)")
    return fig, ax


def plot_fig6(npz):
    """Mean WSR vs K, all four methods."""
    fig, ax = _plot_wsr(_load(npz), "K", "Number of active users $K$")
    ax.legend(loc="best", fontsize=plotting.FONT_SIZE - 2)
    return fig


def plot_fig8(npz):
    """Zero-shot mean WSR vs M; the training M is marked as the reference point."""
    z = _load(npz)
    fig, ax = _plot_wsr(z, "M", "Number of IRS elements $M$")
    ax.axvline(float(z["M_reference"]), color="grey", linestyle=":", linewidth=0.9,
               label=f"Training $M$ = {int(z['M_reference'])}")
    ax.legend(loc="best", fontsize=plotting.FONT_SIZE - 2)
    return fig
