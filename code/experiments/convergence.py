"""Experiments 1-3 (spec §7 rows 1-3, §7.1, §7.2): convergence vs layer/iteration,
WSR vs wall-clock, and the CDF of the optimality gap.

Raw results (``{cfg.root}/results``)
------------------------------------
``exp_01_convergence.npz`` (+ ``exp_01_convergence.csv`` per-method/per-step table,
``exp_01_convergence_summary.json``):
  * ``wsr_{n}``            (B, L+1) bits, n in fonr/maso/tias; complex128 batched on cfg.eval_device,
                            all test samples; column 0 = shared initial state.
  * ``time_{n}``           (S, L) seconds per layer, CPU complex128 batch 1 (``timing.time_steps``).
  * ``wsr_ao_traj``        (B, ao_max_iter+1) teacher (efficient-mode, AO-100) trajectory, padded.
  * ``wsr_ao100``/``n_iter_ao100``, ``wsr_ao1000``/``n_iter_ao1000`` (B,) teacher test labels.
  * ``ao100_mean``, ``ao1000_mean`` scalars.
  * ``ao_cumtime_{mode}``  (S, T+1) cumulative seconds (0 at step 0), held after convergence,
                            mode in explicit/efficient (``timing.time_until_converged``).
  * ``ao_wsr_timed_{mode}`` (S, T+1) WSR of the batch-1 timing run (held), ``ao_niter_{mode}`` (S,).
  * ``timing_idx`` (S,), ``timing_threads``, ``cfg_json``, ``cfg_hash``, ``ao_baseline_max_iter``,
    ``artifact_fingerprint`` (experiments.cache; checkpoints + teacher labels, size:sha256).
  Cache: reused only if cfg_hash and artifact_fingerprint match, else recomputed with a notice.
  S = cfg.timing_subset (first S test samples) or all; the same subset for every method.

``exp_03_cdf.npz`` (+ ``exp_03_cdf.csv``): ``gap_{n}`` = R_AO100 - R_n(L) and
``gap_vs1000_{n}`` = R_AO1000 - R_n(L) per test sample (bits); ``conv_sha256`` of the
exp_01 npz it was built from (rebuilt whenever that file changes).

Curves
------
* Fig. 1: mean WSR ± 1.96·std/sqrt(B) per step 0..L; AO = teacher trajectory columns 0..10,
  dashed lines at the AO-100 mean ("FP-AO (converged)") and AO-1000 mean ("FP-AO (1000 it.)"); a single
  line "FP-AO (converged, 1000 it.)" when ao_baseline_max_iter = 1000 or the two means coincide.
* Fig. 2: mean WSR (all B) vs mean cumulative time (subset S). AO WSR per step = teacher
  trajectory column t (explicit and efficient iterates are identical); AO time = mean over the
  subset of the cumulative time, a converged sample holding its final time. Each AO curve runs to
  the largest iteration count in the timing subset. Log x-axis, so step 0 (t = 0) is not drawn.

``summarize_convergence`` keys (floats; consumed by the report)
--------------------------------------------------------------
* ``ao_baseline_max_iter`` (= cfg.ao_max_iter, the "AO-100" iteration cap)
* ``ao100_mean``, ``ao1000_mean``, ``n_test``, ``n_timing``, ``timing_threads``
* ``recovery_pct_layer{4,6,8}_{fonr,maso,tias,ao}`` = 100·(R̄_l − R̄_0)/(R̄_AO100 − R̄_0)
  (``ao`` uses AO iteration l of the teacher trajectory)
* ``time_to_98pct_{m}_ms``, ``time_to_99pct_{m}_ms``, m in fonr/maso/tias/ao_explicit/ao_efficient:
  mean cumulative time at the first step whose mean WSR ≥ 0.98/0.99·ao100_mean (NaN if never)
* ``ao_conv_time_ms_{explicit,efficient}`` (mean), ``ao_conv_time_median_ms_{mode}``,
  ``ao_conv_iters_mean_{mode}``
* ``layer_time_ms_{n}`` mean total L-layer time
* ``gap_p95_{n}``, ``gap_min_{n}``, ``gap_median_{n}``, ``gap_mean_{n}`` vs AO-100 and the same with
  suffix ``_vs1000`` vs AO-1000 (e.g. ``gap_p95_fonr_vs1000``)
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import ao
import atomic
import data
import fp_core as fp
import plotting
import runlog
import teacher
import timing
import train
from config import Config, config_hash
from experiments.cache import artifact_fingerprint, check_cache, file_sha256, fingerprint_array
from models import build_model

NETS = ("fonr", "maso", "tias")
AO_MODES = ("explicit", "efficient")
C128 = torch.complex128
EXP_KEYS = train.TRAIN_KEYS + ("ao_tol", "ao_max_iter", "n_test", "timing_threads", "timing_warmup",
                               "timing_subset", "eval_device")
RECOVERY_LAYERS = (4, 6, 8)


# ---------------------------------------------------------------- helpers
def _results_dir(cfg: Config) -> str:
    d = os.path.join(cfg.root, "results")
    os.makedirs(d, exist_ok=True)
    return d


def _load(npz):
    return dict(np.load(npz)) if not isinstance(npz, dict) else npz


def _atomic_npz(path: str, **arrs) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    atomic.replace(tmp, path)


def _write_csv(path, cols, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    atomic.replace(tmp, path)


def _ci95(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return 1.96 * x.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(x.shape[1:])


def _timing_idx(cfg: Config) -> np.ndarray:
    n = cfg.n_test if cfg.timing_subset is None else min(int(cfg.timing_subset), cfg.n_test)
    return np.arange(n)


# ---------------------------------------------------------------- models
def load_trained_models(cfg: Config, check_hash: bool = True) -> dict:
    """{name: UnfoldedNet} from ``{cfg.root}/checkpoints/{name}_best.pt``, in eval mode (CPU).

    FONR's mu and MASO's step_scale buffer come from the state_dict, so placeholder
    constructor values are used (no mu probe is run)."""
    out = {}
    for name in NETS:
        p = os.path.join(cfg.root, "checkpoints", f"{name}_best.pt")
        ck = torch.load(p, weights_only=False, map_location="cpu")
        if check_hash and ck.get("cfg_hash") != config_hash(cfg, train.TRAIN_KEYS):
            raise ValueError(f"{p}: checkpoint config hash does not match cfg")
        kw = {"fonr": {"mu_init": 1.0}, "maso": {"step_scale": 1.0}}.get(name, {})
        m = build_model(name, cfg, **kw)
        m.load_state_dict(ck["model_state"])
        out[name] = m.eval()
    return out


def _wsr_per_layer(model, ch, theta0, device, chunk: int = 250) -> np.ndarray:
    """(B, L+1) WSR in bits; batched complex128 under no_grad."""
    net = copy.deepcopy(model).to_precision(C128).to(device).eval()
    rows = []
    with torch.no_grad():
        for lo in range(0, ch.B, chunk):
            sl = slice(lo, min(lo + chunk, ch.B))
            c = ch.slice(sl).to(device, C128)
            p0, phi0 = ao.init_state(c, theta0[sl].to(device))
            tr = net(c, p0, phi0)
            w = torch.stack([fp.wsr(c, tr.p[l], tr.phi[l]) for l in range(tr.p.shape[0])], dim=1)
            rows.append(w.cpu().numpy())
    return np.concatenate(rows, 0)


# ---------------------------------------------------------------- experiment 1/2
def cache_spec(cfg: Config, exp: str = "convergence") -> tuple:
    """(npz path, cfg-hash keys, extra fingerprint files, extra npz keys or None) of exp_01
    ("convergence"); shared by run_convergence and report.py."""
    if exp != "convergence":
        raise KeyError(exp)
    return os.path.join(cfg.root, "results", "exp_01_convergence.npz"), EXP_KEYS, [], None


def is_cached(cfg: Config) -> bool:
    path, keys, files, extra = cache_spec(cfg)
    return check_cache(path, cfg, keys, artifact_fingerprint(cfg, extra=files), extra)


def run_convergence(cfg: Config, force_rerun: bool = False) -> Path:
    """Per-layer WSR (all test samples) and CPU batch-1 timings; cached as exp_01_convergence.npz."""
    _results_dir(cfg)
    path, keys, files, extra = cache_spec(cfg)
    fpr = artifact_fingerprint(cfg, extra=files)
    if not force_rerun and check_cache(path, cfg, keys, fpr, extra):
        return Path(path)

    d = data.build_or_load(cfg, "test")
    ch, theta0 = d["ch"], d["theta0"]
    t = teacher.load_teacher(cfg, "test")
    models = load_trained_models(cfg)
    out = {
        "wsr_ao_traj": t["wsr_traj"].double().numpy(),
        "wsr_ao100": t["wsr_ao"].double().numpy(), "n_iter_ao100": t["n_iter"].numpy(),
        "wsr_ao1000": t["wsr_ref1000"].double().numpy(), "n_iter_ao1000": t["n_iter_ref1000"].numpy(),
    }
    out["ao100_mean"] = np.float64(out["wsr_ao100"].mean())
    out["ao_baseline_max_iter"] = np.int64(cfg.ao_max_iter)
    out["ao1000_mean"] = np.float64(out["wsr_ao1000"].mean())
    log = runlog.get_logger(cfg, "exp01")
    for n, m in runlog.bar(list(models.items()), desc="exp01 WSR per layer", unit="model"):
        out[f"wsr_{n}"] = _wsr_per_layer(m, ch, theta0, cfg.eval_device)
        log.info("%s WSR per layer: %s", n, np.array2string(out[f"wsr_{n}"].mean(0), precision=4,
                                                               max_line_width=10 ** 6))

    idx = _timing_idx(cfg)
    prev = torch.get_num_threads()
    threads = timing.set_timing_threads(cfg.timing_threads)
    try:
        for n, m in models.items():
            net = copy.deepcopy(m).to_precision(C128).to("cpu").eval()
            # the bar advances between samples, never inside a timed step
            out[f"time_{n}"] = np.stack([
                timing.time_steps(net, ch.slice([i]), theta0[[i]], net.L, "cpu", C128,
                                  warmup=cfg.timing_warmup)
                for i in runlog.bar(idx, desc=f"exp01 timing {n}", unit="sample")])
            log.info("%s time/layer %.4f ms (%d samples)", n, 1e3 * out[f"time_{n}"].mean(), len(idx))
        for mode in AO_MODES:
            st = ao.AOStepper(mode)
            runs = [timing.time_until_converged(st, ch.slice([i]), theta0[[i]], cfg.ao_tol,
                                                cfg.ao_max_iter, "cpu", C128)
                    for i in runlog.bar(idx, desc=f"exp01 timing AO {mode}", unit="sample")]
            T = max(len(r[0]) for r in runs)
            cum = np.empty((len(idx), T + 1))
            wt = np.empty((len(idx), T + 1))
            for j, (tt, ww) in enumerate(runs):
                c = np.concatenate([[0.0], np.cumsum(tt)])
                cum[j] = np.pad(c, (0, T + 1 - len(c)), mode="edge")
                wt[j] = np.pad(ww, (0, T + 1 - len(ww)), mode="edge")
            out[f"ao_cumtime_{mode}"] = cum
            out[f"ao_wsr_timed_{mode}"] = wt
            out[f"ao_niter_{mode}"] = np.array([len(r[0]) for r in runs])
            log.info("AO %s conv time %.3f ms", mode, 1e3 * cum[:, -1].mean())
    finally:
        torch.set_num_threads(prev)

    out.update(timing_idx=idx, timing_threads=np.int64(threads), nets=np.array(NETS),
               cfg_json=np.array(json.dumps(cfg.to_dict())), cfg_hash=np.array(config_hash(cfg, EXP_KEYS)),
               artifact_fingerprint=fingerprint_array(fpr))
    _atomic_npz(path, **out)
    _write_convergence_csv(cfg, out)
    with open(os.path.join(_results_dir(cfg), "exp_01_convergence_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summarize_convergence(out), f, indent=2)
    with open(os.path.join(_results_dir(cfg), "exp_01_convergence_config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "seed_test": 3001, "timing_threads": int(threads)}, f, indent=2)
    return Path(path)


def _curves(z: dict) -> dict:
    """{key: (wsr_mean (T+1,), wsr_ci (T+1,), cumtime_mean_s (T+1,), cumtime_median_s (T+1,))}."""
    cur = {}
    for n in NETS:
        w = z[f"wsr_{n}"]
        ct = np.concatenate([np.zeros((z[f"time_{n}"].shape[0], 1)), np.cumsum(z[f"time_{n}"], 1)], 1)
        cur[n] = (w.mean(0), _ci95(w), ct.mean(0), np.median(ct, 0))
    traj = z["wsr_ao_traj"]
    for mode in AO_MODES:
        ct = z[f"ao_cumtime_{mode}"]
        T = min(ct.shape[1], traj.shape[1])
        w = traj[:, :T]
        cur[f"ao_{mode}"] = (w.mean(0), _ci95(w), ct[:, :T].mean(0), np.median(ct[:, :T], 0))
    return cur


def _write_convergence_csv(cfg, z):
    rows = []
    for key, (wm, wc, tm, tmed) in _curves(z).items():
        for s in range(len(wm)):
            rows.append({"method": key, "step": s, "wsr_mean": wm[s], "wsr_ci95": wc[s],
                         "cumtime_mean_ms": 1e3 * tm[s], "cumtime_median_ms": 1e3 * tmed[s]})
    _write_csv(os.path.join(_results_dir(cfg), "exp_01_convergence.csv"),
               ["method", "step", "wsr_mean", "wsr_ci95", "cumtime_mean_ms", "cumtime_median_ms"], rows)


def _gap_stats(gap: np.ndarray, n: str, suffix: str = "") -> dict:
    return {f"gap_p95_{n}{suffix}": float(np.quantile(gap, 0.95)),
            f"gap_min_{n}{suffix}": float(gap.min()),
            f"gap_median_{n}{suffix}": float(np.median(gap)),
            f"gap_mean_{n}{suffix}": float(gap.mean())}


def summarize_convergence(npz) -> dict:
    """Key numbers of experiments 1-3 (key names in the module docstring)."""
    z = _load(npz)
    ao100, ao1000 = float(z["ao100_mean"]), float(z["ao1000_mean"])
    s = {"ao100_mean": ao100, "ao1000_mean": ao1000,
         "ao_baseline_max_iter": float(z["ao_baseline_max_iter"]), "n_test": float(z["wsr_ao100"].shape[0]),
         "n_timing": float(len(z["timing_idx"])), "timing_threads": float(z["timing_threads"])}
    traj_mean = z["wsr_ao_traj"].mean(0)
    for l in RECOVERY_LAYERS:
        for n in NETS:
            w = z[f"wsr_{n}"].mean(0)
            s[f"recovery_pct_layer{l}_{n}"] = float(100 * (w[l] - w[0]) / (ao100 - w[0]))
        if l < len(traj_mean):
            s[f"recovery_pct_layer{l}_ao"] = float(100 * (traj_mean[l] - traj_mean[0]) / (ao100 - traj_mean[0]))
    for key, (wm, _, tm, _) in _curves(z).items():
        for pct in (98, 99):
            hit = np.nonzero(wm >= pct / 100 * ao100)[0]
            s[f"time_to_{pct}pct_{key}_ms"] = float(1e3 * tm[hit[0]]) if hit.size else float("nan")
    for mode in AO_MODES:
        tot = z[f"ao_cumtime_{mode}"][:, -1]
        s[f"ao_conv_time_ms_{mode}"] = float(1e3 * tot.mean())
        s[f"ao_conv_time_median_ms_{mode}"] = float(1e3 * np.median(tot))
        s[f"ao_conv_iters_mean_{mode}"] = float(z[f"ao_niter_{mode}"].mean())
    for n in NETS:
        s[f"layer_time_ms_{n}"] = float(1e3 * z[f"time_{n}"].sum(1).mean())
        wl = z[f"wsr_{n}"][:, -1]
        s.update(_gap_stats(z["wsr_ao100"] - wl, n))
        s.update(_gap_stats(z["wsr_ao1000"] - wl, n, "_vs1000"))
    return s


# ---------------------------------------------------------------- experiment 3
def run_cdf(cfg: Config, force_rerun: bool = False) -> Path:
    """ΔR = R_AO(converged) − R_method(L) per test sample; cached as exp_03_cdf.npz."""
    path = os.path.join(_results_dir(cfg), "exp_03_cdf.npz")
    conv = run_convergence(cfg)
    conv_sha = file_sha256(conv)
    fpr = artifact_fingerprint(cfg)
    if not force_rerun and check_cache(path, cfg, EXP_KEYS, fpr, {"conv_sha256": conv_sha}):
        return Path(path)
    z = _load(conv)
    out = {"wsr_ao100": z["wsr_ao100"], "wsr_ao1000": z["wsr_ao1000"], "nets": np.array(NETS),
           "cfg_json": z["cfg_json"], "cfg_hash": z["cfg_hash"], "conv_sha256": np.array(conv_sha),
           "artifact_fingerprint": fingerprint_array(fpr),
           "ao_baseline_max_iter": z["ao_baseline_max_iter"]}
    rows = [{"sample": i} for i in range(len(z["wsr_ao100"]))]
    for n in NETS:
        wl = z[f"wsr_{n}"][:, -1]
        out[f"wsr_L_{n}"] = wl
        out[f"gap_{n}"] = z["wsr_ao100"] - wl
        out[f"gap_vs1000_{n}"] = z["wsr_ao1000"] - wl
        for i, r in enumerate(rows):
            r[f"gap_{n}"] = out[f"gap_{n}"][i]
            r[f"gap_vs1000_{n}"] = out[f"gap_vs1000_{n}"][i]
    _atomic_npz(path, **out)
    _write_csv(os.path.join(_results_dir(cfg), "exp_03_cdf.csv"), list(rows[0]), rows)
    return Path(path)


# ---------------------------------------------------------------- figures
def _style(label):
    return plotting.STYLE[label]


def plot_fig1(npz):
    """Mean WSR ± 95% CI vs layer / AO iteration 0..L (WSR_vs_iteration)."""
    z = _load(npz)
    fig, ax = plotting.new_fig()
    L = z["wsr_fonr"].shape[1] - 1
    traj = z["wsr_ao_traj"][:, :L + 1]
    series = [("FP-AO", traj)] + [(plotting.NET_LABEL[n], z[f"wsr_{n}"]) for n in NETS]
    for label, w in series:
        x = np.arange(w.shape[1])
        m, c = w.mean(0), _ci95(w)
        st = _style(label)
        ax.plot(x, m, color=st["color"], marker=st["marker"], label=label)
        ax.fill_between(x, m - c, m + c, color=st["color"], alpha=0.2, linewidth=0)
    a100, a1000 = float(z["ao100_mean"]), float(z["ao1000_mean"])
    cap = int(z["ao_baseline_max_iter"]) if "ao_baseline_max_iter" in z else None
    if cap == 1000 or math.isclose(a100, a1000, rel_tol=1e-12, abs_tol=1e-12):
        # the baseline IS the 1000-iteration AO (or the two means coincide): one line, combined label
        ax.axhline(a100, color="black", linestyle=(0, (5, 2)), linewidth=0.9, label="FP-AO (converged, 1000 it.)")
    else:
        ax.axhline(a100, color="black", linestyle=(0, (5, 2)), linewidth=0.9, label="FP-AO (converged)")
        ax.axhline(a1000, color="grey", linestyle=(0, (2, 1.5)), linewidth=0.9, label="FP-AO (1000 it.)")
    ax.set_xticks(np.arange(L + 1))
    ax.set_xlabel("AO iteration / unfolded layer index")
    ax.set_ylabel("WSR (bits/s/Hz)")
    ax.legend(loc="best", ncol=2, fontsize=plotting.FONT_SIZE - 2)
    return fig


def _minor_label(v, _):
    """Label only the 2x and 5x minor ticks of a log axis."""
    if v <= 0:
        return ""
    lead = v / 10 ** np.floor(np.log10(v) + 1e-9)
    return f"{v:g}" if np.isclose(lead, 2) or np.isclose(lead, 5) else ""


def plot_fig2(npz):
    """Mean WSR vs mean cumulative CPU time (ms), log x-axis (WSR_vs_wall_clock)."""
    z = _load(npz)
    fig, ax = plotting.new_fig()
    labels = {"ao_explicit": "FP-AO (explicit)", "ao_efficient": "FP-AO (efficient)",
              **plotting.NET_LABEL}
    for key, (wm, _, tm, _) in _curves(z).items():
        st = _style(labels[key])
        ax.plot(1e3 * tm[1:], wm[1:], color=st["color"], marker=st["marker"], label=labels[key])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(_minor_label))
    ax.set_xlabel("Cumulative wall-clock time (ms)")
    ax.set_ylabel("WSR (bits/s/Hz)")
    ax.legend(loc="best")
    return fig


def plot_fig3(npz):
    """Empirical CDF of ΔR = R_AO(converged) − R_method(L) (CDF)."""
    z = _load(npz)
    fig, ax = plotting.new_fig()
    handles = []
    for n in NETS:
        g = np.sort(z[f"gap_{n}"])
        y = np.arange(1, len(g) + 1) / len(g)
        label = plotting.NET_LABEL[n]
        st = _style(label)
        ax.step(g, y, where="post", color=st["color"])
        k = max(1, len(g) // 10)
        ax.plot(g[::k], y[::k], linestyle="none", color=st["color"], marker=st["marker"])
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"], label=label))
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\Delta R = R_{\mathrm{AO}} - R_{\mathrm{method}}^{(L)}$ (bits/s/Hz)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1)
    ax.legend(handles=handles, loc="lower right")
    return fig


def make_figures(cfg: Config, force_rerun: bool = False) -> dict:
    """Run (or load) experiments 1-3 and save Figs. 1-3 into cfg.paper_dir."""
    import matplotlib.pyplot as plt
    conv = run_convergence(cfg, force_rerun=force_rerun)
    cdf = run_cdf(cfg, force_rerun=force_rerun)
    out = {}
    for name, fn, src in (("WSR_vs_iteration", plot_fig1, conv), ("WSR_vs_wall_clock", plot_fig2, conv),
                          ("CDF", plot_fig3, cdf)):
        fig = fn(src)
        out[name] = plotting.save_fig(fig, name, cfg.paper_dir)
        plt.close(fig)
    return out
