"""Orchestration and display helpers for ``master_pipeline.ipynb`` (spec §12).

The notebook cells only call these functions and display what they return; every function is also run
end to end on a tiny configuration by ``tests/test_pipeline_e2e.py``. No algorithm code lives here: the
work is done by ``teacher``, ``train``, ``experiments.*`` and ``report``.

Conventions: functions that produce a paper figure return ``(png_path, table)`` where ``table`` is a
pandas DataFrame of the figure's key numbers; tables are what the notebook displays.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ao
import plotting
import report
import runlog
import teacher
import train
from config import Config
from experiments import convergence, sweeps

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/ (src/ lives inside it)
NETS = ("fonr", "maso", "tias")
NET = {"fonr": "FONR", "maso": "MASO", "tias": "TIAS"}
TABLE_I_AO_MAX_ITER = 100


# ------------------------------------------------------------------ configuration
def artifact_root(ao_max_iter: int, code_dir: str = CODE_DIR) -> str:
    """``{code}/artifacts`` for the Table-I cap (100), ``{code}/artifacts_ao{N}`` otherwise, so runs with
    different AO caps never overwrite each other."""
    name = "artifacts" if int(ao_max_iter) == TABLE_I_AO_MAX_ITER else f"artifacts_ao{int(ao_max_iter)}"
    return os.path.join(code_dir, name).replace("\\", "/")


def make_config(ao_max_iter: int = TABLE_I_AO_MAX_ITER, timing_subset=200, **overrides) -> Config:
    """The pipeline Config: defaults (spec), the AO cap, the timing subset, the matching artifact root."""
    kw = dict(ao_max_iter=int(ao_max_iter), timing_subset=timing_subset, root=artifact_root(ao_max_iter))
    kw.update(overrides)
    return Config(**kw)


def environment_table(cfg: Config) -> pd.DataFrame:
    """Python / torch / CUDA / threads and the paths used by this run."""
    rows = {"python": f"{platform.python_version()} ({sys.executable})", "torch": torch.__version__,
            "torch threads": torch.get_num_threads(), "CPU count": os.cpu_count(),
            "timing threads (cfg)": cfg.timing_threads}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rows["CUDA device"] = f"{p.name}, {p.total_memory / 2 ** 30:.1f} GiB, capability {p.major}.{p.minor}"
        rows["torch CUDA"] = torch.version.cuda
    else:
        rows["CUDA device"] = "not available (training falls back to CPU; GPU supplement skipped)"
    rows.update({"artifact root": cfg.root, "paper dir": cfg.paper_dir, "pipeline log": runlog.log_path(cfg)})
    return pd.DataFrame({"value": {k: str(v) for k, v in rows.items()}})


def config_table(cfg: Config) -> pd.DataFrame:
    return pd.DataFrame({"value": {k: str(v) for k, v in cfg.to_dict().items()}})


# ------------------------------------------------------------------ verification
def run_unit_tests(code_dir: str = CODE_DIR, extra_args=()) -> tuple:
    """Run ``pytest -q tests`` in a subprocess -> (per-file DataFrame with a TOTAL row, ok, output tail)."""
    xml = os.path.join(tempfile.mkdtemp(prefix="iuf_junit_"), "junit.xml")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={xml}", *extra_args],
                          capture_output=True, text=True, cwd=code_dir)
    rows = {}
    if os.path.exists(xml):
        for tc in ET.parse(xml).getroot().iter("testcase"):
            f = (tc.get("classname", "") or "?").split(".")[-1]
            r = rows.setdefault(f, {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "time_s": 0.0})
            st = ("failed" if tc.find("failure") is not None else "error" if tc.find("error") is not None
                  else "skipped" if tc.find("skipped") is not None else "passed")
            r[st] += 1
            r["time_s"] += float(tc.get("time", 0))
    df = pd.DataFrame(rows).T if rows else pd.DataFrame(columns=["passed", "failed", "error", "skipped", "time_s"])
    df.loc["TOTAL"] = df.sum()
    df["time_s"] = df["time_s"].astype(float).round(1)
    df["status"] = np.where((df["failed"] + df["error"]) == 0, "PASS", "FAIL")
    df.attrs["wall_s"] = time.time() - t0
    ok = proc.returncode == 0
    return df, ok, (proc.stdout + proc.stderr)[-3000:]


def sanity_table(cfg: Config) -> tuple:
    """Spec §5.1 sanity check (full AO vs power-only FP, val split) -> (DataFrame, passed)."""
    s = ao.sanity_check(cfg)
    keep = ("ao_mean", "power_only_mean", "gain_pct", "init_mean", "ao_mean_n_iter", "power_only_mean_n_iter",
            "n_samples", "passed")
    return pd.DataFrame({"value": {k: s[k] for k in keep if k in s}}), bool(s["passed"])


def require(ok_tests: bool, ok_sanity: bool) -> None:
    """Stop the run (RuntimeError) if the unit tests or the sanity check failed."""
    problems = [m for ok, m in ((ok_tests, "unit tests failed"), (ok_sanity, "Sec. 5.1 sanity check failed"))
                if not ok]
    if problems:
        raise RuntimeError("; ".join(problems) + ": fix this before starting hours of compute")


# ------------------------------------------------------------------ data and teacher labels
def build_data(cfg: Config) -> pd.DataFrame:
    """All datasets and the FP-AO teacher labels (cached) -> per-split summary table."""
    teacher.build_all_datasets(cfg)
    teacher.build_teacher_labels(cfg)
    rows = {}
    for s in ("train", "val", "test"):
        t = teacher.load_teacher(cfg, s)
        n_it = t["n_iter"].double()
        rows[s] = {"samples": len(t["wsr_ao"]), "FP-AO WSR mean": float(t["wsr_ao"].mean()),
                   "FP-AO WSR std": float(t["wsr_ao"].std()), "AO iterations mean": float(n_it.mean()),
                   "% at cap": 100 * float((n_it >= cfg.ao_max_iter).double().mean())}
        if "wsr_ref1000" in t:
            rows[s]["FP-AO (1000 it.) WSR mean"] = float(t["wsr_ref1000"].mean())
    df = pd.DataFrame(rows).T
    df.index.name = "split"
    return df


# ------------------------------------------------------------------ models
def lr_pilot(cfg: Config, name: str) -> tuple:
    """LR pilot (resumable) -> (selected lr, table of the final-epoch mean val WSR per lr)."""
    lr = train.run_lr_pilot(name, cfg)
    pilot = pd.read_csv(os.path.join(cfg.root, "logs", f"{name}_lr_pilot.csv"))
    last = pilot.sort_values("epoch").groupby("lr").tail(1).set_index("lr")[["epoch", "val_wsr_mean"]]
    last["selected"] = ["yes" if np.isclose(v, lr) else "" for v in last.index]
    return lr, last.rename(columns={"epoch": "last epoch", "val_wsr_mean": "val WSR (bits/s/Hz)"})


def train_net(cfg: Config, name: str, resume: bool = True, force_retrain: bool = False) -> Path:
    """Two-stage training (resumable; classic tqdm bar over epochs, one log line per epoch)."""
    return train.train_model(name, cfg, resume=resume, force_retrain=force_retrain)


def history(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(cfg.root, "logs", f"{name}_training_history.csv"))


def history_figure(cfg: Config, name: str) -> str:
    """Training curves (loss, Stage-II loss components, WSR) saved to ``{root}/logs/{name}_training_history.png``.
    The Stage-II loss is negative by construction (it minimises −objective in nats)."""
    import matplotlib.pyplot as plt
    h = history(cfg, name)
    ao_val = float(teacher.load_teacher(cfg, "val")["wsr_ao"].mean())
    switch = h["epoch"][h["stage"].ne(h["stage"].shift()) & (h.index > 0)].tolist()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train")
    ax[0].plot(h["epoch"], h["val_loss"], label="val (Stage-II loss)")
    ax[0].set_title(f"{NET[name]}: loss (Stage II < 0 by construction)")
    un = h[h["stage"] == "unsupervised"]
    ax[1].plot(un["epoch"], un["train_wsr_nats_term"], label=r"$\sum_l \rho_l R_l$ (ln-WSR, nats)")
    ax[1].plot(un["epoch"], un["train_fq_term"], label=r"$\sum_l \rho_l f_{q,l}$ (bound, nats)")
    ax[1].set_title("Stage-II loss components (train)")
    ax[2].plot(h["epoch"], h["train_wsr_mean"], label="train WSR")
    ax[2].plot(h["epoch"], h["val_wsr_mean"], label="val WSR")
    ax[2].axhline(ao_val, color="k", ls="--", lw=0.8, label=f"FP-AO val ({ao_val:.3f})")
    ax[2].set_title("WSR (bits/s/Hz)")
    for a in ax:
        for e in switch:
            a.axvline(e, color="grey", ls=":", lw=0.8)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(cfg.root, "logs", f"{name}_training_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def final_metrics(cfg: Config, name: str) -> tuple:
    """best.pt on train/val/test in complex128 -> (metrics table, gap table)."""
    fm = train.final_metrics(name, cfg)
    m = pd.DataFrame({s: {"loss": fm[f"{s}_loss"], "WSR mean": fm[f"{s}_wsr_mean"], "WSR std": fm[f"{s}_wsr_std"],
                          "FP-AO WSR mean": fm[f"{s}_ao_wsr_mean"], "efficiency %": fm[f"{s}_efficiency_pct"]}
                      for s in ("train", "val", "test")}).T
    m.index.name = f"{NET[name]} (lr {fm['selected_lr']:g}, best epoch {fm['best_epoch']})"
    gap = pd.DataFrame({"test gap to FP-AO (bits/s/Hz)": {"median": fm["test_gap_median"],
                                                         "95th percentile": fm["test_gap_95th_pctl"],
                                                         "min (left tail)": fm["test_gap_min_left_tail"]}})
    return m, gap


def comparison_table(cfg: Config) -> pd.DataFrame:
    """Train/val/test WSR of the three models next to FP-AO (baseline cap) and the AO-1000 reference;
    also written to ``{root}/logs/master_comparison.csv``."""
    fm = pd.read_csv(os.path.join(cfg.root, "logs", "all_models_final_metrics.csv")).set_index("model_name")
    T = {s: teacher.load_teacher(cfg, s) for s in ("train", "val", "test")}
    rows = {f"FP-AO ({cfg.ao_max_iter} it.)": {
        "train WSR": float(T["train"]["wsr_ao"].mean()), "val WSR": float(T["val"]["wsr_ao"].mean()),
        "test WSR": float(T["test"]["wsr_ao"].mean()), "test efficiency %": 100.0}}
    for n in NETS:
        if n in fm.index:
            r = fm.loc[n]
            rows[NET[n]] = {"train WSR": r["train_wsr_mean"], "val WSR": r["val_wsr_mean"],
                            "test WSR": r["test_wsr_mean"], "test efficiency %": r["test_efficiency_pct"],
                            "test gap median": r["test_gap_median"], "test gap p95": r["test_gap_95th_pctl"],
                            "selected lr": r["selected_lr"], "best epoch": r["best_epoch"]}
    if "wsr_ref1000" in T["test"]:
        a1000 = float(T["test"]["wsr_ref1000"].mean())
        rows["FP-AO (1000 it.), test reference"] = {
            "test WSR": a1000, "test efficiency %": 100.0 * float(T["test"]["wsr_ao"].mean()) / a1000}
    df = pd.DataFrame(rows).T
    df.to_csv(os.path.join(cfg.root, "logs", "master_comparison.csv"))
    return df


# ------------------------------------------------------------------ figures
def _png(cfg: Config, name: str) -> str:
    return os.path.join(cfg.paper_dir, f"{name}.png")


def _save(cfg: Config, fig, name: str) -> str:
    import matplotlib.pyplot as plt
    png, _ = plotting.save_fig(fig, name, cfg.paper_dir)
    plt.close(fig)
    return png


def pivot(summary: dict, pattern: str, row_name: str = "method", col_name: str = "point") -> pd.DataFrame:
    """Summary keys matching ``pattern`` (group 1 = row, group 2 = integer column) -> DataFrame."""
    rx = re.compile(pattern)
    rec = {}
    for k, v in summary.items():
        m = rx.match(k)
        if m:
            rec.setdefault(m.group(1), {})[int(m.group(2))] = v
    df = pd.DataFrame(rec).T.sort_index(axis=1)
    df.index.name, df.columns.name = row_name, col_name
    return df


def _keys(summary: dict, prefixes, label: str = "value") -> pd.DataFrame:
    return pd.DataFrame({label: {k: summary[k] for k in sorted(summary) if k.startswith(tuple(prefixes))}})


def figure_1(cfg: Config, force_rerun: bool = False) -> tuple:
    """Fig. 1 WSR_vs_iteration; computes exp_01 (shared with Figs. 2-3)."""
    npz = convergence.run_convergence(cfg, force_rerun=force_rerun)
    s = convergence.summarize_convergence(npz)
    z = np.load(npz)
    curves = {("FP-AO" if k == "ao" else NET[k]):
              pd.Series((z["wsr_ao_traj"] if k == "ao" else z[f"wsr_{k}"])[:, :cfg.L + 1].mean(0))
              for k in ("ao",) + NETS}
    tab = pd.DataFrame(curves).T
    tab.columns.name = "layer / iteration"
    tab["recovery % @4"] = [s.get(f"recovery_pct_layer4_{k}") for k in ("ao",) + NETS]
    tab.attrs["ao"] = (s["ao100_mean"], s["ao1000_mean"])
    return _save(cfg, convergence.plot_fig1(npz), "WSR_vs_iteration"), tab


def figure_2(cfg: Config) -> tuple:
    """Fig. 2 WSR_vs_wall_clock from the cached exp_01 (never recomputed here)."""
    npz = convergence.run_convergence(cfg)
    s = convergence.summarize_convergence(npz)
    meths = ("fonr", "maso", "tias", "ao_explicit", "ao_efficient")
    tab = pd.DataFrame({m: {"time to 98 % (ms)": s.get(f"time_to_98pct_{m}_ms"),
                            "time to 99 % (ms)": s.get(f"time_to_99pct_{m}_ms")} for m in meths}).T
    tab["L-layer / converged time (ms)"] = [s[f"layer_time_ms_{m}"] for m in NETS] + \
                                           [s[f"ao_conv_time_ms_{m[3:]}"] for m in meths[3:]]
    return _save(cfg, convergence.plot_fig2(npz), "WSR_vs_wall_clock"), tab


def figure_3(cfg: Config, force_rerun: bool = False) -> tuple:
    """Fig. 3 CDF of the gap to FP-AO (rebuilt from exp_01 when that changes)."""
    npz = convergence.run_cdf(cfg, force_rerun=force_rerun)
    s = convergence.summarize_convergence(convergence.run_convergence(cfg))
    tab = pd.DataFrame({NET[n]: {"median": s[f"gap_median_{n}"], "mean": s[f"gap_mean_{n}"],
                                 "95th pct": s[f"gap_p95_{n}"], "min": s[f"gap_min_{n}"],
                                 "95th pct vs AO-1000": s[f"gap_p95_{n}_vs1000"]} for n in NETS}).T
    return _save(cfg, convergence.plot_fig3(npz), "CDF"), tab


def figure_4(cfg: Config, force_rerun: bool = False) -> tuple:
    _, s = sweeps.run_runtime_vs_M(cfg, force_rerun=force_rerun)
    return _png(cfg, "avg_time_vs_Number_of_RIS_M"), pivot(s, r"^runtime_ms_(.+)_M(\d+)$", "mean ms / iter", "M")


def figure_5(cfg: Config, force_rerun: bool = False) -> tuple:
    _, s = sweeps.run_power_sweep(cfg, force_rerun=force_rerun)
    tab = pivot(s, r"^eta_ratio_of_means_(.+)_P(\d+)$", "eta % (ratio of means)", "CRN dBm")
    return _png(cfg, "etha_vs_CRN_transmit_power"), tab


def figure_6(cfg: Config, force_rerun: bool = False) -> tuple:
    _, s = sweeps.run_wsr_vs_K(cfg, force_rerun=force_rerun)
    w = pivot(s, r"^wsr_(.+)_K(\d+)$", "mean WSR", "K")
    e = pivot(s, r"^eta_(.+)_K(\d+)$", "eta %", "K")
    e.index = [f"{i} eta %" for i in e.index]
    return _png(cfg, "WSR_vs_Number_of_active_users_K"), pd.concat([w, e])


def figure_7(cfg: Config, force_rerun: bool = False) -> tuple:
    _, s = sweeps.run_runtime_vs_K(cfg, force_rerun=force_rerun)
    return _png(cfg, "avg_time_vs_Number_of_active_users_K"), pivot(s, r"^runtime_ms_(.+)_K(\d+)$",
                                                                     "mean ms / iter", "K")


def figure_8(cfg: Config, force_rerun: bool = False) -> tuple:
    _, s = sweeps.run_zeroshot_M(cfg, force_rerun=force_rerun)
    w = pivot(s, r"^wsr_(.+)_M(\d+)_zeroshot$", "mean WSR", "M")
    e = pivot(s, r"^eta_(.+)_M(\d+)_zeroshot$", "eta %", "M")
    e.index = [f"{i} eta %" for i in e.index]
    return _png(cfg, "WSR_vs_M_ZeroShot"), pd.concat([w, e])


# ------------------------------------------------------------------ GPU supplement and report
def gpu_supplement(cfg: Config, force_rerun: bool = False) -> pd.DataFrame:
    """Supplementary GPU timing (complex64; skipped without CUDA) -> default-setting table. FP-AO run to
    convergence on GPU is 'not available' whenever samples failed (Task 16 E)."""
    path, s = sweeps.run_gpu_timing(cfg, force_rerun=force_rerun)
    if path is None:
        return pd.DataFrame({"GPU timing": {"status": "skipped (no CUDA device)"}})
    na = report._gpu_conv_unavailable(s)
    rows = {f"{NET[n]} per layer (ms)": s[f"step_time_ms_{n}"] for n in NETS}
    for mode in sweeps.AO_MODES:
        ok = s.get(f"ao_conv_available_{mode}", 0.0) == 1.0
        pct = s[f"ao_conv_pct_failed_{mode}"]
        rows[f"FP-AO ({mode}) to convergence (ms)"] = f"{s[f'ao_conv_time_ms_{mode}']:.4g}" if ok else na
        rows[f"FP-AO ({mode}) run-to-convergence failures %"] = pct
        k = f"runtime_pct_failed_ao_{mode}_M{cfg.M}"
        if k in s:
            rows[f"FP-AO ({mode}) per iteration at M={cfg.M} sweep point (ms)"] = (
                f"{s[f'runtime_ms_ao_{mode}_M{cfg.M}']:.4g} [{s[k]:.3g} % failed]")
    return pd.DataFrame({"value": {k: str(v) for k, v in rows.items()}})


def findings_report(cfg: Config, n_lines: int = 16) -> tuple:
    """Write ``results_findings.md`` -> (path, head of the claims table as markdown)."""
    path = report.build_report(cfg)
    text = Path(path).read_text(encoding="utf-8")
    head = text.split("## 1. Claims vs. measured", 1)[-1].split("###", 1)[0].strip().splitlines()
    return path, "\n".join(head[:n_lines])


def show(png: str, width: int = 640):
    """Display a saved PNG inline (notebook only)."""
    from IPython.display import Image, display
    display(Image(filename=png, width=width))
