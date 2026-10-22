"""Generate master_pipeline.ipynb (spec section 12). The notebook only orchestrates the modules; it holds no
algorithm code. Run: E:/payam/env/Scripts/python.exe make_notebook.py"""
import nbformat as nbf
from nbformat import v4

cells = []


def md(text):
    cells.append(v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(v4.new_code_cell(text.strip("\n")))


# ------------------------------------------------------------------ title
md("""
# Multi-IRS uplink deep unfolding: master pipeline

Orchestration only: every cell imports the project modules (`config`, `ao`, `teacher`, `train`, `experiments.*`,
`report`) and calls them. Each code cell is preceded by a markdown cell saying what it does and which artifact it
resumes from, and each can be re-run on its own after Cell 1. Artifacts live under `cfg.root`
(`datasets/`, `checkpoints/`, `logs/`, `results/`); figures (PNG 300 dpi + PDF) and `results_findings.md`
go to `cfg.paper_dir`. `main.tex` is never touched.

Full run (hours; resumable, so an interrupted run continues where it stopped):

`E:/payam/env/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 master_pipeline.ipynb`

Smoke run (tiny config, temporary artifact root and paper dir, never touches the real ones): set the environment
variable `IUF_SMOKE=1`.
""")

# ------------------------------------------------------------------ cell 1
md("""
## Cell 1: setup, flags, hardware, config

Imports, seeds, hardware/CUDA report, the `Config`, and the flags `RESUME`, `FORCE_RETRAIN`, `FORCE_RERUN`.
Resumes from nothing (this cell only builds the in-memory config).

**`AO_MAX_ITER`** is the iteration cap of the FP-AO baseline (Table I: 100). Setting it to **1000** makes the
1000-iteration AO both the baseline and the teacher: the teacher labels regenerate automatically because their hash
includes `ao_max_iter`, and the training checkpoints and all experiment results then recompute through their own
config hashes / artifact fingerprints. **`TIMING_SUBSET`** is the number of test samples used for the batch-1
timings (`None` = all 1000).

With `IUF_SMOKE=1` the config shrinks to a tiny problem (M = 16, a few hundred samples, 2-point sweeps, short
epochs, CPU training) with `root` and `paper_dir` in a fresh temporary directory.
""")
code(r'''
import os, sys, time, json, random, re, tempfile, subprocess, dataclasses, warnings
from pathlib import Path

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

import ao, plotting, report, teacher, train
from config import Config
from experiments import convergence, sweeps

# ---- flags
RESUME = True            # continue training from {name}_last.pt if present
FORCE_RETRAIN = False    # True: retrain from scratch, ignoring checkpoints
FORCE_RERUN = False      # True: recompute every experiment even if its cached result is valid
AO_MAX_ITER = 100        # Table I. 1000 => the 1000-iteration AO is the baseline AND the teacher (see markdown)
TIMING_SUBSET = None     # None = all 1000 test samples for batch-1 timings
RUN_TESTS = os.environ.get("IUF_SKIP_TESTS") != "1"
SMOKE = os.environ.get("IUF_SMOKE") == "1"

# ---- seeds (the pipeline modules use their own fixed seed registry; this only fixes notebook-side randomness)
random.seed(0); np.random.seed(0); torch.manual_seed(0)

# ---- config
if SMOKE:
    _tmp = tempfile.mkdtemp(prefix="iuf_smoke_")
    cfg = Config(M=16, maso_M_train=16, K=6, n_train=200, n_val=50, n_test=50, batch_size=50,
                 n_sup_epochs=2, n_unsup_epochs=2, pilot_sup_epochs=1, pilot_unsup_epochs=1, lr_grid=(0.01, 0.001),
                 ao_max_iter=min(AO_MAX_ITER, 30), timing_subset=3, timing_threads=2, timing_warmup=1,
                 sweep_M=(8, 16), sweep_K=(4, 6), sweep_crn_dbm=(20, 30), zeroshot_M=(8, 24),
                 train_device="cpu", eval_device="cpu",
                 root=os.path.join(_tmp, "artifacts"), paper_dir=os.path.join(_tmp, "Paper"))
    assert "payam" not in cfg.root.lower() and "payam" not in cfg.paper_dir.lower()
    print("SMOKE RUN: tiny config; artifacts in", _tmp)
else:
    cfg = Config(ao_max_iter=AO_MAX_ITER, timing_subset=TIMING_SUBSET)

# ---- hardware report
print("python", sys.version.split()[0], sys.executable, "| torch", torch.__version__, "| cpu threads", torch.get_num_threads(),
      "| cpu count", os.cpu_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"CUDA: {p.name}, {p.total_memory / 2**30:.1f} GiB, capability {p.major}.{p.minor}, torch CUDA {torch.version.cuda}")
else:
    print("CUDA: not available (training falls back to CPU; the supplementary GPU timing is skipped)")
print(f"flags: RESUME={RESUME} FORCE_RETRAIN={FORCE_RETRAIN} FORCE_RERUN={FORCE_RERUN} "
      f"AO_MAX_ITER={cfg.ao_max_iter} TIMING_SUBSET={cfg.timing_subset}")
print("root      :", cfg.root)
print("paper_dir :", cfg.paper_dir)
display(pd.Series({k: str(v) for k, v in cfg.to_dict().items()}, name="cfg").to_frame())

# ---- display helpers (presentation only)
def show_png(name):
    png = os.path.join(cfg.paper_dir, f"{name}.png")
    print(png, "|", "pdf ok" if os.path.exists(png.replace(".png", ".pdf")) else "pdf MISSING")
    display(Image(filename=png, width=720))

def show_fig(fig, name):
    """Save PNG (300 dpi) + PDF into cfg.paper_dir, close the figure and show the PNG inline."""
    plotting.save_fig(fig, name, cfg.paper_dir)
    plt.close(fig)
    show_png(name)

def pivot(summary, pattern, row_name="method", col_name="point"):
    """Reshape summary-dict keys matching regex (group 1 = row, group 2 = column) into a DataFrame."""
    rx = re.compile(pattern)
    rec = {}
    for k, v in summary.items():
        m = rx.match(k)
        if m:
            rec.setdefault(m.group(1), {})[int(m.group(2))] = v
    df = pd.DataFrame(rec).T.sort_index(axis=1)
    df.index.name, df.columns.name = row_name, col_name
    return df

def show_keys(summary, prefixes):
    for k in sorted(summary):
        if any(k.startswith(p) for p in prefixes):
            v = summary[k]
            print(f"  {k:48s}", v if isinstance(v, (tuple, str)) else f"{v:.6g}")
''')

# ------------------------------------------------------------------ cell 2
md("""
## Cell 2: unit tests and the Sec. 5.1 sanity check

Runs `pytest -q tests` in a subprocess (junit XML parsed into a per-file pass/fail table), then
`ao.sanity_check(cfg)`: full AO versus the power-only FP baseline on the validation split (pass iff the gain is
at least 10 %). Resumes from the validation dataset in `datasets/` if present (else it is generated);
`logs/sanity_check.json` is rewritten. Set the environment variable `IUF_SKIP_TESTS=1` to skip the pytest part.
""")
code(r'''
import xml.etree.ElementTree as ET

if RUN_TESTS:
    _xml = os.path.join(tempfile.mkdtemp(prefix="iuf_junit_"), "junit.xml")
    _t0 = time.time()
    _proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={_xml}"],
                           capture_output=True, text=True)
    rows = {}
    for tc in ET.parse(_xml).getroot().iter("testcase"):
        cn = tc.get("classname", "")
        f = cn.split(".")[-1] if cn else "?"
        r = rows.setdefault(f, {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "time_s": 0.0})
        status = ("failed" if tc.find("failure") is not None else "error" if tc.find("error") is not None
                  else "skipped" if tc.find("skipped") is not None else "passed")
        r[status] += 1
        r["time_s"] += float(tc.get("time", 0))
    tests_df = pd.DataFrame(rows).T
    tests_df.loc["TOTAL"] = tests_df.sum()
    tests_df["status"] = np.where((tests_df["failed"] + tests_df["error"]) == 0, "PASS", "FAIL")
    display(tests_df)
    print(f"pytest return code {_proc.returncode} in {time.time() - _t0:.0f} s")
    if _proc.returncode != 0:
        print("WARNING: some tests failed; tail of the pytest output follows")
        print(_proc.stdout[-3000:])
else:
    print("pytest skipped (IUF_SKIP_TESTS=1)")

sanity = ao.sanity_check(cfg)
display(pd.Series(sanity, name="sanity_check").astype(str).to_frame())
print("Sec. 5.1 sanity check:", "PASSED" if sanity["passed"] else "FAILED", f"(gain {sanity['gain_pct']:.1f} %)")
''')

# ------------------------------------------------------------------ cell 3
md("""
## Cell 3: datasets and teacher labels

`teacher.build_all_datasets(cfg)` generates (or loads) the train/val/test sets and every sweep-point test set
(fixed seeds); `teacher.build_teacher_labels(cfg)` runs the FP-AO teacher (baseline cap `cfg.ao_max_iter`, plus a
1000-iteration reference on the test split). Resumes from `datasets/*.pt` and `datasets/teacher_labels.pt`
(reused when the file exists and its config hash matches; changing `AO_MAX_ITER` regenerates the labels).
""")
code(r'''
t0 = time.time()
ds_paths = teacher.build_all_datasets(cfg)
teacher_path = teacher.build_teacher_labels(cfg)
print(f"{len(ds_paths)} datasets ready; teacher labels: {teacher_path} ({time.time() - t0:.0f} s)")
rows = []
for s in ("train", "val", "test"):
    t = teacher.load_teacher(cfg, s)
    rows.append({"split": s, "n": len(t["wsr_ao"]), "mean_WSR_AO": float(t["wsr_ao"].mean()),
                 "std_WSR_AO": float(t["wsr_ao"].std()), "mean_AO_iters": float(t["n_iter"].double().mean()),
                 "pct_hit_cap": 100 * float((t["n_iter"] >= cfg.ao_max_iter).double().mean())})
display(pd.DataFrame(rows).set_index("split"))
''')

# ------------------------------------------------------------------ cells 4-6
MODEL_CODE = r'''
NAME = "__NAME__"
t0 = time.time()
lr = train.run_lr_pilot(NAME, cfg)
best_path = train.train_model(NAME, cfg, resume=RESUME, force_retrain=FORCE_RETRAIN)
fm = train.final_metrics(NAME, cfg)
print(f"[{NAME}] selected lr {lr:g}, best epoch {fm['best_epoch']}, checkpoint {best_path}, {time.time() - t0:.0f} s")

logs = os.path.join(cfg.root, "logs")
pilot = pd.read_csv(os.path.join(logs, f"{NAME}_lr_pilot.csv"))
print("LR pilot: final-epoch mean validation WSR per learning rate")
display(pilot.sort_values("epoch").groupby("lr").tail(1).set_index("lr")[["epoch", "val_wsr_mean"]])

hist = pd.read_csv(os.path.join(logs, f"{NAME}_training_history.csv"))
display(hist.tail(8))
ao_val = float(teacher.load_teacher(cfg, "val")["wsr_ao"].mean())
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(hist["epoch"], hist["train_loss"], label="train loss")
ax[0].plot(hist["epoch"], hist["val_loss"], label="val loss")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title(f"{NAME.upper()}: loss")
ax[1].plot(hist["epoch"], hist["train_wsr_mean"], label="train WSR")
ax[1].plot(hist["epoch"], hist["val_wsr_mean"], label="val WSR")
ax[1].axhline(ao_val, color="k", ls="--", lw=0.8, label=f"FP-AO val mean ({ao_val:.3f})")
for e in hist["epoch"][hist["stage"].ne(hist["stage"].shift()) & (hist.index > 0)]:
    ax[1].axvline(e, color="grey", ls=":", lw=0.8)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("WSR (bits/s/Hz)"); ax[1].legend()
ax[1].set_title(f"{NAME.upper()}: WSR (dotted = stage switch)")
plt.tight_layout(); plt.show()

metrics = pd.DataFrame({s: {"loss": fm[f"{s}_loss"], "WSR mean": fm[f"{s}_wsr_mean"], "WSR std": fm[f"{s}_wsr_std"],
                            "FP-AO WSR mean": fm[f"{s}_ao_wsr_mean"], "efficiency %": fm[f"{s}_efficiency_pct"]}
                        for s in ("train", "val", "test")}).T
display(metrics)
print("test gap to AO (bits/s/Hz): median {:.4g}, 95th pctl {:.4g}, min (left tail) {:.4g}".format(
    fm["test_gap_median"], fm["test_gap_95th_pctl"], fm["test_gap_min_left_tail"]))
'''


def model_cell(num, name, title):
    md(f"""
## Cell {num}: {title}

`train.run_lr_pilot` (learning-rate pilot over `cfg.lr_grid`), `train.train_model` (two-stage training, resumable),
`train.final_metrics` (train/val/test evaluation of `best.pt` in complex128), then the training history table and
inline train/validation curves. Resumes from `logs/{name}_lr_pilot.csv`, `checkpoints/{name}_last.pt` /
`{name}_best.pt` (continues at the next epoch; a finished run is not retrained) and
`logs/{name}_training_history.csv`. `FORCE_RETRAIN = True` discards them.
""")
    code(MODEL_CODE.replace("__NAME__", name))


model_cell(4, "fonr", "FONR")
model_cell(5, "maso", "MASO")
model_cell(6, "tias", "TIAS")

# ------------------------------------------------------------------ cell 7
md("""
## Cell 7: master train/val/test comparison

Reads `logs/all_models_final_metrics.csv` (written by `final_metrics` in Cells 4-6) and the teacher means from
`datasets/teacher_labels.pt`. Rows: FP-AO at the baseline cap (train/val/test means from the teacher labels), the
three models, and the AO-1000 reference on the test split. Saves `logs/master_comparison.csv`.
""")
code(r'''
fm_all = pd.read_csv(os.path.join(cfg.root, "logs", "all_models_final_metrics.csv")).set_index("model_name")
ao_label = f"FP-AO ({cfg.ao_max_iter} iter)"
T = {s: teacher.load_teacher(cfg, s) for s in ("train", "val", "test")}
rows = {ao_label: {"train WSR": float(T["train"]["wsr_ao"].mean()), "val WSR": float(T["val"]["wsr_ao"].mean()),
                   "test WSR": float(T["test"]["wsr_ao"].mean()), "test efficiency %": 100.0,
                   "test gap median": 0.0, "test gap p95": 0.0}}
for n in ("fonr", "maso", "tias"):
    r = fm_all.loc[n]
    rows[n.upper()] = {"train WSR": r["train_wsr_mean"], "val WSR": r["val_wsr_mean"], "test WSR": r["test_wsr_mean"],
                       "test efficiency %": r["test_efficiency_pct"], "test gap median": r["test_gap_median"],
                       "test gap p95": r["test_gap_95th_pctl"]}
ao1000 = float(T["test"]["wsr_ref1000"].mean())
rows["FP-AO (1000 iter), test reference"] = {"test WSR": ao1000,
                                             "test efficiency %": 100.0 * float(T["test"]["wsr_ao"].mean()) / ao1000}
master = pd.DataFrame(rows).T
master.to_csv(os.path.join(cfg.root, "logs", "master_comparison.csv"))
display(master)
print("model efficiency is relative to the baseline-cap teacher; the last row's efficiency is the baseline-cap AO "
      "relative to AO-1000.")
''')


# ------------------------------------------------------------------ cells 8-10 (convergence figures)
def conv_fig_cell(num, fig_no, fname, plot_fn, intro, extra):
    md(f"""
## Cell {num}: Fig. {fig_no} ({fname})

{intro} Saves `{fname}.png` (300 dpi) and `{fname}.pdf` into `cfg.paper_dir`, shows the PNG, and prints the key numbers.
Resumes from `results/exp_01_convergence.npz`{extra}; recomputed only if the file is missing/stale (config hash and
artifact fingerprint) or `FORCE_RERUN = True`.
""")
    return plot_fn


conv_fig_cell(8, 1, "WSR_vs_iteration", None,
              "Mean WSR (95 % CI) versus layer / AO iteration for FONR, MASO, TIAS and FP-AO.", "")
code(r'''
conv_npz = convergence.run_convergence(cfg, force_rerun=FORCE_RERUN)
show_fig(convergence.plot_fig1(conv_npz), "WSR_vs_iteration")
conv_sum = convergence.summarize_convergence(conv_npz)
print(f"AO baseline mean {conv_sum['ao100_mean']:.4f}, AO-1000 mean {conv_sum['ao1000_mean']:.4f}")
show_keys(conv_sum, ["recovery_pct_"])
''')

conv_fig_cell(9, 2, "WSR_vs_wall_clock", None,
              "Mean WSR versus cumulative CPU wall-clock time (batch 1, complex128, `cfg.timing_threads` threads).", "")
code(r'''
conv_npz = convergence.run_convergence(cfg, force_rerun=FORCE_RERUN)
show_fig(convergence.plot_fig2(conv_npz), "WSR_vs_wall_clock")
conv_sum = convergence.summarize_convergence(conv_npz)
show_keys(conv_sum, ["time_to_", "ao_conv_time_ms_", "ao_conv_iters_mean_", "layer_time_ms_"])
''')

md("""
## Cell 10: Fig. 3 (CDF)

Empirical CDF of the per-sample WSR gap to the converged AO on the test split, derived by `convergence.run_cdf`
from the convergence result. Saves `CDF.png` / `CDF.pdf` into `cfg.paper_dir`, shows the PNG, prints gap statistics.
Resumes from `results/exp_03_cdf.npz` (rebuilt from `exp_01_convergence.npz` if missing or stale) or
`FORCE_RERUN = True`.
""")
code(r'''
cdf_npz = convergence.run_cdf(cfg, force_rerun=FORCE_RERUN)
show_fig(convergence.plot_fig3(cdf_npz), "CDF")
conv_sum = convergence.summarize_convergence(convergence.run_convergence(cfg))
show_keys(conv_sum, ["gap_"])
''')


# ------------------------------------------------------------------ cells 11-15 (sweeps)
def sweep_cell(num, fig_no, fname, call, what, resume_txt, summary_code):
    md(f"""
## Cell {num}: Fig. {fig_no} ({fname})

{what} Saves `{fname}.png` / `{fname}.pdf` into `cfg.paper_dir`, shows the PNG, prints the key numbers.
Resumes from `results/{resume_txt}`; recomputed only if the file is missing/stale (config hash and artifact
fingerprint) or `FORCE_RERUN = True`.
""")
    code(f'''
npz_path, summ = {call}
show_png("{fname}")
print("result file:", npz_path)
{summary_code}
''')


sweep_cell(11, 4, "avg_time_vs_Number_of_RIS_M", "sweeps.run_runtime_vs_M(cfg, force_rerun=FORCE_RERUN)",
           "Average CPU time per iteration (AO) / per layer (nets) versus the number of IRS elements M "
           "(batch 1, complex128; MASO instantiated untrained at each M).",
           "exp_04_runtime_vs_M.npz",
           'display(pivot(summ, r"^runtime_ms_(.+)_M(\\d+)$", "method (mean ms)", "M"))')
sweep_cell(12, 5, "etha_vs_CRN_transmit_power", "sweeps.run_power_sweep(cfg, force_rerun=FORCE_RERUN)",
           "Efficiency eta = WSR(model) / WSR(FP-AO) versus CRN peak power (PMN = CRN - 10 dB), models trained at 20 dBm.",
           "exp_05_power_sweep.npz",
           'display(pivot(summ, r"^eta_ratio_of_means_(.+)_P(\\d+)$", "eta % (ratio of means)", "CRN dBm"))\n'
           'print("eta range per model (min, max):", {n: summ[f"eta_range_{n}"] for n in ("fonr", "maso", "tias")})')
sweep_cell(13, 6, "WSR_vs_Number_of_active_users_K", "sweeps.run_wsr_vs_K(cfg, force_rerun=FORCE_RERUN)",
           "Mean WSR versus the number of active users K for FP-AO and the three trained models (no retraining).",
           "exp_06_wsr_vs_K.npz",
           'display(pivot(summ, r"^wsr_(.+)_K(\\d+)$", "mean WSR", "K"))\n'
           'display(pivot(summ, r"^eta_(.+)_K(\\d+)$", "eta % vs FP-AO", "K"))')
sweep_cell(14, 7, "avg_time_vs_Number_of_active_users_K", "sweeps.run_runtime_vs_K(cfg, force_rerun=FORCE_RERUN)",
           "Average CPU time per iteration / layer versus K (M = cfg.M, batch 1, complex128).",
           "exp_07_runtime_vs_K.npz",
           'display(pivot(summ, r"^runtime_ms_(.+)_K(\\d+)$", "method (mean ms)", "K"))')
sweep_cell(15, 8, "WSR_vs_M_ZeroShot", "sweeps.run_zeroshot_M(cfg, force_rerun=FORCE_RERUN)",
           "Zero-shot mean WSR versus M with the models trained at M = cfg.M (MASO zero-pads / truncates).",
           "exp_08_zeroshot_M.npz",
           'display(pivot(summ, r"^wsr_(.+)_M(\\d+)_zeroshot$", "mean WSR", "M"))\n'
           'display(pivot(summ, r"^eta_(.+)_M(\\d+)_zeroshot$", "eta % vs FP-AO", "M"))\n'
           'print("training M (reference):", summ["M_reference"])')

# ------------------------------------------------------------------ cell 16
md("""
## Cell 16: supplementary GPU timing and the findings report

`sweeps.run_gpu_timing(cfg)` times every method on CUDA in complex64 (skipped with a message when no GPU is
available), then `report.build_report(cfg)` rebuilds all summaries from the cached result files and writes
`results_findings.md` (claims vs. measured, deviations, metrics, latency, LR pilot, eta, suggested text changes)
into `cfg.paper_dir`. Resumes from `results/exp_supp_gpu_timing.npz` and every other `results/*.npz`.
`main.tex` is not modified.
""")
code(r'''
gpu_path, gpu_summ = sweeps.run_gpu_timing(cfg, force_rerun=FORCE_RERUN)
print("GPU timing:", gpu_path if gpu_path else "skipped (no CUDA)")
if gpu_path:
    show_keys(gpu_summ, ["step_time_ms_", "ao_conv_time_ms_"])

report_path = report.build_report(cfg)
print("report written to:", report_path)
text = Path(report_path).read_text(encoding="utf-8")
head = text.split("## 1. Claims vs. measured", 1)[1].split("###", 1)[0].strip().splitlines()
print(f"report: {len(text.splitlines())} lines; head of the claims section follows (first 14 lines)")
display(Markdown("\n".join(head[:14])))
''')

nb = v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, "master_pipeline.ipynb")
print("wrote master_pipeline.ipynb:", sum(c.cell_type == "code" for c in cells), "code cells,",
      sum(c.cell_type == "markdown" for c in cells), "markdown cells")
