"""Generate ``master_pipeline.ipynb`` (spec §12), unexecuted.

The notebook only orchestrates: every code cell calls ``nb_helpers`` (which calls teacher / train /
experiments / report) and displays the returned tables and figures. The end-to-end check of the same calls
on a tiny configuration lives in ``tests/test_pipeline_e2e.py``.

Run (from E:/payam/code): E:/payam/env/Scripts/python.exe scripts/make_notebook.py
"""
import nbformat as nbf
from nbformat import v4

cells = []


def md(text):
    cells.append(v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(v4.new_code_cell(text.strip("\n")))


# ================================================================== overview
md(r"""
# Multi-IRS uplink deep unfolding — master pipeline

This notebook reproduces every numerical result of the paper from scratch:

1. **Verification** — the unit-test suite and the Sec. 5.1 sanity check (the run stops if either fails).
2. **Data** — train / validation / test sets and one test set per sweep point (fixed seeds), plus the FP-AO
   teacher labels (complex128; a 1000-iteration FP-AO reference on the test split).
3. **Training** — FONR, MASO and TIAS: learning-rate pilot on the validation set, two-stage training
   (50 supervised + 150 unsupervised epochs, GPU complex64), final train / validation / test metrics in complex128.
4. **Experiments** — the eight paper figures (convergence, wall-clock, CDF, runtime vs *M* and *K*, power sweep,
   WSR vs *K*, zero-shot *M*), the supplementary GPU timing, and the findings report.

**Outputs.** Artifacts go to `cfg.root` (`artifacts/` for the Table-I AO cap, `artifacts_ao{N}/` otherwise):
`datasets/`, `checkpoints/`, `logs/` (training histories, LR pilots, final metrics, history plots,
**`pipeline.log`**) and `results/` (raw `.npz` / `.csv` per experiment). Figures (PNG 300 dpi + PDF) and
`results_findings.md` go to `cfg.paper_dir` (`E:/payam/Paper`). `main.tex` is never modified.

**How to run** (from `E:/payam/code`, machine otherwise idle so the timings are clean):

```
E:/payam/env/Scripts/python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 master_pipeline.ipynb > run.log 2>&1
```

Progress is written as it happens to `{cfg.root}/logs/pipeline.log` (one line per epoch, sweep point, AO chunk and
timing point, with timestamps); classic text progress bars appear in the cell outputs.
Use `python -m nbconvert` (not `jupyter nbconvert`, which may start another Python's kernel).

**Expected runtime** (GTX 1660 Ti, 6 CPU threads; estimates): data and teacher labels ≈ 10 min; training
≈ 15–25 min per model including its LR pilot; WSR sweeps ≈ 30 min; batch-1 CPU timing with 200 samples per point
≈ 3–4 h (the dominant cost); GPU supplement ≈ 30 min. Total ≈ 5 h.

**Resume.** Every stage is cached and validated against the configuration hash (and, for experiments, a
fingerprint of the checkpoints, teacher labels and datasets it used). Re-running the same command after an
interruption continues where it stopped: datasets / labels are loaded, training resumes at the next epoch from
`{model}_last.pt` (bit-exact), finished models are not retrained, and experiments reuse their results (runtime
sweeps and the GPU timing per point).

**Reading the training loss.** Stage II minimises −[a·R + (1−a)·f_q] summed over the layers with weights
ρ_l (in nats): R is the ln-WSR and f_q its Shen–Yu lower bound. The loss is therefore **negative by construction**
(≈ −WSR in nats; lower is better); its two components are logged separately.
""")

# ================================================================== 1. configuration
md(r"""
## 1. Configuration and environment

| parameter | meaning |
|---|---|
| `AO_MAX_ITER` | FP-AO iteration cap for the baseline and the teacher labels. 100 = Table I (`artifacts/`); any other value uses `artifacts_ao{N}/`, so both runs are kept. |
| `TIMING_SUBSET` | number of test / sweep samples timed at batch 1 (spec §7.2 pre-declared subset, identical for every method; `None` = all 1000). WSR values always use every sample. |
| `RESUME` | continue training from the last checkpoint. |
| `FORCE_RETRAIN` | retrain the three models from scratch, ignoring their checkpoints. |
| `FORCE_RERUN` | recompute every experiment even when a valid cached result exists. |

All other settings are the frozen defaults of `config.Config` (shown below).
""")
code(r"""
AO_MAX_ITER = 100        # Table I
TIMING_SUBSET = 200      # batch-1 timing samples per point
RESUME = True
FORCE_RETRAIN = False
FORCE_RERUN = False

import os
import sys

SRC = os.path.abspath("src")      # the notebook lives in code/, the library in code/src
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd
from IPython.display import Markdown, display

import nb_helpers as H
import runlog

pd.set_option("display.precision", 4)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

cfg = H.make_config(ao_max_iter=AO_MAX_ITER, timing_subset=TIMING_SUBSET)
log = runlog.get_logger(cfg, "notebook")
log.info("pipeline started: AO_MAX_ITER=%s TIMING_SUBSET=%s RESUME=%s FORCE_RETRAIN=%s FORCE_RERUN=%s root=%s",
         AO_MAX_ITER, TIMING_SUBSET, RESUME, FORCE_RETRAIN, FORCE_RERUN, cfg.root)
display(H.environment_table(cfg))
display(H.config_table(cfg))
""")

# ================================================================== 2. verification
md(r"""
## 2. Verification

The full unit-test suite (`pytest tests`, including the end-to-end pipeline test on a tiny configuration) and the
Sec. 5.1 sanity check (full FP-AO vs. power-only FP on the validation split; passes if the gain is ≥ 10 %).
**The run stops with a `RuntimeError` if any test or the sanity check fails.**
""")
code(r"""
tests, tests_ok, tests_tail = H.run_unit_tests()
display(tests)
log.info("unit tests: %s (%d passed, %d failed)", "PASS" if tests_ok else "FAIL",
         tests.loc["TOTAL", "passed"], tests.loc["TOTAL", "failed"] + tests.loc["TOTAL", "error"])
if not tests_ok:
    print(tests_tail)

sanity, sanity_ok = H.sanity_table(cfg)
display(sanity)
log.info("sanity check: %s", "PASS" if sanity_ok else "FAIL")
H.require(tests_ok, sanity_ok)
""")

# ================================================================== 3. data
md(r"""
## 3. Data and teacher labels

All datasets (fixed seeds) and the FP-AO teacher labels (efficient FP-AO, complex128, cap `AO_MAX_ITER`;
the test split also gets a 1000-iteration reference). Loaded from `datasets/` when present and valid.
""")
code(r"""
display(H.build_data(cfg))
""")

# ================================================================== 4-6. models
MODELS = [("fonr", "FONR", "DUFP-style nested projected-gradient unfolding (L·T = 30 learned step sizes)."),
          ("maso", "MASO", "DeepFP-style learned element-wise preconditioner (one complex MLP per layer and IRS)."),
          ("tias", "TIAS", "Bipartite user–element message passing with one shared node MLP.")]
for i, (name, label, what) in enumerate(MODELS, start=4):
    md(f"""
## {i}. {label}

{what}

**Learning-rate pilot** — {{0.1, 0.01, 0.001}}, 5 supervised + 15 unsupervised epochs each; the best final mean
validation WSR is selected (cached in `logs/{name}_lr_pilot.csv`).
""")
    code(f"""
lr, pilot = H.lr_pilot(cfg, "{name}")
display(pilot)
""")
    md(f"""
**Training** — epochs 1–50 supervised phase alignment, 51–200 unsupervised deep-supervised loss; the checkpoint
with the best validation WSR is kept. Resumes from `checkpoints/{name}_last.pt`.
""")
    code(f"""
H.train_net(cfg, "{name}", resume=RESUME, force_retrain=FORCE_RETRAIN)
""")
    md(f"""
**Training history** — loss (Stage II negative by construction), its two components, and train / validation WSR
against the FP-AO validation mean; dotted line = stage switch. Saved as `logs/{name}_training_history.png`.
""")
    code(f"""
H.show(H.history_figure(cfg, "{name}"), width=980)
display(H.history(cfg, "{name}").tail(5))
""")
    md(f"""
**Final metrics** — `best.pt` evaluated in complex128 on train / validation / test (the only use of the test set).
""")
    code(f"""
metrics, gap = H.final_metrics(cfg, "{name}")
display(metrics)
display(gap)
""")

# ================================================================== 7. comparison
md(r"""
## 7. Model comparison

Mean WSR (bits/s/Hz) on every split for the three models next to FP-AO at the baseline cap, and the
1000-iteration FP-AO reference on the test split. Efficiency is relative to the baseline-cap FP-AO (the last row:
baseline-cap FP-AO relative to FP-AO-1000). Saved as `logs/master_comparison.csv`.
""")
code(r"""
display(H.comparison_table(cfg))
""")

# ================================================================== 8-15. figures
FIGS = [
    (1, "WSR_vs_iteration", "Convergence: mean WSR (95 % CI) per unfolded layer / FP-AO iteration on the test set, "
     "with the converged FP-AO and FP-AO-1000 means. Table: mean WSR per layer and the % of the AO gain recovered "
     "after 4 layers.", "figure_1(cfg, force_rerun=FORCE_RERUN)"),
    (2, "WSR_vs_wall_clock", "Mean WSR against cumulative CPU wall-clock time (batch 1, complex128). Uses the same "
     "convergence result as Fig. 1 (not recomputed). Table: time to reach 98 / 99 % of the FP-AO mean.",
     "figure_2(cfg)"),
    (3, "CDF", "Empirical CDF of the per-sample gap R_AO − R_method(L) on the test set. Table: gap statistics "
     "(bits/s/Hz).", "figure_3(cfg, force_rerun=FORCE_RERUN)"),
    (4, "avg_time_vs_Number_of_RIS_M", "Average CPU time per FP-AO iteration / unfolded layer versus the number of "
     "IRS elements M (batch 1, complex128; MASO untrained at each M). Cached per sweep point.",
     "figure_4(cfg, force_rerun=FORCE_RERUN)"),
    (5, "etha_vs_CRN_transmit_power", "Efficiency η = WSR(model) / WSR(FP-AO) versus the CRN peak power "
     "(PMN = CRN − 10 dB); models trained at 20 dBm, not retrained.", "figure_5(cfg, force_rerun=FORCE_RERUN)"),
    (6, "WSR_vs_Number_of_active_users_K", "Mean WSR versus the number of active users K (models trained at "
     "K = 6, not retrained).", "figure_6(cfg, force_rerun=FORCE_RERUN)"),
    (7, "avg_time_vs_Number_of_active_users_K", "Average CPU time per iteration / layer versus K "
     "(batch 1, complex128). Cached per sweep point.", "figure_7(cfg, force_rerun=FORCE_RERUN)"),
    (8, "WSR_vs_M_ZeroShot", "Zero-shot generalisation: mean WSR versus M with the models trained at M = 80 "
     "(MASO zero-pads / truncates its input).", "figure_8(cfg, force_rerun=FORCE_RERUN)"),
]
for k, (fig_no, fname, what, call) in enumerate(FIGS, start=8):
    md(f"""
## {k}. Fig. {fig_no} — `{fname}`

{what}
""")
    code(f"""
png, table = H.{call}
H.show(png)
display(table)
""")

# ================================================================== 16. GPU + report
md(r"""
## 16. Supplementary GPU timing and findings report

All methods on CUDA in complex64 (MAGMA linear algebra). FP-AO run to convergence is numerically unstable in
complex64 on the GPU, so its GPU time is reported as *not available* with the failure percentage instead of a
mean over the surviving samples. Then `results_findings.md` is written (claims vs. measured, deviations, metrics,
latency, LR pilot, η, suggested text changes); the head of its claims table is shown below.
""")
code(r"""
display(H.gpu_supplement(cfg, force_rerun=FORCE_RERUN))
report_path, claims_head = H.findings_report(cfg)
print("findings report:", report_path)
display(Markdown(claims_head))
log.info("pipeline finished")
""")

NOTEBOOK = v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.validate(NOTEBOOK)

if __name__ == "__main__":
    import os
    nbf.write(NOTEBOOK, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "master_pipeline.ipynb"))
    print("wrote master_pipeline.ipynb:", sum(c.cell_type == "code" for c in cells), "code cells,",
          sum(c.cell_type == "markdown" for c in cells), "markdown cells")
