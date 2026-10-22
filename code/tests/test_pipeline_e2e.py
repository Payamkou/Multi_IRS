"""End-to-end pipeline check (moved out of the notebook, Task 16 D).

Runs the same high-level functions the notebook calls (``nb_helpers``), in the notebook's order, on a tiny
configuration in temporary directories: datasets + teacher labels, LR pilot / training / history figure /
final metrics for the three models, the comparison table, all eight paper figures, the GPU supplement and
the findings report. Real artifacts and Paper/ are never touched. Takes ~2-3 minutes.
"""
import dataclasses
import json
import os

import nbformat
import pandas as pd
import pytest

import nb_helpers as H
from config import Config

FIGS = ["WSR_vs_iteration", "WSR_vs_wall_clock", "CDF", "avg_time_vs_Number_of_RIS_M",
        "etha_vs_CRN_transmit_power", "WSR_vs_Number_of_active_users_K", "avg_time_vs_Number_of_active_users_K",
        "WSR_vs_M_ZeroShot"]


def _tiny(tmp_path):
    return H.make_config(ao_max_iter=20, timing_subset=3, M=16, maso_M_train=16, K=6, n_train=200, n_val=50,
                         n_test=50, batch_size=50, n_sup_epochs=2, n_unsup_epochs=2, pilot_sup_epochs=1,
                         pilot_unsup_epochs=1, lr_grid=(0.01, 0.001), timing_threads=2, timing_warmup=1,
                         sweep_M=(8, 16), sweep_K=(4, 6), sweep_crn_dbm=(20, 30), zeroshot_M=(8, 24),
                         train_device="cpu", eval_device="cpu",
                         root=str(tmp_path / "artifacts"), paper_dir=str(tmp_path / "Paper"))


def test_make_config_roots():
    assert H.make_config().root == H.artifact_root(100) and H.artifact_root(100).endswith("/artifacts")
    assert H.make_config(ao_max_iter=1000).root.endswith("/artifacts_ao1000")
    c = H.make_config()
    assert c.ao_max_iter == 100 and c.timing_subset == 200 and c.M == Config().M


def test_unit_test_runner_table():
    df, ok, _ = H.run_unit_tests(extra_args=("-k", "test_rho_weights_sum_to_one"))
    assert ok and df.loc["TOTAL", "passed"] == 1 and df.loc["TOTAL", "status"] == "PASS"
    with pytest.raises(RuntimeError, match="unit tests failed"):
        H.require(False, True)
    H.require(True, True)


def test_pipeline_end_to_end(tmp_path):
    cfg = _tiny(tmp_path)
    assert os.path.normcase(cfg.root) != os.path.normcase(Config().root)
    assert isinstance(H.environment_table(cfg), pd.DataFrame) and len(H.config_table(cfg)) > 10
    sanity, ok = H.sanity_table(cfg)
    assert "gain_pct" in sanity.index
    data = H.build_data(cfg)
    assert list(data.index) == ["train", "val", "test"] and data.loc["test", "samples"] == 50

    for name in H.NETS:
        lr, pilot = H.lr_pilot(cfg, name)
        assert lr in cfg.lr_grid and (pilot["selected"] == "yes").sum() == 1
        best = H.train_net(cfg, name)
        assert os.path.exists(best)
        h = H.history(cfg, name)
        assert len(h) == 4 and {"train_wsr_nats_term", "train_fq_term"} <= set(h.columns)
        un = h[h["stage"] == "unsupervised"]
        # Stage-II loss = -a R - (1-a) f_q, negative, and f_q is a lower bound of the ln-WSR term
        assert (un["train_loss"] < 0).all()
        assert ((-0.5 * un["train_wsr_nats_term"] - 0.5 * un["train_fq_term"] - un["train_loss"]).abs()
                < 1e-4 * un["train_loss"].abs().max()).all()
        assert (un["train_fq_term"] <= un["train_wsr_nats_term"] + 1e-6).all()
        assert os.path.exists(H.history_figure(cfg, name))
        m, gap = H.final_metrics(cfg, name)
        assert list(m.index) == ["train", "val", "test"] and len(gap) == 3

    comp = H.comparison_table(cfg)
    assert {"FONR", "MASO", "TIAS"} <= set(comp.index) and os.path.exists(
        os.path.join(cfg.root, "logs", "master_comparison.csv"))

    outs = [H.figure_1(cfg), H.figure_2(cfg), H.figure_3(cfg), H.figure_4(cfg), H.figure_5(cfg),
            H.figure_6(cfg), H.figure_7(cfg), H.figure_8(cfg)]
    for name, (png, tab) in zip(FIGS, outs):
        assert os.path.basename(png) == f"{name}.png" and os.path.exists(png), name
        assert os.path.exists(png[:-4] + ".pdf") and isinstance(tab, pd.DataFrame) and not tab.empty, name

    gpu = H.gpu_supplement(cfg)
    assert isinstance(gpu, pd.DataFrame)
    path, head = H.findings_report(cfg)
    assert os.path.exists(path) and str(path).startswith(str(tmp_path)) and "|" in head

    log = open(os.path.join(cfg.root, "logs", "pipeline.log"), encoding="utf-8").read()
    for s in ("iuf.teacher", "iuf.train: [fonr] epoch 4/4", "iuf.train: [pilot tias", "iuf.report"):
        assert s in log, s


def test_notebook_is_clean():
    """The generated notebook is valid, unexecuted, has no smoke mode / env-var switches and only calls the
    pipeline helpers; the notebook on disk has the same cells (its outputs may exist after a real run)."""
    import make_notebook
    nb = make_notebook.NOTEBOOK
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and all(not c.outputs and c.execution_count is None for c in code)
    src = chr(10).join(c.source for c in nb.cells)
    for bad in ("IUF_", "SMOKE", "smoke", "os.environ"):
        assert bad not in src, bad
    for fn in ("figure_1", "figure_8", "gpu_supplement", "findings_report", "run_unit_tests", "train_net"):
        assert f"H.{fn}(" in src, fn
    assert "AO_MAX_ITER = 100" in src and "TIMING_SUBSET = 200" in src
    assert src.count("H.figure_1(") == 1 and "H.figure_2(cfg)" in src     # exp_01 computed once
    disk = nbformat.read(os.path.join(H.CODE_DIR, "master_pipeline.ipynb"), as_version=4)
    assert [c.source for c in disk.cells] == [c.source for c in nb.cells], "run make_notebook.py"


def test_notebook_executes_on_tiny_config(tmp_path):
    """Execute a COPY of the notebook whose config line is patched to the tiny test configuration (the
    notebook itself has no such option) and whose test cell runs one quick test (no recursion)."""
    from nbclient import NotebookClient
    import make_notebook
    nb = nbformat.reads(nbformat.writes(make_notebook.NOTEBOOK), as_version=4)
    tiny = _tiny(tmp_path)
    over = {k: v for k, v in dataclasses.asdict(tiny).items() if v != getattr(Config(), k)}
    cfg_line = "cfg = H.make_config(ao_max_iter=AO_MAX_ITER, timing_subset=TIMING_SUBSET)"
    test_line = "tests, tests_ok, tests_tail = H.run_unit_tests()"
    srcs = [c.source for c in nb.cells]
    assert sum(cfg_line in s for s in srcs) == 1 and sum(test_line in s for s in srcs) == 1
    for c in nb.cells:
        c.source = c.source.replace(cfg_line, f"from config import Config\ncfg = Config(**{over!r})")
        c.source = c.source.replace(test_line, 'tests, tests_ok, tests_tail = H.run_unit_tests(extra_args=("-k", '
                                               '"test_rho_weights_sum_to_one"))')
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": H.CODE_DIR}}).execute()
    errs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errs, errs
    for name in FIGS:
        assert os.path.exists(os.path.join(tiny.paper_dir, f"{name}.png")), name
    assert os.path.exists(os.path.join(tiny.paper_dir, "results_findings.md"))
