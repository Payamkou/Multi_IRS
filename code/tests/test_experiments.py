"""Spec §7 experiments 1-3 (convergence, wall-clock, CDF) and the shared plot style."""
import dataclasses
import os
import time

import numpy as np
import pytest
import torch

import teacher
import train
from config import Config, config_hash
from models import build_model

NETS = ("fonr", "maso", "tias")


def _tiny_cfg(root, paper, **kw):
    base = dict(n_train=8, n_val=8, n_test=6, M=12, K=3, ao_max_iter=20, train_device="cpu",
                eval_device="cpu", timing_threads=2, timing_warmup=1, root=str(root),
                paper_dir=str(paper))
    base.update(kw)
    return dataclasses.replace(Config(), **base)


def _fake_checkpoints(cfg):
    """Untrained models saved with train.py's checkpoint dict format."""
    ck = os.path.join(cfg.root, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    kw = {"fonr": {"mu_init": 0.05}, "maso": {"step_scale": 0.1}, "tias": {}}
    for i, name in enumerate(NETS):
        torch.manual_seed(100 + i)
        m = build_model(name, cfg, **kw[name])
        state = {"model_name": name, "epoch": 1, "stage": "supervised",
                 "model_state": {k: t.detach().cpu().clone() for k, t in m.state_dict().items()},
                 "optimizer_state": {}, "selected_lr": 0.01, "best_val_wsr": 0.0, "best_epoch": 1,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": None, "metrics_history": {},
                 "config": cfg.to_dict(), "cfg_hash": config_hash(cfg, train.TRAIN_KEYS)}
        torch.save(state, os.path.join(ck, f"{name}_best.pt"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp_root")
    paper = tmp_path_factory.mktemp("paper")
    cfg = _tiny_cfg(root, paper)
    teacher.build_teacher_labels(cfg, chunk=8)
    _fake_checkpoints(cfg)
    return cfg


def test_style_fixed():
    import plotting
    assert plotting.STYLE["FP-AO"] == plotting.STYLE["FP-AO (explicit)"]
    assert plotting.STYLE["FP-AO"]["color"] == "black" and plotting.STYLE["FP-AO"]["marker"] == "s"
    assert plotting.STYLE["FP-AO (efficient)"]["color"] == "grey"
    assert plotting.STYLE["FP-AO (efficient)"]["marker"] == "D"
    assert (plotting.STYLE["FONR"]["color"], plotting.STYLE["FONR"]["marker"]) == ("C0", "o")
    assert (plotting.STYLE["MASO"]["color"], plotting.STYLE["MASO"]["marker"]) == ("C1", "^")
    assert (plotting.STYLE["TIAS"]["color"], plotting.STYLE["TIAS"]["marker"]) == ("C2", "v")


def test_load_trained_models_fig1(env):
    from experiments.convergence import load_trained_models
    models = load_trained_models(env)
    assert set(models) == set(NETS)
    sd = torch.load(os.path.join(env.root, "checkpoints", "maso_best.pt"), weights_only=False)
    assert torch.allclose(models["maso"].step_scale.double(), sd["model_state"]["step_scale"].double())
    assert all(not m.training for m in models.values())


def test_exp_cache_reuse(env):
    from experiments.convergence import run_convergence, run_cdf
    p = run_convergence(env)
    assert p.name == "exp_01_convergence.npz" and p.exists()
    assert os.path.normpath(str(p.parent)) == os.path.normpath(os.path.join(env.root, "results"))
    assert (p.parent / "exp_01_convergence.csv").exists()
    m1 = os.path.getmtime(p)
    time.sleep(0.05)
    assert run_convergence(env) == p
    assert os.path.getmtime(p) == m1
    q = run_cdf(env)
    assert q.name == "exp_03_cdf.npz"
    m2 = os.path.getmtime(q)
    time.sleep(0.05)
    run_cdf(env)
    assert os.path.getmtime(q) == m2


def test_convergence_contents_fig1_fig2(env):
    from experiments.convergence import load_trained_models, run_convergence
    import data
    z = np.load(run_convergence(env))
    B, L = env.n_test, env.L
    t = teacher.load_teacher(env, "test")
    for n in NETS:
        assert z[f"wsr_{n}"].shape == (B, L + 1)
        assert z[f"time_{n}"].shape == (B, L)          # timing_subset None -> all samples
        assert (z[f"time_{n}"] > 0).all()
    # layer-L WSR equals train.evaluate's complex128 output
    d = data.build_or_load(env, "test")
    r = train.evaluate(load_trained_models(env)["tias"], d["ch"], d["theta0"])
    assert np.allclose(z["wsr_tias"][:, -1], r["wsr_per_sample"].numpy(), rtol=1e-10)
    assert np.allclose(z["wsr_ao_traj"], t["wsr_traj"].numpy())
    assert np.isclose(float(z["ao100_mean"]), float(t["wsr_ao"].mean()))
    assert np.isclose(float(z["ao1000_mean"]), float(t["wsr_ref1000"].mean()))
    for mode in ("explicit", "efficient"):
        ct = z[f"ao_cumtime_{mode}"]
        assert ct.shape[0] == B and (np.diff(ct, axis=1) >= 0).all()
        assert (z[f"ao_niter_{mode}"] >= 1).all()


def test_timing_subset_fig2(tmp_path, env):
    from experiments.convergence import run_convergence
    import shutil
    root = tmp_path / "r"
    shutil.copytree(env.root, root, ignore=shutil.ignore_patterns("results"))
    cfg = dataclasses.replace(env, root=str(root), timing_subset=3)
    z = np.load(run_convergence(cfg))
    assert z["wsr_fonr"].shape[0] == env.n_test
    assert z["time_fonr"].shape[0] == 3 and z["ao_cumtime_explicit"].shape[0] == 3


def test_cdf_gap_fig3(env):
    from experiments.convergence import run_cdf, run_convergence
    z = np.load(run_convergence(env))
    c = np.load(run_cdf(env))
    t = teacher.load_teacher(env, "test")
    for n in NETS:
        assert np.allclose(c[f"gap_{n}"], t["wsr_ao"].numpy() - z[f"wsr_{n}"][:, -1])
        assert np.allclose(c[f"gap_vs1000_{n}"], t["wsr_ref1000"].numpy() - z[f"wsr_{n}"][:, -1])


def test_summary_keys(env):
    from experiments.convergence import run_convergence, summarize_convergence
    z = run_convergence(env)
    s = summarize_convergence(z)
    keys = ["ao100_mean", "ao1000_mean", "ao_conv_time_ms_explicit", "ao_conv_time_ms_efficient"]
    for n in NETS:
        keys += [f"recovery_pct_layer{l}_{n}" for l in (4, 6, 8)]
        for pre in ("gap_p95", "gap_min", "gap_median"):
            keys += [f"{pre}_{n}", f"{pre}_{n}_vs1000"]
    for m in NETS + ("ao_explicit", "ao_efficient"):
        keys += [f"time_to_98pct_{m}_ms", f"time_to_99pct_{m}_ms"]
    for k in keys:
        assert k in s, k
        assert isinstance(s[k], float)
    zz = np.load(z)
    w = zz["wsr_fonr"].mean(0)
    want = 100 * (w[4] - w[0]) / (float(zz["ao100_mean"]) - w[0])
    assert s["recovery_pct_layer4_fonr"] == pytest.approx(want)
    assert s["ao_conv_time_ms_explicit"] > 0
    # AO reaches its own converged mean, so its 99% time is finite
    assert np.isfinite(s["time_to_99pct_ao_efficient_ms"])


def test_plots_written_fig1_fig2_fig3(env):
    from experiments.convergence import make_figures
    out = make_figures(env)
    for name in ("WSR_vs_iteration", "WSR_vs_wall_clock", "CDF"):
        for ext in (".png", ".pdf"):
            f = os.path.join(env.paper_dir, name + ext)
            assert os.path.exists(f) and os.path.getsize(f) > 0
        assert name in out
    assert os.path.normpath(env.paper_dir) != os.path.normpath("E:/payam/Paper")


def _copy_env(env, tmp_path, tag):
    import shutil
    root = tmp_path / tag
    shutil.copytree(env.root, root, ignore=shutil.ignore_patterns("results"))
    return dataclasses.replace(env, root=str(root))


def test_cache_fingerprint_helper(env, tmp_path):
    from experiments.cache import artifact_fingerprint
    cfg = _copy_env(env, tmp_path, "fp")
    f1 = artifact_fingerprint(cfg)
    assert set(f1) == {"checkpoints/fonr_best.pt", "checkpoints/maso_best.pt",
                       "checkpoints/tias_best.pt", "datasets/teacher_labels.pt"}
    assert artifact_fingerprint(cfg) == f1
    p = os.path.join(cfg.root, "checkpoints", "tias_best.pt")
    ck = torch.load(p, weights_only=False)
    ck["model_state"] = {k: v + 0.5 if v.is_floating_point() else v for k, v in ck["model_state"].items()}
    torch.save(ck, p)
    assert artifact_fingerprint(cfg)["checkpoints/tias_best.pt"] != f1["checkpoints/tias_best.pt"]


def test_cache_recompute_on_checkpoint_change(env, tmp_path, capsys):
    from experiments.convergence import run_cdf, run_convergence
    cfg = _copy_env(env, tmp_path, "rc")
    p = run_convergence(cfg)
    q = run_cdf(cfg)
    w1 = np.load(p)["wsr_tias"].copy()
    m1, mq = os.path.getmtime(p), os.path.getmtime(q)
    time.sleep(0.05)
    ck_p = os.path.join(cfg.root, "checkpoints", "tias_best.pt")
    ck = torch.load(ck_p, weights_only=False)
    torch.manual_seed(12345)
    ck["model_state"] = build_model("tias", cfg).state_dict()
    torch.save(ck, ck_p)
    run_convergence(cfg)
    assert os.path.getmtime(p) > m1
    assert "recomput" in capsys.readouterr().out.lower()
    assert np.abs(np.load(p)["wsr_tias"] - w1).max() > 1e-9  # new weights, new values
    # the CDF follows the new convergence npz
    c = np.load(run_cdf(cfg))
    assert os.path.getmtime(q) > mq
    assert np.allclose(c["gap_tias"], np.load(p)["wsr_ao100"] - np.load(p)["wsr_tias"][:, -1])


def test_cdf_rebuilt_after_convergence_force_rerun(env, tmp_path):
    from experiments.convergence import run_cdf, run_convergence
    cfg = _copy_env(env, tmp_path, "fr")
    run_convergence(cfg)
    q = run_cdf(cfg)
    mq = os.path.getmtime(q)
    time.sleep(0.05)
    run_convergence(cfg, force_rerun=True)
    run_cdf(cfg)
    assert os.path.getmtime(q) > mq


def test_summary_ao_baseline_max_iter(env):
    from experiments.convergence import run_convergence, summarize_convergence
    p = run_convergence(env)
    assert int(np.load(p)["ao_baseline_max_iter"]) == env.ao_max_iter
    assert summarize_convergence(p)["ao_baseline_max_iter"] == float(env.ao_max_iter)


# ================================================================ Task 13: sweeps (Figs. 4-8)
SWEEP_FIGS = {"runtime_vs_M": "avg_time_vs_Number_of_RIS_M", "power": "etha_vs_CRN_transmit_power",
              "wsr_vs_K": "WSR_vs_Number_of_active_users_K", "runtime_vs_K": "avg_time_vs_Number_of_active_users_K",
              "zeroshot": "WSR_vs_M_ZeroShot"}


@pytest.fixture(scope="module")
def sweep_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep_root")
    paper = tmp_path_factory.mktemp("sweep_paper")
    # training M = 80 (so MASO pads at M=20 and truncates at M=100); 80 deliberately NOT in zeroshot_M
    cfg = _tiny_cfg(root, paper, n_test=4, M=80, K=3, sweep_M=(20, 40), sweep_K=(3, 4),
                    sweep_crn_dbm=(20, 30), zeroshot_M=(20, 100), timing_subset=2)
    _fake_checkpoints(cfg)
    return cfg


def _sweep_runs():
    from experiments import sweeps
    return {"runtime_vs_M": sweeps.run_runtime_vs_M, "power": sweeps.run_power_sweep,
            "wsr_vs_K": sweeps.run_wsr_vs_K, "runtime_vs_K": sweeps.run_runtime_vs_K,
            "zeroshot": sweeps.run_zeroshot_M}


def test_sweep_outputs_and_figures(sweep_env):
    from pathlib import Path
    for key, fn in _sweep_runs().items():
        p, s = fn(sweep_env)
        p = Path(p)
        assert p.exists() and p.suffix == ".npz" and p.name.startswith("exp_0"), key
        assert p.with_suffix(".csv").exists(), key
        assert isinstance(s, dict) and s, key
        for ext in (".png", ".pdf"):
            f = os.path.join(sweep_env.paper_dir, SWEEP_FIGS[key] + ext)
            assert os.path.exists(f) and os.path.getsize(f) > 0, f
    names = sorted(os.listdir(os.path.join(sweep_env.root, "results")))
    for pre in ("exp_04_", "exp_05_", "exp_06_", "exp_07_", "exp_08_"):
        assert any(n.startswith(pre) and n.endswith(".npz") for n in names), pre
    assert os.path.normpath(sweep_env.paper_dir) != os.path.normpath("E:/payam/Paper")


def test_sweep_cache_reuse(sweep_env):
    for key, fn in _sweep_runs().items():
        p, s1 = fn(sweep_env)
        m = os.path.getmtime(p)
        time.sleep(0.05)
        p2, s2 = fn(sweep_env)
        assert str(p2) == str(p) and os.path.getmtime(p) == m, key
        assert s1.keys() == s2.keys()


def test_power_eta_both_statistics(sweep_env):
    from experiments.sweeps import run_power_sweep
    p, s = run_power_sweep(sweep_env)
    z = np.load(p)
    assert int(z["ao_baseline_max_iter"]) == sweep_env.ao_max_iter
    assert s["ao_baseline_max_iter"] == float(sweep_env.ao_max_iter)
    for n in NETS:
        etas = []
        for P in sweep_env.sweep_crn_dbm:
            w, a = z[f"wsr_{n}_P{P}"], z[f"wsr_ao_P{P}"]
            assert w.shape == a.shape == (sweep_env.n_test,)
            rom = 100 * w.mean() / a.mean()
            mor = 100 * np.mean(w / a)
            assert s[f"eta_ratio_of_means_{n}_P{P}"] == pytest.approx(rom)
            assert s[f"eta_mean_of_ratios_{n}_P{P}"] == pytest.approx(mor)
            assert float(z[f"eta_ratio_of_means_{n}_P{P}"]) == pytest.approx(rom)
            assert float(z[f"eta_mean_of_ratios_{n}_P{P}"]) == pytest.approx(mor)
            etas.append(rom)
        assert s[f"eta_range_{n}"] == pytest.approx((min(etas), max(etas)))
    # AO is run fresh (efficient, cfg tol / max_iter) on each power set
    import ao, data
    P = sweep_env.sweep_crn_dbm[-1]
    d = data.build_or_load(sweep_env, "sweep_power", P)
    ch = d["ch"].to("cpu", torch.complex128)
    p0, phi0 = ao.init_state(ch, d["theta0"])
    r = ao.run_ao(ch, p0, phi0, mode="efficient", tol=sweep_env.ao_tol, max_iter=sweep_env.ao_max_iter)
    assert np.allclose(z[f"wsr_ao_P{P}"], r.wsr_traj[:, -1].numpy(), rtol=1e-10)


def test_runtime_has_both_ao_curves(sweep_env):
    from experiments.sweeps import run_runtime_vs_K, run_runtime_vs_M
    S = sweep_env.timing_subset
    for fn, tag, pts in ((run_runtime_vs_M, "M", sweep_env.sweep_M), (run_runtime_vs_K, "K", sweep_env.sweep_K)):
        p, s = fn(sweep_env)
        z = np.load(p)
        assert int(z["timing_n_samples"]) == S and s["timing_n_samples"] == float(S)
        assert s["timing_threads"] == float(z["timing_threads"])
        for x in pts:
            for m in ("ao_explicit", "ao_efficient", "fonr", "maso", "tias"):
                t = z[f"time_{m}_{tag}{x}"]
                assert t.shape == (S,) and (t > 0).all()
                assert s[f"runtime_ms_{m}_{tag}{x}"] == pytest.approx(1e3 * t.mean())
    z = np.load(run_runtime_vs_M(sweep_env)[0])
    # MASO for runtime-vs-M is instantiated at each M (M_train = M)
    assert list(z["maso_M_train"]) == list(sweep_env.sweep_M)
    assert str(z["device"]) == "cpu" and str(z["cdtype"]) == "complex128"


def test_wsr_vs_K_contents(sweep_env):
    from experiments.sweeps import run_wsr_vs_K
    from experiments.convergence import load_trained_models
    import data
    p, s = run_wsr_vs_K(sweep_env)
    z = np.load(p)
    K = sweep_env.sweep_K[-1]
    d = data.build_or_load(sweep_env, "sweep_K", K)
    r = train.evaluate(load_trained_models(sweep_env)["fonr"], d["ch"], d["theta0"])
    assert np.allclose(z[f"wsr_fonr_K{K}"], r["wsr_per_sample"].numpy(), rtol=1e-10)
    for m in ("ao",) + NETS:
        for k in sweep_env.sweep_K:
            assert s[f"wsr_{m}_K{k}"] == pytest.approx(z[f"wsr_{m}_K{k}"].mean())
    assert int(z["ao_baseline_max_iter"]) == sweep_env.ao_max_iter


def test_zeroshot_includes_M80(sweep_env):
    from experiments.sweeps import run_zeroshot_M
    from experiments.convergence import load_trained_models
    import data
    p, s = run_zeroshot_M(sweep_env)
    z = np.load(p)
    assert list(z["M_values"]) == [20, 80, 100]
    assert int(z["M_reference"]) == 80
    for m in ("ao",) + NETS:
        for M in (20, 80, 100):
            assert f"wsr_{m}_M{M}_zeroshot" in s
            assert z[f"wsr_{m}_M{M}"].shape == (sweep_env.n_test,)
    d = data.build_or_load(sweep_env, "sweep_M", 100)
    r = train.evaluate(load_trained_models(sweep_env)["maso"], d["ch"], d["theta0"])
    assert np.allclose(z["wsr_maso_M100"], r["wsr_per_sample"].numpy(), rtol=1e-10)


def test_gpu_timing_skips_without_cuda(sweep_env, monkeypatch, capsys):
    from experiments import sweeps
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    p, s = sweeps.run_gpu_timing(sweep_env)
    assert p is None and s.get("gpu_available") == 0.0
    assert "cuda" in capsys.readouterr().out.lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_timing_runs(sweep_env):
    from experiments import sweeps
    p, s = sweeps.run_gpu_timing(sweep_env)
    z = np.load(p)
    assert os.path.basename(p) == "exp_supp_gpu_timing.npz"
    assert os.path.exists(str(p)[:-4] + ".csv")
    assert str(z["device"]) == "cuda" and str(z["cdtype"]) == "complex64"
    for m in ("ao_explicit", "ao_efficient", "fonr", "maso", "tias"):
        assert f"runtime_ms_{m}_M{sweep_env.sweep_M[0]}" in s
        assert f"runtime_ms_{m}_K{sweep_env.sweep_K[0]}" in s
        assert f"step_time_ms_{m}" in s
    assert not any(f.startswith("gpu") for f in os.listdir(sweep_env.paper_dir))


# ================================================================ per-point timing cache (final review I2/I3)
def test_runtime_per_point_cache(sweep_env, monkeypatch):
    """Each runtime point is its own npz; losing one point (and the combined file, as after an
    interruption) recomputes only that point."""
    from experiments import sweeps
    p, s0 = sweeps.run_runtime_vs_K(sweep_env)
    K0, K1 = sweep_env.sweep_K
    pts = {K: sweeps.point_path(sweep_env, "exp_07_runtime_vs_K", f"K{K}") for K in (K0, K1)}
    assert all(os.path.exists(q) for q in pts.values())
    before = {K: os.path.getmtime(q) for K, q in pts.items()}
    keep = np.load(p)[f"time_fonr_K{K0}"].copy()
    os.remove(pts[K1])
    os.remove(p)
    time.sleep(0.05)
    calls, real = [], sweeps._runtime_point
    monkeypatch.setattr(sweeps, "_runtime_point", lambda cfg, split, x, *a, **k: (calls.append(x), real(cfg, split, x, *a, **k))[1])
    p2, s2 = sweeps.run_runtime_vs_K(sweep_env)
    assert calls == [K1]
    assert os.path.getmtime(pts[K0]) == before[K0] and os.path.exists(pts[K1]) and os.path.exists(p2)
    assert np.array_equal(np.load(p2)[f"time_fonr_K{K0}"], keep)
    assert s2.keys() == s0.keys()
    # a valid combined file is reused without touching any point
    calls.clear()
    sweeps.run_runtime_vs_K(sweep_env)
    assert calls == []


def test_runtime_vs_M_per_point_files(sweep_env):
    from experiments import sweeps
    sweeps.run_runtime_vs_M(sweep_env)
    for M in sweep_env.sweep_M:
        q = sweeps.point_path(sweep_env, "exp_04_runtime_vs_M", f"M{M}")
        z = np.load(q)
        assert int(z["point"]) == M and str(z["device"]) == "cpu" and str(z["cdtype"]) == "complex128"
        assert all(f"time_{m}_M{M}" in z.files for m in ("ao_explicit", "ao_efficient", "fonr", "maso", "tias"))


def test_gpu_summary_always_has_failure_counts_and_pct():
    from experiments import sweeps
    S, Ms, Ks = 4, (20,), (3,)
    z = {"timing_threads": 2, "timing_n_samples": S}
    for tag, x in (("M", 20), ("K", 3)):
        for m in sweeps.TIME_METHODS:
            z[f"time_{m}_{tag}{x}"] = np.full(S, 1e-3)
    z["time_ao_explicit_M20"][0] = np.nan
    for n in NETS:
        z[f"step_time_{n}"] = np.full((S, 2), 1e-4)
    for mode in sweeps.AO_MODES:
        z[f"ao_cumtime_{mode}"] = np.full(S, 1e-2)
        z[f"ao_steptime_{mode}"] = np.full(S, 1e-3)
    z["ao_cumtime_efficient"][:] = np.nan          # every sample failed: NaN mean, no crash
    z["ao_steptime_efficient"][:] = np.nan
    s = sweeps.summarize_gpu(z, Ms, Ks)
    for tag, x in (("M", 20), ("K", 3)):
        for m in sweeps.TIME_METHODS:
            assert f"runtime_n_failed_{m}_{tag}{x}" in s and f"runtime_pct_failed_{m}_{tag}{x}" in s
    assert s["runtime_n_failed_ao_explicit_M20"] == 1 and s["runtime_pct_failed_ao_explicit_M20"] == 25.0
    assert s["runtime_pct_failed_ao_efficient_K3"] == 0.0
    assert s["ao_conv_n_failed_explicit"] == 0 and s["ao_conv_pct_failed_explicit"] == 0.0
    assert s["ao_conv_pct_failed_efficient"] == 100.0 and np.isnan(s["ao_conv_time_ms_efficient"])
    # CPU summaries keep their old key set (no failure keys when nothing failed)
    assert not any("failed" in k for k in sweeps._summ_runtime(z, "K", Ks))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_timing_per_point_and_magma(sweep_env, monkeypatch):
    from experiments import sweeps
    prev = torch.backends.cuda.preferred_linalg_library()
    seen, real = [], sweeps._runtime_point
    monkeypatch.setattr(sweeps, "_runtime_point", lambda cfg, split, x, *a, **k: (
        seen.append((x, str(torch.backends.cuda.preferred_linalg_library()))), real(cfg, split, x, *a, **k))[1])
    p, s = sweeps.run_gpu_timing(sweep_env, force_rerun=True)
    assert torch.backends.cuda.preferred_linalg_library() == prev          # restored
    assert seen and all("Magma" in b for _, b in seen)
    assert str(np.load(p)["linalg_library"]) == "magma"
    labels = [f"M{x}" for x in sweep_env.sweep_M] + [f"K{x}" for x in sweep_env.sweep_K] + ["steps"]
    q = {lab: sweeps.point_path(sweep_env, "exp_supp_gpu_timing", lab) for lab in labels}
    assert all(os.path.exists(v) for v in q.values())
    for mode in sweeps.AO_MODES:
        assert f"ao_conv_pct_failed_{mode}" in s
    import csv
    with open(str(p)[:-4] + ".csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {"n_failed", "pct_failed"} <= set(rows[0]) and any(r["experiment"] == "ao_to_convergence" for r in rows)
    before = {lab: os.path.getmtime(v) for lab, v in q.items()}
    lost = f"K{sweep_env.sweep_K[-1]}"
    os.remove(q[lost])
    os.remove(p)
    time.sleep(0.05)
    seen.clear()
    sweeps.run_gpu_timing(sweep_env)
    assert [x for x, _ in seen] == [sweep_env.sweep_K[-1]]
    assert all(os.path.getmtime(q[lab]) == before[lab] for lab in labels if lab != lost)


def test_fig1_single_dashed_line_when_baseline_is_ao1000(env):
    """Final review M5: no duplicated legend entry when the baseline is the 1000-iteration AO."""
    import matplotlib.pyplot as plt
    from experiments.convergence import plot_fig1, run_convergence
    z = dict(np.load(run_convergence(env)))

    def dashed_labels(zz):
        fig = plot_fig1(zz)
        labs = [l.get_label() for l in fig.axes[0].get_lines() if l.get_label().startswith("FP-AO (")]
        plt.close(fig)
        return labs
    z["ao1000_mean"] = z["ao100_mean"] + 0.5
    assert dashed_labels(z) == ["FP-AO (converged)", "FP-AO (1000 it.)"]
    z1 = dict(z, ao_baseline_max_iter=np.int64(1000), ao1000_mean=z["ao100_mean"])
    assert dashed_labels(z1) == ["FP-AO (converged, 1000 it.)"]
    z2 = dict(z, ao1000_mean=z["ao100_mean"])          # coinciding means, cap 20
    assert dashed_labels(z2) == ["FP-AO (converged, 1000 it.)"]
