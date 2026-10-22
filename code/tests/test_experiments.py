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
