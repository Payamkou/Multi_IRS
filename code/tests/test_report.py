"""Task 14: automated findings report (spec §10).

All inputs are synthetic and live in pytest tmp dirs; E:/payam/Paper is never written."""
import csv
import dataclasses
import json
import os

import numpy as np
import pytest
import torch

import train
from config import Config, config_hash
from experiments import convergence as cv
from experiments import sweeps as sw
from experiments.cache import artifact_fingerprint, fingerprint_array

NETS = ("fonr", "maso", "tias")
REAL_PAPER = os.path.normcase(os.path.abspath("E:/payam/Paper"))


# ---------------------------------------------------------------- synthetic artifacts
def _cfg(tmp_path, **kw):
    base = dict(n_test=6, M=80, K=6, L=10, sweep_M=(20, 80, 180), sweep_K=(10, 60), sweep_crn_dbm=(20, 45),
                zeroshot_M=(20, 160), timing_subset=None, ao_max_iter=100, timing_threads=2,
                root=str(tmp_path / "artifacts"), paper_dir=str(tmp_path / "paper"))
    base.update(kw)
    return dataclasses.replace(Config(), **base)


def _meta(cfg, keys, files=()):
    return {"cfg_json": np.array(json.dumps(cfg.to_dict())), "cfg_hash": np.array(config_hash(cfg, keys)),
            "artifact_fingerprint": fingerprint_array(artifact_fingerprint(cfg, extra=list(files)))}


def _save(cfg, name, **arrs):
    d = os.path.join(cfg.root, "results")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, name), **arrs)


def _write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_synthetic(cfg, gpu=True, seed=0):
    """Producer-format npz/CSV/JSON files (same keys as Tasks 8-13 write) with valid cache metadata."""
    rng = np.random.default_rng(seed)
    B, L = cfg.n_test, cfg.L
    # ---- exp_01 (convergence)
    out = {}
    base = 1.5 + 0.1 * rng.random(B)
    for i, n in enumerate(NETS):
        frac = 1 - np.exp(-0.6 * np.arange(L + 1))
        out[f"wsr_{n}"] = base[:, None] + (2.5 - 0.05 * i) * frac[None, :] + 0.01 * rng.random((B, L + 1))
        out[f"wsr_{n}"][:, 0] = base
        out[f"time_{n}"] = (1e-4 * (1 + i)) * (1 + 0.1 * rng.random((B, L)))
    T = cfg.ao_max_iter
    traj = base[:, None] + 2.5 * (1 - np.exp(-0.08 * np.arange(T + 1)))[None, :]
    out["wsr_ao_traj"] = traj
    out["wsr_ao100"] = traj[:, -1].copy()
    out["n_iter_ao100"] = np.full(B, T)
    out["wsr_ao1000"] = traj[:, -1] + 0.2
    out["n_iter_ao1000"] = np.full(B, 400)
    out["ao100_mean"] = np.float64(out["wsr_ao100"].mean())
    out["ao1000_mean"] = np.float64(out["wsr_ao1000"].mean())
    out["ao_baseline_max_iter"] = np.int64(cfg.ao_max_iter)
    for k, mode in enumerate(cv.AO_MODES):
        step = (4e-3 if mode == "explicit" else 2e-3) * (1 + 0.1 * rng.random((B, 30)))
        out[f"ao_cumtime_{mode}"] = np.concatenate([np.zeros((B, 1)), np.cumsum(step, 1)], 1)
        out[f"ao_wsr_timed_{mode}"] = traj[:, :31]
        out[f"ao_niter_{mode}"] = np.full(B, 30)
    out.update(timing_idx=np.arange(B), timing_threads=np.int64(cfg.timing_threads), nets=np.array(NETS),
               **_meta(cfg, cv.EXP_KEYS))
    _save(cfg, "exp_01_convergence.npz", **out)

    # ---- runtime sweeps (exp_04 / exp_07)
    def runtime(tag, pts, split, sweep_key, stem, device="cpu", cdtype="complex128", nan=False):
        o = {}
        for x in pts:
            for m in sw.TIME_METHODS:
                scale = {"ao_explicit": 1e-5 * x ** 1.5, "ao_efficient": 3e-5 * x, "fonr": 2e-5 * x ** 0.5,
                         "maso": 1.8e-5 * x ** 0.5, "tias": 1e-5 * x ** 0.5}[m]
                o[f"time_{m}_{tag}{x}"] = scale * (1 + 0.1 * rng.random(B))
        if nan:
            o[f"time_ao_explicit_{tag}{pts[-1]}"][0] = np.nan
        o.update(points=np.array(pts), timing_idx=np.arange(B), timing_n_samples=np.int64(B),
                 timing_threads=np.int64(cfg.timing_threads), device=np.array(device), cdtype=np.array(cdtype))
        if stem is not None:
            o.update(_meta(cfg, sw.TIME_KEYS + (sweep_key,), sw._ds(split, pts)))
            _save(cfg, stem, **o)
        return o

    runtime("M", tuple(cfg.sweep_M), "sweep_M", "sweep_M", "exp_04_runtime_vs_M.npz")
    runtime("K", tuple(cfg.sweep_K), "sweep_K", "sweep_K", "exp_07_runtime_vs_K.npz")

    # ---- WSR sweeps (exp_05 / exp_06 / exp_08)
    def wsr(tag, pts, split, sweep_key, stem, extra=None):
        o = {}
        for x in pts:
            ao_w = 3 + rng.random(B)
            o[f"wsr_ao_{tag}{x}"] = ao_w
            for i, n in enumerate(NETS):
                o[f"wsr_{n}_{tag}{x}"] = ao_w * (0.995 + 0.004 * i)
            o[f"n_iter_ao_{tag}{x}"] = np.full(B, 50)
            o[f"wsr_init_{tag}{x}"] = ao_w / 2
        o.update(points=np.array(pts), ao_baseline_max_iter=np.int64(cfg.ao_max_iter),
                 **_meta(cfg, sw.WSR_KEYS + (sweep_key,), sw._ds(split, pts)))
        if extra:
            extra(o)
        _save(cfg, stem, **o)

    pts_p = tuple(cfg.sweep_crn_dbm)
    wsr("P", pts_p, "sweep_power", "sweep_crn_dbm", "exp_05_power_sweep.npz", sw._eta_post(pts_p))
    wsr("K", tuple(cfg.sweep_K), "sweep_K", "sweep_K", "exp_06_wsr_vs_K.npz")
    zp = sw.zeroshot_points(cfg)

    def zs(o):
        o["M_values"] = np.array(zp)
        o["M_reference"] = np.int64(cfg.M)

    wsr("M", zp, "sweep_M", "zeroshot_M", "exp_08_zeroshot_M.npz", zs)

    # ---- supplementary GPU timing
    if gpu:
        Ms, Ks = tuple(cfg.sweep_M), tuple(cfg.sweep_K)
        o = runtime("M", Ms, None, None, None, "cuda", "complex64", nan=True)
        o.update(runtime("K", Ks, None, None, None, "cuda", "complex64"))
        for i, n in enumerate(NETS):
            o[f"step_time_{n}"] = 1e-4 * (3 - i) * (1 + 0.1 * rng.random((B, L)))
        for mode in cv.AO_MODES:
            o[f"ao_cumtime_{mode}"] = 0.01 * (1 + rng.random(B))
            o[f"ao_niter_{mode}"] = np.full(B, 30)
            o[f"ao_steptime_{mode}"] = o[f"ao_cumtime_{mode}"] / 30
        o["ao_cumtime_explicit"][1] = np.nan
        files = sw._ds("sweep_M", Ms) + sw._ds("sweep_K", Ks) + ["datasets/test.pt"]
        o.update(timing_idx=np.arange(B), timing_n_samples=np.int64(B), timing_threads=np.int64(cfg.timing_threads),
                 device=np.array("cuda"), cdtype=np.array("complex64"), sweep_M=np.array(Ms), sweep_K=np.array(Ks),
                 gpu_name=np.array("Synthetic GPU"),
                 **_meta(cfg, sw.TIME_KEYS + ("sweep_M", "sweep_K", "ao_tol", "ao_max_iter"), files))
        _save(cfg, "exp_supp_gpu_timing.npz", **o)

    # ---- logs
    logs = os.path.join(cfg.root, "logs")
    os.makedirs(logs, exist_ok=True)
    rows = []
    for i, n in enumerate(NETS):
        r = {"model_name": n, "selected_lr": (0.1, 0.01, 0.001)[i], "best_epoch": 150 + i}
        for s in ("train", "val", "test"):
            r.update({f"{s}_loss": -1.0, f"{s}_wsr_mean": 4.0 + 0.01 * i, f"{s}_wsr_std": 0.5,
                      f"{s}_ao_wsr_mean": 4.02, f"{s}_efficiency_pct": 99.5 + 0.2 * i})
        r.update(test_gap_median=0.01, test_gap_95th_pctl=0.1, test_gap_min_left_tail=-0.5)
        rows.append(r)
    _write_csv(os.path.join(logs, "all_models_final_metrics.csv"), train.FINAL_COLS, rows)
    for i, n in enumerate(NETS):
        prow = [{"lr": lr, "epoch": ep, "stage": "supervised" if ep <= 2 else "unsupervised",
                 "val_wsr_mean": 3.5 + 0.1 * ep + (0.3 if j == i else 0.0)}
                for j, lr in enumerate((0.1, 0.01, 0.001)) for ep in (1, 2, 3)]
        _write_csv(os.path.join(logs, f"{n}_lr_pilot.csv"), train.PILOT_COLS, prow)
        with open(os.path.join(logs, f"{n}_lr_pilot.json"), "w", encoding="utf-8") as f:
            json.dump({"cfg_hash": "x", "selected_lr": (0.1, 0.01, 0.001)[i], "lr_grid": [0.1, 0.01, 0.001]}, f)
        hrow = [{c: 0 for c in train.HISTORY_COLS} | {"epoch": ep, "stage": "supervised", "n_skipped_batches": i}
                for ep in (1, 2)]
        _write_csv(os.path.join(logs, f"{n}_training_history.csv"), train.HISTORY_COLS, hrow)
    with open(os.path.join(logs, "sanity_check.json"), "w", encoding="utf-8") as f:
        json.dump({"ao_mean": 3.98, "power_only_mean": 2.214, "gain_pct": 79.7243125821807, "passed": True,
                   "init_mean": 1.457, "ao_mean_n_iter": 73.0, "n_samples": 1000, "split": "val"}, f)
    with open(os.path.join(logs, "fonr_mu_init.json"), "w", encoding="utf-8") as f:
        json.dump({"abc:mu-v2-qprobe": {
            "median_mu": 61.594, "median_lambda_max": 0.00812, "chosen_q": 0.99, "mu_init": 0.21938,
            "probe_failed": False, "per_q": {"0.5": {"mu": 61.594, "finite": False},
                                             "0.9": {"mu": 2.498, "finite": False},
                                             "0.99": {"mu": 0.21938, "finite": True}}}}, f)


@pytest.fixture()
def synth(tmp_path):
    cfg = _cfg(tmp_path)
    write_synthetic(cfg)
    return cfg


# ---------------------------------------------------------------- real producer summaries
def producer_summaries(cfg, monkeypatch):
    """Summaries computed by the REAL Task 12/13 functions on the synthetic npz files
    (caches are valid, so nothing is recomputed; figures are disabled)."""
    monkeypatch.setattr(sw, "_fig", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    res = os.path.join(cfg.root, "results")
    return {
        "convergence": cv.summarize_convergence(os.path.join(res, "exp_01_convergence.npz")),
        "runtime_M": sw.run_runtime_vs_M(cfg)[1],
        "power": sw.run_power_sweep(cfg)[1],
        "wsr_K": sw.run_wsr_vs_K(cfg)[1],
        "runtime_K": sw.run_runtime_vs_K(cfg)[1],
        "zeroshot_M": sw.run_zeroshot_M(cfg)[1],
        "gpu": sw.run_gpu_timing(cfg)[1],
    }


# ---------------------------------------------------------------- tests
def _claim(claims, cid):
    m = [c for c in claims if c.id == cid]
    assert len(m) == 1, cid
    return m[0]


def test_verdicts_on_synthetic():
    import report
    claims = report.make_claims(Config())
    # 1) 95th-pct gap ≈ 0.08 for FONR (brief's example tolerances)
    c = _claim(claims, "cdf_p95_fonr")
    k = "convergence:gap_p95_fonr"
    assert c.measured_key == (k,)
    assert c.verdict_fn({k: 0.09}) == report.CONFIRMED
    assert c.verdict_fn({k: 0.15}) == report.PARTLY
    assert c.verdict_fn({k: 0.5}) == report.CONTRADICTED
    # 2) over an order of magnitude speed-up at M = 180 (explicit AO)
    c = _claim(claims, "rtM_speedup_M180_ao_explicit")
    m = {"runtime_M:runtime_ms_ao_explicit_M180": 10.0, "runtime_M:runtime_ms_fonr_M180": 0.5,
         "runtime_M:runtime_ms_maso_M180": 0.4, "runtime_M:runtime_ms_tias_M180": 0.3}
    assert c.verdict_fn(m) == report.CONFIRMED
    m["runtime_M:runtime_ms_fonr_M180"] = 1.5  # ratio 6.7 for FONR only
    assert c.verdict_fn(m) == report.PARTLY
    m.update({f"runtime_M:runtime_ms_{n}_M180": 4.0 for n in NETS})  # ratio 2.5
    assert c.verdict_fn(m) == report.CONTRADICTED
    # 3) TIAS tracks AO across unseen M (|eta - 100| <= 1 at every M)
    c = _claim(claims, "zsM_tias_tracks_ao")
    keys = [f"zeroshot_M:eta_tias_M{M}_zeroshot" for M in (20, 40, 60, 80, 100, 120, 160)]
    assert set(c.measured_key) == set(keys)
    assert c.verdict_fn({k: 99.5 for k in keys}) == report.CONFIRMED
    assert c.verdict_fn({k: 97.0 for k in keys}) == report.PARTLY
    assert c.verdict_fn({**{k: 99.5 for k in keys}, keys[0]: 60.0}) == report.CONTRADICTED
    # 4) a "≈7.9-8.8 ms" claim exists for both AO modes
    for mode in ("explicit", "efficient"):
        c = _claim(claims, f"wall_ao_conv_{mode}")
        k = f"convergence:ao_conv_time_ms_{mode}"
        assert c.verdict_fn({k: 8.3}) == report.CONFIRMED
        assert c.verdict_fn({k: 20.0}) == report.PARTLY
        assert c.verdict_fn({k: 100.0}) == report.CONTRADICTED
    # every claim states its tolerance and a section/figure
    for c in claims:
        assert c.tolerance and c.section and c.figure and c.text, c.id


def test_approx_claims_sign_and_direction_tags():
    import report
    claims = report.make_claims(Config())

    def ev(cid, src, key, v):
        return report.evaluate_claim(_claim(claims, cid), {src: {key: v}})[1:]

    # p95 <= 0: the method beats AO on >= 95% of samples -> sign inverted, never "Partly"
    assert ev("cdf_p95_fonr", "convergence", "gap_p95_fonr", -0.3) == \
        (report.CONTRADICTED, "sign inverted: method beats AO")
    assert ev("cdf_p95_fonr", "convergence", "gap_p95_fonr", 0.0) == (report.PARTLY, "below claimed")
    assert ev("cdf_p95_fonr", "convergence", "gap_p95_fonr", 0.01) == (report.PARTLY, "below claimed")
    assert ev("cdf_p95_fonr", "convergence", "gap_p95_fonr", 0.09) == (report.CONFIRMED, "above claimed")
    assert ev("cdf_p95_fonr", "convergence", "gap_p95_fonr", 0.15) == (report.PARTLY, "above claimed")
    # MASO left tail ≈ -2
    assert ev("cdf_maso_left_tail_minus2", "convergence", "gap_min_maso", -10.0) == \
        (report.CONTRADICTED, "below claimed")
    assert ev("cdf_maso_left_tail_minus2", "convergence", "gap_min_maso", -2.5) == (report.CONFIRMED, "below claimed")
    assert ev("cdf_maso_left_tail_minus2", "convergence", "gap_min_maso", -4.0) == (report.PARTLY, "below claimed")
    assert ev("cdf_maso_left_tail_minus2", "convergence", "gap_min_maso", 0.3) == \
        (report.CONTRADICTED, "sign inverted: MASO never beats AO")
    # range claims carry a direction tag too
    assert ev("wall_ao_conv_explicit", "convergence", "ao_conv_time_ms_explicit", 20.0) == \
        (report.PARTLY, "above claimed")
    assert ev("wall_ao_conv_explicit", "convergence", "ao_conv_time_ms_explicit", 8.0) == \
        (report.CONFIRMED, "within claimed range")


def test_nan_error_and_not_run_verdicts():
    import report
    claims = report.make_claims(Config())
    # NaN measurement -> Not run (input unusable), not Contradicted
    m, v, _ = report.evaluate_claim(_claim(claims, "cdf_p95_fonr"), {"convergence": {"gap_p95_fonr": float("nan")}})
    assert v == report.NOT_RUN
    # ...except where NaN is a real outcome: "never reached 98%" is Contradicted
    _, v, _ = report.evaluate_claim(_claim(claims, "wall_tias_98pct_085ms"),
                                    {"convergence": {"time_to_98pct_tias_ms": float("nan")}})
    assert v == report.NEVER_REACHED == "Contradicted (never reached)"
    _, v, _ = report.evaluate_claim(_claim(claims, "wall_tias_99pct_1ms"),
                                    {"convergence": {"time_to_99pct_tias_ms": float("nan")}})
    assert v == report.NEVER_REACHED
    # two-key time claim: NaN from the AO input is Not run; NaN target with a valid AO input is never reached
    c = _claim(claims, "wall_tias98_within_first_ao_iter_explicit")
    (s1, n1), (s2, n2) = [k.split(":", 1) for k in c.measured_key]
    nan = float("nan")
    assert report.evaluate_claim(c, {s1: {n1: 1.0}, s2: {n2: nan}})[1] == report.NOT_RUN
    S = {s1: {n1: nan}}
    S.setdefault(s2, {})[n2] = 1.0
    assert report.evaluate_claim(c, S)[1] == report.NEVER_REACHED
    # an exception inside verdict_fn is reported as an error, never as Contradicted
    c = _claim(claims, "rtM_maso_approx_fonr")
    S = {"runtime_M": {k.split(":", 1)[1]: 0.0 for k in c.measured_key}}
    _, v, _ = report.evaluate_claim(c, S)
    assert v == "Error: ZeroDivisionError"
    # K-sweep latency claim split: nets only (body) and all methods (Fig. 7 caption)
    _claim(claims, "K_tias_lowest_net")
    assert _claim(claims, "K_tias_lowest_all").figure == "Fig. 7 caption"


def test_report_and_producers_share_cache_spec(synth, monkeypatch):
    from experiments import sweeps
    monkeypatch.setattr(sw, "_fig", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runs = {"runtime_M": sw.run_runtime_vs_M, "power": sw.run_power_sweep, "wsr_K": sw.run_wsr_vs_K,
            "runtime_K": sw.run_runtime_vs_K, "zeroshot_M": sw.run_zeroshot_M, "gpu": sw.run_gpu_timing}
    for exp, fn in runs.items():
        p, s = fn(synth)
        assert os.path.samefile(p, sweeps.cache_spec(synth, exp)[0]), exp
        assert sweeps.is_cached(synth, exp), exp
        assert s == sweeps.summarize(synth, exp, sweeps.cache_spec(synth, exp)[0]), exp
    assert os.path.exists(cv.cache_spec(synth)[0]) and cv.is_cached(synth)
    import report
    import inspect
    src = inspect.getsource(report)
    for private in ("_summ_wsr", "_summ_runtime", "_step_times", "_ds("):
        assert private not in src, private


def test_provenance_flags_inconsistent_ao_cap(synth):
    import report
    md = report.build_report(synth).read_text(encoding="utf-8")
    assert "shared-config hash" in md and "Consistency: all sources share" in md
    p = os.path.join(synth.root, "results", "exp_06_wsr_vs_K.npz")
    z = dict(np.load(p))
    z["ao_baseline_max_iter"] = np.int64(1000)
    np.savez(p, **z)
    md = report.build_report(synth).read_text(encoding="utf-8")
    assert "WARNING: ao_baseline_max_iter differs between sources" in md


def test_every_claim_has_measurement_key_in_summaries(synth, monkeypatch):
    import report
    S = producer_summaries(synth, monkeypatch)
    S["final"] = report.final_summary(report.load_artifacts(synth)["final_rows"])
    claims = report.make_claims(synth)
    assert len(claims) >= 40
    for c in claims:
        for mk in c.measured_key + tuple(c.note_key):
            src, key = mk.split(":", 1)
            assert src in S, (c.id, mk)
            assert key in S[src], (c.id, mk)
        # verdicts are computable on real producer output
        assert c.verdict_fn(report.measure(c, S)) in (report.CONFIRMED, report.PARTLY, report.CONTRADICTED)
    # default-config claim list also references real keys of the default sweep grid
    assert any(c.id == "rtM_speedup_M180_ao_efficient" for c in report.CLAIMS)


def test_collected_summaries_match_producers(synth, monkeypatch):
    import report
    got, status = report.collect_summaries(synth)
    want = producer_summaries(synth, monkeypatch)
    for src, s in want.items():
        assert status[src] == "ok", src
        assert set(got[src]) == set(s), src
        for k, v in s.items():
            if isinstance(v, tuple):
                assert got[src][k] == pytest.approx(v)
            elif np.isfinite(v):
                assert got[src][k] == pytest.approx(v), (src, k)


def test_report_sections_present(synth):
    import report
    p = report.build_report(synth)
    assert os.path.normcase(os.path.abspath(p)).startswith(os.path.normcase(os.path.abspath(synth.paper_dir)))
    assert not os.path.normcase(os.path.abspath(p)).startswith(REAL_PAPER)
    assert p.name == "results_findings.md"
    md = p.read_text(encoding="utf-8")
    for h in report.SECTION_HEADINGS:
        assert h in md, h
    for sec in ("### Sec. III", "### Sec. IV", "### Sec. VI"):
        assert sec in md
    for c in report.make_claims(synth):
        assert f"`{c.id}`" in md, c.id
    for d in report.DEVIATIONS:
        assert d.title in md, d.id
    # numbers read from artifacts, not hard-coded
    assert "79.72" in md               # sanity-check gain from sanity_check.json
    assert "q = 0.99" in md            # FONR mu0 probe
    assert "AO-100" in md and "AO-1000" in md
    assert "Synthetic GPU" in md
    assert "LinAlgError" in md         # GPU explicit-AO failures are stated
    # both power statistics present
    assert "ratio of means" in md and "mean of ratios" in md


def test_ao_wording_follows_baseline(tmp_path):
    import report
    cfg = _cfg(tmp_path, ao_max_iter=1000)
    write_synthetic(cfg, gpu=False)
    md = report.build_report(cfg).read_text(encoding="utf-8")
    assert "baseline AO-1000" in md
    assert "baseline AO-100 " not in md and "baseline AO-100)" not in md


def test_numbers_come_from_artifacts(synth):
    import report
    path = os.path.join(synth.root, "logs", "sanity_check.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d["gain_pct"] = 12.3456
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    md = report.build_report(synth).read_text(encoding="utf-8")
    assert "12.35" in md and "79.72" not in md


def test_missing_inputs_say_not_run(tmp_path):
    import report
    cfg = _cfg(tmp_path)
    p = report.build_report(cfg)  # nothing exists: must not crash
    md = p.read_text(encoding="utf-8")
    for h in report.SECTION_HEADINGS:
        assert h in md
    assert "not run" in md.lower()
    assert report.NOT_RUN in md
    # GPU missing only
    cfg2 = _cfg(tmp_path / "b")
    write_synthetic(cfg2, gpu=False)
    md2 = report.build_report(cfg2).read_text(encoding="utf-8")
    assert "GPU timing: not run" in md2


def test_stale_results_are_not_used(synth):
    import report
    cfg = dataclasses.replace(synth, ao_tol=1e-6)  # changes every experiment's cfg_hash
    _, status = report.collect_summaries(cfg)
    assert status["convergence"] == "stale"
    assert status["power"] == "stale"


def test_text_suggestions_from_contradicted_claims(synth):
    import report
    S, _ = report.collect_summaries(synth)
    S["convergence"]["gap_p95_fonr"] = 5.0  # forces a Contradicted CDF claim
    md = report.render_report(synth, S, {k: "ok" for k in S}, report.load_artifacts(synth))
    sugg = md.split(report.SECTION_HEADINGS[-1], 1)[1]
    assert "cdf_p95_fonr" in sugg
    assert "Contradicted" in sugg
