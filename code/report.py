"""Findings report (spec §10): ``{cfg.paper_dir}/results_findings.md``.

Compares every claim of main.tex Sec. VI with the measured value and lists every deviation from
main.tex (with its reason and evidence). Measured numbers are read from the artifacts at report
time; only explanatory text is static. Nothing here runs an experiment: summaries are rebuilt from
the cached result files of Tasks 8-13, and a missing (or stale) result file makes the claims that
need it "Not run" instead of crashing.

Inputs (all under ``cfg.root``)
-------------------------------
* ``results/exp_01_convergence.npz``  -> ``experiments.convergence.summarize_convergence``
* ``results/exp_04_runtime_vs_M.npz``, ``exp_05_power_sweep.npz``, ``exp_06_wsr_vs_K.npz``,
  ``exp_07_runtime_vs_K.npz``, ``exp_08_zeroshot_M.npz``, ``exp_supp_gpu_timing.npz`` -> the same
  summary dicts ``experiments.sweeps.run_*`` return (pinned by tests/test_report.py).
  A file whose cfg_hash / artifact fingerprint no longer match the config is reported as "stale"
  and not used.
* ``logs/all_models_final_metrics.csv``, ``logs/{model}_lr_pilot.csv`` (+ ``.json`` sidecar),
  ``logs/{model}_training_history.csv``, ``logs/sanity_check.json``, ``logs/fonr_mu_init.json``.

Claims
------
``Claim.measured_key`` is a tuple of ``"source:key"`` strings, source in SOURCES. ``verdict_fn``
gets ``{"source:key": value}`` and returns CONFIRMED / PARTLY / CONTRADICTED using the tolerance
written in ``Claim.tolerance`` (derived from the claim's own wording, never from the data).
``make_claims(cfg)`` builds the list for a config's sweep grid; ``CLAIMS`` is the default-config list.

Wording follows ``ao_baseline_max_iter``: the AO baseline is called "AO-<max_iter>"
(e.g. AO-100 under Table I's rule, AO-1000 if cfg.ao_max_iter = 1000).
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import atomic
from config import Config, config_hash

CONFIRMED, PARTLY, CONTRADICTED, NOT_RUN = "Confirmed", "Partly confirmed", "Contradicted", "Not run"
NEVER_REACHED = "Contradicted (never reached)"
NETS = ("fonr", "maso", "tias")
NET = {"fonr": "FONR", "maso": "MASO", "tias": "TIAS"}
AO_MODES = ("explicit", "efficient")
TIME_METHODS = ("ao_explicit", "ao_efficient") + NETS
METHOD_LABEL = {"ao": "FP-AO", "ao_explicit": "FP-AO (explicit)", "ao_efficient": "FP-AO (efficient)", **NET}
SOURCES = ("convergence", "runtime_M", "power", "wsr_K", "runtime_K", "zeroshot_M", "gpu", "final")
SRC_FILES = {
    "convergence": "exp_01_convergence.npz",
    "runtime_M": "exp_04_runtime_vs_M.npz",
    "power": "exp_05_power_sweep.npz",
    "wsr_K": "exp_06_wsr_vs_K.npz",
    "runtime_K": "exp_07_runtime_vs_K.npz",
    "zeroshot_M": "exp_08_zeroshot_M.npz",
    "gpu": "exp_supp_gpu_timing.npz",
}
HALF_DECADE = math.sqrt(10.0)

SECTION_HEADINGS = (
    "# Findings report: measured results vs. main.tex Sec. VI",
    "## 0. Inputs and provenance",
    "## 1. Claims vs. measured",
    "## 2. Deviations from main.tex and builder's calls",
    "## 3. Train / validation / test metrics",
    "## 4. Latency: CPU vs. GPU (supplementary GPU timing)",
    "## 5. Learning-rate pilot",
    "## 6. Power efficiency η (both statistics)",
    "## 7. Suggested text changes (not applied to main.tex)",
)


# ================================================================ claims
@dataclass
class Claim:
    id: str
    section: str
    figure: str
    text: str
    measured_key: tuple
    verdict_fn: Callable[[dict], str]
    tolerance: str = ""
    suggestion: str = ""          # may use {m} (measured, formatted) and {ao} (baseline label)
    note_key: tuple = ()          # shown next to the measurement, never used for the verdict
    text_section: str = "VI"      # main.tex section the suggested edit goes to
    tag_fn: Optional[Callable[[dict], str]] = None  # direction tag for approximate claims
    nan_ok: bool = False          # True: NaN is a real outcome (e.g. "never reached"), not a missing input
    never_key: str = ""           # with nan_ok: NaN in this key = target never reached; NaN elsewhere = Not run


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _v_approx(k, x, partly_mult=2.5):
    """'≈ x' (x ≠ 0): Confirmed if |m − x| ≤ 0.5|x|; Partly if m has the sign of x (m = 0 counts) and
    |m| ≤ partly_mult·|x| (i.e. 0 ≤ m/x < 0.5 or 1.5 < m/x ≤ partly_mult); otherwise Contradicted
    (an inverted sign is always Contradicted)."""
    def f(m):
        v = float(m[k])
        if abs(v - x) <= 0.5 * abs(x):
            return CONFIRMED
        r = v / x
        return PARTLY if 0 <= r <= partly_mult else CONTRADICTED
    return f


def _tag_point(k, x, inverted="sign inverted"):
    """Direction tag of an approximate claim '≈ x'."""
    def f(m):
        v = float(m[k])
        if math.isnan(v):
            return ""
        if v / x < 0:
            return inverted
        if abs(v - x) <= 1e-12 * max(1.0, abs(x)):
            return "as claimed"
        return "below claimed" if v < x else "above claimed"
    return f


def _tag_range(klo, khi, lo, hi):
    """Direction tag of an approximate range claim 'lo–hi' (klo/khi may be the same key)."""
    def f(m):
        a, b = float(m[klo]), float(m[khi])
        if math.isnan(a) or math.isnan(b):
            return ""
        if a >= lo and b <= hi:
            return "within claimed range"
        parts = []
        if a < lo:
            parts.append("below claimed")
        if b > hi:
            parts.append("above claimed")
        return " and ".join(parts)
    return f


def _v_time_within(k, x):
    """'reaches … within ≈ x ms': Confirmed if m ≤ 1.5x; Partly if m ≤ 3x; never reached (NaN) → Contradicted."""
    def f(m):
        v = float(m[k])
        if math.isnan(v):
            return CONTRADICTED
        return CONFIRMED if v <= 1.5 * x else PARTLY if v <= 3 * x else CONTRADICTED
    return f


def _v_range(k, lo, hi):
    """'≈ lo–hi': Confirmed if 0.5·lo ≤ m ≤ 1.5·hi; Partly if lo/3 ≤ m ≤ 3·hi."""
    def f(m):
        v = float(m[k])
        if math.isnan(v):
            return CONTRADICTED
        if 0.5 * lo <= v <= 1.5 * hi:
            return CONFIRMED
        return PARTLY if lo / 3 <= v <= 3 * hi else CONTRADICTED
    return f


def _v_pp(k, x, conf, partly):
    def f(m):
        v = float(m[k])
        if math.isnan(v):
            return CONTRADICTED
        d = abs(v - x)
        return CONFIRMED if d <= conf else PARTLY if d <= partly else CONTRADICTED
    return f


def _v_at_least(k, thr, margin=5.0):
    def f(m):
        v = float(m[k])
        if math.isnan(v):
            return CONTRADICTED
        return CONFIRMED if v >= thr else PARTLY if v >= thr - margin else CONTRADICTED
    return f


def _v_equal(k, x):
    def f(m):
        v = float(m[k])
        return CONFIRMED if math.isclose(v, x, rel_tol=1e-9) else CONTRADICTED
    return f


def _frac(oks) -> str:
    """Point-wise relation: Confirmed if it holds at every point, Partly at ≥ half, else Contradicted."""
    oks = list(oks)
    n = sum(bool(o) for o in oks)
    return CONFIRMED if n == len(oks) else PARTLY if 2 * n >= len(oks) else CONTRADICTED


def zeroshot_points_of(cfg: Config) -> tuple:
    return tuple(sorted({int(m) for m in cfg.zeroshot_M} | {int(cfg.M)}))


def make_claims(cfg: Config) -> list:
    """One Claim per quantitative/qualitative statement of main.tex Sec. VI (+ Table I / captions)."""
    C = []
    add = lambda **kw: C.append(Claim(**kw))  # noqa: E731
    Ms = tuple(int(x) for x in cfg.sweep_M)
    Ks = tuple(int(x) for x in cfg.sweep_K)
    Ps = tuple(int(x) for x in cfg.sweep_crn_dbm)
    M0 = int(cfg.M)
    Mbig = 180 if 180 in Ms else max(Ms)
    ZM = zeroshot_points_of(cfg)
    off = [m for m in ZM if m != M0]
    cv = lambda k: f"convergence:{k}"  # noqa: E731
    rtM = lambda m, M: f"runtime_M:runtime_ms_{m}_M{M}"  # noqa: E731
    rtK = lambda m, K: f"runtime_K:runtime_ms_{m}_K{K}"  # noqa: E731

    # ---------------- VI-A, Table I, captions
    for n in NETS:
        k = f"final:selected_lr_{n}"
        add(id=f"setup_lr_{n}", section="VI-A", figure="Table I",
            text=f"{NET[n]} is trained with Adam at a constant LR = 0.1", measured_key=(k,),
            verdict_fn=_v_equal(k, 0.1),
            tolerance="Confirmed iff the LR selected by the pre-registered validation pilot over {0.1, 0.01, 0.001} "
                      "is 0.1; otherwise Contradicted",
            suggestion=f"Sec. VI-A text and Table I: state the LR actually used for {NET[n]} ({{m}}) and that it was "
                       "selected per model by a short validation pilot over {{0.1, 0.01, 0.001}}.")
    k = cv("ao_baseline_max_iter")
    add(id="setup_ao_stop", section="VI-A", figure="Table I",
        text="FP-AO stops when ΔR/R < 1e-5 or after at most 100 iterations", measured_key=(k,),
        verdict_fn=_v_equal(k, 100.0),
        tolerance="Confirmed iff the reported baseline uses the 100-iteration cap; otherwise Contradicted",
        suggestion="Table I: change the FP-AO iteration cap to the one actually used ({m}).")
    k = cv("n_test")
    add(id="setup_fig1_n80", section="VI-B", figure="Fig. 1 caption",
        text="95% confidence regions over 80 independent test channel realizations", measured_key=(k,),
        verdict_fn=_v_equal(k, 80.0), tolerance="Confirmed iff the number of test samples used is 80",
        suggestion="Fig. 1 caption: replace '80 independent test channel realizations' by the test-set size "
                   "actually used ({m}; Table I's 10^3).")
    add(id="setup_fig3_n200", section="VI-B", figure="Fig. 3 caption",
        text="CDF evaluated over 200 independent test channel realizations", measured_key=(k,),
        verdict_fn=_v_equal(k, 200.0), tolerance="Confirmed iff the number of test samples used is 200",
        suggestion="Fig. 3 caption: replace '200 independent test channel realizations' by {m} (Table I's 10^3).")

    # ---------------- VI-B convergence (Fig. 1)
    rec = {4: ("approximately 90%", _v_pp, 90.0, "Confirmed if |m − 90| ≤ 5 pp; Partly if ≤ 15 pp"),
           6: ("exceeding 96%", _v_at_least, 96.0, "Confirmed if m ≥ 96%; Partly if m ≥ 91%"),
           8: ("surpassing 98%", _v_at_least, 98.0, "Confirmed if m ≥ 98%; Partly if m ≥ 93%")}
    for l, (phr, kind, x, tol) in rec.items():
        for n in NETS:
            k = cv(f"recovery_pct_layer{l}_{n}")
            fn = kind(k, x, 5.0, 15.0) if kind is _v_pp else kind(k, x)
            add(tag_fn=_tag_point(k, x) if kind is _v_pp else None,
                id=f"conv_recovery_l{l}_{n}", section="VI-B", figure="Fig. 1",
                text=f"{NET[n]} recovers {phr} of the steady-state WSR gain by layer {l}", measured_key=(k,),
                verdict_fn=fn,
                tolerance=f"recovery = 100·(R̄_{l} − R̄_0)/(R̄_AO − R̄_0) vs the converged AO baseline; {tol}",
                suggestion=f"Sec. VI-B: replace the layer-{l} recovery figure for {NET[n]} by the measured {{m}} % "
                           "(relative to the {ao} mean).")
    ks = tuple(cv(f"recovery_pct_layer8_{n}") for n in NETS + ("ao",))

    def flat(m, ks=ks):
        v = [float(m[k]) for k in ks]
        if any(math.isnan(x) for x in v):
            return CONTRADICTED
        s = max(v) - min(v)
        return CONFIRMED if s <= 2 else PARTLY if s <= 5 else CONTRADICTED
    add(id="conv_four_curves_indistinguishable", section="VI-B", figure="Fig. 1",
        text="after layer/iteration 8 all four curves (three nets and FP-AO) flatten and become "
             "statistically indistinguishable", measured_key=ks, verdict_fn=flat,
        tolerance="spread (max − min) of the step-8 recovery of FONR, MASO, TIAS and FP-AO iteration 8: "
                  "Confirmed if ≤ 2 pp; Partly if ≤ 5 pp",
        suggestion="Sec. VI-B: the four curves are not indistinguishable at step 8 (recoveries {m}); describe the "
                   "AO-iteration vs layer comparison as measured (AO needs many more iterations).")

    # ---------------- VI-B wall clock (Fig. 2)
    k98, k99 = cv("time_to_98pct_tias_ms"), cv("time_to_99pct_tias_ms")
    add(id="wall_tias_98pct_085ms", section="VI-B", figure="Fig. 2",
        text="TIAS reaches ≈98% of AO's converged WSR within ≈0.85 ms", measured_key=(k98,),
        verdict_fn=_v_time_within(k98, 0.85), nan_ok=True, never_key=k98, tag_fn=_tag_point(k98, 0.85),
        tolerance="mean cumulative CPU time at the first layer with mean WSR ≥ 0.98·AO mean: Confirmed if "
                  "≤ 1.5·0.85 ms; Partly if ≤ 3·0.85 ms; never reached → Contradicted",
        suggestion="Sec. VI-B: replace '≈0.85 ms' by the measured time to 98% ({m} ms; NaN = never reached at "
                   "L = 10) relative to the {ao} mean.")
    for mode in AO_MODES:
        ka = rtM(f"ao_{mode}", M0)

        def first(m, ka=ka):
            t, a = float(m[k98]), float(m[ka])
            if math.isnan(t) or math.isnan(a):
                return CONTRADICTED
            return CONFIRMED if t <= a else PARTLY if t <= 2 * a else CONTRADICTED
        add(id=f"wall_tias98_within_first_ao_iter_{mode}", section="VI-B", figure="Fig. 2",
            text=f"TIAS's 98% time is within the budget AO ({mode}) needs to complete its first phase-shift update",
            measured_key=(k98, ka), verdict_fn=first, nan_ok=True, never_key=k98,
            tolerance=f"AO first-update time taken as the mean per-iteration time of the first L iterations at "
                      f"M = {M0} (Fig. 4 data); Confirmed if t98(TIAS) ≤ t_iter(AO); Partly if ≤ 2·t_iter",
            suggestion=f"Sec. VI-B: compare TIAS's time to 98% with the measured AO ({mode}) per-iteration time "
                       "({m}).")
    add(id="wall_tias_99pct_1ms", section="VI-B", figure="Fig. 2",
        text="TIAS crosses 99% of AO's converged WSR by ≈1 ms", measured_key=(k99,),
        verdict_fn=_v_time_within(k99, 1.0), nan_ok=True, never_key=k99, tag_fn=_tag_point(k99, 1.0),
        tolerance="Confirmed if ≤ 1.5 ms; Partly if ≤ 3 ms; never reached → Contradicted",
        suggestion="Sec. VI-B: replace '≈1 ms' by the measured time to 99% ({m} ms; NaN = never reached at L = 10).")
    for mode in AO_MODES:
        kc = cv(f"ao_conv_time_ms_{mode}")

        def order(m, kc=kc):
            t, a = float(m[k99]), float(m[kc])
            if math.isnan(t) or math.isnan(a) or t <= 0:
                return CONTRADICTED
            r = a / t
            return CONFIRMED if r >= 5 else PARTLY if r > 1 else CONTRADICTED
        add(id=f"wall_tias99_order_before_ao_{mode}", section="VI-B", figure="Fig. 2",
            text=f"TIAS crosses 99% nearly an order of magnitude before AO ({mode}) converges",
            measured_key=(k99, kc), verdict_fn=order, nan_ok=True, never_key=k99,
            tolerance="ratio t_conv(AO)/t99(TIAS): 'nearly an order of magnitude' → Confirmed if ≥ 5; Partly if > 1",
            suggestion=f"Sec. VI-B: state the measured ratio of AO ({mode}) convergence time to TIAS's 99% time "
                       "({m}).")
    for mode in AO_MODES:
        kc = cv(f"ao_conv_time_ms_{mode}")
        add(id=f"wall_ao_conv_{mode}", section="VI-B", figure="Fig. 2",
            text=f"AO ({mode}) converges at ≈7.9–8.8 ms", measured_key=(kc,), verdict_fn=_v_range(kc, 7.9, 8.8),
            tag_fn=_tag_range(kc, kc, 7.9, 8.8),
            tolerance="mean CPU time to the stopping rule: Confirmed if within [0.5·7.9, 1.5·8.8] ms; Partly if "
                      "within [7.9/3, 3·8.8] ms",
            suggestion=f"Sec. VI-B: replace '≈7.9–8.8 ms' by the measured AO ({mode}) convergence time ({{m}} ms) "
                       "and say which AO implementation is timed.")
    for mode in AO_MODES:
        ka = rtM(f"ao_{mode}", M0)
        kl = tuple(cv(f"layer_time_ms_{n}") for n in NETS)

        def fullpass(m, ka=ka, kl=kl):
            a = float(m[ka])
            n_ok = sum(float(m[k]) < a for k in kl)
            return CONFIRMED if n_ok == len(kl) else PARTLY if n_ok else CONTRADICTED
        add(id=f"wall_fullpass_vs_one_ao_iter_{mode}", section="VI-B", figure="Fig. 2",
            text=f"the nets execute their entire L-layer forward pass in a fraction of one FP-AO ({mode}) step",
            measured_key=kl + (ka,), verdict_fn=fullpass,
            tolerance=f"total L-layer CPU time of each net vs AO's mean per-iteration time at M = {M0}: Confirmed if "
                      "all three are smaller; Partly if some; else Contradicted",
            suggestion=f"Sec. VI-B: the full forward passes are not all shorter than one AO ({mode}) step ({{m}}); "
                       "rephrase the 'fraction of a single FP-AO step' sentence.")

    # ---------------- VI-B CDF (Fig. 3)
    for n, x in (("fonr", 0.08), ("maso", 0.09), ("tias", 0.13)):
        k = cv(f"gap_p95_{n}")
        add(id=f"cdf_p95_{n}", section="VI-B", figure="Fig. 3",
            text=f"95th-percentile gap ≈ {x} bps/Hz for {NET[n]}", measured_key=(k,), verdict_fn=_v_approx(k, x),
            tag_fn=_tag_point(k, x, "sign inverted: method beats AO"),
            tolerance=f"Confirmed if |m − {x}| ≤ {0.5 * x:.3g}; Partly if 0 ≤ m < {0.5 * x:.3g} or "
                      f"{1.5 * x:.3g} < m ≤ {2.5 * x:.3g}; otherwise Contradicted (m < 0: sign inverted, the "
                      "method beats AO on ≥ 95% of samples)",
            note_key=(cv(f"gap_p95_{n}_vs1000"),),
            suggestion=f"Sec. VI-B: replace '{x} bps/Hz for {NET[n]}' by the measured 95th-percentile gap {{m}} "
                       "bps/Hz (vs {ao}; the AO-1000 reference value is listed in Sec. 1).")
    km = tuple(cv(f"gap_median_{n}") for n in NETS)
    kp = tuple(cv(f"gap_p95_{n}") for n in NETS)

    def conc(m, km=km, kp=kp):
        med = [abs(float(m[k])) for k in km]
        p95 = [float(m[k]) for k in kp]
        if all(v <= 0.05 for v in med) and all(v <= 0.2 for v in p95):
            return CONFIRMED
        return PARTLY if all(v <= 0.2 for v in med) else CONTRADICTED
    add(id="cdf_concentrated_at_zero", section="VI-B", figure="Fig. 3",
        text="all three gap distributions are tightly concentrated around ΔR ≈ 0 (within a small fraction of a bit)",
        measured_key=km + kp, verdict_fn=conc,
        tolerance="Confirmed if every |median gap| ≤ 0.05 and every 95th pct ≤ 0.2 bps/Hz; Partly if every "
                  "|median| ≤ 0.2 bps/Hz",
        note_key=tuple(cv(f"gap_median_{n}_vs1000") for n in NETS),
        suggestion="Sec. VI-B: describe the measured gap distribution (medians / 95th pct {m}) instead of "
                   "'tightly concentrated around 0'.")
    k = cv("gap_min_maso")

    add(id="cdf_maso_left_tail_minus2", section="VI-B", figure="Fig. 3",
        text="MASO's left tail reaches ΔR ≈ −2 bps/Hz", measured_key=(k,), verdict_fn=_v_approx(k, -2.0),
        tag_fn=_tag_point(k, -2.0, "sign inverted: MASO never beats AO"),
        tolerance="minimum gap: Confirmed if |m + 2| ≤ 1; Partly if −5 ≤ m < −3 or −1 < m ≤ 0; otherwise "
                  "Contradicted (m > 0: sign inverted)",
        note_key=(cv("gap_min_maso_vs1000"),),
        suggestion="Sec. VI-B: replace 'ΔR ≈ −2 bps/Hz' by MASO's measured minimum gap ({m} bps/Hz vs {ao}).")
    kmin = tuple(cv(f"gap_min_{n}") for n in NETS)

    def furthest(m, kmin=kmin):
        f, ma, t = (float(m[k]) for k in kmin)
        if ma <= min(f, t) - 0.5:
            return CONFIRMED
        return PARTLY if ma < min(f, t) else CONTRADICTED
    add(id="cdf_maso_tail_furthest", section="VI-B", figure="Fig. 3",
        text="the left tail extends considerably further for MASO than for FONR or TIAS", measured_key=kmin,
        verdict_fn=furthest,
        tolerance="Confirmed if min gap(MASO) ≤ min(gap_min FONR, TIAS) − 0.5 bps/Hz; Partly if it is merely the "
                  "smallest",
        note_key=tuple(cv(f"gap_min_{n}_vs1000") for n in NETS),
        suggestion="Sec. VI-B: the left-tail ordering differs from the text (minimum gaps {m}); rewrite the "
                   "'MASO escapes further' sentence accordingly.")

    # ---------------- VI-C runtime vs M (Fig. 4)
    keys_all = lambda methods, pts, f: tuple(f(m, x) for x in pts for m in methods)  # noqa: E731
    kM = keys_all(NETS, Ms, rtM)
    add(id="rtM_tias_lowest_net", section="VI-C", figure="Fig. 4",
        text="TIAS consistently records the lowest running time (t_TIAS < t_MASO, t_FONR)", measured_key=kM,
        verdict_fn=lambda m: _frac(float(m[rtM("tias", M)]) < min(float(m[rtM("fonr", M)]), float(m[rtM("maso", M)]))
                                   for M in Ms),
        tolerance="per M: t_TIAS < min(t_FONR, t_MASO); Confirmed at every M, Partly at ≥ half the M points",
        suggestion="Sec. VI-C: the per-layer ordering among the nets is not TIAS-lowest at every M ({m}); state "
                   "the measured ordering.")
    add(id="rtM_maso_approx_fonr", section="VI-C", figure="Fig. 4",
        text="t_MASO ≈ t_FONR", measured_key=kM,
        verdict_fn=lambda m: _frac(2 / 3 <= float(m[rtM("maso", M)]) / float(m[rtM("fonr", M)]) <= 1.5 for M in Ms),
        tolerance="per M: 2/3 ≤ t_MASO/t_FONR ≤ 1.5 (≈ read as within ±50%); Confirmed at every M, Partly at ≥ half",
        suggestion="Sec. VI-C: MASO and FONR per-layer times are not comparable at every M ({m}).")
    for mode in AO_MODES:
        ka = f"ao_{mode}"
        kk = keys_all(NETS + (ka,), Ms, rtM)
        add(id=f"rtM_nets_ll_ao_{mode}", section="VI-C", figure="Fig. 4",
            text=f"t_nets ≪ t_AO ({mode}) across M", measured_key=kk,
            verdict_fn=lambda m, ka=ka: _frac(float(m[rtM(ka, M)]) >= HALF_DECADE * max(float(m[rtM(n, M)]) for n in NETS)
                                              for M in Ms),
            tolerance="per M: t_AO ≥ √10 · max(t_nets) ('≪' read as at least half a decade on the log10 axis); "
                      "Confirmed at every M, Partly at ≥ half",
            suggestion=f"Sec. VI-C: against FP-AO ({mode}) the nets are not much faster at every M ({{m}}); "
                       "qualify '≪' and name the AO implementation.")
        lo, hi = min(Ms), max(Ms)
        kg = tuple(rtM(m, x) for m in NETS + (ka,) for x in (lo, hi))

        def steeper(m, ka=ka, lo=lo, hi=hi):
            g_ao = float(m[rtM(ka, hi)]) / float(m[rtM(ka, lo)])
            g_net = max(float(m[rtM(n, hi)]) / float(m[rtM(n, lo)]) for n in NETS)
            return CONFIRMED if g_ao >= 2 * g_net else PARTLY if g_ao > g_net else CONTRADICTED
        add(id=f"rtM_ao_steeper_growth_{mode}", section="VI-C", figure="Fig. 4",
            text=f"FP-AO ({mode}) latency grows steeply with M (cubic inversion) while the nets scale much more "
                 "gracefully", measured_key=kg, verdict_fn=steeper,
            tolerance=f"growth factor t(M={hi})/t(M={lo}): Confirmed if AO's ≥ 2 × the largest net's ('much more "
                      "gracefully'); Partly if merely larger",
            suggestion=f"Sec. VI-C: AO ({mode}) does not grow much faster with M than the nets ({{m}}); the efficient "
                       "AO has no O(M^3) step — revise the cubic-bottleneck sentence.")
    for mode in AO_MODES:
        ka = f"ao_{mode}"
        kk = tuple(rtM(m, Mbig) for m in (ka,) + NETS)

        def speed(m, ka=ka):
            a = float(m[rtM(ka, Mbig)])
            r = [a / float(m[rtM(n, Mbig)]) for n in NETS]
            if min(r) >= 10:
                return CONFIRMED
            return PARTLY if max(r) >= 10 or min(r) >= 5 else CONTRADICTED
        add(id=f"rtM_speedup_M{Mbig}_{ka}", section="VI-C", figure="Fig. 4",
            text=f"at M = {Mbig} the nets achieve over an order-of-magnitude speed-up vs FP-AO ({mode})",
            measured_key=kk, verdict_fn=speed,
            tolerance="per-iteration speed-up t_AO/t_net: Confirmed if ≥ 10 for all three nets; Partly if ≥ 10 for "
                      "at least one or ≥ 5 for all; else Contradicted",
            suggestion=f"Sec. VI-C: replace 'over an order-of-magnitude speedup' by the measured speed-ups vs AO "
                       f"({mode}) at M = {Mbig} ({{m}}).")
    add(id="rtM_fonr_slower_than_maso", section="VI-C", figure="Fig. 4",
        text="FONR's per-layer runtime is consistently, if modestly, higher than MASO's", measured_key=kM,
        verdict_fn=lambda m: (CONFIRMED if all(1 < float(m[rtM("fonr", M)]) / float(m[rtM("maso", M)]) <= 2 for M in Ms)
                              else PARTLY if 2 * sum(float(m[rtM("fonr", M)]) > float(m[rtM("maso", M)]) for M in Ms)
                              >= len(Ms) else CONTRADICTED),
        tolerance="per M: Confirmed if 1 < t_FONR/t_MASO ≤ 2 at every M ('consistently, modestly'); Partly if "
                  "t_FONR > t_MASO at ≥ half the M points",
        suggestion="Sec. VI-C: FONR is not consistently slower than MASO ({m}); drop or reverse the paragraph that "
                   "explains FONR's higher per-layer cost.")

    # ---------------- VI-D power (Fig. 5)
    for n, (lo, hi) in (("fonr", (99.7, 100.6)), ("maso", (99.4, 101.0)), ("tias", (98.9, 100.4))):
        kl, kh = f"power:eta_min_{n}", f"power:eta_max_{n}"

        def rng(m, kl=kl, kh=kh, lo=lo, hi=hi):
            a, b = float(m[kl]), float(m[kh])
            if a >= lo - 0.5 and b <= hi + 0.5:
                return CONFIRMED
            return PARTLY if a >= 98 and b <= 102 else CONTRADICTED
        add(id=f"pow_eta_range_{n}", section="VI-D", figure="Fig. 5",
            text=f"η_WSR stays within {lo}–{hi}% for {NET[n]} over 20–45 dBm", measured_key=(kl, kh), verdict_fn=rng,
            tag_fn=_tag_range(kl, kh, lo, hi),
            tolerance=f"η = ratio of means (spec §7): Confirmed if [min, max] ⊂ [{lo - 0.5:.1f}, {hi + 0.5:.1f}]; "
                      "Partly if ⊂ [98, 102] ('within about one percentage point', doubled)",
            suggestion=f"Sec. VI-D: replace '{lo}–{hi}% for {NET[n]}' by the measured range {{m}} (ratio of means vs "
                       "{ao}).")
    ke = tuple(f"power:eta_ratio_of_means_{n}_P{P}" for n in NETS for P in Ps)

    def onepp(m, ke=ke):
        d = max(abs(float(m[k]) - 100) for k in ke)
        return CONFIRMED if d <= 1.5 else PARTLY if d <= 3 else CONTRADICTED
    add(id="pow_within_one_pp", section="VI-D", figure="Fig. 5",
        text="all three nets track the benchmark's WSR to within about one percentage point at every power",
        measured_key=ke, verdict_fn=onepp,
        tolerance="max |η − 100| over nets and powers: Confirmed if ≤ 1.5 pp ('about one'); Partly if ≤ 3 pp",
        suggestion="Sec. VI-D: replace 'within about one percentage point' by the measured worst-case deviation "
                   "from 100% ({m}).")
    add(id="pow_points_above_100", section="VI-D", figure="Fig. 5",
        text="several (net, power) points exceed 100%", measured_key=ke,
        verdict_fn=lambda m, ke=ke: (lambda c: CONFIRMED if c >= 2 else PARTLY if c == 1 else CONTRADICTED)(
            sum(float(m[k]) > 100 for k in ke)),
        tolerance="count of η > 100%: Confirmed if ≥ 2 ('several'); Partly if exactly 1; Contradicted if 0",
        suggestion="Sec. VI-D: no (or only one) point exceeds 100% ({m}); remove the 'several points exceeding 100%' "
                   "remark.")

    # ---------------- VI-E K sweep (Figs. 6, 7)
    for n in NETS:
        kk = tuple(f"wsr_K:eta_{n}_K{K}" for K in Ks)
        add(id=f"K_indistinguishable_{n}", section="VI-E", figure="Fig. 6",
            text=f"{NET[n]} is visually indistinguishable from FP-AO across K = {min(Ks)}–{max(Ks)}, with no "
                 "measurable degradation at the largest K", measured_key=kk,
            verdict_fn=lambda m, kk=kk: (lambda d: CONFIRMED if d <= 1 else PARTLY if d <= 3 else CONTRADICTED)(
                max(abs(float(m[k]) - 100) for k in kk)),
            tolerance="η_K = 100·mean R_net/mean R_AO; Confirmed if |η − 100| ≤ 1 at every K (≈ line width); "
                      "Partly if ≤ 3",
            suggestion=f"Sec. VI-E: {NET[n]} deviates from FP-AO in the K sweep (η per K: {{m}}); replace "
                       "'visually indistinguishable … no measurable degradation' by the measured efficiency.")
    for mode in AO_MODES:
        ka = f"ao_{mode}"
        kk = keys_all(NETS + (ka,), Ks, rtK)
        add(id=f"K_ao_most_expensive_{mode}", section="VI-E", figure="Fig. 7",
            text=f"FP-AO ({mode}) remains the most computationally expensive method at every K", measured_key=kk,
            verdict_fn=lambda m, ka=ka: _frac(float(m[rtK(ka, K)]) > max(float(m[rtK(n, K)]) for n in NETS)
                                              for K in Ks),
            tolerance="per K: t_AO > max(t_nets); Confirmed at every K, Partly at ≥ half",
            suggestion=f"Sec. VI-E: FP-AO ({mode}) is not the slowest at every K ({{m}}).")
        kg = (rtK(ka, min(Ks)), rtK(ka, max(Ks)))
        add(id=f"K_ao_latency_grows_{mode}", section="VI-E", figure="Fig. 7",
            text=f"FP-AO ({mode}) latency grows noticeably as K increases", measured_key=kg,
            verdict_fn=lambda m, kg=kg: (lambda g: CONFIRMED if g >= 1.2 else PARTLY if g > 1.05 else CONTRADICTED)(
                float(m[kg[1]]) / float(m[kg[0]])),
            tolerance=f"growth t(K={max(Ks)})/t(K={min(Ks)}): Confirmed if ≥ 1.2 ('noticeably'); Partly if > 1.05",
            suggestion=f"Sec. VI-E: AO ({mode}) latency barely grows with K ({{m}}); revise the sentence.")
    kk = keys_all(NETS, Ks, rtK)
    add(id="K_tias_lowest_net", section="VI-E", figure="Fig. 7",
        text="among the unfolded schemes, TIAS achieves the lowest absolute latency (every K)", measured_key=kk,
        verdict_fn=lambda m: _frac(float(m[rtK("tias", K)]) < min(float(m[rtK(x, K)]) for x in ("fonr", "maso"))
                                   for K in Ks),
        tolerance="per K: t_TIAS < min(t_FONR, t_MASO); Confirmed at every K, Partly at ≥ half",
        suggestion="Sec. VI-E: TIAS is not the fastest unfolded scheme at every K ({m}); state the measured ordering.")
    kk = keys_all(TIME_METHODS, Ks, rtK)
    add(id="K_tias_lowest_all", section="VI-E", figure="Fig. 7 caption",
        text="TIAS maintains the lowest latency profile (of all methods, FP-AO included)", measured_key=kk,
        verdict_fn=lambda m: _frac(float(m[rtK("tias", K)]) < min(float(m[rtK(x, K)]) for x in TIME_METHODS if x != "tias")
                                   for K in Ks),
        tolerance="per K: t_TIAS < every other method (both AO modes included); Confirmed at every K, Partly at ≥ half",
        suggestion="Fig. 7 caption: TIAS does not have the lowest latency of all methods at every K ({m}).")
    kg = tuple(rtK(m, x) for m in ("tias", "ao_explicit", "ao_efficient") for x in (min(Ks), max(Ks)))

    def gentle(m):
        lo, hi = min(Ks), max(Ks)
        g = float(m[rtK("tias", hi)]) / float(m[rtK("tias", lo)])
        g_ao = min(float(m[rtK(a, hi)]) / float(m[rtK(a, lo)]) for a in ("ao_explicit", "ao_efficient"))
        lin = hi / lo
        return CONFIRMED if g <= lin and g <= g_ao else PARTLY if g <= lin else CONTRADICTED
    add(id="K_tias_gentle_growth", section="VI-E", figure="Fig. 7",
        text="TIAS's inference time grows gently with K (O(|E|), linear in the active edges)", measured_key=kg,
        verdict_fn=gentle,
        tolerance=f"growth t(K={max(Ks)})/t(K={min(Ks)}): Confirmed if ≤ {max(Ks)}/{min(Ks)} (at most linear) and "
                  "≤ both AO growth factors; Partly if only at most linear",
        suggestion="Sec. VI-E: TIAS's latency growth with K is {m}; revise the 'gentle growth' sentence.")

    # ---------------- VI-F zero-shot M (Fig. 8)
    for n in ("fonr", "maso"):
        kk = tuple(f"zeroshot_M:eta_{n}_M{M}_zeroshot" for M in off)

        def degrade(m, kk=kk):
            v = [float(m[k]) for k in kk]
            if 2 * sum(x <= 80 for x in v) >= len(v):
                return CONFIRMED
            return PARTLY if any(x < 95 for x in v) else CONTRADICTED
        add(id=f"zsM_massive_degradation_{n}", section="VI-F", figure="Fig. 8",
            text=f"{NET[n]} suffers massive WSR degradation as M deviates from the training M = {M0}",
            measured_key=kk, verdict_fn=degrade,
            tolerance="η_M = 100·mean R_net/mean R_AO at M ≠ M_train: Confirmed if η ≤ 80% ('massive', ≥ 20% loss) "
                      "at ≥ half of those M; Partly if η < 95% at some M; Contradicted if η ≥ 95% everywhere",
            suggestion=(f"Sec. VI-F: {NET[n]} does not degrade massively off M = {M0} (η per M: {{m}}). "
                        + ("FONR has no M-dependent parameters (30 real step sizes), so the 'rigid dimensional "
                           "constraints' argument does not apply to it; rewrite the paragraph."
                           if n == "fonr" else
                           "State the zero-pad/truncate rule actually used and the measured effect.")))
    kk = tuple(f"zeroshot_M:eta_tias_M{M}_zeroshot" for M in ZM)
    add(id="zsM_tias_tracks_ao", section="VI-F", figure="Fig. 8",
        text="TIAS maintains near-optimal performance across the sweep, tracking FP-AO flawlessly", measured_key=kk,
        verdict_fn=lambda m, kk=kk: (lambda d: CONFIRMED if d <= 1 else PARTLY if d <= 5 else CONTRADICTED)(
            max(abs(float(m[k]) - 100) for k in kk)),
        tolerance="|η − 100| at every M (training M included): Confirmed if ≤ 1 pp ('flawlessly'); Partly if ≤ 5 pp",
        suggestion="Sec. VI-F: replace 'tracking the classical FP-AO baseline flawlessly' by the measured "
                   "efficiency per M ({m}).")
    kk = tuple(f"zeroshot_M:eta_{n}_M{M}_zeroshot" for M in off for n in NETS)
    add(id="zsM_tias_beats_mlp", section="VI-F", figure="Fig. 8",
        text="TIAS generalises across unseen M better than the MLP-based FONR and MASO", measured_key=kk,
        verdict_fn=lambda m: _frac(float(m[f"zeroshot_M:eta_tias_M{M}_zeroshot"])
                                   >= max(float(m[f"zeroshot_M:eta_{n}_M{M}_zeroshot"]) for n in ("fonr", "maso"))
                                   for M in off),
        tolerance="per M ≠ M_train: η_TIAS ≥ max(η_FONR, η_MASO); Confirmed at every M, Partly at ≥ half",
        suggestion="Sec. VI-F: the 'structural supremacy' of TIAS is not supported at every M ({m}); soften the "
                   "opening sentence.")
    return C


CLAIMS = make_claims(Config())


def measure(claim: Claim, S: dict) -> Optional[dict]:
    """{"source:key": value} for the claim, or None if any input is missing."""
    out = {}
    for mk in claim.measured_key:
        src, key = mk.split(":", 1)
        if src not in S or S[src] is None or key not in S[src]:
            return None
        out[mk] = S[src][key]
    return out


def _has_nan(m: dict) -> bool:
    for v in m.values():
        vals = v if isinstance(v, (tuple, list)) else (v,)
        if any(isinstance(x, (float, np.floating)) and math.isnan(x) for x in vals):
            return True
    return False


def evaluate_claim(claim: Claim, S: dict) -> tuple:
    """(measured dict or None, verdict, direction tag). Missing input or NaN (unless ``nan_ok``) →
    NOT_RUN; an exception inside verdict_fn → "Error: <type>" (never silently Contradicted)."""
    m = measure(claim, S)
    if m is None or (not claim.nan_ok and _has_nan(m)):
        return m, NOT_RUN, ""
    if claim.nan_ok and claim.never_key and _has_nan(m):
        if _has_nan({k: v for k, v in m.items() if k != claim.never_key}):
            return m, NOT_RUN, ""         # NaN from a missing (e.g. AO) input: cannot judge
        return m, NEVER_REACHED, ""       # the target was never reached
    try:
        v = claim.verdict_fn(m)
    except Exception as e:  # noqa: BLE001 - reported, not hidden
        return m, f"Error: {type(e).__name__}", ""
    tag = ""
    if claim.tag_fn is not None:
        try:
            tag = claim.tag_fn(m)
        except Exception:  # noqa: BLE001
            tag = ""
    return m, v, tag


# ================================================================ inputs
def _load_npz(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _read_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _shared_hash(cfg_json: str) -> str:
    """Hash of the config fields every experiment must share (training + AO rule), from a stored cfg_json."""
    import train
    keys = train.TRAIN_KEYS + ("ao_tol", "ao_max_iter")
    return config_hash(Config.from_dict(json.loads(cfg_json)), keys)


def collect_summaries(cfg: Config, check: bool = True) -> tuple:
    """(summaries {source: dict}, status {source: "ok" | "not run" | "stale"}) from the cached result
    files; never runs an experiment. The cache rule (path, hash keys, fingerprint files) and the
    summary functions are the producers' own (``convergence.cache_spec`` / ``sweeps.cache_spec``,
    ``summarize_convergence`` / ``sweeps.summarize``), so they cannot drift.
    Extra sources: "ao_ref" (capped fractions), "meta" (strings, incl. per-source provenance)."""
    from experiments import convergence as cv
    from experiments import sweeps as sw
    from experiments.cache import artifact_fingerprint, check_cache

    S, status = {"meta": {"provenance": {}}}, {}
    for src in SRC_FILES:
        path, keys, files, extra = cv.cache_spec(cfg) if src == "convergence" else sw.cache_spec(cfg, src)
        if not os.path.exists(path):
            status[src] = "not run"
            continue
        z = _load_npz(path)
        prov = {"cfg_hash": str(z.get("cfg_hash", "")),
                "ao_baseline_max_iter": (int(z["ao_baseline_max_iter"]) if "ao_baseline_max_iter" in z else None)}
        try:
            prov["shared_hash"] = _shared_hash(str(z["cfg_json"]))
        except (KeyError, ValueError, TypeError):
            prov["shared_hash"] = None
        S["meta"]["provenance"][src] = prov
        if check and not check_cache(path, cfg, keys, artifact_fingerprint(cfg, extra=files), extra):
            status[src] = "stale"
            continue
        status[src] = "ok"
        if src == "convergence":
            S[src] = cv.summarize_convergence(z)
            cap = float(z["ao_baseline_max_iter"])
            S["ao_ref"] = {"frac_capped_baseline": float(np.mean(z["n_iter_ao100"] >= cap)),
                           "frac_capped_1000": float(np.mean(z["n_iter_ao1000"] >= 1000)),
                           "n_capped_baseline": float(np.sum(z["n_iter_ao100"] >= cap))}
        else:
            S[src] = sw.summarize(cfg, src, z)
        if src == "gpu":
            S["meta"]["gpu_name"] = str(z.get("gpu_name", "unknown GPU"))
    rows = load_artifacts(cfg)["final_rows"]
    if rows:
        S["final"] = final_summary(rows)
        status["final"] = "ok"
    else:
        status["final"] = "not run"
    return S, status


def final_summary(rows: list) -> dict:
    """{f"{column}_{model}": float} from all_models_final_metrics.csv rows."""
    s = {}
    for r in rows:
        n = r["model_name"]
        for c, v in r.items():
            if c == "model_name":
                continue
            try:
                s[f"{c}_{n}"] = float(v)
            except (TypeError, ValueError):
                pass
    return s


def load_artifacts(cfg: Config) -> dict:
    logs = os.path.join(cfg.root, "logs")
    p = lambda name: os.path.join(logs, name)  # noqa: E731
    art = {"sanity": None, "fonr_mu": None, "fonr_mu_note": "", "final_rows": [], "pilots": {},
           "pilot_meta": {}, "histories": {}}
    if os.path.exists(p("sanity_check.json")):
        art["sanity"] = _read_json(p("sanity_check.json"))
    if os.path.exists(p("fonr_mu_init.json")):
        cache = _read_json(p("fonr_mu_init.json"))
        try:
            from models.fonr import _MU_KEYS, _MU_VERSION
            key = config_hash(cfg, _MU_KEYS) + ":" + _MU_VERSION
        except Exception:  # pragma: no cover - import guard
            key = None
        if key in cache:
            art["fonr_mu"] = cache[key]
        elif cache:
            art["fonr_mu"] = list(cache.values())[-1]
            art["fonr_mu_note"] = " (no entry matches the current config hash; the last stored entry is shown)"
    if os.path.exists(p("all_models_final_metrics.csv")):
        art["final_rows"] = _read_csv(p("all_models_final_metrics.csv"))
    for n in NETS:
        if os.path.exists(p(f"{n}_lr_pilot.csv")):
            art["pilots"][n] = _read_csv(p(f"{n}_lr_pilot.csv"))
        if os.path.exists(p(f"{n}_lr_pilot.json")):
            art["pilot_meta"][n] = _read_json(p(f"{n}_lr_pilot.json"))
        if os.path.exists(p(f"{n}_training_history.csv")):
            art["histories"][n] = _read_csv(p(f"{n}_training_history.csv"))
    return art


# ================================================================ formatting helpers
def _fmt(v) -> str:
    if v is None:
        return "not run"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, str):
        return v
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4g}"


def _fmt_measured(m: Optional[dict], limit: int = 8) -> str:
    if m is None:
        return "not run"
    items = [f"{k.split(':', 1)[1]}={_fmt(v)}" for k, v in m.items()]
    if len(items) > limit:
        return "; ".join(items[:limit]) + f"; … (+{len(items) - limit} more)"
    return "; ".join(items)


def _pct(x):
    return None if x is None else 100.0 * x


def _table(header, rows) -> str:
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _g(S, src, key):
    s = S.get(src)
    return None if s is None else s.get(key)


def ao_label(S: dict, cfg: Config) -> str:
    for src in ("convergence", "power", "wsr_K", "zeroshot_M"):
        v = _g(S, src, "ao_baseline_max_iter")
        if v is not None:
            return f"AO-{int(v)}"
    return f"AO-{int(cfg.ao_max_iter)}"


# ================================================================ deviations
@dataclass
class Deviation:
    id: str
    title: str
    paper_ref: str
    section: str                     # III / IV / VI: where the suggested text edit goes
    change: str
    reason: str
    evidence: str = ""
    evidence_fn: Optional[Callable[[dict], str]] = None
    suggestion: str = ""


def _ev_ao_modes(ctx):
    S = ctx["S"]
    Ms = tuple(ctx["cfg"].sweep_M)
    if "runtime_M" not in S:
        return "Runtime-vs-M timing: not run."
    parts = []
    for M in (min(Ms), max(Ms)):
        e, f = _g(S, "runtime_M", f"runtime_ms_ao_explicit_M{M}"), _g(S, "runtime_M", f"runtime_ms_ao_efficient_M{M}")
        parts.append(f"M = {M}: explicit {_fmt(e)} ms vs efficient {_fmt(f)} ms per iteration")
    return "Measured (CPU, complex128, batch 1): " + "; ".join(parts) + "."


def _ev_ao_cap(ctx):
    S, ao = ctx["S"], ctx["ao"]
    if "convergence" not in S:
        return "Convergence experiment: not run."
    a, b = S["convergence"]["ao100_mean"], S["convergence"]["ao1000_mean"]
    r = S.get("ao_ref", {})
    return (f"Baseline in force for this report: {ao}. Measured on the test set: {ao} mean WSR {_fmt(a)} bits vs AO-1000 reference {_fmt(b)} bits "
            f"({_fmt(100 * (b - a) / a)} % higher); {_fmt(100 * r.get('frac_capped_baseline', float('nan')))} % of "
            f"test samples hit the {ao} cap, {_fmt(100 * r.get('frac_capped_1000', float('nan')))} % still hit the "
            "1000-iteration cap.")


def _ev_fonr_mu(ctx):
    e = ctx["art"]["fonr_mu"]
    if not e:
        return "logs/fonr_mu_init.json: not found (not run)."
    per = e.get("per_q", {})
    probes = ", ".join(f"q = {q}: μ = {_fmt(v.get('mu'))}, finite = {v.get('finite')}" for q, v in per.items())
    ratio = e["median_mu"] / e["mu_init"] if e.get("mu_init") else float("nan")
    return (f"From logs/fonr_mu_init.json{ctx['art']['fonr_mu_note']}: chosen q = {e.get('chosen_q')}, "
            f"μ0 = {_fmt(e.get('mu_init'))}; median-based μ = {_fmt(e.get('median_mu'))} (the MASO anchor), i.e. "
            f"the FONR start step is {_fmt(ratio)}× smaller than the MASO anchor. Probe: {probes}. "
            f"Probe failed: {e.get('probe_failed')}.")


def _ev_maso_scale(ctx):
    e = ctx["art"]["fonr_mu"]
    if not e:
        return "logs/fonr_mu_init.json: not found (not run)."
    mu = e.get("median_mu")
    return (f"From logs/fonr_mu_init.json: median-based μ = {_fmt(mu)}, so c = 2μ/ln 2 = {_fmt(2 * mu / math.log(2))} "
            "and the untrained step c·Softplus(0) = 2μ = 1/median(λ_max).")


def _ev_skips(ctx):
    h = ctx["art"]["histories"]
    if not h:
        return "Training histories: not found (not run)."
    parts = []
    for n in NETS:
        if n in h:
            tot = sum(int(float(r.get("n_skipped_batches") or 0)) for r in h[n])
            parts.append(f"{NET[n]} {tot} skipped batches over {len(h[n])} epochs")
        else:
            parts.append(f"{NET[n]} not run")
    return "From logs/{model}_training_history.csv: " + "; ".join(parts) + "."


def _ev_test_size(ctx):
    S = ctx["S"]
    n = _g(S, "convergence", "n_test")
    return f"Test samples actually used for Figs. 1 and 3: {_fmt(n)} (config n_test = {ctx['cfg'].n_test})."


def _ev_geometry(ctx):
    c = ctx["cfg"]
    return (f"Config: BS {c.bs}, user disk centre {c.disk_center} radius {c.disk_radius} m, IRS positions "
            f"{c.irs_pos} (metres).")


def _ev_sanity(ctx):
    s = ctx["art"]["sanity"]
    if not s:
        return "logs/sanity_check.json: not found (not run)."
    return (f"From logs/sanity_check.json ({s.get('split')} split, {s.get('n_samples')} samples): full FP-AO mean "
            f"WSR {_fmt(s.get('ao_mean'))} vs power-only FP {_fmt(s.get('power_only_mean'))} → gain "
            f"{float(s.get('gain_pct')):.2f} % (threshold 10 %), passed = {s.get('passed')}; geometry frozen.")


def _ev_lr(ctx):
    rows, meta = ctx["art"]["final_rows"], ctx["art"]["pilot_meta"]
    sel = {r["model_name"]: r.get("selected_lr") for r in rows}
    for n, m in meta.items():
        sel.setdefault(n, m.get("selected_lr"))
    if not sel:
        return "LR pilot / final metrics: not run."
    parts = []
    for n in NETS:
        v = sel.get(n)
        if v is None:
            parts.append(f"{NET[n]} not run")
        else:
            flag = "" if math.isclose(float(v), 0.1) else " (≠ Table I's 0.1: deviation)"
            parts.append(f"{NET[n]} LR = {_fmt(float(v))}{flag}")
    return "Selected: " + "; ".join(parts) + "."


def _ev_timing_subset(ctx):
    c, S = ctx["cfg"], ctx["S"]
    nt = _g(S, "convergence", "n_timing")
    if c.timing_subset is None and (nt is None or nt >= c.n_test):
        return f"Not used: every test sample was timed (n_timing = {_fmt(nt)})."
    return (f"Used: timing on the first {_fmt(nt if nt is not None else c.timing_subset)} test/sweep samples "
            "(identical subset for every method); all WSR values use the full sets.")


def _ev_timing(ctx):
    S = ctx["S"]
    th = _g(S, "convergence", "timing_threads") or _g(S, "runtime_M", "timing_threads")
    return (f"torch threads = {_fmt(th)}, warm-up runs discarded = {ctx['cfg'].timing_warmup}, CPU complex128, "
            "batch 1.")


def _ev_gpu_fail(ctx):
    g = ctx["S"].get("gpu")
    if not g:
        return "GPU timing: not run."
    fails = {k: v for k, v in g.items() if (k.startswith("runtime_n_failed_") or k.startswith("ao_conv_n_failed_"))
             and v > 0}
    if not fails:
        return "No GPU sample failed."
    return ("Failed (NaN, excluded from the GPU means): "
            + "; ".join(f"{k} = {int(v)}" for k, v in sorted(fails.items())) + ".")


def _ev_zs_fonr(ctx):
    z = ctx["S"].get("zeroshot_M")
    if not z:
        return "Zero-shot-M experiment: not run."
    pts = zeroshot_points_of(ctx["cfg"])
    return "Measured FONR η per M: " + ", ".join(f"M={M}: {_fmt(z.get(f'eta_fonr_M{M}_zeroshot'))} %" for M in pts) + "."


DEVIATIONS = [
    Deviation("hd_conj", "h_d conjugation in v_n and in the TIAS observation scalar",
              "Sec. III (v_n), Sec. IV (TIAS ψ / H_k)", "III",
              "The code uses h_{d,i} where main.tex writes h*_{d,i} (v_n = Σ_i x_{i,n}(√(w̃_i p_i) ε_i − Ξ p_i h_{d,i}) "
              "b_{i,n}); TIAS's observation scalar is H_k = h_{d,k} + ψ_k.",
              "Expanding |h_{d,i} + b_{i,n}^H φ_n|² gives 2Re{(h_{d,i} b_{i,n})^H φ_n}; the conjugate is a typo.",
              "Unit test (spec §9.1): the matrix-free gradient g_n = v_n − A_n φ_n equals the autograd Wirtinger "
              "gradient of f₃ only with h_{d,i}.",
              suggestion="Sec. III, v_n: replace h*_{d,i} by h_{d,i}; Sec. IV (TIAS): write H_k = h_{d,k} + ψ_k."),
    Deviation("safe_sqrt_sinr", "Numerically safe √(w̃p) and masked-interference SINR in fp_core",
              "Sec. III / IV (not specified; implementation)", "III",
              "√(w̃_k p_k) uses a safe square root with a finite (zero) gradient at p = 0, and the SINR interference "
              "term is computed by masking the user's own signal instead of D − p_k|H_k|².",
              "complex64 training produced NaN gradients from √ at p = 0, and the D − S subtraction loses precision "
              "(~1e-5 relative in complex64). Mathematically identical; applied to AO and all three nets.",
              "Build-time measurement (ledger, Task 9): AO outputs changed by ≤ 3e-15 (rounding level), so the "
              "teacher labels stayed valid."),
    Deviation("ln_log2", "Natural log in the Lagrangian dual / quadratic transforms",
              "Sec. III (f₁)", "III",
              "All transforms and surrogates use ln; reported WSR stays Σ w_k log₂(1+γ_k).",
              "The stationarity α* = γ of the Lagrangian dual transform (Shen–Yu Part II) holds only for ln; "
              "log₂ = ln/ln 2 has the same maximiser.",
              "Unit test (spec §9.4): ∂f₁/∂α = 0 at α = γ and f₁(γ) = ln-WSR.",
              suggestion="Sec. III: write f₁ with ln (or note that log₂ rescales by 1/ln 2 and α* = γ needs ln)."),
    Deviation("singular_An_mm", "Singular A_n: MM phase step instead of φ^unc = A_n⁻¹ v_n (user decision D1)",
              "Sec. III (structure-aware reflection optimisation); Sec. VI-B/C ('exact O(M³) matrix inversion')", "III",
              "AO's phase step is the MM / nonhomogeneous-bound closed form φ_n ← exp(j∠(v_n + (λ_max I − A_n)φ_n)). "
              "No inverse, pseudo-inverse or regularised solve anywhere.",
              "A_n = Ξ Σ_i x_{i,n} p_i b_{i,n} b_{i,n}^H has rank ≤ K_n ≪ M, so A_n⁻¹ does not exist. The MM step is "
              "exact on the unit circle, never lowers f₃ and has no tuning constant (DeepFP Lemma 1).",
              "Unit tests (spec §9.3): f₃ non-decreasing across every AO phase step on random channels.",
              suggestion="Sec. III: replace the φ^unc = A_n⁻¹v_n step by the MM step and note rank(A_n) ≤ K_n. "
                         "Sec. VI-B/C: AO does not perform a matrix inversion; its cubic cost comes only from an "
                         "explicit M×M eigen-decomposition (see the next item)."),
    Deviation("ao_explicit_efficient", "Two AO implementations timed: explicit vs efficient (user decision D2)",
              "Sec. III (complexity), Sec. VI-B/C/E (timing)", "III",
              "'FP-AO (explicit)' builds the M×M A_n and takes λ_max by an M×M eigen-decomposition "
              "(O(K M² + M³)); 'FP-AO (efficient)' computes A_nφ matrix-free and λ_max from a Gram matrix. Both give "
              "identical iterates (tested), so every WSR result is one AO curve; every timing figure shows both. "
              "Implementation note: the efficient Gram is a masked K×K matrix (O(K² M)), not the minimal K_n×K_n.",
              "main.tex's O(M³) claim is true only for the explicit form; an efficient AO is not cubic in M.",
              evidence_fn=_ev_ao_modes,
              suggestion="Sec. III complexity analysis: give both costs (explicit O(KM² + M³), efficient "
                         "O(K M + K² M)) and state which AO the timing figures use."),
    Deviation("ao_cap", "AO iteration cap (Table I: 100) for baseline and teacher; AO-1000 reference reported",
              "Table I; Sec. VI-B (gap definition)", "VI",
              "The reported FP-AO baseline and the teacher labels use Table I's rule (ΔR/R < 1e-5 or the iteration "
              "cap). An additional 1000-iteration AO reference is computed on the default test set and every gap is "
              "also reported against it (Sec. 1). User decision (2026-10-22): proceed under the Table I rule; the "
              "user decides the baseline after the full run (cfg.ao_max_iter is the switch, and the teacher labels "
              "are regenerated when it changes).",
              "Experiment design is binding (spec §3); hiding an under-converged baseline would be a strawman.",
              "Build-time measurement of the AO-100 (Table I rule) baseline, validation split, recorded in the "
              "build ledger before training (these numbers describe AO-100, whatever baseline this report uses): "
              "528/1000 samples stopped by the 100-iteration cap; 1000 iterations gave +6.5 % mean WSR; untrained "
              "TIAS (4.359) and MASO (4.306) already exceeded AO-100 (3.980) and AO-1000 (4.238).",
              evidence_fn=_ev_ao_cap,
              suggestion="Sec. VI-A/B: state that FP-AO with a 100-iteration cap is not converged on a large share of "
                         "samples, report the gap also against a long-run AO, and interpret ΔR < 0 accordingly."),
    Deviation("eps_order", "main.tex Algorithm 1 order kept (ε reused after the power update); c_k = 0 observation",
              "Sec. III (Algorithm 1), Sec. IV (all three nets)", "IV",
              "ε (and Ξ, H, D) come from the layer's input state and are reused after the power update, exactly as "
              "Algorithm 1. A refreshed-ε variant was implemented and reverted.",
              "Both orders are valid block ascents; the refreshed variant was slightly worse for AO (build-time "
              "validation mean 3.966 vs 3.980 bits; old order better on 67 % of samples).",
              "Observation for the authors (build-time measurement): with this order the observation scalar "
              "c_k = √(w̃_k p_new) ε_old − Ξ p_new H_k is analytically zero for every user whose power is not clipped "
              "at P_peak (median |c_k| ≈ 3e-20 vs 0.026 for clipped users; 69 % of users contribute nothing), so the "
              "phase gradient of AO, FONR, MASO and TIAS is driven only by power-clipped users.",
              suggestion="Sec. IV: mention that, under Algorithm 1's update order, only P_peak-clipped users contribute "
                         "to the phase gradient (c_k = 0 otherwise), or refresh ε before the phase step."),
    Deviation("fonr_step", "FONR: real step on θ (DUFP), not a complex scalar weighting; zero-shot-M consequence",
              "Sec. IV (FONR), Sec. VI-F", "IV",
              "θ_n ← θ_n + μ_{l,t}·2Re{−j φ_n* ⊙ g_n}, one real μ per (layer, inner step): L·T = 30 parameters, "
              "independent of K, M and N.",
              "DUFP's base-paper form and main.tex's own angle parameterisation (Sec. V-D). Consequence: FONR runs at "
              "any M without padding, so the 'rigid dimensional constraints' argument of Sec. VI-F cannot apply to it.",
              evidence_fn=_ev_zs_fonr,
              suggestion="Sec. IV: describe N_θ as a learned real step on θ (30 scalars). Sec. VI-F: FONR has no "
                         "M-dependent weights; remove it from the 'MLP dimensional bottleneck' argument."),
    Deviation("fonr_mu0", "FONR initial step μ0 from a pre-registered quantile probe",
              "Sec. IV (FONR initialisation, not specified)", "IV",
              "μ0 = 1/(2·Q_q(λ_max)) with q the smallest of {0.5, 0.9, 0.99, 1.0} for which a complex64 L=10, T=3 "
              "FONR backward on the first 1000 training samples is finite (train data only).",
              "Backprop through 30 fixed-scalar θ steps is chaotic on high-curvature samples (gradients up to ~1e64 "
              "with the median-based μ). The architecture is unchanged; the weakness is reported, not engineered away.",
              "Build-time measurement (ledger): untrained FONR at L = 10 reached 2.611 bits on validation, far below "
              "untrained MASO/TIAS, because its start step is much smaller than MASO's anchor.",
              evidence_fn=_ev_fonr_mu,
              suggestion="Sec. IV/VI-A: state how FONR's step sizes are initialised and that a curvature-quantile start "
                         "was needed for stable training."),
    Deviation("grad_clip", "Gradient clipping and non-finite batch skipping (all three nets)",
              "Sec. V / Table I (not specified)", "VI",
              "clip_grad_norm_(max_norm = 1.0) every step; a batch whose loss or gradient is non-finite is skipped "
              "(no optimizer step) and counted (extra CSV column n_skipped_batches).",
              "Needed for stable complex64 training of the unfolded θ updates (see FONR μ0); applied identically to "
              "FONR, MASO and TIAS.",
              evidence_fn=_ev_skips,
              suggestion="Sec. V / Table I: add gradient-norm clipping (1.0) and the skip rule to the training "
                         "protocol."),
    Deviation("maso_input", "MASO input: DeepFP's [φ_n, ĝ_n] (K-agnostic); zero-pad / truncate for unseen M",
              "Sec. IV (MASO) vs Sec. VI-E", "IV",
              "Ψ_{l,n} sees z_n = [φ_n, g_n/‖g_n‖] ∈ ℂ^{2M}. For M < 80 the input is zero-padded; for M > 80 only the "
              "first 80 elements get a step, elements 81…M keep their phase (λ = 0, literal truncation).",
              "Resolves main.tex's contradiction (Sec. IV: per-user b_{k,n} input, K-dependent; Sec. VI-E: a "
              "K-agnostic aggregate). The truncation rule drives MASO's zero-shot-M result.",
              suggestion="Sec. IV: describe MASO's input as [φ_n, normalised gradient] and state the pad/truncate "
                         "rule used for M ≠ 80."),
    Deviation("maso_step_scale", "MASO step scale c = 2μ_median/ln 2 (builder's call)",
              "Sec. IV (MASO)", "IV",
              "λ = c·Softplus(Ψ(z)) with a non-trainable buffer c, so the untrained step equals the MM/FONR scale "
              "1/median(λ_max).",
              "Without it, untrained MASO steps were ~150× smaller than MM/FONR (strawman risk). Because the inputs "
              "are scale-free, MASO cannot adapt its step to a sample's curvature.",
              evidence_fn=_ev_maso_scale,
              suggestion="Sec. IV: state the step scaling and that MASO's scale-free input cannot see per-sample "
                         "curvature."),
    Deviation("tias_message", "TIAS message carries Ξ·p_k scaling",
              "Sec. IV (TIAS message passing)", "IV",
              "m_{k→(n,m)} = [b_{k,n,m} c_k, Ξ p_k |b_{k,n,m}|²]; sum-pooling gives exactly g_n[m] and [A_n]_mm.",
              "main.tex's message omits Ξ p_k, which makes the pooled message differ from the gradient / curvature.",
              "Unit test (spec §9.5): u¹ = explicit gradient and u² = diag(A_n), including users with no IRS links.",
              suggestion="Sec. IV (TIAS): add the Ξ p_k factor to the curvature part of the message."),
    Deviation("tias_f3", "TIAS feature f3: active-node standardisation; eps caveat",
              "Sec. IV (TIAS node features, not specified)", "IV",
              "f3 = log(|u¹|/(u² + eps) + eps) is standardised over the sample's nodes that HAVE users (K_n > 0); "
              "f3 = 0 at user-less nodes; mean/std are detached.",
              "Otherwise the log(eps) = −27.6 values of empty IRSs (≈16.5 % of default samples) dominate the "
              "statistics. Caveat: eps = 1e-12 is not unit-free, so the features are scale-invariant only while "
              "|u¹|, u² ≫ eps in the σ-normalised units used here.",
              suggestion="Sec. IV (TIAS): list the scale-free node features and the per-sample standardisation."),
    Deviation("test_size", "One test-set size everywhere: 10³", "Fig. 1 caption (80), Fig. 3 caption (200), Table I",
              "VI", "All evaluations use Table I's 10³ test samples (and 10³ per sweep point).",
              "Table I fixes 10³; the captions' 80 and 200 contradict it.",
              evidence_fn=_ev_test_size,
              suggestion="Captions of Figs. 1 and 3: replace 80 / 200 by 10³."),
    Deviation("geometry", "Geometry (builder's call, fixed before any comparison)", "Sec. VI-A (not specified)",
              "VI", "BS, user disk and IRS coordinates as in the config (see evidence).",
              "main.tex does not place the BS or IRSs; the IRSs sit near the BS, outside the user disk, so the path-loss "
              "formula stays in range and cascaded and direct links are comparable.",
              evidence_fn=_ev_geometry,
              suggestion="Sec. VI-A: give the BS, IRS and user-disk coordinates."),
    Deviation("sanity", "The one allowed sanity check (IRS phases matter)", "Sec. VI-A (not in main.tex)", "VI",
              "Before training: full FP-AO vs power-only FP (phases frozen at their random start) on the validation "
              "set; pass if the gain is ≥ 10 %.",
              "Shows the geometry makes the IRS relevant without looking at the nets or the test set.",
              evidence_fn=_ev_sanity),
    Deviation("lr", "Learning-rate choice per model (validation pilot)", "Table I / Sec. VI-A (LR = 0.1)", "VI",
              "Equal pre-registered pilot (5 supervised + 15 unsupervised epochs) over {0.1, 0.01, 0.001}; best mean "
              "validation WSR wins; the test set is never used.",
              "Prevents an untuned-model strawman without favouring any model.",
              evidence_fn=_ev_lr,
              suggestion="Table I / Sec. VI-A: report the per-model LR (see Sec. 5)."),
    Deviation("complex64", "complex64 GPU training; complex128 evaluation", "Sec. VI-A (not specified)", "VI",
              "Training runs in complex64 on the GPU; AO, teacher labels, every evaluation and CPU timing use "
              "complex128.",
              "The GTX 1660 Ti runs FP64 at ~1/32 speed; checkpoints are evaluated in complex128.",
              suggestion="Sec. VI-A: state training and evaluation precision."),
    Deviation("timing_subset", "Timing subset", "Sec. VI (timing protocol)", "VI",
              "Spec §7.2 allows timing on a pre-declared subset (first 200 samples) only if timing all 10³ is too slow.",
              "Same subset for every method; WSR always on all samples.",
              evidence_fn=_ev_timing_subset),
    Deviation("timing_method", "Timing method", "Sec. VI-B/C/E (not specified)", "VI",
              "CPU, complex128, batch 1, torch.no_grad, fixed thread count, warm-up discarded, time.perf_counter. "
              "Per-iteration time (Figs. 4, 7) = time of the first L AO iterations / L neural layers, divided by L. "
              "Eager PyTorch for every method (no compile/custom kernels), so batch-1 times partly reflect "
              "per-operation overhead (op counts) as well as asymptotic cost. Fig. 2 combines the full-test-set mean "
              "WSR with the timing-subset mean times; times to 98/99 % are resolved at whole layers/iterations. "
              "AO is timed with the per-step stepper (explicit and efficient). The runtime-vs-M MASO is built "
              "untrained at each M (timing does not depend on weight values).",
              "A GPU at batch 1 measures launch overhead; CPU timing is the paper protocol (DeepFP does the same).",
              evidence_fn=_ev_timing,
              suggestion="Sec. VI-B/C: state device, precision, batch size and how per-iteration time is defined."),
    Deviation("cpu_matmul_bug", "torch CPU batched complex matmul workaround", "implementation (no paper text)",
              "VI",
              "Contractions over users use broadcast multiply-and-sum instead of einsum/matmul in every method.",
              "torch 2.8 CPU multithreaded batched complex matmul/einsum with a 1-row left operand returns wrong "
              "results (confirmed). Applied identically to all methods, so timing stays fair.",
              ""),
    Deviation("gpu_eigvalsh", "GPU explicit-AO eigvalsh failures (supplementary timing)",
              "supplementary GPU timing (no paper text)", "VI",
              "In GPU complex64 timing, a sample whose cuSOLVER eigvalsh raises LinAlgError (explicit AO, "
              "ill-conditioned A_n) is stored as NaN and excluded from the means; counts are reported. CPU runs never "
              "swallow the error.",
              "complex64 eigen-decomposition of a rank-deficient M×M matrix can fail on cuSOLVER.",
              evidence_fn=_ev_gpu_fail),
    Deviation("builder_calls", "Other builder's calls", "Sec. II / V / VI-A (not specified)", "VI",
              "User k even → CRN, odd → PMN (K = 6 gives 3 + 3 with exactly the listed weights); shadowing drawn "
              "independently per link; initial phases i.i.d. uniform and p⁽⁰⁾ = P_peak, shared by all four methods; "
              "channels normalised by σ (exact rescaling); η = ratio of means (mean of ratios also reported); loss "
              "weight a = 0.5 and deep-supervision weights ρ_l = l/Σj; MASO hidden width 2M with separate weights per "
              "layer and IRS; TIAS MLP 2×32.",
              "Gaps neither main.tex nor the base papers fill; fixed before any comparison (spec §3, §13).",
              ""),
]


# ================================================================ rendering
def _section_inputs(cfg, S, status, art) -> list:
    rows = [[src, SRC_FILES.get(src, "logs/all_models_final_metrics.csv"), status.get(src, "not run")]
            for src in SOURCES]
    rows.append(["sanity check", "logs/sanity_check.json", "ok" if art["sanity"] else "not run"])
    rows.append(["FONR μ0", "logs/fonr_mu_init.json", "ok" if art["fonr_mu"] else "not run"])
    for n in NETS:
        rows.append([f"{NET[n]} LR pilot", f"logs/{n}_lr_pilot.csv", "ok" if n in art["pilots"] else "not run"])
    prov = S.get("meta", {}).get("provenance", {})
    prow = [[src, (p["cfg_hash"] or "")[:12], _fmt(p["ao_baseline_max_iter"]),
             (p["shared_hash"] or "?")[:12]] for src, p in prov.items()]
    shared = {p["shared_hash"] for p in prov.values() if p["shared_hash"]}
    iters = {p["ao_baseline_max_iter"] for p in prov.values() if p["ao_baseline_max_iter"] is not None}
    flags = []
    if len(shared) > 1:
        flags.append("**WARNING: the result files were produced with different training/AO configurations "
                     "(shared-config hash differs between sources).**")
    if len(iters) > 1:
        flags.append(f"**WARNING: ao_baseline_max_iter differs between sources: {sorted(iters)}.**")
    out = [SECTION_HEADINGS[1], "",
           f"AO baseline: **{ao_label(S, cfg)}** (ΔR/R < {cfg.ao_tol:g} or {ao_label(S, cfg)[3:]} iterations). "
           "'stale' = the result file no longer matches the config/artifacts and is not used.", "",
           _table(["input", "file", "status"], rows), ""]
    if prow:
        out += ["Per-source provenance. cfg_hash covers each experiment's own key set, so it differs between "
                "experiments by design; the shared-config hash covers the fields every experiment must agree on "
                "(training config + AO rule).", "",
                _table(["source", "cfg_hash", "ao_baseline_max_iter", "shared-config hash"], prow), ""]
        out += (flags or ["Consistency: all sources share the same training/AO configuration and AO cap."]) + [""]
    return out


def _section_claims(cfg, S, ao) -> tuple:
    claims = make_claims(cfg)
    rows, results, counts = [], [], {CONFIRMED: 0, PARTLY: 0, CONTRADICTED: 0, NOT_RUN: 0, "Error": 0}
    for c in claims:
        m, v, tag = evaluate_claim(c, S)
        counts[CONTRADICTED if v == NEVER_REACHED else v if v in counts else "Error"] += 1
        note = ""
        if c.note_key and ao != "AO-1000":
            nm = {}
            for k in c.note_key:
                src, key = k.split(":", 1)
                nm[k] = _g(S, src, key)
            if all(x is not None for x in nm.values()):
                note = "vs AO-1000: " + _fmt_measured(nm)
        cell = f"**{v}**" + (f" ({tag})" if tag else "")
        rows.append([f"`{c.id}`", c.section, c.figure, c.text, c.tolerance, _fmt_measured(m), note, cell])
        results.append((c, m, v, tag))
    out = [SECTION_HEADINGS[2], "",
           f"Verdicts: {counts[CONFIRMED]} {CONFIRMED}, {counts[PARTLY]} {PARTLY}, {counts[CONTRADICTED]} "
           f"{CONTRADICTED}, {counts[NOT_RUN]} {NOT_RUN}, {counts['Error']} Error (of {len(claims)}). Tolerances are derived from each claim's "
           "own wording and were fixed before any result existed.", "",
           _table(["ID", "Sec.", "Fig.", "Claim (main.tex)", "Tolerance", f"Measured (baseline {ao})",
                   "Note", "Verdict"], rows), ""]
    conv = S.get("convergence")
    out += ["### AO baseline vs AO-1000 reference (default test set)", ""]
    if conv:
        r = S.get("ao_ref", {})
        rows2 = [[f"{ao} mean WSR (bits)", _fmt(conv["ao100_mean"])],
                 ["AO-1000 mean WSR (bits)", _fmt(conv["ao1000_mean"])],
                 [f"% of samples stopped by the {ao} cap", _fmt(_pct(r.get("frac_capped_baseline")))],
                 ["% of samples stopped by the 1000-iteration cap", _fmt(_pct(r.get("frac_capped_1000")))]]
        for n in NETS:
            for st in ("median", "p95", "min"):
                rows2.append([f"{NET[n]} gap {st} vs {ao} / vs AO-1000 (bps/Hz)",
                              f"{_fmt(conv.get(f'gap_{st}_{n}'))} / {_fmt(conv.get(f'gap_{st}_{n}_vs1000'))}"])
        out += [_table(["quantity", "value"], rows2), ""]
    else:
        out += ["Convergence experiment: not run.", ""]
    return out, results


def _section_deviations(ctx) -> list:
    out = [SECTION_HEADINGS[3], "",
           "Every place the code departs from main.tex, every user decision and every builder's call. "
           "'Build-time measurement' marks numbers recorded in the build ledger before training (not re-measured "
           "here); all other numbers are read from the artifacts now.", ""]
    for i, d in enumerate(DEVIATIONS, 1):
        ev = d.evidence
        if d.evidence_fn is not None:
            try:
                extra = d.evidence_fn(ctx)
            except Exception as e:  # never crash the report on a malformed artifact
                extra = f"(evidence unavailable: {type(e).__name__}: {e})"
            ev = (ev + " " + extra).strip()
        out += [f"### Dev. {i}: {d.title}", "", f"- **main.tex:** {d.paper_ref}", f"- **Change:** {d.change}",
                f"- **Reason:** {d.reason}"]
        if ev:
            out.append(f"- **Evidence:** {ev}")
        out.append("")
    return out


def _section_metrics(art) -> list:
    out = [SECTION_HEADINGS[4], ""]
    rows = art["final_rows"]
    if not rows:
        return out + ["Final metrics (logs/all_models_final_metrics.csv): not run.", ""]
    hdr = ["model", "LR", "best epoch", "split", "loss", "WSR mean ± std", "AO WSR mean", "efficiency %"]
    tab = []
    for r in rows:
        for s in ("train", "val", "test"):
            tab.append([NET.get(r["model_name"], r["model_name"]), _fmt(float(r["selected_lr"])), r["best_epoch"], s,
                        _fmt(float(r[f"{s}_loss"])),
                        f"{_fmt(float(r[f'{s}_wsr_mean']))} ± {_fmt(float(r[f'{s}_wsr_std']))}",
                        _fmt(float(r[f"{s}_ao_wsr_mean"])), _fmt(float(r[f"{s}_efficiency_pct"]))])
    out += [_table(hdr, tab), "", "Test-set gaps vs the teacher AO (bps/Hz):", ""]
    out += [_table(["model", "median", "95th pct", "min (left tail)"],
                   [[NET.get(r["model_name"], r["model_name"]), _fmt(float(r["test_gap_median"])),
                     _fmt(float(r["test_gap_95th_pctl"])), _fmt(float(r["test_gap_min_left_tail"]))] for r in rows]), ""]
    return out


def _ordering(vals: dict) -> str:
    ok = {k: v for k, v in vals.items() if v is not None and not math.isnan(v)}
    return " < ".join(METHOD_LABEL[k] for k in sorted(ok, key=ok.get))


def _section_latency(cfg, S) -> list:
    out = [SECTION_HEADINGS[5], ""]
    gpu = S.get("gpu")
    if gpu:
        out += [f"GPU: {S['meta'].get('gpu_name', 'unknown GPU')}, complex64 for every method; CPU: complex128. "
                "Means in ms.", ""]
    else:
        out += ["GPU timing: not run (no CUDA device or experiment not executed); only CPU columns are shown.", ""]
    changes = []
    for tag, src, pts, name in (("M", "runtime_M", tuple(cfg.sweep_M), "Per-iteration / per-layer time vs M"),
                                ("K", "runtime_K", tuple(cfg.sweep_K), "Per-iteration / per-layer time vs K")):
        out += [f"### {name}", ""]
        if src not in S and not gpu:
            out += [f"{name}: not run.", ""]
            continue
        rows = []
        for x in pts:
            row = [x]
            cpu_v, gpu_v = {}, {}
            for m in TIME_METHODS:
                c = _g(S, src, f"runtime_ms_{m}_{tag}{x}")
                g = gpu.get(f"runtime_ms_{m}_{tag}{x}") if gpu else None
                cpu_v[m], gpu_v[m] = c, g
                row += [_fmt(c), _fmt(g)]
            rows.append(row)
            if gpu and src in S:
                a, b = _ordering(cpu_v), _ordering(gpu_v)
                if a != b:
                    changes.append(f"{tag} = {x}: CPU {a}; GPU {b}")
        hdr = [tag] + [f"{METHOD_LABEL[m]} {d}" for m in TIME_METHODS for d in ("CPU", "GPU")]
        out += [_table(hdr, rows), ""]
    out += ["### Default setting (M = %d, K = %d): per step and to convergence" % (cfg.M, cfg.K), ""]
    conv = S.get("convergence")
    rows = []
    for n in NETS:
        c = None if not conv else conv[f"layer_time_ms_{n}"] / cfg.L
        rows.append([f"{NET[n]} per layer", _fmt(c), _fmt(gpu.get(f"step_time_ms_{n}") if gpu else None)])
    for mode in AO_MODES:
        c = None
        if conv and conv.get(f"ao_conv_iters_mean_{mode}"):
            c = conv[f"ao_conv_time_ms_{mode}"] / conv[f"ao_conv_iters_mean_{mode}"]
        rows.append([f"FP-AO ({mode}) per iteration", _fmt(c),
                     _fmt(gpu.get(f"step_time_ms_ao_{mode}") if gpu else None)])
        rows.append([f"FP-AO ({mode}) to convergence", _fmt(conv.get(f"ao_conv_time_ms_{mode}") if conv else None),
                     _fmt(gpu.get(f"ao_conv_time_ms_{mode}") if gpu else None)])
    out += [_table(["quantity (ms)", "CPU", "GPU"], rows), ""]
    if gpu:
        out += ["Method ordering on GPU vs CPU: " + ("unchanged at every point." if not changes else
                                                     "CHANGES at " + "; ".join(changes) + "."), ""]
        fails = {k: v for k, v in gpu.items() if ("n_failed" in k) and v > 0}
        if fails:
            out += ["GPU samples that raised LinAlgError (cuSOLVER complex64 eigvalsh, explicit AO) and were excluded "
                    "from the means: " + "; ".join(f"{k} = {int(v)}" for k, v in sorted(fails.items())) + ".", ""]
        else:
            out += ["No GPU sample raised LinAlgError.", ""]
    return out


def _section_pilot(art) -> list:
    out = [SECTION_HEADINGS[6], ""]
    if not art["pilots"]:
        return out + ["LR pilot: not run.", ""]
    rows = []
    for n in NETS:
        pr = art["pilots"].get(n)
        if not pr:
            rows.append([NET[n], "not run", "", "", ""])
            continue
        last = {}
        for r in pr:
            lr, ep = float(r["lr"]), int(r["epoch"])
            if lr not in last or ep > last[lr][0]:
                last[lr] = (ep, float(r["val_wsr_mean"]))
        chosen = art["pilot_meta"].get(n, {}).get("selected_lr")
        if chosen is None:
            chosen = max(last, key=lambda k: last[k][1] if math.isfinite(last[k][1]) else -math.inf)
        for lr, (ep, v) in last.items():
            flag = "selected" + ("" if math.isclose(float(chosen), 0.1) else " (≠ Table I's 0.1)") \
                if math.isclose(lr, float(chosen)) else ""
            rows.append([NET[n], _fmt(lr), ep, _fmt(v), flag])
    out += ["Final-epoch mean validation WSR (bits) of each pilot run; the best one is selected.", "",
            _table(["model", "LR", "last epoch", "val WSR", "choice"], rows), ""]
    return out


def _section_power(cfg, S, ao) -> list:
    out = [SECTION_HEADINGS[7], ""]
    p = S.get("power")
    if not p:
        return out + ["Power sweep: not run.", ""]
    out += [f"η ratio of means = 100·mean(R_net)/mean(R_{ao}) (plotted in Fig. 5); η mean of ratios = "
            f"100·mean(R_net/R_{ao}). Models trained at 20 dBm, not retrained.", ""]
    rows = []
    for P in cfg.sweep_crn_dbm:
        row = [P, _fmt(p.get(f"wsr_ao_P{P}")), _fmt(p.get(f"ao_n_iter_mean_P{P}"))]
        for n in NETS:
            row += [_fmt(p.get(f"eta_ratio_of_means_{n}_P{P}")), _fmt(p.get(f"eta_mean_of_ratios_{n}_P{P}"))]
        rows.append(row)
    hdr = ["CRN dBm", f"{ao} WSR", "AO iters"] + [f"{NET[n]} {s}" for n in NETS
                                                  for s in ("ratio of means %", "mean of ratios %")]
    out += [_table(hdr, rows), ""]
    return out


def _section_suggestions(results, ao) -> list:
    out = [SECTION_HEADINGS[8], "",
           "Generated from the Contradicted / Partly confirmed claims and from the deviations. The authors decide; "
           "main.tex is not edited.", ""]
    not_run = [c.id for c, _, v, _ in results if v == NOT_RUN]
    errors = [f"`{c.id}` ({v})" for c, _, v, _ in results if v.startswith("Error")]
    for sec in ("III", "IV", "VI"):
        out += [f"### Sec. {sec}", ""]
        items = []
        for c, m, v, tag in results:
            if c.text_section == sec and v in (CONTRADICTED, NEVER_REACHED, PARTLY):
                vt = v + (f", {tag}" if tag else "")
                items.append(f"- [`{c.id}`, {vt}] {c.figure}: \"{c.text}\" — measured {_fmt_measured(m)}. "
                             f"Suggested: {c.suggestion.format(m=_fmt_measured(m, 4), ao=ao)}")
        for d in DEVIATIONS:
            if d.section == sec and d.suggestion:
                items.append(f"- [deviation: {d.title}] {d.suggestion}")
        out += (items or ["- No change suggested by the measured results."]) + [""]
    if errors:
        out += ["Claims whose verdict raised an error (check the inputs): " + ", ".join(errors) + ".", ""]
    if not_run:
        out += ["Claims not evaluated (inputs not run or NaN): " + ", ".join(f"`{i}`" for i in not_run) + ".", ""]
    return out


def render_report(cfg: Config, S: dict, status: dict, art: dict) -> str:
    ao = ao_label(S, cfg)
    ctx = {"cfg": cfg, "S": S, "status": status, "art": art, "ao": ao}
    lines = [SECTION_HEADINGS[0], "",
             "Generated by `code/report.py` from the cached experiment results. Every number below is read from "
             "the artifacts; no value is adjusted toward main.tex.", ""]
    lines += _section_inputs(cfg, S, status, art)
    claims_md, results = _section_claims(cfg, S, ao)
    lines += claims_md
    lines += _section_deviations(ctx)
    lines += _section_metrics(art)
    lines += _section_latency(cfg, S)
    lines += _section_pilot(art)
    lines += _section_power(cfg, S, ao)
    lines += _section_suggestions(results, ao)
    return "\n".join(lines) + "\n"


def build_report(cfg: Config, summaries: Optional[dict] = None, out_path=None) -> Path:
    """Write ``{cfg.paper_dir}/results_findings.md`` (or ``out_path``) and return its path.

    ``summaries`` ({source: summary dict}, e.g. what the notebook got from the run_* calls) overrides the
    summaries rebuilt from the cached result files for the sources it contains."""
    S, status = collect_summaries(cfg)
    if summaries:
        for k, v in summaries.items():
            if v is not None:
                S[k] = v
                status[k] = "ok (passed in)"
    art = load_artifacts(cfg)
    md = render_report(cfg, S, status, art)
    path = Path(out_path) if out_path else Path(cfg.paper_dir) / "results_findings.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(md, encoding="utf-8")
    atomic.replace(tmp, path)
    return path
