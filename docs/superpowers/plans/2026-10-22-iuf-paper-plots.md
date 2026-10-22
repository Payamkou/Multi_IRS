# IUF Paper Simulations and Figures: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyTorch codebase plus a resumable master notebook. Together they implement FP-AO (explicit and efficient), FONR, MASO and TIAS with corrected mathematics, train them, run the eight Sec. VI experiments, write the eight figures into `Paper/`, and write an honest claim-by-claim findings report.

**Architecture:**
- One shared mathematical core (`fp_core.py`) is used by every method.
- Each method exposes the same "stepper" interface (init / one step), so convergence curves and timing treat all five methods identically.
- Every expensive stage (datasets, teacher labels, LR pilot, training, experiments) writes artifacts to disk and resumes from them.
- The notebook only orchestrates; it contains no algorithm code.

**Tech Stack:** Python 3.13 and PyTorch 2.8 (CUDA 12.6) in `E:\payam\env`, plus NumPy, SciPy, Matplotlib, Pandas, PyYAML, pytest, Jupyter. nbconvert/nbformat get installed in Task 1.

**Spec:** `E:\payam\docs\superpowers\specs\2026-10-22-paper-plots-design.md`. Read it fully before any task. Section numbers below ("spec §6.1") refer to it. Also read `E:\payam\Paper\main.tex` §II–VI.

## Global Constraints

- Python is `E:/payam/env/Scripts/python.exe`. Every command runs from `E:/payam/code`, and pytest is invoked as `E:/payam/env/Scripts/python.exe -m pytest`.
- Code goes in `E:/payam/code/`; artifacts go in `E:/payam/code/artifacts/{datasets,checkpoints,logs,results}/`; figures (PNG at 300 dpi plus a PDF copy) and `results_findings.md` go in `E:/payam/Paper/`.
- **Never edit `E:/payam/Paper/main.tex`.**
- **No hard-coding, rescaling, smoothing or post-hoc tweaking of results toward main.tex's numbers.**
- **The test set is never used to choose anything.**
- Geometry, seeds and hyperparameters are frozen before any neural model is compared.
- No Python loops over users K, elements M or IRSs N in any method's math. Loops over layers and iterations are allowed.
- Precision:
  - complex128 for AO, teacher labels, evaluation and CPU timing;
  - complex64 for GPU training;
  - complex64 for every method in the supplementary GPU timing.
- Internal units: noise power σ² = 1. Channels are scaled by 1/σ with σ² = 1e-12 W (−90 dBm). Powers are in W (20 dBm = 0.1 W, 10 dBm = 0.01 W).
- Defaults: N=2, M=80, K=6, L=10, T=3 (FONR), CRN 20 dBm with weights cycling (0.90, 0.80, 0.85), PMN 10 dBm with weights cycling (0.30, 0.20, 0.25), even k → CRN and odd k → PMN, Rician κ=10, NLOS +20 dB on direct links, 8 dB shadowing on every link, PL = 128.1 + 37.6·log10(d/1000).
- Geometry (metres): BS (0,0), user disk centre (100,0) with radius 20, IRS₁ (10,10), IRS₂ (10,−10).
- Data: 10⁴ train / 10³ val / 10³ test, plus 10³ per sweep point. AO stops when ΔR/R < 1e-5 or after 100 iterations.
- Training: Adam, batch 200, epochs 1–50 supervised and 51–200 unsupervised, LR pilot over {0.1, 0.01, 0.001} with 5 supervised + 15 unsupervised epochs, a = 0.5, ρ_l = l/55.
- **Git:** `E:/payam` is not a git repository. The commit steps apply only if the user approves running `git init` in `E:/payam` at the start of execution; otherwise skip every commit step.

## Review Focus

1. **An IRS with zero associated users** (common at small K): A_n = 0, λ_n = 0 and v_n = 0, so the MM step would set every phase to angle(0) = 0. Expected behaviour: that IRS's phases stay unchanged. Pinned by `test_mm_step_keeps_phases_when_irs_has_no_users` (Task 4).
2. **A zero-magnitude MM argument element** (v + (λI − A)φ)_m = 0. Expected: that element keeps its old phase rather than snapping to 0. Pinned by `test_mm_step_zero_argument_keeps_phase` (Task 4).
3. **A user with a vanishing channel or a vanishing Σ|β|²** in the power law. Expected: a finite p in [0, P_peak] and no NaN. Pinned by `test_power_update_tiny_channel_is_finite` (Task 3).
4. **Resuming training with a changed config** (for example a different lr grid or epoch count). Expected: a clear error rather than silently mixing runs. Pinned by `test_resume_refuses_mismatched_config` (Task 10).
5. **Test or sweep data overlapping training data** through seed reuse. Expected: every split and sweep point uses a distinct seed from the registry. Pinned by `test_seed_registry_unique` (Task 2).

---

## File Structure

```
code/
  pytest.ini                pythonpath = . ; testpaths = tests
  config.py                 Config dataclass, JSON/YAML I/O, seed registry, config hash
  channels.py               geometry, path loss, ULA steering, fading, association, ChannelBatch
  data.py                   dataset build/load/cache for default splits and sweep points
  fp_core.py                SINR/WSR, aux vars, power law, f3, v_n, A_n, diag(A_n), matrix-free gradient
  ao.py                     MM phase step (explicit/efficient), run_ao, run_power_only, AOStepper, sanity check
  teacher.py                teacher-label cache (AO on train/val/test)
  models/__init__.py        build_model(name, cfg, **kw)
  models/common.py          Trajectory, UnfoldedNet base (deterministic block + stepper API)
  models/fonr.py  models/maso.py  models/tias.py
  train.py                  losses, LR pilot, resumable training, CSV logs, final metrics
  timing.py                 device/precision-fair per-step timing
  plotting.py               IEEE style + one plot function per figure
  experiments/__init__.py
  experiments/convergence.py   figs 1, 2, 3
  experiments/sweeps.py        figs 4, 5, 6, 7, 8
  report.py                 claims registry, verdicts, results_findings.md
  make_notebook.py          writes master_pipeline.ipynb with nbformat
  master_pipeline.ipynb
  tests/test_config.py test_channels.py test_fp_core.py test_ao.py test_models.py
        test_teacher.py test_train.py test_timing.py test_experiments.py test_report.py
```

---

### Task 1: Scaffolding and Config

**Files:**
- Create: `code/pytest.ini`, `code/config.py`, `code/models/__init__.py` (empty for now), `code/experiments/__init__.py`
- Test: `code/tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass Config` with these fields and defaults:
    - `N=2, M=80, K=6, L=10, T=3`
    - `noise_dbm=-90.0, crn_dbm=20.0, pmn_offset_db=-10.0`
    - `crn_weights=(0.90,0.80,0.85), pmn_weights=(0.30,0.20,0.25)`
    - `kappa=10.0, nlos_db=20.0, shadow_db=8.0`
    - `bs=(0.,0.), disk_center=(100.,0.), disk_radius=20., irs_pos=((10.,10.),(10.,-10.))`
    - `n_train=10000, n_val=1000, n_test=1000`
    - `ao_tol=1e-5, ao_max_iter=100`
    - `n_sup_epochs=50, n_unsup_epochs=150, batch_size=200`
    - `lr_grid=(0.1,0.01,0.001), pilot_sup_epochs=5, pilot_unsup_epochs=15`
    - `loss_a=0.5`
    - `maso_M_train=80, tias_hidden=32`
    - `sweep_M=(20,40,60,80,100,120,140,160,180), sweep_K=(10,15,20,25,30,35,40,45,50,55,60), sweep_crn_dbm=(20,25,30,35,40,45), zeroshot_M=(20,40,60,80,100,120,160)`
    - `timing_threads=6, timing_warmup=3, timing_subset=None`
    - `train_device="cuda", eval_device="cpu", root="E:/payam/code/artifacts", paper_dir="E:/payam/Paper"`
  - `Config.to_json(path)`, `Config.from_json(path) -> Config`, `Config.to_yaml(path)`
  - `config_hash(cfg, keys: tuple[str,...]) -> str` (sha1 of the JSON of the selected fields)
  - `seed_for(split: str, point: int | float | None = None) -> int`. Registry:
    - train → 1001, val → 2001, test → 3001;
    - sweep_M → 40000 + M, sweep_K → 50000 + K, sweep_power → 60000 + int(dbm);
    - init_phases → seed + 7 (applied to the dataset's own seed);
    - model_init → 9001.

- [ ] **Step 1:** Install the notebook tooling, which is not in `env`.
  Run: `E:/payam/env/Scripts/python.exe -m pip install nbconvert nbformat nbclient`
  Expected: "Successfully installed …"
- [ ] **Step 2:** Write the failing tests:
  - `test_defaults_match_spec`: assert `Config().M == 80`, `.K == 6`, `.crn_weights == (0.90,0.80,0.85)`, `.irs_pos == ((10.,10.),(10.,-10.))`, `.sweep_crn_dbm == (20,25,30,35,40,45)`.
  - `test_json_roundtrip`: write to tmp_path, read back, assert equal.
  - `test_config_hash_changes_with_field`.
- [ ] **Step 3:** Run `E:/payam/env/Scripts/python.exe -m pytest tests/test_config.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 4:** Implement `config.py`. `pytest.ini` contains `[pytest]`, `pythonpath = .` and `testpaths = tests`.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit with message `feat: config and scaffolding`.

### Task 2: Channel Model, Association, Datasets

**Files:**
- Create: `code/channels.py`, `code/data.py`
- Test: `code/tests/test_channels.py`

**Interfaces:**
- Consumes: `Config`, `seed_for`.
- Produces:
  - `@dataclass ChannelBatch`:
    - `hd: (B,K) complex` — direct link, scaled by 1/σ;
    - `b: (B,N,K,M) complex` — b_{k,n} = diag(g_n) h*_{r,k,n}, with g scaled by 1/σ;
    - `x: (B,N,K) real {0,1}`;
    - `w: (B,K) real`;
    - `P: (B,K) real` (peak power in W);
    - `sigma2: float = 1.0`.

    Methods: `.to(device, cdtype) -> ChannelBatch` (real tensors become the matching real dtype), `.slice(idx) -> ChannelBatch`, `.B`, `.N`, `.K`, `.M`.
  - `path_loss_db(d_m: Tensor) -> Tensor`
  - `ula_steering(sin_psi: Tensor, M: int) -> Tensor`, where a_m = exp(jπ m sin ψ) for m = 0…M−1. The IRS ULA axis is parallel to the y-axis, so sin ψ = Δy/d, with Δ taken from the IRS to the other node.
  - `generate_channels(cfg, B, seed, M=None, K=None, crn_dbm=None) -> tuple[ChannelBatch, Tensor theta0 (B,N,M)]`, with:
    - θ0 ~ U[0, 2π) from `seed + 7`;
    - Rician: √PL_lin·(√(κ/(1+κ))·a + √(1/(1+κ))·CN(0,I));
    - the LoS of g_n fixed by the geometry, its NLOS part and shadowing drawn per sample;
    - users uniform in the disk (radius √U·R);
    - CRN/PMN assignment and weights per Global Constraints.
  - `associate(b) -> x`: one-hot of argmax_n ‖b_{k,n}‖.
  - `data.build_or_load(cfg, split: str, point=None) -> dict(ch=ChannelBatch, theta0=Tensor, meta=dict)`, cached at `artifacts/datasets/{split}[_{point}].pt` and stored as complex128.

- [ ] **Step 1:** Write the failing tests:
  - `test_path_loss_values`: `path_loss_db(100.) ≈ 90.5 ± 0.01`; `path_loss_db(1000.) == 128.1`.
  - `test_shapes_and_association`: for B=4, every user has exactly one x=1 and it sits at the argmax norm.
  - `test_node_types_k6`: `w[0] == [0.90,0.30,0.80,0.20,0.85,0.25]` and P = [0.1,0.01,0.1,0.01,0.1,0.01].
  - `test_min_user_irs_distance`: over 10⁴ sampled users, the distance to either IRS is greater than 60 m.
  - `test_cache_roundtrip`: a second `build_or_load` returns identical tensors and doesn't regenerate (check file mtime unchanged).
  - `test_seed_registry_unique`: every seed for train/val/test, all sweep points and their +7 init seeds is distinct.
  - `test_ula_unit_modulus`.
- [ ] **Step 2:** Run `…pytest tests/test_channels.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `channels.py` and `data.py`. Use `torch.Generator` seeded from `seed_for`. Generate everything on CPU in float64/complex128 and save with `torch.save`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: channel model and cached datasets`.

### Task 3: FP Mathematical Core (spec §6)

**Files:**
- Create: `code/fp_core.py`
- Test: `code/tests/test_fp_core.py` (spec §9 tests 1, 2, 4)

**Interfaces:**
- Consumes: `ChannelBatch`.
- Produces:
  - `effective_channel(ch, phi) -> H (B,K)`; phi is `(B,N,M)`. H = hd + Σ_n x_{n,k}·(b_{n,k}ᴴ φ_n).
  - `received_power(ch, p, H) -> D (B,)`
  - `sinr(p, H, D) -> gamma (B,K)`
  - `wsr(ch, p, phi, base: str = "2") -> (B,)` (base "2" or "e")
  - `@dataclass Aux(alpha, wt, eps, Xi, H, D)`, all `(B,K)` except Xi and D, which are `(B,)`.
  - `aux_vars(ch, p, phi) -> Aux`, with α = γ, w̃ = w(1+α), ε = √(w̃ p)·H/D, Ξ = Σ|ε|².
  - `power_update(ch, aux, p_cur) -> p_new`. Uses the spec §6 law. Every denominator is clamped with `+ 1e-300`, and the result is clamped to [0, P].
  - `surrogate_f3(ch, p, phi, wt, eps) -> (B,)`, implementing Σ 2Re{ε*√(w̃p)H} − Ξ·D, with Ξ = Σ|ε|² recomputed from the eps argument.
  - `obs_scalar(ch, p, phi, aux) -> c (B,K)`, with c = √(w̃ p)ε − Ξ p H(phi).
  - `grad_matrix_free(ch, p, phi, aux) -> g (B,N,M)`, with g = Σ_k x_{n,k} c_k b_{n,k}.
  - `excitation_v(ch, p, aux) -> v (B,N,M)`, with v = Σ_k x (√(w̃p)ε − Ξ p hd) b (**hd not conjugated**).
  - `curvature_A(ch, p, aux) -> A (B,N,M,M)`, built explicitly (for explicit AO and tests).
  - `diag_A(ch, p, aux) -> (B,N,M)`.
  - `deterministic_block(ch, p, phi) -> tuple[p_new, Aux]`. It computes aux from (p, phi), then updates p. The returned Aux holds α, w̃, ε and Ξ from the **input** state. Every later g/v/A call in that layer or iteration uses `(p_new, that Aux)`, with H recomputed from the current phi inside `obs_scalar`.

- [ ] **Step 1:** Write the failing tests (complex128 on CPU, random ChannelBatch from `generate_channels(cfg, 3, seed=123)` with M=16, K=6):
  - `test_gradient_matches_explicit`: `grad_matrix_free ≈ excitation_v − A@phi` (atol 1e-9, relative to the norm).
  - `test_gradient_matches_autograd`: let phi be a leaf tensor with `requires_grad`. With p, w̃ and ε held fixed (detached), take `f = surrogate_f3(...).sum()`. Then `phi.grad` equals `2 * grad_matrix_free`; assert that with rtol 1e-8. The factor of 2 is verified: for a real loss, torch returns 2·∂f/∂φ*. Checking |z|² gives z.grad = 2z.
  - `test_power_update_maximises_f2`: for each k, f2 restricted to p_k is 2√p_k √w̃_k Re{ε_k*H_k} − p_k|H_k|²Ξ. Check the closed form against a 20001-point grid on [0, P_k]; the grid maximum must not exceed it by more than 1e-9 relative.
  - `test_power_update_bounds` and `test_power_update_tiny_channel_is_finite`: set hd[0,0] = 0 and b[:, :, 0] = 0. Assert all finite and 0 ≤ p ≤ P.
  - `test_lagrangian_dual_stationary`: in natural log, ∂f₁/∂α at α=γ is 0 (via autograd), and f₁(γ) = ln-WSR.
  - `test_quadratic_transform_optimal_eps`: perturbing ε lowers f₃ for fixed (p, phi, w̃).
- [ ] **Step 2:** Run `…pytest tests/test_fp_core.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `fp_core.py` with einsum over (n,k,m), so there are no loops.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: FP math core with corrected v_n and matrix-free gradient`.

### Task 4: FP-AO (explicit and efficient), Power-Only FP, Sanity Check

**Files:**
- Create: `code/ao.py`
- Test: `code/tests/test_ao.py` (spec §9 tests 3, 8)

**Interfaces:**
- Consumes: Task 3 functions.
- Produces:
  - `mm_phase_step(ch, p, phi, aux, mode: str) -> phi_new`, with mode "explicit" or "efficient".
  - `lambda_max(ch, p, aux, mode) -> (B,N)`:
    - explicit: `eigvalsh(curvature_A)[..., -1]`;
    - efficient: the largest eigenvalue of the masked K×K Gram matrix G_n = Ξ·diag(√(x p))·B_nᴴB_n·diag(√(x p)), where B_n = b[:, n] has shape (K, M), so G = Ξ·(√(xp) ⊗ √(xp))·(b_n b_nᴴ).
  - `A_phi(ch, p, phi, aux, mode) -> (B,N,M)`:
    - explicit: `curvature_A @ phi`;
    - efficient: `excitation_v − grad_matrix_free`.
  - `@dataclass AOResult(p, phi, wsr_traj (B, max_iter+1) bits, padded with the final value after convergence; n_iter (B,) int)`
  - `run_ao(ch, p0, phi0, mode="efficient", tol=1e-5, max_iter=100) -> AOResult`. It freezes each sample once that sample converges.
  - `run_power_only(ch, p0, phi0, tol, max_iter) -> AOResult`, which applies the power law only.
  - `class AOStepper(mode)` with `init(ch, p0, phi0) -> state` and `step(i, ch, state) -> state` (one AO iteration); state = (p, phi).
  - `init_state(ch, theta0) -> (p0 = ch.P.clone(), phi0 = exp(jθ0))`.
  - `sanity_check(cfg) -> dict(ao_mean, power_only_mean, gain_pct, passed)`, run on the **val** split.

The MM step body (decided by the spec, written here because the guards are not implied by the signature):
```python
lam = lambda_max(ch, p, aux, mode)                         # (B,N)
arg = v + lam[..., None] * phi - A_phi(ch, p, phi, aux, mode)
has_users = ch.x.sum(-1) > 0                                # (B,N)
ok = (arg.abs() > 1e-300) & has_users[..., None]
phi_new = torch.where(ok, arg / arg.abs().clamp_min(1e-300), phi)
```

One AO iteration:
```python
p_new, aux = deterministic_block(ch, p, phi)
phi = mm_phase_step(ch, p_new, phi, aux, mode)
p = p_new
```

- [ ] **Step 1:** Write the failing tests (complex128):
  - `test_explicit_equals_efficient`: iterates agree to 1e-9 relative after 20 iterations at M=180, K=60, B=2, and also at M=16, K=6.
  - `test_f3_monotone_in_phase_step`: f₃ with fixed (p_new, aux) is non-decreasing across `mm_phase_step` (tolerance −1e-10 relative).
  - `test_wsr_monotone_over_iterations`: `wsr_traj` is non-decreasing (tolerance −1e-9 relative) for B=8.
  - `test_mm_step_keeps_phases_when_irs_has_no_users`: set x[:,1,:] = 0 and route all users to IRS 0. Assert phi[:,1] is unchanged.
  - `test_mm_step_zero_argument_keeps_phase`: craft a state where one argument element is exactly 0 (b[..., m] = 0 for all k and phi_m chosen so the λφ term is masked; simplest is to monkeypatch `A_phi` to return v + λφ at index m). Assert that element is unchanged.
  - `test_stopping_rule`: `n_iter <= 100`, and the relative change at stop is < 1e-5 unless n_iter == 100.
- [ ] **Step 2:** Run `…pytest tests/test_ao.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `ao.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run the spec §5.1 sanity check:
  `E:/payam/env/Scripts/python.exe -c "from config import Config; from ao import sanity_check; print(sanity_check(Config()))"`
  Expected: `passed: True` (gain_pct ≥ 10).
  - If it fails: STOP and report to the user before moving the geometry once (spec §5.1). Record the result in `artifacts/logs/sanity_check.json`.
- [ ] **Step 6:** Commit with message `feat: FP-AO with MM phase step, explicit and efficient`.

### Task 5: Unfolded Base and FONR (spec §6.2, §6.5)

**Files:**
- Create: `code/models/common.py`, `code/models/fonr.py`; modify `code/models/__init__.py`
- Test: `code/tests/test_models.py` (FONR part, plus shared feasibility and size tests)

**Interfaces:**
- Consumes: `deterministic_block`, `grad_matrix_free`, `init_state`, `lambda_max`.
- Produces:
  - `@dataclass Trajectory(p: (L+1,B,K), phi: (L+1,B,N,M), alpha_in: (L,B,K))`; index 0 is the initial state.
  - `class UnfoldedNet(nn.Module)`:
    - abstract `phase_update(l, ch, p_new, phi, aux) -> phi`;
    - `forward(ch, p0, phi0) -> Trajectory`, where layer l runs `p_new, aux = deterministic_block(ch, p, phi)`, then `phi = phase_update(l, …)`, and records `aux.alpha` in `alpha_in[l]`;
    - stepper API `init(ch,p0,phi0)` and `step(i, ch, state)`.
  - `class FONR(UnfoldedNet)(L=10, T=3, mu_init: float)` with `self.mu = nn.Parameter(torch.full((L,T), mu_init))` (real). The inner loop runs over t: recompute g with `grad_matrix_free(ch, p_new, phi, aux)`, then θ ← θ + μ[l,t]·2·Re{−j φ* g}, then φ = e^{jθ}.
  - `fonr_mu_init(cfg) -> float` = 1/(2·median λ_max) over the training set at the initial state after one deterministic block (efficient mode). The result is cached in `artifacts/logs/fonr_mu_init.json`.
  - `build_model(name: str, cfg, **kw) -> UnfoldedNet` for "fonr", "maso" and "tias".

- [ ] **Step 1:** Write the failing tests:
  - `test_fonr_param_count == 30`.
  - `test_feasibility_all_layers[fonr]`, parametrised over models (MASO and TIAS are added in Tasks 6–7): every layer has |φ| = 1 ± 1e-6 and 0 ≤ p ≤ P.
  - `test_size_handling[fonr]`: the same instance runs at (M,K) ∈ {(20,6), (180,6), (80,60)}.
  - `test_same_init_as_ao`: `Trajectory.phi[0]` equals `init_state` phi0.
  - `test_fonr_untrained_close_to_mm`: with μ = 1/(2λ) taken per sample, one inner step changes WSR in the same direction as the MM step on average (sign check only, B=64).
- [ ] **Step 2:** Run `…pytest tests/test_models.py -k fonr -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `common.py`, `fonr.py` and `build_model`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: unfolded base and FONR`.

### Task 6: MASO (spec §6.3)

**Files:**
- Create: `code/models/maso.py`
- Test: extend `code/tests/test_models.py`

**Interfaces:**
- Produces:
  - `class ComplexLinear(nn.Module)(in_f, out_f)` (W_re, W_im, b_re, b_im; dtype follows the input).
  - `complex_relu(z)` = ReLU(Re z) + j·ReLU(Im z).
  - `class MASO(UnfoldedNet)(L=10, N=2, M_train=80)` with `self.psi = nn.ModuleList` of L·N MLPs, each ComplexLinear(2M_train→2M_train) → crelu → ComplexLinear(2M_train→2M_train) → crelu → ComplexLinear(2M_train→M_train) → `.real` → Softplus.
  - `phase_update` does the following:
    1. g, then r = g − Re{g ⊙ φ*} ⊙ φ;
    2. ĝ = g/(‖g‖ + 1e-12), using the norm over the full M;
    3. build z: for M < M_train, zero-pad φ and ĝ to M_train each; for M > M_train, keep the first M_train elements of each; then z = cat([φ_pad, ĝ_pad]);
    4. λ = Ψ_{l,n}(z): keep λ[:M] when M ≤ M_train; when M > M_train, append zeros for elements M_train+1 … M;
    5. φ ← (φ + λ⊙r)/|φ + λ⊙r|.

    Run N MLPs per layer, one per IRS. Looping over the N=2 IRS modules is allowed, because modules differ per IRS (stacked weights are optional).

- [ ] **Step 1:** Write the failing tests:
  - `test_maso_padding_small_M`: at M=20 the output λ has shape (B,N,20). Build the padded inputs and assert the MLP receives zeros beyond index 20 in both halves.
  - `test_maso_truncation_large_M`: at M=160, elements 80…159 keep their phase exactly.
  - Add `maso` to the feasibility and size parametrisation (sizes (20,6), (160,6), (80,60)).
  - `test_maso_K_agnostic_input`: the input dimension is 2·M_train, whatever K is.
- [ ] **Step 2:** Run `…pytest tests/test_models.py -k maso -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `maso.py` and register it in `build_model`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: MASO (DeepFP-style preconditioner)`.

### Task 7: TIAS (spec §6.4)

**Files:**
- Create: `code/models/tias.py`
- Test: extend `code/tests/test_models.py` (spec §9 test 5)

**Interfaces:**
- Produces:
  - `tias_messages(ch, p, phi, aux) -> tuple[u1 (B,N,M) complex, u2 (B,N,M) real]`, with u1 = Σ_k x b c_k and u2 = Σ_k x Ξ p_k |b|², computed by einsum.
  - `tias_features(ch, u1, u2, phi) -> (B,N,M,4)` with the four features of spec §6.4. eps = 1e-12; feature 3 is standardised over the (N,M) nodes of each sample (std clamped ≥ 1e-12).
  - `class TIAS(UnfoldedNet)(L=10, hidden=32)` with one shared `nn.Sequential(Linear(4,32), ReLU, Linear(32,32), ReLU, Linear(32,1), Softplus)` reused for every layer and node. The update is Δ = s·u1/(u2 + δ) with δ = 1e-12·max(u2) per sample, then φ ← (φ + Δ)/|φ + Δ|.

- [ ] **Step 1:** Write the failing tests:
  - `test_tias_messages_equal_gradient_and_diag`: u1 ≈ `grad_matrix_free`, and u2 ≈ `diag_A` (rtol 1e-10). Include a case where IRS 1 has zero users; there u1 = u2 = 0 and phi is unchanged after `phase_update`.
  - `test_tias_param_count_independent_of_size` (1249 parameters whatever M, K and L are).
  - `test_tias_features_scale_invariant`: scaling `ch.hd` and `ch.b` by 10³ leaves the features unchanged to 1e-9.
  - Add `tias` to the feasibility and size parametrisation.
- [ ] **Step 2:** Run `…pytest tests/test_models.py -k tias -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `tias.py` and register it.
- [ ] **Step 4:** Run the full suite `…pytest -v`. Expected: every test from Tasks 1–7 passes. This is the spec §9 gate: **do not start Task 8 until it is 100% green.**
- [ ] **Step 5:** Commit with message `feat: TIAS bipartite message passing`.

### Task 8: Teacher Labels and All Datasets

**Files:**
- Create: `code/teacher.py`
- Test: `code/tests/test_teacher.py`

**Interfaces:**
- Consumes: `data.build_or_load`, `run_ao`, `init_state`.
- Produces:
  - `build_all_datasets(cfg) -> dict[str, path]`, covering the default train/val/test splits and every sweep point (sweep_M ∪ zeroshot_M, sweep_K, sweep_crn_dbm).
  - `build_teacher_labels(cfg, splits=("train","val","test"), device=cfg.eval_device, chunk=500) -> path`. It writes `artifacts/datasets/teacher_labels.pt` = `{split: dict(phi_star, p_star, wsr_ao, n_iter, wsr_traj)}`, computed with efficient AO in complex128 and processed in chunks. If the file exists with matching `config_hash`, it is skipped.
  - `load_teacher(cfg, split) -> dict`.

- [ ] **Step 1:** Write the failing tests:
  - `test_teacher_cached_and_consistent`: with a tiny cfg (n_train=16, n_val=8, n_test=8, tmp root), labels exist and |phi_star| = 1. A second call doesn't recompute (mtime unchanged). wsr_ao equals `wsr(ch, p_star, phi_star)` in bits.
  - `test_teacher_hash_mismatch_rebuilds`.
- [ ] **Step 2:** Run `…pytest tests/test_teacher.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `teacher.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Generate the real data in the background:
  `E:/payam/env/Scripts/python.exe -c "from config import Config; import teacher as t; c=Config(); t.build_all_datasets(c); t.build_teacher_labels(c)"`
  Expected: the files exist under `artifacts/datasets/`; print the mean AO WSR and n_iter for each split.
- [ ] **Step 6:** Commit with message `feat: dataset and teacher-label generation`.

### Task 9: Losses (spec §8)

**Files:**
- Create: `code/train.py` (loss section)
- Test: `code/tests/test_train.py` (loss tests)

**Interfaces:**
- Produces:
  - `loss_supervised(phi_L, phi_star) -> scalar` = mean over (B,N,M) of 1 − Re{φ^{(L)} conj(φ*)}.
  - `loss_unsup_layer(ch, p_l, phi_l, alpha_in_l, a=0.5) -> scalar` = mean_B[ −a·R_nats − (1−a)·f₃ ], computed as:
    ```python
    wt = ch.w * (1 + alpha_in_l.detach())
    H = effective_channel(ch, phi_l); D = received_power(ch, p_l, H)
    eps = torch.sqrt(wt * p_l) * H / D[:, None]          # grads flow through p_l, phi_l
    f3 = surrogate_f3(ch, p_l, phi_l, wt, eps)
    R = wsr(ch, p_l, phi_l, base="e")
    ```
  - `loss_deep_supervision(ch, traj, a=0.5) -> scalar` = Σ_{l=1..L} (l/55)·loss_unsup_layer(ch, traj.p[l], traj.phi[l], traj.alpha_in[l-1], a).

- [ ] **Step 1:** Write the failing tests:
  - `test_supervised_loss_zero_at_label` and `test_supervised_loss_two_at_antipode`.
  - `test_unsup_no_grad_through_alpha`: make alpha_in a leaf with `requires_grad`, compute the loss, and assert `alpha_in.grad is None`.
  - `test_unsup_nats`: with a=1 the loss equals −mean(ln-WSR).
  - `test_rho_weights_sum_to_one`.
- [ ] **Step 2:** Run `…pytest tests/test_train.py -k loss -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the loss section.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: hybrid losses with detached alpha`.

### Task 10: Resumable Training, LR Pilot, Logs, Final Metrics (spec §8)

**Files:**
- Modify: `code/train.py`
- Test: `code/tests/test_train.py` (training tests)

**Interfaces:**
- Consumes: `build_model`, `fonr_mu_init`, `load_teacher`, `data.build_or_load`, the losses.
- Produces:
  - `run_lr_pilot(name, cfg) -> float`. Each lr in `cfg.lr_grid` gets a fresh model (seed `seed_for("model_init")`) trained for 5 supervised + 15 unsupervised epochs; selection is by the final validation WSR mean. Writes `artifacts/logs/{name}_lr_pilot.csv` with columns lr, epoch, stage, val_wsr_mean. If the CSV exists, it returns the stored choice.
  - `train_model(name, cfg, resume=True, force_retrain=False, stop_after_epoch: int | None = None) -> Path(best.pt)`:
    - each epoch shuffles with `torch.Generator().manual_seed(seed_for("train") + epoch)`, so resuming is exact;
    - epochs 1…n_sup use `loss_supervised`, and later epochs use `loss_deep_supervision`;
    - checkpoints follow spec §8: last.pt every epoch, best.pt on a new best mean validation WSR, with exactly the keys listed there, `metrics_history` holding train and val only, and also `cfg_hash = config_hash(cfg, TRAIN_KEYS)`.
  - `TRAIN_KEYS = ("N","M","K","L","T","n_train","n_sup_epochs","n_unsup_epochs","batch_size","lr_grid","loss_a","maso_M_train","tias_hidden")`. Resuming with a different hash raises `ValueError("config mismatch")`.
  - `evaluate(model, ch, theta0, phi_star=None, a=0.5) -> dict(loss, wsr_mean, wsr_std, wsr_per_sample, phase_cos)`. It runs in complex128 on `cfg.eval_device` under `no_grad`.
  - `final_metrics(name, cfg) -> dict`. It loads best.pt, evaluates the train, val and test splits, adds the AO means from the teacher labels, computes efficiency_pct = 100·mean(R)/mean(R_AO) and the test gaps ΔR = R_AO − R (median, 95th percentile, min), then upserts a row into `artifacts/logs/all_models_final_metrics.csv` and writes `{name}_final_metrics.json`.
  - Per-epoch CSV `artifacts/logs/{name}_training_history.csv` with exactly the spec §8 columns:
    - `train_loss` and `train_wsr_*` are means over that epoch's batches;
    - `val_*` come from `evaluate` on val at the training precision.

- [ ] **Step 1:** Write the failing tests (tiny cfg: n_train=64, n_val=16, n_test=16, M=16, n_sup_epochs=2, n_unsup_epochs=2, batch 32, device cpu, lr_grid=(0.01,)):
  - `test_resume_is_exact`, for FONR and TIAS:
    - run A trains 4 epochs straight;
    - run B calls `stop_after_epoch=2` and then resumes;
    - assert the final model state_dicts are equal (atol 0) and the history CSVs are identical apart from `epoch_time_sec`.
  - `test_completed_training_not_rerun`: a second `train_model` call returns best.pt without changing last.pt's mtime.
  - `test_resume_refuses_mismatched_config`: change `batch_size` and assert `ValueError`.
  - `test_history_columns`: the header equals the spec §8 list.
  - `test_no_test_metrics_in_history`: no key in `metrics_history` starts with "test".
  - `test_final_metrics_row`: the CSV holds every spec §8 column, and test_gap_95th_pctl is finite.
- [ ] **Step 2:** Run `…pytest tests/test_train.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Training uses complex64 on `cfg.train_device`; evaluation for selection uses the same device and precision; `final_metrics` uses complex128.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: resumable training, LR pilot, CSV logs, final metrics`.

### Task 11: Fair Timing (spec §7.2)

**Files:**
- Create: `code/timing.py`
- Test: `code/tests/test_timing.py`

**Interfaces:**
- Consumes: the stepper API (`init` and `step`) shared by `AOStepper` and `UnfoldedNet`.
- Produces:
  - `time_steps(stepper, ch1, theta0_1, n_steps: int, device: str, cdtype, warmup: int = 3) -> np.ndarray (n_steps,)` gives the time in seconds per step for one sample.
    - Use `torch.no_grad()`, `time.perf_counter()`, and `torch.cuda.synchronize()` before and after each step on CUDA.
    - The warm-up runs the whole sequence `warmup` times first and is discarded.
  - `time_until_converged(ao_stepper, ch1, theta0_1, tol, max_iter, device, cdtype) -> tuple[np.ndarray step_times, np.ndarray wsr_per_step]`, for Fig. 2's AO curves.
  - `set_timing_threads(n)` calls `torch.set_num_threads(n)` and returns the value for logging.
  - `per_iteration_time(stepper, data, idx, device, cdtype, n_steps=10) -> np.ndarray (len(idx),)`, the mean of the first 10 step times per sample.

- [ ] **Step 1:** Write the failing tests:
  - `test_times_positive_and_shape`: CPU, FONR and AOStepper("efficient"), M=16.
  - `test_same_device_and_dtype_enforced`: passing a model whose parameters sit on another device raises.
  - `test_cuda_path_syncs` (skipped if no CUDA): it runs and returns finite values.
- [ ] **Step 2:** Run `…pytest tests/test_timing.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. `stepper.to(device, cdtype)` must also move the model parameters, so give `UnfoldedNet` a `.to_precision(cdtype)`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: device/precision-fair timing`.

### Task 12: Plot Style and Figures 1–3

**Files:**
- Create: `code/plotting.py`, `code/experiments/convergence.py`
- Test: `code/tests/test_experiments.py` (figures 1–3)

**Interfaces:**
- Consumes: `load_teacher`, `train_model` (paths), `build_model`, `AOStepper`, `timing`.
- Produces:
  - `plotting.STYLE`: the colour and marker per method, fixed across all figures:
    - "FP-AO" and "FP-AO (explicit)": black, marker "s";
    - "FP-AO (efficient)": grey, marker "D";
    - FONR: C0, "o";
    - MASO: C1, "^";
    - TIAS: C2, "v".

    `setup_ieee()` sets a width of 3.5 in, font size 8 and a grid. `save_fig(fig, name, paper_dir)` writes `{name}.png` at 300 dpi and `{name}.pdf`, and returns both paths.
  - `load_trained_models(cfg) -> dict[name, UnfoldedNet]`, which loads the best.pt checkpoints.
  - `run_convergence(cfg, force_rerun=False) -> Path("artifacts/results/exp_01_convergence.npz")`:
    - per test sample, the WSR at layers 0…10 for each model (complex128, batched);
    - the AO wsr_traj from the teacher test labels;
    - CPU batch-1 cumulative step times for every model and for both AO modes (AO runs until convergence), over `cfg.timing_subset` or all 1000;
    - plus a CSV summary.
  - `plot_fig1(npz) -> fig`: the mean ± 95% CI band (1.96·std/√n) against step 0…10, with AO's first 10 iterations and a dashed line at the converged AO mean. Saved as `WSR_vs_iteration`.
  - `plot_fig2(npz) -> fig`: mean WSR against mean cumulative time in ms, with a log x-axis allowed and both AO curves. Saved as `WSR_vs_wall_clock`.
  - `run_cdf(cfg, force_rerun=False) -> Path("exp_03_cdf.npz")` holding ΔR = R_AO − R_method(L=10) per sample.
  - `plot_fig3(npz) -> fig`. Saved as `CDF`.
  - `summarize_convergence(npz) -> dict` with:
    - the % of AO's steady-state gain recovered at layers 4, 6 and 8 (gain measured from WSR at step 0);
    - for each method, the time to reach 98% and 99% of AO's converged mean;
    - AO convergence time per mode;
    - the 95th-percentile gaps and left-tail minima.

- [ ] **Step 1:** Write the failing tests:
  - `test_exp_cache_reuse`: a tiny cfg with fake checkpoints (untrained models saved as best.pt); the second call loads the npz without recomputing (mtime).
  - `test_plots_written`: the PNG and PDF exist for figures 1–3 in a tmp `paper_dir`.
  - `test_summary_keys`.
- [ ] **Step 2:** Run `…pytest tests/test_experiments.py -k "fig1 or fig2 or fig3 or cache or summary" -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: convergence, wall-clock and CDF experiments`.

### Task 13: Sweep Experiments, Figures 4–8

**Files:**
- Create: `code/experiments/sweeps.py`
- Test: extend `code/tests/test_experiments.py`

**Interfaces:**
- Produces (each caches `artifacts/results/exp_{id}_{name}.npz/.csv`, takes `force_rerun`, and saves its figure through `save_fig`):
  - `run_runtime_vs_M(cfg, device="cpu", cdtype=complex128)` → exp_04, figure `avg_time_vs_Number_of_RIS_M`:
    - M ∈ `cfg.sweep_M`;
    - `per_iteration_time` for AO explicit, AO efficient, FONR (trained), TIAS (trained), and MASO **instantiated at that M with untrained weights** (`MASO(M_train=M)`);
    - log₁₀ y-axis.
  - `run_power_sweep(cfg)` → exp_05, figure `etha_vs_CRN_transmit_power`. CRN ∈ `cfg.sweep_crn_dbm`; AO is run fresh (efficient) on each set. η is stored both as the ratio of means and as the mean of ratios; the plot uses the ratio of means.
  - `run_wsr_vs_K(cfg)` → exp_06, figure `WSR_vs_Number_of_active_users_K`; mean WSR for all four methods.
  - `run_runtime_vs_K(cfg, device="cpu", cdtype=complex128)` → exp_07, figure `avg_time_vs_Number_of_active_users_K`, with both AO curves.
  - `run_zeroshot_M(cfg)` → exp_08, figure `WSR_vs_M_ZeroShot`. M ∈ `cfg.zeroshot_M` with the trained models: MASO pads or truncates per spec, and FONR/TIAS run natively.
  - `run_gpu_timing(cfg)`: re-runs exp_04, exp_07 and the step timings of exp_01 with `device="cuda"`, `cdtype=complex64` for all methods. Results go to `exp_supp_gpu_timing.npz/.csv`; there is no paper figure.
  - Each returns `(npz_path, summary_dict)`.

- [ ] **Step 1:** Write the failing tests (tiny cfg, sweeps reduced to 2 points via cfg overrides, untrained models):
  - `test_sweep_outputs_and_figures` for exp_04–08: the npz, CSV, PNG and PDF exist.
  - `test_power_eta_both_statistics`.
  - `test_runtime_has_both_ao_curves`.
  - `test_zeroshot_includes_M80`.
- [ ] **Step 2:** Run `…pytest tests/test_experiments.py -k sweep -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: sweep experiments and figures 4-8`.

### Task 14: Findings Report (spec §10)

**Files:**
- Create: `code/report.py`
- Test: `code/tests/test_report.py`

**Interfaces:**
- Consumes: every exp summary dict, `all_models_final_metrics.csv`, the LR pilot CSVs, `sanity_check.json`, `fonr_mu_init.json`, and the GPU timing.
- Produces:
  - `@dataclass Claim(id, section, figure, text, measured_key, verdict_fn)`. `verdict_fn(measured) -> "Confirmed" | "Partly confirmed" | "Contradicted"`, using explicit tolerances written in the claim. Example: "95th pct gap ≈ 0.08 bps/Hz for FONR" is Confirmed if |m − 0.08| ≤ 0.04, Partly confirmed if m ≤ 0.2, and otherwise Contradicted.
  - The `CLAIMS` list holds one entry per claim in spec §10, plus the "≈7.9–8.8 ms AO convergence" claim, for both AO modes.
  - `DEVIATIONS`: a static list of the spec deviations, with reasons:
    - h_d conjugation;
    - ln versus log₂;
    - singular A_n and the MM step;
    - explicit versus efficient AO;
    - FONR's real θ-step and the zero-shot-M consequence;
    - MASO's input;
    - the TIAS message scaling Ξp_k;
    - test-set size 10³;
    - geometry;
    - the sanity-check result;
    - LR choices;
    - complex64 training;
    - the timing subset, if used.
  - `build_report(cfg) -> Path("Paper/results_findings.md")`, with sections: claims table; deviations; train/val/test table; CPU versus GPU latency tables; LR pilot summary; power η both ways; per-section text suggestions generated from contradicted or partly confirmed claims plus deviations.

- [ ] **Step 1:** Write the failing tests:
  - `test_verdicts_on_synthetic`: hand-made summaries produce the expected verdicts for three claims.
  - `test_report_sections_present`: every heading exists in the generated markdown, built from synthetic inputs in tmp.
  - `test_every_claim_has_measurement_key_in_summaries`: the key names match what Tasks 12–13 return.
- [ ] **Step 2:** Run `…pytest tests/test_report.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: automated findings report`.

### Task 15: Master Notebook, Full Run, Final Verification

**Files:**
- Create: `code/make_notebook.py`, which generates `code/master_pipeline.ipynb`

**Interfaces:**
- Consumes everything above. The notebook cells follow spec §12:
  1. setup and flags: `RESUME=True`, `FORCE_RETRAIN=False`, `FORCE_RERUN=False`;
  2. `subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"])` with a pass/fail table, then `sanity_check`;
  3. `build_all_datasets` and `build_teacher_labels`;
  4–6. for each model in turn: `run_lr_pilot` → `train_model` → `final_metrics` → show the history CSV as a DataFrame plus inline train/val curves;
  7. master comparison table (including FP-AO rows from the teacher means);
  8–15. one experiment per cell (figs 1…8), displaying the figure and printing the summary dict;
  16. `run_gpu_timing` and `build_report`, then print the report path.

  Every cell starts with a markdown cell explaining what it does and which artifact it resumes from.

- [ ] **Step 1:** Implement `make_notebook.py` with `nbformat.v4`, then run `E:/payam/env/Scripts/python.exe make_notebook.py`. Expected: `master_pipeline.ipynb` is written with 16 code cells.
- [ ] **Step 2:** Smoke-run the notebook on the tiny cfg by setting the environment variable `IUF_SMOKE=1`, which Cell 1 reads to shrink sizes and use a temporary artifact root:
  `IUF_SMOKE=1 E:/payam/env/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --output smoke_run.ipynb --ExecutePreprocessor.timeout=-1 master_pipeline.ipynb`
  Expected: exit 0 with no cell errors.
- [ ] **Step 3:** Run the full pipeline **in the background** (expect hours):
  `E:/payam/env/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 master_pipeline.ipynb`
  If it is interrupted, re-run the same command; every stage resumes from its artifacts.
- [ ] **Step 4:** Verify the outputs.
  Run `ls E:/payam/Paper/*.png E:/payam/Paper/*.pdf E:/payam/Paper/results_findings.md`.
  Expected: all 8 names, each as both PNG and PDF:
  - WSR_vs_iteration
  - WSR_vs_wall_clock
  - CDF
  - avg_time_vs_Number_of_RIS_M
  - etha_vs_CRN_transmit_power
  - WSR_vs_Number_of_active_users_K
  - avg_time_vs_Number_of_active_users_K
  - WSR_vs_M_ZeroShot

  Also confirm the report exists, `git diff`/mtime show `main.tex` unchanged, and the notebook has saved outputs in every cell.
- [ ] **Step 5:** Read `results_findings.md` and spot-check three claims against the raw npz by hand. Report the headline findings to the user, including every Contradicted verdict, verbatim from the report.
- [ ] **Step 6:** Commit with message `feat: master notebook and full results`. Commit only the code and notebook, and only if git was approved; artifacts stay out of git.
