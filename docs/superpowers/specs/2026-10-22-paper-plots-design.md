# Design: Real Simulations and Figures for the IUF Multi-IRS Paper

**Date:** 2026-10-22
**Paper:** `Paper/main.tex`, "Structural Inductive Biases in Neural Optimization: Comparative Unfolding Paradigms for Multi-IRS IoT Networks"
**Base papers:** `Base-methods/` (Shen & Yu FP Part I and II, DeepFP, Nested Unfolding (DUFP), Learning-to-Unfold (GNNFP))

---

## 1. Intent

The paper's Numerical Results section (Sec. VI) has eight figures. Their numbers and captions were written from expectation, not from simulation. We want a Python codebase that:

1. implements the FP-AO baseline and the three unfolded architectures (FONR, MASO, TIAS) with correct mathematics, using the base papers wherever `main.tex` is wrong or incomplete;
2. runs the experiments exactly as Sec. VI designs them;
3. produces the eight figures from real runs;
4. reports, claim by claim, where the real results agree with the text and where they don't.

**Why:** the paper must rest on real results. If reality disagrees with the text, we need to know so the text can be fixed. A figure that only looks like the expected one is worthless.

## 2. Goals, non-goals, anti-goals

**Goals**
- Mathematically correct implementations, checked by unit tests (Sec. 9).
- The experiment designs of Sec. VI and Table I reproduced: sweeps, training sizes, zero-shot protocols, and metrics.
- Each of the eight figures saved as PNG in `Paper/` under the filename `main.tex` already uses, so the paper compiles against them unchanged.
- A findings report comparing every quantitative and qualitative claim in Sec. VI with the measured result.
- Every method able to run on CPU or GPU through one device switch.

**Non-goals**
- Editing `main.tex` text, captions or numbers. The report tells the authors what to change; they change it.
- The two drawings `system_model` and `tias_architecture`. They are diagrams, not results.
- Reproducing the base papers' own experiments (multicell MIMO and so on). They are sources of method details only.

**Anti-goals (doing any of these makes the work a failure even if it runs)**
- Hard-coding, rescaling, smoothing or post-hoc adjusting any result so it matches the numbers currently in `main.tex`.
- Choosing geometry, seeds, hyperparameters or the test set *after* looking at the comparisons, in order to make the neural methods look better or the text look right. Every free choice is fixed before the comparison runs and recorded in the config and the report (see Sec. 5.1 for the one allowed sanity check).
- Strawman baselines, such as a deliberately weak AO or an untuned MASO. Every method gets the same care.
- Hidden deviations. Every place the code departs from `main.tex` is listed in the report.

## 3. Sources of truth (decided by the authors)

1. **Mathematics and algorithm structure:** the base papers' correct mathematics wins. Where `main.tex` has a typo or an internal contradiction, the code follows the base papers and the derivation in Sec. 6, and the report lists the fix.
2. **Experiment design:** `main.tex` Sec. VI and Table I win: sweeps, sample counts, layer counts, training protocol (Sec. V), and which model is trained where.
3. **Gaps** that neither covers are filled by the builder with sensible, documented choices (marked *builder's call* below).

Method-to-paper mapping:

| Paper method | Base paper(s) | What we take from it |
|---|---|---|
| FP-AO (teacher, baseline) | Shen & Yu, FP Part I (closed-form FP, Alg. 3) and Part II (Lagrangian dual transform, uplink) | α/β/ε updates, closed-form power law |
| FONR | Xia et al., "Deep Unfolded FP-Based Beamforming in RIS-Aided MISO" (Nested Unfolding / DUFP) | L outer layers × T inner projected-gradient steps on θ, one learned real step per (l,t), loss −WSR + f₃ |
| MASO | Zhu et al., DeepFP (paper and code at github.com/zhujhz/DeepFP) | Per-layer (unshared) DNN that maps (current variable, gradient direction) to step sizes; complex MLP with complex ReLU; hybrid training |
| TIAS | Jiao et al., GNNFP | Single GNN shared across all unfolded layers; graph built from the quadratic structure; permutation-equivariant aggregation |

The only public code found is DeepFP's (checked 2026-10-22). GNNFP, DUFP and Shen–Yu have no public code, so they are implemented from their papers.

## 4. Codebase shape

Python with PyTorch, using the existing `E:\payam\env` (torch 2.8 + CUDA 12.6, numpy, scipy, matplotlib). Code lives in `E:\payam\code\`. Suggested layout (builder may adjust):

```
code/
  config.py          all parameters in one place (Sec. 5), dataclass + YAML/JSON dump per run
  channels.py        geometry, channel generation, association, dataset save/load
  fp_core.py         SINR, WSR, α/β/ε, power layer, A_n / v_n, matrix-free gradient, f₃
  ao.py              FP-AO baseline / teacher
  models/fonr.py, models/maso.py, models/tias.py
  train.py           hybrid protocol (Sec. 8)
  timing.py          fair latency measurement (Sec. 7.2)
  experiments/       one script per figure, each writes raw results (CSV/NPZ) then plots
  plotting.py        shared figure style
  report.py          builds the findings report from the raw results
  tests/             math and shape tests (Sec. 9)
```

All math is batched over samples (first tensor dim) and written in torch so it runs on either device. There are no Python loops over users, elements or IRSs. Loops over layers and iterations are fine.

Precision:
- **complex128** for the AO, teacher labels, all evaluation, and **all CPU timing** (the paper figures), so every method is timed at the same precision.
- The supplementary GPU timing uses **complex64 for every method**, because consumer FP64 throughput would distort it. Fairness only requires that all methods in one timing run share a device and a precision.
- **complex64** for training on the GPU. The machine's GTX 1660 Ti (6 GB) runs FP64 at about 1/32 speed, so float64 training would be impractically slow.
- All model code is dtype-agnostic, and checkpoints are evaluated in complex128.

## 5. System model and data (Sec. II of `main.tex` + Sec. VI-A)

| Item | Value | Source |
|---|---|---|
| BS antennas | 1 | main.tex |
| IRSs N | 2 | main.tex |
| Elements M | 80 default (ULA, half-wavelength) | main.tex |
| Users K | 6 for training | Table I |
| Noise σ² | −90 dBm | main.tex |
| Path loss | 128.1 + 37.6 log₁₀(d/1000), d in m | main.tex |
| Direct-link extra loss | +20 dB NLOS | main.tex |
| Shadowing | 8 dB log-normal, independent per link (direct, user–IRS, IRS–BS) | main.tex (per-link: builder's call) |
| Small-scale fading | Rician κ = 10 on user–IRS and IRS–BS links (LoS from ULA steering vectors at the geometric angles); Rayleigh on direct links | main.tex |
| Users | uniform in a disk of radius 20 m | main.tex |
| CRN peak power | 20 dBm; weights cycle {0.90, 0.80, 0.85} | main.tex |
| PMN peak power | 10 dBm; weights cycle {0.30, 0.20, 0.25} | main.tex |
| CRN/PMN split | user k even → CRN, k odd → PMN (so K=6 gives 3+3 with exactly the listed weights) | builder's call |
| Association | x_{k,n}=1 for n = argmax_n ‖b_{k,n}‖, fixed before optimisation | main.tex |
| Rate | log₂ | main.tex |

### 5.1 Geometry (builder's call, fixed once)

`main.tex` doesn't place the BS or the IRSs. Starting default (2D, metres): BS at (0, 0); user disk centred at (100, 0) with radius 20; IRS₁ at (10, 10), IRS₂ at (10, −10). The IRSs sit near the BS and well outside the user disk, so no user can be closer to an IRS than about 70 m and the path-loss formula never leaves its valid range.

Rough link budget for a typical user (about 90 m from an IRS):
- IRS–BS ≈ 14 m → 58 dB; user–IRS ≈ 90 m → 89 dB; minus the M=80 array gain of 38 dB gives a cascaded loss of about 109 dB.
- Direct link ≈ 100 m → 90.5 + 20 (NLOS) ≈ 110.5 dB.

So the cascaded and direct links are comparable and the IRS actually matters. (An earlier draft placed the IRSs inside the disk, which is invalid; that was fixed before any run and does not count as the allowed adjustment below.)

**The one allowed sanity check:** before any neural model is trained, run it on the **validation** set and compare two converged runs from the same initialisation:
- full FP-AO (power and phases);
- **power-only FP**: the same power law, with the phases frozen at their random initial values.

The check passes if full AO's mean WSR is at least 10% higher, which shows the IRS phases matter. Comparing against the WSR at initialisation instead would mix in the power-optimisation gain and say nothing about the IRS.

If the check fails, move the IRSs or the disk once, record why, and freeze the geometry. The check never looks at FONR, MASO, TIAS or the test set.

### 5.2 Datasets

- Training 10⁴, validation 10³, test 10³ independent realisations at the default setting (Table I), each with fixed seeds, saved to disk.
- Every sweep (M, K, power) gets its own 10³-sample test set generated from the same distribution with only the swept parameter changed.
- **One test-set size everywhere: 10³.** `main.tex` mentions 80 (Fig. convergence) and 200 (CDF caption) in places. We use Table I's 10³ throughout, and the report tells the authors to update those captions.
- Channels are normalised by σ internally (σ²=1 in code units). This is a mathematically exact rescaling, used only for numerical conditioning.

## 6. Mathematics (corrected; this is the spec the code implements)

Notation as in `main.tex`. Cascaded channel b_{k,n} = diag(g_n) h*_{r,k,n} (main.tex's convention). With single-surface association, write n(k) for the IRS user k uses.

- Effective channel: H_k = h_{d,k} + Σ_n x_{k,n} b_{k,n}ᴴ φ_n
- Total received power: D = Σ_i p_i |H_i|² + σ²
- SINR: γ_k = p_k|H_k|² / (D − p_k|H_k|²); WSR R = Σ_k w_k log₂(1+γ_k)

**Lagrangian dual transform (Shen–Yu Part II, Thm 3):** α_k = γ_k, adjusted weight w̃_k = w_k(1+α_k).

**Correction vs main.tex:** main.tex writes f₁ with log₂. The stationarity condition α* = γ only holds for the natural log, which is what Shen–Yu use. All transforms and surrogates therefore use ln. The reported WSR is Σ w_k log₂(1+γ_k), which equals (1/ln 2)·(the ln-WSR) and has the same maximiser. With α fixed, the objective becomes Σ_k w̃_k p_k|H_k|²/D, where the denominator D includes the user's own signal, as in Shen–Yu (23).

**Quadratic transform:**
- β_k = ε_k = √(w̃_k p_k) H_k / D. These are the same closed form; they are re-evaluated whenever p or Φ change.
- f₃(Φ, ε) = Σ_k 2Re{ε_k* √(w̃_k p_k) H_k} − Ξ·D(Φ), where Ξ = Σ_k |ε_k|².

**Power law (Shen–Yu Part I eq. 44, uplink single-receiver form; matches main.tex eq. power_update):**
p_k ← min{ P_k, w̃_k (max(0, Re{β_k* H_k}))² / (|H_k|² Σ_i |β_i|²)² }. The clamp sits before squaring. It is inactive in practice, because Re{β_k*H_k} = √(w̃_k p_k)|H_k|²/D ≥ 0.

**Phase sub-problem, per IRS n:** f₃(φ_n) = 2Re{v_nᴴ φ_n} − φ_nᴴ A_n φ_n + const, where
- A_n = Ξ Σ_i x_{i,n} p_i b_{i,n} b_{i,n}ᴴ
- v_n = Σ_i x_{i,n} ( √(w̃_i p_i) ε_i − Ξ p_i h_{d,i} ) b_{i,n}

**Correction vs main.tex:** main.tex writes h*_{d,i} in v_n. Expanding |h_{d,i} + b_{i,n}ᴴφ_n|² gives 2Re{(h_{d,i} b_{i,n})ᴴ φ_n}, so the correct term is h_{d,i}. Single association means no cross-IRS term.
- Wirtinger gradient: g_n = ∇_{φ_n*} f₃ = v_n − A_n φ_n.
- **Matrix-free form (used by FONR, MASO and TIAS; never builds A_n):** g_n = Σ_{k: x_{k,n}=1} b_{k,n} c_k, where c_k = √(w̃_k p_k) ε_k − Ξ p_k H_k. Note h_{d,k} + b_{k,n}ᴴφ_n = H_k. This is TIAS's observation-node scalar. main.tex's "h*_{d,k} + ψ" is the same typo and is fixed here.
- Curvature diagonal: [A_n]_{mm} = Ξ Σ_k x_{k,n} p_k |b_{k,n,m}|². main.tex's TIAS message omits Ξ p_k; the corrected message is [b_{k,n,m} c_k, Ξ p_k |b_{k,n,m}|²].

**A_n is singular.** Its rank is at most K_n (the users on IRS n), which is ≤ K ≪ M. So main.tex's φ^unc = A_n⁻¹ v_n does not exist as written. The authors chose the MM step in Sec. 6.1 instead (decision D1, Sec. 11).

### 6.1 FP-AO (baseline and teacher)

One iteration:
1. compute α (and so w̃), β and ε from the current (p, Φ);
2. update p by the power law;
3. **refresh ε, Ξ, H and D using the new p**, keeping α and w̃ from step 1;
4. update each φ_n using that refreshed ε;
5. recompute R and stop when ΔR/R < 10⁻⁵ or after 100 iterations (Table I).

**Deviation from main.tex Algorithm 1 (decided during implementation).** main.tex computes ε before the power update and reuses it for the phase step. That order is a valid block ascent, but it is degenerate in practice. Substituting the closed-form p into c_k = √(w̃_k p_k)ε_k − Ξ p_k H_k gives c_k = 0 exactly for every user whose power is not clipped at P_peak. Measured on 200 default samples, the median |c_k| for unclipped users was about 3·10⁻²⁰, against 0.026 for clipped users. So about 69% of users would contribute nothing to the phase gradient, which weakens AO and all three networks.

Refreshing ε after the power update keeps the alternating ascent valid and gives a tighter bound; it follows Shen–Yu's principle of making each auxiliary variable optimal before each primal block. This change is implemented once, in the shared deterministic block, so it applies identically to FP-AO, FONR, MASO and TIAS. The findings report lists it.

**Phase step (decided, D1):** a majorisation–minimisation step using the nonhomogeneous bound from DeepFP Lemma 1 (Shen et al.'s FastFP). Because |φ_m| = 1, φᴴφ = M is constant, which gives the exact closed form
φ_n ← exp(j·∠( v_n + (λ_n I − A_n) φ_n )), with λ_n = λ_max(A_n).

This is mathematically valid on the unit-modulus set and monotonically non-decreasing in f₃. It also equals proj(φ_n + g_n/λ_n) with the Euclidean gradient. FONR and TIAS are this update with a learned step. MASO uses the Riemannian gradient, so for MASO the match is only approximate. This is the DeepFP narrative.

**Cost depends on implementation (see decision D2).** Since A_n = Ξ B_n P_n B_nᴴ with B_n of size M×K_n:
- A_nφ can be computed in O(K_n M);
- λ_max(A_n) is the largest eigenvalue of the K_n×K_n matrix Ξ P^{1/2} B_nᴴB_n P^{1/2}, which costs O(K_n² M + K_n³).

An efficient AO is therefore *not* cubic in M. Only the explicit form (build the M×M matrix A_n, run an M×M eigen-decomposition or solve) is O(K M² + M³). Both forms give identical iterates.

Initialisation (shared by all four methods, same per-sample seed): φ⁽⁰⁾ with i.i.d. uniform phases; p⁽⁰⁾ = P_k^peak (*builder's call*; Shen–Yu suggest a simple heuristic start).

### 6.2 FONR (DUFP structure)

L = 10 outer layers. Each layer: deterministic α → β → power layer → ε (Sec. 6.1 steps 1–3), then T = 3 inner steps per IRS:
θ_n ← θ_n + μ_{l,t} · ∇_θ f₃, with ∇_θ f₃ = 2 Re{ −j φ_n* ⊙ (v_n − A_n φ_n) } = 2 Re{ −j φ_n* ⊙ g_n } computed matrix-free, and φ_n = e^{jθ_n}.

This is DUFP eq. (19): their descent on φᴴUφ − 2Re{φᴴv} is the same as our ascent on f₃.
- Trainable parameters: one real scalar μ_{l,t} per (layer, inner step). That is L·T = 30 parameters, independent of K, M and N, exactly as in DUFP.
- main.tex calls the operator a "complex scalar weighting". DUFP's real step on θ is the base-paper form and is what main.tex's own angle parameterisation (Sec. V-D) describes, so we use it and list the change in the report.
- Initialisation: μ_{l,t} = 1/(2·median λ_max(A_n)) over the training set, so an untrained FONR already behaves like a sensible gradient method. DUFP initialises at 1 in its own scaling; this is the same idea adapted to our units (*builder's call*).
- **Consequence for zero-shot M:** FONR's parameters do not depend on M, so FONR runs at any M without padding. main.tex claims FONR degrades at M ≠ 80 "due to rigid dimensional constraints". That claim cannot hold for the base-paper FONR. We run the experiment as designed and report what happens.

### 6.3 MASO (DeepFP structure)

L = 10 layers. Each layer: the same deterministic α → β → power → ε blocks, then one step per IRS:
- Euclidean gradient g_n (matrix-free); Riemannian gradient r_n = g_n − Re{g_n ⊙ φ_n*} ⊙ φ_n (main.tex eq.).
- Step vector λ_n = Softplus(Ψ_{l,n}(z_n)) ∈ ℝ₊^M, where z_n = [φ_n, g_n] ∈ ℂ^{2M}. Following DeepFP, the network sees the current variable and the gradient direction, not raw per-user channels.

This resolves main.tex's internal contradiction: Sec. IV says the input contains b_{k,n} for the clustered users (K-dependent), Sec. VI says it is a K-agnostic aggregate. The DeepFP input is K-agnostic and M-dependent, which matches Sec. VI and the zero-shot-M design.
- Update: φ_n ← (φ_n + λ_n ⊙ r_n)/|φ_n + λ_n ⊙ r_n| (main.tex eq. maso_update).
- Ψ_{l,n}: complex MLP as in DeepFP. Input z_n = [φ_n, ĝ_n] ∈ ℂ^{2M}, with ĝ_n = g_n/(‖g_n‖ + 10⁻¹²) normalised over the full M before any truncation. Output head: a complex Linear to M, then its real part (as in DeepFP's code), then Softplus. Complex `Linear` layers, ComplexReLU = ReLU(Re) + j·ReLU(Im) (the DeepFP *paper's* definition; its code only applies ReLU to the real part, and we follow the paper), 2 hidden layers, real Softplus output of size M. Hidden width = 2M (main.tex says hidden width scales with M), so 160 at M = 80. Separate weights per layer l and per IRS n (DeepFP: layers do not share; main.tex: θ_n per IRS).
- Gradient-input scaling: g_n is divided by its per-sample norm before entering Ψ (direction only). The step it multiplies stays unnormalised. This guards against the 100× magnitude swings across power levels, and the same rule is applied in TIAS (*builder's call*, recorded).
- **Zero-shot M (designed in main.tex):** for M < 80, zero-pad z_n to 2·80 and use the first M outputs. For M > 80, feed the first 80 elements; elements 81…M get λ = 0, so they keep their initial phase. This is literal truncation. The report states this choice explicitly because it drives MASO's result.
- For the *runtime vs M* experiment, MASO is instantiated at each M with untrained weights. Timing does not depend on weight values.

### 6.4 TIAS (main.tex architecture, GNNFP principles)

L = 10 layers, each with the same deterministic blocks, then message passing on the bipartite graph (edges (k, (n,m)) where x_{k,n} = 1):
1. **Variable→observation:** ψ_{k} = b_{k,n(k)}ᴴ φ_{n(k)}; H_k = h_{d,k} + ψ_k; c_k as in Sec. 6.
2. **Observation→variable:** message m_{k→(n,m)} = [b_{k,n,m} c_k, Ξ p_k |b_{k,n,m}|²]; sum-pool to u_{n,m} = [u¹, u²]. u¹ is exactly element m of g_n and u² is exactly [A_n]_{mm}.
3. **Node update with one shared MLP F_GNN** (weights shared across all N·M nodes and all L layers, as GNNFP shares across layers): φ ← (φ + Δ)/|φ + Δ|, with Δ = s ⊙ u¹ / (u² + δ), where s = Softplus(F_GNN(scale-free node features)). This is a learned correction to the Jacobi (diagonal-Newton) step.

Scale-free features (fixed set, invariant to channel and power scaling so that zero-shot transfer is meaningful), with eps = 10⁻¹²:
1. Re{u¹ φ*}/(|u¹| + eps);
2. Im{u¹ φ*}/(|u¹| + eps);
3. log(|u¹|/(u² + eps) + eps), standardised per sample (zero mean, unit variance over that sample's N·M nodes);
4. the node degree fraction K_n/K.

δ in the Jacobi step: δ = 10⁻¹²·max(u²) per sample. Nodes with no users have u¹ = u² = 0, so Δ = 0 there.

F_GNN: 2 hidden layers of width 32 with ReLU (GNNFP widths). No batch-norm or dropout: inference runs one sample at a time and batch-norm would couple test samples.

TIAS's parameter count is independent of K, M and L. The edge count is |E| = Σ_n K_n·M, giving O(|E|) per layer.

### 6.5 Shared rules for the three neural models

- The power layer and the auxiliary variables are deterministic and have no parameters (main.tex Sec. IV-B).
- Same initialisation as AO for the same sample (Sec. 6.1).
- Every layer's output is feasible: |φ| = 1 and 0 ≤ p ≤ P_peak. Assert this in the tests.

## 7. Experiments (exactly as Sec. VI designs them)

All neural models are trained once at the default setting (M = 80, K = 6, CRN 20 dBm) and evaluated **without retraining** in every sweep. AO is run fresh on every test sample.

| # | Output file (in `Paper/`) | Design | Metric |
|---|---|---|---|
| 1 | `WSR_vs_iteration.png` | default test set; x = AO iteration index or neural layer index 0…10. AO is also shown converged (it may need more than 10 iterations; draw a dashed line at the AO converged mean) | mean WSR with shaded 95% CI over the 10³ samples |
| 2 | `WSR_vs_wall_clock.png` | same runs; x = cumulative wall-clock time per step (CPU, batch 1, Sec. 7.2); AO plotted until it stops | mean WSR vs mean cumulative time |
| 3 | `CDF.png` | default test set | empirical CDF of ΔR = R_AO(converged) − R_method(L=10), one curve per neural method; 95th percentile and left tail reported |
| 4 | `avg_time_vs_Number_of_RIS_M.png` | M ∈ {20, 40, …, 180}, K = 6 | mean per-iteration (AO) / per-layer (neural) time, log₁₀ y-axis |
| 5 | `etha_vs_CRN_transmit_power.png` | CRN peak ∈ {20, 25, …, 45} dBm, PMN = CRN − 10 dB | η = mean(R_method)/mean(R_AO)·100% (ratio of means; *builder's call*, the per-sample mean ratio also goes in the report) |
| 6 | `WSR_vs_Number_of_active_users_K.png` | K ∈ {10, 15, …, 60}, M = 80 | mean WSR, all four methods |
| 7 | `avg_time_vs_Number_of_active_users_K.png` | same K sweep | mean per-iteration/per-layer time |
| 8 | `WSR_vs_M_ZeroShot.png` | M ∈ {20, 40, 60, 100, 120, 160}; also plot M = 80 as the reference point | mean WSR, all four methods |

The timing figures (2, 4, 7) each show two AO curves, "FP-AO (explicit)" and "FP-AO (efficient)" (decision D2). In Fig. 2 both curves have the same WSR values and differ only in time. Fig. 1 and Fig. 2 are separate files because `main.tex` includes them as subfigures. Plot style: IEEE single-column width (3.5 in), readable fonts, one consistent colour and marker per method across all figures, and PNGs at 300 dpi. Also save PDF copies next to them.

### 7.1 What each experiment must also record

Raw per-sample results (WSR per layer or iteration, times, AO iteration count) are saved as NPZ/CSV, so figures and the report can be regenerated without rerunning. Each run also saves its config and seeds.

### 7.2 Fair timing

- One `--device {cpu,cuda}` switch; within a timing run every method uses the same device.
- **Paper figures (2, 4, 7) use CPU timing.** With batch size 1, a GPU measures launch overhead rather than arithmetic and would hide the M³ vs |E| scaling. DeepFP times on CPU for the same reason.
- Batch size 1, warm-up runs discarded, `torch.no_grad()`, `torch.set_num_threads` fixed and recorded, and `time.perf_counter`. Report the mean, with the median in the raw data.
- The same timings are repeated on GPU with `torch.cuda.synchronize()` around every timed region, as a supplementary result in the report. If the method ordering changes on GPU, the report says so.
- AO per-iteration time includes all of its phase-step work. Neural per-layer time includes the deterministic blocks. Nobody's timing excludes work the method really does.
- **Per-iteration (per-layer) time** for Figs. 4 and 7 = total time for the first 10 AO iterations or the 10 neural layers, divided by 10, averaged over the test samples. The wall-clock curve in Fig. 2 uses cumulative time per step over each method's full run.
- If timing all 10³ samples at large M takes too long, a fixed, pre-declared subset of the first 200 test samples may be used for timing only, the same subset for every method. WSR values always use all 10³.
- Every method is timed as the same kind of code: eager PyTorch, written vectorised with no Python loops over elements or users. At batch 1 on CPU, eager mode carries per-operation overhead of a few µs, so measured times partly reflect operation counts as well as asymptotic cost. The report states this. No method gets `torch.compile` or a custom kernel unless all do.

## 8. Training protocol (main.tex Sec. V, applied identically to FONR, MASO and TIAS)

- **Labels:** run FP-AO to convergence on each training sample, from the same initialisation the network will see, and cache Φ*.
- **Stage I:** supervised phase-alignment loss (1/NM) Σ(1 − Re{φ^{(L)} φ*^*}) for 50 epochs.
- **Stage II:** unsupervised, 150 epochs.
  - Composite loss −α·R − (1−α)·f₃, with ε recomputed in closed form from the layer's output.
  - Inside f₃, w̃ uses the α that layer l actually used (its input-state α), **detached**. If α were re-evaluated at the output with gradients flowing through it, f₃ would collapse to the weighted sum-SINR Σ w_k γ_k, which is a different objective.
  - With α detached and ε* closed-form, f₃ is a valid lower-bound surrogate whose gradient at α = γ matches the ln-WSR gradient.
  - Both terms are in nats: R in the loss is the ln-WSR, so the α weighting means what it says. Reported WSR stays in bits.
  - Deep supervision: Σ_l ρ_l·loss_l with ρ_l = l / Σ_j j (monotonically increasing).
  - α = 0.5 (*builder's call*; DUFP uses an unweighted sum, which is equivalent up to scale).
- Optimiser: Adam, constant lr, batch size 200 (DeepFP).
  - Equal, pre-registered tuning budget for every neural model: lr ∈ {0.1 (Table I), 0.01, 0.001}.
  - Each value gets a short pilot (5 supervised + 15 unsupervised epochs). Pick the one with the best mean **validation** WSR, then do the full run.
  - The test set is never used for any choice.
  - If a model's chosen lr is not 0.1, the report flags it as a deviation from Table I and shows the pilot curves. This prevents an untuned-MASO strawman without favouring any model.
  - Keep the checkpoint with the best validation WSR.
- Training runs on GPU by default. Seeds are fixed, and training curves are saved.
- The Adam optimiser continues from Stage I into Stage II without being reset. Epochs are numbered 1–200: 1–50 supervised, 51–200 unsupervised.
- **Checkpoints** go in `artifacts/checkpoints/`:
  - `{model}_last.pt` is written every epoch; `{model}_best.pt` is written whenever the mean validation WSR reaches a new maximum.
  - Each contains: model_name, epoch, stage, model and optimizer state, selected_lr, best_val_wsr, best_epoch, CPU and CUDA RNG state, metrics_history, and the config.
  - **metrics_history holds train and validation metrics only.** The test set is evaluated once, after training, on `best.pt`. Logging test metrics every epoch would invite selection on the test set.
- **Resume:** if `{model}_last.pt` exists with epoch < 200, training resumes at epoch + 1. If epoch = 200, `best.pt` is loaded and nothing is retrained unless `force_retrain=True`. The learning-rate pilot is resumable too: if `{model}_lr_pilot.csv` exists, it is reused.
- **Per-epoch CSV** `artifacts/logs/{model}_training_history.csv`, with columns: epoch, stage, lr, train_loss, val_loss, train_wsr_mean, train_wsr_std, val_wsr_mean, val_wsr_std, val_phase_cosine_sim, is_best, epoch_time_sec.
- **Final metrics CSV** `artifacts/logs/all_models_final_metrics.csv` (plus a JSON per model), computed on `best.pt` in complex128. Columns:
  - model_name, selected_lr, best_epoch;
  - for each of train, val and test: loss, wsr_mean, wsr_std, ao_wsr_mean, efficiency_pct;
  - test only: test_gap_median, test_gap_95th_pctl, test_gap_min_left_tail.

  AO reference values come from the cached teacher runs.

## 9. Verification (must pass before any experiment runs)

1. **Gradient check:** the matrix-free g_n equals v_n − A_n φ_n from explicit matrices, and equals the autograd Wirtinger gradient of f₃ (torch complex autograd gives ∂/∂φ*, up to the factor convention, which must be pinned down in the test).
2. **Power law:** the closed form maximises f₂ for fixed β (compare against a numeric 1-D search) and respects the bounds.
3. **AO monotonicity:** with the MM phase step, f₃ is non-decreasing across the phase step. Track the WSR per iteration and report any decrease. For random test channels, it should not decrease beyond numerical noise.
4. **Transform identities:** with natural log, ∂f₁/∂α = 0 at α = γ and f₁(γ) equals the ln-WSR; β* maximises f₂.
5. **TIAS message passing:** u¹ equals the explicit gradient and u² equals diag(A_n), for random graphs including users with no IRS links.
6. **Feasibility:** |φ| = 1 and 0 ≤ p ≤ P_peak after every layer for every model.
7. **Size handling:** FONR and TIAS run at any K and M; MASO pads and truncates as specified.
8. **AO equivalence:** explicit and efficient FP-AO produce the same iterates (to float64 tolerance) on random channels, including at M = 180 and K = 60.

## 10. Findings report

Output: `Paper/results_findings.md`. Contents:

- **Claims vs measured.** One table with one row per claim in Sec. VI:
  - ~90/96/98% recovery at layers 4/6/8;
  - TIAS ≥ 98% of AO by about 0.85 ms and ≥ 99% by about 1 ms;
  - AO converges at about 7.9–8.8 ms;
  - 95th-percentile gaps of 0.08/0.09/0.13 bps/Hz;
  - MASO's left tail reaching about −2 bps/Hz;
  - the runtime ordering t_TIAS < t_MASO ≈ t_FONR ≪ t_AO;
  - "over an order-of-magnitude" speedup at M = 180;
  - FONR slower per layer than MASO;
  - the η ranges per method;
  - K-sweep indistinguishability;
  - AO latency growing with K;
  - TIAS lowest latency vs K;
  - FONR and MASO degrading at M ≠ 80;
  - TIAS tracking AO across M.

  Each row gives: the claim, its section/figure, the measured value, and a verdict (Confirmed / Partly / Contradicted).
- **Every deviation from main.tex**, with the reason: math corrections, the AO phase step, the FONR step form, the MASO input, the TIAS message, test-set size, lr changes, geometry, and all *builder's calls*.
- **Supplementary GPU timing.**
- **Train / validation / test metrics table** taken from `all_models_final_metrics.csv`.
- **CPU vs GPU latency tables.**
- **Learning-rate pilot summary.**
- **Both power-efficiency statistics:** ratio of means and mean of ratios.
- **Suggested text changes.** Concrete edits to Sec. III, IV and VI. Not applied; the authors decide.

## 11. Open decisions for the authors

- **D1, the AO phase step: decided, MM.** The phase step is the MM / nonhomogeneous-bound closed form from DeepFP Lemma 1 (Sec. 6.1). It is exact on the unit circle, never lowers f₃, and has no constant to tune. **No pseudo-inverse and no regularised solve**, anywhere in the code.
- **D2, which AO gets timed: decided, both.**
  - **"FP-AO (explicit)"** builds the M×M matrix A_n and computes λ_max with an M×M eigen-decomposition. This is the algorithm as `main.tex` Sec. III describes it, at O(K M² + M³).
  - **"FP-AO (efficient)"** computes A_nφ matrix-free and λ_max from the K_n×K_n Gram matrix.
  - Both produce identical iterates; a test asserts that. Every WSR result is therefore one AO curve, while every timing figure (2, 4, 7) shows both AO curves, and the report discusses what that means for the speed claims.

## 12. Artifacts, resumability and the master notebook

**Artifacts** live under `code/artifacts/`:
- `datasets/` holds every dataset (default train, val and test, plus one test set per sweep point) and `teacher_labels.pt`. Each is generated with a fixed seed, and if the file exists it is loaded instead of regenerated.
- `checkpoints/`, `logs/` (CSVs and configs) and `results/` (`exp_{id}_{name}.npz/.csv`).

Every experiment function takes `force_rerun=False`. If its raw result file already exists, it only re-plots.

**Notebook** `code/master_pipeline.ipynb`. It imports the modules and contains no algorithm code of its own. Each cell runs independently and resumes from the artifacts:
1. Setup: imports, seeds, hardware/CUDA report, Config, and the RESUME / FORCE_RETRAIN flags.
2. Unit tests (pytest run as a subprocess, with a pass/fail table) and the Sec. 5.1 sanity check.
3. Datasets and teacher labels.
4. FONR: learning-rate pilot, training, metrics table and curves.
5. MASO: the same.
6. TIAS: the same.
7. Master train/val/test comparison table, including FP-AO.
8–15. One cell per figure: run or load, save PNG and PDF into `Paper/`, show inline, print key numbers.
16. Supplementary GPU timing, then `report.py` writes `Paper/results_findings.md`.

**End-to-end run:** `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1`, launched in the background, because training and timing take hours. `nbconvert` isn't installed in `env` yet, so it is installed there with pip. Because every stage resumes from its artifacts, an interrupted run continues where it stopped.

## 13. Left to the builder

- Exact module layout.
- Float precision for training.
- The exact TIAS scale-free feature set (within the invariance requirement).
- Plot styling details.
- Caching strategy.
- How to parallelise label generation.

Every such choice is recorded in the config and the report.
