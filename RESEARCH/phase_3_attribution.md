# Phase 3 Research — Causal Attribution Engine

**Date:** 2026-05-29
**Scope:** Implementable findings for `attribute/contrastive.py` (single-step resampling
attribution) and `attribute/shapley.py` (budget-bounded Monte-Carlo Shapley over steps).

**Method caveat:** `WebFetch` unavailable; equation-level claims reconstructed from search
summaries + standard results. Items marked **[verify]** need a primary-source check before being
cited verbatim. Several 2026 arXiv IDs are dated at/just after today — treat dates as **[verify]**.

---

## 1. Counterfactual credit assignment (lineage)

- **Mesnard et al., "Counterfactual Credit Assignment in Model-Free RL"** (arXiv:2011.09464,
  ICML 2021). Uses a *hindsight / future-conditional baseline* V(S_t, Φ); unbiased **iff** the
  hindsight statistic Φ is conditionally independent of the action given the state (the
  independence constraint), and provably lower-variance than a forward baseline. Ancestor:
  Harutyunyan et al., "Hindsight Credit Assignment," NeurIPS 2019.
- **COMA** (Foerster et al. 2018) — counterfactual baseline that marginalizes ONE agent's action
  while holding others fixed. This is the cleanest formal analog of CAR's single-step
  `do_resample`.
- 2026 successors (LLM agents, **[verify dates]**): survey "From Reasoning to Agentic: Credit
  Assignment in RL for LLMs" (2604.09459); HCAPO (2603.08754, hindsight counterfactual
  continuation); CCPO (2603.21563, COMA-style counterfactual baselines for multi-agent).

### The run-forward confound (the main pitfall — drives the design)
`do_resample(k)` re-samples a_k *and* re-rolls every downstream stochastic step. So
τ_k = E[Y | do(resample k)] − Y_factual is a **total effect through a stochastic continuation**,
not the isolated effect of a_k. Consequences:
- This is usually what we want for "did this step doom the run" — but a late, irrelevant step can
  show nonzero effect purely by perturbing a noisy continuation. **Guard every claim with a CI:**
  call a step pivotal only when its effect CI excludes 0.
- An *early* irrelevant step also shows an effect (it re-rolls the true pivotal step downstream).
  So magnitude alone cannot localize. **Identification rule:** the causal locus = the **largest k**
  whose resampling effect CI still excludes 0 — the last step where re-rolling still rescues the
  run; beyond it the outcome is locked in. (This is what `contrastive.py` computes.)
- **Common random numbers (CRN):** to isolate a_k from downstream noise, reuse the same downstream
  randomness across factual and counterfactual (paired design) — the single biggest variance
  reducer. Where an early resample diverges the branch so seeds no longer align, fall back to
  averaging K continuations and document τ_k as a through-continuation total effect. (v1 uses the
  total-effect definition + latest-significant-step rule; CRN is a documented refinement.)

---

## 2. Causal influence diagrams (framing only)
Cite for "an agent run is legitimately an SCM with the policy as a causal mechanism":
- **Everitt, Carey et al., "Agent Incentives: A Causal Perspective"** (AAAI 2021, arXiv:2102.01685).
- **Kenton et al., "Discovering Agents"** (AIJ 2023, arXiv:2208.08345) — causal definition of agency;
  SCM ↔ influence-diagram translation.
- Working group: causalincentives.com. No single 2025+ CID paper supersedes these.
This justifies the do-intervention modeling; it does **not** supply the estimator (that's §1/§3).

---

## 3. Monte-Carlo Shapley + variance reduction (the tractable estimator)

- **Permutation sampling / ApproShapley** (Castro, Gómez, Tejada 2009): sample permutations; walk
  each adding steps one at a time; the marginal contribution v(pre∪{i}) − v(pre) is recorded for
  every step in one walk (n+1 coalition evals per permutation). Unbiased; **CLT** error.
- **Stratified sampling** (Maleki et al. 2013, arXiv:1306.4265): stratify marginals by coalition
  size, Neyman allocation; **finite-sample** Hoeffding/Chebyshev bounds.
- **Antithetic sampling** (MDPI 2023): pair each permutation with its **reverse**; negative
  correlation cancels variance. Essentially free; combine with permutation sampling.
- **TMC-Shapley** (Ghorbani & Zou 2019): truncate once v(S) ≈ v(N). **Avoid for CAR** — truncation
  can skip a pivotal *late* step (a late step can flip a near-certain outcome).
- **KernelSHAP**: more sample-efficient (reuses each v(S) for all players) but messier CIs/budget
  and arbitrary-coalition continuations break CRN. Treat as a later optimization, not v1.

### Hard-budget formulation
Each coalition value v(S) = mean of `samples_per_eval` forward rollouts. One permutation =
(n+1) coalition evals. With antithetic, 2 walks per permutation. Total forward samples ≈
M · (n+1) · samples_per_eval · 2. Budget caps M (and a circuit breaker caps total samples).

### Confidence intervals
**CLT on per-step marginal contributions**: φ̂_i = mean of M (paired, if antithetic) marginals;
CI = φ̂_i ± z·s_i/√M. IMPORTANT: marginals must be **i.i.d. across permutations** — so v(S) is
re-evaluated with fresh samples per permutation (do **not** cache v(S) across permutations, or the
per-step variance collapses to 0 and the CI is falsely tight). Bootstrap is the small-M fallback.

### Interaction (Shapley vs contrastive) — verified arithmetic
AND-failure: v(∅)=v({i})=v({j})=0, v({i,j})=1. Shapley φ_i = (0 + 1)/2 = **0.5**, φ_j = **0.5**
(efficiency: sum = 1). Single-step contrastive either double-counts (each ≈ 1, holding the other
bad) or under-attributes (each ≈ 0, holding the other resampled) — it cannot express shared
synergistic responsibility. **This is the headline reason `shapley.py` ships alongside
`contrastive.py`.**

---

## 4. 2025–2026 LLM-agent trace attribution (positioning the writeup)
Closest neighbors and how CAR differs:
- **Who&When** (arXiv:2505.00212, ICML 2025): defines the task; LLM-judge baselines hit only
  **~14% step-level** accuracy. Use as motivation.
- **AgenTracer** (2509.03312): counterfactual replay via **oracle substitution** (replace action
  with the gold action) + a trained scorer. CAR differs: **same-policy resampling** (not oracle),
  distributional outcome shift, not a learned black box.
- **"Seeing the Whole Elephant" Dynamic Agentic** (2604.22708, **[verify date]**): re-runs from a
  candidate step + LLM counterfactual checks — closest to "re-run from step k", but an LLM-judge
  heuristic, not a measured interventional distribution shift with CIs.
- **Ma et al.** (2509.08682): pairs **Shapley + causal inference** with agent failure attribution —
  but via causal-discovery over static logs, not executed do-interventions. CAR must contrast
  here and **not** claim novelty on "Shapley for agent blame" or "counterfactual replay" per se.

**CAR's defensible one-liner:** existing tools are LLM-judge heuristic trace scoring or
causal-discovery over static logs; the few that re-run use oracle substitution or LLM
verification. CAR is distinctive in combining **executed same-policy `do_resample` interventions
+ distributional outcome shifts with CIs + Monte-Carlo Shapley credit-splitting + ground-truth
validation against synthetic SCMs.**

---

## Decisions that shape `contrastive.py` and `shapley.py`
1. **Contrastive effect** = distributional outcome shift P(bad | do_resample at k) − P(bad |
   factual), over K forward samples; **causal locus = largest k with a CI that excludes 0**
   (handles the run-forward confound). Document τ_k as a through-continuation total effect; CRN is
   a future refinement.
2. **Shapley estimator** = permutation sampling + antithetic reverse pairing; **do NOT cache v(S)
   across permutations** (preserves i.i.d. marginals for honest CIs). Stratification is a future
   add; TMC truncation is rejected (late-step risk).
3. **CIs everywhere** — Wilson for proportions (Phase 2), CLT for Shapley marginals; never a bare
   point. Every "step X caused it" claim is CI-gated.
4. **Budget-bounded** — each v(S) charges `samples_per_eval` forward rollouts to a `Budget` with a
   hard sample cap + cost cap + circuit breaker; Shapley is on-demand, never default.
5. **Validate against §7 ground truth** — contrastive recovers the pivotal step; Shapley recovers
   the 0.5/0.5 interaction split; both checked before trusting any heatmap.

### Sources
arxiv.org/abs/2011.09464 · NeurIPS19 Hindsight CA · Foerster COMA 2018 · arxiv.org/abs/2102.01685 ·
arxiv.org/abs/2208.08345 · Castro 2009 (S0305054808000804) · arxiv.org/abs/1306.4265 ·
arxiv.org/abs/1904.02868 · MDPI 3/4/49 (antithetic) · arxiv.org/abs/2505.00212 ·
arxiv.org/abs/2509.03312 · arxiv.org/abs/2509.08682 · arxiv.org/abs/2604.22708. Accessed 2026-05-29.
