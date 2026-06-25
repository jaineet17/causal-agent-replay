# Phase 6 Research — Who&When Benchmark via Surrogate-Policy Replay

**Date:** 2026-06-11
**Verdict: GO-WITH-SCOPE.** Full Algorithm-Generated subset (126, short 5–10-step logs) as the
primary evaluation; Hand-Crafted pilot (20 trajectories, truncated horizon) for the scaling story.
**Method caveat:** snippet-extracted facts; load-bearing numbers re-checked against PDFs before
any paper claim. Items marked **[verify]** rechecked at implementation time.

---

## 1. The dataset

- **Who&When** (Zhang et al., ICML 2025 Spotlight, arXiv:2505.00212): **184 annotated failure
  logs** = **126 Algorithm-Generated** (each from a distinct CaptainAgent-built system; 5–10
  steps) + **58 Hand-Crafted** (Magnetic-One; 5–130 steps). Queries from GAIA/AssistantBench.
- Hosting: HuggingFace `Kevin355/Who_and_When` (JSON per instance + parquet exports, ~8.7 MB).
  Annotations: `mistake_agent`, `mistake_step`, `mistake_reason`, + query and ground-truth answer.
- **License: UNVERIFIED [verify]** — check the HF card/repo before redistributing any derivative
  (incl. counterfactual continuations). Local evaluation is fine.
- Critical structural fact: logs are **output-only** (no original system prompts / full tool I/O)
  — "partial observability" (confirmed by FAMAS, TraceElephant). The surrogate cannot condition
  on what the original agents saw privately. This is the #1 fidelity confound.

## 2. The bar (mid-2026 leaderboard, step-level emphasis)

- Original GPT-4o baselines: 53.5% agent / **14.2% step** (the number our writeup already cites).
- **FAMAS (arXiv:2509.13782) — the fair comparator** (only other replay/resampling method):
  20 full re-executions per task with a local Qwen2.5-72B surrogate; **55.56% agent / 23.81% step
  on AG**; cost ~105 min/task (~322 task-hours total).
- A2P (2509.10401, single-pass counterfactual scaffold; AI-authored, treat with care): 47.5% step
  AG / 29.3% HC. MASPrism (2605.07509): 27.6% step HC at 2.66 s/trace. ECHO (2510.04886): 78.8%
  agent [verify subset]. CDC-MAS: up to 36.2% step [verify subset]. **GraphTracer: WITHDRAWN —
  exclude.**
- MP-Bench (2603.25001) argues the single-root-cause label assumption is flawed — both a threat
  to exact-match scoring and an argument FOR CAR's distribution-over-steps output. Report exact
  match AND ±1/±3 tolerance curves (tolerance reporting precedented in the original paper).

## 3. Method: surrogate-policy replay (the §12 extension, made concrete)

- Precedent: FAMAS re-executes with a surrogate LLM (existence proof, publishable); AgenTracer
  used counterfactual replay for *annotation*; ToolEmu (2309.15817) is the citation for
  LLM-simulated tool/env responses; CF-GPS (1811.06272) + Gumbel-Max SCMs (Oberst & Sontag) are
  the formal grounding for "hold the factual prefix, intervene, roll forward."
- Design: **role-conditioned zero-shot surrogate** ("You are {agent}; here is the conversation;
  produce the next message") for agent steps + an **explicit env-sim role** for tool/observation
  steps (the clean agent-sim/env-sim split is itself novel in this line) + an outcome judge.
- **w/ ground-truth-answer setting primary**: the judge reduces to answer-match → near-
  deterministic outcome labels, killing most judge noise.
- Known failure modes (documented in the Sim2Real literature, 2603.11245: best user-simulator
  USI 76.0 vs human 92.9; simulators too cooperative/uniform): surrogate infidelity inflates
  counterfactual success → over-attribution. Mitigation = the **fidelity audit** (below), which
  no prior Who&When work has done.

## 4. Scoped experiment design

| component | choice |
|---|---|
| primary | full AG subset (126), exhaustive per-step `do_resample`, full remaining horizon |
| secondary | HC pilot (20), horizon m=12, binary search over k (monotonicity stated) |
| surrogate | local Qwen3-8B-class via Ollama; 20-instance stronger-surrogate sensitivity ablation |
| judge | answer-match vs GT (primary); calibrated on 50 hand-checked continuations |
| K | max 16 with **CI-aware early stopping** (stop when CI excludes 0 or width < ε); report K̄ |
| metrics | agent exact, step exact, step ±1/±3 curves, + CAR-native flip-rate & effect CIs |
| sanity floors | factual replay reproduces failure (>90% target); prefix-reproduction similarity per role |
| cost | ~40–55K LLM calls total — days on a local 8B, ~zero dollars |
| target | beat FAMAS 55.6%/23.8% on AG; contextualize vs A2P (1-call class, not directly comparable) |

## 5. Validity threats → honest reporting

1. **Surrogate ≠ original policy** → report prefix-reproduction scores, factual-replay
   reproduction rate, two-surrogate ablation; claim "surrogate-counterfactual attribution," never
   "the causal step."
2. **Human 'decisive error' ≠ counterfactual locus** (MP-Bench) → tolerance curves + qualitative
   disagreement analysis; disagreements are data.
3. **Judge noise + multiplicity** (n CIs per trajectory) → w/GT primary, judge calibration
   reported, FDR/Bonferroni-adjusted CI procedure, seeds and K disclosed; test for GAIA answer
   contamination in surrogate/judge.

## Sources
arXiv:2505.00212 · HF Kevin355/Who_and_When · github.com/ag2ai/Agents_Failure_Attribution ·
2509.13782 (FAMAS) · 2509.03312 · 2509.10401 · 2509.08682 · 2510.04886 · 2603.25001 (MP-Bench) ·
2603.11245 (Sim2Real) · 2604.22708 (TraceElephant) · 2605.07509 (MASPrism) · 2309.15817 (ToolEmu)
· 1811.06272 (CF-GPS). Accessed 2026-06-11.

---

## Results (full AG run) — 2026-06-25

**Setup.** Algorithm-Generated subset, surrogate-counterfactual attribution with a **free local
`llama3.2` (3B)** for all three roles (agent surrogate, GT-grounded environment simulator, and the
extract-then-compare outcome judge). Per-step `do_resample`, K≤8 with CI-aware early stopping
(median realized K = 8/step, mean 69 rollouts/instance), horizon = 5, w/-ground-truth judge.
**N = 121/126 evaluated**; 5 excluded as infrastructure casualties (1 wall-clock timeout, 4
transient connection errors across local-inference wedges / system sleep — not method failures).
The full run was ~a week of intermittent local compute; the runner tuning that made it tractable
is the §"Hard parts" lesson, not a result.

**Sanity floor: 120/121 (99%)** — the unmodified factual log is judged "still failing", so the
counterfactual rescues are measured against a correct baseline.

| rule (re-derived offline on the same rollouts) | predicts on | agent | step exact | step ±1 | step ±3 |
|---|---|---|---|---|---|
| `ci_locus` (CI excludes 0; abstains otherwise) | 22/121 | 7.4% | 4.1% | 7.4% | 14.9% |
| `argmax` (max rescue rate) | 121/121 | **48.8%** | **20.7%** | 41.3% | **71.9%** |
| `latest_tol` (latest within tol of max) | 121/121 | 43.8% | 14.9% | 28.1% | 54.5% |
| `cliff` (largest consecutive rescue drop) | 121/121 | **49.6%** | 15.7% | 38.8% | 68.6% |

(Accuracies over all 121 ok rows; abstentions count as misses.)

**Reading vs the field.** The fair comparator is **FAMAS** (the only other replay/resampling
method): **55.6% agent / 23.8% step** on AG — using a **72B** surrogate. CAR's best point-estimate
rule reaches **~49% agent / 20.7% step exact / 71.9% within ±3 with a 3B surrogate (~24× smaller)**.
Step-exact 20.7% **clears the original LLM-judge baseline (14.2%)** and **approaches FAMAS's
replay-based 23.8%**; agent accuracy sits below FAMAS but near the judge baselines. The honest
headline is *competitiveness with a far larger replay method on a real benchmark*, not SOTA — and
exactly the "surrogate-counterfactual attribution" framing committed to up front.

**The load-bearing limitation (a genuine finding).** The principled CI-gated locus (`ci_locus`),
which is CAR's headline rule on the validated synthetic SCMs, **fires on only 22/121 here**: at
K≤8 with a noisy 3B surrogate, per-step rescue rarely reaches significance, so the rule abstains on
82% of instances. The point-estimate rules (`argmax`/`cliff`) carry the result. This is the
**K-vs-confidence tradeoff made concrete**: CI-gated attribution needs either more rollouts (higher
K, more compute) or a lower-variance surrogate to fire on real, noisy traces. It is a paper-worthy
result in its own right — and the reason the synthetic-SCM validation (where the surrogate IS the
ground-truth policy and CI gating works) remains the trustworthy core, with the benchmark reported
as a surrogate-bounded application.

**Validity threats, as reported (per §5):** (1) surrogate ≠ original 2024-era policies — these are
*surrogate*-counterfactual effects; the 99% factual-replay sanity floor bounds gross infidelity but
not directional bias. (2) human "decisive error step" ≠ counterfactual locus (MP-Bench) — hence the
±1/±3 tolerance curves alongside exact match. (3) GT-grounded env simulation and a w/-GT judge are
disclosed design choices that make the w/-GT setting tractable; the w/o-GT setting was not run.
