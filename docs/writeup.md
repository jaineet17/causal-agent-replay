---
title: "Causal Agent Replay — technical writeup"
---

# Causal Agent Replay: counterfactual attribution for LLM-agent failures

*A technical writeup. Companion to the `causal-agent-replay` package.*

## The question

An LLM agent does something wrong — issues a refund it shouldn't have, calls the wrong tool,
leaks data. You have the trace. **Which step actually caused the failure?**

Observability tools (LangSmith, Langfuse) show you *what happened*. Eval tools (Promptfoo) score
*pass/fail*. Neither answers the causal question, and the obvious heuristics are wrong:

- *"The step that did the bad thing"* — `issue_refund` was called at step 3, but the **decision**
  was made at step 2; step 3 is just the mechanical consequence. Blaming step 3 fixes nothing.
- *"Ask an LLM judge which step is to blame"* — this is correlational pattern-matching over the
  text, and it is unreliable: the strongest public benchmark (Who&When, ICML 2025) reports
  state-of-the-art **step-level** attribution accuracy of only ~14%.

The principled answer is causal: **intervene** on a step and see whether the outcome changes.

## The frame: an agent run is a structural causal model

We model a run as an SCM (Pearl). A trajectory is

```
τ = [ s0, (a1,o1), (a2,o2), …, (an,on), y ]
```

where `s_k` is the exact state the agent decided from (system prompt, tools, full message
history), `a_k ~ π(·|s_k)` is the action drawn from the **stochastic** policy (the LLM), `o_k` is
the tool result, and `y = Y(τ)` is an outcome score. Modeling agents this way follows the causal
influence diagram program (Everitt & Carey, *Agent Incentives*, AAAI 2021; Kenton et al.,
*Discovering Agents*, AIJ 2023): the policy is a causal mechanism, and effects are read off
interventions, not off the surface trace.

An **intervention** is a `do(·)` on one variable, after which the agent **re-decides everything
downstream**. The key consequence of the policy being stochastic: an intervention does not
produce *a* trajectory, it produces a **distribution** over trajectories. Everything downstream
reasons over outcome distributions with confidence intervals — collapsing this to a single path
misses the point.

CAR implements five interventions: `do_resample` (re-draw `a_k` from the same policy — the null
intervention), `do_action` (force `a_k`), `do_observation` (replace `o_k`), `do_context` (edit the
history at `k`), `do_policy` (swap the model from `k` on).

## Faithful replay first

You cannot intervene at step `k` if you cannot reconstruct the exact state there. So Phase 0 is
*faithful deterministic replay*, validated before anything counterfactual is built — the
record-replay debugger discipline (rr, Pernosco) adapted to the LLM setting: **record every
nondeterministic input** so deterministic glue can be re-executed, and the one irreducible
nondeterministic input (the model call) is recorded as ground truth and its replay *measured*.

Providers are not deterministic — even at temperature 0, and current Claude Opus models reject a
temperature parameter entirely. CAR does not pretend otherwise: it reports the **action-match
rate** of replay and the residual nondeterminism, rather than asserting reproducibility. (On a
free local model with a fixed seed, replay is in fact exact — 20/20 in our gallery — because a
single local stream sidesteps the batch-variance that makes hosted inference nondeterministic.)

## Attribution

**Contrastive (single-step) attribution.** For each step `k`, hold `[0,k)` at their factual
actions, `do_resample(k)`, and run forward `K` times. Estimate `P(bad | resample at k)` and its
shift from the observed run, with confidence intervals.

The subtlety — and the thing that makes naive versions wrong — is that under run-forward,
resampling step `k` also re-rolls *every downstream stochastic step*. So an early, irrelevant step
shows an effect too (it re-rolls the true pivotal step). Magnitude alone cannot localize the
cause. The fix: the **causal locus is the latest step whose effect's CI still excludes zero** —
the last point at which re-deciding still rescues the run. Beyond it, the outcome is committed.
This is the COMA/Mesnard counterfactual-baseline idea (resample one action, hold the rest)
specialized to a single-stream agent.

**Shapley attribution.** Contrastive treats steps independently, so it cannot express
*interactions* — two steps that only jointly cause a failure. Consider an AND-failure where the
outcome is bad only if both step `i` and step `j` go wrong: holding the other bad makes each look
fully responsible (effects sum to ~2, double-counting); holding the other resampled makes each
look irrelevant (effects ~0). Neither is the truth, which is *shared* responsibility. The Shapley
value resolves this by averaging each step's marginal contribution over all coalition contexts,
and by the efficiency axiom the values sum to the total effect. For the AND-failure it returns the
0.5/0.5 split.

CAR estimates Shapley by Monte-Carlo permutation sampling (ApproShapley) with antithetic
reverse-permutation pairing, budget-bounded with a circuit breaker. Coalition values are *not*
cached across permutations, so the per-step marginal contributions stay i.i.d. and the CLT
confidence intervals are honest (caching would collapse the variance and report false certainty).
Truncated Monte-Carlo Shapley is deliberately avoided: truncation can skip a pivotal *late* step.

## Validation against ground truth

The credibility of an attribution tool rests entirely on whether it is right on cases where the
answer is known. CAR ships synthetic SCMs with planted causal structure and tests against them:

- a **pivotal-step** SCM (step 1 is the decision): contrastive recovers `causal_locus == 1`, and
  resampling *downstream* of it shows no significant effect.
- a **two-step interaction** SCM (bad only if both steps fail): Shapley recovers
  `φ₀ = 0.44, φ₁ = 0.45, φ₂ ≈ 0`, summing to 0.909 vs the theoretical `1 − q² = 0.91`
  (efficiency); contrastive over-counts the same case — a concrete demonstration of why both
  methods ship.

These are not illustrations; they are the regression tests. A heatmap that has not been checked
this way is exactly the failure mode that makes attribution untrustworthy.

## How CAR relates to recent work

Failure attribution for agents became active in 2025–26. CAR is deliberately *not* claiming
novelty on "counterfactual replay" or "Shapley for agent blame" in the abstract — both appear in
recent work (AgenTracer, arXiv:2509.03312, replays with **oracle substitution** and trains a
scorer; Ma et al., arXiv:2509.08682, pair Shapley with **causal discovery over static logs**;
Who&When, arXiv:2505.00212, uses LLM judges). What CAR combines that those do not:

1. **executed same-policy `do_resample` interventions** (not oracle substitution, not an LLM
   judging the transcript);
2. **distributional outcomes with confidence intervals** throughout;
3. a principled **point-of-commitment** locus rule that handles the run-forward confound;
4. **Shapley credit-splitting** for interactions; and
5. **ground-truth validation** against synthetic SCMs.

## Limitations (honest)

- The contrastive effect is a *total* effect through a stochastic continuation; isolating a step's
  direct effect wants common random numbers across branches, which is hard across divergent LLM
  contexts (a documented refinement, not yet implemented).
- Judge-based outcomes add their own noise; prefer rule-based outcomes for anything you want to
  trust.
- Real tools with side effects are out of scope for now (the demo and tests use mocked,
  reproducible tools).
- Shapley is exponential in the worst case; the Monte-Carlo estimator is budget-bounded, and the
  variance-vs-budget tradeoff is real.

## Try it

```bash
uv sync --extra dev
uv run python scripts/make_demo.py          # -> examples/demo_report.html
```

The report shows, for a support agent that absorbed a prompt injection, that the *decision* step
is the causal locus — resampling it avoids the refund ~half the time, resampling the
already-committed steps never does — with the injection visible in the very context that step
decided from. See `examples/gallery.md` for the same engine running on a live local model.
