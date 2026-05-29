# causal-agent-replay

**Find which step actually caused your agent to fail — by intervening on it and measuring whether the outcome changes.**

When an LLM agent does something wrong — issues a refund it shouldn't have, calls the wrong
tool, leaks data — observability tools (LangSmith, Langfuse) show you *what happened*, and eval
tools (Promptfoo) score *pass/fail*. Neither answers the question that actually matters for
debugging:

> **Which step *caused* the bad outcome?**

`causal-agent-replay` (CAR) answers it the only principled way: it **intervenes** on a step and
**re-runs the agent forward** to see if the outcome changes. The step where changing the
decision changes the outcome — but changing later steps does not — is the **causal locus**.
Proven by counterfactual, not guessed from a trace.

---

## The idea in one picture

A recorded run is modelled as a **structural causal model** (Pearl):

```
τ = [ s0, (a1,o1), (a2,o2), …, (an,on), y ]
```

- `s_k` — the exact state the agent decided from (system prompt, tools, full message history)
- `a_k` — the action it took (a tool call or a final answer), drawn from the **stochastic** policy π
- `o_k` — the tool result
- `y` — the outcome, scored by a user-supplied function `Y(τ)`

An **intervention** is a `do(·)` on one variable, after which the agent **re-decides everything
downstream**. Because the policy is stochastic, running forward `K` times gives a *distribution*
over outcomes — never a single path. The **causal effect** of step `k` is how much that
distribution shifts versus the observed run. **Attribution** ranks steps by causal effect.

The headline primitive is `do_resample(k)`: change *nothing* except re-draw the action at step
`k` from the *same* policy. If the bad outcome usually disappears when you resample step `k` —
but persists when you resample `k+1` — then `k` is where the agent committed to the failure.

## The intervention algebra

| `do(·)` | meaning | question it answers |
|---|---|---|
| `do_resample` | re-draw `a_k` from the unchanged policy | *how sensitive is the outcome to this step?* |
| `do_action` | force a specific `a_k` | *what if it had done X instead?* |
| `do_observation` | replace the tool result `o_k` | *what if the tool had returned X?* |
| `do_context` | edit the message history at `k` | *what if the prompt hadn't contained X?* |
| `do_policy` | swap the model from `k` forward | *did the model upgrade break it?* |

## Status

Early. Built in phases (see `PLAN.md`); each phase is research-gated and has a binary
done-condition.

- **Phase 0 — faithful recording + deterministic replay.** ← in progress
- Phase 1 — intervention algebra + forward replay
- Phase 2 — distributional outcomes + effect estimators (with confidence intervals)
- Phase 3 — causal attribution (contrastive, then budget-bounded Monte-Carlo Shapley)
- Phase 4 — interactive visualization + technical writeup

## Why this is honest about hard things

- **Counterfactual replay yields a distribution, not a path.** The policy is stochastic; every
  effect is an estimate reported *with Monte-Carlo confidence intervals*.
- **Providers are not deterministic** — even at temperature 0, and current Claude Opus models
  don't accept a temperature at all. CAR does not pretend otherwise: it *measures* the
  action-match rate of replay and reports residual nondeterminism as a metric. See
  `RESEARCH/phase_0_foundations.md`.
- **Attribution is validated against synthetic SCMs with known ground truth**, not just against
  plausible-looking heatmaps. A method that can't recover a pivotal step you planted isn't
  trusted.

## Install

```bash
pip install causal-agent-replay     # not yet published
# or, from source:
uv sync --extra dev
```

## Intellectual lineage

Pearl's do-calculus and SCMs · counterfactual credit assignment in RL (Mesnard et al.) · causal
influence diagrams for agents (Everitt, Carey et al.) · Shapley attribution · record-replay /
time-travel debuggers (rr, Pernosco). The novel intersection — causal credit assignment applied
to LLM-agent traces as a practical debugging tool — is what CAR is.

## License

Apache-2.0
