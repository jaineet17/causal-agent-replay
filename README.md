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

## See it

![Attribution report: the causal locus is the decision step, with confidence intervals and the injection visible in context.](docs/assets/demo.png)

```bash
uv sync --extra dev
uv run python scripts/make_demo.py     # -> examples/demo_report.html  (open it)
```

The demo report attributes a support agent that absorbed a prompt injection. It shows the
**causal locus is the decision step** — resampling it avoids the bad refund ~half the time,
resampling the already-committed steps never does — with the injection visible in the exact
context that step decided from, an attribution chart with confidence intervals, and a Shapley
toggle. (`docs/writeup.md` explains the method; `examples/gallery.md` runs the engine on a live
local model.)

## Status

Built in phases (see `PLAN.md`); each phase is research-gated with a binary done-condition.
Phases 0–4 are complete and validated against synthetic SCMs with known ground truth.

- **Phase 0 — faithful recording + deterministic replay.** ✅
- **Phase 1 — intervention algebra + forward replay.** ✅
- **Phase 2 — distributional outcomes + effect estimators (with confidence intervals).** ✅
- **Phase 3 — causal attribution (contrastive + budget-bounded Monte-Carlo Shapley).** ✅
- **Phase 4 — interactive visualization + technical writeup.** ✅
- **Phase 5 — framework adapters: LangGraph/LangChain + OpenAI Agents SDK.** ✅

Record *your* agent and attribute its failures — same one-line wrap on either framework:

| framework | extra | wrap |
|---|---|---|
| LangGraph / LangChain 1.x | `causal-agent-replay[langgraph]` | `create_agent(..., middleware=[LangGraphRecorder()])` |
| OpenAI Agents SDK | `causal-agent-replay[openai-agents]` | `Agent(..., model=OpenAIAgentsRecorder(model))` |

Both produce a CAR `Trajectory` held to the same `verify_reconstruction` faithfulness invariant
as the native recorder; counterfactual replay and attribution then run through the unchanged
core, re-executing *your* tools live on the branches.

### Example: LangGraph / LangChain 1.x

```python
from langchain.agents import create_agent
from car.adapters.langgraph import LangGraphRecorder, LangChainPolicy, LangChainToolEnvironment
from car.record.recorder import codec_for

recorder = LangGraphRecorder()                      # an AgentMiddleware
agent = create_agent(model, tools, system_prompt=..., middleware=[recorder])
await agent.ainvoke({"messages": [HumanMessage("...")]})

trajectory = recorder.trajectory("my-run")
result = await contrastive_attribution(
    trajectory,
    policy=LangChainPolicy(model),                  # resample YOUR model
    environment=LangChainToolEnvironment(tools),    # rerun YOUR tools on counterfactual branches
    codec=codec_for("langchain"),
    outcome_fn=..., bad_label=...,
)
print(result.causal_locus)
```

The OpenAI Agents SDK adapter is identical in shape (`OpenAIAgentsRecorder(model)` as the agent's
`model`, then `OpenAIAgentsPolicy` / `OpenAIAgentsToolEnvironment`). Scope for both (v1, refused
loudly rather than mis-recorded): single-agent tool loops, one tool call per turn, string-returning
tools. See `RESEARCH/phase_5_adapters.md` for the faithfulness analysis.

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

## Quickstart — free, no API key (local Ollama)

CAR runs against any backend behind the `Policy` protocol, so you can record and replay agents
with a **free local model** — no spend. [Ollama](https://ollama.com) exposes an OpenAI-compatible
endpoint, so one policy covers it (and Groq / OpenRouter / vLLM / LM Studio too).

```bash
ollama serve &                 # start the local server
ollama pull llama3.1:8b        # a tool-capable model (qwen2.5:7b also works)

# record the support-agent demo against the local model, then replay it
uv run python scripts/record.py --backend ollama --model llama3.1:8b
uv run car replay support-injection-demo
```

Local seeded inference (`seed` + `temperature=0` + a fixed `num_ctx`) reproduces *far* more
reliably than hosted APIs — a real advantage for faithful replay. Hosted providers stay
supported (`--backend anthropic` with `ANTHROPIC_API_KEY`, or any OpenAI-compatible endpoint via
`OPENAI_BASE_URL`), and their nondeterminism is measured and reported, not hidden. See
[`RESEARCH/phase_0_foundations.md`](RESEARCH/phase_0_foundations.md).

## Intellectual lineage

Pearl's do-calculus and SCMs · counterfactual credit assignment in RL (Mesnard et al.) · causal
influence diagrams for agents (Everitt, Carey et al.) · Shapley attribution · record-replay /
time-travel debuggers (rr, Pernosco). The novel intersection — causal credit assignment applied
to LLM-agent traces as a practical debugging tool — is what CAR is.

## License

Apache-2.0
