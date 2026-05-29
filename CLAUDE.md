# Project: causal-agent-replay (CAR)

## What this is
An open-source engine for counterfactual replay and CAUSAL ATTRIBUTION over LLM-agent
trajectories: given a run that went wrong, find which step actually caused it by
intervening on it and measuring whether the outcome distribution changes. See PLAN.md §1
for the SCM frame — read it before touching the attribute/ or replay/ modules.

This is a frontier OPEN ARTIFACT, not a product. No business logic, no auth, no dashboard.
The bar is: genuinely hard, correct, and impressive.

## The hard core
src/car/attribute/ (causal attribution) and src/car/replay/intervene.py (the do() algebra).
Everything else is scaffolding to make these demonstrable. Spend care there.

## Non-negotiables
- Counterfactual replay yields a DISTRIBUTION over outcomes, never a single path. The policy
  is stochastic; reason over distributions with confidence intervals everywhere.
- Faithful state reconstruction is the foundation: prove deterministic replay before any
  counterfactual work (test_deterministic_replay.py).
- Validate attribution against synthetic SCMs with KNOWN ground truth (§7) — never trust a
  heatmap that hasn't been checked against a case where you know the right answer.
- Python 3.12, asyncio, pydantic v2, mypy --strict on src/.
- Each phase starts with a research checkpoint in RESEARCH/.

## Don't
- Don't turn this into a trace viewer / observability tool. Intervention + causation is the point.
- Don't add framework adapters until PLAN §12 (after the core works on the native tool-loop).
- Don't report effect sizes without Monte-Carlo confidence intervals.
- Don't run Shapley by default — it's budget-bounded and on-demand.

## Dev commands
- Install: `uv sync --extra dev`
- Test: `uv run pytest`
- Types: `uv run mypy`
- Lint: `uv run ruff check src tests` / `uv run ruff format`
