"""Pilot: surrogate-counterfactual attribution on real Who&When instances, free local model.

    uv run python scripts/whowhen_pilot.py --n 5 --model llama3.2:latest

Downloads the first N Algorithm-Generated instances (cached under data/whowhen/), runs
``attribute_log`` with the local Ollama surrogate, and prints predictions vs labels plus the
sanity floors. This is the plumbing-validation step before any full-subset run; numbers from a
handful of instances are NOT results.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from car.bench.attribute_log import attribute_log  # noqa: E402
from car.bench.surrogate import LLMWorldModel, ollama_chat  # noqa: E402
from car.bench.whowhen import fetch_subset  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n: int = typer.Option(5, help="Instances to run (first N by id)."),
    model: str = typer.Option("llama3.2:latest", help="Ollama model for surrogate+judge."),
    k_max: int = typer.Option(8),
    horizon: int = typer.Option(0, help="Roll-forward horizon (0 = to the end of the log)."),
    concurrency: int = typer.Option(4),
) -> None:
    instances = fetch_subset("Algorithm-Generated", REPO_ROOT / "data" / "whowhen", limit=n)
    world = LLMWorldModel(ollama_chat(model))

    async def _run() -> None:
        agent_hits = step_hits = step_pm3 = factual_ok = 0
        for inst in instances:
            result = await attribute_log(
                inst,
                world,
                k_max=k_max,
                horizon=horizon or None,
                max_concurrency=concurrency,
            )
            agent_hits += result.agent_correct
            step_hits += result.step_within(0)
            step_pm3 += result.step_within(3)
            factual_ok += result.factual_still_fails
            typer.echo(
                f"{inst.instance_id}: pred=({result.predicted_agent}, {result.predicted_step}) "
                f"label=({result.label_agent}, {result.label_step}) "
                f"agent={'Y' if result.agent_correct else 'n'} "
                f"step={'Y' if result.step_within(0) else 'n'} "
                f"rollouts={result.total_rollouts} factual_fails={result.factual_still_fails}"
            )
        n_run = len(instances)
        typer.echo(
            f"\npilot ({n_run} instances): agent {agent_hits}/{n_run}  step {step_hits}/{n_run}  "
            f"step±3 {step_pm3}/{n_run}  factual-replay-sanity {factual_ok}/{n_run}"
        )

    asyncio.run(_run())


if __name__ == "__main__":
    app()
