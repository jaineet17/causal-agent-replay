"""Full Who&When evaluation run — detached, resumable, per-instance JSONL.

    nohup caffeinate -i uv run python scripts/whowhen_run.py > data/whowhen/run.log 2>&1 &

Appends one JSON line per instance to data/whowhen/results_ag.jsonl and skips instances already
present, so the run can be killed and restarted at any time at zero cost. Errors are recorded as
result lines (never silently skipped). Progress: tail data/whowhen/run.log
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
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
    model: str = typer.Option("llama3.2:latest"),
    k_max: int = typer.Option(8),
    concurrency: int = typer.Option(4, help="In-flight rollouts (keep moderate: shared machine)."),
    results: Path = typer.Option(REPO_ROOT / "data" / "whowhen" / "results_ag.jsonl"),
    limit: int = typer.Option(0, help="0 = all instances."),
    per_instance_minutes: float = typer.Option(
        30.0,
        help="Hard wall-clock cap per instance; abandons one wedged by inference degradation "
        "(recorded as a timeout row) so it can never eat the whole run.",
    ),
    horizon: int = typer.Option(
        5,
        help="Forward-rollout horizon: resample step k, roll m steps, then judge. Caps the "
        "O(n_steps^2) blow-up (RESEARCH s4 lever; AgentDebug: outcomes flip within a few steps). "
        "0 = roll to the end of the log.",
    ),
    surrogate_max_tokens: int = typer.Option(
        250, help="Per surrogate generation (700 was ~12s/call on a local 3B; 250 is ~5s)."
    ),
) -> None:
    instances = fetch_subset(
        "Algorithm-Generated", REPO_ROOT / "data" / "whowhen", limit=limit or None
    )
    results.parent.mkdir(parents=True, exist_ok=True)
    # Dedupe to the LAST row per instance; an instance is "done" only if its latest outcome is a
    # success or a genuine timeout (inference wedged) — transient connection/API errors are
    # retried on resume rather than permanently skipped.
    last_row: dict[str, dict] = {}
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                row = json.loads(line)
                last_row[row["instance_id"]] = row
            except (json.JSONDecodeError, KeyError):
                continue

    def is_done(row: dict) -> bool:
        err = row.get("error", "")
        return "error" not in row or err.startswith("timeout:")

    done = {iid for iid, row in last_row.items() if is_done(row)}
    todo = [i for i in instances if i.instance_id not in done]
    typer.echo(f"{len(instances)} instances; {len(done)} done; {len(todo)} to run")

    world = LLMWorldModel(
        ollama_chat(model, max_tokens=surrogate_max_tokens),
        judge_chat=ollama_chat(model, temperature=0.0, max_tokens=80),
    )

    async def _run() -> None:
        agent_hits = step_hits = pm3 = sane = 0
        for n_done, inst in enumerate(todo, start=1):
            started = time.time()
            try:
                r = await asyncio.wait_for(
                    attribute_log(
                        inst,
                        world,
                        k_max=k_max,
                        max_concurrency=concurrency,
                        horizon=horizon or None,
                    ),
                    timeout=per_instance_minutes * 60.0,
                )
                row = {
                    "instance_id": r.instance_id,
                    "predicted_agent": r.predicted_agent,
                    "predicted_step": r.predicted_step,
                    "label_agent": r.label_agent,
                    "label_step": r.label_step,
                    "confident": r.prediction_confident,
                    "factual_still_fails": r.factual_still_fails,
                    "agent_correct": r.agent_correct,
                    "step_exact": r.step_within(0),
                    "step_pm1": r.step_within(1),
                    "step_pm3": r.step_within(3),
                    "rollouts": r.total_rollouts,
                    "elapsed_s": round(time.time() - started, 1),
                    "per_step": [
                        {
                            "i": se.index,
                            "agent": se.agent,
                            "env": se.is_env,
                            "p_fail": se.p_fail_after_resample,
                            "ci": [se.effect.ci_low, se.effect.ci_high],
                            "k": se.k_realized,
                        }
                        for se in r.per_step
                    ],
                }
                agent_hits += r.agent_correct
                step_hits += r.step_within(0)
                pm3 += r.step_within(3)
                sane += r.factual_still_fails
            except TimeoutError:
                row = {
                    "instance_id": inst.instance_id,
                    "error": f"timeout: exceeded {per_instance_minutes} min (inference wedged)",
                    "elapsed_s": round(time.time() - started, 1),
                }
            except Exception as exc:  # record, never silently skip (PLAN.md s0.9)
                row = {
                    "instance_id": inst.instance_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.time() - started, 1),
                }
            with results.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            typer.echo(
                f"[{n_done}/{len(todo)}] {inst.instance_id} "
                f"agent={agent_hits}/{n_done} step={step_hits}/{n_done} "
                f"pm3={pm3}/{n_done} sane={sane}/{n_done} ({row.get('elapsed_s')}s)",
                err=False,
            )

    asyncio.run(_run())


if __name__ == "__main__":
    app()
