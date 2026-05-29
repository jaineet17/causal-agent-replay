"""The ``car`` CLI (PLAN.md tech stack: typer).

Phase 0 surface: inspect config, and replay a stored trajectory to measure faithfulness.
Recording the demo agent lives in ``scripts/record.py`` (it imports the example fixture);
attribution commands arrive with Phase 3.
"""

from __future__ import annotations

import asyncio

import typer

from car.config import Settings
from car.record.recorder import codec_for, policy_for
from car.replay.deterministic import DeterministicReplay
from car.store.store import TrajectoryStore

app = typer.Typer(
    add_completion=False,
    help="Counterfactual replay + causal attribution for agents.",
)


@app.command()
def info() -> None:
    """Show resolved configuration."""
    s = Settings.from_env()
    typer.echo("causal-agent-replay")
    for k, v in s.model_dump().items():
        typer.echo(f"  {k}: {v}")


@app.command()
def replay(
    trajectory_id: str,
    n_samples: int = typer.Option(8, help="Re-issue samples for the action-match measurement."),
    db_path: str = typer.Option(None, help="Override DB_PATH."),
) -> None:
    """Replay a stored trajectory and print the faithfulness report (action-match rate)."""
    settings = Settings.from_env()
    store = TrajectoryStore(db_path=db_path or settings.db_path)
    try:
        traj = store.load(trajectory_id)
    finally:
        store.close()

    provider = traj.steps[0].state_before.provider
    model = traj.steps[0].state_before.model
    replayer = DeterministicReplay(codec_for(provider))
    policy = policy_for(provider, model)

    report = asyncio.run(replayer.measure(traj, policy, n_samples=n_samples))
    typer.echo(f"trajectory: {report.trajectory_id}  provider={provider} model={model}")
    typer.echo(f"reconstruction_faithful: {report.reconstruction_faithful}")
    typer.echo(f"sequence_reproduction_rate: {report.sequence_reproduction_rate:.3f}")
    typer.echo(
        f"mean_step_match_rate: {report.mean_step_match_rate:.3f} (over {report.n_samples} samples)"
    )
    for s in report.per_step:
        typer.echo(f"  step {s.index}: match={s.match_rate:.3f}  recorded={s.recorded_signature}")
    typer.echo(report.notes)


if __name__ == "__main__":  # pragma: no cover
    app()
