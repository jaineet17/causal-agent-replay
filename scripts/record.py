"""Record a run of the support-agent demo fixture against a live Anthropic model.

    uv run python scripts/record.py --model claude-opus-4-8

Requires ANTHROPIC_API_KEY. The recorded trajectory is saved to the store; replay it with:

    uv run car replay support-injection-demo

This is the Phase 0 real-provider path. The hard correctness guarantees are proven on the
synthetic fixtures (tests/); this script demonstrates faithful capture on a real model and lets
you measure its action-match rate honestly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.support_agent import (  # noqa: E402
    INJECTION_USER_MESSAGE,
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    build_registry,
)

from car.config import Settings  # noqa: E402
from car.record.recorder import (  # noqa: E402
    AnthropicCodec,
    AnthropicPolicy,
    MockEnvironment,
    record_run,
)
from car.store.store import TrajectoryStore  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option("claude-opus-4-8", help="Anthropic model id."),
    trajectory_id: str = typer.Option("support-injection-demo"),
    max_tokens: int = typer.Option(1024),
) -> None:
    settings = Settings.from_env()
    registry = build_registry()
    policy = AnthropicPolicy(model, max_tokens=max_tokens)

    async def _run() -> None:
        traj = await record_run(
            trajectory_id=trajectory_id,
            policy=policy,
            environment=MockEnvironment(registry),
            codec=AnthropicCodec(),
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=TOOL_SCHEMAS,
            user_input=INJECTION_USER_MESSAGE,
        )
        store = TrajectoryStore(db_path=settings.db_path)
        try:
            path = store.save(traj)
        finally:
            store.close()
        typer.echo(f"recorded {len(traj.steps)} steps -> {path}")
        typer.echo(f"final output: {traj.final_output[:200]}")
        for step in traj.steps:
            a = step.action
            sig = "final" if a.kind == "final" else f"{a.tool_name}({a.tool_args})"
            typer.echo(f"  step {step.index}: {sig}")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
