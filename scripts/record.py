"""Record a run of the support-agent demo fixture against a live model.

Free / local (no API key, no cost) — recommended:

    ollama serve &                     # start the server
    ollama pull llama3.1:8b            # a tool-capable model
    uv run python scripts/record.py --backend ollama --model llama3.1:8b

Anthropic (requires ANTHROPIC_API_KEY):

    uv run python scripts/record.py --backend anthropic --model claude-opus-4-8

Any other OpenAI-compatible endpoint (Groq, OpenRouter, vLLM, ...):

    OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_API_KEY=... \
      uv run python scripts/record.py --backend openai --model llama-3.1-8b-instant

The recorded trajectory is saved to the store; replay it with:

    uv run car replay support-injection-demo

The hard correctness guarantees are proven on the synthetic fixtures (tests/); this script
demonstrates faithful capture on a real model and lets you measure its action-match rate
honestly. Local seeded models (Ollama with seed + temperature=0 + fixed num_ctx) reproduce far
more reliably than hosted APIs — see RESEARCH/phase_0_foundations.md.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

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
    OpenAICodec,
    OpenAICompatiblePolicy,
    ollama_policy,
    record_run,
)
from car.schemas.scm import Policy  # noqa: E402
from car.store.store import TrajectoryStore  # noqa: E402

app = typer.Typer(add_completion=False)


def _build(backend: str, model: str, max_tokens: int) -> tuple[Policy, Any]:
    """Return (policy, codec) for the chosen backend."""
    if backend == "ollama":
        return ollama_policy(model, max_tokens=max_tokens), OpenAICodec()
    if backend == "openai":
        return OpenAICompatiblePolicy(model, max_tokens=max_tokens), OpenAICodec()
    if backend == "anthropic":
        return AnthropicPolicy(model, max_tokens=max_tokens), AnthropicCodec()
    raise typer.BadParameter(f"unknown backend {backend!r} (ollama|openai|anthropic)")


@app.command()
def main(
    backend: str = typer.Option("ollama", help="ollama | openai | anthropic"),
    model: str = typer.Option("llama3.1:8b", help="Model id for the chosen backend."),
    trajectory_id: str = typer.Option("support-injection-demo"),
    seed: int = typer.Option(0, help="Sampling seed (used where the backend supports it)."),
    num_ctx: int = typer.Option(4096, help="Ollama context size; fix it for reproducibility."),
    max_tokens: int = typer.Option(1024),
) -> None:
    settings = Settings.from_env()
    policy, codec = _build(backend, model, max_tokens)

    # Determinism levers: harmless on backends that ignore them; powerful on Ollama.
    sampling: dict[str, Any] = {"seed": seed, "temperature": 0}
    if backend in ("ollama", "openai"):
        sampling["extra_body"] = {"options": {"num_ctx": num_ctx}}

    async def _run() -> None:
        traj = await record_run(
            trajectory_id=trajectory_id,
            policy=policy,
            environment=MockEnvironment(build_registry()),
            codec=codec,
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=TOOL_SCHEMAS,
            user_input=INJECTION_USER_MESSAGE,
            sampling=sampling,
        )
        store = TrajectoryStore(db_path=settings.db_path)
        try:
            path = store.save(traj)
        finally:
            store.close()
        typer.echo(f"[{backend}:{model}] recorded {len(traj.steps)} steps -> {path}")
        typer.echo(f"final output: {traj.final_output[:200]}")
        for step in traj.steps:
            a = step.action
            sig = "final" if a.kind == "final" else f"{a.tool_name}({a.tool_args})"
            typer.echo(f"  step {step.index}: {sig}")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
