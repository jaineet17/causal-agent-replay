"""Surrogate world-model for static-log replay: agent surrogate + env simulator + outcome judge.

The PLAN.md s12 "learned surrogate policy" made concrete (design: RESEARCH/phase_6_benchmark.md):
an LLM conditioned on the logged visible prefix stands in for the original (unavailable) agents
(zero-shot role conditioning — the de-facto standard per FAMAS/AEGIS); a second role simulates
environment outputs on counterfactual branches (the ToolEmu pattern); and an outcome judge labels
whether a continuation still fails, anchored on the ground-truth answer (the low-noise "w/ GT"
setting).

Everything is async and provider-agnostic via ``ChatFn``; ``ollama_chat`` gives the free local
default. Attribution results are explicitly *surrogate-counterfactual* attribution — the surrogate
fidelity gap is reported, never hidden (see the validity-threats section of the research note).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

from car.bench.whowhen import LogStep, WhoWhenInstance

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    ChatFn = Callable[[str, str], Awaitable[str]]

log = structlog.get_logger(__name__)


@runtime_checkable
class LogWorldModel(Protocol):
    """What attribution needs: regenerate a step, simulate env output, judge an outcome."""

    async def next_message(
        self, instance: WhoWhenInstance, prefix: list[LogStep], speaker: str
    ) -> str: ...

    async def env_message(
        self, instance: WhoWhenInstance, prefix: list[LogStep], speaker: str
    ) -> str: ...

    async def still_fails(self, instance: WhoWhenInstance, transcript: list[LogStep]) -> bool: ...


def ollama_chat(model: str) -> ChatFn:
    """A ChatFn against any OpenAI-compatible endpoint (default: local Ollama — free)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    )

    async def chat(system: str, user: str) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.9,  # the surrogate must be STOCHASTIC: resampling needs variation
            max_tokens=700,
        )
        return response.choices[0].message.content or ""

    return chat


def _transcript(steps: list[LogStep], *, last: int | None = None) -> str:
    rows = steps if last is None else steps[-last:]
    return "\n\n".join(f"[{s.index}] {s.agent}: {s.content[:1200]}" for s in rows)


class LLMWorldModel:
    """The default LogWorldModel over a single ChatFn (separate prompts per role)."""

    def __init__(self, chat: ChatFn, *, judge_chat: ChatFn | None = None) -> None:
        self._chat = chat
        self._judge_chat = judge_chat or chat

    async def next_message(
        self, instance: WhoWhenInstance, prefix: list[LogStep], speaker: str
    ) -> str:
        role_prompt = instance.system_prompts.get(
            speaker, f"You are {speaker}, an agent in a multi-agent system solving a task."
        )
        system = (
            f"{role_prompt}\n\nYou are {speaker}. Continue the collaboration: write ONLY your "
            f"next message, exactly as {speaker} would (no preamble, no narration)."
        )
        user = (
            f"TASK:\n{instance.question[:2000]}\n\nCONVERSATION SO FAR:\n"
            f"{_transcript(prefix, last=12)}\n\nYour next message as {speaker}:"
        )
        return (await self._chat(system, user)).strip()

    async def env_message(
        self, instance: WhoWhenInstance, prefix: list[LogStep], speaker: str
    ) -> str:
        system = (
            f"You simulate the output of {speaker} (an execution environment / terminal / user). "
            "Given the conversation, produce ONLY the realistic raw output it would return "
            "(e.g. exit codes, code output, file contents). No commentary."
        )
        user = (
            f"TASK CONTEXT:\n{instance.question[:1200]}\n\nCONVERSATION SO FAR:\n"
            f"{_transcript(prefix, last=8)}\n\nThe {speaker} output now:"
        )
        return (await self._chat(system, user)).strip()

    async def still_fails(self, instance: WhoWhenInstance, transcript: list[LogStep]) -> bool:
        """w/-GT outcome label: does the continuation reach the known-correct answer?"""
        gt = instance.ground_truth.strip()
        # Cheap deterministic pass first: the GT answer literally appearing in the tail is a
        # strong success signal for GAIA-style short answers.
        tail_text = " ".join(s.content for s in transcript[-4:])
        if gt and len(gt) >= 2 and gt.lower() in tail_text.lower():
            return False  # reached the right answer -> no longer fails
        system = (
            "You grade a multi-agent transcript. Answer with exactly one word: SOLVED if the "
            "agents' final answer matches the reference answer, FAILED otherwise."
        )
        user = (
            f"QUESTION:\n{instance.question[:1500]}\n\nREFERENCE ANSWER:\n{gt}\n\n"
            f"TRANSCRIPT END:\n{_transcript(transcript, last=6)}\n\nOne word (SOLVED/FAILED):"
        )
        verdict = (await self._judge_chat(system, user)).strip().upper()
        return "SOLVED" not in verdict


def speaker_schedule(instance: WhoWhenInstance) -> list[tuple[str, bool]]:
    """The factual (speaker, is_env) schedule — held fixed under intervention (PLAN.md s1:
    we intervene on message CONTENT; who speaks when is part of the recorded structure)."""
    return [(s.agent, s.is_env) for s in instance.history]
