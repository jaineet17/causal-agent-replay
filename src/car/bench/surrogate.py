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
import re
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


def ollama_chat(model: str, *, temperature: float = 0.9, max_tokens: int = 700) -> ChatFn:
    """A ChatFn against any OpenAI-compatible endpoint (default: local Ollama — free).

    The surrogate default is hot (resampling needs stochasticity); pass ``temperature=0`` for
    the judge/extraction role, which must be as deterministic as the backend allows.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        # A hung local call must fail fast, not wedge a multi-day run (observed: after a
        # presumed sleep/wake the llama-server degrades and every call crawls, turning a
        # ~40-min instance into 45-73h). Short timeout + single retry; the run script's
        # per-instance wall-clock cap is the hard backstop.
        timeout=60.0,
        max_retries=1,
    )

    async def chat(system: str, user: str) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return chat


def normalize_answer(text: str) -> str:
    """Normalize an answer for comparison: case/whitespace/punctuation-insensitive,
    numeric-aware ('8', '8.0', ' 8 ' all match)."""
    s = text.strip().strip(".:,;!\"'()[]").strip().lower()
    s = re.sub(r"\s+", " ", s)
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def _transcript(steps: list[LogStep], *, last: int | None = None) -> str:
    rows = steps if last is None else steps[-last:]
    return "\n\n".join(f"[{s.index}] {s.agent}: {s.content[:1200]}" for s in rows)


class LLMWorldModel:
    """The default LogWorldModel over a single ChatFn (separate prompts per role)."""

    def __init__(
        self,
        chat: ChatFn,
        *,
        judge_chat: ChatFn | None = None,
        ground_env_with_gt: bool = True,
    ) -> None:
        self._chat = chat
        self._judge_chat = judge_chat or chat
        self._ground_env_with_gt = ground_env_with_gt

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
        """Simulate environment output. Pilot finding (2026-06-11): the logs are partially
        observable (the underlying data — files, pages — is absent), so an ungrounded simulator
        can NEVER produce the true result and nothing rescues. In the w/-GT setting we ground the
        simulator in the true answer (the ToolEmu pattern under partial observability): it may
        emit the correct value ONLY when the preceding agent action would plausibly compute it.
        Disclosed in reporting as GT-grounded environment simulation."""
        grounding = ""
        if self._ground_env_with_gt and instance.ground_truth.strip():
            grounding = (
                f"\n\nGROUND FACT about the true world (for consistency only; NEVER volunteer "
                f"it unprompted): a fully correct solution to the task yields "
                f"{instance.ground_truth.strip()!r}. If the preceding agent action would "
                f"plausibly compute the correct result, your output reflects that value; if the "
                f"action is flawed or computes something else, output what IT would actually "
                f"produce instead."
            )
        system = (
            f"You simulate the output of {speaker} (an execution environment / terminal / user). "
            "Given the conversation, produce ONLY the realistic raw output it would return "
            f"(e.g. exit codes, code output, file contents). No commentary.{grounding}"
        )
        user = (
            f"TASK CONTEXT:\n{instance.question[:1200]}\n\nCONVERSATION SO FAR:\n"
            f"{_transcript(prefix, last=8)}\n\nThe {speaker} output now:"
        )
        return (await self._chat(system, user)).strip()

    async def still_fails(self, instance: WhoWhenInstance, transcript: list[LogStep]) -> bool:
        """w/-GT outcome label: does the continuation reach the known-correct answer?

        Pilot finding (2026-06-11): asking a small judge to *compare* answers is unreliable —
        it graded factual failing logs as solved, breaking the sanity floor. So the LLM does
        only the easy part (EXTRACT the final answer verbatim) and the comparison happens in
        code against the normalized ground truth.
        """
        system = (
            "You read the end of a multi-agent transcript. Reply with ONLY the final answer "
            "the agents committed to for the task (the value itself, no sentence, no units "
            "unless part of the answer). If they never stated a final answer, reply NONE."
        )
        user = (
            f"TASK:\n{instance.question[:1500]}\n\n"
            f"TRANSCRIPT END:\n{_transcript(transcript, last=6)}\n\nFinal answer only:"
        )
        extracted = (await self._judge_chat(system, user)).strip()
        if not extracted or extracted.upper() == "NONE":
            return True  # no final answer -> still failing
        return normalize_answer(extracted) != normalize_answer(instance.ground_truth)


def speaker_schedule(instance: WhoWhenInstance) -> list[tuple[str, bool]]:
    """The factual (speaker, is_env) schedule — held fixed under intervention (PLAN.md s1:
    we intervene on message CONTENT; who speaks when is part of the recorded structure)."""
    return [(s.agent, s.is_env) for s in instance.history]
