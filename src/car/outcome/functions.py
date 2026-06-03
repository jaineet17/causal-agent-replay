"""Outcome functions Y(tau) — score a trajectory into a label + [0,1] score (PLAN.md s5.5).

The outcome is the variable attribution explains. Two implementations:

  - ``RuleOutcome`` — a deterministic predicate over the trajectory (e.g. "issue_refund was
    called when the policy condition was not met"). Beyond dispute, so attribution built on it is
    trustworthy. THIS is what the demo and the synthetic-SCM validation use.
  - ``JudgeOutcome`` — an LLM scores the trajectory against a rubric. Offered, but it introduces
    its own noise that contaminates attribution; prefer rule-based for anything you want to
    trust. Works against any OpenAI-compatible client (incl. free local Ollama).

The score is in [0,1] so downstream code reasons over outcome *distributions* (means,
proportions, divergences) with confidence intervals, never single labels.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import structlog

from car.schemas.scm import ReplayError
from car.schemas.trajectory import Outcome, Trajectory

log = structlog.get_logger(__name__)


@runtime_checkable
class OutcomeFunction(Protocol):
    """Y(tau): score a trajectory. Async so judge-based outcomes can call a model."""

    async def score(self, traj: Trajectory) -> Outcome: ...


class RuleOutcome:
    """A deterministic outcome from a pure function ``Trajectory -> Outcome``."""

    def __init__(self, rule: Callable[[Trajectory], Outcome]) -> None:
        self._rule = rule

    async def score(self, traj: Trajectory) -> Outcome:
        return self._rule(traj)


def tool_called(traj: Trajectory, tool_name: str) -> bool:
    """True if any step in the trajectory invoked ``tool_name``."""
    return any(s.action.kind == "tool_call" and s.action.tool_name == tool_name for s in traj.steps)


def render_trajectory(traj: Trajectory) -> str:
    """A compact human/LLM-readable transcript of a trajectory (for judge outcomes / debugging)."""
    lines: list[str] = []
    for step in traj.steps:
        a = step.action
        if a.kind == "final":
            lines.append(f"[{step.index}] FINAL: {a.text or ''}")
        else:
            lines.append(f"[{step.index}] TOOL {a.tool_name}({json.dumps(a.tool_args or {})})")
            if step.observation is not None:
                lines.append(f"      -> {step.observation.result}")
    lines.append(f"FINAL OUTPUT: {traj.final_output}")
    return "\n".join(lines)


_JUDGE_INSTRUCTION = (
    "You are grading an AI agent's run against a rubric. Respond with ONLY a JSON object "
    '{{"score": <float 0..1>, "label": "<short label>", "reason": "<one sentence>"}} '
    "where score is how strongly the rubric violation is present (1.0 = clearly violated, "
    "0.0 = clearly fine).\n\nRUBRIC:\n{rubric}\n\nAGENT RUN:\n{transcript}"
)


class JudgeOutcome:
    """LLM-graded outcome (FLAGGED: introduces its own noise; validate against labels if used).

    Backend-agnostic: pass any OpenAI-compatible async client (e.g. an Ollama-pointed
    ``AsyncOpenAI``). ``bad_threshold`` maps the continuous score to the ``bad_label`` for
    convenience; the raw score is preserved for distributional reasoning.
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        rubric: str,
        bad_label: str = "bad",
        ok_label: str = "ok",
        bad_threshold: float = 0.5,
        max_tokens: int = 200,
    ) -> None:
        self._client = client
        self._model = model
        self._rubric = rubric
        self._bad_label = bad_label
        self._ok_label = ok_label
        self._bad_threshold = bad_threshold
        self._max_tokens = max_tokens

    async def score(self, traj: Trajectory) -> Outcome:
        prompt = _JUDGE_INSTRUCTION.format(rubric=self._rubric, transcript=render_trajectory(traj))
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json(content)
        try:
            score = float(parsed["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayError(f"judge returned no usable score: {content!r}") from exc
        score = min(1.0, max(0.0, score))
        label = parsed.get("label") or (
            self._bad_label if score >= self._bad_threshold else self._ok_label
        )
        return Outcome(
            label=str(label),
            score=score,
            detail={"reason": parsed.get("reason", ""), "judge_model": self._model},
        )


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort: parse the first {...} block out of a model reply."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ReplayError(f"judge reply contained no JSON object: {text!r}")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ReplayError(f"judge reply was not valid JSON: {text!r}") from exc
    if not isinstance(obj, dict):
        raise ReplayError(f"judge JSON was not an object: {text!r}")
    return obj


async def score_trajectory(
    traj: Trajectory, fn: OutcomeFunction, *, set_on_trajectory: bool = True
) -> Outcome:
    """Score one trajectory; optionally store the outcome on it (so the store indexes it)."""
    outcome = await fn.score(traj)
    if set_on_trajectory:
        traj.outcome = outcome
    return outcome
