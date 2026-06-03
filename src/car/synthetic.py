"""In-process synthetic policies + environment with KNOWN structure (PLAN.md s7).

These are part of the public API on purpose: they let you build small agents whose causal
structure you control, so you can validate attribution against ground truth before trusting it on
a real run — the discipline the whole project rests on (s13). They also drive the reproducible
demo (``scripts/make_demo.py``) without any API calls.

All run against the same ``car.replay`` / ``car.attribute`` code paths as a live LLM policy.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from car.schemas.trajectory import Action, Observation, Provider, State


# --- state helpers ----------------------------------------------------------------------------
def turn_index(state: State) -> int:
    """Which step (0-based) the policy is deciding now = how many assistant turns have happened."""
    return sum(1 for m in state.messages if m.get("role") == "assistant")


def user_text(state: State) -> str:
    """The first user message's text (demos use a string initial message)."""
    for m in state.messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return str(m["content"])
    return ""


def last_tool_result(state: State) -> str:
    """The most recent tool observation visible in the context (SyntheticCodec encoding)."""
    for m in reversed(state.messages):
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            return str(m["content"])
    return ""


# --- action constructors ----------------------------------------------------------------------
def tool_call(name: str, args: dict[str, Any], text: str | None = None) -> Action:
    return Action(
        kind="tool_call",
        text=text,
        tool_name=name,
        tool_args=args,
        raw={"synthetic": True, "kind": "tool_call", "tool_name": name},
    )


def final(text: str) -> Action:
    return Action(kind="final", text=text, raw={"synthetic": True, "kind": "final"})


# --- policies ---------------------------------------------------------------------------------
class ScriptedPolicy:
    """Emits a fixed sequence of actions indexed by step. Fully deterministic."""

    provider: Provider = "synthetic"

    def __init__(self, actions: list[Action], model_id: str = "synthetic:scripted") -> None:
        self._actions = actions
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    async def sample(self, state: State) -> Action:
        idx = turn_index(state)
        if idx >= len(self._actions):
            return final("done")
        return self._actions[idx]


class RulePolicy:
    """A deterministic policy whose action is an arbitrary function of the current state.

    Build agents with KNOWN causal dependence (e.g. "step 1's action depends on step 0's
    observation") so an intervention's downstream effect can be asserted against ground truth.
    """

    provider: Provider = "synthetic"

    def __init__(self, decide: Callable[[State], Action], model_id: str = "synthetic:rule") -> None:
        self._decide = decide
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    async def sample(self, state: State) -> Action:
        return self._decide(state)


class NoisyPolicy:
    """Deterministic except at ``noisy_step``, where it picks ``option_a`` with probability ``p``.

    Seeded per-draw so a whole replay/attribution pass is reproducible while still producing a
    genuine distribution of outcomes across the K samples it draws.
    """

    provider: Provider = "synthetic"

    def __init__(
        self,
        base_actions: list[Action],
        *,
        noisy_step: int,
        option_a: Action,
        option_b: Action,
        p: float = 0.5,
        seed: int = 0,
    ) -> None:
        self._base = base_actions
        self._noisy_step = noisy_step
        self._option_a = option_a
        self._option_b = option_b
        self._p = p
        self._seed = seed
        self._draws = 0

    @property
    def model_id(self) -> str:
        return "synthetic:noisy"

    async def sample(self, state: State) -> Action:
        idx = turn_index(state)
        if idx == self._noisy_step:
            rng = random.Random(self._seed * 1_000_003 + self._draws)
            self._draws += 1
            return self._option_a if rng.random() < self._p else self._option_b
        if idx >= len(self._base):
            return final("done")
        return self._base[idx]


class MultiNoisyPolicy:
    """Independent stochastic choices at several steps — for two-step-interaction SCMs.

    ``noisy`` maps a step index to (option_a, option_b, p). Other steps fall back to
    ``base_actions`` (or a final action). Draws are seeded per-draw for reproducibility.
    """

    provider: Provider = "synthetic"

    def __init__(
        self,
        base_actions: list[Action],
        noisy: dict[int, tuple[Action, Action, float]],
        *,
        seed: int = 0,
    ) -> None:
        self._base = base_actions
        self._noisy = noisy
        self._seed = seed
        self._draws = 0

    @property
    def model_id(self) -> str:
        return "synthetic:multinoisy"

    async def sample(self, state: State) -> Action:
        idx = turn_index(state)
        if idx in self._noisy:
            option_a, option_b, p = self._noisy[idx]
            rng = random.Random(self._seed * 1_000_003 + self._draws)
            self._draws += 1
            return option_a if rng.random() < p else option_b
        if idx >= len(self._base):
            return final("done")
        return self._base[idx]


class DictEnvironment:
    """Returns a fixed result per tool name; deterministic and side-effect-free."""

    def __init__(self, results: dict[str, str]) -> None:
        self._results = results

    async def observe(self, action: Action) -> Observation:
        if action.tool_name is None:
            raise ValueError("DictEnvironment cannot observe an action without a tool_name")
        result = self._results.get(action.tool_name, "{}")
        return Observation(tool_name=action.tool_name, result=result, source="mocked")
