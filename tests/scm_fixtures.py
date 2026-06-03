"""Synthetic SCM fixtures — in-process policies with KNOWN behavior (PLAN.md s7).

These validate the replay/attribution machinery against ground truth, with no API calls and no
provider nondeterminism. ``ScriptedPolicy`` is fully deterministic (action-match rate must be
exactly 1.0 — it validates the machinery). ``NoisyPolicy`` injects seeded stochasticity at a
chosen step (action-match rate strictly between 0 and 1 — it validates the distributional
framing, that replay reasons over a distribution of outcomes, not a single path).

Richer fixtures with a known *causal locus* and a *two-step interaction* (for Phase 3
attribution validation) build on these primitives.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from car.schemas.scm import Environment
from car.schemas.trajectory import Action, Observation, Provider, State


def _n_assistant_turns(state: State) -> int:
    """Step index = how many assistant turns have already happened in the recorded context."""
    return sum(1 for m in state.messages if m.get("role") == "assistant")


def turn_index(state: State) -> int:
    """Public alias: which step (0-based) the policy is deciding now."""
    return _n_assistant_turns(state)


def user_text(state: State) -> str:
    """The first user message's text (the demos use a string initial message)."""
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


class ScriptedPolicy:
    """Emits a fixed sequence of actions, indexed by step. Fully deterministic."""

    provider: Provider = "synthetic"

    def __init__(self, actions: list[Action], model_id: str = "synthetic:scripted") -> None:
        self._actions = actions
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    async def sample(self, state: State) -> Action:
        idx = _n_assistant_turns(state)
        if idx >= len(self._actions):
            return final("done")
        return self._actions[idx]


class RulePolicy:
    """A deterministic policy whose action is an arbitrary function of the current state.

    Lets a test build a synthetic agent with KNOWN causal dependence — e.g. "step 1's action
    depends on step 0's observation" or "step 0's action depends on the user message" — so an
    intervention's downstream effect can be asserted against ground truth.
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

    The RNG is seeded per ``sample`` call from (seed, step, context-hash) so that the *fixture*
    is reproducible across test runs, while still producing a genuine distribution of outcomes
    across the K samples a single replay/attribution pass draws.
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
        idx = _n_assistant_turns(state)
        if idx == self._noisy_step:
            # Draw from a counter so repeated samples in one pass differ but the whole pass is
            # reproducible given the seed (int seed -> deterministic, no hash() dependence).
            rng = random.Random(self._seed * 1_000_003 + self._draws)
            self._draws += 1
            return self._option_a if rng.random() < self._p else self._option_b
        if idx >= len(self._base):
            return final("done")
        return self._base[idx]


class MultiNoisyPolicy:
    """Independent stochastic choices at several steps — for the two-step-interaction SCM.

    ``noisy`` maps a step index to (option_a, option_b, p): at that step, return option_a with
    probability p, else option_b. Other steps fall back to ``base_actions`` (or a final action).
    Draws are seeded per-draw so a whole attribution pass is reproducible.
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
        idx = _n_assistant_turns(state)
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
        assert action.tool_name is not None
        result = self._results.get(action.tool_name, "{}")
        return Observation(tool_name=action.tool_name, result=result, source="mocked")


def support_like_script() -> tuple[list[Action], Environment]:
    """A 3-step deterministic run resembling the support agent: lookup -> refund -> final."""
    actions = [
        tool_call("lookup_order", {"order_id": "A1234"}, text="Let me look that up."),
        tool_call("issue_refund", {"order_id": "A1234", "amount": 99.0}, text="Processing refund."),
        final("Your refund has been processed."),
    ]
    env = DictEnvironment(
        {
            "lookup_order": '{"status": "shipped", "defect_reported": false}',
            "issue_refund": '{"ok": true}',
        }
    )
    return actions, env
