"""The SCM view: the policy and environment interfaces the replay engine runs against.

PLAN.md s1 models an agent run as a structural causal model. Two exogenous mechanisms drive
it forward from any state:

  - the **policy** pi(a_k | context_k) — stochastic; this is the LLM (or, in tests, a
    hand-built synthetic policy with known causal structure);
  - the **environment** — returns the observation o_k for an action's tool call; deterministic
    for mocked/recorded tools, possibly non-reproducible for real ones.

Everything in ``car.replay`` is written against these Protocols, NOT against a concrete
provider. That is what lets the synthetic-SCM fixtures (PLAN.md s7) — where we KNOW which
step is pivotal — exercise the identical forward/intervene/attribute code paths as a live
Anthropic/OpenAI agent, with no API calls and a deterministic ground truth to validate
against. The credibility of the whole project rests on that substitutability.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from car.schemas.trajectory import Action, Observation, Provider, State


@runtime_checkable
class Policy(Protocol):
    """A stochastic action mechanism: sample an action given the current state.

    Implementations:
      - ``car.record.toolloop`` wraps the Anthropic/OpenAI SDKs (the agent under test);
      - synthetic test policies return actions from a known conditional distribution so
        attribution has ground truth to be validated against.

    ``sample`` must be a pure function of ``state`` plus its own randomness — it must not read
    hidden global state — so that replaying from a reconstructed ``state_before`` is sound.
    """

    async def sample(self, state: State) -> Action: ...

    @property
    def model_id(self) -> str:
        """Identifier recorded into ``State.model`` (e.g. "claude-...", or "synthetic:foo")."""
        ...

    @property
    def provider(self) -> Provider:
        """The provider recorded into ``State.provider``; selects the replay codec/SDK."""
        ...


@runtime_checkable
class Environment(Protocol):
    """The observation mechanism: produce the tool result for an action's tool call.

    ``source`` on the returned ``Observation`` records provenance (real/recorded/mocked) so
    non-reproducibility is never silent.
    """

    async def observe(self, action: Action) -> Observation: ...


class ReplayError(RuntimeError):
    """Raised when replay cannot faithfully reconstruct or continue a trajectory.

    Per PLAN.md s0.9 (no silent failures): replay never degrades quietly. If a recorded
    action cannot be reconstructed or an observation cannot be sourced, this is raised with
    context rather than guessed around.
    """
