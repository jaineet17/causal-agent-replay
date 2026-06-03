"""The native instrumented tool-loop — the recorder owns the loop (PLAN.md s5.1).

Because the loop is ours, faithful capture is guaranteed: at every step we snapshot the
EXACT ``state_before`` the action was decided from, the action itself, and the observation.
This is the rr principle (RESEARCH/phase_0_foundations.md s2) — record every nondeterministic
input so the run can be replayed deterministically between those inputs.

The loop is written entirely against the ``Policy`` / ``Environment`` protocols plus a
provider-specific ``MessageCodec``. The same loop therefore drives a live Anthropic/OpenAI
agent and an in-process synthetic policy with known causal structure, with no code change.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, Protocol, runtime_checkable

from car.schemas.scm import Environment, Policy, ReplayError
from car.schemas.trajectory import Action, Observation, State, Step, Trajectory


@runtime_checkable
class MessageCodec(Protocol):
    """Provider-specific serialization of the message history.

    Message threading is the loop's responsibility, but the *encoding* of each message is
    provider-native and lossless (RESEARCH s3): Anthropic ``tool_use``/``tool_result`` keyed by
    ``tool_use_id`` vs OpenAI ``tool_calls``/``role:tool`` keyed by ``tool_call_id`` with args as
    a JSON string. The codec owns that difference so the loop stays provider-agnostic.
    """

    def user_message(self, text: str) -> dict[str, Any]:
        """The initial user turn."""
        ...

    def assistant_message(self, action: Action) -> dict[str, Any]:
        """Serialize a recorded action back into an assistant message, verbatim."""
        ...

    def tool_result_message(self, action: Action, observation: Observation) -> dict[str, Any]:
        """Serialize a tool result, linked to ``action``'s tool call by the provider's id field."""
        ...

    def forge_action(
        self,
        *,
        kind: Literal["tool_call", "final"],
        text: str | None,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
    ) -> Action:
        """Build an ``Action`` with a provider-faithful ``raw`` for a FORCED action (do_action).

        A forced action has no recorded provider response, so the codec synthesizes one that its
        own ``assistant_message`` / ``tool_result_message`` can thread back into history (e.g. an
        Anthropic ``tool_use`` block with an id, or an OpenAI ``tool_calls`` entry).
        """
        ...


class ToolLoop:
    """Runs an agent forward against a policy + environment, recording a faithful trajectory.

    The loop snapshots ``state_before`` (a deep copy of the message history, so later mutation
    cannot corrupt an already-recorded step) before each policy call. The recorded
    ``state_before.messages`` is therefore exactly what was sent to produce the action — the
    invariant Phase 0 proves.
    """

    def __init__(
        self,
        policy: Policy,
        environment: Environment,
        codec: MessageCodec,
        *,
        max_steps: int = 20,
    ) -> None:
        self._policy = policy
        self._environment = environment
        self._codec = codec
        self._max_steps = max_steps

    async def run(
        self,
        *,
        trajectory_id: str,
        system_prompt: str,
        tool_schemas: list[dict[str, Any]],
        user_input: str,
        sampling: dict[str, Any] | None = None,
    ) -> Trajectory:
        sampling = dict(sampling or {})
        messages: list[dict[str, Any]] = [self._codec.user_message(user_input)]
        steps: list[Step] = []
        final_output: str | None = None

        for index in range(self._max_steps):
            state = State(
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
                model=self._policy.model_id,
                provider=self._policy.provider,
                sampling=sampling,
                messages=copy.deepcopy(messages),  # freeze the exact context for this step
            )
            action = await self._policy.sample(state)

            if action.kind == "final":
                steps.append(Step(index=index, state_before=state, action=action, observation=None))
                final_output = action.text or ""
                break

            if action.tool_name is None:
                raise ReplayError(
                    f"step {index}: action.kind=='tool_call' but tool_name is None "
                    f"(policy={self._policy.model_id!r})"
                )

            observation = await self._environment.observe(action)
            steps.append(
                Step(index=index, state_before=state, action=action, observation=observation)
            )
            messages.append(self._codec.assistant_message(action))
            messages.append(self._codec.tool_result_message(action, observation))
        else:
            raise ReplayError(
                f"trajectory {trajectory_id!r} exceeded max_steps={self._max_steps} without a "
                f"final action; refusing to record a truncated run (PLAN.md s0.9)."
            )

        if final_output is None:  # defensive; the break above always sets it
            raise ReplayError(f"trajectory {trajectory_id!r} ended without a final output")

        return Trajectory(
            trajectory_id=trajectory_id,
            steps=steps,
            final_output=final_output,
        )
