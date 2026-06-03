"""Apply an intervention to a recorded trajectory, then run forward K times (PLAN.md s5.3).

This is the operational semantics of the do(.) algebra (s4.2). Each kind reduces to a choice of
{context at k, action at k, observation at k, policy from k onward} handed to ``run_forward``:

  do_resample   : re-sample a_k from the SAME policy                 (the NULL intervention)
  do_action     : force a_k (codec forges its provider-faithful raw)
  do_observation: hold the recorded a_k, replace o_k
  do_context    : edit context_k, then re-sample a_k
  do_policy     : swap the policy from k forward, then re-sample a_k

Because the policy is stochastic, K samples form a *branch*: K child trajectories sharing the
held prefix [0, k) and diverging at k. The branch is one estimate of P(outcome | do(.)).
"""

from __future__ import annotations

import copy
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from car.record.toolloop import MessageCodec
from car.replay.forward import run_forward
from car.schemas.intervention import (
    DoAction,
    DoContext,
    DoObservation,
    DoPolicy,
    DoResample,
    Intervention,
)
from car.schemas.scm import Environment, Policy, ReplayError
from car.schemas.trajectory import Action, Observation, Provider, Trajectory

log = structlog.get_logger(__name__)


class Branch(BaseModel):
    """A set of K counterfactual children produced by one intervention at one step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    intervention_id: str
    step: int
    kind: str
    children: list[Trajectory] = Field(default_factory=list)

    @property
    def k_samples(self) -> int:
        return len(self.children)


class InterventionRunner:
    """Applies interventions and produces tree branches. Codec-bound (provider-specific)."""

    def __init__(self, codec: MessageCodec) -> None:
        self._codec = codec

    async def apply(
        self,
        traj: Trajectory,
        intervention: Intervention,
        *,
        policy: Policy,
        environment: Environment,
        k_samples: int = 1,
        max_extra_steps: int = 20,
        policy_factory: Any | None = None,
    ) -> Branch:
        if k_samples < 1:
            raise ValueError("k_samples must be >= 1")
        k = intervention.step
        self._validate(traj, intervention)

        children: list[Trajectory] = []
        for i in range(k_samples):
            child_id = f"{traj.trajectory_id}:{intervention.intervention_id}:{i}"
            child = await self._one_sample(
                traj=traj,
                intervention=intervention,
                child_id=child_id,
                base_policy=policy,
                environment=environment,
                max_extra_steps=max_extra_steps,
                policy_factory=policy_factory,
            )
            children.append(child)

        log.info(
            "applied intervention",
            intervention_id=intervention.intervention_id,
            kind=intervention.kind,
            step=k,
            k_samples=k_samples,
        )
        return Branch(
            intervention_id=intervention.intervention_id,
            step=k,
            kind=intervention.kind,
            children=children,
        )

    async def _one_sample(
        self,
        *,
        traj: Trajectory,
        intervention: Intervention,
        child_id: str,
        base_policy: Policy,
        environment: Environment,
        max_extra_steps: int,
        policy_factory: Any | None,
    ) -> Trajectory:
        k = intervention.step
        recorded_step = traj.steps[k]
        messages_at_k = copy.deepcopy(recorded_step.state_before.messages)
        sampling = copy.deepcopy(recorded_step.state_before.sampling)
        policy = base_policy
        action_at_k: Action | None = None
        observation_at_k: Observation | None = None

        if isinstance(intervention, DoResample):
            pass  # sample a_k from the same policy; nothing held

        elif isinstance(intervention, DoAction):
            action_at_k = self._codec.forge_action(
                kind=intervention.action_kind,
                text=intervention.text,
                tool_name=intervention.tool_name,
                tool_args=intervention.tool_args,
            )

        elif isinstance(intervention, DoObservation):
            held = recorded_step.action  # hold the recorded action a_k
            action_at_k = held
            observation_at_k = Observation(
                tool_name=held.tool_name or "",
                result=intervention.new_result,
                source=intervention.new_source,
            )

        elif isinstance(intervention, DoContext):
            messages_at_k = apply_message_ops(messages_at_k, intervention.message_ops)

        elif isinstance(intervention, DoPolicy):
            factory = _resolve_policy_factory(policy_factory)
            provider: Provider = intervention.new_provider or base_policy.provider
            policy = factory(provider, intervention.new_model)
            if intervention.new_sampling is not None:
                sampling = dict(intervention.new_sampling)

        else:  # pragma: no cover - exhaustive over the union
            raise ReplayError(f"unknown intervention kind: {intervention!r}")

        return await run_forward(
            trajectory_id=child_id,
            parent=traj,
            k=k,
            policy=policy,
            environment=environment,
            codec=self._codec,
            messages_at_k=messages_at_k,
            sampling=sampling,
            action_at_k=action_at_k,
            observation_at_k=observation_at_k,
            intervention_id=intervention.intervention_id,
            max_extra_steps=max_extra_steps,
        )

    @staticmethod
    def _validate(traj: Trajectory, intervention: Intervention) -> None:
        k = intervention.step
        if not 0 <= k < len(traj.steps):
            raise ReplayError(
                f"intervention step {k} out of range for trajectory with {len(traj.steps)} steps"
            )
        if isinstance(intervention, DoObservation) and traj.steps[k].observation is None:
            raise ReplayError(
                f"do_observation at step {k}: the recorded step has no observation "
                f"(it is a final action), so there is no tool result to replace"
            )
        if (
            isinstance(intervention, DoAction)
            and intervention.action_kind == "tool_call"
            and not intervention.tool_name
        ):
            raise ReplayError("do_action with action_kind='tool_call' requires tool_name")


def _resolve_policy_factory(policy_factory: Any | None) -> Any:
    if policy_factory is not None:
        return policy_factory
    from car.record.recorder import policy_for  # lazy to avoid import cycle

    return policy_for


# --------------------------------------------------------------------------------------------
# do_context message-op grammar
# --------------------------------------------------------------------------------------------
def apply_message_ops(
    messages: list[dict[str, Any]], ops: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply a list of edits to a message history (the do_context grammar).

    Supported ops (each a dict with an ``op`` key):
      - ``replace_substring``: {find, replace, [role]} — substring replace across message
        contents (recurses into list-of-blocks ``text``/``content`` fields); optional role filter.
      - ``set_content``: {index, content} — set a message's content.
      - ``delete_message``: {index}
      - ``append_message``: {message}

    Unknown ops raise (no silent failure, PLAN.md s0.9).
    """
    out = copy.deepcopy(messages)
    for op in ops:
        name = op.get("op")
        if name == "replace_substring":
            role = op.get("role")
            for msg in out:
                if role is not None and msg.get("role") != role:
                    continue
                msg["content"] = _replace_in_content(msg.get("content"), op["find"], op["replace"])
        elif name == "set_content":
            out[_index(op, out)]["content"] = op["content"]
        elif name == "delete_message":
            del out[_index(op, out)]
        elif name == "append_message":
            out.append(op["message"])
        else:
            raise ReplayError(f"unknown do_context op: {name!r}")
    return out


def _index(op: dict[str, Any], messages: list[dict[str, Any]]) -> int:
    idx = int(op["index"])
    if not -len(messages) <= idx < len(messages):
        raise ReplayError(f"do_context op index {idx} out of range ({len(messages)} messages)")
    return idx


def _replace_in_content(content: Any, find: str, replace: str) -> Any:
    if isinstance(content, str):
        return content.replace(find, replace)
    if isinstance(content, list):
        return [_replace_in_content(block, find, replace) for block in content]
    if isinstance(content, dict):
        out = dict(content)
        for key in ("text", "content"):
            if isinstance(out.get(key), str):
                out[key] = out[key].replace(find, replace)
        return out
    return content


def persist_branch(store: Any, branch: Branch) -> None:
    """Save every child of a branch into a ``TrajectoryStore`` (lineage already set on each)."""
    for child in branch.children:
        store.save(child)
