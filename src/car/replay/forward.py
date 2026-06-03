"""Forward replay — run the agent loop forward from step k under the stochastic policy.

One ``run_forward`` call = one counterfactual sample (PLAN.md s5.3). Because Phase 0 records
``state_before`` completely, the context at step k is recorded exactly; forward replay does not
reconstruct it, it *continues* from it. Every intervention in ``intervene.py`` reduces to a
choice of {context at k, action at k, observation at k, policy from k onward} handed to this
engine, which then samples the rest of the trajectory from the (possibly swapped) policy.

The suffix uses a LIVE environment: counterfactual actions may differ from the recorded ones, so
their observations must be produced fresh. For the demo/tests the environment is mocked and
deterministic, which keeps counterfactual samples reproducible. Real-tool side effects are a
deferred concern (PLAN.md s12).
"""

from __future__ import annotations

import copy
from typing import Any

from car.record.toolloop import MessageCodec
from car.schemas.scm import Environment, Policy, ReplayError
from car.schemas.trajectory import Action, Observation, State, Step, Trajectory


async def run_forward(
    *,
    trajectory_id: str,
    parent: Trajectory,
    k: int,
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    messages_at_k: list[dict[str, Any]],
    sampling: dict[str, Any],
    action_at_k: Action | None = None,
    observation_at_k: Observation | None = None,
    intervention_id: str | None = None,
    max_extra_steps: int = 20,
) -> Trajectory:
    """Build a child trajectory: parent's steps [0, k) held, step k seeded, then run to terminal.

    - ``action_at_k=None`` -> sample a_k from ``policy``; else use the given (forced/held) action.
    - ``observation_at_k=None`` -> get o_k from ``environment``; else use the given observation.
    - ``messages_at_k`` is the context at k (recorded, or edited for do_context).
    - ``sampling`` / ``policy`` apply from step k onward (swapped for do_policy).
    """
    if not 0 <= k < len(parent.steps):
        raise ReplayError(f"step {k} out of range for trajectory with {len(parent.steps)} steps")

    system_prompt = parent.steps[k].state_before.system_prompt
    tool_schemas = parent.steps[k].state_before.tool_schemas
    steps: list[Step] = copy.deepcopy(parent.steps[:k])  # held prefix
    messages: list[dict[str, Any]] = copy.deepcopy(messages_at_k)
    final_output: str | None = None

    # --- step k: seeded ----------------------------------------------------------------------
    state_k = State(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        model=policy.model_id,
        provider=policy.provider,
        sampling=sampling,
        messages=copy.deepcopy(messages),
    )
    action_k = action_at_k if action_at_k is not None else await policy.sample(state_k)

    if action_k.kind == "final":
        steps.append(Step(index=k, state_before=state_k, action=action_k, observation=None))
        final_output = action_k.text or ""
    else:
        if action_k.tool_name is None:
            raise ReplayError(f"forward step {k}: tool_call action has no tool_name")
        obs_k = (
            observation_at_k
            if observation_at_k is not None
            else await environment.observe(action_k)
        )
        steps.append(Step(index=k, state_before=state_k, action=action_k, observation=obs_k))
        messages.append(codec.assistant_message(action_k))
        messages.append(codec.tool_result_message(action_k, obs_k))

        # --- steps k+1 .. : normal loop under the (possibly swapped) policy ------------------
        final_output = await _continue_loop(
            steps=steps,
            messages=messages,
            start_index=k + 1,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            sampling=sampling,
            policy=policy,
            environment=environment,
            codec=codec,
            max_index=k + 1 + max_extra_steps,
            trajectory_id=trajectory_id,
        )

    return Trajectory(
        trajectory_id=trajectory_id,
        parent_id=parent.trajectory_id,
        branched_at_step=k,
        intervention_id=intervention_id,
        steps=steps,
        final_output=final_output,
    )


async def coalition_forward(
    *,
    trajectory_id: str,
    factual: Trajectory,
    held: set[int],
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    max_steps: int | None = None,
    intervention_id: str = "coalition",
) -> Trajectory:
    """Run forward holding steps in ``held`` at their FACTUAL actions; resample the rest.

    This generalizes ``do_resample`` to an arbitrary subset (the Shapley coalition primitive):
      - ``held = {0..k-1}`` reproduces contrastive ``do_resample(k)`` (resample k onward);
      - ``held = all steps`` reproduces the factual trajectory exactly (v(N));
      - ``held = {}`` resamples everything (the baseline v(empty)).

    Holding step k means do(a_k = factual a_k) even if the (resampled) context at k differs — the
    Shapley value of a step is the contribution of *fixing that decision to what actually
    happened*. Observations follow from the live environment.
    """
    n = len(factual.steps)
    if max_steps is None:
        max_steps = n + 5
    s0 = factual.steps[0].state_before
    messages: list[dict[str, Any]] = [copy.deepcopy(s0.messages[0])]
    steps: list[Step] = []
    final_output: str | None = None

    for index in range(max_steps):
        state = State(
            system_prompt=s0.system_prompt,
            tool_schemas=s0.tool_schemas,
            model=policy.model_id,
            provider=policy.provider,
            sampling=s0.sampling,
            messages=copy.deepcopy(messages),
        )
        if index in held and index < n:
            action = factual.steps[index].action  # do(a = factual)
        else:
            action = await policy.sample(state)

        if action.kind == "final":
            steps.append(Step(index=index, state_before=state, action=action, observation=None))
            final_output = action.text or ""
            break
        if action.tool_name is None:
            raise ReplayError(f"coalition step {index}: tool_call action has no tool_name")
        observation = await environment.observe(action)
        steps.append(Step(index=index, state_before=state, action=action, observation=observation))
        messages.append(codec.assistant_message(action))
        messages.append(codec.tool_result_message(action, observation))
    else:
        raise ReplayError(
            f"coalition_forward {trajectory_id!r} exceeded max_steps={max_steps} without a final"
        )

    if final_output is None:  # pragma: no cover - the break always sets it
        raise ReplayError(f"coalition_forward {trajectory_id!r} ended without a final output")

    resampled = [i for i in range(n) if i not in held]
    return Trajectory(
        trajectory_id=trajectory_id,
        parent_id=factual.trajectory_id,
        branched_at_step=resampled[0] if resampled else 0,
        intervention_id=intervention_id,
        steps=steps,
        final_output=final_output,
    )


async def _continue_loop(
    *,
    steps: list[Step],
    messages: list[dict[str, Any]],
    start_index: int,
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    sampling: dict[str, Any],
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    max_index: int,
    trajectory_id: str,
) -> str:
    for index in range(start_index, max_index):
        state = State(
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            model=policy.model_id,
            provider=policy.provider,
            sampling=sampling,
            messages=copy.deepcopy(messages),
        )
        action = await policy.sample(state)
        if action.kind == "final":
            steps.append(Step(index=index, state_before=state, action=action, observation=None))
            return action.text or ""
        if action.tool_name is None:
            raise ReplayError(f"forward step {index}: tool_call action has no tool_name")
        observation = await environment.observe(action)
        steps.append(Step(index=index, state_before=state, action=action, observation=observation))
        messages.append(codec.assistant_message(action))
        messages.append(codec.tool_result_message(action, observation))

    raise ReplayError(
        f"forward replay {trajectory_id!r} exceeded max steps without terminating "
        f"(start_index={start_index}, max_index={max_index})"
    )
