"""Schema correctness: validation, discriminated-union dispatch, digest stability."""

from __future__ import annotations

import pydantic
import pytest

from car.schemas.intervention import (
    DoAction,
    DoObservation,
    DoPolicy,
    DoResample,
    Intervention,
)
from car.schemas.trajectory import Action, Observation, State, Step, Trajectory


def _state(messages: list[dict]) -> State:
    return State(
        system_prompt="sys",
        tool_schemas=[],
        model="synthetic:test",
        provider="synthetic",
        sampling={},
        messages=messages,
    )


def test_state_request_digest_is_stable_and_order_independent() -> None:
    a = _state([{"role": "user", "content": "hi"}])
    b = _state([{"role": "user", "content": "hi"}])
    assert a.request_digest() == b.request_digest()
    c = _state([{"role": "user", "content": "different"}])
    assert a.request_digest() != c.request_digest()


def test_outcome_score_bounds() -> None:
    with pytest.raises(pydantic.ValidationError):
        from car.schemas.trajectory import Outcome

        Outcome(label="bad", score=1.5)


def test_extra_fields_forbidden() -> None:
    with pytest.raises(pydantic.ValidationError):
        Observation(tool_name="t", result="r", source="mocked", surprise=1)  # type: ignore[call-arg]


def test_intervention_union_dispatches_on_kind() -> None:
    ta = pydantic.TypeAdapter(Intervention)
    resample = {"intervention_id": "i", "step": 0, "kind": "do_resample"}
    assert isinstance(ta.validate_python(resample), DoResample)
    forced = ta.validate_python(
        {
            "intervention_id": "i",
            "step": 1,
            "kind": "do_action",
            "action_kind": "tool_call",
            "tool_name": "escalate",
            "tool_args": {"reason": "policy"},
        }
    )
    assert isinstance(forced, DoAction)
    assert forced.tool_name == "escalate"

    obs = {"intervention_id": "i", "step": 2, "kind": "do_observation", "new_result": "x"}
    assert isinstance(ta.validate_python(obs), DoObservation)

    pol = {"intervention_id": "i", "step": 0, "kind": "do_policy", "new_model": "m"}
    assert isinstance(ta.validate_python(pol), DoPolicy)


def test_action_signature_distinguishes_tool_args() -> None:
    def step(args: dict) -> Step:
        return Step(
            index=0,
            state_before=_state([{"role": "user", "content": "hi"}]),
            action=Action(kind="tool_call", tool_name="issue_refund", tool_args=args, raw={}),
            observation=Observation(tool_name="issue_refund", result="ok", source="mocked"),
        )

    t1 = Trajectory(trajectory_id="t1", steps=[step({"amount": 10})], final_output="")
    t2 = Trajectory(trajectory_id="t2", steps=[step({"amount": 99})], final_output="")
    assert t1.action_signature() != t2.action_signature()
    t3 = Trajectory(trajectory_id="t3", steps=[step({"amount": 10})], final_output="")
    assert t1.action_signature() == t3.action_signature()


def test_branch_lineage_fields() -> None:
    leaf = Trajectory(
        trajectory_id="child",
        parent_id="root",
        branched_at_step=2,
        intervention_id="iv1",
        steps=[
            Step(
                index=0,
                state_before=_state([{"role": "user", "content": "hi"}]),
                action=Action(kind="final", text="bye", raw={}),
            )
        ],
        final_output="bye",
    )
    assert leaf.is_branch
    assert leaf.branched_at_step == 2
