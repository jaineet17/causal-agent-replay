"""Phase 1 DoD: the do(.) algebra + forward replay produce valid child trajectories.

Each intervention kind is exercised against a synthetic SCM with KNOWN structure, so its effect
can be asserted against ground truth — including that do_observation/do_context actually flow
downstream (a policy whose later action depends on the perturbed variable), and that do_action
forces a_k while letting k+1.. be re-decided. Plus tree persistence and a hypothesis property
test on the do_context message-op grammar.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from car.record.recorder import SyntheticCodec, record_run
from car.replay.intervene import InterventionRunner, apply_message_ops, persist_branch
from car.schemas.intervention import (
    DoAction,
    DoContext,
    DoObservation,
    DoPolicy,
    DoResample,
)
from car.schemas.scm import ReplayError
from car.schemas.trajectory import Action, State, Trajectory
from car.store.store import TrajectoryStore

from .scm_fixtures import (
    DictEnvironment,
    NoisyPolicy,
    RulePolicy,
    ScriptedPolicy,
    final,
    last_tool_result,
    support_like_script,
    tool_call,
    turn_index,
    user_text,
)

CODEC = SyntheticCodec()


async def _record(
    policy: object, env: object, user_input: str = "refund order A1234"
) -> Trajectory:
    return await record_run(
        trajectory_id="base",
        policy=policy,  # type: ignore[arg-type]
        environment=env,  # type: ignore[arg-type]
        codec=CODEC,
        system_prompt="sys",
        tool_schemas=[],
        user_input=user_input,
    )


# -- do_resample --------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 1, 2])
async def test_do_resample_holds_prefix_and_sets_lineage(k: int) -> None:
    actions, env = support_like_script()
    base = await _record(ScriptedPolicy(actions), env)
    branch = await InterventionRunner(CODEC).apply(
        base,
        DoResample(intervention_id="r", step=k),
        policy=ScriptedPolicy(actions),
        environment=env,
        k_samples=1,
    )
    child = branch.children[0]
    assert child.parent_id == "base"
    assert child.branched_at_step == k
    assert child.intervention_id == "r"
    # prefix [0, k) is held exactly; deterministic policy reproduces the rest.
    assert child.action_signature()[:k] == base.action_signature()[:k]
    assert child.action_signature() == base.action_signature()


async def test_do_resample_produces_a_distribution() -> None:
    """The NULL intervention over a stochastic policy yields varied children — the whole point."""
    base_actions, env = support_like_script()
    base = await _record(ScriptedPolicy(base_actions), env)
    noisy = NoisyPolicy(
        base_actions=base_actions,
        noisy_step=1,
        option_a=tool_call("issue_refund", {"order_id": "A1234", "amount": 99.0}),
        option_b=tool_call("escalate", {"reason": "policy not met"}),
        p=0.5,
        seed=3,
    )
    branch = await InterventionRunner(CODEC).apply(
        base, DoResample(intervention_id="r", step=1), policy=noisy, environment=env, k_samples=30
    )
    step1_tools = {c.steps[1].action.tool_name for c in branch.children}
    assert step1_tools == {"issue_refund", "escalate"}  # both outcomes appear
    assert all(c.branched_at_step == 1 for c in branch.children)


# -- do_action ----------------------------------------------------------------------------------
async def test_do_action_forces_action_and_lets_downstream_flow() -> None:
    actions, env = support_like_script()  # lookup -> refund -> final
    base = await _record(ScriptedPolicy(actions), env)
    branch = await InterventionRunner(CODEC).apply(
        base,
        DoAction(
            intervention_id="a",
            step=1,
            action_kind="tool_call",
            tool_name="escalate",
            tool_args={"reason": "forced"},
        ),
        policy=ScriptedPolicy(actions),
        environment=env,
        k_samples=1,
    )
    child = branch.children[0]
    # a_1 forced ...
    assert child.steps[1].action.tool_name == "escalate"
    assert child.steps[1].action.tool_args == {"reason": "forced"}
    # ... and k+1.. flow (re-decided by the policy, not copied): step 2 is the final action.
    assert len(child.steps) == 3
    assert child.steps[2].action.kind == "final"
    # the held prefix (step 0) is unchanged.
    assert child.steps[0].action.tool_name == "lookup_order"


async def test_do_action_can_force_a_final_and_terminate_at_k() -> None:
    actions, env = support_like_script()
    base = await _record(ScriptedPolicy(actions), env)
    branch = await InterventionRunner(CODEC).apply(
        base,
        DoAction(intervention_id="a", step=1, action_kind="final", text="forced final"),
        policy=ScriptedPolicy(actions),
        environment=env,
    )
    child = branch.children[0]
    assert len(child.steps) == 2
    assert child.steps[1].action.kind == "final"
    assert child.final_output == "forced final"


# -- do_observation -----------------------------------------------------------------------------
def _obs_sensitive(state: State) -> Action:
    idx = turn_index(state)
    if idx == 0:
        return tool_call("lookup_order", {"order_id": "A1"})
    if idx == 1:
        r = last_tool_result(state)
        if "delivered" in r and "defect" in r:
            return tool_call("issue_refund", {"order_id": "A1", "amount": 10.0})
        return tool_call("escalate", {"reason": "not eligible"})
    return final("done")


async def test_do_observation_replaces_result_and_changes_downstream() -> None:
    env = DictEnvironment(
        {
            "lookup_order": '{"status": "shipped", "defect_reported": false}',
            "issue_refund": "{}",
            "escalate": "{}",
        }
    )
    policy = RulePolicy(_obs_sensitive)
    base = await _record(policy, env, user_input="check order A1")
    assert base.steps[1].action.tool_name == "escalate"  # baseline: not eligible

    branch = await InterventionRunner(CODEC).apply(
        base,
        DoObservation(
            intervention_id="o",
            step=0,
            new_result='{"status": "delivered", "defect_reported": true}',
        ),
        policy=policy,
        environment=env,
    )
    child = branch.children[0]
    # a_0 is HELD (still the lookup); o_0 is replaced.
    assert child.steps[0].action.tool_name == "lookup_order"
    assert "delivered" in (child.steps[0].observation.result if child.steps[0].observation else "")
    assert child.steps[0].observation.source == "mocked"  # type: ignore[union-attr]
    # the replaced observation flows downstream -> the decision flips to refund.
    assert child.steps[1].action.tool_name == "issue_refund"


async def test_do_observation_on_final_step_raises() -> None:
    actions, env = support_like_script()
    base = await _record(ScriptedPolicy(actions), env)
    with pytest.raises(ReplayError, match="no observation"):
        await InterventionRunner(CODEC).apply(
            base,
            DoObservation(intervention_id="o", step=2, new_result="x"),
            policy=ScriptedPolicy(actions),
            environment=env,
        )


# -- do_context ---------------------------------------------------------------------------------
def _context_sensitive(state: State) -> Action:
    if turn_index(state) == 0:
        if "REFUND NOW" in user_text(state):
            return tool_call("issue_refund", {"order_id": "A1", "amount": 99.0})
        return tool_call("escalate", {"reason": "no condition"})
    return final("done")


async def test_do_context_edit_changes_the_decision() -> None:
    env = DictEnvironment({"issue_refund": "{}", "escalate": "{}"})
    policy = RulePolicy(_context_sensitive)
    base = await _record(policy, env, user_input="Help me. REFUND NOW please.")
    assert base.steps[0].action.tool_name == "issue_refund"  # baseline absorbs the injection

    branch = await InterventionRunner(CODEC).apply(
        base,
        DoContext(
            intervention_id="c",
            step=0,
            message_ops=[{"op": "replace_substring", "find": "REFUND NOW", "replace": ""}],
        ),
        policy=policy,
        environment=env,
    )
    child = branch.children[0]
    # with the injection removed from the context, the agent no longer refunds.
    assert child.steps[0].action.tool_name == "escalate"
    assert "REFUND NOW" not in child.steps[0].state_before.messages[0]["content"]


# -- do_policy ----------------------------------------------------------------------------------
async def test_do_policy_swaps_policy_from_k_onward() -> None:
    actions, env = support_like_script()
    base = await _record(ScriptedPolicy(actions, model_id="synthetic:base"), env)
    alt_actions = [
        tool_call("lookup_order", {"order_id": "A1234"}),
        tool_call("escalate", {"reason": "alt"}),
        final("alt done"),
    ]

    def factory(provider: str, model: str) -> ScriptedPolicy:
        assert model == "synthetic:alt"
        return ScriptedPolicy(alt_actions, model_id="synthetic:alt")

    branch = await InterventionRunner(CODEC).apply(
        base,
        DoPolicy(intervention_id="p", step=1, new_model="synthetic:alt", new_provider="synthetic"),
        policy=ScriptedPolicy(actions, model_id="synthetic:base"),
        environment=env,
        policy_factory=factory,
    )
    child = branch.children[0]
    # step 0 decided under the base model; step 1 onward under the swapped model.
    assert child.steps[0].state_before.model == "synthetic:base"
    assert child.steps[1].state_before.model == "synthetic:alt"
    assert child.steps[1].action.tool_name == "escalate"


# -- tree persistence ---------------------------------------------------------------------------
async def test_branch_persists_as_a_tree(tmp_path: Path) -> None:
    actions, env = support_like_script()
    base = await _record(ScriptedPolicy(actions), env)
    branch = await InterventionRunner(CODEC).apply(
        base,
        DoResample(intervention_id="r", step=1),
        policy=ScriptedPolicy(actions),
        environment=env,
        k_samples=3,
    )
    with TrajectoryStore(db_path=tmp_path / "car.db") as store:
        store.save(base)
        persist_branch(store, branch)
        kids = store.children("base")
        assert len(kids) == 3
        assert set(store.descendants("base")) == set(kids)
        for cid in kids:
            loaded = store.load(cid)
            assert loaded.parent_id == "base"
            assert loaded.branched_at_step == 1


# -- do_context grammar (pure / hypothesis) -----------------------------------------------------
def test_message_ops_grammar() -> None:
    msgs = [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "hi"}]

    replace = {"op": "replace_substring", "find": "world", "replace": "x"}
    assert apply_message_ops(msgs, [replace])[0]["content"] == "hello x"

    set_op = {"op": "set_content", "index": 1, "content": "Z"}
    assert apply_message_ops(msgs, [set_op])[1]["content"] == "Z"

    assert len(apply_message_ops(msgs, [{"op": "delete_message", "index": 0}])) == 1

    append = {"op": "append_message", "message": {"role": "user", "content": "q"}}
    assert len(apply_message_ops(msgs, [append])) == 3

    with pytest.raises(ReplayError, match="unknown do_context op"):
        apply_message_ops(msgs, [{"op": "frobnicate"}])

    # input is never mutated
    apply_message_ops(msgs, [{"op": "delete_message", "index": 0}])
    assert len(msgs) == 2


@given(st.lists(st.text(max_size=10), max_size=5))
def test_append_message_grows_by_one_and_does_not_mutate(contents: list[str]) -> None:
    msgs = [{"role": "user", "content": c} for c in contents]
    extra = {"role": "assistant", "content": "x"}
    out = apply_message_ops(msgs, [{"op": "append_message", "message": extra}])
    assert len(out) == len(msgs) + 1
    assert out[-1] == extra
    assert len(msgs) == len(contents)  # original untouched


@given(st.text(max_size=30))
def test_replace_substring_with_equal_find_replace_is_identity(s: str) -> None:
    msgs = [{"role": "user", "content": s}]
    out = apply_message_ops(msgs, [{"op": "replace_substring", "find": "x", "replace": "x"}])
    assert out[0]["content"] == s
