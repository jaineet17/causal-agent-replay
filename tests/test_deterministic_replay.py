"""Phase 0 DoD: faithful recording + deterministic replay.

The hard claim of Phase 0 is that we can faithfully record a run and replay it to the
provider's determinism limit, measuring residual nondeterminism honestly. We prove the replay
*machinery* against a synthetic deterministic policy (match rate must be exactly 1.0), prove
the faithfulness check actually catches a corrupted recording, and prove the distributional
framing on a seeded-stochastic policy (match rate strictly between 0 and 1, reproducibly).
"""

from __future__ import annotations

import copy

import pytest

from car.record.recorder import SyntheticCodec, record_run
from car.replay.deterministic import DeterministicReplay
from car.schemas.scm import ReplayError

from .scm_fixtures import (
    DictEnvironment,
    NoisyPolicy,
    ScriptedPolicy,
    final,
    support_like_script,
    tool_call,
)


async def _record_support_run() -> tuple:
    actions, env = support_like_script()
    policy = ScriptedPolicy(actions)
    codec = SyntheticCodec()
    traj = await record_run(
        trajectory_id="support-demo",
        policy=policy,
        environment=env,
        codec=codec,
        system_prompt="refund only under policy",
        tool_schemas=[],
        user_input="please refund order A1234",
    )
    return traj, codec


async def test_recording_round_trips_state_reconstruction() -> None:
    traj, codec = await _record_support_run()
    assert len(traj.steps) == 3
    assert traj.action_signature()[0].startswith("tool:lookup_order")
    assert traj.action_signature()[-1] == "final"
    # The load-bearing invariant: state_before is independently reconstructable.
    assert DeterministicReplay(codec).verify_reconstruction(traj) is True


async def test_deterministic_policy_replays_exactly() -> None:
    traj, codec = await _record_support_run()
    actions, _ = support_like_script()
    replay = DeterministicReplay(codec)
    report = await replay.measure(traj, ScriptedPolicy(actions), n_samples=8)

    assert report.reconstruction_faithful
    assert report.sequence_reproduction_rate == 1.0
    assert report.mean_step_match_rate == 1.0
    assert all(s.match_rate == 1.0 for s in report.per_step)


async def test_corrupted_recording_is_caught() -> None:
    """A faithfulness check that never fails is worthless — prove it catches tampering."""
    traj, codec = await _record_support_run()
    corrupted = copy.deepcopy(traj)
    # Tamper with the message history the agent supposedly decided step 1 from.
    corrupted.steps[1].state_before.messages.append({"role": "user", "content": "INJECTED"})
    with pytest.raises(ReplayError, match="state reconstruction diverged at step 1"):
        DeterministicReplay(codec).verify_reconstruction(corrupted)


async def test_stochastic_policy_yields_a_distribution_not_a_path() -> None:
    """Residual nondeterminism is the phenomenon, not a bug: replay reports a distribution."""
    traj, codec = await _record_support_run()
    base_actions, _ = support_like_script()
    # At step 1 the policy sometimes refunds (recorded) and sometimes escalates instead.
    noisy = NoisyPolicy(
        base_actions=base_actions,
        noisy_step=1,
        option_a=tool_call("issue_refund", {"order_id": "A1234", "amount": 99.0}),
        option_b=tool_call("escalate", {"reason": "policy not met"}),
        p=0.5,
        seed=7,
    )
    report = await DeterministicReplay(codec).measure(traj, noisy, n_samples=40)

    assert report.reconstruction_faithful
    # Only the noisy step diverges; surrounding steps reproduce exactly.
    assert report.per_step[0].match_rate == 1.0
    assert report.per_step[2].match_rate == 1.0
    assert 0.0 < report.per_step[1].match_rate < 1.0
    assert 0.0 < report.sequence_reproduction_rate < 1.0
    # The report exposes the full observed distribution at the divergent step.
    assert len(report.per_step[1].observed) == 2


async def test_replay_longer_than_recording_raises() -> None:
    """If the policy diverges into more steps than were recorded, refuse silently continuing."""
    actions = [tool_call("lookup_order", {"order_id": "A1234"}), final("done")]
    env = DictEnvironment({"lookup_order": "{}"})
    codec = SyntheticCodec()
    traj = await record_run(
        trajectory_id="short",
        policy=ScriptedPolicy(actions),
        environment=env,
        codec=codec,
        system_prompt="s",
        tool_schemas=[],
        user_input="go",
    )
    # A policy that never finalizes will request more observations than were recorded.
    runaway = ScriptedPolicy([tool_call("lookup_order", {"order_id": "A1234"})] * 10)
    with pytest.raises(ReplayError, match=r"only 1 were recorded|exceeded max_steps"):
        await DeterministicReplay(codec).replay_once(traj, runaway)
