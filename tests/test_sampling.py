"""coalition_distribution: concurrency actually overlaps rollouts; budget fails before spend."""

from __future__ import annotations

import asyncio

import pytest

from car.attribute.sampling import coalition_distribution
from car.budget.budget import Budget, BudgetExceeded
from car.record.recorder import SyntheticCodec, record_run
from car.schemas.scm import Policy
from car.schemas.trajectory import Action, State, Trajectory
from car.synthetic import final

from .scm_fixtures import ScriptedPolicy
from .synthetic_scms import ENV, PIVOTAL_OBSERVED, refund_outcome

CODEC = SyntheticCodec()


class SlowCountingPolicy:
    """Always finalizes, after a tiny await; records the max number of in-flight samples."""

    provider = "synthetic"

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def model_id(self) -> str:
        return "synthetic:slow"

    async def sample(self, state: State) -> Action:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return final("done")


async def _factual() -> Trajectory:
    return await record_run(
        trajectory_id="obs",
        policy=ScriptedPolicy(PIVOTAL_OBSERVED),
        environment=ENV,
        codec=CODEC,
        system_prompt="sys",
        tool_schemas=[],
        user_input="go",
    )


async def test_rollouts_overlap_under_concurrency() -> None:
    factual = await _factual()
    policy = SlowCountingPolicy()
    dist = await coalition_distribution(
        factual,
        held=set(),
        policy=policy,
        environment=ENV,
        codec=CODEC,
        outcome_fn=refund_outcome(),
        k_samples=12,
        max_concurrency=6,
    )
    assert dist.n == 12
    assert policy.max_in_flight > 1  # rollouts genuinely overlapped
    assert policy.max_in_flight <= 6  # ... bounded by the semaphore


async def test_sequential_default_does_not_overlap() -> None:
    factual = await _factual()
    policy = SlowCountingPolicy()
    await coalition_distribution(
        factual,
        held=set(),
        policy=policy,
        environment=ENV,
        codec=CODEC,
        outcome_fn=refund_outcome(),
        k_samples=4,
    )
    assert policy.max_in_flight == 1


async def test_budget_fails_before_any_spend() -> None:
    factual = await _factual()
    policy = SlowCountingPolicy()
    budget = Budget(max_samples=5)
    with pytest.raises(BudgetExceeded):
        await coalition_distribution(
            factual,
            held=set(),
            policy=policy,
            environment=ENV,
            codec=CODEC,
            outcome_fn=refund_outcome(),
            k_samples=6,  # would exceed -> must raise BEFORE running anything
            budget=budget,
            max_concurrency=4,
        )
    assert budget.used_samples == 0
    assert policy.max_in_flight == 0  # nothing was spent

    _ = isinstance(policy, Policy)  # the counting fixture satisfies the Policy protocol
