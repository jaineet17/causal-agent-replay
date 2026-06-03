"""Shared sampling primitive for attribution: estimate an outcome distribution for a coalition.

Both contrastive (held = a prefix) and Shapley (held = an arbitrary subset) reduce to: hold a set
of steps at their factual actions, resample the rest, run forward K times, score each, and return
the outcome distribution. Every rollout is charged to the budget (PLAN.md s5.7).
"""

from __future__ import annotations

from car.attribute.effects import OutcomeDistribution
from car.budget.budget import Budget
from car.outcome.functions import OutcomeFunction
from car.record.toolloop import MessageCodec
from car.replay.forward import coalition_forward
from car.schemas.scm import Environment, Policy
from car.schemas.trajectory import Outcome, Trajectory


async def coalition_distribution(
    factual: Trajectory,
    held: set[int],
    *,
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    outcome_fn: OutcomeFunction,
    k_samples: int,
    budget: Budget | None = None,
    label: str = "coalition",
) -> OutcomeDistribution:
    """Run-forward-K with ``held`` fixed at factual actions; return the scored distribution."""
    if k_samples < 1:
        raise ValueError("k_samples must be >= 1")
    outcomes: list[Outcome] = []
    for i in range(k_samples):
        if budget is not None:
            budget.charge(1)
        child = await coalition_forward(
            trajectory_id=f"{factual.trajectory_id}:{label}:{i}",
            factual=factual,
            held=held,
            policy=policy,
            environment=environment,
            codec=codec,
        )
        outcomes.append(await outcome_fn.score(child))
    return OutcomeDistribution.from_outcomes(outcomes)
