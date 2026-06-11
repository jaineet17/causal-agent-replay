"""Shared sampling primitive for attribution: estimate an outcome distribution for a coalition.

Both contrastive (held = a prefix) and Shapley (held = an arbitrary subset) reduce to: hold a set
of steps at their factual actions, resample the rest, run forward K times, score each, and return
the outcome distribution. The whole branch is charged to the budget BEFORE any rollout runs
(fail-before-spend; PLAN.md s5.7).

Rollouts are independent counterfactual samples, so they can run concurrently:
``max_concurrency`` bounds the in-flight rollouts. The default of 1 preserves the exact
sequential draw order (which keeps seeded synthetic fixtures byte-reproducible); real-model
callers should raise it — on a hosted or local LLM the wall-clock win is roughly the concurrency
factor.
"""

from __future__ import annotations

import asyncio

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
    max_concurrency: int = 1,
) -> OutcomeDistribution:
    """Run-forward-K with ``held`` fixed at factual actions; return the scored distribution."""
    if k_samples < 1:
        raise ValueError("k_samples must be >= 1")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if budget is not None:
        budget.charge(k_samples)  # the whole branch, before any model call is spent

    semaphore = asyncio.Semaphore(max_concurrency)

    async def one_rollout(i: int) -> Outcome:
        async with semaphore:
            child = await coalition_forward(
                trajectory_id=f"{factual.trajectory_id}:{label}:{i}",
                factual=factual,
                held=held,
                policy=policy,
                environment=environment,
                codec=codec,
            )
            return await outcome_fn.score(child)

    outcomes = await asyncio.gather(*(one_rollout(i) for i in range(k_samples)))
    return OutcomeDistribution.from_outcomes(list(outcomes))
