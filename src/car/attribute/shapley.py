"""Budget-bounded Monte-Carlo Shapley attribution over steps (PLAN.md s5.4 shapley.py).

Contrastive attribution treats steps independently and so misses interactions (two steps that
only jointly cause the outcome). Shapley assigns each step its average marginal contribution over
all coalition contexts, which correctly SPLITS shared credit (for an AND-failure, 0.5 / 0.5).

Estimator (RESEARCH s3): permutation sampling (ApproShapley) + antithetic reverse-permutation
pairing. A "coalition" S = the steps held at their factual actions; the complement is resampled.
The value v(S) = P(bad | held = S) over ``samples_per_eval`` forward rollouts.

  - one permutation walk yields one marginal contribution PER step (n+1 coalition evals);
  - v(S) is NOT cached across permutations, so per-step marginals are i.i.d. and the CLT CI is
    honest (caching would collapse the variance to 0 and report false confidence);
  - antithetic pairing (a permutation and its reverse) reduces variance for free;
  - every rollout is charged to a ``Budget``; the breaker stops cleanly and reports partial work.

Expensive and on-demand only — never run by default.
"""

from __future__ import annotations

import random

import structlog
from pydantic import BaseModel, ConfigDict

from car.attribute.effects import clt_interval
from car.attribute.sampling import coalition_distribution
from car.budget.budget import Budget, BudgetExceeded
from car.outcome.functions import OutcomeFunction
from car.record.toolloop import MessageCodec
from car.schemas.scm import Environment, Policy
from car.schemas.trajectory import Trajectory

log = structlog.get_logger(__name__)


class StepShapley(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    value: float
    ci_low: float
    ci_high: float
    n_marginals: int

    @property
    def is_significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


class ShapleyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    bad_label: str
    per_step: list[StepShapley]
    efficiency_sum: float
    permutations_completed: int
    antithetic: bool
    samples_per_eval: int
    confidence: float
    budget_truncated: bool

    def ranked(self) -> list[StepShapley]:
        return sorted(self.per_step, key=lambda s: s.value, reverse=True)


async def shapley_attribution(
    factual: Trajectory,
    *,
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    outcome_fn: OutcomeFunction,
    bad_label: str,
    n_permutations: int = 64,
    samples_per_eval: int = 1,
    antithetic: bool = True,
    confidence: float = 0.95,
    budget: Budget | None = None,
    seed: int = 0,
    max_concurrency: int = 1,
) -> ShapleyResult:
    """Estimate each step's Shapley value (contribution to the bad outcome), with CLT CIs.

    ``max_concurrency`` bounds in-flight rollouts within each coalition evaluation; raise it for
    real models, keep 1 for byte-reproducible synthetic runs.
    """
    n = len(factual.steps)
    players = list(range(n))
    marginals: dict[int, list[float]] = {k: [] for k in players}
    rng = random.Random(seed)

    completed = 0
    truncated = False
    for _ in range(n_permutations):
        perm = players[:]
        rng.shuffle(perm)
        try:
            m_fwd = await _walk(
                perm,
                factual,
                policy,
                environment,
                codec,
                outcome_fn,
                bad_label,
                samples_per_eval,
                budget,
                max_concurrency,
            )
            if antithetic:
                m_rev = await _walk(
                    list(reversed(perm)),
                    factual,
                    policy,
                    environment,
                    codec,
                    outcome_fn,
                    bad_label,
                    samples_per_eval,
                    budget,
                    max_concurrency,
                )
                for k in players:
                    marginals[k].append((m_fwd[k] + m_rev[k]) / 2.0)
            else:
                for k in players:
                    marginals[k].append(m_fwd[k])
        except BudgetExceeded:
            truncated = True
            log.warning("shapley truncated by budget", permutations_completed=completed)
            break
        completed += 1

    per_step: list[StepShapley] = []
    for k in players:
        value, low, high = clt_interval(marginals[k], confidence)
        per_step.append(
            StepShapley(
                index=k, value=value, ci_low=low, ci_high=high, n_marginals=len(marginals[k])
            )
        )

    result = ShapleyResult(
        trajectory_id=factual.trajectory_id,
        bad_label=bad_label,
        per_step=per_step,
        efficiency_sum=sum(s.value for s in per_step),
        permutations_completed=completed,
        antithetic=antithetic,
        samples_per_eval=samples_per_eval,
        confidence=confidence,
        budget_truncated=truncated,
    )
    log.info(
        "shapley attribution",
        trajectory_id=factual.trajectory_id,
        permutations_completed=completed,
        efficiency_sum=result.efficiency_sum,
        truncated=truncated,
    )
    return result


async def _walk(
    perm: list[int],
    factual: Trajectory,
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    outcome_fn: OutcomeFunction,
    bad_label: str,
    samples_per_eval: int,
    budget: Budget | None,
    max_concurrency: int,
) -> dict[int, float]:
    """Walk one permutation, returning the marginal contribution of each step.

    Fresh coalition evaluations (no cross-permutation cache) keep the marginals i.i.d.
    """
    held: set[int] = set()
    prev = await _value(
        factual,
        held,
        policy,
        environment,
        codec,
        outcome_fn,
        bad_label,
        samples_per_eval,
        budget,
        max_concurrency,
    )
    out: dict[int, float] = {}
    for step in perm:
        held = held | {step}
        cur = await _value(
            factual,
            held,
            policy,
            environment,
            codec,
            outcome_fn,
            bad_label,
            samples_per_eval,
            budget,
            max_concurrency,
        )
        out[step] = cur - prev
        prev = cur
    return out


async def _value(
    factual: Trajectory,
    held: set[int],
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    outcome_fn: OutcomeFunction,
    bad_label: str,
    samples_per_eval: int,
    budget: Budget | None,
    max_concurrency: int,
) -> float:
    """v(S) = P(bad | the steps in ``held`` take their factual actions, the rest are resampled)."""
    dist = await coalition_distribution(
        factual,
        held,
        policy=policy,
        environment=environment,
        codec=codec,
        outcome_fn=outcome_fn,
        k_samples=samples_per_eval,
        budget=budget,
        label="shapley",
        max_concurrency=max_concurrency,
    )
    return dist.prob_label(bad_label)


def estimate_rollouts(
    n_steps: int, n_permutations: int, samples_per_eval: int, *, antithetic: bool = True
) -> int:
    """Forward rollouts a full run would cost — for budgeting before committing."""
    walks = 2 if antithetic else 1
    return n_permutations * walks * (n_steps + 1) * samples_per_eval
