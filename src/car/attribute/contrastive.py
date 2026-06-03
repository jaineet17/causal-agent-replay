"""Single-step contrastive resampling attribution — the headline method (PLAN.md s5.4).

For each step k: hold steps [0, k) at their factual actions, ``do_resample`` step k and let
k+1.. flow, K times. Estimate P(bad | resample at k) and the effect versus the observed run.

Identifying the causal locus (RESEARCH s1/s3): under run-forward, resampling step k also re-rolls
every downstream stochastic step, so an *early* irrelevant step can show an effect (it re-rolls
the true pivotal step). Magnitude alone cannot localize. The locus is the **largest k whose effect
CI excludes 0** — the last step where re-rolling still rescues the run; beyond it the outcome is
committed. τ_k is therefore a through-continuation total effect; CRN (common random numbers) is the
documented variance-reduction refinement.

Cheap (n·K forward runs), interpretable, and the headline result. It does NOT capture
interactions between steps — that is what ``shapley.py`` is for.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, ConfigDict

from car.attribute.effects import (
    EffectEstimate,
    OutcomeDistribution,
    observed_distribution,
    prob_label_effect,
)
from car.attribute.sampling import coalition_distribution
from car.budget.budget import Budget
from car.outcome.functions import OutcomeFunction
from car.record.toolloop import MessageCodec
from car.schemas.scm import Environment, Policy
from car.schemas.trajectory import Trajectory

log = structlog.get_logger(__name__)


class StepAttribution(BaseModel):
    """The per-step contrastive result: the effect of resampling step k on the bad outcome."""

    model_config = ConfigDict(extra="forbid")

    index: int
    effect: EffectEstimate
    p_bad_after_resample: float

    @property
    def rescues(self) -> bool:
        """True if resampling this step significantly REDUCES the bad outcome."""
        return self.effect.is_significant and self.effect.point < 0.0


class ContrastiveResult(BaseModel):
    """Attribution over all steps, plus the identified causal locus."""

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    bad_label: str
    observed_label: str
    per_step: list[StepAttribution]
    causal_locus: int | None
    confidence: float
    k_samples: int

    def ranked(self) -> list[StepAttribution]:
        """Steps ranked by effect magnitude (largest rescue first)."""
        return sorted(self.per_step, key=lambda s: s.effect.point)


async def contrastive_attribution(
    factual: Trajectory,
    *,
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    outcome_fn: OutcomeFunction,
    bad_label: str,
    k_samples: int = 16,
    confidence: float = 0.95,
    budget: Budget | None = None,
) -> ContrastiveResult:
    """Attribute the bad outcome to a single step by contrastive resampling."""
    observed = await observed_distribution(factual, outcome_fn)
    observed_label = observed.labels[0] if observed.labels else ""

    per_step: list[StepAttribution] = []
    for k in range(len(factual.steps)):
        dist: OutcomeDistribution = await coalition_distribution(
            factual,
            held=set(range(k)),  # hold [0,k); resample k onward
            policy=policy,
            environment=environment,
            codec=codec,
            outcome_fn=outcome_fn,
            k_samples=k_samples,
            budget=budget,
            label=f"contrastive-{k}",
        )
        effect = prob_label_effect(observed, dist, bad_label, confidence=confidence)
        per_step.append(
            StepAttribution(index=k, effect=effect, p_bad_after_resample=dist.prob_label(bad_label))
        )

    # Causal locus = the LAST step whose resampling significantly reduces the bad outcome.
    locus: int | None = None
    for sa in per_step:
        if sa.rescues:
            locus = sa.index

    log.info(
        "contrastive attribution",
        trajectory_id=factual.trajectory_id,
        causal_locus=locus,
        n_steps=len(factual.steps),
    )
    return ContrastiveResult(
        trajectory_id=factual.trajectory_id,
        bad_label=bad_label,
        observed_label=observed_label,
        per_step=per_step,
        causal_locus=locus,
        confidence=confidence,
        k_samples=k_samples,
    )
