"""Phase 3 DoD: attribution recovers the KNOWN causal structure of the synthetic SCMs (s7).

- contrastive recovers the single pivotal step (and resampling downstream of it shows no effect);
- Shapley recovers the two-step interaction, splitting credit ~0.5/0.5 (efficiency holds);
- contrastive alone does NOT express the interaction (it reports a single locus) — the concrete
  reason shapley.py exists;
- the budget circuit breaker truncates a Shapley run cleanly.

A heatmap that hasn't been checked against known ground truth is exactly the failure mode that
makes attribution untrustworthy (PLAN.md s13). These checks are non-negotiable.
"""

from __future__ import annotations

from typing import Any

import pytest

from car.attribute.contrastive import contrastive_attribution
from car.attribute.shapley import estimate_rollouts, shapley_attribution
from car.budget.budget import Budget
from car.record.recorder import SyntheticCodec, record_run
from car.schemas.trajectory import Trajectory

from .scm_fixtures import ScriptedPolicy
from .synthetic_scms import (
    ENV,
    INTERACTION_OBSERVED,
    PIVOTAL_OBSERVED,
    PIVOTAL_STEP,
    interaction_outcome,
    interaction_policy,
    pivotal_policy,
    refund_outcome,
)

CODEC = SyntheticCodec()


async def _observed(actions: list[Any]) -> Trajectory:
    return await record_run(
        trajectory_id="obs",
        policy=ScriptedPolicy(actions),
        environment=ENV,
        codec=CODEC,
        system_prompt="sys",
        tool_schemas=[],
        user_input="go",
    )


# -- contrastive recovers the pivotal step ------------------------------------------------------
async def test_contrastive_recovers_the_pivotal_step() -> None:
    factual = await _observed(PIVOTAL_OBSERVED)
    result = await contrastive_attribution(
        factual,
        policy=pivotal_policy(0.3, seed=1),
        environment=ENV,
        codec=CODEC,
        outcome_fn=refund_outcome(),
        bad_label="inappropriate_refund",
        k_samples=80,
    )
    # The causal locus is the pivotal decision step, identified despite the run-forward confound.
    assert result.causal_locus == PIVOTAL_STEP
    # Resampling AT the pivotal step rescues; resampling AFTER it does not.
    assert result.per_step[PIVOTAL_STEP].rescues
    assert not result.per_step[PIVOTAL_STEP + 1].rescues


# -- Shapley recovers the two-step interaction --------------------------------------------------
async def test_shapley_recovers_the_two_step_interaction() -> None:
    factual = await _observed(INTERACTION_OBSERVED)  # bad_a -> bad_b -> final (both bad)
    result = await shapley_attribution(
        factual,
        policy=interaction_policy(0.3, seed=2),
        environment=ENV,
        codec=CODEC,
        outcome_fn=interaction_outcome(),
        bad_label="bad",
        n_permutations=80,
        samples_per_eval=12,
        antithetic=True,
        seed=5,
    )
    phi = {s.index: s.value for s in result.per_step}
    # The two interacting steps split credit roughly equally (true value ~0.455 each).
    assert phi[0] == pytest.approx(0.455, abs=0.15)
    assert phi[1] == pytest.approx(0.455, abs=0.15)
    assert abs(phi[0] - phi[1]) < 0.2
    # The inert final step gets ~0 and both interacting steps are significant.
    assert abs(phi[2]) < 0.1
    assert result.per_step[0].is_significant and result.per_step[1].is_significant
    # Efficiency: sum of Shapley values ~ v(N) - v(empty) = 1 - q^2 = 0.91.
    assert result.efficiency_sum == pytest.approx(0.91, abs=0.2)


async def test_contrastive_underattributes_the_interaction() -> None:
    """Contrastive reports a single locus, not the 0.5/0.5 split — the reason Shapley exists."""
    factual = await _observed(INTERACTION_OBSERVED)
    result = await contrastive_attribution(
        factual,
        policy=interaction_policy(0.3, seed=2),
        environment=ENV,
        codec=CODEC,
        outcome_fn=interaction_outcome(),
        bad_label="bad",
        k_samples=80,
    )
    # Contrastive collapses the joint cause to ONE locus (the latest necessary step) ...
    assert result.causal_locus == 1
    # ... and its single-step effects do NOT sum to ~1 the way a credit decomposition should
    # (they double-count: each early step's resample re-rolls the whole tail).
    total = sum(abs(s.effect.point) for s in result.per_step)
    assert total > 1.2  # over-counts vs the true total contribution of 0.91


# -- budget circuit breaker ---------------------------------------------------------------------
async def test_shapley_respects_the_budget() -> None:
    factual = await _observed(INTERACTION_OBSERVED)
    budget = Budget(max_samples=200)  # far fewer than a full run needs
    result = await shapley_attribution(
        factual,
        policy=interaction_policy(0.3, seed=2),
        environment=ENV,
        codec=CODEC,
        outcome_fn=interaction_outcome(),
        bad_label="bad",
        n_permutations=500,
        samples_per_eval=10,
        budget=budget,
    )
    assert result.budget_truncated
    assert result.permutations_completed < 500
    assert budget.used_samples <= 200


def test_estimate_rollouts() -> None:
    # n=3 steps, 10 permutations, 2 samples/eval, antithetic -> 10*2*(3+1)*2 = 160.
    assert estimate_rollouts(3, 10, 2, antithetic=True) == 160
    assert estimate_rollouts(3, 10, 2, antithetic=False) == 80
