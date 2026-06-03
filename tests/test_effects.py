"""Phase 2 DoD: on a synthetic SCM, the effect estimator recovers the KNOWN effect within its CI.

The pivotal SCM has a ground-truth answer: resampling the pivotal step (or anything upstream,
which re-rolls it) yields P(bad)=p; resampling downstream of it yields no change. We check the
estimator recovers both — with confidence intervals that contain the true values — and that
effects are never reported as bare points. Also: outcome plumbing, distances, and a demonstration
that single-step resampling under-attributes a two-step interaction (why Shapley is needed).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from car.attribute.effects import (
    OutcomeDistribution,
    mean_score_effect,
    observed_distribution,
    prob_label_effect,
    score_branch,
    tv_distance,
    wasserstein,
    wilson_interval,
)
from car.outcome.functions import JudgeOutcome, score_trajectory, tool_called
from car.record.recorder import SyntheticCodec, record_run
from car.replay.intervene import InterventionRunner
from car.schemas.intervention import DoResample
from car.schemas.trajectory import Trajectory

from .scm_fixtures import ScriptedPolicy
from .synthetic_scms import (
    ENV,
    INTERACTION_OBSERVED,
    PIVOTAL_OBSERVED,
    PIVOTAL_STEP,
    interaction_outcome,
    pivotal_policy,
    refund_outcome,
)

CODEC = SyntheticCodec()
P_BAD = 0.3
K = 300


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


# -- the headline DoD ---------------------------------------------------------------------------
async def test_effect_estimator_recovers_known_effect_within_ci() -> None:
    observed = await _observed(PIVOTAL_OBSERVED)
    fn = refund_outcome()
    baseline = await observed_distribution(observed, fn)
    assert baseline.prob_label("inappropriate_refund") == 1.0  # the observed run is bad

    # do_resample at the PIVOTAL step: known P(bad) = P_BAD.
    branch = await InterventionRunner(CODEC).apply(
        observed,
        DoResample(intervention_id="resample-pivotal", step=PIVOTAL_STEP),
        policy=pivotal_policy(P_BAD, seed=1),
        environment=ENV,
        k_samples=K,
    )
    dist = await score_branch(branch, fn)

    # 1) P(bad) CI contains the true p.
    p_hat, lo, hi = dist.prob_label_ci("inappropriate_refund")
    assert lo <= P_BAD <= hi, f"true p={P_BAD} not in CI [{lo:.3f}, {hi:.3f}] (p_hat={p_hat:.3f})"

    # 2) The effect vs the observed bad run is ~ (p - 1) = -0.7, recovered within its CI,
    #    and is statistically distinguishable from zero.
    effect = prob_label_effect(baseline, dist, "inappropriate_refund", seed=7)
    true_effect = P_BAD - 1.0
    assert effect.ci_low <= true_effect <= effect.ci_high
    assert effect.is_significant
    assert effect.n_intervened == K


async def test_resampling_downstream_of_the_locus_has_no_effect() -> None:
    """Resampling AFTER the pivotal step leaves the committed bad decision intact (effect ~ 0)."""
    observed = await _observed(PIVOTAL_OBSERVED)
    fn = refund_outcome()
    baseline = await observed_distribution(observed, fn)
    branch = await InterventionRunner(CODEC).apply(
        observed,
        DoResample(intervention_id="resample-after", step=PIVOTAL_STEP + 1),  # the final step
        policy=pivotal_policy(P_BAD, seed=1),
        environment=ENV,
        k_samples=50,
    )
    dist = await score_branch(branch, fn)
    assert dist.prob_label("inappropriate_refund") == 1.0  # still bad every time
    effect = prob_label_effect(baseline, dist, "inappropriate_refund", seed=7)
    assert effect.ci_low <= 0.0 <= effect.ci_high
    assert not effect.is_significant


# -- estimator unit checks ----------------------------------------------------------------------
def test_wilson_interval_known_values() -> None:
    # All successes, small n -> upper bound 1.0, lower bound well below 1.
    p, lo, hi = wilson_interval(10, 10)
    assert p == 1.0 and hi == pytest.approx(1.0) and lo < 0.95
    # Symmetric case p=0.5.
    p, lo, hi = wilson_interval(50, 100)
    assert p == 0.5 and lo < 0.5 < hi
    # n=0 is handled (maximally uncertain), not a divide-by-zero.
    assert wilson_interval(0, 0) == (0.0, 0.0, 1.0)


def test_outcome_distribution_stats() -> None:
    d = OutcomeDistribution(scores=[1.0, 1.0, 0.0, 0.0], labels=["bad", "bad", "ok", "ok"])
    assert d.n == 4
    assert d.mean() == 0.5
    assert d.prob_label("bad") == 0.5
    assert d.label_indicator("bad") == [1.0, 1.0, 0.0, 0.0]


def test_distribution_distances() -> None:
    same = OutcomeDistribution(scores=[1.0, 1.0, 1.0], labels=["bad"] * 3)
    other = OutcomeDistribution(scores=[0.0, 0.0, 0.0], labels=["ok"] * 3)
    assert tv_distance(same, same) == 0.0
    assert tv_distance(same, other) == pytest.approx(1.0)
    assert wasserstein(same, other) == pytest.approx(1.0)


async def test_mean_score_effect_has_ci() -> None:
    a = OutcomeDistribution(scores=[0.0] * 20, labels=["ok"] * 20)
    b = OutcomeDistribution(scores=[1.0] * 20, labels=["bad"] * 20)
    eff = mean_score_effect(a, b, seed=0)
    assert eff.point == pytest.approx(1.0)
    assert eff.ci_low <= 1.0 <= eff.ci_high
    assert eff.statistic == "mean_score"


# -- outcome plumbing ---------------------------------------------------------------------------
async def test_score_trajectory_sets_outcome() -> None:
    observed = await _observed(PIVOTAL_OBSERVED)
    outcome = await score_trajectory(observed, refund_outcome())
    assert outcome.label == "inappropriate_refund"
    assert observed.outcome is not None and observed.outcome.score == 1.0


def test_tool_called_helper() -> None:
    async def _build() -> Trajectory:
        return await _observed(PIVOTAL_OBSERVED)

    import asyncio

    traj = asyncio.run(_build())
    assert tool_called(traj, "issue_refund")
    assert not tool_called(traj, "escalate")


# -- interaction SCM: single-step resampling under-attributes (motivates Shapley) ---------------
async def test_interaction_outcome_requires_both_steps() -> None:
    observed = await _observed(INTERACTION_OBSERVED)  # bad_a -> bad_b -> final
    fn = interaction_outcome()
    out = await score_trajectory(observed, fn)
    assert out.label == "bad"  # both bad -> bad

    # A run where only ONE of the two pivotal steps is bad is NOT bad.
    from .scm_fixtures import final, tool_call

    only_a = await _observed([tool_call("bad_a", {}), tool_call("good_b", {}), final("x")])
    assert (await fn.score(only_a)).label == "ok"


# -- judge outcome (real path, fake client; no server/key/money) --------------------------------
class _FakeJudgeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    @property
    def chat(self) -> Any:
        async def create(**_: Any) -> Any:
            msg = SimpleNamespace(content=self._reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        return SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_judge_outcome_parses_score() -> None:
    reply = 'Here is my grade: {"score": 0.9, "label": "bad", "reason": "refunded"}'
    client = _FakeJudgeClient(reply)
    judge = JudgeOutcome(client=client, model="local", rubric="bad if it refunds without cause")
    observed = await _observed(PIVOTAL_OBSERVED)
    out = await judge.score(observed)
    assert out.score == pytest.approx(0.9)
    assert out.label == "bad"
    assert out.detail["reason"] == "refunded"
