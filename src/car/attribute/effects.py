"""Outcome distributions and causal-effect estimators (PLAN.md s5.4 effects.py).

Counterfactual replay yields a *distribution* over outcomes, never a single path. Every effect
is therefore an estimate over a finite K, and is reported WITH a Monte-Carlo confidence interval
— never a bare point estimate (CLAUDE.md non-negotiable).

  - ``OutcomeDistribution`` — the K scored outcomes of a branch (or the size-1 observed run).
  - proportion CIs via the Wilson score interval (well-behaved at p near 0/1 and small n).
  - difference-of-distributions effects via bootstrap percentile CIs.
  - distributional distances (total-variation on the score histogram; Wasserstein via scipy).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from car.outcome.functions import OutcomeFunction
from car.replay.intervene import Branch
from car.schemas.trajectory import Outcome, Trajectory

log = structlog.get_logger(__name__)


class OutcomeDistribution(BaseModel):
    """The scored outcomes of a set of counterfactual samples."""

    model_config = ConfigDict(extra="forbid")

    scores: list[float]
    labels: list[str]

    @classmethod
    def from_outcomes(cls, outcomes: Sequence[Outcome]) -> OutcomeDistribution:
        return cls(
            scores=[o.score for o in outcomes],
            labels=[o.label for o in outcomes],
        )

    @property
    def n(self) -> int:
        return len(self.scores)

    def mean(self) -> float:
        if self.n == 0:
            return 0.0
        return float(np.mean(self.scores))

    def prob_label(self, label: str) -> float:
        if self.n == 0:
            return 0.0
        return sum(1 for x in self.labels if x == label) / self.n

    def label_indicator(self, label: str) -> list[float]:
        return [1.0 if x == label else 0.0 for x in self.labels]

    def prob_label_ci(self, label: str, confidence: float = 0.95) -> tuple[float, float, float]:
        """(point, low, high) for P(label) via the Wilson score interval."""
        k = sum(1 for x in self.labels if x == label)
        return wilson_interval(k, self.n, confidence)


class EffectEstimate(BaseModel):
    """A causal-effect estimate of an intervention, with its Monte-Carlo CI."""

    model_config = ConfigDict(extra="forbid")

    statistic: str = Field(description='e.g. "P(label=inappropriate_refund)" or "mean_score"')
    point: float
    ci_low: float
    ci_high: float
    confidence: float
    n_baseline: int
    n_intervened: int

    @property
    def is_significant(self) -> bool:
        """True if the CI for the difference excludes 0 (an effect distinguishable from none)."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p_hat, low, high)."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _bootstrap_diff_ci(
    baseline: Sequence[float],
    intervened: Sequence[float],
    stat: Callable[[np.ndarray], float],
    *,
    confidence: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for stat(intervened) - stat(baseline)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(intervened, dtype=float)
    point = stat(b) - (stat(a) if a.size else 0.0)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        rb = rng.choice(b, size=b.size, replace=True)
        sa = stat(rng.choice(a, size=a.size, replace=True)) if a.size else 0.0
        diffs[i] = stat(rb) - sa
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(diffs, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return (float(point), float(low), float(high))


def prob_label_effect(
    baseline: OutcomeDistribution,
    intervened: OutcomeDistribution,
    label: str,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> EffectEstimate:
    """Effect of the intervention on P(outcome == label): P_intervened - P_baseline, with CI."""
    point, low, high = _bootstrap_diff_ci(
        baseline.label_indicator(label),
        intervened.label_indicator(label),
        lambda x: float(np.mean(x)) if x.size else 0.0,
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
    )
    return EffectEstimate(
        statistic=f"P(label={label})",
        point=point,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n_baseline=baseline.n,
        n_intervened=intervened.n,
    )


def mean_score_effect(
    baseline: OutcomeDistribution,
    intervened: OutcomeDistribution,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> EffectEstimate:
    """Effect of the intervention on the mean outcome score, with a bootstrap CI."""
    point, low, high = _bootstrap_diff_ci(
        baseline.scores,
        intervened.scores,
        lambda x: float(np.mean(x)) if x.size else 0.0,
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
    )
    return EffectEstimate(
        statistic="mean_score",
        point=point,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n_baseline=baseline.n,
        n_intervened=intervened.n,
    )


def tv_distance(a: OutcomeDistribution, b: OutcomeDistribution, *, bins: int = 10) -> float:
    """Total-variation distance between two score distributions (histogram, common binning)."""
    if a.n == 0 or b.n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ha, _ = np.histogram(a.scores, bins=edges, density=False)
    hb, _ = np.histogram(b.scores, bins=edges, density=False)
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    return float(0.5 * np.abs(pa - pb).sum())


def wasserstein(a: OutcomeDistribution, b: OutcomeDistribution) -> float:
    """1-Wasserstein (earth-mover) distance between two score distributions."""
    if a.n == 0 or b.n == 0:
        return 0.0
    return float(stats.wasserstein_distance(a.scores, b.scores))


async def score_branch(branch: Branch, fn: OutcomeFunction) -> OutcomeDistribution:
    """Score every child of a branch (run-forward-K already produced them) into a distribution."""
    outcomes: list[Outcome] = []
    for child in branch.children:
        outcome = await fn.score(child)
        child.outcome = outcome
        outcomes.append(outcome)
    log.info("scored branch", intervention_id=branch.intervention_id, k=branch.k_samples)
    return OutcomeDistribution.from_outcomes(outcomes)


async def observed_distribution(traj: Trajectory, fn: OutcomeFunction) -> OutcomeDistribution:
    """The (degenerate, size-1) outcome distribution of the observed run — a baseline."""
    outcome = await fn.score(traj)
    traj.outcome = outcome
    return OutcomeDistribution.from_outcomes([outcome])
