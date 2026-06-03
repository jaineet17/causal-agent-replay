"""Sampling/cost budget + circuit breaker for attribution (PLAN.md s5.7).

Attribution is many forward rollouts (n steps x K samples for contrastive; M permutations x
(n+1) x samples_per_eval for Shapley). Before any run we can estimate the cost; during a run every
rollout is charged, and the breaker trips before a cap is exceeded rather than after.

For free local models ``cost_per_sample_usd`` is 0, so only the sample cap bounds compute; for
hosted models the USD cap is the hard ceiling the CLI refuses to cross without an explicit --yes.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when a charge would exceed the sample or cost cap (no silent overruns)."""


class Budget:
    """A counter with hard caps. ``charge`` trips the breaker BEFORE exceeding a cap."""

    def __init__(
        self,
        *,
        max_samples: int | None = None,
        max_cost_usd: float | None = None,
        cost_per_sample_usd: float = 0.0,
    ) -> None:
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be >= 0")
        self._max_samples = max_samples
        self._max_cost_usd = max_cost_usd
        self._cost_per_sample_usd = cost_per_sample_usd
        self._used = 0

    @property
    def used_samples(self) -> int:
        return self._used

    @property
    def used_cost_usd(self) -> float:
        return self._used * self._cost_per_sample_usd

    def estimate_cost_usd(self, n_samples: int) -> float:
        return n_samples * self._cost_per_sample_usd

    def would_exceed(self, n_samples: int) -> bool:
        projected = self._used + n_samples
        if self._max_samples is not None and projected > self._max_samples:
            return True
        if self._max_cost_usd is not None:
            return projected * self._cost_per_sample_usd > self._max_cost_usd + 1e-9
        return False

    def charge(self, n_samples: int) -> None:
        if n_samples < 0:
            raise ValueError("cannot charge a negative number of samples")
        if self.would_exceed(n_samples):
            raise BudgetExceeded(
                f"charge of {n_samples} would exceed budget "
                f"(used={self._used}, max_samples={self._max_samples}, "
                f"max_cost_usd={self._max_cost_usd}, "
                f"projected_cost=${(self._used + n_samples) * self._cost_per_sample_usd:.4f})"
            )
        self._used += n_samples

    @classmethod
    def unlimited(cls) -> Budget:
        """A no-op budget (useful for synthetic/free runs where compute is the only limit)."""
        return cls()
