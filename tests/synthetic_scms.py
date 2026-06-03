"""Synthetic SCMs with KNOWN causal structure (PLAN.md s7) — the credibility anchor.

A heatmap that looks plausible but was never checked against a case where we know the right
answer is exactly the failure mode that makes attribution untrustworthy. These hand-built agents
have a ground-truth causal locus, so the effect estimators (Phase 2) and attribution (Phase 3)
can be validated, not merely admired.

  - ``pivotal_scm``: step 1 is deterministically pivotal — resampling it (or anything upstream,
    which re-rolls it) shifts the outcome to P(bad)=p; resampling step 2 (downstream) does not.
  - ``interaction_scm``: steps 1 and 3 form a two-step interaction — the outcome is bad ONLY if
    BOTH go wrong. Single-step resampling understates each step's importance; this is why Shapley
    exists (it recovers the interaction; contrastive does not).
"""

from __future__ import annotations

from car.outcome.functions import RuleOutcome, tool_called
from car.schemas.trajectory import Outcome, Trajectory

from .scm_fixtures import DictEnvironment, NoisyPolicy, final, tool_call

# --- shared actions ---------------------------------------------------------------------------
LOOKUP = tool_call("lookup_order", {"order_id": "A"}, text="looking up")
REFUND = tool_call("issue_refund", {"order_id": "A", "amount": 99.0}, text="refunding")
ESCALATE = tool_call("escalate", {"reason": "policy not met"}, text="escalating")

ENV = DictEnvironment(
    {
        "lookup_order": "{}",
        "issue_refund": "{}",
        "escalate": "{}",
        "bad_a": "{}",
        "good_a": "{}",
        "bad_b": "{}",
        "good_b": "{}",
    }
)


# --- outcome (rule-based; beyond dispute) -----------------------------------------------------
def refund_outcome() -> RuleOutcome:
    """Bad iff ``issue_refund`` was ever called (the support-agent failure)."""

    def rule(traj: Trajectory) -> Outcome:
        bad = tool_called(traj, "issue_refund")
        return Outcome(label="inappropriate_refund" if bad else "ok", score=1.0 if bad else 0.0)

    return RuleOutcome(rule)


def interaction_outcome() -> RuleOutcome:
    """Bad iff BOTH ``bad_a`` and ``bad_b`` were called (a two-step interaction)."""

    def rule(traj: Trajectory) -> Outcome:
        bad = tool_called(traj, "bad_a") and tool_called(traj, "bad_b")
        return Outcome(label="bad" if bad else "ok", score=1.0 if bad else 0.0)

    return RuleOutcome(rule)


# --- pivotal SCM ------------------------------------------------------------------------------
# Observed (bad) run: lookup -> refund -> final. Step 1 is the pivotal decision.
PIVOTAL_OBSERVED = [LOOKUP, REFUND, final("Done.")]
PIVOTAL_STEP = 1


def pivotal_policy(p_bad: float, seed: int = 0) -> NoisyPolicy:
    """At step 1, choose REFUND (bad) with probability ``p_bad``, else ESCALATE (good)."""
    return NoisyPolicy(
        base_actions=PIVOTAL_OBSERVED,
        noisy_step=PIVOTAL_STEP,
        option_a=REFUND,
        option_b=ESCALATE,
        p=p_bad,
        seed=seed,
    )


# --- two-step interaction SCM -----------------------------------------------------------------
BAD_A = tool_call("bad_a", {}, text="a-bad")
GOOD_A = tool_call("good_a", {}, text="a-good")
BAD_B = tool_call("bad_b", {}, text="b-bad")
GOOD_B = tool_call("good_b", {}, text="b-good")

# Observed (bad) run: bad_a -> bad_b -> final. Steps 0 and 1 each independently pivotal; the
# outcome is bad only when BOTH are bad.
INTERACTION_OBSERVED = [BAD_A, BAD_B, final("Done.")]
INTERACTION_STEPS = (0, 1)
