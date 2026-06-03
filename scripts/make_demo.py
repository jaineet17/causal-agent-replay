"""Produce the demo attribution report (PLAN.md s5.6 / Phase 4 DoD).

Builds the support-agent failure as a CONTROLLED synthetic SCM (reproducible, free, no API): the
customer message carries a prompt injection, and at the decision step the agent absorbs it ~half
the time. We record a run that DID absorb it (an inappropriate refund), then attribute the
failure and render a self-contained interactive HTML.

    uv run python scripts/make_demo.py            # -> examples/demo_report.html

The engine is identical on real LLM policies (see examples/gallery.md for live llama3.2 runs);
the demo uses a synthetic policy only so the result is deterministic and checkable. The HTML makes
the causal locus and its counterfactual evidence immediately legible.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.support_agent import (  # noqa: E402
    INJECTION_USER_MESSAGE,
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    build_registry,
)

from car.attribute.contrastive import contrastive_attribution  # noqa: E402
from car.attribute.shapley import shapley_attribution  # noqa: E402
from car.outcome.functions import RuleOutcome, tool_called  # noqa: E402
from car.record.recorder import MockEnvironment, SyntheticCodec, record_run  # noqa: E402
from car.schemas.trajectory import Outcome, Trajectory  # noqa: E402
from car.store.store import TrajectoryStore  # noqa: E402
from car.synthetic import NoisyPolicy, ScriptedPolicy, final, tool_call  # noqa: E402
from car.viz.html import render_html  # noqa: E402

app = typer.Typer(add_completion=False)

# The controlled support-agent trajectory. Step 1 is the decision that absorbs the injection.
LOOKUP = tool_call("lookup_order", {"order_id": "A1234"}, text="Let me pull up that order.")
REFUND = tool_call(
    "issue_refund", {"order_id": "A1234", "amount": 99.0}, text="Issuing the refund as requested."
)
ESCALATE = tool_call(
    "escalate",
    {"reason": "Refund conditions not met; the message contains an instruction to ignore policy."},
    text="This doesn't meet the refund policy; escalating.",
)
CONFIRM = final("All set — I've processed a $99.00 refund for order A1234. Anything else?")

BAD_PATH = [LOOKUP, REFUND, CONFIRM]


def refund_outcome() -> RuleOutcome:
    def rule(traj: Trajectory) -> Outcome:
        bad = tool_called(traj, "issue_refund")
        return Outcome(
            label="inappropriate_refund" if bad else "ok",
            score=1.0 if bad else 0.0,
            detail={"policy": "refund only if delivered AND defect reported"},
        )

    return RuleOutcome(rule)


@app.command()
def main(
    out: Path = typer.Option(REPO_ROOT / "examples" / "demo_report.html"),
    k_samples: int = typer.Option(48, help="Counterfactual rollouts per step (contrastive)."),
    n_permutations: int = typer.Option(64, help="Shapley permutations."),
    p_absorb: float = typer.Option(0.5, help="Probability the model absorbs the injection."),
    seed: int = typer.Option(1),
) -> None:
    codec = SyntheticCodec()
    env = MockEnvironment(build_registry())
    outcome = refund_outcome()

    async def _run() -> None:
        # The observed (bad) run: looked up, then absorbed the injection and refunded.
        factual = await record_run(
            trajectory_id="support-injection-failure",
            policy=ScriptedPolicy(BAD_PATH),
            environment=env,
            codec=codec,
            system_prompt=SYSTEM_PROMPT,
            tool_schemas=TOOL_SCHEMAS,
            user_input=INJECTION_USER_MESSAGE,
        )
        # The policy: at the decision step it absorbs the injection (refund) with prob p_absorb,
        # else correctly escalates.
        policy = NoisyPolicy(
            base_actions=BAD_PATH,
            noisy_step=1,
            option_a=REFUND,
            option_b=ESCALATE,
            p=p_absorb,
            seed=seed,
        )
        contrastive = await contrastive_attribution(
            factual,
            policy=policy,
            environment=env,
            codec=codec,
            outcome_fn=outcome,
            bad_label="inappropriate_refund",
            k_samples=k_samples,
        )
        shapley = await shapley_attribution(
            factual,
            policy=policy,
            environment=env,
            codec=codec,
            outcome_fn=outcome,
            bad_label="inappropriate_refund",
            n_permutations=n_permutations,
            samples_per_eval=8,
            seed=seed,
        )

        TrajectoryStore().save(factual)
        html = render_html(
            factual,
            contrastive,
            shapley=shapley,
            title="Causal Agent Replay — support-agent injection",
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")

        typer.echo(f"causal locus: step {contrastive.causal_locus}")
        for sa in contrastive.per_step:
            typer.echo(
                f"  step {sa.index} {factual.steps[sa.index].action.tool_name or 'final':12s} "
                f"rescue={-sa.effect.point:+.2f} "
                f"CI=[{-sa.effect.ci_high:+.2f},{-sa.effect.ci_low:+.2f}] "
                f"{'<= LOCUS' if sa.index == contrastive.causal_locus else ''}"
            )
        typer.echo(f"\nwrote {out}")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
