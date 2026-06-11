"""Surrogate-counterfactual attribution over a static failure log (Who&When protocol).

The same do_resample semantics as ``car.attribute.contrastive`` (hold the prefix factual,
regenerate step k, roll forward, judge), transplanted from the tool-loop SCM to the message-chain
SCM of a multi-agent log:

  - the **speaker schedule is held fixed** (who talks when is recorded structure; we intervene on
    message *content*) — agent steps regenerate via the agent surrogate, env steps via the env
    simulator;
  - the factual run failed (it's a failure benchmark), so the per-step effect is
    P(fail | resample k) - 1, estimated with the SAME ``prob_label_effect`` bootstrap CIs as the
    core, with CI-aware early stopping in chunks;
  - the **causal locus = the latest step whose rescue CI excludes 0** (the core's
    point-of-commitment rule), and the predicted ``(mistake_agent, mistake_step)`` is the speaker
    and index at the locus.

All effects are *surrogate-counterfactual*: they measure causal structure under the surrogate
world-model, and the result records the sanity floors needed to interpret them (factual-replay
failure reproduction; realized K per step).
"""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel, ConfigDict, Field

from car.attribute.effects import EffectEstimate, OutcomeDistribution, prob_label_effect
from car.bench.surrogate import LogWorldModel
from car.bench.whowhen import LogStep, WhoWhenInstance
from car.schemas.trajectory import Outcome

log = structlog.get_logger(__name__)


class StepEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    agent: str
    is_env: bool
    p_fail_after_resample: float
    effect: EffectEstimate
    k_realized: int

    @property
    def rescues(self) -> bool:
        return self.effect.is_significant and self.effect.point < 0.0


class LogAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    factual_still_fails: bool = Field(
        description="Sanity floor: the judge on the UNMODIFIED log must say 'fails'."
    )
    per_step: list[StepEffect]
    locus: int | None
    predicted_agent: str | None
    predicted_step: int | None
    prediction_confident: bool = Field(
        description="True when the prediction is the CI-gated locus; False when it fell back to "
        "the argmax-rescue step (benchmarks expect a prediction on every instance)."
    )
    label_agent: str
    label_step: int
    total_rollouts: int

    @property
    def agent_correct(self) -> bool:
        return self.predicted_agent == self.label_agent

    def step_within(self, tolerance: int = 0) -> bool:
        if self.predicted_step is None:
            return False
        return abs(self.predicted_step - self.label_step) <= tolerance


async def _rollout(
    instance: WhoWhenInstance,
    world: LogWorldModel,
    k: int,
    *,
    horizon: int | None,
) -> bool:
    """One counterfactual sample: regenerate step k, continue the schedule, judge."""
    transcript: list[LogStep] = list(instance.history[:k])
    end = len(instance.history) if horizon is None else min(len(instance.history), k + horizon)
    for index in range(k, end):
        factual = instance.history[index]
        # Every step from k onward is re-decided under the surrogate world-model (run-forward),
        # following the factual speaker schedule.
        content = (
            await world.env_message(instance, transcript, factual.agent)
            if factual.is_env
            else await world.next_message(instance, transcript, factual.agent)
        )
        transcript.append(
            LogStep(index=index, agent=factual.agent, content=content, is_env=factual.is_env)
        )
    return await world.still_fails(instance, transcript)


async def attribute_log(
    instance: WhoWhenInstance,
    world: LogWorldModel,
    *,
    k_max: int = 8,
    chunk: int = 4,
    horizon: int | None = None,
    confidence: float = 0.95,
    max_concurrency: int = 4,
    seed: int = 0,
) -> LogAttribution:
    """Attribute the failure to a step by per-step surrogate resampling with early stopping."""
    factual_fails = await world.still_fails(instance, instance.history)
    baseline = OutcomeDistribution.from_outcomes([Outcome(label="fail", score=1.0)])
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(k: int) -> bool:
        async with semaphore:
            return await _rollout(instance, world, k, horizon=horizon)

    per_step: list[StepEffect] = []
    total = 0
    for step in instance.history:
        outcomes: list[Outcome] = []
        effect: EffectEstimate | None = None
        while len(outcomes) < k_max:
            n = min(chunk, k_max - len(outcomes))
            fails = await asyncio.gather(*(bounded(step.index) for _ in range(n)))
            total += n
            outcomes.extend(
                Outcome(label="fail" if f else "ok", score=1.0 if f else 0.0) for f in fails
            )
            dist = OutcomeDistribution.from_outcomes(outcomes)
            effect = prob_label_effect(baseline, dist, "fail", confidence=confidence, seed=seed)
            # Early stop once the verdict for this step is settled: a significant rescue, or a
            # full chunk-run with zero rescues observed (the CI cannot become significant).
            if effect.is_significant or (
                dist.prob_label("fail") == 1.0 and len(outcomes) >= chunk * 2
            ):
                break
        assert effect is not None
        dist = OutcomeDistribution.from_outcomes(outcomes)
        per_step.append(
            StepEffect(
                index=step.index,
                agent=step.agent,
                is_env=step.is_env,
                p_fail_after_resample=dist.prob_label("fail"),
                effect=effect,
                k_realized=len(outcomes),
            )
        )

    locus: int | None = None
    for se in per_step:
        if se.rescues and not se.is_env:  # the benchmark labels AGENT mistakes
            locus = se.index

    # The benchmark expects a prediction on every instance: when no step clears the CI gate,
    # fall back to point estimates — but apply the SAME point-of-commitment logic. Under
    # run-forward, resampling an early step re-rolls everything downstream (including the true
    # mistake), so early steps rescue often and a plain argmax is biased early (observed on
    # AG/2 in the pilot). Instead: the LATEST agent step whose rescue rate is within tolerance
    # of the maximum.
    predicted_step = locus
    confident = locus is not None
    if predicted_step is None:
        agent_steps = [se for se in per_step if not se.is_env]
        if agent_steps:
            best_rescue = max(-se.effect.point for se in agent_steps)
            tolerance = max(0.8 * best_rescue, best_rescue - 0.15)
            for se in agent_steps:
                if -se.effect.point >= tolerance:
                    predicted_step = se.index
    predicted_agent = instance.history[predicted_step].agent if predicted_step is not None else None

    result = LogAttribution(
        instance_id=instance.instance_id,
        factual_still_fails=factual_fails,
        per_step=per_step,
        locus=locus,
        predicted_agent=predicted_agent,
        predicted_step=predicted_step,
        prediction_confident=confident,
        label_agent=instance.mistake_agent,
        label_step=instance.mistake_step,
        total_rollouts=total,
    )
    log.info(
        "attributed log",
        instance=instance.instance_id,
        locus=locus,
        label=instance.mistake_step,
        rollouts=total,
        factual_fails=factual_fails,
    )
    return result
