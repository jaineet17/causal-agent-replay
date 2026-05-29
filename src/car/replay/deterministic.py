"""Deterministic replay — the faithfulness proof Phase 0 rests on (PLAN.md s5.2).

If we cannot reconstruct the exact state at step k, we cannot intervene there; if replay does
not reproduce the recorded run (to the provider's determinism limit), nothing downstream is
trustworthy. This module proves two things, honestly:

  1. **State-reconstruction round-trip.** Independently rebuild the message history from the
     recorded actions/observations via the codec, and assert it equals every step's recorded
     ``state_before.messages`` (by digest). This proves ``state_before`` captured *enough* to
     re-issue the exact call.

  2. **Action-match rate.** Re-issue the recorded calls under the policy N times and measure how
     often the action reproduces. For a synthetic deterministic policy this is exactly 1.0,
     validating the replay *machinery* independent of provider noise. For a real provider it is
     a *measured, reported* metric — never asserted to be 1.0, because providers are not
     deterministic (RESEARCH s1: no seed on Anthropic, best-effort on OpenAI; FP/batch-size
     causes residual divergence).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from car.record.toolloop import MessageCodec
from car.schemas.scm import Policy, ReplayError
from car.schemas.trajectory import Action, Observation, Trajectory

log = structlog.get_logger(__name__)

# Replay fidelity is judged at the level of ``Trajectory.action_signature()`` — tool choice +
# structured args (or "final"), deliberately coarser than token identity, since residual FP
# noise rarely flips a confident argmax over tool selection (RESEARCH s1).


class StepMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    recorded_signature: str
    match_rate: float = Field(ge=0.0, le=1.0)
    observed: dict[str, int] = Field(description="signature -> count over the re-issue samples")


class ReplayReport(BaseModel):
    """The honest result of replaying a recorded trajectory."""

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    n_samples: int
    reconstruction_faithful: bool
    per_step: list[StepMatch]
    sequence_reproduction_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of full-loop replays reproducing the entire recorded signature.",
    )
    notes: str = ""

    @property
    def mean_step_match_rate(self) -> float:
        if not self.per_step:
            return 1.0
        return sum(s.match_rate for s in self.per_step) / len(self.per_step)


class RecordedEnvironment:
    """Replays recorded observations in order — the rr principle of injecting recorded inputs.

    During replay we do NOT re-call real tools; we feed back the observations captured at record
    time, so divergence is attributable to the *policy*, not the environment.
    """

    def __init__(self, observations: list[Observation]) -> None:
        self._observations = observations
        self._cursor = 0

    async def observe(self, action: Action) -> Observation:
        if self._cursor >= len(self._observations):
            raise ReplayError(
                f"replay requested observation #{self._cursor} but only "
                f"{len(self._observations)} were recorded; the replayed run is longer than the "
                f"recording (policy diverged)"
            )
        obs = self._observations[self._cursor]
        self._cursor += 1
        # Re-tag provenance: this came from the recording, not a live/mocked tool call.
        return Observation(tool_name=obs.tool_name, result=obs.result, source="recorded")


class DeterministicReplay:
    """Replays a recorded trajectory and reports faithfulness honestly."""

    def __init__(self, codec: MessageCodec) -> None:
        self._codec = codec

    # -- 1. state-reconstruction round-trip ---------------------------------------------------
    def verify_reconstruction(self, traj: Trajectory) -> bool:
        """Rebuild message history from recorded actions/observations; assert it matches each
        recorded ``state_before.messages`` exactly (by digest). Raises on the first divergence.
        """
        if not traj.steps:
            raise ReplayError(f"trajectory {traj.trajectory_id!r} has no steps")

        first_messages = traj.steps[0].state_before.messages
        if len(first_messages) != 1 or first_messages[0].get("role") != "user":
            raise ReplayError(
                f"expected the first state to hold exactly one user message, got {first_messages}"
            )
        rebuilt: list[dict[str, Any]] = [first_messages[0]]

        for step in traj.steps:
            recorded_digest = _digest(step.state_before.messages)
            rebuilt_digest = _digest(rebuilt)
            if recorded_digest != rebuilt_digest:
                raise ReplayError(
                    f"state reconstruction diverged at step {step.index}: recorded messages do "
                    f"not match messages rebuilt from prior actions/observations. The recording "
                    f"is not self-consistent and replay cannot be trusted."
                )
            if step.observation is not None:
                rebuilt.append(self._codec.assistant_message(step.action))
                rebuilt.append(self._codec.tool_result_message(step.action, step.observation))

        log.info("state reconstruction verified", trajectory_id=traj.trajectory_id)
        return True

    # -- 2. replay + action-match measurement -------------------------------------------------
    async def replay_once(self, traj: Trajectory, policy: Policy) -> Trajectory:
        """Re-run the whole loop with the recorded observations injected; return the replay."""
        from car.record.recorder import record_run

        observations = [s.observation for s in traj.steps if s.observation is not None]
        s0 = traj.steps[0].state_before
        return await record_run(
            trajectory_id=f"{traj.trajectory_id}:replay",
            policy=policy,
            environment=RecordedEnvironment(observations),
            codec=self._codec,
            system_prompt=s0.system_prompt,
            tool_schemas=s0.tool_schemas,
            user_input=_user_input_of(traj),
            sampling=s0.sampling,
            max_steps=len(traj.steps) + 2,
        )

    async def measure(
        self, traj: Trajectory, policy: Policy, *, n_samples: int = 8
    ) -> ReplayReport:
        """Verify reconstruction, then measure action-match over ``n_samples`` re-issues."""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        reconstruction_faithful = self.verify_reconstruction(traj)

        recorded_sig = traj.action_signature()
        per_step_observed: list[Counter[str]] = [Counter() for _ in traj.steps]
        sequence_reproductions = 0

        for _ in range(n_samples):
            replay = await self.replay_once(traj, policy)
            replay_sig = replay.action_signature()
            if replay_sig == recorded_sig:
                sequence_reproductions += 1
            for k in range(min(len(replay_sig), len(recorded_sig))):
                per_step_observed[k][replay_sig[k]] += 1

        per_step = [
            StepMatch(
                index=k,
                recorded_signature=recorded_sig[k],
                match_rate=(observed[recorded_sig[k]] / max(1, sum(observed.values()))),
                observed=dict(observed),
            )
            for k, observed in enumerate(per_step_observed)
        ]

        report = ReplayReport(
            trajectory_id=traj.trajectory_id,
            n_samples=n_samples,
            reconstruction_faithful=reconstruction_faithful,
            per_step=per_step,
            sequence_reproduction_rate=sequence_reproductions / n_samples,
            notes=(
                f"provider={traj.steps[0].state_before.provider}; "
                "match rates < 1.0 reflect provider nondeterminism (RESEARCH s1), not a bug."
            ),
        )
        log.info(
            "replay measured",
            trajectory_id=traj.trajectory_id,
            sequence_reproduction_rate=report.sequence_reproduction_rate,
            mean_step_match_rate=report.mean_step_match_rate,
        )
        return report


def _digest(messages: list[dict[str, Any]]) -> str:
    from car.schemas.trajectory import _canonical_digest

    return _canonical_digest(messages)


def _user_input_of(traj: Trajectory) -> str:
    first = traj.steps[0].state_before.messages[0]
    content = first.get("content")
    if isinstance(content, str):
        return content
    raise ReplayError(
        f"cannot extract user_input: first message content is {type(content).__name__}, not str "
        f"(non-string initial user content is not yet supported by deterministic replay)"
    )


# Re-export so ``measure`` and ``replay_once`` share one helper without import drift.
__all__ = [
    "DeterministicReplay",
    "RecordedEnvironment",
    "ReplayReport",
    "StepMatch",
]
