"""Phase 6 harness validation: the log-attribution machinery recovers a PLANTED causal locus.

Same discipline as every phase: before the harness touches real Who&When data, it must recover
ground truth we control. The fake world-model plants the causal structure: resampling step 2
rescues the run with probability 0.7; resampling any later step never does. The loader is tested
on self-authored fixtures shaped like the real instances (verified against the dataset
2026-06-11) — no dataset content is committed.
"""

from __future__ import annotations

import random

from car.bench.attribute_log import attribute_log
from car.bench.whowhen import LogStep, WhoWhenInstance, parse_instance

# -- loader ----------------------------------------------------------------------------------------

AG_SHAPED = {
    "is_correct": False,
    "question": "How many even-numbered addresses?",
    "question_ID": "x-1",
    "level": "2",
    "ground_truth": "8",
    "history": [
        {"content": "Let me compute.", "role": "assistant", "name": "Excel_Expert"},
        {"content": "exitcode: 0\nOutput: 4", "role": "user", "name": "Computer_terminal"},
        {"content": "The answer is 4. TERMINATE", "role": "user", "name": "Verifier"},
    ],
    "mistake_agent": "Excel_Expert",
    "mistake_step": "0",
    "mistake_reason": "Bad edge-case handling.",
    "system_prompt": {"Excel_Expert": "## Your role\nExcel expert."},
}

HC_SHAPED = {
    "is_corrected": False,
    "question": "What homework was assigned?",
    "question_ID": "x-2",
    "level": "1",
    "groundtruth": "pages 5-7",
    "history": [
        {"content": "Hi, I was out sick.", "role": "human"},
        {"content": "Initial plan: ...", "role": "Orchestrator (thought)"},
        {"content": "Address: file:///x.mp3", "role": "FileSurfer"},
    ],
    "mistake_agent": "FileSurfer",
    "mistake_step": "2",
    "mistake_reason": "Misread the file.",
}


def test_parses_ag_shape() -> None:
    inst = parse_instance(AG_SHAPED, instance_id="ag/1", subset="Algorithm-Generated")
    assert inst.mistake_step == 0  # str -> int
    assert inst.ground_truth == "8"
    assert [s.agent for s in inst.history] == ["Excel_Expert", "Computer_terminal", "Verifier"]
    assert [s.is_env for s in inst.history] == [False, True, False]
    assert "Excel_Expert" in inst.system_prompts


def test_parses_hc_shape() -> None:
    inst = parse_instance(HC_SHAPED, instance_id="hc/51", subset="Hand-Crafted")
    assert inst.ground_truth == "pages 5-7"  # 'groundtruth' spelling
    assert [s.agent for s in inst.history] == ["human", "Orchestrator", "FileSurfer"]
    assert inst.history[0].is_env  # 'human' is environment, not an attributable agent
    assert inst.mistake_step == 2


# -- attribution recovers a planted locus ---------------------------------------------------


def _instance(n_steps: int = 5, label_step: int = 2) -> WhoWhenInstance:
    return WhoWhenInstance(
        instance_id="synth/1",
        subset="Algorithm-Generated",
        question="q",
        ground_truth="42",
        history=[
            LogStep(index=i, agent=f"Agent{i}", content=f"factual message {i}", is_env=False)
            for i in range(n_steps)
        ],
        mistake_agent=f"Agent{label_step}",
        mistake_step=label_step,
    )


class PlantedWorld:
    """Resampling step PIVOT rescues w.p. ``p``; later steps never rescue; earlier ones re-roll
    the pivot (run-forward, like the real surrogate). Judge reads the marker off the transcript."""

    PIVOT = 2

    def __init__(self, p: float = 0.7, seed: int = 0) -> None:
        self._p, self._seed, self._draws = p, seed, 0

    async def next_message(
        self, instance: WhoWhenInstance, prefix: list[LogStep], speaker: str
    ) -> str:
        index = len(prefix)
        if index == self.PIVOT:
            rng = random.Random(self._seed * 1_000_003 + self._draws)
            self._draws += 1
            return "GOOD: the answer is 42" if rng.random() < self._p else "bad message"
        return f"resampled message {index}"

    async def env_message(
        self, instance: WhoWhenInstance, prefix: list[LogStep], speaker: str
    ) -> str:
        return "env output"

    async def still_fails(self, instance: WhoWhenInstance, transcript: list[LogStep]) -> bool:
        return not any("GOOD" in s.content for s in transcript)


async def test_recovers_planted_locus() -> None:
    inst = _instance(n_steps=5, label_step=2)
    result = await attribute_log(inst, PlantedWorld(p=0.7, seed=3), k_max=24, chunk=8)

    assert result.factual_still_fails  # sanity floor: the unmodified log fails
    assert result.locus == 2  # the LATEST significantly-rescuing step, not an earlier re-roller
    assert result.predicted_agent == "Agent2"
    assert result.agent_correct
    assert result.step_within(0)
    # Steps after the pivot never rescue; the step before it rescues only via run-forward re-roll.
    assert not result.per_step[3].rescues and not result.per_step[4].rescues
    # Early stopping did its job somewhere (not every step burned the full K).
    assert any(se.k_realized < 24 for se in result.per_step)


async def test_no_locus_when_nothing_rescues() -> None:
    class HopelessWorld(PlantedWorld):
        async def still_fails(self, instance: WhoWhenInstance, transcript: list[LogStep]) -> bool:
            return True  # nothing ever helps

    result = await attribute_log(_instance(), HopelessWorld(), k_max=8, chunk=4)
    assert result.locus is None
    assert not result.prediction_confident  # fallback prediction, flagged as such
    assert result.predicted_step is not None  # benchmarks expect a prediction on every instance


def test_normalize_answer() -> None:
    from car.bench.surrogate import normalize_answer

    assert normalize_answer(" 8. ") == "8"
    assert normalize_answer("8.0") == "8"
    assert normalize_answer("1,234") == "1234"
    assert normalize_answer('"Paris"') == "paris"
    assert normalize_answer("The answer") == "the answer"
    assert normalize_answer("4") != normalize_answer("8")
