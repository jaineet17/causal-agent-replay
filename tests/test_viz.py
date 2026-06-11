"""The HTML report renders, is self-contained, and embeds valid attribution data."""

from __future__ import annotations

import json
import re
from typing import Any

from car.attribute.contrastive import contrastive_attribution
from car.attribute.shapley import shapley_attribution
from car.record.recorder import SyntheticCodec, record_run
from car.schemas.trajectory import Trajectory
from car.viz.html import render_html

from .scm_fixtures import ScriptedPolicy
from .synthetic_scms import ENV, PIVOTAL_OBSERVED, PIVOTAL_STEP, pivotal_policy, refund_outcome

CODEC = SyntheticCodec()


async def _attribution() -> tuple[Trajectory, Any, Any]:
    factual = await record_run(
        trajectory_id="viz",
        policy=ScriptedPolicy(PIVOTAL_OBSERVED),
        environment=ENV,
        codec=CODEC,
        system_prompt="sys",
        tool_schemas=[],
        user_input="go",
    )
    contrastive = await contrastive_attribution(
        factual,
        policy=pivotal_policy(0.3, seed=1),
        environment=ENV,
        codec=CODEC,
        outcome_fn=refund_outcome(),
        bad_label="inappropriate_refund",
        k_samples=40,
    )
    shapley = await shapley_attribution(
        factual,
        policy=pivotal_policy(0.3, seed=1),
        environment=ENV,
        codec=CODEC,
        outcome_fn=refund_outcome(),
        bad_label="inappropriate_refund",
        n_permutations=20,
        samples_per_eval=6,
        seed=1,
    )
    return factual, contrastive, shapley


async def test_render_html_is_self_contained_and_embeds_valid_data() -> None:
    factual, contrastive, shapley = await _attribution()
    html = render_html(factual, contrastive, shapley=shapley)

    # Self-contained: a full document with no external script/style references.
    assert html.startswith("<!DOCTYPE html>")
    assert "<script src=" not in html
    assert "https://" not in html.split("</head>")[0]  # no CDN in <head>

    # The embedded data parses and reflects the attribution.
    match = re.search(r"const DATA = (\{.*?\});", html)
    assert match is not None
    data = json.loads(match.group(1))
    assert data["causal_locus"] == PIVOTAL_STEP
    assert len(data["steps"]) == len(factual.steps)
    assert data["has_shapley"] is True
    assert data["steps"][PIVOTAL_STEP]["is_locus"] is True
    # The locus action and the bad label are surfaced for the viewer.
    assert "issue_refund" in data["steps"][PIVOTAL_STEP]["action"]
    assert data["bad_label"] == "inappropriate_refund"


async def test_render_html_without_shapley() -> None:
    factual, contrastive, _ = await _attribution()
    html = render_html(factual, contrastive)
    data = json.loads(re.search(r"const DATA = (\{.*?\});", html).group(1))  # type: ignore[union-attr]
    assert data["has_shapley"] is False
    assert all(s["shapley"] is None for s in data["steps"])
