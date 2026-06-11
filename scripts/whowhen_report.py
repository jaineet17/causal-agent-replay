"""Analyze a Who&When results JSONL: metrics under several prediction rules, A/B'd offline.

The expensive artifact is the per-step rescue curve (p_fail + CI per step), stored per instance
by whowhen_run.py. The cheap part — turning a curve into a (agent, step) prediction — can be
re-derived offline under any rule, so rules are compared on the SAME rollouts:

  - ci_locus:    latest agent step whose rescue CI excludes 0 (no fallback; abstains otherwise)
  - argmax:      agent step with max rescue rate (early-biased under run-forward)
  - latest_tol:  latest agent step within tolerance of max rescue (late-biased when mistake at 0)
  - cliff:       largest consecutive DROP in rescue rate — the commitment cliff (the discrete
                 derivative of the rescue curve)

    uv run python scripts/whowhen_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
app = typer.Typer(add_completion=False)


def _agent_steps(row: dict) -> list[dict]:
    return [s for s in row.get("per_step", []) if not s["env"]]


def predict_ci_locus(row: dict) -> int | None:
    locus = None
    for s in _agent_steps(row):
        _lo, hi = s["ci"]
        if hi < 0.0 and (1.0 - s["p_fail"]) > 0:  # rescue CI excludes 0
            locus = s["i"]
    return locus


def predict_argmax(row: dict) -> int | None:
    steps = _agent_steps(row)
    return max(steps, key=lambda s: 1.0 - s["p_fail"])["i"] if steps else None


def predict_latest_tol(row: dict) -> int | None:
    steps = _agent_steps(row)
    if not steps:
        return None
    best = max(1.0 - s["p_fail"] for s in steps)
    tol = max(0.8 * best, best - 0.15)
    pick = None
    for s in steps:
        if (1.0 - s["p_fail"]) >= tol:
            pick = s["i"]
    return pick


def predict_cliff(row: dict) -> int | None:
    """The commitment cliff: the agent step with the largest drop to the NEXT step's rescue."""
    steps = sorted(row.get("per_step", []), key=lambda s: s["i"])
    agent_idx = [s["i"] for s in steps if not s["env"]]
    if not agent_idx:
        return None
    rescue = {s["i"]: 1.0 - s["p_fail"] for s in steps}
    indices = [s["i"] for s in steps]
    best_step, best_drop = None, -1.0
    for i in agent_idx:
        later = [rescue[j] for j in indices if j > i]
        nxt = later[0] if later else 0.0
        drop = rescue[i] - nxt
        if drop > best_drop:
            best_step, best_drop = i, drop
    return best_step


RULES = {
    "ci_locus": predict_ci_locus,
    "argmax": predict_argmax,
    "latest_tol": predict_latest_tol,
    "cliff": predict_cliff,
}


@app.command()
def main(
    results: Path = typer.Option(REPO_ROOT / "data" / "whowhen" / "results_ag.jsonl"),
) -> None:
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    sane = [r for r in ok if r.get("factual_still_fails")]
    typer.echo(
        f"{len(rows)} rows: {len(ok)} ok, {len(errors)} errors; "
        f"sanity floor (factual fails): {len(sane)}/{len(ok)}"
    )

    # agent->step map per row for agent accuracy under re-derived step predictions
    def agent_at(row: dict, step: int | None) -> str | None:
        if step is None:
            return None
        for s in row.get("per_step", []):
            if s["i"] == step:
                return str(s["agent"])
        return None

    header = f"{'rule':12s} {'n_pred':>6s} {'agent':>7s} {'step':>7s} {'±1':>7s} {'±3':>7s}"
    typer.echo(header)
    for name, rule in RULES.items():
        n_pred = agent_hit = exact = pm1 = pm3 = 0
        for r in ok:
            pred = rule(r)
            if pred is None:
                continue
            n_pred += 1
            agent_hit += agent_at(r, pred) == r["label_agent"]
            d = abs(pred - r["label_step"])
            exact += d == 0
            pm1 += d <= 1
            pm3 += d <= 3
        n = len(ok) or 1
        typer.echo(
            f"{name:12s} {n_pred:6d} {agent_hit / n:7.1%} {exact / n:7.1%} "
            f"{pm1 / n:7.1%} {pm3 / n:7.1%}"
        )
    typer.echo("(accuracies over ALL ok rows; abstentions count as misses)")


if __name__ == "__main__":
    app()
