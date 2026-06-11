"""Self-contained interactive HTML report for an attribution (PLAN.md s5.6).

Renders, in one file with no external dependencies (inline CSS + vanilla JS + SVG):
  - the **trajectory trunk**: the observed run, each step showing the outcome distribution when
    that step is resampled (the counterfactual branch), so the "point of commitment" is visible —
    resampling at/before the locus often rescues the run; resampling after it never does;
  - the **attribution panel**: per-step causal effect (rescue magnitude) with confidence
    intervals, the causal locus highlighted; the Shapley decomposition if provided;
  - **click a step** -> its context (``state_before``), the action taken, and the resample
    distribution.

The goal (DoD): a viewer immediately sees *which single step* caused the bad outcome and *why*,
with the counterfactual evidence on screen.
"""

from __future__ import annotations

import html
import json
from typing import Any

from car.attribute.contrastive import ContrastiveResult
from car.attribute.shapley import ShapleyResult
from car.schemas.trajectory import Trajectory


def _action_summary(
    kind: str, tool_name: str | None, tool_args: dict[str, Any] | None, text: str | None
) -> str:
    if kind == "final":
        return f"FINAL: {(text or '').strip()[:120]}"
    args = ", ".join(f"{k}={v!r}" for k, v in (tool_args or {}).items())
    return f"{tool_name}({args})"


def _render_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(b.get("text") or b.get("content") or "") for b in content if isinstance(b, dict)
            )
        out.append({"role": str(m.get("role", "?")), "content": str(content)[:600]})
    return out


def build_payload(
    factual: Trajectory,
    contrastive: ContrastiveResult,
    *,
    shapley: ShapleyResult | None = None,
    good_label: str = "ok",
) -> dict[str, Any]:
    """Assemble the JSON the HTML renders from."""
    shap_by_index = {s.index: s for s in shapley.per_step} if shapley else {}
    steps: list[dict[str, Any]] = []
    for sa in contrastive.per_step:
        step = factual.steps[sa.index]
        a = step.action
        shap = shap_by_index.get(sa.index)
        steps.append(
            {
                "index": sa.index,
                "action": _action_summary(a.kind, a.tool_name, a.tool_args, a.text),
                "kind": a.kind,
                "is_locus": sa.index == contrastive.causal_locus,
                # contrastive effect is P(bad|resample)-P(bad|factual) <= 0; rescue = how much it
                # reduces badness.
                "rescue": -sa.effect.point,
                "rescue_low": -sa.effect.ci_high,
                "rescue_high": -sa.effect.ci_low,
                "significant": sa.effect.is_significant,
                "p_bad_after": sa.p_bad_after_resample,
                "k_samples": contrastive.k_samples,
                "shapley": (
                    None
                    if shap is None
                    else {"value": shap.value, "low": shap.ci_low, "high": shap.ci_high}
                ),
                "messages": _render_messages(step.state_before.messages),
                "observation": (step.observation.result if step.observation else None),
            }
        )
    return {
        "trajectory_id": factual.trajectory_id,
        "bad_label": contrastive.bad_label,
        "good_label": good_label,
        "observed_label": contrastive.observed_label,
        "causal_locus": contrastive.causal_locus,
        "final_output": factual.final_output,
        "confidence": contrastive.confidence,
        "has_shapley": shapley is not None,
        "steps": steps,
    }


def render_html(
    factual: Trajectory,
    contrastive: ContrastiveResult,
    *,
    shapley: ShapleyResult | None = None,
    good_label: str = "ok",
    title: str = "Causal Agent Replay — attribution",
) -> str:
    """Return a complete, self-contained interactive HTML document."""
    payload = build_payload(factual, contrastive, shapley=shapley, good_label=good_label)
    data_json = json.dumps(payload)
    return _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data_json)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root{
    --bg:#0e1116; --panel:#171b22; --line:#2a3038; --txt:#e6edf3; --muted:#8b949e;
    --bad:#f85149; --good:#3fb950; --locus:#d29922; --accent:#58a6ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  header{padding:22px 28px;border-bottom:1px solid var(--line)}
  h1{margin:0 0 4px;font-size:19px}
  .sub{color:var(--muted);font-size:13px}
  .verdict{margin-top:12px;font-size:15px}
  .pill{display:inline-block;padding:2px 9px;border-radius:999px;font-weight:600;font-size:12px}
  .pill.bad{background:rgba(248,81,73,.16);color:var(--bad)}
  .pill.locus{background:rgba(210,153,34,.18);color:var(--locus)}
  main{display:grid;grid-template-columns:minmax(380px,1fr) minmax(360px,1fr);gap:0}
  @media(max-width:900px){main{grid-template-columns:1fr}}
  .col{padding:20px 28px}
  .col+.col{border-left:1px solid var(--line)}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 14px}
  .step{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px;
        background:var(--panel);cursor:pointer;transition:border-color .15s}
  .step:hover{border-color:var(--accent)}
  .step.locus{border-color:var(--locus);box-shadow:0 0 0 1px var(--locus) inset}
  .step.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
  .step .top{display:flex;align-items:center;gap:8px}
  .idx{width:22px;height:22px;border-radius:6px;background:#21262d;display:grid;place-items:center;
       font-size:12px;color:var(--muted);flex:0 0 auto}
  .act{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;overflow:hidden;
       text-overflow:ellipsis;white-space:nowrap}
  .bar{height:8px;border-radius:5px;margin-top:9px;display:flex;overflow:hidden;background:#21262d}
  .bar .g{background:var(--good)} .bar .b{background:var(--bad)}
  .barlabel{font-size:11px;color:var(--muted);margin-top:5px;display:flex;justify-content:space-between}
  svg{width:100%;height:auto;overflow:visible}
  .axis{stroke:var(--line);stroke-width:1}
  .tick{fill:var(--muted);font-size:10px}
  .barrect{fill:var(--accent)} .barrect.locus{fill:var(--locus)} .barrect.ns{fill:#39414d}
  .whisk{stroke:var(--txt);stroke-width:1.4;opacity:.85}
  .blabel{fill:var(--txt);font-size:11px} .bsteplabel{fill:var(--muted);font-size:10px}
  .detail{margin-top:6px}
  .detail .meta{color:var(--muted);font-size:12px;margin-bottom:8px}
  .msg{border-left:2px solid var(--line);padding:4px 0 4px 10px;margin:6px 0}
  .msg .role{color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  .msg .content{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;
                color:#c9d1d9}
  .note{color:var(--muted);font-size:12px;margin-top:16px;border-top:1px dashed var(--line);padding-top:12px}
  .toggle{float:right;font-size:12px}
  .toggle button{background:#21262d;border:1px solid var(--line);color:var(--txt);border-radius:6px;
                 padding:3px 9px;cursor:pointer;font-size:12px}
  .toggle button.on{background:var(--accent);border-color:var(--accent);color:#0d1117}
</style>
</head>
<body>
<header>
  <h1>Causal Agent Replay</h1>
  <div class="sub">Which step actually caused the outcome — proven by counterfactual.</div>
  <div class="verdict" id="verdict"></div>
</header>
<main>
  <div class="col">
    <h2>Trajectory &middot; resample distribution per step</h2>
    <div id="trunk"></div>
    <div class="note" id="trunknote"></div>
  </div>
  <div class="col">
    <h2>Attribution
      <span class="toggle" id="toggle"></span>
    </h2>
    <div id="chart"></div>
    <div id="detail" class="detail"></div>
  </div>
</main>
<script>
const DATA = __DATA__;
let mode = "contrastive";
let selected = DATA.causal_locus;

function pct(x){ return Math.round(x*100); }

function renderVerdict(){
  const v = document.getElementById("verdict");
  if(DATA.causal_locus === null){
    v.innerHTML = `Observed outcome: <span class="pill bad">${DATA.observed_label}</span>. `
      + `No single step's resampling significantly changed it.`;
    return;
  }
  const s = DATA.steps[DATA.causal_locus];
  v.innerHTML = `Observed outcome: <span class="pill bad">${DATA.observed_label}</span>. `
    + `Causal locus: <span class="pill locus">step ${DATA.causal_locus}</span> `
    + `&mdash; <code>${escapeHtml(s.action)}</code>. `
    + `Resampling it avoids the bad outcome ${pct(1 - s.p_bad_after)}% of the time; `
    + `resampling later steps never does.`;
}

function renderTrunk(){
  const t = document.getElementById("trunk");
  t.innerHTML = "";
  DATA.steps.forEach(s => {
    const div = document.createElement("div");
    div.className = "step" + (s.is_locus?" locus":"") + (s.index===selected?" sel":"");
    const good = 1 - s.p_bad_after;
    div.innerHTML =
      `<div class="top"><div class="idx">${s.index}</div>`
      + `<div class="act">${escapeHtml(s.action)}</div></div>`
      + `<div class="bar"><div class="g" style="width:${pct(good)}%"></div>`
      + `<div class="b" style="width:${pct(s.p_bad_after)}%"></div></div>`
      + `<div class="barlabel"><span>resample here &rarr; ${pct(good)}% good</span>`
      + `<span>${pct(s.p_bad_after)}% bad</span></div>`;
    div.onclick = () => { selected = s.index; renderAll(); };
    t.appendChild(div);
  });
  document.getElementById("trunknote").textContent =
    `Each bar: the outcome distribution over ${DATA.steps[0].k_samples} counterfactual rollouts `
    + `when that step (and everything after it) is re-decided by the same policy. The locus is the `
    + `last step whose green share is large — the point of commitment.`;
}

function renderChart(){
  const c = document.getElementById("chart");
  const steps = DATA.steps;
  const W = 360, rowH = 34, padL = 40, padR = 40, H = steps.length*rowH + 24;
  const vals = steps.map(s => mode==="contrastive"
      ? {v:s.rescue, lo:s.rescue_low, hi:s.rescue_high, sig:s.significant}
      : {v:(s.shapley?s.shapley.value:0), lo:(s.shapley?s.shapley.low:0),
         hi:(s.shapley?s.shapley.high:0), sig:(s.shapley? (s.shapley.low>0||s.shapley.high<0):false)});
  const maxV = Math.max(0.001, ...vals.map(d=>Math.max(Math.abs(d.v),Math.abs(d.hi),Math.abs(d.lo))));
  const x = v => padL + (v/maxV)*(W-padL-padR);
  let svg = `<svg viewBox="0 0 ${W} ${H}">`;
  svg += `<line class="axis" x1="${padL}" y1="8" x2="${padL}" y2="${H-16}"/>`;
  steps.forEach((s,i)=>{
    const d = vals[i]; const y = 16 + i*rowH;
    const cls = s.is_locus ? "barrect locus" : (d.sig ? "barrect" : "barrect ns");
    const x0 = x(0), x1 = x(d.v);
    svg += `<rect class="${cls}" x="${Math.min(x0,x1)}" y="${y}" width="${Math.abs(x1-x0)}" height="14" rx="3"/>`;
    svg += `<line class="whisk" x1="${x(d.lo)}" y1="${y+7}" x2="${x(d.hi)}" y2="${y+7}"/>`;
    svg += `<line class="whisk" x1="${x(d.lo)}" y1="${y+3}" x2="${x(d.lo)}" y2="${y+11}"/>`;
    svg += `<line class="whisk" x1="${x(d.hi)}" y1="${y+3}" x2="${x(d.hi)}" y2="${y+11}"/>`;
    svg += `<text class="bsteplabel" x="4" y="${y+11}">step ${s.index}</text>`;
    const rightmost = Math.max(x(0), x(d.v), x(d.lo), x(d.hi));
    svg += `<text class="blabel" x="${rightmost+8}" y="${y+11}" text-anchor="start">`
         + `${d.v.toFixed(2)}</text>`;
  });
  svg += `</svg>`;
  const cap = mode==="contrastive"
    ? `Rescue effect: how much resampling each step reduces P(${DATA.bad_label}), with `
      + `${pct(DATA.confidence)}% CIs. Gold = causal locus; grey = CI includes 0 (not significant).`
    : `Shapley value: each step's share of credit for the bad outcome (CIs shown). Interacting `
      + `steps split credit; the sum equals the total effect (efficiency).`;
  c.innerHTML = svg + `<div class="note">${cap}</div>`;
}

function renderToggle(){
  const t = document.getElementById("toggle");
  if(!DATA.has_shapley){ t.innerHTML=""; return; }
  t.innerHTML = `<button id="bc" class="${mode==='contrastive'?'on':''}">contrastive</button> `
              + `<button id="bs" class="${mode==='shapley'?'on':''}">shapley</button>`;
  document.getElementById("bc").onclick=()=>{mode="contrastive";renderAll();};
  document.getElementById("bs").onclick=()=>{mode="shapley";renderAll();};
}

function renderDetail(){
  const d = document.getElementById("detail");
  if(selected===null){ d.innerHTML=""; return; }
  const s = DATA.steps[selected];
  let h = `<h2 style="margin-top:22px">Step ${s.index} detail`
        + (s.is_locus?` &middot; <span class="pill locus">causal locus</span>`:``) + `</h2>`;
  h += `<div class="meta">action: <code>${escapeHtml(s.action)}</code>`;
  if(s.observation!==null) h += ` &middot; observed: <code>${escapeHtml(String(s.observation).slice(0,80))}</code>`;
  h += `<br/>resampling this step &rarr; ${pct(1-s.p_bad_after)}% good / ${pct(s.p_bad_after)}% bad `
     + `over ${s.k_samples} rollouts.</div>`;
  h += `<div class="meta">context it decided from:</div>`;
  s.messages.forEach(m=>{
    h += `<div class="msg"><div class="role">${escapeHtml(m.role)}</div>`
       + `<div class="content">${escapeHtml(m.content)}</div></div>`;
  });
  d.innerHTML = h;
}

function renderAll(){ renderVerdict(); renderTrunk(); renderToggle(); renderChart(); renderDetail(); }
function escapeHtml(s){ return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
renderAll();
</script>
</body>
</html>
"""
