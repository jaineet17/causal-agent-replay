# Phase 0 Research — Foundations (record-replay, provider determinism, deps)

**Date:** 2026-05-29
**Scope:** What must be true before any code: how deterministic replay can even work given
current provider behavior, the state-reconstruction design pattern from record-replay
debuggers, and pinned dependency versions.

**Method caveat (honesty):** `WebFetch` was unavailable in this session, so primary-page
content was read via search-engine summaries rather than direct fetches. Version numbers and
the Opus-4.7/4.8 400 behavior were consistent across multiple results but warrant a ~30s
direct-page confirmation before hard-pinning. Items flagged **[verify]** are the ones to
re-check.

---

## 1. Provider determinism — the constraint that defines the DoD

### Anthropic Messages API
- **No `seed` parameter.** No documented deterministic-sampling mechanism.
- **temperature=0 is not a determinism guarantee.** Anthropic documents that identical inputs
  may produce different outputs across calls even at temperature 0 (applies to first-party and
  Bedrock/Vertex). temperature only *reduces* randomness.
- **MAJOR GOTCHA — sampling params removed on newest models.** `temperature`, `top_p`, `top_k`
  are **not supported on Claude Opus 4.7+ (incl. Opus 4.8)** and return HTTP **400
  `invalid_request_error`**. The Python SDK request types still *define* these fields (so code
  type-checks) but the **server rejects** them. Steer behavior via prompting instead. **[verify]**
- **Documented causes of residual nondeterminism:** floating-point non-associativity; GPU /
  model-parallel sharding; and **batch-size variance** — the same prompt lands in different
  server-side batch sizes under varying load, and standard matmul/attention kernels are not
  batch-invariant, so the numeric path (and the argmax token) can differ.
  (Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference," Sept 2025;
  corroborated by LMSYS/SGLang deterministic-inference blog, 2025-09-22.)

### OpenAI Chat Completions / Responses API
- **`seed` exists in Chat Completions but is best-effort, not guaranteed.** Repeated requests
  with the same seed + params *should* match but "determinism isn't guaranteed"; variability is
  "not uncommon" even with matching `system_fingerprint`.
- **`system_fingerprint`** identifies the backend config (weights + infra + numerics); it
  changes when OpenAI updates numerics. Compare it across runs to detect determinism-affecting
  backend changes.
- **`seed` reportedly deprecated for Chat Completions and absent in the Responses API**
  (community-reported, **not** vendor-confirmed). Responses API is OpenAI's strategic surface and
  has **no seed** → effectively at parity with Anthropic there. **[verify]**
- Image inputs break seed reproducibility even on otherwise-supported models.

### Bottom line → DoD framing
Neither provider guarantees deterministic output even at its most-deterministic settings; both
explicitly disclaim it. Bit-exact replay is achievable only with a controlled inference stack
(batch-invariant kernels, demonstrated by Thinking Machines on vLLM), which hosted APIs do not
expose. Therefore CAR's faithful-replay floor is **provider-bounded**, and the honest Phase 0
DoD is **measure-and-report**, not guarantee:

> Re-issue the recorded call N times; report an **action-match rate** and a residual-
> nondeterminism metric; attribute residual divergence to the documented causes above.

Crucially, **action-match rate ≫ token-identity rate**: residual FP noise rarely flips a
confident argmax over tool selection + structured args, so agent *actions* replay far more
stably than raw tokens. The DoD measures action-match, which is the level attribution operates at.

---

## 2. Record-replay state reconstruction (rr, Pernosco)

- **rr (Mozilla):** record only the **nondeterministic inputs** to a process (syscall results,
  signals, `rdtsc`), then **re-execute the unchanged binary deterministically between them**,
  injecting recorded inputs at the same points. Deterministic computation in between is re-run,
  not logged — which keeps overhead low. (rr-project.org; ACM Queue "To Catch a Failure," 2020.)
- **Pernosco:** consumes an rr recording and, **offline and in parallel**, replays under heavy
  instrumentation to record every memory write into a database, enabling reconstruction of full
  state at any point and dataflow back to causal origins. (pernos.co/about/overview.)

### → Design principles adopted by CAR
1. **The recorder captures every nondeterministic input to the agent**: the exact model request
   (messages, tool schemas, params, model ID, seed if any), the **recorded model response**
   (verbatim), tool outputs, and any clock/RNG/env reads. The **LLM call is CAR's `rdtsc`/
   syscall** — the irreducible nondeterministic input we cannot perfectly reproduce, so we record
   its output as ground truth and **measure divergence** when we re-issue it.
2. **Pernosco split:** faithful replay is the Phase 0 critical path; the heavy causal-attribution
   / state-indexing is a **later offline pass** over a deterministic-enough replay, not part of
   Phase 0.

---

## 3. Pinned dependency versions (PyPI, 2026-05-29)

| Library | Latest stable | Notes |
|---|---|---|
| pydantic | 2.13 | v2 API; floor `>=2.7` is safe. |
| numpy | 2.4.6 | NumPy 2.x. |
| scipy | 1.17.1 | needs NumPy 2.x. |
| anthropic | 0.105.2 | pre-1.0; minor bumps can shift types — pin in CI. |
| openai | 2.38.0 | v2.x. |
| typer | 0.26.3 | — |
| structlog | 25.5.0 | **gotcha:** 25.5.0 made a `styles` import non-backwards-compatible (#769). |
| pytest | 9.0.3 | v9 line. |
| pytest-asyncio | 1.4.0 | v1.x changed defaults — set `asyncio_mode` explicitly (we set `"auto"`). |
| hypothesis | 6.151.11 | — |
| ruff | 0.15.15 | 0.x; rules churn between minors — pin exact in CI. |
| mypy | reported 2.1.0 | **[verify]** — mypy historically 1.x; floor `>=1.10` is safe regardless. |

### Provider tool-call/result shapes (for faithful message reconstruction)
- **Anthropic:** assistant `tool_use` block `{"type":"tool_use","id":"toolu_…","name":…,"input":{…}}`;
  result returned as a `tool_result` block in a **user** message keyed by **`tool_use_id`**.
  Linkage: `tool_use_id ↔ tool_use.id`.
- **OpenAI (Chat Completions):** assistant `tool_calls`:
  `[{"id":"call_…","type":"function","function":{"name":…,"arguments":"<JSON string>"}}]` —
  **`arguments` is a JSON-encoded string, preserve verbatim**; result sent as a message with
  `role:"tool"` and **`tool_call_id`** matching the call id. Linkage: `tool_call_id ↔ tool_calls[].id`.
- The two providers differ structurally in encoding AND id-linkage field. → **Recorder stores
  provider-native message objects verbatim** (no lossy normalization); replay reconstructs
  byte-faithful history per provider.

---

## 4. Decisions that flow into the Phase 0 implementation

1. **DoD = measure & report, not guarantee.** `deterministic.py` re-issues the recorded call N
   times and reports action-match rate + residual nondeterminism. `test_deterministic_replay.py`
   proves this against a **synthetic deterministic policy** (where match rate is exactly 1.0, so
   the replay *machinery* is validated independently of provider noise), and separately measures
   the real-provider rate as a reported metric.
2. **Omit sampling params for Opus 4.7+.** `State.sampling` must not carry `temperature`/`top_p`/
   `top_k` for those models; the replayer strips them. Record the exact model ID so the
   determinism contract that applied is known. Don't architect around seed availability (Anthropic
   has none; OpenAI Responses has none).
3. **Record verbatim provider-native messages + response.** `Action.raw` and `State.messages`
   hold the unmodified provider objects. For OpenAI, capture `seed` and `system_fingerprint`;
   flag fingerprint mismatch on replay as a known determinism-breaking event.
4. **Pernosco split honored by the phase plan** — faithful replay now; attribution is a later
   offline pass.

### Key sources
thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ · lmsys.org/blog/2025-09-22-sglang-deterministic/ ·
docs.anthropic.com (messages, tool-use, migration, errors) · cookbook.openai.com (reproducible_outputs_with_the_seed_parameter) ·
developers.openai.com (function-calling) · rr-project.org · pernos.co/about/overview · queue.acm.org/detail.cfm?id=3391621 ·
pypi.org/project/{pydantic,numpy,scipy,anthropic,openai,typer,structlog,pytest,pytest-asyncio,hypothesis,ruff,mypy}.
All accessed 2026-05-29.

---

## 5. Free / local models (added 2026-05-29) — Ollama + OpenAI-compatible backends

Motivation: run + record agents at **zero cost**. Because the recorder is written against the
`Policy` protocol, a free backend is just another policy.

- **Ollama exposes an OpenAI-compatible endpoint** at `http://localhost:11434/v1` (mirrors
  `/v1/chat/completions`); any code written for the `openai` SDK points at it with a one-line
  `base_url` change. It supports **tool calling**, structured outputs, and streaming — *but
  tool-call reliability is model-dependent*, so test with your specific model.
  (docs.ollama.com/api/openai-compatibility; ollama.com/blog/{openai-compatibility,tool-support}.)
- **Caveat (from a stalled research pass, partially verified):** the `/v1` endpoint reportedly
  has tool-calling reliability issues *especially with streaming*; the **native `/api/chat`**
  endpoint is considered more reliable. We use **non-streaming** calls, which avoids the worst of
  it. If `/v1` tool calling proves flaky in practice, a native-`/api/chat` policy is the
  documented fallback (deferred — its message shapes were not verified this pass).
- **Determinism (a real upside for CAR):** Ollama supports `seed` and `temperature=0`. With a
  fixed seed + `temperature=0` + **fixed `num_ctx`**, local output is reproducible — *much* more
  so than hosted APIs (single local stream, batch size 1). The docs note slight residual
  nondeterminism can still appear (e.g. Llama 3 between runs without a kernel restart), which
  *confirms* the measure-and-report framing applies locally too — it's just a far higher floor.
  Set `num_ctx` via the policy's `extra_body` passthrough: `sampling={"seed":0,"temperature":0,
  "extra_body":{"options":{"num_ctx":4096}}}`.
  (ollama/ollama issues #586, #4660, #5321; tspi.at/2025/08/10/ollamaparams.html.)
- **Tool-capable small models** (good on a 16–32GB Mac): `llama3.1:8b`, `qwen2.5:7b`. Use a
  recent Ollama (the installed 0.5.12 is early-2025; upgrade for best tool support).
- **Free hosted OpenAI-compatible alternatives** (no local GPU): Groq free tier
  (`https://api.groq.com/openai/v1`, supports tools), OpenRouter `:free` models. Same
  `OpenAICompatiblePolicy`, different `base_url`/key.

### → DECISION / IMPLICATION for CAR
Implemented `OpenAICompatiblePolicy` + `OpenAICodec` (one impl covers Ollama, Groq, OpenRouter,
vLLM, LM Studio). Validated end-to-end against a fake OpenAI-shaped client (no server/key/money):
records tool-call→final faithfully, replays exactly under a deterministic backend. The local
seeded path is the recommended **free** way to demo faithful replay; the hosted-provider
nondeterminism story (s1) remains the headline scientific point.

### Section 5 sources
docs.ollama.com/api/openai-compatibility · ollama.com/blog/openai-compatibility · ollama.com/blog/tool-support ·
github.com/ollama/ollama/issues/{586,4660,5321} · tspi.at/2025/08/10/ollamaparams.html. All accessed 2026-05-29.
