# Phase 5 Research — Framework Adapters (LangGraph first)

**Date:** 2026-06-11
**Scope:** How to FAITHFULLY record agents people build with LangGraph/LangChain — without owning
their loop — and re-execute them counterfactually. Plus sizing for the OpenAI Agents SDK adapter.

**Method caveat:** WebFetch unavailable; versions from search snippets of PyPI/pypistats/GitHub.
Items marked **[verify]** were re-checked at dependency-install time (see pins actually resolved
in `uv.lock`).

---

## 1. The lay of the land (mid-2026)

- **LangChain 1.0 + LangGraph 1.0 shipped together (Oct 2025).** The blessed agent surface is now
  **`langchain.agents.create_agent`** — a prebuilt loop that runs on LangGraph and adds a
  **middleware** system. `langgraph.prebuilt.create_react_agent` still works but is **deprecated,
  removal slated for langgraph 2.0**.
- Current stable (2026-06-11): `langgraph` 1.2.4, `langchain` ~1.3.7, `langchain-core` 1.4.3,
  `langchain-openai` ~1.3.0 **[verify]**, `langchain-anthropic` 1.4.0.
- → **Target `create_agent` as the supported surface**; accept `create_react_agent` as legacy.
  Do not build against pre-1.0 (`AgentExecutor`-era) APIs.

## 2. Interception surfaces (ranked)

1. **Middleware — the winner.** `AgentMiddleware.wrap_model_call(request, handler)` receives a
   `ModelRequest` carrying exactly what CAR's `state_before` needs: `messages`, the per-request
   `system_message` (which NEVER appears in graph state — create_agent injects it at request
   time), `tools`, the `model` object, and `model_settings`. `wrap_tool_call` receives the
   `ToolCall` dict (name/args/**id**) and returns the `ToolMessage`. Multiple middleware compose
   (first = outermost), so a CAR recorder placed outermost sees what inner middleware did.
   Retries/short-circuits appear as multiple handler invocations — record per-invocation.
2. **Callbacks** (`on_chat_model_start` + `invocation_params`): the fallback for legacy
   `create_react_agent`. Known lossiness: in 1.x, `invocation_params["tools"]` can contain
   serialized `StructuredTool` reprs rather than clean schemas (langfuse #11850).
3. **Wrapping the chat model**: gives wire payloads only via private partner-package APIs
   (`_get_request_payload` / `_format_messages`); an httpx event-hook tap is the byte-exact
   *verification* option, not the primary surface.
4. **Checkpointer mining**: graph state only — no system message, no tool schemas, no sampling
   params. Cross-check, not capture.

### Provider-native projection
- **OpenAI direction is public and id-preserving:** `convert_to_openai_messages` (LC messages →
  OpenAI dicts; `tool_call_id`s survive both ways) and `convert_to_openai_tool` (tool → schema).
- **Anthropic has no public equivalent** (private `_format_messages` hoists system + merges
  same-role messages). → CAR stores adapter trajectories in **OpenAI-projected message format**
  regardless of the underlying model vendor; the LangChain model object handles its own
  vendor conversion when we re-invoke it.

## 3. Mapping to the linear SCM

- create_agent's loop is exactly CAR's SCM: model → (tool_calls? tools : END) → model → …
- **Parallel tool calls are default-on** and `create_agent` has **no switch to disable them**
  (open request: langchain #34010); `ToolNode` runs them concurrently. CAR v1 keeps one action =
  one tool call (consistent with the native recorder's documented limitation): the adapter
  **refuses parallel sets with an actionable error** and ships a `single_tool_call` middleware
  override (`model_settings={"parallel_tool_calls": False}`) for OpenAI-compatible models.
- Out of scope, detect-and-refuse: Send-API fan-out, multi-agent supervisor/swarm graphs,
  subgraphs, middleware `jump_to` short-circuits, tools returning `Command`.

## 4. Counterfactual re-execution

- **Mode A — policy replay (implemented):** π(a|context) =
  `model.bind_tools(recorded_tool_schemas).ainvoke([SystemMessage(recorded)] + convert_to_messages(state.messages))`.
  Public APIs end to end; `bind_tools` accepts OpenAI-format dict schemas; LC models accept
  OpenAI-format message dicts via `convert_to_messages`.
- **Tools individually:** invoke with the FULL ToolCall dict
  `{"name", "args", "id", "type": "tool_call"}` — required for `InjectedToolCallId` tools since
  1.0.6, and conveniently returns a ready `ToolMessage` with the right `tool_call_id`. Tools
  needing `ToolRuntime`/`InjectedState` are classified non-replayable-live (stub or refuse).
- **Mode B — in-graph replay** (checkpointer + `update_state` time travel): maximally faithful to
  middleware effects; deferred.
- Invariant to enforce before any re-issued call: every `tool_calls[i].id` has exactly one
  downstream tool message before the next assistant turn (providers 400 otherwise).

## 5. OpenAI Agents SDK (sizing)

`openai-agents` ~0.17.x (0.x churn). Hooks now include `on_llm_start`/`on_llm_end` +
`on_tool_start/end`; tracing processors exist but gate payloads
(`trace_include_sensitive_data`) — capture via a wrapping `ModelProvider` + `RunHooks`, not
tracing. Responses-API item types (reasoning items) need schema thought. **~half the LangGraph
effort.** Deferred to Phase 5b.

## Decisions

1. **Capture = a CAR `AgentMiddleware`** (`wrap_model_call` + `wrap_tool_call`), exposed as
   `car.adapters.langgraph`. Callbacks fallback deferred until someone asks.
2. **Storage format = OpenAI-projected messages** (public, id-preserving) with
   `provider="langchain"`; the existing OpenAI codec semantics apply for reconstruction/forging.
3. **Replay = Mode A** via `LangChainPolicy` (wraps the user's `BaseChatModel`) and
   `LangChainToolEnvironment` (tools invoked by ToolCall dict, `source="real"`).
4. **Parallel tool calls: refuse + provide the disable override.** Honest v1 scope.
5. **Faithfulness proven the same way as Phase 0:** record with a scripted fake chat model
   (offline, no keys), assert `verify_reconstruction` passes, then run interventions/attribution
   on the recorded trajectory through the unchanged core.

### Dependency pins (`car[langgraph]` extra)
`langchain>=1.3,<2.0`, `langgraph>=1.2,<2.0`, `langchain-core>=1.4,<2.0`
(providers: `langchain-openai`/`langchain-anthropic` only needed by the user's own model choice).

### Top faithfulness risks → tests
1. **Projection lossiness** → reconstruction round-trip digest test on an adapter-recorded run
   (same invariant as Phase 0), incl. a tool-rich multi-step fixture.
2. **tool_call_id pairing breaks under intervention** → invariant checker + adversarial tests
   (forged action gets a synthetic id; the codec pairs the ToolMessage to it).
3. **Parallel tool calls** → explicit test that a 2-call AIMessage is refused with the
   actionable error, and that the disable-override middleware exists.
