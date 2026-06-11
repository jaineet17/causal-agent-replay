"""LangGraph/LangChain adapter: record ``create_agent`` runs, replay them counterfactually.

Three pieces (design + faithfulness analysis in RESEARCH/phase_5_adapters.md):

  - ``LangGraphRecorder`` — an ``AgentMiddleware`` the user adds to their ``create_agent(...)``
    call. ``wrap_model_call`` is the only surface that exposes the complete logical request
    (messages, the per-request system message — which never appears in graph state — tools,
    model, settings); ``wrap_tool_call`` gives exact per-call args/ids/results. The recorder
    snapshots ``state_before`` at every model call and pairs observations by ``tool_call_id``.

  - ``LangChainPolicy`` — pi(a | context) for replay/interventions: re-invokes the user's
    ``BaseChatModel`` with the reconstructed context (public APIs only:
    ``convert_to_messages`` + ``bind_tools`` accepting OpenAI-format schemas).

  - ``LangChainToolEnvironment`` — live observations on counterfactual branches by invoking the
    agent's actual tools with full ToolCall dicts (returns provenance ``source="real"``).

Messages are stored in the OpenAI-projected wire format via the public, id-preserving
``convert_to_openai_messages``, so the core ``OpenAICodec`` reconstruction semantics apply
unchanged and ``DeterministicReplay.verify_reconstruction`` holds the adapter to the same
faithfulness invariant as the native recorder.

Honest v1 scope (detected and refused, never silently mis-recorded):
  - one tool call per assistant turn (parallel sets raise; see ``disable_parallel_tool_calls``);
  - ``create_agent`` / ``create_react_agent``-shaped loops (model -> tools -> model -> ... ->
    final), single initial user message; not Send-API fan-out / multi-agent graphs;
  - tools that return strings and don't require graph-injected state/runtime;
  - tool handlers returning ``Command`` are unsupported.

Usage::

    from car.adapters.langgraph import LangGraphRecorder

    recorder = LangGraphRecorder()
    agent = create_agent(model, tools, system_prompt=..., middleware=[recorder])
    await agent.ainvoke({"messages": [HumanMessage("...")]})
    trajectory = recorder.trajectory("my-run")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from car.schemas.scm import ReplayError
from car.schemas.trajectory import Action, Observation, Provider, State, Step, Trajectory

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain.agents.middleware.types import (
        ModelCallResult,
        ModelRequest,
        ToolCallRequest,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
    from langchain_core.messages.utils import convert_to_messages, convert_to_openai_messages
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the LangGraph adapter needs the optional extra: "
        "pip install 'causal-agent-replay[langgraph]'"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

log = structlog.get_logger(__name__)

_PARALLEL_REFUSAL = (
    "the model emitted {n} parallel tool calls in one turn; CAR v1 models one action per step "
    "(PLAN.md s5.1 limitation). Add car.adapters.langgraph.disable_parallel_tool_calls() to your "
    "middleware (OpenAI-compatible models), or prompt the agent to act sequentially."
)


def _text_of(content: Any) -> str:
    """Flatten LangChain message content (str or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _project_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """LangChain messages -> OpenAI-format dicts (public, tool_call_id-preserving)."""
    projected = convert_to_openai_messages(list(messages))
    return projected if isinstance(projected, list) else [projected]


def _action_from_ai(ai: AIMessage) -> Action:
    """Parse an AIMessage into a CAR Action; refuse parallel tool calls (v1 scope)."""
    raw = _project_messages([ai])[0]
    tool_calls = ai.tool_calls or []
    if len(tool_calls) > 1:
        raise ReplayError(_PARALLEL_REFUSAL.format(n=len(tool_calls)))
    text = _text_of(ai.content) or None
    if tool_calls:
        call = tool_calls[0]
        return Action(
            kind="tool_call",
            text=text,
            tool_name=call["name"],
            tool_args=dict(call["args"]),
            raw=raw,
        )
    return Action(kind="final", text=text or "", raw=raw)


def _model_name(model: BaseChatModel) -> str:
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


class LangGraphRecorder(AgentMiddleware):
    """Record a ``create_agent`` run faithfully into a CAR ``Trajectory``.

    One recorder instance records one run; call :meth:`trajectory` to consume it (or
    :meth:`reset` to discard). Both sync and async middleware hooks are implemented, so
    ``agent.invoke`` and ``agent.ainvoke`` both record.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending: list[tuple[dict[str, Any], Action]] = []  # (state kwargs, action)
        self._observations: dict[str, Observation] = {}  # tool_call_id -> observation

    # -- model boundary ------------------------------------------------------------------------
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        state_kwargs = self._snapshot(request)
        response = handler(request)
        self._record_model_call(state_kwargs, response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        state_kwargs = self._snapshot(request)
        response = await handler(request)
        self._record_model_call(state_kwargs, response)
        return response

    # -- tool boundary -------------------------------------------------------------------------
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        result = handler(request)
        self._record_tool_result(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        result = await handler(request)
        self._record_tool_result(request, result)
        return result

    # -- assembly --------------------------------------------------------------------------------
    def trajectory(self, trajectory_id: str) -> Trajectory:
        """Assemble the recorded run; raises (never guesses) if it wasn't a clean tool loop."""
        if not self._pending:
            raise ReplayError("nothing recorded — was the recorder passed as middleware?")

        observations = dict(self._observations)
        steps: list[Step] = []
        final_output: str | None = None
        for index, (state_kwargs, action) in enumerate(self._pending):
            state = State(**state_kwargs)
            if action.kind == "final":
                if index != len(self._pending) - 1:
                    raise ReplayError(
                        f"model call {index} produced a final answer but the run continued — "
                        f"not a create_agent-shaped loop CAR v1 can certify"
                    )
                steps.append(Step(index=index, state_before=state, action=action, observation=None))
                final_output = action.text or ""
                continue

            call_id = _tool_call_id_of(action)
            observation = observations.pop(call_id, None)
            if observation is None:
                raise ReplayError(
                    f"step {index}: no recorded tool result for tool_call_id {call_id!r} "
                    f"(interrupted run, Command-returning tool, or a non-linear graph)"
                )
            steps.append(
                Step(index=index, state_before=state, action=action, observation=observation)
            )

        if final_output is None:
            raise ReplayError(
                "the run never produced a final answer (interrupted or still mid-loop); "
                "refusing to record a truncated trajectory (PLAN.md s0.9)"
            )
        if observations:
            raise ReplayError(
                f"unmatched tool results for ids {sorted(observations)} — likely parallel tool "
                f"execution or a graph shape outside the v1 scope"
            )
        log.info("recorded langgraph run", trajectory_id=trajectory_id, n_steps=len(steps))
        return Trajectory(trajectory_id=trajectory_id, steps=steps, final_output=final_output)

    def reset(self) -> None:
        self._pending.clear()
        self._observations.clear()

    # -- internals -------------------------------------------------------------------------------
    @staticmethod
    def _snapshot(request: ModelRequest) -> dict[str, Any]:
        sampling = dict(request.model_settings or {})
        if request.tool_choice is not None:
            sampling["tool_choice"] = request.tool_choice
        system = request.system_message
        return {
            "system_prompt": _text_of(system.content) if system is not None else "",
            "tool_schemas": [convert_to_openai_tool(t) for t in (request.tools or [])],
            "model": _model_name(request.model),
            "provider": "langchain",
            "sampling": sampling,
            "messages": _project_messages(request.messages),
        }

    def _record_model_call(self, state_kwargs: dict[str, Any], response: Any) -> None:
        ai = _ai_of(response)
        self._pending.append((state_kwargs, _action_from_ai(ai)))

    def _record_tool_result(self, request: ToolCallRequest, result: Any) -> None:
        if not isinstance(result, ToolMessage):
            raise ReplayError(
                f"tool {request.tool_call['name']!r} returned {type(result).__name__}; "
                f"Command-returning tools are outside the v1 adapter scope"
            )
        call_id = request.tool_call.get("id") or ""
        self._observations[call_id] = Observation(
            tool_name=request.tool_call["name"],
            result=_text_of(result.content),
            source="real",
        )


def _ai_of(response: Any) -> AIMessage:
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None)
    if isinstance(result, list):
        ai_messages = [m for m in result if isinstance(m, AIMessage)]
        if ai_messages:
            return ai_messages[-1]
    raise ReplayError(f"could not extract an AIMessage from model response {type(response)!r}")


def _tool_call_id_of(action: Action) -> str:
    for call in action.raw.get("tool_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("id"), str):
            return str(call["id"])
    raise ReplayError("recorded tool_call action carries no tool_call id")


class LangChainPolicy:
    """pi(a | context) over a LangChain chat model — the policy CAR resamples during replay.

    Reconstructs the context from a recorded ``State`` (OpenAI-format messages -> LangChain
    messages via the public ``convert_to_messages``), rebinds the recorded tool schemas, and
    re-invokes the model. ``model_settings`` beyond tools/tool_choice are not re-applied in v1
    (documented limitation; the model object carries its own construction-time settings).
    """

    provider: Provider = "langchain"

    def __init__(self, model: BaseChatModel, *, model_id: str | None = None) -> None:
        self._model = model
        self._model_id = model_id or _model_name(model)

    @property
    def model_id(self) -> str:
        return self._model_id

    async def sample(self, state: State) -> Action:
        messages: list[Any] = list(convert_to_messages(state.messages))
        if state.system_prompt:
            messages.insert(0, SystemMessage(state.system_prompt))
        runnable: Any = self._model
        if state.tool_schemas:
            tool_choice = state.sampling.get("tool_choice")
            kwargs: dict[str, Any] = {"tool_choice": tool_choice} if tool_choice else {}
            runnable = self._model.bind_tools(list(state.tool_schemas), **kwargs)
        ai = await runnable.ainvoke(messages)
        if not isinstance(ai, AIMessage):
            raise ReplayError(f"model returned {type(ai).__name__}, expected AIMessage")
        return _action_from_ai(ai)


class LangChainToolEnvironment:
    """Live observations on counterfactual branches by invoking the agent's actual tools.

    Tools are invoked with a FULL ToolCall dict (required for ``InjectedToolCallId`` tools, and
    it returns a ready ``ToolMessage``). Tools that need graph-injected state/runtime are out of
    scope and will raise inside LangChain rather than be silently mis-observed.
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}

    async def observe(self, action: Action) -> Observation:
        if action.kind != "tool_call" or action.tool_name is None:
            raise ReplayError(f"environment cannot observe a non-tool action: {action.kind}")
        tool = self._tools.get(action.tool_name)
        if tool is None:
            raise ReplayError(
                f"tool {action.tool_name!r} not among the agent's tools {sorted(self._tools)}"
            )
        try:
            call_id = _tool_call_id_of(action)
        except ReplayError:
            call_id = "call_counterfactual"
        result = await tool.ainvoke(
            {
                "name": action.tool_name,
                "args": dict(action.tool_args or {}),
                "id": call_id,
                "type": "tool_call",
            }
        )
        content = _text_of(result.content) if isinstance(result, ToolMessage) else str(result)
        return Observation(tool_name=action.tool_name, result=content, source="real")


class _DisableParallelToolCalls(AgentMiddleware):
    """Force one tool call per turn on OpenAI-compatible models (RESEARCH phase_5 s3)."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        return handler(_no_parallel(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        return await handler(_no_parallel(request))


def _no_parallel(request: ModelRequest) -> ModelRequest:
    settings = {**(request.model_settings or {}), "parallel_tool_calls": False}
    return request.override(model_settings=settings)


def disable_parallel_tool_calls() -> AgentMiddleware:
    """Middleware that pins ``parallel_tool_calls=False`` (place BEFORE the recorder)."""
    return _DisableParallelToolCalls()


__all__ = [
    "LangChainPolicy",
    "LangChainToolEnvironment",
    "LangGraphRecorder",
    "disable_parallel_tool_calls",
]
