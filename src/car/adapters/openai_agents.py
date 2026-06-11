"""OpenAI Agents SDK adapter: record ``Runner.run`` agents, replay them counterfactually.

The capture surface (RESEARCH/phase_5_adapters.md s5) is a wrapping ``Model``: the Runner calls
``get_response(system_instructions, input, model_settings, tools, ...)`` once per turn, which hands
over exactly what CAR's ``state_before`` needs — the system instructions, the full Responses-API
input-item history, the model settings, and the tools — plus the response output to parse into an
action. Wrap the user's model in one line:

    from car.adapters.openai_agents import OpenAIAgentsRecorder
    recorder = OpenAIAgentsRecorder(real_model)
    agent = Agent(name=..., instructions=..., tools=[...], model=recorder)
    await Runner.run(agent, "...")
    trajectory = recorder.trajectory("my-run")

Messages are stored as Responses-API input items (``function_call`` / ``function_call_output``,
linked by ``call_id``) via ``OpenAIAgentsCodec``, and the recording is held to the SAME
``verify_reconstruction`` invariant as the native recorder.

v1 scope (refused loudly, never silently mis-recorded): one tool call per turn (parallel
``function_call`` sets raise); single-agent runs (no handoffs); string-returning function tools;
non-streaming ``Runner.run`` (streaming passes through unrecorded).
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import structlog

from car.schemas.scm import ReplayError
from car.schemas.trajectory import Action, Observation, Provider, State, Step, Trajectory

try:
    from agents.model_settings import ModelSettings
    from agents.models.interface import Model, ModelTracing
    from agents.tool_context import ToolContext
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "the OpenAI Agents adapter needs the optional extra: "
        "pip install 'causal-agent-replay[openai-agents]'"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import Sequence

log = structlog.get_logger(__name__)

_PARALLEL_REFUSAL = (
    "the model emitted {n} parallel tool calls in one turn; CAR v1 models one action per step "
    "(PLAN.md s5.1). Set the agent's model_settings to disable parallel tool calls, or prompt it "
    "to act sequentially."
)
_MODEL_SETTINGS_FIELDS = {f.name for f in dataclasses.fields(ModelSettings)}


def _to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dict(dump(exclude_none=True))
    raise ReplayError(f"cannot normalize Responses item of type {type(item).__name__}")


def _item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type", item.get("role", "")))
    return str(getattr(item, "type", getattr(item, "role", "")))


def _normalize_input(input_: str | Sequence[Any]) -> list[dict[str, Any]]:
    if isinstance(input_, str):
        return [{"role": "user", "content": input_}]
    return [_to_dict(i) for i in input_]


def _text_of_message(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("text")]
        return "".join(parts)
    return ""


def _action_from_output(output: Sequence[Any]) -> Action:
    items = [_to_dict(o) for o in output]
    calls = [it for it in items if it.get("type") == "function_call"]
    if len(calls) > 1:
        raise ReplayError(_PARALLEL_REFUSAL.format(n=len(calls)))
    if calls:
        call = calls[0]
        return Action(
            kind="tool_call",
            text=None,
            tool_name=str(call["name"]),
            tool_args=json.loads(call.get("arguments") or "{}"),
            raw=call,
        )
    messages = [it for it in items if it.get("type") == "message" or it.get("role") == "assistant"]
    text = _text_of_message(messages[-1]) if messages else ""
    return Action(kind="final", text=text, raw={"role": "assistant", "content": text})


def _settings_to_dict(model_settings: Any) -> dict[str, Any]:
    if model_settings is None:
        return {}
    try:
        raw = dataclasses.asdict(model_settings)
    except TypeError:
        return {}
    return {k: v for k, v in raw.items() if v is not None and k in _MODEL_SETTINGS_FIELDS}


def _model_id(model: Any) -> str:
    value = getattr(model, "model", None)
    if isinstance(value, str) and value:
        return value
    return type(model).__name__


class OpenAIAgentsRecorder(Model):
    """Wraps the agent's ``Model`` and records each turn into a CAR ``Trajectory``.

    ``get_response`` is the capture point; ``stream_response`` passes through unrecorded (so a
    streamed run still works, but yields no trajectory — call ``Runner.run``, not the streamed
    variant, to record).
    """

    def __init__(self, model: Model, *, model_id: str | None = None) -> None:
        self._inner = model
        self._model_id = model_id or _model_id(model)
        self._pending: list[tuple[dict[str, Any], Action]] = []
        self._observations: dict[str, Observation] = {}

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        **kwargs: Any,
    ) -> Any:
        items = _normalize_input(input)
        self._harvest_observations(items)
        state_kwargs = {
            "system_prompt": system_instructions or "",
            "tool_schemas": [self._tool_schema(t) for t in tools],
            "model": self._model_id,
            "provider": "openai-agents",
            "sampling": _settings_to_dict(model_settings),
            "messages": items,
        }
        response = await self._inner.get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        )
        self._pending.append((state_kwargs, _action_from_output(response.output)))
        return response

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - pass-through
        return self._inner.stream_response(*args, **kwargs)

    # -- assembly --------------------------------------------------------------------------------
    def trajectory(self, trajectory_id: str) -> Trajectory:
        if not self._pending:
            raise ReplayError("nothing recorded — was the recorder set as the agent's model?")
        observations = dict(self._observations)
        steps: list[Step] = []
        final_output: str | None = None
        for index, (state_kwargs, action) in enumerate(self._pending):
            state = State(**state_kwargs)
            if action.kind == "final":
                if index != len(self._pending) - 1:
                    raise ReplayError(
                        f"model call {index} produced a final answer but the run continued — "
                        f"not a single-agent loop CAR v1 can certify"
                    )
                steps.append(Step(index=index, state_before=state, action=action, observation=None))
                final_output = action.text or ""
                continue
            call_id = str(action.raw.get("call_id"))
            observation = observations.pop(call_id, None)
            if observation is None:
                raise ReplayError(
                    f"step {index}: no recorded output for call_id {call_id!r} "
                    f"(interrupted run, handoff, or a non-linear loop)"
                )
            steps.append(
                Step(index=index, state_before=state, action=action, observation=observation)
            )
        if final_output is None:
            raise ReplayError(
                "the run never produced a final answer; refusing a truncated trajectory (s0.9)"
            )
        log.info("recorded openai-agents run", trajectory_id=trajectory_id, n_steps=len(steps))
        return Trajectory(trajectory_id=trajectory_id, steps=steps, final_output=final_output)

    def reset(self) -> None:
        self._pending.clear()
        self._observations.clear()

    def _harvest_observations(self, items: Sequence[dict[str, Any]]) -> None:
        for item in items:
            if item.get("type") == "function_call_output":
                call_id = str(item.get("call_id"))
                if call_id not in self._observations:
                    self._observations[call_id] = Observation(
                        tool_name="", result=str(item.get("output", "")), source="real"
                    )

    @staticmethod
    def _tool_schema(tool: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "name": getattr(tool, "name", "?"),
            "description": getattr(tool, "description", "") or "",
            "parameters": getattr(tool, "params_json_schema", {}) or {},
        }


class OpenAIAgentsPolicy:
    """pi(a | context) over an Agents-SDK ``Model`` — what CAR resamples during replay."""

    provider: Provider = "openai-agents"

    def __init__(
        self, model: Model, *, tools: Sequence[Any] = (), model_id: str | None = None
    ) -> None:
        self._model = model
        self._tools = list(tools)
        self._model_id = model_id or _model_id(model)

    @property
    def model_id(self) -> str:
        return self._model_id

    async def sample(self, state: State) -> Action:
        settings = ModelSettings(
            **{k: v for k, v in state.sampling.items() if k in _MODEL_SETTINGS_FIELDS}
        )
        response = await self._model.get_response(
            state.system_prompt or None,
            list(state.messages),  # type: ignore[arg-type]  # Responses-API input items
            settings,
            self._tools,
            None,
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        return _action_from_output(response.output)


class OpenAIAgentsToolEnvironment:
    """Live observations on counterfactual branches by invoking the agent's real function tools."""

    def __init__(self, tools: Sequence[Any]) -> None:
        self._tools: dict[str, Any] = {getattr(t, "name", "?"): t for t in tools}

    async def observe(self, action: Action) -> Observation:
        if action.kind != "tool_call" or action.tool_name is None:
            raise ReplayError(f"environment cannot observe a non-tool action: {action.kind}")
        tool = self._tools.get(action.tool_name)
        if tool is None:
            raise ReplayError(
                f"tool {action.tool_name!r} not among the agent's tools {sorted(self._tools)}"
            )
        invoke = getattr(tool, "on_invoke_tool", None)
        if invoke is None:
            raise ReplayError(f"tool {action.tool_name!r} is not an invokable FunctionTool")
        arguments = json.dumps(action.tool_args or {})
        call_id = str(action.raw.get("call_id") or "call_counterfactual")
        ctx: Any = ToolContext(
            context=None,
            tool_name=action.tool_name,
            tool_call_id=call_id,
            tool_arguments=arguments,
        )
        result = await invoke(ctx, arguments)
        return Observation(tool_name=action.tool_name, result=str(result), source="real")


__all__ = [
    "OpenAIAgentsPolicy",
    "OpenAIAgentsRecorder",
    "OpenAIAgentsToolEnvironment",
]
