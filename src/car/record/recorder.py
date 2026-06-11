"""Concrete policies, codecs, environment, and the recording entry point.

Contains:
  - ``ToolRegistry`` + ``MockEnvironment`` — pluggable, reproducible tools for the demo and tests.
  - ``AnthropicPolicy`` + ``AnthropicCodec`` — the live agent-under-test on Anthropic, with the
    Opus-4.7+ sampling-param handling the research flagged (RESEARCH s1/s4: those models 400 on
    ``temperature``/``top_p``/``top_k``).
  - ``SyntheticCodec`` — the simple message encoding used by in-process synthetic policies.
  - ``record_run`` — wires a policy + environment + codec into a ``ToolLoop`` and records.

OpenAI support is intentionally deferred (see ``OpenAIPolicy``): the documented shapes are in
RESEARCH s3, but shipping untested live-provider code that claims to work would violate the
no-silent-failure discipline. Anthropic is implemented and exercised; OpenAI raises explicitly.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import structlog

from car.record.toolloop import MessageCodec, ToolLoop
from car.schemas.scm import Environment, Policy, ReplayError
from car.schemas.trajectory import Action, Observation, Provider, State, Trajectory

log = structlog.get_logger(__name__)

ToolFn = Callable[[dict[str, Any]], "str | Awaitable[str]"]


# --------------------------------------------------------------------------------------------
# Tools / environment (reproducible, no real side effects)
# --------------------------------------------------------------------------------------------
class ToolRegistry:
    """A name -> callable map. Tool fns take parsed args and return a result string."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = fn

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def call(self, name: str, args: dict[str, Any]) -> str:
        fn = self._tools.get(name)
        if fn is None:
            raise ReplayError(f"tool {name!r} not in registry {self.names()}")
        result = fn(args)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str):
            raise ReplayError(f"tool {name!r} returned {type(result).__name__}, expected str")
        return result


class MockEnvironment:
    """An ``Environment`` backed by a ``ToolRegistry``; every observation is ``source="mocked"``.

    Mocked tools are deterministic and side-effect-free, which is what makes recorded runs
    reproducible (PLAN.md s5.1). Real tools are a deferred concern (PLAN.md s12).
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def observe(self, action: Action) -> Observation:
        if action.kind != "tool_call" or action.tool_name is None:
            raise ReplayError(f"environment cannot observe a non-tool action: {action.kind}")
        result = await self._registry.call(action.tool_name, action.tool_args or {})
        return Observation(tool_name=action.tool_name, result=result, source="mocked")


# --------------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------------
_ANTHROPIC_NO_SAMPLING = ("opus-4-7", "opus-4-8")  # models that 400 on temperature/top_p/top_k


def _anthropic_accepts_sampling_params(model: str) -> bool:
    """False for Claude Opus 4.7+, which reject ``temperature``/``top_p``/``top_k`` (HTTP 400)."""
    return not any(tag in model for tag in _ANTHROPIC_NO_SAMPLING)


class AnthropicCodec:
    """Lossless Anthropic message serialization (``tool_use`` / ``tool_result``).

    ``Action.raw`` holds the verbatim response ``content`` blocks; we reconstruct the assistant
    turn from them and link the ``tool_result`` to the call by ``tool_use_id`` (RESEARCH s3).
    """

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, action: Action) -> dict[str, Any]:
        content = action.raw.get("content")
        if content is None:
            raise ReplayError("Anthropic action.raw is missing 'content'; cannot reconstruct turn")
        return {"role": "assistant", "content": content}

    def tool_result_message(self, action: Action, observation: Observation) -> dict[str, Any]:
        tool_use_id = self._tool_use_id(action)
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": observation.result,
                }
            ],
        }

    @staticmethod
    def _tool_use_id(action: Action) -> str:
        for block in action.raw.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = block.get("id")
                if isinstance(tid, str):
                    return tid
        raise ReplayError("no tool_use block with an id found in Anthropic action.raw")

    def forge_action(
        self,
        *,
        kind: Literal["tool_call", "final"],
        text: str | None,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
    ) -> Action:
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        if kind == "tool_call":
            if tool_name is None:
                raise ReplayError("forge_action(tool_call) requires tool_name")
            content.append(
                {
                    "type": "tool_use",
                    "id": "toolu_forced",
                    "name": tool_name,
                    "input": tool_args or {},
                }
            )
        raw = {
            "role": "assistant",
            "content": content,
            "stop_reason": ("tool_use" if kind == "tool_call" else "end_turn"),
        }
        return Action(kind=kind, text=text, tool_name=tool_name, tool_args=tool_args, raw=raw)


class AnthropicPolicy:
    """The agent-under-test, backed by the Anthropic Messages API.

    Constructed with an ``AsyncAnthropic`` client (injected for testability). Strips unsupported
    sampling params for Opus 4.7+ and records the verbatim response into ``Action.raw``.
    """

    provider: Provider = "anthropic"

    def __init__(self, model: str, client: Any | None = None, *, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ReplayError("anthropic SDK not installed") from exc
            client = AsyncAnthropic()
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model

    async def sample(self, state: State) -> Action:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "system": state.system_prompt,
            "messages": state.messages,
            "max_tokens": int(state.sampling.get("max_tokens", self._max_tokens)),
        }
        if state.tool_schemas:
            kwargs["tools"] = state.tool_schemas
        if _anthropic_accepts_sampling_params(self._model):
            for key in ("temperature", "top_p", "top_k"):
                if key in state.sampling:
                    kwargs[key] = state.sampling[key]
        elif any(k in state.sampling for k in ("temperature", "top_p", "top_k")):
            log.warning(
                "stripping unsupported sampling params for model",
                model=self._model,
                stripped=[k for k in ("temperature", "top_p", "top_k") if k in state.sampling],
            )

        response = await self._client.messages.create(**kwargs)
        raw = response.model_dump()
        return self._parse(raw)

    @staticmethod
    def _parse(raw: dict[str, Any]) -> Action:
        blocks = raw.get("content", [])
        text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        text = "\n".join(p for p in text_parts if p) or None

        if raw.get("stop_reason") == "tool_use" or tool_uses:
            if len(tool_uses) != 1:
                raise ReplayError(
                    f"expected exactly one tool_use block, got {len(tool_uses)} "
                    f"(parallel tool calls are a deferred v0 limitation)"
                )
            tu = tool_uses[0]
            return Action(
                kind="tool_call",
                text=text,
                tool_name=tu.get("name"),
                tool_args=tu.get("input") or {},
                raw=raw,
            )
        return Action(kind="final", text=text, raw=raw)


# --------------------------------------------------------------------------------------------
# OpenAI-compatible (covers Ollama, Groq, OpenRouter, vLLM, LM Studio, ... — same wire format)
# --------------------------------------------------------------------------------------------
class OpenAICodec:
    """Lossless OpenAI-compatible message serialization (``tool_calls`` / ``role:tool``).

    ``Action.raw`` holds the verbatim assistant message dump. We reconstruct the assistant turn
    from its semantic fields (role/content/tool_calls) — robust across strict servers like
    Ollama — and link tool results by ``tool_call_id`` (RESEARCH s3). Tool-call ``arguments`` is
    a JSON-encoded STRING in this wire format and is preserved as such.
    """

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, action: Action) -> dict[str, Any]:
        raw = action.raw
        msg: dict[str, Any] = {"role": "assistant", "content": raw.get("content")}
        tool_calls = raw.get("tool_calls")
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def tool_result_message(self, action: Action, observation: Observation) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self._tool_call_id(action),
            "content": observation.result,
        }

    @staticmethod
    def _tool_call_id(action: Action) -> str:
        for call in action.raw.get("tool_calls") or []:
            cid = call.get("id") if isinstance(call, dict) else None
            if isinstance(cid, str):
                return cid
        raise ReplayError("no tool_call with an id found in OpenAI action.raw")

    def forge_action(
        self,
        *,
        kind: Literal["tool_call", "final"],
        text: str | None,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
    ) -> Action:
        raw: dict[str, Any] = {"role": "assistant", "content": text}
        if kind == "tool_call":
            if tool_name is None:
                raise ReplayError("forge_action(tool_call) requires tool_name")
            raw["tool_calls"] = [
                {
                    "id": "call_forced",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_args or {})},
                }
            ]
        return Action(kind=kind, text=text, tool_name=tool_name, tool_args=tool_args, raw=raw)


class LangChainProjectionCodec(OpenAICodec):
    """OpenAI codec variant matching LangChain's ``convert_to_openai_messages`` projection.

    The projection adds ``"name"`` to tool messages (from ``ToolMessage.name``); the rebuild must
    emit the identical dict or the faithfulness digest check fails. Lives in core (it is pure
    dict shaping) so ``codec_for("langchain")`` works without the optional langgraph extra.
    """

    def tool_result_message(self, action: Action, observation: Observation) -> dict[str, Any]:
        return {
            "role": "tool",
            "name": observation.tool_name,
            "tool_call_id": self._tool_call_id(action),
            "content": observation.result,
        }


class OpenAIAgentsCodec:
    """Responses-API item serialization for the OpenAI Agents SDK adapter.

    The Agents SDK threads context as Responses-API *input items*, not chat messages: a tool call
    is a ``function_call`` item and its result a ``function_call_output`` item, linked by
    ``call_id`` (not ``tool_call_id``). ``Action.raw`` holds the verbatim ``function_call`` item.
    Lives in core (pure dict shaping) so ``codec_for("openai-agents")`` needs no SDK import.
    """

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, action: Action) -> dict[str, Any]:
        if action.kind == "final":
            return {"role": "assistant", "content": action.text or ""}
        return dict(action.raw)  # the verbatim function_call item

    def tool_result_message(self, action: Action, observation: Observation) -> dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": self._call_id(action),
            "output": observation.result,
        }

    @staticmethod
    def _call_id(action: Action) -> str:
        cid = action.raw.get("call_id")
        if isinstance(cid, str):
            return cid
        raise ReplayError("no call_id found in openai-agents action.raw")

    def forge_action(
        self,
        *,
        kind: Literal["tool_call", "final"],
        text: str | None,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
    ) -> Action:
        if kind == "final":
            return Action(kind="final", text=text, raw={"role": "assistant", "content": text or ""})
        if tool_name is None:
            raise ReplayError("forge_action(tool_call) requires tool_name")
        raw = {
            "type": "function_call",
            "call_id": "call_forced",
            "name": tool_name,
            "arguments": json.dumps(tool_args or {}),
        }
        return Action(
            kind="tool_call", text=text, tool_name=tool_name, tool_args=tool_args, raw=raw
        )


# Sampling params the OpenAI-compatible wire format accepts (Ollama maps seed/temperature/top_p
# onto its native options; RESEARCH: local seeded inference is far more deterministic than hosted).
_OPENAI_SAMPLING_KEYS = ("temperature", "top_p", "seed", "frequency_penalty", "presence_penalty")


class OpenAICompatiblePolicy:
    """Agent-under-test on any OpenAI-compatible endpoint, incl. local Ollama (free).

    ``base_url`` selects the backend (e.g. ``http://localhost:11434/v1`` for Ollama,
    ``https://api.groq.com/openai/v1`` for Groq). When omitted, the ``openai`` SDK reads
    ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` from the environment, which is also how replay
    reconstructs the policy via :func:`policy_for`.
    """

    provider: Provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ReplayError("openai SDK not installed") from exc
            # api_key must be non-empty for the SDK even when the server ignores it (Ollama).
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
            client = AsyncOpenAI(base_url=base_url, api_key=resolved_key)
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model

    async def sample(self, state: State) -> Action:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": state.system_prompt}, *state.messages],
            "max_tokens": int(state.sampling.get("max_tokens", self._max_tokens)),
        }
        if state.tool_schemas:
            kwargs["tools"] = [_as_openai_tool(t) for t in state.tool_schemas]
        for key in _OPENAI_SAMPLING_KEYS:
            if key in state.sampling:
                kwargs[key] = state.sampling[key]
        # Passthrough for backend-native options, e.g. Ollama's {"options": {"num_ctx": N}} —
        # fixing num_ctx is required for fully reproducible local output (RESEARCH s5).
        if "extra_body" in state.sampling:
            kwargs["extra_body"] = state.sampling["extra_body"]

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        raw = message.model_dump()
        return self._parse(raw)

    @staticmethod
    def _parse(raw: dict[str, Any]) -> Action:
        text = raw.get("content")
        tool_calls = raw.get("tool_calls") or []
        if tool_calls:
            if len(tool_calls) != 1:
                raise ReplayError(
                    f"expected exactly one tool_call, got {len(tool_calls)} "
                    f"(parallel tool calls are a deferred v0 limitation)"
                )
            fn = tool_calls[0].get("function", {})
            arguments = fn.get("arguments") or "{}"
            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ReplayError(
                    f"tool_call arguments were not valid JSON: {arguments!r}"
                ) from exc
            return Action(
                kind="tool_call",
                text=text,
                tool_name=fn.get("name"),
                tool_args=parsed_args,
                raw=raw,
            )
        return Action(kind="final", text=text, raw=raw)


def _as_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Accept either a bare {name, description, parameters/input_schema} or an already-wrapped
    {type: function, function: {...}} tool schema, and emit the OpenAI ``tools`` shape."""
    if schema.get("type") == "function" and "function" in schema:
        return schema
    params = schema.get("parameters") or schema.get("input_schema") or {"type": "object"}
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": params,
        },
    }


def ollama_policy(model: str = "llama3.1:8b", *, max_tokens: int = 1024) -> OpenAICompatiblePolicy:
    """Convenience: an OpenAI-compatible policy pointed at a local Ollama server (free).

    Requires ``ollama serve`` running and ``ollama pull <model>``. Tool calling needs a
    tool-capable model (e.g. llama3.1/3.2/3.3, qwen2.5) and a recent Ollama.
    """
    return OpenAICompatiblePolicy(
        model,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # ignored by the server, required non-empty by the SDK
        max_tokens=max_tokens,
    )


class SyntheticCodec:
    """Minimal message encoding for in-process synthetic policies (the ground-truth fixtures).

    Not provider-native — it only needs to be internally consistent so the replay machinery can
    be validated independently of any real provider's nondeterminism.
    """

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, action: Action) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": action.text,
            "tool_call": (
                None
                if action.kind == "final"
                else {"name": action.tool_name, "arguments": json.dumps(action.tool_args or {})}
            ),
        }

    def tool_result_message(self, action: Action, observation: Observation) -> dict[str, Any]:
        return {"role": "tool", "name": observation.tool_name, "content": observation.result}

    def forge_action(
        self,
        *,
        kind: Literal["tool_call", "final"],
        text: str | None,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
    ) -> Action:
        # SyntheticCodec threads from action fields, so raw can be minimal.
        return Action(
            kind=kind,
            text=text,
            tool_name=tool_name,
            tool_args=tool_args,
            raw={"synthetic": True, "forced": True},
        )


# --------------------------------------------------------------------------------------------
# Provider factories (reconstruct a policy/codec from a recorded trajectory's provider)
# --------------------------------------------------------------------------------------------
def codec_for(provider: Provider) -> MessageCodec:
    if provider == "anthropic":
        return AnthropicCodec()
    if provider == "openai":
        return OpenAICodec()
    if provider == "langchain":
        # LangGraph-adapter trajectories store messages via LangChain's OpenAI projection,
        # which adds "name" on tool messages (RESEARCH phase_5).
        return LangChainProjectionCodec()
    if provider == "openai-agents":
        return OpenAIAgentsCodec()
    if provider == "synthetic":
        return SyntheticCodec()
    raise ReplayError(f"no codec for provider {provider!r}")


def policy_for(provider: Provider, model: str) -> Policy:
    if provider == "anthropic":
        return AnthropicPolicy(model)
    if provider == "openai":
        # base_url / api_key come from OPENAI_BASE_URL / OPENAI_API_KEY (set these to point at
        # Ollama, Groq, etc. when replaying an OpenAI-compatible recording).
        return OpenAICompatiblePolicy(model)
    if provider == "langchain":
        raise ReplayError(
            "a 'langchain' trajectory replays against the caller's chat-model object, which "
            "cannot be reconstructed from a model-id string — construct "
            "car.adapters.langgraph.LangChainPolicy(your_model) and pass it explicitly"
        )
    if provider == "openai-agents":
        raise ReplayError(
            "an 'openai-agents' trajectory replays against the caller's Model object — construct "
            "car.adapters.openai_agents.OpenAIAgentsPolicy(your_model, tools=...) and pass it in"
        )
    raise ReplayError(
        f"cannot reconstruct a {provider!r} policy outside its fixture "
        f"(synthetic policies are in-process)"
    )


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
async def record_run(
    *,
    trajectory_id: str,
    policy: Policy,
    environment: Environment,
    codec: MessageCodec,
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    user_input: str,
    sampling: dict[str, Any] | None = None,
    max_steps: int = 20,
) -> Trajectory:
    """Record one agent run faithfully into a ``Trajectory``."""
    loop = ToolLoop(policy, environment, codec, max_steps=max_steps)
    traj = await loop.run(
        trajectory_id=trajectory_id,
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        user_input=user_input,
        sampling=sampling,
    )
    log.info(
        "recorded run",
        trajectory_id=trajectory_id,
        provider=policy.provider,
        model=policy.model_id,
        n_steps=len(traj.steps),
    )
    return traj
