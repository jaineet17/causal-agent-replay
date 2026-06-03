"""The OpenAI-compatible path (Ollama / Groq / OpenRouter / vLLM share this wire format).

Proven against a fake client that mimics the ``openai`` SDK response shape, so these tests need
no server, no key, and no money — yet they exercise the real ``OpenAICompatiblePolicy`` parsing
and the real ``OpenAICodec`` message threading. The fake backend is deterministic (it replies
from the message history), which models the seeded-Ollama ideal: exact replay.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from car.record.recorder import (
    MockEnvironment,
    OpenAICodec,
    OpenAICompatiblePolicy,
    ToolRegistry,
    record_run,
)
from car.replay.deterministic import DeterministicReplay

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "lookup_order",
        "description": "Look up an order by id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    }
]

# Scripted assistant turns keyed by how many assistant turns have already happened.
_SCRIPT: list[dict[str, Any]] = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_order", "arguments": '{"order_id": "A1234"}'},
            }
        ],
    },
    {"role": "assistant", "content": "Your order #A1234 is on its way.", "tool_calls": None},
]


class _FakeMessage:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class _FakeCompletions:
    """Deterministic backend: replies from the conversation state (assistant-turn count)."""

    async def create(self, **kwargs: Any) -> Any:
        n_assistant = sum(1 for m in kwargs["messages"] if m.get("role") == "assistant")
        payload = _SCRIPT[min(n_assistant, len(_SCRIPT) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=_FakeMessage(payload))])


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("lookup_order", lambda args: '{"status": "shipped"}')
    return reg


async def _record() -> tuple:
    policy = OpenAICompatiblePolicy("fake-model", client=FakeOpenAIClient())
    codec = OpenAICodec()
    traj = await record_run(
        trajectory_id="openai-demo",
        policy=policy,
        environment=MockEnvironment(_registry()),
        codec=codec,
        system_prompt="be helpful",
        tool_schemas=_TOOL_SCHEMAS,
        user_input="where is order A1234?",
    )
    return traj, codec, policy


async def test_openai_path_records_tool_call_then_final() -> None:
    traj, _, _ = await _record()
    assert len(traj.steps) == 2
    assert traj.steps[0].action.kind == "tool_call"
    assert traj.steps[0].action.tool_name == "lookup_order"
    assert traj.steps[0].action.tool_args == {"order_id": "A1234"}
    assert traj.steps[1].action.kind == "final"
    assert "on its way" in (traj.steps[1].action.text or "")


async def test_openai_codec_threads_messages_faithfully() -> None:
    """The load-bearing Phase-0 invariant, now for the OpenAI/Ollama codec."""
    traj, codec, _ = await _record()
    assert DeterministicReplay(codec).verify_reconstruction(traj) is True
    # The reconstructed tool turn uses OpenAI linkage: role:tool + tool_call_id.
    tool_msg = codec.tool_result_message(traj.steps[0].action, traj.steps[0].observation)
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"


async def test_openai_path_replays_exactly_under_deterministic_backend() -> None:
    traj, codec, policy = await _record()
    report = await DeterministicReplay(codec).measure(traj, policy, n_samples=5)
    assert report.reconstruction_faithful
    assert report.sequence_reproduction_rate == 1.0
    assert all(s.match_rate == 1.0 for s in report.per_step)


def test_tool_schema_wrapping_accepts_bare_and_wrapped() -> None:
    from car.record.recorder import _as_openai_tool

    bare = _as_openai_tool(_TOOL_SCHEMAS[0])
    assert bare["type"] == "function"
    assert bare["function"]["name"] == "lookup_order"
    assert bare["function"]["parameters"]["required"] == ["order_id"]
    # Already-wrapped passes through unchanged.
    assert _as_openai_tool(bare) == bare
