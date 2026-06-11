"""Core trajectory schema — the recorded structural causal model of an agent run.

A trajectory is one sample from the distribution of possible runs:

    tau = [ s0, (a1, o1), (a2, o2), ..., (an, on), y ]

See PLAN.md s1 for the formal frame. The load-bearing field is ``Step.state_before``:
it must contain *everything* needed to re-issue the exact model call that produced the
step. Faithfulness of that reconstruction is what Phase 0 proves
(``test_deterministic_replay.py``). If it is incomplete, nothing downstream is trustworthy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# "synthetic" is a first-class provider for in-process policies with KNOWN causal structure
# (PLAN.md s7). The ground-truth validation fixtures run the identical replay/attribute code
# paths as a live LLM, so they must be representable in the same schema rather than masquerading
# as a real provider.
# "langchain" marks trajectories recorded through the LangGraph adapter: messages are stored in
# the OpenAI-projected wire format (the public, id-preserving projection — RESEARCH phase_5), and
# the policy is a LangChain chat-model object supplied by the caller, not reconstructable from a
# model-id string.
# "openai-agents" marks trajectories recorded through the OpenAI Agents SDK adapter: messages are
# stored as Responses-API input items (function_call / function_call_output, linked by call_id),
# and the policy wraps the caller's Model object.
Provider = Literal["anthropic", "openai", "synthetic", "langchain", "openai-agents"]


def _canonical_digest(payload: Any) -> str:
    """Stable content hash of an arbitrary JSON-able payload.

    Used to assert that a reconstructed state matches the recorded one byte-for-byte
    at the level that matters (the request we would send to the provider).
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class State(BaseModel):
    """The EXACT, reconstructable state from which an action is decided at a step.

    Everything required to re-issue the model call lives here. ``messages`` is the full
    provider-native context (Anthropic ``messages`` blocks or OpenAI chat messages) up to
    but not including the action this state produces.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    tool_schemas: list[dict[str, Any]]
    model: str
    provider: Provider
    sampling: dict[str, Any] = Field(
        description="temperature, top_p, max_tokens, seed (if the provider supports it), etc."
    )
    messages: list[dict[str, Any]] = Field(
        description="Full provider-native context at this point — enough to reconstruct the call."
    )

    def request_digest(self) -> str:
        """Hash of the request-determining fields.

        Two states with the same digest will, modulo provider nondeterminism, produce the
        same action distribution. This is the equality used by the faithfulness proof.
        """
        return _canonical_digest(
            {
                "system_prompt": self.system_prompt,
                "tool_schemas": self.tool_schemas,
                "model": self.model,
                "provider": self.provider,
                "sampling": self.sampling,
                "messages": self.messages,
            }
        )


class Action(BaseModel):
    """The agent's output at a step: a tool call, or a final answer.

    Drawn from the stochastic policy pi(a_k | context_k). ``raw`` retains the full provider
    response so the action can be reconstructed and re-serialized into message history
    faithfully (tool_use ids, content-block ordering, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool_call", "final"]
    text: str | None = Field(default=None, description="Thought text or final answer.")
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    raw: dict[str, Any] = Field(
        description="Raw provider response, for faithful reconstruction into message history."
    )


class Observation(BaseModel):
    """The environment's response to an action's tool call.

    ``source`` records provenance: ``real`` (live tool, possibly non-reproducible),
    ``recorded`` (replayed from the original run), or ``mocked`` (deterministic fixture).
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    result: str
    source: Literal["real", "recorded", "mocked"]


class Outcome(BaseModel):
    """The label/score produced by a user-supplied outcome function Y(tau).

    ``score`` is in [0, 1] so we can reason over outcome *distributions* (means, divergences,
    confidence intervals), not just discrete labels.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description='e.g. "inappropriate_refund" | "ok"')
    score: float = Field(ge=0.0, le=1.0, description="0..1, for distributional reasoning.")
    detail: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    """One (state -> action -> observation) transition of the agent loop."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    state_before: State
    action: Action
    observation: Observation | None = Field(
        default=None, description="None on the final step (no tool call)."
    )


class Trajectory(BaseModel):
    """A recorded run, or a counterfactual branch of one.

    Counterfactual branches set ``parent_id`` / ``branched_at_step`` / ``intervention_id``,
    so a root trajectory plus its branches form a *tree*, not a flat list of rows.
    """

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    parent_id: str | None = None
    branched_at_step: int | None = None
    intervention_id: str | None = None
    steps: list[Step]
    final_output: str
    outcome: Outcome | None = None

    @property
    def is_branch(self) -> bool:
        return self.parent_id is not None

    def action_signature(self) -> list[str]:
        """Compact per-step action signature, for measuring action-match rate on replay."""
        sig: list[str] = []
        for step in self.steps:
            a = step.action
            if a.kind == "final":
                sig.append("final")
            else:
                sig.append(f"tool:{a.tool_name}:{_canonical_digest(a.tool_args)[:12]}")
        return sig
