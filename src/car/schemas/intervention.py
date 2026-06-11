"""The intervention algebra — the do(.) operations of the SCM.

This algebra is part of the novel contribution (PLAN.md s4.2). An ``Intervention`` names a
step k and a ``do(.)`` operation that perturbs one variable there; the trajectory is then
re-run forward from k under the (unchanged, stochastic) policy, yielding a *distribution*
over counterfactual outcomes.

``do_resample`` is the load-bearing primitive: it changes nothing except re-drawing the
action a_k from the SAME policy. It measures the intrinsic causal sensitivity of the outcome
to step k, which is the foundation of contrastive attribution (PLAN.md s5.4).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

InterventionKind = Literal[
    "do_observation",
    "do_action",
    "do_context",
    "do_policy",
    "do_resample",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_id: str
    step: int = Field(ge=0, description="The step index to intervene at.")


class DoResample(_Base):
    """NULL intervention: re-draw a_k from the unchanged policy. Empty payload.

    The most important primitive. Holding steps < k at observed values and letting steps > k
    flow, repeated K times, estimates P(outcome | resample at k) — the intrinsic sensitivity
    of the outcome to the decision made at k.
    """

    kind: Literal["do_resample"] = "do_resample"


class DoObservation(_Base):
    """Replace the tool result o_k, then run forward. ("what if the tool had returned X?")"""

    kind: Literal["do_observation"] = "do_observation"
    new_result: str
    new_source: Literal["mocked", "real", "recorded"] = "mocked"


class DoAction(_Base):
    """Force a specific action a_k, then let k+1.. flow. ("what if it had done X instead?")"""

    kind: Literal["do_action"] = "do_action"
    action_kind: Literal["tool_call", "final"]
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


class DoContext(_Base):
    """Edit the message history at step k, then run forward. ("what if the prompt lacked X?")

    ``message_ops`` is a list of edits applied to ``state_before.messages`` at k; the exact
    op grammar is defined by the replay engine. Kept as typed dicts here so the schema stays
    stable while the op set evolves.
    """

    kind: Literal["do_context"] = "do_context"
    message_ops: list[dict[str, Any]]


class DoPolicy(_Base):
    """Swap the model from step k forward. (the "the model upgrade broke it" case)"""

    kind: Literal["do_policy"] = "do_policy"
    new_model: str
    new_provider: Literal["anthropic", "openai", "synthetic", "langchain"] | None = None
    new_sampling: dict[str, Any] | None = None


Intervention = Annotated[
    DoResample | DoObservation | DoAction | DoContext | DoPolicy,
    Field(discriminator="kind"),
]
"""Discriminated union over the five do(.) kinds. Use ``car.schemas.intervention.Intervention``
as the field type anywhere an intervention is accepted; pydantic dispatches on ``kind``."""
