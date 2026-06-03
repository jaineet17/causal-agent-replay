"""Test access to the synthetic SCM toolkit (now shipped as ``car.synthetic``) + test builders.

The reusable policies/environment live in the package (``car.synthetic``) so they validate
attribution and drive the demo without importing from tests. This module re-exports them and adds
the test-specific scenario builders.
"""

from __future__ import annotations

from car.schemas.scm import Environment
from car.synthetic import (
    DictEnvironment,
    MultiNoisyPolicy,
    NoisyPolicy,
    RulePolicy,
    ScriptedPolicy,
    final,
    last_tool_result,
    tool_call,
    turn_index,
    user_text,
)

__all__ = [
    "DictEnvironment",
    "MultiNoisyPolicy",
    "NoisyPolicy",
    "RulePolicy",
    "ScriptedPolicy",
    "final",
    "last_tool_result",
    "support_like_script",
    "tool_call",
    "turn_index",
    "user_text",
]


def support_like_script() -> tuple[list, Environment]:
    """A 3-step deterministic run resembling the support agent: lookup -> refund -> final."""
    actions = [
        tool_call("lookup_order", {"order_id": "A1234"}, text="Let me look that up."),
        tool_call("issue_refund", {"order_id": "A1234", "amount": 99.0}, text="Processing refund."),
        final("Your refund has been processed."),
    ]
    env = DictEnvironment(
        {
            "lookup_order": '{"status": "shipped", "defect_reported": false}',
            "issue_refund": '{"ok": true}',
        }
    )
    return actions, env
