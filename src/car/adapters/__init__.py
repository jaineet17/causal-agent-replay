"""Framework adapters: record and counterfactually re-execute agents people already have.

The native recorder (``car.record``) owns its tool loop, so capture is trivially faithful. An
adapter instead instruments someone else's loop — faithfulness has to be *checked*, which is why
every adapter routes its recordings through the same ``DeterministicReplay.verify_reconstruction``
invariant the native path is held to.

Available:
  - ``car.adapters.langgraph`` — LangChain ``create_agent`` / LangGraph
    (``causal-agent-replay[langgraph]``);
  - ``car.adapters.openai_agents`` — the OpenAI Agents SDK
    (``causal-agent-replay[openai-agents]``).
"""
