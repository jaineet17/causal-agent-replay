---
title: Causal Agent Replay
---

# Causal Agent Replay

**Find which step actually caused your agent to fail — by intervening on it and measuring whether the outcome changes.**

Observability shows you *what happened*; eval scores *pass/fail*. Neither answers *which step
caused it*. Causal Agent Replay does — it intervenes (`do(·)`) on a step, re-runs the agent
forward under the same stochastic policy, and measures the shift in the outcome distribution. The
step where re-deciding changes the outcome, but later steps don't, is the **causal locus**.

![Attribution report: the causal locus is the decision step.](assets/demo.png)

- 📝 **Read the post:** [Which step made your agent fail? Prove it by counterfactual.](blog.html)
- 🔬 **Deep dive:** [the technical writeup](writeup.html)
- 💻 **Code:** [github.com/jaineet17/causal-agent-replay](https://github.com/jaineet17/causal-agent-replay)
- 📦 **Install:** `pip install causal-agent-replay`

Apache-2.0.
