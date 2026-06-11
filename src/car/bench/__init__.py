"""Benchmark harnesses (Phase 6): evaluating CAR's attribution on external benchmarks.

``car.bench.whowhen`` loads the Who&When failure-attribution benchmark (Zhang et al., ICML 2025)
and ``car.bench.attribute_log`` runs surrogate-policy counterfactual attribution over its static
logs — the PLAN.md s12 "learned surrogate policy" extension made concrete. Design + validity
analysis: RESEARCH/phase_6_benchmark.md.

The dataset carries no published license: it is fetched at runtime from HuggingFace for local
evaluation and is never committed to this repository.
"""
