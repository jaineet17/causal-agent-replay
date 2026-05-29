"""Runtime configuration from environment (PLAN.md s8).

Defaults match ``.env.example``. Secrets are read lazily where needed (so the synthetic/test
path requires no keys); attribution cost limits live here for later phases.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    db_path: str = Field(default="./data/car.db")
    default_k: int = Field(default=16, ge=1)
    shapley_permutations: int = Field(default=64, ge=1)
    max_attribution_cost_usd: float = Field(default=10.0, ge=0.0)
    judge_model: str | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=os.environ.get("DB_PATH", "./data/car.db"),
            default_k=int(os.environ.get("DEFAULT_K", "16")),
            shapley_permutations=int(os.environ.get("SHAPLEY_PERMUTATIONS", "64")),
            max_attribution_cost_usd=float(os.environ.get("MAX_ATTRIBUTION_COST_USD", "10")),
            judge_model=os.environ.get("JUDGE_MODEL") or None,
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
