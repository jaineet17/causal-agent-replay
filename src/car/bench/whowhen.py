"""Who&When benchmark loader (Zhang et al., ICML 2025; HF ``Kevin355/Who_and_When``).

Normalizes both subsets into one schema (verified against real instances, 2026-06-11):

  - **Algorithm-Generated** (126): keys ``is_correct``, ``question``, ``ground_truth``,
    ``history`` (items ``{content, role, name}``), ``mistake_agent``, ``mistake_step`` (str!),
    ``mistake_reason``, ``system_prompt`` (dict agent -> prompt).
  - **Hand-Crafted** (58): ``groundtruth`` / ``is_corrected`` spellings, NO ``system_prompt``,
    history items ``{content, role}`` with the agent name embedded in ``role``
    (e.g. ``"Orchestrator (-> FileSurfer)"`` -> agent ``"Orchestrator"``).

Environment-ish steps (``Computer_terminal`` outputs, etc.) are flagged ``is_env`` so the
attribution layer resamples them with the environment simulator, not the agent surrogate.

No dataset content ships with this repo (no published license — local evaluation only);
``fetch_subset`` downloads JSONs from HuggingFace into a local cache dir at runtime.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

log = structlog.get_logger(__name__)

Subset = Literal["Algorithm-Generated", "Hand-Crafted"]

_HF_API = "https://huggingface.co/api/datasets/Kevin355/Who_and_When/tree/main"
_HF_RESOLVE = "https://huggingface.co/datasets/Kevin355/Who_and_When/resolve/main"

# Speakers that are environment output, not agent decisions (extend as observed).
_ENV_SPEAKERS = {"computer_terminal", "human"}


class LogStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    agent: str
    content: str
    is_env: bool


class WhoWhenInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    subset: Subset
    question: str
    ground_truth: str
    history: list[LogStep]
    mistake_agent: str
    mistake_step: int
    mistake_reason: str = ""
    system_prompts: dict[str, str] = Field(default_factory=dict)


def _agent_of(item: dict[str, Any]) -> str:
    name = item.get("name")
    if isinstance(name, str) and name:
        return name
    role = str(item.get("role", ""))
    # HC embeds the agent in role, possibly with a parenthetical: "Orchestrator (thought)".
    return re.sub(r"\s*\(.*\)$", "", role) or "unknown"


def _is_env(agent: str) -> bool:
    return agent.lower() in _ENV_SPEAKERS


def parse_instance(raw: dict[str, Any], *, instance_id: str, subset: Subset) -> WhoWhenInstance:
    history = [
        LogStep(
            index=i,
            agent=_agent_of(m),
            content=str(m.get("content", "")),
            is_env=_is_env(_agent_of(m)),
        )
        for i, m in enumerate(raw.get("history", []))
    ]
    if not history:
        raise ValueError(f"instance {instance_id}: empty history")
    mistake_step = int(raw["mistake_step"])  # stored as a string in the dataset
    system_prompt = raw.get("system_prompt") or {}
    return WhoWhenInstance(
        instance_id=instance_id,
        subset=subset,
        question=str(raw.get("question", "")),
        ground_truth=str(raw.get("ground_truth", raw.get("groundtruth", ""))),
        history=history,
        mistake_agent=str(raw["mistake_agent"]),
        mistake_step=mistake_step,
        mistake_reason=str(raw.get("mistake_reason", "")),
        system_prompts={str(k): str(v) for k, v in system_prompt.items()},
    )


def load_instance(path: Path, *, subset: Subset) -> WhoWhenInstance:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_instance(raw, instance_id=f"{subset}/{path.stem}", subset=subset)


def fetch_subset(
    subset: Subset, cache_dir: Path, *, limit: int | None = None
) -> list[WhoWhenInstance]:
    """Download (with local caching) and parse a subset. Network only on cache miss."""
    cache = cache_dir / subset
    cache.mkdir(parents=True, exist_ok=True)
    names = _list_remote(subset)
    if limit is not None:
        names = sorted(names, key=lambda n: int(Path(n).stem))[:limit]
    instances: list[WhoWhenInstance] = []
    for name in names:
        local = cache / name
        if not local.exists():
            url = f"{_HF_RESOLVE}/{urllib.parse.quote(f'Who&When/{subset}/{name}')}"
            log.info("fetching", url=url)
            with urllib.request.urlopen(url, timeout=60) as resp:
                local.write_bytes(resp.read())
        instances.append(load_instance(local, subset=subset))
    log.info("loaded subset", subset=subset, n=len(instances))
    return instances


def _list_remote(subset: Subset) -> list[str]:
    url = f"{_HF_API}/{urllib.parse.quote(f'Who&When/{subset}')}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        listing = json.loads(resp.read())
    return [Path(e["path"]).name for e in listing if e["path"].endswith(".json")]
