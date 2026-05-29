"""Trajectory-tree persistence: round-trip and lineage queries."""

from __future__ import annotations

from pathlib import Path

from car.schemas.trajectory import Action, State, Step, Trajectory
from car.store.store import TrajectoryStore


def _traj(tid: str, parent: str | None = None) -> Trajectory:
    state = State(
        system_prompt="s",
        tool_schemas=[],
        model="synthetic:test",
        provider="synthetic",
        sampling={},
        messages=[{"role": "user", "content": "hi"}],
    )
    return Trajectory(
        trajectory_id=tid,
        parent_id=parent,
        branched_at_step=0 if parent else None,
        steps=[
            Step(
                index=0,
                state_before=state,
                action=Action(kind="final", text="bye", raw={}),
                observation=None,
            )
        ],
        final_output="bye",
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    with TrajectoryStore(db_path=tmp_path / "car.db") as store:
        original = _traj("root")
        store.save(original)
        loaded = store.load("root")
        assert loaded == original


def test_tree_lineage(tmp_path: Path) -> None:
    with TrajectoryStore(db_path=tmp_path / "car.db") as store:
        store.save(_traj("root"))
        store.save(_traj("childA", parent="root"))
        store.save(_traj("childB", parent="root"))
        store.save(_traj("grandchild", parent="childA"))

        assert set(store.children("root")) == {"childA", "childB"}
        assert set(store.descendants("root")) == {"childA", "childB", "grandchild"}
