"""Persist trajectory trees: JSON trace files + a SQLite index (PLAN.md s5.4 / tech stack).

Counterfactual branches form a *tree* (each branch carries ``parent_id`` / ``branched_at_step``
/ ``intervention_id``), not flat rows. The JSON files hold the full trajectories; the SQLite
index makes lineage and outcome queries cheap (find a run's children, filter by outcome).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import structlog

from car.schemas.trajectory import Trajectory

log = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id    TEXT PRIMARY KEY,
    parent_id        TEXT,
    branched_at_step INTEGER,
    intervention_id  TEXT,
    n_steps          INTEGER NOT NULL,
    outcome_label    TEXT,
    outcome_score    REAL,
    path             TEXT NOT NULL,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parent ON trajectories(parent_id);
CREATE INDEX IF NOT EXISTS idx_outcome ON trajectories(outcome_label);
"""


class TrajectoryStore:
    """A file+SQLite store for trajectory trees."""

    def __init__(self, db_path: str | Path = "./data/car.db", traces_dir: str | Path | None = None):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._traces_dir = Path(traces_dir) if traces_dir else self._db_path.parent / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, traj: Trajectory) -> Path:
        path = self._traces_dir / f"{traj.trajectory_id}.json"
        path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
        self._conn.execute(
            """
            INSERT INTO trajectories
                (trajectory_id, parent_id, branched_at_step, intervention_id, n_steps,
                 outcome_label, outcome_score, path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trajectory_id) DO UPDATE SET
                parent_id=excluded.parent_id,
                branched_at_step=excluded.branched_at_step,
                intervention_id=excluded.intervention_id,
                n_steps=excluded.n_steps,
                outcome_label=excluded.outcome_label,
                outcome_score=excluded.outcome_score,
                path=excluded.path
            """,
            (
                traj.trajectory_id,
                traj.parent_id,
                traj.branched_at_step,
                traj.intervention_id,
                len(traj.steps),
                traj.outcome.label if traj.outcome else None,
                traj.outcome.score if traj.outcome else None,
                str(path),
                time.time(),
            ),
        )
        self._conn.commit()
        log.info("saved trajectory", trajectory_id=traj.trajectory_id, path=str(path))
        return path

    def load(self, trajectory_id: str) -> Trajectory:
        row = self._conn.execute(
            "SELECT path FROM trajectories WHERE trajectory_id = ?", (trajectory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"trajectory {trajectory_id!r} not in store")
        return Trajectory.model_validate_json(Path(row["path"]).read_text(encoding="utf-8"))

    def children(self, parent_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT trajectory_id FROM trajectories WHERE parent_id = ? ORDER BY created_at",
            (parent_id,),
        ).fetchall()
        return [r["trajectory_id"] for r in rows]

    def descendants(self, root_id: str) -> list[str]:
        """All trajectory ids in the tree rooted at ``root_id`` (excluding the root)."""
        out: list[str] = []
        frontier = [root_id]
        while frontier:
            current = frontier.pop()
            kids = self.children(current)
            out.extend(kids)
            frontier.extend(kids)
        return out

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TrajectoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
