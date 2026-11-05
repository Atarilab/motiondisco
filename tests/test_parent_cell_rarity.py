"""cell_rarity parent selection: elites of rarely visited descriptor cells
are sampled more often than elites of crowded cells."""

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.parents import CellRaritySamplingStrategy, WeightedSamplingStrategy


def _program(pid: str, score: float, cell: str) -> Program:
    return Program(id=pid, code="x = 1\n", correct=True, combined_score=score,
                   generation=1, island_idx=0, public_metrics={"plan_patches": cell})


def _db(tmpdir: str) -> ProgramDatabase:
    cfg = DatabaseConfig(
        db_path=str(Path(tmpdir) / "db.sqlite"), num_islands=1, archive_size=50,
        archive_selection_strategy="descriptor", archive_descriptor_key="plan_patches",
        archive_min_score=0.0, parent_selection_lambda=2.0, parent_selection_cell_rarity=1.0,
    )
    return ProgramDatabase(config=cfg, embedding_model="")


def _counts(strategy_cls, db, draws=400):
    strategy = strategy_cls(db.cursor, db.conn, db.config, db.get, None, 0)
    return Counter(strategy.sample_parent().public_metrics["plan_patches"] for _ in range(draws))


def test_rare_cell_elite_is_sampled_far_more_often():
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            for i in range(10):  # crowded cell A: ten plans, one elite in the archive
                db.add(_program(f"a{i}", 0.5, "A"))
            db.add(_program("b0", 0.5, "B"))  # rare cell B: one plan
            weighted = _counts(WeightedSamplingStrategy, db)
            rarity = _counts(CellRaritySamplingStrategy, db)
        finally:
            db.close()
    # Same scores: weighted sampling is roughly even, rarity favours B ~5x.
    assert 0.3 < weighted["B"] / 400 < 0.7
    assert rarity["B"] > 3 * rarity["A"]
