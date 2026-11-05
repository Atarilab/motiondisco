"""Descriptor (MAP-Elites) archive, public-metric archive criteria, and
per-island system-message suffixes."""

import tempfile
from pathlib import Path

from shinka.core.sampler import PromptSampler
from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _program(pid: str, score: float, cell: str, novelty: float = 0.0) -> Program:
    return Program(
        id=pid,
        code=f"def f():\n    return {score}\n",
        correct=True,
        combined_score=score,
        generation=1,
        island_idx=0,
        public_metrics={"plan_patches": cell, "plan_novelty": novelty},
    )


def _archive_ids(db: ProgramDatabase) -> set:
    db.cursor.execute("SELECT program_id FROM archive")
    return {row[0] for row in db.cursor.fetchall()}


def _db(tmpdir: str, **overrides) -> ProgramDatabase:
    cfg = DatabaseConfig(
        db_path=str(Path(tmpdir) / "archive.db"),
        num_islands=1,
        archive_size=2,
        archive_selection_strategy="descriptor",
        archive_descriptor_key="plan_patches",
        **overrides,
    )
    return ProgramDatabase(config=cfg, embedding_model="")


def test_descriptor_archive_keeps_one_program_per_cell():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            db.add(_program("a1", 0.5, "A"))
            db.add(_program("a2", 0.9, "A"))  # better in the same cell
            db.add(_program("a3", 0.1, "A"))  # worse in the same cell
            assert _archive_ids(db) == {"a2"}
            db.add(_program("b1", 0.2, "B"))  # new cell, room left
            assert _archive_ids(db) == {"a2", "b1"}
        finally:
            db.close()


def test_descriptor_archive_new_cell_competes_with_global_worst_when_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            db.add(_program("a", 0.9, "A"))
            db.add(_program("b", 0.2, "B"))
            db.add(_program("c", 0.1, "C"))  # worse than every occupant
            assert _archive_ids(db) == {"a", "b"}
            db.add(_program("d", 0.5, "D"))  # beats the worst (b)
            assert _archive_ids(db) == {"a", "d"}
        finally:
            db.close()


def test_archive_criteria_read_numeric_public_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            prog = _program("p", 0.4, "A", novelty=0.75)
            assert db._get_criterion_value(prog, "plan_novelty") == 0.75
            assert db._get_criterion_value(prog, "missing_metric") == 0.0
        finally:
            db.close()


def test_higher_score_replaces_cell_elite_despite_zero_novelty():
    """Inside a cell only the score decides: later members of a cell always
    have zero patch novelty, so a novelty criterion would keep the first
    finder forever (rank saturation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(
            tmpdir,
            archive_criteria={"combined_score": 1.0, "plan_novelty": 0.5},
        )
        try:
            db.add(_program("a", 0.5, "A", novelty=0.8))
            db.add(_program("b", 0.2, "B", novelty=0.3))
            db.add(_program("a2", 0.9, "A", novelty=0.0))  # better plan, same cell
            assert _archive_ids(db) == {"a2", "b"}
            db.add(_program("a3", 0.9, "A", novelty=0.0))  # equal score: newer wins
            assert _archive_ids(db) == {"a3", "b"}
        finally:
            db.close()


def test_novelty_decides_cross_cell_eviction_when_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(
            tmpdir,
            archive_criteria={"combined_score": 1.0, "plan_novelty": 0.5},
        )
        try:
            db.add(_program("x", 0.5, "A", novelty=0.1))
            db.add(_program("y", 0.5, "B", novelty=0.9))
            db.add(_program("z", 0.5, "C", novelty=0.9))  # full: evicts the least novel
            assert _archive_ids(db) == {"y", "z"}
        finally:
            db.close()


def test_archive_min_score_excludes_zero_score_programs():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir, archive_min_score=0.0)
        try:
            db.add(_program("zero", 0.0, "A"))
            db.add(_program("pos", 0.3, "B"))
            assert _archive_ids(db) == {"pos"}
        finally:
            db.close()


def test_initial_copies_do_not_enter_the_archive_below_min_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = DatabaseConfig(
            db_path=str(Path(tmpdir) / "archive.db"),
            num_islands=3,
            archive_size=2,
            archive_selection_strategy="descriptor",
            archive_descriptor_key="plan_patches",
            archive_min_score=0.0,
        )
        db = ProgramDatabase(config=cfg, embedding_model="")
        try:
            init = _program("init", 0.0, "I")
            init.generation = 0
            db.add(init)  # copied to islands 1-2 by the island manager
            db.cursor.execute("SELECT COUNT(*) FROM programs")
            assert db.cursor.fetchone()[0] == 3
            assert _archive_ids(db) == set()
        finally:
            db.close()


def test_descriptor_and_criteria_fall_back_to_private_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            prog = _program("p", 0.4, "A", novelty=0.75)
            prog.public_metrics = {}
            prog.private_metrics = {"plan_patches": "A", "plan_novelty": 0.75}
            assert db._descriptor_cell(prog) == "A"
            assert db._get_criterion_value(prog, "plan_novelty") == 0.75
        finally:
            db.close()


def test_island_suffix_appended_to_mutation_system_message():
    sampler = PromptSampler(
        task_sys_msg="TASK",
        patch_types=["full"],
        patch_type_probs=[1.0],
        task_sys_msg_island_suffixes=["", "HINT-1"],
    )
    parent = _program("parent", 0.5, "A")

    parent.island_idx = 1
    sys_msg, _user, _patch = sampler.sample(parent, [], [])
    assert sys_msg.startswith("TASK\n\nHINT-1")

    for idx in (0, 5):  # empty suffix, island beyond the list
        parent.island_idx = idx
        sys_msg, _user, _patch = sampler.sample(parent, [], [])
        assert "HINT-1" not in sys_msg
        assert sys_msg.startswith("TASK")


def test_full_patch_variants_restrict_rewrite_instructions():
    from shinka.prompts.prompts_full import FULL_SYS_FORMATS, FULL_SYS_FORMAT_NAMES

    sampler = PromptSampler(
        task_sys_msg="TASK",
        patch_types=["full"],
        patch_type_probs=[1.0],
        full_patch_variants=["different_algorithm"],
    )
    expected = FULL_SYS_FORMATS[FULL_SYS_FORMAT_NAMES.index("different_algorithm")]
    for _ in range(5):
        sys_msg, _user, _patch = sampler.sample(_program("p", 0.5, "A"), [], [])
        assert sys_msg.endswith(expected)


def test_full_patch_variants_reject_unknown_names():
    import pytest

    with pytest.raises(ValueError):
        PromptSampler(patch_types=["full"], patch_type_probs=[1.0],
                      full_patch_variants=["no_such_variant"])


def test_full_patch_variant_texts_override_rewrite_instructions():
    import pytest

    custom = ["Alpha <CODE> {language}", "Beta <CODE> {language}"]
    sampler = PromptSampler(
        task_sys_msg="TASK",
        patch_types=["full"],
        patch_type_probs=[1.0],
        full_patch_variant_texts=custom,
    )
    assert sampler.full_sys_formats == custom
    with pytest.raises(ValueError):
        PromptSampler(task_sys_msg="TASK", patch_types=["full"], patch_type_probs=[1.0],
                      full_patch_variant_texts=["no delimiters here"])
    with pytest.raises(ValueError):
        PromptSampler(task_sys_msg="TASK", patch_types=["full"], patch_type_probs=[1.0],
                      full_patch_variant_texts=custom, full_patch_variants=["default"])
