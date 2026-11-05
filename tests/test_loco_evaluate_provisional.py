"""evaluate.py writes a provisional "KSO feasible, TO pending" record
before trajectory optimization, so a job killed during TO still yields a
usable result (the runner loads whatever metrics.json exists)."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import evaluate  # noqa: E402

SCENE = [
    {"id": 1, "label": "floor", "is_movable": False,
     "self_features": {"feat_1": 1}, "env_features": {"feat_1": 1}},
    {"id": 2, "label": "table", "is_movable": False,
     "self_features": {"feat_1": 1}, "env_features": {"feat_1": 2}},
    {"id": 3, "label": "box", "is_movable": True,
     "self_features": {f"feat_{i}": i for i in range(1, 7)},
     "env_features": {f"feat_{i}": i + 2 for i in range(1, 7)}},
]
PLAN = {
    "left_sole": [(1, 1), (1, 1), (1, 1)],
    "right_sole": [(1, 1), (), (1, 1)],
    "box": [(6, 2, 1), (), (6, 1, 1)],
}


def _setup(monkeypatch, tmp_path, solve_to, evaluate_ik=None):
    monkeypatch.setenv("SHINKA_SCORE_MODE", "quality")
    monkeypatch.delenv("SHINKA_DIVERSITY", raising=False)
    monkeypatch.delenv("SHINKA_DISABLE_TO", raising=False)
    task = SimpleNamespace(discrete_goal={"box": (6, 1, 1)}, terminal_goal=None,
                           scene=None, outcome_scorer=None, outcome_scorer_name=None)
    monkeypatch.setattr(evaluate, "load_task", lambda: task)
    stub = types.ModuleType("ik_eval")
    stub.evaluate_ik = evaluate_ik or (
        lambda plan, terminal_goal=None, on_full_failure=None: (0.0, {"obj_xs": {}}, None, "soft", False))
    monkeypatch.setitem(sys.modules, "ik_eval", stub)
    monkeypatch.setattr(evaluate, "solve_to", solve_to)
    results_dir = tmp_path / "gen_1" / "results"
    results_dir.mkdir(parents=True)
    return str(results_dir)


def test_killed_during_kso_diagnosis_leaves_the_full_plan_violations(monkeypatch, tmp_path):
    def evaluate_ik(plan, terminal_goal=None, on_full_failure=None):
        on_full_failure({"kind": "pending", "total_modes": 3, "last_feasible_modes": None,
                         "first_infeasible_modes": None, "kso_attempts": 3, "best_max_violation": 0.4,
                         "violation_details": []})
        raise SystemExit("job cancelled during the prefix walk")

    results_dir = _setup(monkeypatch, tmp_path, lambda *a, **k: None, evaluate_ik)
    with pytest.raises(SystemExit):
        evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)
    metrics = json.loads((Path(results_dir) / "metrics.json").read_text())
    assert metrics["public"]["ik_feasible"] is False
    assert metrics["public"]["to_status"] == "skipped"
    assert metrics["text_feedback"].startswith("KSO could not solve the 3-mode plan")
    assert "break-point search did not finish" in metrics["text_feedback"]
    assert metrics["private"]["ik_diagnosis"]["kind"] == "pending"
    (record,) = json.loads((tmp_path / "seen_plans.json").read_text()).values()
    assert record["provisional"] is True


def _incremental_result(**kw):
    base = dict(status=1, converged=False, cost=2.0, results={"obj_xs": {}}, num_modes=3, num_nodes=60,
                constraint_violation=1.0, max_constraint_violation=0.3, jerk_cost=0.0, violations=[], valid=False,
                physical=None, iterate="final", polish=None, fail_reason="", rescues=0, plan=None, repairs=[],
                kso_feasible=True, cached_modes=0, solved_modes=0, segments=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_incremental_evaluation_skips_the_full_kso_and_reads_the_window_verdict(monkeypatch, tmp_path):
    seen = {}

    def evaluate_ik(*a, **k):
        raise AssertionError("the full-plan KSO must not run in incremental mode")

    def solve_to(task, plan, **kwargs):
        import os

        seen["cache_dir"] = os.environ.get("SHINKA_SEGMENT_CACHE_DIR")
        return _incremental_result(
            kso_feasible=True, solved_modes=2, cached_modes=2,
            segments=[{"modes": [0, 2], "status": 0, "physical_pass": None, "extra": {"cached": True}},
                      {"modes": [2, 3], "status": 1, "physical_pass": False, "fail_reason": "", "extra": {}}])

    results_dir = _setup(monkeypatch, tmp_path, solve_to, evaluate_ik)
    monkeypatch.setattr(evaluate, "incremental_plan", lambda n: True)
    metrics = evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)
    assert seen["cache_dir"] == str(tmp_path / "segment_cache")
    assert metrics["public"]["ik_feasible"] is True and metrics["public"]["to_solved_modes"] == 2
    assert metrics["combined_score"] == pytest.approx(0.1 * (1 + 0.05 * 2 / 3))
    fb = metrics["text_feedback"]
    assert "Modes 0–1 are solved and cached: keep them exactly as they are and change the plan from mode 2 on" in fb
    assert "Modes 0–1 were reused from an earlier evaluation" in fb


def test_incremental_first_window_without_kinematic_solution_is_kso_infeasible(monkeypatch, tmp_path):
    def solve_to(task, plan, **kwargs):
        return _incremental_result(
            status=2, kso_feasible=False, solved_modes=0,
            segments=[{"modes": [0, 3], "status": 2, "physical_pass": None,
                       "fail_reason": "kso: no kinematic solution for modes [0,3)",
                       "extra": {"kso": {"max_violation": 0.4, "violations": [], "error": None}}}])

    results_dir = _setup(monkeypatch, tmp_path, solve_to, lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(evaluate, "incremental_plan", lambda n: True)
    metrics = evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)
    assert metrics["public"]["ik_feasible"] is False and metrics["combined_score"] == 0.0
    assert metrics["private"]["ik_diagnosis"]["kind"] == "window"
    assert metrics["text_feedback"].startswith("Solver feedback:\nKSO found no kinematic solution for modes 0–2 of 3 (the plan's first window).")


def test_killed_during_incremental_evaluation_leaves_a_window_record(monkeypatch, tmp_path):
    def killed(task, plan, **kwargs):
        raise SystemExit("job cancelled")

    results_dir = _setup(monkeypatch, tmp_path, killed, lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(evaluate, "incremental_plan", lambda n: True)
    with pytest.raises(SystemExit):
        evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)
    metrics = json.loads((Path(results_dir) / "metrics.json").read_text())
    assert metrics["public"]["to_status"] == "pending" and metrics["public"]["ik_feasible"] is None
    assert metrics["text_feedback"].startswith("The 3-mode plan is evaluated window by window")


def test_repaired_plan_is_the_evaluated_one(monkeypatch, tmp_path):
    repaired = {ch: seq[:1] + [seq[0], seq[0]] + seq[1:] for ch, seq in PLAN.items()}   # two modes inserted

    def solve_to(task, plan, **kwargs):
        return SimpleNamespace(status=1, converged=False, cost=2.0, results={"obj_xs": {}}, num_modes=5,
                               num_nodes=100, constraint_violation=1.0, max_constraint_violation=0.3,
                               jerk_cost=0.0, violations=[], valid=False, physical=None, iterate="final",
                               segments=[{"modes": [0, 3], "status": 0, "physical_pass": None, "extra": {}},
                                         {"modes": [3, 5], "status": 1, "physical_pass": False, "extra": {}}],
                               polish=None, fail_reason="", rescues=0, solved_modes=3, plan=repaired,
                               repairs=[{"segment": 1, "before_mode": 3, "before_mode_original": 1, "steps": 1,
                                         "modes_added": 2, "reason": "left_sole travels 1.90 m landing at mode 2"}])

    results_dir = _setup(monkeypatch, tmp_path, solve_to)
    metrics = evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)
    # Partial credit: 3 of 5 modes solved -> 0.1 * (1 + 0.05 * 0.6).
    assert metrics["combined_score"] == pytest.approx(0.1 * (1 + 0.05 * 0.6))
    assert metrics["public"]["num_modes"] == 5 and metrics["public"]["plan_repairs"] == 1
    assert metrics["public"]["to_solved_modes"] == 3
    fb = metrics["text_feedback"]
    assert "in segment 2 of 2 (modes 3–4; modes 0–2 were solved)" in fb
    assert "walk(plan, 1) inserted right before mode 1 of the proposed plan" in fb
    extra = json.loads((Path(results_dir) / "extra.json").read_text())
    assert len(extra["contact_mode_seq"]["left_sole"]) == 5
    assert len(extra["original_contact_mode_seq"]["left_sole"]) == 3
    assert extra["plan_repairs"][0]["steps"] == 1 and extra["to_solved_modes"] == 3
    seen = json.loads((tmp_path / "seen_plans.json").read_text())
    assert len(seen) == 2                     # the proposed and the repaired plan
    assert any(r["top_violation"].startswith("evaluated as the repaired form") for r in seen.values())


def test_killed_during_to_leaves_provisional_record(monkeypatch, tmp_path):
    def killed(task, plan, **kwargs):
        raise SystemExit("job cancelled")

    results_dir = _setup(monkeypatch, tmp_path, killed)
    with pytest.raises(SystemExit):
        evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)

    metrics = json.loads((Path(results_dir) / "metrics.json").read_text())
    assert metrics["combined_score"] == pytest.approx(0.1)  # KSO-feasible band
    assert metrics["public"]["ik_feasible"] is True
    assert metrics["public"]["to_status"] == "pending"
    assert metrics["text_feedback"].startswith("KSO found the 3-mode plan feasible")
    assert json.loads((Path(results_dir) / "correct.json").read_text())["correct"] is True
    seen = json.loads((tmp_path / "seen_plans.json").read_text())
    (record,) = seen.values()
    assert record["provisional"] is True and record["first_seen_gen"] == "gen_1"


def test_final_result_replaces_own_provisional_record(monkeypatch, tmp_path):
    calls = {"n": 0}

    def solve_to(task, plan, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SystemExit("job cancelled")
        return SimpleNamespace(status=0, converged=True, cost=2.0, results={"obj_xs": {}},
                               num_modes=3, num_nodes=60, constraint_violation=0.0,
                               max_constraint_violation=0.0, jerk_cost=0.0, violations=[])

    results_dir = _setup(monkeypatch, tmp_path, solve_to)
    with pytest.raises(SystemExit):
        evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)
    # A re-run of the same generation is not a duplicate of its own
    # provisional record and replaces it.
    metrics = evaluate.aggregate_planning_metrics([(PLAN, SCENE)], results_dir)
    assert "is_duplicate" not in metrics["public"]
    assert "to_status" not in metrics["public"]
    assert metrics["combined_score"] == pytest.approx(1.0)
    (record,) = json.loads((tmp_path / "seen_plans.json").read_text()).values()
    assert record["provisional"] is False
    assert record["combined_score"] == pytest.approx(1.0)
    # Another generation proposing the same plan IS a duplicate.
    other = tmp_path / "gen_2" / "results"
    other.mkdir(parents=True)
    dup = evaluate.aggregate_planning_metrics([(PLAN, SCENE)], str(other))
    assert dup["public"]["is_duplicate"] is True
