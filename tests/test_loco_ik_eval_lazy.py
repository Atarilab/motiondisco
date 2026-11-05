"""KSO feasibility in the loco_manipulation example: lazy solver pools per
goal kind, full-plan solves with retries, goal-miss check, bisection
prefix diagnosis and the repair probe (all solver calls stubbed)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import ik_eval  # noqa: E402


def _plan(n_modes: int) -> dict:
    return {"left_sole": [(1, 1)] * n_modes, "right_sole": [(1, 1)] * n_modes,
            "box": [(6, 1, 1)] * n_modes}


def _stub_pool(monkeypatch, feasible_up_to: int, fail_first: int = 0,
               terminal_goal=None, solve=None, probes: int = 0, bucket: int = 0,
               budget: float = 1500.0):
    """Stub solvers: prefixes up to ``feasible_up_to`` modes succeed,
    except that the first ``fail_first`` solves fail (a noisy solver).
    Exact horizon lengths unless ``bucket`` is set."""
    built = []
    calls = {"n": 0}
    task = SimpleNamespace(scene=None, terminal_goal=terminal_goal, outcome_scorer=None)
    monkeypatch.setattr(ik_eval, "load_task", lambda: task)
    monkeypatch.setattr(ik_eval, "to_faro_plan", lambda _scene, plan: plan)
    monkeypatch.setattr(ik_eval, "_pools", {k: None for k in ik_eval.KINDS})
    monkeypatch.setattr(ik_eval, "_failed_attempts", {})
    monkeypatch.setattr(ik_eval, "KSO_ATTEMPTS", 3)
    monkeypatch.setattr(ik_eval, "PREFIX_ATTEMPTS", 4)
    monkeypatch.setattr(ik_eval, "PROBES", probes)
    monkeypatch.setattr(ik_eval, "WALK_MODE", "bisect")
    monkeypatch.setattr(ik_eval, "KSO_BUCKET", bucket)
    monkeypatch.setattr(ik_eval, "EVAL_BUDGET_S", budget)
    monkeypatch.setattr(ik_eval, "_violations_of", lambda n, kind, plan: None)

    def make(_task, goal, n):
        built.append(n)
        return SimpleNamespace(n=n, goal=goal, get_results=lambda: {"n": n, "obj_xs": {"box": [[0.0, 0.0, 0.0]]}})

    def default_solve(kso_task, cms, _scorer):
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return False
        return 0.0 if kso_task.n <= feasible_up_to else False

    monkeypatch.setattr(ik_eval, "_make_kso_task", make)
    monkeypatch.setattr(ik_eval, "_solve_kso", solve or default_solve)
    return built


def test_feasible_plan_solves_only_its_own_length(monkeypatch):
    built = _stub_pool(monkeypatch, feasible_up_to=99)
    cost, results, diag, mode, _ = ik_eval.evaluate_ik(_plan(6))
    assert (cost, results["n"], diag, mode) == (0.0, 6, None, "soft")
    assert built == [6]
    ik_eval.evaluate_ik(_plan(6))  # same length reuses the built solver
    assert built == [6]


def test_noisy_solver_succeeds_on_retry_with_fresh_solver(monkeypatch):
    built = _stub_pool(monkeypatch, feasible_up_to=99, fail_first=2)
    cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(_plan(20))
    assert cost == 0.0 and diag is None
    assert built == [20, 20, 20]  # first solver + two fresh rebuilds


def test_infeasible_plan_is_located_by_bisection(monkeypatch):
    built = _stub_pool(monkeypatch, feasible_up_to=20)
    cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(_plan(34))
    assert cost is False
    assert diag["kind"] == "prefix"
    assert diag["first_infeasible_modes"] == 21
    assert diag["last_feasible_modes"] == 20
    # 3 full-plan attempts, prefix 2, then bisection 18 / 26 / 22 / 20 / 21.
    # Infeasible prefixes get 4 tries (first solver + 3 fresh rebuilds).
    assert built == [34, 34, 34, 2, 18, 26, 26, 26, 26, 22, 22, 22, 22, 20, 21, 21, 21, 21]
    assert [t["modes"] for t in diag["prefix_trials"]] == [2, 18, 26, 22, 20, 21]


def test_noisy_prefix_still_finds_the_boundary(monkeypatch):
    # A feasible prefix fails its first two tries but passes within 4.
    built = _stub_pool(monkeypatch, feasible_up_to=5, fail_first=5)
    _cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(_plan(8))
    assert (diag["last_feasible_modes"], diag["first_infeasible_modes"]) == (5, 6)
    assert built[:3] == [8, 8, 8]


def test_hard_goal_checks_full_length_first(monkeypatch):
    goal = SimpleNamespace(kso="hard", is_empty=lambda: False, objects={}, robots={})
    built = _stub_pool(monkeypatch, feasible_up_to=99, terminal_goal=goal)
    cost, _results, _diag, mode, hard_failed = ik_eval.evaluate_ik(_plan(9), terminal_goal=goal)
    assert (cost, mode, hard_failed) == (0.0, "hard", False)
    assert built == [9]


def test_goal_missed_when_only_the_goal_free_solve_works(monkeypatch):
    target = SimpleNamespace(value=[1.0, 0.0, 0.0], kso_tolerance=SimpleNamespace(position=0.05))
    goal = SimpleNamespace(kso="hard", is_empty=lambda: False, objects={"box": target}, robots={})

    def solve(kso_task, _cms, _scorer):
        return False if kso_task.goal is not None else 0.0

    built = _stub_pool(monkeypatch, feasible_up_to=99, terminal_goal=goal, solve=solve)
    cost, results, diag, mode, hard_failed = ik_eval.evaluate_ik(_plan(7), terminal_goal=goal)
    assert cost is False and hard_failed and mode == "soft"
    assert diag["kind"] == "goal_missed"
    assert results is not None  # the goal-free motion is kept for replay
    res = diag["goal_residuals"]["box"]
    assert res["distance"] == 1.0 and res["tolerance"] == 0.05
    # 3 hard, 3 soft, then one goal-free solve; no prefix walk.
    assert built == [7] * 7


def test_repair_probe_reports_the_edit_that_fixed_the_plan(monkeypatch):
    plan = _plan(10)
    plan["left_ee"] = [()] * 3 + [(3, 1)] * 7  # first grasp at mode 3
    plan["right_ee"] = [()] * 10

    def solve(kso_task, cms, _scorer):
        n = len(next(iter(cms.values())))
        return False if n == 10 else 0.0  # the original plan fails, everything else works

    _stub_pool(monkeypatch, feasible_up_to=99, solve=solve, probes=2)
    cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(plan)
    assert cost is False
    assert diag["first_infeasible_modes"] == 10 and diag["last_feasible_modes"] == 9
    probe = diag["probe"]
    assert probe["fixed_by"].startswith("walk(plan, 2) inserted right before mode 9")
    assert probe["tried"][0]["modes"] == 13 and probe["tried"][0]["result"] == "feasible"
    assert "plan" not in probe


def test_schedule_error_is_not_retried(monkeypatch):
    def solve(_kso_task, _cms, _scorer):
        raise ValueError("invalid contact schedule: Contact union for ('ee', 'robot', 'left_ee') "
                         "has conflicting active geometries")

    built = _stub_pool(monkeypatch, feasible_up_to=99, solve=solve)
    cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(_plan(5))
    assert cost is False
    assert diag["kind"] == "schedule_error"
    assert diag["solver_error"].startswith("ValueError: invalid contact schedule")
    assert built == [5]  # one attempt, no fresh rebuilds, no walk


def test_bucket_helpers():
    import numpy as np

    assert ik_eval.pad_plan({"a": [(1,), (2,)]}, 4, "end") == {"a": [(1,), (2,), (2,), (2,)]}
    assert ik_eval.pad_plan({"a": [(1,), (2,)]}, 4, "start") == {"a": [(1,), (1,), (1,), (2,)]}
    assert ik_eval.pad_plan({"a": [(1,), (2,)]}, 2) == {"a": [(1,), (2,)]}
    with pytest.raises(ValueError):
        ik_eval.pad_plan({"a": []}, 3)
    with pytest.raises(ValueError):
        ik_eval.pad_plan({"a": [(1,)]}, 3, "middle")
    goal = {"robot_xs": {"r": np.arange(11)[:, None]}, "obj_xs": {"box": np.arange(11)[:, None]}}
    none = {"robot_xs": {"r": np.arange(10)[:, None]}}
    # End padding, goal solve: one state per mode plus the endpoint; keep 0..n-1 and the endpoint.
    out = ik_eval.trim_results(goal, 6, 10, "end")
    assert out["robot_xs"]["r"][:, 0].tolist() == [0, 1, 2, 3, 4, 5, 10]
    assert out["obj_xs"]["box"][:, 0].tolist() == [0, 1, 2, 3, 4, 5, 10]
    # End padding, goal-free solve: one state per mode; keep 0..n-1.
    assert ik_eval.trim_results(none, 6, 10, "end")["robot_xs"]["r"][:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    # Start padding: the first length-n states hold the initial state and are dropped.
    assert ik_eval.trim_results(goal, 6, 10, "start")["robot_xs"]["r"][:, 0].tolist() == [4, 5, 6, 7, 8, 9, 10]
    assert ik_eval.trim_results(none, 6, 10, "start")["robot_xs"]["r"][:, 0].tolist() == [4, 5, 6, 7, 8, 9]
    assert ik_eval.trim_results(none, 10, 10) is none and ik_eval.trim_results(None, 6, 10) is None
    # Violations of a start-padded solve come back in the plan's own modes.
    from violation_feedback import Violation

    vs = [Violation("kso", 9, "box", "gap", 0.4, "m"), Violation("kso", 2, "box", "gap", 0.4, "m"),
          Violation("kso", None, "box", "gap", 0.4, "m")]
    assert [v.mode for v in ik_eval.shift_modes(vs, 4)] == [5, 0, None]
    assert ik_eval.shift_modes(vs, 0) == vs


def test_bucket_length_rounds_up_within_the_cap(monkeypatch):
    monkeypatch.setattr(ik_eval, "KSO_BUCKET", 10)
    monkeypatch.setattr(ik_eval, "MAX_NUM_NODES", 45)
    assert [ik_eval.bucket_length(n) for n in (2, 10, 11, 39, 41, 45)] == [10, 10, 20, 40, 45, 45]
    monkeypatch.setattr(ik_eval, "KSO_BUCKET", 0)
    assert ik_eval.bucket_length(11) == 11


def test_bucketed_plans_share_one_padded_solver(monkeypatch):
    seen = []

    def solve(kso_task, cms, _scorer):
        seen.append((kso_task.n, len(next(iter(cms.values()))), cms["left_sole"][-1]))
        return 0.0

    built = _stub_pool(monkeypatch, feasible_up_to=99, solve=solve, bucket=10)
    monkeypatch.setattr(ik_eval, "KSO_PAD", "end")
    plan = _plan(6)
    plan["left_sole"][-1] = (1, 2)              # a distinctive last mode to see repeated
    cost, results, diag, _mode, _ = ik_eval.evaluate_ik(plan)
    ik_eval.evaluate_ik(_plan(8))
    assert cost == 0.0 and diag is None
    assert built == [10]                        # one compile for both lengths
    assert seen == [(10, 10, (1, 2)), (10, 10, (1, 1))]   # both solved on the padded 10-mode plan
    assert results["n"] == 10                   # the stub's results carry no state arrays to trim


def test_full_failure_callback_runs_before_the_walk_and_the_walk_honours_the_deadline(monkeypatch):
    built = _stub_pool(monkeypatch, feasible_up_to=20, budget=-1.0)   # budget already spent
    pending = []
    cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(_plan(34), on_full_failure=pending.append)
    assert cost is False
    assert pending[0]["kind"] == "pending" and pending[0]["total_modes"] == 34
    assert diag["kind"] == "unknown" and diag["timed_out"]
    assert built == [34, 34, 34]                # the three full-plan attempts, no prefix solves


def test_bisection_stops_at_the_deadline_with_a_valid_bracket(monkeypatch):
    import itertools

    built = _stub_pool(monkeypatch, feasible_up_to=20)
    clock = itertools.count()
    # Each call to time.time() advances one "second"; the budget covers the
    # full-plan attempts and the first two prefix solves only.
    monkeypatch.setattr(ik_eval.time, "time", lambda: float(next(clock)))
    monkeypatch.setattr(ik_eval, "EVAL_BUDGET_S", 3.0)
    _cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(_plan(34))
    assert diag["kind"] == "prefix" and diag["timed_out"]
    assert diag["last_feasible_modes"] in (2, 18) and diag["first_infeasible_modes"] in (34, 26)
    assert diag["last_feasible_modes"] <= 20 < diag["first_infeasible_modes"]
    assert len(built) < 18                      # the full bisection builds 18 solvers


def test_failed_plan_reports_pooled_violations_of_every_attempt(monkeypatch):
    from violation_feedback import Violation

    _stub_pool(monkeypatch, feasible_up_to=3)
    calls = []

    def snapshot(n, kind, plan):
        calls.append((n, kind))
        return {"max_violation": 0.4 if len(calls) < 3 else 0.02,
                "violations": [Violation("kso", n - 1, "box", "gap", 0.4, "m")]}

    monkeypatch.setattr(ik_eval, "_violations_of", snapshot)
    _cost, _results, diag, _mode, _ = ik_eval.evaluate_ik(_plan(12))
    assert diag["kso_attempts"] == 3
    assert diag["best_max_violation"] == 0.02
    details = diag["violation_details"]
    assert len(details) == 3 and all(d["attempts"] == 3 for d in details)
    assert diag["prefix_tries"] == 4 and len(diag["prefix_violation_details"]) == 4
