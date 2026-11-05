"""Segmented TO with static junctions (loco_manipulation/to_static.py):
weight and cut bookkeeping, junction stitching, the worker pool, and the
whole pipeline with stubbed static IK and TO solves (no faro)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import to_static  # noqa: E402

NQ, NV = 8, 7           # toy robot: base pose (7) + 1 joint
DIM = NQ + NV + 3 + 6
F, HL, BOX = (1, 1), (9, 1), (6, 1, 1)
# 0-3 walk | 4 grasp | 5-8 carry | 9 release | 10-11 stand; double support at 2, 4, 6, 8, 9-11
ROWS = [(F, F, (), BOX), (F, (), (), BOX), (F, F, (), BOX), ((), F, (), BOX), (F, F, HL, BOX), (F, (), HL, ()),
        (F, F, HL, ()), ((), F, HL, ()), (F, F, HL, ()), (F, F, (), BOX), (F, F, (), BOX), (F, F, (), BOX)]
PLAN = {ch: [row[i] for row in ROWS] for i, ch in enumerate(("left_sole", "right_sole", "left_ee", "box"))}


def _row(x=0.0):
    r = np.zeros(DIM)
    r[0], r[6] = x, 1.0
    return r


def _results(n, x0, x1):
    """n intervals moving the base from x0 to x1."""
    xs = np.tile(_row(), (n + 1, 1))
    xs[:, 0] = np.linspace(x0, x1, n + 1)
    box = np.zeros((n + 1, 13))
    box[:, 6] = 1.0
    return {"robot_xs": {"r": xs}, "obj_xs": {"box": box}, "vdot_rob": {"r": np.zeros((n, NV))},
            "rob_Ws": {"r": {"left_sole": np.ones((n, 6))}}, "obj_Vdots": {"box": np.zeros((n, 6))},
            "obj_Ws": {"box": np.zeros((n, 6))}, "t_scale": np.ones(n), "step_ts": np.full(n, 0.05),
            "best_iterate": {"robot_xs": "dropped"}}


def test_rest_weights_scale_only_the_terminal_velocity_blocks():
    we = np.arange(1, 2 * NV + 10, dtype=float)
    w = {"robots": {"r": {"state_cost_We": we.copy(), "state_cost_W": np.ones(3)}},
         "object_state_cost_We": np.ones(12)}
    to_static.rest_weights(w, {"r": NV}, 10.0)
    out = w["robots"]["r"]["state_cost_We"]
    assert np.array_equal(out[:NV], we[:NV]) and np.array_equal(out[2 * NV:], we[2 * NV:])
    assert np.array_equal(out[NV:2 * NV], 10 * we[NV:2 * NV])
    assert np.array_equal(w["object_state_cost_We"], np.r_[np.ones(6), 10 * np.ones(6)])
    assert np.array_equal(w["robots"]["r"]["state_cost_W"], np.ones(3))
    with pytest.raises(ValueError):
        to_static.rest_weights({"robots": {"r": {"state_cost_We": np.ones(5)}}}, {"r": NV}, 10.0)


def test_merge_cuts_drops_interior_cuts_only():
    assert to_static.merge_cuts([0, 4, 8, 12], [8]) == [0, 8, 12]
    assert to_static.merge_cuts([0, 4, 8, 12], []) == [0, 12]
    assert to_static.merge_cuts([0, 12], [5]) == [0, 12]


def test_keyframes_need_the_robot_rows():
    assert to_static.keyframes(None, "r") is None and to_static.keyframes({}, "r") is None
    k = to_static.keyframes({"robot_xs": {"r": np.zeros((3, NQ))}, "obj_xs": {"box": np.zeros((3, 7))}}, "r")
    assert k["robot"].shape == (3, NQ) and k["objects"]["box"].shape == (3, 7)


def test_junction_stitch_replaces_each_end_with_the_next_start():
    a, b = _results(4, 0.0, 0.97), _results(6, 1.0, 2.0)     # a ends 3 cm short of b's start
    with pytest.raises(ValueError):
        to_static.ts.stitch_results([a, b])
    s = to_static.junction_stitch([a, b], "r")
    assert s["robot_xs"]["r"].shape == (11, DIM) and s["t_scale"].shape == (10,)
    assert s["robot_xs"]["r"][4, 0] == pytest.approx(1.0) and s["robot_xs"]["r"][3, 0] == pytest.approx(0.97 * 3 / 4)
    assert a["robot_xs"]["r"][-1, 0] == pytest.approx(0.97)   # inputs untouched


def _square(x):
    if x < 0:
        raise ValueError("negative")
    return x * x


@pytest.mark.parametrize("workers", [1, 3])
def test_run_jobs_keeps_order_and_returns_exceptions(workers):
    out = to_static.run_jobs(_square, [(3,), (-1,), (2,), (5,)], workers)
    assert out[0] == 9 and out[2] == 4 and out[3] == 25
    assert isinstance(out[1], ValueError)


def _stub(monkeypatch, pose_status=None, pose_shift=None):
    """Stub faro: KSO keyframe k and its static pose put the base at x = k
    (plus ``pose_shift``); a segment's TO moves the base from its start to
    its end pose."""
    to_eval = pytest.importorskip("to_eval")
    pose_status, pose_shift = pose_status or {}, pose_shift or {}

    def static_pose(task, faro_plan, mode, guess):
        q = np.zeros(NQ)
        q[0], q[6] = float(mode) + pose_shift.get(mode, 0.0), 1.0
        return (q, {"box": np.r_[np.zeros(6), 1.0]}), pose_status.get(mode, 0)

    def build(task, plan, **kw):
        n = len(plan["left_sole"])
        return SimpleNamespace(to_task=SimpleNamespace(collision_index=None), task=task, plan=plan,
                               num_modes=n, num_nodes=n * 20, build_time_s=0.0, nodes_per_mode=20,
                               end=task.goal, suffix=kw.get("name_suffix"), weight_fn=kw.get("weight_fn"))

    def solve(prob, start=None, **kw):
        x0 = 0.0 if start is None else start.robot_x[0]
        x1 = prob.end if prob.end is not None else x0 + prob.num_modes
        status = 1 if (x0, x1) == (4.0, 8.0) else 0
        return to_eval._Solved(status=status, fail_reason="", rescues=0, outer_iters=5, best_prim=0.01,
                               results=_results(prob.num_nodes, x0, x1 - 0.01), physical=None, iterate="final",
                               constraint_violation=1.0, max_violation=0.2 if status else 0.02,
                               rows=[{"node": 1}], solve_time_s=0.0)

    def segment_task(task, start, goal):
        return SimpleNamespace(goal=None if goal is None else goal["x"])

    monkeypatch.setattr(to_static, "static_pose", static_pose)
    monkeypatch.setattr(to_static, "junction_goal", lambda task, pose: {"x": float(pose[0][0])})
    monkeypatch.setattr(to_static, "rest_start", lambda task, pose: to_static.ts.SegmentStart(
        robot_x=_row(float(pose[0][0])), obj_x={}))
    monkeypatch.setattr(to_static, "_seed_codegen", lambda task: None)
    monkeypatch.setattr(to_eval, "_build_problem", build)
    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    monkeypatch.setattr(to_eval, "_segment_task", segment_task)
    monkeypatch.setattr(to_eval, "_translate_rows", lambda rows, *a: [{"node": r["node"]} for r in rows])
    monkeypatch.setattr(to_eval, "TO_PHYSICAL_CHECK", False)
    import _task_loader
    monkeypatch.setattr(_task_loader, "to_faro_plan", lambda scene, plan: {})
    robot = SimpleNamespace(nq=NQ, nv=NV)
    task = SimpleNamespace(robot_id="r", terminal_goal=None, scene=SimpleNamespace(robots={"r": robot}, env_objects={}))
    keys = np.tile(_row()[:NQ], (13, 1))
    keys[:, 0] = np.arange(13)
    kso = {"robot_xs": {"r": keys}, "obj_xs": {"box": np.zeros((13, 7))}}
    return to_eval, task, kso


@pytest.mark.parametrize("workers", [1, 2])
def test_independent_segments_run_from_static_pose_to_static_pose(monkeypatch, workers):
    to_eval, task, kso = _stub(monkeypatch)
    sol = to_static.solve(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0, weight_scale=None,
                          kso_results=kso, workers=workers)
    assert [s["modes"] for s in sol.segments] == [[0, 4], [4, 8], [8, 12]]
    assert [s["goal"] for s in sol.segments] == ["static", "static", "task"]
    assert [s["status"] for s in sol.segments] == [0, 1, 0] and sol.status == 1 and not sol.converged
    xs = sol.results["robot_xs"]["r"]
    assert xs.shape == (12 * 20 + 1, DIM) and "best_iterate" not in sol.results
    # Segment 2 starts at the static pose of mode 8 although segment 1 failed, and each
    # segment's last row is the next one's start (ends were 1 cm short of it).
    assert xs[80, 0] == pytest.approx(4.0) and xs[160, 0] == pytest.approx(8.0) and xs[-1, 0] == pytest.approx(11.99)
    assert [v["node"] for v in sol.violations] == [1, 81, 161]
    assert sol.segments[1]["extra"]["start_pose"]["moved"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [{"pose_status": {4: 1}}, {"pose_shift": {4: 0.5}}])
def test_a_failed_or_far_static_pose_merges_its_segments(monkeypatch, bad):
    to_eval, task, kso = _stub(monkeypatch, **bad)
    sol = to_static.solve(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0, weight_scale=None,
                          kso_results=kso)
    assert [s["modes"] for s in sol.segments] == [[0, 8], [8, 12]] and sol.converged


def test_a_close_static_pose_is_kept(monkeypatch):
    to_eval, task, kso = _stub(monkeypatch, pose_shift={4: 0.2})
    sol = to_static.solve(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0, weight_scale=None,
                          kso_results=kso)
    assert [s["modes"] for s in sol.segments] == [[0, 4], [4, 8], [8, 12]]
    assert sol.segments[1]["extra"]["start_pose"]["moved"] == pytest.approx(0.2)


def _parallel_stub(monkeypatch, shift_first=0.3):
    """Stub for solve_parallel: static pose k at base x = k (as the KSO
    keyframe); a window's solve moves the base from its start x by its
    mode count, the first window ``shift_first`` further, so the later
    windows' static starts no longer match and get re-solved."""
    to_eval, task, kso = _stub(monkeypatch)
    calls = []

    def solve(prob, start=None, guess=None, **kw):
        x0 = 0.0 if start is None else float(start.robot_x[0])
        if guess is not None:
            assert guess[0]["robots"]["r"][0, 0] == pytest.approx(x0)   # node 0 of the warm start is the start
        x1 = x0 + prob.num_modes + (shift_first if start is None else 0.0)
        calls.append((prob.num_modes, round(x0, 3), guess is not None))
        return to_eval._Solved(status=0, fail_reason="", rescues=0, outer_iters=5, best_prim=0.01,
                               results=_results(prob.num_nodes, x0, x1), physical=None, iterate="final",
                               constraint_violation=1.0, max_violation=0.02, rows=[{"node": 1}], solve_time_s=0.0)

    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    task.scene.robots["r"].q_min = np.array([-5.0])
    task.scene.robots["r"].q_max = np.array([5.0])
    return to_eval, task, kso, calls


def test_parallel_windows_repair_the_junctions_round_by_round(monkeypatch):
    to_eval, task, kso, calls = _parallel_stub(monkeypatch)
    sol = to_static.solve_parallel(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=kso, workers=1)
    # Round 0: all three windows from the static poses (x = 0, 4, 8). Window 0
    # ends at 4.3, so window 1 is re-solved from there (round 1, warm), then
    # window 2 from window 1's new end 8.3 (round 2).
    assert calls == [(4, 0.0, False), (4, 4.0, False), (4, 8.0, False), (4, 4.3, True), (4, 8.3, True)]
    assert [r["windows"] for r in sol.polish["rounds"]] == [[0, 1, 2], [1], [2]]
    assert sol.polish["rounds"][1]["reasons"] == ["window 1: base differs by 0.300 m"]
    assert sol.polish["unmatched"] == [] and sol.converged and sol.status == 0
    xs = sol.results["robot_xs"]["r"]
    assert xs.shape == (12 * 20 + 1, DIM)
    assert xs[80, 0] == pytest.approx(4.3) and xs[160, 0] == pytest.approx(8.3) and xs[-1, 0] == pytest.approx(12.3)
    assert [s["modes"] for s in sol.segments] == [[0, 4], [4, 8], [8, 12]] and sol.solved_modes == 12


def test_parallel_windows_need_no_repair_when_the_static_poses_match(monkeypatch):
    to_eval, task, kso, calls = _parallel_stub(monkeypatch, shift_first=0.0)
    sol = to_static.solve_parallel(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=kso, workers=1)
    assert len(calls) == 3 and len(sol.polish["rounds"]) == 1 and sol.converged


def test_parallel_windows_stop_after_the_round_limit(monkeypatch):
    to_eval, task, kso, calls = _parallel_stub(monkeypatch)
    sol = to_static.solve_parallel(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=kso, workers=1, max_rounds=1)
    assert len(calls) == 4 and not sol.converged
    assert sol.polish["unmatched"] == ["window 2: base differs by 0.300 m"]
    assert "window 2" in sol.fail_reason
    assert to_static.solve_parallel(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0,
                                    weight_scale=None, kso_results=None, workers=1) is None


def test_start_mismatch_names_the_first_tolerance_exceeded():
    s = to_static.ts.SegmentStart(robot_x=_row(1.0), obj_x={"box": np.zeros(13)})
    e = to_static.ts.SegmentStart(robot_x=_row(1.0), obj_x={"box": np.r_[0.0, 0.0, 0.05, np.zeros(10)]})
    assert to_static.start_mismatch(s, s, NQ, NV) is None
    assert to_static.start_mismatch(s, e, NQ, NV) == "box differs by 0.050 m"
    e2 = to_static.ts.SegmentStart(robot_x=_row(1.0), obj_x={})
    e2.robot_x[NQ] = 0.2                                        # base velocity x
    assert to_static.start_mismatch(s, e2, NQ, NV) == "base velocity differs by 0.200 m/s"


def test_without_kso_keyframes_the_caller_chains(monkeypatch):
    to_eval, task, _ = _stub(monkeypatch)
    assert to_static.solve(task, PLAN, [0, 4, 8, 12], nodes_per_mode=20, time_per_mode=1.0, weight_scale=None,
                           kso_results=None) is None
