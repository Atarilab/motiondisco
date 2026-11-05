"""Segmented TO (loco_manipulation/to_segments.py and to_eval's segmented
path): cut selection, slicing, start states, stitching and aggregation on
toy data, and the segment chaining with stubbed solves (no faro)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import to_segments as ts  # noqa: E402

F, BOX_ON_F, BOX_ON_T = (1, 1), (6, 1, 1), (6, 2, 1)
HL, HR = (9, 1), (11, 1)
# 0-4 walk | 5 grasp | 6 lift | 7-10 carry steps | 11 place | 12 release | 13 stand
ROWS = [  # left_sole, right_sole, left_ee, right_ee, box
    (F, F, (), (), BOX_ON_F), (F, (), (), (), BOX_ON_F), (F, F, (), (), BOX_ON_F),
    ((), F, (), (), BOX_ON_F), (F, F, (), (), BOX_ON_F), (F, F, HL, HR, BOX_ON_F),
    (F, F, HL, HR, ()), (F, (), HL, HR, ()), (F, F, HL, HR, ()), ((), F, HL, HR, ()),
    (F, F, HL, HR, ()), (F, F, HL, HR, BOX_ON_T), (F, F, (), (), BOX_ON_T), (F, F, (), (), BOX_ON_T),
]
CHANNELS = ("left_sole", "right_sole", "left_ee", "right_ee", "box")
PLAN = {ch: [row[i] for row in ROWS] for i, ch in enumerate(CHANNELS)}


def test_cut_ranks_follow_contact_changes_at_double_support():
    rank = [ts.cut_rank(PLAN, k) for k in range(len(ROWS))]
    assert rank[0] is None and rank[1] is None and rank[3] is None and rank[7] is None   # not interior / single support
    assert rank[2] == ts.RANK_STEP and rank[4] == ts.RANK_STEP and rank[8] == ts.RANK_STEP
    assert rank[5] == ts.RANK_HAND and rank[12] == ts.RANK_HAND       # grasp, release
    assert rank[6] == ts.RANK_OBJECT and rank[11] == ts.RANK_OBJECT   # lift, place
    assert rank[13] == ts.RANK_SAME


def test_split_points_respect_lengths_and_never_cut_at_a_hand_change():
    n = len(ROWS)
    assert ts.split_points(PLAN, 20) == [0, n] and ts.split_points(PLAN, 0) == [0, n]
    b = ts.split_points(PLAN, 7, min_modes=2)
    assert b[0] == 0 and b[-1] == n and len(b) == 4            # 2 segments impossible (mode 7 is single support)
    lengths = np.diff(b)
    assert all(2 <= x <= 7 for x in lengths)
    assert all(ts.cut_rank(PLAN, k) in (ts.RANK_SAME, ts.RANK_STEP, ts.RANK_OBJECT) for k in b[1:-1])
    assert sum(ts.cut_rank(PLAN, k) for k in b[1:-1]) == 2      # two walking landings
    assert ts.split_points(PLAN, 7, min_modes=2) == b          # deterministic


def test_hand_cuts_only_on_request_and_no_cut_means_whole_plan():
    # alternate grasp / release at every double-support mode
    rows = [(F, F, (), (), BOX_ON_F) if i % 2 == 0 else (F, F, HL, HR, BOX_ON_F) for i in range(10)]
    plan = {ch: [row[i] for row in rows] for i, ch in enumerate(CHANNELS)}
    assert ts.split_points(plan, 4, min_modes=2) == [0, 10]
    b = ts.split_points(plan, 4, min_modes=2, allow_hand_cut=True)
    assert len(b) > 2 and all(x <= 4 for x in np.diff(b))


def test_a_stretch_without_cuts_longer_than_the_cap_only_lengthens_its_own_segment():
    # walk | grasp, lift | 10 carry steps in single support (no landing to cut at) |
    # land, place, release, stand | walk
    carry = [(F, (), HL, HR, ()) if k % 2 == 0 else ((), F, HL, HR, ()) for k in range(10)]
    rows = ROWS[:7] + carry + [(F, F, HL, HR, ()), (F, F, HL, HR, BOX_ON_T), (F, F, (), (), BOX_ON_T),
                               (F, F, (), (), BOX_ON_T), (F, (), (), (), BOX_ON_T), (F, F, (), (), BOX_ON_T),
                               ((), F, (), (), BOX_ON_T), (F, F, (), (), BOX_ON_T)]
    plan = {ch: [row[i] for row in rows] for i, ch in enumerate(CHANNELS)}
    n = len(rows)
    for prefer in ("balanced", "manip"):
        b = ts.split_points(plan, 7, min_modes=2, prefer=prefer)
        long = [(a, e) for a, e in zip(b, b[1:]) if e - a > 7]
        assert len(b) > 2 and len(long) == 1                   # was [0, n]: the whole plan in one solve
        a, e = long[0]
        assert a <= 6 and e >= 17                              # the one over the cap spans the carry
        assert all(ts.cut_rank(plan, k) in (ts.RANK_SAME, ts.RANK_STEP, ts.RANK_OBJECT) for k in b[1:-1])
    assert all(x <= 12 for x in np.diff(ts.split_points(plan, 12, min_modes=2)))
    assert ts.split_points(plan, n) == [0, n]


def test_manipulation_cuts_are_the_last_landing_before_each_manipulation():
    assert ts.manipulation_cuts(PLAN) == [4, 10]              # before the grasp, before the placement
    assert ts.split_points(PLAN, 7, min_modes=2, prefer="manip") == [0, 4, 10, 14]
    # after: the first landing once the lift is done (carry and placement in one
    # segment) and the standing mode after the release
    assert ts.manipulation_cuts(PLAN, "after") == [8, 13]
    assert ts.split_points(PLAN, 9, min_modes=2, prefer="approach") == [0, 8, 14]
    # first: the grasp (a hand cut) and the placement themselves
    assert ts.manipulation_cuts(PLAN, "first") == [5, 11]
    assert ts.split_points(PLAN, 7, min_modes=2, prefer="first") == [0, 5, 11, 14]
    with pytest.raises(ValueError):
        ts.split_points(PLAN, 7, prefer="nearest")
    with pytest.raises(ValueError):
        ts.manipulation_cuts(PLAN, "during")


def test_greedy_cuts_are_left_to_right_and_prefix_stable():
    # Allowed cuts of PLAN: 2, 4, 6, 8, 10, 11, 13 (double support, no hand
    # change); manipulation cuts 4 (before the grasp) and 10 (before the place).
    assert ts.greedy_cuts(PLAN, 5, 2) == [0, 4, 8, 10, 14]
    assert ts.greedy_cuts(PLAN, 20) == [0, 14] and ts.greedy_cuts(PLAN, 0) == [0, 14]
    # The same plan with more modes after it keeps the same leading cuts (a
    # boundary depends on the plan up to the segment length past it: the
    # farthest manipulation cut within reach wins).
    longer = {ch: seq + [seq[-1]] * 6 for ch, seq in PLAN.items()}
    assert ts.greedy_cuts(longer, 5, 2) == [0, 4, 8, 10, 15, 20]
    # No allowed cut within the length: the first one beyond it is taken.
    walk = {"left_sole": [F, (), F, (), F, (), F, (), F, F], "right_sole": [F, F, (), F, (), F, (), F, F, F],
            "left_ee": [()] * 10, "right_ee": [()] * 10, "box": [BOX_ON_F] * 10}
    assert ts.greedy_cuts(walk, 3, 2) == [0, 8, 10]      # 8 is the first double support


def test_windows_look_past_the_cut_and_the_one_reaching_the_end_is_last():
    assert ts.windows([0, 4, 10, 14]) == [(0, 4, 4), (4, 10, 10), (10, 14, 14)]
    assert ts.windows([0, 4, 10, 14], 3) == [(0, 4, 7), (4, 10, 13), (10, 14, 14)]
    assert ts.windows([0, 4, 10, 14], 5) == [(0, 4, 9), (4, 14, 14)]
    assert ts.windows([0, 14], 5) == [(0, 14, 14)]


def test_truncate_results_keeps_the_first_intervals():
    t = ts.truncate_results(_fake(10, _x0()), 4)
    assert t["robot_xs"]["r"].shape == (5, DIM) and t["obj_xs"]["box"].shape == (5, 13)
    assert t["vdot_rob"]["r"].shape == (4, NV) and t["rob_Ws"]["r"]["left_ee"].shape == (4, 6)
    assert t["torques"]["r"].shape == (4, 1) and t["step_ts"].shape == (4,) and t["t_scale"].shape == (4,)
    assert t["robot_xs"]["r"][-1, 0] == pytest.approx(0.4)


def test_slice_plan():
    sub = ts.slice_plan(PLAN, 4, 9)
    assert set(sub) == set(PLAN) and all(len(v) == 5 for v in sub.values())
    assert sub["left_ee"][1] == HL and sub["box"][0] == BOX_ON_F


NQ, NV = 8, 7           # toy robot: base pose (7) + 1 joint; v has 7 entries
DIM = NQ + NV + 3 + 6


def _fake(n, x_start, joint=0.0):
    """Segment results with n intervals starting at x_start."""
    xs = np.tile(np.asarray(x_start, float), (n + 1, 1))
    xs[:, 0] += np.linspace(0, 1, n + 1)      # base x moves forward
    xs[1:, 7] = joint
    box = np.zeros((n + 1, 13))
    box[:, 6] = 1.0
    return {
        "robot_xs": {"r": xs}, "obj_xs": {"box": box},
        "vdot_rob": {"r": np.arange(n * NV, dtype=float).reshape(n, NV)},
        "rob_Ws": {"r": {"left_sole": np.ones((n, 6)), "left_ee": np.zeros((n, 6))}},
        "obj_Vdots": {"box": np.zeros((n, 6))}, "obj_Ws": {"box": np.ones((n, 6))},
        "t_scale": np.ones(n), "step_ts": np.full(n, 0.05),
        "torques": {"r": np.ones((n, NV - 6))}, "torque_limit": {"r": np.array([5.0])},
    }


def _x0():
    x = np.zeros(DIM)
    x[6] = 1.0
    return x


def test_start_from_results_normalizes_and_clips():
    res = _fake(4, _x0(), joint=1.3)
    res["robot_xs"]["r"][-1, 3:7] = [0, 0, 0, 2.0]       # not a unit quaternion
    res["obj_xs"]["box"][-1, 3:7] = [0, 0, 0, 3.0]
    start = ts.start_from_results(res, "r", NQ, q_min=np.array([-1.0]), q_max=np.array([1.0]))
    assert np.linalg.norm(start.robot_x[3:7]) == pytest.approx(1.0)
    assert np.linalg.norm(start.obj_x["box"][3:7]) == pytest.approx(1.0)
    assert start.robot_x[7] == pytest.approx(1.0) and start.joint_excess == pytest.approx(0.3)
    assert start.robot_x[0] == pytest.approx(1.0)        # the last row
    assert res["robot_xs"]["r"][-1, 7] == pytest.approx(1.3)   # input untouched


def test_clipped_start_keeps_the_stitch_continuous():
    a = _fake(4, _x0(), joint=1.3)                       # ends 0.3 rad past the limit
    start = ts.start_from_results(a, "r", NQ, q_min=np.array([-1.0]), q_max=np.array([1.0]))
    b = _fake(6, start.robot_x)
    with pytest.raises(ValueError):
        ts.stitch_results([a, b])
    a2 = ts.with_end_state(a, start, "r")
    s = ts.stitch_results([a2, b])
    assert s["robot_xs"]["r"][4, 7] == pytest.approx(1.0) and a["robot_xs"]["r"][-1, 7] == pytest.approx(1.3)


def test_tiled_guess_puts_the_start_on_node_zero():
    start = ts.SegmentStart(robot_x=np.arange(DIM, dtype=float), obj_x={"box": np.full(13, 2.0)})
    x0, u0 = ts.tiled_guess(start, {"r": np.zeros(DIM)}, {"box": np.zeros(13)}, 6,
                            {"r": NV}, {"r": ["left_sole", "left_ee"]}, "r")
    assert x0["robots"]["r"].shape == (7, DIM) and x0["objects"]["box"].shape == (7, 13)
    assert np.array_equal(x0["robots"]["r"][0], start.robot_x) and not x0["robots"]["r"][1:].any()
    assert np.array_equal(x0["objects"]["box"][0], start.obj_x["box"])
    assert u0["robots"]["r"]["acceleration"].shape == (6, NV)
    assert set(u0["robots"]["r"]["wrenches"]) == {"left_sole", "left_ee"}
    assert u0["objects"]["box"].shape == (6, 12) and np.array_equal(u0["t_scale"], np.ones(6))


def test_stitch_keeps_the_junction_node_once():
    a = _fake(4, _x0())
    b = _fake(6, a["robot_xs"]["r"][-1])
    s = ts.stitch_results([a, b])
    assert s["robot_xs"]["r"].shape == (11, DIM) and s["obj_xs"]["box"].shape == (11, 13)
    assert s["vdot_rob"]["r"].shape == (10, NV) and s["torques"]["r"].shape == (10, 1)
    assert s["rob_Ws"]["r"]["left_sole"].shape == (10, 6) and s["step_ts"].shape == (10,)
    assert np.array_equal(s["robot_xs"]["r"][4], a["robot_xs"]["r"][-1])
    assert np.array_equal(s["robot_xs"]["r"][5], b["robot_xs"]["r"][1])
    assert s["u0_dict"]["objects"]["box"].shape == (10, 12)
    assert s["x0_dict"]["robots"]["r"].shape == (11, DIM)
    bad = _fake(6, _x0())        # starts back at the origin
    with pytest.raises(ValueError):
        ts.stitch_results([a, bad])


def test_pad_hold_extends_to_the_full_horizon():
    p = ts.pad_hold(_fake(4, _x0()), 10, 0.05)
    assert p["robot_xs"]["r"].shape == (11, DIM) and p["t_scale"].shape == (10,)
    assert np.array_equal(p["robot_xs"]["r"][-1], p["robot_xs"]["r"][4])
    assert not p["vdot_rob"]["r"][4:].any() and np.all(p["step_ts"] > 0)


def test_cost_helpers():
    v = {"r": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 3.0]])}
    assert ts.jerk_cost(v, 0.5) == pytest.approx(2.0 + 6.0)
    assert ts.to_cost(2.0, 100.0, 40, 2) == pytest.approx(1e2 * 2 / 40 + 1e-2 * 100 / 2)
    assert np.allclose(ts.polish_t_scale([0.05, 0.005, 0.5], [0.05, 0.05, 0.05]), [1.0, 0.5, 2.0])
    assert ts.shift_rows([{"node": 3, "name": "x"}], 40) == [{"node": 43, "name": "x"}]


def test_aggregate_names_the_failing_segment():
    o = [ts.SegmentOutcome((0, 13), status=0, max_violation=0.01, constraint_violation=1.0, rescues=1),
         ts.SegmentOutcome((13, 26), status=1, max_violation=0.4, constraint_violation=2.0, iterate="best"),
         ts.SegmentOutcome((26, 40))]
    agg = ts.aggregate(o)
    assert agg["status"] == 1 and not agg["converged"] and agg["rescues"] == 1
    assert agg["constraint_violation"] == pytest.approx(3.0) and agg["max_constraint_violation"] == 0.4
    # An unsolved tail after a non-converged segment is a fail-fast stop.
    assert agg["iterate"] == "mixed" and agg["fail_reason"] == "segment 1 modes [13,26): stopped, status 1, max violation 0.4"
    o[1].status = 0
    assert ts.aggregate(o)["fail_reason"] == ""
    o[1].status = 1
    o[2].status, o[2].fail_reason = 2, "collision: numerical_failure"
    assert ts.aggregate(o)["fail_reason"] == "segment 2 modes [26,40): collision: numerical_failure"
    assert ts.aggregate(o[:1])["converged"]
    assert o[2].to_dict()["modes"] == [26, 40]


def test_segmented_solve_chains_starts_and_stops_on_failure(monkeypatch):
    to_eval = pytest.importorskip("to_eval")
    calls = []
    statuses = iter([0, 1, 2])

    def build(task, plan, **kw):
        n = len(plan["left_sole"])
        return SimpleNamespace(to_task=SimpleNamespace(collision_index=None), task=task, plan=plan,
                               num_modes=n, num_nodes=n * 20, build_time_s=0.0, nodes_per_mode=20)

    def solve(prob, kso_results=None, start=None, **kw):
        x = _x0() if start is None else start.robot_x
        calls.append((prob.num_modes, None if start is None else start.robot_x.copy()))
        return to_eval._Solved(status=next(statuses), fail_reason="collision: x", rescues=0, outer_iters=5,
                               best_prim=0.1, results=_fake(prob.num_nodes, x), physical=None, iterate="final",
                               constraint_violation=1.0, max_violation=0.2,
                               rows=[{"node": 1, "name": "n", "idx": 0, "kind": "eq", "raw": 1.0, "violation": 1.0}],
                               solve_time_s=0.0)

    monkeypatch.setattr(to_eval, "_build_problem", build)
    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    monkeypatch.setattr(to_eval, "_segment_task", lambda task, start, goal: task)
    monkeypatch.setattr(to_eval, "_translate_rows", lambda rows, *a: [{"node": r["node"]} for r in rows])
    monkeypatch.setattr(to_eval, "TO_PHYSICAL_CHECK", False)
    monkeypatch.setattr(to_eval, "TO_SEGMENT_FAIL_FAST", False)
    monkeypatch.setattr(to_eval, "TO_INCREMENTAL", False)
    robot = SimpleNamespace(nq=NQ, q_min=np.array([-5.0]), q_max=np.array([5.0]))
    task = SimpleNamespace(robot_id="r", terminal_goal=None, scene=SimpleNamespace(robots={"r": robot}))
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert [c[0] for c in calls] == [4, 4, 6]
    assert calls[0][1] is None
    assert calls[1][1][0] == pytest.approx(1.0)       # segment 1 starts at segment 0's end (base x = 1)
    assert sol.status == 2 and "segment 2 modes [8,14)" in sol.fail_reason and not sol.converged
    assert sol.results["robot_xs"]["r"].shape == (14 * 20 + 1, DIM)
    assert [v["node"] for v in sol.violations] == [1, 81, 161]
    assert [s["status"] for s in sol.segments] == [0, 1, 2] and sol.num_nodes == 280
    assert sol.solved_modes == 4 and sol.repairs == [] and sol.plan is None


def test_a_numerical_failure_passing_the_physical_check_solves_nothing(monkeypatch):
    """shelf_unstack gen_22: window 0 failed numerically, its held initial
    pose passed the physical check, and the plan was reported fully solved."""
    to_eval = pytest.importorskip("to_eval")

    def build(task, plan, **kw):
        n = len(plan["left_sole"])
        return SimpleNamespace(to_task=SimpleNamespace(collision_index=None), task=task, plan=plan,
                               num_modes=n, num_nodes=n * 20, build_time_s=0.0, nodes_per_mode=20)

    def solve(prob, kso_results=None, start=None, **kw):
        return to_eval._Solved(status=2, fail_reason="presolve: infeasible", rescues=0, outer_iters=1,
                               best_prim=2.7, results=_fake(prob.num_nodes, _x0()), physical={"pass": True, "fails": []},
                               iterate="final", constraint_violation=9.0, max_violation=2.7, rows=[], solve_time_s=0.0)

    monkeypatch.setattr(to_eval, "_build_problem", build)
    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    monkeypatch.setattr(to_eval, "_segment_task", lambda task, start, goal: task)
    monkeypatch.setattr(to_eval, "_translate_rows", lambda rows, *a: [])
    monkeypatch.setattr(to_eval, "TO_INCREMENTAL", False)
    robot = SimpleNamespace(nq=NQ, q_min=np.array([-5.0]), q_max=np.array([5.0]))
    task = SimpleNamespace(robot_id="r", terminal_goal=None, scene=SimpleNamespace(robots={"r": robot}))
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert sol.status == 2 and [s["status"] for s in sol.segments] == [2, None, None]
    assert sol.solved_modes == 0                       # nothing after a numerical failure counts as solved


def test_physical_choice_skips_the_final_iterate_after_a_numerical_failure(monkeypatch):
    to_eval = pytest.importorskip("to_eval")
    tc = pytest.importorskip("trajectory_check")
    checked = []
    monkeypatch.setattr(tc, "check", lambda task, plan, res, npm: checked.append(res["tag"]) or {"pass": res["tag"] == "best", "fails": []})
    final, best = {"tag": "final"}, {"tag": "best"}
    assert to_eval._physical_choice(None, PLAN, final, None, 20, status=2) == (None, final, "final") and checked == []
    rep, res, which = to_eval._physical_choice(None, PLAN, final, best, 20, status=2)
    assert which == "best" and rep["pass"] and checked == ["best"]
    checked.clear()
    rep, res, which = to_eval._physical_choice(None, PLAN, final, best, 20, status=1)
    assert which == "best" and checked == ["final", "best"]     # status 1: the final iterate is tried first


def test_lookahead_solves_past_the_cut_and_keeps_up_to_it(monkeypatch):
    to_eval = pytest.importorskip("to_eval")
    calls = []

    def build(task, plan, **kw):
        n = len(plan["left_sole"])
        return SimpleNamespace(to_task=SimpleNamespace(collision_index=None), task=task, plan=plan,
                               num_modes=n, num_nodes=n * 20, build_time_s=0.0, nodes_per_mode=20)

    def solve(prob, kso_results=None, start=None, **kw):
        x = _x0() if start is None else start.robot_x
        calls.append((prob.num_modes, x[0]))
        return to_eval._Solved(status=0, fail_reason="", rescues=0, outer_iters=5, best_prim=0.01,
                               results=_fake(prob.num_nodes, x), physical=None, iterate="final",
                               constraint_violation=1.0, max_violation=0.02,
                               rows=[{"node": 1}, {"node": 100}], solve_time_s=0.0)

    monkeypatch.setattr(to_eval, "_build_problem", build)
    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    monkeypatch.setattr(to_eval, "_segment_task", lambda task, start, goal: task)
    monkeypatch.setattr(to_eval, "_translate_rows", lambda rows, *a: [{"node": r["node"]} for r in rows])
    monkeypatch.setattr(to_eval, "TO_PHYSICAL_CHECK", False)
    monkeypatch.setattr(to_eval, "TO_INCREMENTAL", False)
    monkeypatch.setattr(to_eval, "TO_SEGMENT_LOOKAHEAD", 3)
    robot = SimpleNamespace(nq=NQ, q_min=np.array([-5.0]), q_max=np.array([5.0]))
    task = SimpleNamespace(robot_id="r", terminal_goal=None, scene=SimpleNamespace(robots={"r": robot}))
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert [c[0] for c in calls] == [7, 7, 6]                 # windows [0,7), [4,11), [8,14)
    assert calls[1][1] == pytest.approx(80 / 140)             # starts at node 80 of a 140-interval window
    assert sol.converged and sol.results["robot_xs"]["r"].shape == (14 * 20 + 1, DIM)
    assert [v["node"] for v in sol.violations] == [1, 81, 161, 260]   # rows past a cut dropped
    assert [s["modes"] for s in sol.segments] == [[0, 4], [4, 8], [8, 14]]
    assert [s["extra"].get("window") for s in sol.segments] == [[0, 7], [4, 11], None]


def test_codegen_cache_seeded_from_the_newest_matching_horizon(tmp_path):
    import os

    to_eval = pytest.importorskip("to_eval")

    def make(name, mtime, files=("f/abc/lib.so", "f/abc/x.lock")):
        for f in files:
            p = tmp_path / name / ".moto_artifacts" / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
        os.utime(tmp_path / name, (mtime, mtime))

    make("scene_800", 100)
    make("scene_820", 200)
    make("scene_900_tohard", 300)
    make("other_scene_999", 400)
    assert to_eval._seed_codegen("scene", "", root=str(tmp_path)).endswith("scene_820")
    so = tmp_path / "scene" / ".moto_artifacts" / "f" / "abc"
    assert (so / "lib.so").read_text() == "scene_820" and not (so / "x.lock").exists()
    assert to_eval._seed_codegen("scene", "", root=str(tmp_path)) is None       # already there
    assert to_eval._seed_codegen("scene", "_tohard", root=str(tmp_path)).endswith("scene_900_tohard")
    assert to_eval._seed_codegen("new_scene", "", root=str(tmp_path)) is None   # nothing to seed from


def test_failed_segment_chains_from_its_best_iterate(monkeypatch):
    to_eval = pytest.importorskip("to_eval")
    starts = []

    def build(task, plan, **kw):
        n = len(plan["left_sole"])
        return SimpleNamespace(to_task=SimpleNamespace(collision_index=None), task=task, plan=plan,
                               num_modes=n, num_nodes=n * 20, build_time_s=0.0, nodes_per_mode=20)

    def solve(prob, kso_results=None, start=None, **kw):
        starts.append(None if start is None else start.robot_x[0])
        x = _x0() if start is None else start.robot_x
        final, best = _fake(prob.num_nodes, x), _fake(prob.num_nodes, x)
        best["robot_xs"]["r"][-1, 0] = 0.5          # the best iterate ends elsewhere
        return to_eval._Solved(status=1, fail_reason="", rescues=0, outer_iters=300, best_prim=1.4,
                               results=final, physical={"pass": False}, iterate="final",
                               constraint_violation=1.0, max_violation=17.4, rows=[], solve_time_s=0.0, best=best)

    monkeypatch.setattr(to_eval, "_build_problem", build)
    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    monkeypatch.setattr(to_eval, "_segment_task", lambda task, start, goal: task)
    monkeypatch.setattr(to_eval, "TO_PHYSICAL_CHECK", False)
    monkeypatch.setattr(to_eval, "TO_SEGMENT_FAIL_FAST", False)
    monkeypatch.setattr(to_eval, "TO_INCREMENTAL", False)
    robot = SimpleNamespace(nq=NQ, q_min=np.array([-5.0]), q_max=np.array([5.0]))
    task = SimpleNamespace(robot_id="r", terminal_goal=None, scene=SimpleNamespace(robots={"r": robot}))
    sol = to_eval._solve_segmented(task, PLAN, [0, 6, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert starts[1] == pytest.approx(0.5)            # from the best iterate, not the final one (1.0)
    assert [s["iterate"] for s in sol.segments] == ["best", "best"] and sol.iterate == "best"
    assert sol.solved_modes == 0


def _segmented_stub(monkeypatch, statuses, physical=None):
    """Stubbed to_eval: segment i gets statuses[i] (a status-1 segment is
    not physically valid); returns the task and the list of
    (num_modes, start x, plan length) per solve call."""
    to_eval = pytest.importorskip("to_eval")
    calls = []

    def build(task, plan, **kw):
        n = len(plan["left_sole"])
        return SimpleNamespace(to_task=SimpleNamespace(collision_index=None), task=task, plan=plan,
                               num_modes=n, num_nodes=n * 20, build_time_s=0.0, nodes_per_mode=20)

    def solve(prob, kso_results=None, start=None, **kw):
        x = _x0() if start is None else start.robot_x
        status = statuses[len(calls)] if len(calls) < len(statuses) else 0
        calls.append((prob.num_modes, x[0], len(prob.plan["left_sole"])))
        return to_eval._Solved(status=status, fail_reason="", rescues=0, outer_iters=100, best_prim=0.3,
                               results=_fake(prob.num_nodes, x), iterate="final", constraint_violation=1.0,
                               max_violation=0.3 if status else 0.02, rows=[{"node": 1}], solve_time_s=0.0,
                               physical=None if status == 0 else (physical or {"pass": False, "fails": ["contact_gap"],
                                                                                "worst": {"contact_gap": 0.1},
                                                                                "where": {"contact_gap": ["mode", 2, "right_ee", "box", "top"]}}))

    monkeypatch.setattr(to_eval, "_build_problem", build)
    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    monkeypatch.setattr(to_eval, "_segment_task", lambda task, start, goal: task)
    monkeypatch.setattr(to_eval, "_translate_rows", lambda rows, *a: [{"node": r["node"]} for r in rows])
    monkeypatch.setattr(to_eval, "TO_PHYSICAL_CHECK", False)
    monkeypatch.setattr(to_eval, "TO_SEGMENT_FAIL_FAST", True)
    monkeypatch.setattr(to_eval, "TO_INCREMENTAL", False)
    robot = SimpleNamespace(nq=NQ, nv=NV, q_min=np.array([-5.0]), q_max=np.array([5.0]), patch_ees=["left_sole"])
    resolved = SimpleNamespace(robots={"r": SimpleNamespace(x=_x0())},
                               env_objects={"box": SimpleNamespace(x=np.r_[np.zeros(6), 1.0, np.zeros(6)])})
    task = SimpleNamespace(robot_id="r", terminal_goal=None, scene=SimpleNamespace(robots={"r": robot}),
                           resolved_state=resolved)
    return to_eval, task, calls


def test_fail_fast_stops_at_the_first_bad_segment_and_holds_the_rest(monkeypatch):
    to_eval, task, calls = _segmented_stub(monkeypatch, [0, 1, 0])
    monkeypatch.setattr(to_eval, "TO_REPAIR", False)
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert len(calls) == 2                                    # segment 2 never solved
    assert [s["status"] for s in sol.segments] == [0, 1, None]
    assert sol.status == 1 and not sol.converged and not sol.valid and sol.solved_modes == 4
    assert "segment 1 modes [4,8): stopped" in sol.fail_reason
    assert sol.results["robot_xs"]["r"].shape == (14 * 20 + 1, DIM)   # padded to the full plan
    # The failing window's physical report, moved to full-plan modes, is the result's.
    assert sol.physical["pass"] is False and sol.physical["where"]["contact_gap"][1] == 6
    assert sol.physical["window_start_mode"] == 4


def test_step_repair_inserts_steps_and_resolves_the_segment(monkeypatch):
    # Segment 1 (modes [4,8), walk + grasp) fails once; the repair inserts
    # two steps before the grasp (mode 5) and the re-solve converges.
    to_eval, task, calls = _segmented_stub(monkeypatch, [0, 1, 0, 0])
    monkeypatch.setattr(to_eval, "TO_REPAIR", True)
    asked = []

    def repair(task, plan, a, e, results, npm):
        asked.append((a, e, len(plan["left_sole"])))
        return (5, 2, "left_sole travels 1.90 m landing at mode 4") if len(asked) == 1 else None

    monkeypatch.setattr(to_eval, "_step_repair", repair)
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results={"robot_xs": {"r": np.zeros((15, NQ))}})
    assert asked == [(4, 8, 14)]
    # Solves: segment 0, segment 1 (fails), segment 1 again on 7 modes from the same start, segment 2.
    assert [(c[0], c[2]) for c in calls] == [(4, 4), (4, 4), (7, 7), (6, 6)]
    assert calls[1][1] == pytest.approx(calls[2][1])
    assert sol.converged and sol.num_modes == 17 and sol.solved_modes == 17
    assert [s["modes"] for s in sol.segments] == [[0, 4], [4, 11], [11, 17]]
    assert sol.segments[1]["extra"]["repairs"] == 1
    assert sol.repairs == [{"segment": 1, "before_mode": 5, "before_mode_original": 5, "steps": 2,
                            "modes_added": 3, "reason": "left_sole travels 1.90 m landing at mode 4"}]
    assert len(sol.plan["left_sole"]) == 17 and sol.plan["left_ee"][5:9] == [(), (), (), HL]
    assert sol.results["robot_xs"]["r"].shape == (17 * 20 + 1, DIM)


def test_repair_caps_and_unrepairable_segments_fall_back_to_fail_fast(monkeypatch):
    to_eval, task, calls = _segmented_stub(monkeypatch, [0, 1, 1, 1, 1])
    monkeypatch.setattr(to_eval, "TO_REPAIR", True)
    monkeypatch.setattr(to_eval, "TO_REPAIR_MAX", 2)
    monkeypatch.setattr(to_eval, "_step_repair", lambda task, plan, a, e, results, npm: (a + 1, 1, "long step"))
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    # Segment 1 is repaired twice (cap), still fails, and the solve stops there.
    assert len(calls) == 4 and len(sol.repairs) == 2 and sol.solved_modes == 4
    assert [s["status"] for s in sol.segments] == [0, 1, None]
    assert sol.repairs[1]["before_mode_original"] == 5         # mode 7 of the once-repaired plan
    assert len(sol.plan["left_sole"]) == 14 + 2 + 2
    # A repair that insert_walk rejects (mode 7 is single support) is not applied.
    to_eval, task, calls = _segmented_stub(monkeypatch, [0, 1, 0])
    monkeypatch.setattr(to_eval, "_step_repair", lambda task, plan, a, e, results, npm: (8, 1, "x"))
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert sol.repairs == [] and sol.plan is None and len(calls) == 2


def test_repeated_repairs_stop_when_the_stride_does_not_shorten(monkeypatch):
    to_eval, task, calls = _segmented_stub(monkeypatch, [0, 1, 1, 1])
    monkeypatch.setattr(to_eval, "TO_REPAIR", True)
    strides = iter([1.9, 1.85, 1.0])

    def repair(task, plan, a, e, results, npm):
        s = next(strides)
        return (a + 1, 1, f"left_sole travels {s:.2f} m", s)

    monkeypatch.setattr(to_eval, "_step_repair", repair)
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    # One repair (1.9 m); the second asks for the same mode with 1.85 m, less
    # than a 20% improvement, so it is refused and the solve stops there.
    assert len(sol.repairs) == 1 and len(calls) == 3 and sol.solved_modes == 4
    assert sol.repairs[0]["reason"] == "left_sole travels 1.90 m"


def test_repair_recuts_the_rest_of_the_plan_within_the_segment_length(monkeypatch):
    to_eval, task, calls = _segmented_stub(monkeypatch, [0, 1])
    monkeypatch.setattr(to_eval, "TO_REPAIR", True)
    monkeypatch.setattr(to_eval, "TO_SEGMENT_MODES", 5)
    monkeypatch.setattr(to_eval, "TO_SEGMENT_MIN_MODES", 2)
    monkeypatch.setattr(to_eval, "TO_SEGMENT_CUTS", "balanced")
    asked = []

    def repair(task, plan, a, e, results, npm):
        asked.append(a)
        return (5, 3, "long step") if len(asked) == 1 else None

    monkeypatch.setattr(to_eval, "_step_repair", repair)
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    modes = [tuple(s["modes"]) for s in sol.segments]
    assert sol.converged and sol.num_modes == 18 and modes[0] == (0, 4) and modes[1][0] == 4
    assert all(b - a <= 5 for a, b in modes[1:]) and modes[-1][1] == 18      # re-cut, not stretched to 4..12
    assert sol.plan is not None and len(sol.plan["left_sole"]) == 18
    # Without a segment length the later cuts just shift.
    to_eval, task, calls = _segmented_stub(monkeypatch, [0, 1])
    monkeypatch.setattr(to_eval, "TO_SEGMENT_MODES", 0)
    monkeypatch.setattr(to_eval, "_step_repair", repair)
    asked.clear()
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert [tuple(s["modes"]) for s in sol.segments] == [(0, 4), (4, 12), (12, 18)]


def _incremental_stub(monkeypatch, tmp_path, statuses, kso_fail=()):
    """_segmented_stub plus the incremental mode: a window cache in
    tmp_path and a window KSO that fails for the windows starting at the
    modes in ``kso_fail``."""
    to_eval, task, calls = _segmented_stub(monkeypatch, statuses)
    monkeypatch.setattr(to_eval, "TO_INCREMENTAL", True)
    monkeypatch.setattr(to_eval, "TO_REPAIR", False)
    monkeypatch.setenv("SHINKA_SEGMENT_CACHE_DIR", str(tmp_path / "segment_cache"))
    kso_calls = []

    def window_kso(seg_task, window, attempts):
        a = len(window["left_sole"])
        kso_calls.append(a)
        start = kso_calls  # noqa: F841 - the window length identifies the call in the asserts
        return (False, None, None) if len(kso_calls) - 1 in kso_fail else (True, None, None)

    monkeypatch.setattr(to_eval, "_window_kso", window_kso)
    task.scene_name, task.ic_name = "scene", "default"
    return to_eval, task, calls, kso_calls


def test_incremental_evaluation_reuses_cached_windows_of_a_shared_prefix(monkeypatch, tmp_path):
    import segment_cache

    to_eval, task, calls, kso_calls = _incremental_stub(monkeypatch, tmp_path, [0, 0, 0])
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert sol.converged and sol.kso_feasible is True and sol.cached_modes == 0
    assert len(calls) == 3 and len(kso_calls) == 1          # KSO on the first window only
    assert len(segment_cache.Cache(str(tmp_path / "segment_cache"))) == 3
    # Same first 8 modes, a different ending: only the last window is solved.
    changed = {ch: list(seq) for ch, seq in PLAN.items()}
    changed["left_ee"][-1] = HL
    to_eval, task, calls, kso_calls = _incremental_stub(monkeypatch, tmp_path, [0])
    sol = to_eval._solve_segmented(task, changed, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert len(calls) == 1 and calls[0][0] == 6 and len(kso_calls) == 0
    assert sol.converged and sol.cached_modes == 8 and sol.kso_feasible is True
    assert [s["extra"].get("cached") for s in sol.segments] == [True, True, None]
    assert sol.results["robot_xs"]["r"].shape == (14 * 20 + 1, DIM)
    # The identical plan again: everything from the cache, nothing solved.
    to_eval, task, calls, kso_calls = _incremental_stub(monkeypatch, tmp_path, [])
    sol = to_eval._solve_segmented(task, changed, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert calls == [] and sol.converged and sol.cached_modes == 14 and sol.solved_modes == 14


def test_incremental_window_kso_gates_the_first_window_and_explains_later_ones(monkeypatch, tmp_path):
    # KSO fails on the first window: nothing is solved, the plan is kinematically infeasible.
    to_eval, task, calls, kso_calls = _incremental_stub(monkeypatch, tmp_path, [0, 0, 0], kso_fail={0})
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert calls == [] and sol.kso_feasible is False and sol.solved_modes == 0 and sol.status == 2
    assert sol.segments[0]["fail_reason"].startswith("kso: no kinematic solution for modes [0,4)")
    assert sol.results["robot_xs"]["r"].shape == (14 * 20 + 1, DIM)
    # Later windows go straight to TO: KSO runs once, on the first window.
    to_eval, task, calls, kso_calls = _incremental_stub(monkeypatch, tmp_path / "b", [0, 0, 0], kso_fail={1})
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert sol.converged and len(calls) == 3 and len(kso_calls) == 1
    # TO fails on a later window: the stop is a TO failure with the prefix solved.
    to_eval, task, calls, kso_calls = _incremental_stub(monkeypatch, tmp_path / "c", [0, 1, 0])
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert not sol.converged and sol.solved_modes == 4 and sol.kso_feasible is True
    assert "kso" not in sol.segments[1]["extra"] and len(kso_calls) == 1
    assert len(segment_cache_entries(tmp_path / "c")) == 1          # only the good first window was stored


def segment_cache_entries(tmp_path):
    return [p for p in (tmp_path / "segment_cache").iterdir() if p.suffix == ".pkl"]


def test_origin_of_and_shift_report():
    to_eval = pytest.importorskip("to_eval")
    origin = [0, 1, None, None, 2, 3]
    assert [to_eval._origin_of(origin, m) for m in (1, 2, 4, 6)] == [1, 2, 2, None]
    rep = to_eval._shift_report({"pass": False, "where": {"penetration": ["node", 30, "a", "b"],
                                                          "dyn_force_ratio": ["mode", 2, "interval", 45]}}, 4, 20)
    assert rep["where"] == {"penetration": ["node", 110, "a", "b"], "dyn_force_ratio": ["mode", 6, "interval", 125]}
    assert to_eval._shift_report(None, 4, 20) is None


def test_a_window_that_fails_cold_is_resolved_from_a_window_kso_seed(monkeypatch):
    """TO_ROUTE_SEED: the failed window is solved again with the KSO
    solution of that window as initial guess, once; the seeded solve is
    kept when it is valid, else the cold one goes on to the step repair."""
    to_eval = pytest.importorskip("to_eval")
    calls, kso_calls = [], []

    def build(task, plan, **kw):
        n = len(plan["left_sole"])
        return SimpleNamespace(to_task=SimpleNamespace(collision_index=None), task=task, plan=plan,
                               num_modes=n, num_nodes=n * 20, build_time_s=0.0, nodes_per_mode=20)

    def solve(prob, kso_results=None, start=None, force_seed=False, **kw):
        x = _x0() if start is None else start.robot_x
        calls.append((prob.num_modes, force_seed))
        status = 0 if (force_seed or len(calls) == 1) else 1      # window 1 fails cold, solves seeded
        return to_eval._Solved(status=status, fail_reason="", rescues=0, outer_iters=5, best_prim=0.1,
                               results=_fake(prob.num_nodes, x), physical=None, iterate="final",
                               constraint_violation=0.0, max_violation=0.05, rows=[], solve_time_s=0.0)

    def window_kso(seg_task, window, attempts):
        kso_calls.append(len(window["left_sole"]))
        return True, None, None, {"robot_xs": {}, "obj_xs": {}}

    monkeypatch.setattr(to_eval, "_build_problem", build)
    monkeypatch.setattr(to_eval, "_solve_problem", solve)
    monkeypatch.setattr(to_eval, "_window_kso_results", window_kso)
    monkeypatch.setattr(to_eval, "_segment_task", lambda task, start, goal: task)
    monkeypatch.setattr(to_eval, "_translate_rows", lambda rows, *a: [])
    monkeypatch.setattr(to_eval, "TO_PHYSICAL_CHECK", False)
    monkeypatch.setattr(to_eval, "TO_INCREMENTAL", False)
    monkeypatch.setattr(to_eval, "TO_ROUTE_SEED", True)
    robot = SimpleNamespace(nq=NQ, q_min=np.array([-5.0]), q_max=np.array([5.0]))
    task = SimpleNamespace(robot_id="r", terminal_goal=None, scene=SimpleNamespace(robots={"r": robot}))
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    # window 0 cold OK; window 1 cold fails -> KSO on it -> seeded solve OK; window 2 cold fails -> seeded OK
    assert calls == [(4, False), (4, False), (4, True), (6, False), (6, True)]
    assert kso_calls == [4, 6]
    assert sol.converged and sol.solved_modes == 14
    assert sol.segments[1]["extra"]["route_seed"]["valid"] is True and "route_seed" not in sol.segments[0]["extra"]

    # KSO has no solution: no seeded solve, the window keeps its cold failure
    calls.clear()
    monkeypatch.setattr(to_eval, "_window_kso_results", lambda *a: (False, "infeasible", None, None))
    monkeypatch.setattr(to_eval, "TO_REPAIR", False)
    sol = to_eval._solve_segmented(task, PLAN, [0, 4, 8, 14], nodes_per_mode=20, time_per_mode=1.0,
                                   weight_scale=None, kso_results=None)
    assert calls == [(4, False), (4, False)] and not sol.converged and sol.solved_modes == 4
    assert sol.segments[1]["extra"]["route_seed"] == {"note": "KSO found no solution (infeasible)", "valid": False,
                                                       "status": None}
