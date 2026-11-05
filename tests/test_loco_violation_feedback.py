"""Plain-language KSO / TO violation feedback (loco_manipulation example).

The KSO fixture holds real solver rows: three failed solves of a
truck_unload_stack plan that sets the big box down on the truck's rear step
instead of the floor.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import violation_feedback as vf  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "loco_truck_gen3_kso_violations.json"


def _resolver(data: dict) -> vf.NameResolver:
    r = data["resolver"]
    plan = {k: [tuple(e) for e in v] for k, v in data["plan"].items()}
    res = vf.NameResolver(None, plan)
    res.interfaces = r["interfaces"]
    res.objects = set(r["objects"])
    res.face_env = {int(k): v for k, v in r["face_env"].items()}
    res.face_self = {o: {int(k): v for k, v in f.items()} for o, f in r["face_self"].items()}
    res.pairs = {k: tuple(v) for k, v in r["pairs"].items()}
    return res


def _truck():
    data = json.loads(FIXTURE.read_text())
    res = _resolver(data)
    # faro hands over numpy arrays, not lists.
    attempts = [{**a, "sorted_vios": np.asarray(a["sorted_vios"]),
                 "sorted_vio_names": np.asarray(a["sorted_vio_names"])}
                for a in data["attempts"]]
    return res, [vf.translate_kso(a, res) for a in attempts]


def test_decode_solver_name():
    assert vf.decode_solver_name("n_6565_726f626f74_6c6566745f6565") == ["ee", "robot", "left_ee"]
    assert vf.decode_solver_name("n_" + "ab" * 32) is None  # SHA-256 digest
    assert vf.decode_solver_name("big_box") is None


def test_kso_rows_map_to_mode_subject_and_meaning():
    res, attempts = _truck()
    first = attempts[0]
    support = first[0]
    assert (support.mode, support.subject, support.family, support.unit) == (30, "big_box", "gap", "m")
    assert support.sign > 0 and abs(support.value - 0.40) < 0.01
    assert support.other == "floor surface"
    # Object-only collision pair (hex-encoded) and robot-link pair (digest).
    coll = [v for v in first if v.family == "collision"]
    assert any(v.subject == "big_box" and v.other == "rear_step" for v in coll)
    assert any(v.subject == "left upper arm" and v.other == "torso" for v in coll)
    # Robot contact patches resolve to the plan's channel names.
    assert {v.subject for v in first if v.family in ("twist", "tilt")} >= {"right_sole", "left_ee"}


def test_summary_explains_box_on_rear_step_and_hides_noise():
    res, attempts = _truck()
    lines = vf.summarize(vf.merge_attempts(attempts), res, n_attempts=3)
    assert lines[0].startswith("modes 29–30: big_box ends up on top of rear_step")
    assert "bottom face on floor surface" in lines[0]
    # The gap and the collision it explains are not repeated.
    assert not any("penetrates rear_step" in ln for ln in lines[1:])
    # Problems seen in a single solve collapse into one noise note.
    assert any("only one solve" in ln for ln in lines)
    assert not any("(in 1 of 3 solves)" in ln for ln in lines)


def test_to_rows_translate_contact_force_and_limit_families():
    res = vf.NameResolver(None, {"left_ee": [(), (3, 1), (3, 1)], "box": [(6, 1, 1)] * 3})
    res.interfaces = {"n_6565_726f626f74_6c6566745f6565": "left_ee"}
    res.objects = {"box"}
    res.face_env = {1: "floor surface", 3: "box left"}
    res.face_self = {"box": {6: "bottom"}}
    rows = [
        {"node": 25, "name": "n_6565_726f626f74_6c6566745f6565_ctrl_ineqs", "idx": 10,
         "kind": "ineq", "raw": 4.0, "violation": 4.0},
        {"node": 30, "name": "n_6565_726f626f74_6c6566745f6565_kin_eqs", "idx": 0,
         "kind": "eq", "raw": 0.05, "violation": 0.05},
        {"node": 41, "name": "obj_box_kin_eqs", "idx": 2, "kind": "eq", "raw": -0.03,
         "violation": 0.03},
        {"node": 12, "name": "n_726f626f74_726f626f74_6a6f696e745f706f735f626f756e64",
         "idx": 3, "kind": "ineq", "raw": 0.2, "violation": 0.2},
        {"node": 5, "name": "box_obj_f_eq_ma", "idx": 2, "kind": "eq", "raw": 1.0,
         "violation": 1.0},
        {"node": 1, "name": "dynamics", "idx": 0, "kind": "dyn", "raw": 1e-4,
         "violation": 1e-4},  # below the report cutoff
    ]
    vs = vf.translate_to(rows, res, nodes_per_mode=20)
    got = {(v.mode, v.subject, v.family) for v in vs}
    assert got == {
        (1, "left_ee", "pulling"), (1, "left_ee", "slip"), (2, "box", "gap"),
        (0, "robot", "joint_limit"), (0, "box", "object_balance"),
    }
    text = "\n".join(vf.summarize(vs, res))
    assert "left_ee on box left would have to pull" in text
    assert "box is pushed 0.03 m into floor surface" in text


def test_to_status_text():
    assert "iterations" in vf.to_status_text(1)
    assert "numerically" in vf.to_status_text(2)


def test_converged_to_slack_is_not_reported():
    """Real rows of a converged TO solve: only slack-inflated inequality
    rows under moto's 0.05 feasibility tolerance, so nothing to report."""
    data = json.loads((Path(__file__).parent / "fixtures" / "loco_floor_stack_to_rows.json").read_text())
    assert data["status"] == 0 and data["rows"]
    res = vf.NameResolver(None, {"left_sole": [(1, 1)] * 12})
    assert vf.translate_to(data["rows"], res, data["nodes_per_mode"]) == []


def test_one_off_noise_hidden_when_only_the_diagnosis_recurs():
    res = vf.NameResolver(None, {"box": [(6, 1, 1)] * 3, "left_sole": [(1, 1)] * 3})
    res.objects = {"box"}
    res.face_env = {1: "floor surface"}
    res.face_self = {"box": {6: "bottom"}}
    solve = [vf.Violation("kso", 2, "box", "gap", 0.4, "m", other="floor surface"),
             vf.Violation("kso", 2, "box", "collision", 0.05, "m", other="step")]
    noise = [vf.Violation("kso", 1, "left_sole", "slip", 0.1, "m", other="floor surface")]
    lines = vf.summarize(vf.merge_attempts([solve, solve + noise, solve]), res, n_attempts=3)
    assert "ends up on top of step" in lines[0]
    assert not any("1 of 3" in ln for ln in lines)
    assert any("only one solve" in ln for ln in lines)


def test_released_contact_falls_back_to_previous_mode():
    """KSO reports the union of incoming/outgoing contacts at a transition,
    so a violation at the release mode names the contact being released."""
    res = vf.NameResolver(None, {"left_ee": [(), (3, 1), ()]})
    res.face_env = {3: "box left"}
    assert res.contact("left_ee", 2) == "box left"
    assert res.contact("left_ee", 0) == ""


def _plain_resolver(plan, objects=("box",), face_env=None, boxes=None):
    res = vf.NameResolver(None, plan)
    res.objects = set(objects)
    res.face_env = face_env or {1: "floor surface", 2: "bed top", 3: "box left", 5: "box right"}
    res.face_owner = {3: "box", 5: "box"}
    res.face_self = {"box": {6: "bottom"}}
    res.boxes = boxes or {}
    return res


def test_object_dragged_before_any_contact_is_reported_as_reach():
    plan = {"left_ee": [(), (), (3, 1)], "right_ee": [(), (), (5, 1)],
            "box": [(6, 2, 1)] * 3}
    res = _plain_resolver(plan)
    dragged = vf.Violation("kso", 0, "box", "slip", 0.36, "m", other="bed top")
    held = vf.Violation("kso", 2, "box", "slip", 0.05, "m", other="bed top")
    lines = vf.summarize([dragged, held], res, hints=True)
    assert any("box moves 0.36 m on bed top before anything touches it" in ln for ln in lines)
    assert any("box slides 0.05 m along bed top" in ln for ln in lines)
    assert any(ln.startswith("Hint (reach)") for ln in lines)


def test_resting_on_wrong_body_includes_clearance_geometry():
    plan = {"box": [(6, 1, 1)] * 2}
    boxes = {"step": (0.6, 0.0, 0.225, 0.2, 0.5, 0.225), "box": (0.5, 0.0, 0.6, 0.2, 0.2, 0.15)}
    res = _plain_resolver(plan, boxes=boxes)
    res.fixed_boxes = {"step"}
    gap = vf.Violation("kso", 1, "box", "gap", 0.40, "m", other="floor surface")
    coll = vf.Violation("kso", 1, "box", "collision", 0.05, "m", other="step")
    (line, *_rest) = vf.summarize([gap, coll], res, hints=False)
    assert "ends up on top of step" in line
    assert "step spans x 0.40–0.80, y -0.50–0.50" in line
    assert "footprint is 0.40 x 0.40" in line and "at least 0.20 m outside" in line


def test_kso_feedback_headline_and_bullets_describe_the_same_prefix():
    plan = {"left_ee": [()] * 20 + [(3, 1)] * 14, "box": [(6, 2, 1)] * 34}
    res = _plain_resolver(plan)
    diag = {
        "kind": "prefix", "total_modes": 34, "last_feasible_modes": 20,
        "first_infeasible_modes": 21, "break_event": "left_ee -> box left",
        "prefix_tries": 4, "kso_attempts": 3,
        "prefix_violation_details": [vf.Violation("kso", 20, "left_ee", "gap", 0.31, "m",
                                                   other="box left", attempts=4).to_dict()],
        "violation_details": [vf.Violation("kso", 33, "box", "gap", 0.4, "m",
                                            other="floor surface", attempts=3).to_dict()],
        "probe": {"fixed_by": "walk(plan, 2) inserted right before mode 20 (left_ee -> box left)",
                  "tried": [], "skipped": None},
    }
    headline, lines = vf.kso_feedback(diag, res)
    assert headline == ("KSO infeasible: first break at mode 20 of 34 (left_ee -> box left); "
                        "modes 0–19 solve on their own.")
    text = "\n".join(lines)
    assert "The 21-mode prefix failed 4 of 4 tries" in text
    assert "mode 20: left_ee should touch box left but stays 0.31 m away" in text
    assert "Full-plan solves additionally report" in text
    assert "Repair probe: walk(plan, 2) inserted right before mode 20" in text


def test_kso_feedback_goal_missed_and_schedule_error():
    res = _plain_resolver({"box": [(6, 1, 1)] * 2})
    headline, lines = vf.kso_feedback({
        "kind": "goal_missed", "total_modes": 30,
        "goal_residuals": {"box": {"final": [0.62, -0.35, 0.15], "target": [0.1, -0.9, 0.15],
                                   "distance": 0.76, "tolerance": 0.05}},
    }, res)
    assert headline.startswith("KSO: every contact in the plan is feasible")
    assert "- box ends at (0.62, -0.35, 0.15), 0.76 m from its target (0.10, -0.90, 0.15) (tolerance 0.05 m)" in lines
    headline, lines = vf.kso_feedback({
        "kind": "schedule_error", "total_modes": 5,
        "errors": ["mode 1->2: left_ee changes from (3, 1) to (4, 1) without a free () mode in between"],
    }, res)
    assert headline.startswith("Invalid contact schedule (no solver run): mode 1->2")
    assert lines[-1].startswith("Every channel (hands, feet, objects)")


def test_score_feedback_line():
    line = vf.score_feedback(0.153, 0.93, 0.18, {"stack_xy_offset": 0.07, "goal_distance": 0.3},
                             0.041, 820.0)
    assert line.startswith("Score 0.153 = 0.1 + 0.9 x task_quality 0.93 x to_quality 0.18.")
    assert "goal_distance 0.300" in line and "control jerk dominates" in line


def test_clearance_for_a_movable_blocker_does_not_quote_its_initial_span():
    plan = {"box": [(6, 1, 1)] * 2}
    boxes = {"other_box": (2.3, -0.4, 1.0, 0.13, 0.13, 0.1), "box": (0.5, 0.0, 0.6, 0.2, 0.2, 0.15)}
    res = _plain_resolver(plan, boxes=boxes)
    gap = vf.Violation("kso", 1, "box", "gap", 0.12, "m", other="floor surface")
    coll = vf.Violation("kso", 1, "box", "collision", 0.25, "m", other="other_box")
    (line, *_rest) = vf.summarize([gap, coll], res, hints=False)
    assert "spans" not in line and "0.26 x 0.26 and sits wherever the plan last put it" in line


def test_large_step_down_gap_is_explained():
    plan = {"left_sole": [(2, 1), (2, 1), (2, 1)], "right_sole": [(2, 1), (), (1, 1)]}
    res = vf.NameResolver(None, plan)
    res.face_env = {1: "floor surface", 2: "table top"}
    res.face_heights = {1: 0.0, 2: 0.73}
    assert res.contact_face_id("right_sole", 1) == 2  # released there: previous contact
    assert res.contact_face_id("right_sole", 2) == 1
    gap = vf.Violation("kso", 2, "right_sole", "gap", 0.22, "m", other="floor surface")
    lines = vf.summarize([gap], res, hints=False)
    assert len(lines) == 1
    assert "drop 0.73 m from table top to floor surface" in lines[0]
    assert "near the legs' limit" in lines[0] and "two walking steps on table top" in lines[0]
    # A small step is an ordinary gap.
    res.face_heights = {1: 0.0, 2: 0.3}
    assert "legs' limit" not in vf.summarize([gap], res, hints=False)[0]
