"""Solver-backed grasp feedback (loco_manipulation/grasp_probe.py) with the
static IK stubbed: which window mode is probed, which face pairs are
offered, and the line the LLM gets."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import grasp_probe as gp  # noqa: E402

# Side-resting box (0.72 x 0.90 x 0.40 m): front face on the floor, the
# plan grips left (15) + right (17), 0.90 m apart.
OWN = [(14, 1, "top"), (15, 2, "left"), (16, 3, "front"), (17, 4, "right"), (18, 5, "back"), (19, 6, "bottom")]
BY_NAME = {("parcel", face): eid for eid, _sid, face in OWN}
BY_NAME[("floor", "surface")] = 1
F = (1, 1)


def _plan():
    n = 8
    plan = {"left_sole": [F] * n, "right_sole": [F] * n, "left_ee": [()] * n, "right_ee": [()] * n,
            "parcel": [(3, 1, 1)] * n}             # (own face 3 = front, env face 1 = floor, type)
    plan["left_ee"][5] = plan["left_ee"][6] = (15, 1)
    plan["right_ee"][6] = (17, 1)
    plan["parcel"][7] = ()
    return plan


def _task():
    scene = SimpleNamespace(env_objects={"parcel": SimpleNamespace(half_extents=[0.36, 0.45, 0.2], mass=2.0)},
                            faces=SimpleNamespace(by_name=BY_NAME), robots={"r": SimpleNamespace(nq=3)})
    return SimpleNamespace(scene=scene, robot_id="r")


def _solve(a, e, n=8, npm=2, lookahead=0):
    return SimpleNamespace(segments=[{"modes": [0, a], "status": 0, "physical_pass": None},
                                     {"modes": [a, e], "status": 1, "physical_pass": False,
                                      "extra": {"window": [a, min(n, e + lookahead)]}}],
                           num_modes=n, num_nodes=n * npm,
                           results={"robot_xs": {"r": np.arange(n * npm + 1.0)[:, None] * np.ones(4)},   # q at node k = [k, k, k]
                                    "obj_xs": {"parcel": np.zeros((n * npm + 1, 13))}})


def test_grasp_mode_is_the_first_mode_with_the_most_hands_on_one_object():
    faces = {eid: ("parcel", face) for eid, _sid, face in OWN}
    plan = _plan()
    assert gp.grasp_mode(plan, 4, 8, faces) == 6          # both hands from mode 6
    assert gp.grasp_mode(plan, 4, 6, faces) == 5          # one hand only in [4, 6)
    assert gp.grasp_mode(plan, 0, 4, faces) is None
    assert gp.hands_on(plan, 6, faces) == {"left_ee": ("parcel", "left"), "right_ee": ("parcel", "right")}


def test_alternatives_skip_the_held_pair_and_faces_on_a_support():
    alts = gp.alternatives(["left", "right"], ["front"], [0.36, 0.45, 0.2])
    assert alts == [("top", "bottom", 0.4)]                # front/back excluded: front rests on the floor
    assert gp.alternatives(["top", "bottom"], [], [0.36, 0.45, 0.2]) == [("front", "back", 0.72), ("left", "right", 0.9)]


def test_feedback_names_the_unreachable_grasp_and_a_pair_a_pose_exists_for(monkeypatch):
    monkeypatch.setattr(gp, "object_faces", lambda scene, obj: OWN)
    calls = []

    def fake_probe(task, plan, mode, guess):
        pair = frozenset({plan["left_ee"][mode][0], plan["right_ee"][mode][0]})
        calls.append((mode, sorted(pair), guess[0].tolist()))
        return 0 if pair == frozenset({14, 19}) else 1    # only top + bottom has a pose

    lines = gp.feedback(_task(), _plan(), _solve(4, 8), probe_fn=fake_probe)
    assert len(lines) == 1
    assert lines[0].startswith("mode 6: no standing pose exists in which the robot holds parcel on its left and right faces (0.90 m apart)")
    assert "A standing pose does exist holding its top and bottom faces (0.40 m apart): grip those instead." in lines[0]
    assert calls[0][0] == 6 and calls[0][1] == [15, 17] and calls[0][2] == [12.0, 12.0, 12.0]   # seeded at the grasp node, nq = 3
    assert calls[1][1] == [15, 17] and calls[1][2] == [8.0, 8.0, 8.0]                           # retried from the window start
    assert calls[2][1] == [14, 19] and len(calls) == 3
    # a pose exists for the planned grasp -> nothing to say
    assert gp.feedback(_task(), _plan(), _solve(4, 8), probe_fn=lambda *a: 0) == []
    # no hand on an object in the failing window
    assert gp.feedback(_task(), _plan(), _solve(1, 4), probe_fn=fake_probe) == []
    # the grasp lies in the window's lookahead part: still probed, seeded at the grasp node
    calls.clear()
    assert gp.feedback(_task(), _plan(), _solve(2, 5, lookahead=3), probe_fn=fake_probe)[0].startswith("mode 6:")
    assert calls[0][2] == [12.0, 12.0, 12.0]
    # the probe could not run
    assert gp.feedback(_task(), _plan(), _solve(4, 8), probe_fn=lambda *a: None) == []
    # no alternative pose: generic advice, within the solve budget (2 seeds x 2 grasps)
    n_calls = []
    out = gp.feedback(_task(), _plan(), _solve(4, 8), probe_fn=lambda *a: n_calls.append(1) or 1)
    assert out and out[0].endswith("support it from below or stage it on another face first.")
    assert len(n_calls) == gp.MAX_PROBES


def test_an_adjacent_grip_skips_its_own_probe_and_names_a_spannable_pair(monkeypatch):
    monkeypatch.setattr(gp, "object_faces", lambda scene, obj: OWN)
    plan = _plan()
    plan["right_ee"][6] = (19, 1)                        # left + bottom: adjacent faces
    calls = []

    def fake_probe(task, p, mode, guess):
        pair = frozenset({p["left_ee"][mode][0], p["right_ee"][mode][0]})
        calls.append(sorted(pair))
        return 0 if pair == frozenset({14, 19}) else 1

    lines = gp.feedback(_task(), plan, _solve(4, 8), probe_fn=fake_probe)
    assert calls == [[14, 19]]                            # straight to the opposite pairs (top + bottom first)
    assert lines[0].startswith("mode 6: parcel is held on its left and bottom faces (not opposite faces) (static IK, 1 solve")
    assert "A standing pose does exist holding its top and bottom faces (0.40 m apart): grip those instead." in lines[0]
