"""Grip feedback for failed TO solves (loco_manipulation/grip_feedback.py):
carried objects held on adjacent / the same faces, on synthetic plans."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import grip_feedback as gf  # noqa: E402

FACES = {1: ("floor", "surface"), 10: ("box", "top"), 11: ("box", "left"), 12: ("box", "front"),
         13: ("box", "right"), 14: ("box", "back"), 15: ("box", "bottom"), 20: ("lid", "bottom")}
MASSES = {"box": 2.0, "lid": 1.0}
F = (1, 1)


def _plan(left, right, lid_on_box=False):
    """0 resting | 1 grasped | 2-3 carried | 4 set down | 5 released."""
    return {
        "left_sole": [F] * 6, "right_sole": [F] * 6,
        "left_ee": [(), left, left, left, left, ()],
        "right_ee": [(), right, right, right, right, ()],
        "box": [(15, 1, 1), (15, 1, 1), (), (), (15, 1, 1), (15, 1, 1)],
        **({"lid": [(6, 10, 1)] * 6} if lid_on_box else {}),
    }


def test_opposite_faces_are_fine():
    assert gf.grip_issues(_plan((11, 1), (13, 1)), FACES, MASSES) == []
    assert gf.grip_issues(_plan((12, 1), (14, 1)), FACES, MASSES) == []


def test_adjacent_faces_flagged_with_carried_load():
    issues = gf.grip_issues(_plan((12, 1), (13, 1), lid_on_box=True), FACES, MASSES)
    assert len(issues) == 1
    i = issues[0]
    assert (i["object"], i["first"], i["last"], i["faces"], i["kind"]) == ("box", 2, 3, ("front", "right"), "adjacent")
    assert i["mass"] == 3.0
    line = gf.grip_lines(issues, MASSES)[0]
    assert line.startswith("modes 2–3: box is carried with left_ee on its front face and right_ee on its right face")
    assert "3.0 kg including what is stacked on it" in line
    assert "Grip two opposite faces" in line


def test_top_and_side_and_same_face():
    i = gf.grip_issues(_plan((10, 1), (14, 1)), FACES, MASSES)[0]
    assert i["faces"] == ("top", "back") and i["kind"] == "adjacent"
    i = gf.grip_issues(_plan((12, 1), (12, 1)), FACES, MASSES)[0]
    assert i["kind"] == "same"
    assert "both hands on its front face" in gf.grip_lines([i])[0]


def test_resting_object_and_one_hand_not_flagged():
    plan = _plan((12, 1), (13, 1))
    plan["box"] = [(15, 1, 1)] * 6          # never lifted: pushing, not carrying
    assert gf.grip_issues(plan, FACES, MASSES) == []
    plan = _plan((12, 1), ())
    assert gf.grip_issues(plan, FACES, MASSES) == []


def test_unreadable_task_gives_no_lines():
    assert gf.to_grip_feedback(object(), _plan((12, 1), (13, 1))) == []
