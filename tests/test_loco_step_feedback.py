"""Walking-step feedback for failed TO solves (loco_manipulation/step_feedback.py):
stretches between manipulation events and the suggested edit, on given
foot-travel numbers (no faro / pinocchio)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import step_feedback as sf  # noqa: E402

F, T = (1, 1), (2, 1)   # floor, table top
# 0 start | 1-3 walk | 4 grasp | 5 lift | 6-8 walk | 9 set down | 10 release
PLAN = {
    "left_sole":  [F, F, (), F, F, F, F, (), F, F, F],
    "right_sole": [F, (), F, F, F, F, (), F, F, F, F],
    "left_ee":    [(), (), (), (), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), ()],
    "right_ee":   [(), (), (), (), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), ()],
    "box":        [(6, 1, 1)] * 5 + [(), (), (), ()] + [(6, 2, 1)] * 2,
}


def test_manipulation_events_skip_walking():
    assert sf.manipulation_events(PLAN) == [4, 5, 9, 10]


def test_long_steps_reported_per_stretch_with_insert_point():
    lands = [(2, "right_sole", 0.4), (3, "left_sole", 0.7),        # before the grasp: fine
             (7, "right_sole", 1.1), (8, "left_sole", 0.95)]       # carrying: too long
    st = sf.long_step_stretches(PLAN, lands)
    assert len(st) == 1
    s = st[0]
    assert (s["first"], s["last"], s["steps"], s["long"], s["before"]) == (7, 8, 2, 2, 9)
    assert s["longest"] == (7, "right_sole", 1.1)
    assert s["extra"] == 2
    line = sf.step_lines(PLAN, st, describe=lambda m: "box set down on table top")[0]
    assert line.startswith("modes 7–8: in 2 of 2 steps a foot travels more than 0.8 m")
    assert "up to 1.10 m, right_sole at mode 7" in line
    assert "before mode 9 (box set down on table top), e.g. walk(plan, 2)" in line


def test_one_extra_step_per_long_step_worst_first_and_end_of_plan():
    lands = [(2, "right_sole", 0.85), (3, "left_sole", 0.9), (8, "left_sole", 1.3)]
    plan = {k: v[:9] for k, v in PLAN.items()}   # ends while carrying: no set-down
    st = sf.long_step_stretches(plan, lands)
    assert [s["first"] for s in st] == [8, 2]
    assert st[0]["before"] is None and st[1]["before"] == 4
    assert st[1]["extra"] == 2
    assert "at the end of the plan" in sf.step_lines(plan, st)[0]
    many = [(6, "r", 0.9), (7, "l", 0.9), (8, "r", 0.9)]
    assert sf.long_step_stretches(PLAN, many)[0]["extra"] == 3


def test_nothing_reported_for_normal_steps():
    lands = [(2, "right_sole", 0.4), (3, "left_sole", 0.8), (7, "right_sole", 0.79)]
    assert sf.long_step_stretches(PLAN, lands) == []
    assert sf.step_lines(PLAN, []) == []


def test_unreadable_solution_gives_no_lines():
    assert sf.to_step_feedback(object(), PLAN, {}) == []
