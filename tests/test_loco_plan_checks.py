"""Pure plan checks and edits (loco_manipulation): faro's release-before-
change rule, contact events, walk insertion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import plan_checks  # noqa: E402
from env import append_mode, walk  # noqa: E402


def _base(n=1):
    return {"left_sole": [(1, 1)] * n, "right_sole": [(1, 1)] * n,
            "left_ee": [()] * n, "right_ee": [()] * n, "box": [(6, 2, 1)] * n}


def test_schedule_errors_flag_direct_surface_changes_on_every_channel():
    plan = _base()
    append_mode(plan, left_ee=(3, 1))
    append_mode(plan, left_ee=(4, 1))          # hand jumps to another face
    append_mode(plan, box=(6, 1, 1))           # object teleports to the floor
    append_mode(plan, right_sole=(8, 1))       # foot steps up without lifting
    errors = plan_checks.schedule_errors(plan)
    assert [e.split(":")[0] for e in errors] == ["mode 1->2", "mode 2->3", "mode 3->4"]
    assert "left_ee changes from (3, 1) to (4, 1)" in errors[0]
    assert "box changes from (6, 2, 1) to (6, 1, 1)" in errors[1]


def test_sticking_to_sliding_on_the_same_surfaces_is_allowed():
    plan = _base()
    append_mode(plan, box=(6, 2, 2))
    append_mode(plan, box=(6, 2, 1))
    assert plan_checks.schedule_errors(plan) == []


def test_event_modes_ignore_walking_steps():
    plan = _base()
    walk(plan, 3)                              # modes 1-4: no events
    append_mode(plan, left_ee=(3, 1), right_ee=(5, 1))   # mode 5: grasp
    append_mode(plan, box=())                  # mode 6: lift
    walk(plan, 2)                              # modes 7-9
    append_mode(plan, right_sole=())           # mode 10: lift foot ...
    append_mode(plan, right_sole=(8, 1))       # mode 11: ... onto another surface
    append_mode(plan, box=(6, 1, 1))           # mode 12: place
    assert plan_checks.event_modes(plan) == [5, 6, 11, 12]
    assert plan_checks.first_grasp_mode(plan) == 5


def test_insert_walk_matches_env_walk_pattern():
    plan = _base()
    append_mode(plan, left_ee=(3, 1))          # mode 1: grasp
    append_mode(plan, box=())                  # mode 2: lift
    edited = plan_checks.insert_walk(plan, before_mode=1, n_steps=2)
    assert plan_checks.n_modes(edited) == 3 + plan_checks.walk_mode_count(2)
    assert edited["right_sole"][1:4] == [(), (1, 1), (1, 1)]
    assert edited["left_sole"][1:4] == [(1, 1), (), (1, 1)]
    assert edited["left_ee"][1:4] == [(), (), ()]      # hands keep mode 0's values
    assert edited["left_ee"][4] == (3, 1) and edited["box"][5] == ()
    assert plan_checks.schedule_errors(edited) == []
    assert all(len(seq) == len(edited["left_sole"]) for seq in edited.values())
    # Matches env.walk exactly.
    reference = _base()
    walk(reference, 2)
    assert {ch: edited[ch][:4] for ch in reference} == reference


def test_insert_walk_refuses_a_lifted_foot_and_bad_indices():
    plan = _base()
    append_mode(plan, right_sole=())
    append_mode(plan, left_ee=(3, 1))
    assert plan_checks.insert_walk(plan, before_mode=2, n_steps=2) is None  # mode 1 has a lifted foot
    assert plan_checks.insert_walk(plan, before_mode=0, n_steps=2) is None
    assert plan_checks.insert_walk(plan, before_mode=1, n_steps=0) == plan


def test_describe_transition_uses_face_names():
    plan = _base()
    append_mode(plan, left_ee=(3, 1), right_ee=(5, 2))
    append_mode(plan, box=())
    append_mode(plan, box=(6, 1, 1), left_ee=())
    names = {3: "box left", 5: "box right", 1: "floor surface"}
    describe = lambda m: plan_checks.describe_transition(  # noqa: E731
        plan, m, face_name=lambda i: names.get(i, f"face {i}"), objects={"box"})
    assert describe(1) == "left_ee -> box left, right_ee -> box right (sliding)"
    assert describe(2) == "box lifted"
    assert describe(3) == "box set down on floor surface, left_ee released"
    assert describe(0) == ""
