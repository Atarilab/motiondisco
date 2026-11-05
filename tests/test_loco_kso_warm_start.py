"""Level-aware KSO warm start (loco_manipulation/kso_warm_start.py): state
bookkeeping and the per-state guess, on synthetic geometry (no faro)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import kso_warm_start as ws  # noqa: E402

FLOOR, TABLE = 1, 2
L, R = ("ee", "robot", "left_sole"), ("ee", "robot", "right_sole")
CENTRES = {FLOOR: np.array([0.0, 0.0, 0.0]), TABLE: np.array([-0.7, 0.0, 0.73])}
START = np.array([-1.7, 0.0])
GOAL = np.array([0.7, 0.0])
STANDING = 0.77


def _plan(step_up=True, step_down=True, table_steps=2):
    """Flat-floor walk, optional climb, optional descent, as mode lists.
    Like env.walk, a lifted foot returns to the surface it last stood on."""
    left, right = [(FLOOR, 1)], [(FLOOR, 1)]
    on = {"l": (FLOOR, 1), "r": (FLOOR, 1)}

    def step(l_, r_):
        left.append(l_)
        right.append(r_)
        on["l"], on["r"] = l_ or on["l"], r_ or on["r"]

    def walk(n):
        for i in range(n):
            step(on["l"], ()) if i % 2 == 0 else step((), on["r"])
        step(on["l"], on["r"])

    def level_change(face):
        step(on["l"], ())
        step(on["l"], (face, 1))
        step((), on["r"])
        step((face, 1), on["r"])

    walk(2)
    if step_up:
        level_change(TABLE)
        walk(table_steps)
    if step_down:
        level_change(FLOOR)
        walk(2)
    return {L: left, R: right}


def test_state_modes_cover_initial_transitions_and_terminal():
    assert ws.state_modes(4, True, True) == [(0,), (0, 1), (1, 2), (2, 3), (3,)]
    assert ws.state_modes(4, False, False) == [(0, 1), (1, 2), (2, 3)]


def test_sole_faces_is_the_union_over_the_modes():
    cms = {L: [(FLOOR, 1), (FLOOR, 1)], R: [(FLOOR, 1), ()], ("ee", "robot", "left_ee"): [(TABLE, 1)] * 2}
    assert ws.sole_faces(cms, (0, 1)) == {"left_sole": FLOOR, "right_sole": FLOOR}
    assert ws.sole_faces(cms, (1,)) == {"left_sole": FLOOR}


def test_flat_plan_is_left_alone():
    cms = _plan(step_up=False, step_down=False)
    stages = ws.state_modes(len(cms[L]), True, True)
    assert not any(ws.plan_guess(stages, cms, CENTRES, START, GOAL, STANDING))


def test_climb_faces_the_table_from_the_start_and_descends_backwards():
    cms = _plan()
    stages = ws.state_modes(len(cms[L]), True, True)
    guesses = ws.plan_guess(stages, cms, CENTRES, START, GOAL, STANDING)
    straddles = [i for i, g in enumerate(guesses) if g is not None and g.legs]
    assert len(straddles) == 4  # two states per level change
    up, down = straddles[:2], straddles[2:]
    for i in up:
        assert guesses[i].yaw == pytest.approx(0.0)
        assert guesses[i].legs == {"right": ws.FOLDED_LEG, "left": ws.STRAIGHT_LEG}
        assert guesses[i].pelvis_z == pytest.approx(min(0.80, 0.73 + ws.MIN_LEG_FOLD))
    for i in down:
        assert abs(guesses[i].yaw) == pytest.approx(math.pi)
        assert guesses[i].legs == {"left": ws.FOLDED_LEG, "right": ws.STRAIGHT_LEG}
    # On the table the pelvis stands at table height and the heading turns
    # gradually from the climb's facing to the descent's.
    on_table = range(up[-1] + 1, down[0])
    yaws = [guesses[i].yaw for i in on_table]
    assert all(guesses[i].pelvis_z == pytest.approx(0.73 + STANDING) for i in on_table)
    assert all(abs(a) <= abs(b) + 1e-9 for a, b in zip(yaws, yaws[1:]))
    assert 0.0 < abs(yaws[0]) < abs(yaws[-1]) < math.pi
    # After the last step down the heading is held.
    tail = [guesses[i] for i in range(down[-1] + 1, len(guesses))]
    assert all(abs(g.yaw) == pytest.approx(math.pi) and g.pelvis_z == pytest.approx(STANDING) for g in tail)


def test_descent_without_goal_turns_back_toward_the_start():
    cms = _plan()
    stages = ws.state_modes(len(cms[L]), True, False)
    guesses = ws.plan_guess(stages, cms, CENTRES, START, None, STANDING)
    down = [i for i, g in enumerate(guesses) if g is not None and g.legs][2:]
    assert all(guesses[i].yaw == pytest.approx(0.0) for i in down)


def test_small_steps_are_ignored():
    centres = {**CENTRES, TABLE: np.array([-0.7, 0.0, 0.2])}
    cms = _plan()
    stages = ws.state_modes(len(cms[L]), True, True)
    assert not any(ws.plan_guess(stages, cms, centres, START, GOAL, STANDING))


def test_apply_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.setenv("SHINKA_KSO_WARM_START", "0")
    assert ws.apply(object(), {}) == 0
    monkeypatch.setenv("SHINKA_KSO_WARM_START", "1")
    assert ws.apply(object(), {}) == 0  # broken task: swallowed, logged


def test_face_centres_offsets_box_top_and_bottom():
    class Cfg:
        half_extents = (0.35, 0.7, 0.365)

    class State:
        q = np.array([-0.7, 0.0, 0.365, 0, 0, 0, 1])

    class Faces:
        by_id = {1: ("floor", 1), 2: ("table", 1), 7: ("table", 6), 3: ("table", 2)}

    class Scene:
        faces = Faces()
        objects = {"table": Cfg(), "floor": type("P", (), {"half_extents": [1.9, 0.7]})()}

    class Resolved:
        fixed_env_objects = {"table": State(), "floor": type("S", (), {"q": np.zeros(7)})()}
        env_objects = {}

    c = ws.face_centres(Scene(), Resolved())
    assert c[2][2] == pytest.approx(0.73) and c[7][2] == pytest.approx(0.0)
    assert c[3][2] == pytest.approx(0.365) and c[1][2] == pytest.approx(0.0)


def test_goal_falls_back_to_the_goal_objects_targets():
    """A task that only places objects (box past the table) still gives the
    descent a direction: the centre of the object targets."""
    import numpy as np
    from types import SimpleNamespace as NS

    robot_goal = NS(terminal_goal=NS(robots={"r": NS(value=[1.0, 2.0, 0.7])}, objects={"b": NS(value=[5.0, 5.0, 0.1])}))
    assert ws._goal_xy(robot_goal, "r").tolist() == [1.0, 2.0]
    objects_only = NS(terminal_goal=NS(robots={}, objects={"a": NS(value=[1.9, 0.1, 0.1, 0, 0, 0, 1]),
                                                           "b": NS(value=[2.1, -0.1, 0.4])}))
    assert np.allclose(ws._goal_xy(objects_only, "r"), [2.0, 0.0])
    assert ws._goal_xy(NS(terminal_goal=None), "r") is None
    assert ws._goal_xy(NS(terminal_goal=NS(robots={}, objects={})), "r") is None
