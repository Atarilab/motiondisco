"""TO initial guess from a KSO solution (loco_manipulation/to_warm_start.py):
keyframe / node bookkeeping and the contact-force split, without faro."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import to_warm_start as tw  # noqa: E402


def test_keyframes_sit_on_mode_boundaries():
    # Hard initial + 3 transitions + terminal for 4 modes of 5 nodes.
    assert tw.keyframe_nodes(5, 4, 5) == [0, 5, 10, 15, 20]
    # No terminal goal: the last keyframe is the last transition.
    assert tw.keyframe_nodes(4, 4, 5) == [0, 5, 10, 15]


def test_interpolation_follows_node_times_and_holds_the_last_keyframe():
    # A short (switch) interval right after node 5.
    dt = [0.2] * 5 + [0.02] + [0.2] * 4
    times = tw.node_times(dt)
    assert times[-1] == pytest.approx(0.2 * 9 + 0.02)
    w = tw.interpolation_weights(times, [0, 5, 10])
    assert w[0] == (0, 1, 0.0) and w[5] == (1, 2, 0.0) and w[10] == (2, 2, 0.0)
    # Node 6 is only 0.02 s past keyframe 1, of 0.82 s to keyframe 2.
    assert w[6][:2] == (1, 2) and w[6][2] == pytest.approx(0.02 / 0.82)
    fr = [f for a, b, f in w[5:10]]
    assert fr == sorted(fr)
    w = tw.interpolation_weights(times, [0, 5])
    assert all(x == (1, 1, 0.0) for x in w[5:])


def test_node_modes():
    assert tw.node_modes(10, 2, 5).tolist() == [0] * 5 + [1] * 5


PLAN = {
    "left_sole": [(1, 1), (1, 1), ()],
    "right_sole": [(1, 1), (), (1, 1)],
    "left_ee": [(), (7, 1), (7, 1)],
    "right_ee": [(), (8, 1), ()],
    "box": [(6, 1, 1), (), ()],
    "lid": [(6, 9, 1), (6, 9, 1), (6, 9, 1)],   # rests on the box's top (face 9)
}
OWNER = {1: "floor", 7: "box", 8: "box", 9: "box", 12: "lid"}
MASSES = {"box": 2.0, "lid": 1.0}
NORMALS = {7: np.array([1.0, 0, 0]), 8: np.array([-1.0, 0, 0]), 9: np.array([0, 0, 1.0])}


def test_carried_load_includes_what_is_stacked_on_it():
    assert tw.carried_loads(PLAN, 0, MASSES, OWNER) == {}
    assert tw.carried_loads(PLAN, 1, MASSES, OWNER) == pytest.approx({"box": 3.0})


def test_forces_on_the_robot_for_standing_and_carrying():
    g = tw.GRAVITY
    f0 = tw.contact_forces(PLAN, 0, 30.0, MASSES, OWNER, NORMALS.get)
    assert set(f0) == {"left_sole", "right_sole"}
    assert f0["left_sole"] == pytest.approx([0, 0, g * 15.0])
    # Box (+ lid) carried with both hands: each hand is pulled down by half
    # the load and pushed out of its side face by the squeeze; the planted
    # sole carries robot + load.
    f1 = tw.contact_forces(PLAN, 1, 30.0, MASSES, OWNER, NORMALS.get)
    share = g * 1.5
    squeeze = tw.SQUEEZE_MARGIN * share / tw.MU
    assert f1["left_ee"] == pytest.approx([squeeze, 0, -share])
    assert f1["right_ee"] == pytest.approx([-squeeze, 0, -share])
    assert f1["left_sole"] == pytest.approx([0, 0, g * 33.0])
    assert "right_sole" not in f1
    # Friction holds the share with the squeeze.
    assert share <= tw.MU * squeeze
    # One hand alone: it takes the whole load.
    f2 = tw.contact_forces(PLAN, 2, 30.0, MASSES, OWNER, NORMALS.get)
    assert f2["left_ee"][2] == pytest.approx(-g * 3.0)
    # Without normals there is no squeeze.
    assert tw.contact_forces(PLAN, 1, 30.0, MASSES, OWNER)["left_ee"] == pytest.approx([0, 0, -share])


def test_hands_support_the_robot_when_no_sole_is_planted():
    plan = {"left_sole": [()], "right_sole": [()], "left_ee": [(2, 1)], "right_ee": [(2, 1)]}
    f = tw.contact_forces(plan, 0, 30.0, {}, {2: "table"})
    assert f["left_ee"] == pytest.approx([0, 0, tw.GRAVITY * 15.0])


def test_interp_pose_endpoints():
    pin = pytest.importorskip("pinocchio")
    a = np.array([0.0, 0.0, 0.5, 0, 0, 0, 1.0])
    b = np.r_[1.0, 2.0, 0.5, pin.Quaternion(pin.utils.rpyToMatrix(0, 0, 1.0)).coeffs()]
    assert tw._interp_pose(a, b, 0.0) == pytest.approx(a)
    assert tw._interp_pose(a, b, 1.0) == pytest.approx(b, abs=1e-9)
