"""Physical trajectory checks (loco_manipulation/trajectory_check.py): the
pure metrics, without faro / pinocchio / MuJoCo."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import trajectory_check as tc  # noqa: E402


def test_power_excess_matches_faros_limit():
    tau_max, vel_max = np.array([25.0, 139.0]), np.array([37.0, 20.0])
    # within the limit, at the torque limit with no speed, and over it
    tau = np.array([[10.0, 0.0], [25.0, 0.0], [24.9, 140.0]])
    v = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.0]])
    ex = tc.power_excess(tau, v, tau_max, vel_max)
    assert ex[0] == pytest.approx([0.0, 0.0]) and ex[1] == pytest.approx([0.0, 0.0])
    assert ex[2][0] == pytest.approx(24.9 + 25 / 37 * 0.5 - 25)
    assert ex[2][1] == pytest.approx(1.0)


def test_contact_force_pull_and_friction_pyramid():
    n, t1, t2 = np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    assert tc.contact_force_metrics(np.array([0.0, 0, 100]), n, t1, t2) == (0.0, 0.0)
    # pyramid: each tangential axis is bounded by mu f_n on its own
    assert tc.contact_force_metrics(np.array([50.0, 50, 100]), n, t1, t2) == (0.0, 0.0)
    pull, fric = tc.contact_force_metrics(np.array([60.0, 0, 100]), n, t1, t2)
    assert pull == 0.0 and fric == pytest.approx(10.0)
    pull, fric = tc.contact_force_metrics(np.array([5.0, 0, -20]), n, t1, t2)
    assert pull == pytest.approx(20.0) and fric == pytest.approx(5.0)


def test_face_geometry_of_a_rotated_box_and_a_plane():
    rot = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])   # yaw +90 deg
    c, n, t1, t2 = tc.face_geometry(np.zeros(3), rot, (0.1, 0.2, 0.3), "front")
    assert n == pytest.approx([0, 1, 0]) and c == pytest.approx([0, 0.1, 0])
    c, n, _, _ = tc.face_geometry(np.zeros(3), np.eye(3), (0.1, 0.2, 0.3), "right")
    assert n == pytest.approx([0, -1, 0]) and c == pytest.approx([0, -0.2, 0])
    c, n, _, _ = tc.face_geometry(np.array([0, 0, 0.0]), np.eye(3), (10, 10, 0), "surface")
    assert n == pytest.approx([0, 0, 1]) and c == pytest.approx([0, 0, 0])


def test_verdict_and_mode_of_node():
    ok = {k: 0.0 for k in tc.THRESHOLDS}
    assert tc.verdict(ok) == {"pass": True, "fails": []}
    bad = dict(ok, penetration=0.05, pull_force=1.0)
    assert tc.verdict(bad) == {"pass": False, "fails": ["penetration"]}
    assert [tc.mode_of_node(k, 20, 3) for k in (0, 19, 20, 59, 60)] == [0, 0, 1, 2, 2]


def test_normalized_and_describe_name_the_worst_failure_first():
    worst = {k: 0.0 for k in tc.THRESHOLDS}
    worst.update(penetration=0.08, contact_gap=0.03)
    rep = {"worst": worst, "fails": ["contact_gap", "penetration"], "mass": 35.0, "nodes_per_mode": 20,
           "where": {"penetration": ["node", 125, "white_box", "yellow_box"],
                     "contact_gap": ["mode", 9, "left_ee", "banana", "left"]}}
    assert tc.normalized(worst) == pytest.approx(0.08 / tc.THRESHOLDS["penetration"])
    lines = tc.describe(rep)
    assert lines[0] == "mode 6: white_box penetrates yellow_box by 8 cm in the trajectory"
    assert "left_ee stays 3 cm off banana left" in lines[1]


def test_to_solve_validity_and_scored_violation():
    to_eval = pytest.importorskip("to_eval")
    ok = {k: 0.0 for k in tc.THRESHOLDS}
    base = dict(cost=1.0, results={}, num_modes=3, num_nodes=60, max_constraint_violation=2.5)
    conv = to_eval.ToSolve(status=0, converged=True, **base)
    phys = to_eval.ToSolve(status=1, converged=False, physical={"pass": True, "worst": dict(ok, penetration=0.015)}, **base)
    bad = to_eval.ToSolve(status=1, converged=False, physical={"pass": False, "worst": ok}, **base)
    assert conv.valid and phys.valid and not bad.valid
    assert conv.effective_max_violation == 2.5 and bad.effective_max_violation == 2.5
    # half the penetration threshold -> half of faro's 0.05 tolerance
    assert phys.effective_max_violation == pytest.approx(0.025)
