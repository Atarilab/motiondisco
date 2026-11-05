"""The terminal-goal check on TO trajectories (loco_manipulation/to_eval.py
``goal_check`` and ``ToSolve.valid``): the physical check knows nothing
about the task goal, so a non-converged but physically valid trajectory
must also end at the hard TO goal to count as valid."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import to_eval as te  # noqa: E402


def _target(value, tol):
    """A faro-like target: ``value`` and componentwise TO tolerances."""
    return SimpleNamespace(value=np.asarray(value, float), tolerance=lambda solver: np.asarray(tol, float))


def _error(target, state):
    return state[: target.value.size] - target.value


def _task(goal_mode="hard"):
    goal = SimpleNamespace(to=goal_mode, robots={},
                           objects={"box": _target([1.0, 2.0, 0.5, 0.0, 0.0, 0.0], [0.05] * 3 + [0.1] * 3)})
    return SimpleNamespace(terminal_goal=goal)


def test_goal_check_measures_the_final_state_against_the_to_tolerance():
    final = [1.03, 2.0, 0.5, 0.0, 0.0, 0.3]            # 3 cm off (within 5 cm), 0.3 rad yaw off
    res = {"obj_xs": {"box": np.array([[0.0] * 6, final])}}
    g = te.goal_check(_task(), res, slack=0.1, error_fn=_error)
    assert g["mode"] == "hard" and g["met"] is False
    assert abs(g["excess"] - 0.2) < 1e-6                 # 0.3 - 0.1 tolerance, beyond the 0.1 slack
    box = g["targets"]["box"]
    assert box["position_error"] == 0.03 and box["position_tolerance"] == 0.05
    assert box["rotation_error"] == 0.3 and box["rotation_tolerance"] == 0.1
    res["obj_xs"]["box"][-1, 5] = 0.15                    # 0.05 beyond the tolerance: within the slack
    assert te.goal_check(_task(), res, slack=0.1, error_fn=_error)["met"] is True
    assert te.goal_check(_task("off"), res, slack=0.1, error_fn=_error)["mode"] == "off"
    assert te.goal_check(SimpleNamespace(terminal_goal=None), res, error_fn=_error) is None
    assert te.goal_check(_task(), {"obj_xs": {}}, error_fn=_error) is None      # nothing checkable
    assert te.goal_check(SimpleNamespace(), res, error_fn=_error) is None      # stub task without a goal


def test_valid_requires_the_hard_goal_when_it_was_checked():
    def solve(converged, physical, goal):
        return te.ToSolve(status=0 if converged else 1, converged=converged, cost=0.0, results={},
                          num_modes=5, num_nodes=100, physical=physical, goal=goal)

    miss = {"mode": "hard", "met": False, "slack": 0.1, "excess": 0.3, "targets": {}}
    assert solve(False, {"pass": True}, dict(miss, met=True)).valid
    assert not solve(False, {"pass": True}, miss).valid
    assert not solve(False, {"pass": True}, miss).goal_met
    assert solve(False, {"pass": True}, dict(miss, mode="off")).valid     # TO goal off: not gated
    assert solve(False, {"pass": True}, None).valid                        # unchecked: as before
    assert not solve(False, {"pass": False}, dict(miss, met=True)).valid
    assert solve(True, None, None).valid and not solve(True, None, miss).valid
