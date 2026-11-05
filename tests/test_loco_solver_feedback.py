"""The TO part of evaluate._solver_feedback on a stubbed solve: a numerical
failure (status 2) must not present the wrecked iterate's constraint
violations as properties of the plan (shelf_unstack gen_29: soles
"needing" 19 N of friction in a plain walking window)."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import evaluate  # noqa: E402
import to_eval  # noqa: E402

F = (1, 1)
PLAN = {"left_sole": [F] * 6, "right_sole": [F, (), F, (), F, F], "left_ee": [()] * 6, "right_ee": [()] * 6,
        "box": [(6, 2, 1)] * 6}
ROW = {"solver": "to", "mode": 1, "subject": "left_sole", "family": "friction", "value": 19.1, "unit": "N",
       "sign": 1.0, "other": "floor surface", "raw": "x", "attempts": 1}
TASK = SimpleNamespace(scene=None, resolved_state=None, robot_id="r", terminal_goal=None)


def _solve(status, segments):
    return to_eval.ToSolve(status=status, converged=False, cost=1.0, results={}, num_modes=6, num_nodes=120,
                           violations=[ROW], fail_reason="segment 0 modes [0,4): collision: restoration_failed",
                           segments=segments, solved_modes=0, kso_feasible=True)


def test_numerical_failure_feedback_lists_no_wreck_violations():
    segs = [{"modes": [0, 4], "status": 2, "fail_reason": "collision: restoration_failed", "physical_pass": None,
             "extra": {"window": [0, 6]}},
            {"modes": [4, 6], "status": None, "physical_pass": None, "extra": {}}]
    _headline, block = evaluate._solver_feedback(TASK, PLAN, None, [], _solve(2, segs))
    assert "failed numerically" in block and "not a trajectory" in block
    assert "Largest violations" not in block and "friction" not in block
    assert "in segment 1 of 2 (modes 0–5)" in block
    # a status-2 window whose wreck passed the physical check is still the failing one
    segs[0]["physical_pass"] = True
    assert evaluate._segment_failure_text(_solve(2, segs)) == " in segment 1 of 2 (modes 0–5)"


def test_a_plain_non_convergence_still_lists_its_violations():
    segs = [{"modes": [0, 4], "status": 1, "fail_reason": "", "physical_pass": False, "extra": {"window": [0, 6]}},
            {"modes": [4, 6], "status": None, "physical_pass": None, "extra": {}}]
    _headline, block = evaluate._solver_feedback(TASK, PLAN, None, [], _solve(1, segs))
    assert "Largest violations" in block and "friction" in block and "not a trajectory" not in block
