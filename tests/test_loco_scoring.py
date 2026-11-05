"""Quality-mode score bands (loco_manipulation/scoring.py): a failed
segmented TO climbs with the modes it solved, but never reaches the
valid-trajectory band."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import scoring  # noqa: E402


def _score(**kw):
    base = dict(goal_match=True, n_modes=70, kso_feasible=True, kso_diagnosis=None, kso_task_quality=0.7,
                to_ran=True, to_disabled=False, to_converged=False, to_cost=3.0, to_task_quality=0.7,
                to_quality_value=0.05)
    base.update(kw)
    return scoring.quality_score(**base)


def test_partial_to_credit_stays_between_the_failed_and_valid_bands():
    failed = _score()
    assert failed == pytest.approx(0.1)
    partial = [_score(to_progress=p) for p in (0.0, 0.25, 0.5, 1.0)]
    assert partial[0] == pytest.approx(0.1) and partial == sorted(partial)
    assert partial[-1] == pytest.approx(0.105)
    # The weakest valid trajectory seen in the long-horizon runs (three_shelf
    # gen_17: task quality 0.676, TO quality 0.014) scored 0.1085.
    weakest_valid = _score(to_converged=True, to_task_quality=0.676, to_quality_value=0.014)
    assert partial[-1] < weakest_valid == pytest.approx(0.1085, abs=1e-4)
    # A KSO failure stays below every TO result; progress is clipped.
    assert _score(kso_feasible=False, kso_diagnosis={"last_feasible_modes": 60, "total_modes": 70}) < failed
    assert _score(to_progress=7.0) == pytest.approx(0.105)
    # No TO run: no credit from progress.
    assert _score(to_ran=False, to_progress=0.5) == pytest.approx(0.1)
