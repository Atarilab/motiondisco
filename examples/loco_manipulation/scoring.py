"""
Score modes for loco_manipulation plans.

``legacy`` (default) is the sharp gate the CoRL ablations use::

    combined_score = 1 / (1 + to_cost)   if TO ran and to_cost > 0
                   = 0                    otherwise

``quality`` (opt-in, ``task.score_mode=quality``) fixes three blind spots
of the legacy rule — it ignores TO convergence, ignores how well the
task was achieved, and normalizes constraint violation by plan length
(so padding a plan dilutes a violation). The worst single violation is
used instead: it is not diluted by extra nodes, yet stays on the same
scale for any horizon (the raw sum grows with it and would crush
to_quality towards 0 for every real plan). Bands, highest first::

    TO converged             0.1 + 0.9 * task_quality * to_quality
    KSO feasible, TO failed  0.1           (not converged, error, cost 0)
    ... partway              0.1 * (1 + 0.05 * solved_modes / n_modes)
                             (segmented TO stopped at a failing segment:
                             0.100-0.105, below any valid trajectory seen,
                             whose lowest score was 0.1085)
    KSO feasible, TO off     0.1 + 0.9 * task_quality   (SHINKA_DISABLE_TO)
    KSO goal missed          0.1 * (0.9 + 0.1 / (1 + miss distance))
                             (every contact works without the goal target)
    KSO infeasible           0.1 * 0.9 * events_feasible / events_total
                             (contact events — grasps, releases, placements,
                             steps onto another surface — before the break;
                             walking steps do not count, so padding a plan
                             does not raise it; falls back to the mode
                             fraction when the plan has no events)
    goal mismatch, < 2 modes 0

    to_quality   = 1 / (1 + 1e2 * max_constraint_violation
                          + 1e-2 * jerk_cost / num_modes)
    task_quality = scene outcome scorer, mapped to (0, 1]:
        lower-is-better scorers (goal residuals):  1 / (1 + value)
        higher-is-better scorers (box height):     value / (value + baseline)
                                                   baseline = value at the
                                                   scene's initial state
        no scorer:                                 1

Scorer direction comes from ``SCORER_DIRECTION`` (keyed by the scene's
``OUTCOME_SCORERS`` name); unlisted scorers are treated as costs, which
matches faro's own MCTS (it minimizes them). Every non-legacy score is
strictly ordered: any converged TO beats any TO failure, which beats any
KSO failure.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

SCORE_MODE_ENV = "SHINKA_SCORE_MODE"
SCORE_MODES = ("legacy", "quality")

# Outcome scorers whose value should be maximized. Everything else is a
# residual / cost (lower is better).
SCORER_DIRECTION = {
    "box_stack_height": "higher",
}

PARTIAL_BAND = 0.1
# Width, relative to PARTIAL_BAND, of the band a failed segmented TO solve
# climbs with the fraction of the plan it solved (see quality_score).
TO_PROGRESS_BAND = 0.05


def score_mode() -> str:
    mode = (os.environ.get(SCORE_MODE_ENV) or "legacy").strip().lower()
    if mode not in SCORE_MODES:
        raise ValueError(f"{SCORE_MODE_ENV}={mode!r}; expected one of {SCORE_MODES}")
    return mode


def legacy_score(to_cost: Optional[float]) -> float:
    if to_cost is not None and to_cost > 0.0:
        return 1.0 / (1.0 + to_cost)
    return 0.0


def to_quality(max_constraint_violation: float, jerk_cost: float, num_modes: int) -> float:
    return 1.0 / (
        1.0 + 1e2 * max_constraint_violation + 1e-2 * jerk_cost / max(num_modes, 1)
    )


def _static_results(task: Any) -> Dict[str, Any]:
    """Single-sample results at the scene's resolved initial state."""
    rs = task.resolved_state
    return {
        "robot_xs": {rid: st.q[None, :] for rid, st in rs.robots.items()},
        "obj_xs": {name: st.q[None, :] for name, st in rs.env_objects.items()},
    }


def task_quality(task: Any, results: Optional[Dict[str, Any]]) -> Optional[float]:
    """Outcome-scorer value of *results* mapped to (0, 1].

    Returns 1.0 for scenes without a scorer and None when *results* is
    missing or the scorer fails on it.
    """
    if task.outcome_scorer is None:
        return 1.0
    if results is None:
        return None
    from faro.search.goal import SolverOutcome

    def evaluate(res: Dict[str, Any]) -> float:
        return float(task.outcome_scorer(
            SolverOutcome(converged=True, results=res, scene=task.scene)
        ))

    try:
        value = evaluate(results)
    except Exception as e:  # noqa: BLE001 — a scorer bug must not kill eval
        print(f"[scoring] outcome scorer failed: {type(e).__name__}: {e}")
        return None
    if SCORER_DIRECTION.get(task.outcome_scorer_name) == "higher":
        baseline = max(evaluate(_static_results(task)), 1e-6)
        return max(value, 0.0) / (max(value, 0.0) + baseline)
    return 1.0 / (1.0 + max(value, 0.0))


def quality_score(
    *,
    goal_match: bool,
    n_modes: int,
    kso_feasible: Optional[bool],
    kso_diagnosis: Optional[Dict[str, Any]],
    kso_task_quality: Optional[float],
    to_ran: bool,
    to_disabled: bool,
    to_converged: Optional[bool],
    to_cost: Optional[float],
    to_task_quality: Optional[float],
    to_quality_value: Optional[float],
    to_progress: Optional[float] = None,
) -> float:
    """``quality``-mode combined score (see module docstring).
    ``to_progress`` is the fraction of the plan a segmented TO solve got
    through before it stopped (``ToSolve.solved_modes / num_modes``)."""
    if not goal_match or n_modes < 2:
        return 0.0
    if not kso_feasible:
        diag = kso_diagnosis or {}
        if diag.get("kind") == "goal_missed":
            residuals = diag.get("goal_residuals") or {}
            miss = max((float(r.get("distance") or 0.0) for r in residuals.values()), default=0.0)
            return PARTIAL_BAND * (0.9 + 0.1 / (1.0 + miss))
        ev_total, ev_ok = diag.get("events_total"), diag.get("events_feasible")
        if ev_total:
            progress = float(ev_ok or 0) / float(ev_total)
        else:
            last, total = diag.get("last_feasible_modes"), diag.get("total_modes")
            progress = (last / total) if last and total else 0.0
        return PARTIAL_BAND * 0.9 * min(1.0, progress)
    if to_disabled:
        return PARTIAL_BAND + (1 - PARTIAL_BAND) * (kso_task_quality or 0.0)
    if (
        not to_ran
        or not to_converged
        or not to_cost
        or to_quality_value is None
        or to_task_quality is None
    ):
        if to_ran and to_progress:
            return PARTIAL_BAND * (1.0 + TO_PROGRESS_BAND * max(0.0, min(1.0, float(to_progress))))
        return PARTIAL_BAND
    return PARTIAL_BAND + (1 - PARTIAL_BAND) * to_task_quality * to_quality_value


def quality_parts(task: Any, results: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Named residuals behind the outcome scorer, when it exposes
    ``parts(outcome)`` (e.g. StackAtGoal); None otherwise."""
    scorer = task.outcome_scorer
    if scorer is None or results is None or not hasattr(scorer, "parts"):
        return None
    try:
        from faro.search.goal import SolverOutcome

        parts = scorer.parts(SolverOutcome(converged=True, results=results, scene=task.scene))
        return {str(k): float(v) for k, v in parts.items()}
    except Exception as e:  # noqa: BLE001 — feedback is best effort
        print(f"[scoring] outcome scorer parts failed: {type(e).__name__}: {e}")
        return None
