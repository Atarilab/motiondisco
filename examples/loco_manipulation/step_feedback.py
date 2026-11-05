"""Walking-step feedback for a failed TO solve.

KSO places feet anywhere reachable, with no limit on how far a foot moves
between modes, so a plan can pass KSO while giving the robot too few
walking steps for what it has to do; TO then has to take metre-long
strides and may not converge. This module measures, in TO's own solution,
how far each sole travels every time it lands, and reports the stretches
of the plan where feet travel much further than in normal walking, with
how many walking steps to add and where.

The measure is foot travel, not body advance: with 0.4 m steps a foot
moves up to about 0.8 m when it lands (it passes the other foot). On
truck_unload_stack gen_2 the body never advanced more than 0.5 m per step,
but turning around with the boxes swept the feet up to 1.09 m; with four
more walking steps before the set-down the feet stayed under 0.7 m and TO
converged. Long steps alone do not make TO fail (many converged
long_dist_pick_place plans have them), so this is reported only when TO
did not converge, as one possible cause.

    landings(plan, robot_qs, model, nodes_per_mode)
        [(mode, sole, stride_m)] for every sole that goes free -> contact.
    long_step_stretches(plan, landings)
        stretches between manipulation events whose steps are too long.
    step_lines(plan, stretches, describe)
        plain-language feedback lines.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import plan_checks

NORMAL_STEP = 0.4   # m, the prompt's "step length about 0.4 m"
LONG_STEP = 2 * NORMAL_STEP  # m of foot travel per landing beyond which a step is long
MIN_EXTRA = 2       # suggest at least this many extra steps (one per long step otherwise)
MAX_LINES = 2

Landing = Tuple[int, str, float]


def landings(plan: Dict[str, List[Any]], robot_qs: Any, model: Any, nodes_per_mode: int) -> List[Landing]:
    """Every sole landing (free in mode j-1, in contact in mode j) with the
    xy distance from where that sole last stood, measured mid-mode in TO's
    solution (node ``nodes_per_mode * j + nodes_per_mode // 2``)."""
    import numpy as np
    import pinocchio as pin

    data = model.createData()
    qs = np.asarray(robot_qs, dtype=float)
    last: Dict[str, Any] = {}
    out: List[Landing] = []
    n = plan_checks.n_modes(plan)
    for j in range(n):
        node = min(nodes_per_mode * j + nodes_per_mode // 2, len(qs) - 1)
        pin.framesForwardKinematics(model, data, qs[node][:model.nq])
        for sole in plan_checks.SOLE_CHANNELS:
            if sole not in plan or not plan_checks._t(plan[sole][j]):
                continue
            p = data.oMf[model.getFrameId(sole)].translation[:2].copy()
            if j > 0 and not plan_checks._t(plan[sole][j - 1]) and sole in last:
                out.append((j, sole, float(np.linalg.norm(p - last[sole]))))
            last[sole] = p
    return out


def manipulation_events(plan: Dict[str, List[Any]]) -> List[int]:
    """Event modes where something other than a sole changes (grasp,
    release, object lifted / set down)."""
    out = []
    for m in plan_checks.event_modes(plan):
        if any(ch not in plan_checks.SOLE_CHANNELS for ch, _p, _c in plan_checks.transition_changes(plan, m - 1)):
            out.append(m)
    return out


def long_step_stretches(plan: Dict[str, List[Any]], lands: Sequence[Landing]) -> List[Dict[str, Any]]:
    """Stretches between consecutive manipulation events (and the plan's
    start / end) containing a step with more than LONG_STEP of foot
    travel, worst first. Each: first / last landing mode, number of steps,
    number of long ones, the longest, the mode to insert steps before, and
    how many to add (one per long step, at least MIN_EXTRA)."""
    n = plan_checks.n_modes(plan)
    bounds = [0] + manipulation_events(plan) + [n]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        inside = [x for x in lands if a < x[0] <= b and x[0] < n]
        long_ = [x for x in inside if x[2] > LONG_STEP]
        if not long_:
            continue
        out.append({
            "first": inside[0][0], "last": inside[-1][0], "steps": len(inside),
            "long": len(long_), "longest": max(long_, key=lambda x: x[2]),
            "before": b if b < n else None, "extra": max(len(long_), MIN_EXTRA),
        })
    return sorted(out, key=lambda s: -s["longest"][2])


def step_lines(
    plan: Dict[str, List[Any]], stretches: Sequence[Dict[str, Any]],
    describe: Optional[Callable[[int], str]] = None,
) -> List[str]:
    """Feedback lines for the worst stretches."""
    lines = []
    for s in stretches[:MAX_LINES]:
        mode, sole, length = s["longest"]
        span = f"mode {s['first']}" if s["first"] == s["last"] else f"modes {s['first']}–{s['last']}"
        where = "at the end of the plan"
        if s["before"] is not None:
            what = describe(s["before"]) if describe else ""
            where = f"before mode {s['before']}" + (f" ({what})" if what else "")
        lines.append(
            f"{span}: in {s['long']} of {s['steps']} steps a foot travels more than "
            f"{LONG_STEP:.1f} m (up to {length:.2f} m, {sole} at mode {mode}); with normal "
            f"{NORMAL_STEP} m steps a foot travels at most about {LONG_STEP:.1f} m. Too few steps "
            f"for the distance or the turning. Give it about {s['extra']} more walking steps "
            f"{where}, e.g. walk(plan, {s['extra']})"
        )
    return lines


def to_step_feedback(task: Any, plan: Dict[str, List[Any]], to_results: Dict[str, Any],
                     describe: Optional[Callable[[int], str]] = None) -> List[str]:
    """Lines for a failed TO solve, [] when nothing is too long or the
    solution cannot be read. Single-robot scenes only, like the rest of
    the evaluator."""
    try:
        robot_id = next(iter(task.scene.robots))
        model = task.scene.robots[robot_id].pin_model
        qs = to_results["robot_xs"][robot_id]
        n = plan_checks.n_modes(plan)
        nodes_per_mode = (len(qs) - 1) // max(n, 1)
        if nodes_per_mode < 1:
            return []
        return step_lines(plan, long_step_stretches(plan, landings(plan, qs, model, nodes_per_mode)), describe)
    except Exception:  # noqa: BLE001 - feedback is best effort
        return []
