"""
Bounded repair probe for KSO-infeasible plans (pure; the solve is injected).

When the full plan fails the kinematic check, the violations say where
the constraints broke, not which decision caused it (a box set down on
the truck's step was caused by where the robot stood when it gripped it,
twenty modes earlier). The probe tests the one edit that fixed every such
case so far: insert a couple of walking steps before an event, and
re-solve once. Candidates, in order: right before the mode where the
prefix walk found the break, then before the first grasp. At most
``max_probes`` solves; stops at the first success.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import plan_checks


def candidate_modes(plan: Dict[str, List[Any]], first_infeasible_modes: Optional[int]) -> List[int]:
    """Modes before which to insert a walk: the breaking mode (the last mode
    of the first infeasible prefix), then the first grasp."""
    out: List[int] = []
    if first_infeasible_modes and first_infeasible_modes >= 2:
        out.append(first_infeasible_modes - 1)
    grasp = plan_checks.first_grasp_mode(plan)
    if grasp is not None and grasp not in out:
        out.append(grasp)
    return out


def probe(
    plan: Dict[str, List[Any]],
    first_infeasible_modes: Optional[int],
    solve_once: Callable[[Dict[str, List[Any]]], bool],
    *,
    max_probes: int = 2,
    n_steps: int = 2,
    max_modes: Optional[int] = None,
    deadline: Optional[float] = None,
    clock: Callable[[], float] = time.time,
    describe: Optional[Callable[[int], str]] = None,
) -> Dict[str, Any]:
    """Try the candidate edits. Returns ``{"fixed_by": str | None,
    "tried": [...], "skipped": str | None, "plan": repaired plan | None}``."""
    result: Dict[str, Any] = {"fixed_by": None, "tried": [], "skipped": None, "plan": None}
    if max_probes <= 0:
        result["skipped"] = "disabled"
        return result
    for k in candidate_modes(plan, first_infeasible_modes)[:max_probes]:
        if deadline is not None and clock() > deadline:
            result["skipped"] = "time budget"
            break
        event = describe(k) if describe else ""
        entry: Dict[str, Any] = {"before_mode": k, "n_steps": n_steps, "event": event}
        edited = plan_checks.insert_walk(plan, k, n_steps)
        if edited is None:
            entry["result"] = "no double support at that mode"
            result["tried"].append(entry)
            continue
        modes = plan_checks.n_modes(edited)
        entry["modes"] = modes
        if max_modes is not None and modes > max_modes:
            entry["result"] = f"over the {max_modes}-mode cap"
            result["tried"].append(entry)
            continue
        feasible = bool(solve_once(edited))
        entry["result"] = "feasible" if feasible else "still infeasible"
        result["tried"].append(entry)
        if feasible:
            where = f"right before mode {k}" + (f" ({event})" if event else "")
            result["fixed_by"] = f"walk(plan, {n_steps}) inserted {where}"
            result["plan"] = edited
            break
    return result


def describe_result(result: Dict[str, Any]) -> Optional[str]:
    """One feedback sentence, or None when nothing was tried."""
    if result.get("fixed_by"):
        return (f"Repair probe: {result['fixed_by']} made the whole plan KSO-feasible "
                f"in one solve; try that edit.")
    tried = [t for t in result.get("tried", []) if t.get("result") in ("feasible", "still infeasible")]
    if tried:
        spots = ", ".join(
            f"before mode {t['before_mode']}" + (f" ({t['event']})" if t.get("event") else "")
            for t in tried
        )
        return (f"Repair probe: inserting walk(plan, {tried[0]['n_steps']}) {spots} "
                f"did not make the plan feasible; the fix is elsewhere.")
    if result.get("skipped") == "time budget":
        return "Repair probe skipped (evaluation time budget)."
    return None
