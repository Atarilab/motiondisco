"""Segmented TO: solve a long contact plan as consecutive shorter TO problems.

A 40-mode plan is an 800-node TO problem; long problems converge much less
often than short ones (truck_unload_stack 0/29; long_dist_pick_place 100% at
<= 13 modes, 69% at >= 18). ``to_eval`` can instead cut the plan into
segments of at most ``SHINKA_TO_SEGMENT_MODES`` modes, solve them in order
and start each from the previous segment's final state, then stitch the
results into one trajectory that the rest of the pipeline uses unchanged.

faro fixes a TO's initial condition to node 0 of the warm start (robot
q, v, CoM, momentum; object pose and twist), so a segment starts exactly
where the previous one ended, moving. A cut keeps the full problem's
constraints when no hand contact changes at it: a solve's first interval is
never a hand-switch interval (faro shortens a hand's landing interval to a
tenth and forces its wrench to zero), whereas foot and object changes carry
no interval-level special case. Cuts are therefore placed at double-support
modes where no hand contact changes; walking landings and object placements
are fine.

Pure numpy and plan bookkeeping; the faro-touching orchestration lives in
``to_eval._solve_segmented``.

    split_points(plan, max_modes, min_modes)   segment boundaries [0, ..., n]
    windows(bounds, lookahead)                 solved / kept mode ranges
    slice_plan(plan, a, b)                     modes [a, b)
    truncate_results(results, k)               first k intervals of a solve
    start_from_results(results, rid, nq, ...)  next segment's start state
    tiled_guess(start, rest_robot_x, ...)      warm start with row 0 = start
    stitch_results(parts)                      one results dict
    aggregate(outcomes)                        status / violation numbers
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import plan_checks

Plan = Dict[str, List[Any]]

# Cut ranks, lower is better; None means "never cut here".
RANK_SAME = 0       # nothing changes between mode k-1 and mode k
RANK_STEP = 1       # only the soles change (a walking landing)
RANK_OBJECT = 2     # an object channel changes (placement / lift), no hand change
RANK_HAND = 3       # a hand contact changes: only with allow_hand_cut


def _double_support(plan: Plan, k: int) -> bool:
    return all(ch in plan and plan_checks._t(plan[ch][k]) for ch in plan_checks.SOLE_CHANNELS)


def cut_rank(plan: Plan, k: int) -> Optional[int]:
    """Rank of starting a new segment at mode ``k`` (cut between k-1 and k);
    None when mode k is not double support or k is not an interior mode."""
    n = plan_checks.n_modes(plan)
    if not 0 < k < n or not _double_support(plan, k):
        return None
    changed = {ch for ch, _prev, _curr in plan_checks.transition_changes(plan, k - 1)}
    if not changed:
        return RANK_SAME
    if changed & set(plan_checks.EE_CHANNELS):
        return RANK_HAND
    if changed <= set(plan_checks.SOLE_CHANNELS):
        return RANK_STEP
    return RANK_OBJECT


def manipulation_cuts(plan: Plan, where: str = "before") -> List[int]:
    """Walking cuts (``cut_rank`` <= RANK_STEP) around the manipulation
    stretches (consecutive modes entered by a hand or object contact change).
    ``where="before"``: the last one before each stretch, so a segment
    starts with the robot already at the object. ``where="after"``: the
    first one after each stretch, so the walk to the next object and its
    manipulation are solved together. ``where="first"``: the stretch's
    first mode itself when it is double support (a hand cut when a hand
    contact starts there: the segment's first interval then is not
    faro's shortened, zero-wrench switch interval)."""
    if where not in ("before", "after", "first"):
        raise ValueError(f"unknown cut side {where!r}")
    n = plan_checks.n_modes(plan)
    soles = set(plan_checks.SOLE_CHANNELS)
    cuts: List[int] = []
    cand: Optional[int] = None
    after_stretch = False
    for k in range(1, n):
        changed = {ch for ch, _prev, _curr in plan_checks.transition_changes(plan, k - 1)}
        if changed - soles:
            if where == "before" and cand is not None:
                cuts.append(cand)
            if where == "first" and not after_stretch and cut_rank(plan, k) is not None:
                cuts.append(k)
            cand, after_stretch = None, True
            continue
        if where == "first":
            after_stretch = False
            continue
        r = cut_rank(plan, k)
        if r is None or r > RANK_STEP:
            continue
        if where == "before":
            cand = k
        elif after_stretch:
            cuts.append(k)
            after_stretch = False
    return cuts


def split_points(plan: Plan, max_modes: int, min_modes: int = 4,
                 allow_hand_cut: bool = False, prefer: str = "balanced") -> List[int]:
    """Segment boundaries ``[0, c1, ..., n]``: every segment has at least
    ``min_modes`` modes, at most ``max_modes`` wherever the cuts allow it,
    and starts at an allowed cut (``cut_rank`` <= RANK_OBJECT, or RANK_HAND
    with ``allow_hand_cut``). A stretch with no allowed cut longer than
    ``max_modes`` (e.g. a long walk without double support) makes only its
    own segment longer: the split first minimises the modes over
    ``max_modes`` summed over segments.
    ``prefer="balanced"``: then fewest segments, then lowest rank sum, then
    the most even lengths. ``prefer="manip"`` / ``"approach"`` / ``"first"``:
    then cut at as many ``manipulation_cuts`` (before / after each
    manipulation, at its first mode; "first" allows hand cuts) as the length
    bounds allow and elsewhere only where a segment would be too long.
    ``[0, n]`` (no segmentation) when ``n <= max_modes``, ``max_modes <= 0``
    or no allowed cut exists."""
    n = plan_checks.n_modes(plan)
    if max_modes <= 0 or n <= max_modes:
        return [0, n]
    sides = {"manip": "before", "approach": "after", "first": "first"}
    if prefer != "balanced" and prefer not in sides:
        raise ValueError(f"unknown cut preference {prefer!r}")
    min_modes = max(1, min(min_modes, max_modes))
    worst = RANK_HAND if allow_hand_cut or prefer == "first" else RANK_OBJECT
    ranks = {k: cut_rank(plan, k) for k in range(1, n)}
    manip: set = set()
    if prefer != "balanced":
        manip = set(manipulation_cuts(plan, sides[prefer]))
    target = n / math.ceil(n / max_modes)
    best: List[Optional[Tuple[int, int, int, float]]] = [None] * (n + 1)
    prev = [-1] * (n + 1)
    best[0] = (0, 0, 0, 0.0)
    for j in range(1, n + 1):
        if j < n:
            r = ranks[j]
            if r is None or r > worst:
                continue
        else:
            r = 0
        for i in range(0, j - min_modes + 1):
            b = best[i]
            if b is None:
                continue
            excess = b[0] + max(0, j - i - max_modes)
            if prefer != "balanced":
                # (excess, cuts outside manipulation_cuts, minus cuts at them, evenness)
                at = j in manip
                cand = (excess, b[1] + int(j < n and not at), b[2] - int(at), b[3] + (j - i - target) ** 2)
            else:
                cand = (excess, b[1] + 1, b[2] + r, b[3] + (j - i - target) ** 2)
            if best[j] is None or cand < best[j]:
                best[j], prev[j] = cand, i
    if best[n] is None:
        return [0, n]
    out, j = [n], n
    while j > 0:
        j = prev[j]
        out.append(j)
    return out[::-1]


def greedy_cuts(plan: Plan, max_modes: int, min_modes: int = 1, where: str = "before") -> List[int]:
    """Segment boundaries chosen left to right, so that a boundary depends
    only on the plan before it (and the manipulation it precedes): from
    each boundary the next one is the last manipulation cut
    (``manipulation_cuts``) within ``max_modes``, else the last allowed cut
    (``cut_rank`` <= RANK_OBJECT) at least ``min_modes`` away within
    ``max_modes``, else the first allowed cut beyond (the stretch lengthens
    its own segment), else the plan's end. Plans sharing a prefix share its
    boundaries, which the incremental evaluation's window cache needs.
    ``[0, n]`` when ``n <= max_modes`` or ``max_modes <= 0``."""
    n = plan_checks.n_modes(plan)
    if max_modes <= 0 or n <= max_modes:
        return [0, n]
    min_modes = max(1, min(min_modes, max_modes))
    manip = set(manipulation_cuts(plan, where))
    allowed = [k for k in range(1, n) if (r := cut_rank(plan, k)) is not None and r <= RANK_OBJECT]
    bounds = [0]
    while bounds[-1] < n:
        a = bounds[-1]
        if n - a <= max_modes:
            bounds.append(n)
            break
        near = [c for c in allowed if a + min_modes <= c <= a + max_modes]
        at_manip = [c for c in near if c in manip]
        if at_manip:
            nxt = at_manip[-1]
        elif near:
            nxt = near[-1]
        else:
            beyond = [c for c in allowed if c > a + max_modes]
            nxt = beyond[0] if beyond else n
        bounds.append(nxt)
    return bounds


def windows(bounds: Sequence[int], lookahead: int = 0) -> List[Tuple[int, int, int]]:
    """``(a, b, e)`` per segment: solve modes [a, e), keep [a, b). An
    intermediate segment looks ``lookahead`` modes past its cut (``e = b +
    lookahead``, clipped to the plan end), so its kept part is planned
    knowing what follows; the first window that reaches the end of the plan
    is kept whole and is the last one."""
    n = int(bounds[-1])
    out: List[Tuple[int, int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        e = n if b >= n else min(n, b + max(0, int(lookahead)))
        if e >= n:
            out.append((int(a), n, n))
            break
        out.append((int(a), int(b), int(e)))
    return out


def slice_plan(plan: Plan, a: int, b: int) -> Plan:
    return {ch: list(seq[a:b]) for ch, seq in plan.items() if isinstance(seq, list)}


def _unit_quat(x: np.ndarray, at: int = 3) -> None:
    q = x[at:at + 4]
    norm = float(np.linalg.norm(q))
    if norm > 0:
        x[at:at + 4] = q / norm


@dataclass
class SegmentStart:
    robot_x: np.ndarray                 # full robot state row [q | v | com | h]
    obj_x: Dict[str, np.ndarray]        # object rows [pos | quat xyzw | twist]
    joint_excess: float = 0.0           # largest joint clip applied (rad)


def start_from_results(results: Mapping[str, Any], robot_id: str, nq: int,
                       q_min: Optional[np.ndarray] = None, q_max: Optional[np.ndarray] = None,
                       clip_tol: float = 1e-6) -> SegmentStart:
    """Final state of a solve as the next segment's start: the last robot
    and object rows, with unit quaternions (faro's resolve requires them)
    and joints clipped into their limits when a non-converged iterate ended
    slightly outside."""
    robot_xs = results["robot_xs"]
    x = np.array(robot_xs[robot_id] if isinstance(robot_xs, Mapping) else robot_xs, dtype=float)[-1].copy()
    _unit_quat(x)
    excess = 0.0
    if q_min is not None and q_max is not None:
        qj = x[7:nq]
        lo, hi = np.asarray(q_min, float), np.asarray(q_max, float)
        over = np.maximum(lo - qj, 0) + np.maximum(qj - hi, 0)
        excess = float(over.max()) if over.size else 0.0
        if excess > clip_tol:
            x[7:nq] = np.clip(qj, lo, hi)
    objs = {}
    for name, xs in (results.get("obj_xs") or {}).items():
        row = np.array(xs, dtype=float)[-1].copy()
        _unit_quat(row)
        objs[name] = row
    return SegmentStart(robot_x=x, obj_x=objs, joint_excess=excess)


def with_end_state(results: Mapping[str, Any], start: SegmentStart, robot_id: str) -> Dict[str, Any]:
    """Copy of *results* whose last robot / object rows are *start* (the
    next segment's normalised, joint-clipped start), so the stitched
    trajectory stays continuous after a clip."""
    out = dict(results)
    xs = {o: np.array(x, float) for o, x in results["robot_xs"].items()}
    xs[robot_id][-1] = start.robot_x
    out["robot_xs"] = xs
    objs = {o: np.array(x, float) for o, x in (results.get("obj_xs") or {}).items()}
    for name, row in start.obj_x.items():
        if name in objs:
            objs[name][-1] = row
    out["obj_xs"] = objs
    return out


def tiled_guess(start: SegmentStart, rest_robot_x: Mapping[str, np.ndarray],
                rest_obj_x: Mapping[str, np.ndarray], num_nodes: int, nv: Mapping[str, int],
                patches: Mapping[str, Sequence[str]], robot_id: str
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """faro's own guess (the start pose at rest on every node, zero
    controls, unit time scale) with node 0 replaced by the true start, so
    the segment's fixed initial condition carries the previous velocity."""
    n = int(num_nodes)
    x0 = {
        "robots": {rid: np.tile(np.asarray(x, float), (n + 1, 1)) for rid, x in rest_robot_x.items()},
        "objects": {name: np.tile(np.asarray(x, float), (n + 1, 1)) for name, x in rest_obj_x.items()},
    }
    u0 = {
        "robots": {rid: {"acceleration": np.zeros((n, nv[rid])),
                         "wrenches": {p: np.zeros((n, 6)) for p in patches[rid]}}
                   for rid in rest_robot_x},
        "objects": {name: np.zeros((n, 12)) for name in rest_obj_x},
        "t_scale": np.ones(n),
    }
    inject_start(x0, start, robot_id)
    return x0, u0


def inject_start(x0: Dict[str, Any], start: SegmentStart, robot_id: str) -> None:
    """Overwrite node 0 of an existing guess with the start state."""
    x0["robots"][robot_id][0] = start.robot_x
    for name, row in start.obj_x.items():
        if name in x0["objects"]:
            x0["objects"][name][0] = row


_NODE_KEYS = ("robot_xs", "obj_xs")                      # (N+1, ...) per owner
_INTERVAL_OWNER_KEYS = ("vdot_rob", "obj_Vdots", "obj_Ws", "torques")   # (N, ...) per owner
_INTERVAL_KEYS = ("t_scale", "step_ts")                  # (N,)


def stitch_results(parts: Sequence[Mapping[str, Any]], junction_atol: float = 1e-2) -> Dict[str, Any]:
    """One results dict from consecutive segment results: node arrays
    ``vstack(A, B[1:])`` (the junction node once), interval arrays
    concatenated, warm-start dicts rebuilt from the stitched arrays. Raises
    ValueError when a segment does not start where the previous one ended."""
    if not parts:
        raise ValueError("no segment results to stitch")
    first = parts[0]
    out: Dict[str, Any] = {}
    for key in _NODE_KEYS:
        owners = first.get(key) or {}
        out[key] = {}
        for owner in owners:
            arrs = [np.asarray(p[key][owner], float) for p in parts]
            for i in range(1, len(arrs)):
                gap = float(np.max(np.abs(arrs[i - 1][-1] - arrs[i][0]))) if arrs[i].size else 0.0
                if gap > junction_atol:
                    raise ValueError(f"{key}[{owner}]: segment {i} starts {gap:.3g} away from segment {i - 1}'s end")
            out[key][owner] = np.vstack([arrs[0]] + [a[1:] for a in arrs[1:]])
    for key in _INTERVAL_OWNER_KEYS:
        if key in first:
            out[key] = {o: np.concatenate([np.asarray(p[key][o], float) for p in parts]) for o in first[key]}
    if "rob_Ws" in first:
        out["rob_Ws"] = {rid: {patch: np.concatenate([np.asarray(p["rob_Ws"][rid][patch], float) for p in parts])
                               for patch in first["rob_Ws"][rid]} for rid in first["rob_Ws"]}
    for key in _INTERVAL_KEYS:
        if key in first:
            out[key] = np.concatenate([np.asarray(p[key], float).reshape(-1) for p in parts])
    if "torque_limit" in first:
        out["torque_limit"] = first["torque_limit"]
    out["x0_dict"], out["u0_dict"] = warm_start_dicts(out)
    return out


def truncate_results(results: Mapping[str, Any], num_intervals: int) -> Dict[str, Any]:
    """The first ``num_intervals`` intervals of a solve (nodes 0 ..
    num_intervals): the kept part of a lookahead window."""
    k = int(num_intervals)
    out: Dict[str, Any] = {}
    for key in _NODE_KEYS:
        if key in results:
            out[key] = {o: np.asarray(x, float)[:k + 1] for o, x in results[key].items()}
    for key in _INTERVAL_OWNER_KEYS:
        if key in results:
            out[key] = {o: np.asarray(x, float)[:k] for o, x in results[key].items()}
    if "rob_Ws" in results:
        out["rob_Ws"] = {r: {p: np.asarray(w, float)[:k] for p, w in ws.items()}
                         for r, ws in results["rob_Ws"].items()}
    for key in _INTERVAL_KEYS:
        if key in results:
            out[key] = np.asarray(results[key], float).reshape(-1)[:k]
    if "torque_limit" in results:
        out["torque_limit"] = results["torque_limit"]
    return out


def warm_start_dicts(results: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """moto's warm-start layout from a results dict (copies)."""
    x0 = {"robots": {r: np.array(x, float) for r, x in results["robot_xs"].items()},
          "objects": {o: np.array(x, float) for o, x in results.get("obj_xs", {}).items()}}
    u0 = {"robots": {r: {"acceleration": np.array(results["vdot_rob"][r], float),
                         "wrenches": {p: np.array(w, float) for p, w in results["rob_Ws"][r].items()}}
                     for r in results["robot_xs"]},
          "objects": {o: np.hstack([np.asarray(results["obj_Vdots"][o], float),
                                    np.asarray(results["obj_Ws"][o], float)])
                      for o in results.get("obj_xs", {})},
          "t_scale": np.array(results["t_scale"], float).reshape(-1)}
    return x0, u0


def hold_results(robot_x: Any, obj_x: Mapping[str, Any], patches: Sequence[str], nv: int, num_nodes: int,
                 dt: float, robot_id: str) -> Dict[str, Any]:
    """A trajectory that holds one state for ``num_nodes`` intervals with
    zero controls: the result of an evaluation stopped before its first
    window was solved. Not a physical trajectory."""
    n = int(num_nodes)
    out: Dict[str, Any] = {
        "robot_xs": {robot_id: np.tile(np.asarray(robot_x, float), (n + 1, 1))},
        "obj_xs": {o: np.tile(np.asarray(v, float), (n + 1, 1)) for o, v in obj_x.items()},
        "vdot_rob": {robot_id: np.zeros((n, int(nv)))},
        "rob_Ws": {robot_id: {p: np.zeros((n, 6)) for p in patches}},
        "obj_Vdots": {o: np.zeros((n, 6)) for o in obj_x}, "obj_Ws": {o: np.zeros((n, 6)) for o in obj_x},
        "t_scale": np.ones(n), "step_ts": np.full(n, float(dt)),
    }
    out["x0_dict"], out["u0_dict"] = warm_start_dicts(out)
    return out


def pad_hold(results: Mapping[str, Any], num_nodes: int, dt: float) -> Dict[str, Any]:
    """Extend a partial (failed) stitched trajectory to ``num_nodes``
    intervals by holding its last state with zero controls, so consumers
    keep full-plan shapes. Not a physical trajectory."""
    out = dict(results)
    have = len(np.asarray(results["t_scale"]).reshape(-1))
    extra = int(num_nodes) - have
    if extra <= 0:
        return out
    for key in _NODE_KEYS:
        out[key] = {o: np.vstack([x, np.repeat(np.asarray(x)[-1:], extra, axis=0)]) for o, x in results[key].items()}
    for key in ("vdot_rob", "obj_Vdots", "obj_Ws"):
        if key in results:
            out[key] = {o: np.vstack([x, np.zeros((extra, np.asarray(x).shape[1]))]) for o, x in results[key].items()}
    if "torques" in results:
        out["torques"] = {o: np.vstack([x, np.repeat(np.asarray(x)[-1:], extra, axis=0)]) for o, x in results["torques"].items()}
    if "rob_Ws" in results:
        out["rob_Ws"] = {r: {p: np.vstack([w, np.zeros((extra, 6))]) for p, w in ws.items()}
                         for r, ws in results["rob_Ws"].items()}
    out["t_scale"] = np.r_[np.asarray(results["t_scale"], float).reshape(-1), np.ones(extra)]
    out["step_ts"] = np.r_[np.asarray(results["step_ts"], float).reshape(-1), np.full(extra, float(dt))]
    out["x0_dict"], out["u0_dict"] = warm_start_dicts(out)
    return out


def jerk_cost(vdot_rob: Mapping[str, np.ndarray], dt: float) -> float:
    """faro's ``ToTask.get_finite_diff_jerk_cost`` on given accelerations."""
    return sum(float(np.sum(np.linalg.norm(np.diff(np.asarray(v, float), axis=0) / dt, axis=1)))
               for v in vdot_rob.values())


def to_cost(constraint_violation: float, jerk: float, num_nodes: int, num_modes: int) -> float:
    """``to_eval``'s TO cost formula."""
    return 1e2 * constraint_violation / max(num_nodes, 1) + 1e-2 * jerk / max(num_modes, 1)


def shift_rows(rows: Sequence[Mapping[str, Any]], node_offset: int) -> List[Dict[str, Any]]:
    """Violation rows of a segment with node indices in the full plan."""
    return [dict(r, node=int(r["node"]) + int(node_offset)) for r in rows]


def polish_t_scale(step_ts: np.ndarray, dt_traj: np.ndarray, lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    """Time scale that reproduces the stitched node durations in the full
    problem (whose switch intervals differ), within faro's bounds."""
    return np.clip(np.asarray(step_ts, float).reshape(-1) / np.asarray(dt_traj, float).reshape(-1), lo, hi)


@dataclass
class SegmentOutcome:
    modes: Tuple[int, int]
    status: Optional[int] = None            # None: not solved (an earlier segment failed)
    fail_reason: str = ""
    outer_iters: int = 0
    best_prim: float = float("inf")
    rescues: int = 0
    constraint_violation: float = 0.0
    max_violation: float = 0.0
    iterate: str = "final"
    physical_pass: Optional[bool] = None
    goal: str = "none"
    build_time_s: float = 0.0
    solve_time_s: float = 0.0
    joint_excess: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modes"] = list(self.modes)
        if not math.isfinite(d["best_prim"]):
            d["best_prim"] = None
        return d


def aggregate(outcomes: Sequence[SegmentOutcome]) -> Dict[str, Any]:
    """Whole-plan numbers from the solved segments: worst status, converged
    only if every segment converged, summed / max violation, the first
    failure named by its segment, and which iterates were used."""
    solved = [o for o in outcomes if o.status is not None]
    if not solved:
        return {"status": 2, "converged": False, "rescues": 0, "constraint_violation": 0.0,
                "max_constraint_violation": 0.0, "fail_reason": "no segment solved", "iterate": "final"}
    failed = next((i for i, o in enumerate(solved) if o.status == 2), None)
    reason = ""
    if failed is not None:
        o = solved[failed]
        reason = f"segment {failed} modes [{o.modes[0]},{o.modes[1]}): {o.fail_reason}"
    elif len(solved) < len(outcomes) and solved[-1].status != 0:
        # Stopped early (to_eval fail-fast) at a segment that did not converge
        # and was not physically valid.
        o = solved[-1]
        reason = (f"segment {len(solved) - 1} modes [{o.modes[0]},{o.modes[1]}): stopped, status {o.status}, "
                  f"max violation {o.max_violation:.3g}")
    iterates = {o.iterate for o in solved}
    return {
        "status": max(int(o.status) for o in solved),
        "converged": len(solved) == len(outcomes) and all(o.status == 0 for o in solved),
        "rescues": sum(o.rescues for o in solved),
        "constraint_violation": float(sum(o.constraint_violation for o in solved)),
        "max_constraint_violation": float(max(o.max_violation for o in solved)),
        "fail_reason": reason,
        "iterate": iterates.pop() if len(iterates) == 1 else "mixed",
    }
