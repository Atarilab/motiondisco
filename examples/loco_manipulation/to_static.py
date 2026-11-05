"""Segmented TO with statically stable junctions, solved in parallel.

The chained segmented TO (``to_eval._solve_segmented``) starts each segment
where the previous one ended, so a segment that fails hands its bad end
state to every later one. Here each cut gets its own junction pose instead:
a statically stable configuration from faro's static IK (``SsikTask``: the
cut mode's contacts with object force balance), started from the KSO
keyframe at the cut. Every segment then runs from rest at its start
junction to a hard terminal target at its end junction (robot
``ConfigurationTarget``, box ``PoseTarget``s, terminal velocity cost scaled
up so it arrives at rest), which makes the segments independent: the static
poses and then the segments are solved in a pool of forked worker processes
(``SHINKA_TO_SEGMENT_WORKERS``). The stitched trajectory stops at every
junction; ``SHINKA_TO_SEGMENT_POLISH=1`` then solves the whole plan once,
warm-started from it (``to_eval._polish``), which removes the stops.

Selected with ``SHINKA_TO_SEGMENT_JUNCTION=static``; needs the KSO solution
(``solve_to(..., kso_results=...)``, which ``evaluate.py`` passes), and
falls back to the chained path without it.

``SHINKA_TO_SEGMENT_JUNCTION=parallel`` (``solve_parallel``) keeps the
chained formulation (free segment ends, lookahead windows) but solves the
windows in parallel and repairs the junctions afterwards: round 0 starts
every window from rest at the static pose of its cut; each later round
re-solves, in parallel, only the windows whose start differs from the
previous window's truncated end (``start_mismatch``: base / joint / object
pose and velocity tolerances), from that end and warm-started with the
window's own previous solution. It ends when every junction matches or
after as many rounds as windows (then it has cost a sequential pass plus
round 0). The stitched trajectory is then exactly the chained one.

faro's own static-IK regularisation (1e-4) let the pose slide up to 5 m
along the floor, hence the anchor weights below; with them the gen_27
junctions (three_shelf, 76 modes) sit 0.2-20 cm from the KSO keyframes with
joints changed by 0.08-0.5 rad. The junctions inherit KSO's choices, which
TO may not like (KSO has no dynamics or stride limit): there the segment
starting from the KSO keyframe at mode 72 failed where the chained solve,
free to place the robot, converged.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import to_segments as ts

# Static IK pull toward the KSO keyframe (faro's default is 1e-4 for each);
# with 1.0 on the base, gen_27's cut 63 (hands free) still slid 1.6 m.
STATIC_W_BASE = 10.0
STATIC_W_JOINT = 0.1
STATIC_W_OBJ = 10.0
# A static pose whose base lands farther than this from the KSO keyframe is
# not used: the cut is dropped and its two segments merge.
MAX_POSE_SHIFT = 0.3        # m
# Hard terminal tolerances of an intermediate segment around its end junction
# (compiled into the codegen, hence constants).
JUNCTION_TOL_POS = 0.02     # m, robot base and boxes
JUNCTION_TOL_ROT = 0.05     # rad
JUNCTION_TOL_JOINT = 0.05   # rad
# Factor on the terminal velocity cost of an intermediate segment, so it
# arrives at its end junction at rest (x100 still left |v| ~ 1.9 at the end).
JUNCTION_REST_WEIGHT = 1000.0
CODEGEN_SUFFIX = "_segstatic"
# solve_parallel: a window is re-solved when its start differs from the
# previous window's end by more than this (position m, joints rad, base
# velocity m/s, joint velocity rad/s, object position m).
MATCH_POS = 0.05
MATCH_JOINT = 0.1
MATCH_VEL = 0.1
MATCH_JOINT_VEL = 0.5
MATCH_OBJ = 0.03

Pose = Tuple[np.ndarray, Dict[str, np.ndarray]]   # robot q, movable object poses (xyz + xyzw)

# Set by solve() before the worker pool forks; workers read it, results come
# back pickled. faro's Scene holds a MuJoCo MjSpec and cannot be pickled.
_JOB: Dict[str, Any] = {}


def _unit(x: np.ndarray, at: int = 3) -> np.ndarray:
    x = np.array(x, dtype=float)
    norm = float(np.linalg.norm(x[at:at + 4]))
    if norm > 0:
        x[at:at + 4] /= norm
    return x


def keyframes(kso_results: Optional[Mapping[str, Any]], robot_id: str) -> Optional[Dict[str, Any]]:
    """KSO keyframes {"robot": (n+1, >=nq), "objects": {name: (n+1, >=7)}},
    keyframe k being the transition into mode k; None when missing."""
    if not kso_results or "robot_xs" not in kso_results:
        return None
    robot = kso_results["robot_xs"]
    robot = robot.get(robot_id) if isinstance(robot, Mapping) else robot
    if robot is None:
        return None
    return {"robot": np.asarray(robot, float),
            "objects": {k: np.asarray(v, float) for k, v in (kso_results.get("obj_xs") or {}).items()}}


def merge_cuts(bounds: Sequence[int], keep: Sequence[int]) -> List[int]:
    """``bounds`` without the interior cuts not in ``keep`` (a cut whose
    static pose failed joins its two segments)."""
    kept = set(keep)
    return [bounds[0]] + [c for c in bounds[1:-1] if c in kept] + [bounds[-1]]


def rest_weights(weights: Dict[str, Any], nv: Mapping[str, int], scale: float) -> Dict[str, Any]:
    """Scale the velocity block of the robot and object terminal state cost
    (faro's ``state_cost_We`` layout: base / joint position, base / joint
    velocity, CoM, momentum; objects: pose, twist) in place."""
    for rid, rw in weights["robots"].items():
        we = np.array(rw["state_cost_We"], dtype=float)
        n = int(nv[rid])
        if we.size != 2 * n + 9:
            raise ValueError(f"state_cost_We of {rid!r} has {we.size} entries, expected {2 * n + 9}")
        we[n:2 * n] *= scale
        rw["state_cost_We"] = we
    if "object_state_cost_We" in weights:
        ow = np.array(weights["object_state_cost_We"], dtype=float)
        ow[6:] *= scale
        weights["object_state_cost_We"] = ow
    return weights


def junction_stitch(parts: Sequence[Mapping[str, Any]], robot_id: str) -> Dict[str, Any]:
    """Stitch independent segments: each segment's last robot / object row
    becomes the next segment's first (its junction pose at rest), which its
    end matches only within the junction tolerances."""
    parts = list(parts)
    for i in range(len(parts) - 1):
        nxt = parts[i + 1]
        start = ts.SegmentStart(robot_x=np.asarray(nxt["robot_xs"][robot_id], float)[0],
                                obj_x={k: np.asarray(v, float)[0] for k, v in (nxt.get("obj_xs") or {}).items()})
        parts[i] = ts.with_end_state(parts[i], start, robot_id)
    return ts.stitch_results(parts)


def run_jobs(fn: Callable[..., Any], args_list: Sequence[Tuple], workers: int) -> List[Any]:
    """``[fn(*args) for args in args_list]``, in a pool of forked processes
    when ``workers`` > 1, in order; an exception (also a crashed worker) is
    returned in place of its result."""
    if workers <= 1 or len(args_list) <= 1:
        out: List[Any] = []
        for args in args_list:
            try:
                out.append(fn(*args))
            except Exception as e:  # noqa: BLE001 - reported by the caller
                out.append(e)
        return out
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(int(workers), len(args_list)),
                             mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [pool.submit(fn, *args) for args in args_list]
        out = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                out.append(e)
    return out


def static_pose(task: Any, faro_plan: Mapping[Any, Sequence], mode: int, guess: Pose) -> Tuple[Pose, int]:
    """Statically stable configuration for plan mode ``mode``'s contacts
    near ``guess`` (faro static IK with the anchor weights above);
    returns ((q, objects), solver status)."""
    from faro.cost.ssik_state_ctrl_cost import make_ocp_weights
    from faro.solvers.ipopt_single_stage import IpoptSingleStageConfig
    from faro.tasks.ssik_task import SsikTask

    import to_eval

    q0, objs0 = guess
    near = to_eval._segment_task(task, ts.SegmentStart(robot_x=q0, obj_x=objs0), None)
    ik = SsikTask(scene=task.scene, resolved_state=near.resolved_state)
    ik.build_solver(build_cfg=IpoptSingleStageConfig(
        max_iter=500, tol=1e-2, acceptable_tol=1e-2, expand=False, print_level=0, print_time=False,
        ipopt_options={"print_level": 0, "sb": "yes"}, casadi_options={"print_time": False}))
    ik.update_mode_oh_via_tup({key: [modes[mode]] for key, modes in faro_plan.items()})
    weights = make_ocp_weights(task.scene.robots)
    for rid, robot in task.scene.robots.items():
        weights["robots"][rid]["rob_state_cost_W"] = np.r_[STATIC_W_BASE * np.ones(6),
                                                            STATIC_W_JOINT * np.ones(robot.nv - 6)]
    weights["object_state_cost_W"] = STATIC_W_OBJ * np.ones(6)
    ik.set_fixed_params(n_nodes=1, weights=weights)
    ik.initialize_solver()
    status = ik.solve(max_nlp_iters=20, nlp_steps_per_fixed_collision_params=20, constraint_violation_tol=1e-2)
    res = ik.get_results()
    nq = task.scene.robots[task.robot_id].nq
    q = _unit(np.asarray(res["robot_xs"][task.robot_id], float)[-1][:nq])
    objs = {k: _unit(np.asarray(v, float)[-1][:7]) for k, v in res["obj_xs"].items()}
    return (q, objs), int(status)


def junction_goal(task: Any, pose: Pose) -> Any:
    """Hard terminal target at a junction pose: the robot configuration and
    every movable object's pose, within the junction tolerances."""
    from faro.search.goal import ConfigurationTarget, GoalTolerances, PoseTarget, TerminalGoal

    q, objs = pose
    tol = GoalTolerances(position=JUNCTION_TOL_POS, rotation=JUNCTION_TOL_ROT, joints=JUNCTION_TOL_JOINT)
    return TerminalGoal(kso="off", to="hard", robots={task.robot_id: ConfigurationTarget(q, to_tolerance=tol)},
                        objects={k: PoseTarget(v, to_tolerance=tol) for k, v in objs.items()
                                 if k in task.scene.env_objects})


def rest_start(task: Any, pose: Pose) -> ts.SegmentStart:
    """Full robot / object state rows of a junction pose at rest."""
    import to_eval

    q, objs = pose
    rs = to_eval._segment_task(task, ts.SegmentStart(robot_x=q, obj_x=objs), None).resolved_state
    return ts.SegmentStart(robot_x=np.asarray(rs.robots[task.robot_id].x, float),
                           obj_x={k: np.asarray(s.x, float) for k, s in rs.env_objects.items()})


def _pose_job(cut: int) -> Dict[str, Any]:
    t0 = time.time()
    task, keys = _JOB["task"], _JOB["keys"]
    nq = task.scene.robots[task.robot_id].nq
    guess = (_unit(keys["robot"][cut][:nq]), {k: _unit(v[cut][:7]) for k, v in keys["objects"].items()})
    (q, objs), status = static_pose(task, _JOB["faro_plan"], cut, guess)
    return {"cut": cut, "pose": (q, objs), "status": status, "time_s": time.time() - t0,
            "moved": float(np.linalg.norm(q[:3] - guess[0][:3])),
            "joint_change": float(np.abs(q[7:] - guess[0][7:]).max()) if q.size > 7 else 0.0,
            "object_moved": max([float(np.linalg.norm(objs[k][:3] - guess[1][k][:3])) for k in objs if k in guess[1]],
                                default=0.0)}


def _segment_job(i: int, a: int, b: int, start_pose: Optional[Pose], end_pose: Optional[Pose]
                 ) -> Tuple[ts.SegmentOutcome, Dict[str, Any], List[Dict[str, Any]]]:
    """Solve modes [a, b) from rest at ``start_pose`` (None: the task's
    initial state) to ``end_pose`` (None: the task's own terminal goal)."""
    import functools

    import to_eval

    task, plan, npm = _JOB["task"], _JOB["plan"], _JOB["npm"]
    rid = task.robot_id
    robot = task.scene.robots[rid]
    start = None if start_pose is None else rest_start(task, start_pose)
    goal = task.terminal_goal if end_pose is None else junction_goal(task, end_pose)
    weight_fn = None if end_pose is None else functools.partial(
        rest_weights, nv={r: c.nv for r, c in task.scene.robots.items()}, scale=JUNCTION_REST_WEIGHT)
    prob = to_eval._build_problem(to_eval._segment_task(task, start, goal), ts.slice_plan(plan, a, b),
                                  nodes_per_mode=npm, time_per_mode=_JOB["tpm"], weight_scale=_JOB["weight_scale"],
                                  name_suffix="" if end_pose is None else CODEGEN_SUFFIX, weight_fn=weight_fn)
    solved = to_eval._solve_problem(prob, start=start)
    chosen, iterate = solved.results, solved.iterate
    if (solved.status != 0 and not (solved.physical and solved.physical.get("pass"))
            and solved.best is not None and iterate != "best"):
        chosen, iterate = solved.best, "best"
    end = np.asarray(chosen["robot_xs"][rid], float)[-1]
    extra: Dict[str, Any] = {"v_end": float(np.linalg.norm(end[robot.nq:robot.nq + robot.nv]))}
    if end_pose is not None:
        extra["end_err_base"] = float(np.linalg.norm(end[:3] - end_pose[0][:3]))
        extra["end_err_joint"] = float(np.abs(end[7:robot.nq] - end_pose[0][7:]).max())
    o = ts.SegmentOutcome(
        modes=(a, b), status=solved.status, fail_reason=solved.fail_reason, outer_iters=solved.outer_iters,
        best_prim=solved.best_prim, rescues=solved.rescues, constraint_violation=solved.constraint_violation,
        max_violation=solved.max_violation, iterate=iterate,
        physical_pass=None if solved.physical is None else bool(solved.physical.get("pass")),
        goal="task" if end_pose is None else "static", build_time_s=prob.build_time_s,
        solve_time_s=solved.solve_time_s, extra=extra)
    violations = to_eval._translate_rows(ts.shift_rows(solved.rows, a * npm), task, plan, npm,
                                         getattr(prob.to_task, "collision_index", None))
    print(f"[to_static] segment {i} modes [{a},{b}): status {o.status}, max violation {o.max_violation:.3g}, "
          f"|v_end| {extra['v_end']:.3f}, build {o.build_time_s:.0f}s solve {o.solve_time_s:.0f}s", flush=True)
    return o, {k: v for k, v in chosen.items() if k not in ("best_iterate", "final_iterate")}, violations


def start_mismatch(start: ts.SegmentStart, end: ts.SegmentStart, nq: int, nv: int) -> Optional[str]:
    """Why a window's ``start`` differs from the previous window's ``end``
    (the first tolerance exceeded), None when they match."""
    s, e = np.asarray(start.robot_x, float), np.asarray(end.robot_x, float)
    checks = [
        ("base", float(np.linalg.norm(s[:3] - e[:3])), MATCH_POS, "m"),
        ("joints", float(np.abs(s[7:nq] - e[7:nq]).max()) if nq > 7 else 0.0, MATCH_JOINT, "rad"),
        ("base velocity", float(np.linalg.norm(s[nq:nq + 3] - e[nq:nq + 3])), MATCH_VEL, "m/s"),
        ("joint velocity", float(np.abs(s[nq + 6:nq + nv] - e[nq + 6:nq + nv]).max()) if nv > 6 else 0.0,
         MATCH_JOINT_VEL, "rad/s"),
    ]
    for name in start.obj_x:
        if name in end.obj_x:
            checks.append((name, float(np.linalg.norm(np.asarray(start.obj_x[name], float)[:3]
                                                      - np.asarray(end.obj_x[name], float)[:3])), MATCH_OBJ, "m"))
    for name, value, tol, unit in checks:
        if value > tol:
            return f"{name} differs by {value:.3f} {unit}"
    return None


def _window_job(i: int, a: int, b: int, e: int, start: Optional[ts.SegmentStart],
                guess: Optional[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """Solve modes [a, e) from ``start`` (None: the task's initial state)
    with a free end (the task's own goal when e is the plan end), keep
    [a, b). ``guess`` warm-starts the solve (its node 0 becomes ``start``)."""
    import to_eval

    task, plan, npm = _JOB["task"], _JOB["plan"], _JOB["npm"]
    rid = task.robot_id
    robot = task.scene.robots[rid]
    n = len(next(iter(plan.values())))
    goal = task.terminal_goal if e == n else None
    prob = to_eval._build_problem(to_eval._segment_task(task, start, goal), ts.slice_plan(plan, a, e),
                                  nodes_per_mode=npm, time_per_mode=_JOB["tpm"], weight_scale=_JOB["weight_scale"])
    if guess is not None and start is not None:
        ts.inject_start(guess[0], start, rid)
    solved = to_eval._solve_problem(prob, start=start, guess=guess)
    chosen, iterate = solved.results, solved.iterate
    if (solved.status != 0 and not (solved.physical and solved.physical.get("pass"))
            and solved.best is not None and iterate != "best"):
        chosen, iterate = solved.best, "best"
    keep = (b - a) * npm
    part, rows = chosen, solved.rows
    if e != b:
        part = ts.truncate_results(chosen, keep)
        rows = [r for r in rows if int(r["node"]) < keep]
    end = ts.start_from_results(part, rid, robot.nq, robot.q_min, robot.q_max)
    o = ts.SegmentOutcome(
        modes=(a, b), status=solved.status, fail_reason=solved.fail_reason, outer_iters=solved.outer_iters,
        best_prim=solved.best_prim, rescues=solved.rescues, constraint_violation=solved.constraint_violation,
        max_violation=solved.max_violation, iterate=iterate,
        physical_pass=None if solved.physical is None else bool(solved.physical.get("pass")),
        goal="task" if e == n else "none", build_time_s=prob.build_time_s, solve_time_s=solved.solve_time_s,
        joint_excess=0.0 if start is None else start.joint_excess, extra={"window": [a, e]} if e != b else {})
    violations = to_eval._translate_rows(ts.shift_rows(rows, a * npm), task, plan, npm,
                                         getattr(prob.to_task, "collision_index", None))
    print(f"[to_static] window {i} modes [{a},{b}) over [{a},{e}): status {o.status}, max violation "
          f"{o.max_violation:.3g}, {'warm' if guess is not None else 'cold'} start, build {o.build_time_s:.0f}s "
          f"solve {o.solve_time_s:.0f}s", flush=True)
    full = {k: v for k, v in chosen.items() if k not in ("best_iterate", "final_iterate")}
    return {"outcome": o, "part": ts.with_end_state(part, end, rid), "full": full, "rows": violations, "end": end}


def solve_parallel(task: Any, plan: Dict[str, List[Tuple]], bounds: List[int], *, nodes_per_mode: int,
                   time_per_mode: float, weight_scale: Optional[Mapping[str, float]],
                   kso_results: Optional[Mapping[str, Any]], workers: int = 1, polish: bool = False,
                   lookahead: int = 0, max_rounds: Optional[int] = None) -> Any:
    """Chained windows solved in parallel with junction repair (module
    docstring); returns a ``to_eval.ToSolve`` or None without KSO
    keyframes (the caller then chains)."""
    import plan_checks
    import to_eval
    from _task_loader import to_faro_plan

    keys = keyframes(kso_results, task.robot_id)
    n = plan_checks.n_modes(plan)
    if keys is None or len(keys["robot"]) <= bounds[-2]:
        print("[to_static] no KSO keyframes for the window starts; chaining segments instead", flush=True)
        return None
    npm, rid = nodes_per_mode, task.robot_id
    robot = task.scene.robots[rid]
    _JOB.clear()
    _JOB.update(task=task, plan=plan, faro_plan=to_faro_plan(task.scene, plan), keys=keys, npm=npm,
                tpm=time_per_mode, weight_scale=weight_scale)
    t0 = time.time()
    poses: Dict[int, Dict[str, Any]] = {}
    for cut, p in zip(bounds[1:-1], run_jobs(_pose_job, [(c,) for c in bounds[1:-1]], workers)):
        if isinstance(p, Exception) or p["status"] != 0 or p["moved"] > MAX_POSE_SHIFT:
            why = (f"{type(p).__name__}: {p}" if isinstance(p, Exception) else
                   f"status {p['status']}" if p["status"] != 0 else f"base {p['moved']:.2f} m from the KSO keyframe")
            print(f"[to_static] no static pose at mode {cut} ({why}); merging its segments", flush=True)
            continue
        poses[cut] = p
    bounds = merge_cuts(bounds, poses)
    wins = ts.windows(bounds, lookahead)
    k = len(wins)
    t_poses = time.time() - t0
    starts: List[Optional[ts.SegmentStart]] = [None] + [rest_start(task, poses[a]["pose"]) for a, _, _ in wins[1:]]
    print(f"[to_static] {n} modes as {k} windows {[(a, e) for a, _, e in wins]} in parallel, "
          f"{max(1, int(workers))} workers; static poses {t_poses:.0f}s", flush=True)

    def run(todo: Sequence[int], warm: bool) -> List[Dict[str, Any]]:
        args = []
        for i in todo:
            a, b, e = wins[i]
            guess = None
            if warm:
                x0, u0 = ts.warm_start_dicts(done[i]["full"])
                guess = (x0, u0)
            args.append((i, a, b, e, starts[i], guess))
        out = run_jobs(_window_job, args, workers)
        for i, d in zip(todo, out):
            if isinstance(d, Exception):
                a, b, _ = wins[i]
                raise RuntimeError(f"window {i} modes [{a},{b}) raised {type(d).__name__}: {d}") from d
        return out

    t1 = time.time()
    done: List[Dict[str, Any]] = run(list(range(k)), warm=False)
    rounds: List[Dict[str, Any]] = [{"round": 0, "windows": list(range(k)), "time_s": time.time() - t1}]
    limit = k if max_rounds is None else int(max_rounds)

    def mismatches() -> Tuple[List[int], List[str]]:
        todo, why = [], []
        for i in range(1, k):
            reason = start_mismatch(starts[i], done[i - 1]["end"], robot.nq, robot.nv)  # type: ignore[arg-type]
            if reason is not None:
                todo.append(i)
                why.append(f"window {i}: {reason}")
        return todo, why

    todo, unmatched = mismatches()
    for r in range(1, limit + 1):
        if not todo:
            break
        for i in todo:
            starts[i] = done[i - 1]["end"]
        print(f"[to_static] round {r}: re-solving windows {todo} from the previous ends ({'; '.join(unmatched)})",
              flush=True)
        t_r = time.time()
        for i, d in zip(todo, run(todo, warm=True)):
            done[i] = d
        rounds.append({"round": r, "windows": todo, "reasons": unmatched, "time_s": time.time() - t_r})
        todo, unmatched = mismatches()
    t_windows = time.time() - t1
    outcomes = [d["outcome"] for d in done]
    for i, (a, _, _) in enumerate(wins):
        if a in poses:
            outcomes[i].extra["start_pose"] = {kk: poses[a][kk] for kk in ("moved", "joint_change", "object_moved")}
    # Each window's node 0 is its start; after the rounds that start is the
    # previous end, so this is the chained stitch (an unmatched junction
    # left after the last round is hidden in the end row, as in solve()).
    stitched = junction_stitch([d["part"] for d in done], rid)
    agg = ts.aggregate(outcomes)
    failed = any(o.status == 2 for o in outcomes)
    physical = None
    if to_eval.TO_PHYSICAL_CHECK and (agg["status"] != 0 or unmatched) and not failed:
        try:
            import trajectory_check

            physical = trajectory_check.check(task, plan, stitched, npm)
        except Exception as e:  # noqa: BLE001 - the check is advisory
            print(f"[to_static] physical check of the stitched trajectory failed: {type(e).__name__}: {e}", flush=True)
    converged = agg["converged"] and not unmatched
    first_bad = next((o.modes[0] for o in outcomes if o.status != 0 and not o.physical_pass), None)
    jerk = ts.jerk_cost(stitched["vdot_rob"], time_per_mode / npm)
    out = to_eval.ToSolve(
        status=agg["status"] if converged else max(agg["status"], 1), converged=converged,
        cost=ts.to_cost(agg["constraint_violation"], jerk, n * npm, n), results=stitched, num_modes=n,
        num_nodes=n * npm, constraint_violation=agg["constraint_violation"],
        max_constraint_violation=agg["max_constraint_violation"], jerk_cost=jerk,
        violations=[v for d in done for v in d["rows"]],
        fail_reason=agg["fail_reason"] or ("; ".join(unmatched) if unmatched else ""), rescues=agg["rescues"],
        physical=physical, iterate=agg["iterate"], goal=to_eval.goal_check(task, stitched),
        segments=[o.to_dict() for o in outcomes],
        solved_modes=n if first_bad is None else first_bad)
    out.polish = {"rounds": rounds, "unmatched": unmatched}
    solves = sum(len(rd["windows"]) for rd in rounds)
    print(f"[to_static] parallel windows: {sum(o.status == 0 for o in outcomes)}/{k} converged, {len(rounds)} rounds, "
          f"{solves} window solves, junctions {'matched' if not unmatched else 'NOT matched'}; "
          f"windows {t_windows:.0f}s wall", flush=True)
    if polish and not failed and not unmatched and k > 1 and not out.converged:
        polished = to_eval._polish(task, plan, out, nodes_per_mode=npm, time_per_mode=time_per_mode,
                                   weight_scale=weight_scale)
        polished.polish = dict(polished.polish or {}, rounds=rounds, unmatched=unmatched)
        return polished
    return out


def _seed_codegen(task: Any) -> None:
    """Create the junction-goal codegen folder from the scene's goal-free one
    (only the goal functions then compile); copied under a temporary name
    and renamed, so a concurrent build never sees half a folder."""
    import shutil

    import to_eval

    base = task.scene.name.replace(".", "_")
    dst = os.path.join(to_eval.MOTO_GEN, f"{base}_tohard{CODEGEN_SUFFIX}")
    src = os.path.join(to_eval.MOTO_GEN, base)
    if os.path.isdir(dst) or not os.path.isdir(os.path.join(src, ".moto_artifacts")):
        return
    tmp = f"{dst}.tmp{os.getpid()}"
    try:
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("*.lock"))
        os.rename(tmp, dst)
        print(f"[to_static] codegen cache {dst} seeded from {src}", flush=True)
    except OSError as e:
        print(f"[to_static] codegen seed from {src} failed: {e}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)


def solve(task: Any, plan: Dict[str, List[Tuple]], bounds: List[int], *, nodes_per_mode: int,
          time_per_mode: float, weight_scale: Optional[Mapping[str, float]],
          kso_results: Optional[Mapping[str, Any]], workers: int = 1, polish: bool = False) -> Any:
    """Static junction poses, independent segments, stitching, physical
    check and optional polish; returns a ``to_eval.ToSolve``, or None
    without KSO keyframes (the caller then chains segments)."""
    import plan_checks
    import to_eval
    from _task_loader import to_faro_plan

    keys = keyframes(kso_results, task.robot_id)
    n = plan_checks.n_modes(plan)
    if keys is None or len(keys["robot"]) <= bounds[-2]:
        print("[to_static] no KSO keyframes for the junction poses; chaining segments instead", flush=True)
        return None
    npm, rid = nodes_per_mode, task.robot_id
    _JOB.clear()
    _JOB.update(task=task, plan=plan, faro_plan=to_faro_plan(task.scene, plan), keys=keys, npm=npm,
                tpm=time_per_mode, weight_scale=weight_scale)
    t0 = time.time()
    poses: Dict[int, Dict[str, Any]] = {}
    for cut, p in zip(bounds[1:-1], run_jobs(_pose_job, [(c,) for c in bounds[1:-1]], workers)):
        if isinstance(p, Exception) or p["status"] != 0 or p["moved"] > MAX_POSE_SHIFT:
            why = (f"{type(p).__name__}: {p}" if isinstance(p, Exception) else
                   f"status {p['status']}" if p["status"] != 0 else f"base {p['moved']:.2f} m from the KSO keyframe")
            print(f"[to_static] no static pose at mode {cut} ({why}); merging its segments", flush=True)
            continue
        poses[cut] = p
        print(f"[to_static] static pose at mode {cut}: base moved {p['moved']:.3f} m, joints "
              f"{p['joint_change']:.3f} rad, objects {p['object_moved']:.3f} m, {p['time_s']:.0f}s", flush=True)
    bounds = merge_cuts(bounds, poses)
    t_poses = time.time() - t0
    _seed_codegen(task)
    jobs = [(i, a, b, poses[a]["pose"] if a > 0 else None, poses[b]["pose"] if b < n else None)
            for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    print(f"[to_static] {n} modes as {len(jobs)} independent segments {[(a, b) for _, a, b, _, _ in jobs]}, "
          f"{max(1, int(workers))} workers", flush=True)
    t1 = time.time()
    done = run_jobs(_segment_job, jobs, workers)
    t_segments = time.time() - t1
    for (i, a, b, _, _), d in zip(jobs, done):
        if isinstance(d, Exception):
            raise RuntimeError(f"segment {i} modes [{a},{b}) raised {type(d).__name__}: {d}") from d
    outcomes = [d[0] for d in done]
    for o, (_, a, b, _, _) in zip(outcomes, jobs):
        if a in poses:
            o.extra["start_pose"] = {k: poses[a][k] for k in ("moved", "joint_change", "object_moved", "time_s")}
    stitched = junction_stitch([d[1] for d in done], rid)
    agg = ts.aggregate(outcomes)
    failed = any(o.status == 2 for o in outcomes)
    physical = None
    if to_eval.TO_PHYSICAL_CHECK:
        try:
            import trajectory_check

            physical = trajectory_check.check(task, plan, stitched, npm)
        except Exception as e:  # noqa: BLE001 - the check is advisory
            print(f"[to_static] physical check of the stitched trajectory failed: {type(e).__name__}: {e}", flush=True)
    # Junction rows are replaced when stitching, so all-converged segments
    # count as converged only if the stitched motion also passes the check.
    converged = agg["converged"] and (physical is None or bool(physical.get("pass")))
    jerk = ts.jerk_cost(stitched["vdot_rob"], time_per_mode / npm)
    out = to_eval.ToSolve(
        status=agg["status"] if converged else max(agg["status"], 1), converged=converged,
        cost=ts.to_cost(agg["constraint_violation"], jerk, n * npm, n), results=stitched, num_modes=n,
        num_nodes=n * npm, constraint_violation=agg["constraint_violation"],
        max_constraint_violation=agg["max_constraint_violation"], jerk_cost=jerk,
        violations=[v for d in done for v in d[2]], fail_reason=agg["fail_reason"], rescues=agg["rescues"],
        physical=physical, iterate=agg["iterate"], goal=to_eval.goal_check(task, stitched),
        segments=[o.to_dict() for o in outcomes])
    print(f"[to_static] stitched: {sum(o.status == 0 for o in outcomes)}/{len(outcomes)} segments converged, "
          f"physical {None if physical is None else physical.get('pass')}; static poses {t_poses:.0f}s, "
          f"segments {t_segments:.0f}s wall", flush=True)
    if polish and not failed and len(jobs) > 1:
        out = to_eval._polish(task, plan, out, nodes_per_mode=npm, time_per_mode=time_per_mode,
                              weight_scale=weight_scale)
    return out
