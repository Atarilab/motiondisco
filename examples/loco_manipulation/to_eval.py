"""
Single TO (trajectory optimization) solve for a loco_manipulation plan.

Shared by ``evaluate.py``, ``run_to_all.py`` and ``recompute_to_costs.py``
so the three agree on solver settings and the cost formula:

    to_cost = 1e2 * sum_constraint_violations / num_nodes
            + 1e-2 * finite_diff_jerk_cost / num_modes

This mirrors ``faro.evaluators.to.create_to_func`` except for
``nodes_per_mode`` (20 here, 10 there) and the zero-cost fallback,
which ``evaluate.py`` handles itself (``to_cost == 0`` scores 0).

The terminal goal's own ``to=`` mode ("soft" / "hard" / "off") decides
how ToTask treats its targets.

``solve_to`` takes optional overrides (timing, cost-weight scaling) used
by the level-2 variant sweep (``sweep_to_variants.py``); the defaults
reproduce the evaluator's solve exactly. Start-state and terminal-goal
variants are passed in through ``task`` (a ``TaskContext`` copy with a
different ``resolved_state`` / ``terminal_goal``).

Callers redirect solver output themselves; nothing here touches stdout
beyond what faro / moto print.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import segment_cache  # noqa: E402
from _task_loader import TaskContext, to_faro_plan  # noqa: E402

NODES_PER_MODE = 20
TIME_PER_MODE = 1.0
# Moto worker threads per TO solve (faro default 6). Lower it with
# SHINKA_TO_N_JOB=N when the machine is shared or busy.
TO_N_JOB = int(os.environ.get("SHINKA_TO_N_JOB", "6"))
# Seed TO from the KSO solution when one is passed to solve_to
# (to_warm_start.py): "1" poses, velocities and contact forces, "pose"
# poses and contact forces with zero velocities; "0" keeps faro's guess
# (initial pose everywhere).
# Off by default: on truck_unload_stack both variants made TO worse than
# faro's own guess (max violation 20-23 vs 1.1 after 100 iterations).
TO_WARM_START = os.environ.get("SHINKA_TO_WARM_START", "0").strip().lower()
# Collision-free presolve iterations (faro default 50). With a KSO seed the
# presolve may drag the motion through obstacles before collisions apply.
TO_PRESOLVE = os.environ.get("SHINKA_TO_PRESOLVE")
# Outer TO iterations. 100 stopped the best truck_unload_stack plan at
# max violation 1.09; it converged after 188 (16.5 min, one thread).
TO_MAX_NLP_ITERS = int(os.environ.get("SHINKA_TO_MAX_ITERS", "300"))
# Primal residual below which a TO solve (or segment) counts as converged;
# faro's default is 0.05.
TO_FEASIBILITY_TOL = float(os.environ.get("SHINKA_TO_FEASIBILITY_TOL", "0.05"))
# Iterations given to moto's restoration phase when an update had no budget
# left for it (to_solve.py); faro's own loop aborts there with status 2.
# 0 aborts at that point like faro's ToTask.solve (to_solve still records
# the failure reason and the best iterate). SHINKA_TO_MAX_RESCUES caps how
# often per solve.
TO_RESCUE_ITERS = int(os.environ.get("SHINKA_TO_RESCUE", "10"))
TO_MAX_RESCUES = int(os.environ.get("SHINKA_TO_MAX_RESCUES", "20"))
# Cold-start the interior point after a rescue (barrier back to mu0, slacks
# and multipliers re-initialised) instead of continuing from restoration's
# exit state, which drifted (banana_real gen_31: best residual 2.76 with
# faro's loop, 11-14 after a warm rescue, 0.69 with the reset).
# Judge non-converged trajectories physically (trajectory_check.py):
# "0" disables; then only TO convergence counts.
TO_PHYSICAL_CHECK = os.environ.get("SHINKA_TO_PHYSICAL_CHECK", "1").strip().lower() not in ("0", "false", "no", "off")
TO_RESET_AFTER_RESCUE = os.environ.get("SHINKA_TO_RESET_AFTER_RESCUE", "1").strip().lower() in ("1", "true", "yes", "on")
# Segmented TO (to_segments.py): plans longer than SHINKA_TO_SEGMENT_MODES
# modes are solved as consecutive segments of at most that many modes, each
# starting from the previous segment's final state, and stitched. 0 = off.
TO_SEGMENT_MODES = int(os.environ.get("SHINKA_TO_SEGMENT_MODES", "0"))
TO_SEGMENT_MIN_MODES = int(os.environ.get("SHINKA_TO_SEGMENT_MIN_MODES", "4"))
# Lookahead: an intermediate segment is solved this many modes past its cut
# and only the part up to the cut is kept, so its end state is planned for
# what follows instead of being left free (to_segments.windows). 0 = none.
TO_SEGMENT_LOOKAHEAD = int(os.environ.get("SHINKA_TO_SEGMENT_LOOKAHEAD", "0"))
# Where to cut: "balanced" (fewest, most even segments) or "manip" (at the
# last step landing before each grasp / release / placement, so a segment
# starts with the robot at the object; to_segments.manipulation_cuts).
TO_SEGMENT_CUTS = os.environ.get("SHINKA_TO_SEGMENT_CUTS", "balanced").strip().lower()
# "kso": intermediate segments end on hard base / object position targets
# taken from the KSO solution at the cut (tolerance SHINKA_TO_SEGMENT_GOAL_TOL
# metres), against greedy drift between segments; "off": no target.
TO_SEGMENT_GOAL = os.environ.get("SHINKA_TO_SEGMENT_GOAL", "off").strip().lower()
TO_SEGMENT_GOAL_TOL = float(os.environ.get("SHINKA_TO_SEGMENT_GOAL_TOL", "0.1"))
# State-cost reference of later segments: "segment" (faro's semantics: the
# segment's own start pose at rest) or "initial" (the task's start state, as
# in the full problem).
TO_SEGMENT_REF = os.environ.get("SHINKA_TO_SEGMENT_REF", "segment").strip().lower()
# Polish: one full-horizon solve warm-started from the stitched trajectory
# (kept when it converges, or is physically valid with a lower violation).
TO_SEGMENT_POLISH = os.environ.get("SHINKA_TO_SEGMENT_POLISH", "0").strip().lower() in ("1", "true", "yes", "on")
TO_SEGMENT_POLISH_ITERS = int(os.environ.get("SHINKA_TO_SEGMENT_POLISH_ITERS", "50"))
# The collision-free presolve would pull a good seed through obstacles.
TO_SEGMENT_POLISH_PRESOLVE = int(os.environ.get("SHINKA_TO_SEGMENT_POLISH_PRESOLVE", "0"))
# Junctions between segments: "chain" (each segment starts where the
# previous one ended), "static" (to_static.py: statically stable poses at
# the cuts, segments independent from rest to rest and solved in parallel by
# SHINKA_TO_SEGMENT_WORKERS forked processes; the polish then always runs
# when SHINKA_TO_SEGMENT_POLISH is on) or "parallel" (to_static.solve_parallel:
# the chained windows solved in parallel from the static poses, then only
# the windows whose start no longer matches the previous end re-solved,
# round by round, until the junctions match).
TO_SEGMENT_JUNCTION = os.environ.get("SHINKA_TO_SEGMENT_JUNCTION", "chain").strip().lower()
TO_SEGMENT_WORKERS = int(os.environ.get("SHINKA_TO_SEGMENT_WORKERS", "1"))
# Stop a chained segmented solve at the first segment that neither converges
# nor passes the physical check; the rest of the plan is held in place
# (to_segments.pad_hold). Every later segment would start from a bad state
# and burn its iteration budget (layered_pallet gen_1: 26 min on five such
# segments after segment 2 failed). "0" solves them all as before.
TO_SEGMENT_FAIL_FAST = os.environ.get("SHINKA_TO_SEGMENT_FAIL_FAST", "1").strip().lower() in ("1", "true", "yes", "on")
# Walking-step repair of a failing segment (_step_repair): when TO's own
# solution of the window has a foot travelling more than
# step_feedback.LONG_STEP at a landing, the steps step_feedback asks for are
# inserted into the plan (plan_checks.insert_walk) and the segment is
# re-solved from the same start; the repaired plan is the one the result
# describes (ToSolve.plan / repairs). Caps per segment and per plan.
TO_REPAIR = os.environ.get("SHINKA_TO_REPAIR", "1").strip().lower() in ("1", "true", "yes", "on")
TO_REPAIR_MAX = int(os.environ.get("SHINKA_TO_REPAIR_MAX", "2"))
# Steps per repair: step_feedback asks for one per long step, but a badly
# non-converged iterate (layered_pallet gen_6: residual 8.4, "8 long steps")
# is no measure of the distance; a few steps, then re-solve and look again.
TO_REPAIR_STEPS = int(os.environ.get("SHINKA_TO_REPAIR_STEPS", "3"))
# Per plan: layered_pallet gen_1 needed 6 repairs over 4 segments and still
# failed the last one after 2.2 h; a plan that short of steps everywhere is
# better handed back to the LLM with the first repairs as feedback.
TO_REPAIR_TOTAL = int(os.environ.get("SHINKA_TO_REPAIR_TOTAL", "4"))
# Stop a TO solve whose best primal residual improved by less than 10% over
# this many outer iterations (to_solve.solve stall_iters); 0 = off.
TO_STALL_ITERS = int(os.environ.get("SHINKA_TO_STALL_ITERS", "0"))
# Incremental evaluation of plans longer than SHINKA_TO_SEGMENT_MODES: the
# windows are cut left to right (to_segments.greedy_cuts, so plans sharing
# a prefix share its windows), each window is KSO-checked and TO-solved
# from the previous window's end state (no full-plan KSO; evaluate.py skips
# it), every window that converged or was physically valid is stored in the
# run's cache (segment_cache.py, keyed by the plan prefix, the start state
# and the solver settings) and reused by later plans that keep the prefix,
# and the first failing window stops the evaluation. One code path for any
# plan length: a plan within the segment length is a single window. On by
# default ("0" restores the full-plan KSO and the global cut placement); on
# three_shelf gen_25 (73 modes) the first pass converged all 8 windows (same
# TO cost as the whole chained solve, 39.1), the second pass of the same plan
# took 7 windows from the cache and solved one in 4 min.
TO_INCREMENTAL = os.environ.get("SHINKA_TO_INCREMENTAL", "1").strip().lower() in ("1", "true", "yes", "on")
# KSO attempts per window (KSO is noisy near its tolerance).
TO_WINDOW_KSO_ATTEMPTS = int(os.environ.get("SHINKA_TO_WINDOW_KSO_ATTEMPTS", "2"))
# Route seeding: a window that neither converged nor passed the physical
# check is re-solved once from a KSO solution of that window, solved from
# the window's own start state (kso_warm_start's level-change seed
# included), as TO's initial guess (to_warm_start). TO's cold guess holds
# the start pose and lets the solver find the motion; when the motion must
# pass a fixed box (walk around a table, step onto or off it) the iterate
# goes through the box and the collision constraint, linearised there,
# cannot push it back out ("table penetrates torso by 40 cm"). KSO is only
# a seed here, never a verdict: from a TO end state it has called windows
# infeasible that TO then solved (three_shelf gen_25).
TO_ROUTE_SEED = os.environ.get("SHINKA_TO_ROUTE_SEED", "1").strip().lower() not in ("0", "false", "no", "off")


def incremental_plan(num_modes: int) -> bool:
    """Whether a plan of ``num_modes`` is evaluated incrementally (window
    by window, without a full-plan KSO)."""
    return TO_INCREMENTAL and TO_SEGMENT_MODES > 0 and num_modes > TO_SEGMENT_MODES


@dataclass
class ToSolve:
    status: int
    converged: bool
    cost: float
    results: Dict[str, Any]
    num_modes: int
    num_nodes: int
    # Raw cost components: summed / max constraint violation and the
    # finite-difference jerk cost, before the per-node / per-mode
    # normalization folded into ``cost``.
    constraint_violation: float = 0.0
    max_constraint_violation: float = 0.0
    jerk_cost: float = 0.0
    # Violations above violation_feedback.REPORT_TOL, translated to plan
    # vocabulary (Violation.to_dict() records, worst first).
    violations: List[Dict[str, Any]] = field(default_factory=list)
    # "<phase>: <moto result>" when status is 2 (to_solve.SolveInfo), and
    # how often restoration was given a budget during the solve.
    fail_reason: str = ""
    rescues: int = 0
    # trajectory_check report of the trajectory in ``results`` (None when
    # the check is off or failed to run), and which iterate that is.
    physical: Optional[Dict[str, Any]] = None
    iterate: str = "final"
    # Final state of ``results`` against the terminal goal's TO targets
    # (goal_check; None when there is no goal or the check could not run).
    goal: Optional[Dict[str, Any]] = None
    # Segmented solves: one record per segment (to_segments.SegmentOutcome
    # dicts) and the optional full-horizon polish; None for a single solve.
    segments: Optional[List[Dict[str, Any]]] = None
    polish: Optional[Dict[str, Any]] = None
    # Segmented solves: modes solved before the first segment that neither
    # converged nor passed the physical check (num_modes when none did), for
    # the partial-credit score; None for a single solve.
    solved_modes: Optional[int] = None
    # Walking-step repairs applied during the solve (_step_repair records)
    # and the plan they produced, which is the plan this result describes;
    # ``plan`` is None when nothing was changed.
    repairs: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[Dict[str, List[Tuple]]] = None
    # Incremental evaluation: whether the first window had a kinematic
    # solution (None when KSO ran on the whole plan instead), and how many
    # leading modes were reused from the run's window cache.
    kso_feasible: Optional[bool] = None
    cached_modes: int = 0

    @property
    def physically_valid(self) -> bool:
        return bool(self.physical and self.physical.get("pass"))

    @property
    def goal_met(self) -> bool:
        """The terminal goal's hard TO targets reached within the solve's
        feasibility slack (True when the TO goal is soft / off or unchecked)."""
        return not (self.goal and self.goal.get("mode") == "hard" and not self.goal.get("met"))

    @property
    def valid(self) -> bool:
        """A usable trajectory: converged, or physically valid without
        converging (trajectory_check), and either way at the hard TO goal
        (the physical check knows nothing about the task goal)."""
        return (self.converged or self.physically_valid) and self.goal_met

    @property
    def effective_max_violation(self) -> float:
        """Violation used for scoring: the solver's for a converged solve;
        for a physically valid one, the physical check's worst metric
        relative to its threshold, scaled to the feasibility tolerance
        (TO_FEASIBILITY_TOL, so it never exceeds a converged solve's);
        otherwise the solver's."""
        if self.converged or not self.physically_valid:
            return self.max_constraint_violation
        import trajectory_check

        return TO_FEASIBILITY_TOL * trajectory_check.normalized(self.physical["worst"])


def to_mode(task: TaskContext) -> str:
    """The TO treatment of the task's terminal goal: soft / hard / off."""
    return task.terminal_goal.to if task.terminal_goal is not None else "off"


def goal_check(task: Any, results: Mapping[str, Any], slack: Optional[float] = None,
               error_fn: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None) -> Optional[Dict[str, Any]]:
    """The trajectory's final state against every target of the task's
    terminal goal, with the solver's own residual (faro's target error:
    position, rotation log in the target frame, joints) and the targets'
    componentwise TO tolerances. ``met`` allows ``slack`` (default
    TO_FEASIBILITY_TOL, the primal residual at which a solve counts as
    converged) beyond the tolerance, so a trajectory that did not converge
    is held to the same standard as one that did. None when the task has
    no terminal goal or no target could be checked; never raises."""
    goal = getattr(task, "terminal_goal", None)
    if goal is None:
        return None
    if error_fn is None:
        try:
            from faro.search.goal import _target_error as error_fn
        except Exception as e:  # noqa: BLE001
            print(f"[to_eval] goal check unavailable: {type(e).__name__}: {e}", flush=True)
            return None
    slack = TO_FEASIBILITY_TOL if slack is None else float(slack)
    targets: Dict[str, Dict[str, Any]] = {}
    for key, group in (("obj_xs", getattr(goal, "objects", None) or {}),
                       ("robot_xs", getattr(goal, "robots", None) or {})):
        states = results.get(key) or {}
        for name, target in group.items():
            try:
                state = np.asarray(states[name], dtype=float)[-1]
                err = np.abs(np.asarray(error_fn(target, state), dtype=float).reshape(-1))
                tol = np.asarray(target.tolerance("to"), dtype=float).reshape(-1)
                excess = np.maximum(err - tol, 0.0)
            except Exception as e:  # noqa: BLE001 - diagnostics only
                print(f"[to_eval] goal check of {name} failed: {type(e).__name__}: {e}", flush=True)
                continue
            rec: Dict[str, Any] = {"excess": round(float(excess.max(initial=0.0)), 4),
                                   "position_error": round(float(np.linalg.norm(err[:3])), 4),
                                   "position_tolerance": round(float(tol[:3].max(initial=0.0)), 4)}
            if err.size >= 6:
                rec["rotation_error"] = round(float(np.linalg.norm(err[3:6])), 4)
                rec["rotation_tolerance"] = round(float(tol[3:6].max(initial=0.0)), 4)
            if err.size > 6:
                rec["joint_error"] = round(float(err[6:].max()), 4)
            targets[name] = rec
    if not targets:
        return None
    worst = max(t["excess"] for t in targets.values())
    return {"mode": getattr(goal, "to", "off"), "met": bool(worst <= slack), "slack": slack,
            "excess": worst, "targets": targets}


def scale_weights(weights: Dict[str, Any], scale: Mapping[str, float]) -> Dict[str, Any]:
    """Multiply named cost-weight entries in place and return *weights*.

    A key may name a shared top-level entry (object / time weights) or a
    per-robot entry under ``weights["robots"][robot_id]`` (e.g.
    ``state_cost_W``); per-robot keys are scaled for every robot.
    Unknown keys raise ``KeyError`` rather than silently doing nothing.
    """
    for key, factor in scale.items():
        hit = False
        if key in weights and key != "robots":
            weights[key] = weights[key] * factor
            hit = True
        for robot_weights in weights.get("robots", {}).values():
            if key in robot_weights:
                robot_weights[key] = robot_weights[key] * factor
                hit = True
        if not hit:
            raise KeyError(f"unknown TO cost weight {key!r}")
    return weights


def _goal_suffix(task: TaskContext) -> str:
    """'' when TO has no terminal goal (the original cache names), else
    '_to<mode>'."""
    mode = getattr(task.terminal_goal, "to", "off") if task.terminal_goal is not None else "off"
    return "" if mode == "off" else f"_to{mode}"


MOTO_GEN = "moto_gen"   # faro's MotoBuildConfig.code_gen_path (relative to the cwd)


def codegen_name(task: TaskContext, name_suffix: str = "") -> Tuple[str, str]:
    """(scene part, suffix) of a TO problem's moto codegen folder
    ``<scene><goal suffix><suffix>``."""
    return task.scene.name.replace(".", "_"), _goal_suffix(task) + name_suffix


def _seed_codegen(base: str, rest: str, root: str = MOTO_GEN) -> Optional[str]:
    """Create ``root/<base><rest>`` from the newest per-horizon folder of
    the same problem (``<base>_<nodes><rest>``, the layout before per-scene
    folders) so already compiled functions are not rebuilt; moto's hash check
    recompiles only functions whose code changed. Returns the source folder."""
    import re
    import shutil

    dst = os.path.join(root, base + rest)
    if os.path.isdir(dst) or not os.path.isdir(root):
        return None
    pat = re.compile(rf"^{re.escape(base)}_\d+{re.escape(rest)}$")
    cands = [os.path.join(root, d) for d in os.listdir(root) if pat.match(d)
             and os.path.isdir(os.path.join(root, d, ".moto_artifacts"))]
    if not cands:
        return None
    src = max(cands, key=os.path.getmtime)
    try:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.lock"))
        print(f"[to_eval] codegen cache {dst} seeded from {src}", flush=True)
        return src
    except OSError as e:
        print(f"[to_eval] codegen seed from {src} failed: {e}", flush=True)
        shutil.rmtree(dst, ignore_errors=True)
        return None


def solve_to(
    task: TaskContext,
    plan: Dict[str, List[Tuple]],
    *,
    nodes_per_mode: int = NODES_PER_MODE,
    time_per_mode: float = TIME_PER_MODE,
    weight_scale: Optional[Mapping[str, float]] = None,
    kso_results: Optional[Mapping[str, Any]] = None,
    segment_modes: Optional[int] = None,
) -> ToSolve:
    """Build, solve and score the TO problem for a flat-channel *plan*.

    ``results`` additionally carries ``step_ts`` (per-node durations
    ``dt_traj[i] * t_scale[i]``) so replays reproduce the optimizer's
    timing. Plans longer than ``segment_modes`` (default
    ``SHINKA_TO_SEGMENT_MODES``; 0 = off) are solved segment by segment
    (``to_segments``). Exceptions propagate.
    """
    import plan_checks

    seg = TO_SEGMENT_MODES if segment_modes is None else int(segment_modes)
    n = plan_checks.n_modes(plan)
    if seg > 0 and n > seg:
        import to_segments

        if TO_INCREMENTAL:
            sides = {"balanced": "before", "manip": "before", "approach": "after", "first": "first"}
            bounds = to_segments.greedy_cuts(plan, seg, TO_SEGMENT_MIN_MODES, where=sides.get(TO_SEGMENT_CUTS, "before"))
        else:
            bounds = to_segments.split_points(plan, seg, TO_SEGMENT_MIN_MODES, prefer=TO_SEGMENT_CUTS)
        if len(bounds) > 2:
            if TO_SEGMENT_JUNCTION in ("static", "parallel"):
                import to_static

                if TO_SEGMENT_JUNCTION == "static":
                    out = to_static.solve(task, plan, bounds, nodes_per_mode=nodes_per_mode,
                                          time_per_mode=time_per_mode, weight_scale=weight_scale,
                                          kso_results=kso_results, workers=TO_SEGMENT_WORKERS,
                                          polish=TO_SEGMENT_POLISH)
                else:
                    out = to_static.solve_parallel(task, plan, bounds, nodes_per_mode=nodes_per_mode,
                                                   time_per_mode=time_per_mode, weight_scale=weight_scale,
                                                   kso_results=kso_results, workers=TO_SEGMENT_WORKERS,
                                                   polish=TO_SEGMENT_POLISH, lookahead=TO_SEGMENT_LOOKAHEAD)
                if out is not None:
                    return out
            return _solve_segmented(task, plan, bounds, nodes_per_mode=nodes_per_mode,
                                    time_per_mode=time_per_mode, weight_scale=weight_scale,
                                    kso_results=kso_results)
        print(f"[to_eval] {n}-mode plan has no eligible cut for {seg}-mode segments; solving it whole",
              flush=True)
    prob = _build_problem(task, plan, nodes_per_mode=nodes_per_mode, time_per_mode=time_per_mode,
                          weight_scale=weight_scale)
    return _finalize(prob, _solve_problem(prob, kso_results=kso_results))


@dataclass
class _Problem:
    """One built TO problem (owns its ToTask)."""
    to_task: Any
    task: TaskContext
    plan: Dict[str, List[Tuple]]
    faro_plan: Any
    num_modes: int
    num_nodes: int
    nodes_per_mode: int
    time_per_mode: float
    build_time_s: float = 0.0


@dataclass
class _Solved:
    status: int
    fail_reason: str
    rescues: int
    outer_iters: int
    best_prim: float
    results: Dict[str, Any]
    physical: Optional[Dict[str, Any]]
    iterate: str
    constraint_violation: float
    max_violation: float
    rows: List[Dict[str, Any]]
    solve_time_s: float
    # Full best (lowest-residual) iterate with torques and step_ts, when the
    # solve did not converge; segmented solves chain from it.
    best: Optional[Dict[str, Any]] = None
    # Stopped early because the residual stopped improving (TO_STALL_ITERS).
    stalled: bool = False


def _build_problem(task: TaskContext, plan: Dict[str, List[Tuple]], *, nodes_per_mode: int,
                   time_per_mode: float, weight_scale: Optional[Mapping[str, float]] = None,
                   name_suffix: str = "", presolve: Optional[int] = None,
                   weight_fn: Optional[Callable[[Dict[str, Any]], Any]] = None) -> _Problem:
    import time

    from faro.cost.state_ctrl_cost import make_ocp_weights
    from faro.search.evaluation import num_modes_cms
    from faro.solvers.moto import MotoBuildConfig
    from faro.tasks.to_task import ToTask

    t0 = time.time()
    faro_plan = to_faro_plan(task.scene, plan)
    num_modes = num_modes_cms(faro_plan)
    num_nodes = num_modes * nodes_per_mode
    if presolve is None and TO_PRESOLVE:
        presolve = int(TO_PRESOLVE)

    # One codegen folder per scene and TO goal mode under moto_gen/. The
    # generated code does not depend on the horizon (node count and contact
    # schedule are runtime parameters: the 800- and 820-node truck folders
    # held identical code for all 348 functions), and moto reuses a compiled
    # function when its code hash matches, recompiling only what changed.
    base, rest = codegen_name(task, name_suffix)
    _seed_codegen(base, rest)
    name = base + rest
    to_task = ToTask(
        scene=task.scene,
        resolved_state=task.resolved_state,
        terminal_goal=task.terminal_goal,
    )
    to_task.build_solver(
        cfg=MotoBuildConfig(
            name=name,
            num_modes=num_modes,
            nodes_per_mode=nodes_per_mode,
            time_per_mode=time_per_mode,
            tol=1e-2,
            feasibility_tol=TO_FEASIBILITY_TOL,
            n_job=TO_N_JOB,
            **({"presolve_max_iters": int(presolve)} if presolve is not None else {}),
        ),
    )
    to_task.initialize_solver(faro_plan)
    to_task.update_mode_oh_via_tup(faro_plan)
    weights = make_ocp_weights(task.scene.robots)
    if weight_scale:
        scale_weights(weights, weight_scale)
    if weight_fn is not None:   # in-place edits beyond whole-entry scales (to_static.rest_weights)
        weight_fn(weights)
    to_task.set_fixed_params(n_nodes=num_nodes, weights=weights)
    return _Problem(to_task, task, plan, faro_plan, num_modes, num_nodes, nodes_per_mode,
                    time_per_mode, build_time_s=time.time() - t0)


def _set_warm_start(prob: _Problem, x0: Dict[str, Any], u0: Dict[str, Any]) -> None:
    """Install a guess on both SQPs (the presolve one only exists when
    presolve iterations > 0); node 0 becomes the fixed initial state."""
    to_task = prob.to_task
    if to_task.sqp is not None:
        to_task.moto_wrap.set_warm_start(to_task.sqp, x0, u0)
    to_task.moto_wrap.set_warm_start(to_task.sqp_coll, x0, u0)


def _solve_problem(prob: _Problem, *, kso_results: Optional[Mapping[str, Any]] = None,
                   start: Any = None, guess: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
                   max_iters: Optional[int] = None, force_seed: bool = False) -> _Solved:
    """Warm start (an explicit guess, the KSO seed, or faro's own guess
    with node 0 set to ``start``), solve, pick the trajectory and read the
    violations while the ToTask is alive."""
    import time

    import to_solve

    to_task = prob.to_task
    seed = kso_results is not None and (force_seed or TO_WARM_START not in ("0", "false", "no", "off"))
    if guess is not None:
        _set_warm_start(prob, *guess)
    elif seed:
        import to_warm_start

        x0, u0 = to_warm_start.build_guess(to_task, kso_results, prob.plan, prob.num_modes,
                                           prob.nodes_per_mode, with_velocities=TO_WARM_START != "pose")
        if start is not None:
            import to_segments

            to_segments.inject_start(x0, start, prob.task.robot_id)
        _set_warm_start(prob, x0, u0)
    elif start is not None:
        import to_segments

        resolved = to_task.resolved
        robots = prob.task.scene.robots
        x0, u0 = to_segments.tiled_guess(
            start, {rid: st.x for rid, st in resolved.robots.items()},
            {name: st.x for name, st in resolved.env_objects.items()}, prob.num_nodes,
            {rid: cfg.nv for rid, cfg in robots.items()},
            {rid: list(cfg.patch_ees) for rid, cfg in robots.items()}, prob.task.robot_id)
        _set_warm_start(prob, x0, u0)

    t0 = time.time()
    info = to_solve.solve(to_task, TO_MAX_NLP_ITERS if max_iters is None else int(max_iters),
                          TO_RESCUE_ITERS, TO_MAX_RESCUES,
                          log=lambda msg: print(msg, flush=True), reset_after_rescue=TO_RESET_AFTER_RESCUE,
                          stall_iters=TO_STALL_ITERS)
    status, fail_reason, rescues = info.status, info.fail_reason, info.rescues
    results = to_task.get_results(include_torque=True, coll=True)
    results["step_ts"] = to_task.get_step_ts(results)
    best = None
    if status != 0 and info.best_results is not None:
        # Cost and violations describe the final iterate; the best one is
        # kept for replay / tracking (max violation = its primal residual).
        best = _with_torques(to_task, info.best_results)
        best["step_ts"] = to_task.get_step_ts(best)
        results["best_iterate"] = {
            "robot_xs": best["robot_xs"], "obj_xs": best["obj_xs"],
            "step_ts": best["step_ts"], "max_violation": info.best_prim, "outer_iter": info.best_iter,
        }
    physical, iterate = None, "final"
    if TO_PHYSICAL_CHECK and status != 0:
        physical, chosen, iterate = _physical_choice(prob.task, prob.plan, results, best, prob.nodes_per_mode,
                                                     status=int(status))
        if iterate == "best":
            # The trajectory everything downstream uses (replay, dataset,
            # task quality) becomes the physically valid best iterate.
            final = results
            results = dict(chosen)
            results["final_iterate"] = {k: final[k] for k in ("robot_xs", "obj_xs", "step_ts")}
    return _Solved(
        status=int(status), fail_reason=fail_reason, rescues=int(rescues),
        outer_iters=int(getattr(info, "outer_iters", 0)), best_prim=float(getattr(info, "best_prim", float("inf"))),
        results=results, physical=physical, iterate=iterate,
        constraint_violation=float(to_task.sum_constraint_violations()),
        max_violation=float(to_task.max_constraint_violation()),
        rows=_violation_rows(to_task), solve_time_s=time.time() - t0, best=best,
        stalled=bool(getattr(info, "stalled", False)),
    )


def _finalize(prob: _Problem, solved: _Solved) -> ToSolve:
    ctrl_margin = prob.to_task.get_finite_diff_jerk_cost(solved.results["vdot_rob"])
    cost = 1e2 * solved.constraint_violation / prob.num_nodes + 1e-2 * ctrl_margin / prob.num_modes
    return ToSolve(
        violations=_translate_rows(solved.rows, prob.task, prob.plan, prob.nodes_per_mode,
                                   getattr(prob.to_task, "collision_index", None)),
        status=solved.status,
        converged=(solved.status == 0),
        cost=float(cost),
        results=solved.results,
        num_modes=prob.num_modes,
        num_nodes=prob.num_nodes,
        constraint_violation=solved.constraint_violation,
        max_constraint_violation=solved.max_violation,
        jerk_cost=float(ctrl_margin),
        fail_reason=solved.fail_reason,
        rescues=solved.rescues,
        physical=solved.physical,
        iterate=solved.iterate,
        goal=goal_check(prob.task, solved.results),
    )


def _segment_task(task: TaskContext, start: Any, goal: Any) -> TaskContext:
    """Copy of *task* with *goal* as terminal goal and, given a start, a
    resolved state at the start pose (velocities zero there: the state-cost
    reference is the segment's start at rest; the true velocity enters only
    through node 0 of the warm start)."""
    import copy

    seg = copy.copy(task)
    seg.terminal_goal = goal
    if start is None:
        return seg
    from faro.configs.scene import Pose, RobotConfiguration, StateSpec

    scene, rs = task.scene, task.resolved_state
    nq = scene.robots[task.robot_id].nq
    objects = {}
    for name in scene.objects:
        if name in scene.env_objects:
            objects[name] = Pose(q=tuple(float(v) for v in start.obj_x[name][:7]))
        else:
            objects[name] = Pose(q=tuple(float(v) for v in rs.fixed_env_objects[name].q))
    robot = RobotConfiguration(q=tuple(float(v) for v in start.robot_x[:nq]))
    seg.resolved_state = scene.resolve(StateSpec(robots={task.robot_id: robot}, objects=objects))
    return seg


def _segment_goal(task: TaskContext, kso_results: Optional[Mapping[str, Any]], cut: int) -> Tuple[Any, str]:
    """(goal, kind) for an intermediate segment ending at mode ``cut``:
    hard base and object position targets from KSO state ``cut`` (the
    transition into that mode) with SHINKA_TO_SEGMENT_GOAL=kso, else none."""
    if TO_SEGMENT_GOAL != "kso" or not kso_results:
        return None, "none"
    try:
        from faro.search.goal import GoalTolerances, PositionTarget, TerminalGoal

        robot_keys = kso_results["robot_xs"]
        robot_keys = robot_keys[task.robot_id] if isinstance(robot_keys, Mapping) else robot_keys
        if len(robot_keys) <= cut:
            return None, "none"
        tol = GoalTolerances(position=TO_SEGMENT_GOAL_TOL)
        objects = {name: PositionTarget(xyz=np.asarray(kso_results["obj_xs"][name][cut][:3], float), to_tolerance=tol)
                   for name in task.scene.env_objects if name in (kso_results.get("obj_xs") or {})}
        goal = TerminalGoal(kso="off", to="hard", objects=objects, robots={
            task.robot_id: PositionTarget(xyz=np.asarray(robot_keys[cut][:3], float), to_tolerance=tol)})
        return goal, "kso"
    except Exception as e:  # noqa: BLE001 - fall back to an open segment end
        print(f"[to_eval] KSO segment goal unavailable: {type(e).__name__}: {e}", flush=True)
        return None, "none"


def _slice_kso(kso_results: Optional[Mapping[str, Any]], a: int, b: int) -> Optional[Dict[str, Any]]:
    """KSO keyframes for modes [a, b): states a .. b (b = the transition
    into the next segment)."""
    if not kso_results:
        return None
    out = dict(kso_results)
    for key in ("robot_xs", "obj_xs"):
        if key in kso_results and isinstance(kso_results[key], Mapping):
            out[key] = {o: np.asarray(x)[a:b + 1] for o, x in kso_results[key].items()}
    return out


def _set_reference(prob: _Problem, resolved: Any) -> None:
    """Point the running / terminal state cost at *resolved* (the task's
    start state) instead of the segment's start (SHINKA_TO_SEGMENT_REF=initial)."""
    from faro.configs.identifiers import robot_name

    params = {robot_name(rid, "x0"): st.x for rid, st in resolved.robots.items()}
    params.update({f"{name}_x0": st.x for name, st in resolved.env_objects.items()})
    to_task = prob.to_task
    for sqp in (to_task.sqp, to_task.sqp_coll):
        if sqp is not None:
            to_task.moto_wrap.update_params_const(sqp, params)


def _step_repair(task: TaskContext, plan: Dict[str, List[Tuple]], a: int, e: int,
                 results: Mapping[str, Any], nodes_per_mode: int) -> Optional[Tuple[int, int, str, float]]:
    """Walking-step repair for a failing window [a, e): the stretch of the
    window with the longest foot travel at a landing (step_feedback) whose
    end is a manipulation inside the window -> ``(before_mode, steps,
    reason, longest stride in m)`` in the plan's numbering. None when every
    step is normal or the long stretch runs to the window's end (the
    lookahead's end is not an event)."""
    import step_feedback
    import to_segments as ts

    try:
        model = task.scene.robots[task.robot_id].pin_model
        qs = results["robot_xs"][task.robot_id]
    except Exception:  # noqa: BLE001 - no robot model / trajectory to read
        return None
    window = ts.slice_plan(plan, a, e)
    try:
        stretches = step_feedback.long_step_stretches(
            window, step_feedback.landings(window, qs, model, nodes_per_mode))
    except Exception as ex:  # noqa: BLE001 - the repair is advisory
        print(f"[to_eval] step analysis of modes [{a},{e}) failed: {type(ex).__name__}: {ex}", flush=True)
        return None
    for s in stretches:
        if s["before"] is None:
            continue
        before = a + int(s["before"])
        if before <= a:
            continue
        mode, sole, length = s["longest"]
        steps = max(1, min(int(s["extra"]), TO_REPAIR_STEPS))
        return before, steps, f"{sole} travels {length:.2f} m landing at mode {a + int(mode)}", float(length)
    return None


def _shift_report(report: Optional[Mapping[str, Any]], a: int, nodes_per_mode: int) -> Optional[Dict[str, Any]]:
    """A window's trajectory_check report with its node / mode / interval
    locations moved to the full plan (window start ``a``)."""
    if not report:
        return None
    out = dict(report)
    where: Dict[str, List[Any]] = {}
    for key, w in (report.get("where") or {}).items():
        w = list(w)
        if len(w) >= 2 and w[0] == "node":
            w[1] = int(w[1]) + a * nodes_per_mode
        elif len(w) >= 2 and w[0] == "mode":
            w[1] = int(w[1]) + a
            if len(w) >= 4 and w[2] == "interval":
                w[3] = int(w[3]) + a * nodes_per_mode
        where[key] = w
    out["where"] = where
    out["window_start_mode"] = a
    return out


def _resplit(bounds: List[int], a: int, before: int, added: int, plan: Dict[str, List[Tuple]]) -> List[int]:
    """Segment bounds after ``added`` modes were inserted at ``before``
    (inside the segment starting at ``a``): with TO_SEGMENT_MODES set the
    plan from ``a`` on is cut afresh (to_segments.split_points), so the
    repaired segment and its window stay within the segment length
    instead of growing with every repair (gen_1's grew to 18 modes, 17 min
    a solve); without it the later cuts just shift."""
    if TO_SEGMENT_MODES > 0:
        import to_segments as ts

        fresh = ts.split_points(plan, TO_SEGMENT_MODES, TO_SEGMENT_MIN_MODES, prefer=TO_SEGMENT_CUTS)
        return [x for x in bounds if x <= a] + [c for c in fresh if c > a]
    return [x if x < before else x + added for x in bounds]


def _cache_settings(task: TaskContext, nodes_per_mode: int, time_per_mode: float,
                    weight_scale: Optional[Mapping[str, float]]) -> Dict[str, Any]:
    """Everything besides the plan prefix and the start state that a
    window's solution depends on (segment_cache key)."""
    return {"scene": getattr(task, "scene_name", None), "ic": getattr(task, "ic_name", None),
            "goal_to": to_mode(task), "npm": int(nodes_per_mode), "tpm": float(time_per_mode),
            "weights": dict(weight_scale or {}), "lookahead": TO_SEGMENT_LOOKAHEAD, "max_iters": TO_MAX_NLP_ITERS,
            "tol": TO_FEASIBILITY_TOL, "rescue": [TO_RESCUE_ITERS, TO_MAX_RESCUES, TO_RESET_AFTER_RESCUE],
            "presolve": TO_PRESOLVE, "stall": TO_STALL_ITERS, "ref": TO_SEGMENT_REF, "goal": TO_SEGMENT_GOAL,
            "physical": TO_PHYSICAL_CHECK, "warm": TO_WARM_START, "n_job": TO_N_JOB,
            "kso_attempts": TO_WINDOW_KSO_ATTEMPTS, "margin": _task_loader_margin(),
            "goal_override": _task_loader_goal()}


def _task_loader_goal() -> Any:
    import _task_loader

    return [_task_loader.TO_GOAL, _task_loader.TO_GOAL_TOL]


def _task_loader_margin() -> Any:
    import _task_loader

    return [float(getattr(_task_loader, "COLLISION_MARGIN", 0.0)),
            list(getattr(_task_loader, "COLLISION_MARGIN_CAPSULES", ()))]


def _window_kso(seg_task: TaskContext, window: Dict[str, List[Tuple]], attempts: int
                ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """KSO on one window from its start state (``ik_eval.check_plan``):
    ``(feasible, error, violations of the last failed solve)``."""
    ok, err, snapshot, _results = _window_kso_results(seg_task, window, attempts)
    return ok, err, snapshot


def _window_kso_results(seg_task: TaskContext, window: Dict[str, List[Tuple]], attempts: int
                        ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """``_window_kso`` plus the KSO results of a feasible solve (None otherwise)."""
    import ik_eval

    cost, results, err, snapshot = ik_eval.check_plan_violations(seg_task, window, attempts)
    return cost is not False, err, snapshot, (results if cost is not False else None)


def _route_seeded(seg_task: TaskContext, window: Dict[str, List[Tuple]], start: Any, *, nodes_per_mode: int,
                  time_per_mode: float, weight_scale: Optional[Mapping[str, float]], name_suffix: str,
                  ) -> Tuple[Optional[_Problem], Optional[_Solved], str]:
    """One re-solve of a failed window seeded from KSO on that window (see
    TO_ROUTE_SEED): ``(problem, solved, note)``, or ``(None, None, why not)``
    when KSO finds no solution or the build fails. Never raises."""
    try:
        ok, err, snap, kso = _window_kso_results(seg_task, window, TO_WINDOW_KSO_ATTEMPTS)
        if not ok or not kso:
            worst = sorted(((snap or {}).get("violations") or []), key=lambda v: -float(getattr(v, "value", 0) or 0))[:3]
            if worst:
                print("[to_eval] route seed KSO violations: " + "; ".join(
                    f"mode {getattr(v, 'mode', None)} {getattr(v, 'subject', '')} {getattr(v, 'family', '')} "
                    f"{getattr(v, 'other', '')} {float(getattr(v, 'value', 0) or 0):.3g}" for v in worst), flush=True)
            return None, None, "KSO found no solution" + (f" ({err})" if err else "")
        prob = _build_problem(seg_task, window, nodes_per_mode=nodes_per_mode, time_per_mode=time_per_mode,
                              weight_scale=weight_scale, name_suffix=name_suffix)
        return prob, _solve_problem(prob, kso_results=kso, start=start, force_seed=True), "seeded from window KSO"
    except Exception as e:  # noqa: BLE001 - the retry is best effort
        print(f"[to_eval] route seed failed: {type(e).__name__}: {e}", flush=True)
        return None, None, f"error: {type(e).__name__}"


def _origin_of(origin: Sequence[Optional[int]], mode: int) -> Optional[int]:
    """Original-plan index of repaired-plan ``mode`` (the next original
    mode when ``mode`` lies inside an inserted walk)."""
    for m in range(mode, len(origin)):
        if origin[m] is not None:
            return origin[m]
    return None


def _solve_segmented(task: TaskContext, plan: Dict[str, List[Tuple]], bounds: List[int], *,
                     nodes_per_mode: int, time_per_mode: float,
                     weight_scale: Optional[Mapping[str, float]],
                     kso_results: Optional[Mapping[str, Any]]) -> ToSolve:
    """Solve plan segments [bounds[i], bounds[i+1]) in order, each from the
    previous segment's final state, and stitch them (to_segments). With
    SHINKA_TO_SEGMENT_LOOKAHEAD each intermediate segment is solved past its
    cut (to_segments.windows) and cut back before stitching. A segment that
    neither converges nor passes the physical check is first repaired
    (TO_REPAIR: the walking steps step_feedback asks for are inserted where
    TO's own solution strides too far, _step_repair; the plan grows and the
    later cuts shift) and re-solved from the same start; when it still
    fails, TO_SEGMENT_FAIL_FAST stops there and the rest of the plan is held
    in place (to_segments.pad_hold). The result describes the repaired plan
    (``ToSolve.plan`` / ``repairs``)."""
    import gc

    import plan_checks
    import to_segments as ts

    plan = {ch: list(seq) for ch, seq in plan.items()}
    bounds = [int(x) for x in bounds]
    origin: List[Optional[int]] = list(range(plan_checks.n_modes(plan)))   # repaired mode -> original
    npm = nodes_per_mode
    rid = task.robot_id
    cfg = task.scene.robots[rid]

    def outcome(w: Tuple[int, int, int]) -> ts.SegmentOutcome:
        a, b, e = w
        return ts.SegmentOutcome(modes=(a, b), extra={"window": [a, e]} if e != b else {})

    wins = ts.windows(bounds, TO_SEGMENT_LOOKAHEAD)
    outcomes = [outcome(w) for w in wins]
    parts: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    repairs: List[Dict[str, Any]] = []
    repaired_here = 0          # repairs of the segment being solved
    route_seeded_here = False  # the segment being solved was re-solved from a KSO seed
    route_note: Optional[Dict[str, Any]] = None
    last_stride: Dict[Optional[int], float] = {}   # longest stride seen per repaired (original) mode
    stop_report: Optional[Dict[str, Any]] = None
    stopped = False
    start, collision_index = None, None
    cache = segment_cache.open_cache() if TO_INCREMENTAL else None
    settings = _cache_settings(task, npm, time_per_mode, weight_scale) if cache is not None else {}
    kso_feasible: Optional[bool] = None
    cached_modes, prefix_cached = 0, True
    print(f"[to_eval] segmented TO: {plan_checks.n_modes(plan)} modes as {len(wins)} segments "
          f"{[(a, b) for a, b, _ in wins]}, solved over {[(a, e) for a, _, e in wins]}"
          + (f"; incremental, cache {cache.dir} ({len(cache)} windows)" if cache is not None else ""), flush=True)
    i = 0
    while i < len(wins):
        a, b, e = wins[i]
        n = plan_checks.n_modes(plan)
        last = e == n
        goal, kind = (task.terminal_goal, "task") if last else _segment_goal(task, kso_results, e)
        win_start = start
        if cache is not None:
            cache_key = segment_cache.key(plan, e, win_start, settings)
            rec = cache.get(cache_key)
            if rec is not None:
                o = rec["outcome"]
                o.extra = dict(o.extra, cached=True)
                outcomes[i] = o
                rows += ts.shift_rows(rec["rows"], a * npm)
                parts.append(rec["part"])
                start = rec["end"]
                if prefix_cached:
                    cached_modes = b
                if kso_feasible is None:
                    kso_feasible = True
                print(f"[to_eval] segment {i} modes [{a},{b}) reused from the cache (status {o.status})", flush=True)
                repaired_here = 0
                i += 1
                continue
        prefix_cached = False
        seg_task = _segment_task(task, start, goal)
        if TO_INCREMENTAL and a == 0:
            # KSO gates only the first window, from the task's own initial
            # state, where its verdict is the full-plan KSO's for a short plan.
            # From a TO end state it is not reliable: on three_shelf gen_25 it
            # found no solution for windows TO then converged on, so later
            # windows go straight to TO (its violations and the physical
            # check explain a failure).
            ok, err, snap = _window_kso(seg_task, ts.slice_plan(plan, a, e), TO_WINDOW_KSO_ATTEMPTS)
            kso_feasible = ok
            if not ok:
                from dataclasses import replace as _replace

                shifted = [_replace(v, mode=None if v.mode is None else int(v.mode) + a)
                           for v in (snap or {}).get("violations", [])]
                o = outcomes[i]
                o.status, o.goal = 2, kind
                o.fail_reason = f"kso: no kinematic solution for modes [{a},{e})" + (f" ({err})" if err else "")
                o.extra["kso"] = {"max_violation": None if not snap else snap.get("max_violation"),
                                  "violations": [v.to_dict() for v in shifted], "error": err}
                print(f"[to_eval] segment {i} modes [{a},{e}): KSO found no kinematic solution"
                      + (f" ({err})" if err else ""), flush=True)
                stopped = True
                break
        prob = _build_problem(seg_task, ts.slice_plan(plan, a, e), nodes_per_mode=npm,
                              time_per_mode=time_per_mode, weight_scale=weight_scale,
                              name_suffix="_segkso" if kind == "kso" else "")
        if TO_SEGMENT_REF == "initial" and start is not None:
            _set_reference(prob, task.resolved_state)
        solved = _solve_problem(prob, kso_results=_slice_kso(kso_results, a, e), start=start)
        if collision_index is None:
            collision_index = getattr(prob.to_task, "collision_index", None)
        # A numerical failure is never a solved window, whatever the physical
        # check says of its iterate (see _physical_choice).
        bad = solved.status == 2 or (solved.status != 0 and not (solved.physical and solved.physical.get("pass")))
        if bad and TO_ROUTE_SEED and not route_seeded_here:
            route_seeded_here = True
            print(f"[to_eval] segment {i} modes [{a},{e}) failed cold; re-solving seeded from window KSO", flush=True)
            del prob
            gc.collect()
            prob2, solved2, note = _route_seeded(seg_task, ts.slice_plan(plan, a, e), start, nodes_per_mode=npm,
                                                 time_per_mode=time_per_mode, weight_scale=weight_scale,
                                                 name_suffix="_segkso" if kind == "kso" else "")
            ok2 = solved2 is not None and not (solved2.status == 2 or (
                solved2.status != 0 and not (solved2.physical and solved2.physical.get("pass"))))
            print(f"[to_eval] segment {i} route seed: {note}" + ("" if solved2 is None else
                  f", status {solved2.status}, max violation {solved2.max_violation:.3g}, valid {ok2}"), flush=True)
            route_note = {"note": note, "valid": bool(ok2),
                          "status": None if solved2 is None else int(solved2.status)}
            if ok2:
                prob, solved, bad = prob2, solved2, False
            else:
                # Keep the cold solve (its rows and report describe the failure
                # the step repair reads); rebuild so the repair path below and
                # the bookkeeping see a live problem.
                del prob2
                gc.collect()
                prob = _build_problem(seg_task, ts.slice_plan(plan, a, e), nodes_per_mode=npm,
                                      time_per_mode=time_per_mode, weight_scale=weight_scale,
                                      name_suffix="_segkso" if kind == "kso" else "")
        if bad and TO_REPAIR and repaired_here < TO_REPAIR_MAX and len(repairs) < TO_REPAIR_TOTAL:
            fix = _step_repair(task, plan, a, e, solved.results, npm)
            if fix is not None:
                before, steps, reason, *rest = fix
                stride = float(rest[0]) if rest else None
                orig = _origin_of(origin, before)
                prev = last_stride.get(orig)
                if stride is not None and prev is not None and stride >= 0.8 * prev:
                    # The steps added before this mode did not shorten the stride
                    # (layered_pallet gen_9: 1.49 -> 2.08 -> 1.85 -> 1.93 m over four
                    # repairs): the window fails for another reason, stop repairing.
                    print(f"[to_eval] repair before mode {orig} did not shorten the stride ({prev:.2f} -> "
                          f"{stride:.2f} m); no further repair", flush=True)
                    fix = None
                elif stride is not None:
                    last_stride[orig] = stride
            new_plan = None if fix is None else plan_checks.insert_walk(plan, fix[0], fix[1])
            if fix is not None and new_plan is not None:
                before, steps, reason = fix[0], fix[1], fix[2]
                added = plan_checks.walk_mode_count(steps)
                repairs.append({"segment": i, "before_mode": before, "before_mode_original": _origin_of(origin, before),
                                "steps": steps, "modes_added": added, "reason": reason})
                print(f"[to_eval] repair: walk(plan, {steps}) inserted before mode {before} ({reason}); "
                      f"re-solving segment {i}", flush=True)
                plan = new_plan
                origin = origin[:before] + [None] * added + origin[before:]
                bounds = _resplit(bounds, a, before, added, plan)
                wins = ts.windows(bounds, TO_SEGMENT_LOOKAHEAD)
                outcomes = outcomes[:i] + [outcome(w) for w in wins[i:]]
                kso_results = None       # its keyframes no longer line up with the plan
                repaired_here += 1
                del prob
                gc.collect()
                continue
        chosen, iterate = solved.results, solved.iterate
        if bad and solved.best is not None and iterate != "best":
            # Neither iterate is valid: continue from the lowest-residual one,
            # not from wherever the solve stopped.
            chosen, iterate = solved.best, "best"
        keep = (b - a) * npm
        if e != b:
            chosen = ts.truncate_results(chosen, keep)
            solved.rows = [r for r in solved.rows if int(r["node"]) < keep]
        o = outcomes[i]
        o.status, o.fail_reason, o.outer_iters = solved.status, solved.fail_reason, solved.outer_iters
        o.best_prim, o.rescues, o.iterate = solved.best_prim, solved.rescues, iterate
        o.constraint_violation, o.max_violation = solved.constraint_violation, solved.max_violation
        o.physical_pass = None if solved.physical is None else bool(solved.physical.get("pass"))
        o.goal, o.build_time_s, o.solve_time_s = kind, prob.build_time_s, solved.solve_time_s
        o.joint_excess = 0.0 if start is None else start.joint_excess
        if repaired_here:
            o.extra["repairs"] = repaired_here
        if route_note is not None:
            o.extra["route_seed"] = route_note
        if solved.stalled:
            o.extra["stalled"] = True
        rows += ts.shift_rows(solved.rows, a * npm)
        parts.append(chosen)
        print(f"[to_eval] segment {i} modes [{a},{b}) solved over [{a},{e}): status {o.status}, max violation "
              f"{o.max_violation:.3g}, best {o.best_prim:.3g}, physical {o.physical_pass}, "
              f"build {o.build_time_s:.0f}s solve {o.solve_time_s:.0f}s", flush=True)
        del prob
        gc.collect()
        if solved.status == 2:
            break
        if bad and TO_SEGMENT_FAIL_FAST:
            stopped, stop_report = True, _shift_report(solved.physical, a, npm)
            print(f"[to_eval] stopping at segment {i}: not converged and not physically valid", flush=True)
            break
        nxt = ts.start_from_results(chosen, rid, cfg.nq, cfg.q_min, cfg.q_max)
        # A non-converged end can sit past a joint limit; the next segment
        # starts clipped, so end this one there too.
        parts[-1] = ts.with_end_state(chosen, nxt, rid)
        if cache is not None and solved.status == 0:
            # Converged windows only: a physically valid but non-converged
            # window still counts for this plan, but its end state is not a
            # start to build on (layered_pallet gen_32 reused such a window
            # and the next one's presolve was infeasible at once).
            try:
                cache.put(segment_cache.key(plan, e, win_start, settings),
                          {"outcome": o, "part": parts[-1], "rows": list(solved.rows), "end": nxt,
                           "window": ts.slice_plan(plan, a, e)})
            except Exception as ex:  # noqa: BLE001 - the cache is an optimisation
                print(f"[to_eval] cache write failed: {type(ex).__name__}: {ex}", flush=True)
        start = nxt
        repaired_here = 0
        route_seeded_here, route_note = False, None
        i += 1

    n = plan_checks.n_modes(plan)
    agg = ts.aggregate(outcomes)
    failed = any(o.status == 2 for o in outcomes)
    if parts:
        stitched = ts.stitch_results(parts)
        if failed or stopped:
            stitched = ts.pad_hold(stitched, n * npm, time_per_mode / npm)
    else:
        # Stopped before the first window was solved: hold the initial state.
        rs = task.resolved_state
        stitched = ts.hold_results(rs.robots[rid].x, {o: st.x for o, st in rs.env_objects.items()},
                                   list(cfg.patch_ees), cfg.nv, n * npm, time_per_mode / npm, rid)
    physical = stop_report
    if TO_PHYSICAL_CHECK and agg["status"] != 0 and not failed and not stopped:
        try:
            import trajectory_check

            physical = trajectory_check.check(task, plan, stitched, npm)
        except Exception as e:  # noqa: BLE001 - the check is advisory
            print(f"[to_eval] physical check of the stitched trajectory failed: {type(e).__name__}: {e}", flush=True)
    # Modes solved: up to the first window that failed, or that was never
    # solved (status None after a stop) — a stop after a "physically valid"
    # numerical failure used to report the whole plan as solved.
    first_bad = next((o.modes[0] for o in outcomes
                      if o.status is None or o.status == 2 or (o.status != 0 and not o.physical_pass)), None)
    jerk = ts.jerk_cost(stitched["vdot_rob"], time_per_mode / npm)
    out = ToSolve(
        status=agg["status"], converged=agg["converged"],
        cost=ts.to_cost(agg["constraint_violation"], jerk, n * npm, n),
        results=stitched, num_modes=n, num_nodes=n * npm,
        constraint_violation=agg["constraint_violation"],
        max_constraint_violation=agg["max_constraint_violation"], jerk_cost=jerk,
        violations=_translate_rows(rows, task, plan, npm, collision_index),
        fail_reason=agg["fail_reason"], rescues=agg["rescues"], physical=physical,
        iterate=agg["iterate"], goal=goal_check(task, stitched), segments=[o.to_dict() for o in outcomes],
        solved_modes=n if first_bad is None else first_bad,
        repairs=repairs, plan=plan if repairs else None,
        kso_feasible=kso_feasible, cached_modes=cached_modes,
    )
    if TO_SEGMENT_POLISH and not failed and not stopped and not out.converged:
        out = _polish(task, plan, out, nodes_per_mode=npm, time_per_mode=time_per_mode,
                      weight_scale=weight_scale)
    return out


def _polish(task: TaskContext, plan: Dict[str, List[Tuple]], stitched: ToSolve, *, nodes_per_mode: int,
            time_per_mode: float, weight_scale: Optional[Mapping[str, float]]) -> ToSolve:
    """Full-horizon solve warm-started from the stitched trajectory; kept
    when it converges, or is physically valid with a lower max violation."""
    import time

    import to_segments as ts

    t0 = time.time()
    try:
        prob = _build_problem(task, plan, nodes_per_mode=nodes_per_mode, time_per_mode=time_per_mode,
                              weight_scale=weight_scale, presolve=TO_SEGMENT_POLISH_PRESOLVE)
        x0, u0 = ts.warm_start_dicts(stitched.results)
        u0["t_scale"] = ts.polish_t_scale(stitched.results["step_ts"], prob.to_task.dt_traj)
        polished = _finalize(prob, _solve_problem(prob, guess=(x0, u0), max_iters=TO_SEGMENT_POLISH_ITERS))
        del prob
    except Exception as e:  # noqa: BLE001 - keep the stitched result
        print(f"[to_eval] polish failed: {type(e).__name__}: {e}", flush=True)
        stitched.polish = {"error": f"{type(e).__name__}: {e}", "kept": False, "time_s": time.time() - t0}
        return stitched
    keep = polished.converged or (polished.valid and polished.max_constraint_violation
                                  < stitched.max_constraint_violation)
    record = {"status": polished.status, "max_violation": polished.max_constraint_violation,
              "valid": polished.valid, "kept": bool(keep), "time_s": time.time() - t0}
    print(f"[to_eval] polish: status {polished.status}, max violation "
          f"{polished.max_constraint_violation:.3g}, valid {polished.valid}, kept {keep}", flush=True)
    chosen = polished if keep else stitched
    chosen.segments, chosen.polish = stitched.segments, record
    chosen.solved_modes, chosen.repairs, chosen.plan = stitched.solved_modes, stitched.repairs, stitched.plan
    chosen.kso_feasible, chosen.cached_modes = stitched.kso_feasible, stitched.cached_modes
    return chosen


def _with_torques(to_task: Any, results: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of moto *results* with joint torques, computed exactly as faro's
    ``ToTask.get_results(include_torque=True)`` does for the final iterate."""
    import numpy as np
    from faro.ocp.contact_wrenches import robot_wrench_trajectory

    out = dict(results)
    total = robot_wrench_trajectory(to_task.scene, to_task.formulation.robots, out["robot_xs"], out["obj_xs"],
                                    out["rob_Ws"], out["obj_Ws"], to_task.interface_ohs)
    torques, limits = {}, {}
    for rid, robot in to_task.formulation.robots.items():
        xs = out["robot_xs"][rid]
        torques[rid] = np.zeros((to_task.N, robot.nv - 6))
        for k in range(to_task.N):
            torques[rid][k] = np.asarray(to_task.formulation.get_torques_funcs[rid](
                xs[k + 1, :robot.nq], xs[k + 1, robot.nq:robot.nq + robot.nv], out["vdot_rob"][rid][k],
                *(total[rid][patch][k] for patch in robot.cfg.patch_ees))[1]).reshape(-1)
        limits[rid] = robot.cfg.tau_max.copy()
    out["torques"], out["torque_limit"] = torques, limits
    return out


def _physical_choice(task: TaskContext, plan: Dict[str, List[Tuple]], final: Dict[str, Any],
                     best: Optional[Dict[str, Any]], nodes_per_mode: int, status: int = 1
                     ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
    """(report, results, 'final'|'best'): the final iterate if it passes
    the physical check, else the best iterate if that passes, else the
    final one with its report. After a numerical failure (status 2) the
    final iterate is the solver's wreck — after a presolve failure the
    initial pose held still, which a walking window passes trivially
    (shelf_unstack gen_22: "physically valid", nothing moved) — so only
    the best iterate is a candidate then. Never raises."""
    try:
        import trajectory_check

        if status == 2:
            if best is None:
                return None, final, "final"
            brep = trajectory_check.check(task, plan, best, nodes_per_mode)
            return (brep, best, "best") if brep["pass"] else (None, final, "final")
        rep = trajectory_check.check(task, plan, final, nodes_per_mode)
        if rep["pass"] or best is None:
            return rep, final, "final"
        brep = trajectory_check.check(task, plan, best, nodes_per_mode)
        if brep["pass"]:
            return brep, best, "best"
        return rep, final, "final"
    except Exception as e:  # noqa: BLE001 - the check is advisory
        print(f"[to_eval] physical check failed: {type(e).__name__}: {e}", flush=True)
        return None, final, "final"


def _violation_rows(to_task: Any) -> List[Dict[str, Any]]:
    """Raw constraint-violation rows of the collision SQP's current iterate
    (``node`` = interval index; diagnostics only: never raises)."""
    try:
        from violation_feedback import REPORT_TOL

        return list(to_task.moto_wrap.filtered_constraint_violations(to_task.sqp_coll, tol=REPORT_TOL))
    except Exception as e:  # noqa: BLE001
        print(f"[to_eval] violation rows failed: {type(e).__name__}: {e}")
        return []


def _translate_rows(rows: List[Dict[str, Any]], task: TaskContext, plan: Dict[str, List[Tuple]],
                    nodes_per_mode: int, collision_index: Any) -> List[Dict[str, Any]]:
    """Violation rows in plan vocabulary (diagnostics only: never raises)."""
    try:
        from violation_feedback import NameResolver, translate_to

        resolver = NameResolver(task.scene, plan, collision_index)
        return [v.to_dict() for v in translate_to(rows, resolver, nodes_per_mode)]
    except Exception as e:  # noqa: BLE001
        print(f"[to_eval] violation translation failed: {type(e).__name__}: {e}")
        return []
