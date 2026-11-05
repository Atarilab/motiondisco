"""
Wrapper around faro's KSO (kinematic sequence optimization,
formerly "seqik") evaluator.

Keeps up to three lazy KSO solver pools per process, keyed by how the
terminal goal is treated: ``hard`` (as authored), ``soft`` (a copy with
``kso="soft"``) and ``none`` (no goal at all; the same pool as ``soft``
when the task has no KSO goal). A pool holds one acados solver per
horizon length, built (or loaded from the codegen cache) the first time
a plan of that length needs it. Lengths are capped at MAX_NUM_NODES. The
pools mirror ``faro.evaluators.kso.create_kso_func`` (same solver
settings and feasibility rule) but are built here so the per-horizon
``KsoTask``s stay reachable for ``get_results()`` and
``get_performance_metrics()``.

Public entrypoint
    evaluate_ik(plan, terminal_goal=None)
        -> (cost, results, diagnosis, ik_mode, ik_hard_failed)

``plan`` uses flat channel names (``left_ee``, ``white_box``, ...); it
is converted to faro interface keys here.

Feasibility is decided by solving the FULL plan, like faro's own
``create_kso_func``. A KSO solve is not deterministic: it stops just under
the 1e-2 violation cutoff, and identical inputs pass or fail from run to
run (measured ~80-100% per solve on feasible truck_unload_stack carries).
So each full-plan solve gets up to KSO_ATTEMPTS tries (default 3,
``SHINKA_KSO_ATTEMPTS``), each retry on a freshly built solver.

Verdict, selected by ``terminal_goal.kso``:

  - "hard": hard KSO on the full plan (with retries) -> ik_mode="hard";
    otherwise soft KSO on the full plan -> ik_mode="soft",
    ik_hard_failed=True.
  - "soft", "off", or terminal_goal=None: soft KSO on the full plan.

Diagnosis, only for plans that failed every attempt (all on the ``none``
pool, so prefixes are not pulled toward the goal):

  1. Goal miss: if the task has a KSO goal and the full plan solves
     without it (GOAL_CHECK_ATTEMPTS tries), the contacts are fine and
     only the target is missed -> ``kind="goal_missed"`` with per-target
     final position, distance and tolerance.
  2. Prefix walk: bisection over prefix lengths (2..n), PREFIX_ATTEMPTS
     tries each (``SHINKA_KSO_PREFIX_ATTEMPTS``, default 4; feasible
     prefixes usually pass on the first try), locating the first
     infeasible prefix -> ``kind="prefix"`` with that prefix's own pooled
     violations and the transition it ends with (``break_event``).
     ``SHINKA_KSO_WALK=linear`` keeps the old 2..n walk.
  3. Repair probe (``SHINKA_KSO_PROBES``, default 2, 0 disables): insert
     two walking steps before the breaking mode, then before the first
     grasp, and re-solve the edited plan once on the plan's own goal
     path; the first edit that works is reported (``repair_probe.py``).

Everything diagnostic stops when the evaluation time budget
(``SHINKA_EVAL_BUDGET_S``, default 1500 s) is spent: the goal-miss check
is skipped, the prefix walk returns the bracket it has (``timed_out``),
the probe is skipped. ``evaluate_ik(on_full_failure=fn)`` calls ``fn`` with
a provisional diagnosis (``kind="pending"``, the full-plan attempts'
violations) after the full-plan solves failed and before the diagnosis,
so the caller can persist a record before the slow part.

Horizon buckets: an acados solver is compiled for one horizon length, so
a plan is padded (``SHINKA_KSO_PAD``: its first mode repeated before it,
the default, or its last mode after it; no contact changes either way, so
the padded nodes hold a state and feasibility is unchanged, and the goal
still applies at the end) to the next multiple of ``SHINKA_KSO_BUCKET``
modes (default 10, 0 = exact lengths) and solved on that bucket's solver;
results and violations are mapped back to the plan's own states.
Without buckets a 4-box search compiled 4-7 solvers per plan (3-14 min
each); with them every plan of the run shares at most
``3 * ceil(MAX_NUM_NODES / bucket)`` compiles. Concurrent evaluators take
a file lock per solver name while compiling.

Returns:
    cost          : False on infeasibility, otherwise float outcome-scorer
                    cost (0.0 when the scene has no scorer).
    results       : KsoTask.get_results() of the accepted solve; for a goal
                    miss, of the goal-free solve (so the motion can still
                    be replayed); otherwise None.
    diagnosis     : None when feasible, else a dict with ``kind``
                    (prefix | goal_missed | unknown), last_feasible_modes,
                    first_infeasible_modes, total_modes, solver_error,
                    kso_attempts, best_max_violation, violation_details,
                    prefix_attempts, prefix_trials, prefix_violation_details,
                    break_event, probe, goal_residuals, [ik_hard_failed].
    ik_mode       : "hard" or "soft" — which path produced the verdict.
    ik_hard_failed: True iff hard KSO was attempted and did not converge
                    on the full plan (independent of whether soft rescued).

The active scene comes from the SHINKA_LOCOMANIP_{SCENE,IC} env vars
(see _task_loader.load_task); the terminal goal and outcome scorer are
auto-discovered from the scene's goals.py (if any).

Building a pool triggers acados / pinocchio / casadi codegen (about two
minutes per new horizon length). Make sure ACADOS_SOURCE_DIR and
LD_LIBRARY_PATH are set in the worker environment. Codegen caches land
in the launch cwd under ``solvers/``.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import os
import sys
import time
from dataclasses import is_dataclass, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Make sibling modules importable regardless of caller cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import kso_warm_start  # noqa: E402
import plan_checks  # noqa: E402
import repair_probe  # noqa: E402
from _task_loader import load_task, scene_faces, to_faro_plan  # noqa: E402


# Maximum number of contact modes the wrapper can score. Solvers are
# built lazily per horizon length, so this only caps plan length.
# Override at runtime with: SHINKA_IK_MAX_NUM_NODES=24 python ...
MAX_NUM_NODES = int(os.environ.get("SHINKA_IK_MAX_NUM_NODES", "50"))

# Attempts per full-plan KSO solve (each retry on a fresh solver), per
# prefix in the diagnostic walk, and for the goal-free full-plan check.
KSO_ATTEMPTS = max(1, int(os.environ.get("SHINKA_KSO_ATTEMPTS", "3")))
PREFIX_ATTEMPTS = max(1, int(os.environ.get("SHINKA_KSO_PREFIX_ATTEMPTS", "4")))
GOAL_CHECK_ATTEMPTS = 2
# Repair probes per failed plan (0 disables) and walk steps per probe.
PROBES = max(0, int(os.environ.get("SHINKA_KSO_PROBES", "2")))
PROBE_STEPS = 2
# Diagnostics stop once this much of the evaluation's wall clock is used.
EVAL_BUDGET_S = float(os.environ.get("SHINKA_EVAL_BUDGET_S", "1500"))
# "bisect" (default) or "linear" prefix walk.
WALK_MODE = os.environ.get("SHINKA_KSO_WALK", "bisect").strip().lower()
# Horizon bucket (modes): plans are padded to a multiple of it and share
# that length's compiled solver (module docstring). 0 = one solver per
# exact length.
KSO_BUCKET = max(0, int(os.environ.get("SHINKA_KSO_BUCKET", "10")))
# Where the padding goes. "start" repeats the plan's first mode before it:
# the initial state is fixed, so the padded nodes hold it at zero cost
# (the state cost's reference is that pose) and the plan itself is solved
# as if unpadded. "end" repeats the last mode after it: those nodes then
# carry the end pose's distance-from-start cost once more each, which
# changes the objective's balance for long plans.
KSO_PAD = os.environ.get("SHINKA_KSO_PAD", "start").strip().lower()

# Same solver settings as faro.evaluators.kso.create_kso_func.
_CODE_GEN_PATH = "solvers/c_generated_code"
_SOLVER_CONFIG_PATH = "solvers"
_FEASIBILITY_TOL = 1e-2
_SQP_ITERS_PER_COLL = 5

KINDS = ("hard", "soft", "none")

Evaluator = Callable[[Dict[tuple, List[Tuple]]], Any]
Pool = Tuple[Evaluator, Dict[int, Any]]

_pools: Dict[str, Optional[Pool]] = {kind: None for kind in KINDS}


def _kso_goal(terminal_goal: Any, kind: str) -> Any:
    """Terminal goal for a pool: as authored (hard), a soft copy, or none."""
    if kind == "none" or terminal_goal is None:
        return None
    if kind == "soft" and terminal_goal.kso == "hard":
        if is_dataclass(terminal_goal):
            return replace(terminal_goal, kso="soft")
        soft = copy.copy(terminal_goal)
        soft.kso = "soft"
        return soft
    return terminal_goal


def _has_kso_goal(terminal_goal: Any) -> bool:
    return (
        terminal_goal is not None
        and getattr(terminal_goal, "kso", "off") != "off"
        and not (hasattr(terminal_goal, "is_empty") and terminal_goal.is_empty())
    )


def bucket_length(n: int) -> int:
    """Horizon an *n*-mode plan is solved on: *n* rounded up to a multiple
    of KSO_BUCKET, capped at MAX_NUM_NODES (*n* itself without buckets)."""
    if KSO_BUCKET <= 0 or n >= MAX_NUM_NODES:
        return n
    return min(MAX_NUM_NODES, -(-n // KSO_BUCKET) * KSO_BUCKET)


def pad_plan(plan: Dict[Any, List[Tuple]], length: int, where: Optional[str] = None) -> Dict[Any, List[Tuple]]:
    """Copy of *plan* (flat or faro keys) padded to *length* modes by
    repeating its first mode before it (``where="start"``) or its last
    mode after it (``"end"``; default KSO_PAD); unchanged when it is
    already that long."""
    where = KSO_PAD if where is None else where
    if where not in ("start", "end"):
        raise ValueError(f"unknown padding side {where!r}")
    out: Dict[Any, List[Tuple]] = {}
    for ch, seq in plan.items():
        seq = list(seq)
        if not seq:
            raise ValueError(f"channel {ch!r} has no modes")
        pad = [seq[0] if where == "start" else seq[-1]] * max(0, length - len(seq))
        out[ch] = pad + seq if where == "start" else seq + pad
    return out


def trim_results(results: Optional[Dict[str, Any]], n: int, length: int,
                 where: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """KSO results of a *length*-mode padded solve cut back to the *n*-mode
    plan's states (mode 0, one per transition, plus the endpoint state when
    the solve had a goal, i.e. one state more than modes). Start padding
    holds the initial state over the first ``length - n`` states, which are
    dropped; end padding holds the final pose between state n-1 and the
    endpoint, and those states are dropped."""
    if results is None or length <= n:
        return results
    import numpy as np

    where = KSO_PAD if where is None else where
    pad = length - n
    out = dict(results)
    for key in ("robot_xs", "obj_xs"):
        arrays = results.get(key)
        if not isinstance(arrays, dict):
            continue
        trimmed = {}
        for name, x in arrays.items():
            x = np.asarray(x)
            if len(x) not in (length, length + 1):   # not a per-state array
                trimmed[name] = x
                continue
            if where == "start":
                keep = list(range(pad, len(x)))
            else:
                keep = list(range(n)) + ([len(x) - 1] if len(x) > length else [])
            trimmed[name] = x[keep]
        out[key] = trimmed
    return out


def shift_modes(violations: List[Any], pad: int) -> List[Any]:
    """Violations translated against a start-padded plan, moved back to
    the plan's own mode numbers (the held initial states become mode 0)."""
    if pad <= 0:
        return list(violations)
    out = []
    for v in violations:
        mode = getattr(v, "mode", None)
        out.append(v if mode is None else replace(v, mode=max(0, int(mode) - pad)))
    return out


@contextlib.contextmanager
def _build_lock(name: str) -> Iterator[None]:
    """One acados build at a time across evaluator processes: two of them
    compiling the same bucket would write the same directory, and two
    compiles of different buckets together exceed the memory the search
    runs under (an 80-mode 4-box compile was OOM-killed at 16 GB on its
    own). Loading an already compiled solver holds the lock only briefly."""
    os.makedirs(_SOLVER_CONFIG_PATH, exist_ok=True)
    with open(os.path.join(_SOLVER_CONFIG_PATH, ".kso_build.lock"), "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _make_kso_task(task: Any, terminal_goal: Any, num_modes: int) -> Any:
    """Build one KsoTask + acados solver for a *num_modes* horizon."""
    from faro.configs.identifiers import solver_name
    from faro.solvers.acados import AcadosBuildConfig
    from faro.tasks.kso_task import KsoTask

    scene = task.scene
    if terminal_goal is not None:
        terminal_goal.validate(scene)
    suffix = terminal_goal.kso if terminal_goal is not None else "none"
    kso_task = KsoTask(
        scene=scene,
        resolved_state=task.resolved_state,
        terminal_goal=terminal_goal,
    )
    # Hex-encoded like faro's own evaluator: scene IDs are dotted module
    # paths, which acados cannot use in a C identifier.
    name = solver_name("kso", scene.name, str(num_modes), suffix)
    with _build_lock(name):
        kso_task.build_solver(
            num_modes=num_modes,
            build_cfg=AcadosBuildConfig(
                name=name,
                solver_config_path=_SOLVER_CONFIG_PATH,
                code_gen_path=_CODE_GEN_PATH,
                force_recompile=False,
                qp_max_iter=100,
                max_nlp_iters=500,
                tol=_FEASIBILITY_TOL,
                free_init_state=False,
            ),
        )
    return kso_task


def _solve_kso(kso_task: Any, cms: Dict[tuple, List[Tuple]], outcome_scorer: Any) -> Any:
    """One KSO solve on a prepared task: False if infeasible, else the
    outcome-scorer cost (0.0 without a scorer)."""
    from faro.cost.kso_state_ctrl_cost import make_ocp_weights
    from faro.search.goal import SolverOutcome

    try:
        kso_task.update_mode_oh_via_tup(cms)
    except ValueError as e:
        # faro rejects the schedule itself (e.g. a contact that changes
        # surface without passing through free). Not a solver verdict:
        # raise so the message reaches the feedback as a solver error.
        raise ValueError(f"invalid contact schedule: {e}") from e
    kso_task.set_fixed_params(weights=make_ocp_weights(kso_task.scene.robots))
    kso_task.initialize_solver()
    # faro seeds every state, and linearises every collision pair, at the
    # initial standing pose. Plans that change support level (onto a table
    # and down its far side) get a level-aware seed and collisions
    # re-resolved there instead; otherwise the SQP stays in a basin with the
    # stepping foot 0.2 m short.
    kso_warm_start.apply(kso_task, cms)
    status = kso_task.solve(
        nlp_iters=100,
        print_stats=False,
        sqp_iters_per_collision=_SQP_ITERS_PER_COLL,
        res_ineq_tol=_FEASIBILITY_TOL,
    )
    perf_metrics = kso_task.get_performance_metrics()
    if status == 1 and perf_metrics["max_violation"] > _FEASIBILITY_TOL:
        return False
    if status not in (0, 1):
        raise ValueError(f"KSO solver returned unsupported status {status}.")
    if outcome_scorer is None:
        return 0.0
    outcome = SolverOutcome(
        converged=(status == 0),
        results=kso_task.get_results(),
        perf_metrics=perf_metrics,
        scene=kso_task.scene,
    )
    return float(outcome_scorer(outcome))


def _build_pool(kind: str) -> Pool:
    """Create an (evaluator, tasks_by_num_modes) pair for one goal kind.

    Solvers are built lazily, one per horizon length on first use.
    ``tasks_by_num_modes`` is returned alongside so callers can pull the
    trajectory and violations from the KsoTask of the most recent solve.
    """
    task = load_task()
    terminal_goal = _kso_goal(task.terminal_goal, kind)
    tasks_by_num_modes: Dict[int, Any] = {}

    def evaluate(cms: Dict[tuple, List[Tuple]]) -> Any:
        num_modes = len(next(iter(cms.values())))
        if not 2 <= num_modes <= MAX_NUM_NODES:
            raise ValueError(f"Num modes {num_modes} not in range 2 - {MAX_NUM_NODES}.")
        length = bucket_length(num_modes)
        if length not in tasks_by_num_modes:
            tasks_by_num_modes[length] = _make_kso_task(task, terminal_goal, length)
        return _solve_kso(tasks_by_num_modes[length], pad_plan(cms, length), task.outcome_scorer)

    return evaluate, tasks_by_num_modes


def check_plan(
    task: Any, plan: Dict[str, List[Tuple]], attempts: int = 1,
) -> Tuple[Any, Optional[Dict[str, Any]], Optional[str]]:
    """KSO-check one flat-channel *plan* against an explicit *task*.

    For callers whose ``TaskContext`` differs from the env-selected one
    (e.g. a perturbed start state in sweep_to_variants.py), so the cached
    pools do not apply. Builds a single solver for the plan's horizon
    (acados codegen is cached by formulation, so repeats are cheap).
    Same hard -> soft fallback as ``evaluate_ik``, up to ``attempts``
    solves per goal kind. Returns ``(cost_or_False, results_or_None,
    error_or_None)``; ``check_plan_violations`` has the last failed
    solve's violations.
    """
    cost, results, err, _ = _check_plan(task, plan, attempts)
    return cost, results, err


def _check_plan(task: Any, plan: Dict[str, List[Tuple]], attempts: int = 1,
                ) -> Tuple[Any, Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    """``check_plan`` plus the translated violations of the last failed
    solve (``{"max_violation", "violations"}``; None when there was none)."""
    from violation_feedback import NameResolver, translate_kso

    n = len(next(iter(plan.values()), []))
    if n < 2:
        return False, None, "plan has fewer than 2 modes", None
    length = bucket_length(n)
    cms = pad_plan(to_faro_plan(task.scene, plan), length)
    kinds = ["hard", "soft"] if (
        task.terminal_goal is not None and task.terminal_goal.kso == "hard"
    ) else ["soft"]
    err: Optional[str] = None
    snapshot: Optional[Dict[str, Any]] = None
    for kind in kinds:
        for _attempt in range(max(1, int(attempts))):
            try:
                kso_task = _make_kso_task(task, _kso_goal(task.terminal_goal, kind), length)
                cost = _solve_kso(kso_task, cms, task.outcome_scorer)
            except Exception as e:  # noqa: BLE001 — reported to the caller
                err = f"{type(e).__name__}: {e}"
                if err.startswith("ValueError: invalid contact schedule"):
                    return False, None, err, None
                continue
            if cost is not False:
                return float(cost), trim_results(kso_task.get_results(), n, length), None, snapshot
            try:
                pm = kso_task.get_performance_metrics()
                padded = KSO_PAD == "start" and length > n
                resolver = NameResolver(kso_task.scene, pad_plan(plan, length) if padded else plan,
                                        getattr(kso_task, "collision_index", None),
                                        resolved_state=getattr(kso_task, "resolved", None))
                violations = translate_kso(pm, resolver)
                snapshot = {"max_violation": float(pm["max_violation"]),
                            "violations": shift_modes(violations, length - n) if padded else violations}
            except Exception as e:  # noqa: BLE001 — diagnostics only
                print(f"[ik_eval] violation translation failed: {type(e).__name__}: {e}")
    return False, None, err, snapshot


def check_plan_violations(task: Any, plan: Dict[str, List[Tuple]], attempts: int = 1,
                          ) -> Tuple[Any, Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    """``check_plan`` returning also the last failed solve's violations."""
    return _check_plan(task, plan, attempts)


def _pool_kind(kind: str) -> str:
    """``none`` shares the ``soft`` pool when the task has no KSO goal
    (identical formulations; avoids a second set of solver binaries)."""
    if kind == "none" and not _has_kso_goal(load_task().terminal_goal):
        return "soft"
    return kind


def _get_pool(kind: str) -> Pool:
    kind = _pool_kind(kind)
    if _pools[kind] is None:
        _pools[kind] = _build_pool(kind)
    return _pools[kind]  # type: ignore[return-value]


def _truncate_plan(
    plan: Dict[str, List[Tuple]], length: int,
) -> Dict[str, List[Tuple]]:
    """Return a copy of *plan* with every channel truncated to *length* modes."""
    return {ch: seq[:length] for ch, seq in plan.items()}


def _run_ik_once(
    plan: Dict[str, List[Tuple]], n: int, kind: str, fresh: bool = False,
) -> Tuple[Any, Optional[Dict[str, Any]], Optional[str]]:
    """
    Run the KSO evaluator on *plan* (assumed to have *n* modes) using
    the pool of *kind*. Returns (cost_or_False, results_or_None,
    error_or_None) — any solver exception message is returned alongside
    so the caller can surface it in feedback instead of losing it to
    stdout. ``fresh`` rebuilds the n-mode solver first (codegen is
    cached), so a retry does not inherit a failed solve's state.
    """
    length = bucket_length(n)
    try:
        ev, pool = _get_pool(kind)
        if fresh:
            pool.pop(length, None)
        cost = ev(to_faro_plan(load_task().scene, plan))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ik_eval] {err}")
        return False, None, err
    if cost is False:
        return False, None, None
    try:
        results = trim_results(pool[length].get_results(), n, length)
    except Exception as e:
        print(f"[ik_eval] failed to extract results: {type(e).__name__}: {e}")
        results = None
    return float(cost), results, None


def _violations_of(n: int, kind: str, plan: Dict[str, List[Tuple]]) -> Optional[Dict[str, Any]]:
    """Translated violations of the most recent n-mode solve in a pool:
    ``{"max_violation", "violations": [Violation, ...]}`` (None if unavailable)."""
    from violation_feedback import NameResolver, translate_kso

    pool = _pools.get(_pool_kind(kind))
    length = bucket_length(n)
    try:
        kso_task = pool[1].get(length) if pool else None
        if kso_task is None:
            return None
        pm = kso_task.get_performance_metrics()
        # Solver stages count the padded plan: translate against it and move
        # the modes back (end padding folds its extra stages onto the last
        # mode through the resolver's own clamp).
        padded = KSO_PAD == "start" and length > n
        resolver = NameResolver(
            kso_task.scene, pad_plan(plan, length) if padded else plan, getattr(kso_task, "collision_index", None),
            resolved_state=getattr(kso_task, "resolved", None),
        )
        violations = translate_kso(pm, resolver)
        if padded:
            violations = shift_modes(violations, length - n)
        return {"max_violation": float(pm["max_violation"]), "violations": violations}
    except Exception as e:  # noqa: BLE001 — diagnostics only
        print(f"[ik_eval] violation translation failed: {type(e).__name__}: {e}")
        return None


# Failed attempts of the last _solve_with_retries call per (n, kind):
# their translated violations, for feedback.
_failed_attempts: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}


def _solve_with_retries(
    plan: Dict[str, List[Tuple]], n: int, kind: str, attempts: int = KSO_ATTEMPTS,
) -> Tuple[Any, Optional[Dict[str, Any]], Optional[str]]:
    """Full-plan KSO solve with up to *attempts* tries (fresh solver per
    retry). Returns the first success, else (False, None, last_error)."""
    err: Optional[str] = None
    failed = _failed_attempts[(n, kind)] = []
    for attempt in range(attempts):
        cost, results, e = _run_ik_once(plan, n, kind, fresh=attempt > 0)
        if cost is False and not e:
            snap = _violations_of(n, kind, plan)
            if snap is not None:
                failed.append(snap)
        if cost is not False:
            if attempt:
                print(f"[ik_eval] {n}-mode {kind} KSO succeeded on attempt {attempt + 1}")
            return cost, results, None
        err = e or err
        if e and e.startswith("ValueError: invalid contact schedule"):
            break  # deterministic: retrying cannot help
    return False, None, err


def _pooled(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """kso_attempts / best_max_violation / violation_details of pooled attempts."""
    from violation_feedback import merge_attempts

    if not attempts:
        return {}
    return {
        "attempts": len(attempts),
        "best_max_violation": min(a["max_violation"] for a in attempts),
        "details": [v.to_dict() for v in merge_attempts([a["violations"] for a in attempts])],
    }


def _prefix_feasible(
    plan: Dict[str, List[Tuple]], k: int, n: int, attempts: int, trials: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    prefix = plan if k == n else _truncate_plan(plan, k)
    cost, _results, err = _solve_with_retries(prefix, k, "none", attempts)
    tries = len(_failed_attempts.get((k, "none"), [])) + (1 if cost is not False else 0)
    trials.append({"modes": k, "feasible": cost is not False, "tries": max(tries, 1)})
    return cost is not False, err


def _past(deadline: Optional[float]) -> bool:
    return deadline is not None and time.time() > deadline


def _no_walk(n: int, attempts: int) -> Dict[str, Any]:
    return {"kind": "unknown", "last_feasible_modes": None, "first_infeasible_modes": None,
            "total_modes": n, "solver_error": None, "prefix_attempts": attempts, "prefix_trials": [],
            "timed_out": True, "reason": "the evaluation time budget was spent before the prefix walk"}


def _bisect_prefix(
    plan: Dict[str, List[Tuple]], n: int, attempts: int, deadline: Optional[float] = None,
) -> Dict[str, Any]:
    """Locate the first infeasible prefix by bisection; the full plan is
    already known to be infeasible on the ``none`` pool. Past *deadline*
    the bracket found so far is returned (``timed_out``): ``hi`` is still
    a known infeasible prefix, ``lo`` a known feasible one."""
    if _past(deadline):
        return _no_walk(n, attempts)
    trials: List[Dict[str, Any]] = []
    err: Optional[str] = None
    timed_out = False
    ok, err = _prefix_feasible(plan, 2, n, attempts, trials)
    if not ok:
        lo, hi = None, 2
    else:
        lo, hi = 2, n
        while hi - lo > 1:
            if _past(deadline):
                timed_out = True
                break
            mid = (lo + hi) // 2
            ok, e = _prefix_feasible(plan, mid, n, attempts, trials)
            err = e or err
            lo, hi = (mid, hi) if ok else (lo, mid)
    out = {"kind": "prefix", "last_feasible_modes": lo, "first_infeasible_modes": hi,
           "total_modes": n, "solver_error": err, "prefix_attempts": attempts,
           "prefix_trials": trials}
    if timed_out:
        out["timed_out"] = True
    return out


def _linear_prefix(
    plan: Dict[str, List[Tuple]], n: int, attempts: int, deadline: Optional[float] = None,
) -> Dict[str, Any]:
    """The old walk: prefixes 2..n in order, stop at the first failure (or
    at the deadline, with the longest feasible prefix so far)."""
    if _past(deadline):
        return _no_walk(n, attempts)
    trials: List[Dict[str, Any]] = []
    last: Optional[int] = None
    for k in range(2, n + 1):
        if _past(deadline):
            return {"kind": "prefix", "last_feasible_modes": last, "first_infeasible_modes": n,
                    "total_modes": n, "solver_error": None, "prefix_attempts": attempts,
                    "prefix_trials": trials, "timed_out": True}
        ok, err = _prefix_feasible(plan, k, n, attempts, trials)
        if not ok:
            return {"kind": "prefix", "last_feasible_modes": last, "first_infeasible_modes": k,
                    "total_modes": n, "solver_error": err, "prefix_attempts": attempts,
                    "prefix_trials": trials}
        last = k
    return {"kind": "unknown", "last_feasible_modes": n, "first_infeasible_modes": None,
            "total_modes": n, "solver_error": None, "prefix_attempts": attempts,
            "prefix_trials": trials, "reason": "every prefix solved in the walk (solver noise)"}


def _goal_residuals(results: Dict[str, Any], terminal_goal: Any) -> Dict[str, Dict[str, Any]]:
    """Final position vs target for every owned position target."""
    import numpy as np

    out: Dict[str, Dict[str, Any]] = {}
    for targets, key in ((getattr(terminal_goal, "objects", {}), "obj_xs"),
                         (getattr(terminal_goal, "robots", {}), "robot_xs")):
        for name, target in targets.items():
            try:
                final = np.asarray(results[key][name])[-1, :3]
                goal = np.asarray(target.value, dtype=float)[:3]
            except Exception:  # noqa: BLE001 — diagnostics only
                continue
            tol = getattr(getattr(target, "kso_tolerance", None), "position", None)
            out[name] = {
                "final": [round(float(v), 3) for v in final],
                "target": [round(float(v), 3) for v in goal],
                "distance": round(float(np.linalg.norm(final - goal)), 3),
                "tolerance": None if tol is None else float(tol),
            }
    return out


def _transition_describer(plan: Dict[str, List[Tuple]]) -> Callable[[int], str]:
    names: Dict[int, str] = {}
    objects: set = set()
    try:
        scene = load_task().scene
        objects = set(scene.env_objects)
        for obj, faces in scene_faces(scene).items():
            for env_id, _self_id, face in faces:
                names[env_id] = f"{obj} {face}"
    except Exception:  # noqa: BLE001 — names are cosmetic
        pass
    return lambda mode: plan_checks.describe_transition(
        plan, mode, face_name=lambda i: names.get(i, f"face {i}"), objects=objects,
    )


def evaluate_ik(
    plan: Dict[str, List[Tuple]],
    terminal_goal: Any = None,
    on_full_failure: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[
    Any,
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    str,
    bool,
]:
    """
    Score a contact_mode_seq with the KSO evaluator.

    Full-plan solves with retries decide feasibility (hard first when
    ``terminal_goal.kso == "hard"``, then soft); the goal-miss check,
    prefix bisection and repair probe only diagnose plans that failed.
    ``on_full_failure`` is called once with a provisional diagnosis
    (``kind="pending"``, the full-plan attempts' pooled violations) when
    the full-plan solves failed, before the diagnosis starts. See the
    module docstring.

    Returns (cost, results, diagnosis, ik_mode, ik_hard_failed).
    """
    if not plan:
        return False, None, None, "soft", False
    try:
        n = len(next(iter(plan.values())))
    except (StopIteration, TypeError):
        return False, None, None, "soft", False
    if n < 2 or n > MAX_NUM_NODES:
        return False, None, None, "soft", False

    started = time.time()
    deadline = started + EVAL_BUDGET_S
    use_hard = terminal_goal is not None and terminal_goal.kso == "hard"
    err_h: Optional[str] = None

    if use_hard:
        cost_h, results_h, err_h = _solve_with_retries(plan, n, "hard")
        if cost_h is not False:
            return cost_h, results_h, None, "hard", False
        print("[ik_eval] hard-mode KSO failed; trying soft KSO on full plan")

    cost_s, results_s, err_s = _solve_with_retries(plan, n, "soft")
    if cost_s is not False:
        return cost_s, results_s, None, "soft", use_hard
    full = _pooled(list(_failed_attempts.get((n, "soft"), [])))
    error = err_s or err_h

    def with_full(diag: Dict[str, Any]) -> Dict[str, Any]:
        diag["solver_error"] = error if not diag.get("solver_error") else diag["solver_error"]
        if full:
            diag["kso_attempts"] = full["attempts"]
            diag["best_max_violation"] = full["best_max_violation"]
            diag["violation_details"] = full["details"]
        if use_hard:
            diag["ik_hard_failed"] = True
        return diag

    diag: Dict[str, Any]
    results_out: Optional[Dict[str, Any]] = None
    if error and error.startswith("ValueError: invalid contact schedule"):
        diag = {"kind": "schedule_error", "total_modes": n, "last_feasible_modes": None,
                "first_infeasible_modes": None}
    else:
        if on_full_failure is not None:
            try:
                on_full_failure(with_full({"kind": "pending", "total_modes": n, "last_feasible_modes": None,
                                           "first_infeasible_modes": None}))
            except Exception as e:  # noqa: BLE001 — the record is best effort
                print(f"[ik_eval] on_full_failure failed: {type(e).__name__}: {e}")
        if _has_kso_goal(terminal_goal) and not _past(deadline):
            # Without the goal: do the contacts work at all?
            cost_n, results_n, err_n = _solve_with_retries(plan, n, "none", GOAL_CHECK_ATTEMPTS)
            if cost_n is not False:
                diag = {"kind": "goal_missed", "total_modes": n, "last_feasible_modes": n,
                        "first_infeasible_modes": None,
                        "goal_residuals": _goal_residuals(results_n or {}, terminal_goal)}
                results_out = results_n
            else:
                error = error or err_n
                diag = _walk(plan, n, deadline)
        else:
            diag = _walk(plan, n, deadline)

    with_full(diag)
    if diag.get("kind") == "prefix":
        hi = diag["first_infeasible_modes"]
        prefix = _pooled(list(_failed_attempts.get((hi, "none"), [])))
        if prefix:
            diag["prefix_violation_details"] = prefix["details"]
            diag["prefix_tries"] = prefix["attempts"]
        describe = _transition_describer(plan)
        diag["break_event"] = describe(hi - 1) if hi else ""
        kind = "hard" if use_hard else "soft"

        def solve_once(edited: Dict[str, List[Tuple]]) -> bool:
            cost, _r, _e = _run_ik_once(edited, plan_checks.n_modes(edited), kind)
            return cost is not False

        diag["probe"] = repair_probe.probe(
            plan, hi, solve_once, max_probes=PROBES, n_steps=PROBE_STEPS,
            max_modes=MAX_NUM_NODES, deadline=deadline, describe=describe,
        )
        diag["probe"].pop("plan", None)
    diag["diagnosis_seconds"] = round(time.time() - started, 1)
    print(f"[ik_eval] infeasibility diagnosis: kind={diag.get('kind')}, "
          f"last_feasible={diag.get('last_feasible_modes')}, "
          f"first_infeasible={diag.get('first_infeasible_modes')}, total_modes={n}"
          + (", timed out" if diag.get("timed_out") else ""))
    return False, results_out, diag, "soft", use_hard


def _walk(plan: Dict[str, List[Tuple]], n: int, deadline: Optional[float] = None) -> Dict[str, Any]:
    print(f"[ik_eval] full-plan KSO failed; locating the break ({WALK_MODE} walk)")
    if WALK_MODE == "linear":
        return _linear_prefix(plan, n, PREFIX_ATTEMPTS, deadline)
    return _bisect_prefix(plan, n, PREFIX_ATTEMPTS, deadline)
