"""TO solve loop that leaves moto's restoration phase an iteration budget.

faro's ``ToTask.solve`` re-linearises the collisions after every single
SQP iteration (``sqp_coll.update(1)``). moto gives a restoration phase only
the iterations left in the current ``update`` call
(``min(restoration.max_iter, budget - used)``, moto
``src/solver/sqp_impl/restoration.cpp``), which inside ``update(1)`` is
zero: the first line-search failure skips restoration, returns
``restoration_reached_max_iter`` and faro aborts with status 2. That was
every status-2 TO on banana_real (24 of 25) and truck_unload_stack
(12 of 29); raising ``restoration.max_iter`` changes nothing.

``solve`` mirrors faro's loop (faro is not modified) and, when an
update ends in ``restoration_reached_max_iter``, calls it once more with
``rescue_iters`` iterations so restoration can actually run, then goes back
to re-linearising after every iteration. ``rescue_iters=0`` is faro's
behaviour.

    solve(to_task, max_nlp_iters, rescue_iters, max_rescues) -> SolveInfo

It also keeps the collision-phase iterate with the lowest primal residual
(``SolveInfo.best_*``); to_eval stores it beside the final one.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Callable, Dict, Optional

CONVERGED, MAX_ITERS, FAILED = 0, 1, 2


@dataclass
class SolveInfo:
    status: int
    # "<phase>: <moto result>" for a failed solve ('presolve' or
    # 'collision'), "" otherwise.
    fail_reason: str = ""
    outer_iters: int = 0
    rescues: int = 0
    # Lowest primal residual reached in the collision phase, its outer
    # iteration and a copy of that iterate's results (None if the phase
    # never ran). A non-converged solve often passes a better point than
    # the one it stops at.
    best_prim: float = float("inf")
    best_iter: int = 0
    best_results: Optional[Dict[str, Any]] = None
    # True when the solve stopped because the best residual stopped
    # improving (``stall_iters``); status is MAX_ITERS then.
    stalled: bool = False


def result_name(moto: Any, result: Any) -> str:
    for name in dir(moto.sqp):
        if name.startswith("iter_result_") and getattr(moto.sqp, name) == result:
            return name[len("iter_result_"):]
    return str(result)


def solve(to_task: Any, max_nlp_iters: int, rescue_iters: int = 10, max_rescues: int = 20,
          log: Optional[Callable[[str], None]] = None, moto: Any = None,
          reset_after_rescue: bool = False, stall_iters: int = 0, stall_ratio: float = 0.1,
          stall_near_tol: float = 3.0) -> SolveInfo:
    """Run ``to_task``'s presolve and collision SQP like ``ToTask.solve``,
    giving restoration ``rescue_iters`` iterations when an update ran out
    of budget for it (at most ``max_rescues`` times per solve).
    ``reset_after_rescue`` cold-starts the interior point (barrier back to
    mu0, slacks and multipliers re-initialised) on the update after a
    rescue, instead of warm-starting from restoration's exit state.
    ``stall_iters`` > 0 stops a collision-phase solve whose best primal
    residual improved by less than ``stall_ratio`` over the last
    ``stall_iters`` outer iterations (status MAX_ITERS, ``stalled`` set):
    a solve that has stopped making progress otherwise burns its whole
    iteration budget. A solve whose best residual is already below
    ``stall_near_tol`` times the feasibility tolerance is never stalled."""
    if moto is None:
        import moto
    max_nlp_iters = int(max_nlp_iters)
    if max_nlp_iters < 1:
        raise ValueError("max_nlp_iters must be at least 1")
    cfg = to_task.moto_wrap.build_cfg
    batch = max(1, int(cfg.collision_update_batch_size))
    info = SolveInfo(status=MAX_ITERS)
    rescued = [False]
    say = log or (lambda _msg: None)

    def keep_best(outer: int, res: Any, results: Dict[str, Any]) -> None:
        prim = float(getattr(res, "inf_prim_res", float("inf")))
        if prim < info.best_prim:
            info.best_prim, info.best_iter = prim, outer
            info.best_results = copy.deepcopy(results)

    def update(sqp: Any, n: int) -> Any:
        res = sqp.update(n, verbose=False)
        if (rescue_iters > 0 and info.rescues < max_rescues and res is not None
                and getattr(res, "result", None) == moto.sqp.iter_result_restoration_reached_max_iter):
            info.rescues += 1
            say(f"[to_solve] restoration had no budget; rescue {info.rescues} with {rescue_iters} iterations")
            res = sqp.update(int(rescue_iters), verbose=False)
            rescued[0] = True
        return res

    def failed(phase: str, res: Any) -> SolveInfo:
        info.status = FAILED
        info.fail_reason = f"{phase}: " + ("non-finite results" if res is None
                                          else result_name(moto, getattr(res, "result", None)))
        say(f"[to_solve] failed in {info.fail_reason}")
        return info

    presolve = int(cfg.presolve_max_iters)
    if presolve > 0:
        res = update(to_task.sqp, presolve)
        if to_task._sqp_update_failed(res):
            return failed("presolve", res)
        results = to_task.moto_wrap.get_results(to_task.sqp)
        if to_task._has_nonfinite_results(results):
            return failed("presolve", None)
        to_task.moto_wrap.set_warm_start(to_task.sqp_coll, results["x0_dict"], results["u0_dict"])
    else:
        results = to_task.moto_wrap.get_results(to_task.sqp_coll)
        if to_task._has_nonfinite_results(results):
            return failed("presolve", None)

    to_task.resolve_collisions(results["robot_xs"], results["obj_xs"])
    best_hist: list = []   # best residual after each outer iteration
    for outer in range(1, max_nlp_iters + 1):
        info.outer_iters = outer
        res = update(to_task.sqp_coll, batch)
        if to_task._sqp_update_failed(res):
            return failed("collision", res)
        results = to_task.moto_wrap.get_results(to_task.sqp_coll)
        if to_task._has_nonfinite_results(results):
            return failed("collision", None)
        keep_best(outer, res, results)
        if (res.result in (moto.sqp.iter_result_success, moto.sqp.iter_result_exceed_max_iter)
                and res.inf_prim_res < cfg.feasibility_tol):
            info.status = CONVERGED
            say(f"[to_solve] feasibility reached after {outer} outer iterations ({info.rescues} rescues)")
            return info
        best_hist.append(info.best_prim)
        # A solve that has stopped improving is given up, unless its best
        # residual is already within reach of the tolerance (layered_pallet
        # windows were cut at 0.147 against a 0.1 tolerance): those keep their
        # full budget, a plateau there often precedes convergence.
        if (stall_iters > 0 and outer > stall_iters
                and info.best_prim > stall_near_tol * cfg.feasibility_tol
                and info.best_prim > (1.0 - stall_ratio) * best_hist[-1 - stall_iters]):
            info.stalled = True
            say(f"[to_solve] stalled: best residual {info.best_prim:.3g} improved less than "
                f"{100 * stall_ratio:.0f}% over the last {stall_iters} outer iterations; stopping at {outer}")
            return info
        to_task.resolve_collisions(results["robot_xs"], results["obj_xs"])
        to_task.sqp_coll.settings.ipm.warm_start = not (reset_after_rescue and rescued[0])
        rescued[0] = False
    return info
