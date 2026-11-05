"""TO solve loop with a restoration budget (loco_manipulation/to_solve.py),
on a stub moto / ToTask (no faro)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import to_solve  # noqa: E402

R = SimpleNamespace(
    iter_result_success=0, iter_result_exceed_max_iter=1, iter_result_numerical_failure=2,
    iter_result_restoration_reached_max_iter=3, iter_result_restoration_failed=4,
    iter_result_infeasible_stationary=5,
)
MOTO = SimpleNamespace(sqp=R)
FATAL = (R.iter_result_numerical_failure, R.iter_result_restoration_reached_max_iter,
         R.iter_result_restoration_failed, R.iter_result_infeasible_stationary)


class Sqp:
    """Returns the scripted (result, inf_prim_res) per update call."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.settings = SimpleNamespace(ipm=SimpleNamespace(warm_start=False))

    def update(self, n, verbose=False):
        self.calls.append(n)
        result, prim = self.script.pop(0)
        return SimpleNamespace(result=result, inf_prim_res=prim, num_iter=n)


class Task:
    def __init__(self, presolve, coll, presolve_iters=50):
        self.sqp, self.sqp_coll = Sqp(presolve), Sqp(coll)
        self.moto_wrap = SimpleNamespace(
            build_cfg=SimpleNamespace(collision_update_batch_size=1, presolve_max_iters=presolve_iters,
                                      feasibility_tol=0.05),
            get_results=lambda sqp: {"robot_xs": {}, "obj_xs": {}, "x0_dict": {}, "u0_dict": {}},
            set_warm_start=lambda *a: None,
        )
        self.relinearised = 0

    def _sqp_update_failed(self, res):
        return res is None or res.result in FATAL

    def _has_nonfinite_results(self, results):
        return False

    def resolve_collisions(self, *a):
        self.relinearised += 1


def run(task, rescue=10, max_rescues=20, iters=10):
    return to_solve.solve(task, iters, rescue, max_rescues, moto=MOTO)


def test_converges_like_faro_without_restoration():
    t = Task([(R.iter_result_exceed_max_iter, 9.0)],
             [(R.iter_result_exceed_max_iter, 1.0), (R.iter_result_exceed_max_iter, 0.01)])
    info = run(t)
    assert (info.status, info.outer_iters, info.rescues, info.fail_reason) == (0, 2, 0, "")
    assert t.sqp.calls == [50] and t.sqp_coll.calls == [1, 1]


def test_restoration_without_budget_gets_a_rescue_update():
    t = Task([(R.iter_result_exceed_max_iter, 9.0)],
             [(R.iter_result_restoration_reached_max_iter, 3.1),   # faro would abort here
              (R.iter_result_exceed_max_iter, 2.0),                # the rescue update(10)
              (R.iter_result_success, 0.02)])
    info = run(t)
    assert info.status == 0 and info.rescues == 1
    assert t.sqp_coll.calls == [1, 10, 1]


def test_faro_behaviour_with_rescue_off_and_failure_reason():
    t = Task([(R.iter_result_exceed_max_iter, 9.0)], [(R.iter_result_restoration_reached_max_iter, 3.1)])
    info = run(t, rescue=0)
    assert info.status == 2 and info.fail_reason == "collision: restoration_reached_max_iter"
    assert t.sqp_coll.calls == [1]


def test_failed_rescue_and_rescue_cap_abort():
    t = Task([(R.iter_result_exceed_max_iter, 9.0)],
             [(R.iter_result_restoration_reached_max_iter, 3.1), (R.iter_result_restoration_failed, 3.1)])
    assert run(t).fail_reason == "collision: restoration_failed"
    t = Task([(R.iter_result_exceed_max_iter, 9.0)],
             [(R.iter_result_restoration_reached_max_iter, 3.1), (R.iter_result_exceed_max_iter, 2.0),
              (R.iter_result_restoration_reached_max_iter, 2.0)])
    info = run(t, max_rescues=1)
    assert info.status == 2 and info.rescues == 1


def test_presolve_failure_is_reported_and_rescued():
    t = Task([(R.iter_result_numerical_failure, 14.0)], [])
    assert run(t).fail_reason == "presolve: numerical_failure"
    t = Task([(R.iter_result_restoration_reached_max_iter, 14.0), (R.iter_result_exceed_max_iter, 8.0)],
             [(R.iter_result_success, 0.01)])
    info = run(t)
    assert info.status == 0 and info.rescues == 1 and t.sqp.calls == [50, 10]


def test_max_iterations_and_bad_budget():
    t = Task([], [(R.iter_result_exceed_max_iter, 1.0)] * 3, presolve_iters=0)
    info = run(t, iters=3)
    # Once before the loop and after every unconverged iteration, as faro.
    assert (info.status, info.outer_iters) == (1, 3) and t.relinearised == 4
    with pytest.raises(ValueError):
        run(t, iters=0)


def test_best_iterate_is_kept_when_the_solve_drifts():
    t = Task([(R.iter_result_exceed_max_iter, 9.0)],
             [(R.iter_result_exceed_max_iter, 5.0), (R.iter_result_exceed_max_iter, 2.5),
              (R.iter_result_exceed_max_iter, 7.0)])
    info = run(t, iters=3)
    assert info.status == 1 and (info.best_prim, info.best_iter) == (2.5, 2)
    assert info.best_results is not None and "robot_xs" in info.best_results
    t = Task([(R.iter_result_numerical_failure, 14.0)], [])
    assert run(t).best_results is None


def test_stall_stop_ends_a_solve_whose_residual_stops_improving():
    # Residual 5 -> 2.5 -> then flat: with stall_iters=2 the solve stops two
    # iterations after the last 10% improvement instead of using all 10.
    script = [(R.iter_result_exceed_max_iter, p) for p in (5.0, 2.5, 2.4, 2.35, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3)]
    t = Task([(R.iter_result_exceed_max_iter, 9.0)], list(script))
    info = to_solve.solve(t, 10, 10, 20, moto=MOTO, stall_iters=2)
    assert info.status == 1 and info.stalled and info.outer_iters == 4
    assert (info.best_prim, info.best_iter) == (2.35, 4)
    # Off by default: all 10 iterations run.
    t = Task([(R.iter_result_exceed_max_iter, 9.0)], list(script))
    info = to_solve.solve(t, 10, 10, 20, moto=MOTO)
    assert info.outer_iters == 10 and not info.stalled
    # A solve still improving is not stopped.
    t = Task([(R.iter_result_exceed_max_iter, 9.0)],
             [(R.iter_result_exceed_max_iter, p) for p in (5.0, 4.0, 3.0, 2.0, 1.0)] + [(R.iter_result_success, 0.01)])
    info = to_solve.solve(t, 10, 10, 20, moto=MOTO, stall_iters=2)
    assert info.status == 0 and not info.stalled and info.outer_iters == 6
    # A plateau within 3x the tolerance (0.05 here) keeps its budget and may converge.
    t = Task([(R.iter_result_exceed_max_iter, 9.0)],
             [(R.iter_result_exceed_max_iter, p) for p in (0.14, 0.14, 0.14, 0.14, 0.14)] + [(R.iter_result_success, 0.04)])
    info = to_solve.solve(t, 10, 10, 20, moto=MOTO, stall_iters=2)
    assert info.status == 0 and not info.stalled and info.outer_iters == 6
    t = Task([(R.iter_result_exceed_max_iter, 9.0)], [(R.iter_result_exceed_max_iter, 0.14)] * 6)
    assert to_solve.solve(t, 10, 10, 20, moto=MOTO, stall_iters=2, stall_near_tol=1.0).stalled


def test_reset_after_rescue_cold_starts_the_next_update():
    script = [(R.iter_result_restoration_reached_max_iter, 3.1), (R.iter_result_exceed_max_iter, 2.0),
              (R.iter_result_exceed_max_iter, 1.5), (R.iter_result_exceed_max_iter, 1.2)]
    seen = []

    class Watch(Sqp):
        def update(self, n, verbose=False):
            seen.append(self.settings.ipm.warm_start)
            return super().update(n, verbose)

    t = Task([(R.iter_result_exceed_max_iter, 9.0)], [])
    t.sqp_coll = Watch(script)
    to_solve.solve(t, 3, 10, 20, moto=MOTO, reset_after_rescue=True)
    # update(1) -> rescue update(10) -> cold update -> warm update
    assert seen == [False, False, False, True]
