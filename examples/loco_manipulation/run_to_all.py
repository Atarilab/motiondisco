"""Run TO unconditionally on every saved contact plan under a results dir.

Unlike ``evaluate.py`` (which only solves TO when ``goal_match`` *and*
``ik_feasible`` both hold), this script solves TO for *every* plan it finds
and writes a ``to_run.pkl`` next to each plan's ``extra.json`` — including
plans that failed the discrete-goal or IK gate during evolution and so never
got a TO trajectory.

The solve mirrors ``evaluate.py`` exactly (``nodes_per_mode=20``,
``time_per_mode=1.0``, ``step_ts`` recorded), so the resulting pickles are
replayable with ``replay_to_run.py``.

Usage::

    python examples/loco_manipulation/run_to_all.py \\
        results/loco_abl3_ik_fb/corl_climb_table_up_down_with_box

The path may be a single run dir (one with a ``.hydra/`` subdir), a scene dir
whose children are run dirs, or any ancestor — every run dir found at or below
it is processed. Existing ``to_run.pkl`` files are skipped unless ``--force``.

Requires the same environment as ``evaluate.py``::

    export ACADOS_SOURCE_DIR=/path/to/faro/extern/acados
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=submodules/faro/src:$PYTHONPATH
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Tuple


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _run_common import plan_to_tuples, read_hydra_task, redirect_fds_to  # noqa: E402

# Deferred import: only the worker needs _task_loader (and through it
# faro / acados / moto / pinocchio / casadi). The coordinator stays light so
# none of those C-extensions are touched in the dispatch process.
TaskContext = Any  # type alias; real class loaded in the worker


# ---------------------------------------------------------------------------
# I/O helpers (mirrors recompute_to_costs.py / evaluate.py)
# ---------------------------------------------------------------------------


_redirect_fds_to = redirect_fds_to
_plan_to_tuples = plan_to_tuples
_read_hydra_task = read_hydra_task


def _discover_runs(root: str) -> List[str]:
    """Return absolute paths of every run dir (one containing ``.hydra/``)
    at ``root`` or anywhere below it."""
    runs: List[str] = []
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, ".hydra")):
        return [root]
    for dirpath, dirnames, _ in os.walk(root):
        if ".hydra" in dirnames:
            runs.append(dirpath)
            # Don't descend into a run dir's own subtree.
            dirnames[:] = [d for d in dirnames if d != ".hydra"]
    return sorted(runs)


def _discover_plan_dirs(run_dir: str) -> List[Tuple[str, str]]:
    """Sorted ``(label, results_dir)`` for every subdir of ``run_dir`` that
    holds a ``results/extra.json`` (``gen_N`` and ``best``)."""
    out: List[Tuple[int, str, str]] = []
    for entry in os.listdir(run_dir):
        results_dir = os.path.join(run_dir, entry, "results")
        if not os.path.isfile(os.path.join(results_dir, "extra.json")):
            continue
        # gen_N sorts numerically; "best" and anything else sort last.
        if entry.startswith("gen_"):
            try:
                idx = int(entry.split("_", 1)[1])
            except ValueError:
                idx = 1 << 30
        else:
            idx = 1 << 30
        out.append((idx, entry, results_dir))
    out.sort(key=lambda t: (t[0], t[1]))
    return [(label, rd) for _, label, rd in out]


# ---------------------------------------------------------------------------
# TO solve + pickle (mirrors evaluate.py:480-599)
# ---------------------------------------------------------------------------


def _solve_to_and_save(
    contact_plan: Dict[str, List[Tuple]],
    task: TaskContext,
    results_dir: str,
    log_path: str,
) -> Dict[str, Any]:
    """Solve TO for ``contact_plan`` and write ``to_run.pkl`` into
    ``results_dir``. Returns a small status dict. All exceptions are caught
    and reported in ``error`` (matches evaluate.py's posture: a bad solve
    must never abort the sweep)."""
    result: Dict[str, Any] = {
        "to_cost": None,
        "to_converged": None,
        "to_status": None,
        "num_modes": None,
        "to_mode": None,
        "saved": False,
        "error": None,
    }
    from to_eval import solve_to, to_mode

    result["to_mode"] = to_mode(task)
    try:
        with _redirect_fds_to(log_path):
            to_solve = solve_to(task, contact_plan)
            result["num_modes"] = to_solve.num_modes
            result["to_status"] = to_solve.status
            result["to_cost"] = to_solve.cost
            result["to_converged"] = to_solve.converged

            to_run_path = os.path.join(results_dir, "to_run.pkl")
            with open(to_run_path, "wb") as f:
                pickle.dump(
                    {
                        "contact_mode_seq": contact_plan,
                        "to_cost": to_solve.cost,
                        "to_converged": result["to_converged"],
                        "results": to_solve.results,
                    },
                    f,
                )
            result["saved"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        try:
            with open(log_path, "a") as f:
                f.write("\n--- python traceback ---\n")
                f.write(traceback.format_exc())
        except OSError:
            pass
    return result


# ---------------------------------------------------------------------------
# Worker (one subprocess per run for acados/pinocchio C-state isolation)
# ---------------------------------------------------------------------------


def _run_worker(run_dir: str, force: bool) -> int:
    from _task_loader import load_task  # noqa: F401  (acados/pinocchio side effects)

    run_name = os.path.basename(run_dir)
    hydra = _read_hydra_task(run_dir)
    if hydra is None:
        print(f"[skip] {run_name}: no .hydra/config.yaml")
        return 0
    scene, ic, desc = hydra
    try:
        task = load_task(scene=scene, initial_condition=ic, description=desc)
    except Exception as e:
        print(f"[skip] {run_name}: load_task({scene!r}, {ic!r}) failed: "
              f"{type(e).__name__}: {e}")
        return 0

    plan_dirs = _discover_plan_dirs(run_dir)
    print(f"[run ] {run_name}  scene={scene} ic={ic}  ({len(plan_dirs)} plans)")

    for label, results_dir in plan_dirs:
        to_run_path = os.path.join(results_dir, "to_run.pkl")
        if os.path.isfile(to_run_path) and not force:
            print(f"       {label:>8s}  skip (to_run.pkl exists; --force to redo)")
            continue
        with open(os.path.join(results_dir, "extra.json")) as f:
            extra = json.load(f)
        plan = extra.get("contact_mode_seq")
        if not isinstance(plan, dict) or not plan:
            print(f"       {label:>8s}  skip (empty plan)")
            continue

        log_path = os.path.join(results_dir, "to_run_all.log")
        res = _solve_to_and_save(_plan_to_tuples(plan), task, results_dir, log_path)
        if res["error"]:
            print(f"       {label:>8s}  error: {res['error']}")
        else:
            print(f"       {label:>8s}  to_cost={res['to_cost']:.4f}  "
                  f"converged={res['to_converged']}  modes={res['num_modes']}  "
                  f"-> to_run.pkl")
    return 0


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Run dir, scene dir, or any ancestor. Every run dir at or below "
             "it is processed.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-solve and overwrite even where a to_run.pkl already exists.",
    )
    parser.add_argument(
        "--worker", action="store_true",
        help="Internal: process the single run given by --run-dir, then exit.",
    )
    parser.add_argument("--run-dir", help="Internal: run dir for --worker.")
    args = parser.parse_args()

    if args.worker:
        if not args.run_dir:
            parser.error("--worker requires --run-dir")
        sys.exit(_run_worker(os.path.abspath(args.run_dir), args.force))

    if not args.path:
        parser.error("path is required")
    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        parser.error(f"path {root!r} is not a directory")

    runs = _discover_runs(root)
    print(f"[scan] {len(runs)} run(s) found under {root}")
    if not runs:
        return

    # One worker subprocess per run, serially. A crash deep in acados is
    # logged and skipped; the rest of the tree continues. No shared mutable
    # state, so no locking needed.
    for run_dir in runs:
        cmd = [sys.executable, os.path.abspath(__file__),
               "--worker", "--run-dir", run_dir]
        if args.force:
            cmd.append("--force")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            sig = -proc.returncode if proc.returncode < 0 else None
            print(f"[warn] worker for {os.path.basename(run_dir)} exited with "
                  f"code {proc.returncode}"
                  + (f" (signal {sig})" if sig is not None else "")
                  + " — continuing")


if __name__ == "__main__":
    main()
