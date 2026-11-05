"""Recompute TO cost for every saved contact plan under a results tree.

Walks ``<results_root>/<scene>/<run>/{gen_*,best}/results/extra.json``,
attempts a TO solve for every plan where ``goal_match=True`` and
``ik_feasible=True``, and aggregates the lowest TO cost per run into a
single CSV.

The JSONL cache only treats a row as final once it's either a structural
skip (empty plan / goal mismatch / IK infeasible — won't change on
retry) or a converged TO solve. Non-converged solves and solves that
raised an exception are retried on every invocation, since a script or
solver-config change may let them succeed on a later attempt; each
retry appends a new cache line and the latest one wins.

Requires the same environment as ``evaluate.py``::

    export ACADOS_SOURCE_DIR=/path/to/faro/extern/acados
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=submodules/faro/src:$PYTHONPATH
"""

import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import subprocess
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _run_common import plan_to_tuples, read_hydra_task, redirect_fds_to  # noqa: E402

# Deferred import: only the worker needs _task_loader (and through it,
# faro / acados / moto / pinocchio / casadi). The coordinator stays light so
# none of those C-extensions are touched in the dispatch process — that
# isolation is the whole point of the worker subprocess design.
TaskContext = Any  # type alias for annotations; real class loaded in worker

CACHE_FILENAME = ".to_recompute_cache.jsonl"
CSV_FILENAME = "to_costs.csv"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


_redirect_fds_to = redirect_fds_to
_plan_to_tuples = plan_to_tuples


def _plan_hash(plan: Dict[str, List[Any]]) -> str:
    # Keys of .to_recompute_cache.jsonl: deliberately NOT the evaluator's
    # canonical hash (changing it would invalidate existing caches).
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


_read_hydra_task = read_hydra_task


def _discover_runs(results_root: str) -> List[Tuple[str, str, str]]:
    """Return a list of ``(scene_dir_name, run_dir_name, run_dir_abs)``."""
    out: List[Tuple[str, str, str]] = []
    for scene_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(scene_dir):
            continue
        scene_name = os.path.basename(scene_dir)
        for run_dir in sorted(glob.glob(os.path.join(scene_dir, "*"))):
            if not os.path.isdir(run_dir):
                continue
            if not os.path.isdir(os.path.join(run_dir, ".hydra")):
                continue
            out.append((scene_name, os.path.basename(run_dir), run_dir))
    return out


def _discover_gens(run_dir: str) -> List[str]:
    """Sorted list of ``gen_N`` dir names that contain ``results/extra.json``.

    ``best/`` is intentionally excluded — it's always a byte-duplicate of
    some ``gen_N`` (the first to reach the original evolution's top
    ``combined_score``), so re-solving TO on it wastes a long acados
    solve and surfaces a misleading row in the per-run output.
    """
    entries: List[Tuple[int, str]] = []
    for entry in os.listdir(run_dir):
        if not entry.startswith("gen_"):
            continue
        path = os.path.join(run_dir, entry, "results", "extra.json")
        if not os.path.isfile(path):
            continue
        try:
            idx = int(entry.split("_", 1)[1])
        except ValueError:
            continue
        entries.append((idx, entry))
    entries.sort()
    return [name for _, name in entries]


# ---------------------------------------------------------------------------
# TO solve (copied from evaluate.py:494-533)
# ---------------------------------------------------------------------------


def _solve_to(
    contact_plan: Dict[str, List[Tuple]],
    task: TaskContext,
    log_path: str,
) -> Dict[str, Any]:
    """Run a TO solve on ``contact_plan`` for the given task. Returns a
    dict with ``to_cost`` / ``to_converged`` / ``error`` / ``num_modes``.
    All exceptions are caught and returned in ``error`` (matches the
    posture of ``evaluate.py:534``)."""
    result: Dict[str, Any] = {
        "to_cost": None,
        "to_converged": None,
        "to_status": None,
        "num_modes": None,
        "num_nodes": None,
        "to_mode": None,
        "error": None,
    }
    from to_eval import solve_to, to_mode

    result["to_mode"] = to_mode(task)
    try:
        with _redirect_fds_to(log_path):
            to_solve = solve_to(task, contact_plan)
            result["num_modes"] = to_solve.num_modes
            result["num_nodes"] = to_solve.num_nodes
            result["to_status"] = to_solve.status
            result["to_cost"] = to_solve.cost
            result["to_converged"] = to_solve.converged
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        # Stash the traceback into the log too, since the fd redirect is gone by here.
        try:
            with open(log_path, "a") as f:
                f.write("\n--- python traceback ---\n")
                f.write(traceback.format_exc())
        except OSError:
            pass
    return result


# ---------------------------------------------------------------------------
# Cache + aggregation
# ---------------------------------------------------------------------------


def _load_cache(cache_path: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not os.path.isfile(cache_path):
        return cache
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("scene"), row.get("run"), row.get("gen"))
            if None in key:
                continue
            cache[key] = row
    return cache


def _append_cache(cache_path: str, row: Dict[str, Any]) -> None:
    with open(cache_path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _is_final(row: Dict[str, Any]) -> bool:
    """Whether a cached row is authoritative and should NOT be retried.

    True for structural skips (empty plan / goal mismatch / IK
    infeasible — the outcome can't change without new IK/goal data) and
    for converged TO solves. False for a solve that ran but didn't
    converge and for a solve that raised an exception — both get
    re-attempted on the next invocation.
    """
    return bool(row.get("skipped")) or bool(row.get("to_converged"))


def _aggregate_csv(
    cache_rows: Iterable[Dict[str, Any]],
    csv_path: str,
    valid_keys: Optional[Set[Tuple[str, str, str]]] = None,
) -> int:
    """Write one row per ``(scene, run)`` with the lowest converged TO
    cost (fallback: lowest TO cost regardless of convergence). Returns
    the number of rows written.

    ``valid_keys`` (optional) filters cache rows to those whose
    ``(scene, run, gen)`` triple still exists on disk — drops orphans
    like the ``gen=best`` rows left over from earlier script versions.
    """
    by_run: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in cache_rows:
        key = (row.get("scene"), row.get("run"), row.get("gen"))
        if valid_keys is not None and key not in valid_keys:
            continue
        by_run.setdefault((row["scene"], row["run"]), []).append(row)

    out_rows: List[Dict[str, Any]] = []
    for (scene, run), rows in sorted(by_run.items()):
        eligible = [r for r in rows if not r.get("skipped")]
        success = [r for r in eligible if r.get("to_cost") is not None]
        converged = [r for r in success if r.get("to_converged")]
        pool = converged or success
        if pool:
            best = min(pool, key=lambda r: r["to_cost"])
            out_rows.append({
                "scene": scene,
                "run": run,
                "best_gen": best["gen"],
                "best_to_cost": best["to_cost"],
                "best_to_converged": best.get("to_converged"),
                "ik_cost_at_best": best.get("ik_cost"),
                "plan_hash": best.get("plan_hash"),
                "n_eligible_gens": len(eligible),
                "n_to_success": len(success),
                "n_to_fail": len(eligible) - len(success),
                "reason": "",
            })
        else:
            out_rows.append({
                "scene": scene,
                "run": run,
                "best_gen": "",
                "best_to_cost": "",
                "best_to_converged": "",
                "ik_cost_at_best": "",
                "plan_hash": "",
                "n_eligible_gens": len(eligible),
                "n_to_success": 0,
                "n_to_fail": len(eligible),
                "reason": (
                    "no_eligible_plans"
                    if not eligible
                    else "all_to_solves_failed"
                ),
            })

    fieldnames = [
        "scene", "run", "best_gen",
        "best_to_cost", "best_to_converged", "ik_cost_at_best",
        "plan_hash",
        "n_eligible_gens", "n_to_success", "n_to_fail", "reason",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    return len(out_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _process_gen(
    scene_dir_name: str,
    run_dir_name: str,
    run_dir_abs: str,
    gen_name: str,
    task: TaskContext,
    cache: Dict[Tuple[str, str, str], Dict[str, Any]],
    cache_path: str,
    to_log_path: str,
) -> Dict[str, Any]:
    key = (scene_dir_name, run_dir_name, gen_name)
    if key in cache and _is_final(cache[key]):
        return cache[key]

    extra_path = os.path.join(run_dir_abs, gen_name, "results", "extra.json")
    with open(extra_path) as f:
        extra = json.load(f)

    plan = extra.get("contact_mode_seq")
    goal_match = bool(extra.get("goal_match"))
    ik_feasible = bool(extra.get("ik_feasible"))
    ik_cost = extra.get("ik_cost")

    row: Dict[str, Any] = {
        "scene": scene_dir_name,
        "run": run_dir_name,
        "gen": gen_name,
        "plan_hash": _plan_hash(plan) if isinstance(plan, dict) else None,
        "goal_match": goal_match,
        "ik_feasible": ik_feasible,
        "ik_cost": ik_cost,
        "to_cost": None,
        "to_converged": None,
        "to_status": None,
        "num_modes": None,
        "num_nodes": None,
        "to_mode": None,
        "error": None,
        "skipped": None,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    if not isinstance(plan, dict) or not plan:
        row["skipped"] = "empty_plan"
    elif not goal_match:
        row["skipped"] = "goal_match_false"
    elif not ik_feasible:
        row["skipped"] = "ik_infeasible"
    else:
        tup_plan = _plan_to_tuples(plan)
        to_result = _solve_to(tup_plan, task, to_log_path)
        row.update({
            "to_cost": to_result["to_cost"],
            "to_converged": to_result["to_converged"],
            "to_status": to_result["to_status"],
            "num_modes": to_result["num_modes"],
            "num_nodes": to_result["num_nodes"],
            "to_mode": to_result["to_mode"],
            "error": to_result["error"],
        })

    cache[key] = row
    _append_cache(cache_path, row)
    return row


def _run_worker(run_dir_abs: str, cache_path: str) -> int:
    """Worker mode: process every gen in a single run, append to cache,
    print the per-run summary, and exit. Imports acados / pinocchio /
    casadi lazily (via ``_task_loader.load_task`` and ``_solve_to``)
    so the long-lived C-side state lives and dies inside this process
    only — the next run gets a fresh process via subprocess."""
    from _task_loader import load_task  # noqa: F401  (acados/pinocchio side effects)

    scene_dir_name = os.path.basename(os.path.dirname(run_dir_abs))
    run_dir_name = os.path.basename(run_dir_abs)

    hydra = _read_hydra_task(run_dir_abs)
    if hydra is None:
        print(f"[skip] {scene_dir_name}/{run_dir_name}: no .hydra/config.yaml")
        return 0
    scene, ic, desc = hydra
    try:
        task = load_task(scene=scene, initial_condition=ic, description=desc)
    except Exception as e:
        print(
            f"[skip] {scene_dir_name}/{run_dir_name}: "
            f"load_task({scene!r}, {ic!r}) failed: {type(e).__name__}: {e}"
        )
        return 0

    cache = _load_cache(cache_path)
    gens = _discover_gens(run_dir_abs)
    to_log_path = os.path.join(run_dir_abs, "_to_recompute.log")
    print(
        f"[run ] {scene_dir_name}/{run_dir_name}  scene={scene} ic={ic}  "
        f"({len(gens)} gens)"
    )

    for gen_name in gens:
        key = (scene_dir_name, run_dir_name, gen_name)
        if key in cache and _is_final(cache[key]):
            tag = cache[key].get("skipped") or "converged"
            print(f"       {gen_name:>8s}  cached ({tag})")
            continue
        retry = key in cache  # present but not final: to_failed / to_nonconverged
        row = _process_gen(
            scene_dir_name, run_dir_name, run_dir_abs,
            gen_name, task, cache, cache_path, to_log_path,
        )
        if row.get("skipped"):
            print(f"       {gen_name:>8s}  skipped: {row['skipped']}")
        elif row.get("error"):
            print(f"       {gen_name:>8s}  error: {row['error']}" +
                  (" (retry)" if retry else ""))
        else:
            print(
                f"       {gen_name:>8s}  to_cost={row['to_cost']:.4f}  "
                f"converged={row['to_converged']}  "
                f"modes={row['num_modes']}" +
                (" (retry)" if retry else "")
            )

    # Per-run summary line. Mirrors _aggregate_csv's selection rule.
    run_rows = [
        cache[(scene_dir_name, run_dir_name, g)]
        for g in gens
        if (scene_dir_name, run_dir_name, g) in cache
    ]
    success = [r for r in run_rows if r.get("to_cost") is not None]
    converged = [r for r in success if r.get("to_converged")]
    pool = converged or success
    if pool:
        b = min(pool, key=lambda r: r["to_cost"])
        print(
            f"       [best] gen={b['gen']}  to_cost={b['to_cost']:.4f}  "
            f"converged={b['to_converged']}  modes={b['num_modes']}"
        )
    else:
        eligible = [r for r in run_rows if not r.get("skipped")]
        reason = "all_to_solves_failed" if eligible else "no_eligible_plans"
        print(f"       [best] none ({reason})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        help="Directory whose children are <scene>/<run>/ (e.g. "
             "results/loco_abl3_ik_fb). Required in coordinator and "
             "worker modes.",
    )
    parser.add_argument(
        "--rebuild-csv-only", action="store_true",
        help="Skip TO solves; rebuild to_costs.csv from the existing "
             "JSONL cache.",
    )
    parser.add_argument(
        "--worker", action="store_true",
        help="Internal: worker subprocess mode. Process a single run "
             "given by --run-dir, then exit. The coordinator spawns "
             "one of these per run so acados/pinocchio C-side state "
             "from one scene cannot leak into the next.",
    )
    parser.add_argument(
        "--run-dir",
        help="Internal: absolute path to the run dir the worker should "
             "process (only with --worker).",
    )
    args = parser.parse_args()

    if not args.results_root:
        parser.error("--results-root is required")
    results_root = os.path.abspath(args.results_root)
    if not os.path.isdir(results_root):
        parser.error(f"results-root {results_root!r} is not a directory")
    cache_path = os.path.join(results_root, CACHE_FILENAME)
    csv_path = os.path.join(results_root, CSV_FILENAME)

    if args.worker:
        if not args.run_dir:
            parser.error("--worker requires --run-dir")
        sys.exit(_run_worker(os.path.abspath(args.run_dir), cache_path))

    if args.rebuild_csv_only:
        cache = _load_cache(cache_path)
        # Build the on-disk valid-keys set so orphan rows (e.g. legacy
        # gen=best entries from earlier script versions) drop out.
        valid_keys: Set[Tuple[str, str, str]] = set()
        for scene_dir_name, run_dir_name, run_dir_abs in _discover_runs(results_root):
            for g in _discover_gens(run_dir_abs):
                valid_keys.add((scene_dir_name, run_dir_name, g))
        n = _aggregate_csv(cache.values(), csv_path, valid_keys=valid_keys)
        print(f"[csv] wrote {n} rows to {csv_path}")
        return

    runs = _discover_runs(results_root)
    print(f"[scan] {len(runs)} runs found under {results_root}")
    if not runs:
        return

    # Spawn one worker subprocess per run, serially. A crash in one
    # worker (e.g. segfault deep in acados) is logged and skipped — the
    # rest of the tree continues. The cache file is the only shared
    # state; since workers run serially and append-only, no locking is
    # needed.
    valid_keys = set()
    for scene_dir_name, run_dir_name, run_dir_abs in runs:
        for g in _discover_gens(run_dir_abs):
            valid_keys.add((scene_dir_name, run_dir_name, g))

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker",
            "--results-root", results_root,
            "--run-dir", run_dir_abs,
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            sig = -proc.returncode if proc.returncode < 0 else None
            print(
                f"[warn] worker for {scene_dir_name}/{run_dir_name} "
                f"exited with code {proc.returncode}"
                + (f" (signal {sig})" if sig is not None else "")
                + " — continuing with next run"
            )

    cache = _load_cache(cache_path)
    n = _aggregate_csv(cache.values(), csv_path, valid_keys=valid_keys)
    print(f"[csv ] wrote {n} rows to {csv_path}")


if __name__ == "__main__":
    main()
