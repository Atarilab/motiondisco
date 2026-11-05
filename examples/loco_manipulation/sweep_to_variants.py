"""
Level-2 diversity: re-solve the successful plans of a finished search
under controlled variations to get many different trajectories.

The search (level 1) decides *which* contacts are made and broken; this
script varies *how* a fixed strategy is executed. After a run:

    python examples/loco_manipulation/sweep_to_variants.py \\
        --run-dir results/loco_abl3_ik_fb_diverse/corl_box_stacking/<run>

1. Collect plans whose TO ran with a usable cost (to_cost > 0) from
   ``<run>/gen_*/results/extra.json``; ``--converged-only`` keeps only
   converged solves.
2. Group them into strategy families (``plan_signature``) and keep the
   best ``--plans-per-family`` plans of the best ``--top-families``
   families (lowest to_cost, converged first).
3. For every kept plan, solve each variant once. Variants change one
   factor at a time (``--variants`` picks groups):

       start   robot start pose: base x / y offsets and yaw (Standing
               starts only; the task is unchanged)
       timing  time_per_mode (slower / faster execution)
       weights TO cost-weight scaling (state vs acceleration effort)
       goal    looser terminal-goal TO tolerances (scenes with to="hard")
       plan    plan edits: one more / one fewer single-support step
               pair in the longest walk, and the mirrored plan
               (left <-> right hands and soles)
       objects movable-object start offsets -- changes the TASK
               layout, so it is not in the default set

   Variants that change kinematics (start, plan, objects) are KSO-checked
   first and skipped when infeasible.

Outputs, under ``--out`` (default ``<run>/to_variants``)::

    <family>/<plan>/<variant>/to_run.pkl     same format as evaluate.py
                                             (replay_to_run.py works)
    <family>/<plan>/<variant>/variant.json   variant spec + metrics
    results.jsonl                            one row per solve (resume key)
    summary.csv                              the same rows as a table

Reruns skip variants already in results.jsonl (``--force`` re-solves).
Each kept plan runs in its own worker subprocess (acados / moto C-state
isolation, like run_to_all.py); ``--jobs`` runs several in parallel.
``--dry-run`` lists the plans and variants without solving.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from plan_signature import SOLE_CHANNELS, plan_signature  # noqa: E402
from run_to_all import (  # noqa: E402
    _discover_plan_dirs,
    _plan_to_tuples,
    _read_hydra_task,
    _redirect_fds_to,
)

RESULTS_JSONL = "results.jsonl"
SUMMARY_CSV = "summary.csv"
DEFAULT_GROUPS = ("start", "timing", "weights", "goal", "plan")
ALL_GROUPS = DEFAULT_GROUPS + ("objects",)
# Variants whose kinematics differ from the evaluated plan.
KSO_CHECK_GROUPS = {"start", "plan", "objects"}


# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------


def variant_catalogue(groups: List[str]) -> List[Dict[str, Any]]:
    """One-factor-at-a-time variants for the selected groups."""
    out: List[Dict[str, Any]] = [{"name": "reference", "group": "reference"}]
    if "start" in groups:
        for dx, dy, dyaw in [
            (-0.15, 0, 0), (0.15, 0, 0), (0, -0.15, 0), (0, 0.15, 0),
            (0, 0, -0.2), (0, 0, 0.2),
        ]:
            name = "start_" + "_".join(
                f"{k}{v:+.2f}" for k, v in (("x", dx), ("y", dy), ("yaw", dyaw)) if v
            )
            out.append({"name": name, "group": "start", "robot_xy_yaw": [dx, dy, dyaw]})
    if "timing" in groups:
        for tpm in (0.7, 1.4):
            out.append({"name": f"timing_tpm{tpm:g}", "group": "timing",
                        "time_per_mode": tpm})
    if "weights" in groups:
        for key, factor in (("acc_cost_W", 0.3), ("acc_cost_W", 3.0),
                            ("state_cost_W", 0.3), ("state_cost_W", 3.0)):
            out.append({"name": f"weights_{key}x{factor:g}", "group": "weights",
                        "weight_scale": {key: factor}})
    if "goal" in groups:
        out.append({"name": "goal_loose", "group": "goal",
                    "goal_tolerance": {"position": 0.1, "rotation": 0.2}})
    if "plan" in groups:
        out += [
            {"name": "plan_walk_plus1", "group": "plan", "plan_edit": "walk+1"},
            {"name": "plan_walk_minus1", "group": "plan", "plan_edit": "walk-1"},
            {"name": "plan_mirror", "group": "plan", "plan_edit": "mirror"},
        ]
    if "objects" in groups:
        for dx, dy in ((-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1)):
            out.append({"name": f"objects_x{dx:+.2f}_y{dy:+.2f}", "group": "objects",
                        "object_xy": [dx, dy]})
    return out


# ---------------------------------------------------------------------------
# Plan edits (pure, flat-channel plans)
# ---------------------------------------------------------------------------


def _is_step(plan: Dict[str, List[Tuple]], k: int, support: Dict[str, Any]) -> bool:
    changed = [ch for ch in plan if plan[ch][k] != plan[ch][k + 1]]
    return bool(changed) and all(
        ch in SOLE_CHANNELS
        and (plan[ch][k + 1] == () or plan[ch][k + 1] == support.get(ch))
        for ch in changed
    )


def _walk_segments(plan: Dict[str, List[Tuple]]) -> List[Tuple[int, int]]:
    """``(start, end)`` mode-index ranges of maximal walking runs: modes
    start..end where every transition inside is a locomotion step."""
    n = len(next(iter(plan.values())))
    support = {ch: plan[ch][0] for ch in SOLE_CHANNELS if ch in plan}
    segs, start = [], None
    for k in range(n - 1):
        step = _is_step(plan, k, support)
        for ch in SOLE_CHANNELS:
            if ch in plan and plan[ch][k + 1] != ():
                support[ch] = plan[ch][k + 1]
        if step and start is None:
            start = k
        if not step and start is not None:
            segs.append((start, k))
            start = None
    if start is not None:
        segs.append((start, n - 1))
    return segs


def edit_plan(plan: Dict[str, List[Tuple]], edit: str) -> Optional[Dict[str, List[Tuple]]]:
    """Return an edited copy of *plan*, or None if the edit does not apply.

    ``walk+1`` / ``walk-1`` add / remove one single-support step pair in
    the longest walk (two modes, so the walk still ends where it did);
    ``mirror`` swaps every left_/right_ channel pair.
    """
    plan = {ch: list(seq) for ch, seq in plan.items()}
    if edit == "mirror":
        out = dict(plan)
        for ch in plan:
            if ch.startswith("left_") and "right_" + ch[5:] in plan:
                other = "right_" + ch[5:]
                out[ch], out[other] = plan[other], plan[ch]
        return out
    segs = _walk_segments(plan)
    if not segs:
        return None
    s, e = max(segs, key=lambda se: se[1] - se[0])
    # Single-support modes inside the walk: s+1 .. e-1.
    inner = list(range(s + 1, e))
    if edit == "walk+1":
        if len(inner) < 2:
            return None
        a, b = inner[0], inner[1]
        return {ch: seq[: b + 1] + [seq[a], seq[b]] + seq[b + 1:] for ch, seq in plan.items()}
    if edit == "walk-1":
        if len(inner) < 3:
            return None
        a, b = inner[0], inner[1]
        return {ch: seq[:a] + seq[b + 1:] for ch, seq in plan.items()}
    raise ValueError(f"unknown plan edit {edit!r}")


# ---------------------------------------------------------------------------
# Task variants (faro StateSpec / TerminalGoal)
# ---------------------------------------------------------------------------


def task_variant(task: Any, variant: Dict[str, Any]) -> Any:
    """A TaskContext copy with the variant's start state / terminal goal.

    Returns None when the variant does not apply to this scene (e.g. a
    start-pose offset for a robot not initialized with ``Standing``).
    """
    from faro.configs.scene import OnFace, Standing
    from faro.search.goal import GoalTolerances

    from _task_loader import initial_spec

    out = copy.copy(task)
    if "robot_xy_yaw" in variant or "object_xy" in variant:
        spec = initial_spec(task.scene_name, task.ic_name, task.scene)
        robots, objects = dict(spec.robots), dict(spec.objects)
        if "robot_xy_yaw" in variant:
            start = robots.get(task.robot_id)
            if not isinstance(start, Standing):
                return None
            robots[task.robot_id] = replace(start, xy_yaw=tuple(
                a + b for a, b in zip(start.xy_yaw, variant["robot_xy_yaw"])
            ))
        if "object_xy" in variant:
            dx, dy = variant["object_xy"]
            moved = False
            for name in task.scene.env_objects:
                placement = objects.get(name)
                if isinstance(placement, OnFace):
                    x, y, yaw = placement.xy_yaw
                    objects[name] = replace(placement, xy_yaw=(x + dx, y + dy, yaw))
                    moved = True
            if not moved:
                return None
        out.resolved_state = task.scene.resolve(replace(spec, robots=robots, objects=objects))
    if "goal_tolerance" in variant:
        goal = task.terminal_goal
        # Tolerances only bound hard goals; soft goals use weights.
        if goal is None or goal.is_empty() or goal.to != "hard":
            return None
        tol = variant["goal_tolerance"]

        def loosen(target: Any) -> Any:
            return replace(target, to_tolerance=GoalTolerances(
                position=tol.get("position", 0.0), rotation=tol.get("rotation", 0.0),
            ))
        out.terminal_goal = replace(
            goal,
            robots={k: loosen(v) for k, v in goal.robots.items()},
            objects={k: loosen(v) for k, v in goal.objects.items()},
        )
    return out


# ---------------------------------------------------------------------------
# Plan collection (coordinator, no solver imports)
# ---------------------------------------------------------------------------


def collect_plans(run_dir: str, converged_only: bool) -> List[Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}
    for label, results_dir in _discover_plan_dirs(run_dir):
        if label == "best":
            continue  # byte-duplicate of some gen_N
        try:
            with open(os.path.join(results_dir, "extra.json")) as f:
                extra = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        plan, cost = extra.get("contact_mode_seq"), extra.get("to_cost")
        if not isinstance(plan, dict) or not cost or cost <= 0:
            continue
        converged = bool(extra.get("to_converged"))
        if converged_only and not converged:
            continue
        key = json.dumps(plan, sort_keys=True)
        cand = {"gen": label, "plan": plan, "to_cost": float(cost), "converged": converged,
                "signature": plan_signature(_plan_to_tuples(plan))}
        best = plans.get(key)
        if best is None or (not best["converged"], best["to_cost"]) > (not converged, cost):
            plans[key] = cand
    return list(plans.values())


def select_plans(plans: List[Dict[str, Any]], top_families: int,
                 per_family: int) -> List[Dict[str, Any]]:
    families: Dict[str, List[Dict[str, Any]]] = {}
    for p in plans:
        families.setdefault(p["signature"], []).append(p)
    ranked = sorted(
        (sorted(ps, key=lambda p: (not p["converged"], p["to_cost"])) for ps in families.values()),
        key=lambda ps: (not ps[0]["converged"], ps[0]["to_cost"]),
    )
    out = []
    for fam_idx, members in enumerate(ranked[:top_families]):
        for p in members[:per_family]:
            out.append({**p, "family": f"fam{fam_idx:02d}", "plan_id": p["gen"]})
    return out


# ---------------------------------------------------------------------------
# Worker: all variants of one plan
# ---------------------------------------------------------------------------


def _save_variant(vdir: str, plan: Dict[str, List[Tuple]], solve: Any,
                  row: Dict[str, Any], variant: Dict[str, Any]) -> None:
    os.makedirs(vdir, exist_ok=True)
    if solve is not None:
        with open(os.path.join(vdir, "to_run.pkl"), "wb") as f:
            pickle.dump({
                "contact_mode_seq": plan,
                "to_cost": solve.cost,
                "to_converged": solve.converged,
                "results": solve.results,
                "variant": variant,
            }, f)
    with open(os.path.join(vdir, "variant.json"), "w") as f:
        json.dump({"variant": variant, **row}, f, indent=2)


def run_worker(job_path: str) -> int:
    with open(job_path) as f:
        job = json.load(f)
    from _task_loader import load_task
    from ik_eval import check_plan
    from scoring import task_quality, to_quality
    from to_eval import TIME_PER_MODE, solve_to

    task = load_task(scene=job["scene"], initial_condition=job["ic"],
                     description=job["description"])
    base_plan = _plan_to_tuples(job["plan"])
    plan_dir = os.path.join(job["out"], job["family"], job["plan_id"])
    log_path = os.path.join(plan_dir, "solver.log")
    os.makedirs(plan_dir, exist_ok=True)

    for variant in job["variants"]:
        row: Dict[str, Any] = {
            "family": job["family"], "plan_id": job["plan_id"], "variant": variant["name"],
            "group": variant["group"], "signature": job["signature"], "status": None,
            "converged": None, "to_cost": None, "task_quality": None, "to_quality": None,
            "kso_feasible": None, "skipped": None, "error": None, "seconds": None,
        }
        t0 = time.time()
        solve = None
        plan = base_plan
        try:
            vtask = task_variant(task, variant)
            if vtask is None:
                row["skipped"] = "not applicable to this scene"
            elif "plan_edit" in variant and (
                plan := edit_plan(base_plan, variant["plan_edit"])
            ) is None:
                row["skipped"] = "plan edit not applicable"
            else:
                with _redirect_fds_to(log_path):
                    if variant["group"] in KSO_CHECK_GROUPS:
                        cost, _res, err = check_plan(vtask, plan)
                        row["kso_feasible"] = cost is not False
                        if cost is False:
                            row["skipped"] = "KSO infeasible" + (f" ({err})" if err else "")
                    if not row["skipped"]:
                        solve = solve_to(
                            vtask, plan,
                            time_per_mode=variant.get("time_per_mode", TIME_PER_MODE),
                            weight_scale=variant.get("weight_scale"),
                        )
                if solve is not None:
                    row.update(status=solve.status, converged=solve.converged,
                               to_cost=solve.cost,
                               task_quality=task_quality(vtask, solve.results),
                               to_quality=to_quality(solve.max_constraint_violation,
                                                     solve.jerk_cost, solve.num_modes),
                               max_violation=solve.max_constraint_violation,
                               jerk_cost=solve.jerk_cost)
        except Exception as e:  # noqa: BLE001 — one bad variant must not stop the sweep
            row["error"] = f"{type(e).__name__}: {e}"
            with open(log_path, "a") as f:
                f.write(f"\n--- {variant['name']} traceback ---\n{traceback.format_exc()}")
        row["seconds"] = round(time.time() - t0, 1)
        _save_variant(os.path.join(plan_dir, variant["name"]), plan, solve, row, variant)
        with open(os.path.join(job["out"], RESULTS_JSONL), "a") as f:
            f.write(json.dumps(row) + "\n")
        state = row["skipped"] or row["error"] or (
            f"to_cost={row['to_cost']:.4f} converged={row['converged']}")
        print(f"[{job['family']}/{job['plan_id']}] {variant['name']}: {state}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def _done_keys(out: str) -> set:
    path = os.path.join(out, RESULTS_JSONL)
    done = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["family"], r["plan_id"], r["variant"]))
    return done


def _write_summary(out: str) -> int:
    path = os.path.join(out, RESULTS_JSONL)
    if not os.path.isfile(path):
        return 0
    rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["family"], r["plan_id"], r["variant"])] = r  # last write wins
    fields = ["family", "plan_id", "variant", "group", "converged", "to_cost",
              "task_quality", "to_quality", "max_violation", "jerk_cost", "kso_feasible", "skipped", "error",
              "seconds", "signature"]
    with open(os.path.join(out, SUMMARY_CSV), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    return sum(1 for r in rows.values() if r.get("converged"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Run dir (contains .hydra/ and gen_*/)")
    p.add_argument("--out", help="Output dir (default: <run-dir>/to_variants)")
    p.add_argument("--top-families", type=int, default=5)
    p.add_argument("--plans-per-family", type=int, default=1)
    p.add_argument("--variants", default=",".join(DEFAULT_GROUPS),
                   help=f"comma-separated groups from {ALL_GROUPS}")
    p.add_argument("--converged-only", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="parallel plan workers")
    p.add_argument("--force", action="store_true", help="re-solve recorded variants")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        sys.exit(run_worker(args.worker))
    if not args.run_dir:
        p.error("--run-dir is required")

    run_dir = os.path.abspath(args.run_dir)
    task_id = _read_hydra_task(run_dir)
    if task_id is None:
        sys.exit(f"{run_dir}: no .hydra/config.yaml with task.scene")
    scene, ic, desc = task_id
    groups = [g.strip() for g in args.variants.split(",") if g.strip()]
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        p.error(f"unknown variant groups {sorted(unknown)}; choose from {ALL_GROUPS}")
    out = os.path.abspath(args.out or os.path.join(run_dir, "to_variants"))
    variants = variant_catalogue(groups)

    plans = collect_plans(run_dir, args.converged_only)
    chosen = select_plans(plans, args.top_families, args.plans_per_family)
    n_fam = len({p["signature"] for p in plans})
    print(f"[scan] {scene}/{ic}: {len(plans)} successful plan(s) in {n_fam} "
          f"famil{'y' if n_fam == 1 else 'ies'}; sweeping {len(chosen)} plan(s) x "
          f"{len(variants)} variant(s) -> {out}")
    done = set() if args.force else _done_keys(out)
    jobs = []
    for c in chosen:
        todo = [v for v in variants if (c["family"], c["plan_id"], v["name"]) not in done]
        print(f"  {c['family']} {c['plan_id']} to_cost={c['to_cost']:.4f} "
              f"converged={c['converged']} todo={len(todo)}  {c['signature'][:90]}")
        if todo:
            jobs.append({**c, "variants": todo, "scene": scene, "ic": ic,
                         "description": desc, "out": out})
    if args.dry_run or not jobs:
        print("[dry-run] nothing solved." if args.dry_run else "[done] nothing to do.")
        return

    os.makedirs(out, exist_ok=True)

    def launch(job: Dict[str, Any]) -> int:
        job_path = os.path.join(out, job["family"], job["plan_id"], "job.json")
        os.makedirs(os.path.dirname(job_path), exist_ok=True)
        with open(job_path, "w") as f:
            json.dump(job, f)
        return subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", job_path],
            check=False,
        ).returncode

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        codes = list(pool.map(launch, jobs))
    n_conv = _write_summary(out)
    print(f"[done] {n_conv} converged variant trajectories; summary: "
          f"{os.path.join(out, SUMMARY_CSV)}"
          + (f"; {sum(c != 0 for c in codes)} worker(s) crashed" if any(codes) else ""))


if __name__ == "__main__":
    main()
