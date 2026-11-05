"""
Re-solve the contact plans of finished searches under randomly varied
physics and start states, and export every usable trajectory with the
scene and parameters that produced it (RL-tracker dataset).

    python examples/loco_manipulation/generate_dataset.py \\
        --run-dir results/<exp>/<scene>/<run> [--run-dir ...] \\
        --out datasets/loco --samples 8

(or ``--runs-root results/data_<campaign>`` to take every run below it).

1. Plans: every distinct plan of each run whose TO trajectory is valid
   (converged, or physically valid per trajectory_check; runs from before
   the check use ``--near-feasible`` on the solver residual instead),
   at most ``--max-per-family`` per strategy family (``plan_signature``).
2. Samples: per plan a ``reference`` sample (the search's own physics)
   plus ``--samples`` random combined variants, each with a seed derived
   from (``--seed``, run, plan, index) so reruns reproduce them. Factors
   (``--factors``; ranges via ``--range name=lo,hi``):

       mass     movable-object mass scale            U[0.5, 2.0]
       mass_kg  movable-object mass in kg, absolute  U[0.5, 10.0]
                (replaces the object's own mass; wins over ``mass``)
       size     movable-object half-extent scale,
                per axis                              U[0.85, 1.15]
       robot    robot start offset: xy U[-0.15, 0.15] m, yaw U[-0.2, 0.2]
       objects  start offset of each movable object resting on a fixed
                surface (stacked objects ride along): xy U[-0.08, 0.08] m,
                yaw U[-0.15, 0.15]
       timing   time_per_mode                         U[0.8, 1.3] s
       weights  TO cost-weight scales acc_cost_W / state_cost_W,
                log-uniform [0.5, 2]

   Friction is not varied (a constant, mu = 0.5, inside faro's TO).
   Mass and size are runtime TO parameters, so no recompilation; goals
   that are world positions get their height corrected for size changes.
3. Solve: samples that change geometry or start poses are KSO-checked
   first (skipped when infeasible); then TO (``to_eval.solve_to``).
4. Export: converged solves, and non-converged ones whose trajectory
   passes trajectory_check (flagged ``physical``; the final or the best
   iterate), go to
   ``<out>/<scene>/<ic>/<run>/<plan12>/<sample>/`` (traj.npz, scene.xml,
   params.json; see dataset_export.py). Every attempt gets a row in
   ``<out>/manifest.jsonl``; ``summary.csv`` counts them.

One worker subprocess per plan (its samples run serially); ``--jobs``
plans in parallel. Keep ``SHINKA_TO_N_JOB=1`` (multi-threaded moto
segfaulted on large problems) and run under ``nice``. Reruns skip
samples already in the manifest (``--force`` redoes them);
``--dry-run`` prints the plan x sample matrix.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import dataset_export as dx  # noqa: E402
from _run_common import canonical_plan_hash, plan_to_tuples, read_hydra_task, redirect_fds_to  # noqa: E402
from plan_signature import SOLE_CHANNELS, plan_patches, plan_signature  # noqa: E402
from _task_loader import canonical_scene_name  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
EE_CHANNELS = ("left_ee", "right_ee")
ALL_FACTORS = ("mass", "mass_kg", "size", "robot", "objects", "timing", "weights")


@dataclass
class Ranges:
    mass: Tuple[float, float] = (0.5, 2.0)
    mass_kg: Tuple[float, float] = (0.5, 10.0)
    size: Tuple[float, float] = (0.85, 1.15)
    robot_xy: Tuple[float, float] = (-0.15, 0.15)
    robot_yaw: Tuple[float, float] = (-0.2, 0.2)
    object_xy: Tuple[float, float] = (-0.08, 0.08)
    object_yaw: Tuple[float, float] = (-0.15, 0.15)
    time_per_mode: Tuple[float, float] = (0.8, 1.3)
    weights: Tuple[float, float] = (0.5, 2.0)


# ---------------------------------------------------------------------------
# Sampling (coordinator; pure)
# ---------------------------------------------------------------------------


def sample_seed(seed: int, run_id: str, plan_hash: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}|{run_id}|{plan_hash}|{index}".encode()).hexdigest()
    return int(digest[:15], 16)


def sample_factors(rng: np.random.Generator, objects: Sequence[str],
                   factors: Sequence[str], r: Ranges) -> Dict[str, Any]:
    """One random combined variant (JSON-serializable)."""
    out: Dict[str, Any] = {}
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
    if "mass" in factors and objects:
        out["mass_scale"] = {o: u(r.mass) for o in objects}
    if "mass_kg" in factors and objects:
        out["mass_kg"] = {o: round(u(r.mass_kg), 3) for o in objects}
    if "size" in factors and objects:
        out["size_scale"] = {o: [u(r.size) for _ in range(3)] for o in objects}
    if "robot" in factors:
        out["robot_offset"] = [u(r.robot_xy), u(r.robot_xy), u(r.robot_yaw)]
    if "objects" in factors and objects:
        out["object_offsets"] = {o: [u(r.object_xy), u(r.object_xy), u(r.object_yaw)]
                                 for o in objects}
    if "timing" in factors:
        out["time_per_mode"] = u(r.time_per_mode)
    if "weights" in factors:
        lo, hi = np.log(r.weights[0]), np.log(r.weights[1])
        out["weight_scale"] = {k: float(np.exp(rng.uniform(lo, hi)))
                               for k in ("acc_cost_W", "state_cost_W")}
    return out


def plan_samples(run_id: str, plan_hash: str, objects: Sequence[str], n: int, seed: int,
                 factors: Sequence[str], r: Ranges) -> List[Dict[str, Any]]:
    samples = [{"sample_id": "s000_reference", "seed": None, "factors": {}}]
    for i in range(1, n + 1):
        s = sample_seed(seed, run_id, plan_hash, i)
        samples.append({"sample_id": f"s{i:03d}", "seed": s,
                        "factors": sample_factors(np.random.default_rng(s), objects, factors, r)})
    return samples


def plan_objects(plan: Dict[str, List[Any]]) -> List[str]:
    return sorted(ch for ch in plan if ch not in SOLE_CHANNELS and ch not in EE_CHANNELS)


# ---------------------------------------------------------------------------
# Plan collection (coordinator; no solver imports)
# ---------------------------------------------------------------------------


def collect_plans(run_dir: str, near_feasible: float, max_per_family: int) -> List[Dict[str, Any]]:
    from run_to_all import _discover_plan_dirs

    by_hash: Dict[str, Dict[str, Any]] = {}
    for label, results_dir in _discover_plan_dirs(run_dir):
        if label == "best":
            continue
        try:
            with open(os.path.join(results_dir, "extra.json")) as f:
                extra = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        plan = extra.get("contact_mode_seq")
        if not isinstance(plan, dict):
            continue
        parts = extra.get("to_cost_parts") or {}
        max_vio = parts.get("constraint_violation_max")
        converged = bool(extra.get("to_converged"))
        # Runs evaluated with the physical check record to_valid; older
        # runs fall back to the solver-residual near-feasible rule.
        valid = extra.get("to_valid")
        if valid is None:
            valid = converged or (max_vio is not None and float(max_vio) <= near_feasible)
        if not valid:
            continue
        h = canonical_plan_hash(plan)
        cand = {"gen": label, "plan": plan, "plan_hash": h, "converged": converged, "valid": bool(valid),
                "max_violation": max_vio, "to_cost": extra.get("to_cost"),
                "signature": plan_signature(plan_to_tuples(plan)),
                "patches": plan_patches(plan_to_tuples(plan))}
        best = by_hash.get(h)
        rank = lambda c: (not c["converged"], c["max_violation"] or 0.0)  # noqa: E731
        if best is None or rank(cand) < rank(best):
            by_hash[h] = cand
    families: Dict[str, List[Dict[str, Any]]] = {}
    for c in sorted(by_hash.values(), key=lambda c: (not c["converged"], c["max_violation"] or 0.0)):
        families.setdefault(c["signature"], []).append(c)
    out = []
    for members in families.values():
        out += members[:max_per_family] if max_per_family > 0 else members
    return out


# ---------------------------------------------------------------------------
# Variant task (worker; faro)
# ---------------------------------------------------------------------------


def vary_task(task: Any, factors: Dict[str, Any]) -> Tuple[Any, bool]:
    """TaskContext copy with the sampled physics / start state, and whether
    kinematics changed (geometry or start pose -> KSO re-check)."""
    from faro.configs.scene import OnFace, Scene, Standing

    from _task_loader import initial_spec, scene_goal_factories
    import layouts

    mass = factors.get("mass_scale") or {}
    mass_kg = factors.get("mass_kg") or {}
    size = factors.get("size_scale") or {}
    objects = dict(task.scene.objects)
    old_half: Dict[str, np.ndarray] = {}
    for name, cfg in task.scene.env_objects.items():
        new_mass = float(mass_kg[name]) if name in mass_kg else float(cfg.mass) * float(mass.get(name, 1.0))
        s = [float(v) for v in size.get(name, (1.0, 1.0, 1.0))]
        if new_mass != float(cfg.mass) or any(v != 1.0 for v in s):
            old_half[name] = np.asarray(cfg.half_extents, float)
            objects[name] = replace(cfg, mass=new_mass,
                                    half_extents=tuple(float(h) * f for h, f in zip(cfg.half_extents, s)))
    vscene = Scene(name=task.scene.name, robots=task.scene.robots, objects=objects,
                   include_robot_ees=task.scene.include_robot_ees)

    spec = initial_spec(task.scene_name, task.ic_name, vscene)
    robots, placements = dict(spec.robots), dict(spec.objects)
    robot_off = factors.get("robot_offset")
    moved = False
    if robot_off and any(robot_off):
        start = robots.get(task.robot_id)
        if isinstance(start, Standing):
            robots[task.robot_id] = replace(start, xy_yaw=tuple(
                float(a) + float(b) for a, b in zip(start.xy_yaw, robot_off)))
            moved = True
    for name, off in (factors.get("object_offsets") or {}).items():
        p = placements.get(name)
        # Objects stacked on a movable object ride along with their support.
        if isinstance(p, OnFace) and p.on[0] not in task.scene.env_objects and any(off):
            placements[name] = replace(p, xy_yaw=tuple(
                float(a) + float(b) for a, b in zip(p.xy_yaw, off)))
            moved = True
    vresolved = vscene.resolve(replace(spec, robots=robots, objects=placements))

    out = copy.copy(task)
    out.scene, out.resolved_state = vscene, vresolved
    if old_half and task.terminal_goal is not None:
        # Goals are world positions: correct each target for the size
        # change. Factories that derive the goal from the scene (e.g. a
        # goal IC) already account for it; literal positions get the
        # change in the object's half-height.
        tg_f, _ = scene_goal_factories(task.scene_name, task.terminal_goal_name,
                                       task.outcome_scorer_name)
        g_base = tg_f(task.scene, task.resolved_state) if tg_f else None
        g_var = tg_f(vscene, vresolved) if tg_f else None
        offsets: Dict[str, np.ndarray] = {}
        for name in task.terminal_goal.objects:
            d = np.zeros(3)
            if g_base is not None and g_var is not None and name in g_base.objects:
                d = np.asarray(g_var.objects[name].value[:3], float) - np.asarray(
                    g_base.objects[name].value[:3], float)
            if not np.any(np.abs(d) > 1e-9) and name in old_half:
                d = np.array([0.0, 0.0, float(objects[name].half_extents[2]) - old_half[name][2]])
            offsets[name] = d
        out.terminal_goal = layouts.shift_goal(task.terminal_goal, offsets)
        out.outcome_scorer = layouts.shift_scorer(task.outcome_scorer, offsets)
    return out, bool(old_half and size) or moved


def _params(job: Dict[str, Any], sample: Dict[str, Any], vtask: Any, solve: Any,
            row: Dict[str, Any], tpm: float, nodes_per_mode: int,
            plan_used: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
    import layouts

    base_ic, layout = layouts.split_ic(job["ic"])
    repaired = plan_used is not None and plan_used != job["plan"]
    return {
        # The plan the trajectory follows: the search's plan, or the one the
        # segmented solve repaired by inserting walking steps (to_eval).
        "contact_mode_seq": plan_used if plan_used is not None else job["plan"],
        "plan_repairs": list(getattr(solve, "repairs", None) or []),
        "original_contact_mode_seq": job["plan"] if repaired else None,
        "dataset_schema": 1,
        "scene": job["scene"], "initial_condition": job["ic"], "base_ic": base_ic,
        "layout": layout, "description": job["description"],
        "run_id": job["run_id"], "run_dir": job["run_dir"], "generation": job["gen"],
        "plan_hash": job["plan_hash"], "plan_signature": job["signature"],
        "plan_patches": job["patches"],
        "num_modes": solve.num_modes, "nodes_per_mode": nodes_per_mode,
        "time_per_mode": tpm, "num_nodes": solve.num_nodes,
        "sample_id": sample["sample_id"], "seed": sample["seed"], "factors": sample["factors"],
        "physics": dx.physics_record(vtask.scene, vtask.resolved_state),
        "terminal_goal": dx.goal_record(vtask.terminal_goal),
        "to": {"status": solve.status, "converged": solve.converged, "cost": solve.cost,
               "max_violation": solve.max_constraint_violation,
               "violation_sum": solve.constraint_violation, "jerk_cost": solve.jerk_cost},
        "task_quality": row.get("task_quality"), "quality": row.get("quality"),
        "versions": job.get("versions"),
    }


def run_worker(job_path: str) -> int:
    with open(job_path) as f:
        job = json.load(f)
    from _task_loader import load_task, to_faro_plan
    from ik_eval import check_plan
    from scoring import task_quality
    from to_eval import NODES_PER_MODE, TIME_PER_MODE, solve_to

    task = load_task(scene=job["scene"], initial_condition=job["ic"], description=job["description"])
    plan = plan_to_tuples(job["plan"])
    plan_dir = os.path.join(job["out"], job["rel_dir"])
    os.makedirs(plan_dir, exist_ok=True)
    log_path = os.path.join(plan_dir, "solver.log")

    for sample in job["samples"]:
        t0 = time.time()
        row: Dict[str, Any] = {
            "run_id": job["run_id"], "run_dir": job["run_dir"], "scene": job["scene"],
            "initial_condition": job["ic"], "gen": job["gen"], "plan_hash": job["plan_hash"],
            "signature": job["signature"], "sample_id": sample["sample_id"],
            "seed": sample["seed"], "factors": sample["factors"], "kso_feasible": None,
            "status": None, "converged": None, "max_violation": None, "to_cost": None,
            "task_quality": None, "quality": None, "skipped": "", "error": None,
            "path": None, "seconds": None,
        }
        try:
            vtask, kinematics_changed = vary_task(task, sample["factors"])
            tpm = float(sample["factors"].get("time_per_mode", TIME_PER_MODE))
            with redirect_fds_to(log_path):
                if kinematics_changed:
                    cost, _res, err = check_plan(vtask, plan)
                    row["kso_feasible"] = cost is not False
                    if cost is False:
                        row["skipped"] = "KSO infeasible" + (f" ({err})" if err else "")
                solve = None
                if not row["skipped"]:
                    solve = solve_to(vtask, plan, time_per_mode=tpm,
                                     weight_scale=sample["factors"].get("weight_scale"))
            if solve is not None:
                row.update(status=solve.status, converged=solve.converged,
                           max_violation=solve.max_constraint_violation, to_cost=solve.cost,
                           task_quality=task_quality(vtask, solve.results))
                row["quality"] = dx.classify(solve.converged, solve.max_constraint_violation,
                                             job["near_feasible"],
                                             None if solve.physical is None else solve.physically_valid)
                row["iterate"] = solve.iterate
                row["goal"] = getattr(solve, "goal", None)
                if row["quality"] and not getattr(solve, "goal_met", True):
                    # A usable trajectory that leaves the object off its hard
                    # TO goal is no sample of the task (to_eval.goal_check).
                    row["quality"] = None
                    row["skipped"] = f"misses the task goal (excess {solve.goal.get('excess')})"
                if row["quality"]:
                    # A segmented solve may have inserted walking steps: the
                    # trajectory follows the repaired plan (to_eval.ToSolve.plan).
                    used = plan_to_tuples(solve.plan) if getattr(solve, "plan", None) else plan
                    row["plan_repairs"] = len(getattr(solve, "repairs", None) or [])
                    sample_dir = os.path.join(plan_dir, sample["sample_id"])
                    params = _params(job, sample, vtask, solve, row, tpm, NODES_PER_MODE,
                                     plan_used={ch: [list(m) for m in seq] for ch, seq in used.items()})
                    dx.export_sample(
                        sample_dir, vtask.scene, vtask.resolved_state,
                        to_faro_plan(vtask.scene, used), solve.results, params,
                        robot_id=vtask.robot_id, num_modes=solve.num_modes,
                        nodes_per_mode=NODES_PER_MODE, frequency=job["frequency"],
                    )
                    row["path"] = os.path.relpath(sample_dir, job["out"])
                elif not row["skipped"]:
                    row["skipped"] = (f"TO status {solve.status}, not physically valid "
                                      f"({', '.join((solve.physical or {}).get('fails', []))})"
                                      if solve.physical else
                                      f"TO status {solve.status}, max violation "
                                      f"{solve.max_constraint_violation:.3g} > {job['near_feasible']}")
        except Exception as e:  # noqa: BLE001 — one bad sample must not stop the plan
            row["error"] = f"{type(e).__name__}: {e}"
            with open(log_path, "a") as f:
                f.write(f"\n--- {sample['sample_id']} traceback ---\n{traceback.format_exc()}")
        row["seconds"] = round(time.time() - t0, 1)
        dx.append_manifest(job["out"], row)
        state = row["error"] or row["skipped"] or f"{row['quality']} (max violation {row['max_violation']:.3g})"
        print(f"[{job['run_id']} {job['plan_hash'][:12]}] {sample['sample_id']}: {state}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def _parse_ranges(items: Sequence[str]) -> Ranges:
    r = Ranges()
    for item in items or []:
        key, _, val = item.partition("=")
        key = key.strip().replace("-", "_")
        if not hasattr(r, key):
            raise SystemExit(f"--range: unknown factor {key!r}; one of {list(vars(Ranges()))}")
        lo, hi = (float(v) for v in val.split(","))
        setattr(r, key, (lo, hi))
    return r


def _run_dirs(args: argparse.Namespace) -> List[str]:
    from run_to_all import _discover_runs

    runs = [os.path.abspath(d) for d in args.run_dir or []]
    for root in args.runs_root or []:
        runs += _discover_runs(root)
    return sorted(dict.fromkeys(runs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", help="finished run dir (repeatable)")
    p.add_argument("--runs-root", action="append", help="take every run below this dir")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "datasets", "loco"))
    p.add_argument("--samples", type=int, default=8, help="random samples per plan (+ reference)")
    p.add_argument("--factors", default=",".join(ALL_FACTORS))
    p.add_argument("--range", action="append", default=[], metavar="NAME=LO,HI",
                   help=f"override a sampling range; names: {', '.join(vars(Ranges()))}")
    p.add_argument("--near-feasible", type=float, default=0.1,
                   help="export non-converged solves with max violation <= this")
    p.add_argument("--max-per-family", type=int, default=3)
    p.add_argument("--max-plans", type=int, default=0, help="per run (0 = all); for smoke tests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frequency", type=float, default=50.0, help="npz sample rate (Hz)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        sys.exit(run_worker(args.worker))

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    if set(factors) - set(ALL_FACTORS):
        p.error(f"unknown factors {sorted(set(factors) - set(ALL_FACTORS))}; choose from {ALL_FACTORS}")
    ranges = _parse_ranges(args.range)
    out = os.path.abspath(args.out)
    runs = _run_dirs(args)
    if not runs:
        p.error("give --run-dir or --runs-root")
    done = set() if args.force else dx.done_keys(out)
    versions = dx.git_versions(REPO_ROOT)

    jobs = []
    for run_dir in runs:
        task_id = read_hydra_task(run_dir)
        if task_id is None:
            print(f"[skip] {run_dir}: no .hydra/config.yaml with task.scene")
            continue
        scene, ic, desc = task_id
        scene = canonical_scene_name(scene)
        run_id = f"{scene}/{ic}/{os.path.basename(run_dir)}"
        plans = collect_plans(run_dir, args.near_feasible, args.max_per_family)
        if args.max_plans:
            plans = plans[: args.max_plans]
        print(f"[scan] {run_id}: {len(plans)} plan(s) "
              f"({sum(c['converged'] for c in plans)} converged)")
        for c in plans:
            samples = plan_samples(run_id, c["plan_hash"], plan_objects(c["plan"]), args.samples,
                                   args.seed, factors, ranges)
            todo = [s for s in samples if (run_id, c["plan_hash"], s["sample_id"]) not in done]
            print(f"  {c['gen']:>8} {c['plan_hash'][:12]} converged={c['converged']} "
                  f"todo={len(todo)}/{len(samples)}  {c['signature'][:80]}")
            if todo:
                jobs.append({**c, "samples": todo, "scene": scene, "ic": ic, "description": desc,
                             "run_id": run_id, "run_dir": run_dir, "out": out,
                             "rel_dir": os.path.join(scene, ic, os.path.basename(run_dir),
                                                     c["plan_hash"][:12]),
                             "near_feasible": args.near_feasible, "frequency": args.frequency,
                             "versions": versions})
    n = sum(len(j["samples"]) for j in jobs)
    print(f"[plan] {len(jobs)} plan worker(s), {n} sample(s) -> {out}")
    if args.dry_run or not jobs:
        print("[dry-run] nothing solved." if args.dry_run else "[done] nothing to do.")
        return

    os.makedirs(out, exist_ok=True)
    env = {**os.environ, "SHINKA_TO_N_JOB": os.environ.get("SHINKA_TO_N_JOB", "1")}

    def launch(job: Dict[str, Any]) -> int:
        job_path = os.path.join(out, job["rel_dir"], "job.json")
        os.makedirs(os.path.dirname(job_path), exist_ok=True)
        with open(job_path, "w") as f:
            json.dump(job, f)
        return subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", job_path],
                              cwd=_THIS_DIR, env=env, check=False).returncode

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        codes = list(pool.map(launch, jobs))
    rows = dx.write_summary(out)
    exported = sum(r["exported"] for r in rows)
    print(f"[done] {exported} trajectories exported in total; summary: "
          f"{os.path.join(out, dx.SUMMARY)}"
          + (f"; {sum(c != 0 for c in codes)} worker(s) crashed" if any(codes) else ""))


if __name__ == "__main__":
    main()
