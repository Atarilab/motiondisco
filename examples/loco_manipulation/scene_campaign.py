"""
Solve as many registered scenes as possible: one MotionDisco search per
scene, easiest first, each stopped as soon as it has enough unique valid
plans, or at a time cap, then the next scene.

    python examples/loco_manipulation/scene_campaign.py --valid 5 --hours 4 [--dry-run]
    python examples/loco_manipulation/scene_campaign.py --only generated.two_box_counter_swap_001
    python examples/loco_manipulation/scene_campaign.py --to-goal hard \\
        --scenes motiondisco.long_dist_pick_place motiondisco.under_table_pick_place floor_stack_carry

Scenes come from ``scenes/generated/*/spec.json`` (``register_generated.py``)
and are ordered by number of boxes, then terrain and manipulation difficulty
(``order_key``); the LLM-facing description is the spec's. ``--scenes`` runs
any registry scenes instead, in the given order, with run_search.sh's
descriptions; ``--to-goal hard`` forces a hard TO goal (5 cm / 0.1 rad by
default, `_task_loader.with_to_goal`) on scenes whose goals.py leaves TO
free, so that a valid plan means the object reached its target.
``--dataset-out DIR`` adds the dataset follower: every solved scene's run is
handed to ``generate_dataset.py`` (physics-varied re-solves of its plans:
``--dataset-samples`` per plan, ``--dataset-range``, ``--dataset-jobs``
workers in a ``--dataset-cpus`` / ``--dataset-mem`` scope), one scene at a
time in a worker thread while the search moves on; ``--dataset-since`` also
queues scenes solved before this start. Per scene:

1. The TO code is generated once (``warm_codegen``: a solve of the initial
   mode plus two steps with the task's goal and once without it, for the
   ``_tohard`` and the goal-free folders the windows use; ~20 min; skipped
   when ``moto_gen/<scene>`` exists), outside any evaluation's time limit.
2. ``run_search.sh`` runs the search (diverse, quality
   score, the long-horizon settings, ``--eval-jobs`` evaluations at a time,
   ``--job-time`` per evaluation) under a systemd scope with a CPU quota and
   a memory cap. The campaign polls the run directory: at ``--valid`` unique
   valid plans (distinct ``contact_mode_seq`` with ``to_valid``) it stops the
   search; when the node budget runs out it resumes with ``--extra`` more
   nodes up to ``--cap``; at ``--hours`` of wall time it stops and moves on.
3. One line per scene goes to ``<out>/campaign_summary.jsonl`` (scene, boxes,
   difficulty, outcome, valid plans and their mode counts, evaluations,
   hours) and ``campaign_summary.csv`` is rewritten. Rerunning the campaign
   skips scenes whose outcome is ``solved`` (``--redo`` repeats them).

A KSO solver build interrupted by the stop is deleted from the cache (it
would otherwise be loaded half built). The G1's shin capsules get the 2 cm
collision margin by default (``--leg-margin`` / ``--margin-capsules``; see
`_task_loader` and ``LEG_CAPSULES`` for why not the feet).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
SEARCH_SCRIPT = os.path.join(_THIS_DIR, "run_search.sh")
SCENES_DIR = os.path.join(_THIS_DIR, "scenes", "generated")
SOLVERS_DIR = os.path.join(_THIS_DIR, "solvers")
MOTO_GEN = os.path.join(_THIS_DIR, "moto_gen")
EXP = "loco_abl3_ik_fb_diverse_quality"
# Capsules the collision margin applies to. Not the feet: standing on a box
# support their capsule bottom sits 5 mm above the surface, so a 2 cm
# margin put it 1.5 cm inside the raised_dock scene's dock at mode 0 and
# every KSO failed there (the floor plane is exempt for a planted sole, a
# box support is not).
LEG_CAPSULES = "l_shin,r_shin"

# Long-horizon search settings (the ones the pallet run used).
SEARCH_ENV = {
    "DIVERSE": "1", "SCORE_MODE": "quality", "SHINKA_IK_MAX_NUM_NODES": "130", "SHINKA_TO_N_JOB": "1",
    "SHINKA_TO_SEGMENT_MODES": "13", "SHINKA_TO_SEGMENT_CUTS": "manip", "SHINKA_TO_SEGMENT_LOOKAHEAD": "5",
    "SHINKA_TO_SEGMENT_MIN_MODES": "2", "SHINKA_TO_MAX_ITERS": "100", "SHINKA_TO_FEASIBILITY_TOL": "0.1",
    "SHINKA_TO_STALL_ITERS": "30", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "MAKEFLAGS": "-j2", "PYTHONUNBUFFERED": "1",
}


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def scene_specs(scenes_dir: str = SCENES_DIR) -> List[Tuple[str, Dict[str, Any]]]:
    """``(scene id, spec)`` for every registered generated scene."""
    out = []
    for path in sorted(glob.glob(os.path.join(scenes_dir, "*", "spec.json"))):
        with open(path) as f:
            spec = json.load(f)
        out.append((f"generated.{os.path.basename(os.path.dirname(path))}", spec))
    return out


FARO_SCENES = os.path.join(REPO_ROOT, "submodules", "faro", "src", "scenes")
LOCAL_SCENES = os.path.join(_THIS_DIR, "scenes")


def scene_module_path(scene: str, roots: Optional[Tuple[str, str]] = None) -> Optional[str]:
    """``scene.py`` of a registry scene: ``family.name`` lives in
    faro's scene families, a bare name in this example's
    ``scenes/``; None when neither exists."""
    tamp, local = roots or (FARO_SCENES, LOCAL_SCENES)
    if "." in scene:
        family, name = scene.split(".", 1)
        candidates = [os.path.join(tamp, family, name, "scene.py"), os.path.join(local, family, name, "scene.py")]
    else:
        candidates = [os.path.join(local, scene, "scene.py")]
    return next((c for c in candidates if os.path.isfile(c)), None)


def explicit_specs(scenes: Sequence[str], roots: Optional[Tuple[str, str]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """``(scene id, spec)`` for scenes named on the command line, in that
    order: the box count is read off ``scene.py`` (one ``MovableBox`` per
    movable object), there is no difficulty, and no description — the
    search then uses run_search.sh's own table."""
    out = []
    for scene in scenes:
        path = scene_module_path(scene, roots)
        n = 0
        if path is not None:
            with open(path) as f:
                n = f.read().count("MovableBox(")
        out.append((scene, {"id": scene, "boxes": [{}] * n, "difficulty": None, "description": None}))
    return out


def order_key(spec: Dict[str, Any]) -> Tuple[int, float, float, str]:
    """Easy first: fewer boxes, then terrain, then manipulation difficulty."""
    diff = spec.get("difficulty") or {}
    return (len(spec.get("boxes") or []), float(diff.get("terrain") or 0.0),
            float(diff.get("manipulation") or 0.0), str(spec.get("id", "")))


def plan_hash(plan: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def valid_plans(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """Unique valid plans of a run: hash -> {gen, modes, score}."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "gen_*", "results", "extra.json")),
                       key=lambda p: int(p.split("gen_")[-1].split("/")[0])):
        try:
            with open(path) as f:
                extra = json.load(f)
        except (OSError, ValueError):
            continue
        plan = extra.get("contact_mode_seq")
        if extra.get("to_valid") is not True or not plan:
            continue
        gen = os.path.basename(os.path.dirname(os.path.dirname(path)))
        try:
            with open(os.path.join(os.path.dirname(path), "metrics.json")) as f:
                score = float(json.load(f).get("combined_score") or 0.0)
        except (OSError, ValueError):
            score = 0.0
        out.setdefault(plan_hash(plan), {"gen": gen, "modes": len(next(iter(plan.values()))), "score": score})
    return out


def evaluations(run_dir: str) -> int:
    return len(glob.glob(os.path.join(run_dir, "gen_*", "results", "extra.json")))


def nodes(run_dir: str) -> int:
    """Programs in the run's archive minus the island copies of generation 0."""
    import sqlite3

    try:
        con = sqlite3.connect(f"file:{run_dir}/programs.sqlite?mode=ro", uri=True)
        return max(0, int(con.execute("select count(*) from programs").fetchone()[0]) - 3)
    except Exception:  # noqa: BLE001 - no archive yet
        return 0


def latest_run(results_dir: str, scene: str) -> Optional[str]:
    runs = sorted(glob.glob(os.path.join(results_dir, EXP, scene, "*_diverse")), key=os.path.getmtime)
    return runs[-1] if runs else None


def codegen_dir(scene: str) -> str:
    return os.path.join(MOTO_GEN, scene.replace(".", "_"))


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

def launcher_pid(scene: str) -> Optional[int]:
    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()
    for line in ps:
        pid, _, args = line.strip().partition(" ")
        if "motiondisco_launch task=loco_manipulation" in args and f"task.scene={scene}" in args:
            return int(pid)
    return None


def _children(pid: int) -> List[int]:
    try:
        kids = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True).stdout.split()
    except Exception:  # noqa: BLE001
        return []
    out: List[int] = []
    for k in map(int, kids):
        out += [k] + _children(k)
    return out


def stop_search(scene: str, log=print) -> None:
    """SIGINT the launcher, SIGTERM what is left of its tree, and drop a KSO
    solver whose compile the stop interrupted."""
    pid = launcher_pid(scene)
    if pid is None:
        return
    tree = _children(pid)
    builds = []
    for k in tree:
        try:
            with open(f"/proc/{k}/cmdline") as f:
                if f.read().startswith("make"):
                    builds.append(os.readlink(f"/proc/{k}/cwd"))
        except OSError:
            pass
    os.kill(pid, signal.SIGINT)
    time.sleep(10)
    for k in [pid] + tree:
        try:
            os.kill(k, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for d in builds:
        if not glob.glob(os.path.join(d, "libacados_ocp_solver_*.so")):
            shutil.rmtree(d, ignore_errors=True)
            try:
                os.remove(os.path.join(SOLVERS_DIR, os.path.basename(d) + ".json"))
            except OSError:
                pass
            log(f"removed half-built KSO solver {os.path.basename(d)[:50]}...")
    log(f"stopped the search of {scene} (pid {pid}, {len(tree)} child processes)")


def scoped(cmd: Sequence[str], args: argparse.Namespace, cpus: Optional[int] = None,
           mem: Optional[str] = None) -> List[str]:
    """``cmd`` under a transient systemd scope with the campaign's CPU and
    memory caps (or the given ones), niced."""
    return ["systemd-run", "--user", "--scope", "-p", f"CPUQuota={int(cpus or args.cpus) * 100}%",
            "-p", f"MemoryMax={mem or args.mem}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue", "--quiet",
            "nice", "-n", "19", *cmd]


# ---------------------------------------------------------------------------
# Dataset follower: physics-varied re-solves of every solved scene's plans
# (generate_dataset.py), one scene at a time beside the running search.
# ---------------------------------------------------------------------------

# The search's TO settings minus the incremental window cache (its key does
# not include the varied masses and sizes); the goal override and the
# collision margin come from the campaign arguments.
DATASET_ENV = {k: v for k, v in SEARCH_ENV.items() if k.startswith("SHINKA_TO") or k.endswith("_THREADS")}
DATASET_ENV.update({"SHINKA_TO_INCREMENTAL": "0", "SHINKA_TO_N_JOB": "1", "PYTHONUNBUFFERED": "1"})
DATASET_FACTORS = "mass_kg,size,robot,objects,timing"
DATASET_RANGES = ("mass_kg=0.5,10", "size=0.9,1.1")


def dataset_command(args: argparse.Namespace, run_dir: str) -> List[str]:
    """The generate_dataset.py call for one finished run."""
    cmd = [sys.executable, os.path.join(_THIS_DIR, "generate_dataset.py"), "--run-dir", run_dir,
           "--out", args.dataset_out, "--samples", str(int(args.dataset_samples)),
           "--factors", args.dataset_factors, "--jobs", str(int(args.dataset_jobs))]
    for r in args.dataset_range or DATASET_RANGES:
        cmd += ["--range", r]
    return cmd


def dataset_backlog(records: Dict[str, Dict[str, Any]], since: Optional[str]) -> List[Tuple[str, str]]:
    """``(scene, run_dir)`` of solved scenes whose run started at or after
    ``since`` (a run-dir timestamp prefix such as ``2026.11.02``) — what a
    (re)started campaign owes the dataset. Runs already in the manifest are
    queued too: generate_dataset skips the samples it has and so finishes a
    job that a restart interrupted, at the cost of a few seconds for a
    complete one."""
    out = []
    for scene, rec in records.items():
        run = rec.get("run_dir")
        if rec.get("outcome") != "solved" or not run:
            continue
        if since and os.path.basename(run) < since:
            continue
        out.append((scene, run))
    return out


class DatasetQueue:
    """Runs dataset generation for solved scenes sequentially in a worker
    thread, each under its own CPU / memory scope, logging to
    ``<dataset_out>/dataset_<scene>.log``."""

    def __init__(self, args: argparse.Namespace, log=print) -> None:
        import queue
        import threading

        self.args, self.log = args, log
        self.todo: "queue.Queue[Optional[Tuple[str, str]]]" = queue.Queue()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def put(self, scene: str, run_dir: str) -> None:
        self.todo.put((scene, run_dir))

    def close(self) -> None:
        self.todo.put(None)
        self.thread.join()

    def _work(self) -> None:
        while True:
            item = self.todo.get()
            if item is None:
                return
            scene, run_dir = item
            os.makedirs(self.args.dataset_out, exist_ok=True)
            env = {**os.environ, **DATASET_ENV, **goal_env(self.args)}
            if self.args.leg_margin > 0:
                env["SHINKA_COLLISION_MARGIN"] = str(self.args.leg_margin)
                env["SHINKA_COLLISION_MARGIN_CAPSULES"] = self.args.margin_capsules
            self.log(f"[{scene}] dataset: {self.args.dataset_samples} samples per plan -> {self.args.dataset_out}")
            t0 = time.time()
            with open(os.path.join(self.args.dataset_out, f"dataset_{scene}.log"), "a") as f:
                code = subprocess.run(scoped(dataset_command(self.args, run_dir), self.args,
                                             cpus=self.args.dataset_cpus, mem=self.args.dataset_mem),
                                      env=env, cwd=_THIS_DIR, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
            self.log(f"[{scene}] dataset {'done' if code == 0 else f'exited {code}'} in {(time.time() - t0) / 60:.0f} min")


def manifest_runs(dataset_out: str) -> List[str]:
    """Run dirs that already have rows in the dataset manifest."""
    runs = set()
    try:
        with open(os.path.join(dataset_out, "manifest.jsonl")) as f:
            for line in f:
                if line.strip():
                    runs.add(json.loads(line).get("run_dir"))
    except OSError:
        pass
    return sorted(r for r in runs if r)


def safe_description(text: str) -> str:
    """The spec's description as run_search.sh can pass it to Hydra
    (``task.description='...'``): straight quotes become typographic ones
    (an apostrophe in "the rack's upper slab" ended the override early),
    backslashes become slashes, whitespace is collapsed."""
    out = str(text).replace("'", "’").replace('"', "”").replace("\\", "/")
    return " ".join(out.split())


def goal_env(args: argparse.Namespace) -> Dict[str, str]:
    """The TO-goal override (``--to-goal`` / ``--to-goal-tol``; see
    `_task_loader.with_to_goal`) for the search and the codegen."""
    mode = getattr(args, "to_goal", None)
    if not mode:
        return {}
    return {"SHINKA_TO_GOAL": str(mode), "SHINKA_TO_GOAL_TOL": str(float(getattr(args, "to_goal_tol", 0.05)))}


def search_env(scene: str, description: str, args: argparse.Namespace, resume_extra: Optional[int]) -> Dict[str, str]:
    env = {**os.environ, **SEARCH_ENV, "SCENES": scene, "ICS": "default", "RESULTS_DIR": args.out,
           "DESCRIPTION": safe_description(description), "MAX_NODES": str(int(args.nodes)),
           "MAX_EVAL_JOBS": str(int(args.eval_jobs)), "JOB_TIME": args.job_time}
    if args.leg_margin > 0:
        env["SHINKA_COLLISION_MARGIN"] = str(args.leg_margin)
        env["SHINKA_COLLISION_MARGIN_CAPSULES"] = args.margin_capsules
    env.update(goal_env(args))
    if resume_extra is not None:
        env["RESUME"], env["RESUME_EXTRA"] = "1", str(int(resume_extra))
    else:
        env.pop("RESUME", None)
        env.pop("RESUME_EXTRA", None)
    return env


CODEGEN_ATTEMPTS = 3


def warm_codegen(scene: str, args: argparse.Namespace, log=print) -> bool:
    """Generate the scene's TO code (a 3-mode solve) unless its folder exists.
    A failed attempt is retried up to ``CODEGEN_ATTEMPTS`` times: gcc crashes
    now and then on this machine (an internal compiler error 17 s into
    tall_carton_low_bay_stocking_001's codegen; the box logs hardware MCEs)
    and moto keeps one artifact per compiled function, so a retry resumes
    at the function that crashed."""
    if os.path.isdir(os.path.join(codegen_dir(scene), ".moto_artifacts")):
        return True
    code = ("import os, sys, time\n"
            f"os.environ['SHINKA_LOCOMANIP_SCENE'] = {scene!r}\n"
            "os.environ['SHINKA_TO_SEGMENT_MODES'] = '0'\n"
            f"sys.path.insert(0, {_THIS_DIR!r})\n"
            "import env, to_eval\n"
            "from _task_loader import load_task\n"
            f"task = load_task({scene!r}, 'default', '')\n"
            "plan = {ch: list(seq) for ch, seq in env.get_initial_mode().items()}\n"
            "env.walk(plan, 2)\n"
            "t = time.time(); sol = to_eval.solve_to(task, plan)\n"
            "import copy\n"
            "free = copy.copy(task); free.terminal_goal = None   # the goal-free code the intermediate windows use\n"
            "sol2 = to_eval.solve_to(free, plan)\n"
            "print(f'WARM_DONE status={sol.status},{sol2.status} time={time.time() - t:.0f}s', flush=True)\n")
    env = {**os.environ, **SEARCH_ENV, **goal_env(args), "MAKEFLAGS": f"-j{int(args.cpus)}",
           "OMP_NUM_THREADS": str(int(args.cpus))}
    log(f"[{scene}] generating TO code ({codegen_dir(scene)})")
    for attempt in range(1, CODEGEN_ATTEMPTS + 1):
        t0 = time.time()
        proc = subprocess.run(scoped([sys.executable, "-c", code], args), env=env, cwd=_THIS_DIR,
                              capture_output=True, text=True, timeout=3 * 3600)
        ok = "WARM_DONE" in proc.stdout
        log(f"[{scene}] codegen {'done' if ok else 'FAILED'} in {(time.time() - t0) / 60:.0f} min"
            + ("" if ok else f": {proc.stdout[-300:]} {proc.stderr[-500:]}"))
        if ok:
            return True
        if attempt < CODEGEN_ATTEMPTS:
            log(f"[{scene}] retrying codegen ({attempt + 1}/{CODEGEN_ATTEMPTS}); compiled functions are kept")
    return False


# ---------------------------------------------------------------------------
# One scene
# ---------------------------------------------------------------------------

def run_scene(scene: str, spec: Dict[str, Any], args: argparse.Namespace, log=print) -> Dict[str, Any]:
    t0 = time.time()
    record: Dict[str, Any] = {"scene": scene, "boxes": len(spec.get("boxes") or []),
                              "difficulty": spec.get("difficulty"), "family": spec.get("family")}
    if not warm_codegen(scene, args, log):
        record.update(outcome="codegen_failed", hours=round((time.time() - t0) / 3600, 2))
        return record
    deadline = t0 + args.hours * 3600
    description = str(spec.get("description") or "")
    run_dir = latest_run(args.out, scene)
    resume_extra: Optional[int] = None
    if run_dir is not None:
        found = valid_plans(run_dir)
        if len(found) >= args.valid:      # an earlier run already solved it
            record.update(outcome="solved", run_dir=run_dir, valid=len(found),
                          plans=[{"gen": v["gen"], "modes": v["modes"], "score": round(v["score"], 4)} for v in found.values()],
                          evaluations=evaluations(run_dir), nodes=nodes(run_dir), hours=0.0)
            return record
        resume_extra = args.extra          # continue an earlier run of this scene
    budget_exhausted = False
    proc = subprocess.Popen(scoped(["bash", SEARCH_SCRIPT], args), env=search_env(scene, description, args, resume_extra),
                            cwd=REPO_ROOT, stdout=open(os.path.join(args.out, f"campaign_{scene}.log"), "a"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    outcome = "running"
    try:
        while True:
            time.sleep(args.poll)
            run_dir = latest_run(args.out, scene)
            found = valid_plans(run_dir) if run_dir else {}
            if len(found) >= args.valid:
                outcome = "solved"
                break
            if time.time() > deadline:
                outcome = "capped"
                break
            if proc.poll() is not None:
                have = nodes(run_dir) if run_dir else 0
                if run_dir is None or have >= args.cap:
                    outcome = "budget" if run_dir else "no_run"
                    budget_exhausted = True
                    break
                log(f"[{scene}] node budget used ({have} nodes, {len(found)} valid); resuming with {args.extra} more")
                proc = subprocess.Popen(scoped(["bash", SEARCH_SCRIPT], args),
                                        env=search_env(scene, description, args, args.extra), cwd=REPO_ROOT,
                                        stdout=open(os.path.join(args.out, f"campaign_{scene}.log"), "a"),
                                        stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        if not budget_exhausted:
            stop_search(scene, log)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    run_dir = latest_run(args.out, scene)
    found = valid_plans(run_dir) if run_dir else {}
    record.update(outcome=outcome, run_dir=run_dir, valid=len(found),
                  plans=[{"gen": v["gen"], "modes": v["modes"], "score": round(v["score"], 4)} for v in found.values()],
                  evaluations=evaluations(run_dir) if run_dir else 0, nodes=nodes(run_dir) if run_dir else 0,
                  hours=round((time.time() - t0) / 3600, 2))
    return record


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

DONE_OUTCOMES = ("solved", "capped", "budget")


def should_run(record: Optional[Dict[str, Any]], redo: bool = False) -> bool:
    """Whether a scene is still due: never attempted, or attempted without
    a verdict (a codegen failure, an interrupted run). A scene that was
    solved, or that spent its time cap or node cap without solving, is
    left alone unless ``--redo``; the user decides whether a hard scene
    gets another four hours."""
    return redo or not record or record.get("outcome") not in DONE_OUTCOMES


def load_summary(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["scene"]] = rec
    except OSError:
        pass
    return out


def write_csv(records: Dict[str, Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "boxes", "terrain", "manipulation", "outcome", "valid", "modes", "evaluations", "hours"])
        for rec in records.values():
            diff = rec.get("difficulty") or {}
            w.writerow([rec["scene"], rec.get("boxes"), diff.get("terrain"), diff.get("manipulation"), rec.get("outcome"),
                        rec.get("valid", 0), " ".join(str(p["modes"]) for p in rec.get("plans", [])),
                        rec.get("evaluations", 0), rec.get("hours", 0)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes-dir", default=SCENES_DIR)
    p.add_argument("--out", default="results/long_horizon", help="results root (run_search.sh RESULTS_DIR)")
    p.add_argument("--valid", type=int, default=5, help="unique valid plans that solve a scene")
    p.add_argument("--hours", type=float, default=4.0, help="wall-time cap per scene")
    p.add_argument("--nodes", type=int, default=40, help="first node budget per scene")
    p.add_argument("--extra", type=int, default=20, help="nodes added when the budget runs out")
    p.add_argument("--cap", type=int, default=100, help="node cap per scene")
    p.add_argument("--eval-jobs", type=int, default=2)
    p.add_argument("--job-time", default="02:00:00")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="32G")
    p.add_argument("--leg-margin", type=float, default=0.02, help="collision margin on --margin-capsules (0 = off)")
    p.add_argument("--margin-capsules", default=LEG_CAPSULES)
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--only", nargs="*", help="scene ids to run (generated.<id> or <id>)")
    p.add_argument("--scenes", nargs="*",
                   help="run exactly these registry scenes in this order instead of the generated ones "
                        "(e.g. motiondisco.long_dist_pick_place floor_stack_carry); descriptions come from "
                        "run_search.sh's table")
    p.add_argument("--to-goal", choices=("hard", "soft", "off"), default=None,
                   help="force how TO treats every scene's terminal goal (SHINKA_TO_GOAL); 'hard' makes a valid "
                        "plan mean the object reached its target")
    p.add_argument("--to-goal-tol", type=float, default=0.05, help="TO goal tolerance (m / rad) with --to-goal hard")
    p.add_argument("--dataset-out", default=None,
                   help="also run generate_dataset.py on every solved scene (one at a time, beside the search) into this root")
    p.add_argument("--dataset-samples", type=int, default=10)
    p.add_argument("--dataset-jobs", type=int, default=2, help="plan workers of each dataset job")
    p.add_argument("--dataset-cpus", type=int, default=3)
    p.add_argument("--dataset-mem", default="24G")
    p.add_argument("--dataset-factors", default=DATASET_FACTORS)
    p.add_argument("--dataset-range", action="append", default=None, metavar="NAME=LO,HI",
                   help=f"generate_dataset ranges (default {' '.join(DATASET_RANGES)})")
    p.add_argument("--dataset-since", default=None,
                   help="also queue scenes already solved whose run started at/after this run-dir timestamp "
                        "prefix (e.g. 2026.11.02) and that are not in the manifest yet")
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--max-boxes", type=int, default=None)
    p.add_argument("--redo", action="store_true", help="run scenes already marked solved / capped / budget")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.out = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "campaign_summary.jsonl")
    records = load_summary(summary_path)

    def log(msg: str) -> None:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    if args.scenes:
        scenes = explicit_specs(args.scenes)          # the given order is the run order
    else:
        scenes = sorted(scene_specs(args.scenes_dir), key=lambda s: order_key(s[1]))
    if args.only:
        wanted = {s if s.startswith("generated.") else f"generated.{s}" for s in args.only}
        scenes = [s for s in scenes if s[0] in wanted]
    if args.max_boxes is not None:
        scenes = [s for s in scenes if len(s[1].get("boxes") or []) <= args.max_boxes]
    todo = [s for s in scenes if should_run(records.get(s[0]), args.redo)]
    if args.max_scenes is not None:
        todo = todo[:args.max_scenes]
    log(f"{len(scenes)} scenes, {len(todo)} to run; target {args.valid} valid plans, cap {args.hours} h each")
    datasets = None
    if args.dataset_out and not args.dry_run:
        args.dataset_out = args.dataset_out if os.path.isabs(args.dataset_out) else os.path.join(REPO_ROOT, args.dataset_out)
        datasets = DatasetQueue(args, log)
        for scene, run in dataset_backlog(records, args.dataset_since):
            log(f"[{scene}] dataset backlog: {run}")
            datasets.put(scene, run)
    for scene, spec in todo:
        diff = spec.get("difficulty") or {}
        state = records.get(scene, {}).get("outcome", "-")
        line = (f"  {scene:<60} boxes {len(spec.get('boxes') or []):>2}  terrain {diff.get('terrain', 0):<4} "
                f"manip {diff.get('manipulation', 0):<4} [{state}]")
        if args.dry_run:
            print(line)
            continue
        log(line.strip())
        rec = run_scene(scene, spec, args, log)
        records[scene] = rec
        with open(summary_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        write_csv(records, os.path.join(args.out, "campaign_summary.csv"))
        log(f"[{scene}] {rec['outcome']}: {rec.get('valid', 0)} valid plans "
            f"{[p['modes'] for p in rec.get('plans', [])]} in {rec.get('hours')} h, {rec.get('evaluations', 0)} evaluations")
        if datasets is not None and rec.get("outcome") == "solved" and rec.get("run_dir"):
            datasets.put(scene, rec["run_dir"])
    if not args.dry_run:
        solved = sum(1 for r in records.values() if r.get("outcome") == "solved")
        log(f"campaign: {solved} scenes solved of {len(records)} attempted")
    if datasets is not None:
        log("campaign: waiting for the dataset jobs to finish")
        datasets.close()


if __name__ == "__main__":
    main()
