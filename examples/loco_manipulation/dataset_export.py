"""
Write one trajectory sample of the RL-tracker dataset, plus the manifest.

Per accepted TO solve, under ``<out>/<scene>/<ic>/<run>/<plan12>/<sample>/``:

    traj.npz      faro.utils.results.pin_to_mj at ``frequency`` Hz (root /
                  joint / object pos, rot (wxyz), world-frame velocities,
                  ``contact_id`` per sample, schema_version 2/3), plus the
                  native solver nodes: ``solver_time`` (N+1), ``solver_t_scale``,
                  ``solver_torques`` (N, nj), ``solver_wrench_<patch>`` and
                  ``solver_obj_wrench_<obj>`` (N, 6), ``solver_robot_xs`` /
                  ``solver_obj_xs_<obj>``, and ``params_json`` (params.json
                  as a string). Load with ``faro.utils.results.
                  load_trajectory_archive`` (no pickle).
    scene.xml     faro.utils.results.save_scene_xml of the varied scene
                  (geometry, masses, start poses)
    params.json   everything needed to rebuild the solve: scene / IC /
                  layout, run and plan identity, the contact plan, the
                  physical parameters (per-object mass, inertia,
                  half-extents, start pose; fixed objects; robot start q;
                  mu), the sampled factors and seed, TO settings and result

``<out>/manifest.jsonl`` gets one row per attempted sample (exported or
not, with the reason); ``write_summary`` turns it into ``summary.csv``.

faro is imported lazily; ``to_mj`` / ``scene_xml`` are injectable so the
bookkeeping is testable without solvers.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

MANIFEST = "manifest.jsonl"
SUMMARY = "summary.csv"
FRICTION_MU = 0.5  # faro's TO friction coefficient (constant in its formulation)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _floats(values: Any) -> List[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def physics_record(scene: Any, resolved: Any) -> Dict[str, Any]:
    """Physical and geometric parameters of a (varied) scene + start state."""
    objects = {}
    for name, cfg in scene.env_objects.items():
        st = resolved.env_objects[name]
        objects[name] = {
            "mass": float(cfg.mass),
            "inertia_diag": _floats(cfg.inertia),
            "half_extents": _floats(cfg.half_extents),
            "start_q_xyzw": _floats(st.q[:7]),
        }
    fixed = {}
    for name, cfg in scene.fixed_env_objects.items():
        st = resolved.fixed_env_objects[name]
        half = list(getattr(cfg, "half_extents", []) or [])
        fixed[name] = {
            "kind": type(cfg).__name__,
            "half_extents": _floats(half),
            "q_xyzw": _floats(st.q[:7]),
        }
    robots = {
        rid: {"start_q": _floats(st.q), "start_v": _floats(st.v),
              "joint_names": list(getattr(scene.robots[rid], "joint_names", []) or [])}
        for rid, st in resolved.robots.items()
    }
    return {"objects": objects, "fixed_objects": fixed, "robots": robots,
            "friction_mu": FRICTION_MU, "gravity": 9.81}


def goal_record(goal: Any) -> Optional[Dict[str, Any]]:
    if goal is None:
        return None
    def target(t: Any) -> Dict[str, Any]:
        kso_tol = getattr(getattr(t, "kso_tolerance", None), "position", None)
        to_tol = getattr(getattr(t, "to_tolerance", None), "position", None)
        return {"type": type(t).__name__, "value": _floats(t.value),
                "kso_position_tol": kso_tol, "to_position_tol": to_tol}
    return {"kso": goal.kso, "to": goal.to,
            "objects": {k: target(v) for k, v in goal.objects.items()},
            "robots": {k: target(v) for k, v in goal.robots.items()}}


def classify(converged: Optional[bool], max_violation: Optional[float],
             near_feasible: float, physically_valid: Optional[bool] = None) -> Optional[str]:
    """``converged`` / ``physical`` (not converged, but trajectory_check
    passes) / ``near_feasible`` / None (not exported). The solver-residual
    ``near_feasible`` rule only applies when no physical check ran
    (``physically_valid`` is None): that residual includes IPM slack."""
    if converged:
        return "converged"
    if physically_valid is not None:
        return "physical" if physically_valid else None
    if max_violation is not None and float(max_violation) <= near_feasible:
        return "near_feasible"
    return None


def git_versions(repo_root: str) -> Dict[str, Optional[str]]:
    def rev(path: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                                 text=True, timeout=10, check=True)
            dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"], capture_output=True,
                                   text=True, timeout=10).stdout.strip()
            return out.stdout.strip() + ("+dirty" if dirty else "")
        except Exception:  # noqa: BLE001
            return None
    return {"motiondisco": rev(repo_root),
            "faro": rev(os.path.join(repo_root, "submodules", "faro"))}


# ---------------------------------------------------------------------------
# Writing one sample
# ---------------------------------------------------------------------------


def _solver_arrays(results: Dict[str, Any], robot_id: str) -> Dict[str, np.ndarray]:
    step_ts = np.asarray(results["step_ts"], dtype=float).reshape(-1)
    out = {
        "solver_time": np.r_[0.0, np.cumsum(step_ts)],
        "solver_t_scale": np.asarray(results.get("t_scale", []), dtype=float).reshape(-1),
    }
    robot_xs = results["robot_xs"]
    out["solver_robot_xs"] = np.asarray(robot_xs[robot_id] if isinstance(robot_xs, dict) else robot_xs)
    torques = results.get("torques")
    if torques is not None:
        out["solver_torques"] = np.asarray(torques[robot_id] if isinstance(torques, dict) else torques)
    for patch, w in ((results.get("rob_Ws") or {}).get(robot_id) or {}).items():
        out[f"solver_wrench_{patch}"] = np.asarray(w, dtype=float)
    for name, xs in (results.get("obj_xs") or {}).items():
        out[f"solver_obj_xs_{name}"] = np.asarray(xs, dtype=float)
    for name, w in (results.get("obj_Ws") or {}).items():
        out[f"solver_obj_wrench_{name}"] = np.asarray(w, dtype=float)
    return out


def _atomic_npz(path: str, arrays: Dict[str, Any]) -> None:
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def export_sample(
    sample_dir: str,
    scene: Any,
    resolved: Any,
    faro_plan: Dict[Any, List[Tuple]],
    results: Dict[str, Any],
    params: Dict[str, Any],
    *,
    robot_id: str,
    num_modes: int,
    nodes_per_mode: int,
    frequency: float = 50.0,
    to_mj: Optional[Callable[..., Dict[str, Any]]] = None,
    scene_xml: Optional[Callable[..., None]] = None,
) -> Dict[str, str]:
    """Write traj.npz, scene.xml and params.json; returns their paths."""
    if to_mj is None or scene_xml is None:
        from faro.utils.results import pin_to_mj, save_scene_xml

        to_mj = to_mj or pin_to_mj
        scene_xml = scene_xml or save_scene_xml
    os.makedirs(sample_dir, exist_ok=True)
    solver = _solver_arrays(results, robot_id)
    traj = dict(to_mj(scene, results, solver["solver_time"], faro_plan, num_modes,
                      nodes_per_mode, frequency=frequency))
    traj.update(solver)
    traj["params_json"] = np.array(json.dumps(params))
    paths = {
        "npz": os.path.join(sample_dir, "traj.npz"),
        "scene_xml": os.path.join(sample_dir, "scene.xml"),
        "params": os.path.join(sample_dir, "params.json"),
    }
    _atomic_npz(paths["npz"], traj)
    scene_xml(scene, resolved, paths["scene_xml"])
    _atomic_json(paths["params"], params)
    return paths


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def append_manifest(out: str, row: Dict[str, Any]) -> None:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, MANIFEST)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def manifest_rows(out: str) -> List[Dict[str, Any]]:
    path = os.path.join(out, MANIFEST)
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def sample_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (row["run_id"], row["plan_hash"], row["sample_id"])


def done_keys(out: str) -> set:
    """Samples already attempted: their latest manifest row has a verdict.
    A row that ended in an exception (``error``) is not one, so a rerun
    retries it; the retry's row then wins (last row per key).
    """
    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in manifest_rows(out):
        latest[sample_key(r)] = r
    return {k for k, r in latest.items() if not r.get("error")}


SUMMARY_FIELDS = ["scene", "initial_condition", "attempted", "exported", "converged",
                  "physical", "near_feasible", "kso_infeasible", "failed", "errors"]


def write_summary(out: str) -> List[Dict[str, Any]]:
    """Counts per (scene, IC) from the manifest (last row per key wins)."""
    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in manifest_rows(out):
        latest[sample_key(r)] = r
    table: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in latest.values():
        k = (r.get("scene", ""), r.get("initial_condition", ""))
        t = table.setdefault(k, {"scene": k[0], "initial_condition": k[1], **{
            f: 0 for f in SUMMARY_FIELDS[2:]}})
        t["attempted"] += 1
        q = r.get("quality")
        if q:
            t["exported"] += 1
            t[q] += 1
        if r.get("skipped", "").startswith("KSO infeasible"):
            t["kso_infeasible"] += 1
        elif r.get("error"):
            t["errors"] += 1
        elif not q:
            t["failed"] += 1
    rows = [table[k] for k in sorted(table)]
    with open(os.path.join(out, SUMMARY), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows
