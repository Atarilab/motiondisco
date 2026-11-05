"""Physical quality of a TO trajectory, independent of solver convergence.

TO "converged" means moto's primal residual fell below 0.05. That residual
is a slack-inflated, mixed-unit number (inequality rows include the IPM
slack), so trajectories that are physically fine can count as failed and
the reported "violations" can be solver artefacts. This module judges the
saved trajectory itself, in physical units:

    dynamics     centroidal momentum change vs gravity + contact wrenches
                 (N, as a fraction of the robot's weight)
    actuation    power limit |tau| + tau_max/v_max |v| <= tau_max, joint
                 position / velocity limits
    contacts     every planned robot contact: normal force >= 0, friction
                 within mu (pyramid, as faro), distance to its face, slip
                 of sticking contacts within a mode
    penetration  MuJoCo replay (the tracker's simulator): deepest
                 interpenetration robot-environment, robot-object or of a
                 movable object (robot self-contact is not counted)

    check(task, plan, results) -> report dict (worst values + pass flags)
    python trajectory_check.py <run_dir> [...]   tables for stored runs

Wrenches are faro's ``rob_Ws``: the force / moment on the robot at the
patch origin in world axes, for interval k at the interval's end pose
(node k+1), which is also where faro evaluates its contact constraints.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MU = 0.5            # faro's friction coefficient (to.py, all contacts)
GRAVITY = 9.81
ROBOT_PATCHES = ("left_sole", "right_sole", "left_ee", "right_ee")

# Pass thresholds, calibrated on stored runs: all 68 converged
# long_dist_pick_place plans pass, and 11 of its 17 non-converged ones do.
# A trajectory passes when every worst value is within them.
THRESHOLDS = {
    "dyn_force_ratio": 0.05,     # |dh/dt - sum f - m g| / (m g)
    "power_excess_ratio": 0.05,  # (|tau| + slope |v| - tau_max) / tau_max
    "joint_pos_excess": 0.02,    # rad beyond the limit
    "joint_vel_excess_ratio": 0.05,
    "pull_force": 5.0,           # N of negative normal force
    "friction_excess_ratio": 0.10,   # (|f_t|_inf - mu f_n) / max(f_n, m g / 20)
    "contact_gap": 0.02,         # m between a planned patch and its face
    "contact_slip": 0.02,        # m of sticking-patch drift within a mode
    "penetration": 0.03,         # m; converged long_dist plans reach 2.6 cm
                                 # (MuJoCo collision shapes vs faro's patches)
}

# Face name -> (outward axis index, sign, in-face axis indices)
FACE_AXES = {
    "top": (2, 1.0, (0, 1)), "bottom": (2, -1.0, (0, 1)),
    "front": (0, 1.0, (1, 2)), "back": (0, -1.0, (1, 2)),
    "left": (1, 1.0, (0, 2)), "right": (1, -1.0, (0, 2)),
}


# ---------------------------------------------------------------------------
# Pure metrics (unit-tested without faro / pinocchio / MuJoCo)
# ---------------------------------------------------------------------------

def power_excess(tau: np.ndarray, v: np.ndarray, tau_max: np.ndarray, vel_max: np.ndarray) -> np.ndarray:
    """Per-node, per-joint excess over faro's power limit (N·m, >= 0)."""
    return np.maximum(np.abs(tau) + (tau_max / vel_max) * np.abs(v) - tau_max, 0.0)


def contact_force_metrics(force: np.ndarray, normal: np.ndarray, t1: np.ndarray, t2: np.ndarray,
                          mu: float = MU) -> Tuple[float, float]:
    """(pull, friction excess) of a force on the robot against a face with
    outward ``normal`` and in-face axes t1, t2: pull = max(-f_n, 0);
    friction excess = max(|f_t1|, |f_t2|) - mu f_n (pyramid, as faro)."""
    fn = float(force @ normal)
    ft = max(abs(float(force @ t1)), abs(float(force @ t2)))
    return max(-fn, 0.0), max(ft - mu * max(fn, 0.0), 0.0)


def face_geometry(center: np.ndarray, rot: np.ndarray, half: Sequence[float], face: str
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(face centre, outward normal, in-face axis 1, axis 2) of a box face;
    a plane's single 'surface' face is its +z."""
    if face == "surface":
        return center.copy(), rot[:, 2].copy(), rot[:, 0].copy(), rot[:, 1].copy()
    axis, sign, (a, b) = FACE_AXES[face]
    normal = sign * rot[:, axis]
    return center + normal * float(half[axis]), normal, rot[:, a].copy(), rot[:, b].copy()


def mode_of_node(k: int, nodes_per_mode: int, num_modes: int) -> int:
    return min(k // nodes_per_mode, num_modes - 1)


def normalized(worst: Mapping[str, float], thresholds: Mapping[str, float] = THRESHOLDS) -> float:
    """Worst metric relative to its threshold (<= 1 passes)."""
    return max((float(worst.get(k, 0.0)) / lim for k, lim in thresholds.items()), default=0.0)


def describe(report: Mapping[str, Any], max_items: int = 4) -> List[str]:
    """Plain-language lines for the failed checks, worst (relative to its
    threshold) first."""
    worst, where = report["worst"], report.get("where", {})
    npm = int(report.get("nodes_per_mode", 20))
    weight = float(report.get("mass", 35.0)) * GRAVITY

    def mode(w: Sequence[Any]) -> str:
        if not w:
            return "somewhere"
        return f"mode {int(w[1]) // npm if w[0] == 'node' else int(w[1])}"

    lines = []
    for key in sorted((k for k in report.get("fails", [])), key=lambda k: -worst[k] / THRESHOLDS[k]):
        v, w = worst[key], list(where.get(key) or [])
        if key == "penetration":
            a, b = (w[2], w[3]) if len(w) >= 4 else ("a body", "another")
            lines.append(f"{mode(w)}: {a} penetrates {b} by {100 * v:.0f} cm in the trajectory")
        elif key == "contact_gap":
            lines.append(f"{mode(w)}: {w[2]} stays {100 * v:.0f} cm off {w[3]} {w[4]}; the planned contact is not made")
        elif key == "contact_slip":
            lines.append(f"{mode(w)}: {w[2]} slides {100 * v:.0f} cm on {w[3]} {w[4]} although the contact is sticking")
        elif key == "dyn_force_ratio":
            lines.append(f"{mode(w)}: the motion is not dynamically consistent (it needs {100 * v:.0f}% of the "
                         f"body weight, {v * weight:.0f} N, that no contact provides)")
        elif key == "power_excess_ratio":
            lines.append(f"{mode(w)}: {w[2]} exceeds its torque-speed limit by {100 * v:.0f}%")
        elif key == "joint_pos_excess":
            lines.append(f"{mode(w)}: {w[2] if len(w) > 2 else 'a joint'} goes {v:.2f} rad past its limit")
        elif key == "joint_vel_excess_ratio":
            lines.append(f"{mode(w)}: a joint exceeds its speed limit by {100 * v:.0f}%")
        elif key == "friction_excess_ratio":
            lines.append(f"{mode(w)}: {w[2]} on {w[3]} {w[4]} needs more friction than available "
                         f"({100 * v:.0f}% beyond the cone)")
        elif key == "pull_force":
            lines.append(f"{mode(w)}: {w[2]} pulls on {w[3]} {w[4]} ({v:.0f} N) instead of pushing")
    return lines[:max_items]


def verdict(worst: Mapping[str, float], thresholds: Mapping[str, float] = THRESHOLDS) -> Dict[str, Any]:
    fails = [k for k, lim in thresholds.items() if worst.get(k, 0.0) > lim]
    return {"pass": not fails, "fails": fails}


# ---------------------------------------------------------------------------
# Trajectory check (faro task, pinocchio, MuJoCo)
# ---------------------------------------------------------------------------

def _geom_for(mj_model: Any, name: str) -> int:
    import mujoco
    for cand in (name, f"{name}-{name}_geom"):
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, cand)
        if gid >= 0:
            return gid
    raise KeyError(f"no MuJoCo geom for object {name!r}")


def check(task: Any, plan: Mapping[str, Sequence[Any]], results: Mapping[str, Any],
          nodes_per_mode: int = 20) -> Dict[str, Any]:
    """Worst physical violations of a TO trajectory (see module doc)."""
    import mujoco
    import pinocchio as pin

    from _replay_common import build_qpos_trajectory, build_replay_model

    rid = task.robot_id
    cfg = task.scene.robots[rid]
    model = cfg.pin_model
    pdata = model.createData()
    nq, nv = model.nq, model.nv
    nj = len(cfg.joint_names)
    xs = np.asarray(results["robot_xs"][rid] if isinstance(results["robot_xs"], dict) else results["robot_xs"], float)
    tau = np.asarray(results["torques"][rid] if isinstance(results["torques"], dict) else results["torques"], float)
    W = {ee: np.asarray(w, float) for ee, w in (results.get("rob_Ws") or {}).get(rid, {}).items()}
    dts = np.asarray(results["step_ts"], float).reshape(-1)
    n_int = len(tau)
    num_modes = len(plan["left_sole"])
    mass = float(pin.computeTotalMass(model))
    weight = mass * GRAVITY

    faces = {fid: (obj, face) for (obj, face), fid in task.scene.faces.by_name.items()}
    mj_model = build_replay_model(task)
    mj_data = mujoco.MjData(mj_model)
    qpos = build_qpos_trajectory(results, task, mj_model)
    movable = set(task.scene.env_objects)
    robot_geoms = {g for g in range(mj_model.ngeom) if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) is None
                   or str(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)).startswith("n_")}
    movable_geoms = {_geom_for(mj_model, o) for o in movable}

    def geom_label(g: int) -> str:
        if g in robot_geoms:
            body = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, mj_model.geom_bodyid[g]) or "robot"
            return "robot " + body.split("__")[-1]
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "?"
        return name.split("-")[0]

    worst = {k: 0.0 for k in THRESHOLDS}
    where: Dict[str, Any] = {}
    gravity = np.array([0.0, 0.0, -weight])
    patch_start: Dict[str, np.ndarray] = {}

    def bump(key: str, value: float, info: Any) -> None:
        if value > worst[key]:
            worst[key], where[key] = float(value), info

    h_prev = None
    for k in range(n_int + 1):
        q = xs[k][:nq]
        v = xs[k][nq:nq + nv]
        pin.computeCentroidalMomentum(model, pdata, q, v)
        h = np.asarray(pdata.hg.np, float).copy()
        com = pdata.com[0].copy()
        pin.framesForwardKinematics(model, pdata, q)

        # joint limits at every node
        qj, vj = q[7:7 + nj], v[6:6 + nj]
        jp = np.maximum(cfg.q_min - qj, 0) + np.maximum(qj - cfg.q_max, 0)
        bump("joint_pos_excess", float(np.max(jp)), ("node", k, cfg.joint_names[int(np.argmax(jp))]))
        bump("joint_vel_excess_ratio", float(np.max(np.maximum(np.abs(vj) - cfg.vel_max, 0) / cfg.vel_max)), ("node", k))

        # MuJoCo penetration at this node
        mj_data.qpos[:] = qpos[min(k, len(qpos) - 1)]
        mujoco.mj_forward(mj_model, mj_data)
        for c in mj_data.contact[:mj_data.ncon]:
            g1r, g2r = c.geom1 in robot_geoms, c.geom2 in robot_geoms
            # robot self-contact (G1 meshes of neighbouring links) is not counted
            if c.dist < 0 and not (g1r and g2r) and (g1r or g2r
                                                     or c.geom1 in movable_geoms or c.geom2 in movable_geoms):
                bump("penetration", -float(c.dist), ("node", k, geom_label(c.geom1), geom_label(c.geom2)))

        if k == 0:
            h_prev = h
            continue
        i = k - 1                      # interval i ends at node k
        m = mode_of_node(i, nodes_per_mode, num_modes)

        # actuation (interval torque, end-node velocity)
        pe = power_excess(tau[i], vj, cfg.tau_max, cfg.vel_max)
        j = int(np.argmax(pe / cfg.tau_max))
        bump("power_excess_ratio", float(pe[j] / cfg.tau_max[j]), ("mode", m, cfg.joint_names[j], round(float(pe[j]), 2)))

        # centroidal dynamics: (h_k - h_{k-1}) / dt = sum f + m g (linear part)
        f_sum = gravity.copy()
        for ee, w in W.items():
            f_sum += w[i, :3]
        dh = (h[:3] - h_prev[:3]) / max(dts[i], 1e-9)
        bump("dyn_force_ratio", float(np.linalg.norm(dh - f_sum)) / weight, ("mode", m, "interval", i))
        h_prev = h

        # planned robot contacts
        for ee in ROBOT_PATCHES:
            c = plan.get(ee, [()] * num_modes)[m]
            if not c or ee not in W:
                patch_start.pop(ee, None)
                continue
            obj, face = faces[int(c[0])]
            gid = _geom_for(mj_model, obj)
            centre, normal, t1, t2 = face_geometry(mj_data.geom_xpos[gid].copy(), mj_data.geom_xmat[gid].reshape(3, 3).copy(),
                                                   mj_model.geom_size[gid], face)
            p = pdata.oMf[model.getFrameId(ee)].translation.copy()
            # faro's pyramid is aligned with the patch axes, not the face's
            rot = pdata.oMf[model.getFrameId(ee)].rotation
            pull, fric = contact_force_metrics(W[ee][i, :3], normal, rot[:, 0], rot[:, 1])
            fn = max(float(W[ee][i, :3] @ normal), weight / 20.0)
            bump("pull_force", pull, ("mode", m, ee, obj, face))
            bump("friction_excess_ratio", fric / fn, ("mode", m, ee, obj, face, round(fric, 2)))
            bump("contact_gap", abs(float((p - centre) @ normal)), ("mode", m, ee, obj, face))
            sticking = int(c[-1]) == 1
            key = f"{ee}:{m}"
            if sticking and obj not in movable:
                start = patch_start.setdefault(key, p)
                drift = p - start
                bump("contact_slip", float(np.linalg.norm(drift - (drift @ normal) * normal)), ("mode", m, ee, obj, face))
    report = {"worst": worst, "where": {k: list(v) for k, v in where.items()}, "mass": mass,
              "nodes": n_int + 1, "nodes_per_mode": nodes_per_mode}
    report.update(verdict(worst))
    return report


# ---------------------------------------------------------------------------
# CLI over stored runs
# ---------------------------------------------------------------------------

def _load(run_gen: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    try:
        extra = json.load(open(os.path.join(run_gen, "results", "extra.json")))
        data = pickle.load(open(os.path.join(run_gen, "results", "to_run.pkl"), "rb"))
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        return None
    return extra, data, data.get("results", data)


def main(argv: List[str]) -> None:
    from _task_loader import canonical_scene_name, load_task

    for run in argv:
        cfg_path = os.path.join(run, ".hydra", "config.yaml")
        scene = os.environ.get("SHINKA_LOCOMANIP_SCENE")
        if os.path.exists(cfg_path):
            import yaml
            scene = yaml.safe_load(open(cfg_path)).get("task", {}).get("scene", scene)
        task = load_task(canonical_scene_name(scene), "default", "")
        print(f"== {run} ({task.scene_name})")
        for g in sorted(glob.glob(os.path.join(run, "gen_*")), key=lambda x: int(x.rsplit("_", 1)[1])):
            loaded = _load(g)
            if loaded is None:
                continue
            extra, data, res = loaded
            plan = {k: [tuple(x) if x else () for x in v] for k, v in extra["contact_mode_seq"].items()}
            variants = [("final", res)]
            if res.get("best_iterate"):
                b = dict(res); b.update(res["best_iterate"])
                variants.append(("best", b))
            for label, r in variants:
                try:
                    rep = check(task, plan, r)
                except Exception as e:  # noqa: BLE001 - report and continue
                    print(f"{os.path.basename(g):7s} {label:5s} error: {type(e).__name__}: {e}")
                    continue
                w = rep["worst"]
                print(f"{os.path.basename(g):7s} {label:5s} conv={str(extra.get('to_converged')):5s} "
                      f"PASS={str(rep['pass']):5s} " + " ".join(f"{k}={w[k]:.3g}" for k in THRESHOLDS)
                      + (f"  fails={rep['fails']}" if rep["fails"] else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
