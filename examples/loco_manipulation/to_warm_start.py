"""TO initial guess from a KSO solution.

faro starts every TO node at the initial standing pose with zero
velocities and zero contact forces (``ToTask.reset_initial_state``) and
never uses the KSO result. For plans that travel (carry a box off a
truck, climb a table) the solver then has to discover the whole motion
from a robot standing still at the start.

The KSO solution already holds one kinematically feasible configuration
per mode boundary. This module spreads it over the TO nodes:

* configurations: interpolated on the configuration manifold
  (``pin.interpolate``) between the KSO keyframes, in time, using the
  node durations TO itself uses (``ToTask.dt_traj``, which shortens
  contact-switch intervals);
* velocities / accelerations: finite differences of that path;
* centre of mass and centroidal momentum: computed from (q, v), as faro
  does for the initial state;
* contact forces (world frame, the force on the robot, as faro stores
  them): the soles push up with the robot's weight plus every carried
  load (a carried object and anything stacked on it), shared over the
  planted soles (over the hands when no sole is planted); a hand holding
  a carried object is pulled down by its share of that load and pushed
  out of the gripped face by a squeeze large enough for friction
  (``MU``) to hold it;
* objects: KSO object poses interpolated the same way;
* ``t_scale`` = 1.

KSO states sit at mode boundaries: state 0 is the initial pose (start of
mode 0), state j the transition from mode j-1 to mode j (TO node
``j * nodes_per_mode``), and, with a terminal goal, a last state at the
end of the horizon. Keyframes beyond the TO horizon are dropped; a
missing terminal keyframe holds the last one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

GRAVITY = 9.81
MU = 0.5            # faro's friction coefficient (constant in its TO)
SQUEEZE_MARGIN = 1.3


def keyframe_nodes(num_states: int, num_modes: int, nodes_per_mode: int) -> List[int]:
    """TO node index of each KSO state (capped at the horizon end)."""
    n = num_modes * nodes_per_mode
    return [min(j * nodes_per_mode, n) for j in range(num_states)]


def node_times(dt_traj: Sequence[float]) -> np.ndarray:
    """Time of each of the N+1 nodes for interval durations *dt_traj* (N)."""
    dt = np.asarray(dt_traj, dtype=float).reshape(-1)
    return np.r_[0.0, np.cumsum(dt)]


def interpolation_weights(times: np.ndarray, key_nodes: Sequence[int]) -> List[Tuple[int, int, float]]:
    """For every node: (keyframe a, keyframe b, fraction in time). Nodes
    at or after the last keyframe hold it."""
    key_nodes = list(key_nodes)
    out: List[Tuple[int, int, float]] = []
    for k in range(len(times)):
        j = max(i for i, node in enumerate(key_nodes) if node <= k)
        if j == len(key_nodes) - 1:
            out.append((j, j, 0.0))
            continue
        t0, t1 = times[key_nodes[j]], times[key_nodes[j + 1]]
        frac = 0.0 if t1 <= t0 else float(np.clip((times[k] - t0) / (t1 - t0), 0.0, 1.0))
        out.append((j, j + 1, frac))
    return out


def node_modes(num_nodes: int, num_modes: int, nodes_per_mode: int) -> np.ndarray:
    """Mode index of each of the *num_nodes* intervals."""
    return np.minimum(np.arange(num_nodes) // max(nodes_per_mode, 1), num_modes - 1)


def _active(plan: Mapping[str, Sequence[Tuple]], ch: str, mode: int) -> bool:
    seq = plan.get(ch)
    return bool(seq) and mode < len(seq) and bool(tuple(seq[mode]))


def carried_loads(
    plan: Mapping[str, Sequence[Tuple]], mode: int, object_masses: Mapping[str, float],
    face_owner: Mapping[int, str],
) -> Dict[str, float]:
    """Mass each carried object (object channel free) bears in *mode*:
    itself plus everything resting on it, directly or stacked."""
    parent: Dict[str, str] = {}
    for o in object_masses:
        if _active(plan, o, mode):
            entry = tuple(plan[o][mode])
            owner = face_owner.get(int(entry[1])) if len(entry) == 3 else None
            if owner in object_masses and owner != o:
                parent[o] = owner
    load = {o: float(m) for o, m in object_masses.items()}
    for o, m in object_masses.items():
        p, seen = parent.get(o), {o}
        while p is not None and p not in seen:
            load[p] += float(m)
            seen.add(p)
            p = parent.get(p)
    return {o: load[o] for o in object_masses if o in plan and not _active(plan, o, mode)}


def contact_forces(
    plan: Mapping[str, Sequence[Tuple]], mode: int, robot_mass: float,
    object_masses: Mapping[str, float], face_owner: Mapping[int, str],
    face_normal: Optional[Any] = None,
    sole_channels: Sequence[str] = ("left_sole", "right_sole"),
    hand_channels: Sequence[str] = ("left_ee", "right_ee"),
) -> Dict[str, np.ndarray]:
    """Force on the robot (world frame, N) per contact channel in *mode*.
    ``face_normal(face_id)`` gives a face's outward normal in the world
    (for the squeeze); without it hands get no squeeze."""
    z = np.array([0.0, 0.0, 1.0])
    loads = carried_loads(plan, mode, object_masses, face_owner)
    forces: Dict[str, np.ndarray] = {}
    for name, mass in loads.items():
        holders = [h for h in hand_channels if _active(plan, h, mode)
                   and face_owner.get(int(tuple(plan[h][mode])[0])) == name]
        for h in holders:
            share = GRAVITY * mass / len(holders)
            f = -share * z
            n = face_normal(int(tuple(plan[h][mode])[0])) if face_normal is not None else None
            if n is not None and abs(float(n[2])) < 0.7:  # side face: squeeze it
                # Horizontally: a carried box is often tilted in the KSO
                # poses, which would otherwise tip the squeeze up or down.
                nh = np.array([float(n[0]), float(n[1]), 0.0])
                f = f + (SQUEEZE_MARGIN * share / MU) * nh / max(np.linalg.norm(nh), 1e-9)
            forces[h] = forces.get(h, np.zeros(3)) + f
    total = GRAVITY * (robot_mass + sum(loads.values()))
    supports = [c for c in sole_channels if _active(plan, c, mode)] or \
        [h for h in hand_channels if _active(plan, h, mode) and h not in forces]
    for c in supports:
        forces[c] = forces.get(c, np.zeros(3)) + (total / len(supports)) * z
    return forces


def build_guess(
    to_task: Any, kso_results: Mapping[str, Any], plan: Mapping[str, Sequence[Tuple]],
    num_modes: int, nodes_per_mode: int, with_velocities: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(x0_dict, u0_dict) for ``moto_wrap.set_warm_start``. With
    ``with_velocities=False`` velocities, momenta and accelerations stay
    zero (faro's choice for its own guess) and only poses and contact
    forces come from the plan."""
    import pinocchio as pin

    scene = to_task.scene
    n = num_modes * nodes_per_mode
    times = node_times(to_task.dt_traj[:n])
    dts = np.diff(times)
    x0: Dict[str, Dict[str, np.ndarray]] = {"robots": {}, "objects": {}}
    u0: Dict[str, Any] = {"robots": {}, "objects": {}, "t_scale": np.ones(n)}
    modes = node_modes(n, num_modes, nodes_per_mode)
    face_owner = {fid: name for fid, (name, _local) in scene.faces.by_id.items()}
    object_masses = {name: float(cfg.mass) for name, cfg in scene.env_objects.items()}
    robot_parts: Dict[str, Tuple[float, np.ndarray, Any]] = {}

    for rid, cfg in scene.robots.items():
        model = cfg.pin_model
        data = model.createData()
        keys = np.asarray(kso_results["robot_xs"][rid], dtype=float)
        w = interpolation_weights(times, keyframe_nodes(len(keys), num_modes, nodes_per_mode))
        qs = np.stack([pin.interpolate(model, keys[a], keys[b], f) for a, b, f in w])
        vs = np.zeros((n + 1, model.nv))
        if with_velocities:
            for k in range(n):
                vs[k] = pin.difference(model, qs[k], qs[k + 1]) / max(dts[k], 1e-6)
        acc = np.diff(vs, axis=0) / np.maximum(dts, 1e-6)[:, None]
        coms = np.stack([np.asarray(pin.centerOfMass(model, data, q)) for q in qs])
        moms = np.stack([np.asarray(pin.computeCentroidalMomentum(model, data, q, v).vector)
                         for q, v in zip(qs, vs)])
        x0["robots"][rid] = np.hstack((qs, vs, coms, moms))
        mass = float(pin.computeTotalMass(model))
        robot_parts[rid] = (mass, acc, cfg)

    obj_qs: Dict[str, np.ndarray] = {}
    for name in scene.env_objects:
        keys = np.asarray(kso_results["obj_xs"][name], dtype=float)
        w = interpolation_weights(times, keyframe_nodes(len(keys), num_modes, nodes_per_mode))
        qs = np.stack([_interp_pose(keys[a], keys[b], f) for a, b, f in w])
        obj_qs[name] = qs
        vs = np.zeros((n + 1, 6))
        for k in range(n if with_velocities else 0):
            vs[k] = pin.log6(pin.XYZQUATToSE3(qs[k]).actInv(pin.XYZQUATToSE3(qs[k + 1]))).vector \
                / max(dts[k], 1e-6)
        x0["objects"][name] = np.hstack((qs, vs))
        u0["objects"][name] = np.zeros((n, 12))

    local_normals = _local_face_normals(scene)
    for rid, (mass, acc, cfg) in robot_parts.items():
        wrenches = {patch: np.zeros((n, 6)) for patch in cfg.patch_ees}
        for k in range(n):
            def normal(face_id: int, k: int = k) -> Optional[np.ndarray]:
                name = face_owner.get(face_id)
                if name not in obj_qs or face_id not in local_normals:
                    return local_normals.get(face_id)
                return pin.XYZQUATToSE3(obj_qs[name][k]).rotation @ local_normals[face_id]
            forces = contact_forces(plan, int(modes[k]), mass, object_masses, face_owner, normal)
            for patch, f in forces.items():
                if patch in wrenches:
                    wrenches[patch][k, :3] = f
        u0["robots"][rid] = {"acceleration": acc, "wrenches": wrenches}
    return x0, u0


def _local_face_normals(scene: Any) -> Dict[int, np.ndarray]:
    """Outward normal of every face in its object's frame (the z axis of
    faro's face feature frame)."""
    out: Dict[int, np.ndarray] = {}
    for face_id, (name, local_id) in scene.faces.by_id.items():
        try:
            feature = scene.objects[name].features[:, local_id - 1]
            out[face_id] = np.asarray(feature[3:12], dtype=float).reshape(3, 3, order="F")[:, 2]
        except (AttributeError, IndexError, KeyError, ValueError):
            continue
    return out


def _interp_pose(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    import pinocchio as pin

    A, B = pin.XYZQUATToSE3(np.asarray(a, dtype=float)), pin.XYZQUATToSE3(np.asarray(b, dtype=float))
    return np.asarray(pin.SE3ToXYZQUAT(A * pin.exp6(f * pin.log6(A.actInv(B)))), dtype=float)


def apply(to_task: Any, kso_results: Mapping[str, Any], plan: Mapping[str, Sequence[Tuple]],
          num_modes: int, nodes_per_mode: int, with_velocities: bool = True) -> None:
    """Install the guess on both of the task's SQPs (the collision-free
    presolve and the collision SQP). Call after ``update_mode_oh_via_tup``
    (which sets ``dt_traj``)."""
    x0, u0 = build_guess(to_task, kso_results, plan, num_modes, nodes_per_mode,
                         with_velocities=with_velocities)
    if to_task.sqp is not None:  # no presolve SQP when SHINKA_TO_PRESOLVE=0
        to_task.moto_wrap.set_warm_start(to_task.sqp, x0, u0)
    to_task.moto_wrap.set_warm_start(to_task.sqp_coll, x0, u0)
