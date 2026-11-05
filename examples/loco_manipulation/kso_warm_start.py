"""Level-aware KSO warm start.

faro initialises every KSO state with the scene's initial standing pose
and resolves the collision pairs (witness points, normals) at that pose
for every state. For a plan that changes support level — climbing onto a
table and down its far side — both are far from the solution: states on
and beside the table start with the robot metres away, and the
linearised collision constraints describe the table as seen from the
start. The SQP then settles in a basin where the lower leg stays straight
and the stepping foot ends ~0.2 m short of the floor, or the robot walks
back down the near side and misses the goal.

This module seeds such plans instead: the pelvis walks in xy from the
start, over the edge of each upper face, to the goal; it stands at
standing height over its support level; at a straddle state (soles on
faces >= MIN_STEP apart) it sits at the straddle height with the upper
leg folded; and the heading turns to face the upper surface (from the
start when climbing, from the goal when descending — the posture the G1
folds into most easily). The goal is the robot's terminal target, else
the centre of the goal objects' targets (``_goal_xy``). The collision
pairs are then re-resolved at the seed. Plans without a level change are left untouched.

``SHINKA_KSO_WARM_START=0`` disables it.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Below this level difference the default guess solves fine.
MIN_STEP = 0.3
# Pelvis above the lower sole: just under the straight-leg maximum.
MAX_LEG_EXTENSION = 0.80
# Pelvis above the upper sole when the upper leg is folded.
MIN_LEG_FOLD = 0.12
# Folded-leg joints (hip pitch, knee, ankle pitch): deep flexion with the
# foot slightly ahead of the pelvis, inside the G1 limits.
FOLDED_LEG = (-2.3, 2.6, -0.3)
STRAIGHT_LEG = (0.0, 0.0, 0.0)

BOX_FACE_NORMAL_Z = {1: 1.0, 6: -1.0}  # top / bottom; side faces at centre height


def enabled() -> bool:
    return os.environ.get("SHINKA_KSO_WARM_START", "1") not in ("0", "false", "no")


def face_centres(scene: Any, resolved: Any) -> Dict[int, np.ndarray]:
    """World centre of every face, keyed by global face ID (movable objects
    at their initial pose; boxes taken as axis-aligned)."""
    out: Dict[int, np.ndarray] = {}
    for env_id, (name, local_id) in scene.faces.by_id.items():
        state = (getattr(resolved, "fixed_env_objects", {}) or {}).get(name)
        if state is None:
            state = (getattr(resolved, "env_objects", {}) or {}).get(name)
        if state is None:
            continue
        centre = np.asarray(state.q, dtype=float)[:3].copy()
        cfg = scene.objects.get(name) if hasattr(scene, "objects") else None
        half = getattr(cfg, "half_extents", None)
        if half is not None and len(half) >= 3:
            centre[2] += BOX_FACE_NORMAL_Z.get(int(local_id), 0.0) * float(half[2])
        out[env_id] = centre
    return out


def state_modes(num_modes: int, hard_initial: bool, terminal: bool) -> List[Tuple[int, ...]]:
    """Mode indices whose contacts hold at each solver state, in stage
    order: the initial state (mode 0), one state per transition (union of
    the modes on either side), the terminal state (last mode)."""
    states: List[Tuple[int, ...]] = [(0,)] if hard_initial else []
    states += [(k, k + 1) for k in range(num_modes - 1)]
    if terminal:
        states.append((num_modes - 1,))
    return states


def sole_faces(cms: Mapping[tuple, List[Tuple]], modes: Tuple[int, ...]) -> Dict[str, int]:
    """``{sole patch: face ID}`` for every sole in contact in any of *modes*."""
    faces: Dict[str, int] = {}
    for key, seq in cms.items():
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == "ee"):
            continue
        patch = str(key[2])
        if not patch.endswith("sole"):
            continue
        for m in modes:
            entry = tuple(seq[m]) if m < len(seq) else ()
            if entry:
                faces[patch] = int(entry[0])
    return faces


def _yaw(quat_xyzw: Sequence[float]) -> float:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heading(src_xy: np.ndarray, dst_xy: np.ndarray) -> Optional[float]:
    d = np.asarray(dst_xy, dtype=float)[:2] - np.asarray(src_xy, dtype=float)[:2]
    if np.linalg.norm(d) < 1e-6:
        return None
    return math.atan2(d[1], d[0])


class StateGuess:
    """Per-state guess: pelvis xy / z (world), base yaw (world), leg joints."""

    __slots__ = ("pelvis_z", "yaw", "legs", "xy")

    def __init__(self, pelvis_z: Optional[float] = None, yaw: Optional[float] = None,
                 legs: Optional[Dict[str, Tuple[float, float, float]]] = None,
                 xy: Optional[np.ndarray] = None):
        self.pelvis_z, self.yaw, self.legs, self.xy = pelvis_z, yaw, legs or {}, xy


def face_edge_toward(centre: np.ndarray, half: Optional[Sequence[float]], src_xy: np.ndarray) -> np.ndarray:
    """The point on the (axis-aligned) face's boundary toward *src_xy*, or
    the centre when the face has no extent."""
    c = np.asarray(centre, dtype=float)[:2]
    d = np.asarray(src_xy, dtype=float)[:2] - c
    n = np.linalg.norm(d)
    if n < 1e-6 or half is None or len(half) < 2:
        return c
    d = d / n
    scale = min(abs(float(half[k]) / d[k]) if abs(d[k]) > 1e-9 else np.inf for k in range(2))
    return c + d * min(scale, n)


def plan_guess(
    stages: Sequence[Tuple[int, ...]], cms: Mapping[tuple, List[Tuple]],
    centres: Mapping[int, np.ndarray], start_xy: np.ndarray, goal_xy: Optional[np.ndarray],
    standing_height: float, start_yaw: float = 0.0,
    half_extents: Optional[Mapping[int, Sequence[float]]] = None,
) -> List[Optional[StateGuess]]:
    """One guess per solver state, or None where faro's default stands.
    All None when the plan never changes level. The pelvis path runs from
    the start through each straddle (over the edge of the upper face on the
    side it is approached from) to the goal, if there is one."""
    half_extents = half_extents or {}
    per_state = [sole_faces(cms, modes) for modes in stages]
    levels = [{p: float(centres[f][2]) for p, f in faces.items() if f in centres}
              for faces in per_state]
    guesses: List[Optional[StateGuess]] = [None] * len(stages)
    straddle_yaw: Dict[int, float] = {}
    for i, lv in enumerate(levels):
        if len(lv) < 2:
            continue
        (hi_patch, z_hi), (lo_patch, z_lo) = sorted(lv.items(), key=lambda kv: -kv[1])
        if z_hi - z_lo < MIN_STEP:
            continue
        hi_xy = centres[per_state[i][hi_patch]][:2]
        # Climbing or descending: look at the last state before this level
        # change whose feet shared a level (a level change spans two states).
        j = i - 1
        while j >= 0 and guesses[j] is not None and guesses[j].legs:
            j -= 1
        prev = levels[j] if j >= 0 else {}
        descending = bool(prev) and max(prev.values()) >= z_hi - 1e-6
        src = goal_xy if (descending and goal_xy is not None) else start_xy
        yaw = _heading(src, hi_xy)
        pelvis = min(z_lo + MAX_LEG_EXTENSION, z_hi + MIN_LEG_FOLD)
        legs = {hi_patch.split("_")[0]: FOLDED_LEG, lo_patch.split("_")[0]: STRAIGHT_LEG}
        hi_face = per_state[i][hi_patch]
        xy = face_edge_toward(centres[hi_face], half_extents.get(hi_face), src)
        guesses[i] = StateGuess(pelvis, yaw, legs, xy)
        if yaw is not None:
            straddle_yaw[i] = yaw
    if not any(guesses):
        return guesses
    anchors: Dict[int, np.ndarray] = {0: np.asarray(start_xy, dtype=float)[:2]}
    anchors.update({i: g.xy for i, g in enumerate(guesses) if g is not None and g.xy is not None})
    last = len(stages) - 1
    if goal_xy is not None and last not in anchors:
        anchors[last] = np.asarray(goal_xy, dtype=float)[:2]
    a_keys = sorted(anchors)

    def path_xy(i: int) -> np.ndarray:
        before = [k for k in a_keys if k <= i]
        after = [k for k in a_keys if k > i]
        if not after:
            return anchors[before[-1]]
        k0, k1 = before[-1], after[0]
        t = (i - k0) / max(k1 - k0, 1)
        return anchors[k0] + t * (anchors[k1] - anchors[k0])
    # Non-straddle states: pelvis at standing height over their level, and
    # the heading turned gradually over the steps between one straddle's
    # facing and the next (the feet are planted, so a jump in the guess
    # only fights the sticking constraints); held after the last one.
    keys = sorted(straddle_yaw)
    for i, lv in enumerate(levels):
        if guesses[i] is not None or not lv:
            continue
        before = [k for k in keys if k < i]
        after = [k for k in keys if k > i]
        if after:
            k1, y1 = after[0], straddle_yaw[after[0]]
            k0, y0 = (before[-1], straddle_yaw[before[-1]]) if before else (0, start_yaw)
            t = (i - k0) / max(k1 - k0, 1)
            yaw = y0 + t * ((y1 - y0 + math.pi) % (2 * math.pi) - math.pi)
        elif before:
            yaw = straddle_yaw[before[-1]]
        else:
            yaw = None
        guesses[i] = StateGuess(max(lv.values()) + standing_height, yaw, {}, path_xy(i))
    return guesses


def _leg_indices(model: Any, side: str) -> Optional[Tuple[int, int, int]]:
    idx = []
    for joint in ("hip_pitch", "knee", "ankle_pitch"):
        name = f"{side}_{joint}_joint"
        if not model.existJointName(name):
            return None
        idx.append(int(model.joints[model.getJointId(name)].idx_v))
    return idx[0], idx[1], idx[2]


def _target_xy(target: Any) -> Optional[np.ndarray]:
    try:
        xy = np.asarray(getattr(target, "value", None), dtype=float).reshape(-1)[:2]
    except (TypeError, ValueError):
        return None
    return xy if xy.shape == (2,) and np.all(np.isfinite(xy)) else None


def _goal_xy(kso_task: Any, rid: str) -> Optional[np.ndarray]:
    """Where the pelvis path ends: the robot's own terminal target, else the
    centre of the goal objects' targets. Most tasks only place objects
    (a box on the floor past the table); without this fallback a descent
    had no goal and was seeded back toward the start, so KSO stepped off
    the near edge and set the box down behind the start
    (climb_table_up_down_with_box gen_31, box goal x = 1.9, ended x = -0.29)."""
    goal = getattr(kso_task, "terminal_goal", None)
    if goal is None:
        return None
    target = (getattr(goal, "robots", None) or {}).get(rid)
    if target is not None:
        return _target_xy(target)
    points = [xy for xy in (_target_xy(t) for t in (getattr(goal, "objects", None) or {}).values()) if xy is not None]
    return np.mean(points, axis=0) if points else None


def apply(kso_task: Any, cms: Mapping[tuple, List[Tuple]]) -> int:
    """Seed *kso_task*'s solver (call after ``initialize_solver()``).
    Returns the number of stages changed; 0 for plans without a level
    change, or when disabled."""
    if not enabled():
        return 0
    try:
        return _apply(kso_task, cms)
    except Exception as exc:  # noqa: BLE001 - a guess must never break a solve
        logger.warning("KSO warm start skipped: %s", exc)
        return 0


def _apply(kso_task: Any, cms: Mapping[tuple, List[Tuple]]) -> int:
    import pinocchio as pin
    from faro.configs.identifiers import robot_name

    scene, resolved = kso_task.scene, kso_task.resolved
    centres = face_centres(scene, resolved)
    half_extents = {env_id: getattr(scene.objects.get(name), "half_extents", None)
                    for env_id, (name, _local) in scene.faces.by_id.items()}
    stages = state_modes(kso_task.num_modes, kso_task.hard_initial, kso_task.has_terminal_goal)
    wrapper = kso_task.solver.solver
    per_robot: Dict[str, Tuple[np.ndarray, List[Optional[StateGuess]]]] = {}
    for rid, cfg in scene.robots.items():
        q0 = np.asarray(resolved.robots[rid].q, dtype=float)
        own = {k: v for k, v in cms.items() if len(k) == 3 and k[1] == rid}
        standing = float(getattr(cfg, "default_standing_height", None) or q0[2])
        per_robot[rid] = (q0, plan_guess(stages, own, centres, q0[:2], _goal_xy(kso_task, rid),
                                         standing, _yaw(q0[3:7]), half_extents))
    if not any(any(g) for _, g in per_robot.values()):
        return 0
    changed = 0
    # Configurations of the seed, per robot and state, for re-linearising
    # the collision constraints (faro linearises them at the initial pose).
    seed_qs = {rid: np.tile(np.asarray(resolved.robots[rid].q, dtype=float), (len(stages), 1))
               for rid in scene.robots}
    for stage in range(len(stages)):
        if stage == 0 and kso_task.hard_initial:
            continue  # pinned to the initial state
        x_blocks: Dict[str, np.ndarray] = {}
        for rid, cfg in scene.robots.items():
            q0, guesses = per_robot[rid]
            dq = np.hstack((np.zeros(6), q0[7:]))
            g = guesses[stage]
            if g is not None:
                # KSO integrates the base on SE3 (faro.ocp.kinematics), so the
                # guess is the exact SE3 tangent from the initial base pose to
                # an upright pose at the guessed position and heading.
                target = q0[:7].copy()
                if g.xy is not None:
                    target[:2] = g.xy
                if g.pelvis_z is not None:
                    target[2] = g.pelvis_z
                yaw = g.yaw if g.yaw is not None else _yaw(q0[3:7])
                target[3:7] = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
                dq[:6] = pin.liegroups.SE3().difference(q0[:7], target)
                for side, joints in g.legs.items():
                    idx = _leg_indices(cfg.pin_model, side)
                    if idx is not None:
                        dq[list(idx)] = joints
            x_blocks[robot_name(rid, "dq_rob")] = dq
            seed_qs[rid][stage, :7] = pin.liegroups.SE3().integrate(q0[:7], dq[:6])
            seed_qs[rid][stage, 7:] = dq[6:]
        for name in kso_task.env_objs:
            x_blocks[name] = np.zeros(kso_task.formulation.obj_n_v)
        wrapper.solver.set(stage, "x", wrapper._stack_block("x", x_blocks))
        if stage >= 1:
            u_blocks = {k.replace("dq_rob", "dq_next_rob"): v for k, v in x_blocks.items()}
            try:
                wrapper.solver.set(stage - 1, "u", wrapper._stack_block("u", u_blocks))
            except KeyError:
                pass  # input layout differs from the state layout: x alone is enough
        changed += 1
    # faro resolves the collision pairs (witness points, normals) at the
    # initial standing pose for every state, so the first SQP rounds would
    # see the table as it looks from the start, metres away. Resolve them at
    # the seed instead.
    obj_qs = {name: np.tile(np.asarray(resolved.env_objects[name].q, dtype=float), (len(stages), 1))
              for name in kso_task.env_objs}
    kso_task.resolve_collisions(seed_qs, obj_qs)
    kso_task.apply_params()
    logger.debug("KSO warm start: %d state(s) seeded (level change in plan)", changed)
    return changed
