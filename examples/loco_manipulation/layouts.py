"""
Scene layout variants for searches and datasets (MotionDisco side).

faro's initial-condition registry takes only the scene, and its
scenes are read-only here, so start layouts are varied on top of a base
IC: an IC name ``<base_ic>@<layout>`` (e.g. ``default@far``) resolves the
base ``StateSpec`` and applies the named layout's offsets. The name works
everywhere an IC name does (``task.initial_condition``,
``SHINKA_LOCOMANIP_IC``, ``.hydra/config.yaml``), so prompts, evaluator,
replays and post-search tools all see the same layout.

A ``Layout`` holds:

    robot       (dx, dy, dyaw) added to the robot's ``Standing.xy_yaw``
    objects     {name: (dx, dy, dyaw)} added to an object's ``OnFace``
                xy_yaw (in its support face's frame) or to a ``Pose``
                position; fixed objects (tables, trucks) and movable
                ones alike. Objects resting on a moved object move with it.
    goal_shift  (dx, dy, dz) added to every terminal-goal target, or
    goal_follows  name of an object whose world displacement the goal
                targets follow (a goal "on the table" moves with the table)

Terminal goals that the scene defines as fixed world positions are
shifted accordingly, and an outcome scorer bound to that goal (the
common ``goal.score`` pattern, or a scorer holding ``.goal``) is rebound
to the shifted goal. ``check_layout`` rejects layouts in which fixed
boxes overlap or the robot starts inside one.

Pure Python on faro's frozen dataclasses; faro is imported lazily.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LAYOUT_SEP = "@"

Offset = Tuple[float, float, float]


@dataclass(frozen=True)
class Layout:
    robot: Offset = (0.0, 0.0, 0.0)
    objects: Mapping[str, Offset] = field(default_factory=dict)
    goal_shift: Optional[Tuple[float, float, float]] = None
    goal_follows: Optional[str] = None
    note: str = ""


# Named layouts per scene. Offsets are relative to the base IC; keep the
# goal consistent (goal_follows / goal_shift) when moving what it sits on.
LAYOUTS: Dict[str, Dict[str, Layout]] = {
    "truck_unload_stack": {
        "near": Layout(objects={"truck_bed": (-0.3, 0, 0), "rear_step": (-0.3, 0, 0)},
                       note="truck 0.3 m closer to the robot"),
        "far": Layout(objects={"truck_bed": (0.4, 0, 0), "rear_step": (0.4, 0, 0)},
                      note="truck 0.4 m further away"),
        "side_start": Layout(robot=(0.0, 0.6, -0.4),
                             note="robot starts 0.6 m to the left, turned toward the truck"),
        "boxes_swapped": Layout(objects={"big_box": (0.0, -0.8, 0), "small_box": (0.0, 0.8, 0)},
                                note="big and small box swap sides on the bed"),
    },
    "floor_stack_carry": {
        "side_start": Layout(robot=(0.0, 0.5, -0.5), note="robot starts beside the stack"),
        "far_stack": Layout(objects={"big_box": (0.4, 0.0, 0)}, goal_follows="big_box",
                            note="stack 0.4 m further away; target moves with it"),
    },
    "motiondisco.long_dist_pick_place": {
        "table_left": Layout(objects={"table_short": (0.0, 0.8, 0)}, goal_follows="table_short",
                             note="target table 0.8 m to the left"),
        "table_far": Layout(objects={"table_short": (0.7, 0.0, 0)}, goal_follows="table_short",
                            note="target table 0.7 m further away"),
        "box_side": Layout(objects={"yellow_box": (0.0, -0.4, 0.6)},
                           note="box 0.4 m to the right, rotated"),
    },
    "motiondisco.climb_table_with_box": {
        # Goal: standing on the table holding the box (robot and box
        # targets are world positions), so they follow the table.
        "table_far": Layout(objects={"table_short": (0.3, 0.0, 0)}, goal_follows="table_short",
                            note="table 0.3 m further away; targets move with it"),
        "box_left": Layout(objects={"white_box": (0.0, 0.3, 0.4)},
                           note="box 0.3 m further left, rotated"),
        "start_back": Layout(robot=(-0.4, 0.0, 0.0), note="robot starts 0.4 m further back"),
    },
    "motiondisco.under_table_pick_place": {
        # The table is four boxes (top, two sides, back); move them together.
        "table_far": Layout(
            objects={"table_top_short": (0.3, 0.0, 0), "table_side_left": (0.3, 0.0, 0),
                     "table_side_right": (0.3, 0.0, 0), "table_back": (0.3, 0.0, 0),
                     "white_box": (0.3, 0.0, 0)},
            goal_follows="table_top_short", note="table (with the box under it) 0.3 m further away"),
        "box_left": Layout(objects={"white_box": (0.0, 0.25, 0.3)},
                           note="box under the table, 0.25 m to the left, rotated"),
        "start_side": Layout(robot=(-0.2, 0.4, -0.3), note="robot starts to the left, turned"),
    },
    "motiondisco.box_stacking": {
        "table_far": Layout(objects={"table_short": (0.3, 0.0, 0)}, note="table 0.3 m further away"),
        "side_start": Layout(robot=(0.0, -0.5, 0.5), note="robot starts to the right of the table"),
    },
}


def split_ic(ic: str) -> Tuple[str, Optional[str]]:
    """``"default@far"`` -> ``("default", "far")``; plain names -> ``(ic, None)``."""
    base, sep, layout = str(ic).partition(LAYOUT_SEP)
    return base, (layout or None) if sep else None


def get_layout(scene_name: str, name: str) -> Layout:
    try:
        return LAYOUTS[scene_name][name]
    except KeyError:
        known = sorted(LAYOUTS.get(scene_name, {}))
        raise ValueError(
            f"unknown layout {name!r} for scene {scene_name!r}; known: {known}"
        ) from None


def _add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def apply_layout(spec: Any, layout: Layout) -> Any:
    """A ``StateSpec`` copy with the layout's offsets applied."""
    from faro.configs.scene import OnFace, Pose, Standing

    robots = dict(spec.robots)
    if any(layout.robot):
        if len(robots) != 1:
            raise ValueError("robot offsets need a single-robot scene")
        rid, start = next(iter(robots.items()))
        if not isinstance(start, Standing):
            raise ValueError(f"robot offset needs a Standing start, got {type(start).__name__}")
        robots[rid] = replace(start, xy_yaw=_add(start.xy_yaw, layout.robot))
    objects = dict(spec.objects)
    for name, offset in layout.objects.items():
        placement = objects.get(name)
        if isinstance(placement, OnFace):
            objects[name] = replace(placement, xy_yaw=_add(placement.xy_yaw, offset))
        elif isinstance(placement, Pose):
            q = list(placement.q)
            q[0] += float(offset[0])
            q[1] += float(offset[1])
            if offset[2]:
                raise ValueError(f"yaw offsets on Pose placements are not supported ({name!r})")
            objects[name] = replace(placement, q=tuple(q))
        else:
            raise ValueError(f"layout moves {name!r}, which the base IC does not place")
    return replace(spec, robots=robots, objects=objects)


def goal_offset(layout: Layout, base_resolved: Any, resolved: Any) -> np.ndarray:
    """World offset (3,) to add to terminal-goal targets."""
    if layout.goal_follows:
        name = layout.goal_follows
        before = {**base_resolved.fixed_env_objects, **base_resolved.env_objects}
        after = {**resolved.fixed_env_objects, **resolved.env_objects}
        if name not in before or name not in after:
            raise ValueError(f"goal_follows names unknown object {name!r}")
        return np.asarray(after[name].q[:3], float) - np.asarray(before[name].q[:3], float)
    if layout.goal_shift is not None:
        return np.asarray(layout.goal_shift, float)
    return np.zeros(3)


def shift_target(target: Any, offset: np.ndarray) -> Any:
    from faro.search.goal import PositionTarget

    if isinstance(target, PositionTarget):
        return replace(target, xyz=np.asarray(target.xyz, float) + offset)
    q = np.asarray(target.q, float).copy()
    q[:3] += offset
    return replace(target, q=q)


def _offset_for(offset: Any, name: str) -> np.ndarray:
    if isinstance(offset, Mapping):
        return np.asarray(offset.get(name, np.zeros(3)), float)
    return np.asarray(offset, float)


def _any_offset(offset: Any) -> bool:
    if isinstance(offset, Mapping):
        return any(np.any(np.asarray(v, float)) for v in offset.values())
    return bool(np.any(np.asarray(offset, float)))


def shift_goal(goal: Any, offset: Any) -> Any:
    """Terminal goal with its targets moved: *offset* is one (3,) vector
    for every target or ``{owner name: (3,)}`` per target (None stays None)."""
    if goal is None or not _any_offset(offset):
        return goal
    return replace(
        goal,
        robots={k: shift_target(t, _offset_for(offset, k)) for k, t in goal.robots.items()},
        objects={k: shift_target(t, _offset_for(offset, k)) for k, t in goal.objects.items()},
    )


def shift_scorer(scorer: Any, offset: Any) -> Any:
    """Move the goal behind a goal-based outcome scorer by *offset*.

    Scene scorers build their own ``TerminalGoal`` instance (e.g.
    ``box_on_table(scene, resolved).score``), so the goal is found on the
    scorer itself: a ``TerminalGoal.score`` bound method, or a scorer
    object with a ``.goal`` attribute (e.g. ``StackAtGoal``). Anything
    else is returned unchanged (its residual does not use a position).
    """
    from faro.search.goal import TerminalGoal

    if scorer is None or not _any_offset(offset):
        return scorer
    owner = getattr(scorer, "__self__", None)
    if isinstance(owner, TerminalGoal) and getattr(scorer, "__name__", "") == "score":
        return shift_goal(owner, offset).score
    goal = getattr(scorer, "goal", None)
    if isinstance(goal, TerminalGoal):
        out = copy.copy(scorer)
        out.goal = shift_goal(goal, offset)
        return out
    return scorer


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _box_xy_extent(q: Sequence[float], half: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned xy bounds of a yawed box (centre q[:3], quaternion xyzw)."""
    qx, qy, qz, qw = (float(v) for v in q[3:7])
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    hx = c * half[0] + s * half[1]
    hy = s * half[0] + c * half[1]
    centre = np.asarray(q[:2], float)
    return centre - [hx, hy], centre + [hx, hy]


def _fixed_boxes(scene: Any, resolved: Any) -> Dict[str, Tuple[np.ndarray, np.ndarray, float, float]]:
    boxes = {}
    for name, cfg in scene.fixed_env_objects.items():
        half = list(getattr(cfg, "half_extents", []) or [])
        if len(half) == 3:
            lo, hi = _box_xy_extent(resolved.fixed_env_objects[name].q, half)
            z = float(resolved.fixed_env_objects[name].q[2])
            boxes[name] = (lo, hi, z - half[2], z + half[2])
    return boxes


def _overlapping_pairs(boxes: Mapping[str, Any], tol: float) -> set:
    names = sorted(boxes)
    out = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (alo, ahi, az0, az1), (blo, bhi, bz0, bz1) = boxes[a], boxes[b]
            overlap_xy = np.minimum(ahi, bhi) - np.maximum(alo, blo)
            overlap_z = min(az1, bz1) - max(az0, bz0)
            if np.all(overlap_xy > tol) and overlap_z > tol:
                out.add((a, b))
    return out


def check_layout(scene: Any, resolved: Any, base_resolved: Any = None, *,
                 robot_clearance: float = 0.15, tol: float = 1e-3) -> List[str]:
    """Problems a layout introduces: fixed boxes that overlap (touching is
    fine; overlaps already present in *base_resolved*, e.g. a table built
    from interpenetrating parts, are ignored), or a robot standing within
    ``robot_clearance`` of a fixed box it does not stand on."""
    problems: List[str] = []
    boxes = _fixed_boxes(scene, resolved)
    known = _overlapping_pairs(_fixed_boxes(scene, base_resolved), tol) if base_resolved else set()
    for a, b in sorted(_overlapping_pairs(boxes, tol) - known):
        problems.append(f"fixed objects {a!r} and {b!r} overlap")
    for rid, state in resolved.robots.items():
        base = np.asarray(state.q[:3], float)
        for name, (lo, hi, _z0, z1) in boxes.items():
            standing_on_top = base[2] > z1
            inside = np.all(base[:2] > lo - robot_clearance) and np.all(base[:2] < hi + robot_clearance)
            if inside and not standing_on_top:
                problems.append(f"robot {rid!r} starts within {robot_clearance} m of {name!r}")
    return problems


def describe(scene_name: str) -> List[str]:
    """``name: note`` lines for the scene's layouts (for CLIs / docs)."""
    return [f"{n}: {lay.note}" for n, lay in sorted(LAYOUTS.get(scene_name, {}).items())]
