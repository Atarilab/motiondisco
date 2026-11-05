"""
Solver-backed grasp feedback for a failed TO window.

Incremental evaluation checks kinematics (KSO) on the first window only,
so a grasp the arms cannot make — two faces further apart than the hands
can span, a face resting on the floor — reaches TO unexamined and fails
there with wrists past their limits and the box inside the torso, which
tells the LLM nothing about the cause. On occupied_side_resting_support_
reorientation_001 (box 0.72 x 0.90 x 0.40 m) every plan of the first hour
gripped the 0.90 m left / right pair, or left + front with the front face
on the floor, and none converged.

``feedback`` asks faro's static IK (`to_static.static_pose`: a statically
stable configuration with the mode's contacts, near a guess) whether a
standing pose exists for the failing window's grasp, starting from the
window's own start state. When there is none, it probes the object's other
opposite-face pairs (closest together first, faces resting on a support
excluded) and names one a pose exists for. The robot model decides, not a
span constant. Feedback only: nothing is scored or gated on it.

The static IK is anchored to its seed, so the seed matters: from the
window's start, metres from the object, it found no pose even for pairs
that have one, and the failed window's own iterate at the grasp can be
garbage (a foot 11 cm off the floor) and fail just the same. Each grasp
is therefore tried near the trajectory's state at the grasp first and,
failing that, near the window's start (the last converged state); "no
pose" means both failed. ``SHINKA_GRASP_PROBE=0`` turns it off;
``SHINKA_GRASP_PROBES`` caps the static IK solves per evaluation
(default 4, ~15-25 s each).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

GRASP_PROBE = os.environ.get("SHINKA_GRASP_PROBE", "1").strip().lower() not in ("0", "false", "no", "off")
MAX_PROBES = int(os.environ.get("SHINKA_GRASP_PROBES", "4"))
HANDS = ("left_ee", "right_ee")
# Opposite faces of a box lie on one local axis; their separation is twice
# the half extent on it (faro's face frames: front/back x, left/right y,
# top/bottom z).
PAIRS = (("front", "back"), ("left", "right"), ("top", "bottom"))
FACE_AXIS = {"front": 0, "back": 0, "left": 1, "right": 1, "top": 2, "bottom": 2}


def _t(entry: Any) -> tuple:
    return tuple(entry) if entry else ()


def object_faces(scene: Any, obj: str) -> List[Tuple[int, int, str]]:
    """(global face id, own face id, face name) of ``obj``'s faces."""
    from _task_loader import scene_faces

    return list(scene_faces(scene)[obj])


def hands_on(plan: Mapping[str, Sequence[Any]], mode: int, faces: Mapping[int, Tuple[str, str]]
             ) -> Dict[str, Tuple[str, str]]:
    """hand channel -> (object, face) for every hand on an object face at ``mode``."""
    out: Dict[str, Tuple[str, str]] = {}
    for hand in HANDS:
        seq = plan.get(hand)
        if seq is None or mode >= len(seq):
            continue
        c = _t(seq[mode])
        if c and int(c[0]) in faces:
            out[hand] = faces[int(c[0])]
    return out


def grasp_mode(plan: Mapping[str, Sequence[Any]], a: int, e: int, faces: Mapping[int, Tuple[str, str]]
               ) -> Optional[int]:
    """The first mode of ``[a, e)`` at which the most hands hold one
    object; None when no hand touches an object in the window."""
    best, best_n = None, 0
    for m in range(a, e):
        on = hands_on(plan, m, faces)
        for obj in {o for o, _f in on.values()}:
            n = sum(1 for o, _f in on.values() if o == obj)
            if n > best_n:
                best, best_n = m, n
    return best


def alternatives(held: Sequence[str], resting: Sequence[str], half: Sequence[float]
                 ) -> List[Tuple[str, str, float]]:
    """Opposite face pairs other than the held one with neither face on a
    support, closest together first: (face, face, separation in m)."""
    out = []
    for fa, fb in PAIRS:
        if {fa, fb} == set(held) or fa in resting or fb in resting:
            continue
        out.append((fa, fb, 2.0 * float(half[FACE_AXIS[fa]])))
    return sorted(out, key=lambda x: x[2])


def start_state(task: Any, results: Mapping[str, Any], node: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """(robot q, object poses) at trajectory node ``node``."""
    rid = task.robot_id
    nq = task.scene.robots[rid].nq
    xs = results["robot_xs"][rid] if isinstance(results["robot_xs"], dict) else results["robot_xs"]
    q = np.asarray(xs, float)[node][:nq]
    objs = {k: np.asarray(v, float)[node][:7] for k, v in results["obj_xs"].items()}
    return q, objs


def probe(task: Any, plan: Mapping[str, Sequence[Any]], mode: int,
          guess: Tuple[np.ndarray, Dict[str, np.ndarray]]) -> Optional[int]:
    """Static IK status (0 = a pose exists) for ``mode``'s contacts near
    ``guess``; None when the solve could not run."""
    try:
        import to_static
        from _task_loader import to_faro_plan

        tup = {ch: [_t(m) for m in seq] for ch, seq in plan.items()}
        _pose, status = to_static.static_pose(task, to_faro_plan(task.scene, tup), mode, guess)
        return int(status)
    except Exception as e:  # noqa: BLE001 - feedback is best effort
        print(f"[grasp_probe] static IK failed: {type(e).__name__}: {e}", flush=True)
        return None


def _describe(obj: str, held: Sequence[str], half: Sequence[float]) -> str:
    if len(held) == 1:
        return f"{obj} on its {held[0]} face with one hand"
    if held[0] == held[1]:
        return f"{obj} on its {held[0]} face with both hands"
    if {held[0], held[1]} in (set(p) for p in PAIRS):
        return f"{obj} on its {held[0]} and {held[1]} faces ({2.0 * float(half[FACE_AXIS[held[0]]]):.2f} m apart)"
    return f"{obj} on its {held[0]} and {held[1]} faces (not opposite faces)"


def feedback(task: Any, plan: Mapping[str, Sequence[Any]], to_solve: Any,
             probe_fn=probe) -> List[str]:
    """One line for a failed segmented TO solve whose failing window holds
    an object with no standing pose for that grasp; [] when the window has
    no hand-object contact, a pose exists for the planned grasp, or the
    probe cannot run. Never raises."""
    if not GRASP_PROBE:
        return []
    try:
        bad = next((s for s in (getattr(to_solve, "segments", None) or [])
                    if s.get("status") not in (0, None) and not s.get("physical_pass")), None)
        if bad is None:
            return []
        # The solved window (kept modes plus the lookahead): a grasp the
        # lookahead reaches fails the solve just the same.
        win = (bad.get("extra") or {}).get("window") or bad["modes"]
        a, e = int(bad["modes"][0]), int(win[1])
        scene = task.scene
        objects = set(scene.env_objects)
        faces = {fid: (name, face) for (name, face), fid in scene.faces.by_name.items() if name in objects}
        m = grasp_mode(plan, a, e, faces)
        if m is None:
            return []
        on = hands_on(plan, m, faces)
        obj = max({o for o, _f in on.values()}, key=lambda o: sum(1 for oo, _f in on.values() if oo == o))
        held = [f for o, f in on.values() if o == obj]
        npm = max(1, int(to_solve.num_nodes) // max(1, int(to_solve.num_modes)))
        # Seeds (see the module doc): the trajectory's state at the grasp,
        # then the window's start.
        nodes = [min(m * npm, int(to_solve.num_nodes))]
        if a * npm != nodes[0]:
            nodes.append(a * npm)
        seeds = [start_state(task, to_solve.results, node) for node in nodes]
        t0 = time.time()
        probes = 0

        def exists(p: Mapping[str, Sequence[Any]]) -> Optional[bool]:
            """A standing pose near any seed; None when the probe could not run."""
            nonlocal probes
            for guess in seeds:
                if probes >= MAX_PROBES:
                    return None
                probes += 1
                status = probe_fn(task, p, m, guess)
                if status is None:
                    return None
                if status == 0:
                    return True
            return False

        opposite = len(held) == 2 and {held[0], held[1]} in (set(p) for p in PAIRS)
        if opposite:
            # A squeeze grip: is there a standing pose for it at all?
            if exists(plan) is not False:
                return []
        # Otherwise (adjacent / same face, one hand) a static pose usually
        # exists — friction holds the object — but grip_feedback already
        # says that grip is weak; what the LLM needs is which opposite pair
        # the hands can actually span, so probe those directly.
        half = np.asarray(scene.env_objects[obj].half_extents, float)
        own = object_faces(scene, obj)
        by_name = {face: eid for eid, _sid, face in own}
        self_face = {sid: face for _eid, sid, face in own}
        # Object entries are (own face, env face, type): the own face rests on the support.
        rest = _t(plan[obj][m]) if obj in plan and m < len(plan[obj]) else ()
        resting = [self_face[int(rest[0])]] if rest and int(rest[0]) in self_face else []
        found: Optional[Tuple[str, str, float]] = None
        for fa, fb, sep in alternatives(held, resting, half):
            if probes >= MAX_PROBES:
                break
            alt = {ch: list(seq) for ch, seq in plan.items()}
            alt["left_ee"][m], alt["right_ee"][m] = (by_name[fa], 1), (by_name[fb], 1)
            if exists(alt):
                found = (fa, fb, sep)
                break
        if probes == 0:
            return []
        solves = f"static IK, {probes} solve{'s' if probes > 1 else ''}, {time.time() - t0:.0f} s"
        if opposite:
            line = (f"mode {m}: no standing pose exists in which the robot holds {_describe(obj, held, half)} "
                    f"({solves}), so TO cannot make that grasp.")
        else:
            line = f"mode {m}: {obj} is held {_describe(obj, held, half)[len(obj) + 1:]} ({solves})."
        if found is not None:
            line += (f" A standing pose does exist holding its {found[0]} and {found[1]} faces ({found[2]:.2f} m apart): "
                     "grip those instead.")
        else:
            line += (" No standing pose was found for its other opposite face pairs either: "
                     "support it from below or stage it on another face first.")
        return [line]
    except Exception as e:  # noqa: BLE001 - feedback is best effort
        print(f"[grasp_probe] feedback failed: {type(e).__name__}: {e}", flush=True)
        return []
