"""Grip feedback for a failed TO solve.

KSO checks only geometry, so it accepts a carried object held by two
hands on faces at right angles (front + right, top + back) or on the same
face. Such a grip cannot squeeze the object between the hands: friction
alone has to hold its weight (and anything stacked on it), the object
slips and tilts, and TO reports hand-friction violations. In the stored
runs no TO with such a grip converged (2 of 2 failed; opposite-face grips
converged 66 of 81 times on long_dist_pick_place).

Feedback only, and only when TO did not converge, so unusual grips stay
open to the search.

    grip_issues(plan, faces, masses)   carry stretches with a weak grip
    grip_lines(issues)                 plain-language feedback lines
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import to_warm_start

OPPOSITE = {frozenset(("left", "right")), frozenset(("front", "back")), frozenset(("top", "bottom"))}
HANDS = ("left_ee", "right_ee")
MAX_LINES = 2


def _t(entry: Any) -> tuple:
    return tuple(entry) if entry else ()


def grip_issues(
    plan: Mapping[str, Sequence[Any]], faces: Mapping[int, Tuple[str, str]],
    masses: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Stretches of consecutive modes in which a carried object (its
    channel free) is held by both hands on the same face or on two faces
    that are not opposite. ``faces`` maps a global face ID to (object,
    face name). Each issue: object, first / last mode, the two face names,
    kind ('adjacent' | 'same'), carried mass in kg (with stacked objects)."""
    n = len(plan.get("left_sole", []))
    owner = {fid: obj for fid, (obj, _f) in faces.items()}
    issues: List[Dict[str, Any]] = []
    open_: Dict[str, Dict[str, Any]] = {}
    for m in range(n):
        loads = to_warm_start.carried_loads(plan, m, masses, owner)
        seen = set()
        for obj, mass in loads.items():
            held = [faces.get(int(_t(plan[h][m])[0])) for h in HANDS if h in plan and _t(plan[h][m])]
            on = [f for o, f in (x for x in held if x) if o == obj]
            if len(on) != 2:
                continue
            kind = "same" if on[0] == on[1] else None if frozenset(on) in OPPOSITE else "adjacent"
            if kind is None:
                continue
            key = (obj, tuple(on))
            cur = open_.get(obj)
            if cur is not None and cur["key"] == key and cur["last"] == m - 1:
                cur["last"] = m
                cur["mass"] = max(cur["mass"], mass)
            else:
                cur = {"key": key, "object": obj, "first": m, "last": m, "faces": tuple(on),
                       "kind": kind, "mass": mass}
                open_[obj] = cur
                issues.append(cur)
            seen.add(obj)
        for obj in list(open_):
            if obj not in seen:
                del open_[obj]
    for i in issues:
        i.pop("key", None)
    return sorted(issues, key=lambda i: -(i["last"] - i["first"]))


def grip_lines(issues: Sequence[Dict[str, Any]], masses: Optional[Mapping[str, float]] = None) -> List[str]:
    lines = []
    for i in issues[:MAX_LINES]:
        span = f"mode {i['first']}" if i["first"] == i["last"] else f"modes {i['first']}–{i['last']}"
        left, right = i["faces"]
        obj = i["object"]
        own = (masses or {}).get(obj)
        weight = f"{i['mass']:.1f} kg"
        if own is not None and i["mass"] > own + 1e-9:
            weight += " including what is stacked on it"
        if i["kind"] == "same":
            how = f"both hands on its {left} face"
        else:
            how = f"left_ee on its {left} face and right_ee on its {right} face, which are at right angles"
        lines.append(
            f"{span}: {obj} is carried with {how}, so the hands cannot squeeze it between them "
            f"and friction alone has to hold its {weight}, so it tends to slip and tilt. Grip two opposite "
            f"faces the arms can span, or support it from below"
        )
    return lines


def to_grip_feedback(task: Any, plan: Mapping[str, Sequence[Any]]) -> List[str]:
    """Lines for a failed TO solve; [] when every carried object is held
    on opposite faces (or the scene cannot be read)."""
    try:
        scene = task.scene
        faces = {fid: (name, face) for (name, face), fid in scene.faces.by_name.items()}
        masses = {name: float(cfg.mass) for name, cfg in scene.env_objects.items()}
        return grip_lines(grip_issues(plan, faces, masses), masses)
    except Exception:  # noqa: BLE001 - feedback is best effort
        return []
