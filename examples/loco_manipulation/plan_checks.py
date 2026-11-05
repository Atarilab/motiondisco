"""
Pure checks and edits on flat-channel contact plans (no solver, no faro).

    schedule_errors(plan)      faro's contact rule: a channel must pass
                               through free () before touching a different
                               surface (contact_mode_union); violations are
                               rejected without solving.
    event_modes(plan)          modes where something other than a plain
                               walking step happens (grasp, release, object
                               placed / lifted, foot onto another surface).
    first_grasp_mode(plan)     first mode where a hand makes contact.
    describe_transition(...)   "left_ee -> big_box left, big_box lifted".
    insert_walk(plan, k, n)    the plan with n walking steps (same pattern
                               as env.walk, n + 1 modes) spliced in right
                               before mode k.

Mode indices are 0-based; mode 0 is the scene's initial state. The
"event" rule mirrors plan_signature.plan_tokens so both agree on what a
walking step is.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

SOLE_CHANNELS = ("left_sole", "right_sole")
EE_CHANNELS = ("left_ee", "right_ee")

Plan = Dict[str, List[Any]]
Change = Tuple[str, tuple, tuple]


def _t(entry: Any) -> tuple:
    return tuple(entry) if isinstance(entry, (list, tuple)) else (entry,)


def n_modes(plan: Plan) -> int:
    lengths = [len(seq) for seq in plan.values() if isinstance(seq, list)]
    return min(lengths) if lengths else 0


def schedule_errors(plan: Plan) -> List[str]:
    """Transitions faro rejects: a contact that changes surface (or an
    object that changes faces) without a free mode in between. Sticking
    <-> sliding on the same surfaces is allowed."""
    found: List[Tuple[int, str, str]] = []
    for ch in sorted(plan):
        seq = plan[ch]
        if not isinstance(seq, list):
            continue
        for k in range(len(seq) - 1):
            a, b = _t(seq[k]), _t(seq[k + 1])
            if a and b and a[:-1] != b[:-1]:
                found.append((k, ch, (
                    f"mode {k}->{k + 1}: {ch} changes from {a} to {b} without a free () "
                    f"mode in between; release it first: {a} -> () -> {b}"
                )))
    return [msg for _k, _ch, msg in sorted(found)]  # plan order, then channel


def transition_changes(plan: Plan, k: int) -> List[Change]:
    """Channels that differ between mode k and mode k+1."""
    out: List[Change] = []
    for ch in sorted(plan):
        seq = plan[ch]
        if not isinstance(seq, list) or k + 1 >= len(seq):
            continue
        prev, curr = _t(seq[k]), _t(seq[k + 1])
        if prev != curr:
            out.append((ch, prev, curr))
    return out


def _is_walk_step(changes: List[Change], support: Dict[str, Optional[tuple]]) -> bool:
    return bool(changes) and all(
        ch in SOLE_CHANNELS and (curr == () or (support.get(ch) is not None and curr == support[ch]))
        for ch, _prev, curr in changes
    )


def event_modes(plan: Plan) -> List[int]:
    """Modes (index of the new mode) at which a non-walking event happens."""
    n = n_modes(plan)
    support: Dict[str, Optional[tuple]] = {
        ch: (_t(plan[ch][0]) or None) if ch in plan and n else None for ch in SOLE_CHANNELS
    }
    events: List[int] = []
    for k in range(n - 1):
        changes = transition_changes(plan, k)
        if not changes:
            continue
        if not _is_walk_step(changes, support):
            events.append(k + 1)
        for ch, _prev, curr in changes:
            if ch in SOLE_CHANNELS and curr != ():
                support[ch] = curr
    return events


def first_grasp_mode(plan: Plan) -> Optional[int]:
    """First mode at which a hand goes from free to a contact."""
    for k in range(n_modes(plan) - 1):
        for ch, prev, curr in transition_changes(plan, k):
            if ch in EE_CHANNELS and prev == () and curr != ():
                return k + 1
    return None


def describe_transition(
    plan: Plan, mode: int,
    face_name: Callable[[int], str] = str,
    objects: Optional[set] = None,
) -> str:
    """Plain words for what changes between mode-1 and *mode*."""
    if mode < 1:
        return ""
    objects = objects or set()
    parts = []
    for ch, prev, curr in transition_changes(plan, mode - 1):
        if ch in objects or len(curr) == 3 or len(prev) == 3:
            if curr and len(curr) == 3:
                parts.append(f"{ch} set down on {face_name(curr[1])}")
            elif prev and not curr:
                parts.append(f"{ch} lifted")
            else:
                parts.append(f"{ch} -> {curr or 'free'}")
        elif curr:
            kind = " (sliding)" if curr[-1] == 2 else ""
            parts.append(f"{ch} -> {face_name(curr[0])}{kind}")
        else:
            parts.append(f"{ch} released")
    return ", ".join(parts)


def insert_walk(plan: Plan, before_mode: int, n_steps: int) -> Optional[Plan]:
    """Copy of *plan* with ``n_steps`` walking steps inserted right before
    ``before_mode`` (same pattern as ``env.walk``: alternating single
    support starting with the right foot, then double support; n + 1
    modes). Non-sole channels keep the values of mode ``before_mode - 1``.
    Returns None when that mode is not in double support (a walk cannot
    start from a lifted foot)."""
    out: Plan = {ch: list(seq) for ch, seq in plan.items() if isinstance(seq, list)}
    if n_steps <= 0:
        return out
    if before_mode < 1 or before_mode > n_modes(out):
        return None
    if any(ch not in out for ch in SOLE_CHANNELS):
        return None
    anchor = {ch: _t(seq[before_mode - 1]) for ch, seq in out.items()}
    if not anchor["left_sole"] or not anchor["right_sole"]:
        return None
    new_modes = []
    for i in range(n_steps):
        mode = dict(anchor)
        mode["right_sole" if i % 2 == 0 else "left_sole"] = ()
        new_modes.append(mode)
    new_modes.append(dict(anchor))
    return {
        ch: seq[:before_mode] + [m[ch] for m in new_modes] + seq[before_mode:]
        for ch, seq in out.items()
    }


def walk_mode_count(n_steps: int) -> int:
    """Modes ``env.walk(plan, n_steps)`` appends."""
    return n_steps + 1 if n_steps > 0 else 0
