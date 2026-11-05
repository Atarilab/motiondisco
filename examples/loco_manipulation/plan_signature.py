"""
Plan-level descriptors for diversity: signature, novelty, families.

A contact plan is turned into a sequence of *strategy tokens*, one per
mode transition, that ignores how many steps a walk takes:

    - A transition that only lifts a sole or puts it back down on the
      surface it last stood on is a locomotion step. Any run of such
      steps collapses into a single ``walk`` token.
    - Every other transition becomes one token listing each changed
      channel as ``channel:before>after`` (channels sorted, contact
      tuples written as their integer IDs, ``-`` for free), e.g.
      ``left_ee:->9.1`` (left hand grasps env 9) or
      ``white_box:6.1.1>-`` (box lifted).

The signature is the ``|``-joined token list. Two plans share a
signature iff they make and break the same contacts in the same order
and only differ in step counts — i.e. they are the same *strategy*
("plan family"). ``plan_novelty`` is the normalized Levenshtein distance
over tokens to the nearest previously seen plan.

A second, coarser descriptor ignores order entirely: ``plan_patches``
lists, per channel, the set of contacts it ever uses (which faces the
hands touch, which surfaces the feet and objects rest on, sticking or
sliding). Two plans with the same grasp and support patches share it
even if they order the contacts differently. ``combined_novelty``
weights patch novelty (Jaccard distance over the patch sets) above
ordering novelty, so new contact patches count for more than reordering
the same ones.

Pure Python on the flat-channel plan format; no faro import.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Set to "1" (task.diversity=true) to enable plan-level diversity in
# evaluate.py and the matching prompt section in prompts.py.
DIVERSITY_ENV = "SHINKA_DIVERSITY"

SOLE_CHANNELS = ("left_sole", "right_sole")
WALK_TOKEN = "walk"


def _fmt(entry: Any) -> str:
    entry = tuple(entry) if isinstance(entry, (list, tuple)) else (entry,)
    return ".".join(str(v) for v in entry) if entry else "-"


def plan_tokens(plan: Dict[str, List[Any]]) -> List[str]:
    """Strategy tokens for *plan* (see module docstring)."""
    if not plan:
        return []
    channels = sorted(ch for ch, seq in plan.items() if isinstance(seq, list))
    if not channels:
        return []
    n = min(len(plan[ch]) for ch in channels)
    # Last surface each sole stood on, to tell a step from a step-up.
    support = {ch: None for ch in SOLE_CHANNELS}
    for ch in SOLE_CHANNELS:
        if ch in plan and n and tuple(plan[ch][0]):
            support[ch] = tuple(plan[ch][0])

    tokens: List[str] = []
    for k in range(n - 1):
        changes: List[Tuple[str, Any, Any]] = []
        for ch in channels:
            prev, curr = tuple(plan[ch][k]), tuple(plan[ch][k + 1])
            if prev != curr:
                changes.append((ch, prev, curr))
        if not changes:
            continue

        def is_step_change(ch: str, curr: Any) -> bool:
            return ch in SOLE_CHANNELS and (
                curr == () or (support[ch] is not None and curr == support[ch])
            )

        # A plain step that coincides with another change (a grasp made
        # while a foot lifts) is still just a step: keep the token equal
        # to the sequential version's.
        events = [c for c in changes if not is_step_change(c[0], c[2])]
        for ch, _prev, curr in changes:
            if ch in SOLE_CHANNELS and curr != ():
                support[ch] = curr
        if not events:
            if not tokens or tokens[-1] != WALK_TOKEN:
                tokens.append(WALK_TOKEN)
            continue
        tokens.append(
            ",".join(f"{ch}:{_fmt(prev)}>{_fmt(curr)}" for ch, prev, curr in events)
        )
    return tokens


def plan_signature(plan: Dict[str, List[Any]]) -> str:
    """Strategy signature: the ``|``-joined strategy tokens."""
    return "|".join(plan_tokens(plan)) or "(no transitions)"


def signature_tokens(signature: str) -> List[str]:
    if not signature or signature == "(no transitions)":
        return []
    return signature.split("|")


def token_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Levenshtein distance over tokens, normalized to [0, 1]."""
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        curr = [i]
        for j, tb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = curr
    return prev[-1] / max(len(a), len(b))


def plan_novelty(signature: str, prior_signatures: Iterable[str]) -> float:
    """Distance from *signature* to its nearest prior signature.

    1.0 when nothing has been seen yet; 0.0 for a strategy already
    explored (same family, e.g. only step counts differ).
    """
    tokens = signature_tokens(signature)
    best: Optional[float] = None
    for prior in prior_signatures:
        d = token_distance(tokens, signature_tokens(prior))
        best = d if best is None else min(best, d)
        if best == 0.0:
            break
    return 1.0 if best is None else best


PATCH_WEIGHT = 0.7


def plan_patches(plan: Dict[str, List[Any]]) -> str:
    """Order-free contact-patch descriptor.

    ``channel:c1,c2|...`` with channels sorted and each channel's
    distinct non-free contacts sorted (contact tuples written as in
    ``plan_tokens``). E.g. ``left_ee:9.1|left_sole:1.1|...|box:6.1.1,6.2.1``.
    """
    parts = []
    for ch in sorted(ch for ch, seq in plan.items() if isinstance(seq, list)):
        used = sorted({_fmt(c) for c in plan[ch] if tuple(c)})
        parts.append(f"{ch}:{','.join(used) if used else '-'}")
    return "|".join(parts)


def patch_set(patches: str) -> set:
    """``{"channel:contact", ...}`` from a ``plan_patches`` string."""
    out = set()
    for part in (patches or "").split("|"):
        ch, _, used = part.partition(":")
        out.update(f"{ch}:{u}" for u in used.split(",") if u and u != "-")
    return out


def patch_novelty(
    patches: str, prior_patches: Iterable[str], ignore: Iterable[str] = (),
) -> float:
    """Jaccard distance from *patches* to the nearest prior patch set
    (1.0 when nothing has been seen yet).

    ``ignore`` lists ``channel:contact`` tokens shared by every plan of the
    task (the initial contacts, the discrete goal), which would otherwise
    dilute the distance: a new grasp pair on top of them scored ~0.5.
    """
    skip = set(ignore)
    mine = patch_set(patches) - skip
    best: Optional[float] = None
    for prior in prior_patches:
        other = patch_set(prior) - skip
        union = mine | other
        d = 1.0 - (len(mine & other) / len(union) if union else 1.0)
        best = d if best is None else min(best, d)
        if best == 0.0:
            break
    return 1.0 if best is None else best


def combined_novelty(patch_nov: float, order_nov: float) -> float:
    """Novelty that ranks new contact patches above reordering."""
    return PATCH_WEIGHT * patch_nov + (1.0 - PATCH_WEIGHT) * order_nov


def family_stats(
    records: Iterable[Dict[str, Any]],
    key: str = "signature",
) -> List[Dict[str, Any]]:
    """Group seen-plan records by ``key`` (``signature`` or ``patches``).

    Each record needs ``key`` and ``combined_score``. Returns one entry
    per family, sorted best-first: ``{signature, count, best_score}``
    (``signature`` holds the grouping value).
    """
    fams: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        sig = rec.get(key)
        if not sig:
            continue
        fam = fams.setdefault(sig, {"signature": sig, "count": 0, "best_score": 0.0})
        fam["count"] += 1
        fam["best_score"] = max(fam["best_score"], float(rec.get("combined_score") or 0.0))
    return sorted(fams.values(), key=lambda f: (-f["best_score"], -f["count"]))


def shared_patch_tokens(
    initial_mode: Dict[str, Any], discrete_goal: Optional[Dict[str, Any]],
) -> set:
    """``channel:contact`` tokens every plan of a task carries: the scene's
    initial contacts and the discrete goal's final contacts (for
    ``patch_novelty(ignore=...)``)."""
    out = set()
    for ch, entry in (initial_mode or {}).items():
        if tuple(entry) if isinstance(entry, (list, tuple)) else entry:
            out.add(f"{ch}:{_fmt(entry)}")
    for ch, entry in (discrete_goal or {}).items():
        if entry:
            out.add(f"{ch}:{_fmt(entry)}")
    return out
