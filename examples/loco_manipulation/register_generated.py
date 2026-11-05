"""
Register generated scenes under ``scenes/generated/`` from a folder of
generated tasks (one sub-folder per task with ``spec.json`` and
``source/{scene.py, initial_conditions.py, goals.py}``).

    python examples/loco_manipulation/register_generated.py --examples <scenes_dir> --all
    python examples/loco_manipulation/register_generated.py --examples <scenes_dir> two_box_counter_swap_001

Only the scene definition is taken: ``scene.py``, ``initial_conditions.py``
and ``spec.json`` are copied unchanged, and ``goals.py`` is rewritten to
MotionDisco's conventions (``transform_goals``): every box within
``GOAL_TOLERANCE_M`` of its target position and ``GOAL_ROTATION_TOLERANCE_RAD``
of its target orientation, as a hard goal in KSO and in TO (the generated
goals are exact poses), and the goal residual as the outcome scorer. Both
solvers need the goal: a position-only goal let an in-place quarter-turn
task count as solved by plans that never turned the box, and with the TO
goal off, TO left boxes up to 80° off the yaw KSO had reached. The
generated discrete goal (which face of each box rests on which support,
hands free) is kept as is. Plans or solutions that may sit next to
the sources are never read. Scenes already registered are skipped unless
``--force``; ``--dry-run`` lists what would be done.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR = os.path.join(_THIS_DIR, "scenes", "generated")
SOURCE_FILES = ("scene.py", "initial_conditions.py")
GOAL_TOLERANCE_M = 0.05
GOAL_ROTATION_TOLERANCE_RAD = 0.1

_TERMINAL_SRC = re.compile(
    r"def task_terminal\(scene, resolved_state\):\n"
    r"    return TerminalGoal\.from_placements\(scene, resolved_state, goal_placements\(\),\n"
    r"\s+kind='pose', kso='hard', to='hard'\)\n")
_TERMINAL_DST = (
    "def task_terminal(scene, resolved_state):\n"
    "    goal = TerminalGoal.from_placements(scene, resolved_state, goal_placements(),\n"
    "                                        kind='pose', kso='hard', to='hard')\n"
    "    tol = GoalTolerances(position=GOAL_TOLERANCE_M, rotation=GOAL_ROTATION_TOLERANCE_RAD)\n"
    "    return replace(goal, objects={k: replace(v, kso_tolerance=tol, to_tolerance=tol)\n"
    "                                  for k, v in goal.objects.items()})\n"
    "\n"
    "def task_score(scene, resolved_state):\n"
    "    return task_terminal(scene, resolved_state).score\n")
_IMPORT_SRC = "from faro.search.goal import DiscreteGoal, TerminalGoal\n"
_IMPORT_DST = ("from dataclasses import replace\n\n"
               "from faro.search.goal import DiscreteGoal, GoalTolerances, TerminalGoal\n")
_TOLERANCE_BLOCK = (
    "\n# MotionDisco: box poses within 5 cm / 0.1 rad of the generated goal in KSO\n"
    "# and in TO (generated: exact poses). The discrete goal (which face rests on\n"
    "# which support, hands free) is as generated.\n"
    f"GOAL_TOLERANCE_M = {GOAL_TOLERANCE_M}\n"
    f"GOAL_ROTATION_TOLERANCE_RAD = {GOAL_ROTATION_TOLERANCE_RAD}\n")


def transform_goals(source: str) -> str:
    """The generated ``goals.py`` rewritten to MotionDisco's conventions.
    Raises ValueError when the source does not have the generated shape
    (the imports, the ``task_terminal`` pose goal, the two registries)."""
    if _IMPORT_SRC not in source:
        raise ValueError("goals.py: expected the generated import line")
    if not _TERMINAL_SRC.search(source):
        raise ValueError("goals.py: expected the generated task_terminal (pose goal, kso and to hard)")
    if "TERMINAL_GOALS = {'task': task_terminal}" not in source or "DISCRETE_GOALS = {'task': task_contacts}" not in source:
        raise ValueError("goals.py: expected the DISCRETE_GOALS / TERMINAL_GOALS registries")
    out = source.replace(_IMPORT_SRC, _IMPORT_DST, 1)
    out = _TERMINAL_SRC.sub(lambda _m: _TERMINAL_DST, out, count=1)
    # The tolerance constant goes after the imports (the first blank line after them).
    head, sep, rest = out.partition("\n\n")
    out = head + "\n" + _TOLERANCE_BLOCK + sep[1:] + rest if sep else out + _TOLERANCE_BLOCK
    out = out.replace("TERMINAL_GOALS = {'task': task_terminal}\n",
                      "TERMINAL_GOALS = {'task': task_terminal}\nOUTCOME_SCORERS = {'task': task_score}\n", 1)
    return out


def scene_ids(examples: str, wanted: Optional[List[str]]) -> List[str]:
    ids = sorted(d for d in os.listdir(examples)
                 if os.path.isfile(os.path.join(examples, d, "spec.json"))
                 and os.path.isdir(os.path.join(examples, d, "source")))
    if wanted:
        missing = sorted(set(wanted) - set(ids))
        if missing:
            raise SystemExit(f"not found under {examples}: {missing}")
        return [i for i in ids if i in set(wanted)]
    return ids


def register(examples: str, scene_id: str, dest_root: str = SCENES_DIR, force: bool = False,
             dry_run: bool = False) -> Tuple[str, str]:
    """Register one scene; returns ``(scene_id, outcome)`` with outcome
    ``registered`` / ``skipped`` (already there) / ``invalid: <why>``."""
    src = os.path.join(examples, scene_id, "source")
    dst = os.path.join(dest_root, scene_id)
    if os.path.isdir(dst) and not force:
        return scene_id, "skipped"
    for name in SOURCE_FILES + ("goals.py",):
        if not os.path.isfile(os.path.join(src, name)):
            return scene_id, f"invalid: source/{name} missing"
    with open(os.path.join(src, "goals.py")) as f:
        try:
            goals = transform_goals(f.read())
        except ValueError as e:
            return scene_id, f"invalid: {e}"
    with open(os.path.join(examples, scene_id, "spec.json")) as f:
        spec = json.load(f)
    if spec.get("id") not in (None, scene_id):
        return scene_id, f"invalid: spec id {spec.get('id')!r} differs from the folder name"
    if dry_run:
        return scene_id, "registered (dry run)"
    os.makedirs(dst, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    shutil.copyfile(os.path.join(examples, scene_id, "spec.json"), os.path.join(dst, "spec.json"))
    with open(os.path.join(dst, "goals.py"), "w") as f:
        f.write(goals)
    return scene_id, "registered"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--examples", required=True, help="folder of generated tasks")
    p.add_argument("ids", nargs="*", help="scene ids (folder names); --all for every one")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite scenes already registered")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.ids and not args.all:
        raise SystemExit("give scene ids or --all")
    examples = os.path.expanduser(args.examples)
    counts: dict = {}
    for sid in scene_ids(examples, None if args.all else args.ids):
        _, outcome = register(examples, sid, force=args.force, dry_run=args.dry_run)
        counts[outcome.split(":")[0]] = counts.get(outcome.split(":")[0], 0) + 1
        if not outcome.startswith("skipped"):
            print(f"{sid:<70} {outcome}")
    print("summary:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
