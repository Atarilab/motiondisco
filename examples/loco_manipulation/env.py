"""
Adapter from the active loco_manipulation task's faro scene
to the dict-of-dicts representation that ``evaluate.py``'s structural
validators consume.

Nothing in this file is hardcoded per scene. ``DEFAULT_SCENE``, the
per-object surface dicts, and ``get_initial_mode()`` are all derived
at module load time from the scene returned by
``_task_loader.load_task()``.

Task selection flows through three env vars (set by ``evaluate.main``
from Hydra kwargs, or by ``run_evo.py`` from its yaml ``task:`` block):

    SHINKA_LOCOMANIP_SCENE             scene registry name
    SHINKA_LOCOMANIP_IC                initial-condition name
    SHINKA_LOCOMANIP_DESCRIPTION       LLM prose (prompt-only)

Scene + IC are resolved against the faro registry
(``scenes/<scene>/{initial_conditions,scene}.py``); discrete and
continuous goals are auto-discovered from ``scenes/<scene>/goals.py``.
Unset vars fall back to defaults; see ``_task_loader.load_task``.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Make sibling modules importable regardless of caller cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _task_loader import flat_contact_mode, load_task, scene_faces  # noqa: E402

SceneObject = Dict[str, Any]


# ---------------------------------------------------------------------------
# Scene introspection (rebuilds DEFAULT_SCENE from the live faro scene)
# ---------------------------------------------------------------------------

def _build_scene_object(label: str, faces: List[Tuple[int, int, str]],
                        is_movable: bool, obj_id: int) -> SceneObject:
    """One entry in the dict-style ``DEFAULT_SCENE`` list."""
    self_features = {f"feat_{self_id}": self_id for _, self_id, _ in faces}
    env_features = {f"feat_{self_id}": env_id for env_id, self_id, _ in faces}
    return {
        "id": obj_id,
        "label": label,
        "is_movable": is_movable,
        "self_features": self_features,
        "env_features": env_features,
    }


def _build_default_scene_from_task() -> List[SceneObject]:
    task = load_task()
    scene = task.scene
    grouped = scene_faces(scene)

    movable_labels = set(scene.env_objects.keys())
    objs: List[SceneObject] = []
    for next_id, (label, faces) in enumerate(grouped.items(), start=1):
        objs.append(
            _build_scene_object(
                label,
                faces,
                is_movable=label in movable_labels,
                obj_id=next_id,
            )
        )
    return objs


# Lazy: env vars (SHINKA_LOCOMANIP_SCENE etc.) are set by
# evaluate.main()'s _export_task_env *after* this module is imported, so
# eager resolution at import time would freeze DEFAULT_SCENE to the
# defaults (example_scene) before the active scene is known. Build on
# first call instead.
_default_scene_cache: Optional[List[SceneObject]] = None


def get_default_scene() -> List[SceneObject]:
    """Return the active scene's SceneObject list, building once per process."""
    global _default_scene_cache
    if _default_scene_cache is None:
        _default_scene_cache = _build_default_scene_from_task()
    return _default_scene_cache


# ---------------------------------------------------------------------------
# Public helpers (signatures preserved for evaluate.py)
# ---------------------------------------------------------------------------

def all_surface_ids(scene: List[SceneObject]) -> Set[int]:
    """Every global env_id that may appear as the env_id of a tuple."""
    out: Set[int] = set()
    for obj in scene:
        out.update(obj.get("env_features", {}).values())
    return out


def object_surface_ids(scene: List[SceneObject]) -> Dict[str, Set[int]]:
    """Per-object self_id sets — only the movable objects matter."""
    return {
        obj["label"]: set(obj.get("self_features", {}).values())
        for obj in scene
    }


def movable_object_labels(scene: List[SceneObject]) -> List[str]:
    """Labels that need their own channel in contact_mode_seq."""
    return [obj["label"] for obj in scene if obj.get("is_movable")]


def get_initial_mode() -> Dict[str, List[Tuple]]:
    """
    Return the active scene's initial contact mode as a length-1
    contact_mode_seq dict: each channel maps to a single-entry list so
    the result can be extended by appending further modes.

    Thin wrapper around ``resolved_state.contact_mode``, re-keyed from
    faro's ``("ee", robot, patch)`` / ``("obj", name)`` keys to flat
    channel names; here for convenience so initial.py can
    ``from env import get_initial_mode`` without touching the task
    loader directly.
    """
    cm = flat_contact_mode(load_task().resolved_state.contact_mode)
    return {ch: [entry] for ch, entry in cm.items()}


def append_mode(plan: Dict[str, List[Tuple]], **channel_updates) -> None:
    """
    Append exactly one new mode to ``plan``. Any channel not mentioned
    in ``channel_updates`` keeps its previous value (inheritance), so
    grasps and object-carry states persist automatically.

    Mutates ``plan`` in place. ``plan`` must already contain at least
    one mode (start from ``get_initial_mode()``). Updates that name a
    channel not present in ``plan`` raise ``ValueError``.
    """
    if not plan or not next(iter(plan.values()), None):
        raise ValueError(
            "plan must contain the initial mode before append_mode "
            "(seed it with get_initial_mode())"
        )
    unknown = set(channel_updates) - set(plan)
    if unknown:
        raise ValueError(
            f"append_mode got unknown channel(s): {sorted(unknown)}; "
            f"known channels: {sorted(plan)}"
        )
    for ch, seq in plan.items():
        seq.append(channel_updates.get(ch, seq[-1]))


def walk(plan: Dict[str, List[Tuple]], n_steps: int) -> None:
    """
    Append a standard alternating-foot walking pattern of ``n_steps``
    steps to ``plan``, ending in double support. Mutates ``plan`` in
    place.

    Pattern (n_steps = 4 example, soles only; non-sole channels
    inherit, so an active grasp / carried object persist through
    walking)::

        left_sole:  (1,1), (),    (1,1), (),    (1,1)
        right_sole: (),    (1,1), (),    (1,1), (1,1)

    The right foot lifts first; the two soles alternate single-support
    with no intermediate double-support modes; the final appended mode
    is double support on the same support env_ids the feet had before
    walk was called. ``n_steps`` steps appends ``n_steps + 1`` modes.
    Calling with ``n_steps <= 0`` is a no-op.

    The feet stay on the support surfaces they were already on — IK
    and TO decide the actual xy displacement. To change support
    surfaces (step up onto a box, step down to the floor), use
    ``append_mode`` directly.
    """
    if n_steps <= 0:
        return
    if "left_sole" not in plan or "right_sole" not in plan:
        raise ValueError(
            "walk requires 'left_sole' and 'right_sole' channels in plan"
        )
    last_left = plan["left_sole"][-1]
    last_right = plan["right_sole"][-1]
    for i in range(n_steps):
        if i % 2 == 0:
            append_mode(plan, left_sole=last_left, right_sole=())
        else:
            append_mode(plan, left_sole=(), right_sole=last_right)
    append_mode(plan, left_sole=last_left, right_sole=last_right)
