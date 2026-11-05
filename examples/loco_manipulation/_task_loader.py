"""
Resolve a loco-manipulation task from faro registry names.

A task is three strings:

    scene              -- key into scenes.common.registry.make_scene
    initial_condition  -- key into registry.get_initial_condition(scene, ...)
    description        -- LLM-facing prose (shinka-side prompt field only)

Discrete goals, terminal goals, and outcome scorers are
**scene-intrinsic**: each scene's ``goals.py`` exposes three dict
registries at module scope —

    DISCRETE_GOALS   : name -> factory(scene) -> DiscreteGoal
                       (stage-1 gate)
    TERMINAL_GOALS   : name -> factory(scene, resolved_state) -> TerminalGoal
                       (KSO / TO target; its kso= / to= modes pick
                       soft vs hard vs off per solver)
    OUTCOME_SCORERS  : name -> factory(scene, resolved_state) -> OutcomeScorer
                       (KSO cost)

The loader picks the first non-empty entry from each registry. A scene
with no ``goals.py`` (or only ``none``/empty factories) resolves to
``discrete_goal={}``, ``terminal_goal=None``, ``outcome_scorer=None``
— which makes stage-1 gating vacuous, applies no terminal cost, and
yields a constant zero IK cost.

Sliding contacts
----------------
faro scenes admit only sticking contacts (type 1) by default.
``load_task`` rebuilds the loaded scene with sliding (type 2) admitted on
every object and plane face (``with_sliding``), so plans may push or slide
objects and slide hands / feet along surfaces. Face IDs are unchanged;
faro's KSO and TO solve type-2 phases natively. The faro scene
files themselves are not modified.

Channel names vs faro interface keys
------------------------------------
faro keys contacts by owner: ``("ee", robot_id, patch)`` for robot
patches and ``("obj", name)`` for movable objects. The LLM-facing plan
keeps the flat single-robot channel names (``left_ee``, ``right_sole``,
``white_box``, ...) — the patch / object name alone. This module owns
both directions of that mapping: ``channel_name`` flattens a faro key,
and ``to_faro_plan`` converts a flat plan back via faro's own
``import_legacy_contacts`` at the solver boundary. Only single-robot
scenes are supported.

Consumers with no kwargs fall back to the
``SHINKA_LOCOMANIP_{SCENE,IC,DESCRIPTION}`` env vars (set by
``evaluate.main`` from Hydra kwargs, or by ``run_evo.py`` from its yaml
``task:`` block).
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from _run_common import NONE_STRS, TASK_ENV_VARS  # noqa: E402
import layouts  # noqa: E402

# Make the repo root importable so dotted faro paths resolve
# regardless of the cwd that called us.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_DEFAULTS = {
    "scene": "example_scene",
    "initial_condition": "default",
    "description": "",
}
_ENV = TASK_ENV_VARS
_NONE_STRS = NONE_STRS

# faro groups its scenes into families addressed by dotted
# module paths: corl_<x> -> motiondisco.<x>, faro_<x> -> faro.<x>. Old run
# dirs (.hydra/config.yaml, results paths) still carry the flat names.
_SCENE_FAMILY_PREFIXES = {"corl_": "motiondisco.", "faro_": "faro."}


def canonical_scene_name(name: str) -> str:
    """Map a pre-reorganization scene name onto its dotted module path.

    Returns *name* unchanged when it resolves as is (MotionDisco-side
    scenes under ``examples/loco_manipulation/scenes/`` keep bare names).
    """
    import importlib.util

    for prefix, family in _SCENE_FAMILY_PREFIXES.items():
        if not name.startswith(prefix) or "." in name:
            continue
        try:
            found = importlib.util.find_spec(f"scenes.{name}.scene") is not None
        except (ImportError, ValueError):
            found = False
        if found:
            return name
        alias = family + name[len(prefix):]
        logger.warning("scene %r renamed upstream; using %r", name, alias)
        return alias
    return name


@dataclass
class TaskContext:
    scene: Any
    resolved_state: Any
    robot_id: str
    discrete_goal: Dict[str, Tuple]
    terminal_goal: Any
    outcome_scorer: Any
    description: str
    scene_name: str
    ic_name: str
    discrete_goal_name: Optional[str]
    terminal_goal_name: Optional[str]
    outcome_scorer_name: Optional[str]
    # World offset applied to the scene's terminal goal / scorer by an IC
    # layout (``<ic>@<layout>``); zeros otherwise.
    goal_offset: Any = None


_cache: Dict[tuple, TaskContext] = {}


def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    return None if s in _NONE_STRS else s


def _pick(field: str, override: Optional[str]) -> str:
    val = _norm(override)
    if val is not None:
        return val
    val = _norm(os.environ.get(_ENV[field]))
    if val is not None:
        return val
    return _DEFAULTS[field]


CONTACT_TYPE_STICKING = 1
CONTACT_TYPE_SLIDING = 2
CONTACT_TYPES = (CONTACT_TYPE_STICKING, CONTACT_TYPE_SLIDING)


def with_sliding(scene: Any) -> Any:
    """Copy of *scene* with sliding (type 2) admitted on every face."""
    from dataclasses import replace

    from faro.configs.scene import Scene

    objects = {
        name: replace(cfg, contact_types=tuple(dict.fromkeys((*cfg.contact_types, *CONTACT_TYPES))))
        for name, cfg in scene.objects.items()
    }
    return Scene(
        name=scene.name,
        robots=scene.robots,
        objects=objects,
        include_robot_ees=scene.include_robot_ees,
    )


# Clearance the robot keeps from objects, fixtures and planes (metres):
# every capsule of the robot's collision model is inflated by it, so the
# solvers' distance constraints (dist >= 0, no margin of their own) hold the
# body that far off. Pairs in a declared contact (sole on the floor, hand on
# a box) are exempt as before. Capsule sizes are runtime parameters of the
# generated code, so no recompile. Off by default: the same capsules serve
# the self-collision pairs and the fixed initial state, and on the G1 even
# 1 cm made the standing start pose infeasible for KSO (layered_pallet
# gen_14's first window: no kinematic solution, where 0 solves it). The
# slack that lets a converged window clip a box (tolerance 0.1: 2.9 cm on
# gen_6) is taken out of the tolerance instead.
COLLISION_MARGIN_ENV = "SHINKA_COLLISION_MARGIN"
COLLISION_MARGIN = max(0.0, float(os.environ.get(COLLISION_MARGIN_ENV, "0") or 0.0))
# Capsules the margin applies to (comma-separated names of the robot's
# collision model; empty = all). The legs (shins, feet) tolerate a margin
# in the standing pose, the torso and arms do not.
COLLISION_MARGIN_CAPSULES = tuple(
    s.strip() for s in os.environ.get("SHINKA_COLLISION_MARGIN_CAPSULES", "").split(",") if s.strip())


# TO goal override. Each scene's goals.py decides how TO treats the
# terminal goal (to="hard" | "soft" | "off"; most MotionDisco scenes say
# "off" so TO is free to drift for trajectory diversity). SHINKA_TO_GOAL
# forces one treatment on every scene — the scene campaign uses "hard" so
# that a valid plan means the object reached its target (to_eval.goal_check
# gates on it) — and SHINKA_TO_GOAL_TOL (m, and rad for rotations) replaces
# the targets' TO tolerance, since faro's default is an exact equality that
# a forced hard goal could never meet.
TO_GOAL_ENV = "SHINKA_TO_GOAL"
TO_GOAL = os.environ.get(TO_GOAL_ENV, "").strip().lower() or None
TO_GOAL_TOL = float(os.environ.get("SHINKA_TO_GOAL_TOL", "0.05") or 0.05)


def with_to_goal(goal: Any, mode: Optional[str], tol: float, tolerances_cls: Any = None) -> Any:
    """``goal`` with its TO treatment set to ``mode`` and, for ``"hard"``,
    every target's TO tolerance set to ``tol`` (position in m, rotation in
    rad; joint tolerances kept); ``goal`` itself when ``mode`` is None."""
    if goal is None or not mode:
        return goal
    if mode not in ("hard", "soft", "off"):
        raise ValueError(f"{TO_GOAL_ENV} must be hard, soft or off, not {mode!r}")
    from dataclasses import replace

    if tolerances_cls is None:
        from faro.search.goal import GoalTolerances as tolerances_cls

    def retol(target: Any) -> Any:
        if mode != "hard":
            return target
        joints = getattr(getattr(target, "to_tolerance", None), "joints", 0.0)
        return replace(target, to_tolerance=tolerances_cls(position=float(tol), rotation=float(tol), joints=joints))

    return replace(goal, to=mode,
                   objects={k: retol(v) for k, v in (getattr(goal, "objects", None) or {}).items()},
                   robots={k: retol(v) for k, v in (getattr(goal, "robots", None) or {}).items()})


def with_collision_margin(scene: Any, margin: float, capsules: Optional[Tuple[str, ...]] = None) -> Any:
    """Copy of *scene* whose robots' collision capsules (all, or the named
    ones) are ``margin`` m thicker; *scene* itself when the margin is zero."""
    if margin <= 0.0:
        return scene
    from dataclasses import replace

    from faro.configs.scene import Scene

    only = set(COLLISION_MARGIN_CAPSULES if capsules is None else capsules)
    robots = {}
    for rid, cfg in scene.robots.items():
        cm = cfg.collision_model
        unknown = only - set(cm.capsules)
        if unknown:
            raise ValueError(f"robot {rid!r} has no collision capsules {sorted(unknown)}; known: {sorted(cm.capsules)}")
        caps = {name: (replace(cap, radius=float(cap.radius) + float(margin)) if not only or name in only else cap)
                for name, cap in cm.capsules.items()}
        robots[rid] = replace(cfg, collision_model=replace(cm, capsules=caps))
    return Scene(
        name=scene.name,
        robots=robots,
        objects=scene.objects,
        include_robot_ees=scene.include_robot_ees,
    )


def channel_name(key: Any) -> str:
    """Flatten a faro interface key to its LLM-facing channel name.

    ``("ee", robot_id, patch)`` -> ``patch``; ``("obj", name)`` -> ``name``.
    Bare strings pass through unchanged.
    """
    if isinstance(key, tuple):
        return key[-1]
    return key


def flat_contact_mode(contact_mode: Dict[Any, Tuple]) -> Dict[str, Tuple]:
    """Re-key a faro contact mode (one entry per interface) by channel name."""
    return {channel_name(k): tuple(v) for k, v in contact_mode.items()}


def scene_faces(scene: Any) -> Dict[str, List[Tuple[int, int, str]]]:
    """Group the scene's global face IDs by object, in global-ID order.

    Returns ``{object_name: [(env_id, self_id, face_name), ...]}`` where
    ``self_id`` is the object-local face index (planes have one face,
    ``surface``; boxes have ``top/left/front/right/back/bottom`` = 1..6).
    """
    face_names = {
        env_id: face for (_name, face), env_id in scene.faces.by_name.items()
    }
    grouped: Dict[str, List[Tuple[int, int, str]]] = {}
    for env_id in sorted(scene.faces.by_id):
        name, self_id = scene.faces.by_id[env_id]
        grouped.setdefault(name, []).append((env_id, self_id, face_names[env_id]))
    return grouped


def to_faro_plan(scene: Any, plan: Dict[str, List[Tuple]]) -> Dict[tuple, List[Tuple]]:
    """Convert a flat-channel plan to faro's owned interface keys.

    Raises ``ValueError`` for unknown or duplicate channels.
    """
    from faro.configs.identifiers import import_legacy_contacts

    return import_legacy_contacts(scene, plan)


def _pick_first_from_registry(
    registry: Dict[str, Any],
    is_empty_fn,
    scene_name: str,
    kind: str,
    *factory_args: Any,
) -> Tuple[Optional[str], Any]:
    """Pick the first non-empty entry from a scene's goal registry.

    ``registry`` maps name -> factory, called with ``factory_args``.
    ``is_empty_fn`` receives
    the built object and returns True when it should be skipped (e.g.
    an empty ``TerminalGoal`` or a ``DiscreteGoal`` with no discrete
    assignments). Returns ``(name, object)`` or ``(None, None)``.
    """
    non_empty: list[Tuple[str, Any]] = []
    for name, factory in registry.items():
        try:
            obj = factory(*factory_args)
        except Exception as exc:
            # Silently running without a goal would score every plan on
            # the wrong objective; fail loudly instead.
            raise RuntimeError(
                f"scene {scene_name!r}: {kind} factory {name!r} raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if is_empty_fn(obj):
            continue
        non_empty.append((name, obj))
    if not non_empty:
        return None, None
    if len(non_empty) > 1:
        logger.warning(
            "scene %r: %s registry has %d non-empty entries (%s); "
            "picking the first. Extend _task_loader if you need to "
            "select a specific one.",
            scene_name, kind, len(non_empty),
            [n for n, _ in non_empty],
        )
    return non_empty[0]


def _discover_scene_goals(
    scene_name: str,
    scene_obj: Any,
    resolved: Any,
) -> Tuple[Dict[str, Tuple], Optional[str], Any, Optional[str], Any, Optional[str]]:
    """Return ``(flat_discrete, d_name, terminal_goal, t_name, scorer, s_name)``.

    Reads the scene's ``goals.py`` module registries. Missing
    ``goals.py`` or empty registries resolve to empty/None. A terminal
    goal with no targets (e.g. ``TerminalGoal(kso="off", to="off")``)
    counts as empty.
    """
    try:
        goals_mod = importlib.import_module(f"scenes.{scene_name}.goals")
    except ModuleNotFoundError as exc:
        if exc.name == f"scenes.{scene_name}.goals":
            return {}, None, None, None, None, None
        raise

    disc_reg = getattr(goals_mod, "DISCRETE_GOALS", {}) or {}
    term_reg = getattr(goals_mod, "TERMINAL_GOALS", {}) or {}
    scorer_reg = getattr(goals_mod, "OUTCOME_SCORERS", {}) or {}

    d_name, d_goal = _pick_first_from_registry(
        disc_reg, lambda g: g.is_empty(), scene_name, "DISCRETE_GOALS",
        scene_obj,
    )
    t_name, t_goal = _pick_first_from_registry(
        term_reg, lambda g: g.is_empty(), scene_name, "TERMINAL_GOALS",
        scene_obj, resolved,
    )
    # OutcomeScorer is a Protocol (callable), no is_empty — take the
    # first factory unconditionally.
    s_name, s_obj = _pick_first_from_registry(
        scorer_reg, lambda _s: False, scene_name, "OUTCOME_SCORERS",
        scene_obj, resolved,
    )

    if d_goal is not None:
        alternatives = d_goal.normalized_alternatives()
        if len(alternatives) > 1:
            logger.warning(
                "scene %r: discrete goal %r has %d alternatives; "
                "gating against the first one only. Full "
                "disjunction semantics are not yet wired.",
                scene_name, d_name, len(alternatives),
            )
        first_alt = alternatives[0] if alternatives else {}
        # Keys in DiscreteGoal.discrete are faro interface keys; the
        # downstream gate only needs channel_name -> target tuple.
        flat = {}
        for k, v in first_alt.items():
            if not isinstance(v, tuple):
                # e.g. motiondisco.climb_table_up_down_with_box writes a bare
                # int for its box; faro documents it as invalid.
                logger.warning(
                    "scene %r: discrete goal %r entry %r=%r is not a "
                    "contact tuple; ignoring it",
                    scene_name, d_name, k, v,
                )
                continue
            flat[channel_name(k)] = v
    else:
        flat = {}

    return flat, d_name, t_goal, t_name, s_obj, s_name


def initial_spec(scene_name: str, ic: str, scene_obj: Any) -> Any:
    """The ``StateSpec`` of IC name *ic*: a registry IC, or
    ``<base_ic>@<layout>`` (the base IC with a ``layouts.py`` layout)."""
    from scenes.common import registry

    base_ic, layout_name = layouts.split_ic(ic)
    spec = registry.get_initial_condition(scene_name, base_ic, scene=scene_obj)
    if layout_name:
        spec = layouts.apply_layout(spec, layouts.get_layout(scene_name, layout_name))
    return spec


def load_task(
    scene: Optional[str] = None,
    initial_condition: Optional[str] = None,
    description: Optional[str] = None,
) -> TaskContext:
    """Resolve a task by name from the faro registry.

    Any kwarg set to None (or one of the yaml/resolver stringy-None
    forms like ``"None"``/``"null"``/``""``) falls back to the matching
    ``SHINKA_LOCOMANIP_*`` env var, then to the built-in default. Goals
    and scorers are discovered from the scene's ``goals.py``; see
    ``_discover_scene_goals``.
    """
    from scenes.common import registry

    s = canonical_scene_name(_pick("scene", scene))
    ic = _pick("initial_condition", initial_condition)

    if description is None:
        description = os.environ.get(_ENV["description"], _DEFAULTS["description"])
    desc = description or ""

    key = (s, ic, desc, COLLISION_MARGIN, COLLISION_MARGIN_CAPSULES, TO_GOAL, TO_GOAL_TOL)
    if key in _cache:
        return _cache[key]

    scene_obj = with_collision_margin(with_sliding(registry.make_scene(s)), COLLISION_MARGIN)
    base_ic, layout_name = layouts.split_ic(ic)
    base_spec = registry.get_initial_condition(s, base_ic, scene=scene_obj)
    layout = layouts.get_layout(s, layout_name) if layout_name else None
    resolved = scene_obj.resolve(initial_spec(s, ic, scene_obj))
    if layout is not None:
        problems = layouts.check_layout(scene_obj, resolved, scene_obj.resolve(base_spec))
        if problems:
            raise ValueError(f"layout {ic!r} of scene {s!r} is invalid: " + "; ".join(problems))
    if len(scene_obj.robots) != 1:
        raise ValueError(
            f"scene {s!r} has {len(scene_obj.robots)} robots; the "
            "loco_manipulation channel format supports exactly one."
        )
    robot_id = next(iter(scene_obj.robots))

    flat_goal, d_name, t_goal, t_name, scorer, s_name = _discover_scene_goals(
        s, scene_obj, resolved,
    )
    offset = None
    if layout is not None:
        # Scene goals are world positions laid out for the base IC: move
        # them with the layout (goal_follows / goal_shift) and rebind a
        # goal-based scorer to the moved goal.
        offset = layouts.goal_offset(layout, scene_obj.resolve(base_spec), resolved)
        t_goal = layouts.shift_goal(t_goal, offset)
        scorer = layouts.shift_scorer(scorer, offset)
    t_goal = with_to_goal(t_goal, TO_GOAL, TO_GOAL_TOL)

    ctx = TaskContext(
        scene=scene_obj,
        resolved_state=resolved,
        robot_id=robot_id,
        discrete_goal=flat_goal,
        terminal_goal=t_goal,
        outcome_scorer=scorer,
        description=desc,
        scene_name=s,
        ic_name=ic,
        discrete_goal_name=d_name,
        terminal_goal_name=t_name,
        outcome_scorer_name=s_name,
        goal_offset=offset,
    )
    _cache[key] = ctx
    return ctx


def scene_goal_factories(scene_name: str, terminal_goal_name: Optional[str],
                         outcome_scorer_name: Optional[str]) -> Tuple[Any, Any]:
    """The scene's ``TERMINAL_GOALS`` / ``OUTCOME_SCORERS`` factories by
    name (None when absent), to rebuild goals for a varied scene."""
    try:
        mod = importlib.import_module(f"scenes.{scene_name}.goals")
    except ModuleNotFoundError:
        return None, None
    tg = (getattr(mod, "TERMINAL_GOALS", {}) or {}).get(terminal_goal_name) if terminal_goal_name else None
    sc = (getattr(mod, "OUTCOME_SCORERS", {}) or {}).get(outcome_scorer_name) if outcome_scorer_name else None
    return tg, sc
