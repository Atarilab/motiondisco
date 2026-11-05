"""
Build the loco_manipulation system prompt by introspecting the active
task's faro scene.

Both the standalone runner (``run_evo.py``) and the Hydra task config
(``shinka/configs/task/loco_manipulation.yaml`` via the ``${py:...}``
resolver) call ``build_task_sys_msg(scene, ic, description)``. Unset
args fall back to the ``SHINKA_LOCOMANIP_*`` env vars and then to
built-in defaults (see ``_task_loader.load_task``). Discrete and
continuous goals are auto-discovered from ``scenes.<scene>.goals``.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

# Make sure the example dir is on sys.path so the relative import of
# _task_loader works whether we're invoked from the repo root, the
# example dir, or the Hydra resolver.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _task_loader import flat_contact_mode, load_task, scene_faces  # noqa: E402  (post-sys.path tweak)
from plan_signature import DIVERSITY_ENV  # noqa: E402
from scoring import score_mode as active_score_mode  # noqa: E402


# ---------------------------------------------------------------------------
# Scene introspection helpers
# ---------------------------------------------------------------------------

def _format_id_table(scene) -> str:
    """
    Pretty-print per-object feature tables so the LLM sees every
    env_id → object mapping explicitly (not just compressed ranges).
    Each row is tagged with faro's face name in the object's body
    frame (top / bottom / front / back / left / right; ``surface`` for
    planes).
    """
    lines = []
    for label, faces in scene_faces(scene).items():
        lines.append(f"  {label}:")
        for env_id, self_id, face in faces:
            lines.append(
                f"    env_id {env_id:<3} = {label}_feat{self_id}  (self_id {self_id})  [{face}]"
            )
    return "\n".join(lines)


def _format_channel_list(scene) -> str:
    """List the required channels for this scene."""
    base = ['"left_ee"', '"right_ee"', '"left_sole"', '"right_sole"']
    obj_channels = [f'"{name}"' for name in scene.env_objects.keys()]
    return ", ".join(base + obj_channels)


def _format_initial_mode(contact_mode: Dict[str, Tuple]) -> str:
    """Pretty-print the scene's initial mode (length-1 contact_mode_seq)."""
    lines = ["    contact_mode_seq = {"]
    width = max((len(repr(k)) for k in contact_mode), default=10)
    for ch, entry in contact_mode.items():
        lines.append(f"        {repr(ch):<{width}}: [{tuple(entry)!r}],")
    lines.append("    }")
    return "\n".join(lines)


def _format_movable_summary(scene) -> str:
    movables = list(scene.env_objects.keys())
    if not movables:
        return "(no movable objects in this scene)"
    return ", ".join(f'"{m}"' for m in movables)


def _format_scene_geometry(scene, resolved_state) -> str:
    """
    Render the scene's initial geometry (robot + object positions + sizes)
    so the LLM can reason about distances and reachability.
    """
    lines = ["The scene's initial geometry (meters; xyz is the body centre, "
             "half-extents are half the side lengths):"]
    rq = next(iter(resolved_state.robots.values())).q
    lines.append(
        f"  robot base: xyz=[{rq[0]:.2f}, {rq[1]:.2f}, {rq[2]:.2f}]"
    )

    def spans(q, half) -> str:
        return (f"x {q[0] - half[0]:.2f}..{q[0] + half[0]:.2f}, "
                f"y {q[1] - half[1]:.2f}..{q[1] + half[1]:.2f}, "
                f"top z {q[2] + half[2]:.2f}")

    for name, cfg in scene.fixed_env_objects.items():
        st = resolved_state.fixed_env_objects[name]
        # Planes have 2-D half_extents; report them as "(plane)" as before.
        half = list(cfg.half_extents) if len(cfg.half_extents) == 3 else None
        if half:
            size_str = f"half-extents={[round(v, 3) for v in half]} ({spans(st.q, half)})"
        else:
            size_str = "(plane)"
        lines.append(
            f"  {name} (fixed): xyz=[{st.q[0]:.2f}, {st.q[1]:.2f}, {st.q[2]:.2f}] "
            f"{size_str}"
        )
    for name, cfg in scene.env_objects.items():
        st = resolved_state.env_objects[name]
        half = list(cfg.half_extents)
        lines.append(
            f"  {name} (movable): xyz=[{st.q[0]:.2f}, {st.q[1]:.2f}, {st.q[2]:.2f}] "
            f"half-extents={[round(v, 3) for v in half]} "
            f"(footprint {2 * half[0]:.2f} x {2 * half[1]:.2f}, height {2 * half[2]:.2f})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Static prompt template — placeholders are filled at build time.
#
# Deliberately neutral: it states what the system allows and how plans are
# judged, but suggests no particular strategy (grasp, gait, order). Robot
# figures are approximate guidance; the solvers decide feasibility.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a contact planner for a humanoid robot. Given a task and a scene with
labeled surface IDs, you write Python code that builds a contact_mode_seq: a
dictionary describing how the robot's hands (end-effectors, "EEs"), feet
("soles") and the movable objects make and break contact over a sequence of
modes.

# Notation

EEs and soles:   (env_id, contact_type)          or () for free
Objects:         (self_id, env_id, contact_type) or () for free

env_id:          ID of the surface being contacted (see the ID table below)
self_id:         ID of the face on the object itself
contact_type:    1 = sticking: the two surfaces do not move relative to each
                     other
                 2 = sliding: the contact is kept while one surface slides
                     along the other (for example an object pushed across a
                     surface, or a hand or foot sliding along a face).
                     Friction limits the tangential force.
                 Both types are available on every surface in this scene.
                 Switching a contact between 1 and 2 on the same pair of
                 surfaces keeps it; it does not need to pass through ().

A mode is a full snapshot of all active contacts. The channels are left_ee,
right_ee, left_sole, right_sole, and one channel per movable object.

# System properties

- Channels are independent. A single mode may change any number of them.
  A mode that changes no channel is not a valid mode.
- Every channel in the returned dict must have the same length.
- Object channels are passive: they describe the object's own contact with
  a support. An object's support contact breaks when the object is lifted,
  and reappears only when it is placed on, or lands on, a surface. When it
  rests on another object, env_id is that object's face ID.
- EEs may contact environment surfaces as well as object surfaces.
- Soles may likewise be planted on an object's surface as well as on an
  environment surface. That object then acts as the support surface.
- No channel is reserved for a role. EE contacts are not restricted to
  manipulation and sole contacts are not restricted to locomotion; any channel
  may contact any surface within reach, for any purpose, including to add
  support. Nothing requires the two soles to change in the same mode, the two
  EEs to change in the same mode, or any limb to be free while another moves.

# Robot figures (approximate)

Rough values to orient planning. They are not hard limits: the IK and TO
solvers decide what is feasible, and plans near these values may still work.

  step length    about 0.4 m per walking step
  walking steps  count them from the geometry table for every move of the
                 base: at least ceil(distance / 0.4) + 1 steps, the distance
                 measured from the base's xy before the move to about 0.5 m
                 short of the next object or placement. A 2 m trip needs
                 walk(plan, 6); giving it 2-3 steps is the most common
                 reason TO fails (a foot would have to travel 1-2 m).
  base pose      base_xy ~ midpoint of the planted soles' xy positions
                 base_z  ~ support surface z + 0.77
  EE reach       about 0.8 m from the base to the contacted surface
  step height    about 0.45 m between support surfaces; a single 0.9 m
                 step is beyond the robot, use an intermediate surface
  stepping down  a drop of more than about 0.5 m is near the legs' limit;
                 give the solvers at least two walking steps on the upper
                 surface before the step down so they can position the
                 robot at the edge.
  mode duration  the trajectory optimizer allots about 1 s per mode

# Solver rules

- A contact must be released before it touches a different surface, for
  every channel: hands and soles (env_id, t) -> () -> (new_env_id, t), objects
  (self_id, env_id, t) -> () -> (self_id', env_id', t). A direct change is
  rejected without solving.
- A mode says WHICH surfaces touch, not where. The solvers choose foot and
  hand placement and the body path; the plan only decides what must be
  reachable next and in what order. Walking steps give the solver room to
  move: too few steps before a grasp, a placement or a step onto another
  surface is the most common cause of infeasibility.
- The fully expanded plan may have at most {max_modes} modes; longer plans are
  not evaluated.
- Long plans are evaluated window by window from the start. The feedback
  names the first window that fails and the modes before it that are
  solved; those modes are cached, so keep them exactly as they are and
  change the plan from the failing window on. A plan that gets further
  before failing scores higher than one that fails earlier.

# Helpers available in scope

get_initial_mode() -> Dict[str, List[Tuple]]
    Returns the scene's starting mode as a length-1 contact_mode_seq dict.
    Use it as mode 0 of your plan:
        plan = {{ch: list(seq) for ch, seq in get_initial_mode().items()}}
    The list(seq) copy is required so each channel is a mutable list the
    helpers can append to.

append_mode(plan, **channel_updates) -> None
    Appends exactly one new mode to `plan`. Any channel not mentioned keeps
    its previous value. Any sequence of contacts can be built from this.

walk(plan, n_steps) -> None
    Optional shorthand for plain stepping. Appends n_steps alternating
    single-support steps (right foot lifts first) and ends in double support
    on the same surfaces the feet started on; it appends n_steps + 1 modes.
    Non-sole channels keep their values throughout. It cannot change the
    support surface. Any other way of moving the feet can be written with
    append_mode.

# Feedback

"IK" below means the kinematic check (KSO). If it finds the expanded plan
infeasible, the evaluator locates the first break. Mode indices start at 0,
where mode 0 is the scene's initial state, and refer to the FULLY EXPANDED
plan:

    "KSO infeasible: first break at mode 20 of 34 (left_ee -> big_box left,
     right_ee -> big_box right); modes 0–19 solve on their own."

Solver feedback then explains that break, per mode and contact, in the plan's
own names: a contact that stays a distance away from the face it should touch
(gap) or is pushed into it, a sticking contact that slides, an object that is
dragged before anything touches it (the next contact is out of reach), a patch
that hangs over a face's edge, two bodies that penetrate each other, an object
resting on the wrong body (with the clearance needed), and for the trajectory
optimizer forces a contact cannot provide (more friction than available,
pulling instead of pushing, tipping over a patch edge) or joint / torque
limits. When every contact works but the goal target is missed, the final
position and the miss distance are reported instead. A repair probe may add
that inserting two walking steps before a given mode made the plan feasible;
that edit is verified. Problems that recur across repeated solves are listed
first; the solver is slightly noisy, so a problem seen in only one solve may
not be a flaw of the plan.

# How this is wired into ShinkaEvolve

Only generate_contact_plan(scene) inside the EVOLVE-BLOCK in initial.py is
mutated by the search. It must return a contact_mode_seq dict in the notation
above, with all channels the same length.

# Approach (a suggestion, not a requirement)

It often helps to decide first what should happen in the world, in whatever
terms the task suggests, and only then derive which contacts must be made and
broken, and in what order. If part of the idea turns out not to be realisable,
rethink the idea rather than force it.

# Output format

Briefly describe your plan, then emit the body of generate_contact_plan(scene).
It must start from get_initial_mode() and return the resulting dict.

# ---- Active task: {task_name} ----

{task_description}

{goal_section}

# ---- Scene geometry ----

{scene_geometry}

# ---- Active scene surface IDs ----

There are TWO distinct ID spaces in every contact tuple:

  env_id  (global feature index, used in EVERY tuple as the env_id)
  self_id (per-object face index, used only in object channels)

{id_table}

Face names are directions in the object's own frame: top / bottom = +Z / -Z,
front / back = +X / -X, left / right = +Y / -Y. Planes have a single
`surface` face.

Required channels: {channel_list}

The scene's initial mode (use this as mode 0):

{initial_mode}

Movable objects (each needs its own channel): {movable_summary}

{scoring_section}
"""


_SOLVER_MODE_TEXT = {
    "hard": ("the targets must be reached within the tolerance; if no solve with the "
             "hard target succeeds, a solve that only pulls toward it is tried, the miss "
             "distance is reported and the plan ranks below one that reaches the target"),
    "soft": "the solution is pulled toward the targets as a cost",
    "off": "the targets are not used",
}


def _format_terminal_goal(
    terminal_goal,
    terminal_goal_name: Optional[str],
) -> str:
    """Return a '# Goal targets' prompt block, or '' if absent.

    Lists TerminalGoal's owned targets (``robots`` keyed by robot ID,
    ``objects`` keyed by movable-object name; pose / configuration
    targets report their xyz) and states how each solver treats them
    (TerminalGoal.kso / .to: hard, soft or off).
    """
    if terminal_goal is None or terminal_goal.is_empty():
        return ""

    lines = ["# Goal targets"]
    if terminal_goal_name:
        lines.append(f"(goal: {terminal_goal_name})")
    lines.append("")
    for tgt in terminal_goal.robots.values():
        tx, ty, tz = tgt.value[:3]
        lines.append(
            f"  - robot base final xyz near ({tx:.3f}, {ty:.3f}, {tz:.3f})"
        )
    for obj_name, tgt in terminal_goal.objects.items():
        tx, ty, tz = tgt.value[:3]
        tol = getattr(getattr(tgt, "kso_tolerance", None), "position", None)
        lines.append(
            f"  - object {obj_name!r} final xyz near ({tx:.3f}, {ty:.3f}, {tz:.3f})"
            + (f" (within {float(tol):.2f} m)" if tol else "")
        )
    lines += [
        "",
        f"  IK solver: {_SOLVER_MODE_TEXT[terminal_goal.kso]}.",
        f"  TO solver: {_SOLVER_MODE_TEXT[terminal_goal.to]}.",
    ]
    return "\n".join(lines)


def _format_scoring_section(ctx) -> str:
    """'# ---- Scoring ----' block for task.score_mode=legacy."""
    gate = (
        "its final mode matches the discrete contact goal, "
        if ctx.discrete_goal else ""
    )
    return "\n".join([
        "# ---- Scoring ----",
        "",
        f"A plan scores 0 unless {gate}the IK solver finds the whole plan",
        "feasible and trajectory optimization (TO) then produces a cost.",
        "Feasible plans are ranked by that TO cost: lower constraint violation",
        "and smoother control rank higher. Plans identical to one already",
        "evaluated in this run, and plans over the mode cap, also score 0.",
    ])


def _format_quality_scoring_section(ctx) -> str:
    """'# ---- Scoring ----' block for task.score_mode=quality (scoring.py)."""
    zero = (
        ["  4. Score 0: the final mode does not match the discrete contact goal,",
         "     the plan duplicates an earlier one, or it is longer than the mode cap."]
        if ctx.discrete_goal else
        ["  4. Score 0: the plan duplicates an earlier one, or it is longer than",
         "     the mode cap."]
    )
    return "\n".join([
        "# ---- Scoring ----",
        "",
        "Plans are ranked in tiers; every plan in a tier outranks every plan in",
        "the tiers below it:",
        "",
        "  1. TO converged: ranked by how well the final state achieves the task,",
        "     and by dynamic feasibility and smoothness.",
        "  2. IK feasible, but TO did not converge.",
        "  3. IK infeasible: ranked by how many of the plan's contact events",
        "     (grasps, releases, placements, steps onto another surface; walking",
        "     steps do not count) lie before the first break. A plan whose contacts",
        "     all work but which misses the goal target ranks just below tier 2.",
        *zero,
    ])


_DIVERSITY_SECTION = """# ---- Strategy diversity ----

Plans are also ranked by how different they are from the plans already
evaluated in this run, and the best plan for each distinct set of contact
patches is kept. Two things are compared: WHICH contact patches a plan uses
(the faces the hands touch, the surfaces the feet and objects rest on,
sticking or sliding), which counts most, and the ORDER in which contacts are
made and broken, which counts less. Walking step counts do not count.
Feedback lists the patch sets and orders explored so far. A plan that is
different in these respects is valued even when its score is lower."""


def _format_discrete_goal(discrete_goal: dict) -> str:
    """Format the discrete contact goal for the LLM prompt."""
    if not discrete_goal:
        return (
            "# Discrete contact goal\n"
            "This scene has no discrete contact goal; any final mode is "
            "accepted."
        )
    lines = ["# Discrete contact goal",
             "The FINAL mode of your contact_mode_seq must satisfy:"]
    for channel, target_tuple in discrete_goal.items():
        if len(target_tuple) == 3:
            self_id, env_id, ct = target_tuple
            lines.append(
                f'  "{channel}": final entry must be {target_tuple}  '
                f"(self_id={self_id}, env_id={env_id}, contact_type={ct})"
            )
        elif len(target_tuple) == 2:
            env_id, ct = target_tuple
            lines.append(
                f'  "{channel}": final entry must be {target_tuple}  '
                f"(env_id={env_id}, contact_type={ct})"
            )
        else:
            lines.append(f'  "{channel}": final entry must be {target_tuple}')
    lines.append("")
    lines.append(
        "Only plans whose final mode matches every entry above are passed to "
        "the IK and TO solvers; any other plan scores 0."
    )
    return "\n".join(lines)


def build_task_sys_msg(
    scene: Optional[str] = None,
    initial_condition: Optional[str] = None,
    description: Optional[str] = None,
    score_mode: Optional[str] = None,
    diversity: Optional[str] = None,
) -> str:
    """
    Compose the system prompt for the active loco_manipulation task.

    Static template + scene-derived bits (ID table, channel list,
    initial mode, task name/description). Args are registry names
    (scene, IC) plus a free-form LLM-facing description. Goals are
    auto-discovered from the scene's ``goals.py``. None kwargs fall
    back to SHINKA_LOCOMANIP_* env vars and then to defaults; see
    ``_task_loader.load_task``. Both ``run_evo.py`` and the Hydra
    ``${py:...}`` resolver call this.

    ``score_mode`` / ``diversity`` mirror ``task.score_mode`` /
    ``task.diversity`` (strings, since the resolver passes strings);
    None falls back to SHINKA_SCORE_MODE / SHINKA_DIVERSITY. They select
    the scoring paragraph and add the diversity paragraph; the rest of
    the prompt is the same in every mode.
    """
    ctx = load_task(scene, initial_condition, description)
    mode = (str(score_mode).strip().lower() if score_mode not in (None, "", "None", "null")
            else active_score_mode())
    if diversity in (None, "", "None", "null"):
        diverse = os.environ.get(DIVERSITY_ENV) == "1"
    else:
        diverse = str(diversity).strip().lower() in ("1", "true", "yes", "on")
    scoring_section = (
        _format_scoring_section(ctx) if mode == "legacy"
        else _format_quality_scoring_section(ctx)
    )
    if diverse:
        scoring_section += "\n\n" + _DIVERSITY_SECTION

    if ctx.discrete_goal_name:
        task_name = f"{ctx.scene_name}/{ctx.ic_name}/{ctx.discrete_goal_name}"
    else:
        task_name = f"{ctx.scene_name}/{ctx.ic_name}"
    disc = _format_discrete_goal(ctx.discrete_goal)
    cont = _format_terminal_goal(ctx.terminal_goal, ctx.terminal_goal_name)
    goal_section = disc + ("\n\n" + cont if cont else "")

    return PROMPT_TEMPLATE.format(
        task_name=task_name,
        task_description=ctx.description,
        goal_section=goal_section,
        scene_geometry=_format_scene_geometry(ctx.scene, ctx.resolved_state),
        id_table=_format_id_table(ctx.scene),
        channel_list=_format_channel_list(ctx.scene),
        initial_mode=_format_initial_mode(
            flat_contact_mode(ctx.resolved_state.contact_mode)
        ),
        movable_summary=_format_movable_summary(ctx.scene),
        scoring_section=scoring_section,
        max_modes=os.environ.get("SHINKA_IK_MAX_NUM_NODES", "50"),
    )


# ---------------------------------------------------------------------------
# Full-rewrite instruction variants (EvolutionConfig.full_patch_variant_texts)
#
# shinka's built-in variants talk about algorithms and performance; these
# talk about contact strategy. Each keeps the <NAME>/<DESCRIPTION>/<CODE>
# delimiters shinka parses and the {language} placeholder.
# ---------------------------------------------------------------------------

_REWRITE_TAIL = """
You MUST respond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the plan you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
What happens in the world, step by step, and why this plan differs from the ones already explored.
</DESCRIPTION>

<CODE>
```{language}
# The new rewritten program here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code. Do not change the code outside of these markers.
* Make sure your rewritten program keeps the same inputs and outputs as the original program.
* Make sure the file still runs after your changes.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.
"""

FULL_REWRITE_VARIANTS = [
    ("Write a new plan that uses a SET of contact patches not in the explored list in "
     "the feedback: different faces for the hands, different support surfaces for the "
     "feet or the objects, or sticking versus sliding. Reusing an explored set with a "
     "new order is not the goal here." + _REWRITE_TAIL).strip(),
    ("Write a new plan that reaches the goal with a different ORDER of contact changes "
     "than the explored orders: what is made first, what is held while something else "
     "changes, what is released when. Reuse contact sets the feedback lists as feasible "
     "(score above zero) when that helps." + _REWRITE_TAIL).strip(),
    ("Write a new plan that assigns roles differently from the explored plans: which "
     "hand grasps, carries or supports; one hand versus two; which foot moves when; "
     "whether the body braces on the environment. Then finish the task in a way no "
     "listed plan does." + _REWRITE_TAIL).strip(),
]


def full_rewrite_variants() -> list:
    """Contact-strategy rewrite instructions for ``full_patch_variant_texts``."""
    return list(FULL_REWRITE_VARIANTS)
