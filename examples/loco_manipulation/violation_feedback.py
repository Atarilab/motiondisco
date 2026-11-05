"""
Plain-language translation of KSO / TO constraint violations for the LLM.

faro reports solver violations under encoded names: robot contact patches
and collision pairs go through ``faro.configs.identifiers.solver_name``
(hex tokens, or a SHA-256 digest once the name passes 120 chars, which
is every robot-link collision pair). This module maps them back to the
plan's own vocabulary — mode index, contact channel (``left_ee``,
``big_box``), face names, link names — and writes one sentence per
problem plus a hint per kind of problem.

Two entry points produce ``Violation`` records:

    translate_kso(perf_metrics, resolver)
        acados rows ``k=<node>:<role>:<name>[<row>]`` from
        ``KsoTask.get_performance_metrics()``; KSO node k is plan mode k
        (a terminal goal adds one node that holds the last mode).
    translate_to(rows, resolver, nodes_per_mode)
        moto rows ``{node, name, idx, kind, violation}`` from
        ``Moto.filtered_constraint_violations``; mode = node // nodes_per_mode.

``summarize(violations)`` turns them into feedback lines. Row meanings
follow faro's constraint builders (``ocp/kso_contact``,
``ocp/moto_contact``). TO inequality values include the interior-point
slack, so they over-state small violations; equalities are exact.

Nothing here imports faro at module load, so it is testable with stubs.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPORT_TOL = 1e-2  # same cutoff KSO uses for feasibility
# TO inequality values include the interior-point slack; below moto's
# primal feasibility tolerance (faro default 0.05) they are not evidence
# of a real violation.
TO_INEQ_TOL = 5e-2


@dataclass
class Violation:
    solver: str            # "kso" | "to"
    mode: Optional[int]    # plan mode index (0 = initial state)
    subject: str           # channel / object / robot / collision body
    family: str            # see _SENTENCES
    value: float           # magnitude in ``unit``
    unit: str              # "m" | "rad" | "N" | "N·m" | ""
    sign: float = 1.0      # sign of the raw residual where it matters
    other: str = ""        # collision partner / goal component / joint
    raw: str = ""          # original solver name, for debugging
    attempts: int = 1      # in how many repeated solves it appeared

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

_HEX = re.compile(r"^(?:[0-9a-f]{2})+$")


def decode_solver_name(name: str) -> Optional[List[str]]:
    """``n_<hex>_<hex>...`` -> its parts; None when not decodable (digest)."""
    if not name.startswith("n_"):
        return None
    toks = name[2:].split("_")
    if len(toks) == 1 and len(toks[0]) == 64:
        return None  # SHA-256 digest: only resolvable via the scene index
    parts = []
    for tok in toks:
        if not _HEX.match(tok):
            return None
        try:
            parts.append(bytes.fromhex(tok).decode("utf-8"))
        except UnicodeDecodeError:
            return None
    return parts


def _readable_capsule(cap: str) -> str:
    """'l_forearm' -> 'left forearm'."""
    side = {"l": "left", "r": "right"}
    head, _, rest = cap.partition("_")
    if head in side and rest:
        return f"{side[head]} {rest.replace('_', ' ')}"
    return cap.replace("_", " ")


class NameResolver:
    """Maps solver identifiers of one scene + plan to readable names."""

    def __init__(self, scene: Any, plan: Dict[str, List[Tuple]],
                 collision_index: Any = None, resolved_state: Any = None) -> None:
        self.plan = plan
        self.n_modes = len(next(iter(plan.values()), [])) if plan else 0
        self.interfaces: Dict[str, str] = {}
        self.objects: set = set()
        self.face_env: Dict[int, str] = {}
        self.face_self: Dict[str, Dict[int, str]] = {}
        self.face_owner: Dict[int, str] = {}
        # name -> (cx, cy, cz, hx, hy, hz) of every box (fixed and movable)
        # at the scene's initial state, for clearance geometry; only fixed
        # boxes keep that pose, so spans are quoted for those alone.
        self.boxes: Dict[str, Tuple[float, ...]] = {}
        self.fixed_boxes: set = set()
        # env face id -> world z of the face centre (initial state)
        self.face_heights: Dict[int, float] = {}
        self.pairs: Dict[str, Tuple[str, str, str]] = {}
        if scene is not None:
            self._from_scene(scene, resolved_state)
        if collision_index is not None:
            self._from_collision_index(collision_index)

    def _from_scene(self, scene: Any, resolved_state: Any = None) -> None:
        from faro.configs.identifiers import interface_name

        from _task_loader import channel_name, scene_faces

        for key in getattr(scene, "active_patches", ()):
            self.interfaces[interface_name(key)] = channel_name(key)
        self.objects = set(getattr(scene, "env_objects", {}) or {})
        for obj, faces in scene_faces(scene).items():
            for env_id, self_id, face in faces:
                self.face_env[env_id] = f"{obj} {face}"
                self.face_owner[env_id] = obj
                self.face_self.setdefault(obj, {})[self_id] = face
        if resolved_state is None:
            return
        try:
            import kso_warm_start

            self.face_heights = {env_id: float(c[2]) for env_id, c
                                 in kso_warm_start.face_centres(scene, resolved_state).items()}
        except Exception:  # noqa: BLE001 - heights only feed one diagnosis
            self.face_heights = {}
        for group, states in (("fixed_env_objects", getattr(resolved_state, "fixed_env_objects", {})),
                              ("env_objects", getattr(resolved_state, "env_objects", {}))):
            for name, cfg in (getattr(scene, group, {}) or {}).items():
                half = list(getattr(cfg, "half_extents", []) or [])
                st = states.get(name) if isinstance(states, dict) else None
                if len(half) != 3 or st is None:
                    continue
                q = list(st.q[:3])
                self.boxes[name] = tuple(float(v) for v in (*q, *half))
                if group == "fixed_env_objects":
                    self.fixed_boxes.add(name)

    def touching(self, subject: str, mode: Optional[int]) -> List[str]:
        """Non-object channels in contact with a face of *subject* at
        *mode* or one mode on either side."""
        if mode is None:
            return []
        out = set()
        for ch, seq in self.plan.items():
            if ch == subject or ch in self.objects or not isinstance(seq, list):
                continue
            for m in (mode - 1, mode, mode + 1):
                if 0 <= m < len(seq):
                    e = tuple(seq[m])
                    if e and len(e) == 2 and self.face_owner.get(e[0]) == subject:
                        out.add(ch)
        return sorted(out)

    def _from_collision_index(self, index: Any) -> None:
        for group, pairs in index.groups.items():
            for name, pd in pairs.items():
                a = _readable_capsule(pd.local_i) if pd.robot_i else pd.name_i
                if group in ("self_coll", "robot_robot"):
                    b = _readable_capsule(pd.local_j or pd.name_j)
                else:
                    b = pd.local_j or pd.name_j
                self.pairs[name] = (group, a, b)

    # -- lookups -----------------------------------------------------------
    def subject(self, prefix: str) -> str:
        """Contact-interface prefix -> channel / object name."""
        if prefix in self.interfaces:
            return self.interfaces[prefix]
        if prefix.startswith("obj_") and prefix[4:] in self.objects:
            return prefix[4:]
        parts = decode_solver_name(prefix)
        if parts and parts[0] == "ee":
            return parts[-1]
        return prefix

    def pair(self, name: str) -> Tuple[str, str, str]:
        """Collision pair name -> (group, body a, body b)."""
        if name in self.pairs:
            return self.pairs[name]
        parts = decode_solver_name(name)
        if parts and parts[0] == "collision" and len(parts) >= 4:
            a, b = parts[2], parts[3]
            return parts[1], self._capsule_or(a), self._capsule_or(b)
        return "", "a robot link", "another body"

    def _capsule_or(self, part: str) -> str:
        inner = decode_solver_name(part)
        if inner and inner[0] == "capsule":
            return _readable_capsule(inner[-1])
        return part

    def contact(self, subject: str, mode: Optional[int]) -> str:
        """The face *subject* touches at *mode* per the plan ('big_box left')."""
        return self.contact_faces(subject, mode)[1]

    def contact_face_id(self, subject: str, mode: Optional[int]) -> Optional[int]:
        """Env face ID *subject* touches at *mode* (previous mode's contact
        when it is free there, as in ``contact_faces``), or None."""
        seq = self.plan.get(subject)
        if not seq or mode is None or not 0 <= mode < len(seq):
            return None
        entry = tuple(seq[mode]) or (tuple(seq[mode - 1]) if mode > 0 else ())
        if not entry:
            return None
        return int(entry[1] if len(entry) == 3 else entry[0])

    def contact_faces(self, subject: str, mode: Optional[int]) -> Tuple[str, str]:
        """(own face, touched face) of *subject* at *mode*; own face is only
        known for objects ('bottom', 'floor surface'). Empty when free.

        KSO constrains the union of the incoming and outgoing contacts at
        each transition, so a violation reported at a mode where the
        channel is free refers to the contact released there: fall back to
        the previous mode's contact."""
        seq = self.plan.get(subject)
        if not seq or mode is None or not 0 <= mode < len(seq):
            return "", ""
        entry = tuple(seq[mode])
        if not entry and mode > 0:
            entry = tuple(seq[mode - 1])
        if not entry:
            return "", ""
        if len(entry) == 3:  # object: (self face, env face, type)
            own = self.face_self.get(subject, {}).get(entry[0], f"face {entry[0]}")
            return own, self.face_env.get(entry[1], f"face {entry[1]}")
        return "", self.face_env.get(entry[0], f"face {entry[0]}")


# ---------------------------------------------------------------------------
# Row meanings
# ---------------------------------------------------------------------------

# KSO kin_eqs (ocp/kso_contact): [0] normal gap, [1-2] tilt, [3-4] slip
# between mode k and k+1, [5] normal drift, [6-8] rotation drift. The
# terminal node only has [0-2].
_KSO_KIN_EQS = {0: ("gap", "m"), 1: ("tilt", "rad"), 2: ("tilt", "rad"),
                3: ("slip", "m"), 4: ("slip", "m"), 5: ("drift", "m"),
                6: ("twist", "rad"), 7: ("twist", "rad"), 8: ("twist", "rad")}
# TO kin_eqs (ocp/moto_contact): [0-1] slip, [2] gap, [3-4] tilt, [5] yaw twist.
_TO_KIN_EQS = {0: ("slip", "m"), 1: ("slip", "m"), 2: ("gap", "m"),
               3: ("tilt", "rad"), 4: ("tilt", "rad"), 5: ("twist", "rad")}
# Sliding variant drops the two slip rows.
_TO_SLIDING_KIN_EQS = {0: ("gap", "m"), 1: ("tilt", "rad"), 2: ("tilt", "rad"),
                       3: ("twist", "rad")}
# Patch wrench cone: [0-3] friction pyramid, [4-5] torsion, [6-9] centre of
# pressure inside the patch, [10] normal force >= 0.
_TO_CTRL_INEQS = {**{i: ("friction", "N") for i in range(4)},
                  4: ("torsion", "N·m"), 5: ("torsion", "N·m"),
                  **{i: ("tipping", "N·m") for i in range(6, 10)},
                  10: ("pulling", "N")}

_CONTACT_SUFFIXES = (  # longest first
    ("_sliding_ctrl_ineqs", "sliding"), ("_sliding_ctrl_eqs", "sliding"),
    ("_sliding_kin_eqs", "sliding_kin"), ("_ctrl_ineqs", "ctrl_ineqs"),
    ("_ctrl_eqs", "switch_force"), ("_kin_ineqs", "overhang"), ("_kin_eqs", "kin_eqs"),
)

_ROBOT_FIELDS = {
    "Momentum_constr": ("dynamics", ""), "Power_limit_pos": ("torque", "N·m"),
    "Power_limit_neg": ("torque", "N·m"), "joint_pos_bound": ("joint_limit", "rad"),
    "joint_vel_bound": ("joint_speed", "rad/s"),
}

_GOAL_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _goal(parts: Sequence[str], row: int, value: float, solver: str,
          mode: Optional[int], raw: str) -> Optional[Violation]:
    """Hard terminal-goal rows: solver_name(kind, owner, 'goal'...)."""
    if len(parts) < 3 or not parts[2].startswith("goal"):
        return None
    owner = parts[1] if parts[0] == "object" else "robot base"
    axis = _GOAL_AXES[row] if row < len(_GOAL_AXES) and "bounds" not in parts[2] else f"component {row}"
    return Violation(solver, mode, owner, "goal", value, "m" if row < 3 else "rad",
                     other=axis, raw=raw)


# ---------------------------------------------------------------------------
# KSO
# ---------------------------------------------------------------------------

_KSO_ROW = re.compile(r"^k=(\d+):(\w+):(.*)\[(\d+)\]$")


def translate_kso(perf_metrics: Dict[str, Any], resolver: NameResolver,
                  tol: float = REPORT_TOL) -> List[Violation]:
    """KSO performance metrics -> violations above *tol*, worst first."""
    # faro returns numpy arrays here: no truthiness tests.
    names = list(perf_metrics.get("sorted_vio_names", []))
    vios = [float(x) for x in perf_metrics.get("sorted_vios", [])]
    details = perf_metrics.get("sorted_vio_details")
    details = list(details) if details is not None else [None] * len(names)
    out: List[Violation] = []
    for value, name, det in zip(vios, names, details):
        if value < tol:
            break
        m = _KSO_ROW.match(name)
        if not m:
            out.append(Violation("kso", None, name, "other", float(value), "", raw=name))
            continue
        k, role, body, row = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        mode = min(k, resolver.n_modes - 1) if resolver.n_modes else k
        signed = float(det["value"]) if isinstance(det, dict) and "value" in det else float(value)
        sign = 1.0 if signed >= 0 else -1.0
        v = _kso_violation(body, row, role, mode, float(value), sign, resolver, name)
        out.append(v)
    return out


def _kso_violation(body: str, row: int, role: str, mode: Optional[int], value: float,
                   sign: float, resolver: NameResolver, raw: str) -> Violation:
    if body.startswith("Stage_") and body.endswith(("_kin_eqs", "_kin_ineqs")):
        ineq = body.endswith("_kin_ineqs")
        prefix = body[len("Stage_"):-len("_kin_ineqs" if ineq else "_kin_eqs")]
        subject = resolver.subject(prefix)
        if ineq:
            return Violation("kso", mode, subject, "overhang", value, "m",
                             other=resolver.contact(subject, mode), raw=raw)
        family, unit = _KSO_KIN_EQS.get(row, ("contact", ""))
        return Violation("kso", mode, subject, family, value, unit, sign=sign,
                         other=resolver.contact(subject, mode), raw=raw)
    if body.endswith("_distance"):
        group, a, b = resolver.pair(body[: -len("_distance")])
        return Violation("kso", mode, a, "collision", value, "m", other=b, raw=raw)
    parts = decode_solver_name(body)
    if parts:
        goal = _goal(parts, row, value, "kso", mode, raw)
        if goal is not None:
            return goal
    return Violation("kso", mode, "/".join(parts) if parts else body, "other", value, "", raw=raw)


# ---------------------------------------------------------------------------
# TO
# ---------------------------------------------------------------------------

def translate_to(rows: Iterable[Dict[str, Any]], resolver: NameResolver,
                 nodes_per_mode: int, tol: float = REPORT_TOL,
                 ineq_tol: float = TO_INEQ_TOL) -> List[Violation]:
    """moto violation rows -> violations above *tol* (*ineq_tol* for
    inequalities, whose values include slack), worst first."""
    out: List[Violation] = []
    for r in sorted(rows, key=lambda r: -float(r["violation"])):
        value = float(r["violation"])
        if value < tol:
            break
        if r.get("kind") == "ineq" and value < ineq_tol:
            continue
        mode = int(r["node"]) // max(1, nodes_per_mode)
        if resolver.n_modes:
            mode = min(mode, resolver.n_modes - 1)
        sign = 1.0 if float(r.get("raw", value)) >= 0 else -1.0
        out.append(_to_violation(str(r["name"]), int(r["idx"]), mode, value, sign, resolver))
    return out


def _to_violation(name: str, row: int, mode: int, value: float, sign: float,
                  resolver: NameResolver) -> Violation:
    if name == "dynamics":
        return Violation("to", mode, "robot", "dynamics", value, "", raw=name)
    if name == "time_scale_bound":
        return Violation("to", mode, "timing", "timing", value, "", raw=name)
    if name.endswith("_obj_f_eq_ma"):
        obj = name[: -len("_obj_f_eq_ma")]
        return Violation("to", mode, obj, "object_balance", value, "N" if row < 3 else "N·m", raw=name)
    if name.endswith("_distance"):
        _group, a, b = resolver.pair(name[: -len("_distance")])
        return Violation("to", mode, a, "collision", value, "m", other=b, raw=name)
    for suffix, kind in _CONTACT_SUFFIXES:
        if name.endswith(suffix):
            subject = resolver.subject(name[: -len(suffix)])
            if kind == "kin_eqs":
                family, unit = _TO_KIN_EQS.get(row, ("contact", ""))
            elif kind == "sliding_kin":
                family, unit = _TO_SLIDING_KIN_EQS.get(row, ("contact", ""))
            elif kind == "ctrl_ineqs":
                family, unit = _TO_CTRL_INEQS.get(row, ("friction", "N"))
            elif kind == "overhang":
                family, unit = "overhang", "m"
            else:
                family, unit = kind, "N"
            return Violation("to", mode, subject, family, value, unit, sign=sign,
                             other=resolver.contact(subject, mode), raw=name)
    parts = decode_solver_name(name)
    if parts and parts[0] == "robot" and len(parts) >= 3:
        family, unit = _ROBOT_FIELDS.get(parts[2], ("other", ""))
        return Violation("to", mode, "robot", family, value, unit, raw=name)
    if parts:
        goal = _goal(parts, row, value, "to", mode, name)
        if goal is not None:
            return goal
    return Violation("to", mode, "/".join(parts) if parts else name, "other", value, "", raw=name)


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

def _fmt(v: Violation) -> str:
    if v.unit == "rad":
        return f"{math.degrees(v.value):.0f}°"
    if v.unit == "m":
        return f"{v.value:.2f} m" if v.value >= 0.01 else f"{v.value * 1000:.0f} mm"
    if v.unit:
        return f"{v.value:.3g} {v.unit}"
    return f"{v.value:.3g}"


def _sentence(v: Violation, resolver: NameResolver) -> str:
    where = (v.other if v.family not in ("collision", "goal") else "") or resolver.contact(v.subject, v.mode)
    on = f" on {where}" if where else ""
    surface = where or "its contact surface"
    amount = _fmt(v)
    f = v.family
    if f == "gap":
        own, _ = resolver.contact_faces(v.subject, v.mode)
        if v.sign >= 0 and own:
            return (f"{v.subject} should rest with its {own} face on {surface} "
                    f"but stays {amount} away from it")
        if v.sign >= 0:
            return f"{v.subject} should touch {surface} but stays {amount} away from it"
        return f"{v.subject} is pushed {amount} into {surface}"
    if f == "tilt":
        return f"{v.subject} is tilted {amount} against {surface} (cannot lie flat on it)"
    if f == "reach":
        return (f"{v.subject} moves {amount} on {surface} before anything touches it: the "
                f"solver dragged it to make a later contact reachable (an object only moves "
                f"while a hand holds it)")
    if f == "slip":
        return f"{v.subject} slides {amount} along {surface} although the contact is sticking"
    if f == "drift":
        return f"{v.subject} moves {amount} off {surface} while it should stay in contact"
    if f == "twist":
        return f"{v.subject} rotates {amount} on {surface} while it should stay in contact"
    if f == "overhang":
        return f"{v.subject} hangs {amount} over the edge of {surface}"
    if f == "collision":
        return f"{v.subject} penetrates {v.other} by {amount}"
    if f == "goal":
        return f"final {v.subject} position is off target by {amount} ({v.other})"
    if f == "friction":
        return f"{v.subject}{on} needs more friction than available ({amount} outside the friction cone)"
    if f == "torsion":
        return f"{v.subject}{on} has to resist a twisting moment larger than friction allows ({amount})"
    if f == "tipping":
        return f"{v.subject}{on} would tip over the edge of its contact patch ({amount})"
    if f == "pulling":
        return f"{v.subject}{on} would have to pull ({amount}); contacts can only push"
    if f == "switch_force":
        return f"{v.subject}{on} still carries force at the moment it switches contact ({amount})"
    if f == "sliding":
        return f"{v.subject}{on}: sliding friction is inconsistent with the motion ({amount})"
    if f == "object_balance":
        return (f"forces on {v.subject} do not balance ({amount}): its contacts cannot "
                f"hold or support it")
    if f == "torque":
        return f"a joint torque / power limit is exceeded ({amount})"
    if f == "joint_limit":
        return f"a joint position limit is exceeded ({amount})"
    if f == "joint_speed":
        return f"a joint speed limit is exceeded ({amount})"
    if f == "timing":
        return "a mode's duration hits its limit (mode too short or too long for the motion)"
    if f == "dynamics":
        return f"the robot motion is inconsistent with its dynamics (residual {amount})"
    return f"constraint {v.subject} violated ({amount})"


_HINTS = {
    "gap": "a contact that cannot be reached usually means the body is in the wrong place "
           "(walk closer, or put the object down somewhere it can rest)",
    "collision": "move the robot or the object so the bodies are clear of each other "
                 "before making the contact",
    "slip": "use a sliding contact (type 2) or re-grasp instead of keeping a sticking contact",
    "reach": "the next contact with that object is out of reach from where the robot "
             "stands; step closer or climb onto the same surface before making it",
    "overhang": "the patch is too far toward the face's edge; approach so it sits fully on the face",
    "tilt": "the face orientation cannot be matched; use a different face or approach direction",
    "friction": "add support (second hand / other face) or move the load closer to the body",
    "pulling": "contacts can only push; grasp opposite faces or support the object from below",
    "tipping": "the load is too far off-centre for that contact; use a wider stance or two hands",
    "object_balance": "the object needs contacts that can carry its weight (opposite faces or from below)",
    "torque": "the load or posture is too demanding; keep the object close to the body",
    "joint_limit": "the target is out of reach; step closer before the contact",
    "goal": "the final placement does not reach the target; adjust where you walk or place",
}


# A sole gap at a level change this large is a step down near the legs'
# limit (0.73 m on the G1 needs a deep fold of the upper leg; see
# kso_warm_start).
STEP_DOWN_MIN = 0.5


def _step_down(
    violations: Sequence[Violation], resolver: NameResolver,
) -> Tuple[List[str], set]:
    """A sole that cannot reach a face far below the other sole's face.
    Returns the diagnosis lines and the problem keys they explain."""
    out, explained, seen = [], set(), set()
    if not resolver.face_heights:
        return out, explained
    for g in violations:
        if g.family != "gap" or g.sign < 0 or not g.subject.endswith("sole") or g.mode is None:
            continue
        other = "left_sole" if g.subject == "right_sole" else "right_sole"
        lo, hi = resolver.contact_face_id(g.subject, g.mode), resolver.contact_face_id(other, g.mode)
        if lo is None or hi is None or lo not in resolver.face_heights or hi not in resolver.face_heights:
            continue
        drop = resolver.face_heights[hi] - resolver.face_heights[lo]
        if drop < STEP_DOWN_MIN:
            continue
        explained.add(_problem_key(g))
        if (g.subject, lo, hi) in seen:
            continue
        seen.add((g.subject, lo, hi))
        gs = [x for x in violations if _problem_key(x) == _problem_key(g)]
        hi_name, lo_name = resolver.face_env.get(hi, f"face {hi}"), resolver.face_env.get(lo, f"face {lo}")
        out.append(
            f"{_ranges([x.mode for x in gs if x.mode is not None])}: {g.subject} has to drop "
            f"{drop:.2f} m from {hi_name} to {lo_name} and stops {_fmt(g)} short. A drop that "
            f"large is near the legs' limit; give the robot at least two walking steps on "
            f"{hi_name} before the step down so it can reach the edge (climbing down backwards, "
            f"facing {hi_name}, also works)"
        )
    return out, explained


def _ranges(modes: Sequence[int]) -> str:
    modes = sorted(set(modes))
    spans, start, prev = [], modes[0], modes[0]
    for m in modes[1:]:
        if m != prev + 1:
            spans.append((start, prev))
            start = m
        prev = m
    spans.append((start, prev))
    txt = ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in spans)
    return ("mode " if len(modes) == 1 else "modes ") + txt


def _resting_on_wrong_body(
    violations: Sequence[Violation], resolver: NameResolver,
) -> Tuple[List[str], set]:
    """Object kept off its support while it penetrates another body in the
    same mode: it is sitting on that body instead (e.g. a box set down on
    the truck's rear step instead of the floor). Returns the diagnosis
    lines and the problem keys they explain."""
    out, explained, seen = [], set(), set()
    for g in violations:
        if g.family != "gap" or g.sign < 0 or g.subject not in resolver.objects:
            continue
        for c in violations:
            if not (c.family == "collision" and c.mode == g.mode
                    and g.subject in (c.subject, c.other)):
                continue
            other = c.other if c.subject == g.subject else c.subject
            explained.update({_problem_key(g), _problem_key(c)})
            if (g.subject, other) in seen:
                continue
            seen.add((g.subject, other))
            gs = [x for x in violations if _problem_key(x) == _problem_key(g)]
            own, target = resolver.contact_faces(g.subject, g.mode)
            target = f"its {own} face on {target}" if own else (target or "its support")
            geometry = _clearance(resolver, g.subject, other)
            out.append(
                f"{_ranges([x.mode for x in gs if x.mode is not None])}: {g.subject} "
                f"ends up on top of {other} instead of resting with {target} "
                f"({_fmt(g)} too high, {_fmt(c)} into {other}). There is no room for it "
                f"there; move clear of {other} before putting it down"
                + (f" ({geometry})" if geometry else "")
            )
    return out, explained


def _as_reach(v: Violation, resolver: NameResolver) -> Violation:
    """A resting object that slides / drifts while nothing touches it was
    dragged by the solver toward an unreachable later contact: report
    that (family ``reach``) instead of a slip."""
    if (v.solver == "kso" and v.family in ("slip", "drift", "twist")
            and v.subject in resolver.objects and not resolver.touching(v.subject, v.mode)):
        return Violation(**{**v.to_dict(), "family": "reach"})
    return v


def _clearance(resolver: NameResolver, obj: str, other: str) -> str:
    """'rear_step spans x 0.40–0.80, y -0.50–0.50; big_box's footprint is
    0.40 x 0.40, so its centre must be at least 0.20 m outside that span'."""
    box, blocker = resolver.boxes.get(obj), resolver.boxes.get(other)
    if not blocker:
        return ""
    cx, cy, _cz, hx, hy, _hz = blocker
    if other in resolver.fixed_boxes:
        text = f"{other} spans x {cx - hx:.2f}–{cx + hx:.2f}, y {cy - hy:.2f}–{cy + hy:.2f}"
    else:
        text = f"{other} is {2 * hx:.2f} x {2 * hy:.2f} and sits wherever the plan last put it"
    if box:
        _x, _y, _z, bx, by, _bz = box
        text += (f"; {obj}'s footprint is {2 * bx:.2f} x {2 * by:.2f}, so its centre must be "
                 f"at least {max(bx, by):.2f} m outside that span")
    return text


def _problem_key(v: Violation) -> Tuple:
    return (v.solver, v.subject, v.family, v.other, v.sign if v.family == "gap" else 1.0)


def merge_attempts(attempts: Sequence[Sequence[Violation]]) -> List[Violation]:
    """Pool the violations of repeated solves of one plan.

    KSO is noisy: a failed solve shows the real blocker plus incidental
    violations that move from solve to solve. Each pooled violation's
    ``attempts`` counts the solves in which the same problem (subject,
    kind) appeared at the same mode, so ``summarize`` can rank recurring
    problems first.
    """
    seen_in: Dict[Tuple, int] = {}
    for vs in attempts:
        for key in {(_problem_key(v), v.mode) for v in vs}:
            seen_in[key] = seen_in.get(key, 0) + 1
    return [Violation(**{**v.to_dict(), "attempts": seen_in[(_problem_key(v), v.mode)]})
            for vs in attempts for v in vs]


def summarize(violations: Sequence[Violation], resolver: NameResolver,
              max_items: int = 5, hints: bool = True, n_attempts: int = 1) -> List[str]:
    """Feedback lines: diagnoses, then one line per problem (same subject /
    kind merged over modes; problems recurring in more solves first, then
    worst first), then hints for the kinds seen."""
    if not violations:
        return []
    violations = [_as_reach(v, resolver) for v in violations]
    lines, explained = _resting_on_wrong_body(violations, resolver)
    more, keys = _step_down(violations, resolver)
    lines, explained = lines + more, explained | keys
    groups: Dict[Tuple, List[Violation]] = {}
    for v in violations:
        if _problem_key(v) not in explained:
            groups.setdefault(_problem_key(v), []).append(v)
    ranked = sorted(groups.values(),
                    key=lambda vs: (-max(x.attempts for x in vs), -max(x.value for x in vs)))
    one_off = 0
    if n_attempts > 1 and any(x.attempts > 1 for x in violations):
        one_off = sum(1 for vs in ranked if max(x.attempts for x in vs) == 1)
        ranked = [vs for vs in ranked if max(x.attempts for x in vs) > 1]
    budget = max(0, max_items - len(lines))
    for vs in ranked[:budget]:
        worst = max(vs, key=lambda x: x.value)
        recurring = max(x.attempts for x in vs)
        modes = [x.mode for x in vs if x.mode is not None
                 and (n_attempts == 1 or x.attempts == recurring)]
        prefix = f"{_ranges(modes)}: " if modes else ""
        suffix = ""
        if n_attempts > 1:
            suffix = (f" (in all {n_attempts} solves)" if recurring >= n_attempts
                      else f" (in {recurring} of {n_attempts} solves)")
        lines.append(prefix + _sentence(worst, resolver) + suffix)
    if len(ranked) > budget:
        lines.append(f"... and {len(ranked) - budget} smaller problems")
    if one_off:
        lines.append(f"({one_off} other problems appeared in only one solve each: "
                     f"solver noise, not necessarily a flaw of the plan)")
    if hints:
        fams: List[str] = []
        for fam in (vs[0].family for vs in ranked[:budget]):
            if fam in _HINTS and fam not in fams:
                fams.append(fam)
        lines += [f"Hint ({fam}): {_HINTS[fam]}" for fam in fams[:3]]
    return lines


TO_STATUS_TEXT = {
    0: "converged",
    1: "ran out of iterations without reaching a feasible trajectory",
    2: "failed numerically (infeasible start, restoration failure or non-finite values)",
}


def to_status_text(status: Optional[int]) -> str:
    return TO_STATUS_TEXT.get(status, f"returned status {status}") if status is not None else "did not run"


# ---------------------------------------------------------------------------
# Feedback blocks
# ---------------------------------------------------------------------------

def _bullets(lines: Sequence[str]) -> List[str]:
    return [ln if ln.startswith(("Hint", "(", "...")) else f"- {ln}" for ln in lines]


def kso_feedback(
    diag: Dict[str, Any], resolver: NameResolver,
) -> Tuple[Optional[str], List[str]]:
    """(headline, lines) for a failed KSO verdict. The headline names one
    break point and the bullets explain that same solve, so the two
    never disagree; full-plan violations follow, marked as secondary."""
    kind = diag.get("kind", "prefix")
    total = diag.get("total_modes")
    lines: List[str] = []
    if kind == "schedule_error":
        errors = list(diag.get("errors") or [])
        headline = "Invalid contact schedule (no solver run): " + (errors[0] if errors else "")
        lines += _bullets(errors[:5])
        lines.append("Every channel (hands, feet, objects) must pass through () before "
                     "touching a different surface.")
        return headline, lines
    if kind == "goal_missed":
        headline = ("KSO: every contact in the plan is feasible, but the final state "
                    "misses the goal target.")
        for name, r in (diag.get("goal_residuals") or {}).items():
            f, t = r.get("final", []), r.get("target", [])
            tol = r.get("tolerance")
            lines.append(
                f"{name} ends at ({', '.join(f'{v:.2f}' for v in f)}), {r.get('distance', 0):.2f} m "
                f"from its target ({', '.join(f'{v:.2f}' for v in t)})"
                + (f" (tolerance {tol:.2f} m)" if tol is not None else "")
            )
        lines = _bullets(lines)
        lines.append("Hint (goal): the contacts work; change where the robot walks before the "
                     "placement so the object is put down at the target.")
        return headline, lines
    full = [Violation(**d) for d in diag.get("violation_details") or []]
    n_att = int(diag.get("kso_attempts") or 1)
    if kind == "window":
        # Incremental evaluation: the first window had no kinematic solution.
        a = int(diag.get("last_feasible_modes") or 0)
        e = diag.get("first_infeasible_modes") or total
        headline = (f"KSO found no kinematic solution for modes {a}–{int(e) - 1} of {total}"
                    + (" (the plan's first window)." if a == 0 else f" from the state reached at mode {a}."))
        if full:
            lines.append("Its violations:")
            lines += _bullets(summarize(full, resolver, max_items=4, n_attempts=n_att))
        return headline, lines
    if kind == "pending":
        # Provisional record written before the break-point search; it
        # survives only when the evaluation was killed during that search.
        headline = (f"KSO infeasible on the full {total}-mode plan ({n_att} attempt{'s' if n_att != 1 else ''}); "
                    f"the break-point search did not finish within the time limit.")
        if full:
            lines.append(f"Violations of the full-plan solves ({n_att}):")
            lines += _bullets(summarize(full, resolver, max_items=4, n_attempts=n_att))
        return headline, lines
    if kind == "prefix" and diag.get("first_infeasible_modes"):
        hi = int(diag["first_infeasible_modes"])
        last = diag.get("last_feasible_modes")
        event = diag.get("break_event") or ""
        if diag.get("timed_out"):
            # The bisection stopped at the time budget: hi is still a known
            # infeasible prefix and last a known feasible one.
            headline = (f"KSO infeasible: the first break lies between modes {last} and {hi - 1} of {total} "
                        f"(the break-point search was cut off by the time budget)" if last
                        else f"KSO infeasible: the first break lies at or before mode {hi - 1} of {total} "
                        f"(the break-point search was cut off by the time budget)")
            headline += f"; modes 0–{last - 1} solve on their own." if last else "."
        else:
            headline = f"KSO infeasible: first break at mode {hi - 1} of {total}"
            if event:
                headline += f" ({event})"
            headline += f"; modes 0–{last - 1} solve on their own." if last else "; even the first transition fails."
        prefix = [Violation(**d) for d in diag.get("prefix_violation_details") or []]
        tries = int(diag.get("prefix_tries") or diag.get("prefix_attempts") or 1)
        if prefix:
            lines.append(f"The {hi}-mode prefix failed {tries} of {tries} tries. Its violations:")
            lines += _bullets(summarize(prefix, resolver, max_items=3, n_attempts=tries))
        if full:
            lines.append("Full-plan solves additionally report (later modes; these may follow "
                         "from the break above):")
            lines += _bullets(summarize(full, resolver, max_items=3, hints=not prefix,
                                        n_attempts=n_att))
        probe_line = None
        try:
            import repair_probe

            probe_line = repair_probe.describe_result(diag.get("probe") or {})
        except Exception:  # noqa: BLE001
            probe_line = None
        if probe_line:
            lines.append(probe_line)
        return headline, lines
    reason = diag.get("reason") or "the break point was not located"
    headline = f"KSO failed on the full {total}-mode plan ({reason})."
    if full:
        lines.append(f"Violations of the full-plan solves ({n_att}):")
        lines += _bullets(summarize(full, resolver, max_items=4, n_attempts=n_att))
    return headline, lines


def score_feedback(
    combined_score: float, task_quality: Optional[float], to_quality: Optional[float],
    quality_parts: Optional[Dict[str, float]], max_violation: Optional[float],
    jerk_per_mode: Optional[float], band: float = 0.1,
) -> Optional[str]:
    """One line saying what the score is made of (quality mode)."""
    if task_quality is None or to_quality is None:
        return None
    text = (f"Score {combined_score:.3f} = {band:.1f} + {1 - band:.1f} x task_quality "
            f"{task_quality:.2f} x to_quality {to_quality:.2f}.")
    if quality_parts:
        text += " task_quality residuals: " + ", ".join(
            f"{k} {v:.3f}" for k, v in quality_parts.items()) + "."
    if max_violation is not None and jerk_per_mode is not None:
        dom = ("max constraint violation" if 1e2 * max_violation >= 1e-2 * jerk_per_mode
               else "control jerk")
        text += (f" to_quality = 1/(1 + 100 x max violation {max_violation:.3f} + 0.01 x jerk/mode "
                 f"{jerk_per_mode:.0f}); {dom} dominates.")
    return text
