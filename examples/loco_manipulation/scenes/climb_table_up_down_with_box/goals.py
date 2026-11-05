from __future__ import annotations

from dataclasses import replace

from faro.search.goal import DiscreteGoal, GoalTolerances, OutcomeScorer, TerminalGoal
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL

from scenes.climb_table_up_down_with_box.initial_conditions import goal as goal_state


def box_and_robot_on_floor(scene) -> DiscreteGoal:
    floor = scene.face_id("floor", "surface")
    return DiscreteGoal(discrete={
        ("ee", "robot", "left_sole"): (floor, CONTACT_TYPE_UNILATERAL),
        ("ee", "robot", "right_sole"): (floor, CONTACT_TYPE_UNILATERAL),
        ("obj", "white_box"): (6, floor, CONTACT_TYPE_UNILATERAL),
    })


def box_on_far_side(scene, resolved_state) -> TerminalGoal:
    """Box on the floor past the table's far edge, within 0.3 m of
    (1.9, 0.1) so any spot about 0.25-0.85 m beyond the edge counts.

    Enforced in KSO only; TO keeps it off for trajectory diversity
    (faro's soft TO goal does not build)."""
    g = TerminalGoal.from_state(scene, goal_state(scene), objects={"white_box": "position"},
                                kso="hard", to="off")
    tol = GoalTolerances(position=0.3)
    return replace(g, objects={k: replace(v, kso_tolerance=tol) for k, v in g.objects.items()})


def box_on_far_side_score(scene, resolved_state) -> OutcomeScorer:
    return box_on_far_side(scene, resolved_state).score


DISCRETE_GOALS = {"box_and_robot_on_floor": box_and_robot_on_floor}
TERMINAL_GOALS = {"box_on_far_side": box_on_far_side}
OUTCOME_SCORERS = {"box_on_far_side": box_on_far_side_score}
