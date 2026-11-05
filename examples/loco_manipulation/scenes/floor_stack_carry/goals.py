from __future__ import annotations

from faro.search.goal import DiscreteGoal, GoalTolerances, OutcomeScorer, TerminalGoal
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL

from scenes.floor_stack_carry.initial_conditions import goal as goal_state


def stacked_on_floor(scene) -> DiscreteGoal:
    return DiscreteGoal(discrete={
        ("obj", "big_box"): (6, scene.face_id("floor", "surface"), CONTACT_TYPE_UNILATERAL),
        ("obj", "small_box"): (6, scene.face_id("big_box", "top"), CONTACT_TYPE_UNILATERAL),
    })


def stack_moved(scene, resolved_state) -> TerminalGoal:
    """Both boxes at the goal-state positions (stack moved 1.5 m).

    Enforced in KSO only: faro's soft TO goal cost fails to build
    (std::bad_cast in moto.cost.from_vector), so TO keeps it off.
    """
    g = TerminalGoal.from_state(scene, goal_state(scene),
                                objects={"big_box": "position", "small_box": "position"},
                                kso="hard", to="off")
    tol = GoalTolerances(position=0.05)
    from dataclasses import replace
    return replace(g, objects={k: replace(v, kso_tolerance=tol) for k, v in g.objects.items()})


def stack_moved_score(scene, resolved_state) -> OutcomeScorer:
    return stack_moved(scene, resolved_state).score


DISCRETE_GOALS = {"stacked_on_floor": stacked_on_floor}
TERMINAL_GOALS = {"stack_moved": stack_moved}
OUTCOME_SCORERS = {"stack_moved": stack_moved_score}
