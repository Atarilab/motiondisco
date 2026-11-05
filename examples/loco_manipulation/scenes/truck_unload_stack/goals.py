from __future__ import annotations

from dataclasses import replace

import numpy as np

from faro.search.goal import (
    DiscreteGoal, GoalTolerances, OutcomeScorer, SolverOutcome, TerminalGoal,
)
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL

from scenes.truck_unload_stack.initial_conditions import goal as goal_state
from scenes.truck_unload_stack.scene import BIG_HALF, SMALL_HALF

# How close (m) the boxes must end to the `goal` initial condition's
# positions for the hard KSO target to count as reached.
GOAL_POSITION_TOL = 0.05


def stacked_on_floor(scene) -> DiscreteGoal:
    """Big box resting on the floor, small box resting on the big box's top."""
    return DiscreteGoal(
        discrete={
            ("obj", "big_box"): (6, scene.face_id("floor", "surface"), CONTACT_TYPE_UNILATERAL),
            ("obj", "small_box"): (6, scene.face_id("big_box", "top"), CONTACT_TYPE_UNILATERAL),
        },
    )


def stack_at_goal(scene, resolved_state) -> TerminalGoal:
    """Both boxes at the `goal` initial condition's positions: big box on
    the floor beside the robot at (0.1, -0.9), clear of the rear step
    (x 0.4-0.8, |y| <= 0.5), small box centred on top.

    Without a position target KSO sets the stack down wherever the robot
    stands when it steps off the truck, which is on the rear step. KSO
    enforces the target hard (within GOAL_POSITION_TOL). TO keeps it off
    so trajectories stay diverse: a converged TO may then set the stack
    down away from the target (1.6-2.3 m on the plans tried). to="hard"
    with the same to_tolerance builds and converged on one plan; faro's
    soft TO goal fails to build (std::bad_cast).
    """
    g = TerminalGoal.from_state(
        scene, goal_state(scene),
        objects={"big_box": "position", "small_box": "position"},
        kso="hard", to="off",
    )
    tol = GoalTolerances(position=GOAL_POSITION_TOL)
    return replace(g, objects={k: replace(v, kso_tolerance=tol) for k, v in g.objects.items()})


class StackAlignment:
    """Residual of the final stack (lower is better, 0 = perfect).

    Horizontal offset between the box centers plus the vertical gap
    between the small box's bottom and the big box's top, in meters.
    """

    def __call__(self, outcome: SolverOutcome) -> float:
        return sum(self.parts(outcome).values())

    def parts(self, outcome: SolverOutcome) -> dict:
        big = np.asarray(outcome.results["obj_xs"]["big_box"])[-1, :3]
        small = np.asarray(outcome.results["obj_xs"]["small_box"])[-1, :3]
        # Box sizes from the solved scene when known (datasets vary them).
        objs = getattr(getattr(outcome, "scene", None), "env_objects", None) or {}
        big_h = float(objs["big_box"].half_extents[2]) if "big_box" in objs else BIG_HALF[2]
        small_h = float(objs["small_box"].half_extents[2]) if "small_box" in objs else SMALL_HALF[2]
        xy_offset = float(np.linalg.norm(small[:2] - big[:2]))
        z_gap = abs(float(small[2] - big[2]) - (big_h + small_h))
        return {"stack_xy_offset": xy_offset, "stack_z_gap": z_gap}


class StackAtGoal(StackAlignment):
    """Stack alignment plus the distance of both boxes from their targets."""

    def __init__(self, goal: TerminalGoal) -> None:
        self.goal = goal

    def parts(self, outcome: SolverOutcome) -> dict:
        out = super().parts(outcome)
        out["goal_distance"] = float(self.goal.score(outcome))
        return out


def stack_at_goal_score(scene, resolved_state) -> OutcomeScorer:
    return StackAtGoal(stack_at_goal(scene, resolved_state))


DISCRETE_GOALS = {"stacked_on_floor": stacked_on_floor}
TERMINAL_GOALS = {"stack_at_goal": stack_at_goal}
OUTCOME_SCORERS = {"stack_at_goal": stack_at_goal_score}
