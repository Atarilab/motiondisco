"""The TO-goal override (loco_manipulation/_task_loader.py ``with_to_goal``,
SHINKA_TO_GOAL / SHINKA_TO_GOAL_TOL): forcing a hard TO goal on a scene
whose goals.py leaves TO free, with a usable tolerance."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import _task_loader as tl  # noqa: E402


@dataclass(frozen=True)
class Tol:
    position: Any = 0.0
    rotation: Any = 0.0
    joints: Any = 0.0


@dataclass(frozen=True)
class Target:
    value: Any
    to_tolerance: Tol = field(default_factory=Tol)


@dataclass(frozen=True)
class Goal:
    kso: str
    to: str
    objects: Dict[str, Target] = field(default_factory=dict)
    robots: Dict[str, Target] = field(default_factory=dict)


def test_hard_override_sets_mode_and_tolerances():
    goal = Goal(kso="hard", to="off", objects={"box": Target([1, 2, 3])},
                robots={"robot": Target([0] * 10, Tol(joints=0.2))})
    out = tl.with_to_goal(goal, "hard", 0.05, tolerances_cls=Tol)
    assert out.to == "hard" and out.kso == "hard"
    assert out.objects["box"].to_tolerance == Tol(position=0.05, rotation=0.05, joints=0.0)
    assert out.robots["robot"].to_tolerance == Tol(position=0.05, rotation=0.05, joints=0.2)   # joints kept
    assert goal.to == "off"                                                                  # input untouched


def test_other_modes_and_no_override():
    goal = Goal(kso="hard", to="hard", objects={"box": Target([1, 2, 3], Tol(0.02, 0.1))})
    assert tl.with_to_goal(goal, None, 0.05, tolerances_cls=Tol) is goal
    assert tl.with_to_goal(None, "hard", 0.05, tolerances_cls=Tol) is None
    off = tl.with_to_goal(goal, "off", 0.05, tolerances_cls=Tol)
    assert off.to == "off" and off.objects["box"].to_tolerance == Tol(0.02, 0.1)              # tolerances untouched
    with pytest.raises(ValueError):
        tl.with_to_goal(goal, "firm", 0.05, tolerances_cls=Tol)
