"""Scene layout variants (loco_manipulation): IC-name parsing, offsets on
start placements, goal shifting / scorer rebinding, layout validation.
faro's frozen dataclasses are stubbed so this runs without faro."""

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import layouts  # noqa: E402


@dataclass(frozen=True)
class OnFace:
    on: tuple
    xy_yaw: tuple = (0, 0, 0)


@dataclass(frozen=True)
class Pose:
    q: tuple = (0, 0, 0, 0, 0, 0, 1)


@dataclass(frozen=True)
class Standing:
    on: tuple
    xy_yaw: tuple = (0, 0, 0)


@dataclass(frozen=True)
class StateSpec:
    robots: Mapping[str, Any]
    objects: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionTarget:
    xyz: Any

    @property
    def value(self):
        return self.xyz


@dataclass(frozen=True)
class TerminalGoal:
    objects: Mapping[str, Any] = field(default_factory=dict)
    robots: Mapping[str, Any] = field(default_factory=dict)

    def score(self, outcome):
        return 0.0


@pytest.fixture(autouse=True)
def fake_faro(monkeypatch):
    scene_mod = types.ModuleType("faro.configs.scene")
    scene_mod.OnFace, scene_mod.Pose, scene_mod.Standing = OnFace, Pose, Standing
    goal_mod = types.ModuleType("faro.search.goal")
    goal_mod.PositionTarget, goal_mod.TerminalGoal = PositionTarget, TerminalGoal
    for name in ("faro", "faro.configs", "faro.search"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "faro.configs.scene", scene_mod)
    monkeypatch.setitem(sys.modules, "faro.search.goal", goal_mod)


def _spec():
    return StateSpec(
        robots={"robot": Standing(on=("floor", "surface"))},
        objects={"floor": Pose(), "table": OnFace(("floor", "surface"), (2.5, 0, 0)),
                 "box": OnFace(("floor", "surface"), (0.5, 0, 0))},
    )


def test_split_ic():
    assert layouts.split_ic("default") == ("default", None)
    assert layouts.split_ic("default@far") == ("default", "far")
    assert layouts.split_ic("default@") == ("default", None)


def test_apply_layout_offsets_robot_and_objects():
    lay = layouts.Layout(robot=(0.1, 0.2, 0.3), objects={"table": (0.0, 0.8, 0.0)})
    spec = layouts.apply_layout(_spec(), lay)
    assert spec.robots["robot"].xy_yaw == pytest.approx((0.1, 0.2, 0.3))
    assert spec.objects["table"].xy_yaw == pytest.approx((2.5, 0.8, 0.0))
    assert spec.objects["box"] == _spec().objects["box"]


def test_apply_layout_rejects_unknown_object():
    with pytest.raises(ValueError, match="does not place"):
        layouts.apply_layout(_spec(), layouts.Layout(objects={"ghost": (1, 0, 0)}))


def test_goal_follows_object_displacement():
    lay = layouts.Layout(goal_follows="table")
    state = lambda x: types.SimpleNamespace(  # noqa: E731
        fixed_env_objects={"table": types.SimpleNamespace(q=np.array([x, 0.0, 0.36, 0, 0, 0, 1]))},
        env_objects={})
    off = layouts.goal_offset(lay, state(2.5), state(3.2))
    assert off == pytest.approx([0.7, 0.0, 0.0])


def test_shift_goal_and_scorer_move_targets():
    goal = TerminalGoal(objects={"box": PositionTarget(np.array([2.5, 0, 0.9])),
                                 "lid": PositionTarget(np.array([2.5, 0, 1.0]))})
    moved = layouts.shift_goal(goal, np.array([0.0, 0.8, 0.0]))
    assert moved.objects["box"].xyz == pytest.approx([2.5, 0.8, 0.9])
    per = layouts.shift_goal(goal, {"lid": np.array([0, 0, 0.1])})
    assert per.objects["box"].xyz == pytest.approx([2.5, 0, 0.9])
    assert per.objects["lid"].xyz == pytest.approx([2.5, 0, 1.1])
    # A scorer bound to its own goal instance is rebound to a moved copy.
    scorer = TerminalGoal(objects={"box": PositionTarget(np.array([2.5, 0, 0.9]))}).score
    rebound = layouts.shift_scorer(scorer, np.array([0.0, 0.8, 0.0]))
    assert rebound.__self__.objects["box"].xyz == pytest.approx([2.5, 0.8, 0.9])

    class WithGoal:
        def __init__(self, goal):
            self.goal = goal
    holder = layouts.shift_scorer(WithGoal(goal), {"box": np.array([0, 0, 0.05])})
    assert holder.goal.objects["box"].xyz == pytest.approx([2.5, 0, 0.95])
    assert layouts.shift_scorer(len, np.ones(3)) is len  # not goal-based: unchanged


def test_check_layout_flags_overlaps_and_blocked_start():
    def box(x, y, half=(0.5, 0.5, 0.4)):
        return types.SimpleNamespace(q=np.array([x, y, half[2], 0, 0, 0, 1]))
    scene = types.SimpleNamespace(fixed_env_objects={
        "a": types.SimpleNamespace(half_extents=(0.5, 0.5, 0.4)),
        "b": types.SimpleNamespace(half_extents=(0.5, 0.5, 0.4)),
    })
    robot = types.SimpleNamespace(q=np.array([0.0, 0.0, 0.77]))
    touching = types.SimpleNamespace(fixed_env_objects={"a": box(2.0, 0), "b": box(3.0, 0)},
                                     robots={"robot": robot})
    assert layouts.check_layout(scene, touching) == []
    overlapping = types.SimpleNamespace(fixed_env_objects={"a": box(2.0, 0), "b": box(2.5, 0)},
                                        robots={"robot": robot})
    assert any("overlap" in p for p in layouts.check_layout(scene, overlapping))
    blocked = types.SimpleNamespace(fixed_env_objects={"a": box(0.3, 0), "b": box(3.0, 0)},
                                    robots={"robot": robot})
    assert any("robot" in p for p in layouts.check_layout(scene, blocked))
    # Overlaps the base layout already has (parts of one table) are not new.
    assert layouts.check_layout(scene, overlapping, base_resolved=overlapping) == []
    shifted = types.SimpleNamespace(fixed_env_objects={"a": box(2.0, 0), "b": box(2.3, 0)},
                                    robots={"robot": robot})
    assert layouts.check_layout(scene, shifted, base_resolved=overlapping) == []
    assert layouts.check_layout(scene, overlapping, base_resolved=touching) != []


def test_registered_layouts_are_well_formed():
    for scene, named in layouts.LAYOUTS.items():
        for name, lay in named.items():
            assert isinstance(lay, layouts.Layout), (scene, name)
            assert len(lay.robot) == 3 and all(len(o) == 3 for o in lay.objects.values())
            assert not (lay.goal_shift is not None and lay.goal_follows), (scene, name)
