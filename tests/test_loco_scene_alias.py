"""Old flat scene names map onto faro's dotted scene families."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import _task_loader  # noqa: E402


def test_old_names_alias_when_not_importable(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _task_loader.canonical_scene_name("corl_box_stacking") == "motiondisco.box_stacking"
    assert _task_loader.canonical_scene_name("faro_box_pushing") == "faro.box_pushing"
    # Already dotted, bare MotionDisco-side and unrelated names pass through.
    for name in ("motiondisco.box_stacking", "truck_unload_stack", "example_scene"):
        assert _task_loader.canonical_scene_name(name) == name


def test_old_name_kept_when_it_still_resolves(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert _task_loader.canonical_scene_name("corl_box_stacking") == "corl_box_stacking"


def test_collision_margin_inflates_every_capsule(monkeypatch):
    import types
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Capsule:
        radius: float
        length: float = 1.0

    @dataclass(frozen=True)
    class Model:
        capsules: dict
        self_collision_pairs: list = field(default_factory=list)

    @dataclass(frozen=True)
    class Robot:
        collision_model: Model
        nq: int = 8

    @dataclass(frozen=True)
    class Scene:
        name: str
        robots: dict
        objects: dict
        include_robot_ees: bool = True

    fake = types.ModuleType("faro.configs.scene")
    fake.Scene = Scene
    monkeypatch.setitem(sys.modules, "faro", types.ModuleType("faro"))
    monkeypatch.setitem(sys.modules, "faro.configs", types.ModuleType("faro.configs"))
    monkeypatch.setitem(sys.modules, "faro.configs.scene", fake)
    scene = Scene("s", {"r": Robot(Model({"torso": Capsule(0.1), "l_foot": Capsule(0.03)}))}, {"floor": 1})
    out = _task_loader.with_collision_margin(scene, 0.03)
    caps = out.robots["r"].collision_model.capsules
    assert caps["torso"].radius == 0.13 and caps["l_foot"].radius == 0.06 and caps["torso"].length == 1.0
    assert out.objects is scene.objects and out.robots["r"].nq == 8
    assert scene.robots["r"].collision_model.capsules["torso"].radius == 0.1     # the original is untouched
    assert _task_loader.with_collision_margin(scene, 0.0) is scene
    legs = _task_loader.with_collision_margin(scene, 0.02, ("l_foot",)).robots["r"].collision_model.capsules
    assert legs["l_foot"].radius == 0.05 and legs["torso"].radius == 0.1
    with pytest.raises(ValueError):
        _task_loader.with_collision_margin(scene, 0.02, ("r_hand",))
