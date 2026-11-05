"""MotionDisco-side climb_table_up_down_with_box: the box must end on the
floor on the far side of the table (upstream's goal drops the box entry)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

pytest.importorskip("faro")
pytest.importorskip("pinocchio")


def test_goals_require_box_on_floor_past_the_table():
    import _task_loader

    t = _task_loader.load_task("climb_table_up_down_with_box", "default", "")
    floor = t.scene.face_id("floor", "surface")
    assert t.discrete_goal == {"left_sole": (floor, 1), "right_sole": (floor, 1), "white_box": (6, floor, 1)}
    g = t.terminal_goal
    assert (g.kso, g.to) == ("hard", "off")
    target = g.objects["white_box"]
    far_edge = 1.0 + 0.35 + 0.1325        # table centre + half depth + box half depth
    assert target.xyz[0] - target.kso_tolerance.position >= far_edge - 0.05
