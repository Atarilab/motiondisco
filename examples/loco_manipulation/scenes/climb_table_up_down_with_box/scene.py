"""Carry the white box over the table and set it down on the far side.

MotionDisco-side copy of faro's
``motiondisco.climb_table_up_down_with_box`` (same objects and start),
with working goals: upstream's discrete goal gives the box channel a bare
``1`` instead of a contact tuple, so it is dropped and only "both feet on
the floor" is checked, and it has no target, so the box is never required
to reach the other side. See truck_unload_stack/scene.py for how scenes/
is found.
"""

from faro.configs.objects import FixedBox, MovableBox, Plane
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = "climb_table_up_down_with_box"


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={"robot": make_g1_robot()},
        objects={
            "floor": Plane(),
            "table_short": FixedBox(half_extents=[0.35, 0.7, 0.365], color=[0.5, 0.5, 0.5, 1.0]),
            "white_box": MovableBox(
                half_extents=[0.1325, 0.1275, 0.135],
                mass=1.18,
                color=[1.0, 1.0, 1.0, 1.0],
            ),
        },
    )
