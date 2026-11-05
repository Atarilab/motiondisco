"""Carry a two-box stack across the floor (simplified truck_unload_stack).

Same box sizes as truck_unload_stack; the boxes start already stacked on
the floor in front of the robot. MotionDisco-side scene (see
truck_unload_stack/scene.py for how scenes/ is found).
"""

from faro.configs.objects import MovableBox, Plane
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = "floor_stack_carry"
BIG_HALF = (0.20, 0.20, 0.15)
SMALL_HALF = (0.13, 0.13, 0.10)


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={"robot": make_g1_robot()},
        objects={
            "floor": Plane(),
            "big_box": MovableBox(half_extents=list(BIG_HALF), mass=2.0, color=[0.2, 0.45, 0.85, 1.0]),
            "small_box": MovableBox(half_extents=list(SMALL_HALF), mass=1.0, color=[0.95, 0.55, 0.15, 1.0]),
        },
    )
