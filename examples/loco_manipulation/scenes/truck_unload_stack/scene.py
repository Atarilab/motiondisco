"""Truck unloading: two boxes of different sizes deep on a truck-height bed.

The boxes sit ~1.5 m from the bed's front edge, out of reach from the
ground, so the robot has to climb onto the bed (via the rear step) to
pick them up, then bring them down and stack them on the floor.

A MotionDisco-side scene (faro is read-only). ``scenes`` is a
namespace package, so ``examples/loco_manipulation/scenes/<name>/`` is
found by ``scenes.common.registry`` whenever examples/loco_manipulation
is on sys.path (as for every loco_manipulation entry point).
"""

from faro.configs.objects import FixedBox, MovableBox, Plane
from faro.configs.robot import make_g1_robot
from faro.configs.scene import Scene

SCENE_NAME = "truck_unload_stack"

# Truck bed: 3.0 m long, 2.0 m wide, top surface at 0.9 m (pickup / van
# bed height). The rear step (bumper) is half that height so the bed can
# be climbed in two ~0.45 m steps; a single 0.9 m step-up is beyond the G1.
BED_HALF = (1.5, 1.0, 0.45)
STEP_HALF = (0.2, 0.5, 0.225)
BIG_HALF = (0.20, 0.20, 0.15)
SMALL_HALF = (0.13, 0.13, 0.10)


def make_scene():
    return Scene(
        name=SCENE_NAME,
        robots={"robot": make_g1_robot()},
        objects={
            "floor": Plane(),
            "truck_bed": FixedBox(half_extents=list(BED_HALF), color=[0.35, 0.35, 0.4, 1.0]),
            "rear_step": FixedBox(half_extents=list(STEP_HALF), color=[0.55, 0.55, 0.6, 1.0]),
            "big_box": MovableBox(half_extents=list(BIG_HALF), mass=2.0,
                                  color=[0.2, 0.45, 0.85, 1.0]),
            "small_box": MovableBox(half_extents=list(SMALL_HALF), mass=1.0,
                                    color=[0.95, 0.55, 0.15, 1.0]),
        },
    )


if __name__ == "__main__":
    from scenes.common.viz import view_initial_condition
    view_initial_condition(SCENE_NAME, "default")
