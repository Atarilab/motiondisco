from dataclasses import replace

from faro.configs.scene import OnFace, Pose, Standing, StateSpec

# Bed front face 0.8 m ahead of the robot (center at 0.8 + 1.5 m); the rear
# step sits flush against it, its own front face 0.4 m from the robot.
BED_XY_YAW = (2.3, 0.0, 0.0)
STEP_XY_YAW = (0.6, 0.0, 0.0)


def default(scene):
    """Robot on the floor facing the truck; both boxes in the middle of the
    bed, ~1.5 m from its front edge (out of reach from the ground)."""
    return StateSpec(
        robots={"robot": Standing(on=("floor", "surface"))},
        objects={
            "floor": Pose(),
            "truck_bed": OnFace(on=("floor", "surface"), xy_yaw=BED_XY_YAW),
            "rear_step": OnFace(on=("floor", "surface"), xy_yaw=STEP_XY_YAW),
            "big_box": OnFace(on=("truck_bed", "top"), xy_yaw=(0.0, 0.4, 0.0)),
            "small_box": OnFace(on=("truck_bed", "top"), xy_yaw=(0.0, -0.4, 0.0)),
        },
    )


def goal(scene):
    """One example goal: big box on the floor beside the robot, small box
    centered on top of it. Only the stacking is required; the floor
    location is free (see goals.py)."""
    initial = default(scene)
    return replace(
        initial,
        objects={
            **initial.objects,
            "big_box": OnFace(on=("floor", "surface"), xy_yaw=(0.1, -0.9, 0.0)),
            "small_box": OnFace(on=("big_box", "top")),
        },
    )


INITIAL_CONDITIONS = {"default": default, "goal": goal}
