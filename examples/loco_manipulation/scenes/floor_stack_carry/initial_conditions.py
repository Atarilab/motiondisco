from dataclasses import replace

from faro.configs.scene import OnFace, Pose, Standing, StateSpec

STACK_START_XY = (0.55, 0.0)
STACK_GOAL_XY = (2.05, 0.0)   # 1.5 m further


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=("floor", "surface"))},
        objects={
            "floor": Pose(),
            "big_box": OnFace(on=("floor", "surface"), xy_yaw=(*STACK_START_XY, 0.0)),
            "small_box": OnFace(on=("big_box", "top")),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(initial, objects={**initial.objects,
                                     "big_box": OnFace(on=("floor", "surface"), xy_yaw=(*STACK_GOAL_XY, 0.0))})


INITIAL_CONDITIONS = {"default": default, "goal": goal}
