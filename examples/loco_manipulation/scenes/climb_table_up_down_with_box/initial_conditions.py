from dataclasses import replace

from faro.configs.scene import OnFace, Pose, Standing, StateSpec

TABLE_XY = (1.0, 0.0)          # table spans x 0.65..1.35
BOX_START_XY = (0.4, 0.1)      # near side, as upstream
BOX_GOAL_XY = (1.9, 0.1)       # far side: 0.55 m past the far edge


def default(scene):
    return StateSpec(
        robots={"robot": Standing(on=("floor", "surface"))},
        objects={
            "floor": Pose(),
            "table_short": OnFace(on=("floor", "surface"), xy_yaw=(*TABLE_XY, 0.0)),
            "white_box": OnFace(on=("floor", "surface"), xy_yaw=(*BOX_START_XY, 0.0)),
        },
    )


def goal(scene):
    initial = default(scene)
    return replace(initial, objects={**initial.objects,
                                     "white_box": OnFace(on=("floor", "surface"), xy_yaw=(*BOX_GOAL_XY, 0.0))})


INITIAL_CONDITIONS = {"default": default, "goal": goal}
