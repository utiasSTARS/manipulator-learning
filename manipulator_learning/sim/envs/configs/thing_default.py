# Default config file for thing envs.

# note: as a rule of thumb, these values should be as reusable as possible between envs, while values that
#       are expected to change between each env should be actual class parameters.

import copy
import numpy as np
from manipulator_learning.sim.envs.configs.all_default import ROBOT_URDF_BASE, ALL_DEF_CONFIG

DEFAULT_RC = dict(
    num_controllable_joints=8,
    num_gripper_joints=2,
    base_link_index=1,
    ee_link_index=17,
    arm_indices=[11, 12, 13, 14, 15, 16],
    gripper_max=[.04, 0],
    base_constraint=True
)

TWO_FINGER_RC = copy.deepcopy(DEFAULT_RC)
TWO_FINGER_RC.update(dict(
    robot='thing_2_finger',
    urdf_root=ROBOT_URDF_BASE + '/models/thing/assets/combined_urdf/ridgeback_ur10_pr2grip_fixed.urdf',
    tool_link_index=22,
    gripper_indices=[18, 20],
    gripper_max=[1, 0],
))

PANDA_GRIPPER_RC = copy.deepcopy(DEFAULT_RC)
PANDA_GRIPPER_RC.update(dict(
    robot='thing_panda_gripper',
    urdf_root=ROBOT_URDF_BASE + '/models/thing/assets/combined_urdf/ridgeback_ur10_pandagrip.urdf',
    tool_link_index=21,
    gripper_indices=[19, 20],
))

ROD_RC = copy.deepcopy(DEFAULT_RC)
ROD_RC.update(dict(
    robot='thing_rod',
    urdf_root=ROBOT_URDF_BASE + '/models/thing/assets/combined_urdf/ridgeback_ur10_rod_fixed.urdf',
    num_controllable_joints=6,
    num_gripper_joints=0,
    ee_link_index=16,
    tool_link_index=18,
    gripper_indices=[],
))


def get_thing_def_config(robot):
    if robot == 'thing_2_finger':
        rc = TWO_FINGER_RC
    elif robot == 'thing_panda_gripper':
        rc = PANDA_GRIPPER_RC
    elif robot == 'thing_rod':
        rc = ROD_RC
    else:
        raise NotImplementedError(f"No thing default config for robot {robot}")

    config = copy.deepcopy(ALL_DEF_CONFIG)
    config.update(dict(
        robot_config=rc,

        # cam
        robot_base_ws_cam_tf=((-.4, .65, .9), (-2.45, 0, -.4)),

        # arm
        init_gripper_pose=((-.15, .85, .75), (-.75 * np.pi, 0, np.pi/2)),
        gripper_force=10,
        gripper_default_close=False,
        max_gripper_vel=0.8,
        init_gripper_random_lim=None,

        # base
        random_base_theta_bounds=(0, 0),
        robot_base_pose=((0, -.5, 0), (0, 0, -1.5707963)),
        base_pose_from_workspace_center=True,

        # objects
        workspace_center=(0.75, -.15, .7),
        block_style='cube',
        goal_type=None,
        block_random_lim=[],
        init_block_pos=[],
        init_rod_pos=None,
        rod_random_lim=None,
        block_colors=None,

    ))

    return config


TWO_FINGER_CONFIG = get_thing_def_config('thing_2_finger')
PANDA_GRIPPER_CONFIG = get_thing_def_config('thing_panda_gripper')
ROD_CONFIG = get_thing_def_config('thing_rod')
