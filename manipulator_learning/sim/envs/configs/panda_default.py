# Default config file for panda envs.

# note: as a rule of thumb, these values should be as reusable as possible between envs, while values that
#       are expected to change between each env should be actual class parameters.

import copy
import numpy as np
from manipulator_learning.sim.envs.configs.all_default import ROBOT_URDF_BASE, ALL_DEF_CONFIG


CONFIG = copy.deepcopy(ALL_DEF_CONFIG)
CONFIG.update(dict(
    # specific config for panda robot
    robot_config=dict(
        robot='panda',
        urdf_root=ROBOT_URDF_BASE + "/models/franka_panda/panda.urdf",
        num_controllable_joints=9,
        num_gripper_joints=2,
        base_link_index=0,
        ee_link_index=9,
        tool_link_index=12,
        gripper_indices=[10, 11],
        arm_indices=[1, 2, 3, 4, 5, 6, 7],
        gripper_max=[.04, 0],
        base_constraint=True
    ),

    # cam
    robot_base_ws_cam_tf=((-.4, .6, .3), (-2, 0, -1.85)),

    # arm
    init_gripper_pose=((0.0, .5, .25), (np.pi, 0, 0)),

    # base
    robot_base_pose=((1.0, -.7, .52), (0, 0, 0)),
    base_pose_from_workspace_center=False,

    # objects -- all obj_init_pos are relative to this
    workspace_center=(1.0, -.2, .7),

    # force torque sensor on EE
    force_torque_gravity_sub=12.11,
))
