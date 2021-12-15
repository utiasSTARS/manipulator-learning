# Default config file for all envs. Other envs will use these values unless they are specifically overwritten
# with their own config files.

# note: as a rule of thumb, these values should be as reusable as possible between envs, while values that
#       are expected to change between each env should be actual class parameters.

import os
import numpy as np

import manipulator_learning


# generic defaults for all envs
MAN_LEARN_DIR = os.path.dirname(manipulator_learning.__file__)
MODEL_BASE = MAN_LEARN_DIR + '/sim'
ROBOT_URDF_BASE = MODEL_BASE + '/robots'
OBJECT_URDF_BASE = MODEL_BASE + '/objects/models/urdf'

ALL_DEF_CONFIG = dict(
    # sim
    time_step=.01,
    render_opengl_gui=False,

    # cam
    debug_cam_params=(.20, -.41, .59, 1.6, -29.4, 156.),
    render_shadows=True,
    render_ground_plane=True,

    # arm
    gripper_default_close=False,
    gripper_force=10,
    max_gripper_vel=0.8,
    init_gripper_random_lim=None,

    # base
    random_base_theta_bounds=(0, 0),
    base_random_lim=((0, 0, 0), (0, 0, 0)),
    base_pose_from_workspace_center=True,
    cam_workspace_distance=.3,

    # objects
    object_urdf_root=OBJECT_URDF_BASE,
    block_style='',
    block_random_lim=[],
    init_block_pos=[],
    init_rod_pos=None,
    rod_random_lim=None,
    block_colors=None,

    # task
    goal_type=None,
    goal_pos=None
)

# task-specific params
XYZ_CONFIG = dict(
    valid_trans_dof=[1, 1, 1],
    valid_rot_dof=[0, 0, 0]
)

LIFT_DEFAULTS = dict(
    block_random_lim=((0.25, 0.25)),
    goal_type=None,
)

MULTIVIEW_DEFS = dict(
    random_base_theta_bounds=(-3 * np.pi / 16, np.pi / 16),
    base_random_lim=((.02, .02, .002), (0, 0, .02))
)
