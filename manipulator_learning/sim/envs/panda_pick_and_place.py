from gym import spaces
import numpy as np
import copy

from manipulator_learning.sim.envs.thing_pick_and_place import ThingPickAndPlaceGeneric


PANDA_ADDS_DEF = dict(
    block_random_lim=[[.05, .05]],
    init_block_pos=[[.05, -0.05]],
    goal_type='air',
    init_gripper_pose=[[0.0, .5, .25], [np.pi, 0, 0]],
    robot_base_ws_cam_tf=((-.4, .6, .3), (-2, 0, -1.85)),
    pos_limits=[[.85, -.35, .655], [1.15, -0.05, 0.8]],
    block_style='small'
)


class PandaLiftXYZState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=18, n_substeps=10, dense_reward=True, action_multiplier=0.1, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)

        CONFIG = copy.deepcopy(PANDA_ADDS_DEF)
        CONFIG.update(dict(
            valid_r_dofs=[0, 0, 0],
            block_random_lim=[[0.25, 0.25]],
            init_block_pos=[[-.025, -.05]],
            init_gripper_random_lim=[.25, .25, .06, 0., 0., 0.],
            goal_type=None
        ))

        # for lift env, reach radius is height above table
        super().__init__('lift_xyz', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.1, robot='panda',
                         limits_cause_failure=False, failure_causes_done=False, success_causes_done=False,
                         control_frame='b', **CONFIG, **kwargs)


class PandaBringXYZState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=18, n_substeps=10, dense_reward=True, action_multiplier=0.1, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)

        CONFIG = copy.deepcopy(PANDA_ADDS_DEF)
        CONFIG.update(dict(
            valid_r_dofs=[0, 0, 0],
            block_random_lim=[[0.25, 0.25]],
            init_block_pos=[[-.025, -.05]],
            init_gripper_random_lim=[.25, .25, .06, 0., 0., 0.],
            goal_type='coaster',
            goal_pos=[0.0, .1]
        ))

        super().__init__('bring_xyz', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.025, robot='panda',
                         control_frame='b', **CONFIG, **kwargs)


class PandaPickAndPlaceAirGoal6DofState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, action_multiplier=0.1, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)

        # for lift env, reach radius is height above table
        super().__init__('pick_and_place_air_6dof_small_box', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z_sym', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.015, robot='panda',
                         limits_cause_failure=False, failure_causes_done=False, success_causes_done=False,
                         control_frame='b', **PANDA_ADDS_DEF, **kwargs)

