import numpy as np
import gym
from gym import spaces
import time
import copy

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.thing_default import ROD_CONFIG as DEF_CONFIG
from manipulator_learning.sim.envs.configs.all_default import MULTIVIEW_DEFS


class ThingPushingGeneric(ManipulatorEnv):
    def __init__(self,
                 task,
                 camera_in_state,
                 dense_reward,
                 poses_ref_frame,
                 init_gripper_pose=((-.15, .85, 0.75), (-.75 * np.pi, 0, np.pi / 2)),
                 init_gripper_random_lim=None,
                 state_data=('pos', 'obj_pos', 'obj_rot_z_first_only'),
                 max_real_time=5,
                 n_substeps=10,
                 reach_radius=.085,
                 gap_between_prev_pos=.2,
                 image_width=160,
                 image_height=120,
                 control_method='v',
                 pos_limits=((.55, -.45, .64), (1.05, .05, 1.0)),
                 control_frame='b',
                 random_base_theta_bounds=(0, 0),
                 base_random_lim=((0, 0, 0), (0, 0, 0)),
                 goal_pos=(.05, .15),
                 goal_type='plate',
                 block_style='cube',
                 valid_t_dofs=(1, 1, 1),
                 valid_r_dofs=(1, 1, 1),
                 init_block_pos=(),
                 block_random_lim=(),
                 **kwargs):

        config_dict = copy.deepcopy(DEF_CONFIG)
        config_dict.update(dict(
            init_gripper_pose=init_gripper_pose,
            init_gripper_random_lim=init_gripper_random_lim,
            control_method=control_method,
            random_base_theta_bounds=random_base_theta_bounds,
            base_random_lim=base_random_lim,
            pos_limits=pos_limits,
            goal_pos=goal_pos,
            goal_type=goal_type,
            block_style=block_style,
            init_block_pos=init_block_pos,
            block_random_lim=block_random_lim
        ))

        super().__init__(task, camera_in_state,
                         dense_reward, False, poses_ref_frame, state_data, max_real_time=max_real_time,
                         n_substeps=n_substeps, gap_between_prev_pos=gap_between_prev_pos,
                         image_width=image_width, image_height=image_height, valid_t_dofs=valid_t_dofs,
                         valid_r_dofs=valid_r_dofs, control_frame=control_frame, config_dict=config_dict, **kwargs)
        self.reach_radius = reach_radius
        self.reach_radius_time = .5
        self.reach_radius_start_time = None
        self.in_reach_radius = False

    def _calculate_reward_and_done(self, dense_reward, limit_reached, limits_cause_failure=False):
        block_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[0])
        if 'coaster' in self.task:
            goal_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.goal_id)
        else:
            goal_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[1])
        ee_pose_world = self.env.gripper.manipulator.get_link_pose(
            self.env.gripper.manipulator._tool_link_ind, ref_frame_index=None)
        block_ee_dist = np.linalg.norm(np.array(block_pose[0]) - np.array(ee_pose_world[:3]))
        block_goal_dist = np.linalg.norm(np.array(block_pose[0]) - np.array(goal_pose[0]))
        reward = 3*(1 - np.tanh(10.0 * block_goal_dist)) + 1 - np.tanh(10.0 * block_ee_dist)
        done_success = False
        if block_goal_dist < self.reach_radius:
            if self.reach_radius_start_time is None:
                self.reach_radius_start_time = self.ep_timesteps
            elif (self.ep_timesteps - self.reach_radius_start_time) * self.real_t_per_ts > self.reach_radius_time:
                done_success = True
        else:
            self.reach_radius_start_time = None
        done_failure = False
        if limits_cause_failure and limit_reached:
            done_failure = True
            done_success = False
        if dense_reward:
            return reward, done_success, done_failure
        else:
            return done_success, done_success, done_failure


XY_DEFS = dict(
    block_style='low_fric',
    goal_type=None,
    control_frame='w',
    valid_t_dofs=[1, 1, 0],
    valid_r_dofs=[0, 0, 0],
    block_random_lim=[[.25, .1], [0, 0]],
    init_block_pos=[[0, 0], [0, .2]],
    pos_limits=[[.55, -.45], [1.05, .05]],
    init_gripper_pose=[[-.15, .85, 0.64], [-.75 * np.pi, 0, np.pi/2]],
)


class ThingPushingXYState(ThingPushingGeneric):
    def __init__(self, max_real_time=7, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        super().__init__('pushing_xy', False, dense_reward, 'w', max_real_time=max_real_time, n_substeps=n_substeps,
                         **XY_DEFS, **kwargs)


class ThingPushingXYImage(ThingPushingGeneric):
    def __init__(self, max_real_time=7, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            'img': spaces.Box(0, 255, (160, 120, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 255, (160, 120), dtype=np.uint8),
        })
        super().__init__('pushing_xy', True, dense_reward, 'b', state_data=('pos', 'prev_pos',),
                         max_real_time=max_real_time, n_substeps=n_substeps, **XY_DEFS, **kwargs)


class ThingPushing6DofMultiview(ThingPushingGeneric):
    def __init__(self, max_real_time=12, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })

        CONFIG = dict(
            block_style='low_fric',
            block_random_lim=[[.1, .1]],
            init_block_pos=[[0, -.05]],
            goal_type='coaster',
            pos_limits=((.55, -.45, .64), (.9, .15, 0.8)),
            control_frame='b',
            goal_pos=[.05, .15],
            valid_t_dofs=[1, 1, 1],
            valid_r_dofs=[1, 1, 1],
        )
        CONFIG.update(MULTIVIEW_DEFS)
        CONFIG['random_base_theta_bounds'] = (-np.pi / 8, np.pi / 8)

        super().__init__('pushing_6dof_coaster', True, dense_reward, 'b',
                         state_data=('pos',),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         reach_radius=.033, **CONFIG, **kwargs)
