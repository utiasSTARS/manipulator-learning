import numpy as np
import gym
from gym import spaces
import time
import copy

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.thing_default import TWO_FINGER_CONFIG as DEF_CONFIG
from manipulator_learning.sim.envs.configs.panda_default import CONFIG as PANDA_DEF_CONFIG
from manipulator_learning.sim.envs.configs.all_default import MULTIVIEW_DEFS


class ThingReachingGeneric(ManipulatorEnv):
    def __init__(self,
                 task,
                 camera_in_state,
                 dense_reward,
                 poses_ref_frame,
                 init_gripper_pose=((-.15, .85, 0.65), (-.65 * np.pi, 0, np.pi / 2)),
                 init_gripper_random_lim=None,
                 state_data=('pos', 'prev_pos', 'obj_pos'),
                 max_real_time=5,
                 n_substeps=10,
                 action_multiplier=1.0,
                 reach_radius=.035,
                 gap_between_prev_pos=.2,
                 image_width=160,
                 image_height=120,
                 control_method='v',
                 pos_limits=((.55, -.45, .64), (1.05, .05, 1.0)),
                 control_frame='b',
                 random_base_theta_bounds=(0, 0),
                 base_random_lim=((0, 0, 0), (0, 0, 0)),
                 block_style='cube',
                 valid_t_dofs=(1, 1, 1),
                 valid_r_dofs=(1, 1, 1),
                 init_block_pos=(),
                 block_random_lim=(),
                 limits_cause_failure=False,
                 failure_causes_done=False,
                 success_causes_done=False,
                 robot='thing_2_finger',
                 robot_base_ws_cam_tf=((-.4, .65, .9), (-2.45, 0, -.4)),
                 **kwargs):

        if robot == 'thing_2_finger':
            config_dict = copy.deepcopy(DEF_CONFIG)
        elif robot == 'panda':
            config_dict = copy.deepcopy(PANDA_DEF_CONFIG)
        else:
            raise NotImplementedError()
        config_dict.update(dict(
            init_gripper_pose=init_gripper_pose,
            init_gripper_random_lim=init_gripper_random_lim,
            control_method=control_method,
            random_base_theta_bounds=random_base_theta_bounds,
            base_random_lim=base_random_lim,
            pos_limits=pos_limits,
            block_style=block_style,
            init_block_pos=init_block_pos,
            block_random_lim=block_random_lim,
            robot_base_ws_cam_tf=robot_base_ws_cam_tf
        ))

        super().__init__(task, camera_in_state,
                         dense_reward, False, poses_ref_frame, state_data, max_real_time=max_real_time,
                         n_substeps=n_substeps, action_multiplier=action_multiplier, gap_between_prev_pos=gap_between_prev_pos,
                         image_width=image_width, image_height=image_height,
                         failure_causes_done=failure_causes_done, success_causes_done=success_causes_done,
                         control_frame=control_frame, valid_t_dofs=valid_t_dofs,
                         valid_r_dofs=valid_r_dofs, config_dict=config_dict, **kwargs)
        self.reach_radius = reach_radius
        self.reach_radius_time = .5
        self.reach_radius_start_time = None
        self.in_reach_radius = False
        self.limits_cause_failure = limits_cause_failure
        self.done_success_reward = 100  # hard coded for now, may not work
        self.done_failure_reward = -5  # hard coded for now, may not work

    def _calculate_reward_and_done(self, dense_reward, limit_reached, limits_cause_failure=False):
        block_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[0])
        ee_pose_world = self.env.gripper.manipulator.get_link_pose(
            self.env.gripper.manipulator._tool_link_ind, ref_frame_index=None)
        block_ee_dist = np.linalg.norm(np.array(block_pose[0]) - np.array(ee_pose_world[:3]))
        reward = 1 - np.tanh(10.0 * block_ee_dist)
        done_success = False
        if block_ee_dist < self.reach_radius:
            if self.reach_radius_start_time is None:
                self.reach_radius_start_time = self.ep_timesteps
            elif (self.ep_timesteps - self.reach_radius_start_time) * self.real_t_per_ts > self.reach_radius_time:
                done_success = True
        else:
            self.reach_radius_start_time = None
        done_failure = False
        # num_contact_points = len(self.env._pb_client.getContactPoints(self.env.block_ids[0], self.env.table))
        if limits_cause_failure and limit_reached:
            done_failure = True
            done_success = False
        if dense_reward:
            return reward, done_success, done_failure
        else:
            return done_success, done_success, done_failure


XY_DEFS = dict(
    block_style='vis_only',
    control_frame='w',
    valid_t_dofs=[1, 1, 0],
    valid_r_dofs=[0, 0, 0],
    block_random_lim=[[.35, .35]],
    init_block_pos=[[.1, 0.0]],
    pos_limits=[[.55, -.45], [1.05, .05]],
)


class ThingReachingXYState(ThingReachingGeneric):
    def __init__(self, max_real_time=4, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        super().__init__('reaching_xy', False, dense_reward, 'w',
                         state_data=('pos', 'obj_pos'), max_real_time=max_real_time, n_substeps=n_substeps,
                         reach_radius=.045, **XY_DEFS, **kwargs)

class ThingReachingXYImage(ThingReachingGeneric):
    def __init__(self, max_real_time=4, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            'img': spaces.Box(0, 255, (160, 120, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 255, (160, 120), dtype=np.uint8),
        })

        spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
        super().__init__('reaching_xy', True, dense_reward, 'b', state_data=('pos', 'prev_pos'), max_real_time=max_real_time,
                         n_substeps=n_substeps, **XY_DEFS, **kwargs)