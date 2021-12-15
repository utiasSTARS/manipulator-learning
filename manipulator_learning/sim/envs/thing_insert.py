import copy
import numpy as np
from gym import spaces

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.thing_default import TWO_FINGER_CONFIG as DEF_CONFIG
from manipulator_learning.sim.envs.rewards.generic import get_done_suc_fail


class ThingInsertGeneric(ManipulatorEnv):
    def __init__(self,
                 task,
                 camera_in_state,
                 dense_reward,
                 poses_ref_frame,
                 init_gripper_pose=((-.15, .85, 0.75), (-.75 * np.pi, 0, np.pi/2)),
                 state_data=('pos', 'obj_pos', 'grip_feedback', 'goal_pos'),
                 max_real_time=5,
                 n_substeps=10,
                 gap_between_prev_pos=.2,
                 image_width=160,
                 image_height=120,
                 limits_cause_failure=False,
                 failure_causes_done=False,
                 success_causes_done=False,
                 control_method='v',
                 gripper_control_method='bool_p',
                 pos_limits=((.55, -.45, .64), (1.05, .05, 1.0)),
                 control_frame='b',
                 random_base_theta_bounds=(0, 0),
                 base_random_lim=((0, 0, 0), (0, 0, 0)),
                 gripper_default_close=False,
                 init_rod_pos=(-.025, -.1),
                 rod_random_lim=(0.025, 0.025),
                 max_gripper_vel=0.8,
                 **kwargs):

        config_dict = copy.deepcopy(DEF_CONFIG)
        config_dict.update(dict(
            init_gripper_pose=init_gripper_pose,
            control_method=control_method,
            gripper_control_method=gripper_control_method,
            random_base_theta_bounds=random_base_theta_bounds,
            base_random_lim=base_random_lim,
            pos_limits=pos_limits,
            gripper_default_close=gripper_default_close,
            init_rod_pos=init_rod_pos,
            rod_random_lim=rod_random_lim,
            max_gripper_vel=max_gripper_vel,
        ))

        super().__init__(task, camera_in_state,
                         dense_reward, True, poses_ref_frame, state_data, max_real_time=max_real_time,
                         n_substeps=n_substeps, gap_between_prev_pos=gap_between_prev_pos,
                         image_width=image_width, image_height=image_height,
                         failure_causes_done=failure_causes_done, success_causes_done=success_causes_done,
                         control_frame=control_frame, config_dict=config_dict, **kwargs)
        self.reach_radius = .0015
        self.reach_radius_time = .01
        self.reach_radius_start_time = None
        self.in_reach_radius = False
        self.limits_cause_failure = limits_cause_failure
        self.done_success_reward = 500  # hard coded for now, may not work
        self.done_failure_reward = -5  # hard coded for now, may not work

    def _calculate_reward_and_done(self, dense_reward, limit_reached, limits_cause_failure=False):
        rod_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.insertion_rod)
        box_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.insertion_box)
        ee_pose_world = self.env.gripper.manipulator.get_link_pose(
            self.env.gripper.manipulator._tool_link_ind, ref_frame_index=None)
        rod_ee_dist = np.linalg.norm(np.array(rod_pose[0]) - np.array(ee_pose_world[:3]))
        rod_box_dist = np.linalg.norm(np.array(rod_pose[0]) - np.array(box_pose[0]))
        reward = 3 * (1 - np.tanh(10.0 * rod_ee_dist)) + 10 * (1 - np.tanh(10.0 * rod_box_dist))

        return get_done_suc_fail(rod_box_dist, reward, limit_reached, dense_reward, self)


class ThingInsertImage(ThingInsertGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('insertion', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         gripper_default_close=True, pos_limits=[[.55, -.45, .64], [.9, .05, 0.8]], **kwargs)


class ThingInsertMultiview(ThingInsertImage):
    def __init__(self, **kwargs):
        super().__init__(random_base_theta_bounds=(-3 * np.pi / 16, np.pi / 16),
                         base_random_lim=((.02, .02, .002), (0, 0, .02)),
                         **kwargs)


class ThingPickAndInsertSucDoneImage(ThingInsertGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, success_causes_done=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('pick_insertion_small_fast_grip', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, success_causes_done=success_causes_done,
                         image_width=image_width, image_height=image_height, control_frame='b',
                         max_gripper_vel=2.4, pos_limits=[[.55, -.45, .64], [.9, .05, 0.9]], **kwargs)


class ThingPickAndInsertSucDoneMultiview(ThingPickAndInsertSucDoneImage):
    def __init__(self, **kwargs):
        super().__init__(random_base_theta_bounds=(-3 * np.pi / 16, np.pi / 16),
                         base_random_lim=((.02, .02, .002), (0, 0, .02)),
                         **kwargs)
