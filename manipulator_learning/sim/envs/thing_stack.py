import numpy as np
from gym import spaces
import copy

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.thing_default import TWO_FINGER_CONFIG as DEF_CONFIG
from manipulator_learning.sim.envs.configs.panda_default import CONFIG as PANDA_DEF_CONFIG
from manipulator_learning.sim.envs.configs.all_default import MULTIVIEW_DEFS


STACK_MULTIVIEW_DEFS = copy.deepcopy(MULTIVIEW_DEFS)
STACK_MULTIVIEW_DEFS['random_base_theta_bounds'] = (-np.pi / 8, np.pi / 8)


class ThingStackGeneric(ManipulatorEnv):
    def __init__(self,
                 task,
                 camera_in_state,
                 dense_reward,
                 poses_ref_frame,
                 init_gripper_pose=((-.15, .85, 0.75), (-.75 * np.pi, 0, np.pi / 2)),
                 init_gripper_random_lim=None,
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
                 pos_limits=((.55, -.45, .64), (.9, .05, 0.8)),
                 control_frame='b',
                 init_block_pos=((-.05, 0), (0, .15)),
                 block_random_lim=((.1, .1), (.1, .1)),
                 random_base_theta_bounds=(0, 0),
                 base_random_lim=((0, 0, 0), (0, 0, 0)),
                 block_style='cube',
                 robot_base_ws_cam_tf=((-.4, .65, .9), (-2.45, 0, -.4)),
                 robot='thing_2_finger',
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
            gripper_control_method=gripper_control_method,
            random_base_theta_bounds=random_base_theta_bounds,
            base_random_lim=base_random_lim,
            pos_limits=pos_limits,
            init_block_pos=init_block_pos,
            block_random_lim=block_random_lim,
            block_style=block_style,
            robot_base_ws_cam_tf=robot_base_ws_cam_tf
        ))

        super().__init__(task, camera_in_state,
                         dense_reward, True, poses_ref_frame, state_data, max_real_time=max_real_time,
                         n_substeps=n_substeps, gap_between_prev_pos=gap_between_prev_pos,
                         image_width=image_width, image_height=image_height,
                         failure_causes_done=failure_causes_done, success_causes_done=success_causes_done,
                         control_frame=control_frame, config_dict=config_dict, **kwargs)
        self.stack_time = 0.5  # time for 3 conditions: blocks touching, 2nd block off table, and grip not touch block
        self.stack_start_time = None
        self.limits_cause_failure = limits_cause_failure
        self.done_success_reward = 100  # hard coded for now, may not work
        self.done_failure_reward = -5  # hard coded for now, may not work

    def _calculate_reward_and_done(self, dense_reward, limit_reached, limits_cause_failure=False):
        if len(self.env.block_ids) == 3:
            num_stack = 3
        else:
            num_stack = 2
        b_poses = []
        for i in range(num_stack):
            b_poses.append(self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[i]))

        ee_pose_world = self.env.gripper.manipulator.get_link_pose(
            self.env.gripper.manipulator._tool_link_ind, ref_frame_index=None)
        block1_ee_dist = np.linalg.norm(np.array(b_poses[0][0]) - np.array(ee_pose_world[:3]))
        block1_block2_dist = np.linalg.norm(np.array(b_poses[0][0]) - np.array(b_poses[1][0]))
        if num_stack == 2:
            reward = 3*(1 - np.tanh(10.0 * block1_ee_dist)) + 1 - np.tanh(10.0 * block1_block2_dist)
        elif num_stack == 3:
            block2_block3_dist = np.linalg.norm(np.array(b_poses[2][0]) - np.array(b_poses[1][0]))
            reward = (1 - np.tanh(10. * block1_block2_dist) + 1 - np.tanh(10.0 * block2_block3_dist))
        done_success = False

        # need all three conditions to determine success
        b2b_contact = len(self.env._pb_client.getContactPoints(self.env.block_ids[0], self.env.block_ids[1])) > 0
        b2t_contact = len(self.env._pb_client.getContactPoints(self.env.block_ids[0], self.env.table)) > 0
        b2ee_contact = len(self.env._pb_client.getContactPoints(self.env.gripper.manipulator._arm[0],
                                                                self.env.block_ids[0])) > 0
        if num_stack == 3:
            b32ee_contact = len(self.env._pb_client.getContactPoints(self.env.gripper.manipulator._arm[0],
                                                                self.env.block_ids[2])) > 0
            b32b2_contact = len(self.env._pb_client.getContactPoints(self.env.block_ids[2], self.env.block_ids[1])) > 0
            b22t_contact = len(self.env._pb_client.getContactPoints(self.env.block_ids[1], self.env.table)) > 0
            b32t_contact = len(self.env._pb_client.getContactPoints(self.env.block_ids[2], self.env.table)) > 0

        if (num_stack == 2 and (b2b_contact and not b2t_contact and not b2ee_contact)) or \
            (num_stack == 3 and (b2b_contact and not b32ee_contact and b32b2_contact and not b22t_contact
                                 and not b32t_contact)):
            if self.stack_start_time is None:
                self.stack_start_time = self.ep_timesteps
            elif (self.ep_timesteps - self.stack_start_time) * self.real_t_per_ts > self.stack_time:
                done_success = True
        else:
            self.stack_start_time = None
        done_failure = False
        if limits_cause_failure and limit_reached:
            done_failure = True
            done_success = False
        if dense_reward:
            return reward, done_success, done_failure
        else:
            return done_success, done_success, done_failure


class ThingStackImage(ThingStackGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('stack_2', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height, **kwargs)


class ThingStackMultiview(ThingStackGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('stack_2', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height, **STACK_MULTIVIEW_DEFS, **kwargs)

TWO_SMALL_DEFAULTS = dict(
    block_random_lim=[[.05, .05], [.05, .05]],
    init_block_pos=[[-.05, 0], [0, .15]]
)


class ThingStackSmallMultiview(ThingStackGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })

        super().__init__('stack_2_small', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         **TWO_SMALL_DEFAULTS, **STACK_MULTIVIEW_DEFS, **kwargs)


TWO_SAME_DEFAULTS = dict(
    block_random_lim=[[.15, .15], [.15, .15]],
    init_block_pos=[[0., 0.], [0., 0.]]
)


class ThingStackSameMultiview(ThingStackGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('stack_2_same', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         **TWO_SAME_DEFAULTS, **STACK_MULTIVIEW_DEFS, **kwargs)


class ThingStackSameMultiviewV2(ThingStackGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, success_causes_done=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })

        super().__init__('stack_2_same', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         success_causes_done=success_causes_done,
                         **TWO_SAME_DEFAULTS, **STACK_MULTIVIEW_DEFS, **kwargs)


class ThingStackSameImageV2(ThingStackGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, success_causes_done=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('stack_2_same', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         success_causes_done=success_causes_done, **TWO_SAME_DEFAULTS, **kwargs)


class ThingStack3Multiview(ThingStackGeneric):
    def __init__(self, max_real_time=15, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })

        CONFIG = copy.deepcopy(STACK_MULTIVIEW_DEFS)
        CONFIG.update(dict(
            block_random_lim=[[.05, .05], [.05, .05], [.05, .05]],
            init_block_pos=[[0.0, .1], [.1, 0], [-.1, 0]],
            pos_limits=[[.55, -.45, .64], [.9, .05, 0.9]]
        ))

        super().__init__('stack_3', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         **CONFIG, **kwargs)


class ThingStackTallMultiview(ThingStackGeneric):
    def __init__(self, max_real_time=10, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })

        CONFIG = copy.deepcopy(TWO_SMALL_DEFAULTS)
        CONFIG.update(dict(
            block_style='tall_narrow',
            pos_limits=[[.55, -.45, .64], [.9, .05, 0.9]]
        ))

        super().__init__('stack_2_narrow_small', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         image_width=image_width, image_height=image_height,
                         **CONFIG, **STACK_MULTIVIEW_DEFS, **kwargs)
