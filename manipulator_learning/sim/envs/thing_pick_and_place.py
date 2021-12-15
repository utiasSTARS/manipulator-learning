import numpy as np
from gym import spaces
import copy

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.thing_default import TWO_FINGER_CONFIG as DEF_CONFIG
from manipulator_learning.sim.envs.configs.panda_default import CONFIG as PANDA_DEF_CONFIG
from manipulator_learning.sim.envs.configs.all_default import MULTIVIEW_DEFS


class ThingPickAndPlaceGeneric(ManipulatorEnv):
    def __init__(self,
                 task,
                 camera_in_state,
                 dense_reward,
                 poses_ref_frame,
                 init_gripper_pose=((-.15, .85, 0.75), (-.75 * np.pi, 0, np.pi/2)),
                 init_gripper_random_lim=None,
                 state_data=('pos', 'obj_pos', 'grip_feedback', 'goal_pos'),
                 max_real_time=5,
                 n_substeps=10,
                 action_multiplier=1.0,
                 reach_radius=.05,
                 gap_between_prev_pos=.2,
                 image_width=160,
                 image_height=120,
                 limits_cause_failure=False,
                 failure_causes_done=False,
                 success_causes_done=False,
                 control_method='v',
                 gripper_control_method='bool_p',
                 pos_limits=((.55, -.45, .64), (1.05, .05, 1.0)),
                 control_frame='w',
                 random_base_theta_bounds=(0, 0),
                 base_random_lim=((0, 0, 0), (0, 0, 0)),
                 goal_pos=(.05, .15),
                 goal_type='plate',
                 block_style='cube',
                 valid_t_dofs=(1, 1, 1),
                 valid_r_dofs=(1, 1, 1),
                 init_block_pos=(),
                 block_random_lim=(),
                 robot_base_ws_cam_tf=((-.4, .65, .9), (-2.45, 0, -.4)),
                 block_colors=None,  # just means default to blue then green
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
            goal_pos=goal_pos,
            goal_type=goal_type,
            block_style=block_style,
            init_block_pos=init_block_pos,
            block_random_lim=block_random_lim,
            robot_base_ws_cam_tf=robot_base_ws_cam_tf,
            block_colors=block_colors
        ))

        super().__init__(task, camera_in_state,
                         dense_reward, True, poses_ref_frame, state_data, max_real_time=max_real_time,
                         n_substeps=n_substeps, action_multiplier=action_multiplier, gap_between_prev_pos=gap_between_prev_pos,
                         image_width=image_width, image_height=image_height,
                         failure_causes_done=failure_causes_done, success_causes_done=success_causes_done,
                         control_frame=control_frame, config_dict=config_dict, valid_t_dofs=valid_t_dofs,
                         valid_r_dofs=valid_r_dofs, **kwargs)
        self.reach_radius = reach_radius
        self.reach_radius_time = .5
        self.reach_radius_start_time = None
        self.in_reach_radius = False
        self.limits_cause_failure = limits_cause_failure
        self.done_success_reward = 100  # untuned
        self.done_failure_reward = -5   # untuned

    def _calculate_reward_and_done(self, dense_reward, limit_reached):
        if 'clutter' in self.task:
            return 0, False, False

        if 'lift' in self.task:
            height_above_table_for_suc = self.reach_radius
            table_height = .6247
            des_block_above_table_height = 1.2 * height_above_table_for_suc
            block_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[0])
            ee_pose_world = self.env.gripper.manipulator.get_link_pose(
                self.env.gripper.manipulator._tool_link_ind, ref_frame_index=None)
            block_ee_dist = np.linalg.norm(np.array(block_pose[0]) - np.array(ee_pose_world[:3]))
            block_table_dist = block_pose[0][2] - table_height
            block_table_dist_scaled = block_table_dist / des_block_above_table_height
            reward = 3 * min(block_table_dist_scaled, 1.0) + 1 - np.tanh(5.0 * block_ee_dist)
            done_success = False
            if block_table_dist > height_above_table_for_suc:
                if self.reach_radius_start_time is None:
                    self.reach_radius_start_time = self.ep_timesteps
                elif (self.ep_timesteps - self.reach_radius_start_time) * self.real_t_per_ts > self.reach_radius_time:
                    done_success = True
            else:
                self.reach_radius_start_time = None
            done_failure = False
        else:
            block_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[0])
            goal_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.goal_id)
            if self.task == 'sort_2' or self.task == 'sort_3':
                block2_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[1])
                goal2_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.goal2_id)
            if self.task == 'sort_3':
                block3_pose = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[2])
            ee_pose_world = self.env.gripper.manipulator.get_link_pose(
                self.env.gripper.manipulator._tool_link_ind, ref_frame_index=None)
            block_ee_dist = np.linalg.norm(np.array(block_pose[0]) - np.array(ee_pose_world[:3]))
            block_goal_dist = np.linalg.norm(np.array(block_pose[0]) - np.array(goal_pose[0]))
            if self.task == 'sort_2' or self.task == 'sort_3':
                block2_goal_dist = np.linalg.norm(np.array(block2_pose[0]) - np.array(goal2_pose[0]))
            if self.task == 'sort_2':
                reward = 3 * (1 - np.tanh(10.0 * block_goal_dist)) + 1 - np.tanh(10.0 * block2_goal_dist)
            if self.task == 'sort_3':
                block3_goal_dist = np.linalg.norm(np.array(block3_pose[0]) - np.array(goal_pose[0]))
                reward =  (1 - np.tanh(10.0 * block_goal_dist)) + 1 - np.tanh(10.0 * block2_goal_dist) + \
                          1 - np.tanh(10.0 * block3_goal_dist)
            else:
                reward = 3*(1 - np.tanh(10.0 * block_goal_dist)) + 1 - np.tanh(10.0 * block_ee_dist)
            done_success = False
            if (self.task != 'sort_2' and self.task != 'sort_3' and block_goal_dist < self.reach_radius) or \
                (self.task == 'sort_2' and block_goal_dist < self.reach_radius and block2_goal_dist < self.reach_radius) or \
                (self.task == 'sort_3' and block_goal_dist < self.reach_radius and block2_goal_dist < self.reach_radius and
                block3_goal_dist < self.reach_radius):
                if self.reach_radius_start_time is None:
                    self.reach_radius_start_time = self.ep_timesteps
                elif (self.ep_timesteps - self.reach_radius_start_time) * self.real_t_per_ts > self.reach_radius_time:
                    done_success = True
            else:
                self.reach_radius_start_time = None
            done_failure = False
            # num_contact_points = len(self.env._pb_client.getContactPoints(self.env.block_ids[0], self.env.table))

        if self.limits_cause_failure and limit_reached:
            done_failure = True
            done_success = False

        if self.success_causes_done and done_success:
            reward = self.done_success_reward
        if self.failure_causes_done and done_failure:
            reward = self.done_failure_reward

        if dense_reward:
            return reward, done_success, done_failure
        else:
            return done_success, done_success, done_failure


# ----------------------------------------------------------------------------------------------------------
# State Envs
# ----------------------------------------------------------------------------------------------------------

# XY
# ----------------------------------------------------------------------------------------------------------
XY_DEFS = dict(
    block_style='cylinder',
    goal_type='plate',
    control_frame='w',
    valid_t_dofs=[1, 1, 0],
    valid_r_dofs=[0, 0, 0],
    block_random_lim=[[.25, .1]],
    init_block_pos=[[.05, 0.15]],
    pos_limits=[[.55, -.45], [1.05, .05]],
    init_gripper_pose=[[-.15, .85, 0.64], [-.75 * np.pi, 0, np.pi/2]],
    goal_pos=[.05, 0.0]
)


class ThingPickAndPlaceXYState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=7, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        super().__init__('pick_and_place_xy', False, dense_reward, 'w', max_real_time=max_real_time,
                         n_substeps=n_substeps, **XY_DEFS, **kwargs)


class ThingPickAndPlacePrevPosXYState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=7, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        super().__init__('pick_and_place_xy', False, dense_reward, 'w',
                         state_data=('pos', 'prev_pos', 'obj_pos', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         **XY_DEFS, **kwargs)


class ThingPickAndPlaceGripPosXYState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=7, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)

        XY_MOD = copy.deepcopy(XY_DEFS)
        XY_MOD['block_random_lim'] = [[.1, .25]]
        XY_MOD['init_block_pos'] = [[.15, .05]]
        XY_MOD['goal_pos'] = [-.05, 0.0]

        super().__init__('pick_and_place_xy_vert', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'obj_pos', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         **XY_MOD, **kwargs)


# XYZ
# ----------------------------------------------------------------------------------------------------------
XYZ_DEFS = dict(
    block_style='cube',
    goal_type='plate',
    control_frame='w',
    valid_t_dofs=[1, 1, 1],
    valid_r_dofs=[0, 0, 0],
    block_random_lim=[[.25, .1]],
    init_block_pos=[[.05, 0.15]],
    pos_limits=[[.55, -.45, .64], [1.0, .05, 0.8]],
    goal_pos=[0.0, 0.15]
)


class ThingPickAndPlaceXYZState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        super().__init__('pick_and_place_xyz', False, dense_reward, 'w', max_real_time=max_real_time,
                         n_substeps=n_substeps, **XYZ_DEFS, **kwargs)


class ThingPickAndPlaceGripPosXYZState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), dtype=np.float32)
        super().__init__('pick_and_place_xyz', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'obj_pos', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, **XYZ_DEFS, **kwargs)


# goal in air, designed to mimic FetchPickAndPlace Env
class ThingPickAndPlaceAirGoalXYZState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)

        XYZ_MOD = copy.deepcopy(XYZ_DEFS)
        XYZ_MOD['goal_type'] = 'air'

        super().__init__('pick_and_place_air_xyz', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.015,
                         **XYZ_MOD, **kwargs)


# 6DOF
# ----------------------------------------------------------------------------------------------------------
SIXDOF_DEFS = dict(
    block_style='cube',
    goal_type='plate',
    control_frame='w',
    valid_t_dofs=[1, 1, 1],
    valid_r_dofs=[1, 1, 1],
    block_random_lim=[[.25, .1]],
    init_block_pos=[[.05, -.05]],
    pos_limits=[[.55, -.45, .64], [.9, .05, 0.8]],
    goal_pos=[0.0, 0.15]
)


class ThingPickAndPlace6DofState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)
        super().__init__('pick_and_place_6dof', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'obj_pos', 'obj_rot_z', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, **SIXDOF_DEFS, **kwargs)


class ThingPickAndPlace6DofLongState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)

        SIXDOF_MOD = copy.deepcopy(SIXDOF_DEFS)
        SIXDOF_MOD['block_style'] = 'long'

        super().__init__('pick_and_place_6dof_long', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'obj_pos', 'obj_rot_z_sym', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.075, **SIXDOF_MOD, **kwargs)


class ThingPickAndPlace6DofSmallState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)

        SIXDOF_MOD = copy.deepcopy(SIXDOF_DEFS)
        SIXDOF_MOD['block_random_lim'] = [[.05, .05]]
        SIXDOF_MOD['block_style'] = 'long'

        super().__init__('pick_and_place_6dof_small_box', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'obj_pos', 'obj_rot_z_sym', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.075, **SIXDOF_MOD, **kwargs)


# goal in air, designed to mimic FetchPickAndPlace Env
class ThingPickAndPlaceAirGoal6DofState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)

        SIXDOF_MOD = copy.deepcopy(SIXDOF_DEFS)
        SIXDOF_MOD['block_random_lim'] = [[.05, .05]]
        SIXDOF_MOD['block_style'] = 'long'
        SIXDOF_MOD['goal_type'] = 'air'
        SIXDOF_MOD['pos_limits'] = [[.55, -.45, .64], [1.0, .05, 0.8]]

        super().__init__('pick_and_place_air_6dof_small_box', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'obj_pos', 'obj_rot_z_sym', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.015, **SIXDOF_MOD, **kwargs)


# ----------------------------------------------------------------------------------------------------------
# SAC-X Envs -- envs designed to mimic environments from SAC-X paper
# ----------------------------------------------------------------------------------------------------------
# Time limit based on appendix of paper --  1200 episodes corresponds to 10 hours = 36000s
# 36000s / 1200 eps = 30s, in their case = 600 timesteps
# objects are on 60cm x 30cm table in any position
# gripper initialized randomly above table-top with height offset 10-20cm above table
# for XYZ envs, gripper always starts facing down
# intention length in paper is set to 150 timesteps = 7.5s

LIFTBRING_DEFS = dict(
    block_style='cube',
    goal_type=None,
    control_frame='b',
    valid_t_dofs=[1, 1, 1],
    valid_r_dofs=[0, 0, 0],
    block_random_lim=[[.25, .25]],
    init_block_pos=[[0.0, 0.0]],
    pos_limits=[[.55, -.45, .66], [0.85, .05, 0.8]],
    init_gripper_pose=[[-.29, .64, 0.75], [-.75 * np.pi, 0, .75 * np.pi]],
    init_gripper_random_lim=[.25, .25, .06, 0., 0., 0.],
    robot_base_ws_cam_tf=((-.4, .5, 1.0), (-2.55, 0, -.4))
)


class ThingBringXYZState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=18, n_substeps=10, dense_reward=True, action_multiplier=0.1, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)

        BRING_MOD = copy.deepcopy(LIFTBRING_DEFS)
        BRING_MOD['goal_type'] = 'coaster'
        BRING_MOD['goal_pos'] = [0.0, .1]

        super().__init__('bring_xyz', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z', 'goal_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, 
                         action_multiplier=action_multiplier, reach_radius=.015, **BRING_MOD, **kwargs)


# ----------------------------------------------------------------------------------------------------------
# Multiview Paper Envs -- envs to be used with Multiview BC paper
# ----------------------------------------------------------------------------------------------------------

LIFT_MULTIVIEW_DEFS = copy.deepcopy(LIFTBRING_DEFS)
LIFT_MULTIVIEW_DEFS.update(MULTIVIEW_DEFS)


# Use this one to get an expert, and then create an image-based version with shorter max real time
# to use the expert in to generate non-human expert data
class ThingLiftXYZStateMultiview(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=18, n_substeps=10, dense_reward=True, action_multiplier=0.05, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        # for lift env, reach radius is height above table
        super().__init__('lift_xyz', False, dense_reward, 'b',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.1,
                         image_width=64, image_height=48, **LIFT_MULTIVIEW_DEFS, **kwargs)


class ThingLiftXYZState(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=18, n_substeps=10, dense_reward=True, action_multiplier=0.05, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        # for lift env, reach radius is height above table
        super().__init__('lift_xyz', False, dense_reward, 'b',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.1,
                         image_width=64, image_height=48, **LIFTBRING_DEFS, **kwargs)


class ThingLiftXYZMultiview(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=6, n_substeps=10, dense_reward=False, action_multiplier=0.05,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        # for lift env, reach radius is height above table
        super().__init__('lift_xyz', True, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.1,
                         image_width=image_width, image_height=image_height, **LIFT_MULTIVIEW_DEFS, **kwargs)


class ThingLiftXYZImage(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=6, n_substeps=10, dense_reward=False, action_multiplier=0.05,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        # for lift env, reach radius is height above table
        super().__init__('lift_xyz', True, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.1,
                         image_width=image_width, image_height=image_height, **LIFTBRING_DEFS, **kwargs)


# ----------------------------------------------------------------------------------------------------------
# Image Envs
# ----------------------------------------------------------------------------------------------------------

SIXDOF_IMAGE_DEFS = copy.deepcopy(SIXDOF_DEFS)
SIXDOF_IMAGE_DEFS['control_frame'] = 'b'
SIXDOF_IMAGE_DEFS['block_style'] = 'long'
SIXDOF_IMAGE_DEFS['block_random_lim'] = [[.05, .05]]


class ThingPickAndPlace6DofSmallImage(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('pick_and_place_6dof_small_box_image', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.075,
                         image_width=image_width, image_height=image_height, **SIXDOF_IMAGE_DEFS, **kwargs)


class ThingPickAndPlace6DofSmall160120Image(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True, **kwargs):
        image_height = 120
        image_width = 160
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('pick_and_place_6dof_small_box_image', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.075,
                         image_width=image_width, image_height=image_height, **SIXDOF_IMAGE_DEFS, **kwargs)


class ThingPickAndPlace6DofSmallMultiview(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=8, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })

        CONFIG = copy.deepcopy(SIXDOF_IMAGE_DEFS)
        CONFIG.update(MULTIVIEW_DEFS)

        super().__init__('pick_and_place_6dof_small_box_image', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.075,
                         image_width=image_width, image_height=image_height, **CONFIG, **kwargs)


SORT_DEFS = copy.deepcopy(SIXDOF_IMAGE_DEFS)
SORT_DEFS.update(MULTIVIEW_DEFS)
SORT_DEFS['pos_limits'] = [[.55, -.45, .64], [1.0, .05, 0.8]]
SORT_DEFS['init_block_pos'] = [[0.0, -0.05], [0.0, -.05]]
SORT_DEFS['block_random_lim'] = [[.1, .1], [.1, .1]]
SORT_DEFS['block_style'] = 'cube'
SORT_DEFS['goal_pos'] = [.2, .15]


class ThingSort2Multiview(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=12, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })
        super().__init__('sort_2', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.075,
                         image_width=image_width, image_height=image_height, **SORT_DEFS, **kwargs)


class ThingSort3Multiview(ThingPickAndPlaceGeneric):
    def __init__(self, max_real_time=15, n_substeps=10, dense_reward=True,
                 image_width=64, image_height=48, **kwargs):
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
        })

        CONFIG = copy.deepcopy(SORT_DEFS)
        CONFIG['init_block_pos'] = [[0.0, -0.05], [0.0, -.05], [0.0, -.05]]
        CONFIG['block_random_lim'] = [[.05, .05], [.05, .05], [.05, .05]]
        CONFIG['block_colors'] = ['blue', 'green', 'blue']

        super().__init__('sort_3', True, dense_reward, 'b',
                         state_data=('pos', 'grip_pos'),
                         max_real_time=max_real_time, n_substeps=n_substeps, reach_radius=.1,
                         image_width=image_width, image_height=image_height, **CONFIG, **kwargs)
