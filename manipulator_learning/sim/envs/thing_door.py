import copy
from gym import spaces
import numpy as np
from numpy.linalg import norm
import transforms3d as tf3d

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.thing_default import PANDA_GRIPPER_CONFIG as DEF_CONFIG
from manipulator_learning.sim.envs.rewards.generic import get_done_suc_fail
import manipulator_learning.sim.utils.general as utils_general


class ThingDoorGeneric(ManipulatorEnv):
  def __init__(self,
               task,
               camera_in_state,
               dense_reward,
               poses_ref_frame,
               init_gripper_pose=((-.29, .7, 0.75), (-.75 * np.pi, 0, .125 * np.pi)),
               init_gripper_random_lim=(.12, .05, .05, 0., 0., 0.),
               state_data=('pos', 'obj_rot_z', 'grip_feedback'),
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
               pos_limits=((.45, -.45, .64), (1.0, .15, 0.9)),
               control_frame='w',
               gripper_force=500,
               random_base_theta_bounds=(0, 0),
               base_random_lim=((0, 0, 0), (0, 0, 0)),
               **kwargs):

    config_dict = copy.deepcopy(DEF_CONFIG)
    config_dict.update(dict(
        init_gripper_pose=init_gripper_pose,
        init_gripper_random_lim=init_gripper_random_lim,
        control_method=control_method,
        gripper_control_method=gripper_control_method,
        gripper_force=gripper_force,
        random_base_theta_bounds=random_base_theta_bounds,
        base_random_lim=base_random_lim,
        pos_limits=pos_limits
    ))

    super().__init__(task, camera_in_state,
                     dense_reward, True, poses_ref_frame, state_data, max_real_time=max_real_time,
                     n_substeps=n_substeps, action_multiplier=action_multiplier,
                     gap_between_prev_pos=gap_between_prev_pos,
                     image_width=image_width, image_height=image_height,
                     failure_causes_done=failure_causes_done, success_causes_done=success_causes_done,
                     control_frame=control_frame, config_dict=config_dict, **kwargs)
    self.reach_radius = reach_radius
    self.reach_radius_time = .5
    self.reach_radius_start_time = None
    self.in_reach_radius = False
    self.limits_cause_failure = limits_cause_failure
    self.done_success_reward = 500  # untuned
    self.done_failure_reward = -5  # untuned

  def _calculate_reward_and_done(self, dense_reward, limit_reached):
    door_quat = self.env._pb_client.getLinkState(self.env.door, linkIndex=1)[1]
    # door is -1.57 closed, -3.14 open, crosses to 3.14 when open a little bit more
    door_angle = tf3d.euler.quat2euler(utils_general.q_convert(door_quat))[2]

    # this converts it to be 4.71 when closed, 3.14 (or lower) when opened
    door_angle = (door_angle + 2 * np.pi) % (2 * np.pi)

    # just in case the door gets pushed open, no false positive success -- wraps back to 0 when it's 90 degrees open
    if door_angle < 1.0:
        door_angle += 2 * np.pi

    door_max = 2.0
    door_min = 0.0
    door_dist = min(max(door_angle - np.pi, door_min), door_max)

    ee_pose_world = self.env.gripper.manipulator.get_link_pose(
        self.env.gripper.manipulator._tool_link_ind, ref_frame_index=None)
    handle_pos_world = self.env._pb_client.getLinkState(self.env.door, linkIndex=4)[0]
    ee_handle_dist = norm(ee_pose_world[:3] - handle_pos_world)
    reward = 1 - np.tanh(10.0 * max(ee_handle_dist, .02)) + (door_max - door_dist)

    return get_done_suc_fail(door_dist, reward, limit_reached, dense_reward, self)


class ThingDoorImage(ThingDoorGeneric):
  def __init__(self, max_real_time=12, n_substeps=10, dense_reward=True,
               image_width=64, image_height=48, success_causes_done=True,
               failure_causes_done=True, **kwargs):
    self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
    self.observation_space = spaces.Dict({
      'obs': spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32),
      'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
      'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
    })
    super().__init__('door', True, dense_reward, 'b',
                     state_data=('pos', 'grip_pos', 'prev_grip_pos'),
                     max_real_time=max_real_time, n_substeps=n_substeps,
                     image_width=image_width, image_height=image_height,
                     success_causes_done=success_causes_done, failure_causes_done=failure_causes_done,
                     reach_radius=.2, control_frame='b', **kwargs)


class ThingDoorMultiview(ThingDoorImage):
  def __init__(self, **kwargs):
    super().__init__(random_base_theta_bounds=(-3 * np.pi / 16, np.pi / 16),
                     base_random_lim=((.02, .02, .002), (0, 0, .02)),
                     **kwargs)