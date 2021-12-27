import time
import numpy as np
from multiprocessing import Process, Value

from manipulator_learning.learning.imitation.device_utils import Button
from manipulator_learning.learning.imitation.device_utils import force_feedback


class Intervenor:
  BUTTONS = ['start_stop_intervention', 'bad_failure_prediction', 'good_failure_prediction', 'too_late']
  BUTTONS_TABLE = dict(
    keyboard=['enter', 'b', 'g', 't'],
    gamepad=['B', 'X', 'A', 'Y'],
    vr=['trackpad_right_click', 'trackpad_left_click', 'trackpad_down_click', 'trackpad_up_click']
  )

  def __init__(self, device_type, env, intervene_on_touch=True, real_time_multiplier=1.0, grip_mag=.5):
    """
    Class for allowing a user to intervene on a running agent with an input device.
    ***Note that the speed that button states are updated is limited by how often update and update_buttons are called,
    meaning that if a button press occurs in between calls to update_buttons, it will not be detected.
      - this could be fixed with a separate thread local to this class, but that's low priority

    :param device_type:
    :param env:
    :param intervene_on_touch:   Initial intervention is on a non-zero action, but intervention still
                                 canceled with button
    :param real_time_multiplier: Once an intervention has started, the playback speed that is enforced
                                 in the environment, compared to real time.
    :param grip_mag:             Magnitude of binary grip actions, so will be [-grip_mag, +grip_mag]. Should
                                 be set to be close to the mean magnitudes of the movement actions.
    """
    self.device_type = device_type
    self.real_time_multiplier=real_time_multiplier
    self.intervene_on_touch = intervene_on_touch
    self.grip_mag = grip_mag
    self._env = env
    self.env_id = str(env)
    self.ros_env = True if 'ThingRos' in str(type(env)) else False
    if device_type == 'gamepad':
      from manipulator_learning.learning.imitation.devices.gamepad_control import GamepadSteer
      self.device = GamepadSteer()
    elif device_type == 'keyboard':
      from manipulator_learning.learning.imitation.devices.keyboard_control import KeyboardSteer
      self.device = KeyboardSteer()
    elif device_type == 'vr':
      from manipulator_learning.learning.imitation.devices.vr_control import VRSteer
      if 'ThingRos' in self.env_id:
        if 'Door' in self.env_id:
          up_axis = (0, 0, -1)
        elif 'Drawer' in self.env_id:
          up_axis = (1, 0, 0)
        else:
          up_axis = (0, 0, 1)
        self.device = VRSteer(des_forward_axis=(0, 1, 0), des_up_axis=up_axis)
        self.device.vel_pid.Kp = 0.7
      else:
        self.device = VRSteer(des_forward_axis=(0, 0, 1), des_up_axis=(0, -1, 0))

    self.buttons = dict()
    for b in Intervenor.BUTTONS:
      self.buttons[b] = Button(hold_time_length=2)

    self.want_intervention = False
    self.bad_failure_prediction = False
    self.good_failure_prediction = False
    self.too_late = False
    self._last_enforce_call = time.time()
    self.fast_button_update = False

  def button_update_worker(self, w_shared, b_shared, g_shared, t_shared):
    while self.fast_button_update:
      self._raw_update()
      time.sleep(.01)
      w_shared.value = self.want_intervention
      b_shared.value = self.bad_failure_prediction
      g_shared.value = self.good_failure_prediction
      t_shared.value = self.too_late

  def start_fast_button_updater(self):
    """ Seperate thread for button updates so that we don't miss presses b/c we're reading at 10Hz """
    self.fast_button_update = True
    self.want_shared = Value('i', 0)
    self.bad_shared = Value('i', 0)
    self.good_shared = Value('i', 0)
    self.too_shared = Value('i', 0)
    self.button_worker = Process(target=self.button_update_worker, args=(self.want_shared, self.bad_shared,
                                                                         self.good_shared, self.too_shared))
    self.button_worker.start()
    print('fast button updater started')

  def stop_fast_button_updater(self):
    self.fast_button_update = False
    self.button_worker.join()
    print('fast button updater finished.')

  def _raw_update(self):
    self.device.process_events()
    self.update_buttons()

  def update(self):
    self.device.process_events()
    self.update_buttons()

  def update_buttons(self):
    if self.device_type == 'vr':
      button_edges_dict = self.device.get_latest_button_state()

    cur_time = time.time()
    for but, actual in zip(Intervenor.BUTTONS, Intervenor.BUTTONS_TABLE[self.device_type]):
      if self.device_type == 'vr':
        re = button_edges_dict[actual]['re']
        fe = button_edges_dict[actual]['fe']
        rhe = button_edges_dict[actual]['rhe']
        fhe = button_edges_dict[actual]['fhe']
      else:
        re, fe, rhe, fhe = self.buttons[but].get_update(self.device.btn_state[actual], cur_time)

      if but == 'start_stop_intervention':
        if fe:
          if self.device_type == 'vr':
            self.device.trigger_haptic()
          if not self.want_intervention:
            self.want_intervention = True
          else:
            self.want_intervention = False

      if but == 'bad_failure_prediction':
        if fe:
          if self.device_type == 'vr':
            self.device.trigger_haptic()
          self.bad_failure_prediction = True

      if but == 'good_failure_prediction':
        if fe:
          if self.device_type == 'vr':
            self.device.trigger_haptic()
          self.good_failure_prediction = True

      if but == 'too_late':
        if fe:
          if self.device_type == 'vr':
            self.device.trigger_haptic()
          self.too_late = True

  def _get_action_internal(self, ee_pose=None, base_pose=None):
    if self.device_type == 'vr':
      if base_pose is not None:
        trans_vel, rot_vel, grip = self.device.move_robot(ee_pose[:3], ee_pose[3:],
                                                          base_pose[:3], base_pose[3:], output_vel=True)
      else:
        trans_vel, rot_vel, grip = self.device.move_robot(ee_pose[:3], ee_pose[3:], output_vel=True)
    elif self.device_type == 'keyboard':
      trans_vel, rot_vel, grip = self.device.move_robot()

    if 'Thing' in self.env_id:
      if 'XYState' in self.env_id:
        if self.device_type == 'gamepad':
          action = .3 * np.array([-self.device.normalized_btn_state['LY'],
                                  -self.device.normalized_btn_state['LX']])
          grip = self.device.btn_state['RT']
      elif 'XYZState' in self.env_id:
        action = trans_vel
      elif '6Dof' in self.env_id or self._env.action_space.shape[0] >= 6:
        action = np.concatenate([trans_vel, rot_vel])

    if self._env.action_space.shape[0] >= 6:
      if grip:
        grip = np.array([self.grip_mag])
      else:
        grip = np.array([-self.grip_mag])
      action = np.concatenate([action, grip])

    return action

  def get_action(self, policy_action=None, ee_pose=None, base_pose=None, vr_p_mult=10.0, grip_pos=None):
    # user calls in episode rollout after policy has selected action, before action applied to env
    # returns False if no action from user
    self.update()
    if not self.want_intervention:
      if self.device_type == 'vr':
        self.device.reset_ref_poses()
      return False
    if self.device_type == 'vr':
      if base_pose is not None:
        trans_vel, rot_vel, grip = self.device.move_robot(ee_pose[:3], ee_pose[3:],
                                                          base_pose[:3], base_pose[3:],
                                                          output_vel_p=vr_p_mult, output_vel=True)
      else:
        if 'ThingRos' in self.env_id:
          trans_vel, rot_vel, grip = self.device.move_robot(ee_pose[:3], ee_pose[3:], output_vel_p=1.0, output_vel=True,
                                                            current_gripper_pos=grip_pos)
        else:
          trans_vel, rot_vel, grip = self.device.move_robot(ee_pose[:3], ee_pose[3:], output_vel=True)
    elif self.device_type == 'keyboard':
      trans_vel, rot_vel, grip = self.device.move_robot()
    if 'Thing' in self.env_id:
      if 'XYState' in self.env_id:
        if self.device_type == 'gamepad':
          action = .3 * np.array([-self.device.normalized_btn_state['LY'],
                                  -self.device.normalized_btn_state['LX']])
          grip = self.device.btn_state['RT']
      elif 'XYZState' in self.env_id:
        action = trans_vel
      elif '6Dof' in self.env_id or self._env.action_space.shape[0] >= 6:
        action = np.concatenate([trans_vel, rot_vel])

    # if self._env.action_space.shape[0] >= 6:
    if self._env.grip_in_action:
      if grip:
        grip = np.array([self.grip_mag])
      else:
        grip = np.array([-self.grip_mag])
      action = np.concatenate([action, grip])

    if self.real_time_multiplier is not None:
      self.enforce_realtime_mult()

    if self.intervene_on_touch:
      if np.any(action):
        self.want_intervention = True

    return action

  def wait_for_action(self, ee_pose=None):
    print('waiting for expert action..')
    action = np.array([0])
    if self.device_type == 'vr':
      self.device.reset_ref_poses()

    while np.allclose(action, 0):
      self.update()
      if 'Thing' in self.env_id and 'XYState' in self.env_id:
        if self.device_type == 'gamepad':
          action = .3 * np.array([-self.device.normalized_btn_state['LY'],
                                  -self.device.normalized_btn_state['LX']])
      elif self.device_type == 'vr':
        # take without the gripper so that current grip state isn't overwritten
        action = self._get_action_internal(ee_pose)[:-1]

      elif self.device_type == 'keyboard':
        action = self._get_action_internal()[:-1]
      time.sleep(.01)
    print('expert action received.')

    # need this one to get the grip state again
    if self.device_type == 'vr':
      action = self._get_action_internal(ee_pose)
    elif self.device_type == 'keyboard':
      action = self._get_action_internal()

    return action

  def wait_for_keep_traj(self):
    print('Keep traj? A/down=keep, X/left=discard')
    self.good_failure_prediction = False
    self.bad_failure_prediction = False
    while True:
      self.update()
      time.sleep(.01)
      if self.good_failure_prediction:
        return True
      if self.bad_failure_prediction:
        return False

  def will_policy_fail(self, policy_action=None, get_user_correction=True):
    # user calls when failure is detected. Pauses episode execution until intervenor either adds correction or
    # indicates incorrect failure prediction
    user_query_start = time.time()
    while not (
        (get_user_correction and (self.want_intervention or self.bad_failure_prediction)) or
        (not get_user_correction and (self.good_failure_prediction or self.bad_failure_prediction
                                     or self.too_late))
    ):
      if get_user_correction:
        action = self.get_action(policy_action)  # also updates buttons for bad_failure_prediction
      else:
        self.update()
      time.sleep(.01)
      if (time.time() - user_query_start) % 5 < 0.0105:
        print('Waited %2.3f seconds for user input...' % (time.time() - user_query_start))
      if time.time() > user_query_start + 30:
        print('timed out waiting for user input, assuming no failure')
        self.bad_failure_prediction = True

    if self.want_intervention and get_user_correction:
      return action
    elif self.bad_failure_prediction:
      print('User says incorrect failure prediction')
      self.bad_failure_prediction = False
      return False
    elif self.too_late:
      print('User says good failure prediction, but too late to fix')
      self.too_late = False
      return 2
    elif not get_user_correction and self.good_failure_prediction:
      print('User agreees with failure prediction')
      self.good_failure_prediction = False
      self.want_intervention = True
      return True

  def get_failure_prediction(self):
    self.update()

    # if self.real_time_multiplier is not None:
    #   self.enforce_realtime_mult()

    if self.want_intervention:
      # self.want_intervention = False  # todo is this necessary?
      return True
    else:
      if self.device_type == 'vr':
        self.device.reset_ref_poses()
      return False

  def enforce_realtime_mult(self):
    step_time = time.time() - self._last_enforce_call
    leftover = max(1 / self.real_time_multiplier * self._env.real_t_per_ts - step_time, 0)  # if below 0, already too slow
    time.sleep(leftover)
    self._last_enforce_call = time.time()

  def reset_env_with_teleop(self, env):
    """ For resetting envs with teleop when the env is instantiated with that ability. """
    self.reset()
    if self.device_type == 'vr':
      self.device.reset_ref_poses()
    self.want_intervention = True
    reset = False

    while not reset:
      reset = self.bad_failure_prediction
      if env._control_type == 'delta_tool':
        if self.device_type == 'vr':
          cur_pos = env.get_cur_base_tool_pose()
          cur_grip_pos = env.latest_grip_bool if env.grip_in_action else None
          self.force_feedback(env)
          act = self.get_action(ee_pose=cur_pos, vr_p_mult=1.0, grip_pos=cur_grip_pos)
        else:
          act = self.get_action()
        env.step(act, reset_teleop_step=True)
        env.ep_timesteps = 0
    env.set_reset_teleop_complete()
    if self.device_type == 'vr':
      self.device.reset_ref_poses()
    self.reset()
    return env.reset()

  def get_suc_fail_fb(self):
    """ For allowing user to indicate whether an episode was successful or not after done is set by env. """
    self.reset()
    success_fb_suc, success_fb_fail = False, False
    while not (success_fb_suc or success_fb_fail):
      self.update()
      # up for success: too_late, down for fail: good_failure_prediction
      success_fb_suc = self.too_late
      success_fb_fail = self.good_failure_prediction
      time.sleep(.01)
    return success_fb_suc

  def reset(self):
    # user calls at end of episode
    self.want_intervention = False
    self.good_failure_prediction = False
    self.bad_failure_prediction = False
    self.too_late = False

  def force_feedback(self, env):
    force_feedback(self.device, env)
