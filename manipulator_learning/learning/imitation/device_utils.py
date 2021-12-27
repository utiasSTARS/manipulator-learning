import time
import numpy as np


def force_feedback(dev_obj, env):
    """ give user some degree of force feedback in control device if it is available --
    currently only designed to work with Thing Ros envs and vr"""
    norm = np.linalg.norm
    high_force = 10
    high_torque = 1
    ft = env.latest_ft_raw
    f_norm = norm(ft[:3])
    t_norm = norm(ft[3:])

    if f_norm < high_force: f_norm = 0
    if t_norm < high_torque: t_norm = 0
    if f_norm == 0 and t_norm == 0:
        dev_obj.force_feedback_dur = 0
        return
    f_scaled = f_norm / high_force
    t_scaled = t_norm / high_torque
    f_nonlin = min(np.exp(f_scaled - 4), 1.0)
    t_nonlin = min(np.exp(t_scaled - 4), 1.0)
    # f_feedback = min(max_force, f_norm) / max_force
    # t_feedback = min(max_torque, t_norm) / max_torque
    # dom_feedback = max(f_feedback, t_feedback)
    dom_feedback = max(f_nonlin, t_nonlin)
    feedback_dur = int(dom_feedback * 3999)

    dev_obj.force_feedback_dur = feedback_dur


# user controlled booleans for recording state
class Button():
    def __init__(self, hold_time_length):
        self.state = False
        self.last_state = False
        self.hold_state = False
        self.hold_time_start = time.time()
        self.last_hold_state = False
        self.hold_time_length = hold_time_length
        self.stored_state = dict(re=False, fe=False, rhe=False, fhe=False)

    def get_update(self, raw_state, cur_time):
        """
        Update the button state and hold state and return the rising and falling edges.

        :param raw_state: The raw state of the button from its source.
        :return: Whether there is a rising edge, falling edge, rising edge of being held, and falling edge of
                 being held.
        """
        self.last_hold_state = self.hold_state
        self.last_state = self.state
        self.state = raw_state
        if self.state and not self.last_state:
            self.hold_time_start = cur_time
            rising_edge = True
        else:
            rising_edge = False
        if not self.state and self.last_state:
            falling_edge = True
        else:
            falling_edge = False

        # hold state stuff
        if cur_time - self.hold_time_start > self.hold_time_length and self.state:
            self.hold_state = True
        else:
            self.hold_state = False
        if self.hold_state and not self.last_hold_state:
            hold_rising_edge = True
        else:
            hold_rising_edge = False
        if not self.hold_state and self.last_hold_state:
            hold_falling_edge = True
        else:
            hold_falling_edge = False

        return rising_edge, falling_edge, hold_rising_edge, hold_falling_edge

    def get_and_store_update(self, raw_state, cur_time):
        """ Only allows changing False to True, stores between calls to reset_state. """
        re, fe, hre, hfe = self.get_update(raw_state, cur_time)
        self.stored_state['re'] = re or self.stored_state['re']
        self.stored_state['fe'] = fe or self.stored_state['fe']
        self.stored_state['rhe'] = hre or self.stored_state['rhe']
        self.stored_state['fhe'] = hfe or self.stored_state['fhe']

    def reset_state(self):
        for k in self.stored_state:
            self.stored_state[k] = False


class CollectDevice:
    BUTTONS = ['start_save_cancel', 'delete', 'reset_save', 'success_fb_fail']
    BUTTONS_TABLE = dict(
        keyboard=['enter', 'backspace', 'r_shift'],
        gamepad=['B', 'Y', 'X'],
        vr=['trackpad_right_click', 'trackpad_up_click', 'trackpad_left_click', 'trackpad_down_click']
    )

    def __init__(self, device, valid_t_dof=(1, 1, 1), valid_r_dof=(1, 1, 1), output_grip=True,
                 action_multiplier=1.0, des_forward_axis=(0, 0, 1), des_up_axis=(0, -1, 0)):
        self.dev_type = device
        self.dev = self.initialize_device(device, des_forward_axis, des_up_axis)
        self.recording = False
        self.buttons = dict()
        for b in CollectDevice.BUTTONS:
            self.buttons[b] = Button(hold_time_length=2)
        self.valid_t_dof = np.array(valid_t_dof)
        self.valid_r_dof = np.array(valid_r_dof)
        self.output_grip = output_grip
        self.action_multiplier = action_multiplier

    def initialize_device(self, device, des_forward_axis, des_up_axis):
        if device == 'keyboard':
            from manipulator_learning.learning.imitation.devices.keyboard_control import KeyboardSteer
            dev = KeyboardSteer()
        elif device == 'gamepad':
            from manipulator_learning.learning.imitation.devices.gamepad_control import GamepadSteer
            dev = GamepadSteer()
        elif device == 'vr':
            from manipulator_learning.learning.imitation.devices.vr_control import VRSteer
            dev = VRSteer(des_forward_axis=des_forward_axis, des_up_axis=des_up_axis)
        return dev

    def update_and_get_state(self):
        cur_time = time.time()
        cancel, save, start, reset, delete, success_fb_suc, success_fb_fail = False, False, False, False, False, False, False
        self.dev.process_events()
        if self.dev_type == 'vr':
            button_edges_dict = self.dev.get_latest_button_state()

        for but, actual in zip(CollectDevice.BUTTONS, CollectDevice.BUTTONS_TABLE[self.dev_type]):
            if self.dev_type == 'vr':
                re = button_edges_dict[actual]['re']
                fe = button_edges_dict[actual]['fe']
                rhe = button_edges_dict[actual]['rhe']
                fhe = button_edges_dict[actual]['fhe']
            else:
                re, fe, rhe, fhe = self.buttons[but].get_update(self.dev.btn_state[actual], cur_time)

            if but == 'start_save_cancel':
                if fhe:
                    if self.recording:
                        if self.dev_type == 'vr':
                            self.dev.trigger_haptic()
                        cancel = True
                        self.recording = False
                elif fe:
                    if not self.recording:
                        if self.dev_type == 'vr':
                            self.dev.trigger_haptic()
                        start = True
                        self.recording = True
                        print("-----------------")
                        print("RECORDING START!!")
            elif but == 'reset_save':
                if fe:
                    if self.dev_type == 'vr':
                        self.dev.trigger_haptic()
                    reset = True
                    if self.recording:
                        save = True
                    self.recording = False
            elif but == 'delete':
                if fhe:
                    if self.dev_type == 'vr':
                        self.dev.trigger_haptic()
                    delete = True
                elif fe:
                    if self.dev_type == 'vr':
                        self.dev.trigger_haptic()
                    success_fb_suc = True

            elif but == 'success_fb_fail':
                if fe:
                    if self.dev_type == 'vr':
                        self.dev.trigger_haptic()
                    success_fb_fail = True

        return cancel, save, start, reset, delete, success_fb_suc, success_fb_fail

    def get_ee_vel_action(self, ee_pose=None, base_pose=None, vr_p_mult=10.0, grip_mag=.05):
        """ poses used for vr actions, given as 7-dim arrays with 3 for pos and 4 for xyzw quat.

            grip_mag should be set to be approximately the same as the mean action from the other
            dimenstions, since the grip action will be the same regardless. """
        if self.dev_type == 'gamepad':
            trans_vel = self.action_multiplier * np.array([-self.dev.normalized_btn_state['LY'],
                         -self.dev.normalized_btn_state['LX'],
                         self.dev.normalized_btn_state['LT'] - self.dev.normalized_btn_state['RT']])
            rot_vel = self.action_multiplier * np.array([self.dev.normalized_btn_state['RX'],
                                -self.dev.normalized_btn_state['RY'],
                                self.dev.btn_state['LB'] - self.dev.btn_state['RB']])
            # grip = self.dev.btn_state['A']
            grip = self.dev.btn_state['RT']
        elif self.dev_type == 'keyboard':
            trans_vel = self.action_multiplier * np.array([self.dev.btn_state['d'] - self.dev.btn_state['a'],
                                  self.dev.btn_state['w'] - self.dev.btn_state['s'],
                                  self.dev.btn_state['e'] - self.dev.btn_state['q']])
            rot_vel = self.action_multiplier * np.array([self.dev.btn_state['u'] - self.dev.btn_state['j'],
                                  self.dev.btn_state['i'] - self.dev.btn_state['k'],
                                  self.dev.btn_state['o'] - self.dev.btn_state['l']])
            grip = self.dev.btn_state['space']
        elif self.dev_type == 'vr':
            # since vr needs to output a position, output a position, and use a simple p(id) controller
            # to output a velocity to match the position
            if base_pose is not None:
                trans_vel, rot_vel, grip = self.dev.move_robot(ee_pose[:3], ee_pose[3:],
                                             base_pose[:3], base_pose[3:], output_vel=True, output_vel_p=vr_p_mult)
            else:
                trans_vel, rot_vel, grip = self.dev.move_robot(ee_pose[:3], ee_pose[3:], output_vel=True,
                                                               output_vel_p=vr_p_mult)

        trans_vel = trans_vel[self.valid_t_dof.nonzero()[0]]
        rot_vel = rot_vel[self.valid_r_dof.nonzero()[0]]
        return_act = np.concatenate((trans_vel, rot_vel))
        if self.output_grip:
            if grip:
                grip = grip_mag
            else:
                grip = -grip_mag
            return_act = np.concatenate((return_act, np.array((grip,))))
            # return_act = (return_act, int(grip))
        return return_act

    def force_feedback(self, env):
        """ give user some degree of force feedback in control device if it is available --
        currently only designed to work with Thing Ros envs and vr"""
        force_feedback(self.dev, env)

    def get_ee_pos_action(self, cur_pose, cur_base_pose):
        raise NotImplementedError()
