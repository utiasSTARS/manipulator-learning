# Contains classes and methods for allowing control with a gamepad.

import inputs  # the forked version with non-blocking-gamepad branch
import time
import numpy as np
import transforms3d as tf3d

x360_event_dict = {
    'ABS_Y': 'LY',
    'ABS_X': 'LX',
    'ABS_RX': 'RX',
    'ABS_RY': 'RY',
    'BTN_NORTH': 'X',  # reversed with west for some reason
    'BTN_WEST': 'Y',
    'BTN_SOUTH': 'A',
    'BTN_EAST': 'B',
    'ABS_HAT0X': 'DX',
    'ABS_HAT0Y': 'DY',
    'BTN_SELECT': 'SEL',
    'BTN_START': 'STA',
    'BTN_TR': 'RB',
    'BTN_TL': 'LB',
    'ABS_Z': 'LT',
    'ABS_RZ': 'RT'
}


class GamepadSteer(object):
    """Class for steering the robot with an xbox 360 controller."""
    STICKS = ['LX', 'RX', 'LY', 'RY']
    TRIGS = ['LT', 'RT']

    def __init__(self, gamepad=None, abbrevs=x360_event_dict, trans_rotation=None):
        self.btn_state = {}
        self.normalized_btn_state = {}
        self.old_btn_state = {}
        self.abbrevs = abbrevs
        for key, value in self.abbrevs.items():
            self.btn_state[value] = 0
            self.old_btn_state[value] = 0
        for s in GamepadSteer.STICKS:
            self.normalized_btn_state[s] = 0
        for t in GamepadSteer.TRIGS:
            self.normalized_btn_state[t] = 0
        self._other = 0
        self.gamepad = gamepad
        if not gamepad:
            self._get_gamepad()
        self.trans_rotation = np.eye(4)
        if trans_rotation is not None:
            self.trans_rotation[:3, :3] = tf3d.euler.euler2mat(*trans_rotation, axes='sxyz')

        self.gripper_toggle = False
        # various toggles for use in color blocks environment
        self.enter_toggle = False
        self.space_toggle = False
        self.d_pressed = False
        self.s_toggle = False
        self.right_start_time = None

    def _get_gamepad(self):
        """Get a gamepad object."""
        try:
            self.gamepad = inputs.devices.gamepads[0]
        except IndexError:
            raise inputs.UnpluggedError("No gamepad found.")

    def process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return
        if event.code in self.abbrevs:
            key = self.abbrevs[event.code]
            self.btn_state[key] = event.state

    def process_events(self):
        """Process available events."""
        self._set_old_button_states()
        try:
            events = self.gamepad.read()
        except EOFError:
            events = []
        for event in events:
            self.process_event(event)

        self.normalize_state()
        self._handle_button_toggles()

    def _set_old_button_states(self):
        for key in self.btn_state.keys():
            self.old_btn_state[key] = self.btn_state[key]

    def normalize_state(self):
        """Make sticks be between -1 and 1, and trigs be between 0 and 1."""
        min_stick_abs = 7000
        for s in GamepadSteer.STICKS:
            if abs(self.btn_state[s]) < min_stick_abs:
                self.normalized_btn_state[s] = 0
            else:
                self.normalized_btn_state[s] = self.btn_state[s] / 32768
        for t in GamepadSteer.TRIGS:
            self.normalized_btn_state[t] = self.btn_state[t] / 255

    def move_robot(self, max_trans_vel, max_rot_vel, control_method):
        """
        Move a robot's end effector using an XBOX 360 controller.

        :param max_trans_vel: Max translational velocity in any particular dimension, not total, in m/s.
        :param max_rot_vel:  Max rotational velocity in any particular dimension, rad/s.
        :param control_method: Control method being used for a Manipulator object, 'p' or 'v' for now.
        :return:
        """
        if control_method == 'p':
            raise NotImplementedError('Control with p not yet implemented for gamepad.')

        g_command = self.gripper_toggle

        forward_back = self.normalized_btn_state['LY']
        left_right = self.normalized_btn_state['LX']
        up_down = self.normalized_btn_state['RT'] - self.normalized_btn_state['LT']

        t_perm_change = np.dot(self.trans_rotation, np.array([forward_back, left_right, up_down, 1]))
        forward_back = t_perm_change[0]
        left_right = t_perm_change[1]
        up_down = t_perm_change[2]

        if control_method == 'v':
            trans_vel = np.array([-up_down, -left_right, -forward_back]) * max_trans_vel

            if self.btn_state['LB']:  # allows modifying z rotation
                rot_modify = np.array([0, self.normalized_btn_state['RY'],
                                       self.normalized_btn_state['RX']]) * max_rot_vel
            else:
                rot_modify = np.array([self.normalized_btn_state['RX'],
                                       self.normalized_btn_state['RY'], 0]) * max_rot_vel
            rot_vel = rot_modify

            return trans_vel, rot_vel, g_command

    def _handle_button_toggles(self):
        """ Handle button toggles the same was was done for KeyMouseSteer. Not a long term sol'n. """
        cur_time = time.time()
        hold_time = 2

        if self.btn_state['STA'] and not self.old_btn_state['STA']:
            if not self.enter_toggle:
                self.enter_toggle = True
            else:
                self.enter_toggle = False
        if self.btn_state['SEL'] and not self.old_btn_state['SEL']:
            self.space_toggle = True
        if self.btn_state['X'] and not self.old_btn_state['X']:
            self.d_pressed = True
        if self.btn_state['RB'] and not self.old_btn_state['RB']:
            if not self.gripper_toggle:
                self.gripper_toggle = True
            else:
                self.gripper_toggle = False
        if self.btn_state['DX'] == -1 and self.old_btn_state['DX'] != -1:
            if not self.s_toggle:
                self.s_toggle = True
            else:
                self.s_toggle = False
        if self.btn_state['STA'] and not self.old_btn_state['STA']:
            self.right_start_time = cur_time
        if self.btn_state['STA'] and (cur_time - self.right_start_time > 2):
            self.enter_hold = True
        else:
            self.enter_hold = False


if __name__ == '__main__':
    gs = GamepadSteer()

    while gs.btn_state['STA'] == 0:
        gs.process_events()
        # print(gs.normalized_btn_state['LX'], gs.btn_state['A'])
        print(gs.move_robot(.2, 1.5, 'v'))
        time.sleep(.01)