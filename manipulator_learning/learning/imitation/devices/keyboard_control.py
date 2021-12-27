# Contains classes and methods for allowing control with a gamepad.

import inputs  # the forked version with non-blocking-gamepad branch
import time
import numpy as np
import transforms3d as tf3d

keyboard_event_dict = {
    'KEY_W': 'w',
    'KEY_A': 'a',
    'KEY_S': 's',
    'KEY_D': 'd',
    'KEY_E': 'e',
    'KEY_Q': 'q',
    'KEY_U': 'u',
    'KEY_J': 'j',
    'KEY_I': 'i',
    'KEY_K': 'k',
    'KEY_O': 'o',
    'KEY_L': 'l',
    'KEY_B': 'b',
    'KEY_G': 'g',
    'KEY_F': 'f',
    'KEY_T': 't',
    'KEY_SPACE': 'space',
    'KEY_R': 'r',
    'KEY_ENTER': 'enter',
    'KEY_BACKSPACE': 'backspace',
    'KEY_RIGHTSHIFT': 'r_shift',
}


class KeyboardSteer(object):
    """Class for steering the robot with a keyboard."""
    STICKS = ['LX', 'RX', 'LY', 'RY']
    TRIGS = ['LT', 'RT']

    def __init__(self, keyboard=None, abbrevs=keyboard_event_dict, trans_rotation=None, action_multiplier=.3):
        self.btn_state = {}
        self.normalized_btn_state = {}
        self.old_btn_state = {}
        self.action_multiplier = action_multiplier
        self.abbrevs = abbrevs
        for key, value in self.abbrevs.items():
            self.btn_state[value] = 0
            self.old_btn_state[value] = 0
        self._other = 0
        self.kb = keyboard
        if not keyboard:
            self._get_keyboard()
        if trans_rotation is None:
            self.trans_rotation = np.eye(4)
        else:
            self.trans_rotation = tf3d.euler.euler2mat(*trans_rotation, axes='sxyz')

        self.gripper_toggle = False
        # various toggles for use in color blocks environment
        self.enter_toggle = False
        self.space_toggle = False
        self.d_pressed = False
        self.s_toggle = False
        self.right_start_time = None

    def _get_keyboard(self):
        """Get a gamepad object."""
        try:
            self.kb = inputs.devices.keyboards[0]
        except IndexError:
            raise inputs.UnpluggedError("No keyboard found.")

    def _process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return
        if event.code in self.abbrevs:
            key = self.abbrevs[event.code]
            self.btn_state[key] = bool(event.state)  # so no diff between held (2) and pushed (1)

    def process_events(self):
        """Process available events. Call this one."""
        self._set_old_button_states()
        try:
            events = self.kb.read()
        except EOFError:
            events = []
        for event in events:
            self._process_event(event)

        # self._handle_button_toggles()

    def _set_old_button_states(self):
        for key in self.btn_state.keys():
            self.old_btn_state[key] = self.btn_state[key]

    def normalize_state(self):
        """Make sticks be between -1 and 1, and trigs be between 0 and 1."""
        min_stick_abs = 7000
        for s in KeyboardSteer.STICKS:
            if abs(self.btn_state[s]) < min_stick_abs:
                self.normalized_btn_state[s] = 0
            else:
                self.normalized_btn_state[s] = self.btn_state[s] / 32768
        for t in KeyboardSteer.TRIGS:
            self.normalized_btn_state[t] = self.btn_state[t] / 255

    def move_robot(self):
        trans_vel = self.action_multiplier * np.array([self.btn_state['d'] - self.btn_state['a'],
                                                       self.btn_state['w'] - self.btn_state['s'],
                                                       self.btn_state['e'] - self.btn_state['q']])
        rot_vel = self.action_multiplier * np.array([self.btn_state['u'] - self.btn_state['j'],
                                                     self.btn_state['i'] - self.btn_state['k'],
                                                     self.btn_state['o'] - self.btn_state['l']])
        grip = self.btn_state['space']

        return trans_vel, rot_vel, grip

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
    ks = KeyboardSteer()

    while True:
        ks.process_events()
        # print(gs.normalized_btn_state['LX'], gs.btn_state['A'])
        print(ks.btn_state)
        print(bool(ks.btn_state['w']))
        time.sleep(.1)