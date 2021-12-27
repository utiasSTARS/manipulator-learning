import openvr
import numpy as np
import time
import copy
from threading import Thread, Lock
from queue import Queue
import queue
import signal
import sys

from simple_pid import PID
import liegroups as lie
import transforms3d as tf3d

from manipulator_learning.sim.utils.general import convert_quat_tf_to_pb, convert_quat_pb_to_tf, TransformMat, q_convert
from manipulator_learning.learning.imitation.device_utils import Button


def openvr_pose_to_numpy(TrackedDevicePose_t):
    """ Convert a TrackedDevicePose_t to a 4x4 numpy array for its pose, a (3,) array for velocity,
    and another (3,) array for angular velocity"""
    pose_mat34_ovr = TrackedDevicePose_t.mDeviceToAbsoluteTracking
    vel_ovr = TrackedDevicePose_t.vVelocity
    avel_ovr = TrackedDevicePose_t.vAngularVelocity
    pose_mat = np.ndarray([4, 4])
    pose_mat[3, :4] = [0, 0, 0, 1]
    vel_vec = np.ndarray([3])
    avel_vec = np.ndarray([3])
    for i in range(3):
        pose_mat[i] = pose_mat34_ovr[i]
        vel_vec[i] = vel_ovr[i]
        avel_vec[i] = avel_ovr[i]
    return pose_mat, vel_vec, avel_vec


class VRTrackedDevice:
    def __init__(self, device_type, vr_obj, index, init_pose):
        self.device_type = device_type
        self.index = index
        self.vr = vr_obj
        self.ovr_pose = None  # set by update_pose
        self.np_pose = None  # set by update_pose
        self.np_vel = None  # set by update_pose
        self.np_avel = None  # set by update_pose
        self.update_pose(init_pose)

    def update_pose(self, TrackedDevicePose_t):
        """ Update openvr pose (TrackedDevicePose_t) as well as the numpy poses. This should be called
         instead of directly setting ovr_pose, np_pose, np_vel, or np_avel. """
        self.ovr_pose = TrackedDevicePose_t
        self.np_pose, self.np_vel, self.np_avel = openvr_pose_to_numpy(TrackedDevicePose_t)


class ViveController(VRTrackedDevice):
    def __init__(self, vr_obj, index, init_pose, button_hold_time=2.0):
        super().__init__(device_type=openvr.TrackedDeviceClass_Controller,
                         vr_obj=vr_obj, index=index, init_pose=init_pose)
        self.ovr_state = None  # set by update_buttons
        self.trackpad = [0, 0]
        self.trigger = 0
        self.button_state = dict(
            squeeze=False,
            trigger_bool=False,
            menu=False,
            trackpad_button=False,
            trackpad_up_click=False,  # these four are for a "D-pad" using the trackpad
            trackpad_down_click=False,
            trackpad_left_click=False,
            trackpad_right_click=False,
        )
        self.old_button_state = copy.deepcopy(self.button_state)
        self.button_edges_objs = dict()
        for k in self.button_state:
            self.button_edges_objs[k] = Button(hold_time_length=button_hold_time)
        self.thread_lock = Lock()

    def update_buttons(self, VRControllerState_t):
        """ Update openvr state as well as more readable states. Should only be used for devices that
        have buttons. """
        cur_time = time.time()
        self.ovr_state = VRControllerState_t
        self.trackpad[0] = VRControllerState_t.rAxis[0].x
        self.trackpad[1] = VRControllerState_t.rAxis[0].y
        self.trigger = VRControllerState_t.rAxis[1].x
        self.old_button_state = copy.deepcopy(self.button_state)
        button_str = format(VRControllerState_t.ulButtonPressed, '64b')
        self.button_state['menu'] = button_str[62] == '1'
        self.button_state['squeeze'] = button_str[61] == '1'
        self.button_state['trackpad_button'] = button_str[31] == '1'
        self.button_state['trigger_bool'] = button_str[30] == '1'
        min_trackpad_dpad = .5
        self.button_state['trackpad_right_click'] = self.button_state['trackpad_button'] \
                                                 and self.trackpad[0] > min_trackpad_dpad
        self.button_state['trackpad_left_click'] = self.button_state['trackpad_button'] \
                                                 and self.trackpad[0] < -min_trackpad_dpad
        self.button_state['trackpad_up_click'] = self.button_state['trackpad_button'] \
                                                 and self.trackpad[1] > min_trackpad_dpad
        self.button_state['trackpad_down_click'] = self.button_state['trackpad_button'] \
                                                 and self.trackpad[1] < -min_trackpad_dpad
        self.update_edges(cur_time)

    def update_edges(self, cur_time):
        for k in self.button_state:
            self.button_edges_objs[k].get_and_store_update(self.button_state[k], cur_time)

    def get_latest_button_data(self):
        ret_dict = dict()
        self.thread_lock.acquire()
        for k in self.button_state:
            if k == 'menu':
                continue  # only want in get_latest_menu_data
            ret_dict[k] = copy.deepcopy(self.button_edges_objs[k].stored_state)
            if any(ret_dict[k][edge] for edge in ret_dict[k].keys()):
                self.button_edges_objs[k].reset_state()
        self.thread_lock.release()
        return ret_dict

    def get_latest_menu_data(self):
        """ Called at a higher frequency by VRSteer object itself. """
        stored_menu_state = copy.deepcopy(self.button_edges_objs['menu'].stored_state)
        if any(stored_menu_state[edge] for edge in stored_menu_state.keys()):
            self.button_edges_objs['menu'].reset_state()
        return stored_menu_state


class VRSteer:
    def __init__(self, des_forward_axis, des_up_axis, headset_viewing=False, buttons_thread=True, vel_pid_p=5.0):
        self.vr = openvr.init(openvr.VRApplication_Other)

        # initialization borrowed from triad_openvr
        # Initializing object to hold indexes for various tracked objects
        self.object_names = {"Tracking Reference": [], "HMD": [], "Controller": [], "Tracker": []}
        self.devices = {}
        poses = self.__get_ovr_poses()
        # Iterate through the pose list to find the active devices and determine their type
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bPoseIsValid:
                device_class = self.vr.getTrackedDeviceClass(i)
                if device_class == openvr.TrackedDeviceClass_Controller:
                    device_name = "controller_" + str(len(self.object_names["Controller"]))
                    self.object_names["Controller"].append(device_name)
                    self.devices[device_name] = ViveController(self.vr, i, poses[i])
                elif device_class == openvr.TrackedDeviceClass_HMD:
                    device_name = "hmd_" + str(len(self.object_names["HMD"]))
                    self.object_names["HMD"].append(device_name)
                    self.devices[device_name] = VRTrackedDevice(device_class, self.vr, i, poses[i])
                elif device_class == openvr.TrackedDeviceClass_GenericTracker:
                    device_name = "tracker_" + str(len(self.object_names["Tracker"]))
                    self.object_names["Tracker"].append(device_name)
                    self.devices[device_name] = VRTrackedDevice(device_class, self.vr, i, poses[i])
                elif device_class == openvr.TrackedDeviceClass_TrackingReference:
                    device_name = "tracking_reference_" + str(len(self.object_names["Tracking Reference"]))
                    self.object_names["Tracking Reference"].append(device_name)
                    self.devices[device_name] = VRTrackedDevice(device_class, self.vr, i, poses[i])

        # various toggles for use in color blocks environment
        self.enter_toggle = False
        self.space_toggle = False
        self.d_pressed = False
        self.s_toggle = False
        self.menu_toggle = False
        self.d_pressed_start_time = None
        self.trackpad_down_click_toggle = False
        self.right_start_time = None
        self.enter_hold = False
        self.policy_play = False

        # needed for pose based control
        self.accept_control_toggle = False
        self._tool_ref_pose = None
        self._vr_ref_pose = None

        # set the desired forward and up axes
        self.pose = TransformMat(mat=np.eye(4))  # pose of the 0th controller
        self.vive_T_fix = np.eye(4)
        self.set_axes(des_forward_axis, des_up_axis)

        # for adding random uniform noise to the control outputs that manifests as a random walk
        self.random_walk_T = TransformMat(mat=np.eye(4))
        self.perturb_T = np.eye(4)
        self.last_control_noise = time.time()

        # for viewing in headset
        if headset_viewing:
            self.headset_pose = np.eye(4)

        # for giving user force feedback on hitting things
        self.force_feedback_dur = 0

        # for making a separate thread that polls the buttons faster and ensures presses aren't missed
        if buttons_thread:
            self.buttons_q = Queue()
            self.buttons_thread = Thread(target=self.buttons_worker, args=(self.buttons_q,))
            self.buttons_thread.daemon = True
            self.buttons_thread.start()
            signal.signal(signal.SIGINT, self.signal_handler)
        else:
            self.buttons_thread = None

        # for working the same as gamepadsteer and keyboardsteer
        self.btn_state = dict()

        self.vel_pid = PID(vel_pid_p, 0.0)

        self.last_grip_command = False

        self._last_des_pose = None

    def buttons_worker(self, q: Queue):
        while(True):
            try:
                q_data = q.get_nowait()
                if q_data == 'shutdown':
                    print('shutting down buttons thread in vr control')
                    break
            except queue.Empty:
                pass
            self.update_buttons()
            if self.force_feedback_dur > 0:
                self.trigger_haptic(self.force_feedback_dur)
            time.sleep(.01)  # 100Hz

    def signal_handler(self, sig, frame):
        self.shutdown_buttons_worker()
        sys.exit()

    def shutdown_buttons_worker(self):
        self.buttons_q.put('shutdown')
        self.buttons_thread.join()
        print('vr buttons thread shut down')

    def update_poses(self):
        """ Update all poses. Should be called once per iteration of a control loop. """
        poses = self.__get_ovr_poses()
        for dev in self.devices.keys():
            self.devices[dev].update_pose(poses[self.devices[dev].index])

        self.pose = TransformMat(mat=self.devices['controller_0'].np_pose)

    def update_buttons(self):
        """ Update all button states. Should be called once per iteration of a control loop if needed. """
        self.devices['controller_0'].thread_lock.acquire()
        for dev in self.devices.keys():
            if self.devices[dev].device_type == openvr.TrackedDeviceClass_Controller:
                index = self.devices[dev].index
                state = self.vr.getControllerState(index)
                assert (bool(state[0]) is True), "Cannot get state of %s. Is it off?" % dev
                self.devices[dev].update_buttons(state[1])

        self.btn_state = self.devices['controller_0'].button_state
        self.devices['controller_0'].thread_lock.release()

    def get_latest_button_state(self, dev='controller_0'):
        return self.devices[dev].get_latest_button_data()

    def process_events(self):
        """ Convenience function to give same functionality as KeyboardSteer and GamepadSteer"""
        self.update_poses()
        if self.buttons_thread is None:
            self.update_buttons()

    def __get_ovr_poses(self):
        """ Get the current openvr poses for all devices. """
        return self.vr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0,
                                                       openvr.k_unMaxTrackedDeviceCount)

    def set_axes(self, forward_axis, up_axis):
        """
        Set the desired forward axis and up axis of the frame to be controlled.

        rotation from the VR controller frame to the desired tool frame. I.e. the vr z axis points inward,
        but if the tool frame z axis points outward,  we need a 180 deg rotation about x to get that.

        If forward_axis is [0, 0, 1], and up_axis is [0, -1, 0], then this is the equivalent of
        tf3d.euler.euler2mat(np.pi, 0, 0, 'sxyz').

        :param forward_axis: The desired "forward movement" axis of the robot tool frame
        :param up_axis: The desired "upward movement" axis of the robot tool frame
        """
        # not 100% sure that this function is implemented correctly, but it seems okay.
        vive_forward_axis = [0, 0, -1]  # found from running various steam experiments
        # vive_forward_axis = [0, -1, -1]  # could use this as forward axis instead
        vive_up_axis = [0, 1, 0]
        # vive_up_axis = [0, 1, -1]

        vive_fa = np.array(vive_forward_axis) / np.linalg.norm(vive_forward_axis)
        vive_ua = np.array(vive_up_axis) / np.linalg.norm(vive_up_axis)
        vive_ra = np.cross(vive_fa, vive_ua)
        des_fa = np.array(forward_axis)
        des_ua = np.array(up_axis)
        des_ra = np.cross(des_fa, des_ua)
        vive_rot_mat = np.outer(des_fa, vive_fa) + np.outer(des_ua, vive_ua) + \
                       np.outer(des_ra, vive_ra)
        vive_T_fix = np.eye(4)
        vive_T_fix[:3, :3] = vive_rot_mat.T  # transpose to get the conversion in the direction we want

        # vive_T_fix = np.eye(4)
        # vive_T_fix[:3, :3] = tf3d.euler.euler2mat(np.pi, 0, 0, 'sxyz')

        self.vive_T_fix = vive_T_fix

    def reset_ref_poses(self):
        """ Allows an outside caller to cleanly reset ref poses. """
        self.devices['controller_0'].thread_lock.acquire()
        self._vr_ref_pose = None
        self._tool_ref_pose = None
        self._last_des_pose = None
        self.menu_toggle = False
        self.devices['controller_0'].thread_lock.release()

    def trigger_haptic(self, duration_micros=3999, axis=0):
        index = self.devices['controller_0'].index
        self.vr.triggerHapticPulse(index, axis, duration_micros)

    def move_robot(self, current_robot_pos,
                   current_robot_rot,
                   current_robot_base_pos=None,
                   current_robot_base_rot=None,
                   random_walk_noise=None,
                   drift_noise=None,
                   match_controller_orientation=True,
                   controller_index=0,
                   output_vel=False,
                   output_vel_p=10.0,
                   current_gripper_pos=None):
        """
         Move a robot using an HTC Vive Controller. Current robot pos and rot are only used for setting
         the initial frame by pressing the menu button.

        :param current_robot_pos: 3 element tuple, the current position of the robot in the robot control ref
            frame (probably either the base frame or the world frame).
        :param current_robot_rot: 4 element tuple, the current orientation (xyzw quat) of the robot in the
            robot control ref frame (probably either the base frame or the world frame).
        :param current_robot_base_pos: 3 element tuple, current base position, used for setting new ee relative
            positions. If None, all ee positions assumed to be in world frame.
        :param current_robot_base_rot: 4 element tuple, current base orietation (xyzw quat), used for setting new
        ee relative positions. If None, all ee positions assumed to be in world frame.
        :param controller_index: Controller index to use (if more than one detected)
        :param output_vel: Whether to output velocity instead of position.
        :return:
        """

        if self.buttons_thread is not None:
            self.devices['controller_0'].thread_lock.acquire()

        menu_state_dict = self.devices['controller_0'].get_latest_menu_data()
        if menu_state_dict['fe']:
            if self.menu_toggle:
                self.menu_toggle = False
            else:
                self.menu_toggle = True

        c_str = 'controller_' + str(controller_index)

        if self._tool_ref_pose is None:  # should only occur on first call
            self._tool_ref_pose = TransformMat(pb_pose=(current_robot_pos, current_robot_rot))
            self._last_des_pose = copy.deepcopy(self._tool_ref_pose)

        # Get tool pose in world frame
        if current_robot_base_pos is not None:
            current_robot_base_pose = TransformMat(pb_pose=(current_robot_base_pos, current_robot_base_rot))
            current_tool_world_pose = TransformMat(mat=np.dot(current_robot_base_pose(), self._tool_ref_pose()))
        else:
            current_tool_world_pose = self._tool_ref_pose

        if self.menu_toggle and not self.accept_control_toggle:
            # this ensures tool ref pose only gets set on the initial press of the menu button
            self.trigger_haptic()
            self.accept_control_toggle = True
            self._tool_ref_pose = TransformMat(pb_pose=(current_robot_pos, current_robot_rot))
            self._vr_ref_pose = TransformMat(mat=np.dot(self.pose(), self.vive_T_fix))
            self._tool_to_vr_ref_pose = TransformMat(mat=np.dot(self._tool_ref_pose(), self._vr_ref_pose()))
        elif not self.menu_toggle and self.accept_control_toggle:
            self.trigger_haptic()
            self.accept_control_toggle = False
            self._tool_ref_pose = TransformMat(pb_pose=(current_robot_pos, current_robot_rot))
            self.random_walk_T = TransformMat(mat=np.eye(4))

        if self.accept_control_toggle:
            vr_current_pose_mat = TransformMat(mat=self.pose())
            vr_fixed_pose = TransformMat(np.dot(vr_current_pose_mat(), self.vive_T_fix))
            if match_controller_orientation:
                # the desired pose should be set to the tool_ref + (vr_current - vr_ref)
                T_vr_ref_to_cur = TransformMat(mat=np.dot(np.linalg.inv(self._vr_ref_pose()), vr_fixed_pose()))
                if current_robot_base_pos is not None:
                    T_world_to_tool = TransformMat(mat=np.dot(current_robot_base_pose(), self._tool_ref_pose()))
                else:
                    T_world_to_tool = self._tool_ref_pose
                T_rob_ref_to_des = TransformMat(mat=np.dot(T_world_to_tool(), T_vr_ref_to_cur()))

                # add in random walk uniform noise if squeeze is being pressed
                if random_walk_noise is not None and self.devices[c_str].button_state['trackpad_down_click']:
                    c_noise_flat = np.array(random_walk_noise).flatten()
                    perturb_xi = np.random.uniform(low=-c_noise_flat, high=c_noise_flat, size=6)
                    perturb_T = lie.SE3.exp(perturb_xi).as_matrix()
                    self.random_walk_T = TransformMat(mat=np.dot(perturb_T, self.random_walk_T.pose_mat))
                    self.last_control_noise = time.time()
                    
                # try with smooth drift added instead of random noise
                if drift_noise is not None and self.devices[c_str].button_state['trackpad_down_click']:
                    if self.trackpad_down_click_toggle:  # only set the drift values once
                        self.trackpad_down_click_toggle = False
                        c_noise_flat = np.array(drift_noise).flatten()
                        perturb_xi = np.random.uniform(low=-c_noise_flat, high=c_noise_flat, size=6)
                        self.perturb_T = lie.SE3.exp(perturb_xi).as_matrix()
                    self.random_walk_T = TransformMat(mat=np.dot(self.perturb_T, self.random_walk_T.pose_mat))
                    self.last_control_noise = time.time()

                # add in noise if it has changed, otherwise keep it the same as the last noise added output
                T_rob_ref_to_des = TransformMat(mat=np.dot(T_rob_ref_to_des.pose_mat, self.random_walk_T.pose_mat))

                pos, rot = T_rob_ref_to_des.to_pb()
            else:
                # get position by subtracting world coordinates of v0 from vi directly
                subtracted_pos = vr_fixed_pose()[:3, 3] - self._vr_ref_pose()[:3, 3]
                pos = current_tool_world_pose()[:3, 3] + subtracted_pos

                # rotations gotten based on C_w_ti = C_w_t0 * C_t0_v0 * C_v0_vi * C_v0_t0
                # equation can be simplified to C_w_vi * C_v0_w * C_w_t0
                C_w_vi = vr_fixed_pose()[:3, :3]
                C_v0_w = self._vr_ref_pose()[:3, :3].T
                C_w_t0 = current_tool_world_pose()[:3, :3]
                C_w_ti = C_w_vi @ C_v0_w @ C_w_t0
                quat = tf3d.quaternions.mat2quat(C_w_ti)
                rot = [quat[1], quat[2], quat[3], quat[0]]  # convert from wxyz to xyzw

                # uncomment to use position only
                # _, rot = TransformMat(mat=np.dot(current_robot_base_pose(), self._tool_ref_pose())).to_pb()
        else:
            # pos, rot = self._tool_ref_pose.to_pb()
            if current_robot_base_pos is not None:
                pos, rot = TransformMat(mat=np.dot(current_robot_base_pose(), self._tool_ref_pose())).to_pb()
            else:
                pos, rot = self._tool_ref_pose.to_pb()

        # safety checking
        last_pos, last_rot = self._last_des_pose.to_pb()
        if np.linalg.norm(np.array(pos) - np.array(last_pos)) > .08:
            print("Des pose greater than 8cm from cur pose, ignoring for safety.. is vr obscured?")
            print("Press menu to reenable")
            self.menu_toggle = False
            self.trigger_haptic()
            pos, rot = self._last_des_pose.to_pb()
        rot_wxyz = [rot[3], rot[0], rot[1], rot[2]]  # convert from xyzw to wxyz
        last_rot_wxyz = [last_rot[3], last_rot[0], last_rot[1], last_rot[2]]  # convert from xyzw to wxyz
        q_diff = tf3d.quaternions.qmult(tf3d.quaternions.qinverse(rot_wxyz), last_rot_wxyz)
        if q_diff[0] < 0:
            q_diff = -q_diff
        ax, ang = tf3d.quaternions.quat2axangle(q_diff)
        if abs(ang) > .7:
            print("Des orientation greater than .7 rad from cur, ignoring for safety.. is vr obscured?")
            print("Press menu to reenable")
            self.trigger_haptic()
            self.menu_toggle = False
            pos, rot = self._last_des_pose.to_pb()

        self._last_des_pose = TransformMat(pb_pose=(pos, rot))

        if self.accept_control_toggle:
            g_command = self.devices[c_str].button_state['trigger_bool']
        else:
            if current_gripper_pos is None:
                g_command = self.last_grip_command
            else:
                g_command = current_gripper_pos
        self.last_grip_command = g_command

        if output_vel:
            if rot[3] < 0:
                rot = -rot
            if current_robot_rot[3] < 0:
                current_robot_rot = -current_robot_rot
            rot = q_convert(rot, 'xyzw', 'wxyz')
            current_robot_rot = q_convert(current_robot_rot, 'xyzw', 'wxyz')

            self.vel_pid.setpoint = pos
            trans_vel = self.vel_pid(current_robot_pos)

            # if velocity is being given in tool frame
            # trans_vel = np.dot(tf3d.quaternions.quat2mat(current_robot_rot), trans_vel)

            q_diff = tf3d.quaternions.qmult(tf3d.quaternions.qinverse(current_robot_rot), rot)
            if q_diff[0] < 0:
                q_diff = -q_diff
            ax, ang = tf3d.quaternions.quat2axangle(q_diff)
            rot_vel = np.dot(tf3d.quaternions.quat2mat(current_robot_rot), ang * ax * output_vel_p)

            if self.buttons_thread is not None:
                self.devices['controller_0'].thread_lock.release()

            return trans_vel, rot_vel, g_command

        if self.buttons_thread is not None:
            self.devices['controller_0'].thread_lock.release()

        return pos, rot, g_command

    def _handle_button_toggles(self, controller_index):
        """ Handle button toggles the same as was done for KeyMouseSteer. Not a long term sol'n. """
        cur_time = time.time()
        hold_time = 2
        c_str = 'controller_' + str(controller_index)
        btn_state = self.devices[c_str].button_state
        old_btn_state = self.devices[c_str].old_button_state
        if btn_state['trackpad_right_click'] and not old_btn_state['trackpad_right_click']:
            self.right_start_time = cur_time
        if not btn_state['trackpad_right_click'] and old_btn_state['trackpad_right_click']:  # falling edge
            if not self.enter_toggle:
                self.enter_toggle = True
            else:
                self.enter_toggle = False
        if btn_state['trackpad_right_click'] and (cur_time - self.right_start_time > 2):
            self.enter_hold = True
        else:
            self.enter_hold = False
        if btn_state['trackpad_left_click'] and not old_btn_state['trackpad_left_click']:
            self.space_toggle = True
        if btn_state['trackpad_up_click'] and self.d_pressed_start_time is None and not \
                old_btn_state['trackpad_up_click']:
            self.d_pressed_start_time = cur_time
        if btn_state['trackpad_up_click'] and self.d_pressed_start_time is not None and \
                (cur_time - self.d_pressed_start_time) > 2:
            self.d_pressed = True
            self.d_pressed_start_time = None
        if not btn_state['trackpad_up_click']:
            self.d_pressed_start_time = None
        if btn_state['trackpad_up_click'] and not old_btn_state['trackpad_up_click']:
            if not self.policy_play:
                self.policy_play = True
            else:
                self.policy_play = False
        if btn_state['menu'] and not old_btn_state['menu']:
            if not self.menu_toggle:
                self.menu_toggle = True
            else:
                self.menu_toggle = False
        if btn_state['trackpad_down_click'] and not old_btn_state['trackpad_down_click']:
            if not self.trackpad_down_click_toggle:
                self.trackpad_down_click_toggle = True


if __name__ == '__main__':
    vrs = VRSteer((0, 0, 1), (0, -1, 0))

    test_tool_ref_rot = convert_quat_tf_to_pb(tf3d.euler.euler2quat(0, 0, 0, 'rxyz'))

    while 1:
        vrs.update_poses()
        vrs.update_buttons()
        des_pose = vrs.move_robot([1, 2, 3], test_tool_ref_rot)
        des_rot = tf3d.euler.quat2euler(convert_quat_pb_to_tf(des_pose[1]), 'rxyz')
        print("pos %.3f, %.3f, %.3f, rot %.3f, %.3f, %.3f, grip %s" % (*des_pose[0], *des_rot, str(des_pose[2])))
        time.sleep(.05)
