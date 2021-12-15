"""
Class and methods for making a Manipulator more gym environment friendly.
"""
import numpy as np
import time
import copy

import transforms3d as tf3d

from manipulator_learning.sim.robots.manipulator import Manipulator, pose_error
from manipulator_learning.sim.utils.general import convert_quat_pb_to_tf, convert_quat_tf_to_pb, convert_pose_to_3_pts
import manipulator_learning.sim.utils.general as sim_utils


class ManipulatorWrapper:
    def __init__(self,
                 pb_client,
                 robot_config,
                 control_method,
                 gripper_control_method,
                 timestep,
                 base_pos=(0., 0., 0.),
                 base_rot=(0., 0., 0., 1.),
                 self_collision=False,
                 valid_r_dof=None,
                 valid_t_dof=None,
                 action_ref_frame='w',
                 three_pts_ee_distance=.1,
                 gripper_default_close=False,
                 max_gripper_vel=0.8,
                 gripper_force=10,
                 pos_limits=None,
                 pos_limits_frame='w',
                 force_torque_gravity_sub=0,
                 pos_ctrl_max_arm_force=None,
                 finger_constraint=True):

        self.rc = robot_config
        urdf_path = self.rc['urdf_root']
        ee_link_index = self.rc['ee_link_index']
        tool_link_index = self.rc['tool_link_index']
        gripper_indices = self.rc['gripper_indices']
        arm_indices = self.rc['arm_indices']
        base_constraint = self.rc['base_constraint']

        man_control_method = control_method
        man_gripper_control_method = gripper_control_method

        # dp and bool_p are handled in this class, Manipulator only uses 'p' or 'v'
        if control_method == 'dp':
            man_control_method = 'p'
        if gripper_control_method == 'dp' or gripper_control_method == 'bool_p':
            man_gripper_control_method = 'p'

        self.manipulator = Manipulator(pb_client, urdf_path, ee_link_index, tool_link_index,
                                       man_control_method, man_gripper_control_method, gripper_indices,
                                       arm_indices, self.rc['gripper_max'],
                                       base_pos, base_rot, get_velocities=True, self_collision=self_collision,
                                       max_gripper_vel=max_gripper_vel, gripper_force=gripper_force,
                                       force_gravity_sub=force_torque_gravity_sub,
                                       pos_ctrl_max_arm_force=pos_ctrl_max_arm_force)

        self._pb_client = pb_client
        self._timestep = timestep

        self.control_method = control_method
        self.gripper_control_method = gripper_control_method
        self._step_counter = 0
        self.body_id = self.manipulator._arm[0]

        # set initial values
        self.t_command = np.array([0,0,0])
        if control_method == 'p':
            self.r_command = np.array([0,0,0,1])
        elif control_method == 'v' or control_method == 'dp':
            self.r_command = np.array([0,0,0])

        self.gripper_default_close = gripper_default_close
        self.g_command = False

        self.ref_frame_indices = dict(t=self.manipulator._tool_link_ind, b=self.rc['base_link_index'], w=None)

        if base_constraint:
            self.base_constraint = self._pb_client.createConstraint(self.body_id, -1, -1, -1,
                self._pb_client.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
        else:
            self.base_constraint = None

        # keep fingers same distance from one another, in theory
        if finger_constraint and len(gripper_indices) > 0:
            self.finger_constraint = self._pb_client.createConstraint(self.manipulator._arm[0],
                                                    gripper_indices[0],
                                                    self.manipulator._arm[0],
                                                    gripper_indices[1],
                                                    jointType=self._pb_client.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
            self._pb_client.changeConstraint(self.finger_constraint, gearRatio=-1, erp=0.1, maxForce=50)
        else:
            self.finger_constraint = None

        # frame in which actions are given
        self.action_ref_frame = action_ref_frame

        # the distance from the tool frame origin to the other two points used for the
        # "three points" pose representation used in many papers from Levine's group
        self.three_pts_ee_distance = three_pts_ee_distance

        # variables for enforcing velocity and acceleration limits
        self.prev_vel = np.zeros([6,])
        self.prev_vel_step = 0

        # valid rotational DOF
        if valid_r_dof is not None:
            self.valid_r_dof = np.array(valid_r_dof)
        else:
            self.valid_r_dof = None
        if valid_t_dof is not None:
            self.valid_t_dof = np.array(valid_t_dof)
        else:
            self.valid_t_dof = None

        self.init_gripper_pose = None  # set by reset
        self.init_gripper_rot_eul = None  # set by reset
        self.dp_desired_pose = None  # set by reset
        self.gripper_dp_desired_pose = None

        # if self.action_ref_frame == 'b' and pos_limits_frame == 'w' and pos_limits is not None:
        if pos_limits_frame == 'w' and pos_limits is not None:
            # transform pos limits to be in base frame
            base_pose = self.manipulator.get_link_pose(self.ref_frame_indices['b'])
            base_pose_T = sim_utils.trans_quat_to_mat(base_pose[:3], base_pose[3:])
            pos_limits_mod = np.ones([2, 4])
            pos_limits_mod[0, self.valid_t_dof.nonzero()[0]] = pos_limits[0]
            pos_limits_mod[1, self.valid_t_dof.nonzero()[0]] = pos_limits[1]
            pos_limits = np.linalg.inv(base_pose_T).dot(np.array(pos_limits_mod).T).T[:, :3][:, self.valid_t_dof.nonzero()[0]]

            # make first entry the minimum
            pos_limits_fixed_order = np.vstack((np.min(pos_limits, axis=0), np.max(pos_limits, axis=0)))
            pos_limits = pos_limits_fixed_order

        self.pos_limits = pos_limits

    def step(self, action, pos_limits=None):
        """
        Apply action and step simulation forward. Apply pos_limits if not pre-defined in this class.
        """
        if pos_limits is not None and self.pos_limits is None:  # if self.pos_limits exists, handled in apply
            action[:2], limit_reached = self.limit_action(action[:2], pos_limits, 5, 5)
        self.apply_action(action[0], action[1], action[2], self.action_ref_frame)
        self._pb_client.stepSimulation()
        self._step_counter += 1

    def apply_action(self, t_command, r_command, g_command, ref_frame):
        """
        Apply the action using the Manipulator object.

        :param t_command: Translational command.
        :param r_command: Rotational command. 3 floats for acc or vel command, 4 floats(xyzw quat) for pos.
        :param g_command: Gripper command.
        :param ref_frame: Reference frame for action. Should be t or b.
        :return:
        """

        self.g_command = g_command

        if self.valid_r_dof is not None:
            r_command = self.valid_r_dof * r_command

        if self.valid_t_dof is not None:
            t_command = self.valid_t_dof * t_command

        # move the end effector
        if self.control_method == 'dp':
            # delta positions
            # save initial orientation if gripper is inadvertently pushed out of initial orientation and valid r dof
            # is not (1, 1, 1)

            # get current gripper pose in control frame
            tool_ind = self.manipulator._tool_link_ind
            ref_ind = self.ref_frame_indices[ref_frame]

            # cur_pose = self.manipulator.get_link_pose(tool_ind, ref_ind)  # feedback, but doesn't work as well
            cur_pose = copy.deepcopy(self.dp_desired_pose)  # des pose

            # to force non-valid rot axes to not be modified, set cur pose in those axes to match reset pose
            cur_pose_eul = tf3d.euler.quat2euler(sim_utils.q_convert(cur_pose[3:], 'xyzw', 'wxyz'), 'sxyz')
            fixed_cur_pose_eul = np.array(cur_pose_eul)
            invalid_r = (1 - self.valid_r_dof).astype(bool)
            fixed_cur_pose_eul[invalid_r] = self.init_gripper_rot_eul[invalid_r]
            cur_pose[3:] = sim_utils.q_convert(tf3d.euler.euler2quat(*fixed_cur_pose_eul, axes='sxyz'), 'wxyz', 'xyzw')

            # treat input rot command as ax ang, fine for relatively small delta commands
            cur_pose_T = sim_utils.trans_quat_to_mat(cur_pose[:3], cur_pose[3:])
            delta_T = np.eye(4)
            ang = np.linalg.norm(r_command)
            ax = r_command / (ang + 1e-8)
            delta_T[:3, :3] = tf3d.axangles.axangle2mat(ax, ang, is_normalized=True)
            delta_T[:3, 3] = t_command

            # to give commands in desired frame, need to have pose with pos given by tool frame, but orientation
            # given by desired ref frame, and transform from that -- only applies if ref frame is not tool frame
            if ref_frame == 't':
                new_pose_T = cur_pose_T.dot(delta_T)
                new_pos = new_pose_T[:3, 3]
                new_q = sim_utils.q_convert(tf3d.quaternions.mat2quat(new_pose_T[:3, :3]), 'wxyz', 'xyzw')
            else:
                ref_pos_quat = self.manipulator.get_link_pose(ref_ind)
                delta_ref_pose_T = np.eye(4)
                delta_ref_pose_T[:3, :3] = sim_utils.trans_quat_to_mat(ref_pos_quat[:3], ref_pos_quat[3:])[:3, :3]
                delta_ref_pose_T[:3, 3] = cur_pose_T[:3, 3]
                new_pos = delta_ref_pose_T.dot(delta_T)[:3, 3]
                new_q = sim_utils.q_convert(tf3d.quaternions.mat2quat(
                    delta_T[:3, :3].dot(cur_pose_T[:3, :3])), 'wxyz', 'xyzw')

            # enforce instance defined pos limits
            if self.pos_limits is not None:
                if ref_frame == 't':
                    raise NotImplementedError("pos limits not yet implemented for tool frame actions")
                new_pos = np.clip(new_pos, self.pos_limits[0], self.pos_limits[1])

            self.dp_desired_pose = np.array([*new_pos, *new_q])
            self.manipulator.set_frame_pose_goal(self.manipulator._tool_link_ind, new_pos, new_q, ref_ind, 3.334)

        elif self.manipulator._control_method == 'p':
            t_command = np.array(t_command)
            r_command = np.array(r_command)

            # Needed for LPF or rot dof limiting
            if self.valid_r_dof is not None:
                obs_dict = self.receive_observation('w', 'w')
                if r_command[3] < 0:  # fix quat
                    r_command = -r_command
                if obs_dict['orient'][3] < 0:
                    obs_dict['orient'] = -obs_dict['orient']
                q_rot_diff = tf3d.quaternions.qmult(tf3d.quaternions.qinverse(convert_quat_pb_to_tf(obs_dict['orient'])),
                                                    convert_quat_pb_to_tf(r_command))
                axis, theta = tf3d.quaternions.quat2axangle(q_rot_diff)

            if self.valid_r_dof is not None:
                new_axis = self.valid_r_dof * axis
                new_q_rot_diff = tf3d.quaternions.axangle2quat(new_axis, theta)
                r_command = convert_quat_tf_to_pb(tf3d.quaternions.qmult(convert_quat_pb_to_tf(obs_dict['orient']),
                                                                         new_q_rot_diff))
                r_command = np.array(r_command)

            # max joint velocity taken from Thing/UR10 settings -> 191 deg/s = 3.334 rad/s
            self.manipulator.set_frame_pose_goal(self.manipulator._tool_link_ind,
                                                 t_pos=t_command, t_rot=r_command,
                                                 ref_frame_index=self.ref_frame_indices[ref_frame],
                                                 max_joint_velocity=3.334)
            self.t_command = t_command
            self.r_command = r_command

        elif self.manipulator._control_method == 'v':

            if self.valid_r_dof is not None:
                r_command = self.valid_r_dof * r_command

            if self.valid_t_dof is not None:
                t_command = self.valid_t_dof * t_command

            if ref_frame == 't':
                tool_pose_base_ref = self.manipulator.get_link_pose(self.manipulator._tool_link_ind,
                                                                    ref_frame_index=self.ref_frame_indices['b'])
                rot_mat = tf3d.quaternions.quat2mat(convert_quat_pb_to_tf(tool_pose_base_ref[3:]))
                t_command_rot = np.dot(rot_mat, np.array(t_command))
                r_command_rot = np.dot(rot_mat, np.array(r_command))
            elif ref_frame == 'b':
                t_command_rot = t_command
                r_command_rot = r_command
            elif ref_frame == 'w':
                base_world_rot = self.manipulator.get_link_pose(self.ref_frame_indices['b'])[3:]
                rot_mat = np.linalg.inv(tf3d.quaternions.quat2mat(convert_quat_pb_to_tf(base_world_rot)))
                t_command_rot = np.dot(rot_mat, np.array(t_command))
                r_command_rot = np.dot(rot_mat, np.array(r_command))

            ## joint control override, can be useful for debugging
            # keys = self._pb_client.getKeyboardEvents()
            # if len(keys) > 0:
            #     for k in keys:
            #         speed = .5
            #         if k == ord('q'):
            #             self.manipulator.vel_cmd = [speed, 0, 0, 0, 0, 0, 0, 0]
            #         elif k == ord('a'):
            #             self.manipulator.vel_cmd = [-speed, 0, 0, 0, 0, 0, 0, 0]
            #         elif k == ord('w'):
            #             self.manipulator.vel_cmd = [0, speed, 0, 0, 0, 0, 0, 0]
            #         elif k == ord('s'):
            #             self.manipulator.vel_cmd = [0, -speed, 0, 0, 0, 0, 0, 0]
            #         elif k == ord('e'):
            #             self.manipulator.vel_cmd = [0, 0, speed, 0, 0, 0, 0, 0]
            #         elif k == ord('d'):
            #             self.manipulator.vel_cmd = [0, 0, -speed, 0, 0, 0, 0, 0]
            #         elif k == ord('r'):
            #             self.manipulator.vel_cmd = [0, 0, 0, speed, 0, 0, 0, 0]
            #         elif k == ord('f'):
            #             self.manipulator.vel_cmd = [0, 0, 0, -speed, 0, 0, 0, 0]
            #         elif k == ord('t'):
            #             self.manipulator.vel_cmd = [0, 0, 0, 0, speed, 0, 0, 0]
            #         elif k == ord('g'):
            #             self.manipulator.vel_cmd = [0, 0, 0, 0, -speed, 0, 0, 0]
            #         elif k == ord('y'):
            #             self.manipulator.vel_cmd = [0, 0, 0, 0, 0, speed, 0, 0]
            #         elif k == ord('h'):
            #             self.manipulator.vel_cmd = [0, 0, 0, 0, 0, -speed, 0, 0]
            # else:
            #     self.manipulator.vel_cmd = [0, 0, 0, 0, 0, 0, 0, 0]
            #
            # print(self.manipulator.get_joint_states()[0])

            # enforce acceleration limits -- not currently functional
            # assume new pose is achieved after a single timestep...might not be valid
            accel_limits = False
            if accel_limits:
                max_t_accel = .005  # m/s^2
                max_r_accel = .01  # rot/s^2
                obs_dict = self.receive_observation('b', 'b')
                old_vel = obs_dict['vel']['b']
                expected_t_accel = (t_command_rot - old_vel[:3]) / self._timestep
                expected_r_accel = (r_command_rot - old_vel[3:]) / self._timestep
                t_accel_mag = np.linalg.norm(expected_t_accel)
                r_accel_mag = np.linalg.norm(expected_r_accel)

                # need to scale the entire vel vectors (trans or rot) to keep directions the same
                if t_accel_mag > max_t_accel:
                    t_command_rot = old_vel[:3] + expected_t_accel / t_accel_mag * max_t_accel
                if r_accel_mag > max_r_accel:
                    r_command_rot = old_vel[3:] + expected_r_accel / r_accel_mag * max_r_accel

            # enforce velocity limits -- not currently functional
            vel_limits = False
            if vel_limits:
                max_t_vel = .05  # m/s
                max_r_vel = .1  # rad/s
                t_vel_mag = np.linalg.norm(t_command_rot)
                r_vel_mag = np.linalg.norm(r_command_rot)
                if t_vel_mag > max_t_vel:
                    t_command_rot = t_command_rot / t_vel_mag * max_t_vel
                if r_vel_mag > max_r_vel:
                    r_command_rot = r_command_rot / r_vel_mag * max_r_vel

            # enforce position limits if implemented as an instance member
            if self.pos_limits is not None:
                # since both the action and the pos limits are now rotated into the base frame,
                # use the pose given in the base frame
                pose = self.manipulator.get_link_pose(self.ref_frame_indices['t'], self.ref_frame_indices['b'])
                pos = pose[self.valid_t_dof.nonzero()[0]]

                for i in range(sum(self.valid_t_dof)):  # for x, y, and z limits
                    if pos[i] < self.pos_limits[0][i] and t_command_rot[i] < 0:
                        t_command_rot[i] = 0
                    elif pos[i] > self.pos_limits[1][i] and t_command_rot[i] > 0:
                        t_command_rot[i] = 0

            self.manipulator.set_frame_velocity_goal(self.manipulator._tool_link_ind,
                                                     t_vel=np.array([*t_command_rot, *r_command_rot]),
                                                     task=list(range(6)))

            self.t_command = t_command_rot
            self.r_command = r_command_rot

        else:
            raise NotImplementedError("The current options for control_method are p and v, "
                                      "got %s" % (str(self.manipulator._control_method)))

        # open/close the gripper
        if self.gripper_control_method == 'dp':
            cur_pos = np.array(self.manipulator.jnt_pos[-self.manipulator._num_jnt_gripper:])
            cur_pos_match = np.ones_like(cur_pos) * np.min(cur_pos)  # to ensure fingers don't get off center
            # cur_pos_match = self.gripper_dp_desired_pose  # using direct control based on actual pos, instead of des

            grip_range = self.rc['gripper_max'][0] - self.rc['gripper_max'][1]

            # rescale so that g_command of 1 fully opens, 0 fully closes
            g_command_scaled = grip_range * g_command

            # default g_command is negative, fully open pos is 1, so we want positive des pos to correspond to open
            des_pos = cur_pos_match - g_command_scaled

            self.manipulator.set_gripper_cmd(des_pos)
        else:
            if self.gripper_default_close:
                if g_command:
                    self.manipulator.open_gripper()
                else:
                    self.manipulator.close_gripper()
            else:
                if g_command:
                    self.manipulator.close_gripper()
                else:
                    self.manipulator.open_gripper()

        self.manipulator.update()

    def limit_action(self, action, pos_limits, t_vel_limit, r_vel_limit):
        # Ways to limit action (with positional, velocity, rotational, etc. limits)
        # can be overwritten by child classes, but this is a basic box limit
        pose = self.manipulator.get_link_pose(self.ref_frame_indices['t'],
                                                          self.ref_frame_indices['w'])
        pos = pose[:3]
        limit_reached = False
        if self.manipulator._control_method == 'v' or self.control_method == 'dp':
            # get action in world frame
            if self.action_ref_frame == 't':
                q = pose[3:]
                raise NotImplementedError("pos limits not yet implemented for action ref frame of t")
            elif self.action_ref_frame == 'b':
                q = self.manipulator.get_link_pose(self.ref_frame_indices['b'],
                                                    self.ref_frame_indices['w'])[3:]
                wxyz_q = [q[3], q[0], q[1], q[2]]
                rot_mat = tf3d.quaternions.quat2mat(wxyz_q)
                r_action = np.zeros_like(action)
                r_action[0] = rot_mat.dot(action[0])
                r_action[1] = rot_mat.dot(action[1])
            elif self.action_ref_frame == 'w':
                # if action already in world frame no need to change
                r_action = action

            # enforce limits
            # print(pos, pos_limits)
            for i in range(sum(self.valid_t_dof)):  # for x, y, and z limits
                if pos[i] < pos_limits[0][i] and r_action[0][i] < 0:
                    r_action[0][i] = 0
                    limit_reached = True
                elif pos[i] > pos_limits[1][i] and r_action[0][i] > 0:
                    r_action[0][i] = 0
                    limit_reached = True

            # rotate action back
            if self.action_ref_frame == 't':
                q = pose[3:]
                raise NotImplementedError("pos limits not yet implemented for action ref frame of t")
            elif self.action_ref_frame == 'b':
                action[0] = rot_mat.T.dot(r_action[0])
                action[1] = rot_mat.T.dot(r_action[1])
            elif self.action_ref_frame == 'w':
                # if action already in world frame no need to change
                action = r_action

            if np.linalg.norm(action[0]) > t_vel_limit:
                action[0] = action[0] / np.linalg.norm(action[0]) * t_vel_limit

            if np.linalg.norm(action[1]) > r_vel_limit:
                action[1] = action[1] / np.linalg.norm(action[1]) * r_vel_limit

        elif self.manipulator._control_method == 'p':
            # enforce limits
            action = np.clip(action, pos_limits[0], pos_limits[1])
            raise NotImplementedError('Currently untested, since no envs use this, but could work as is')

        return action, limit_reached

    def receive_observation(self, ref_frame_pose='b', ref_frame_vel='t'):
        """
        Get the current data from the gripper (pose, velocity, acceleration, gripper position).

        Ref frames are t for tool, b for base, w for world, or a for all.
        :return:
        """
        obs_dict = {}
        pose = self.manipulator.get_link_pose(self.manipulator._tool_link_ind,
                                              self.ref_frame_indices[ref_frame_pose])

        obs_dict['pos'] = np.array(pose[:3])
        obs_dict['orient'] = np.array(pose[3:])
        # also save the pose in the "three points" style used in some papers from Berkeley
        obs_dict['pose_3_pts'] = convert_pose_to_3_pts(
            [pose[:3], pose[3:]], dist=self.three_pts_ee_distance, axes='xy')

        obs_dict['vel'] = {}
        if ref_frame_vel == 'a':
            for frame in self.ref_frame_indices:
                obs_dict['vel'][frame] = np.array(self.manipulator.get_link_vel(self.manipulator._tool_link_ind,
                                                                self.ref_frame_indices[frame]))
        else:
            obs_dict['vel'][ref_frame_vel] = np.array(self.manipulator.get_link_vel(self.manipulator._tool_link_ind,
                                                                self.ref_frame_indices[ref_frame_vel]))
        self.prev_vel_step = self._step_counter

        obs_dict['acc'] = None
        if self.manipulator._num_jnt_gripper > 0:
            obs_dict['grip'] = np.array(self.manipulator.jnt_pos[-self.manipulator._num_jnt_gripper:])
        else:
            obs_dict['grip'] = np.atleast_1d(0)

        return obs_dict

    def receive_action(self):
        """
        Get the current commanded data from the gripper.
        :return:
        """

        com_dict = {}
        com_dict['control_type'] = self.manipulator._control_method
        if self.manipulator._control_method == 'v':
            com_dict['vel'] = np.array([*self.t_command, *self.r_command])
        elif self.manipulator._control_method == 'p':
            com_dict['pos'] = np.array(self.t_command)
            com_dict['orient'] = np.array(self.r_command)
        com_dict['grip'] = np.array(self.g_command)

        return com_dict

    def reset(self, initially_hard_set_robot_up=False,
              base_pose=((0, 0, 0), (0, 0, 0, 1)),
              init_gripper_pose=((0, 0, 1), (0, 0, 0, 1)),
              reload_urdf=False):
        """
        Reset the arm to any pose.
        """
        if reload_urdf:
            self.manipulator.reload_urdf()

        self.manipulator._e = 0
        base_pos = np.array(base_pose[0])
        base_rot = np.array(base_pose[1])
        init_gripper_pos = np.array(init_gripper_pose[0])
        init_gripper_rot = np.array(init_gripper_pose[1])
        self.manipulator.reset_commands()

        # set body pose
        self._pb_client.resetBasePositionAndOrientation(self.body_id, base_pos, base_rot)
        self.manipulator.update()

        if self.base_constraint is not None:
            self._pb_client.changeConstraint(self.base_constraint, base_pose[0],
                                            base_pose[1], 5000)

        self._step_counter = 0

        original_man_control_methods = [self.manipulator._control_method, self.manipulator._gripper_control_method]
        self.manipulator._control_method = 'p'
        self.manipulator._gripper_control_method = 'p'

        if initially_hard_set_robot_up:
            # set robot straight up
            if self.rc['robot'] == 'jaco_2_finger':
                if not self.gripper_default_close:
                    joint_positions = [3.14 / 2, 3.14, 3.14, 0, 0, 0, 0.3, 0.3]
                else:
                    joint_positions = [3.14 / 2, 3.14, 3.14, 0, 0, 0, 1.6, 1.6]
            elif self.rc['robot'] == 'jaco_3_finger':
                joint_positions = [3.14 / 2, 3.14, 3.14, 0, 0, 0, 0, 0, 0]
            elif self.rc['robot'] == 'panda':
                # joint_positions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
                joint_positions = [1.253, - 0.272, 0.417, -2.473, 0.137, 2.219, 2.359, .02, .02]
            elif self.rc['robot'] == 'thing_2f85':
                if not self.gripper_default_close:
                    joint_positions = [-1.57, -1.57, .8, -.5, -1.57, 3.14,
                                    self.rc['gripper_max'][0], 0, 0, 0,
                                    self.rc['gripper_max'][0], 0, 0, 0]
                else:
                    joint_positions = [-1.57, -1.57, .8, -.5, -1.57, 3.14,
                                    self.rc['gripper_max'][1], self.rc['gripper_max'][1]]
            elif 'thing' in self.rc['robot']:
                if not self.gripper_default_close:
                    joint_positions = [-1.57, -1.57, .8, -.5, -1.57, 3.14,
                                    self.rc['gripper_max'][0], self.rc['gripper_max'][0]]
                else:
                    joint_positions = [-1.57, -1.57, .8, -.5, -1.57, 3.14,
                                    self.rc['gripper_max'][1], self.rc['gripper_max'][1]]
            elif self.rc['robot'] == 'thing_rod':
                joint_positions = [-1.57, -1.57, .8, -.5, -1.57, 3.14]

            self.manipulator._hard_set_joint_positions(joint_positions)

        # set gripper pose
        i = 0
        t_error = [1e3, 1e3, 1e3]
        r_error = [1e3, 1e3, 1e3]
        epsilon = 3e-2

        max_timeout_iterations = 1000

        while (np.linalg.norm(t_error) > epsilon or np.linalg.norm(r_error) > epsilon) and \
                i < max_timeout_iterations:
            pose = self.manipulator.get_link_pose(self.manipulator._tool_link_ind,
                                                  ref_frame_index=self.ref_frame_indices['b'])
            cur_pos = pose[0:3]
            cur_rot = pose[3:7]

            t_error, r_error = pose_error(cur_pos, cur_rot, init_gripper_pos, init_gripper_rot)
            self.manipulator.update()
            self.manipulator.set_frame_pose_goal(self.manipulator._tool_link_ind, init_gripper_pos,
                                                 init_gripper_rot, ref_frame_index=self.ref_frame_indices['b'],
                                                 max_joint_velocity=3.334)
            if self.gripper_default_close:
                self.manipulator.close_gripper()
            else:
                self.manipulator.open_gripper()

            self._pb_client.stepSimulation()
            i += 1
            # if i % 100 == 0:
            #     print('cur_pos: ', cur_pos)
            #     print('des_pos: ', init_gripper_pos)

        # save initial pose for use in delta p control
        self.init_gripper_pose = init_gripper_pose
        self.init_gripper_rot_eul = np.array(tf3d.euler.quat2euler(
            sim_utils.q_convert(self.init_gripper_pose[1], 'xyzw', 'wxyz'), 'sxyz'))
        self.dp_desired_pose = np.array([*init_gripper_pose[0], *init_gripper_pose[1]])
        if self.gripper_default_close:
            self.gripper_dp_desired_pose = np.array([self.rc['gripper_max'][0]] * self.manipulator._num_jnt_gripper)
        else:
            self.gripper_dp_desired_pose = np.array([self.rc['gripper_max'][1]] * self.manipulator._num_jnt_gripper)

        self.manipulator._control_method, self.manipulator._gripper_control_method = original_man_control_methods

        if i >= max_timeout_iterations:
            print("Reset failure! Might be a bug.")
            return False
        else:
            return True
