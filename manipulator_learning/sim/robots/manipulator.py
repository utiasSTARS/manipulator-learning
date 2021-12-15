import copy
from liegroups import SO3, SE3
import transforms3d as tf3d
from transforms3d.quaternions import mat2quat
from numpy.linalg import lstsq
import numpy as np


TASK_DIM = 6
JOINT_NAMES = 1
JOINT_ACTIVE = 3
LINK_NAMES = 12
ZERO_DISP = [0, 0, 0]
POS = range(0, 3)
ROT = range(3, 6)
KI = .01


# originally from https://github.com/utiasSTARS/pyb-manipulator/tree/manipulator-learning

def pose_error(p1, q1, p2, q2):
    """
    Twist representing the distance between two poses in the world frame
    The transform between poses is returned in the pose 1 frame, then we rotate it back to world frame
    """
    r1 = SO3.from_quaternion(q1, 'xyzw')
    r2 = SO3.from_quaternion(q2, 'xyzw')

    t1 = SE3(r1, p1)
    t2 = SE3(r2, p2)

    xi = SE3.log((t1.inv().dot(t2)))

    return r1.dot(xi[POS]), r1.dot(xi[ROT])


class Manipulator:
    """
    Provides a pybullet API wrapper for simpler interfacing and manipulator-specific functions.
    The update() function should be called in a loop in order to store joint states and update joint controls.
    """

    def __init__(self,
                 pb_client,
                 urdf_path,
                 ee_link_index,
                 tool_link_index,
                 control_method,
                 gripper_control_method,
                 gripper_indices=(),
                 arm_indices=(),
                 gripper_max=(),
                 base_pos=(0,0,0),
                 base_rot=(0, 0, 0, 1),
                 get_velocities=False,
                 self_collision=False,
                 get_ee_ft=True,
                 use_ft_impedance=False,
                 force_gravity_sub=0,
                 max_gripper_vel=0.8,
                 gripper_force=10,
                 pos_ctrl_max_arm_force=None):
        self._pb_client = pb_client
        self.self_collision = self_collision
        self.urdf_path = urdf_path
        self.init_base_pos = base_pos
        self.init_base_rot = base_rot

        # user selected parameters -- non-private can be modified on the fly
        if not self_collision:
            self._arm = [self._pb_client.loadURDF(urdf_path)]  # arm object
        else:
            self._arm = [self._pb_client.loadURDF(urdf_path, flags=pb_client.URDF_USE_SELF_COLLISION)]  # arm object
        self._num_jnt = self._pb_client.getNumJoints(self._arm[0])  # number of joints
        self._num_lnk = self._pb_client.getNumJoints(self._arm[0])  # Equal to the number of joints I think
        self._jnt_infos = [self._pb_client.getJointInfo(self._arm[0], i) for i in range(self._num_jnt)]  # list of joint info objects

        self._active_ind = [j for j, i in zip(range(len(self._jnt_infos)), self._jnt_infos) if
                            i[JOINT_ACTIVE] > -1]  # indices of active joints
        self._true_active_ind = self._active_ind[:]
        self._gripper_ind = gripper_indices  # gripper join indices
        if len(arm_indices) > 0:
            self._arm_ind = list(arm_indices)
            self._active_ind = list(arm_indices) + list(gripper_indices)
        else:
            self._arm_ind = [e for e in self._active_ind if e not in tuple(self._gripper_ind)]  # arm joint indices

        self._num_jnt_gripper = len(self._gripper_ind)  # number of gripper joints
        self._num_jnt_arm = len(self._active_ind) - self._num_jnt_gripper  # number of arm joints

        self._control_method = control_method  # ee control method
        self._gripper_control_method = gripper_control_method  # gripper control method
        self._ee_link_ind = ee_link_index  # index of end effector link
        self._tool_link_ind = tool_link_index
        self._get_velocities = get_velocities
        self.pos_control_max_velocity = 1e10  # max joint velocity in pos control, can be set by user in set_frame_pose_goal
        self.pos_ctrl_max_arm_force = pos_ctrl_max_arm_force

        # define containers for states, poses, jacobians
        self.lnk_state = [None] * self._num_lnk
        self.lnk_pose = [None] * self._num_lnk
        self.lnk_pose_ref_frames = {}
        self.lnk_vel = [None] * self._num_lnk
        self.lnk_vel_ref_frames = {}
        self.J = np.zeros([self._num_lnk, TASK_DIM, self._num_jnt_arm])
        self.H = np.zeros([self._num_lnk, TASK_DIM, self._num_jnt_arm, self._num_jnt_arm])

        # gripper options
        self.gripper_max = gripper_max  # open max and closed max respectively
        self.max_gripper_vel = max_gripper_vel
        self.gripper_force = gripper_force
        self.gripper_p = 5  # for p control

        # initial values
        self.vel_cmd = np.zeros(len(self._active_ind))
        self.pos_cmd = np.zeros(len(self._active_ind))

        # set starting base position and orientation
        self._pb_client.resetBasePositionAndOrientation(self._arm[0], base_pos, base_rot)

        self.get_joint_states()
        self._reset_all_flags()  # reset all flags

        # error used in I PID component
        self._e = 0

        # force torque
        if get_ee_ft:
            self.ee_ft = np.zeros(6)
            self._pb_client.enableJointForceTorqueSensor(self._arm[0], self._arm_ind[-1], enableSensor=True)
            self.ft_gravity = [0, 0, force_gravity_sub]

        self.use_ft_impedance = False
        if use_ft_impedance:
            self.use_ft_impedance = True
            self.ft_gravity_buffer_size = 50
            self.ft_gravity_buffer = []

            # get fixed transform from force torque to tool frame
            ft_pos, ft_orient = self._pb_client.getLinkState(self._arm[0], self._arm_ind[-1])[4:6]
            t_pos, t_orient = self._pb_client.getLinkState(self._arm[0], self._tool_link_ind)[4:6]
            T_world_ft = self.pb_pos_orient_to_mat(ft_pos, ft_orient)
            T_world_tool = self.pb_pos_orient_to_mat(t_pos, t_orient)
            self.T_ft_to_tool = np.linalg.inv(T_world_ft).dot(T_world_tool)
            self.T_tool_to_ft = np.linalg.inv(self.T_ft_to_tool)

    def _reset_all_flags(self):
        """
        Reset all flags to false
        """
        self.__have_state = [False] * self._num_lnk
        self.__have_pose = [False] * self._num_lnk
        self.__have_pose_with_ref = []
        self.__have_vel_with_ref = []
        self.__have_vel = [False] * self._num_lnk
        self.__have_J = [False] * self._num_lnk
        self.__have_H = [False] * self._num_lnk

    # GET - PRIVATE
    # --------------------------------------------------------------------------------------------------------------

    def get_link_names(self):
        """
        Returns a list of all link names
        """
        names = []
        for info in self._jnt_infos:
            names.append(info[LINK_NAMES])

        return names

    def get_joint_names(self):
        """
        Returns a list of all joint names
        """
        names = []
        for info in self._jnt_infos:
            names.append(info[JOINT_NAMES])

        return names

    def get_ee_ft(self):
        """
        Get force torque reading of final joint
        """
        jnt_state = self._pb_client.getJointState(self._arm[0], self._arm_ind[-1])
        ft = np.array(jnt_state[2])

        # rotate gravity to subtract from force measurement
        ft_pos, ft_orient = self._pb_client.getLinkState(self._arm[0], self._arm_ind[-1])[4:6]
        R_ft = self.pb_pos_orient_to_mat(ft_pos, ft_orient)[:3, :3]
        gravity_in_ft = R_ft.dot(self.ft_gravity)
        ft[:3] = ft[:3] - gravity_in_ft
        self.ee_ft = ft

        return self.ee_ft

    def get_joint_states(self):
        """
        Get positions, velocities and torques of active joints (as opposed to passive, fixed joints)
        """
        jnt_states = self._pb_client.getJointStates(self._arm[0], range(self._pb_client.getNumJoints(self._arm[0])))
        # jnt_states = [j for j, i in zip(jnt_states, self._jnt_infos) if i[3] > -1]  # get only active states
        jnt_states = [jnt_states[i] for i in self._active_ind]
        self.jnt_pos = np.array([state[0] for state in jnt_states])
        self.jnt_vel = np.array([state[1] for state in jnt_states])
        self.jnt_torq = np.array([state[3] for state in jnt_states])

        return self.jnt_pos, self.jnt_vel, self.jnt_torq

    def get_link_state(self, link_index):
        """
        Returns information on the link URDF frame and centre of mass poses in the world frame
        """
        if not self.__have_state[link_index]:
            self.lnk_state[link_index] = self._pb_client.getLinkState(self._arm[0],
                                                        linkIndex=link_index,
                                                        computeLinkVelocity=self._get_velocities)
            self.__have_state[link_index] = True

        return self.lnk_state[link_index]

    def get_link_pose(self, link_index, ref_frame_index=None):
        """
        Get a links pose in the world frame as a 7 dimensional vector containing the
        position (x,y,z) and quaternion (x,y,z,w)
        :param link_index: Index for link to get pose of.
        :param ref_frame_index: Index for link to get pose relative to.
        """
        key = str([ref_frame_index, link_index])

        if [ref_frame_index, link_index] not in self.__have_pose_with_ref:
            lnk_state = self.get_link_state(link_index)
            lnk_frame_pos = np.asarray(lnk_state[4])
            lnk_frame_rot = np.asarray(lnk_state[5])
            if ref_frame_index is not None:
                ref_pose = self.get_link_pose(ref_frame_index)
                ref_pose_mat = np.eye(4)
                ref_pose_mat[:3, :3] = SO3.from_quaternion(ref_pose[3:], 'xyzw').as_matrix()
                ref_pose_mat[:3, 3] = ref_pose[:3]
                tf_pose_world_to_ref = np.linalg.inv(ref_pose_mat)
                lnk_pose_mat = np.eye(4)
                lnk_pose_mat[:3, :3] = SO3.from_quaternion(lnk_frame_rot, 'xyzw').as_matrix()
                lnk_pose_mat[:3, 3] = lnk_frame_pos
                lnk_pose_in_ref_mat = np.dot(tf_pose_world_to_ref, lnk_pose_mat)
                lnk_frame_pos = lnk_pose_in_ref_mat[:3, 3]
                lnk_frame_rot_wxyz = mat2quat(lnk_pose_in_ref_mat[:3, :3])
                lnk_frame_rot = np.array([*lnk_frame_rot_wxyz[1:4], lnk_frame_rot_wxyz[0]])

            self.lnk_pose_ref_frames[key] = np.concatenate(
                (lnk_frame_pos, lnk_frame_rot))  # transform from x,y,z,w to w,x,y,z
            self.__have_pose_with_ref.append([ref_frame_index, link_index])

        return self.lnk_pose_ref_frames[key]

    def get_link_vel(self, link_index, ref_frame_index=None):
        """
        Get a link's velocity in the given reference frame as a 6 dimensional vector containing
        translational and rotational velocity.
        :param link_index:
        :return:
        """
        if not self._get_velocities:
            raise AttributeError("Manipulator variable _get_velocities must be True to get velocity values.")

        key = str([ref_frame_index, link_index])

        if [ref_frame_index, link_index] not in self.__have_vel_with_ref:
            lnk_state = self.get_link_state(link_index)
            lnk_frame_lin_vel = np.asarray(lnk_state[6])
            lnk_frame_rot_vel = np.asarray(lnk_state[7])
            if ref_frame_index is not None:
                cur_rot = self.get_link_pose(ref_frame_index)[3:]
                lnk_frame_lin_vel = lnk_frame_lin_vel.dot(SO3.from_quaternion(cur_rot, 'xyzw').as_matrix())
                lnk_frame_rot_vel = lnk_frame_rot_vel.dot(SO3.from_quaternion(cur_rot, 'xyzw').as_matrix())

            # self.lnk_vel[link_index] = np.concatenate((lnk_frame_lin_vel, lnk_frame_rot_vel))
            # self.__have_vel[link_index] = True

        # return self.lnk_vel[link_index]
            self.lnk_vel_ref_frames[key] = np.concatenate((lnk_frame_lin_vel, lnk_frame_rot_vel))
            self.__have_vel_with_ref.append([ref_frame_index, link_index])

        return self.lnk_vel_ref_frames[key]

    def get_link_acc(self):
        """
        Get a link's acceleration in the given reference frame as a 6 dimensional vector containing
        translational and rotational acceleration.
        :param link_index:
        :return:
        """
        raise NotImplementedError("Getting accelerations not yet implemented.")

    def _get_link_jacobian(self, link_index):
        """
        Get the Jacobian of a link frame in the form 6xN [J_trans; J_rot]
        """
        if not self.__have_J[link_index]:
            jnt_pos = self.jnt_pos

            j_t, j_r = self._pb_client.calculateJacobian(self._arm[0], link_index, ZERO_DISP, list(jnt_pos),
                                           [0] * len(jnt_pos), [0] * len(jnt_pos))

            j = np.concatenate((j_t, j_r), axis=0)
            self.J[link_index, :, :] = j[:, :self._num_jnt_arm]  # we don't need columns associated with the gripper
            self.__have_J[link_index] = True

        return self.J[link_index, :, :]

    def _get_link_hessian(self, link_index):
        """
        Compute the Jacobian derivative w.r.t joint angles
        Ref: Arjang Hourtash, 2005.
        """
        if not self.__have_H[link_index]:
            j = self._get_link_jacobian(link_index)

            for k in range(1, self._num_jnt_arm):
                j_k = j[:, k]
                for l in range(1, self._num_jnt_arm):
                    j_l = j[:, l]

                    h = (np.cross(j_k[ROT], j_l[POS]), np.cross(j_k[ROT], j_l[ROT]))
                    self.H[link_index, :, l, k] = np.concatenate(h, axis=0).T

            self.__have_H[link_index] = True

        return self.H[link_index, :, :, :]

    # JOINT CONTROL - PRIVATE
    # --------------------------------------------------------------------------------------------------------------

    def _hard_set_joint_positions(self, cmd):
        """
        Set joint positions without simulating actual control loops
        """
        k = 0
        cmd_ind = [j for j, i in zip(range(self._pb_client.getNumJoints(self._arm[0])), self._jnt_infos) if i[3] > -1]
        for j in cmd_ind:
            self._pb_client.resetJointState(self._arm[0], j, cmd[k])
            k = k + 1

    def _joint_position_control(self, cmd, gripper_only=False, arm_only=False):
        """
        Position control of joints.
        """
        if not gripper_only:
            args = [self._arm[0]]
            for i in range(self._num_jnt_arm):
                kwargs = dict(jointIndex=self._active_ind[i], controlMode=self._pb_client.POSITION_CONTROL,
                              targetPosition=cmd[i], maxVelocity=self.pos_control_max_velocity)
                if self.pos_ctrl_max_arm_force is not None:
                    kwargs['force'] = self.pos_ctrl_max_arm_force
                self._pb_client.setJointMotorControl2(*args, **kwargs)

        # enforce max velocity for gripper joints
        if not arm_only:
            for i in range(1, self._num_jnt_gripper + 1):
                self._pb_client.setJointMotorControl2(
                    self._arm[0], jointIndex=self._active_ind[-i],
                    controlMode=self._pb_client.POSITION_CONTROL, targetPosition=cmd[-i],
                    maxVelocity=self.max_gripper_vel, force=self.gripper_force
                )

    def _joint_velocity_control(self, cmd, arm_only=False):
        """
        Velocity control of joints.
        """
        self._e = self._e + (cmd - self.jnt_vel)  # integrate error
        self._e[-self._num_jnt_gripper:] = 0  # no integral for gripper joints

        forces = [1000] * self._num_jnt_arm
        # forces = [50] * self._num_jnt_arm

        if arm_only:  # don't include gripper joints
            if self._num_jnt_gripper > 0:
                cmd = cmd[:-self._num_jnt_gripper]
                e = self._e[:-self._num_jnt_gripper]
                ji = self._active_ind[:-self._num_jnt_gripper]
            else:
                e = self._e
                ji = self._active_ind
            self._pb_client.setJointMotorControlArray(
                self._arm[0], jointIndices=ji,
                controlMode=self._pb_client.VELOCITY_CONTROL, targetVelocities=cmd + KI * e,
                forces=forces)
        else:  # include gripper joints
            forces = forces + [1] * self._num_jnt_gripper
            self._pb_client.setJointMotorControlArray(self._arm[0], jointIndices=self._active_ind,
                                                      controlMode=self._pb_client.VELOCITY_CONTROL,
                                                      targetVelocities=cmd + KI * self._e, forces=forces)

    # OTHER
    # ----------------------------------------------------------------------------------------------------------------

    def check_contact(self, objects=()):
        """
        Checks for contacts between the manipulator and given list of links indices.
        """
        if not objects:
            objects = range(self._num_jnt)

        for i in objects:
            cont = self._pb_client.getContactPoints(self._arm[0], -1, i)
            if cont:
                return True

        return False

    # SET GOALS
    # ----------------------------------------------------------------------------------------------------------------

    def set_control_method(self, m):
        """
        Sets the control method variable
        """
        self._control_method = m

    def set_joint_position_goal(self, cmd):
        """
        Set goal joint position
        """
        self.pos_cmd = cmd

    def set_joint_velocity_goal(self, cmd):
        """
        Set goal joint velocity
        """
        self.vel_cmd = cmd

    def set_frame_pose_goal(self, index, t_pos, t_rot, ref_frame_index=None, max_joint_velocity=None):
        ''' set a pose goal for an arbitrary frame'''
        if ref_frame_index is not None:
            ref_frame_pose = self.get_link_pose(link_index=ref_frame_index)
            T_world_to_ref = np.eye(4)
            T_world_to_ref[:3, 3] = ref_frame_pose[:3]
            T_world_to_ref[:3, :3] = SO3.from_quaternion(ref_frame_pose[3:], 'xyzw').as_matrix()
            T_ref_to_des = np.eye(4)
            T_ref_to_des[:3, 3] = t_pos
            T_ref_to_des[:3, :3] = SO3.from_quaternion(t_rot, 'xyzw').as_matrix()
            T_world_to_des = np.dot(T_world_to_ref, T_ref_to_des)
            t_pos = T_world_to_des[:3, 3]
            t_rot_wxyz = mat2quat(T_world_to_des[:3, :3])
            t_rot = np.array([*t_rot_wxyz[1:4], t_rot_wxyz[0]])

        result = self._pb_client.calculateInverseKinematics(self._arm[0], index, targetPosition=t_pos.tolist(),
                                              targetOrientation=t_rot.tolist(), maxNumIterations=200,
                                              residualThreshold=0.002)

        help = np.array(result)
        if max_joint_velocity is not None:
            self.pos_control_max_velocity = max_joint_velocity
        self.set_joint_position_goal(np.concatenate((help[:self._num_jnt_arm], np.zeros(self._num_jnt_gripper))))

    def set_frame_velocity_goal(self, index, t_vel, task, impedance_K=np.diag([300] * 3 + [30] * 3)):
        """
        Set Cartesian velocity goal for arbitrary frame, optionally modifying action for simple impedance control
        with ee ft sensor.
        """
        j = self._get_link_jacobian(index)

        if self.use_ft_impedance and index == self._tool_link_ind:
            t_vel = self.impedance_mod_vel(t_vel, 5, .5, impedance_K)  # this doesn't work too well at the moment

        dq, res, rank, a = lstsq(j[task, :],t_vel[task],rcond = None) # LS solver

        self.set_joint_velocity_goal(np.concatenate((dq, np.zeros(self._num_jnt_gripper))))  # Add zeros for gripper

    def impedance_mod_vel(self, vel, f_max, t_max, K=np.eye(6) * 1e3):
        """
        Modify a velocity command using ee force torque sensor with basic impedance control
        """
        norm = np.linalg.norm
        inv = np.linalg.inv
        vel_mod = copy.deepcopy(vel)
        ft = np.array(self.get_ee_ft())

        # for automatically generating ft_gravity, but can cause issues for use with real env, so currently unused
        if self.ft_gravity is None:
            self.ft_gravity_buffer.append(ft[:3])
            if len(self.ft_gravity_buffer) == self.ft_gravity_buffer_size:
                self.ft_gravity = np.array([0, 0, np.linalg.norm(np.array(self.ft_gravity_buffer).mean(axis=0))])
            return vel_mod

        # rotate gravity to subtract from force measurement
        ft_pos, ft_orient = self._pb_client.getLinkState(self._arm[0], self._arm_ind[-1])[4:6]
        R_ft = self.pb_pos_orient_to_mat(ft_pos, ft_orient)[:3, :3]
        gravity_in_ft = R_ft.dot(self.ft_gravity)
        ft[:3] = ft[:3] - gravity_in_ft

        # transform ee ft measurement to tool pose
        force_tool = self.T_ft_to_tool[:3, :3].dot(ft[:3])
        torque_tool = self.T_ft_to_tool[:3, :3].dot(ft[3:])

        t_norm = norm(torque_tool)
        if t_norm > t_max and False:
            new_torque = inv(K[3:, 3:]).dot(ft[3:])
            new_t_norm = norm(new_torque)
            new_t_max = 1 / K[3, 3] * t_max
            R_t_ext = tf3d.axangles.axangle2mat(new_torque / new_t_norm, new_t_norm - new_t_max, )
            T_R_mod = np.eye(4)
            T_R_mod[:3, :3] = R_t_ext
            T_mod_torque = self.T_tool_to_ft.dot(T_R_mod).dot(self.T_ft_to_tool)

            # get T_mod_torque as ax angle and delta pos
            T_mod_torque_ax, T_mod_torque_ang = tf3d.axangles.mat2axangle(T_mod_torque[:3, :3])
            vel_mod[:3] = vel[:3] - T_mod_torque[:3, 3]
            vel_mod[3:] = vel[3:] - T_mod_torque_ax * T_mod_torque_ang

        f_norm = norm(force_tool)
        if f_norm > f_max:
            new_force = force_tool - (force_tool / f_norm) * f_max
            vel_mod[:3] = inv(K[:3, :3]).dot(new_force) + vel_mod[:3]

        return vel_mod

    def pb_pos_orient_to_mat(self, pos, orient):
        """
        Get a 4x4 transformation matrix given a pb pos and orientation
        """
        mat = np.eye(4)
        mat[:3, :3] = SO3.from_quaternion(orient, 'xyzw').as_matrix()
        mat[:3, 3] = pos
        return mat

    def invert_transform(self, mat):
        """
        Inverse transform of 4x4 matrix
        """
        mat_out = np.eye(4)
        C_out_inv = mat[:3, :3].T
        mat_out[:3, :3] = C_out_inv
        mat_out[:3, 3] = -C_out_inv.dot(mat[:3, 3])
        return mat_out

    def close_gripper(self):
        """
        Close the robot gripper (modifies the current joint position command)
        """
        if self._num_jnt_gripper > 0:
            if self._gripper_control_method == 'p':
                self.pos_cmd[-self._num_jnt_gripper:] = self.gripper_max[1] * np.ones(self._num_jnt_gripper)
            elif self._gripper_control_method == 'v':
                # self.vel_cmd[-self._num_jnt_gripper:] = self.gripper_p * (self.jnt_pos[-self._num_jnt_gripper:]
                #                                          <= self.gripper_max[1]).astype(float)
                self.vel_cmd[-self._num_jnt_gripper:] = self.gripper_p * (self.gripper_max[1] - self.jnt_pos[-self._num_jnt_gripper:]).astype(float)
                self.vel_cmd[-self._num_jnt_gripper:] = np.clip(self.vel_cmd[-self._num_jnt_gripper:],
                                                                -np.inf, self.max_gripper_vel)

    def open_gripper(self):
        """
        Open the robot gripper (modifies the current joint position command)
        """
        if self._num_jnt_gripper > 0:
            if self._gripper_control_method == 'p':
                self.pos_cmd[-self._num_jnt_gripper:] = self.gripper_max[0] * np.ones(self._num_jnt_gripper)
            elif self._gripper_control_method == 'v':
                self.vel_cmd[-self._num_jnt_gripper:] = -self.gripper_p * (self.jnt_pos[-self._num_jnt_gripper:]
                                                                             - self.gripper_max[0]).astype(float)
                self.vel_cmd[-self._num_jnt_gripper:] = np.clip(self.vel_cmd[-self._num_jnt_gripper:],
                                                                -self.max_gripper_vel, np.inf)

    def set_gripper_cmd(self, cmd):
        """
        Set the current gripper command to a specific value, clipped to limits.
        """
        if self._num_jnt_gripper > 0:
            if self._gripper_control_method == 'p':
                self.pos_cmd[-self._num_jnt_gripper:] = np.clip(cmd, self.gripper_max[1], self.gripper_max[0])
            elif self._gripper_control_method == 'v':
                self.vel_cmd[-self._num_jnt_gripper] = np.clip(cmd, -self.max_gripper_vel, self.max_gripper_vel)


    # UPDATE INTERNALLY
    # ----------------------------------------------------------------------------------------------------------------
    def update(self):
        """
        This function should be configurable
        """

        # run iteration of control loop
        if self._control_method == 'p' and self._gripper_control_method == 'p':
            self._joint_position_control(self.pos_cmd, arm_only=False)
        elif self._control_method == 'v' and self._gripper_control_method == 'v':
            self._joint_velocity_control(self.vel_cmd, arm_only=False)
        elif self._control_method == 'v' and self._gripper_control_method == 'p':
            self._joint_velocity_control(self.vel_cmd, arm_only=True)
            self._joint_position_control(self.pos_cmd, gripper_only=True)

        # get joint positions, velocities, torques
        self.get_joint_states()
        self._reset_all_flags()

    def reset_commands(self):
        """
        Set all commands to zero.
        """
        self.vel_cmd = np.zeros(len(self._active_ind))
        self.pos_cmd = np.zeros(len(self._active_ind))

    def reload_urdf(self):
        """
        Reload the urdf of the robot after deleting the current robot.
        """

        # this checks to see if the object currently exists
        if self._pb_client.getBodyUniqueId(self._arm[0]) >= 0:
            self._pb_client.removeBody(self._arm[0])

        self.__init__(self._pb_client, self.urdf_path, self._ee_link_ind, self._tool_link_ind,
                      self._control_method, self._gripper_control_method,
                      self._gripper_ind, self._arm_ind, self.gripper_max,
                      self.init_base_pos, self.init_base_rot, self._get_velocities, self.self_collision)

        # if not self.self_collision:
        #     self._arm = [self._pb_client.loadURDF(self.urdf_path)]  # arm object
        # else:
        #     self._arm = [self._pb_client.loadURDF(self.urdf_path,
        #                                           flags=self._pb_client.URDF_USE_SELF_COLLISION)]  # arm object
