import pybullet
import pybullet_data
import gym
import numpy as np
import time
import copy
import pkgutil
from gym.utils import seeding
import transforms3d as tf3d

from manipulator_learning.sim.utils.pb_tools import BulletClient, load_table, add_pb_frame_marker, \
    add_pb_frame_marker_by_pose
import manipulator_learning.sim.utils.general as sim_utils
from manipulator_learning.sim.utils.general import TransformMat, invert_transform, q_convert, convert_quat_tf_to_pb, \
    trans_quat_to_mat, convert_quat_pb_to_tf
from manipulator_learning.sim.robots.cameras import EyeInHandCam, WorkspaceCam
from manipulator_learning.sim.robots.manipulator_wrapper import ManipulatorWrapper


SHOW_PB_FRAME_MARKERS = False


class PBEnv(gym.Env):
    """ Create a pybullet env. Although this inherits gym.Env, it shouldn't be used on its own, but rather as
    an instance in part of another class. """
    RENDER_MODES = ['human', 'vr_headset', 'workspace', 'eye-in-hand', 'both', 'overhead',
                    'robot_facing', 'robot_facing_alt', 'robot_side_cam', 'rgb_and_true_depth',
                    'panda_play_iso_cam']

    def __init__(
            self,
            object_urdf_root,
            robot_config,  # dict with robot, urdf_root, num_controllable_joints, num_gripper_joints, base_link_index,
                           # ee_link_index, tool_link_index, gripper_indices, arm_indices, gripper_max, base_constraint
            seed=None,

            # robot parameters
            robot_base_pose=((1, 0, 0.9), (0, 0, 0)),  # robot base pose in world coordinates, rotating-xyz euler
            workspace_center=(0.95, -.15, .7),  # the center point of the workspace
            robot_base_ws_cam_tf=((.45, -.15, .65), (160 / 180 * np.pi, 0, np.pi)), # (trans, static-xyz eul rot) from base to main workspace cam
            init_gripper_random_lim=None,  # None or [+-x, +-y, +-z, +-x_rot, +-y_rot, +-z_rot], but magnitudes will
                                           # correspond to size of a full box, e.g. [.25, 0, 0, 0, 0, 0] means
                                           # self.init_gripper_pose[0] + uniform(-.125, .125)
            gripper_control_method='bool_p',  # bool_p means T/F for close/open, dp means delta pos
            init_gripper_pose=((.5, -.225, .8), (0, 0, 0, 1)),  # can be (pos, xyzw-quat) or (pos, static-xyz euler)
            control_frame='t',  # reference frame that control commands are given in
            valid_rot_dof=None,  # valid rotational degrees of freedom in given control frame
            valid_trans_dof=None,  # valid translational degrees of freedom in given control frame
            gripper_default_close=False,  # whether the default state of the gripper should be closed
            max_gripper_vel=0.8,  # max velocity for gripper fingers
            gripper_force=10,  # max force for gripper fingers
            control_method='v',  # v for velocity or dp for delta pos
            pos_limits=None,  # new-style pos_limits, (3-tuple, 3-tuple) of min xyz, max xyz in world coordinates
            pos_limits_frame='w',
            force_torque_gravity_sub=0,  # amount to subtract from force-torque sensor to give 0 reading when static (i.e., norm of xyz force reading when static)
            pos_ctrl_max_arm_force=None,  # max force of arm when using position control (including dp), default is undocumented by pybullet

            # multiview robot config
            base_random_lim=None,  # after randomizing base theta, amount to uniformly random each dimension as ((pos), (static-xyz euler))
            cam_workspace_distance=.3,  # distance from cam frame to workspace center, used if base_pose_from_workspace_center is True
            base_pose_from_workspace_center=False,  # whether to set the base pose using the workspace center..if true, only theta from robot_base_pose is used
            random_base_theta_bounds=(0, 0),  # if base_pose_from_workspace_center is True, this is the min and max values for theta

            # pybullet
            existing_pb_client=None,
            render_opengl_gui=False,  # pybullet opengl gui shown or not
            render_ground_plane=True,  # render ground plane in pybullet
            use_egl=True,  # egl renderer in pybullet
            time_step=0.01,  # used by pybullet as real time per pybullet step, .01 should be maximum
            renderer='opengl',  # camera renderer, 'opengl' for gpu or anything else for cpu

            force_pb_direct=False,  # make pybullet use direct mode

            # env observation/action config
            poses_ref_frame='w',  # 'b' or 'w' for base or world, frame that all observations/states of poses are given in
            vel_ref_frame='t',  # 't', 'b', 'w', reference frame for manipulator velocities
            render_shadows=False,  # render shadows in pybullet
            pose_3_pts_dist=0.1,  # distance between ref frame and points for 3 pts representations of rotations
            image_height=120,  # main camera image height
            image_width=160,  # main camera image width

            # objects config
            mark_on_table=False,  # load a table with a mark on it instead of the regular one
            tray_type=None,  # options: [None, 'normal', '2_cube_insert']

            # new-style object design and randomization. currently no support for orientation, and only randomizes z axis
            # each tuple must be the same length
            obj_random_lim=None,  # tuple of 3-tuples for 'box sizes' of starting positions of objects, with init pos in center
            obj_init_pos=None,  # tuple of 3-tuples for initial pos, relative to workplace center, of objects
            obj_rgba=None,  # tuple of 4-tuples for rgba of each object
            obj_urdf_names=None,  # tuple of strings of urdfs of each object
            objs_in_state=None,  # tuple of indices of objects that should be included in state/observation
            rel_pos_in_state=None,  # tuple of indices or tuples of indices for relative positions that should be in state
                                    # e.g., (0, 1, (0, 1)) means include ee-->0, ee-->1, and 0-->1

            # debug
            debug_cam_params=(.12, -.38, .78, 1.4, -27.8, 107.6),  # debug cam view in opengl gui

            # deprecated
            goal_type=None, # deprecated, include goals in obj_urdf_names and in specific env code, see panda_play for ex
            goal_pos=None,  # deprecated, see above
            init_block_pos=None,  # deprecated, use obj_init_pos
            block_style='cube',  # deprecated, use obj_urdf_names
            task='grasp_and_place',  # deprecated, handle task-specific things in top level env classes
            block_colors=None,  # deprecated, for setting block colors
            block_random_lim=None,  # deprecated, use obj_random_lim

            # insertion (deprecated)
            rod_random_lim=None,  # deprecated, for old insertion tasks only
            init_rod_pos=None,  # deprecated, for old insertion tasks only
    ):
        # gym
        self.np_random = None
        self.seed(seed)

        # robot config
        self.rc = robot_config
        self.robot_base_pose = robot_base_pose
        self.robot_base_pose_pb = (
            robot_base_pose[0], convert_quat_tf_to_pb(tf3d.euler.euler2quat(*robot_base_pose[1], 'rxyz')))
        self.workspace_center = workspace_center
        self.robot_base_ws_cam_tf = robot_base_ws_cam_tf
        self.robot_base_ws_cam_T = TransformMat(pb_pose=(
            robot_base_ws_cam_tf[0],
            convert_quat_tf_to_pb(tf3d.euler.euler2quat(*robot_base_ws_cam_tf[1], 'sxyz'))))
        self.init_gripper_random_lim = init_gripper_random_lim
        self.gripper_control_method = gripper_control_method

        # user can supply orientation as sxyz euler or as quat, ends up stored as pos, xyzw-quat
        if len(init_gripper_pose[1]) == 3:
            pose = init_gripper_pose
            self.init_gripper_pose = \
                [pose[0], q_convert(tf3d.euler.euler2quat(*pose[1], axes='sxyz'), 'wxyz', 'xyzw')]
        else:
            self.init_gripper_pose = init_gripper_pose

        # multiview additional robot config
        self.base_random_lim = base_random_lim
        self.cam_workspace_distance = cam_workspace_distance
        self.base_pose_from_workspace_center = base_pose_from_workspace_center
        self.random_base_theta_bounds = random_base_theta_bounds

        # pybullet config
        self._render_opengl_gui = render_opengl_gui
        self.render_ground_plane = render_ground_plane

        # env observation/action config
        assert poses_ref_frame in ['b', 'w'], f"poses_ref_frame must be b or w, got {poses_ref_frame}"
        self.poses_ref_frame = poses_ref_frame
        self.vel_ref_frame = vel_ref_frame
        self.initial_move_made = False
        self.first_gripper_close = False
        self.last_gripper_close = False

        # objects config
        self.object_urdf_root = object_urdf_root
        self.mark_on_table = mark_on_table
        self.tray_type = tray_type
        if obj_init_pos is not None:
            assert len(obj_random_lim) == len(obj_init_pos) == len(obj_rgba) == len(obj_urdf_names)
        self.obj_random_lim = obj_random_lim
        self.obj_init_pos = obj_init_pos
        self.obj_rgba = obj_rgba
        self.obj_urdf_names = obj_urdf_names
        self.objs_in_state = objs_in_state
        self.rel_pos_in_state = rel_pos_in_state
        self._obj_ids = []
        self._prev_obj_vels = None
        self._prev_step_count_for_acc = 0
        self.green_on_blue = False
        self.table = None
        self.tray = None

        # image generation, debug
        self.workspace_cam_frame_marker_ids = []
        self.kinect = None
        self._base_theta_rel_to_fixed = None
        self.debug_cam_params = debug_cam_params

        # deprecated
        if block_random_lim is not None:
            assert (len(block_random_lim) == len(init_block_pos)), "block_random_lim and init_block_pos must have the" \
                                                                   "same number of entries."
        self.goal_type = goal_type
        self.goal_pos = goal_pos
        self.init_block_pos = init_block_pos
        self.block_style = block_style
        self.task = task
        self.block_colors = block_colors
        self.block_random_lim = block_random_lim
        self.block_ids = []
        self.goal_id = None
        self.goal2_id = None

        # insertion objects (deprecated)
        if 'insertion' in self.task:
            self.insertion_box = None
            self.insertion_rod = None
            self.insertion_timer_start = time.time()
            self.insertion_rod_const = None
            self.rod_random_lim = rod_random_lim
            self.init_rod_pos = init_rod_pos

        # door objects (deprecated)
        if 'door' in self.task:
            self.door = None
            self.door_back = None

        # start of env generation
        # start pybullet
        if force_pb_direct:  # useful for multiprocessing or similar
            self._pb_client = BulletClient(connection_mode=pybullet.DIRECT)
        else:
            if existing_pb_client is None:
                if self._render_opengl_gui:
                    self._pb_client = BulletClient(connection_mode=pybullet.GUI_SERVER)
                else:
                    self._pb_client = BulletClient(connection_mode=pybullet.SHARED_MEMORY_SERVER)
            else:
                self._pb_client = existing_pb_client
        self._pb_client.setGravity(0, 0, -10)
        self._pb_client.setTimeStep(time_step)
        self._pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # egl options for faster headless rendering when generating camera images..no effect if camera not needed
        self.egl_plugin = None
        if use_egl:
            egl = pkgutil.get_loader('eglRenderer')
            if egl:
                self.egl_plugin = self._pb_client.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                self.egl_plugin = self._pb_client.loadPlugin("eglRendererPlugin")
            light_direction = (-50, 30, 40)
        else:
            light_direction = None

        # caused various issues, so they needed to be turned off
        if self._render_opengl_gui:
            self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            if render_shadows:
                self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_SHADOWS, 1)
            else:
                self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_SHADOWS, 0)

        # create camera objects
        base_to_ws_cam_q = convert_quat_tf_to_pb(
            tf3d.euler.euler2quat(*self.robot_base_ws_cam_tf[1], 'sxyz'))
        self.workspace_cam = EyeInHandCam(
            pb_client=self._pb_client, frame_rel_trans=self.robot_base_ws_cam_tf[0],
            frame_rel_rot=base_to_ws_cam_q, forward_axis=[0, 0, 1], up_axis=[0, -1, 0],
            width=image_width, height=image_height, aspect=image_width / image_height,
            renderer=renderer, render_shadows=render_shadows,
            light_direction=light_direction
        )

        self.workspace_cam_high_res = EyeInHandCam(
            pb_client=self._pb_client, frame_rel_trans=self.robot_base_ws_cam_tf[0],
            frame_rel_rot=base_to_ws_cam_q, forward_axis=[0, 0, 1], up_axis=[0, -1, 0],
            width=160, height=120, aspect=160 / 120, renderer=renderer, render_shadows=render_shadows,
            light_direction=light_direction
        )

        quat = convert_quat_tf_to_pb(tf3d.euler.euler2quat(0, 0.4, 0, 'sxyz'))
        self.wrist_cam = EyeInHandCam(pb_client=self._pb_client, frame_rel_trans=[-.1, 0, -0.1],
                                      frame_rel_rot=quat, forward_axis=[0, 0, 1], up_axis=[-1, 0, 0],
                                      width=image_width, height=image_height, aspect=image_width / image_height,
                                      renderer=renderer, render_shadows=render_shadows,
                                      light_direction=light_direction)

        if self.render_ground_plane:
            self.ground_plane = self._pb_client.loadURDF(pybullet_data.getDataPath() + "/plane.urdf",
                                     [0.000000, 0.000000, 0.000000],
                                     [0.000000, 0.000000, 0.000000, 1.000000])

        # create manipulator object
        self.gripper = ManipulatorWrapper(
            pb_client=self._pb_client, robot_config=self.rc, control_method=control_method,
            gripper_control_method=self.gripper_control_method, timestep=time_step, base_pos=self.robot_base_pose_pb[0],
            base_rot=self.robot_base_pose_pb[1], self_collision=False, action_ref_frame=control_frame,
            valid_r_dof=valid_rot_dof, valid_t_dof=valid_trans_dof,
            three_pts_ee_distance=pose_3_pts_dist, gripper_default_close=gripper_default_close,
            max_gripper_vel=max_gripper_vel, gripper_force=gripper_force, pos_limits=pos_limits,
            pos_limits_frame=pos_limits_frame, force_torque_gravity_sub=force_torque_gravity_sub,
            pos_ctrl_max_arm_force=pos_ctrl_max_arm_force
        )
        self._time_step = time_step  # also sets it in client and manipulator through setter

        # extra cameras
        self.overhead_cam = WorkspaceCam(self._pb_client, 640, 480, 640 / 480, [.05, -.2, 1.9], [.13, -.2, .79],
                                         renderer=renderer, render_shadows=render_shadows,
                                         light_direction=light_direction)

        width = 852; height = 480
        self.robot_facing_cam = WorkspaceCam(self._pb_client, width, height, width / height, [1.023, -.225, .981],
                                             [-.252, -.215, -.559],
                                         renderer=renderer, render_shadows=render_shadows,
                                         light_direction=light_direction)

        width = 1920; height = 1080
        self.robot_facing_alt = WorkspaceCam(self._pb_client, width, height, width / height,
                                                  [0.95684942, 0.13307215, 1.13294473],
                                                  [-0.13771601, -0.77616041, -0.27248142],
                                                  renderer=renderer, render_shadows=render_shadows,
                                                  light_direction=light_direction)

        width = 852; height = 480
        self.robot_side_cam = WorkspaceCam(self._pb_client, width, height, width / height,
                                           [0.58844697, 0.17096335, 1.07960373],
                                           [0.6382586, -1.17904437, -0.3951837],
                                           # [0.67347074, -0.50665982, 1.03154717],
                                           # [0.19469497,  0.83332193, -0.37387898],
                                           # [0.36, 2.0, 1.62],
                                           # [0.36, -0.19, 0.62],
                                           renderer=renderer, render_shadows=render_shadows,
                                           light_direction=light_direction)

        width = 640; height = 480
        self.panda_play_iso_cam = WorkspaceCam(self._pb_client, width, height, width / height,
                                               [0.75, 0.15, 0.97], [1.631, -1.134, -0.285],
                                               renderer=renderer, render_shadows=render_shadows,
                                               light_direction=light_direction)

        width = 640; height = 480
        self.panda_play_side_cam = WorkspaceCam(self._pb_client, width, height, width / height,
                                               [0.55, -0.1, 0.92], [2.243, -0.633, -0.001],
                                               renderer=renderer, render_shadows=render_shadows,
                                               light_direction=light_direction)

        width = 960; height = 720
        self.panda_play_closer_side_cam = WorkspaceCam(self._pb_client, width, height, width / height,
                                                [0.8, -0.1, 0.82], [2.296, -0.529, -0.436],
                                                renderer=renderer, render_shadows=render_shadows,
                                                light_direction=light_direction)

        width = 960; height = 720
        self.panda_play_blue_side_cam = WorkspaceCam(self._pb_client, width, height, width / height,
                                                       [1.2, -0.05, 0.82], [0.075, -1.125, -0.436],
                                                       renderer=renderer, render_shadows=render_shadows,
                                                       light_direction=light_direction)

        width = 960; height = 720
        self.panda_play_alt_blue_side_cam = WorkspaceCam(self._pb_client, width, height, width / height,
                                                     [1.2, 0, 0.82], [0.022, -1.017, -0.436],
                                                     renderer=renderer, render_shadows=render_shadows,
                                                     light_direction=light_direction)

        if SHOW_PB_FRAME_MARKERS:
            add_pb_frame_marker(self._pb_client, self.gripper.body_id, self.gripper.manipulator._arm_ind[-1])
            add_pb_frame_marker(self._pb_client, self.gripper.body_id, self.gripper.manipulator._tool_link_ind)
            add_pb_frame_marker(self._pb_client, self.gripper.body_id, self.gripper.ref_frame_indices['b'])

    @property
    def _time_step(self):
        return self.__time_step

    @_time_step.setter
    def _time_step(self, val):
        self._pb_client.setTimeStep(val)
        self.gripper.time_step = val
        self.__time_step = val

    def seed(self, seed=None):
        """ Seed for random numbers, for e.g. resetting the environment """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, pos_limits=None):
        """ Take a step in the environment. This does not actually give a camera/depth image, for that, use render.

        pos_limits here is deprecated. pos_limits should be set when this class is instantiated."""

        self.gripper.step(action, pos_limits=pos_limits)

        # reward function depends on task/environment setup -- deprecated, handled by top level classes now
        if self.task == 'grasp_and_place':
            # reward based on block and plate distance
            bowl_pose = self._pb_client.getBasePositionAndOrientation(self.goal_id)
            block_pose = self._pb_client.getBasePositionAndOrientation(self.block_ids[0])

            # negative so higher distance -> lower reward
            reward = -np.linalg.norm(np.array(bowl_pose[0]) - np.array(block_pose[0]))
        elif self.task == 'reaching':
            # reward based on "graspability"
            block_pose = self._pb_client.getBasePositionAndOrientation(self.block_ids[0])
            ee_pose_world = self.gripper.manipulator.get_link_pose(
                self.gripper.manipulator._tool_link_ind, ref_frame_index=None)
            reward = -np.linalg.norm(np.array(block_pose[0]) - np.array(ee_pose_world[:3]))

            # additional reward based on non contact of block with table
            num_contact_points = len(self._pb_client.getContactPoints(self.block_ids[0], self.table))
            reward -= num_contact_points
        elif self.task == 'pushing_easy' or self.task == 'pushing_hard' or self.task == 'pushing_xy':
            push_block_pose = self._pb_client.getBasePositionAndOrientation(self.block_ids[0])
            stat_block_pose = self._pb_client.getBasePositionAndOrientation(self.block_ids[1])
            reward = -np.linalg.norm(np.array(push_block_pose[0]) - np.array(stat_block_pose[0]))
        elif self.task == 'insertion':
            insertion_box_pose = self._pb_client.getBasePositionAndOrientation(self.insertion_box)
            rod_pose = self._pb_client.getBasePositionAndOrientation(self.insertion_rod)
            reward = -np.linalg.norm(np.array(insertion_box_pose[0]) - np.array(rod_pose[0]))
        else:
            reward = 0

        # prepare observations dict after step in env is taken
        obs_dict = self._prepare_obs_dict_new()
        self.initial_move_made = True

        if not self.first_gripper_close and obs_dict['command']['grip']:
            self.first_gripper_close = True

        if self.first_gripper_close and not self.last_gripper_close and not obs_dict['command']['grip']:
            self.last_gripper_close = True

        return obs_dict, reward, False, {}

    def _prepare_obs_dict_new(self):
        obs_dict = {'actual': self.gripper.receive_observation(ref_frame_pose=self.poses_ref_frame,
                                                               ref_frame_vel=self.vel_ref_frame),
                    'command': self.gripper.receive_action(),
                    'block_poses': self._prepare_block_poses_dict()}
        return obs_dict

    def _prepare_block_poses_dict(self):
        block_dict = {'pos': [], 'orient': [], 'vel': [], 'acc': []}

        if self.rel_pos_in_state is None:
            obj_poses, obj_vels, obj_accs = self._get_block_poses()
        else:
            obj_poses, obj_vels, obj_accs, obj_rel_poss = self._get_block_poses()
            block_dict['rel_pos'] = []
            for rel_pos in obj_rel_poss:
                block_dict['rel_pos'].append(rel_pos)

        for pos_quat, vel, acc in zip(obj_poses, obj_vels, obj_accs):
            block_dict['pos'].append(pos_quat[0])
            block_dict['orient'].append(pos_quat[1])
            block_dict['vel'].append(vel)
            block_dict['acc'].append(acc)

        # treat door pose as if it was a block for old-style environment
        if 'door' in self.task:
            door_state = self._pb_client.getLinkState(self.door, linkIndex=1)
            pose = (door_state[0], door_state[1])
            if self.poses_ref_frame == 'b':
                world_to_rel_frame = self.gripper.manipulator.get_link_pose(
                    self.gripper.ref_frame_indices[self.poses_ref_frame])
                T_w_to_rel = trans_quat_to_mat(world_to_rel_frame[:3], world_to_rel_frame[3:])
                T_rel_to_w = invert_transform(T_w_to_rel)
                T_world_obj = trans_quat_to_mat(*pose)
                T_rel_to_obj = T_rel_to_w.dot(T_world_obj)
                pose = TransformMat(T_rel_to_obj).to_pb()
            block_dict['pos'].append(pose[0])
            block_dict['orient'].append(pose[1])

        block_dict['pos'] = np.array(block_dict['pos'])
        block_dict['orient'] = np.array(block_dict['orient'])
        block_dict['vel'] = np.array(block_dict['vel'])
        block_dict['acc'] = np.array(block_dict['acc'])

        return block_dict

    def _get_block_poses(self):
        """ Return the positions of the blocks in the environment as a list """
        poses = []
        vels = []
        accs = []

        # get the new ref frame for quickly converting world poses as given by pb
        if self.poses_ref_frame == 'b':
            w_to_r_frame = self.gripper.manipulator.get_link_pose(
                self.gripper.ref_frame_indices[self.poses_ref_frame])
            rel_to_world_tq = sim_utils.TransformMat(
                mat=sim_utils.invert_transform(trans_quat_to_mat(w_to_r_frame[:3], w_to_r_frame[3:]))).to_pb(single_tuple=True)
            rel_to_world_tq = (rel_to_world_tq[:3], rel_to_world_tq[3:])

        if len(self._obj_ids) > 0:  # new-style objects
            poses_dict = dict.fromkeys(range(len(self._obj_ids)))  # dict to reuse for relative positions if desired
            objs_in_state_arr = np.array(self.objs_in_state)
            obj_id_list = np.array(self._obj_ids)[objs_in_state_arr]
        else:  # old-style objects
            obj_id_list = self.block_ids
        for b_index, b_id in enumerate(obj_id_list):
            # need to transform positions to be relative to desired poses ref frame.
            # necessary to mimic having a sensor attached to either the tool (eye-in-hand) or the base
            obj_world_pose = self._pb_client.getBasePositionAndOrientation(b_id)
            obj_world_vel = self._pb_client.getBaseVelocity(b_id)

            if self.poses_ref_frame == 'b':
                pose = sim_utils.change_pose_ref_frame(obj_world_pose, rel_to_world_tq, ref_pose_inverted=True)
                vel = sim_utils.change_vel_ref_frame(obj_world_vel, w_to_r_frame)
            else:
                pose = obj_world_pose
                vel = np.concatenate([obj_world_vel[0], obj_world_vel[1]])

            poses.append(pose)
            vels.append(np.array(vel).flatten())
            if self.gripper._step_counter == 0:
                accs.append(np.zeros(6))
            else:
                accs.append((np.array(vel) - self._prev_obj_vels[b_index]) / self._time_step)
            self._prev_obj_vels[b_index] = vel

            if self.rel_pos_in_state is not None:
                poses_dict[objs_in_state_arr[b_index]] = pose

        # also get relative positions of gripper/objects or object/object if desired
        if self.rel_pos_in_state is not None:
            rel_poss = []
            rel_frame_to_gripper = None
            all_obj_id_list = np.array(self._obj_ids)
            for obj_list in self.rel_pos_in_state:
                if type(obj_list) == int:  # means rel pos between obj and EE
                    if rel_frame_to_gripper is None:
                        rel_frame_to_gripper = self.gripper.manipulator.get_link_pose(
                            self.gripper.manipulator._tool_link_ind, self.gripper.ref_frame_indices[self.poses_ref_frame])
                    if poses_dict[obj_list] is None:
                        obj_world_pose = self._pb_client.getBasePositionAndOrientation(all_obj_id_list[obj_list])
                        poses_dict[obj_list] = sim_utils.change_pose_ref_frame(
                            obj_world_pose, rel_to_world_tq, ref_pose_inverted=True)
                    rel_poss.append(np.array(poses_dict[obj_list][0]) - np.array(rel_frame_to_gripper[:3]))
                elif len(obj_list) == 2:  # rel pos between obj and other obj
                    for o in obj_list:
                        if poses_dict[o] is None:
                            obj_world_pose = self._pb_client.getBasePositionAndOrientation(all_obj_id_list[o])
                            poses_dict[o] = sim_utils.change_pose_ref_frame(
                                obj_world_pose, rel_to_world_tq, ref_pose_inverted=True)
                    rel_poss.append(np.array(poses_dict[obj_list[1]][0]) - np.array(poses_dict[obj_list[0]][0]))
                else:
                    raise ValueError("Entries in rel_pos_in_state must be length 1 or 2, got %s" % obj_list)

            return poses, vels, accs, rel_poss

        return poses, vels, accs

    def get_random_gripper_pose(self):
        """ Get a random gripper pose on a reset """
        # interpreted as 6-tuple as defined in __init__
        # sampling random rotations as euler angles is definitely a bad idea for a large sample space,
        # but for a small one it's adequate
        main_init_gripper_rot_eul = tf3d.euler.quat2euler(convert_quat_pb_to_tf(self.init_gripper_pose[1]), axes='sxyz')
        init_gripper_pose_mod = self.np_random.uniform(low=-np.array(self.init_gripper_random_lim) / 2,
                                                       high=np.array(self.init_gripper_random_lim) / 2, size=6)
        new_init_gripper_trans = self.init_gripper_pose[0] + init_gripper_pose_mod[:3]
        new_init_gripper_rot_eul = main_init_gripper_rot_eul + init_gripper_pose_mod[3:]
        new_init_gripper_rot = convert_quat_tf_to_pb(tf3d.euler.euler2quat(*new_init_gripper_rot_eul, axes='sxyz'))
        init_gripper_pose = [list(new_init_gripper_trans), new_init_gripper_rot]
        return init_gripper_pose

    def get_reset_base_pose(self, mb_base_angle=None):
        """ Get a reset base pose, uniform random using random_base_theta_bounds if mb_base_angle is not None. """

        if self.base_pose_from_workspace_center:
            T_BC = self.robot_base_ws_cam_T.pose_mat
            if mb_base_angle is not None:
                theta = self.robot_base_pose[1][2] + mb_base_angle
            else:
                theta = self.np_random.uniform(
                    low=self.robot_base_pose[1][2] + self.random_base_theta_bounds[0],
                    high=self.robot_base_pose[1][2] + self.random_base_theta_bounds[1],
                    size=1
                ).item()
            self._base_theta_rel_to_fixed = theta - self.robot_base_pose[1][2]
            u_d = self.cam_workspace_distance
            u_x, u_y, u_z = self.workspace_center
            b_z = -u_d * T_BC[2, 2] - T_BC[2, 3] + u_z
            b_x = (np.sin(theta) * (u_d * T_BC[1, 2] + T_BC[1, 3]) -
                   np.cos(theta) * (u_d * T_BC[0, 2] + T_BC[0, 3]) + u_x).item()
            b_y = (-np.cos(theta) * (u_d * T_BC[1, 2] + T_BC[1, 3]) -
                   np.sin(theta) * (u_d * T_BC[0, 2] + T_BC[0, 3]) + u_y).item()

            # add in small random noise
            b_x, b_y, b_z = np.array([b_x, b_y, b_z]) + self.np_random.uniform(
                low=-np.array(self.base_random_lim[0]), high=np.array(self.base_random_lim[0]), size=3)
            rb_x, rb_y, rb_z = np.array([0, 0, theta]) + self.np_random.uniform(
                low=-np.array(self.base_random_lim[1]), high=np.array(self.base_random_lim[1]), size=3)

            b_pose = ((b_x, b_y, b_z), convert_quat_tf_to_pb(tf3d.euler.euler2quat(rb_x, rb_y, rb_z)))
            return b_pose
        else:
            if self.base_random_lim is None:
                return self.robot_base_pose_pb
            else:
                robot_base_rot_euler = tf3d.euler.quat2euler(convert_quat_pb_to_tf(self.robot_base_pose_pb[1]))
                base_trans = self.np_random.uniform(
                    low=np.array(self.robot_base_pose_pb[0]) - np.array(self.base_random_lim[0]) / 2,
                    high=np.array(self.robot_base_pose_pb[0]) + np.array(self.base_random_lim[0]) / 2,
                    size=3
                )
                base_rot = self.np_random.uniform(
                    low=np.array(robot_base_rot_euler) - np.array(self.base_random_lim[1]) / 2,
                    high=np.array(robot_base_rot_euler) + np.array(self.base_random_lim[1]) / 2,
                    size=3
                )
                base_rot_quat = convert_quat_tf_to_pb(tf3d.euler.euler2quat(*base_rot, axes='rxyz'))
                return base_trans, base_rot_quat

    def reset(self, hard_reset=False, mb_base_angle=None):

        self.initial_move_made = False
        robot_reset_success = False
        while not robot_reset_success:
            self.first_gripper_close = False
            self.last_gripper_close = False

            if self.init_gripper_random_lim is not None:
                init_gripper_pose = self.get_random_gripper_pose()
            else:
                init_gripper_pose = self.init_gripper_pose

            if hard_reset:  # resets pb client, reloads urdf...currently unused, probably needs to be fixed
                self.block_ids = []
                # self.table = None
                self.goal_id = None
                # self.insertion_box = None
                # self.insertion_rod = None
                self._pb_client.resetSimulation()  # for some reason, not reclaiming memory?!?!
                self._pb_client.setGravity(0, 0, -10)
                if self.render_ground_plane:
                    self._pb_client.loadURDF(pybullet_data.getDataPath() + "/plane.urdf",
                                    [0.000000, 0.000000, 0.000000],
                                    [0.000000, 0.000000, 0.000000, 1.000000])
                robot_reset_success = self.gripper.reset(
                    base_pose=self.robot_base_pose_pb, init_gripper_pose=init_gripper_pose,
                    initially_hard_set_robot_up=True, reload_urdf=True)

            base_pose = self.get_reset_base_pose(mb_base_angle=mb_base_angle)
            robot_reset_success = self.gripper.reset(
                base_pose=base_pose, init_gripper_pose=init_gripper_pose, initially_hard_set_robot_up=True)

        if self.table is None:
            self.table = load_table(pb_client=self._pb_client,
                                    urdf_root=self.object_urdf_root,
                                    mark_on_table=self.mark_on_table)

        if self.tray_type is not None and self.tray is None:
            tray_pos = list(copy.deepcopy(self.workspace_center))
            tray_pos[0] -= .24  # tray from freecad has 10cm x 10cm base plus 3cm wide slope on each side, scaled x3
            tray_pos[1] -= .24
            tray_pos[2] -= .07
            if self.tray_type == 'normal':
                self.tray = self._pb_client.loadURDF(self.object_urdf_root + "/tray/tray.urdf", tray_pos,
                                                     [0, 0, 0, 1], globalScaling=3.0, useFixedBase=True)
            elif self.tray_type == '2_cube_insert':
                self.tray = self._pb_client.loadURDF(self.object_urdf_root + "/tray/tray_with_inserts.urdf", tray_pos,
                                                     [0, 0, 0, 1], useFixedBase=True)
            else:
                raise NotImplementedError("No implementation for tray type %s" % self.tray_type)

        # workspace center, manipulator body id, base link, tool link for debugging
        # add_pb_frame_marker_by_pose(self._pb_client, self.workspace_center, [0, 0, 0, 1])
        # add_pb_frame_marker(self._pb_client, self.gripper.body_id, 0)
        # add_pb_frame_marker(self._pb_client, self.gripper.body_id, self.rc['base_link_index'] + 1, line_length=1,
        #                     line_width=10)
        # add_pb_frame_marker(self._pb_client, self.gripper.body_id, self.rc['tool_link_index'], line_length=.5,
        #                     line_width=10)

        # plot workspace corners for debugging
        # wc = list(copy.deepcopy(self.workspace_center))
        # corners = [copy.deepcopy(wc) for i in range(4)]
        # corners[0][0] += .1; corners[0][1] += .1
        # corners[1][0] -= .1; corners[1][1] += .1
        # corners[2][0] += .1; corners[2][1] -= .1
        # corners[3][0] -= .1; corners[3][1] -= .1
        # for c in corners:
        #     add_pb_frame_marker_by_pose(self._pb_client, c, [0, 0, 0, 1])

        if 'door' in self.task:  # deprecated
            if self.door is not None:
                self._pb_client.removeBody(self.door)
            self.door = self._pb_client.loadURDF(self.object_urdf_root + "/door.urdf", [0.8, -.05, .625],
                                                 [0, 0, -.707, .707], globalScaling=0.25)

            if self.door_back is not None:
                self._pb_client.removeBody(self.door_back)
            self.door_back = self._pb_client.loadURDF(self.object_urdf_root + "/door_back.urdf", [0.934, -.17275, .625],
                                                      [0, 0, -.707, .707], globalScaling=0.25)

        if 'insertion' in self.task:  # deprecated
            # note: insertion box may still cause a small memory leak
            if self.insertion_box is not None:
                self._pb_client.removeBody(self.insertion_box)
            if self.insertion_rod is not None:
                self._pb_client.removeBody(self.insertion_rod)
            box_pos = list(copy.deepcopy(self.workspace_center))
            box_pos[1] += 0.1
            self.insertion_box = self._pb_client.loadURDF(self.object_urdf_root +
                                                          "/insertion_box/insertion_block.urdf",
                                                         box_pos, [0, 0, 0, 1])
            if self.task == 'insertion':
                rod_z = 1.0
                rod_xy = [.85, -.15]
                rod_file = '/rod.urdf'
            elif 'pick_insertion' in self.task:
                rod_z = .73
                base_rod_xy_position = self.workspace_center[:2]
                rod_xy = []
                for i in range(2):
                    rod_xy.append(self.np_random.uniform(
                        low=base_rod_xy_position[i] + self.init_rod_pos[i] - self.rod_random_lim[i] / 2,
                        high=base_rod_xy_position[i] + self.init_rod_pos[i] + self.rod_random_lim[i] / 2,
                        size=1
                    ))
                rod_file = '/rod_long.urdf'
            self.insertion_rod = self._pb_client.loadURDF(self.object_urdf_root + rod_file,
                                                          [rod_xy[0], rod_xy[1], rod_z], [0, 0, 0, 1])
            rod_rot = convert_quat_tf_to_pb(tf3d.euler.euler2quat(.75 * np.pi, 0, 0))

            if self.task == 'insertion':
                for i in range(50):
                    self.gripper.manipulator.open_gripper()
                    self.gripper.manipulator.update()
                    self._pb_client.stepSimulation()

                self.insertion_rod_const = self._pb_client.createConstraint(self.gripper.body_id,
                                                 self.gripper.manipulator._tool_link_ind,
                                                 self.insertion_rod, -1, self._pb_client.JOINT_FIXED, [0, 0, 0],
                                                 [0, 0, 0], [0, 0, 0], rod_rot, [0, 0, 0, 1])
                self._pb_client.changeConstraint(self.insertion_rod_const, maxForce=10)

                for i in range(50):
                    self._pb_client.stepSimulation()

                for i in range(100):
                    self.gripper.manipulator.close_gripper()
                    self.gripper.manipulator.update()
                    self._pb_client.stepSimulation()

                self._pb_client.removeConstraint(self.insertion_rod_const)

        # new style object loading -- still needs a way to specify initial orientation of objects:
        if self.obj_init_pos is not None:
            for o in self._obj_ids:
                self._pb_client.removeBody(o)
            self._obj_ids = []
            wc = np.array(self.workspace_center)
            for r_lim, init_pos, rgba, name in \
                    zip(self.obj_random_lim, self.obj_init_pos, self.obj_rgba, self.obj_urdf_names):
                r_lim = np.array(r_lim)
                init_pos = np.array(init_pos)

                if np.any(r_lim):
                    if self.green_on_blue:  # for unstack-->stack env
                        if rgba == (0, 0, 1, 1) and name == 'cube_blue_small':  # just a single pos for both b and g
                            pos = self.np_random.uniform(low=wc + init_pos - r_lim / 2, high=wc + init_pos + r_lim / 2)
                        elif rgba == (0, 1, 0, 1) and name == 'cube_blue_small':
                            pos[2] = pos[2] + .06
                    else:
                        pos = self.np_random.uniform(low=wc + init_pos - r_lim/2, high=wc + init_pos + r_lim/2)

                    # all objects currently assumed to have flat bottom, so only randomize z rot

                    # minor bug: this actually converts from wxyz to zwxy, which is then interpreted by pybullet as xyzw,
                    # so the rotation ends up being xyzw = [0, val, val, 0], and specifically eul'sxyz' = [180, 0, val]
                    rot = q_convert(tf3d.euler.euler2quat(0, 0, self.np_random.uniform(low=0, high=2 * np.pi)))

                    # for cube, randomize one of six faces for cube to initialize on
                    # rot = q_convert(tf3d.euler.euler2quat(self.np_random.randint(low=-2, high=3) * 0.5 * np.pi,
                    #                                       self.np_random.randint(low=-2, high=3) * 0.5 * np.pi,
                    #                                       self.np_random.uniform(low=0, high=2 * np.pi)))

                    # this would allow randomizing in all rotational DOF, but this isn't a balanced distribution for rotations
                    # rot = q_convert(tf3d.euler.euler2quat(self.np_random.uniform(low=0, high=2 * np.pi),
                    #                                       self.np_random.uniform(low=0, high=2 * np.pi),
                    #                                       self.np_random.uniform(low=0, high=2 * np.pi)))

                else:
                    pos = wc + init_pos
                    rot = [0, 0, 0, 1]

                self._obj_ids.append(self._pb_client.loadURDF(self.object_urdf_root + '/' + name + '.urdf', pos, rot))
                self.update_body_visual(self._obj_ids[-1], *rgba)

        else:
            # setup for randomizing positions
            if self.block_colors is None:
                colors = ('blue', 'green', 'red', 'white', 'black')
            else:
                colors = self.block_colors
            base_cube_xy_position = self.workspace_center[:2]

            block_pos = [[], []]
            block_rot = []
            for b in range(len(self.init_block_pos)):
                for i in range(2):
                    block_pos[i].append(
                        self.np_random.uniform(
                            low=base_cube_xy_position[i] + self.init_block_pos[b][i] - \
                                self.block_random_lim[b][i]/2,
                            high=base_cube_xy_position[i] + self.init_block_pos[b][i] + \
                                 self.block_random_lim[b][i]/2,
                            size=1
                        )
                    )

                    # for debugging limits
                    # if i == 0:
                    #     # block_pos[i].append(base_cube_xy_position[i] + self.init_block_pos[b][i] -
                    #     #                     self.block_random_lim[b][i]/2)
                    #     block_pos[i].append(base_cube_xy_position[i] + self.init_block_pos[b][i])
                    # else:
                    #     # block_pos[i].append(base_cube_xy_position[i] + self.init_block_pos[b][i] -
                    #     #                     self.block_random_lim[b][i] / 2)
                    #     block_pos[i].append(base_cube_xy_position[i] + self.init_block_pos[b][i])

                    # also want to randomize block yaw, 0 to pi/2
                    if self.block_style == 'long':
                        yaw_lim = np.pi
                    else:
                        yaw_lim = np.pi/2
                    block_rot.append(convert_quat_tf_to_pb(tf3d.euler.euler2quat(
                        0, 0, self.np_random.uniform(low=0, high=yaw_lim, size=1)
                    )))
            xy_pos = tuple(zip(block_pos[0], block_pos[1]))

        # load plane and blocks and plate -- deletes existing blocks and plate
        for body_id in self.block_ids:
            self._pb_client.removeBody(body_id)
        if self.goal_id is not None:
            self._pb_client.removeBody(self.goal_id)
        if self.goal2_id is not None:
            self._pb_client.removeBody(self.goal2_id)
        self.block_ids = []
        self.goal_id = None

        if self.obj_init_pos is None:  # if obj_init_pos is None, old-style deprecated objects
            if self.block_style == 'cylinder':
                i = 0
                self.block_ids.append(self._pb_client.loadURDF(self.object_urdf_root + "/cylinder_upright.urdf",
                                                               [xy_pos[i][0], xy_pos[i][1], .7], block_rot[i]))
            else:
                for i in range(len(self.init_block_pos)):
                    if self.block_style != '' and self.block_style != 'cube':
                        block_string_suffix = '_' + self.block_style
                    else:
                        block_string_suffix = ''

                    if self.task == 'pushing_xy' and i == 1:
                        block_string_suffix = ''

                    color = colors[i % len(colors)]

                    block_height = .7
                    if self.task == 'reaching_xy':
                        block_height = .655
                    self.block_ids.append(self._pb_client.loadURDF(self.object_urdf_root + "/cube_" + color +
                                                                   block_string_suffix + ".urdf",
                                                                   [xy_pos[i][0], xy_pos[i][1], block_height],
                                                                   block_rot[i]))

        if self.goal_type is not None:  # deprecated
            if self.task == 'pick_and_place_xy' or self.task == 'pick_and_place_xy_vert'\
                    or self.goal_type == 'coaster':
                plate_z = .6247
                self.goal_id = self._pb_client.loadURDF(self.object_urdf_root + '/coaster.urdf',
                                                        [base_cube_xy_position[0] + self.goal_pos[0],
                                                          base_cube_xy_position[1] + self.goal_pos[1],
                                                          plate_z],
                                                        [0, 0, 0, 1])
            # goal is a non-collision sphere in the air
            elif 'air' in self.goal_type:
                plate_z = .78
                self.goal_id = self._pb_client.loadURDF(self.object_urdf_root + '/sphere_no_col_no_mass.urdf',
                                                        [base_cube_xy_position[0] + self.goal_pos[0],
                                                          base_cube_xy_position[1] + self.goal_pos[1],
                                                          plate_z], [0, 0, 0, 1])

            else:
                plate_z = .7
                if self.task == 'pick_and_place_6dof':
                    plate_str = 'plate_small.urdf'
                elif self.task == 'sort_2' or self.task == 'sort_3':
                    plate_str = 'plate_blue.urdf'
                    plate2_str = 'plate_green.urdf'
                    plate2_pos = [0., .15]
                    self.goal2_id = self._pb_client.loadURDF(self.object_urdf_root + "/dinnerware/" + plate2_str,
                                                             [base_cube_xy_position[0] + plate2_pos[0],
                                                              base_cube_xy_position[1] + plate2_pos[1],
                                                              plate_z],
                                                             [0, 0, 0, 1])

                else:
                    plate_str = 'plate_wide.urdf'
                self.goal_id = self._pb_client.loadURDF(self.object_urdf_root + "/dinnerware/" + plate_str,
                                                        [base_cube_xy_position[0] + self.goal_pos[0],
                                      base_cube_xy_position[1] + self.goal_pos[1],
                                      plate_z],
                                                        [0, 0, 0, 1])

        # set main (debug) camera position to supplied view
        # self._pb_client.resetDebugVisualizerCamera(cameraTargetPosition=self.debug_cam_params[:3],
        #         cameraDistance=self.debug_cam_params[3], cameraPitch=self.debug_cam_params[4],
        #         cameraYaw=self.debug_cam_params[5])

        self.gripper.manipulator.update()  # added to fix an occasional reset bug where arm would drift off at start
        for i in range(50):  # allow blocks, etc. to settle
            self._pb_client.stepSimulation()


        # For generating images
        # ----------------------------------------------------------------------------------------
        generate_pretty_images = False
        if generate_pretty_images:
            pbc = self._pb_client
            man = self.gripper.manipulator
            cur_base_pose = man.get_link_pose(0)
            cur_base_pose = [cur_base_pose[:3], cur_base_pose[3:]]
            cur_base_pose_mat = trans_quat_to_mat(*cur_base_pose)
            # show cam pose in sim
            cam_pose_T = np.dot(cur_base_pose_mat, self.workspace_cam.frame_rel_tf)

            # move cam back so it looks more "attached" to robot
            cam_pose_adjustment = np.eye(4)
            cam_pose_adjustment[:3, 3] = [0, 0, -0.275]
            cam_pose_T = np.dot(cam_pose_T, cam_pose_adjustment)

            cam_pose_q = convert_quat_tf_to_pb(tf3d.quaternions.mat2quat(cam_pose_T[:3, :3]))
            if self.kinect is not None:
                pbc.removeBody(self.kinect)
            self.kinect = pbc.loadURDF(self.object_urdf_root + "/kinect.urdf",
                                       cam_pose_T[:3, 3],
                                       cam_pose_q, useFixedBase=True)

            # add frame
            frame_width = 25
            frame_length = .2
            cam_orient = convert_quat_tf_to_pb(tf3d.quaternions.mat2quat(cam_pose_T[:3, :3]))
            if not hasattr(self, 'kinect_frame_marker_id'): self.kinect_frame_marker_id = []
            self.kinect_frame_marker_id = add_pb_frame_marker_by_pose(
                pbc, cam_pose_T[:3, 3], cam_orient, self.kinect_frame_marker_id, frame_length, frame_width)

            # inertial frame
            # if not hasattr(self, 'inertial_frame_marker_id'): self.inertial_frame_marker_id = []
            # self.inertial_frame_marker_id = add_pb_frame_marker_by_pose(
            #     pbc, [0, -.65, .01],  [0, 0, 0, 1], self.inertial_frame_marker_id, .2, frame_width)

            # workspace frame
            if not hasattr(self, 'workspace_frame_marker_id'): self.workspace_frame_marker_id = []
            self.workspace_frame_marker_id = add_pb_frame_marker_by_pose(
                pbc, self.workspace_center, [0, 0, 0, 1], self.workspace_frame_marker_id,
                frame_length, frame_width)

            # robot frame
            if not hasattr(self, 'robot_frame_marker_id'): self.robot_frame_marker_id = []
            base_pose = man.get_link_pose(9)
            base_pose[2] += .01

            self.robot_frame_marker_id = add_pb_frame_marker_by_pose(
                pbc, base_pose[:3], base_pose[3:], self.robot_frame_marker_id, frame_length, frame_width)

            # remove explorer
            self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_GUI, 0)

            draw_multirobot_fig = True
            shuffle_base_angles = True
            if draw_multirobot_fig:
                # make robot (and kinect) transparent
                alpha = 0.2
                color = .5
                self.update_body_visual(self.gripper.body_id, color, color, color, alpha)
                self.update_body_visual(self.kinect, color, color, color, alpha)

                # make EVERYTHING transparent
                # pbc.removeBody(self.ground_plane)
                # for obj in [self.table, self.door]:
                #     for i in range(pbc.getNumJoints(obj)):
                #         pbc.changeVisualShape(obj, i, rgbaColor=rgba)

                # make other robots to show all poses
                num_extra_robots = 6  # for some reason can't render more than 6 (7 total)??? pybullet bug
                # take all but first angle, since first angle is the already generated one
                multi_image_base_angles = np.linspace(np.pi / 16, -3 * np.pi / 16, num=num_extra_robots + 1)[:-1]

                if shuffle_base_angles:
                    rng = np.random.default_rng(seed=0)
                    rng.shuffle(multi_image_base_angles)
                # import ipdb; ipdb.set_trace()  # stop between each to allow screenshots for video

                for i in range(num_extra_robots):
                    new_robot = ManipulatorWrapper(
                        pb_client=self._pb_client,
                        robot_config=self.rc,
                        control_method=self.gripper.control_method,
                        gripper_control_method='p', timestep=.01, base_pos=self.robot_base_pose_pb[0],
                        base_rot=self.robot_base_pose_pb[1], self_collision=False,
                        action_ref_frame=self.gripper.action_ref_frame,
                        valid_r_dof=self.gripper.valid_r_dof, valid_t_dof=self.gripper.valid_t_dof,
                        three_pts_ee_distance=self.gripper.three_pts_ee_distance, gripper_default_close=self.gripper.gripper_default_close,
                        max_gripper_vel=self.gripper.manipulator.max_gripper_vel, gripper_force=self.gripper.manipulator.gripper_force
                    )

                    # make new base poses for each new robot
                    new_gripper_pose = self.get_random_gripper_pose()
                    # new_b_pose = self.get_reset_base_pose()
                    new_b_pose = self.get_reset_base_pose(mb_base_angle=multi_image_base_angles[i])
                    robot_reset_success = False
                    while not robot_reset_success:
                        robot_reset_success = new_robot.reset(base_pose=new_b_pose,
                                                              init_gripper_pose=new_gripper_pose,
                                                              initially_hard_set_robot_up=True)
                    self.update_body_visual(new_robot.body_id, color, color, color, alpha)

                    # this will only work for a single reset
                    base_pose = new_robot.manipulator.get_link_pose(9)
                    base_pose[2] += .01
                    new_robot_marker_id = add_pb_frame_marker_by_pose(
                        pbc, base_pose[:3], base_pose[3:], [], frame_length, frame_width)

                    # new kinect for each robot
                    cur_base_pose = new_robot.manipulator.get_link_pose(0)
                    cur_base_pose = [cur_base_pose[:3], cur_base_pose[3:]]
                    cur_base_pose_mat = trans_quat_to_mat(*cur_base_pose)
                    # show cam pose in sim
                    cam_pose_T = np.dot(cur_base_pose_mat, self.workspace_cam.frame_rel_tf)

                    # move cam back so it looks more "attached" to robot
                    cam_pose_adjustment = np.eye(4)
                    cam_pose_adjustment[:3, 3] = [0, 0, -0.275]
                    cam_pose_T = np.dot(cam_pose_T, cam_pose_adjustment)

                    cam_pose_q = convert_quat_tf_to_pb(tf3d.quaternions.mat2quat(cam_pose_T[:3, :3]))
                    new_kinect = pbc.loadURDF(self.object_urdf_root + "/kinect.urdf",
                                               cam_pose_T[:3, 3],
                                               cam_pose_q, useFixedBase=True)
                    self.update_body_visual(new_kinect, color, color, color, alpha)
                    new_kinect_frame_marker_id = add_pb_frame_marker_by_pose(
                        pbc, cam_pose_T[:3, 3], cam_orient, [], frame_length, frame_width)

                    # import ipdb; ipdb.set_trace()  # allow stopping to take individual screenshot of each for video

                import ipdb; ipdb.set_trace()
        # ----------------------------------------------------------------------------------------

        if len(self._obj_ids) > 0:
            self._prev_obj_vels = np.zeros([len(self._obj_ids), 6])
        else:
            self._prev_obj_vels = np.zeros([len(self.block_ids), 6])
        self._prev_step_count_for_acc = 0

        return self._prepare_obs_dict_new()

    def update_body_visual(self, body_id, r=.5, g=.5, b=.5, a=.2):
        """ Update the visual effects of every link of a body """
        pbc = self._pb_client
        rgba = (r, g, b, a)
        for i in range(-1, pbc.getNumJoints(body_id)):
            pbc.changeVisualShape(body_id, i, rgbaColor=rgba)

    def save_body_texture_ids(self, body_id):
        """ Save the textureUniqueIds for a particular body as a list. """
        pbc = self._pb_client
        data = pbc.getVisualShapeData(body_id, flags=pbc.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS)
        link_shape_text_rgba_ids = []  # 3 items per list item -- link id, shape id, rgba, texture id
        shape_id = -1
        cur_link_id = 0
        for vis_data in data:
            if vis_data[1] == cur_link_id:
                shape_id += 1
            else:
                cur_link_id += 1
                shape_id = 0
            link_shape_text_rgba_ids.append([vis_data[1], shape_id, vis_data[-2], vis_data[-1]])
        return link_shape_text_rgba_ids

    def update_body_visual_with_saved(self, body_id, link_shape_text_rgba_ids, use_rgba=False):
        """ Update the visual effects of every link of a body using a list of [link_id, shape_id, textureUniqueId]s """
        pbc = self._pb_client
        for link_shape_text_rgba_id in link_shape_text_rgba_ids:
            link_id, shape_id, rgba, text_id = link_shape_text_rgba_id
            if use_rgba:
                pbc.changeVisualShape(body_id, link_id, shapeIndex=shape_id, rgbaColor=rgba)
            else:
                pbc.changeVisualShape(body_id, link_id, shapeIndex=shape_id, textureUniqueId=text_id)

    def render(self, mode='human', depth_type='original', segment_mask=False):
        assert mode in PBEnv.RENDER_MODES, f"{mode} is not a valid render mode, must be one of {PBEnv.RENDER_MODES}"

        # cam pose override for choosing a new cam pose, comment out during regular operation ---------------
        # if not hasattr(self, "cam_pose_override"):
        #     self.cam_pose_override = list([list(self.robot_base_ws_cam_tf[0]), list(self.robot_base_ws_cam_tf[1])])
        # keys = self._pb_client.getKeyboardEvents()
        # if len(keys) > 0:
        #     for k in keys:
        #         speed = .05
        #         if k == ord('q'):
        #             self.cam_pose_override[0][0] += speed
        #         elif k == ord('a'):
        #             self.cam_pose_override[0][0] -= speed
        #         elif k == ord('w'):
        #             self.cam_pose_override[0][1] += speed
        #         elif k == ord('s'):
        #             self.cam_pose_override[0][1] -= speed
        #         elif k == ord('e'):
        #             self.cam_pose_override[0][2] += speed
        #         elif k == ord('d'):
        #             self.cam_pose_override[0][2] -= speed
        #         elif k == ord('r'):
        #             self.cam_pose_override[1][0] += speed
        #         elif k == ord('f'):
        #             self.cam_pose_override[1][0] -= speed
        #         elif k == ord('t'):
        #             self.cam_pose_override[1][1] += speed
        #         elif k == ord('g'):
        #             self.cam_pose_override[1][1] -= speed
        #         elif k == ord('y'):
        #             self.cam_pose_override[1][2] += speed
        #         elif k == ord('h'):
        #             self.cam_pose_override[1][2] -= speed
        #
        # new_cam_pose = np.eye(4)
        # new_cam_pose[:3, 3] = self.cam_pose_override[0]
        # new_cam_pose[:3, :3] = tf3d.euler.euler2mat(*self.cam_pose_override[1], 'sxyz')
        # self.workspace_cam.frame_rel_tf = new_cam_pose
        # self.workspace_cam_high_res.frame_rel_tf = new_cam_pose
        #
        # target_tf = np.eye(4)
        # target_tf[:3, 3] = np.dot(np.array(self.workspace_cam.forward_axis), self.workspace_cam.focus_dist)
        # self.workspace_cam.frame_rel_target_tf = np.dot(self.workspace_cam.frame_rel_tf, target_tf)
        # self.workspace_cam_high_res.frame_rel_target_tf = np.dot(self.workspace_cam.frame_rel_tf, target_tf)

        # print(self.cam_pose_override)

        # NOTE! for adding a new camera, print out the cam_pose and the cam_target_pose from EyeInHandCam.get_img,
        # and use those as the eye and target for a new camera

        # end of cam pose override--------------------------------------------------------------------------

        cur_pose = self.gripper.manipulator.get_link_pose(self.gripper.manipulator._tool_link_ind)
        cur_pose = [cur_pose[:3], cur_pose[3:]]
        cur_base_pose = self.gripper.manipulator.get_link_pose(0)
        cur_base_pose = [cur_base_pose[:3], cur_base_pose[3:]]

        if mode == 'human':
            cur_base_pose_mat = trans_quat_to_mat(*cur_base_pose)
            if segment_mask:
                cam_img, depth_img, seg_img = self.workspace_cam_high_res.get_img(cur_base_pose_mat, depth_type=depth_type,
                                                                         segment_mask=segment_mask)
                return cam_img, depth_img, seg_img
            else:
                cam_img, depth_img = self.workspace_cam_high_res.get_img(cur_base_pose_mat, depth_type=depth_type,
                                                                         segment_mask=segment_mask)
                return cam_img, depth_img

        elif mode == 'workspace':
            cur_base_pose_mat = trans_quat_to_mat(*cur_base_pose)
            if segment_mask:
                cam_img, depth_img, seg_img = self.workspace_cam.get_img(cur_base_pose_mat, depth_type=depth_type,
                                                                segment_mask=segment_mask)
            else:
                cam_img, depth_img = self.workspace_cam.get_img(cur_base_pose_mat, depth_type=depth_type,
                                                                         segment_mask=segment_mask)

            # uncomment out the following to show cam pose in sim and/or add a kinect
            # -------------------------------------------------------------------------------------------------
            # cam_pose_T = np.dot(cur_base_pose_mat, self.workspace_cam.frame_rel_tf)
            # cam_pose_q = convert_quat_tf_to_pb(tf3d.quaternions.mat2quat(cam_pose_T[:3, :3]))
            # if self._render:
            #     self.workspace_cam_frame_marker_ids = add_pb_frame_marker_by_pose(
            #         self._pb_client, cam_pose_T[:3, 3], cam_pose_q, ids=self.workspace_cam_frame_marker_ids,
            #         line_width=10
            #     )

            # render a kinect object
            # self.kin_con = self._pb_client.createConstraint(self.kinect, -1, -1, -1, self._pb_client.JOINT_FIXED,
            #                                  [0, 0, 0], [0, 0, 0], cam_pose_T[:3, 3], [0, 0, 0, 1],
            #                                  cam_pose_q)
            # self._pb_client.changeConstraint(self.kin_con, cam_pose_T[:3, 3], cam_pose_q, maxForce=500.0)
            # -------------------------------------------------------------------------------------------------

            if segment_mask:
                return cam_img, depth_img, seg_img
            else:
                return cam_img, depth_img
        elif mode == 'eye-in-hand':
            cur_pose_mat = trans_quat_to_mat(cur_pose[0], cur_pose[1])
            cam_img, depth_img = self.wrist_cam.get_img(cur_pose_mat)
            return cam_img, depth_img
        elif mode == 'both':
            cur_pose_mat = trans_quat_to_mat(cur_pose[0], cur_pose[1])
            cam_wrist, depth_wrist = self.wrist_cam.get_img(cur_pose_mat)
            cur_base_pose_mat = trans_quat_to_mat(*cur_base_pose)
            cam_work, depth_work = self.workspace_cam.get_img(cur_base_pose_mat)
            return [cam_work, depth_work], [cam_wrist, depth_wrist]
        else:
            return getattr(self, mode).get_img()

    def close_env(self):
        self._pb_client.disconnect()
