import copy

import gym
from gym import spaces
import numpy as np
import transforms3d as tf3d
import tkinter as tk
import PIL.Image, PIL.ImageTk

from manipulator_learning.sim.envs.pb_env import PBEnv


PROPRIOCEPTIVE_STATES = ['pos', 'prev_pos', 'grip_pos', 'prev_grip_pos', 'timestep']


class ManipulatorEnv(gym.Env):
    def __init__(self,
                 task,  # specific task name as string, many tasks won't need this and then 'None' can be used
                 camera_in_state,
                 dense_reward,
                 grip_in_action,
                 poses_ref_frame='b',  # 'b' or 'w' for base or world
                 state_data=('pos', 'prev_pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'timestep',
                             'force_torque', 'vel', 'obj_vel', 'obj_rot_vel'),
                 num_prev_pos=5,
                 num_prev_grip_pos=2,  # excluding current grip pos
                 gap_between_prev_pos=.1,  # in seconds
                 max_real_time=5,  # in seconds
                 n_substeps=10,  # number of sim frames to execute action for.. all sim timesteps are .01s
                 action_multiplier=1.0,
                 image_width=160,
                 image_height=120,
                 success_causes_done=False,
                 failure_causes_done=False,
                 egl=True,
                 control_frame='t',  # 't', 'b', or 'w' for tool, base, or world
                 new_env_with_fixed_depth=False,  # see note in cameras.py for issue on bad depth values on old envs
                 valid_t_dofs=(1, 1, 1),
                 valid_r_dofs=(1, 1, 1),
                 config_dict=None,  # dictionary of arguments for pb_env
                 generate_spaces=False,
                 vel_ref_frame='t',  # 't', 'b', or 'w', tool, base or, world
                 grip_multiplier=1.0,
                 render_opengl_gui=False,  # render the pybullet gui with opengl...raises error if egl also True.
                 force_pb_direct=False,  # force pybullet Direct connection..overriden to False if above is True
                 objs_no_rot_no_vel=(),  # object indices with rotation pos and all vel removed from state
                 ):

        if config_dict is None:
            config_dict = dict()

        if egl and render_opengl_gui:
            raise ValueError("Can't set egl to True and pb_render to True at the same time. User must specifically"
                             "specify egl=False if they want pb_render=True, since egl and pb_render actually"
                             "render slightly differently.")

        if render_opengl_gui and force_pb_direct:
            print("WARNING! force_pb_direct and render_opengl_gui both set to True. Setting force_pb_direct to False.")
            force_pb_direct = False

        config_dict.update(dict(
            image_width=image_width,
            image_height=image_height,
            valid_trans_dof=valid_t_dofs,
            valid_rot_dof=valid_r_dofs,
            task=task,
            use_egl=egl,
            render_opengl_gui=render_opengl_gui,
            control_frame=control_frame,
            poses_ref_frame=poses_ref_frame,
            vel_ref_frame=vel_ref_frame,
            force_pb_direct=force_pb_direct
        ))
        self.env = PBEnv(**config_dict)

        self._control_type = config_dict['control_method']  # used by some old other stuff
        self.task = task
        self.camera_in_state = camera_in_state
        self.image_width = image_width
        self.image_heigh = image_height
        self.state_data = state_data
        self.dense_reward = dense_reward
        self.grip_in_action = grip_in_action
        self.on_screen_render = False
        self.success_causes_done = success_causes_done
        self.failure_causes_done = failure_causes_done
        self.real_t_per_ts = self.env._time_step * n_substeps
        self._max_episode_steps = int(max_real_time / self.real_t_per_ts)
        self.n_substeps = n_substeps
        self.action_multiplier = action_multiplier
        self.grip_multiplier = grip_multiplier
        self.ep_timesteps = 0
        self._return_arr = None
        self._return_obs = None
        self._new_env_with_fixed_depth = new_env_with_fixed_depth
        self.objs_no_rot_no_vel = objs_no_rot_no_vel
        self._cube_rot_fix = False

        # prev pose info
        self.gap_between_prev_pos = int(1 / self.env._time_step * gap_between_prev_pos / n_substeps)
        self.num_prev_pos=num_prev_pos
        self.prev_pos=None

        # grip feedback info
        self._grip_feedback_delay = 0
        self._prev_grip_feedback = None

        # grip pos info
        self._prev_grip_pos_delay = 0
        self._prev_grip_pos = None
        self.num_prev_grip_pos = num_prev_grip_pos

        self.img_rendered = False  # to prevent re-rendering images
        self.human_img_rendered = False
        self.human_rgb = None

        # for basic position limiting -- the Thing environments use this method, but newer Panda envs set it in the config_dict instead
        # self.pos_limits = [[.55, -.45, .64], [1.0, .06, 1.0]]
        self.pos_limits_visual = None
        self.pos_limits_visual_body = None
        self.show_pos_limits = False

        # img env state -- for getting state info from a full state env that would be available in an img env
        self.valid_t_dofs = self.env.gripper.valid_t_dof.nonzero()[0]
        self.valid_r_dofs = self.env.gripper.valid_r_dof.nonzero()[0]
        self.num_grip_fingers = len(self.env.gripper.manipulator.gripper_max)
        pos_shape = ('pos' in state_data) * (len(self.valid_t_dofs) + (len(self.valid_r_dofs) > 0) * 4)
        img_state_env_highest_ind = pos_shape + \
                                    int('prev_pos' in state_data) * self.num_prev_pos * pos_shape + \
                                    int('grip_pos' in state_data) * self.num_grip_fingers + \
                                    int('prev_grip_pos' in state_data) * self.num_grip_fingers * self.num_prev_grip_pos + \
                                    int('timestep' in state_data) + \
                                    int('force_torque' in state_data) * 6 + \
                                    int('vel' in state_data) * (len(self.valid_t_dofs) + len(self.valid_r_dofs))

        self._img_env_state_indices = slice(0, img_state_env_highest_ind)

        if generate_spaces:
            # generate action and observation spaces automatically
            num_actions = sum(valid_t_dofs) + sum(valid_r_dofs) + grip_in_action
            self.action_space = spaces.Box(-1, 1, (num_actions,), dtype=np.float32)
            if camera_in_state:
                num_obs = img_state_env_highest_ind
                self.observation_space = ({
                    'obs': spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32),
                    'img': spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
                    'depth': spaces.Box(0, 1, (image_height, image_width), dtype=np.float32)
                })
            else:
                num_obs = img_state_env_highest_ind + \
                          int('obj_pos' in state_data) * len(self.valid_t_dofs) * len(self.env.objs_in_state) + \
                          int('obj_rot' in state_data) * 4 * len(self.env.objs_in_state) + \
                          int('obj_rot_z' in state_data or 'obj_rot_z_sym' in state_data) * 2 * len(self.env.objs_in_state) + \
                          int('obj_rot_z_first_only' in state_data) * 2 + \
                          int(self.env.rel_pos_in_state is not None) * 3 * len(self.env.rel_pos_in_state) + \
                          int('obj_vel' in state_data) * len(self.valid_t_dofs) * len(self.env.objs_in_state) + \
                          int('obj_rot_vel' in state_data) * 3 * len(self.env.objs_in_state)
                self.observation_space = spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def set_max_episode_steps(self, n):
        self._max_episode_steps = n

    def seed(self, seed=None):
        ret_seed = self.env.seed(seed)
        self.np_random = self.env.np_random
        return ret_seed

    def step(self, action):
        action = self.action_multiplier * action
        
        # adjust action shape if coming from dof limited env
        # action should come in as (n,) shape array, where n is total number of valid dofs
        # if gripper in action, add one more to shape for gripper out, anything below 0 is open, anything above
        # zero is closed
        fixed_action = [np.zeros(3), np.zeros(3), False]
        valid_t_dofs = self.env.gripper.valid_t_dof.nonzero()[0]
        valid_r_dofs = self.env.gripper.valid_r_dof.nonzero()[0]
        valid_len = len(valid_t_dofs) + len(valid_r_dofs) + self.grip_in_action
        assert len(action) == valid_len, "len of action is %d, must be %d for this environment" % \
                                                  (len(action), valid_len)

        fixed_action[0][valid_t_dofs] = action[:len(valid_t_dofs)]
        fixed_action[1][valid_r_dofs] = action[len(valid_t_dofs):(len(valid_t_dofs) + len(valid_r_dofs))]
        if self.grip_in_action:
            if self.env.gripper_control_method == 'bool_p':
                if action[-1] < 0:  # no support for position based grasping yet
                    fixed_action[2] = False
                else:
                    fixed_action[2] = True
            elif self.env.gripper_control_method == 'dp':
                fixed_action[2] = action[-1] / self.action_multiplier * self.grip_multiplier
            else:
                raise NotImplementedError("Not implemented for gripper control method %s" %
                                          self.env.gripper_control_method)
        action = fixed_action

        # now action is a 2 or 3 tuple, depending on if gripper in action, with first 2 elements trans and rot vel
        limit_reached = False

        for i in range(self.n_substeps):
            # obs_dict, reward, done, _ = self.env.step(action, self.pos_limits) # pos_limits in this argument is deprecated
            obs_dict, reward, done, _ = self.env.step(action)
        self.img_rendered = False
        self.human_img_rendered = False

        return_arr, return_obs = self._get_obs_from_obs_dict(obs_dict)

        reward, done_success, done_failure = self._calculate_reward_and_done(self.dense_reward, limit_reached)

        # panda gripper can move fingers to extremely unrealistic positions -- if this happens, trigger failure
        if self.grip_in_action:
            if any([fing_pos > 3.0 for fing_pos in return_obs['grip_pos']]):
                done_failure = True
                done_success = False

        return_obs['done_success'] = done_success
        return_obs['done_failure'] = done_failure

        self.ep_timesteps += 1
        done = False
        if self.ep_timesteps >= self._max_episode_steps:
            done = True
        if return_obs['done_success'] and self.success_causes_done:
            done = True
        if return_obs['done_failure'] and self.failure_causes_done:
            done = True

        return return_arr, reward, done, return_obs

    def get_img_obs(self):
        # allows user to manually grab an img-based obs dictionary from a state based env,
        # meaning it will be a dict with img, depth, and obs, and obs will contain everything but object based info
        img, depth = self.render('rgb_array')
        obs = self._return_arr[self._img_env_state_indices]
        return dict(img=img, depth=depth, obs=obs)

    def _get_obs_from_obs_dict(self, obs_dict):
        # kinematics state info always included
        return_obs = dict()
        info_dict = dict()  # extra state data included for debugging/aux rewards, but not officially in state of env
        img_env_obs = []  # non-img based envs will also include an obs in info that includes proprioceptive data only
        valid_t_dofs = self.env.gripper.valid_t_dof.nonzero()[0]
        valid_r_dofs = self.env.gripper.valid_r_dof.nonzero()[0]
        valid_t_and_r_dofs = np.concatenate((valid_t_dofs, valid_r_dofs))
        if 'pos' in self.state_data:
            if len(valid_r_dofs) > 0:
                if obs_dict['actual']['orient'][-1] < 0:
                    obs_dict['actual']['orient'] = -obs_dict['actual']['orient']
                return_obs['pos'] = np.concatenate((obs_dict['actual']['pos'][valid_t_dofs],
                                                    obs_dict['actual']['orient']))
            else:
                return_obs['pos'] = obs_dict['actual']['pos'][valid_t_dofs]
        if 'vel' in self.state_data:
            return_obs['vel'] = obs_dict['actual']['vel'][self.env.vel_ref_frame]
            return_obs['vel'] = return_obs['vel'][valid_t_and_r_dofs]
        if 'prev_pos' in self.state_data:
            if self.prev_pos is None:
                self.prev_pos = np.tile(return_obs['pos'], (self.num_prev_pos * self.gap_between_prev_pos, 1))
            self.prev_pos = np.roll(self.prev_pos, -1, axis=0)
            self.prev_pos[-1] = return_obs['pos']
            return_obs['prev_pos'] = self.prev_pos[np.array([range(0, self.num_prev_pos * self.gap_between_prev_pos,
                                                                   self.gap_between_prev_pos)])]

        if 'goal_pos' in self.state_data or 'goal_rot' in self.state_data:  # assuming goal is plate
            goal_pos, goal_rot = self.env._pb_client.getBasePositionAndOrientation(self.env.goal_id)
            if 'goal_pos' in self.state_data:
                return_obs['goal_pos'] = np.array(goal_pos)[valid_t_dofs]
            if 'goal_rot' in self.state_data:
                return_obs['goal_rot'] = np.array(goal_rot)

        # if 'grip_pos' in self.state_data or 'prev_grip_pos' in self.state_data:
        # gripper_max contains [upper_limit, lower_limit], so scale this to be between 0 and 1
        if self.grip_in_action:
            gm = self.env.rc['gripper_max']
            cur_grip_pos = (obs_dict['actual']['grip'] - gm[1]) / gm[0]
            info_dict['grip_pos'] = cur_grip_pos

        if 'grip_pos' in self.state_data:
            return_obs['grip_pos'] = cur_grip_pos
        if 'prev_grip_pos' in self.state_data:
            if self._prev_grip_pos is None:
                self._prev_grip_pos = np.tile(cur_grip_pos, (self.num_prev_grip_pos + 1, 1))
            self._prev_grip_pos = np.roll(self._prev_grip_pos, -1, axis=0)
            self._prev_grip_pos[-1] = cur_grip_pos
            return_obs['prev_grip_pos'] = np.array(self._prev_grip_pos[:-1])
        if 'obj_pos' in self.state_data and 'obj_rot' in self.state_data:
            quats = obs_dict['block_poses']['orient']
            quats_fixed = []

            for q in quats:
                if self._cube_rot_fix:
                    # check current rot as ax angle, if the angle mag is >45 degrees and <135 deg, rotate
                    # the representation 90 degrees in the OPPOSITE direction along the same axis, if it's
                    # greater than 135, rotate it 180 degrees in the opposite direction
                    # all of this is only for x or y axis though, since z axis rotation is fine

                    # working around an existing bug in pb_env, where all blocks reset to (np.pi, 0, ran_z)
                    wxyz_q = np.array([q[3], q[0], q[1], q[2]])
                    eul_static = list(tf3d.euler.quat2euler(wxyz_q, 'sxyz'))
                    eul_static_orig = copy.deepcopy(eul_static)
                    # remove z component since full rotation is okay there
                    eul_static[2] = 0
                    q_reset_minus_z = np.array([0, 1, 0, 0])  # 180 about x due to bug
                    q_curr_minus_z = tf3d.euler.euler2quat(*eul_static, 'sxyz')
                    q_diff = tf3d.quaternions.qmult(tf3d.quaternions.qinverse(q_reset_minus_z), q_curr_minus_z)
                    ax, ang = tf3d.quaternions.quat2axangle(q_diff)

                    if 1.75 * np.pi > ang >= .25 * np.pi:
                        if 0.75 * np.pi > ang >= 0.25 * np.pi:
                            q_mod = tf3d.quaternions.axangle2quat(ax, -1 * .5 * np.pi)
                        elif 1.25 * np.pi > ang >= 0.75 * np.pi:
                            q_mod = tf3d.quaternions.axangle2quat(ax, -1 * np.pi)
                        elif 1.75 * np.pi > ang >= 1.25 * np.pi:
                            q_mod = tf3d.quaternions.axangle2quat(ax, -1 * 1.5 * np.pi)

                        new_q_minus_z = tf3d.quaternions.qmult(q_curr_minus_z, q_mod)
                        new_eul = list(tf3d.euler.quat2euler(new_q_minus_z))
                        new_eul[2] = eul_static_orig[2]
                        new_q = tf3d.euler.euler2quat(*new_eul, 'sxyz')
                        q = np.array([new_q[1], new_q[2], new_q[3], new_q[0]])

                if q[-1] < 0:
                    quats_fixed.append(-q)
                else:
                    quats_fixed.append(q)
            return_obs['obj_pos'] = np.column_stack((obs_dict['block_poses']['pos'][:, valid_t_dofs],
                                               np.array(quats_fixed))).flatten()

            # remove obj rot if included in objs_no_rot_no_vel
            ind_to_remove = []
            for obj_ind in self.objs_no_rot_no_vel:
                ind_to_remove.extend(list(range(7 * obj_ind + 3, 7 * obj_ind + 7)))
            return_obs['obj_pos'] = np.delete(return_obs['obj_pos'], ind_to_remove)

        elif 'obj_pos' in self.state_data and ('obj_rot_z' in self.state_data or
                                               'obj_rot_z_first_only' in self.state_data or
                                               'obj_rot_z_sym' in self.state_data):
            # Note: obj_rot_z is for objects that only have 90 degrees of variation in rotation, e.g. cubes
            #       obj_rot_z_sym is for blocks that have 180 degrees of variation in rotation, e.g. square prisms
            quats = obs_dict['block_poses']['orient']
            z_rots = []
            for q in quats:
                wxyz_q = np.array([q[3], q[0], q[1], q[2]])
                yaw = tf3d.euler.quat2euler(wxyz_q)[2]
                if 'obj_rot_z_sym' in self.state_data:
                    z_rots.append([np.cos(2 * yaw), np.sin(2 * yaw)])
                else:
                    z_rots.append([np.cos(4 * yaw), np.sin(4 * yaw)])
                if 'obj_rot_z_first_only' in self.state_data: break

            if 'obj_rot_z_first_only' in self.state_data:
                return_obs['obj_pos'] = np.concatenate([obs_dict['block_poses']['pos'][:, valid_t_dofs].flatten(),
                                                        np.array(z_rots).squeeze()])
            else:
                return_obs['obj_pos'] = np.hstack([obs_dict['block_poses']['pos'][:, valid_t_dofs],
                                                   np.array(z_rots)]).flatten()

            # remove obj rot if included in objs_no_rot_no_vel
            ind_to_remove = []
            for obj_ind in self.objs_no_rot_no_vel:
                if 'obj_rot_z_first_only' in self.state_data and obj_ind > 0:
                    continue
                else:
                    ind_to_remove.extend(list(range(5 * obj_ind + 3, 5 * obj_ind + 5)))
            return_obs['obj_pos'] = np.delete(return_obs['obj_pos'], ind_to_remove)

        elif 'obj_pos' in self.state_data:
            return_obs['obj_pos'] = np.hstack((obs_dict['block_poses']['pos'][:, valid_t_dofs]))

        if 'obj_vel' in self.state_data or 'obj_rot_vel' in self.state_data:
            obj_vel_r_dofs = [3, 4, 5] if 'obj_rot_vel' in self.state_data else []
            obj_vel_dof = np.concatenate([valid_t_dofs, obj_vel_r_dofs])
            return_obs['obj_vel'] = np.hstack((obs_dict['block_poses']['vel'][:, obj_vel_dof]))

            # remove obj vel if included in objs_no_rot_no_vel
            ind_to_remove = []
            for obj_ind in self.objs_no_rot_no_vel:
                ind_to_remove.extend(list(range(6 * obj_ind, 6 * obj_ind + 6)))
            return_obs['obj_vel'] = np.delete(return_obs['obj_vel'], ind_to_remove)

        if 'obj_pos' in self.state_data and self.env.rel_pos_in_state is not None:
            return_obs['obj_rel_pos'] = np.hstack(obs_dict['block_poses']['rel_pos'])

        if 'force_torque' in self.state_data:
            # rescale and clip for learning using 100N and 10Nm
            np.set_printoptions(suppress=True, precision=3)
            ft_raw = self.env.gripper.manipulator.get_ee_ft()
            ft_env = np.zeros(6)
            ft_env[:3] = np.tanh(ft_raw[:3] / 100)
            ft_env[3:] = np.tanh(ft_raw[3:] / 10)
            return_obs['force_torque'] = ft_env

        # for rewards that need it
        if len(self.env.block_ids) > 0 and 'door' not in self.task:
            info_dict['obj_pos_world'] = self.env._pb_client.getBasePositionAndOrientation(self.env.block_ids[0])
        elif 'door' in self.task:
            info_dict['obj_pos_world'] = self.env._pb_client.getLinkState(self.env.door, linkIndex=1)[:2]

        # pos limit info, potentially useful for rewards
        limit_thresh = .005

        # accomodate both old and new style pos limits
        # pos_limits = self.env.gripper.pos_limits if self.env.gripper.pos_limits is not None else self.pos_limits
        pos_limits = self.env.gripper.pos_limits
        pos = return_obs['pos'][:3]  # if we have rot in return_obs['pos'] as well

        info_dict['at_limits'] = np.concatenate([
            [pos < (np.array(pos_limits[0]) + limit_thresh)],
            [pos > (np.array(pos_limits[1]) - limit_thresh)]
        ])

        # obj acceleration, useful for rewards
        info_dict['obj_acc'] = obs_dict['block_poses']['acc'].flatten()

        # extra info, if defined
        if hasattr(self, "_extra_info_dict"):
            info_dict.update(self._extra_info_dict)

        # moving base info
        info_dict['base_theta'] = self.env._base_theta_rel_to_fixed

        # if 'grip_feedback' in self.state_data:
        # always include grip_feedback in info
        # gripper links are 18 - 21, 18-19 for left, 20-21 for right, 19/21 for tips
        if self._prev_grip_feedback is None:
            self._prev_grip_feedback = np.ones([self._grip_feedback_delay + 1]) * -1
        self._prev_grip_feedback = np.roll(self._prev_grip_feedback, -1, axis=0)
        bodies = []

        if 'insertion' in self.task:
            contact_id = self.env.insertion_rod
        elif 'door' in self.task:
            contact_id = self.env.door
        elif len(self.env.block_ids) > 0:
            contact_id = self.env.block_ids[0]
        elif len(self.env._obj_ids) > 0:
            contact_id = self.env._obj_ids[0]
        else:
            raise NotImplementedError("Getting contact points not set up when there are 0 blocks, no door, or not insertion")
        for c in self.env._pb_client.getContactPoints(contact_id, self.env.gripper.manipulator._arm[0]):
            bodies.append(c[4])
        if (18 in bodies or 19 in bodies) and (20 in bodies or 21 in bodies):
            self._prev_grip_feedback[-1] = 1
        else:
            self._prev_grip_feedback[-1] = -1
        info_dict['grip_feedback'] = np.array([self._prev_grip_feedback[0]])
        if 'grip_feedback' in self.state_data:
            return_obs['grip_feedback'] = np.array([self._prev_grip_feedback[0]])

        if 'timestep' in self.state_data:
            # rescale timestep to be between -1 and 1
            return_obs['timestep'] = (self.ep_timesteps / self._max_episode_steps - .5) * 2

        return_arr = []
        for k in return_obs.keys():
            return_arr.append(return_obs[k].flatten())
            if k in PROPRIOCEPTIVE_STATES:
                img_env_obs.append(return_obs[k].flatten())
        return_arr = np.concatenate(return_arr)

        # add info dict to return_obs dict after return_arr has already been generated
        info_dict['img_env_state'] = np.concatenate(img_env_obs)
        for k in info_dict:
            return_obs[k] = info_dict[k]

        if self.camera_in_state:
            rgb, depth = self.render('rgb_array')
            return_obs['img'] = rgb
            return_obs['depth'] = depth
            return_arr = dict(
                obs=return_arr,
                img=rgb,
                depth=depth
            )

        self._return_arr = return_arr
        self._return_obs = return_obs
        return return_arr, return_obs

    def _calculate_reward_and_done(self, dense_reward, limit_reached):
        # this should be overwritten by child classes
        return 0, False, False

    def _tk_esc_pressed(self, event):
        self.window.destroy()

    def render(self, mode='human'):
        if not self.img_rendered:  # ensures only one render per step
            if self._new_env_with_fixed_depth:
                self.rgb, self.depth = self.env.render('workspace', depth_type='fixed')
            else:
                self.rgb, self.depth = self.env.render('workspace')
            self.rgb = self.rgb[:, :, :3]
            self.img_rendered = True
        if mode == 'human':
            # rerender if too small
            if not self.human_img_rendered:
                if self.image_width < 160:
                    self.human_rgb, _ = self.env.render('human')
                    self.human_rgb = self.human_rgb[:, :, :3]
                else:
                    self.human_rgb = self.rgb
                self.human_img_rendered = True
            if not self.on_screen_render:
                display_width = self.human_rgb.shape[1] * 6
                display_height = self.human_rgb.shape[0] * 6

                self.window = tk.Tk()
                self.window.title("Manipulator Renderer")
                self.window_text = tk.Label(text="Timesteps remaining: ????", fg='black', bg='lightgrey', height=1,
                                            anchor=tk.E, font=("Arial", 20, "bold"))
                self.window_text.pack(fill="x", side=tk.BOTTOM)
                self.canvas = tk.Canvas(self.window, width=display_width, height=display_height)
                self.canvas.pack(fill="both", expand=True)
                self.window.bind("<KeyRelease-Escape>", self._tk_esc_pressed)

                self.window.update()
                self.on_screen_render = True

            display_width = self.window.winfo_width()
            display_height = self.window.winfo_height()

            img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.human_rgb).resize(
                (display_width, display_height), resample=PIL.Image.NEAREST))
            self.canvas.create_image(display_width / 2, display_height / 2, image=img, anchor=tk.CENTER)

            # terrible workaround for window.update() not getting all keyboard input otherwise
            # confirmed that even with this, still only takes 1.5ms which is fine.
            for i in range(50):
                self.window.update()

            self.window_text.config(text='Timesteps remaining: %s' % str(self._max_episode_steps - self.ep_timesteps).zfill(4))

        elif mode == 'rgb_array':
            return self.rgb, self.depth

        elif mode == 'rgb_and_true_depth_and_segment_mask':
            return self.env.render('workspace', depth_type='true', segment_mask=True)

    def reset(self, mb_base_angle=None, reset_dict=None):
        self.img_rendered = False
        obs_dict = self.env.reset(mb_base_angle=mb_base_angle, reset_dict=reset_dict)
        self.prev_pos = None
        self._prev_grip_pos = None
        self._prev_grip_feedback = None
        self.ep_timesteps = 0

        # for debugging, show pos limits as shape
        if self.show_pos_limits:
            pbc = self.env._pb_client
            pl = self.env.gripper.pos_limits
            x_tot, y_tot, z_tot = (pl[1][0] - pl[0][0], pl[1][1] - pl[0][1], pl[1][2] - pl[0][2])

            if self.pos_limits_visual_body is not None:
                pbc.removeBody(self.pos_limits_visual_body)
            if self.pos_limits_visual is None:
                self.pos_limits_visual = pbc.createVisualShape(
                    shapeType=pbc.GEOM_BOX, rgbaColor=[0, 1, 0, .4], halfExtents=[x_tot/2, y_tot/2, z_tot/2])
            self.pos_limits_visual_body = pbc.createMultiBody(
                baseMass=0, baseVisualShapeIndex=self.pos_limits_visual,
                basePosition=[pl[0][0] + x_tot/2, pl[0][1] + y_tot/2, pl[0][2] + z_tot/2])

        return self._get_obs_from_obs_dict(obs_dict)[0]



if __name__ == '__main__':
    import time
    from manipulator_learning.sim.envs.thing_reaching import ThingReachingXYState

    # env = ThingEnv('reaching_xy', 'thing_2_finger', True, False, False)
    env = ThingReachingXYState()
    # env = ThingEnv('pushing_xy', 'thing_rod', True, False, False)
    obs = env.reset()

    # from manipulator_learning.sim.utils.gamepad_control import GamepadSteer
    # gs = GamepadSteer()
    # for i in range(10000):
    #     gs.process_events()
    #     t_vel = [gs.normalized_btn_state['LX'], -gs.normalized_btn_state['LY'], 0]
    #     r_vel = [0, 0, 0]
    #     grip = gs.normalized_btn_state['RT']
    #     env.step([t_vel, r_vel, grip])
    #     env.render()
    #     if gs.btn_state['A']:
    #         env.reset()
    #     time.sleep(.01)

    from manipulator_learning.learning.imitation.devices.keyboard_control import KeyboardSteer
    ks = KeyboardSteer()
    first_move = False
    loop_start = time.time()
    for i in range(10000):
        ks.process_events()
        mult = .3
        t_vel = mult * np.array([ks.btn_state['d'] - ks.btn_state['a'],
                 ks.btn_state['w'] - ks.btn_state['s']])
        # grip = ks.btn_state['space']
        env.step([t_vel])
        if ks.btn_state['d'] > 0:
            first_move = True
        if first_move:
            env.render()
        if bool(ks.btn_state['r']):
            env.reset()
        if i % 100 == 0:
            print('avg fps: ', 100 / (time.time() - loop_start))
            loop_start = time.time()
        # time.sleep(.01)
