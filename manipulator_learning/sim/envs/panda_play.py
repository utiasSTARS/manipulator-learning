""" A set of environments with a shared observation/state space for multi-task and transfer learning """

import numpy as np
import copy

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.panda_default import CONFIG as DEF_CONFIG
import manipulator_learning.sim.envs.rewards.generic as rew_tools
import manipulator_learning.sim.envs.rewards.lift as lift_rew
import manipulator_learning.sim.envs.rewards.reach as reach_rew
import manipulator_learning.sim.envs.rewards.stack as stack_rew
import manipulator_learning.sim.envs.rewards.bring as bring_rew
import manipulator_learning.sim.envs.rewards.move as move_rew


PLAY_TASKS = ('stack', 'insert', 'bring', 'lift', 'reach', 'together', 'pick_and_place')
PLAY_TASK_SUFFIXES = ('_0', '_1', '_01', '_10')


class PandaPlayGeneric(ManipulatorEnv):
    VALID_AUX_TASKS = ['open_action',
                       'close_action',
                       'stack_0', 'stack_1',
                       'insert', 'insert_0', 'insert_1',
                       'bring', 'bring_0', 'bring_1',
                       'lift_0', 'lift_1',
                       'reach_0', 'reach_1',
                       'move_obj_0', 'move_obj_1',
                       'pick_and_place_0',
                       'unstack_stack_0']

    def __init__(self,
                 task,
                 camera_in_state,
                 dense_reward,
                 init_gripper_pose=((0.0, 0.5, 0.25), (np.pi, 0., 0.)),
                 init_gripper_random_lim=(.15, .15, .06, 0., 0., 0.),
                 obj_random_lim=((.15, .15, 0), (.15, .15, 0), (0, 0, 0), (0, 0, 0)),
                 obj_init_pos=((0, 0, 0), (0, 0, 0), (0.05, 0.0, -.0675), (-0.05, 0.0, -.0675)),  # for non-insert tray
                 obj_rgba=((0, 0, 1, 1), (0, 1, 0, 1), (0, 0, 1, .25), (0, 1, 0, .25)),
                 obj_urdf_names=('cube_blue_small', 'cube_blue_small', 'coaster', 'coaster'),
                 objs_in_state=(0, 1),
                 obj_targets=(2, 3),
                 rel_pos_in_state=(0, 1, (0, 1), (0, 2), (1, 3)),
                 tray_type='normal',
                 state_data=('pos', 'obj_pos', 'grip_pos', 'goal_pos'),
                 max_real_time=18,
                 n_substeps=5,
                 image_width=160,
                 image_height=120,
                 limits_cause_failure=False,
                 failure_causes_done=False,
                 success_causes_done=False,
                 egl=True,
                 action_multiplier=0.1,
                 valid_t_dofs=(1, 1, 1),
                 valid_r_dofs=(1, 1, 1),
                 control_method='v',
                 gripper_control_method='bool_p',
                 pos_limits=((.85, -.35, .655), (1.15, -0.05, 0.8)),   # for non-insert tray
                 main_task='stack_01',  # suffix must be integer, series of integers (for certain tasks), or nothing
                 config_dict_mods=None,
                 force_pb_direct=False,
                 sparse_cond_time=0.5,
                 pos_ctrl_max_arm_force=50,
                 **kwargs):

        config_dict = copy.deepcopy(DEF_CONFIG)
        config_dict.update(dict(
            init_gripper_pose=init_gripper_pose,
            init_gripper_random_lim=init_gripper_random_lim,
            obj_random_lim=obj_random_lim,
            obj_init_pos=obj_init_pos,
            obj_rgba=obj_rgba,
            obj_urdf_names=obj_urdf_names,
            objs_in_state=objs_in_state,
            rel_pos_in_state=rel_pos_in_state,
            tray_type=tray_type,
            control_method=control_method,
            gripper_control_method=gripper_control_method,
            pos_limits=pos_limits,
            pos_ctrl_max_arm_force=pos_ctrl_max_arm_force
        ))

        if config_dict_mods is not None:
            config_dict.update(config_dict_mods)

        super().__init__(task, camera_in_state,
                         dense_reward, True, 'b', state_data, max_real_time=max_real_time,
                         n_substeps=n_substeps, gap_between_prev_pos=.2,
                         image_width=image_width, image_height=image_height,
                         failure_causes_done=failure_causes_done, success_causes_done=success_causes_done,
                         egl=egl,
                         control_frame='b', action_multiplier=action_multiplier,
                         valid_t_dofs=valid_t_dofs, valid_r_dofs=valid_r_dofs,
                         new_env_with_fixed_depth=True, config_dict=config_dict,
                         generate_spaces=True, vel_ref_frame='b', force_pb_direct=force_pb_direct, **kwargs)
        self.sparse_cond_time = sparse_cond_time   # time to "hold" conditions for triggering sparse reward
        self.sparse_cond_start_time = None
        self.limits_cause_failure = limits_cause_failure
        self.done_success_reward = 100  # hard coded for now, may not work
        self.done_failure_reward = -5  # hard coded for now, may not work

        assert len(objs_in_state) == len(obj_targets), "Number of objects in states must equal number of objects" \
                                                       "acting as target positions, got %s state objects and %s" \
                                                       "target objects" % (objs_in_state, obj_targets)
        self.obj_targets = obj_targets  # indices of objects that act as targets for objs in state

        # for defining specific task reward
        # suffix must be integer or series of integers (for stack)
        und_loc = main_task.rfind('_')
        if und_loc > -1:
            self.main_task = main_task[:und_loc]
            self.task_suffix = [int(suf_char) for suf_char in list(main_task[und_loc + 1:])]

            # for stack, we're hardcoding that _0 or _1 indicates _01 or _10 for now.
            if self.main_task == 'stack' or self.main_task == 'unstack_stack' or self.main_task == 'unstack_stack_env_only':
                if self.task_suffix == [0]:
                    self.task_suffix = [0, 1]
                elif self.task_suffix == [1]:
                    self.task_suffix = [1, 0]
        else:
            self.main_task = main_task
            self.task_suffix = None

        # initial setting for unstack_stack -- can be modified for eval
        if self.main_task in ('unstack_stack', 'unstack_move_obj', 'unstack_lift', 'unstack_stack_env_only') and \
                hasattr(self.env, 'green_on_blue'):
            self.env.green_on_blue = True
            self._cube_rot_fix = True

    def reset_episode_success_data(self):
        """
        Call to reset latched_task_successes and all_task_sparse_timers properly.
        """
        if hasattr(self, "_latched_task_successes"):
            for task in self._latched_task_successes:
                self._latched_task_successes[task] = False
        if hasattr(self, "all_task_sparse_timers"):
            for task in self.all_task_sparse_timers:
                self.all_task_sparse_timers[task] = None

    def get_task_successes(self, tasks, observation, action, env_info):
        """
        Get success eval for list of tasks.

        Current options for tasks:
            - open_action (includes sparse reach and low velocity as well)
            - close_action (includes sparse reach and low velocity as well)
            - stack_0, stack_1
            - insert, insert_0, insert_1
            - bring, bring_0, bring_1
            - lift_0, lift_1
            - reach_0, reach_1
            - move_obj_0, move_obj_1
            - pick_and_place_0
            - unstack_stack_0
            - unstack_0
        """
        table_height, ee_pos, task_obj_indices, task_obj_poss, target_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id = \
            self._get_reward_state_info(task_suffix=[0, 1])

        successes = []

        if not hasattr(self, "_latched_task_successes") or self.ep_timesteps <= 1:
            self._latched_task_successes = dict()

        if not hasattr(self, "all_task_sparse_timers"):
            self.all_task_sparse_timers = dict()

        obj_vel = [env_info['obj_vel'][:3], env_info['obj_vel'][6:9]]
        obj_acc = [env_info['obj_acc'][:3], env_info['obj_acc'][6:9]]
        arm_vel = env_info['vel'][:3]

        for task in tasks:
            if task not in self.all_task_sparse_timers.keys():
                self.all_task_sparse_timers[task] = None

            und_loc = task.rfind('_')
            task_suffix = None
            if und_loc > -1:
                try:
                    task_suffix = int(task[und_loc + 1:])
                    assert task_suffix in [0, 1], "get_task_successes only implemented for 0 and 1 suffixes for now"
                    main_task = task[:und_loc]
                except ValueError:  # then it's a task that has an underscore but no integer on the end
                    task_suffix = None
                    main_task = task

            if main_task in ("pure_open", "pure_close"):
                if task == 'pure_open':
                    open_or_close = True if observation[-1, 6] >= 0.8 and observation[-1, 7] >= 0.8 and action[-1] < 0 else 0
                else:
                    open_or_close = True if observation[-1, 6] <= 0.2 and observation[-1, 7] <= 0.2 and action[-1] > 0 else 0
                suc_cur_timestep = open_or_close
            elif main_task in ("open_action", "close_action"):
                reach_0 = reach_rew.reach_sparse(task_obj_poss[0], ee_pos, .1)
                reach_1 = reach_rew.reach_sparse(task_obj_poss[1], ee_pos, .1)
                reach = reach_0 or reach_1
                if task == 'open_action':
                    open_or_close = True if action[-1] < 0 else 0
                else:
                    open_or_close = True if action[-1] > 0 else 0
                arm_vel_norm = np.linalg.norm(arm_vel)
                suc_cur_timestep = bool(reach and open_or_close and arm_vel_norm < .08)
                # suc_cur_timestep = bool(reach and open_or_close and arm_vel_norm < .25)  # for single traj shared data in lfgp

            elif main_task == 'unstack':
                # opposite block to task_suffix
                suc_cur_timestep = bool(task_obj_poss[1 - task_suffix, 2] <= table_height + .001)

            elif main_task == 'stack' or main_task == 'stack_pp_env' or main_task == 'unstack_stack' or main_task == 'unstack_stack_env_only':
                if task_suffix == 1:
                    suc_cur_timestep = stack_rew.stack_sparse(pbc, task_obj_pb_ids[::-1], arm_pb_id, table_pb_id)
                else:
                    suc_cur_timestep = stack_rew.stack_sparse(pbc, task_obj_pb_ids, arm_pb_id, table_pb_id)

            elif main_task == 'insert' or main_task == 'bring' or main_task == 'pick_and_place':
                if main_task == 'insert':
                    cur_target_obj_poss = copy.deepcopy(self._tray_insert_poss_world)
                    thresh = .0025
                elif main_task == 'bring':
                    cur_target_obj_poss = copy.deepcopy(target_obj_poss)
                    thresh = .03
                elif main_task == 'pick_and_place':
                    cur_target_obj_poss = copy.deepcopy(target_obj_poss)
                    # thresh = .01
                    thresh = .025
                if task_suffix is not None:
                    cur_target_obj_poss = cur_target_obj_poss[task_suffix]
                    cur_task_obj_poss = task_obj_poss[task_suffix]
                    cur_task_obj_pb_ids = np.atleast_1d(task_obj_pb_ids[task_suffix])

                if main_task == 'insert' or main_task == 'bring':
                    contacts = bring_rew.bring_contact_bonus_list(pbc, cur_task_obj_pb_ids, arm_pb_id, table_pb_id)
                else:
                    contacts = None
                all_suc_list = bring_rew.bring_sparse_multiple_list(cur_task_obj_poss, cur_target_obj_poss, thresh, contacts)
                suc_cur_timestep = all(all_suc_list)

            elif main_task == 'lift':
                suc_height = .06
                # suc_height = .03  # for single traj shared data in lfgp
                suc_cur_timestep = lift_rew.lift_sparse_multiple(task_obj_poss[task_suffix], suc_height, bottom_height=table_height)

            elif main_task == 'reach':
                thresh = .015
                # thresh = .03  # for single traj shared data in lfgp
                suc_cur_timestep = reach_rew.reach_sparse(task_obj_poss[task_suffix], ee_pos, thresh)

            elif main_task == 'move_obj':
                # if task_suffix == 0:
                #     obj_vel = env_info['obj_vel'][:3]
                #     obj_acc = env_info['obj_acc'][:3]
                # elif task_suffix == 1:
                #     obj_vel = env_info['obj_vel'][6:9]
                #     obj_acc = env_info['obj_acc'][6:9]
                suc_cur_timestep = move_rew.move_sparse(obj_vel[task_suffix], obj_acc)

            if main_task == 'move_obj':
                sparse_cond_time_override = 1.0
            else:
                sparse_cond_time_override = self.sparse_cond_time

            suc, self.all_task_sparse_timers[task] = rew_tools.hold_timer(
                suc_cur_timestep, self.ep_timesteps, self.real_t_per_ts, sparse_cond_time_override, self.all_task_sparse_timers[task])
            if task in self._latched_task_successes:
                self._latched_task_successes[task] = suc or self._latched_task_successes[task]
            else:
                self._latched_task_successes[task] = suc

            successes.append(self._latched_task_successes[task])

        return successes

    def _get_reward_state_info(self, task_suffix=None):
        if task_suffix is None:
            task_suffix = self.task_suffix

        if self.env.tray_type is not None:
            table_height = .665
        else:
            table_height = .6247

        ee_pos = np.array(rew_tools.get_world_ee_pose(self.env)[:3])
        obj_poss = []
        for obj_id in self.env._obj_ids:
            obj_poss.append(rew_tools.get_world_obj_pose(self.env, obj_id)[:3])
        obj_poss = np.array(obj_poss)

        # sort between obj pos and task obj pos, "task" objects are the ones that directly contribute to reward
        if task_suffix is not None:
            task_obj_indices = np.array(task_suffix)
        else:
            task_obj_indices = np.array(self.env.objs_in_state)

        task_obj_poss = obj_poss[task_obj_indices]

        # target objects
        obj_targets_array = np.array(self.obj_targets)
        target_obj_poss = obj_poss[obj_targets_array[task_obj_indices]]

        # get pb ids of objects, needed for some rewards
        pbc = self.env._pb_client
        task_obj_pb_ids = np.array(self.env._obj_ids)[task_obj_indices]
        arm_pb_id = self.env.gripper.body_id
        table_pb_id = self.env.table if self.env.tray_type is None else self.env.tray

        return table_height, ee_pos, task_obj_indices, task_obj_poss, target_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id

    def _calculate_reward_and_done(self, dense_reward, limit_reached):
        reward = 0
        done_failure = False

        table_height, ee_pos, task_obj_indices, task_obj_poss, target_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id = \
            self._get_reward_state_info()

        if self.main_task in ('stack', 'stack_open_close', 'stack_pp_env', 'unstack_stack', 'unstack_stack_env_only'):
            # unstack_stack should include bonus for green being moved off, but not implemented since currently unused
            obj_height = .04
            all_suc = stack_rew.stack_sparse(pbc, task_obj_pb_ids, arm_pb_id, table_pb_id)
            if dense_reward:
                reward = stack_rew.stack_dense(all_suc, task_obj_poss, obj_height, ee_pos)
        elif self.main_task == 'insert':
            # hard-coded success positions given particular tray urdf
            assert len(task_obj_indices) <= 2, "Tray is only set up for 1 or 2 insertions"
            assert hasattr(self, "_tray_insert_poss_world"), "Need tray insert positions to be defined as object attribute."
            target_obj_poss = self._tray_insert_poss_world
            if len(task_obj_indices) == 1:
                target_obj_poss = target_obj_poss[task_obj_indices]
            thresh = .0025
            contacts = bring_rew.bring_contact_bonus_list(pbc, task_obj_pb_ids, arm_pb_id, table_pb_id)
            all_suc_list = bring_rew.bring_sparse_multiple_list(task_obj_poss, target_obj_poss, thresh, contacts)
            all_suc = all(all_suc_list)
            if dense_reward:
                reward = bring_rew.bring_dense_multiple(task_obj_poss, ee_pos, target_obj_poss,
                                                        insert_bonuss=all_suc_list)
        elif self.main_task == 'bring' or self.main_task == 'bring_and_remove':
            thresh = .03
            correct_contact = bring_rew.bring_contact_bonus(pbc, task_obj_pb_ids, arm_pb_id, table_pb_id)
            all_suc = bring_rew.bring_sparse_multiple(task_obj_poss, target_obj_poss, thresh) and correct_contact
            if dense_reward:
                reward = bring_rew.bring_dense_multiple(task_obj_poss, ee_pos, target_obj_poss)
        elif self.main_task == 'lift' or self.main_task == 'lift_open_close' or self.main_task == 'unstack_lift':
            suc_height = .06
            all_suc = lift_rew.lift_sparse_multiple(task_obj_poss, suc_height, bottom_height=table_height)
            if dense_reward:
                reward = lift_rew.lift_dense_multiple(task_obj_poss, ee_pos, suc_height, bottom_height=table_height)
        elif self.main_task == 'reach':
            assert len(task_obj_indices) == 1, "Can't set main_task as reach and also have multiple suffix objects," \
                                               "suffix objects set to %s" % task_obj_indices
            thresh = .015
            all_suc = reach_rew.reach_sparse(task_obj_poss[0], ee_pos, thresh)
            if dense_reward:
                reward = reach_rew.dist_tanh(task_obj_poss[0], ee_pos)
        elif self.main_task == 'move_obj' or self.main_task == 'unstack_move_obj':
            # TODO implement if needed
            all_suc = False
            if dense_reward:
                reward = 0
        elif self.main_task == 'together':
            assert len(task_obj_indices) == 2, "main_task together only set up for 2 objects, object indices " \
                                               "set to %s" % task_obj_indices
            obj_dist_thresh = .05
            all_suc = bring_rew.bring_sparse(task_obj_poss[0], task_obj_poss[1], obj_dist_thresh)
            if dense_reward:
                reward = bring_rew.bring_dense(task_obj_poss[0], ee_pos, task_obj_poss[1])
        elif self.main_task == 'pick_and_place':
            assert type(self) == PandaPlayInsertTrayPlusPickPlaceXYZState, \
                "For main task pick_and_place_X, env must be PandaPlayInsertTrayPlusPickPlaceXYZState."
            thresh = .01
            all_suc = bring_rew.bring_sparse_multiple(task_obj_poss, target_obj_poss, thresh)
            if dense_reward:
                reward = bring_rew.bring_dense_multiple(task_obj_poss, ee_pos, target_obj_poss)
        elif self.main_task == 'all':
            # useful if auxiliary rewards are being used
            all_suc = False
            reward = 0
        else:
            raise NotImplementedError("Play env not implemented for main_task \"%s\"" % self.main_task)

        done_success, self.sparse_cond_start_time = rew_tools.hold_timer(
           all_suc, self.ep_timesteps, self.real_t_per_ts, self.sparse_cond_time, self.sparse_cond_start_time)
        if not dense_reward:
            reward = done_success

        return reward, done_success, done_failure


class PandaPlayXYZState(PandaPlayGeneric):
    # obs space is 59, act space is 4
    # n_substeps of 5 plus action multiplier of .002 means action mag of 1.0 moves desired position 2mm*5 = 1cm
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='stack_01', force_pb_direct=True, **kwargs):
        super().__init__('None', False, dense_reward, max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier,
                         state_data=('pos', 'vel', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'obj_vel',
                                     'obj_rot_vel', 'force_torque'),
                         valid_t_dofs=(1, 1, 1), valid_r_dofs=(0, 0, 0), control_method='dp',
                         main_task=main_task, force_pb_direct=force_pb_direct,
                         **kwargs)


class PandaPlayInsertTrayXYZState(PandaPlayGeneric):
    # obs space is 59, act space is 4
    # obs indices:
    #   - pos:               0:3
    #   - vel:               3:6
    #   - grip_pos:          6:8
    #   - prev_grip_pos:    8:12
    #   - obj_pos:         12:26
    #   - obj_vel:         26:38
    #   - obj_rel_pos:     38:53
    #   - force_torque:    53:59
    # n_substeps of 5 plus action multiplier of .002 means action mag of 1.0 moves desired position 2mm*5 = 1cm
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='insert', force_pb_direct=True, **kwargs):
                 # main_task='unstack_stack_0', force_pb_direct=True, **kwargs):
        super().__init__('None', False, dense_reward, max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier,
                         state_data=('pos', 'vel', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'obj_vel',
                                     'obj_rot_vel', 'force_torque'),
                         tray_type='2_cube_insert',
                         obj_init_pos=((0, 0, 0), (0, 0, 0), (0.075, 0.0, -.055), (-0.075, 0.0, -.055)), # for insert tray
                         valid_t_dofs=(1, 1, 1), valid_r_dofs=(0, 0, 0), control_method='dp',
                         pos_limits=((.85, -.35, .665), (1.15, -0.05, 0.8)),  # for insert tray
                         main_task=main_task, force_pb_direct=force_pb_direct,
                         **kwargs)

        # hardcode the insertion locations, since they correspond to the urdf itself
        self._tray_insert_poss_world = np.array([[1.075, -.2, .655], [.925, -.2, .655]])

        # extra info for LFGP
        self._extra_info_dict = dict(
            insert_goal_poss=np.array([[.075, .5, .135], [-.075, .5, .135]]),  # relative to robot base frame
            bring_goal_poss=np.array([[.075, .5, .145], [-.075, .5, .145]])
        )


class PandaPlayInsertTrayDPGripXYZState(PandaPlayInsertTrayXYZState):
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='insert', **kwargs):
        super().__init__(max_real_time, n_substeps, dense_reward, action_multiplier, main_task,
                         gripper_control_method='dp', grip_multiplier=0.25)


class PandaPlayInsertTrayPlusPickPlaceXYZState(PandaPlayGeneric):
    # Same as PandaPlayInsertTrayXYZState, but with added Pick and Place main task
    #
    # obs space is 65, act space is 4
    # obs indices:
    #   - pos:               0:3
    #   - vel:               3:6
    #   - grip_pos:          6:8
    #   - prev_grip_pos:    8:12
    #   - obj_pos:         12:29    # sphere target has no rotation in state, so 7 + 7 + 3
    #   - obj_vel:         29:41    # sphere not included here
    #   - obj_rel_pos:     41:59    # sphere rel to blue block added here (at end)
    #   - force_torque:    59:65
    # n_substeps of 5 plus action multiplier of .002 means action mag of 1.0 moves desired position 2mm*5 = 1cm
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='pick_and_place_01', force_pb_direct=True, **kwargs):
        super().__init__('None', False, dense_reward, max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier,
                         obj_random_lim=((.15, .15, 0), (.15, .15, 0), (.15, .15, .1), (0, 0, 0), (0, 0, 0)),
                         # for non-insert tray
                         obj_rgba=((0, 0, 1, 1), (0, 1, 0, 1), (.5, .8, .95, .75), (0, 0, 1, .25), (0, 1, 0, .25)),
                         obj_urdf_names=('cube_blue_small', 'cube_blue_small', 'sphere_no_col_fit_small_cube_bigger', 'coaster', 'coaster'),
                         objs_in_state=(0, 1, 2),
                         objs_no_rot_no_vel=(2,),
                         obj_targets=({'pick_and_place': 2, 'bring': 3}, 4, -1),
                         rel_pos_in_state=(0, 1, (0, 1), (0, 3), (1, 4), (0, 2)),
                         state_data=('pos', 'vel', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'obj_vel',
                                     'obj_rot_vel', 'force_torque'),
                         tray_type='2_cube_insert',
                         obj_init_pos=((0, 0, 0), (0, 0, 0), (0, 0, .04), (0.075, 0.0, -.055), (-0.075, 0.0, -.055)), # for insert tray
                         valid_t_dofs=(1, 1, 1), valid_r_dofs=(0, 0, 0), control_method='dp',
                         pos_limits=((.85, -.35, .665), (1.15, -0.05, 0.8)),  # for insert tray
                         main_task=main_task, force_pb_direct=force_pb_direct,
                         **kwargs)

        # hardcode the insertion locations, since they correspond to the urdf itself
        self._tray_insert_poss_world = np.array([[1.075, -.2, .655], [.925, -.2, .655]])

        # extra info for LFGP
        self._extra_info_dict = dict(
            insert_goal_poss=np.array([[.075, .5, .135], [-.075, .5, .135]]),  # relative to robot base frame
            bring_goal_poss=np.array([[.075, .5, .145], [-.075, .5, .145]])
        )

    def _get_reward_state_info(self, task_suffix=None):
        if task_suffix is None:
            task_suffix = self.task_suffix

        if self.env.tray_type is not None:
            table_height = .665
        else:
            table_height = .6247

        ee_pos = np.array(rew_tools.get_world_ee_pose(self.env)[:3])
        obj_poss = []
        for obj_id in self.env._obj_ids:
            obj_poss.append(rew_tools.get_world_obj_pose(self.env, obj_id)[:3])
        obj_poss = np.array(obj_poss)

        # sort between obj pos and task obj pos, "task" objects are the ones that directly contribute to reward
        if task_suffix is not None:
            task_obj_indices = np.array(task_suffix)
        else:
            task_obj_indices = np.array(self.env.objs_in_state)
            task_obj_indices = task_obj_indices[:-1]  # remove pick and place target from task_obj_indices

        task_obj_poss = obj_poss[task_obj_indices]

        # target objects
        obj_targets = []
        for target in self.obj_targets:
            if type(target) == dict:
                if self.main_task in target.keys():
                    obj_targets.append(target[self.main_task])
                else:
                    obj_targets.append(0)
            elif type(target) == int and target > -1:
                obj_targets.append(target)

        obj_targets_array = np.array(obj_targets)

        target_obj_poss = obj_poss[obj_targets_array[task_obj_indices]]

        # get pb ids of objects, needed for some rewards
        pbc = self.env._pb_client
        task_obj_pb_ids = np.array(self.env._obj_ids)[task_obj_indices]
        arm_pb_id = self.env.gripper.body_id
        table_pb_id = self.env.table if self.env.tray_type is None else self.env.tray

        return table_height, ee_pos, task_obj_indices, task_obj_poss, target_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id
