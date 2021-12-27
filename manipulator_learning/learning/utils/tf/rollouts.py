from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.special import expit as sigmoid

from manipulator_learning.learning.data.tf.replay_buffer import ReplayBuffer
from manipulator_learning.learning.data.tf.img_replay_buffer import ImgReplayBufferRAM
from manipulator_learning.learning.imitation.intervenor import Intervenor
from manipulator_learning.learning.algorithms.fire.failure_predictors.failure_prediction_heuristic import \
    FailurePreidctorHeuristic
from manipulator_learning.learning.algorithms.fire.failure_predictors.failure_prediction_statistical import \
    FailurePredictorStatistical
from manipulator_learning.learning.agents.tf.ensemble_actors import EnsembleActor
from manipulator_learning.learning.eval.data_recording import DataRecorder, FP_STATUS_TAB
from manipulator_learning.learning.agents.tf.common import convert_env_obs_to_tuple
import manipulator_learning.learning.eval.plot.utils as plot_utils
from manipulator_learning.learning.agents.tf.bc_policy import BCWithResidualPolicy
from manipulator_learning.learning.utils.absorbing_state import Mask


def do_rollout(env,
               actor,
               replay_buffer: ReplayBuffer,
               noise_scale=0.1,
               num_trajectories=1,
               rand_actions=0,
               sample_random=False,
               add_absorbing_state=False,
               render=False,
               expert_replay_buffer: ReplayBuffer = None,
               intervenor: Intervenor = None,
               intervenor_reset_help_only=False,
               always_allow_intervention=False,
               failure_predictor=None,
               pretrain_rand_actions=0,
               pretrain_noise_scale=0.3,
               generate_vid_dict=None,
               all_successful_in_expert_rb=False,
               data_recorder: DataRecorder = None,
               human_expert_replay_buffer=None):
    """Do N rollouts. Note that this function has extensive changes to accommodate human interventions, and
    it would be relatively easy to generate something much, much simpler if all you need is regular environment
    rollouts.

    Args:
        env: environment to train on.
        actor: policy to take actions.
        replay_buffer: replay buffer to collect samples.
        noise_scale: std of gaussian noise added to a policy output.
        num_trajectories: number of trajectories to collect.
        rand_actions: number of random actions before using policy.
        sample_random: whether to sample a random trajectory or not.
        add_absorbing_state: whether to add an absorbing state.
        render: render the environment.
    Returns:
      An episode reward and a number of episode steps.
    """
    import time
    total_reward = 0
    total_timesteps = 0
    successful_episodes = []
    autonomous_successful_episodes = []

    obs_is_dict = True if type(env.observation_space) == gym.spaces.dict.Dict else False
    ros_env = True if 'ThingRos' in str(type(env)) else False

    for _ in range(num_trajectories):
        # add ability for user to restart to same initial seed if they make a mistake
        start_ep_total_reward = total_reward
        start_ep_total_timesteps = total_timesteps
        if add_absorbing_state:
            initial_random_state = env.unwrapped.np_random.get_state()
        else:
            initial_random_state = env.np_random.get_state()
        user_accepts_rollout = False
        while not user_accepts_rollout:
            if intervenor is None:
                user_accepts_rollout = True

            ep_start = time.time()
            if obs_is_dict:
                ep_buffer = ImgReplayBufferRAM()
                exp_ep_buffer = ImgReplayBufferRAM()
            else:
                ep_buffer = ReplayBuffer()
                exp_ep_buffer = ReplayBuffer()
            obs = env.reset()

            # reset with teleop in ros env
            if ros_env and env._reset_teleop_available:
                obs = intervenor.reset_env_with_teleop(env=env)

            prev_action = np.zeros_like(env.action_space.shape)
            episode_rewards = []
            episode_timesteps = 0
            successful_episode = False
            autonomous_success = False
            if intervenor is not None:
                intervenor.reset()
            inter_action = False
            prev_inter_action = False
            human_pred_failure = False
            ready_for_human_feedback = False
            ep_buffer_timesteps = []
            exp_ep_buffer_timesteps = []
            ep_d_values = []
            ep_raw_d_values = []
            ep_policy_variance = []
            ep_exp_in_control = []
            ep_failure_predict_status = []
            fail_predict_status = FP_STATUS_TAB['tn']
            ep_failure_predict_thresholds = []
            ep_failure_predict_posterior = []
            if type(failure_predictor) == FailurePreidctorHeuristic or type(
                    failure_predictor) == FailurePredictorStatistical:
                # failure_predictor.update_pc_list_from_ep()
                # failure_predictor.update_q_values()
                failure_predictor.new_ep_reset()
            if type(failure_predictor) == FailurePreidctorHeuristic:
                failure_predictor.include_q_cond = False

            ready_for_human_feedback = (intervenor is not None and
                                        not intervenor_reset_help_only and
                                        not sample_random and
                                        (replay_buffer is not None and len(replay_buffer) >= pretrain_rand_actions +
                                         rand_actions) and
                                        (failure_predictor is not None and
                                         failure_predictor.d_model_step >= failure_predictor.d_model_step_delay))

            while True:
                fail_predict_status = 'tn'
                if (replay_buffer is not None and
                    len(replay_buffer) < rand_actions) or sample_random:
                    action = env.action_space.sample()
                    inter_action = False

                else:
                    # give force feedback to user
                    if intervenor is not None and ros_env and intervenor.device_type == 'vr':
                        intervenor.force_feedback(env)

                    if obs_is_dict:
                        tfe_obs = convert_env_obs_to_tuple(obs)
                    else:
                        tfe_obs = tf.Variable([obs.astype('float32')])
                    forward_start = time.time()

                    if replay_buffer is not None and len(replay_buffer) < pretrain_rand_actions + rand_actions and \
                            type(actor) == BCWithResidualPolicy:
                        action = actor.bc_actor(tfe_obs).numpy()[
                            0]  # so that we can update the resid policy without using it
                    else:
                        if type(actor) == EnsembleActor or actor.num_ensemble > 1:
                            action, variance = actor.inference(tfe_obs)
                            action = action.numpy();
                            variance = variance.numpy()

                            ep_policy_variance.append(variance)
                            # action = actor.get_action(tfe_obs).numpy()
                        else:
                            action = actor(tfe_obs).numpy()[0]

                    # print('forward time %.4f' % (time.time() - forward_start))

                    if replay_buffer is not None and len(replay_buffer) < pretrain_rand_actions + rand_actions and \
                            pretrain_noise_scale > 0:
                        action += np.random.normal(size=action.shape) * pretrain_noise_scale
                    else:
                        if noise_scale > 0:
                            action += np.random.normal(size=action.shape) * noise_scale

                    action = action.clip(-1, 1)

                    fail_predict_start = time.time()

                    if ready_for_human_feedback or always_allow_intervention:
                        if intervenor is not None and intervenor.device_type == 'vr':
                            if ros_env:
                                cur_pos = None  # will get it right when it is needed so no delay
                            else:
                                if add_absorbing_state:
                                    obs_dict = env.unwrapped.env.gripper.receive_observation(
                                        ref_frame_pose=env.unwrapped.env.poses_ref_frame,
                                        ref_frame_vel=env.unwrapped.env.vel_ref_frame)
                                else:
                                    obs_dict = env.env.gripper.receive_observation(
                                        ref_frame_pose=env.env.poses_ref_frame,
                                        ref_frame_vel=env.env.vel_ref_frame)
                                cur_pos = np.concatenate([obs_dict['pos'], obs_dict['orient']])
                        else:
                            cur_pos = None

                    # always get failure_predictor data, even if we don't need it
                    if ready_for_human_feedback:
                        if type(failure_predictor) == FailurePreidctorHeuristic:
                            pred_failure, thresholds_reached = failure_predictor.get_failure_prediction(obs, action)
                        elif type(failure_predictor) == FailurePredictorStatistical:
                            pred_failure = failure_predictor.get_failure_prediction(obs, action, episode_timesteps)
                            ep_raw_d_values.append(failure_predictor.latest_raw_d_value)

                    if ready_for_human_feedback:
                        if failure_predictor is not None and not human_pred_failure:
                            # if type(failure_predictor) == FailurePreidctorHeuristic:
                            #   pred_failure, thresholds_reached = failure_predictor.get_failure_prediction(obs, action)
                            # elif type(failure_predictor) == FailurePredictorStatistical:
                            #   pred_failure = failure_predictor.get_failure_prediction(obs, action, episode_timesteps)
                            #   ep_raw_d_values.append(failure_predictor.latest_raw_d_value)
                            if pred_failure:
                                print('Failure predicted! Waiting for human response...')
                                human_pred_failure = intervenor.will_policy_fail(get_user_correction=False)
                                if human_pred_failure == 2:  # True positive, but not possible to correct -- "too_late"
                                    if type(failure_predictor) == FailurePredictorStatistical:
                                        failure_predictor.false_negative()
                                    fail_predict_status = 'tp_late'
                                elif human_pred_failure:  # true positive
                                    if type(failure_predictor) == FailurePreidctorHeuristic:
                                        failure_predictor.update_statistics('tp')
                                    fail_predict_status = 'tp'

                                    if ros_env:
                                        cur_pos = env.get_cur_base_tool_pose()
                                    action = intervenor.wait_for_action(ee_pose=cur_pos)
                                else:  # false positive
                                    if type(failure_predictor) == FailurePreidctorHeuristic:
                                        failure_predictor.update_statistics('fp')
                                    fail_predict_status = 'fp'

                                    if type(failure_predictor) == FailurePreidctorHeuristic:
                                        failure_predictor.update_thresh_false_positive(thresholds_reached)
                                        failure_predictor.update_consec_counts(
                                            -5 * thresholds_reached['consec_non_inc'],
                                            -5 * thresholds_reached['consec_neg_d'])
                                    elif type(failure_predictor) == FailurePredictorStatistical:
                                        failure_predictor.false_positive()

                # False negative failure prediction -- human_pred_failure is actually agreement with failure predictor
                if intervenor is not None and not human_pred_failure and \
                        (ready_for_human_feedback or always_allow_intervention):
                    human_pred_failure = intervenor.get_failure_prediction()
                    if human_pred_failure:  # false negative
                        print("Human predicts failure. Waiting for new action...")
                        if failure_predictor is not None and type(failure_predictor) == FailurePreidctorHeuristic:
                            failure_predictor.update_statistics('fn')
                        fail_predict_status = 'fn'

                        if type(failure_predictor) == FailurePreidctorHeuristic:
                            failure_predictor.update_consec_thresh(-3, -3)
                        elif type(failure_predictor) == FailurePredictorStatistical:
                            failure_predictor.false_negative()
                        # this action doesn't actually get used, since it is overwritten by get_action and then action = int_chosen_action
                        if ros_env:
                            time.sleep(.3)  # allow robot to (hopefully) stop moving
                            cur_pos = env.get_cur_base_tool_pose()
                            print('pos before wait: ', cur_pos)

                        action = intervenor.wait_for_action(ee_pose=cur_pos)
                        cur_pos = env.get_cur_base_tool_pose()
                        print('pos after wait: ', cur_pos)

                if intervenor is not None and not intervenor.want_intervention \
                        and failure_predictor is not None and failure_predictor.d_model_step > failure_predictor.d_model_step_delay \
                        and not ros_env:
                    intervenor.enforce_realtime_mult()

                #
                if intervenor is not None and (ready_for_human_feedback or always_allow_intervention):
                    if ros_env:
                        cur_pos = env.get_cur_base_tool_pose()
                    int_chosen_action = intervenor.get_action(ee_pose=cur_pos)
                else:
                    int_chosen_action = False

                # intervenor has taken over
                if int_chosen_action is not False:
                    action = int_chosen_action
                    if not inter_action and obs_is_dict and len(ep_buffer) > 0:
                        # on switch from non-expert to expert, to give a valid next_obs for learning q
                        ep_buffer.push_back(obs, np.zeros_like(action), 0, done_mask, done, False)
                    inter_action = True
                    if fail_predict_status == 'tn':  # means that no other prediction event has happened
                        fail_predict_status = 'exp'
                # intervenor has either returned control or not taken over
                else:
                    # on switch from expert to non-expert, append next_obs to exp_ep_buffer for learning valid q
                    # also set human_pred_failure back to false
                    if inter_action and obs_is_dict:
                        print('Control returned to agent.')
                        exp_ep_buffer.push_back(obs, np.zeros_like(action), 0, done_mask, done, False)
                        human_pred_failure = False
                        if type(failure_predictor) == FailurePreidctorHeuristic:
                            failure_predictor.reset_consec_counts()
                    inter_action = False

                # keep track of which timesteps had an expert acting for statistical failure predictor
                if inter_action:
                    ep_exp_in_control.append(True)
                    exp_ep_buffer_timesteps.append(episode_timesteps)
                else:
                    ep_exp_in_control.append(False)
                    ep_buffer_timesteps.append(episode_timesteps)

                # print('fail predict time: %.4f' % (time.time() - fail_predict_start))

                # get done success/failure
                if ros_env and (env.success_causes_done or env.failure_causes_done):
                    if intervenor.too_late:
                        env.set_done(True)
                    elif intervenor.good_failure_prediction:
                        env.set_done(False)

                env_start = time.time()
                next_obs, reward, done, info = env.step(action)

                # set success bit with vr if ros env
                if done and ros_env and env._reset_teleop_available and env._success_feedback_available \
                        and not (env.done_success or env.done_failure):
                    info['done_success'] = intervenor.get_suc_fail_fb()

                # print('env time: %.4f' % (time.time() - env_start))
                # for generating figs for video -------------------------------------------------------------------
                if generate_vid_dict is not None:
                    import time
                    vd = generate_vid_dict
                    # screenshots, not done at end since the saving happens relatively quickly
                    img, _ = env.unwrapped.env.render(mode='robot_facing')
                    import cv2
                    vd['imgs'].append(img)
                    img_fixed = copy.deepcopy(img)
                    img_fixed[:, :, 0] = img[:, :, 2]
                    img_fixed[:, :, 2] = img[:, :, 0]
                    cv2.imwrite(vd['ep_screenshots_dir'] + '/' + str(episode_timesteps).zfill(4) + '.png', img_fixed)

                    if type(failure_predictor) == FailurePredictorStatistical:
                        # d for making animation, done when episode is finished
                        ep_d_values.append(sigmoid(failure_predictor.get_discrim_output(obs, action)))
                        ep_failure_predict_thresholds.append(failure_predictor.qual_thresh)
                        # ep_failure_predict_posterior.append(failure_predictor.get_gda_pred_single_step(ep_d_values[-1],
                        #                                                                                episode_timesteps))
                        ep_failure_predict_posterior.append(np.mean(failure_predictor._prev_pred_outs))
                    elif type(failure_predictor) == FailurePreidctorHeuristic:
                        font_size = 20
                        # plot current ep d
                        d_fig = vd['d_fig']
                        d_fig_ax = vd['d_fig_ax']
                        tf_obs, tf_action = failure_predictor._prep_obs_act_for_model(obs, action)
                        d_out = sigmoid(failure_predictor._get_d_output([tf_obs], [tf_action]).numpy())
                        if d_out >= .5:
                            vd['beta_exp_x_val'] = episode_timesteps + failure_predictor.neg_d_thresh
                            vd['good_ep_d'].append(d_out)
                            vd['good_ep_ind'].append(episode_timesteps)
                            # d_fig_ax.scatter(d_out, episode_timesteps, label='Current Episode ($D\\geq0.5$)', s=30, color='green')
                        else:
                            vd['bad_ep_d'].append(d_out)
                            vd['bad_ep_ind'].append(episode_timesteps)
                            # d_fig_ax.scatter(d_out, episode_timesteps, label='Current Episode ($D<0.5$)', s=30, color='red')
                        d_fig_ax.scatter(vd['good_ep_ind'], vd['good_ep_d'], label='Current Episode ($D\\geq0.5$)',
                                         s=30, color='green')
                        d_fig_ax.scatter(vd['bad_ep_ind'], vd['bad_ep_d'], label='Current Episode ($D<0.5$)', s=30,
                                         color='red')
                        if vd['beta_exp_x_val'] < env._max_episode_steps:
                            if vd['v_line'] is not None:
                                vd['v_line'].remove()
                            vd['v_line'] = d_fig_ax.axvline(vd['beta_exp_x_val'], linestyle='--', color='red',
                                                            label='$\\beta$-set failure point',
                                                            linewidth=3)
                        if episode_timesteps == 0:
                            # d_fig_ax.scatter([], [], label='Current Episode ($D\\geq0.5$)', s=30, color='green')
                            # d_fig_ax.scatter([], [], label='Current Episode ($D<0.5$)', s=30, color='red')
                            d_fig_ax.legend(loc='upper left', fontsize=font_size - 8)

                        d_fig.tight_layout()
                        d_fig.savefig(vd['ep_d_dir'] + '/' + str(episode_timesteps).zfill(4) + '.png',
                                      bbox_inches='tight')

                # ---------------------------------------------------------------------------------------------------

                # Extremely important, otherwise Q function is not stationary!
                # Taken from: https://github.com/sfujim/TD3/blob/master/main.py#L123
                if not done or episode_timesteps + 1 == env._max_episode_steps:
                    done_mask = Mask.NOT_DONE.value
                else:
                    done_mask = Mask.DONE.value

                # absorbing states to timeout states...didn't appear to work
                # if done and episode_timesteps < env._max_episode_steps:
                #   done_mask = Mask.NOT_DONE.value

                episode_rewards.append(reward)
                total_reward += reward
                episode_timesteps += 1
                total_timesteps += 1
                ep_failure_predict_status.append(FP_STATUS_TAB[fail_predict_status])

                if type(info) == dict and 'done_success' in info.keys() and info['done_success']:
                    successful_episode = True

                if replay_buffer is not None:
                    if (add_absorbing_state and done and episode_timesteps < env._max_episode_steps):
                        next_obs = env.get_absorbing_state()
                    if obs_is_dict:
                        if inter_action:
                            exp_ep_buffer.push_back(obs, action, reward, done_mask, done, True)
                        else:
                            ep_buffer.push_back(obs, action, reward, done_mask, done, True)
                    else:
                        if inter_action:
                            exp_ep_buffer.push_back(obs, action, next_obs, [reward], [done_mask], done)
                        else:
                            ep_buffer.push_back(obs, action, next_obs, [reward], [done_mask], done)

                if render or ready_for_human_feedback:
                    env.render()

                if done:

                    # for generating plots for vid --------------------------------------------------------------------
                    if generate_vid_dict is not None:
                        np.set_printoptions(suppress=True, precision=4, linewidth=120)
                        plot_utils.setup_pretty_plotting()
                        font_size = 20
                        fig, axs = plt.subplots(2, 1, figsize=[6.4, 4.8 * 2], sharex='col')
                        ax = fig.add_subplot(111, frameon=False)
                        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                        ax.set_xlabel('Timestep', fontsize=font_size)
                        # ax.xaxis.labelpad = 23
                        ax.tick_params(axis='both', which='major')

                        # first the d fig
                        x_vals = np.array(range(env._max_episode_steps))
                        d_ax = axs[0]
                        d_ln, = d_ax.plot([], [], c='black', lw=3, label='Current Episode', linestyle='--')
                        d_scat_fail = d_ax.scatter([], [], s=250, c='C3', marker='X')
                        d_scat_return = d_ax.scatter([], [], s=250, c='C2', marker='P')
                        d_ax.plot(x_vals, failure_predictor.d_suc_means, label='Successes (avg)', c='C2', lw=2)
                        d_ax.plot(x_vals, failure_predictor.d_fail_means, label='Failures (avg)', c='C3', lw=2)
                        d_ax.fill_between(x_vals, failure_predictor.d_suc_means - failure_predictor.d_suc_stds,
                                          failure_predictor.d_suc_means + failure_predictor.d_suc_stds, alpha=.2,
                                          color='C2')
                        d_ax.fill_between(x_vals, failure_predictor.d_fail_means - failure_predictor.d_fail_stds,
                                          failure_predictor.d_fail_means + failure_predictor.d_fail_stds, alpha=.2,
                                          color='C3')
                        d_ax.set_ylabel('Discriminator Output', fontsize=font_size)
                        d_ax.yaxis.labelpad = 10
                        d_ax.xaxis.set_tick_params(labelsize=font_size - 8)
                        d_ax.yaxis.set_tick_params(labelsize=font_size - 8)
                        # d_ax.legend(fontsize=font_size - 4, loc='lower right')

                        # setup the posterior fig
                        p_ax = axs[1]
                        p_ln, = p_ax.plot([], [], c='black', label='Current Episode', lw=3, linestyle='--')
                        p_scat_fail = p_ax.scatter([], [], s=250, c='C3', marker='X')
                        p_scat_return = p_ax.scatter([], [], s=250, c='C2', marker='P')
                        p_ax.axhline(1 - ep_failure_predict_thresholds[0], lw=2, color='C3', label='Threshold')
                        p_ax.set_ylabel('$p(fail | D_t)$ (5-step avg)', fontsize=font_size)
                        p_ax.yaxis.labelpad = 10
                        p_ax.xaxis.set_tick_params(labelsize=font_size - 8)
                        p_ax.yaxis.set_tick_params(labelsize=font_size - 8)
                        # p_ax.legend(fontsize=font_size - 4, loc='lower right')
                        ep_failure_predict_posterior = 1 - np.array(ep_failure_predict_posterior)

                        # points where expert took over or returned control
                        consecutive_timesteps = np.split(exp_ep_buffer_timesteps,
                                                         np.where(np.diff(exp_ep_buffer_timesteps) != 1)[0] + 1)
                        fail_points, return_to_agent_points = [], []
                        if len(consecutive_timesteps[0]) > 0:
                            for sec in consecutive_timesteps:
                                fail_points.append(sec[0])
                                return_to_agent_points.append(sec[-1])

                        # anim stuff
                        d_anim_x, d_anim_y, p_anim_x, p_anim_y = [], [], [], []
                        fail_x, ret_x = [], []
                        d_fail_y, d_ret_y, p_fail_y, p_ret_y = [], [], [], []

                        def init():
                            return d_ln, p_ln, d_scat_fail, d_scat_return, p_scat_fail, p_scat_return

                        def update(index):
                            d_anim_x.append(index)
                            d_anim_y.append(ep_d_values[index])
                            d_ln.set_data(d_anim_x, d_anim_y)
                            p_anim_y.append(ep_failure_predict_posterior[index])
                            p_ln.set_data(d_anim_x, p_anim_y)
                            if index in fail_points:
                                fail_x.append(index);
                                d_fail_y.append(ep_d_values[index]);
                                p_fail_y.append(ep_failure_predict_posterior[index])
                            elif index in return_to_agent_points:
                                ret_x.append(index);
                                d_ret_y.append(ep_d_values[index]);
                                p_ret_y.append(ep_failure_predict_posterior[index])
                            d_scat_fail.set_offsets(np.c_[fail_x, d_fail_y]);
                            d_scat_return.set_offsets(np.c_[ret_x, d_ret_y])
                            p_scat_fail.set_offsets(np.c_[fail_x, p_fail_y]);
                            p_scat_return.set_offsets(np.c_[ret_x, p_ret_y])

                            d_ax.legend(fontsize=font_size - 4, loc='lower right')
                            p_ax.legend(fontsize=font_size - 4, loc='lower right')
                            return d_ln, p_ln, d_scat_fail, d_scat_return, p_scat_fail, p_scat_return

                        plt.tight_layout()
                        # ani = FuncAnimation(fig, update, frames=x_vals, init_func=init, blit=True, interval=10, repeat=False)
                        # plt.show()
                        # import ipdb; ipdb.set_trace()

                        writer = FFMpegWriter(fps=10)
                        with writer.saving(fig, vd['ep_d_dir'] + '/vid.mp4', 100):
                            init()
                            for i in x_vals:
                                update(i)
                                writer.grab_frame()
                    # -------------------------------------------------------------------------------------------------

                    # allow user to discard bad accidental traj, or discard a traj where failure prediction was correct but
                    # happened too late, corresponds to human_pred_failure == 2
                    if (len(exp_ep_buffer) > 0 and intervenor is not None) or human_pred_failure == 2:
                        keep_traj = intervenor.wait_for_keep_traj()
                        if not keep_traj:

                            # a hardcoded option for not adding new data
                            # user_accepts_rollout = True
                            # total_timesteps = env._max_episode_steps
                            # break

                            if add_absorbing_state:
                                env.unwrapped.np_random.set_state(initial_random_state)
                            else:
                                env.np_random.set_state(initial_random_state)
                            total_reward = start_ep_total_reward
                            total_timesteps = start_ep_total_timesteps

                            break
                        else:
                            user_accepts_rollout = True

                    else:
                        user_accepts_rollout = True

                    if intervenor is not None and intervenor.device_type == 'vr':
                        intervenor.device.reset_ref_poses()
                    # always say true negative, even if not successful, since these stats are based on user preference
                    if failure_predictor is not None and type(failure_predictor) == FailurePreidctorHeuristic:
                        failure_predictor.update_statistics('tn')

                    # append data to replay buffers
                    if replay_buffer is not None:
                        if intervenor is None and failure_predictor is None:
                            new_rb_indices = replay_buffer.combine(ep_buffer)
                            all_new_erb_indices = []

                        # FIRE stuff
                        if expert_replay_buffer is not None and intervenor is not None:

                            if obs_is_dict:
                                # last piece of data for q
                                if inter_action:
                                    exp_ep_buffer.push_back(next_obs, np.zeros_like(action), 0, done_mask, done, False)

                                if human_expert_replay_buffer is not None:
                                    new_human_exp_rb_indices = human_expert_replay_buffer.combine(exp_ep_buffer)

                                if not inter_action:
                                    # if successful and exp not in control, add last point to expert, all other data to non expert,
                                    # unless all_successful_in_expert_rb, then add all data (since last intervention) to exp
                                    if successful_episode and not all_successful_in_expert_rb:
                                        # if all_successful_in_expert_rb, we'll add this data to expert_rb later anyways
                                        exp_ep_buffer.push_back(obs, action, reward, done_mask, done, True)
                                        exp_ep_buffer.push_back(next_obs, np.zeros_like(action), 0, done_mask, done,
                                                                False)
                                        ep_buffer.valid_indices.pop()
                                        exp_ep_buffer_timesteps.append(ep_buffer_timesteps[-1])
                                        ep_buffer_timesteps.pop()
                                    else:
                                        ep_buffer.push_back(next_obs, np.zeros_like(action), 0, done_mask, done, False)

                            # find consecutive exections of policy
                            consecutive_timesteps = np.split(ep_buffer_timesteps,
                                                             np.where(np.diff(ep_buffer_timesteps) != 1)[0] + 1)
                            last_consecutive = consecutive_timesteps[-1]
                            # not inter_action b/c that means we had agent, then expert, and expert stayed on until end
                            autonomous_success = len(
                                consecutive_timesteps) == 1 and successful_episode and not inter_action

                            # add expert rb data to exp rb
                            all_new_erb_indices = []
                            new_erb_indices = expert_replay_buffer.combine(exp_ep_buffer)
                            all_new_erb_indices.extend(new_erb_indices)
                            if type(failure_predictor) == FailurePredictorStatistical and len(
                                    exp_ep_buffer_timesteps) > 0:
                                failure_predictor.append_traj(True, True, new_erb_indices, exp_ep_buffer_timesteps,
                                                              False)

                            # add non-expert data to exp rb if successful and all success in exp rb
                            if successful_episode and all_successful_in_expert_rb and not inter_action:
                                exp_ep_buffer_timesteps.extend(last_consecutive)
                                ep_buffer_timesteps = ep_buffer_timesteps[:-len(last_consecutive)]
                                new_erb_indices = expert_replay_buffer.combine(ep_buffer,
                                                                               start_index=-len(last_consecutive))
                                all_new_erb_indices.extend(new_erb_indices)
                                if len(consecutive_timesteps) > 1:
                                    new_rb_indices = replay_buffer.combine(ep_buffer, start_index=0,
                                                                           end_index=-len(last_consecutive))
                                else:
                                    new_rb_indices = []
                            else:
                                new_rb_indices = replay_buffer.combine(ep_buffer)

                            if type(failure_predictor) == FailurePredictorStatistical:
                                # take care of case when policy is last executing and succeeds
                                if successful_episode and not inter_action:
                                    if all_successful_in_expert_rb:
                                        last_consec_ep_buffer_indices = new_erb_indices
                                    else:
                                        last_consec_ep_buffer_indices = new_rb_indices[-len(last_consecutive):]

                                    if autonomous_success:  # means that agent executed without interruption and succeeded
                                        failure_predictor.append_traj(True, all_successful_in_expert_rb,
                                                                      last_consec_ep_buffer_indices, last_consecutive,
                                                                      True)
                                    else:  # means that this is just the end of a trajectory
                                        failure_predictor.append_traj(True, all_successful_in_expert_rb,
                                                                      last_consec_ep_buffer_indices, last_consecutive,
                                                                      False)
                                        if len(consecutive_timesteps) > 1:  # this is not true when we have agent -> expert -> end
                                            all_others = np.concatenate(consecutive_timesteps[:-1])
                                            if all_successful_in_expert_rb:
                                                all_other_ep_buffer_indices = new_rb_indices
                                            else:
                                                all_other_ep_buffer_indices = new_rb_indices[:-len(last_consecutive)]
                                            failure_predictor.append_traj(False, False, all_other_ep_buffer_indices,
                                                                          all_others, True)
                                else:  # if failure OR expert is executing at the end during success
                                    failure_predictor.append_traj(False, False, new_rb_indices, ep_buffer_timesteps,
                                                                  True)

                    # add all relevant data to data recorder
                    if data_recorder is not None:
                        cur_tot_ts = data_recorder.internal_data['total_timesteps']
                        new_tot_ts_to_add = range(cur_tot_ts, cur_tot_ts + episode_timesteps)
                        per_timestep_data_dict = dict(total_timesteps=new_tot_ts_to_add,
                                                      timesteps=range(episode_timesteps))
                        per_ts_keys = data_recorder.per_timestep_data.keys()
                        if 'reward' in per_ts_keys: per_timestep_data_dict['reward'] = episode_rewards
                        if 'fp_raw_d_output' in per_ts_keys: per_timestep_data_dict['fp_raw_d_output'] = ep_raw_d_values
                        if 'fp_status' in per_ts_keys: per_timestep_data_dict['fp_status'] = ep_failure_predict_status
                        if 'int_exp_in_control' in per_ts_keys: per_timestep_data_dict[
                            'int_exp_in_control'] = ep_exp_in_control
                        if 'in_exp_rb' in per_ts_keys:
                            exp_truth_array = np.isin(range(episode_timesteps), exp_ep_buffer_timesteps)
                            per_timestep_data_dict['in_exp_rb'] = exp_truth_array
                        if 'rb_index' in per_ts_keys and replay_buffer is not None and expert_replay_buffer is not None:
                            rb_indices = np.array(range(episode_timesteps))
                            rb_indices[exp_ep_buffer_timesteps] = all_new_erb_indices
                            rb_indices[ep_buffer_timesteps] = new_rb_indices
                            per_timestep_data_dict['rb_index'] = rb_indices
                        if 'actor_variance' in per_ts_keys: per_timestep_data_dict[
                            'actor_variance'] = ep_policy_variance
                        data_recorder.append_per_timestep_data(per_timestep_data_dict)

                    break
                # end of if done

                obs = next_obs
                prev_action = action

                tf.keras.backend.clear_session()  # without this, each run takes longer and longer..doesn't seem to break anything

            # Add an absorbing state that is extremely important for GAIL.
            if user_accepts_rollout:
                if add_absorbing_state and (replay_buffer is not None and
                                            episode_timesteps < env._max_episode_steps):
                    if obs_is_dict: raise NotImplementedError()
                    action = np.zeros(env.action_space.shape)
                    absorbing_state = env.get_absorbing_state()

                    # done=False is set to the absorbing state because it corresponds to
                    # a state where gym environments stopped an episode.
                    replay_buffer.push_back(absorbing_state, action, absorbing_state, [0.0],
                                            [Mask.ABSORBING.value], False)

                successful_episodes.append(successful_episode)
                autonomous_successful_episodes.append(autonomous_success)

        # print('ep: time %.4f' % (time.time() - ep_start))
    return total_reward / num_trajectories, total_timesteps // num_trajectories, \
           sum(successful_episodes) / len(successful_episodes), \
           sum(autonomous_successful_episodes) / len(autonomous_successful_episodes)
