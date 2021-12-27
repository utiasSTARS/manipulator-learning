import numpy as np
from scipy.special import expit as sigmoid, logsumexp
from scipy.signal import convolve2d
from scipy.stats import norm as norm_dist
import sys, os
import copy
import matplotlib.pyplot as plt

from manipulator_learning.learning.data.tf.img_replay_buffer import ImgReplayBufferDisk
from manipulator_learning.learning.eval.plot.utils import get_roc_stats_qual, get_roc_stats


class FailurePredictorStatistical:
    MODE_OPTIONS = ['qual_per_step', 'cum_qual_full_traj', 'multistep_zero_qual', 'cum_qual_multistep',
                    'gda']

    def __init__(self, discriminator, replay_buffer, expert_replay_buffer,
                 d_model_step, ep_len, num_suc_fail_trajs_avg=25, d_model_step_delay=3000,
                 save_avg_d_fig_dir=None, fp_ignore_quantity=5, percent_change_qual=.05, num_prev_for_prior=25,
                 prev_ts_mean_filter=5, min_tpr_thresh_set=.8, mode='gda'):
        """
        A failure predictor that grounds its predictions based on the average predictions of previous successful
        and failed trajectories, given ground truth labels of success and failure.

        :param discriminator: Discriminator object.
        :param d_model_step: The d_model_step object.
        :param replay_buffer: object containing the replay_buffer
        :param expert_replay_buffer: object containing the expert replay buffer.
        :param num_suc_fail_trajs_avg: Number of previous data to use per timestep for both successful
          and failure data (e.g. 25 here would mean 50 total (s,a) pairs per timestep, or 50 total "trajs".
        :param d_model_step_delay: d_model_step must be higher than this before any predictions are made.
        """
        self.discriminator = discriminator
        self.d_model_step = d_model_step
        self.d_model_step_delay = d_model_step_delay
        self.rb = replay_buffer
        self.erb = expert_replay_buffer
        assert mode in FailurePredictorStatistical.MODE_OPTIONS, 'Invalid mode %s' % mode
        self.mode = mode
        if type(self.rb) == ImgReplayBufferDisk:
            self.obs_is_dict = True
        else:
            raise NotImplementedError('Implement if needed')
        self.ep_len = ep_len

        # indices marked with -1 in these buffers indicate no data yet. Otherwise, these arrays contain the equivalent
        # of num_suc_fail_trajs_avg trajectories of data, but each "trajectory" might not be from one contiguous traj
        # (since trajectories of autonomous data, in intervention learning, do not necessarily all come in at same time)
        #
        # the first index of the final dimension contains the actual data index, the second index contains a
        # 1 if the data comes from the expert buffer, and a 0 if it comes from the non-expert buffer
        self.suc_traj_indices = (np.ones([num_suc_fail_trajs_avg, ep_len, 2]) * -1).astype('int')
        self.fail_traj_indices = (np.ones([num_suc_fail_trajs_avg, ep_len, 2]) * -1).astype('int')

        # variables for keeping track of historical stat data
        self.d_suc_means = np.ones(ep_len) * -1000
        self.d_suc_stds = np.ones(ep_len) * -1000
        self.d_fail_means = np.ones(ep_len) * -1000
        self.d_fail_stds = np.ones(ep_len) * -1000
        self.hell_dist = np.ones(ep_len) * -1000
        self.indicator = np.zeros(ep_len)

        # variables for doing failure prediction
        self.qual_thresh = 0
        self._cur_traj_cum_qual = 0
        self._cur_consec_cond = 0
        self.fp_ignore_quantity = fp_ignore_quantity
        self._ignore_count = 0  # ignore a series of predictions after a false positive
        self._qual_change_amount = [0, 0]  # amount to change qual thresh by after incorrect predictions for [fp, fn]
        if self.mode == 'multistep_zero_qual':
            self._qual_change_amount = [-1, -3]
        self.percent_change_qual = percent_change_qual  # percent of the full range of possible values

        self.save_avg_d_fig_dir = save_avg_d_fig_dir

        self.user_feedback_given = False
        self.print_freq = 10
        self._next_print_count = self.print_freq
        self.latest_raw_d_value = None

        # gda (gaussian discriminant analysis) variables
        self.suc_dists = [None] * ep_len
        self.fail_dists = [None] * ep_len
        self.log_suc_prior = np.log(0.5)
        self.log_fail_prior = np.log(0.5)
        self.prev_results = np.zeros(num_prev_for_prior)
        self._prev_pred_outs = np.ones(prev_ts_mean_filter)
        self.min_tpr_thresh_set = min_tpr_thresh_set

        self.gda_ready = False

        self._always_predict_false = False

    def append_traj(self, successful, expert_rb, traj_data_indices, timesteps, add_to_prev_suc_results=False):
        """
        Append a traj of data indices to the historical data.

        :param successful: Whether the trajectory was successful or not. If only adding a partial trajectory
          if it's partial because an expert intervened, counts as an unsuccessful traj.
        :param: Whether the trajectory is stored in the expert replay buffer. if false, stored in non-expert rb.
        :param traj_data_indices: The trajectory of data indices to add.
        :param timesteps: The t values for each piece of new data.
        :param add_to_prev_suc_results: whether to count this trajectory towards the prev suc results. If this is
          being called for only a small amount of data at the end of a traj, or as a result of an expert demo, this
          should be False.
        :return:
        """
        if len(traj_data_indices) != len(timesteps):
            print("traj data indices: ", traj_data_indices)
            print("timesteps: ", timesteps)
            import ipdb;
            ipdb.set_trace()

        # assert len(traj_data_indices) == len(timesteps), "Num new data indices %d does not match num of " \
        #                                                  "new timesteps %d" % (len(traj_data_indices), len(timesteps))
        if len(traj_data_indices) == 0:
            return

        if successful:
            # only roll the timesteps for which we have new data
            self.suc_traj_indices[:, timesteps, :] = np.roll(self.suc_traj_indices[:, timesteps], -1, axis=0)
            self.suc_traj_indices[-1, timesteps, 0] = traj_data_indices
            self.suc_traj_indices[-1, timesteps, 1] = expert_rb
        else:
            self.fail_traj_indices[:, timesteps, :] = np.roll(self.fail_traj_indices[:, timesteps], -1, axis=0)
            self.fail_traj_indices[-1, timesteps, 0] = traj_data_indices
            self.fail_traj_indices[-1, timesteps, 1] = expert_rb

        if add_to_prev_suc_results:
            self.prev_results = np.roll(self.prev_results, -1)
            self.prev_results[-1] = successful
            self.update_priors()

    def update_priors(self):
        suc_prop = self.prev_results.sum() / len(self.prev_results)
        fail_prop = 1 - suc_prop
        suc_prior = np.clip(suc_prop, .05, .95)
        fail_prior = np.clip(fail_prop, .05, .95)
        self.log_suc_prior = np.log(suc_prior)
        self.log_fail_prior = np.log(fail_prior)

    def discrim_as_batches(self, rb_data, rb, batch_size=512):
        num_data = rb_data[0].shape[0]
        num_batches = num_data // batch_size + 1
        rb_d_values = np.zeros([num_data, 1])
        for i in range(num_batches):
            sl = slice(i * batch_size, min((i + 1) * batch_size, num_data))
            batch_data = tuple([rb_data[d_ind][sl] for d_ind in range(len(rb_data))])
            if len(rb_d_values[sl]) > 0:
                if self.discriminator.num_ensemble == 1:
                    rb_d_values[sl] = self.discriminator(rb.tuple_batch_to_d_input(batch_data)).numpy()
                else:
                    rb_d_values[sl] = np.expand_dims(
                        self.discriminator.get_mean_discrim(rb.tuple_batch_to_d_input(batch_data)).numpy(), -1)

        return rb_d_values

    def calc_d_traj_avgs(self, total_numsteps):
        """ Calculate new statistics. Should be called everytime the Discriminator is updated before trying to
            predict failuures. """
        for traj_indices in [self.suc_traj_indices, self.fail_traj_indices]:
            rb_valid_indices = traj_indices[:, :, 1] == 0
            rb_indices = traj_indices[rb_valid_indices][:, 0]
            rb_data = self.rb.dataset.get_data(rb_indices.astype('int'))

            # instead of running all data thru discrim at the same time, run it through in batches to not overuse gpu mem
            # if batch_size is too high, uses too much mem, too low, too slow
            rb_d_values = self.discrim_as_batches(rb_data, self.rb)

            erb_valid_indices = traj_indices[:, :, 1] == 1
            erb_indices = traj_indices[erb_valid_indices][:, 0]
            if len(erb_indices) > 0:
                erb_data = self.erb.dataset.get_data(erb_indices.astype('int'))
                erb_d_values = self.discrim_as_batches(erb_data, self.erb)
            else:
                erb_d_values = np.array([])

            d_values = np.ones(traj_indices.shape[:2]) * -1
            d_values[rb_valid_indices] = rb_d_values.flatten()
            d_values[erb_valid_indices] = erb_d_values.flatten()
            d_values_sig = sigmoid(d_values)
            if traj_indices is self.suc_traj_indices:
                d_suc_values = copy.deepcopy(d_values)
                d_suc_values_sig = copy.deepcopy(d_values_sig)
                self.d_suc_means = d_values_sig.mean(axis=0)
                self.d_suc_stds = d_values_sig.std(axis=0)
                d_suc_means_sig = d_values_sig.mean(axis=0)
                d_suc_stds_sig = d_values_sig.std(axis=0)
            elif traj_indices is self.fail_traj_indices:
                d_fail_values = copy.deepcopy(d_values)
                d_fail_values_sig = copy.deepcopy(d_values_sig)
                self.d_fail_means = d_values_sig.mean(axis=0)
                self.d_fail_stds = d_values_sig.std(axis=0)
                d_fail_means_sig = d_values_sig.mean(axis=0)
                d_fail_stds_sig = d_values_sig.std(axis=0)

        if self.save_avg_d_fig_dir is not None:
            avg_d_fig = plt.figure()
            avg_d_fig_ax = avg_d_fig.add_subplot(111)
            ep_len = self.suc_traj_indices.shape[1]
            avg_d_fig_ax.axhline(.5, linestyle='--', color='black')
            avg_d_fig_ax.plot(range(ep_len), d_suc_means_sig, label='Successes')
            avg_d_fig_ax.plot(range(ep_len), d_fail_means_sig, label='Failures')
            avg_d_fig_ax.fill_between(range(ep_len), d_suc_means_sig - d_suc_stds_sig, d_suc_means_sig + d_suc_stds_sig,
                                      alpha=.1)
            avg_d_fig_ax.fill_between(range(ep_len), d_fail_means_sig - d_fail_stds_sig,
                                      d_fail_means_sig + d_fail_stds_sig,
                                      alpha=.1)
            avg_d_fig_ax.legend(fontsize=14, loc='lower right')
            # plt.show()
            os.makedirs(self.save_avg_d_fig_dir, exist_ok=True)
            avg_d_fig.savefig(self.save_avg_d_fig_dir + '/d_avg_ckpt_' + str(total_numsteps) + '.png')
            plt.close(avg_d_fig)

        if self.mode == 'gda':
            self.suc_dists = [norm_dist(m, s) for m, s in zip(self.d_suc_means, self.d_suc_stds)]
            self.fail_dists = [norm_dist(m, s) for m, s in zip(self.d_fail_means, self.d_fail_stds)]
            self.update_priors()

            # vectorized dists for getting optimal thresh
            vec_suc_dists = norm_dist(self.d_suc_means, self.d_suc_stds)
            vec_fail_dists = norm_dist(self.d_fail_means, self.d_fail_stds)
            all_d = np.concatenate([d_suc_values_sig, d_fail_values_sig])
            suc_log_probs = np.log(vec_suc_dists.pdf(all_d))
            fail_log_probs = np.log(vec_fail_dists.pdf(all_d))
            log_ds = logsumexp([suc_log_probs + self.log_suc_prior, fail_log_probs + self.log_fail_prior], axis=0)
            per_step_pred_suc = np.exp(suc_log_probs + self.log_suc_prior - log_ds)
            conv_op = np.ones([1, len(self._prev_pred_outs)]) * 1 / len(self._prev_pred_outs)
            conv_out = convolve2d(per_step_pred_suc, conv_op, fillvalue=1.0)[:, :self.ep_len]
            conv_out_min = conv_out.min(axis=1)
            num_suc_trajs = d_suc_values_sig.shape[0]
            num_fail_trajs = d_fail_values_sig.shape[0]
            test_values = np.linspace(conv_out_min.min() - .01, conv_out_min.max() + .01,
                                      (num_suc_trajs + num_fail_trajs) * 2)
            successes = np.concatenate([np.ones(num_suc_trajs), np.zeros(num_fail_trajs)], axis=0).astype('bool')
            roc_points = get_roc_stats_qual(successes, conv_out_min, test_values)
            acceptable_indices = np.argwhere(roc_points[:, 1] > self.min_tpr_thresh_set)
            acceptable_best_thresh = test_values[acceptable_indices][np.linalg.norm(
                roc_points - np.array([0, 1]), axis=1)[acceptable_indices].argmin()]
            self._qual_change_amount = [self.percent_change_qual * (conv_out_min.max() - conv_out_min.min())] * 2
            self.qual_thresh = acceptable_best_thresh

        # Hellinger distance -- see https://en.wikipedia.org/wiki/Hellinger_distance
        elif self.mode in ['cum_qual_full_traj', 'qual_per_step', 'multistep_zero_qual']:
            self.hell_dist = np.sqrt(1 - (np.sqrt(2 * self.d_suc_stds * self.d_fail_stds /
                                                  (self.d_suc_stds ** 2 + self.d_fail_stds ** 2)) *
                                          np.exp(-.25 * (self.d_suc_means - self.d_fail_means) ** 2 /
                                                 (self.d_suc_stds ** 2 + self.d_fail_stds ** 2))))
            self.indicator = (self.d_suc_means > self.d_fail_means)

            # set the threshold optimally based on best ROC point
            if not self.user_feedback_given:
                qual_per_ts_suc, qual_cum_suc, qual_per_ts_min_suc = self.get_quality(d_suc_values_sig)
                qual_per_ts_fail, qual_cum_fail, qual_per_ts_min_fail = self.get_quality(d_fail_values_sig)

                num_suc_trajs = d_suc_values_sig.shape[0]
                num_fail_trajs = d_fail_values_sig.shape[0]
                if self.mode == 'cum_qual_full_traj':
                    qual = np.concatenate([qual_cum_suc, qual_cum_fail], axis=0)
                elif self.mode == 'qual_per_step':
                    qual = np.concatenate([qual_per_ts_min_suc, qual_per_ts_min_fail], axis=0)
                elif self.mode == 'multistep_zero_qual':
                    qual_per_ts = np.concatenate([qual_per_ts_suc, qual_per_ts_fail], axis=0)
                    qual_lt_zero = (qual_per_ts < 0)
                    qual = self.get_max_consecutive_cond(qual_lt_zero)

                qual_test_values = np.linspace(qual.min() - .1, qual.max() + .1,
                                               (num_suc_trajs + num_fail_trajs) * 2)
                successes = np.concatenate([np.ones(num_suc_trajs), np.zeros(num_fail_trajs)], axis=0).astype('bool')

                if self.mode == 'multistep_zero_qual':
                    qual_roc_points = get_roc_stats_qual(successes, qual, qual_test_values,
                                                         qual_should_be_lt_param=False)
                    self._qual_change_amount = [-1, -3]  # since we use - and + for fp and fn respectively
                    self.qual_thresh = int(
                        qual_test_values[np.linalg.norm(qual_roc_points - np.array([0, 1]), axis=1).argmin()])
                elif self.mode in ['cum_qual_full_traj', 'qual_per_step']:
                    qual_roc_points = get_roc_stats_qual(successes, qual, qual_test_values)
                    self._qual_change_amount = [self.percent_change_qual * (qual.max() - qual.min()),
                                                self.percent_change_qual * (qual.max() - qual.min())]
                    self.qual_thresh = qual_test_values[
                        np.linalg.norm(qual_roc_points - np.array([0, 1]), axis=1).argmin()]

        self.gda_ready = True

    def get_max_consecutive_cond(self, cond):
        """ Get the maximum consecutive condition for a set of trajectories. cond should be a (N, ep_len) array
            of booleans indicating the condition."""
        # this is implemented with an outer for loop, but it's such a simple task that it shouldn't significantly slow
        # anything down
        max_consecs = []
        for traj in cond:
            diff = np.diff(traj)
            idx, = diff.nonzero()
            idx += 1
            if traj[0]:
                idx = np.r_[0, idx]
            if traj[-1]:
                idx = np.r_[idx, traj.size]
            idx = idx.reshape(-1, 2)
            lens = idx[:, 1] - idx[:, 0]
            if len(lens) == 0:
                max_consecs.append(0.)
            else:
                max_consecs.append(lens.max())
        return np.array(max_consecs)

    def get_quality(self, d_values):
        """ d_values should be a (N, ep_len) array"""
        qual_per_ts = (self.indicator * self.hell_dist * ((d_values - self.d_suc_means) / self.d_suc_stds +
                                                          (d_values - self.d_fail_means) / self.d_fail_stds))
        qual_cum = qual_per_ts.sum(axis=1)
        qual_per_ts_min = qual_per_ts.min(axis=1)

        return qual_per_ts, qual_cum, qual_per_ts_min

    def get_quality_single_step(self, d_value, t):
        """ Quality for a single d value. needs to have associated timestep. """
        return (self.indicator[t] * self.hell_dist[t] * ((d_value - self.d_suc_means[t]) / self.d_suc_stds[t] +
                                                         (d_value - self.d_fail_means[t]) / self.d_fail_stds[t]))

    def get_gda_pred_single_step(self, d_value, t):
        """ GDA prediction for a single d value and timestep. """
        if not self.gda_ready:
            return 0.5
        suc_log_prob = np.log(self.suc_dists[t].pdf(d_value))
        fail_log_prob = np.log(self.fail_dists[t].pdf(d_value))
        log_d = logsumexp([suc_log_prob + self.log_suc_prior, fail_log_prob + self.log_fail_prior], axis=0)
        pred_suc = np.exp(suc_log_prob + self.log_suc_prior - log_d)
        return pred_suc

    def get_discrim_output(self, obs, action):
        raw_discrim = self.discriminator(self.rb.obs_act_to_d_input(obs, action)).numpy().flatten()
        self.latest_raw_d_value = raw_discrim
        if self.discriminator.num_ensemble > 1:
            return np.mean(raw_discrim)
        else:
            return raw_discrim

    def get_failure_prediction(self, obs, action, t):

        d_out = self.get_discrim_output(obs, action)
        d_out_sig = sigmoid(d_out)
        if self.mode in ['cum_qual_full_traj', 'qual_per_step', 'multistep_zero_qual']:
            qual = self.get_quality_single_step(d_out_sig, t)
        elif self.mode == 'gda':
            qual = self.get_gda_pred_single_step(d_out_sig, t)
        if len(self._prev_pred_outs) > 1:
            self._prev_pred_outs = np.roll(self._prev_pred_outs, -1)
            self._prev_pred_outs[-1] = qual
            qual = np.mean(self._prev_pred_outs)
        if self.mode == 'multistep_zero_qual':
            if qual < 0:
                self._cur_consec_cond += 1  # still update consec values even if ignore count is above 0
            else:
                self._cur_consec_cond = 0
        if self._ignore_count > 0:
            self._ignore_count -= 1
            return False

        # print outputs
        if self._next_print_count == 0:
            if self.mode == 'multistep_zero_qual':
                print('Qual t: %.3f, cur consec: %d, qual thresh: %d' % (qual, self._cur_consec_cond, self.qual_thresh))
            elif self.mode == 'gda':
                print('pred suc t: %.4f, thresh: %.4f' % (qual, self.qual_thresh))
            self._next_print_count = self.print_freq
        else:
            self._next_print_count -= 1

        if self._always_predict_false:
            return False

        if self.mode in ['cum_qual_full_traj', 'qual_per_step', 'gda']:
            if self.mode == 'cum_qual_full_traj':
                self._cur_traj_cum_qual += qual
                qual_test = self._cur_traj_cum_qual
                # print('Qual t: %.3f, cum qual: %.3f, qual thresh: %.3f' % (qual, self._cur_traj_cum_qual, self.qual_thresh))
            elif self.mode in ['qual_per_step', 'gda']:
                qual_test = qual
            return (True if qual_test < self.qual_thresh else False)
        if self.mode == 'multistep_zero_qual':
            return (True if self._cur_consec_cond >= self.qual_thresh else False)

    def false_positive(self):
        """ Called by user after a false positive. """
        self._ignore_count = self.fp_ignore_quantity
        self.qual_thresh -= self._qual_change_amount[0]
        self.user_feedback_given = True

    def false_negative(self):
        """ Called by the user after a false negative """
        self.qual_thresh += self._qual_change_amount[1]
        self.user_feedback_given = True

    def new_ep_reset(self):
        self._ignore_count = 0
        self._cur_traj_cum_qual = 0
        self._cur_consec_cond = 0
        self._next_print_count = self.print_freq
        self._prev_pred_outs.fill(1)

    def set_always_predict_false(self, new_setting):
        self._always_predict_false = new_setting
